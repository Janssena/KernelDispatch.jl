#!/usr/bin/env python3
"""Generate the trifast fixtures for KernelDispatch's tests: trifast's REAL CUDA kernels (triangle attention, forward and
the three backward kernels), compiled by real Triton, with the inputs and outputs torch/Triton produced for them.

Run on a machine with an NVIDIA GPU, Triton, a CUDA build of torch and trifast importable (`pip install -e <trifast>` or
`PYTHONPATH=<trifast>/src`):

    python3 test/fixtures/make_trifast_fixture.py --kernels <trifast>/src/trifast/triton-original.py

`--kernels` is the file with the CUDA kernels (`trifast/triton.py` in a trifast checkout without the Metal changes; in the
checkout this was written against it is `triton-original.py`). It can be left out when `import trifast.triton` works.

What it does, for each of `_fwd`, `_bwd_q`, `_bwd_kv`, `_bwd_b` and a few (BLOCK_J, BLOCK_K, num_warps, num_stages) configs
taken from trifast's own autotune lists:
  1. launches the kernel's JIT function directly with that config (no autotuning, so the binary is known), with a fresh
     TRITON_CACHE_DIR so the files are exactly what Triton wrote;
  2. copies `<kernel>.cubin/.ptx/.json` into test/fixtures/trifast/<instance>/;
  3. records the call as `call.json`: the grid and the arguments the COMPILED kernel takes, in order (arrays by name,
     scalars by value) - Triton drops constexpr arguments and integers equal to 1, and the script checks its list against
     the PTX entry's parameters, so the Julia test needs no knowledge of trifast;
  4. saves Triton's own output for it (`expected_<name>.bin`) and compares it with a DENSE torch reference computed in
     float64 (`data/ref_*.bin`), so the fixture itself is checked before it is committed.
Inputs are shared in test/fixtures/trifast/data/ (raw little-endian float32, the mask as bytes; row-major, shapes in
`case.json`). Everything is float32 here: this is a plumbing test of the launch path, not of trifast's bf16 numerics.

It also prints a specialisation probe: how many distinct binaries Triton compiles for the same kernel and config as the
sequence length changes. A Triton binary is compiled FOR the arguments it was first launched with (integers equal to 1 and
integers/pointers divisible by 16 are specialised), so a binary registered with KernelDispatch is valid only for launches
with the same specialisation; paste the report back.

Nothing here is needed to RUN the Julia tests: the generated files are committed.
"""
import argparse
import glob
import importlib
import importlib.util
import json
import math
import os
import platform
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "trifast")
CACHE = tempfile.mkdtemp(prefix="kd_trifast_cache_")
os.environ["TRITON_CACHE_DIR"] = CACHE            # before Triton is imported

import numpy as np  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402
from triton.runtime.jit import JITFunction  # noqa: E402

if not torch.cuda.is_available():
    sys.exit("torch cannot see a CUDA device: this script needs the NVIDIA GPU")
torch.backends.cuda.matmul.allow_tf32 = False

DEV = "cuda"
# Sizes: n is NOT a multiple of any block size (masks matter) and not divisible by 16 (see the specialisation probe).
BS, H, N, D = 2, 2, 40, 32

# Configs taken from trifast's own autotune lists (autotune_helpers.py). The first of each kernel is the "primary" one:
# its outputs feed the next kernels' inputs.
CONFIGS = {
    "_fwd": [(32, 32, 1, 1), (64, 32, 4, 2), (128, 32, 2, 2)],
    "_bwd_q": [(32, 32, 1, 1), (64, 32, 2, 2)],
    "_bwd_kv": [(32, 32, 1, 2), (64, 32, 2, 1)],
    "_bwd_b": [(32, 32, 1, 2), (16, 16, 2, 2)],
}
# The tensors each kernel WRITES, by the base name of its `<base>_ptr` argument.
OUTPUTS = {"_fwd": ["o", "lse"], "_bwd_q": ["d", "dq"], "_bwd_kv": ["dk", "dv"], "_bwd_b": ["db"]}
# Base names that are another tensor's name in the data directory.
ALIAS = {"m": "mask", "lse": "l"}
# The dense reference each output is compared with.
REFERENCE = {"o": "ref_o", "lse": "ref_l", "d": "ref_d", "dq": "ref_dq", "dk": "ref_dk", "dv": "ref_dv", "db": "ref_db"}


def load_kernels(path):
    if path:
        spec = importlib.util.spec_from_file_location("trifast_cuda_kernels", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod                  # Triton reads the kernels' source through the module
        spec.loader.exec_module(mod)
        return mod
    try:
        return importlib.import_module("trifast.triton")
    except ModuleNotFoundError:
        sys.exit("pass --kernels <trifast>/src/trifast/triton-original.py (the file with the CUDA kernels)")


def jit_of(kernel):
    """The JITFunction under trifast's Autotuner wrapper (or the JITFunction itself)."""
    return kernel if isinstance(kernel, JITFunction) else kernel.fn


def param_names(jit):
    return [(p.name, p.is_constexpr) for p in jit.params]


def value_of(name, tensors, scalars):
    """The Python value of kernel argument `name`: `<base>_ptr` -> the tensor, `stride_<base><axis>` -> its stride."""
    if name.endswith("_ptr"):
        return tensors[name[:-4]]
    if name.startswith("stride_"):
        rest = name[len("stride_"):]
        base, axis = rest[:-1], rest[-1]
        return tensors[base].stride("hmnd".index(axis))
    return scalars[name]


def entry_dirs(entry):
    return {os.path.dirname(p) for p in glob.glob(os.path.join(CACHE, "**", entry + ".json"), recursive=True)}


def ptx_params(ptx):
    start = ptx.find(".entry")
    end = ptx.find("{", start)
    return [l.strip().rstrip(",") for l in ptx[start:end].splitlines() if ".param" in l]


def save_tensor(directory, name, t):
    arr = t.detach().cpu().contiguous()
    if arr.dtype == torch.bool:
        arr.numpy().astype(np.uint8).tofile(os.path.join(directory, name + ".bin"))
    else:
        arr.numpy().astype(np.float32).tofile(os.path.join(directory, name + ".bin"))


def dense(q, k, v, b, mask):
    """Triangle attention, written densely: q,k,v (bs,h,i,j,d), b (bs,h,j,k), mask (bs,i,k) True = masked out."""
    s = torch.einsum("bhijd,bhikd->bhijk", q, k) * (q.shape[-1] ** -0.5) + b[:, :, None]
    s = s.masked_fill(mask[:, None, :, None, :], torch.finfo(torch.float32).min)
    p = torch.softmax(s, -1)
    return torch.einsum("bhijk,bhikd->bhijd", p, v), torch.logsumexp(s, -1)


def launch(kernel, tensors, scalars, grid, cfg, require_new=True):
    """Launch one config of `kernel`. Returns (its new cache dir, the arguments it was launched with)."""
    jit = jit_of(kernel)
    bj, bk, warps, stages = cfg
    scalars = dict(scalars, BLOCK_J=bj, BLOCK_K=bk)
    kwargs, runtime = {}, []
    for name, is_constexpr in param_names(jit):
        v = value_of(name, tensors, scalars)
        kwargs[name] = v
        if not is_constexpr:
            runtime.append((name, v))
    entry = jit.fn.__name__
    before = entry_dirs(entry)
    jit[grid](**kwargs, num_warps=warps, num_stages=stages)
    torch.cuda.synchronize()
    new = entry_dirs(entry) - before
    if not require_new:
        return new, runtime
    assert len(new) == 1, f"expected one new {entry}.json in Triton's cache, found {new}"
    return new.pop(), runtime


def describe_args(runtime, tensors):
    """The arguments the compiled kernel takes: Triton drops integers equal to 1 (and constexprs, already out)."""
    args = []
    for name, v in runtime:
        if name.endswith("_ptr"):
            base = name[:-4]
            args.append(dict(kind="tensor", name=name, tensor=ALIAS.get(base, base), base=base,
                             aligned16=(v.data_ptr() % 16 == 0)))
        elif isinstance(v, bool):
            raise SystemExit(f"unexpected bool argument {name}")
        elif isinstance(v, int):
            if v == 1:
                continue                      # specialised away by Triton
            args.append(dict(kind="i32", name=name, value=v, divisible16=(v % 16 == 0)))
        elif isinstance(v, float):
            args.append(dict(kind="f32", name=name, value=v))
        else:
            raise SystemExit(f"unexpected argument {name}: {type(v)}")
    return args


def specialisation_probe(kernel):
    """How many binaries Triton compiles for ONE config as the sequence length changes."""
    jit = jit_of(kernel)
    entry = jit.fn.__name__
    rows = []
    for n in (33, 40, 41, 48, 64):
        g = torch.Generator(device=DEV).manual_seed(1)
        bh = 2
        q = torch.randn(bh, n, n, D, generator=g, device=DEV)
        o = torch.zeros_like(q)
        l = torch.zeros(bh, n, n, device=DEV)
        b = torch.randn(bh, n, n, generator=g, device=DEV)
        mask = torch.zeros(1, n, n, dtype=torch.bool, device=DEV)
        tensors = dict(o=o, lse=l, q=q, k=q, v=q, b=b, mask=mask)
        scalars = dict(sm_scale=D ** -0.5, neg_inf=-3.4e38, N=n, H=2, DIM=D, CLOSEST_N=2 ** math.ceil(math.log2(n)))
        before = len(entry_dirs(entry))
        launch(kernel, tensors, scalars, (math.ceil(n / 32), n, bh), (32, 32, 1, 1), require_new=False)
        rows.append((n, len(entry_dirs(entry)) - before))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", help="the file with trifast's CUDA Triton kernels")
    args = ap.parse_args()
    mod = load_kernels(args.kernels)
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)

    # --- the data -------------------------------------------------------------------------------------------------
    g = torch.Generator(device=DEV).manual_seed(0)
    rnd = lambda *shape: torch.randn(*shape, generator=g, device=DEV, dtype=torch.float32)
    q, k, v = (rnd(BS, H, N, N, D) for _ in range(3))
    b = rnd(BS, H, N, N)
    mask = torch.rand(BS, N, N, generator=g, device=DEV) < 0.3
    do = rnd(BS, H, N, N, D)
    assert not mask.all(-1).any(), "a fully masked row: pick another seed"
    bh = BS * H
    flat = lambda t: t.reshape(bh, *t.shape[2:]).contiguous()
    tensors = dict(q=flat(q), k=flat(k), v=flat(v), b=flat(b), mask=mask.contiguous(), do=flat(do))
    scalars = dict(sm_scale=D ** -0.5, neg_inf=torch.finfo(torch.float32).min, N=N, H=H, DIM=D,
                   CLOSEST_N=2 ** math.ceil(math.log2(N)))

    # --- the dense float64 reference ------------------------------------------------------------------------------
    qd, kd, vd, bd = (t.double().requires_grad_() for t in (q, k, v, b))
    od, ld = dense(qd, kd, vd, bd, mask)
    od.backward(do.double())
    ref = dict(ref_o=flat(od.detach()), ref_l=ld.detach().reshape(bh, N, N), ref_dq=flat(qd.grad), ref_dk=flat(kd.grad),
               ref_dv=flat(vd.grad), ref_db=flat(bd.grad))
    for name, t in tensors.items():
        save_tensor(os.path.join(OUT, "data"), name, t)

    def zeros_like(name):
        shape = {"o": (bh, N, N, D), "dq": (bh, N, N, D), "dk": (bh, N, N, D), "dv": (bh, N, N, D),
                 "lse": (bh, N, N), "d": (bh, N, N), "db": (bh, N, N)}[name]
        return torch.zeros(shape, device=DEV, dtype=torch.float32)

    grids = {
        "_fwd": lambda c: (math.ceil(N / c[0]), N, bh),
        "_bwd_q": lambda c: (math.ceil(N / c[0]), N, bh),
        "_bwd_kv": lambda c: (math.ceil(N / c[1]), N, bh),
        "_bwd_b": lambda c: (math.ceil(N / c[0]), math.ceil(N / c[1]), bh),
    }

    instances, report = [], []
    primary = {}                                     # kernel -> outputs of its first config, feeding the later kernels

    def run_kernel(kname, cfg, is_primary):
        kernel = getattr(mod, kname)
        outs = {o: zeros_like(o) for o in OUTPUTS[kname]}
        # The dict a kernel's argument names are looked up in: its outputs, the shared inputs, and the earlier kernels'
        # outputs. `m` is the mask under another name in _bwd_kv/_bwd_b, `lse`/`l` the log-sum-exp.
        t = dict(tensors)
        t.update(m=tensors["mask"])
        t.update(primary.get("_fwd", {}))
        t.update(primary.get("_bwd_q", {}))
        t.update(outs)
        try:
            src, runtime = launch(kernel, t, scalars, grids[kname](cfg), cfg)
        except Exception as e:  # an unsupported config on this GPU (out of shared memory, ...)
            if is_primary:
                raise
            print(f"   skipped {kname} {cfg}: {type(e).__name__}: {str(e).splitlines()[0]}")
            return None
        entry = jit_of(kernel).fn.__name__
        name = f"{kname.lstrip('_')}_J{cfg[0]}_K{cfg[1]}_w{cfg[2]}_s{cfg[3]}"
        dst = os.path.join(OUT, name)
        os.makedirs(dst, exist_ok=True)
        for ext in ("cubin", "ptx", "json"):
            shutil.copy(os.path.join(src, f"{entry}.{ext}"), os.path.join(dst, f"{entry}.{ext}"))
        meta = json.load(open(os.path.join(dst, entry + ".json")))
        params = ptx_params(open(os.path.join(dst, entry + ".ptx")).read())
        call_args = describe_args(runtime, t)
        nptr = sum(1 for key in ("global_scratch_size", "profile_scratch_size") if key in meta)
        assert len(params) == len(call_args) + nptr, (
            f"{name}: the compiled kernel has {len(params)} parameters, I expected {len(call_args)} + {nptr} hidden pointers")
        outputs, checks = {}, {}
        for o in OUTPUTS[kname]:
            save_tensor(dst, "expected_" + o, outs[o])
            outputs[o] = dict(file=f"expected_{o}.bin", tensor=ALIAS.get(o, o), reference=REFERENCE[o])
            # Check Triton's output against the dense reference; ref_d is the delta from the Triton forward's o.
            want = ref.get(REFERENCE[o])
            if o == "d":
                want = (primary["_fwd"]["o"].double() * flat(do).double()).sum(-1)
            got = outs[o].double()
            checks[o] = float((got - want).abs().max() / max(want.abs().max(), 1e-30))
        for o, err in checks.items():
            flag = "" if err < 1e-3 else "   <-- MISMATCH: Triton's own output disagrees with the dense reference"
            print(f"   {name}: {o} max error vs the dense float64 reference, relative to its largest entry: {err:.2e}{flag}")
        if is_primary:
            primary[kname] = outs
        instances.append(dict(
            name=name, kernel=kname, entry=entry, dir=name, grid=list(grids[kname](cfg)), config=dict(
                BLOCK_J=cfg[0], BLOCK_K=cfg[1], num_warps=cfg[2], num_stages=cfg[3]),
            shared=meta.get("shared"), num_warps=meta.get("num_warps"), args=call_args, outputs=outputs,
            ptx_params=params, triton_cache_dir=src))
        report.append(dict(name=name, shared=meta.get("shared"), num_warps=meta.get("num_warps"),
                           hidden_pointers=nptr, ptx_parameters=len(params), errors=checks))
        print(f"== {name}: shared={meta.get('shared')} num_warps={meta.get('num_warps')} "
              f"args={len(call_args)} ptx_params={len(params)}")
        return outs

    for kname in ("_fwd", "_bwd_q", "_bwd_kv", "_bwd_b"):
        for i, cfg in enumerate(CONFIGS[kname]):
            outs = run_kernel(kname, cfg, i == 0)
            if i == 0 and kname == "_fwd":
                # The backward kernels read o and the log-sum-exp the forward kernel produced.
                primary["_fwd"] = dict(o=outs["o"], l=outs["lse"], lse=outs["lse"])
                for nm in ("o", "l"):
                    save_tensor(os.path.join(OUT, "data"), nm, primary["_fwd"][nm])
            if i == 0 and kname == "_bwd_q":
                primary["_bwd_q"] = dict(d=outs["d"], dq=outs["dq"])
                save_tensor(os.path.join(OUT, "data"), "d", outs["d"])

    for name, t in ref.items():
        save_tensor(os.path.join(OUT, "data"), name, t)
    save_tensor(os.path.join(OUT, "data"), "ref_d", (primary["_fwd"]["o"].double() * flat(do).double()).sum(-1))

    shapes = {name: dict(shape=list(t.shape), dtype="bool" if t.dtype == torch.bool else "f32")
              for name, t in tensors.items()}
    shapes.update(o=dict(shape=[bh, N, N, D], dtype="f32"), l=dict(shape=[bh, N, N], dtype="f32"),
                  d=dict(shape=[bh, N, N], dtype="f32"))
    for r in ("ref_o", "ref_dq", "ref_dk", "ref_dv"):
        shapes[r] = dict(shape=[bh, N, N, D], dtype="f32")
    for r in ("ref_l", "ref_db", "ref_d"):
        shapes[r] = dict(shape=[bh, N, N], dtype="f32")
    with open(os.path.join(OUT, "case.json"), "w") as f:
        json.dump(dict(bs=BS, h=H, n=N, d=D, bh=bh, tensors=shapes, instances=instances), f, indent=2, default=str)

    print("\n== specialisation probe (_fwd, one config): new binaries compiled per sequence length")
    probe = specialisation_probe(mod._fwd)
    for n, new in probe:
        print(f"   n={n:3d}: {'compiled a new binary' if new else 'reused a binary'}")
    report.append(dict(specialisation_probe=probe))

    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(os.path.join(OUT, "versions.txt"), "w") as f:
        cap = torch.cuda.get_device_capability(0)
        f.write(f"triton {triton.__version__}\ntorch {torch.__version__}\npython {platform.python_version()}\n"
                f"gpu {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}\nkernels {mod.__file__}\n")
    print(f"\nWrote {OUT} (case.json, report.json, versions.txt, data/ and one directory per instance).")


if __name__ == "__main__":
    main()
