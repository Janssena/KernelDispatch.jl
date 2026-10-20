#!/usr/bin/env python3
"""Generate the Triton fixtures for KernelDispatch's tests: real kernels compiled by real Triton.

Run on a machine with an NVIDIA GPU, Triton and a CUDA build of torch (torch is only used to launch the kernels once
and to produce Triton's own reference outputs):

    python3 test/fixtures/make_triton_fixture.py              # float32 -> test/fixtures/triton/
    python3 test/fixtures/make_triton_fixture.py float16      # -> test/fixtures/triton_float16/
    python3 test/fixtures/make_triton_fixture.py bfloat16     # -> test/fixtures/triton_bfloat16/   (needs sm_80 or newer)

The half-precision runs make the tensors in that dtype and keep the scalars `fp32` (Python floats), exactly what a Julia
`forward` binds. The binaries differ per dtype, so each has its own directory. The atomic kernel is float32 only (atomic adds
on half types depend on the architecture, and this package does not test CUDA's atomics).

For each kernel it
  1. compiles and launches it through Triton with a fresh TRITON_CACHE_DIR (so the files are exactly what Triton writes),
  2. copies `<name>.cubin`, `<name>.ptx` and `<name>.json` into test/fixtures/triton/<name>/,
  3. saves the inputs and Triton's own output as raw little-endian words of the dtype (`x.bin`, `y.bin`, `expected.bin`, ...:
     float32, or the 16-bit patterns of float16 / bfloat16, which NumPy cannot hold),
  4. prints (and records in test/fixtures/triton/manifest.json) the facts the Julia side needs: Triton's metadata, and
     the PTX entry's real parameter list, which shows which Python arguments Triton dropped (constexpr, integers equal to
     1) and how many hidden trailing parameters it added.

Nothing here is needed to RUN the Julia tests: the generated files are committed, together with versions.txt. Paste the
printed report back when something looks off. The kernels are small on purpose:

  scale_kernel      y = a * x            inputs before the output; constexpr BLOCK is dropped from the signature
  block_sum_kernel  out[pid] = sum(...)  a cross-warp reduction: uses shared memory (metadata `shared` > 0)
  strided_axpy      out = a*x + y        OUTPUT FIRST, strides after every pointer, one stride equal to 1 (specialised away)
  accumulate_kernel y += a*x             READS its output before updating it (needs the output's old values)
  atomic_sum_kernel out[0] += sum(block) every program atomically adds into ONE zero-initialised slot
"""
import glob
import json
import os
import platform
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DTYPE = sys.argv[1] if len(sys.argv) > 1 else "float32"
if DTYPE not in ("float32", "float16", "bfloat16"):
    sys.exit(f"dtype must be float32, float16 or bfloat16, got {DTYPE!r}")
OUT = os.path.join(HERE, "triton" if DTYPE == "float32" else f"triton_{DTYPE}")
# Before Triton is imported: its cache is where the artifacts are read from, and a fresh one has only ours.
CACHE = tempfile.mkdtemp(prefix="kd_triton_cache_")
os.environ["TRITON_CACHE_DIR"] = CACHE

import numpy as np  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

if not torch.cuda.is_available():
    sys.exit("torch cannot see a CUDA device: this script needs the NVIDIA GPU")

DT = dict(float32=torch.float32, float16=torch.float16, bfloat16=torch.bfloat16)[DTYPE]
if DTYPE == "bfloat16" and torch.cuda.get_device_capability(0)[0] < 8:
    sys.exit("bfloat16 needs an sm_80 or newer GPU")
# Tolerances of the script's own sanity checks against a float64 host formula (the Julia tests have their own).
RTOL, ATOL = (1e-6, 1e-6) if DTYPE == "float32" else (2e-2, 2e-2)


@triton.jit
def scale_kernel(x_ptr, y_ptr, a, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, a * x, mask=mask)


@triton.jit
def block_sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
    tl.store(out_ptr + pid, tl.sum(x, axis=0))


@triton.jit
def strided_axpy(out_ptr, stride_o, x_ptr, stride_x, y_ptr, stride_y, a, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(x_ptr + i * stride_x, mask=m)
    y = tl.load(y_ptr + i * stride_y, mask=m)
    tl.store(out_ptr + i * stride_o, a * x + y, mask=m)


@triton.jit
def accumulate_kernel(y_ptr, x_ptr, a, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    y = tl.load(y_ptr + offs, mask=m)
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(y_ptr + offs, y + a * x, mask=m)


@triton.jit
def atomic_sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
    tl.atomic_add(out_ptr, tl.sum(x, axis=0))


def to_t(x):
    """A float32 NumPy array as a device tensor of the fixture's dtype (rounded to it)."""
    return torch.from_numpy(x).to("cuda").to(DT)


def back(t):
    """A tensor of the fixture's dtype as float32 NumPy: exact, every 16-bit float is a float32."""
    return t.float().cpu().numpy()


def save(directory, **tensors):
    """Raw little-endian words of the dtype: float32, or the 16-bit patterns (NumPy has no bfloat16)."""
    for name, t in tensors.items():
        t = t.detach().contiguous().cpu()
        arr = t.numpy() if DTYPE == "float32" else t.view(torch.int16).numpy()
        arr.tofile(os.path.join(directory, name + ".bin"))


def ptx_params(ptx):
    """The `.param` declarations of the first `.entry`: the compiled kernel's real parameter list."""
    start = ptx.find(".entry")
    end = ptx.find("{", start)
    return [l.strip().rstrip(",") for l in ptx[start:end].splitlines() if ".param" in l]


def collect(kernel, extra):
    """Copy what Triton wrote for `kernel` into OUT/<name>/ and describe it."""
    name = kernel.__name__
    found = glob.glob(os.path.join(CACHE, "**", name + ".json"), recursive=True)
    assert len(found) == 1, f"expected one {name}.json in Triton's cache, found {found}"
    src = os.path.dirname(found[0])
    dst = os.path.join(OUT, name)
    os.makedirs(dst, exist_ok=True)
    for ext in ("cubin", "ptx", "json"):
        shutil.copy(os.path.join(src, f"{name}.{ext}"), os.path.join(dst, f"{name}.{ext}"))
    meta = json.load(open(os.path.join(dst, name + ".json")))
    ptx = open(os.path.join(dst, name + ".ptx")).read()
    params = ptx_params(ptx)
    info = dict(name=name, directory=name, triton_cache_dir=src, ptx_params=params,
                metadata={k: meta[k] for k in sorted(meta) if k not in ("hash", "extern_libs", "supported_fp8_dtypes",
                                                                         "deprecated_fp8_dtypes", "allowed_dot_input_precisions")},
                **extra)
    print(f"\n== {name}")
    print(f"   files copied from {src}")
    print(f"   metadata: {json.dumps(info['metadata'], indent=2, default=str)}")
    print(f"   PTX parameters ({len(params)}):")
    for p in params:
        print(f"      {p}")
    return info


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    dev = "cuda"
    report = []

    # --- scale_kernel -----------------------------------------------------------------------------------------
    n, a, block = 100, 2.0, 64
    xt = to_t(rng.standard_normal(n).astype(np.float32))
    yt = torch.zeros(n, device=dev, dtype=DT)
    grid = (triton.cdiv(n, block),)
    scale_kernel[grid](xt, yt, a, n, BLOCK=block, num_warps=2)
    torch.cuda.synchronize()
    xr = back(xt)
    assert np.allclose(back(yt), a * xr.astype(np.float64), rtol=RTOL, atol=ATOL), "scale_kernel disagrees with the host formula"
    info = collect(scale_kernel, dict(grid=list(grid), num_warps=2, python_args="x_ptr, y_ptr, a=2.0, n=100 (BLOCK=64 constexpr)",
                                      scalars=dict(a=a, n=n)))
    save(os.path.join(OUT, "scale_kernel"), x=xt, expected=yt)
    report.append(info)

    # --- block_sum_kernel -------------------------------------------------------------------------------------
    n, block = 1000, 128
    xt = to_t(rng.standard_normal(n).astype(np.float32))
    nblocks = triton.cdiv(n, block)
    ot = torch.zeros(nblocks, device=dev, dtype=DT)
    block_sum_kernel[(nblocks,)](xt, ot, n, BLOCK=block, num_warps=4)
    torch.cuda.synchronize()
    xr = back(xt).astype(np.float64)
    ref = np.array([xr[i * block:(i + 1) * block].sum() for i in range(nblocks)])
    assert np.allclose(back(ot), ref, rtol=max(RTOL, 1e-4), atol=max(ATOL, 1e-4)), "block_sum_kernel disagrees with the host sum"
    info = collect(block_sum_kernel, dict(grid=[nblocks], num_warps=4, python_args="x_ptr, out_ptr, n=1000 (BLOCK=128 constexpr)",
                                          scalars=dict(n=n)))
    save(os.path.join(OUT, "block_sum_kernel"), x=xt, expected=ot)
    report.append(info)

    # --- strided_axpy: output first, strides after each pointer, stride_x == 1 is specialised away --------------
    n, a, block = 50, 3.0, 32
    so, sx, sy = 3, 1, 2
    xt = to_t(rng.standard_normal(n * sx).astype(np.float32))
    yt = to_t(rng.standard_normal(n * sy).astype(np.float32))
    ot = torch.zeros(n * so, device=dev, dtype=DT)
    grid = (triton.cdiv(n, block),)
    strided_axpy[grid](ot, so, xt, sx, yt, sy, a, n, BLOCK=block, num_warps=1)
    torch.cuda.synchronize()
    xr, yr = back(xt).astype(np.float64), back(yt).astype(np.float64)
    ref = np.zeros(n * so)
    ref[::so] = a * xr[::sx] + yr[::sy]
    assert np.allclose(back(ot), ref, rtol=RTOL, atol=ATOL), "strided_axpy disagrees with the host formula"
    info = collect(strided_axpy, dict(grid=list(grid), num_warps=1,
                                      python_args="out_ptr, stride_o=3, x_ptr, stride_x=1, y_ptr, stride_y=2, a=3.0, n=50 (BLOCK=32 constexpr)",
                                      scalars=dict(a=a, n=n, stride_o=so, stride_x=sx, stride_y=sy)))
    save(os.path.join(OUT, "strided_axpy"), x=xt, y=yt, expected=ot)
    report.append(info)

    # --- accumulate_kernel: the output is also an input (y += a*x) ----------------------------------------------
    n, a, block = 100, 0.5, 64
    xt = to_t(rng.standard_normal(n).astype(np.float32))
    yt = to_t(rng.standard_normal(n).astype(np.float32))
    y_init = yt.clone()                          # the kernel updates yt in place
    grid = (triton.cdiv(n, block),)
    accumulate_kernel[grid](yt, xt, a, n, BLOCK=block, num_warps=2)
    torch.cuda.synchronize()
    ref = back(y_init).astype(np.float64) + a * back(xt).astype(np.float64)
    assert np.allclose(back(yt), ref, rtol=RTOL, atol=ATOL), "accumulate_kernel disagrees with the host formula"
    info = collect(accumulate_kernel, dict(grid=list(grid), num_warps=2, python_args="y_ptr (in/out), x_ptr, a=0.5, n=100 (BLOCK=64 constexpr)",
                                           scalars=dict(a=a, n=n)))
    save(os.path.join(OUT, "accumulate_kernel"), x=xt, y_init=y_init, expected=yt)
    report.append(info)

    # --- atomic_sum_kernel: many programs atomically add into out[0] (which starts at zero) --------------------
    # float32 only: atomic adds on half types depend on the architecture.
    if DTYPE == "float32":
        n, block = 1000, 128
        xt = to_t(rng.standard_normal(n).astype(np.float32))
        nblocks = triton.cdiv(n, block)
        ot = torch.zeros(1, device=dev, dtype=DT)
        atomic_sum_kernel[(nblocks,)](xt, ot, n, BLOCK=block, num_warps=4)
        torch.cuda.synchronize()
        assert np.allclose(back(ot), back(xt).sum(dtype=np.float64), rtol=1e-4, atol=1e-4), "atomic_sum_kernel disagrees with the host sum"
        info = collect(atomic_sum_kernel, dict(grid=[nblocks], num_warps=4, python_args="x_ptr, out_ptr (atomic), n=1000 (BLOCK=128 constexpr)",
                                               scalars=dict(n=n)))
        save(os.path.join(OUT, "atomic_sum_kernel"), x=xt, expected=ot)
        report.append(info)

    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(os.path.join(OUT, "versions.txt"), "w") as f:
        f.write(f"triton {triton.__version__}\ntorch {torch.__version__}\npython {platform.python_version()}\ndtype {DTYPE}\n"
                f"gpu {torch.cuda.get_device_name(0)} sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}\n")
    print(f"\nWrote {OUT} (manifest.json, versions.txt and one directory per kernel). Triton's cache was {CACHE}.")


if __name__ == "__main__":
    main()
