# 4. Compiling a Triton kernel to a `.metallib` with triton-msl

[triton-msl](https://github.com/bledden/triton-msl) is a Metal backend for Triton: it lowers a
`@triton.jit` kernel through Triton's own pipeline to Metal Shading Language and on to a
`.metallib`. Compiled ahead of time, that file is just one more binary for KernelDispatch, registered exactly
like the hand-written one in [tutorial 2](2-registering-kernels.md).

This is for example how [trifast](https://github.com/latkins/trifast)'s triangle-attention kernels reach
LuxTriangleAttention. `trifast/src/trifast/metal.py` (`build(dim=32)`) is the working reference this
tutorial follows.

> **Status.** Tutorials 1–3 are checked by the example package's test suite. This one is not: it
> needs a Python toolchain, and the SDPA kernel below follows trifast's build rather than being run as
> part of `SDPAOps`. Treat the Triton kernel as a starting point and test it (`record_bindings`,
> comparison with the host path) before relying on it.

## The toolchain

- An Apple-silicon Mac, macOS 14 or later, with the Xcode command-line tools. On macOS 26 the Metal
  compiler is a separate component: if `xcrun metal --version` reports it missing, run
  `sudo xcodebuild -downloadComponent MetalToolchain` once.
- Triton **3.7.0**, which has no macOS wheel on PyPI. Either build it from source (about 12 minutes)
  or use the unofficial Python 3.14 wheel triton-msl attaches to its GitHub releases.
- triton-msl itself, from PyPI or a checkout.
- torch is **not** needed to compile ahead of time (see the `get_module_map` line below).

With [uv](https://docs.astral.sh/uv/), in a directory of your choice:

```bash
uv venv --python 3.14 .venv && source .venv/bin/activate
uv pip install "https://github.com/bledden/triton-msl/releases/download/triton-wheel-3.7.0-cp314-macos-arm64/triton-3.7.0+git4da2e268-cp314-cp314-macosx_15_0_arm64.whl"
uv pip install triton-msl            # or: uv pip install -e path/to/triton-msl
```

## Writing the kernel for a Julia caller

A Triton kernel compiled for KernelDispatch is called with Julia's arrays, so it has to follow the
conventions of the binary it becomes.

- **Pointers are raw buffers of Julia arrays**, which are **column-major**: element `(r, c)` of a
  `rows×cols` matrix is at `c * rows + r` (0-based). Index them that way, not with PyTorch's
  row-major strides.
- **Scalars must have the types the signature says**: `i32` ↔ `Int32`, `fp32` ↔ `Float32`.
- **`tl.program_id(axis)`** is the threadgroup's position in the grid KernelDispatch dispatches. With
  `tile=1` on the binary, one threadgroup per item, `tl.program_id(0)` runs over the call's `extent`.
- **`tl.constexpr` arguments are compiled in.** They are not passed at launch, so they belong in the
  binary's name and in the variant key.
- **Keep the argument list short.** Metal has 31 buffer slots and every argument, scalars included,
  takes one. trifast derives its strides from the sizes instead of passing ~20 of them.

Here is SDPA in the layout of the example package (`q: d×n`, `k: d×m`, `v: dv×m`, `o: dv×n`), with
the same argument order as the hand-written Metal kernel, so it can share that kernel's `forward`. One
program handles one query; keys are streamed in blocks with an online softmax:

```python
# sdpa_triton.py
import triton
import triton.language as tl

@triton.jit
def sdpa_fwd(q_ptr, k_ptr, v_ptr, o_ptr, N, M, D, DV, scale,
             BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_DV: tl.constexpr):
    i = tl.program_id(0)                              # this program's query (column of q and o)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    q = tl.load(q_ptr + i * D + offs_d, mask=offs_d < D, other=0.0)

    m_i = -float("inf")                               # running max, denominator, weighted sum
    l_i = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
    for j0 in range(0, M, BLOCK_M):
        offs_m = j0 + tl.arange(0, BLOCK_M)
        live = offs_m < M
        # k[:, offs_m] as a [BLOCK_D, BLOCK_M] tile: column j starts at j * D.
        k = tl.load(k_ptr + offs_m[None, :] * D + offs_d[:, None],
                    mask=(offs_d[:, None] < D) & live[None, :], other=0.0)
        s = tl.sum(q[:, None] * k, axis=0) * scale
        s = tl.where(live, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)
        v = tl.load(v_ptr + offs_m[None, :] * DV + offs_dv[:, None],
                    mask=(offs_dv[:, None] < DV) & live[None, :], other=0.0)
        acc = acc * alpha + tl.sum(p[None, :] * v, axis=1)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
    tl.store(o_ptr + i * DV + offs_dv, acc / l_i, mask=offs_dv < DV)
```

It avoids `tl.dot`. triton-msl requires every dimension of a `tl.dot` tile to be at least 32, too
large for this example's head dims; a production kernel with larger tiles would use it.

## Compiling to a `.metallib`

Ahead of time, through Triton's compiler with triton-msl's target, as `trifast.metal.build` does:

```python
# build_sdpa.py
import json
from pathlib import Path

from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, compile as triton_compile
from triton_msl.backend.compiler import MetalBackend

from sdpa_triton import sdpa_fwd

# The libdevice module map lives behind torch; a kernel without extern math doesn't need it.
MetalBackend.get_module_map = lambda self: {}

signature = {"q_ptr": "*fp32", "k_ptr": "*fp32", "v_ptr": "*fp32", "o_ptr": "*fp32",
             "N": "i32", "M": "i32", "D": "i32", "DV": "i32", "scale": "fp32",
             "BLOCK_D": "constexpr", "BLOCK_M": "constexpr", "BLOCK_DV": "constexpr"}
constexprs = {"BLOCK_D": 16, "BLOCK_M": 64, "BLOCK_DV": 8}

compiled = triton_compile(ASTSource(fn=sdpa_fwd, signature=signature, constexprs=constexprs),
                          target=GPUTarget("mps", 0, 32))

out = Path("sdpa_triton_d16_m64_dv8")
out.mkdir(exist_ok=True)
(out / "sdpa_fwd.metallib").write_bytes(compiled.asm["metallib"])

msl = compiled.asm["msl"]
assert "_bpk" not in msl, "packed-scalar ABI: KernelDispatch binds one buffer per argument"
# The file is named after the Triton function, so its entry needs no recording: KernelDispatch defaults
# `entry` to the file name.
(out / "sdpa_fwd.json").write_text(json.dumps({
    "threadgroup": compiled.metadata.block_size,   # NOT num_warps * 32
    "constexprs": constexprs,
}, indent=2))
```

What to take from each line:

- **`signature`** lists every argument in the kernel's order: `*fp32` for a pointer, `i32`/`fp32`
  for a scalar, `constexpr` for a compiled-in value, whose values go in `constexprs`. Keep it in
  Triton's parameter order: trifast builds it from the kernel's own `fn.params` to be sure.
  If the kernel is wrapped in `@autotune`, pass the bare `JITFunction` underneath.
- **`GPUTarget("mps", 0, 32)`** selects triton-msl's backend.
- **`compiled.asm["metallib"]`** is the binary. `compiled.asm["msl"]` is the generated Metal source,
  worth keeping for inspection.
- **The threadgroup is `compiled.metadata.block_size`.** triton-msl derives it from the kernel's tile
  shape, the product of its largest `tl.arange` extents (for this kernel that should be
  `16 × 64 = 1024`; use the reported value, not your own arithmetic).
  `metadata.num_warps` is unrelated: a fixed 4. Dispatching with `num_warps * 32` launches too few
  threads, and most of the tile silently goes unwritten.
- **Check the binding ABI.** Most kernels are lowered generically, and every argument, scalars
  included, gets its own buffer in order: what KernelDispatch binds. A kernel that routes to one of
  triton-msl's specialised templates instead takes its pointers first and **packs every scalar into
  one trailing `constant uint* _bpk` buffer**. Binding that positionally doesn't error: a scalar
  lands where the kernel reads a loop bound, which can read as garbage and **hang the GPU**. Refuse
  it at build time, as above, and re-check after every triton-msl upgrade. trifast's
  `_buffer_binding` parses the emitted signature in full.
- **1024 threads is the ceiling.** Metal caps a threadgroup at 1024 threads. A kernel whose largest
  tile exceeds that is lowered through a multi-element-per-thread loop, which trifast found
  mis-compiles some constructs. trifast refuses to build past it by default.

## Registering it in Julia

The result is a binary like any other. Here as a third kernel of `SDPAOps`' op, sharing the Metal
kernel's argument order:

```julia
struct SDPATriton <: AbstractKernel end
kernelname(::SDPATriton) = :triton                  # optional, like the example's other names

function forward(op::SDPA, kern::SDPATriton, q, k, v)
    d, n, m, dv = _sizes(q, k, v)
    return only(call_binary(op, kern, :fwd, q, k, v, OutArray(Float32, dv, n), n, m, d, dv, _scale(q);
        extent=n))
end
# The key carries what the binary was compiled for: d ≤ 16, dv ≤ 8.
variant_key(::SDPA, ::SDPATriton, q, k, v, o, n, m, d, dv, scale) =
    (nameof(KernelDispatch.realtype(eltype(q))), d <= 16 && dv <= 8)

const TRITON_DIR = joinpath(@__DIR__, "..", "kernels", "sdpa_triton_d16_m64_dv8")

function build!(op::SDPA, kern::SDPATriton, key::Tuple)
    key == (:Float32, true) || return nothing          # Float32, and head dims within the build
    meta = JSON.parsefile(joinpath(TRITON_DIR, "sdpa_fwd.json"))
    add_variant!(op, kern, key; fwd=KernelBinary(joinpath(TRITON_DIR, "sdpa_fwd.metallib");
        threadgroup=meta["threadgroup"], tile=1))   # entry `sdpa_fwd`, from the file; one program per query
    return nothing
end

# select with `use_kernel!(SDPA(), SDPATriton())`, which registers it
```

- **`entry`** is the Triton function's name. Naming the file after it (`sdpa_fwd.metallib`) lets the
  default, the file name, supply it. **`threadgroup`** is the `block_size` the build recorded, never a
  constant written by hand.
- **`tile=1`** makes one threadgroup per item, so `extent=n` launches `n` programs and
  `tl.program_id(0)` runs over `0:n-1`. The `forward` is the hand-written kernel's, unchanged: only the
  binaries' `tile`s differ.
- **The variant key** separates what the binary can serve from what it can't. Calls with larger head
  dims find no variant and fail loudly instead of reading past the tile. Add `nearest` or more builds
  to cover them.
- **`JSON`** stands for whatever reads the sidecar. Writing `threadgroup` into the Julia code works
  too, as long as it is re-checked when the binary is rebuilt.
- Under Reactant with jax-mps, the same binary is embedded as a custom call with no further work.
  Scalars are then passed as bytes rather than buffers, which the generated `constant` parameters read
  the same way.

Then check the bindings against the Triton signature, and the output against the host path:

```julia
julia> use_kernel!(:sdpa, :triton)

julia> map(first, only(KernelDispatch.record_bindings(() -> SDPA()(q, k, v))).slots)
(:in, :in, :in, :out, :scalar, :scalar, :scalar, :scalar, :scalar)   # q k v o N M D DV scale

julia> Array(SDPA()(q, k, v)) ≈ SDPA()(Array(q), Array(k), Array(v))
```

## In practice: trifast

`trifast.metal.build(dim=32)` applies all of this to four kernels (a forward and a three-way
backward), and writes the metallibs plus a JSON descriptor with each kernel's threadgroup, buffer
table and grid formula. They register as one kernel of LuxTriangleAttention's `SDPA` op, with four
launch names in one variant. Each threadgroup of 1024 threads covers a block of 32 rows, or a 32×32
block for `_bwd_b`, and that is what each binary's `tile` records:

```julia
bin(s; tile=(32, 1, 1)) = KernelBinary(joinpath(dir, "$s.metallib"); threadgroup=1024, tile)
add_variant!(SDPA(), Trifast(), (:Float32, 32);
    fwd=bin("_fwd"), dq=bin("_bwd_q"), dkv=bin("_bwd_kv"), dbias=bin("_bwd_b"; tile=(32, 32, 1)))
```

Every launch then passes the same `extent=(n, n, H * B)`, and gets trifast's grids:
`(n/32, n, H·B)` and, for `:dbias`, `(n/32, n/32, H·B)`.

Its `forward` launches `:fwd`, and its `backward` launches `:dq`, `:dkv` and `:dbias` in turn (see
[tutorial 3](3-forward-backward-enzyme.md) for the backward contract). The kernels' own source,
`trifast/src/trifast/triton-metal.py`, documents the changes a CUDA Triton kernel needed for this:
reading Julia's column-major arrays in place, deriving strides instead of passing them, and taking its
arguments in the order the Julia side binds them.
