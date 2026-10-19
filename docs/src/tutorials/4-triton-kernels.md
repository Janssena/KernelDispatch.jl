# 4. Triton kernels, on Metal and on NVIDIA

[Triton](https://triton-lang.org) compiles a Python kernel to a GPU binary. Compiled ahead of time, that binary is just
one more source for KernelDispatch, registered and launched like the hand-written ones in
[tutorial 2](2-registering-kernels.md). The compiler and what it writes differ by backend, so this tutorial has a part for
each:

| | **A. Metal** | **B. NVIDIA** |
|:--|:--|:--|
| compiler | [triton-msl](https://github.com/bledden/triton-msl), a third-party Metal backend for Triton | Triton itself |
| what it writes | a `.metallib`, plus a descriptor you write | `<name>.cubin`, `.ptx` and `<name>.json`, by Triton |
| turned into a `KernelBinary` by | `KernelBinary(...)` | [`triton_binary`](@ref) |
| `threadgroup` | `compiled.metadata.block_size` | `num_warps * 32`, read from the JSON |
| scalar arguments | one buffer each; the argument list is short (31 buffer slots) | by value; constexprs and integers equal to 1 are dropped from the signature |
| hidden parameters | none (checked: no packed-scalar buffer) | two trailing scratch pointers, passed for you |
| shared memory | declared in the kernel | the JSON's `shared`, raised past 48 KB for you |
| `tile` | what one threadgroup covers | `1`: `extent` is the grid of programs |
| status in this repository | follows trifast's build; not run by the test suite | tested on an NVIDIA A40 with Triton 3.8.0 (see B) |

[trifast](https://github.com/latkins/trifast)'s triangle-attention kernels are the working example for both: on Metal they
reach LuxTriangleAttention through `trifast/src/trifast/metal.py`, and on NVIDIA their original CUDA kernels run through
KernelDispatch unmodified.

What both have in common, from the Julia side:

- **Pointers are the raw buffers of Julia arrays**, which are **column-major**.
- **Scalars must have the types the signature says**: `i32` is an `Int32`, `fp32` a `Float32`.
- **`tl.program_id(axis)`** is the position in the grid KernelDispatch launches: with `tile=1`, one threadgroup per item,
  `tl.program_id(0)` runs over the call's `extent`.
- **`tl.constexpr` arguments are compiled in.** They are not passed at launch, so they belong in the binary's name and in the
  variant key.
- **Pass exactly the arguments the compiled signature declares**, in its order.

## A. Metal: triton-msl to a `.metallib`

[triton-msl](https://github.com/bledden/triton-msl) lowers a `@triton.jit` kernel through Triton's own pipeline to Metal
Shading Language and on to a `.metallib`. `trifast/src/trifast/metal.py` (`build(dim=32)`) is the working reference this part
follows.

> **Status.** Tutorials 1–3 are checked by the example package's test suite. Part A is not: it
> needs a Python toolchain, and the SDPA kernel below follows trifast's build rather than being run as
> part of `SDPAOps`. Treat the Triton kernel as a starting point and test it (`record_bindings`,
> comparison with the host path) before relying on it. Part B is run by the test suite, from committed fixtures.

### The toolchain

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

### Writing the kernel for a Julia caller

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

### Compiling to a `.metallib`

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

### Registering it in Julia

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

### In practice: trifast

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

## B. NVIDIA: a Triton `.cubin`

Triton's own backend produces a `.cubin` (machine code for one architecture) and a `.ptx` (portable text) for NVIDIA, and
writes a metadata JSON beside them (`<name>.json`). [`triton_binary`](@ref) turns that JSON into a
[`KernelBinary`](@ref); nothing needs compiling on the Julia side. The [Backends](../backends.md#Triton-on-CUDA) page
describes how CUDA launches such a binary, eagerly and traced.

### The toolchain

An NVIDIA GPU, Python with `triton` and a CUDA build of `torch` (Triton launches the kernel once to compile it; torch
allocates the arrays for that launch). This part is tested with Triton **3.8.0** on Python 3.10 and an sm_86 GPU (an A40).
Other Triton versions have not been tried and may write different metadata or declare a different number of hidden
parameters.

### Writing the kernel for a Julia caller

Julia arrays are column-major, and a Triton kernel can follow either of two conventions:

- **Write it for Julia's layout**, as in part A: element `(r, c)` of a `rows×cols` matrix is at `c * rows + r`.
- **Keep a PyTorch-style kernel and hand Julia the same memory with the dimensions reversed.** A torch tensor of shape
  `(bh, n, n, d)` and a Julia array of size `(d, n, n, bh)` are the same bytes, so a kernel that takes torch's strides as
  arguments works unchanged. This is how trifast's CUDA kernels run here: the strides are integer arguments, computed
  from the torch shape.

Either way the scalars are `Int32` for Python ints and `Float32` for Python floats. (Triton compiles an integer above
2^31 as 64-bit; that case has not been tested here.) The smallest example, used by the test suite, is:

```python
# scale_kernel.py
import triton
import triton.language as tl

@triton.jit
def scale_kernel(x_ptr, y_ptr, a, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, a * x, mask=mask)
```

### Compiling and collecting the files

Compile ahead, into a directory you control (the snippet is a sketch of what the fixture script does). Launch the kernel once from Python with `TRITON_CACHE_DIR` pointing at a
fresh directory, then copy `<name>.cubin`, `<name>.ptx` and `<name>.json` out of the `<hash>/` directory Triton made in it.
(Triton's default JIT cache holds the same files under opaque hash directories that Triton may rewrite or clear, so it is not
a place to read from at run time.) `test/fixtures/make_triton_fixture.py` does exactly this for a few small kernels.

```python
import os, tempfile, glob, shutil
os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp()          # before importing triton
os.makedirs("kernels", exist_ok=True)
import torch, triton
from scale_kernel import scale_kernel

x, y = torch.randn(100, device="cuda"), torch.zeros(100, device="cuda")
scale_kernel[(triton.cdiv(100, 64),)](x, y, 2.0, 100, BLOCK=64, num_warps=2)
for f in glob.glob(os.path.join(os.environ["TRITON_CACHE_DIR"], "**", "scale_kernel.*"), recursive=True):
    if f.endswith((".cubin", ".ptx", ".json")):
        shutil.copy(f, "kernels/")
```

If the GPU is only available inside a job, see [Building without a GPU](../backends.md#Building-without-a-GPU): the files can
be produced elsewhere and registered on the GPU node.

A binary is one *configuration*: the block sizes, `num_warps` and `num_stages` it was compiled with. For a kernel wrapped
in an autotuner (trifast's), launch the `JITFunction` underneath with an explicit configuration, once per configuration you
want to ship; `test/fixtures/make_trifast_fixture.py` does that for trifast's four kernels.

### Registering it

```julia
bin = triton_binary("kernels/scale_kernel.json")        # scale_kernel.cubin beside it; PTX if there is no cubin
add_variant!(op, kernel, (:Float32,); fwd=bin)
```

It reads the kernel's `entry`, `threadgroup` (`num_warps * 32`), `shared_mem` (the dynamic shared memory Triton asked for,
raised past the default 48 KB for you), the architecture a cubin was built for (checked against the device on every launch)
and how many hidden trailing pointer parameters the compiled kernel takes (`abi = :triton` makes the launcher pass them as
null pointers, eagerly and traced). `tile` defaults to `1`: `extent` is the grid of programs.

### Calling it

The arguments are the compiled signature, in order: `x_ptr`, `y_ptr`, `a`, `n` (`BLOCK` is a constexpr and is dropped):

```julia
y, = run_binary(bin, x, OutArray(Float32, 100), 2.0f0, Int32(100); extent=cld(100, 64))    # eager, on CuArrays
```

and the same call, with traced arrays, under `@jit`. From an op, `forward` calls `call_binary(op, kernel, :fwd, args...;
extent=...)` instead, and nothing about it is specific to Triton. The buffer order is the kernel's own: a kernel with its
output first or between other arguments works, because a traced launch passes each output as an aliased operand in its
position.

!!! note "Calling `run_binary` from inside a trace"
    A `KernelBinary` captured by a traced function is not traceable. Build it inside the function (from its file and
    fields) or look it up with `call_binary`, which does that inside the trace.

### What is the caller's job

The compiled signature is what counts, and Triton changes it:

- **Pass exactly the arguments the compiled kernel declares.** Triton removes `constexpr` arguments from the signature and
  integers equal to 1 (a stride of a contiguous array), so the `forward` must not bind them. For trifast's `_fwd` with
  contiguous inputs, four constexprs and seven strides equal to 1 are dropped from its forty Python parameters, leaving 29. `test/triton_fixtures.jl` has `ptx_param_kinds`, which lists the parameters of
  the PTX entry to compare with. Nothing checks this at launch yet.
- **A binary is valid only for the specialisation it was compiled for.** Triton also compiles integers and pointers
  divisible by 16 in as such. For trifast's `_fwd`, sequence lengths 33, 40 and 41 shared one binary and 48 and 64
  another. Register one binary per class, and put the class in the variant key; a launch outside it is not checked and can
  compute wrongly.
- **No clusters or scratch memory.** `triton_binary` refuses `num_ctas > 1` and a nonzero `global_scratch_size`, with the
  reason.

### Kernels that update an array

A kernel that reads and then updates an output (`y += a * x`) takes that array as an [`InOutArray`](@ref):

```julia
y = run_binary(bin, InOutArray(y), x, 0.5f0, Int32(100); extent=cld(100, 64)) |> only
```

Eagerly the kernel writes `y` itself; in a trace the launch starts from `y`'s values and returns the updated array, and `y`
itself is unchanged. A plain [`OutArray`](@ref) starts from zeros (also in a trace), which is what a kernel that fills its
output, or accumulates atomically into a zeroed one, expects. `InOutArray` is not available in a Metal trace.

### In practice: trifast on CUDA

trifast's four triangle-attention kernels (`_fwd`, and the three backward kernels `_bwd_q`, `_bwd_kv`, `_bwd_b`) are the
largest test here. They use `tl.dot`, a three-dimensional grid `(cld(n, BLOCK_J), n, batch·heads)`, between 29 and 40
arguments each (a dozen strides per kernel), a Bool mask, outputs first or last, and dynamic shared memory up to 57 KB. The
test suite launches each at several configurations, cubin and PTX, eagerly and from `@jit`, and compares them with Triton's
own output and a float64 torch reference; it also chains the four (the backward kernels read what the forward and `_bwd_q`
wrote) in one traced program. All of that is in `test/ext/cuda/trifast.test.jl`, driven by a description of each call that
`make_trifast_fixture.py` writes (`test/fixtures/trifast/case.json`: the grid and the compiled argument list).

### What is and is not validated

Tested on an NVIDIA A40 with Triton 3.8.0: small kernels (shared memory, a dropped argument, an output first), a
read-modify-write and an atomic accumulation, and trifast's four kernels (Float32), all eager and traced. Not covered:
other Triton versions and GPUs, Float16/BFloat16, differentiation through the kernels, autotuning, and performance.
