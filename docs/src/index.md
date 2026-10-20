# KernelDispatch.jl

```@meta
CurrentModule = KernelDispatch
```

Simplify working with compiled kernel binaries built anywhere, directly in Julia!

- Define an op
- Register kernels for it
- Select the default kernel and run the op on CPU and device arrays simply through multiple dispatch
- Kernels can even run embedded in a Reactant-compiled program as a `stablehlo.custom_call`

An operation can have several **kernels** (implementations) you switch between at runtime, each with
**tuned variants** picked from the inputs. Pure-Julia
[KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl) kernels are also supported as one
more source of binaries: [`ka_compile`](@ref) turns one into a binary, and from there it is launched like
any other.

The [tutorials](tutorials/1-switching-kernels.md) walk through a complete example: switching kernels and
persisting defaults, registering a compiled binary or a KA kernel, custom `forward`/`backward` with Enzyme
gradients, and compiling a Triton kernel to a `.metallib` or a `.cubin`. Every exported and public name is documented in
the [API reference](api.md).

**Backends.** Metal runs prebuilt `.metallib` binaries and KernelAbstractions kernels (compiled to `.metallib`); CUDA
runs prebuilt `.cubin` / `.ptx` files, KernelAbstractions kernels (compiled to `.cubin`) and kernels compiled by Triton
(`triton_binary`), once `CUDA.jl` is loaded. A prebuilt `.metallib` is Metal-only, so an op that ships one (like the
`FlashAttn` kernel of the examples) needs a KA kernel to run on CUDA. Under Reactant a binary is embedded as a
`stablehlo.custom_call`: on Metal through jax-mps, on CUDA through XLA's `triton_kernel_call_ffi` handler. CUDA's KA
binaries are built without CUDA.jl's kernel state, so the same file launches eagerly and from a trace; the price is
that such a kernel must not be able to throw. A kernel's buffers may come in any order, and a kernel that updates an
array in place takes it as an [`InOutArray`](@ref).

**Precisions.** `Float32`, `Float16` and `BFloat16` are all supported, eagerly and from a Reactant trace, on Metal and on
CUDA: buffers of any of the three, outputs allocated in their own type (`OutArray(Float16, n)`), and launches that mix
them (a `BFloat16` output next to a `Float32` one, half-precision buffers with a `Float32` scalar). A binary is chosen by the
element type through the variant key, so one op can ship a binary per precision, and what a kernel accumulates in is its
own choice (the example `SDPAOps` reads and writes `T` and sums in `Float32`). A binary carries the types it was compiled
for, and a launch with other types is refused instead of computing garbage (see *Argument types are checked* below).
`Float64` is not a goal. Kernels compiled by Triton are the exception to "all three": the Triton fixtures are `float32`, so
Triton in half precision has not been tested. A 2-byte scalar cannot be passed to XLA's handler on CUDA, so it is refused
in a trace (pass a `Float32` scalar, as the half-precision kernels here do).

**Argument types are checked.** A binary takes raw pointers and bits scalars, so one compiled for `float*` and handed a `Float16`
array would run and compute something else. `ka_compile` records the signature of the kernel it builds, and a launch whose
types differ is refused with the argument named (`check_types=false` skips the check). For a prebuilt binary, say what
it takes with `KernelBinary(...; signature=(Vector{Float32}, Float32, Int32))`; without one the first launch warns once
and prints the `signature=` that fits (`signature=false` silences that). Which types, in which order, and what is not
checked: [Signatures](signatures.md).

**What is tested.** Metal (eager and traced) on Apple silicon, in `Float32`, `Float16` and `BFloat16`: prebuilt,
KernelAbstractions and `SDPAOps` kernels, gradients through Enzyme, mixed-type outputs. CUDA on one NVIDIA A40 (sm_86): prebuilt,
KernelAbstractions and Triton binaries, eager and traced, in `Float32`, and `Float16` and `BFloat16` for prebuilt PTX and cubin,
KernelAbstractions kernels, `SDPAOps` and gradients, eager and traced. Kernels compiled by Triton are tested with Triton 3.2.0 and 3.8.0
(small test kernels, which differ in their hidden trailing pointers) in `Float32`, and with 3.2.0 in `Float16` and `BFloat16` too;
[trifast](https://github.com/latkins/trifast)'s four triangle-attention kernels (3.2.0, `Float32`) are checked against torch. ROCm is not implemented. A Triton
binary is compiled for the specialisation of the arguments it was first launched with (integers equal to 1 or divisible
by 16), so a registered one serves only launches in the same class. See the [Backends](backends.md) page for the
matrix and the limits.

## Concepts

| | what | you write |
|:--|:--|:--|
| **op** | an operation: `SDPA <: AbstractKernelOp` | [`host`](@ref), [`@default_kernel`](@ref) |
| **kernel** | one implementation: `Trifast <: AbstractKernel` | [`forward`](@ref) (builds the binary's arguments), optionally [`backward`](@ref), [`variant_key`](@ref), [`extras`](@ref), [`nearest`](@ref), [`build!`](@ref), [`source_tag`](@ref) |
| **variant** | one (tuned) build of a kernel for a class of inputs: launch name → `KernelBinary` | [`add_variant!`](@ref) |
| **[`KernelBinary`](@ref)** | one compiled launch: file, entry point, `threadgroup`, `tile`, tuning `params`, `abi`, `target`, `shared_mem`, `signature` | — ([`ka_compile`](@ref) and [`triton_binary`](@ref) make them) |

Ops and kernels are fieldless singletons; behaviour comes from dispatch. Which kernel is *selected* is a
method, so an op call is type-stable.

## Example

Registering a single kernel for a scaled dot-product attention operation (`q: d×n`, `k: d×m`,
`v: dv×m` → `o: dv×n`):

```julia
import KernelDispatch: host, forward

using KernelDispatch

# Define your operation
struct SDPA <: AbstractKernelOp end

# CPU default:
softmax(s) = (p = exp.(s .- maximum(s; dims=1)); p ./ sum(p; dims=1))
host(::SDPA, q, k, v) = v * softmax((k' * q) ./ sqrt(Float32(size(q, 1))))

# Define our kernel:
struct FlashAttn <: AbstractKernel end

# For package developers
KernelDispatch.@default_kernel SDPA FlashAttn()

add_variant!(
  SDPA(), FlashAttn(); fwd = KernelBinary("flash.metallib"; threadgroup=64)
)

# Every kernel needs a `forward`: it builds the binary's arguments from the op's and launches it.
function forward(op::SDPA, kernel::FlashAttn, q, k, v)
    (d, n), m, dv = Int32.(size(q)), Int32(size(k, 2)), Int32(size(v, 1))
    o = OutArray(Float32, dv, n) # Provide the shapes of the output
    scale = inv(sqrt(Float32(d)))
    # `extent`: the problem size (one query each); KernelDispatch divides it by the binary's `tile`.
    # :fwd should match the kwarg that is used in add_variant!
    result = call_binary(op, kernel, :fwd, q, k, v, o, n, m, d, dv, scale; extent=n) # Returns a Tuple
    return only(result)                                       # of the OutArrays: here, just `o`
end
```

Calling the op picks the path from its arguments, and switching kernels is one line:

```julia
q, k, v = (randn(Float32, 16, 100), randn(Float32, 16, 70), randn(Float32, 8, 70)) .|> MtlArray

o = SDPA()(q, k, v)                                       # MtlArray: the default kernel, FlashAttn
SDPA()(Array(q), Array(k), Array(v))                      # Array: `host` · traced (jax-mps): a custom call
use_kernel!(SDPA(), SDPAKA())                             # this session, from the next top-level statement
SDPA()(q, k, v)                                           # runs the KA kernel
forward(SDPA(), FlashAttn(), q, k, v)                     # one call with a given kernel: no global switch
```

The example is Metal; on CUDA only the binary and the array type change. A `.cubin` (built for one architecture, which
`target` records and the launch checks) or a portable `.ptx` takes the place of the `.metallib`, and the arrays are
`CuArray`s:

```julia
add_variant!(SDPA(), FlashAttn(); fwd = KernelBinary("flash.cubin"; entry="sdpa_fwd", threadgroup=64, target="sm_86"))
```

A prebuilt binary belongs to one backend, so an op that must run on both ships one for each, or a KernelAbstractions
kernel, which is compiled for whichever backend is loaded. [Tutorial 2](tutorials/2-registering-kernels.md) shows both
backends side by side.

A KernelAbstractions kernel joins the op the same way, with its own `forward` and a binary from
[`ka_compile`](@ref) instead of a file; see [tutorial 2](tutorials/2-registering-kernels.md).

## Automatic differentiation

- **Eager Enzyme:** one rule on `forward(op, kernel, args...)` covers every op and runs your
  [`backward`](@ref), so Enzyme never enters your `forward` or the binary.
- **Under Reactant:** call `backward` directly. A gradient traced *through* a kernel throws, because
  Enzyme-JAX cannot differentiate a custom call yet. There is a work in progress PR in Enzyme to address this:
  [EnzymeAD/Enzyme#2516](https://github.com/EnzymeAD/Enzyme/issues/2516).

See [tutorial 3](tutorials/3-forward-backward-enzyme.md) for details.
