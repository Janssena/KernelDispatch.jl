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
gradients, and compiling a Triton kernel to a `.metallib`. Every exported and public name is documented in
the [API reference](api.md).

**Backends.** Metal runs prebuilt `.metallib` binaries and KernelAbstractions kernels (compiled to `.metallib`);
CUDA runs KernelAbstractions kernels (compiled to `.cubin`) once `CUDA.jl` is loaded. A prebuilt `.metallib` is
Metal-only, so an op that ships one (like the `FlashAttn` kernel of the examples) needs a KA kernel to run on CUDA.
Traced (Reactant) launches are Metal-only for now.

## Concepts

| | what | you write |
|:--|:--|:--|
| **op** | an operation: `SDPA <: AbstractKernelOp` | [`host`](@ref), [`@default_kernel`](@ref) |
| **kernel** | one implementation: `Trifast <: AbstractKernel` | [`forward`](@ref) (builds the binary's arguments), optionally [`backward`](@ref), [`variant_key`](@ref), [`extras`](@ref), [`nearest`](@ref), [`build!`](@ref), [`source_tag`](@ref) |
| **variant** | one (tuned) build of a kernel for a class of inputs: launch name → `KernelBinary` | [`add_variant!`](@ref) |
| **[`KernelBinary`](@ref)** | one compiled launch: file, entry point, `threadgroup`, `tile`, tuning `params`, `is_ka` | — |

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

A KernelAbstractions kernel joins the op the same way, with its own `forward` and a binary from
[`ka_compile`](@ref) instead of a file; see [tutorial 2](tutorials/2-registering-kernels.md).

## Automatic differentiation

- **Eager Enzyme:** one rule on `forward(op, kernel, args...)` covers every op and runs your
  [`backward`](@ref), so Enzyme never enters your `forward` or the binary.
- **Under Reactant:** call `backward` directly. A gradient traced *through* a kernel throws, because
  Enzyme-JAX cannot differentiate a custom call yet. There is a work in progress PR in Enzyme to address this:
  [EnzymeAD/Enzyme#2516](https://github.com/EnzymeAD/Enzyme/issues/2516).

See [tutorial 3](tutorials/3-forward-backward-enzyme.md) for details.
