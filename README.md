<p align="left">
  <a href="https://github.com/Janssena/KernelDispatch.jl/actions/workflows/test.yml"><img src="https://github.com/Janssena/KernelDispatch.jl/actions/workflows/test.yml/badge.svg?branch=main" alt="CPU tests"></a>
  <a href="https://janssena.github.io/KernelDispatch.jl/dev/"><img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="Documentation (dev)"></a>
  <a href="https://julialang.org"><img src="https://img.shields.io/badge/Julia-1.12%20(%E2%89%A5%201.11)-9558B2?logo=julia&logoColor=white" alt="Julia 1.12, requires 1.11 or newer"></a>
</p>

# KernelDispatch.jl

<p align="center">
  <img src="assets/logo.png" alt="KernelDispatch.jl logo" width="200">
</p>

Simplify working with compiled kernel binaries built anywhere, directly in Julia!

- Define an op
- Register kernels for it
- Select a default kernel and run the op on CPU and device arrays through multiple dispatch
- Run kernels embedded in a Reactant-compiled program as a `stablehlo.custom_call`

An op can have several **kernels** (implementations) you switch between at runtime, each with **tuned
variants** picked from the inputs. Binaries come from files (`.metallib`, `.cubin`, `.ptx`), from
[KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl) kernels (`ka_compile`) or from Triton
(`triton_binary`), and are launched the same way.

**Documentation:** <https://janssena.github.io/KernelDispatch.jl/dev/>

## Example

Register one kernel for a scaled dot-product attention op (`q: d×n`, `k: d×m`, `v: dv×m` → `o: dv×n`).

```julia
using KernelDispatch
import KernelDispatch: host, forward

# The op, with its CPU implementation
struct SDPA <: AbstractKernelOp end

softmax(s) = (p = exp.(s .- maximum(s; dims=1)); p ./ sum(p; dims=1))
host(::SDPA, q, k, v) = v * softmax((k' * q) ./ sqrt(Float32(size(q, 1))))

# A kernel for it, and its default
struct FlashAttn <: AbstractKernel end
KernelDispatch.@default_kernel SDPA FlashAttn()

# Register the compiled binary under the name `:fwd`
add_variant!(SDPA(), FlashAttn(); fwd = KernelBinary("flash.metallib"; threadgroup=64))

# `forward` builds the binary's arguments from the op's and launches it
function forward(op::SDPA, kernel::FlashAttn, q, k, v)
    (d, n), m, dv = Int32.(size(q)), Int32(size(k, 2)), Int32(size(v, 1))
    o = OutArray(Float32, dv, n)               # an output: state its type and shape
    scale = inv(sqrt(Float32(d)))
    result = call_binary(op, kernel, :fwd, q, k, v, o, n, m, d, dv, scale; extent=n)
    return only(result)                        # the outputs, here just `o`
end
```

Calling the op picks the path from its arguments, and switching kernels is one line:

```julia
o = SDPA()(q, k, v)                  # MtlArray or CuArray: the default kernel, FlashAttn
SDPA()(Array(q), Array(k), Array(v)) # Array: `host`. Under Reactant: a custom call
use_kernel!(SDPA(), SDPAKA())        # another kernel, for this session
forward(SDPA(), FlashAttn(), q, k, v) # one call with a given kernel, no global switch
```

On CUDA only the binary changes: `KernelBinary("flash.cubin"; entry="sdpa_fwd", target="sm_86")`.

## Documentation

| | |
|:--|:--|
| [Switching kernels](https://janssena.github.io/KernelDispatch.jl/dev/tutorials/1-switching-kernels/) | list, select and persist default kernels |
| [Registering kernels](https://janssena.github.io/KernelDispatch.jl/dev/tutorials/2-registering-kernels/) | prebuilt binaries, KernelAbstractions kernels, `build!` |
| [Forward, backward, Enzyme](https://janssena.github.io/KernelDispatch.jl/dev/tutorials/3-forward-backward-enzyme/) | custom `forward`/`backward` and gradients |
| [Triton kernels](https://janssena.github.io/KernelDispatch.jl/dev/tutorials/4-triton-kernels/) | Triton to `.metallib`, or a Triton `.cubin` on NVIDIA |
| [Backends](https://janssena.github.io/KernelDispatch.jl/dev/backends/) | what runs where, how a launch goes, what is verified and the limits |
| [API reference](https://janssena.github.io/KernelDispatch.jl/dev/api/) | |

## Concepts

| | what | you write |
|:--|:--|:--|
| **op** | an operation: `SDPA <: AbstractKernelOp` | `host`, `@default_kernel` |
| **kernel** | one implementation: `FlashAttn <: AbstractKernel` | `forward`, optionally `backward`, `variant_key`, `extras`, `build!` |
| **variant** | one (tuned) build of a kernel for a class of inputs | `add_variant!` |
| **`KernelBinary`** | one compiled launch: file, entry point, `threadgroup`, `tile`, `abi`, `target`, `signature`, ... | made by `ka_compile`, `triton_binary` or by hand |

Ops and kernels are fieldless singletons and the selected kernel is a method, so an op call is type-stable.

## KernelAbstractions kernels

A KernelAbstractions kernel is one more source of binaries. It joins the op as another kernel with its own
`forward`; `ka_compile` turns it into a binary for whichever backend is loaded.

```julia
using KernelAbstractions

@kernel unsafe_indices = true function sdpa_ka!(q, k, v, o, n::Int32, m::Int32,
        d::Int32, dv::Int32, scale::Float32, ::Val{TG}) where {TG}
    # ... one query per thread, read and written with unsafe_load / unsafe_store!
end

struct SDPAKA <: AbstractKernel end
KernelDispatch.extras(::SDPA, ::SDPAKA, b::KernelBinary) = (Val(b.threadgroup),)

argtypes = (Vector{Float32}, Vector{Float32}, Vector{Float32}, Vector{Float32},
            Int32, Int32, Int32, Int32, Float32, Val{64})
fwd = ka_compile(KernelDispatch.device_backend(), sdpa_ka!, argtypes; tg=64, name="sdpa_ka_v1")
add_variant!(SDPA(), SDPAKA(); fwd)
use_kernel!(SDPA(), SDPAKA())
```

`SDPAKA` needs a `forward` like `FlashAttn`'s (same argument list). See the
[tutorial](https://janssena.github.io/KernelDispatch.jl/dev/tutorials/2-registering-kernels/#B.-From-a-KernelAbstractions-kernel)
for the complete kernel, and for what differs between Metal and CUDA.

## Triton kernels

Compile ahead with Triton (Python), keep the `.cubin`, `.ptx` and `.json` it writes, and register them:

```julia
bin = triton_binary("scale_kernel.json")
add_variant!(op, kernel, (:Float32,); fwd=bin)
```

Pass exactly the arguments the compiled kernel declares. A Triton binary is only valid for launches whose
integer arguments fall in the same classes (equal to 1, divisible by 16) as when it was compiled. Details in
the [tutorial](https://janssena.github.io/KernelDispatch.jl/dev/tutorials/4-triton-kernels/).

## Backends

- **Metal** runs `.metallib` binaries and KernelAbstractions kernels.
- **CUDA** runs `.cubin` / `.ptx` files, KernelAbstractions kernels and Triton kernels.
- **Reactant:** a binary is embedded as a `stablehlo.custom_call`, through jax-mps on Metal and XLA's
  `triton_kernel_call_ffi` on CUDA.

A prebuilt binary is backend-specific, so an op that ships one also offers a KernelAbstractions kernel to run
anywhere. ROCm is not implemented. The [Backends](https://janssena.github.io/KernelDispatch.jl/dev/backends/)
page has the verification matrix and the limits.

## Precisions

`Float32`, `Float16` and `BFloat16` are supported on Metal and CUDA, eagerly and from a Reactant trace,
including launches that mix them. State the element type where you allocate an output
(`OutArray(Float16, n)`), and register one binary per precision through the variant key. `Float64` is not a goal.
Triton kernels in half precision are tested with Triton 3.2.0 only. See
[`docs/half-precision.md`](docs/half-precision.md) for what is covered and the decisions behind it.

## Argument types are checked

A binary takes raw pointers, so one compiled for `float*` and handed a `Float16` array would run and potentially 
result in silently incorrect operations. `ka_compile` records the signature of what it builds, and a launch with 
other types is refused, with the argument named. For a prebuilt binary, declare it:

```julia
KernelBinary("scale.metallib"; signature=(Vector{Float32}, Float32, Int32))
```

Without a signature the first launch warns once and prints the one that fits. `signature=false` declares the binary
unchecked, and `check_types=false` skips the check for one launch.

## Automatic differentiation

- **Eager Enzyme:** one rule on `forward(op, kernel, args...)` covers every op and runs your `backward`.
- **Under Reactant:** call `backward` directly. Enzyme-JAX cannot differentiate a custom call yet
  (EnzymeAD/Enzyme#2516), so a gradient traced *through* a kernel throws.

## Tests

```
julia --project=test test/runtests.jl          # everything the machine can run
julia --project=test test/runtests.jl cpu      # skip any test that requires a GPU to run.
julia --project=test test/runtests.jl gpu      # Metal on Apple, CUDA on NVIDIA; skipped without a device
julia --project=test test/runtests.jl gpu ext  # runs both the gpu tests and the extension tests (enzyme, reactant, etc.)
```

`KD_TEST_GPU=none|metal|cuda` overrides the GPU backend choice. The Triton tests run from recorded fixtures kept
in an artifact (`test/Artifacts.toml`) that is downloaded the first time a test needs it, so no Python is required.
`KD_FIXTURES=/dir` points the tests at a different local directory holding `triton/` and `trifast/`;
`test/fixtures/make_triton_fixture.py float32` regenerates the fixtures on a NVIDIA machine using float32 precision 
so you can test local triton versions (alternatives are float16 and bfloat16). 
