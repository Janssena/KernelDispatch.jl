"""
    KernelBackend

Where a kernel call will execute. Not the array type: a `TracedRArray` is the same Julia type
whether it will run on CPU or GPU, and the distinction only exists in the XLA client.
"""
abstract type KernelBackend end

"""Host execution: plain Julia arrays, or tracing against a CPU client."""
struct CPUBackend <: KernelBackend end

"""Apple GPUs, via `MtlArray` or an `mps` client."""
struct MetalBackendTag <: KernelBackend end

"""NVIDIA GPUs, via `CuArray` or a CUDA client."""
struct CUDABackendTag <: KernelBackend end

"""AMD GPUs, via `ROCArray` or a ROCm client."""
struct ROCmBackendTag <: KernelBackend end

"""A device no kernel exists for. Always falls back to the host path rather than failing."""
struct UnknownBackend <: KernelBackend end

"""The union of the device backend tags: [`MetalBackendTag`](@ref), [`CUDABackendTag`](@ref), [`ROCmBackendTag`](@ref)."""
const GPUBackend = Union{MetalBackendTag,CUDABackendTag,ROCmBackendTag}

"""
    backend_of(x) -> KernelBackend

The backend an array will execute on.

Plain `AbstractArray`s are host memory. Backend extensions add methods for their own array types
(`MtlArray`, and views of one), and the Reactant extension adds one for traced arrays that inspects
the compiling client — the only way to tell a CPU trace from a GPU trace, since the Julia type is
identical.
"""
backend_of(::AbstractArray) = CPUBackend()
backend_of(x, xs...) = backend_of(x)

"""
    device_dense(backend, x) -> x′

`x` as an array the backend's launcher can bind directly: a plain, contiguous device array.

A device launcher binds raw buffers, but callers hand it wrappers — a `view` of a fused projection's
output, a `reshape`. The default is the identity (host arrays, traced arrays, `nothing`). A backend
extension whose launcher needs plain arrays copies a wrapper here, once, on the device.
"""
device_dense(::KernelBackend, x) = x

"""
    realtype(T) -> Type

The element type behind a possibly-wrapped number type: e.g. Reactant's `TracedRNumber{Float32}` → 
`Float32`. Extended by the Reactant extension.
"""
realtype(::Type{T}) where T = T

"""
    traced_autodiff(x) -> Bool

Whether `x` is being traced inside an `Enzyme.autodiff`/`gradient` call under Reactant. `false` by
default; the Reactant extension answers for traced arrays (`Reactant.WITHIN_AUTODIFF`).

Enzyme-JAX cannot yet differentiate a `stablehlo.custom_call` through an attached reverse rule
(EnzymeAD/Enzyme#2516), so a kernel call traced inside autodiff THROWS rather than quietly running
something else in both directions. See [`KERNEL_IN_TRACED_AUTODIFF`](@ref).
"""
traced_autodiff(x) = false

"""
    KERNEL_IN_TRACED_AUTODIFF

`Ref{Bool}`, default `false`. With `false`, a kernel call traced inside Reactant autodiff throws (see
[`traced_autodiff`](@ref)). Set it to `true` once the Enzyme-JAX in use can differentiate a custom
call through its attached reverse rule.
"""
const KERNEL_IN_TRACED_AUTODIFF = Ref(false)

throw_traced_autodiff(what::AbstractString) = throw(ArgumentError(
    "$what cannot be differentiated under Reactant: Enzyme-JAX cannot differentiate the kernel's " *
    "custom call yet (no custom reverse rules; see EnzymeAD/Enzyme#2516). Use the kernel for " *
    "forward-only programs, and switch it off for any program that is differentiated. " *
    "`KernelDispatch.KERNEL_IN_TRACED_AUTODIFF[] = true` disables this check once Enzyme-JAX supports it."))

check_traced_autodiff(x, what::AbstractString) =
    (traced_autodiff(x) && !KERNEL_IN_TRACED_AUTODIFF[]) && throw_traced_autodiff(what)

"""
    tracing() -> Bool

Whether we are inside a Reactant trace. `false` without the Reactant extension.

Building a kernel on demand (a lazy tune, say) runs eager device work, which the tracer would
capture into the graph; owners check this and refuse instead.
"""
tracing() = _TRACING[]()
const _TRACING = Ref{Function}(() -> false)

"""
    cache_root() -> String

Root of KernelDispatch' on-disk cache, overridable with `KERNELDISPATCH_CACHE`. An op's own registry lives
under [`cache_root(op)`](@ref), which an op may point elsewhere.
"""
cache_root() = get(ENV, "KERNELDISPATCH_CACHE", joinpath(homedir(), ".cache", "KernelDispatch"))

"""
    LOADED_BACKENDS

Backends whose extension has loaded AND whose device is usable, as `:metal`/`:cuda`/`:rocm`. Each
backend extension adds itself in `__init__`. [`use_kernel!`](@ref) consults it so selecting a CUDA
kernel on a Metal machine fails at the point of selection rather than at the first launch.
"""
const LOADED_BACKENDS = Set{Symbol}()

backend_loaded(backend::Symbol) = backend in LOADED_BACKENDS

"""
    device_available() -> Bool

Whether a backend extension is loaded AND its device is usable.
"""
device_available() = !isempty(LOADED_BACKENDS)

"""
    ka_backend_object(backend) -> KernelAbstractions.Backend

The KernelAbstractions backend object for `backend` (`MetalBackend()` for the Metal tag). Implemented
by the backend extension for its own tag. Dispatching on the tag, rather than a hook the extension
installs, lets any extension reach it: the Reactant extension hands the tag of the compiling client.
"""
ka_backend_object(be::KernelBackend) = throw(ArgumentError(
    "no KernelAbstractions backend for $(typeof(be)); load the package for that backend"))

"""
    encode_launch(backend, b::KernelBinary, slots, grid::NTuple{3,Int})

The lowest layer: bind `slots` (device arrays and bits scalars, in slot order, prelude included) to
`b`'s entry point and dispatch it on a 3-D `grid` of `b.threadgroup`-thread groups, waiting for
completion. Refuses a threadgroup over [`max_threads`](@ref). Implemented by the backend extension;
callers want [`run_binary`](@ref), which builds the slots and grid.
"""
function encode_launch end

"""
    device_backend() -> KernelBackend

The tag of the one usable device backend (see [`LOADED_BACKENDS`](@ref)). Throws when there is none,
or several, since then the choice is the caller's.
"""
function device_backend()
    tags = (metal=MetalBackendTag(), cuda=CUDABackendTag(), rocm=ROCmBackendTag())
    length(LOADED_BACKENDS) == 1 || throw(ArgumentError(
        "no single device backend: loaded are $(sort(collect(LOADED_BACKENDS))); pass `backend=`"
    ))
    return tags[only(LOADED_BACKENDS)]
end

"""
    custom_call(backend, entry, path, operands, result_shapes, layout; grid, threadgroup, result_eltypes)
        -> results

Embed the kernel `entry` of the compiled binary at `path` in a traced program as a
`stablehlo.custom_call`, in the form `backend`'s XLA client understands. `layout` is the kernel's
argument table in order, as `(:in, i)`, `(:out, i)`, `(:out_zero, i)` or `(:bytes, payload)`.

Which binaries a backend's custom call can carry is handled by [`traced_support`](@ref).

A generic, so each backend extension adds a method for its own tag (the Reactant extension provides Metal's, for
jax-mps, and CUDA's, for XLA's `triton_kernel_call` handler, which launches binaries that take only pointers and scalars). The traced
`execute` calls it with the tag of the compiling client.
"""
custom_call(be::KernelBackend, args...; kw...) = throw(ArgumentError(
    "no stablehlo.custom_call embedding for backend $(typeof(be)); load an extension that provides one"
))
