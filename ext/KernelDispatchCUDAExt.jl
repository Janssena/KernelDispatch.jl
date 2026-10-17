module KernelDispatchCUDAExt

# CUDA's half of KernelDispatch, the counterpart of KernelDispatchMetalExt: binding arguments to a compiled
# `.cubin` and launching it (the eager `execute` every `call_binary` / `run_binary` ends in), and
# ahead-of-time compilation of KernelAbstractions kernels.
#
# Written against CUDA.jl 6 (CUDACore). The pieces that depend on CUDA.jl / GPUCompiler internals are kept
# in a few small functions so a version difference is a local fix:
#
#   `_compile_cubin`   KA kernel + argument types -> cubin image   (GPUCompiler job, `CUDACore.compile`)
#   `_function`        cubin file -> `CuFunction`, cached           (`CuModule`)
#   `_kernel_state`    CUDA.jl's hidden leading kernel argument     (`KernelState`)
#   `_launch`          one launch of a `CuFunction`                 (`cudacall`)
#
# `JULIA_DEBUG=KernelDispatchCUDAExt` logs the entry point, image size and launch geometry.

using CUDA
using CUDA: CUDACore
using KernelDispatch

const KO = KernelDispatch
const GPUCompiler = CUDACore.GPUCompiler

dptr(::Type{T}) where {T} = Core.LLVMPtr{T,CUDACore.AS.Global}

# --- the kernel state -------------------------------------------------------------------------------

# Every kernel CUDA.jl compiles takes a `KernelState` (the device-side exception flag and an RNG seed) as
# a hidden first argument, the analogue of Metal.jl's kernel-state word. `exception_info` is a
# host-mapped flag per context that a failing kernel writes to; `CUDACore.synchronize` checks it and
# throws a `KernelException` host-side.
function _kernel_state()
    ctx = CUDACore.context()
    mem = get!(CUDACore.exception_infos, ctx) do
        CUDACore.alloc(CUDACore.HostMemory, sizeof(CUDACore.ExceptionInfo_st),
            CUDACore.MEMHOSTALLOC_DEVICEMAP)
    end
    return CUDACore.KernelState(convert(CUDACore.ExceptionInfo, mem), UInt32(0))
end

# --- loading ----------------------------------------------------------------------------------------

# Module loading is expensive; on the launch path it would dominate small problems. Per context, because
# a `CuFunction` belongs to the context that loaded it.
const _FUNCTIONS = Dict{Tuple{Any,String,String},Any}()

function _function(path::AbstractString, entry::AbstractString)
    get!(_FUNCTIONS, (CUDACore.context(), String(path), String(entry))) do
        image = read(path)
        endswith(path, ".ptx") && push!(image, 0x00)         # PTX is text: the driver wants it NUL-terminated
        mod = CUDACore.CuModule(image)
        CUDACore.create_exceptions!(mod)       # resets the module context's exception flag
        f = CUDACore.CuFunction(mod, entry)
        @debug "loaded CUDA kernel" path entry
        f
    end
end

# The compiled function's own threads-per-block ceiling: what its register use leaves of the
# device's. Launching over it fails with an invalid-configuration error, so launches refuse it first.
function _limit(b::KO.KernelBinary)
    f = _function(b.file, b.entry)
    return Int(CUDACore.attributes(f)[CUDACore.FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK])
end

KO.max_threads(::KO.CUDABackendTag, b::KO.KernelBinary) = _limit(b)

function _check_limit(b::KO.KernelBinary)
    lim = _limit(b)
    b.threadgroup <= lim || throw(ArgumentError(
        "threadgroup $(b.threadgroup) exceeds this kernel's limit of $lim threads per block"))
    return nothing
end

# --- launching --------------------------------------------------------------------------------------

# A device array is passed as its raw device pointer (offset included, so a view works); every other
# slot is a bits value passed by value, the KernelState and KA's launch context included.
_arg(a::CuArray) = pointer(a)
_arg(a) = a

function _launch(f, slots, grid::NTuple{3,Int}, tg::Integer)
    args = Tuple(map(_arg, slots))
    @debug "launching CUDA kernel" blocks = grid threads = tg nargs = length(args)
    CUDACore.cudacall(f, Tuple{map(typeof, args)...}, args...; blocks=grid, threads=(Int(tg), 1, 1))
    return nothing
end

function KO.encode_launch(::KO.CUDABackendTag, b::KO.KernelBinary, slots, grid::NTuple{3,Int})
    _check_limit(b)
    _launch(_function(b.file, b.entry), slots, grid, b.threadgroup)
    CUDACore.synchronize()          # also throws a device-side exception host-side
    return nothing
end

# The repetitions are recorded between two events on one stream: the elapsed time excludes host latency.
function KO.device_time(::KO.CUDABackendTag, items::AbstractVector; reps::Int)
    reps >= 1 || throw(ArgumentError("`reps` must be at least 1, got $reps"))
    for (b, _, _) in items
        _check_limit(b)
    end
    CUDACore.synchronize()
    start, stop = CUDACore.CuEvent(), CUDACore.CuEvent()
    CUDACore.record(start)
    for _ in 1:reps, (b, slots, grid) in items
        _launch(_function(b.file, b.entry), slots, grid, b.threadgroup)
    end
    CUDACore.record(stop)
    CUDACore.synchronize(stop)
    return CUDACore.elapsed(start, stop) / reps
end

# --- the eager executor -----------------------------------------------------------------------------

"""
    execute(proto::CuArray, l::Launch) -> Tuple

Allocate every `OutArray` of the launch's arguments, bind `(prelude..., args..., extras...)` in order
(a `Val` takes no slot), launch, and return the allocated outputs. On `CuArray` rather than
`AbstractArray`, so that it coexists with the other backends' methods.
"""
function KO.execute(proto::CuArray, l::KO.Launch)
    bound, slots = KO.bind_slots(proto, l.prelude, l.args, l.extras)
    if KO.recording()
        KO.record!(l, Any[(:bytes => p for p in l.prelude)...,
            (a isa KO.OutArray ? (:out => b) : b isa AbstractArray ? (:in => b) : (:scalar => b)
             for (a, b) in zip((l.args..., l.extras...), (bound..., l.extras...)) if !(b isa Val))...])
    else
        KO.encode_launch(KO.CUDABackendTag(), l.binary, slots, l.grid)
    end
    return KO._pick_outs(l.args, bound)
end

# --- compiling --------------------------------------------------------------------------------------

"""
    _compile_cubin(kernel, signature, tg, entry) -> Vector{UInt8}

Ahead-of-time compile the KernelAbstractions `kernel` (instantiated at workgroup size `tg`) for the
argument tuple type `signature`, as a cubin whose entry point is `entry`.
"""
function _compile_cubin(kernel, sig::Type, tg::Integer, entry::AbstractString)
    obj = kernel(CUDABackend(), Int(tg))
    source = GPUCompiler.methodinstance(typeof(obj.f), sig)
    config = CUDACore.compiler_config(CUDACore.device(); kernel=true, name=String(entry))
    res = CUDACore.compile(GPUCompiler.CompilerJob(source, config))
    # A relocation is a word patched at load time with the address of a host (Julia) value: a boxed
    # constant or interned symbol the kernel refers to. A bare cubin file has no way to carry it.
    # (CUDA.jl 6.2 returns no `relocations` field: there was nothing to relocate yet.)
    isempty(get(res, :relocations, ())) || error(
        "kernel needs launch-time relocations (interned symbols or boxed constants), which an " *
        "ahead-of-time cubin cannot carry; remove non-isbits constants from the kernel")
    res.entry == entry || error(
        "the compiled kernel's entry point is `$(res.entry)`, expected `$entry`")
    @debug "compiled CUDA kernel" entry bytes = length(res.image)
    return res.image
end

_ka_argtype(::Type{A}) where {A<:AbstractArray} = dptr(eltype(A))
_ka_argtype(::Type{T}) where {T} = T

function KO._ka_compile(::KO.CUDABackendTag, kernel, argtypes::Tuple, tg::Int, name::String,
        dir::String)
    mkpath(dir)
    path = joinpath(dir, name * ".cubin")
    isfile(path) && return path
    ctx, _ = KO.ka_context(KO.CUDABackendTag(), (1, 1, 1), tg)
    # CUDA.jl prepends the KernelState itself (see `_kernel_state`); it is not part of the signature.
    sig = Tuple{typeof(ctx),map(_ka_argtype, argtypes)...}
    write(path, _compile_cubin(kernel, sig, tg, name))
    return path
end

# --- routing ----------------------------------------------------------------------------------------

# Wrappers over device memory that CUDA.jl leaves as a `SubArray` / `ReshapedArray` / `Adjoint` /
# `Transpose` rather than a `CuArray`.
const WrappedCuArray = Union{
    SubArray{<:Any,<:Any,<:CuArray},
    Base.ReshapedArray{<:Any,<:Any,<:CuArray},
    CUDACore.LinearAlgebra.Adjoint{<:Any,<:CuArray},
    CUDACore.LinearAlgebra.Transpose{<:Any,<:CuArray},
}

KO.backend_of(::CuArray) = KO.CUDABackendTag()
KO.backend_of(::WrappedCuArray) = KO.CUDABackendTag()

# The launcher binds a raw pointer and no strides, so a wrapper that is not a `CuArray` (a strided
# `view`, an adjoint) is materialised once, on the device. Anything else is left alone: a host array
# among the arguments should fail at binding, not be copied.
KO.device_dense(::KO.CUDABackendTag, x::CuArray) = x
KO.device_dense(::KO.CUDABackendTag, x::WrappedCuArray) = CuArray(x)

KO.ka_backend_object(::KO.CUDABackendTag) = CUDABackend()

# CUDA.jl's KA kernels take its `KernelState` ahead of KA's context.
KO.ka_state_words(::KO.CUDABackendTag) = (_kernel_state(),)

# KA's CUDA backend addresses a threadgroup by `blockIdx().x` alone and rebuilds the N-D indices from the
# context, so the binary is launched on a flat 1-D grid.
KO.ka_grid(::KO.CUDABackendTag, groups::NTuple{3,Int}) = (prod(groups), 1, 1)

function __init__()
    ok = try
        CUDACore.functional()
    catch
        false
    end
    ok && push!(KO.LOADED_BACKENDS, :cuda)
    return nothing
end

end # module
