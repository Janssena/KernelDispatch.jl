module KernelDispatchMetalExt

# Metal's half of KernelDispatch: binding arguments to a compiled `.metallib` and dispatching it (the eager
# `execute` every `call_binary` / `run_binary` ends in), and ahead-of-time compilation of
# KernelAbstractions kernels.

import Metal: AS, GPUCompiler, LinearAlgebra

using Metal
using KernelDispatch

const KD = KernelDispatch

dptr(::Type{T}) where {T} = Core.LLVMPtr{T,AS.Device}

# Pipeline creation (metallib load -> MTLFunction -> pipeline state) is expensive; on the launch path
# it put a large constant on every measurement and dominated small problems.
const _PIPES = Dict{Tuple{String,String},Any}()
function pipeline(path, entry)
    get!(_PIPES, (path, entry)) do
        dev = Metal.device()
        Metal.MTLComputePipelineState(dev,
            Metal.MTLFunction(Metal.MTLLibraryFromFile(dev, path), entry))
    end
end

# Metal caps live command queues at ~64; one per launch survived a few reps and then threw
# `UndefRefError`, after first producing impossibly fast timings. One queue per process.
const _QUEUE = Ref{Any}(nothing)
queue() = (isnothing(_QUEUE[]) && (_QUEUE[] = Metal.MTLCommandQueue(Metal.device())); _QUEUE[])

# A bits value bound by value lives in a tiny device buffer of its own. Cached per value: the same
# scalars (and KA contexts) recur on every call.
const _ARG_BUFFERS = Dict{Any,MtlArray}()
arg_buffer(a) = get!(_ARG_BUFFERS, (typeof(a), a)) do
    MtlArray([a])
end

# The compiled pipeline's own threadgroup ceiling: what a kernel's register use leaves of the
# device's. Dispatching over it returns garbage AND a fast time, so launches refuse it.
pipeline_limit(path, entry) = Int(pipeline(path, entry).maxTotalThreadsPerThreadgroup)

KD.max_threads(::KD.MetalBackendTag, b::KD.KernelBinary) = pipeline_limit(b.file, b.entry)

function _check_limit(path, entry, tg)
    lim = pipeline_limit(path, entry)
    tg <= lim || throw(ArgumentError(
        "threadgroup $tg exceeds this pipeline's limit of $lim; " *
        "dispatching over it returns garbage AND a fast time"
    ))
    return nothing
end

# Bind `args` (in slot order) to the encoder's buffer slots.
function _set_slots!(enc, args)
    for (i, a) in enumerate(args)
        if a isa MtlArray
            Metal.set_buffer!(enc, a.data[], a.offset, i)   # 1-based, despite Metal's 0-based convention
        else
            Metal.set_buffer!(enc, arg_buffer(a).data[], 0, i)
        end
    end

    return nothing
end

# Encode `reps` repetitions of a sequence of dispatches — `(binary, slots, grid)` triples, in order —
# into one command buffer, commit and wait for it.
function _run(items, reps::Int)
    for (b, _, _) in items
        _check_limit(b.file, b.entry, b.threadgroup)
    end
    # Metal.jl batches its own commands (fills, copies, argument-buffer uploads) on ITS queue; this
    # launch commits to another. Without draining Metal.jl first the kernel can run before they land.
    Metal.synchronize()
    cb = Metal.MTLCommandBuffer(queue())
    enc = Metal.MTLComputeCommandEncoder(cb)
    one = length(items) == 1

    if one          # one binary: bind once, dispatch `reps` times
        b, slots, _ = only(items)
        Metal.set_function!(enc, pipeline(b.file, b.entry))
        _set_slots!(enc, slots)
    end
    
    for _ in 1:reps, (b, slots, grid) in items
        if !one
            Metal.set_function!(enc, pipeline(b.file, b.entry))
            _set_slots!(enc, slots)
        end
        Metal.dispatchThreadgroups!(enc, Metal.MTLSize(grid...), Metal.MTLSize(Int(b.threadgroup), 1, 1))
    end
    Metal.endEncoding!(enc)
    Metal.commit!(cb)
    Metal.wait_completed(cb)
    
    return cb
end

KD.encode_launch(::KD.MetalBackendTag, b::KD.KernelBinary, slots, grid::NTuple{3,Int}) =
    (_run([(b, slots, grid)], 1); nothing)

# The repetitions share one command buffer: its GPU timestamps exclude all host latency.
function KD.device_time(::KD.MetalBackendTag, items::AbstractVector; reps::Int)
    reps >= 1 || throw(ArgumentError("`reps` must be at least 1, got $reps"))
    cb = _run(items, reps)
    return (cb.GPUEndTime - cb.GPUStartTime) / reps
end

"""
    execute(proto::AbstractArray, l::Launch) -> Tuple

Allocate every `OutArray` of the launch's arguments (an `InOutArray` is bound as its array), bind `(prelude..., args..., extras...)` in order
(a `Val` takes no slot), dispatch, and return the allocated outputs. Dispatches on `MtlArray`, so it
coexists with the other backends' methods (CUDA's on `CuArray`, Reactant's on traced arrays).
"""
function KD.execute(proto::MtlArray, l::KD.Launch)
    bound, slots = KD.bind_slots(proto, l.prelude, l.args, l.extras)
    if KD.recording()
        KD.record!(l, Any[(:bytes => p for p in l.prelude)...,
            (a isa KD.OutArray ? (:out => b) : a isa KD.InOutArray ? (:inout => b) : b isa AbstractArray ? (:in => b) : (:scalar => b)
             for (a, b) in zip((l.args..., l.extras...), (bound..., l.extras...)) if !(b isa Val))...])
    else
        KD.encode_launch(KD.MetalBackendTag(), l.binary, slots, l.grid)
    end
    return KD._pick_outs(l.args, bound)
end

"""
    compile_metallib(kernel, signature, tg, entry) -> Vector{UInt8}

Ahead-of-time compile the KernelAbstractions `kernel` (instantiated at workgroup size `tg`) for the
argument tuple type `signature`, as a `.metallib` whose entry point is `entry`.
"""
function compile_metallib(kernel, sig::Type, tg::Integer, entry::AbstractString)
    obj = kernel(MetalBackend(), Int(tg))
    job = GPUCompiler.CompilerJob(GPUCompiler.methodinstance(typeof(obj.f), sig),
        Metal.compiler_config(Metal.device(); name=String(entry), kernel=true))
    return _metallib(job)
end

# Metal 1.10 exposes `Metal.compile(job)`; 1.11 (GPUCompiler 2) renamed it `compile_to_metallib` and
# its result may carry `relocations`, a table resolved at launch time that a bare `.metallib` has no
# way to supply (it only arises for kernels referencing interned symbols / boxed constants).
function _metallib(job)
    res = isdefined(Metal, :compile_to_metallib) ? Metal.compile_to_metallib(job) : Metal.compile(job)
    isnothing(get(res, :relocations, nothing)) || error(
        "kernel needs launch-time relocations (interned symbols or boxed constants), which an " *
        "ahead-of-time `.metallib` cannot carry; remove non-isbits constants from the kernel")
    return res.metallib
end

_ka_argtype(::Type{A}) where {A<:AbstractArray} = dptr(eltype(A))
_ka_argtype(::Type{T}) where {T} = T

function KD._ka_compile(::KD.MetalBackendTag, kernel, argtypes::Tuple, tg::Int, name::String,
        dir::String)
    mkpath(dir)
    path = joinpath(dir, name * ".metallib")
    isfile(path) && return path
    ctx, _ = KD.ka_context(KD.MetalBackendTag(), (1, 1, 1), tg)
    sig = Tuple{typeof(ctx),map(_ka_argtype, argtypes)...}
    write(path, compile_metallib(kernel, sig, tg, name))
    return path
end

# Wrappers over device memory that Metal.jl leaves as a `SubArray` / `ReshapedArray` / `Adjoint` /
# `Transpose` rather than an `MtlArray` (it returns an `MtlArray` with an offset for a contiguous
# `view` or a `reshape`).
const WrappedMtlArray = Union{
    SubArray{<:Any,<:Any,<:MtlArray},
    Base.ReshapedArray{<:Any,<:Any,<:MtlArray},
    LinearAlgebra.Adjoint{<:Any,<:MtlArray},
    LinearAlgebra.Transpose{<:Any,<:MtlArray},
}

KD.backend_of(::MtlArray) = KD.MetalBackendTag()
KD.backend_of(::WrappedMtlArray) = KD.MetalBackendTag()

# The launcher binds a raw buffer (plus offset) and no strides, so a wrapper that is not an `MtlArray`
# (a strided `view`, an adjoint) is copied once, on the device. Anything else is left alone: a host array among
# the arguments should fail at binding, not be copied into another host array.
KD.device_dense(::KD.MetalBackendTag, x::MtlArray) = x
KD.device_dense(::KD.MetalBackendTag, x::WrappedMtlArray) = copy(x)

KD.ka_backend_object(::KD.MetalBackendTag) = MetalBackend()

# Metal.jl's KA kernels take a kernel-state word (a `UInt32`) ahead of KA's context.
KD.ka_state_words(::KD.MetalBackendTag) = (UInt32(0),)

# KA's Metal backend reads only the x component of the threadgroup position and rebuilds the N-D
# indices from the context, so the binary is dispatched as a flat 1-D grid.
KD.ka_grid(::KD.MetalBackendTag, groups::NTuple{3,Int}) = (prod(groups), 1, 1)

const _AVAILABLE = Ref{Union{Nothing,Bool}}(nothing)
function available()
    v = _AVAILABLE[]
    isnothing(v) || return v
    ok = try
        Metal.functional() && !isnothing(Metal.device())
    catch
        false
    end
    _AVAILABLE[] = ok
    return ok
end

function __init__()
    available() && push!(KD.LOADED_BACKENDS, :metal)
    return nothing
end

end # module
