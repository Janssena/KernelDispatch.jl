module KernelDispatchCUDAExt

# CUDA's half of KernelDispatch: binding arguments to a compiled `.cubin` and launching it (the eager 
# `execute` every `call_binary` / `run_binary` ends in), and ahead-of-time compilation of 
# KernelAbstractions kernels.
#
# Binary formats: `.cubin` (machine code for one architecture, what `ka_compile` writes) and `.ptx` (text the
# driver compiles when the module loads, so it runs on any newer device). Any other file is refused with an
# `ArgumentError` rather than handed to the driver.

using CUDA
using CUDA: CUDACore
using KernelAbstractions
using KernelDispatch

const KD = KernelDispatch
const KA = KernelAbstractions
const GPUCompiler = CUDACore.GPUCompiler

dptr(::Type{T}) where {T} = Core.LLVMPtr{T,CUDACore.AS.Global}

const IMAGE_EXTENSIONS = (".cubin", ".ptx")

# The bytes the driver loads: a cubin as is, PTX NUL-terminated (it is text).
function _image(path::AbstractString)
    ext = lowercase(splitext(path)[2])
    ext in IMAGE_EXTENSIONS || throw(ArgumentError(
        "CUDA binary `$path`: expected a .cubin (machine code) or .ptx (text the driver compiles), not `$ext`"
    ))
    image = read(path)
    ext == ".ptx" && push!(image, 0x00)

    return image
end

# Module loading is expensive; on the launch path it would dominate small problems. Per context, because
# a `CuFunction` belongs to the context that loaded it.
const _FUNCTIONS = Dict{Tuple{Any,String,String},Any}()

function _function(path::AbstractString, entry::AbstractString)
    get!(_FUNCTIONS, (CUDACore.context(), String(path), String(entry))) do
        mod = CUDACore.CuModule(_image(path))
        CUDACore.create_exceptions!(mod)       # resets the module context's exception flag
        f = CUDACore.CuFunction(mod, entry)
        @debug "loaded CUDA kernel" path entry
        f
    end
end

# The compiled function's own threads-per-block ceiling: what its register use leaves of the
# device's. Launching over it fails with an invalid-configuration error, so launches refuse it first.
function _limit(b::KD.KernelBinary)
    f = _function(b.file, b.entry)
    return Int(CUDACore.attributes(f)[CUDACore.FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK])
end

KD.max_threads(::KD.CUDABackendTag, b::KD.KernelBinary) = _limit(b)

function _check_limit(b::KD.KernelBinary)
    lim = _limit(b)
    b.threadgroup <= lim || throw(ArgumentError(
        "threadgroup $(b.threadgroup) exceeds this kernel's limit of $lim threads per block"
    ))

    return nothing
end

# A device array is passed as its raw device pointer (offset included, so a view works); every other
# slot is a bits value passed by value (a flat KA binary's context integers included).
_arg(a::CuArray) = pointer(a)
_arg(a) = a

# Dynamic shared memory above the default 48 KB must be allowed on the function before a launch asks for it.
const _DEFAULT_SHARED = 48 * 1024
const _SHARED_RAISED = Dict{Tuple{Any,Int},Nothing}()     # (function, bytes) already allowed

function _allow_shared(f, bytes::Integer)
    bytes > _DEFAULT_SHARED || return nothing
    get!(_SHARED_RAISED, (f, Int(bytes))) do
        CUDACore.attributes(f)[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = Int(bytes)
        nothing
    end
    return nothing
end

# One launch of `b`: its function, its dynamic shared memory, and (for a Triton binary) the hidden trailing null
# pointers its compiled signature ends with, after the slots.
function _launch(b::KD.KernelBinary, slots, grid::NTuple{3,Int})
    f = _function(b.file, b.entry)
    args = (
        map(_arg, slots)..., 
        ntuple(Returns(CUDACore.CuPtr{Cvoid}(0)), KD.trailing_pointers(b))...
    )
    _allow_shared(f, b.shared_mem)
    @debug "launching CUDA kernel" blocks = grid threads = b.threadgroup shmem = b.shared_mem nargs = length(args)
    
    CUDACore.cudacall(f, Tuple{map(typeof, args)...}, args...; 
        blocks=grid, threads=(b.threadgroup, 1, 1), shmem=b.shared_mem
    )
    
    return nothing
end

function KD.encode_launch(::KD.CUDABackendTag, b::KD.KernelBinary, slots, grid::NTuple{3,Int})
    _check_limit(b)
    _launch(b, slots, grid)
    CUDACore.synchronize()          # also throws a device-side exception host-side
    return nothing
end

# The repetitions are recorded between two events on one stream: the elapsed time excludes host latency.
function KD.device_time(::KD.CUDABackendTag, items::AbstractVector; reps::Int)
    reps >= 1 || throw(ArgumentError("`reps` must be at least 1, got $reps"))
    for (b, _, _) in items
        _check_limit(b)
    end
    CUDACore.synchronize()
    start, stop = CUDACore.CuEvent(), CUDACore.CuEvent()
    CUDACore.record(start)
    for _ in 1:reps, (b, slots, grid) in items
        _launch(b, slots, grid)
    end
    CUDACore.record(stop)
    CUDACore.synchronize(stop)
    return CUDACore.elapsed(start, stop) / reps
end

"""
    execute(proto::CuArray, l::Launch) -> Tuple

Allocate every `OutArray` of the launch's arguments, bind `(prelude..., args..., extras...)` in order
(a `Val` takes no slot), launch, and return the allocated outputs. On `CuArray` rather than
`AbstractArray`, so that it coexists with the other backends' methods.
"""
function KD.execute(proto::CuArray, l::KD.Launch)
    bound, slots = KD.bind_slots(proto, l.prelude, l.args, l.extras)
    if KD.recording()
        KD.record!(l, Any[(:bytes => p for p in l.prelude)...,
            (a isa KD.OutArray ? (:out => b) : b isa AbstractArray ? (:in => b) : (:scalar => b)
             for (a, b) in zip((l.args..., l.extras...), (bound..., l.extras...)) if !(b isa Val))...])
    else
        KD.encode_launch(KD.CUDABackendTag(), l.binary, slots, l.grid)
    end
    return KD._pick_outs(l.args, bound)
end

# KA's launch context for a static workgroup size and a dynamic ND-range is six integers: the ND-range and the
# block counts, as two `CartesianIndices`. Rebuilt from them here, with none of `KA.launch_config` / `partition`
# (those have error paths, which a state-free device build cannot contain). Equal to what `KD.ka_context` builds,
# type and bits; test/ext/cuda/flat_layout.test.jl checks that for the KA in use.
@inline function _rebuild_ctx(::Val{TG}, nd1::Int, nd2::Int, nd3::Int, b1::Int, b2::Int, b3::Int) where {TG}
    ND = KA.NDIteration
    iterspace = ND.NDRange{3,ND.DynamicSize,ND.StaticSize{(TG, 1, 1)}}(CartesianIndices((b1, b2, b3)), nothing)
    return KA.CompilerMetadata{ND.DynamicSize,ND.DynamicCheck}(CartesianIndices((nd1, nd2, nd3)), iterspace)
end

# The kernel entry that is compiled: the six context integers, then the KA kernel's own arguments unchanged
# (`Val`s are ghost arguments: they stay in the Julia signature and leave the PTX one). `F` is the type of the
# KA kernel function, whose singleton instance is called.
struct FlatKA{F,TG} end

function (::FlatKA{F,TG})(nd1::Int, nd2::Int, nd3::Int, b1::Int, b2::Int, b3::Int, args...) where {F,TG}
    F.instance(_rebuild_ctx(Val(TG), nd1, nd2, nd3, b1, b2, b3), args...)
    return nothing
end

# A compiler job whose compiler does not add CUDA.jl's `KernelState` first parameter: the same params, under a type
# GPUCompiler is told has no kernel state.
struct NoStateParams <: CUDACore.AbstractCUDACompilerParams
    inner::CUDACore.CUDACompilerParams
end

Base.getproperty(p::NoStateParams, s::Symbol) = getproperty(getfield(p, :inner), s)
GPUCompiler.kernel_state_type(::GPUCompiler.CompilerJob{GPUCompiler.PTXCompilerTarget,NoStateParams}) = Nothing

function _stateless_config(name::AbstractString)
    cfg = CUDACore.compiler_config(CUDACore.device(); kernel=true, name=String(name))
    return GPUCompiler.CompilerConfig(cfg; params=NoStateParams(cfg.params))
end

# The distinct `Reason:` lines of a failed device compile, which are what a user can act on.
function _ir_reasons(err)
    lines = split(sprint(showerror, err), '\n')
    return unique(strip.(filter(l -> startswith(l, "Reason:"), lines)))
end

"""
    _compile_cubin(ftype, sig, entry) -> Vector{UInt8}

Ahead-of-time compile `ftype` (the [`FlatKA`](@ref)) for the argument tuple type `sig` as a state-free cubin whose
entry point is `entry`. PTX comes from GPUCompiler, run in the latest world: `CUDACore.compile` runs in a world
frozen at CUDA's initialisation, which cannot see the methods this extension defines (`NoStateParams`). `ptxas`
then assembles it, the way `CUDACore.compile` does.
"""
function _compile_cubin(ftype::Type, sig::Type, entry::AbstractString)
    source = GPUCompiler.methodinstance(ftype, sig)
    job = GPUCompiler.CompilerJob(source, _stateless_config(entry))
    asm, relocations = try
        GPUCompiler.JuliaContext() do _
            asm, meta = Base.invokelatest(GPUCompiler.compile, :asm, job)
            asm, hasproperty(meta, :relocations) ? meta.relocations : ()      # older GPUCompiler: none
        end
    catch err
        err isa GPUCompiler.InvalidIRError || rethrow()
        reasons = _ir_reasons(err)
        throw(ArgumentError("kernel `$entry` cannot be built for CUDA without CUDA.jl's kernel state, which a " *
            "KernelDispatch CUDA binary does not have (so that XLA's kernel-call handler can launch it): its " *
            "compiled code can still throw or otherwise needs the runtime. " *
            (isempty(reasons) ? "" : "Remaining:\n  " * join(reasons, "\n  ")) *
            "\nIndex with unsafe_load / unsafe_store!, and avoid code that can throw."
        ))
    end
    # A relocation is a word patched at load time with the address of a host (Julia) value: a boxed
    # constant or interned symbol the kernel refers to. A bare cubin file has no way to carry it.
    isempty(relocations) || error(
        "kernel needs launch-time relocations (interned symbols or boxed constants), which an " *
        "ahead-of-time cubin cannot carry; remove non-isbits constants from the kernel"
    )
    dir = mktempdir()
    ptx_file, cubin_file = joinpath(dir, "kernel.ptx"), joinpath(dir, "kernel.cubin")
    write(ptx_file, asm)
    log = IOBuffer()
    cmd = `$(CUDACore.CUDA_Compiler.ptxas()) --gpu-name $(KD.target_id(KD.CUDABackendTag())) --output-file $cubin_file $ptx_file`
    success(run(pipeline(ignorestatus(cmd); stdout=log, stderr=log))) ||
        error("ptxas failed on the PTX for `$entry`:\n" * String(take!(log)))
    image = read(cubin_file)
    rm(dir; recursive=true, force=true)
    @debug "compiled CUDA kernel" entry bytes = length(image)
    
    return image
end

_ka_argtype(::Type{A}) where {A<:AbstractArray} = dptr(eltype(A))
_ka_argtype(::Type{T}) where {T} = T

function KD._ka_compile(::KD.CUDABackendTag, kernel, argtypes::Tuple, tg::Int, name::String, dir::String)
    mkpath(dir)
    path = joinpath(dir, name * ".cubin")
    isfile(path) && return path
    obj = kernel(CUDABackend(), tg)
    ftype = FlatKA{typeof(obj.f),tg}
    # The six context integers, then the kernel's arguments, buffers as raw device pointers.
    sig = Tuple{Int,Int,Int,Int,Int,Int,map(_ka_argtype, argtypes)...}
    write(path, _compile_cubin(ftype, sig, name))
    return path
end

# Wrappers over device memory that CUDA.jl leaves as a `SubArray` / `ReshapedArray` / `Adjoint` /
# `Transpose` rather than a `CuArray`.
const WrappedCuArray = Union{
    SubArray{<:Any,<:Any,<:CuArray},
    Base.ReshapedArray{<:Any,<:Any,<:CuArray},
    CUDACore.LinearAlgebra.Adjoint{<:Any,<:CuArray},
    CUDACore.LinearAlgebra.Transpose{<:Any,<:CuArray},
}

KD.backend_of(::CuArray) = KD.CUDABackendTag()
KD.backend_of(::WrappedCuArray) = KD.CUDABackendTag()

# The launcher binds a raw pointer and no strides, so a wrapper that is not a `CuArray` (a strided
# `view`, an adjoint) is materialised once, on the device. Anything else is left alone: a host array
# among the arguments should fail at binding, not be copied.
KD.device_dense(::KD.CUDABackendTag, x::CuArray) = x
KD.device_dense(::KD.CUDABackendTag, x::WrappedCuArray) = CuArray(x)

KD.ka_backend_object(::KD.CUDABackendTag) = CUDABackend()

# The device's compute capability as a binary's `target` spells it: "sm_86".
function KD.target_id(::KD.CUDABackendTag)
    cap = CUDACore.capability(CUDACore.device())
    return "sm_$(cap.major)$(cap.minor)"
end

# A cubin is machine code for one architecture family (same major, device minor at least the cubin's); PTX is
# compiled by the driver when the module loads, so any device at or above its `.target` takes it.
function KD.compatible_target(be::KD.CUDABackendTag, b::KD.KernelBinary)
    # A KA binary built before they were flat takes CUDA.jl's kernel state and a context blob: launched with the
    # flat prelude it would read shifted arguments.
    b.abi === :ka && get(b.params, :ka_layout, "") != "flat" && return "KernelAbstractions binary `$(b.entry)` " *
        "was built by an older KernelDispatch, with CUDA.jl's kernel state; its layout is not the flat one launched " *
        "now. Delete the stale `ka` build directory and the registry entry, and rebuild it with `ka_compile`"
    
    (isempty(b.target) || KD.binary_format(b) === :ptx) && return nothing
    dev = KD.target_id(be)
    KD.cuda_sm_compatible(b.target, dev) && return nothing
    
    return "cubin `$(b.entry)` was built for $(b.target), but this device is $dev: rebuild it for this " *
           "device, or ship PTX, which the driver compiles for any newer device"
end

# A flat KA binary takes no kernel state, and KA's context as its six integers: the ND-range, then the block counts.
KD.ka_state_words(::KD.CUDABackendTag) = ()
function KD.ka_prelude(::KD.CUDABackendTag, ctx, ghost::Bool)
    ghost && throw(ArgumentError(
        "a KA context with a static ND-range has no flat form; KernelDispatch builds " *
        "its binaries for a dynamic ND-range"
    ))
    return Any[size(ctx.ndrange)..., size(KA.NDIteration.blocks(ctx.iterspace))...]
end
KD.ka_layout(::KD.CUDABackendTag) = :flat

# KA's CUDA backend addresses a threadgroup by `blockIdx().x` alone and rebuilds the N-D indices from the
# context, so the binary is launched on a flat 1-D grid.
KD.ka_grid(::KD.CUDABackendTag, groups::NTuple{3,Int}) = (prod(groups), 1, 1)

function __init__()
    ok = try
        CUDACore.functional()
    catch
        false
    end
    ok && push!(KD.LOADED_BACKENDS, :cuda)
    
    return nothing
end

end # module
