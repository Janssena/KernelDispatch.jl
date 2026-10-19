"""
    supported_formats(backend) -> Tuple{Vararg{Symbol}}

The binary formats `backend` launches, as the file extensions [`binary_format`](@ref) reads: Metal
`(:metallib,)`, CUDA `(:cubin, :ptx)`, ROCm `(:hsaco,)`. `()` (the default, for the host and unknown backends)
means unchecked. A fact about the vendor's formats, so it lives here and not in an extension: a traced launch
needs it even when only Reactant is loaded.
"""
supported_formats(::KernelBackend) = ()
supported_formats(::MetalBackendTag) = (:metallib,)
supported_formats(::CUDABackendTag) = (:cubin, :ptx)
supported_formats(::ROCmBackendTag) = (:hsaco,)

"""
    binary_format(b::KernelBinary) -> Symbol

The format of `b.file`, from its extension (`:cubin` for `kernel.cubin`).
"""
binary_format(b::KernelBinary) = Symbol(lowercase(lstrip(splitext(b.file)[2], '.')))

"""
    target_id(backend) -> String

The architecture of the device `backend` runs on, as a binary's [`KernelBinary`](@ref)`.target` spells it
(`"sm_86"` for CUDA). `""` (the default, and Metal) means the device has no architecture that matters: its
binaries are portable. Implemented by the backend extension, which can ask the device.
"""
target_id(::KernelBackend) = ""

"""
    compatible_target(backend, b::KernelBinary) -> Union{Nothing,String}

`nothing` when `b` can run on `backend`'s device, else the reason it cannot (a cubin built for another
architecture). The default accepts everything. Implemented by the backend extension.
"""
compatible_target(::KernelBackend, ::KernelBinary) = nothing

"""
    traced_support(backend, b::KernelBinary) -> Union{Nothing,String}

`nothing` when `b` can be launched from a Reactant trace on `backend`, else why not. Asked at the top of the
traced `execute`, before any argument is converted, so the reason reaches the user instead of whatever the
unsupported binary would have tripped over first. The default accepts everything, and so does every backend today:
CUDA's KA binaries are built flat (pointers and scalars only), which XLA's kernel-call handler can pass. A backend
whose custom call cannot carry some binary (struct arguments, say) overrides this to say why.
"""
traced_support(::KernelBackend, ::KernelBinary) = nothing

"""
    check_binary(backend, b::KernelBinary)

Refuse, with an explanation, a binary `backend` cannot launch: a format it does not run, or a target it cannot
run on. Cheap and static; done once per [`prepare_launch`](@ref), eager or traced. Unchecked for the host.
"""
function check_binary(be::KernelBackend, b::KernelBinary)
    fmts = supported_formats(be)
    isempty(fmts) || binary_format(b) in fmts || throw(ArgumentError(
        "binary `$(b.file)`: $(nameof(typeof(be))) launches " *
        join(string.('.', fmts), " or ") * " files, not `.$(binary_format(b))`"
    ))
    reason = compatible_target(be, b)
    isnothing(reason) || throw(ArgumentError(reason))

    return nothing
end

# Can a cubin built for `built` ("sm_80") run on a device that is `device` ("sm_86")? Same major, and the
# device's minor at least the cubin's. `""` (portable) always can.
function cuda_sm_compatible(built::AbstractString, device::AbstractString)
    isempty(built) && return true
    pat = r"^sm_(\d+?)(\d)[af]?$"
    (mb, md) = (match(pat, built), match(pat, device))
    (isnothing(mb) || isnothing(md)) && return built == device
    return mb[1] == md[1] && parse(Int, md[2]) >= parse(Int, mb[2])
end

"""
    current_target() -> String

The architecture id ([`target_id`](@ref)) of the device this session's launches run on, which names the
subdirectory of the registry and of the `ka_compile` cache its binaries live in. `""` when no device backend is
loaded, or several are (then there is no single answer); computed once per backend. A session has one device
architecture: switching devices mid-session is not modelled.
"""
function current_target()
    length(LOADED_BACKENDS) == 1 || return ""
    return get!(() -> target_id(device_backend()), _TARGETS, only(LOADED_BACKENDS))
end

const _TARGETS = Dict{Symbol,String}()
