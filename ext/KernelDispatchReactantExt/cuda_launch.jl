# Traced CUDA launches: a `call_binary` under Reactant becomes a `stablehlo.custom_call` to XLA's
# `triton_kernel_call` handler. Despite the name the handler is not Triton-specific at launch time: given a
# kernel image (a cubin, or PTX it assembles itself), a kernel name, a grid, a thread count and a list of
# parameters (device buffers and plain scalars), it loads the image once, binds the buffers XLA hands it
# and calls `cuLaunchKernel` on XLA's own stream. KernelDispatch speaks its protobuf (jaxlib's
# `triton.proto`: `TritonAnyKernelCall`) directly, so there is no Python and no native code of our own.
#
# What the handler cannot do bounds what can be launched this way:
#   - parameters are device buffers and bool/32-bit/64-bit scalars: no by-value structs. That is why CUDA's
#     KernelAbstractions binaries (`abi = :ka`) are built flat (no `KernelState`, KA's context as six integers; see
#     the CUDA extension), and why plain pointer-and-scalar binaries (nvcc, Triton, hand-written PTX) work.
#   - threads per block are `num_warps * 32`, one-dimensional: `threadgroup` is a multiple of 32.
#   - XLA hands the handler its operands, then its results, as buffers; each buffer parameter takes the
#     next one in that order. So a kernel's buffer parameters must list every input before every output.
#
# Confirmed with `test/cuda_probe.jl` on an NVIDIA machine: Reactant's CUDA client launches a plain cubin or PTX
# through `triton_kernel_call_ffi@4` (the typed-FFI form jax-triton uses; the proto travels in an `opaque`
# attribute), which is the default. `triton_kernel_call@2` launched a cubin but not PTX there, and
# `triton_kernel_call@4` was rejected. The convention is selectable for other XLA builds:
# `KERNELDISPATCH_CUDA_CUSTOM_CALL=<target>@<api_version>`; the probe tries the combinations.
#
# Note for Reactant's own client: its compile pipeline has a `raise_triton_custom_call` pass that rewrites a
# custom call named exactly `triton_kernel_call` into a Triton-dialect op and compiles the Triton IR in the
# payload itself (and fails for a payload without any). The `_ffi` name is not matched, which is why it is the
# default; to use the legacy name anyway, compile with
# `CompileOptions(; raise_triton_custom_call=false)`.

# --- protobuf, just enough ----------------------------------------------------------------------------

function _pb_varint(io::IO, x::Integer)
    u = x % UInt64
    while u >= 0x80
        write(io, UInt8((u & 0x7f) | 0x80))
        u >>= 7
    end
    write(io, UInt8(u))
    return nothing
end

_pb_tag(io::IO, field::Integer, wire::Integer) = _pb_varint(io, (field << 3) | wire)

# Proto3 omits fields at their default; so do we.
function _pb_uint(io::IO, field::Integer, x::Integer)
    iszero(x) || (_pb_tag(io, field, 0); _pb_varint(io, x))
    return nothing
end

function _pb_bytes(io::IO, field::Integer, data::AbstractVector{UInt8}; always::Bool=false)
    (isempty(data) && !always) && return nothing
    _pb_tag(io, field, 2)
    _pb_varint(io, length(data))
    write(io, data)
    return nothing
end

_pb_message(f, io::IO, field::Integer) = (b = IOBuffer(); f(b); _pb_bytes(io, field, take!(b); always=true))

# A zlib stream (RFC 1950) of "stored" deflate blocks: valid input for any inflater, with no compression
# and no dependency on a zlib binding.
function _zlib_stored(data::AbstractVector{UInt8})
    out = IOBuffer()
    write(out, 0x78, 0x01)
    n = length(data)
    pos = 1
    while true
        len = min(n - pos + 1, 0xffff)
        final = pos + len > n
        write(out, UInt8(final))
        write(out, htol(UInt16(len)))              # little-endian LEN, then NLEN
        write(out, htol(~UInt16(len)))
        write(out, view(data, pos:(pos+len-1)))
        pos += len
        final && break
    end
    a, b = UInt32(1), UInt32(0)
    for x in data
        a = (a + x) % UInt32(65521)
        b = (b + a) % UInt32(65521)
    end
    write(out, hton((b << 16) | a))               # Adler-32, big-endian
    return take!(out)
end

# --- TritonAnyKernelCall ------------------------------------------------------------------------------

"""
    _kernel_call_proto(; entry, image, is_ptx, num_warps, shared_mem, grid, params, name) -> Vector{UInt8}

jaxlib's `TritonAnyKernelCall` for one kernel launch. `params` is a vector of `(:array, bytes_to_zero)` and
`(:scalar, bytes)` in the kernel's parameter order; a scalar's `bytes` are its little-endian bytes, passed
to the kernel by address, so only their size (1, 4 or 8) matters, not their type.
"""
function _kernel_call_proto(; entry, image, is_ptx, num_warps, shared_mem, grid, params, name)
    kernel = IOBuffer()
    _pb_bytes(kernel, 1, codeunits(String(entry)))                       # kernel_name
    _pb_uint(kernel, 2, num_warps)
    _pb_uint(kernel, 3, shared_mem)
    if is_ptx
        _pb_bytes(kernel, 4, image)                                      # ptx
    else
        _pb_message(kernel, 11) do mi                                    # module_image
            _pb_bytes(mi, 1, image)                                      # object_file
            _pb_bytes(mi, 2, codeunits(String(entry)))
            _pb_uint(mi, 3, shared_mem)
        end
    end

    call = IOBuffer()
    _pb_message(io -> write(io, take!(kernel)), call, 1)                 # kernel
    for (i, g) in enumerate(grid)
        _pb_uint(call, 1 + i, g)                                         # grid_0..2
    end
    for (kind, v) in params
        _pb_message(call, 5) do p                                        # parameters
            if kind === :array
                _pb_message(p, 1) do a
                    _pb_uint(a, 1, v)                                    # bytes_to_zero
                    _pb_uint(a, 2, 1)                                    # ptr_divisibility: no constraint
                end
            elseif length(v) == 1
                _pb_tag(p, 2, 0); _pb_varint(p, v[1])                    # bool_
            elseif length(v) == 4
                _pb_tag(p, 4, 0); _pb_varint(p, reinterpret(UInt32, v)[1])   # u32: same bits as an i32/f32
            elseif length(v) == 8
                _pb_tag(p, 6, 0); _pb_varint(p, reinterpret(UInt64, v)[1])   # u64
            else
                throw(ArgumentError(
                    "a $(length(v))-byte scalar argument cannot be passed to XLA's kernel-call handler " *
                    "(it takes 1-, 4- and 8-byte scalars); widen it in `forward`"))
            end
        end
    end

    any_call = IOBuffer()
    _pb_message(io -> write(io, take!(call)), any_call, 1)               # kernel_call
    _pb_bytes(any_call, 4, codeunits(String(name)))                      # name
    return take!(any_call)
end

# --- the custom call ------------------------------------------------------------------------------------

# "triton_kernel_call_ffi@4" -> ("triton_kernel_call_ffi", 4)
function _cuda_call_convention()
    spec = get(ENV, "KERNELDISPATCH_CUDA_CUSTOM_CALL", "triton_kernel_call_ffi@4")
    i = findfirst('@', spec)
    i === nothing && return (String(spec), 4)
    return (String(spec[1:i-1]), parse(Int, spec[i+1:end]))
end

# The bytes of a `(:bytes, payload)` layout entry: `payload` is the "[1,2,3]" string `_le_bytes` made.
_payload_bytes(payload::AbstractString) =
    UInt8[parse(UInt8, strip(x)) for x in split(strip(payload, ['[', ']']), ',') if !isempty(strip(x))]

# The kernel's parameters in order, and the check that its buffers list inputs before outputs.
function _cuda_params(layout, result_shapes, result_eltypes)
    params = Tuple{Symbol,Any}[]
    buffers = Tuple{Symbol,Int}[]
    for (kind, val) in layout
        if kind === :bytes
            push!(params, (:scalar, _payload_bytes(val)))
        elseif kind === :in
            push!(params, (:array, 0)); push!(buffers, (:in, val))
        else                                       # :out or :out_zero
            zero_bytes = kind === :out_zero ? prod(result_shapes[val+1]; init=1) * sizeof(result_eltypes[val+1]) : 0
            push!(params, (:array, zero_bytes)); push!(buffers, (:out, val))
        end
    end
    ins = [i for (k, i) in buffers if k === :in]
    outs = [i for (k, i) in buffers if k === :out]
    expected = vcat([(:in, i) for i in 0:length(ins)-1], [(:out, i) for i in 0:length(outs)-1])
    buffers == expected || throw(ArgumentError(
        "XLA hands a kernel call its inputs and then its outputs as buffers, in order, so the binary's " *
        "buffer parameters must list every input before every output (and each group in order); got " *
        "$(join(string.(first.(buffers), ":", last.(buffers)), ", ")). Reorder the kernel's parameters."))
    return params
end

"""
    KernelDispatch.custom_call(::CUDABackendTag, entry, path, operands, result_shapes, layout; grid,
        threadgroup, result_eltypes, shared_mem=0)

The CUDA embedding: a `stablehlo.custom_call` to a `triton_kernel_call` handler launching the cubin (or
`.ptx`) at `path`; see this file's header for what it can and cannot launch, and that it is unverified.
"""
function KO.custom_call(::KO.CUDABackendTag, entry, path, operands, result_shapes, layout;
        grid, threadgroup, result_eltypes=ntuple(Returns(Float32), length(result_shapes)),
        shared_mem::Integer=0, reverse=nothing, active_operands=nothing)
    (threadgroup % 32 == 0 && 0 < threadgroup <= 1024) || throw(ArgumentError(
        "threadgroup $threadgroup: XLA's kernel-call handler launches `num_warps * 32` threads per block, " *
        "so it must be a multiple of 32 and at most 1024"))
    isfile(path) || throw(ArgumentError("kernel image `$path` not found"))

    proto = _kernel_call_proto(; entry, image=read(path), is_ptx=lowercase(splitext(path)[2]) == ".ptx",
        num_warps=threadgroup ÷ 32, shared_mem, grid,
        params=_cuda_params(layout, result_shapes, result_eltypes), name=entry)
    opaque = String(_zlib_stored(proto))
    target, api = _cuda_call_convention()
    config = api >= 4 ? Dict("opaque" => opaque) : opaque
    return _embed_custom_call(target, operands, result_shapes; backend_config=config,
        api_version=api, reverse, active_operands, label=entry, result_eltypes)
end
