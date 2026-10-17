"""
    SDPAOps

The example package of the KernelDispatch tutorials (see the tutorials in `docs/src/tutorials/`): scaled dot-product attention as
an op with two kernels — a hand-written Metal binary (standing in for a Triton one; Metal only) and a
KernelAbstractions kernel compiled to a binary (on whichever GPU backend is loaded).

    o = SDPA()(q, k, v)     # q: d×n, k: d×m, v: dv×m  →  o: dv×n (one column per query)
"""
module SDPAOps

using KernelDispatch, KernelAbstractions, LinearAlgebra
import KernelDispatch: opname, kernelname, host, forward, backward, variant_key, extras, build!,
    source_tag

export SDPA, FlashAttn, SDPAKA

# --- the op ---------------------------------------------------------------------------------------

struct SDPA <: AbstractKernelOp end
opname(::SDPA) = :sdpa

_scale(q) = inv(sqrt(Float32(size(q, 1))))
_softmax(s) = (p = exp.(s .- maximum(s; dims=1)); p ./ sum(p; dims=1))

# The op on host arrays: plain Julia.
host(::SDPA, q, k, v) = v * _softmax((k' * q) .* _scale(q))

# --- the kernels ----------------------------------------------------------------------------------

struct FlashAttn <: AbstractKernel end      # a prebuilt Metal binary: kernels/sdpa.metallib
struct SDPAKA <: AbstractKernel end         # a KernelAbstractions kernel, `sdpa_ka!` below
kernelname(::FlashAttn) = :flash
kernelname(::SDPAKA) = :ka

const SDPAKernels = Union{FlashAttn,SDPAKA}       # for what both share: the backward

# `:flash` unless the package's preference names `:ka` (`set_default_kernel!`).
KernelDispatch.@default_kernel SDPA FlashAttn() SDPAKA()

_sizes(q, k, v) = (Int32(size(q, 1)), Int32(size(q, 2)), Int32(size(k, 2)), Int32(size(v, 1)))

# The binary's arguments, in its order: inputs, the output, sizes, scale (see kernels/sdpa.metal).
# One item per query: `extent` is divided into threadgroups by the binary's `tile`.
function forward(op::SDPA, kern::FlashAttn, q, k, v)
    d, n, m, dv = _sizes(q, k, v)
    out = OutArray(Float32, dv, n)
    scale = _scale(q)
    result = call_binary(op, kern, :fwd, q, k, v, out, n, m, d, dv, scale; extent=n)  # returns a tuple
    return only(result)
end
# `variant_key` sees the arguments as `call_binary` got them: this binary's order. One variant per
# element type (`realtype` also names it for a traced array under Reactant).
variant_key(::SDPA, ::FlashAttn, q, k, v, o, n, m, d, dv, scale) = (
    nameof(KernelDispatch.realtype(eltype(q))),
)

# The KernelAbstractions kernel: the output first, its own order of sizes, the tile size a `Val`.
@kernel unsafe_indices = true function sdpa_ka!(o, q, k, v, d::Int32, n::Int32, m::Int32, dv::Int32,
        scale::Float32, ::Val{TG}) where {TG}
    i = (@index(Group, Linear) - 1) * TG + @index(Local, Linear)
    if i <= n
        qi, oi = (i - 1) * d, (i - 1) * dv              # where column `i` starts
        mx = -Inf32
        for j in 1:m
            s = 0.0f0
            for t in 1:d
                s += unsafe_load(q, qi + t) * unsafe_load(k, (j - 1) * d + t)
            end
            mx = max(mx, s * scale)
        end
        for c in 1:dv
            unsafe_store!(o, 0.0f0, oi + c)
        end
        denom = 0.0f0
        for j in 1:m
            s = 0.0f0
            for t in 1:d
                s += unsafe_load(q, qi + t) * unsafe_load(k, (j - 1) * d + t)
            end
            p = exp(s * scale - mx)
            denom += p
            for c in 1:dv
                unsafe_store!(o, unsafe_load(o, oi + c) + p * unsafe_load(v, (j - 1) * dv + c), oi + c)
            end
        end
        for c in 1:dv
            unsafe_store!(o, unsafe_load(o, oi + c) / denom, oi + c)
        end
    end
end

function forward(op::SDPA, kern::SDPAKA, q, k, v)
    d, n, m, dv = _sizes(q, k, v)
    out = OutArray(Float32, dv, n)
    scale = _scale(q)
    result = call_binary(op, kern, :fwd, out, q, k, v, d, n, m, dv, scale; extent=n)
    return only(result)
end
variant_key(::SDPA, ::SDPAKA, o, q, k, v, d, n, m, dv, scale) = (
    nameof(KernelDispatch.realtype(eltype(q))),
)
# The tile size is compiled in: appended as a `Val`, it takes no argument slot.
extras(::SDPA, ::SDPAKA, b::KernelBinary) = (Val(b.params.tg),)

# --- building variants on first use ---------------------------------------------------------------

const METALLIB = joinpath(@__DIR__, "..", "kernels", "sdpa.metallib")
include_dependency(METALLIB)        # a new binary re-precompiles the package, and so its tag below

# A manifest recorded under another tag is ignored, so changed kernels are rebuilt, not reused.
source_tag(::FlashAttn) = string(hash(read(METALLIB)); base=36)
source_tag(::SDPAKA) = string(hash(read(@__FILE__)); base=36)

function build!(op::SDPA, kern::FlashAttn, key::Tuple)
    key == (:Float32,) || return nothing                # the binary is Float32 only
    # A `.metallib` runs on Metal only: on another backend there is no binary to register.
    KernelDispatch.device_backend() isa KernelDispatch.MetalBackendTag || return nothing
    add_variant!(op, kern, key; fwd=KernelBinary(METALLIB; entry="sdpa_fwd", threadgroup=64))
    return nothing
end

function build!(op::SDPA, kern::SDPAKA, key::Tuple)
    key == (:Float32,) || return nothing
    tg = 64
    argtypes = (Vector{Float32}, Vector{Float32}, Vector{Float32}, Vector{Float32},
        Int32, Int32, Int32, Int32, Float32, Val{tg})
    fwd = ka_compile(KernelDispatch.device_backend(), sdpa_ka!, argtypes;
        tg, name="sdpa_fwd_$(source_tag(kern))", params=(; tg))
    add_variant!(op, kern, key; fwd)
    return nothing
end

# --- gradients ------------------------------------------------------------------------------------

# Plain Julia on whatever arrays the forward got (device arrays included): no binary needed. One
# gradient per argument of `forward`, in order.
function backward(::SDPA, ::SDPAKernels, (q, k, v), o, dout)
    scale = _scale(q)
    p = _softmax((k' * q) .* scale)
    dp = v' * dout
    ds = p .* (dp .- sum(p .* dp; dims=1))
    return ((k * ds) .* scale, (q * ds') .* scale, dout * p')
end

end # module
