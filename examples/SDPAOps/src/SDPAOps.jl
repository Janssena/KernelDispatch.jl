"""
    SDPAOps

The example package of the KernelDispatch tutorials (see the tutorials in `docs/src/tutorials/`): scaled dot-product attention as
an op with two kernels — a hand-written Metal binary (standing in for a Triton one; Metal only) and a
KernelAbstractions kernel compiled to a binary (on whichever GPU backend is loaded).

    o = SDPA()(q, k, v)     # q: d×n, k: d×m, v: dv×m  →  o: dv×n (one column per query)

On the device it runs in the element type of its inputs: `Float32`, `Float16` or `BFloat16` (one variant per type,
keyed `(:Float32,)`, `(:Float16,)`, `(:BFloat16,)`). The half types read and write that type but accumulate in
`Float32` (the scores, the softmax denominator and the weighted sum of values), and convert once at the end: the
kernels take a `Float32` workspace for the sum, a second output that `forward` drops. `Float32` needs no workspace
and accumulates in the output itself. On the host the op is plain Julia and returns whatever that gives.
"""
module SDPAOps

using KernelDispatch, KernelAbstractions, LinearAlgebra
using BFloat16s: BFloat16
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

# The element types with a variant, by the name their key uses.
const ELTYPES = (Float32=Float32, Float16=Float16, BFloat16=BFloat16)

# The one element type of the three inputs: a mix has no variant.
function _eltype(q, k, v)
    T = KernelDispatch.realtype(eltype(q))
    (KernelDispatch.realtype(eltype(k)) === T && KernelDispatch.realtype(eltype(v)) === T) || throw(ArgumentError(
        "q, k and v must have one element type, got $(eltype(q)), $(eltype(k)), $(eltype(v))"))
    return T
end

# The binary's arguments, in its order: inputs, the output, sizes, scale (see kernels/sdpa.metal).
# One item per query: `extent` is divided into threadgroups by the binary's `tile`.
function forward(op::SDPA, kern::FlashAttn, q, k, v)
    T = _eltype(q, k, v)
    d, n, m, dv = _sizes(q, k, v)
    scale = _scale(q)
    if T === Float32               # accumulates in the output itself
        result = call_binary(op, kern, :fwd, q, k, v, OutArray(T, dv, n), n, m, d, dv, scale; extent=n)
    else                           # a half type: the sum is kept in a Float32 workspace (the second output)
        result = call_binary(op, kern, :fwd, q, k, v, OutArray(T, dv, n), OutArray(Float32, dv, n),
            n, m, d, dv, scale; extent=n)
    end
    return first(result)
end
# `variant_key` sees the arguments as `call_binary` got them: this binary's order. One variant per
# element type (`realtype` also names it for a traced array under Reactant).
variant_key(::SDPA, ::FlashAttn, q, k, v, o, n, m, d, dv, scale) = (
    nameof(KernelDispatch.realtype(eltype(q))),
)
variant_key(::SDPA, ::FlashAttn, q, k, v, o, acc, n, m, d, dv, scale) = (
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

# The same for a half type: the buffers are `T`, every sum is `Float32`. The weighted sum of values lives in `acc`,
# a `Float32` workspace (a second output), and `o` is written once, converted, at the end.
@kernel unsafe_indices = true function sdpa_ka_acc!(o, acc, q, k, v, d::Int32, n::Int32, m::Int32, dv::Int32,
        scale::Float32, ::Val{TG}) where {TG}
    i = (@index(Group, Linear) - 1) * TG + @index(Local, Linear)
    if i <= n
        qi, oi = (i - 1) * d, (i - 1) * dv
        mx = -Inf32
        for j in 1:m
            s = 0.0f0
            for t in 1:d
                s += Float32(unsafe_load(q, qi + t)) * Float32(unsafe_load(k, (j - 1) * d + t))
            end
            mx = max(mx, s * scale)
        end
        for c in 1:dv
            unsafe_store!(acc, 0.0f0, oi + c)
        end
        denom = 0.0f0
        for j in 1:m
            s = 0.0f0
            for t in 1:d
                s += Float32(unsafe_load(q, qi + t)) * Float32(unsafe_load(k, (j - 1) * d + t))
            end
            p = exp(s * scale - mx)
            denom += p
            for c in 1:dv
                unsafe_store!(acc, unsafe_load(acc, oi + c) + p * Float32(unsafe_load(v, (j - 1) * dv + c)), oi + c)
            end
        end
        for c in 1:dv
            unsafe_store!(o, unsafe_load(acc, oi + c) / denom, oi + c)    # converted to `T` by the store
        end
    end
end

function forward(op::SDPA, kern::SDPAKA, q, k, v)
    T = _eltype(q, k, v)
    d, n, m, dv = _sizes(q, k, v)
    scale = _scale(q)
    if T === Float32
        result = call_binary(op, kern, :fwd, OutArray(T, dv, n), q, k, v, d, n, m, dv, scale; extent=n)
    else
        result = call_binary(op, kern, :fwd, OutArray(T, dv, n), OutArray(Float32, dv, n), q, k, v,
            d, n, m, dv, scale; extent=n)
    end
    return first(result)
end
variant_key(::SDPA, ::SDPAKA, o, q, k, v, d, n, m, dv, scale) = (
    nameof(KernelDispatch.realtype(eltype(q))),
)
variant_key(::SDPA, ::SDPAKA, o, acc, q, k, v, d, n, m, dv, scale) = (
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

# The entry point of each element type in kernels/sdpa.metallib.
const FLASH_ENTRIES = (Float32="sdpa_fwd", Float16="sdpa_fwd_f16", BFloat16="sdpa_fwd_bf16")

# A key is `(:Float32,)`, `(:Float16,)` or `(:BFloat16,)`; any other has no variant.
_key_eltype(key::Tuple) = (length(key) == 1 && key[1] isa Symbol && haskey(ELTYPES, key[1])) ? ELTYPES[key[1]] : nothing

function build!(op::SDPA, kern::FlashAttn, key::Tuple)
    isnothing(_key_eltype(key)) && return nothing
    # A `.metallib` runs on Metal only: on another backend there is no binary to register.
    KernelDispatch.device_backend() isa KernelDispatch.MetalBackendTag || return nothing
    T = _key_eltype(key)
    # The types `forward` binds for this kernel (see `forward(::SDPA, ::FlashAttn, ...)`): q, k, v and the output in `T`,
    # then, for a half type, the Float32 workspace, then the sizes and the scale.
    sig = T === Float32 ?
        (Vector{T}, Vector{T}, Vector{T}, Vector{T}, Int32, Int32, Int32, Int32, Float32) :
        (Vector{T}, Vector{T}, Vector{T}, Vector{T}, Vector{Float32}, Int32, Int32, Int32, Int32, Float32)
    add_variant!(op, kern, key; fwd=KernelBinary(METALLIB; entry=FLASH_ENTRIES[key[1]], threadgroup=64, signature=sig))
    return nothing
end

function build!(op::SDPA, kern::SDPAKA, key::Tuple)
    T = _key_eltype(key)
    isnothing(T) && return nothing
    tg = 64
    if T === Float32
        argtypes = (Vector{Float32}, Vector{Float32}, Vector{Float32}, Vector{Float32},
            Int32, Int32, Int32, Int32, Float32, Val{tg})
        kernel = sdpa_ka!
    else
        argtypes = (Vector{T}, Vector{Float32}, Vector{T}, Vector{T}, Vector{T},
            Int32, Int32, Int32, Int32, Float32, Val{tg})
        kernel = sdpa_ka_acc!
    end
    # The binary's file name gets the signature of `argtypes` (`f16x...`), so the variants never share one.
    fwd = ka_compile(KernelDispatch.device_backend(), kernel, argtypes;
        tg, name="sdpa_fwd_$(source_tag(kern))", params=(; tg))
    add_variant!(op, kern, key; fwd)
    return nothing
end

# --- gradients ------------------------------------------------------------------------------------

# Plain Julia on whatever arrays the forward got (device arrays included): no binary needed. One
# gradient per argument of `forward`, in order.
# A half type is computed in Float32 and the gradients converted back, so they come back in the inputs' type.
_f32(x) = KernelDispatch.realtype(eltype(x)) === Float32 ? x : Float32.(x)
_back(T, g) = T === Float32 ? g : T.(g)

function backward(::SDPA, ::SDPAKernels, (q, k, v), o, dout)
    T = KernelDispatch.realtype(eltype(q))
    qf, kf, vf, df = _f32(q), _f32(k), _f32(v), _f32(dout)
    scale = _scale(qf)
    p = _softmax((kf' * qf) .* scale)
    dp = vf' * df
    ds = p .* (dp .- sum(p .* dp; dims=1))
    return (_back(T, (kf * ds) .* scale), _back(T, (qf * ds') .* scale), _back(T, df * p'))
end

end # module
