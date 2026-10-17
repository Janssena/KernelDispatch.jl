# Shared by the SDPAOps tests of both sections (cpu/ and gpu/): a reference implementation, test sizes,
# and the argument lists of the example's two binaries, each in its own order (see SDPAOps.jl).

using KernelDispatch, SDPAOps
import KernelDispatch: variant_key, extras

const KOI = KernelDispatch

# Independent of `SDPAOps.host`: one query at a time, in Float64.
function sdpa_reference(q, k, v)
    d, n = size(q)
    o = zeros(Float64, size(v, 1), n)
    for i in 1:n
        s = [sum(Float64.(q[:, i]) .* Float64.(k[:, j])) / sqrt(d) for j in 1:size(k, 2)]
        p = exp.(s .- maximum(s))
        o[:, i] = Float64.(v) * (p ./ sum(p))
    end
    return o
end

inputs(rng, d, n, m, dv) = (randn(rng, Float32, d, n), randn(rng, Float32, d, m), randn(rng, Float32, dv, m))

# (d, n, m, dv): `n` below, at, and above a threadgroup of 64; a single query and key.
const SDPA_SIZES = ((16, 100, 70, 8), (4, 1, 1, 1), (8, 64, 64, 4), (32, 129, 5, 16))

# `extras` of a length known only at run time: merged into the arguments, they would hide which
# argument is an output. Bound separately, they cannot reach `call_binary`'s result.
struct SDPADyn <: AbstractKernel end
KOI.kernelname(::SDPADyn) = :dyn
KOI.variant_key(op::SDPA, ::SDPADyn, args...) = variant_key(op, SDPAKA(), args...)
KOI.extras(::SDPA, ::SDPADyn, b::KernelBinary) = Tuple(Val(b.params.tg) for _ in 1:1)


# The binaries' argument lists, each in its own order (see SDPAOps.jl).
sdpa_scale(q) = inv(sqrt(Float32(size(q, 1))))
sdpa_binary_args(::FlashAttn, o, q, k, v) = (q, k, v, o, Int32(size(q, 2)), Int32(size(k, 2)),
    Int32(size(q, 1)), Int32(size(v, 1)), sdpa_scale(q))
sdpa_binary_args(::Union{SDPAKA,SDPADyn}, o, q, k, v) = (o, q, k, v, Int32(size(q, 1)), Int32(size(q, 2)),
    Int32(size(k, 2)), Int32(size(v, 1)), sdpa_scale(q))
