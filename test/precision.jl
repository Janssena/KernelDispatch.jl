# The element types the suite covers and how closely a result in each must match a Float64 reference computed from the
# SAME rounded inputs (docs/half-precision.md, "Tests"). Starting points, to be tightened against measurements; not
# derived bounds.

using BFloat16s: BFloat16

const ALL_ELTYPES = (Float32, Float16, BFloat16)

# Largest error relative to the largest reference value (an element-wise `rtol` would blow up around zero).
const ELEMENTWISE_RTOL = Dict(Float32 => 1e-6, Float16 => 2e-3, BFloat16 => 1.5e-2)
const REDUCTION_RTOL = Dict(Float32 => 1e-4, Float16 => 1e-2, BFloat16 => 5e-2)      # attention-like sums

maxrelerr(got, ref) = maximum(abs.(Float64.(got) .- Float64.(ref))) / max(maximum(abs.(Float64.(ref))), eps())

"`x` rounded to `T`: what a kernel in `T` sees, so a reference computed from it has no input rounding error."
rounded(::Type{T}, x) where {T} = T.(x)
