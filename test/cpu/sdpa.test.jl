# The tutorial example package (examples/SDPAOps) as the integration fixture: a real op with two
# real kernels (a prebuilt `.metallib` and a KernelAbstractions kernel) and a plain-Julia `host` and
# `backward`. This file is the part that needs no device; gpu/sdpa.test.jl runs the kernels. The fixture
# ops in `fixtures.jl` cover what it cannot (a backward binary, registry edge cases, binary-level launch
# features); this file checks that the machinery composes into what the tutorials teach.
#
#   host        no device needed: against a column-by-column reference
#   backward    no device needed: against finite differences

using Test, Random, KernelDispatch, SDPAOps, BFloat16s
isdefined(@__MODULE__, :REDUCTION_RTOL) || include(joinpath(@__DIR__, "..", "precision.jl"))
isdefined(@__MODULE__, :Residual) || include(joinpath(@__DIR__, "..", "fixtures.jl"))     # also keeps SDPAOps' manifests out of the user's cache
isdefined(@__MODULE__, :sdpa_reference) || include(joinpath(@__DIR__, "..", "sdpa_helpers.jl"))

@testset "host" begin
    rng = Xoshiro(1)
    for (d, n, m, dv) in SDPA_SIZES
        q, k, v = inputs(rng, d, n, m, dv)
        o = SDPA()(q, k, v)                         # host arrays: the host path
        @test size(o) == (dv, n)
        @test o ≈ sdpa_reference(q, k, v) rtol = 1e-4
    end
end

@testset "host, half types" begin
    # The host path is plain Julia: whatever `v * softmax(k'q)` gives for these arrays, which the caller casts as it likes
    # (here it comes back wider than the inputs). It must be close to the Float64 reference from the same rounded inputs.
    rng = Xoshiro(5)
    for T in (Float16, BFloat16), (d, n, m, dv) in SDPA_SIZES
        q, k, v = inputs(rng, d, n, m, dv, T)
        o = SDPA()(q, k, v)
        @test size(o) == (dv, n)
        @test maxrelerr(o, sdpa_reference(q, k, v)) <= REDUCTION_RTOL[T]
    end
end

@testset "backward against finite differences" begin
    rng = Xoshiro(2)
    d, n, m, dv = 4, 3, 5, 2
    q, k, v = (Float64.(x) for x in inputs(rng, d, n, m, dv))
    w = randn(rng, dv, n)
    loss(q, k, v) = sum(KernelDispatch.host(SDPA(), q, k, v) .* w)
    dq, dk, dv_ = KernelDispatch.backward(SDPA(), FlashAttn(), (q, k, v), nothing, w)
    function fd(f, x, i; h=1e-6)
        xp, xm = copy(x), copy(x)
        xp[i] += h
        xm[i] -= h
        return (f(xp) - f(xm)) / 2h
    end
    for i in eachindex(q)
        @test dq[i] ≈ fd(x -> loss(x, k, v), q, i) atol = 1e-6
    end
    for i in eachindex(k)
        @test dk[i] ≈ fd(x -> loss(q, x, v), k, i) atol = 1e-6
    end
    for i in eachindex(v)
        @test dv_[i] ≈ fd(x -> loss(q, k, x), v, i) atol = 1e-6
    end
    # The gradients do not depend on which kernel produced the forward.
    @test KernelDispatch.backward(SDPA(), SDPAKA(), (q, k, v), nothing, w) == (dq, dk, dv_)
end
