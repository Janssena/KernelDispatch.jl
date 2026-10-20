# Gradients through an op, as a user takes them: Enzyme differentiates a function that calls the op on DEVICE arrays, and the
# package's reverse rule runs the op's `backward`. For each element type the loaded backend supports, the device gradient is
# checked against the gradient Enzyme takes of the op's HOST implementation in Float64, from the same rounded inputs, and the
# launches the gradient made are checked with `record_bindings`: the forward kernel, then the backward kernel, and nothing else.
#
# Nothing here reaches into the package: an op, a seed, `Enzyme`, and the public `record_bindings` to see what was launched.
# (Enzyme through Metal.jl code OTHER than the rule is not supported, so the differentiated function is the op call alone and the
# cotangent is seeded by hand, as in the package's own tutorial.)
#
# Two ops: `Residual` (fixtures.jl), whose backward is a launched binary, so the test can see THAT kernel run; and SDPAOps'
# `SDPA`, whose backward is plain Julia on the device arrays, so the only launch is the forward.

using Test, Random, KernelDispatch, SDPAOps, Enzyme, BFloat16s
isdefined(@__MODULE__, :Residual) || include(joinpath(@__DIR__, "..", "..", "fixtures.jl"))
isdefined(@__MODULE__, :sdpa_reference) || include(joinpath(@__DIR__, "..", "..", "sdpa_helpers.jl"))
isdefined(@__MODULE__, :ELEMENTWISE_RTOL) || include(joinpath(@__DIR__, "..", "..", "precision.jl"))

# --- the two gradients ---------------------------------------------------------------------------------------

# Reverse through `f` on device arrays: the primal and, for each array argument, its gradient for the output cotangent `dz`.
# Scalars are constants.
function device_pullback(f, args::Tuple, dz)
    anns = map(a -> a isa AbstractArray ? Duplicated(a, zero(a)) : Const(a), args)
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f)}, Duplicated, map(typeof, anns)...)
    tape, primal, shadow = fwd(Const(f), anns...)
    shadow .= dz
    rev(Const(f), anns..., tape)
    return primal, map(a -> a isa Duplicated ? a.dval : nothing, anns)
end

# Reverse through the op's host implementation on Float64 copies of the same arguments: the reference.
function host_pullback(f, args::Tuple, dz)
    args64 = map(a -> a isa AbstractArray ? Float64.(a) : a, args)
    w = Float64.(dz)
    shadows = map(a -> a isa AbstractArray ? zero(a) : nothing, args64)
    anns = map((a, s) -> s === nothing ? Const(a) : Duplicated(a, s), args64, shadows)
    loss(a...) = sum(f(a...) .* w)
    Enzyme.autodiff(Reverse, Const(loss), Active, anns...)
    return f(args64...), shadows
end

# What a call to `f` launches, without running it: the labels of the binaries, in order, and the binaries.
launched(f) = [(; r.label, r.launch.binary) for r in KD.record_bindings(f)]

@testset "Residual: the gradient of a binary forward and a binary backward" begin
    rng = Xoshiro(51)
    n = 50
    a = 3.0f0
    @testset "$T" for T in supported_eltypes()
        register_residual!(T)
        key = (nameof(T),)
        x, y, dz = (rounded(T, randn(rng, Float32, n)) for _ in 1:3)
        args = (DeviceArray(x), DeviceArray(y), a)
        # through the op call (its selected kernel) and through each kernel directly: all reach the rule
        fs = (:op => ((x, y, a) -> Residual()(x, y, a)),
              :ka => ((x, y, a) -> KD.forward(Residual(), ResidualKA(), x, y, a)),
              :plain => ((x, y, a) -> KD.forward(Residual(), ResidualPlain(), x, y, a)))
        @testset "$name" for (name, f) in fs
            primal, (gx, gy, _) = device_pullback(f, args, DeviceArray(dz))
            hprimal, (hx, hy, _) = host_pullback(Residual(), (x, y, a), dz)   # the op on host arrays: its `host`
            tol = ELEMENTWISE_RTOL[T]
            @test eltype(primal) === T && eltype(gx) === T && eltype(gy) === T
            @test maxrelerr(Array(primal), hprimal) <= tol
            @test maxrelerr(Array(gx), hx) <= tol
            @test maxrelerr(Array(gy), hy) <= tol

            # the gradient launched the forward kernel, then the backward kernel, once each
            ls = launched(() -> device_pullback(f, args, DeviceArray(dz)))
            @test [l.label for l in ls] == [:fwd, :bwd]
            kernel = name === :plain ? ResidualPlain() : ResidualKA()
            variant = KD.variant(Residual(), kernel, key)
            @test ls[1].binary.entry == variant[:fwd].entry
            @test ls[2].binary.entry == variant[:bwd].entry
            @test ls[2].binary.entry != ls[1].binary.entry
        end
    end
end

@testset "SDPA: the gradient of a binary forward and a plain-Julia backward" begin
    rng = Xoshiro(52)
    d, n, m, dv = 8, 12, 9, 4
    @testset "$kern, $T" for kern in (prebuilt_sdpa_kernels()..., SDPAKA()), T in supported_eltypes()
        q, k, v = inputs(rng, d, n, m, dv, T)
        dz = rounded(T, randn(rng, Float32, dv, n))
        f(q, k, v) = KD.forward(SDPA(), kern, q, k, v)
        args = map(DeviceArray, (q, k, v))
        primal, grads = device_pullback(f, args, DeviceArray(dz))
        hprimal, hgrads = host_pullback(SDPA(), (q, k, v), dz)
        tol = REDUCTION_RTOL[T]
        @test eltype(primal) === T && all(g -> eltype(g) === T, grads)
        @test maxrelerr(Array(primal), hprimal) <= tol
        for (g, h) in zip(grads, hgrads)
            @test maxrelerr(Array(g), h) <= tol
        end

        # SDPAOps' backward is Julia code on the device arrays, so the gradient launches the forward kernel and nothing more
        ls = launched(() -> device_pullback(f, args, DeviceArray(dz)))
        @test [l.label for l in ls] == [:fwd]
    end
end
