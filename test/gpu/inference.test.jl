# Type stability on the device: routing on the backend's arrays, the KA hooks, and the launch paths.
# `@inferred` alone for launches (a second run would dispatch a second kernel); `@test_nowarn` where a
# warning is the failure. The op is SDPAOps: `FlashAttn` is an `:explicit` (prebuilt, Metal-only) binary where the backend has prebuilt kernels,
# `SDPAKA` a KA binary with `extras`. The device-free half is in cpu/inference.test.jl.
#
# Each testset runs once per element type the backend supports (`supported_eltypes()`: Float32, Float16, BFloat16 on
# Metal). One loop does both jobs: every launch is `@inferred`, and its result is checked for its element type and against
# a Float64 reference from the same rounded inputs, within that type's tolerance (test/precision.jl).

using Test, KernelDispatch, SDPAOps, KernelAbstractions, BFloat16s, Random
isdefined(@__MODULE__, :ALL_ELTYPES) || include(joinpath(@__DIR__, "..", "precision.jl"))
isdefined(@__MODULE__, :Residual) || include(joinpath(@__DIR__, "..", "fixtures.jl"))
isdefined(@__MODULE__, :sdpa_binary_args) || include(joinpath(@__DIR__, "..", "sdpa_helpers.jl"))

# The kernels with a binary on this backend: `FlashAttn` is a `.metallib`.
const INFER_KERNELS = (prebuilt_sdpa_kernels()..., SDPAKA(), SDPADyn())

@testset "routing on device arrays, $T" for T in supported_eltypes()
    W = wrapped_array_type()
    x = DeviceArray(T.(rand(Float32, 4, 6)))
    tag = GPU_TAG
    plain = (x, reshape(x, 6, 4), view(x, :, 2:3))        # contiguous: the backend hands back a plain array
    wrapped = (view(x, 2:3, :), x', transpose(x))         # these stay wrappers
    for a in plain
        @test @inferred(KD.backend_of(a)) === tag
        @test @inferred(KD.device_dense(tag, a)) === a
    end
    for a in wrapped
        @test a isa W
        @test @inferred(KD.backend_of(a)) === tag
        @test @inferred(KD.device_dense(tag, a)) isa DeviceArray
    end
    @test @inferred(KD.backend_of(x, rand(T, 3))) === tag     # the first array decides
    @test @inferred(KD.device_dense(tag, 2.0f0)) === 2.0f0
end

@testset "KA backend hooks" begin
    tag = GPU_TAG
    words = @inferred KD.ka_state_words(tag)
    @test words isa Tuple
    @test words === EXPECTED_KA_STATE_WORDS                  # the backend's hidden leading arguments
    @test KD.ka_backend_object(tag) isa KernelAbstractions.Backend
    @test @inferred(KD.ka_grid(tag, (2, 3, 4))) === (24, 1, 1)             # flat
    ctx, ghost = KD.ka_context(tag, (2, 1, 1), 64)
    @test ghost isa Bool
    # No extension, no default: a backend that forgets a hook fails loudly. ROCm has no extension.
    rocm = KD.ROCmBackendTag()
    for f in (KD.ka_state_words, KD.ka_backend_object)
        @test_throws ArgumentError f(rocm)
    end
    @test_throws ArgumentError KD.ka_grid(rocm, (1, 1, 1))
    @test_throws ArgumentError KD.ka_context(rocm, (1, 1, 1), 64)
end

@testset "launch path, $T" for T in supported_eltypes()
    d, n, m, dv = 16, 8, 6, 4
    q, k, v = inputs(Random.Xoshiro(21), d, n, m, dv, T)
    qd, kd, vd = DeviceArray(q), DeviceArray(k), DeviceArray(v)
    ref = sdpa_reference(q, k, v)                             # Float64, from the same rounded inputs
    key = (nameof(T),)
    register_kernel!(SDPA(), SDPADyn())
    add_variant!(SDPA(), SDPADyn(), key; KD.variant(SDPA(), SDPAKA(), key)...)
    @test !isconcretetype(Base.promote_op(extras, SDPA, SDPADyn, KernelBinary))   # length unknown
    # `extras` read `KernelBinary.params` (typed at run time): the result still infers.
    for kern in INFER_KERNELS, zero in (false, true)
        args = sdpa_binary_args(kern, OutArray(T, dv, n; zero), qd, kd, vd)
        z = @inferred KD.call_binary(SDPA(), kern, :fwd, args...; extent=n)
        # a half-type binary has a second output, the Float32 workspace: the result is the first
        @test eltype(first(z)) === T
        @test maxrelerr(Array(first(z)), ref) <= REDUCTION_RTOL[T]
    end
    @test @inferred(forward(SDPA(), SDPAKA(), qd, kd, vd)) isa DeviceArray{T,2}
    # The op call infers too: its kernel is a method (`@default_kernel`), not a table entry. (The default
    # is `FlashAttn`, which exists on Metal only.)
    if !isempty(prebuilt_sdpa_kernels())
        o = @inferred SDPA()(qd, kd, vd)
        @test eltype(o) === T && maxrelerr(Array(o), ref) <= REDUCTION_RTOL[T]
        @test_nowarn @inferred SDPA()(qd, kd, vd)
        user(q, k, v) = sum(SDPA()(q, k, v)) + 1.0f0
        @test @inferred(user(qd, kd, vd)) isa Float32       # the sum is `T`; adding a Float32 widens it
    end
end

# A fixture op with a backward binary, per type: forward, backward and the dispatch above them all infer, in `T`.
@testset "Residual op, $T" for T in supported_eltypes()
    rng = Random.Xoshiro(22)
    register_residual!(T)
    n = 50
    x, y, dz = (rounded(T, randn(rng, Float32, n)) for _ in 1:3)
    xd, yd, dzd = DeviceArray(x), DeviceArray(y), DeviceArray(dz)
    z = @inferred Residual()(xd, yd, 3.0f0)
    @test z isa DeviceArray{T,1}
    @test maxrelerr(Array(z), 3.0 .* Float64.(x) .+ Float64.(y)) <= ELEMENTWISE_RTOL[T]
    @test_nowarn @inferred KD.forward(Residual(), ResidualKA(), xd, yd, 3.0f0)
    grads = @inferred KD.backward(Residual(), ResidualKA(), (xd, yd, 3.0f0), z, dzd)
    @test grads[1] isa DeviceArray{T,1} && grads[2] isa DeviceArray{T,1} && grads[3] === nothing
    @test maxrelerr(Array(grads[1]), 3.0 .* Float64.(dz)) <= ELEMENTWISE_RTOL[T]
    @test maxrelerr(Array(grads[2]), Float64.(dz)) <= ELEMENTWISE_RTOL[T]
end

# The binary-level layers, for a KA-compiled binary (Vector{Any} prelude, flat grid) and, on Metal, an `:explicit`
# one: the `Launch` keeps the argument types, so running and timing it infer.
@testset "Launch path, $T" for T in supported_eltypes()
    d, n, m, dv = 16, 300, 6, 4
    q, k, v = inputs(Random.Xoshiro(23), d, n, m, dv, T)
    qd, kd, vd = DeviceArray(q), DeviceArray(k), DeviceArray(v)
    ref = sdpa_reference(q, k, v)
    key = (nameof(T),)
    ka = KD.variant(SDPA(), SDPAKA(), key)[:fwd]
    out() = OutArray(T, dv, n)
    cases = Pair[ka => (sdpa_binary_args(SDPAKA(), out(), qd, kd, vd)..., Val(64))]
    for pk in prebuilt_sdpa_kernels()
        plain = KD.variant(SDPA(), pk, key)[:fwd]
        push!(cases, plain => sdpa_binary_args(pk, out(), qd, kd, vd))
    end
    for (b, args) in cases
        l = @inferred prepare_launch(b, args...; extent=n)
        @test isconcretetype(typeof(l))
        o = first(@inferred KD.execute(l))
        @test o isa DeviceArray{T,2} && maxrelerr(Array(o), ref) <= REDUCTION_RTOL[T]
        o = first(@inferred run_binary(b, args...; extent=n))
        @test o isa DeviceArray{T,2} && maxrelerr(Array(o), ref) <= REDUCTION_RTOL[T]
        @test @inferred(time_binary(l; reps=1, warmup=0)) isa Float64
        @test @inferred(time_binary(b, args...; extent=n, reps=1, warmup=0)) isa Float64
    end
    # By op and kernel: the registered variant, with `extras` bound for it.
    kern, (_, args) = SDPAKA(), first(cases)
    @test @inferred(time_binary(SDPA(), kern, :fwd, args[1:end-1]...; extent=n, reps=1, warmup=0)) isa Float64
end
