# Type stability on the device: routing on the backend's arrays, the KA hooks, and the launch paths.
# `@inferred` alone for launches (a second run would dispatch a second kernel); `@test_nowarn` where a
# warning is the failure. The op is SDPAOps: `FlashAttn` is an `:explicit` (prebuilt, Metal-only) binary where the backend has prebuilt kernels,
# `SDPAKA` a KA binary with `extras`. The device-free half is in cpu/inference.test.jl.

using Test, KernelDispatch, SDPAOps, KernelAbstractions
isdefined(@__MODULE__, :Residual) || include(joinpath(@__DIR__, "..", "fixtures.jl"))
isdefined(@__MODULE__, :sdpa_binary_args) || include(joinpath(@__DIR__, "..", "sdpa_helpers.jl"))

const KOI = KernelDispatch
# The kernels with a binary on this backend: `FlashAttn` is a `.metallib`.
const INFER_KERNELS = (prebuilt_sdpa_kernels()..., SDPAKA(), SDPADyn())

@testset "routing on device arrays" begin
    W = wrapped_array_type()
    x = DeviceArray(rand(Float32, 4, 6))
    tag = GPU_TAG
    plain = (x, reshape(x, 6, 4), view(x, :, 2:3))        # contiguous: the backend hands back a plain array
    wrapped = (view(x, 2:3, :), x', transpose(x))         # these stay wrappers
    for a in plain
        @test @inferred(KOI.backend_of(a)) === tag
        @test @inferred(KOI.device_dense(tag, a)) === a
    end
    for a in wrapped
        @test a isa W
        @test @inferred(KOI.backend_of(a)) === tag
        @test @inferred(KOI.device_dense(tag, a)) isa DeviceArray
    end
    @test @inferred(KOI.backend_of(x, rand(Float32, 3))) === tag     # the first array decides
    @test @inferred(KOI.device_dense(tag, 2.0f0)) === 2.0f0
end

@testset "KA backend hooks" begin
    tag = GPU_TAG
    words = @inferred KOI.ka_state_words(tag)
    @test words isa Tuple
    @test words === EXPECTED_KA_STATE_WORDS                  # the backend's hidden leading arguments
    @test KOI.ka_backend_object(tag) isa KernelAbstractions.Backend
    @test @inferred(KOI.ka_grid(tag, (2, 3, 4))) === (24, 1, 1)             # flat
    ctx, ghost = KOI.ka_context(tag, (2, 1, 1), 64)
    @test ghost isa Bool
    # No extension, no default: a backend that forgets a hook fails loudly. ROCm has no extension.
    rocm = KOI.ROCmBackendTag()
    for f in (KOI.ka_state_words, KOI.ka_backend_object)
        @test_throws ArgumentError f(rocm)
    end
    @test_throws ArgumentError KOI.ka_grid(rocm, (1, 1, 1))
    @test_throws ArgumentError KOI.ka_context(rocm, (1, 1, 1), 64)
end

@testset "launch path" begin
    d, n, m, dv = 16, 8, 6, 4
    q, k, v = (DeviceArray(rand(Float32, r, c)) for (r, c) in ((d, n), (d, m), (dv, m)))
    ref = KernelDispatch.host(SDPA(), Array(q), Array(k), Array(v))
    register_kernel!(SDPA(), SDPADyn())
    add_variant!(SDPA(), SDPADyn(), (:Float32,); KOI.variant(SDPA(), SDPAKA(), (:Float32,))...)
    @test !isconcretetype(Base.promote_op(extras, SDPA, SDPADyn, KernelBinary))   # length unknown
    # `extras` read `KernelBinary.params` (typed at run time): the result still infers.
    for kern in INFER_KERNELS, zero in (false, true)
        args = sdpa_binary_args(kern, OutArray(Float32, dv, n; zero), q, k, v)
        z = @inferred KOI.call_binary(SDPA(), kern, :fwd, args...; extent=n)
        @test Array(only(z)) ≈ ref rtol = 1e-3
    end
    @test @inferred(forward(SDPA(), SDPAKA(), q, k, v)) isa DeviceArray
    # The op call infers too: its kernel is a method (`@default_kernel`), not a table entry. (The default
    # is `FlashAttn`, which exists on Metal only.)
    if !isempty(prebuilt_sdpa_kernels())
        @test_nowarn @inferred SDPA()(q, k, v)
        user(q, k, v) = sum(SDPA()(q, k, v)) + 1.0f0
        @test @inferred(user(q, k, v)) isa Float32
    end
end

# The binary-level layers, for a KA-compiled binary (Vector{Any} prelude, flat grid) and, on Metal, an `:explicit`
# one: the `Launch` keeps the argument types, so running and timing it infer.
@testset "Launch path" begin
    d, n, m, dv = 16, 300, 6, 4
    q, k, v = (DeviceArray(rand(Float32, r, c)) for (r, c) in ((d, n), (d, m), (dv, m)))
    ka = KOI.variant(SDPA(), SDPAKA(), (:Float32,))[:fwd]
    out() = OutArray(Float32, dv, n)
    cases = Pair[ka => (sdpa_binary_args(SDPAKA(), out(), q, k, v)..., Val(64))]
    for pk in prebuilt_sdpa_kernels()
        plain = KOI.variant(SDPA(), pk, (:Float32,))[:fwd]
        push!(cases, plain => sdpa_binary_args(pk, out(), q, k, v))
    end
    for (b, args) in cases
        l = @inferred prepare_launch(b, args...; extent=n)
        @test isconcretetype(typeof(l))
        @test only(@inferred KOI.execute(l)) isa DeviceArray{Float32,2}
        @test only(@inferred run_binary(b, args...; extent=n)) isa DeviceArray{Float32,2}
        @test @inferred(time_binary(l; reps=1, warmup=0)) isa Float64
        @test @inferred(time_binary(b, args...; extent=n, reps=1, warmup=0)) isa Float64
    end
    # By op and kernel: the registered variant, with `extras` bound for it.
    kern, (_, args) = SDPAKA(), first(cases)
    @test @inferred(time_binary(SDPA(), kern, :fwd, args[1:end-1]...; extent=n, reps=1, warmup=0)) isa Float64
end
