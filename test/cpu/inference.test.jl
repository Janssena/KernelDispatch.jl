# Type stability, the part that needs no device: the small routing/constructor functions on the call
# path must infer concretely. (The launch paths, which need a device, are in gpu/inference.test.jl.)
# The op is SDPAOps: `FlashAttn` is an `:explicit` binary, `SDPAKA` a KA binary with `extras`.

using Test, KernelDispatch, SDPAOps, BFloat16s
isdefined(@__MODULE__, :ALL_ELTYPES) || include(joinpath(@__DIR__, "..", "precision.jl"))
isdefined(@__MODULE__, :Residual) || include(joinpath(@__DIR__, "..", "fixtures.jl"))

isdefined(@__MODULE__, :sdpa_binary_args) || include(joinpath(@__DIR__, "..", "sdpa_helpers.jl"))

# Every testset below runs once per element type: the same call, `@inferred`, in Float32, Float16 and BFloat16.
@testset "OutArray, $T" for T in ALL_ELTYPES
    @test (@inferred OutArray(T, 8)) isa OutArray{T,1}
    @test (@inferred OutArray(T, 2, 3)) isa OutArray{T,2}
    @test (@inferred OutArray(T, (2, 3, 4))) isa OutArray{T,3}
    @test (@inferred OutArray(T, 8; zero=true)).zero
    @test @inferred(eltype(OutArray(T, 4))) === T
end

@testset "InOutArray, $T" for T in ALL_ELTYPES
    # An array the kernel updates: bound as the array itself, returned with the outputs, in argument order.
    x, y = T.(rand(Float32, 8)), T.(rand(Float32, 8))
    args = (OutArray(T, 8), InOutArray(y), x, 2.0f0)
    bound, slots = KD.bind_slots(x, Any[], args, ())
    @test bound[2] === y && length(slots) == 4 && slots[2] === y
    outs = KD._pick_outs(args, bound)
    @test length(outs) == 2 && outs[2] === y && outs[1] isa Vector{T}       # the output is allocated in `T`
    @test (@inferred KD._pick_outs(args, bound)) isa Tuple{Vector{T},Vector{T}}
    # It counts as an array: it picks the backend, and a launch with only an InOutArray array has one.
    @test KD._first_array((2.0f0, InOutArray(y))) === y
    @test KD._first_array((2.0f0, Val(1))) === nothing
    @test (@inferred KD._first_array((OutArray(T, 2), InOutArray(y), 2.0f0))) === y
    @test (@inferred KD._first_array((2.0f0, x, InOutArray(y)))) === x
    b = KernelBinary("unused.metallib"; threadgroup=64)
    l = KD.prepare_launch(b, InOutArray(y), 2.0f0; extent=8)
    @test l.args[1] isa InOutArray && l.args[1].array === y
    # A wrapper that would be copied (and the update lost) is refused where the backend densifies; the host never does.
    @test KD._dense_arg(KD.CPUBackend(), InOutArray(y)).array === y
end

@testset "routing on host arrays, $T" for T in ALL_ELTYPES
    x = T.(rand(Float32, 4, 6))
    @test @inferred(KD.backend_of(x)) isa KD.CPUBackend
    @test @inferred(KD.backend_of(x, x)) isa KD.CPUBackend
    @test @inferred(KD.backend_of(view(x, :, 1:2))) isa KD.CPUBackend
    be = KD.backend_of(x)
    @test @inferred(KD.device_dense(be, x)) === x
    @test @inferred(KD.device_dense(be, 2.0f0)) === 2.0f0
    @test @inferred(KD.device_dense(be, Val(64))) === Val(64)
end

@testset "op hooks, $T" for T in ALL_ELTYPES
    q, k, v = T.(rand(Float32, 16, 8)), T.(rand(Float32, 16, 6)), T.(rand(Float32, 4, 6))
    op = SDPA()
    for kern in (SDPAKA(), FlashAttn())          # each in its own argument order, with the workspace for a half type
        args = sdpa_binary_args(kern, OutArray(T, 4, 8), q, k, v)
        @test @inferred(variant_key(op, kern, args...)) === (nameof(T),)
    end
    kern = SDPAKA()
    b = KernelBinary("unused.metallib"; threadgroup=64, params=(; tg=64))
    # `KernelBinary.params` is an abstract `NamedTuple` (binaries live in a `Dict` and reload from a manifest),
    # so a hook that reads a tuning number from it cannot infer: the documented one dynamic dispatch
    # per launch. `call_binary` binds `extras` apart from `args`. The launch size reads only `tile`.
    @test extras(op, kern, b) === (Val(64),)
    @test_broken (@inferred extras(op, kern, b); true)
    x, y = T.(rand(Float32, 8)), T.(rand(Float32, 8))
    l = @inferred KD.prepare_launch(b, OutArray(T, 8), x, y, 2.0f0; extent=8)
    @test l.groups === l.grid === (1, 1, 1)
    @test isconcretetype(typeof(l))
end

@testset "host path of an op, $T" for T in ALL_ELTYPES
    # The host runs its plain Julia and returns whatever that gives (wider than the inputs for a half type); stable either way.
    q, k, v = T.(rand(Float32, 16, 8)), T.(rand(Float32, 16, 6)), T.(rand(Float32, 4, 6))
    @test_nowarn @inferred SDPA()(q, k, v)
end

@testset "backends without an implementation fail loudly" begin
    # Metal (jax-mps) and CUDA (XLA's kernel-call handler) have a custom-call convention; a traced launch
    # under a backend without one (ROCm) must say so.
    @test_throws ArgumentError KD.custom_call(KD.ROCmBackendTag(), "entry", "path", (), (), [])
end
