# Type stability, the part that needs no device: the small routing/constructor functions on the call
# path must infer concretely. (The launch paths, which need a device, are in gpu/inference.test.jl.)
# The op is SDPAOps: `FlashAttn` is an `:explicit` binary, `SDPAKA` a KA binary with `extras`.

using Test, KernelDispatch, SDPAOps
isdefined(@__MODULE__, :Residual) || include(joinpath(@__DIR__, "..", "fixtures.jl"))

const KOI = KernelDispatch
isdefined(@__MODULE__, :sdpa_binary_args) || include(joinpath(@__DIR__, "..", "sdpa_helpers.jl"))

@testset "OutArray" begin
    @test (@inferred OutArray(Float32, 8)) isa OutArray{Float32,1}
    @test (@inferred OutArray(Float32, 2, 3)) isa OutArray{Float32,2}
    @test (@inferred OutArray(Float32, (2, 3, 4))) isa OutArray{Float32,3}
    @test (@inferred OutArray(Float32, 8; zero=true)).zero
    @test @inferred(eltype(OutArray(Float16, 4))) === Float16
end

@testset "InOutArray" begin
    # An array the kernel updates: bound as the array itself, returned with the outputs, in argument order.
    x, y = rand(Float32, 8), rand(Float32, 8)
    args = (OutArray(Float32, 8), InOutArray(y), x, 2.0f0)
    bound, slots = KOI.bind_slots(x, Any[], args, ())
    @test bound[2] === y && length(slots) == 4 && slots[2] === y
    outs = KOI._pick_outs(args, bound)
    @test length(outs) == 2 && outs[2] === y && outs[1] isa Vector{Float32}
    @test (@inferred KOI._pick_outs(args, bound)) isa Tuple{Vector{Float32},Vector{Float32}}
    # It counts as an array: it picks the backend, and a launch with only an InOutArray array has one.
    @test KOI._first_array((2.0f0, InOutArray(y))) === y
    @test KOI._first_array((2.0f0, Val(1))) === nothing
    b = KernelBinary("unused.metallib"; threadgroup=64)
    l = KOI.prepare_launch(b, InOutArray(y), 2.0f0; extent=8)
    @test l.args[1] isa InOutArray && l.args[1].array === y
    # A wrapper that would be copied (and the update lost) is refused where the backend densifies; the host never does.
    @test KOI._dense_arg(KOI.CPUBackend(), InOutArray(y)).array === y
end

@testset "routing on host arrays" begin
    x = rand(Float32, 4, 6)
    @test @inferred(KOI.backend_of(x)) isa KOI.CPUBackend
    @test @inferred(KOI.backend_of(x, x)) isa KOI.CPUBackend
    @test @inferred(KOI.backend_of(view(x, :, 1:2))) isa KOI.CPUBackend
    be = KOI.backend_of(x)
    @test @inferred(KOI.device_dense(be, x)) === x
    @test @inferred(KOI.device_dense(be, 2.0f0)) === 2.0f0
    @test @inferred(KOI.device_dense(be, Val(64))) === Val(64)
end

@testset "op hooks" begin
    q, k, v = rand(Float32, 16, 8), rand(Float32, 16, 6), rand(Float32, 4, 6)
    op, kern = SDPA(), SDPAKA()
    args = sdpa_binary_args(kern, OutArray(Float32, 4, 8), q, k, v)
    @test @inferred(variant_key(op, kern, args...)) === (:Float32,)
    b = KernelBinary("unused.metallib"; threadgroup=64, params=(; tg=64))
    # `KernelBinary.params` is an abstract `NamedTuple` (binaries live in a `Dict` and reload from a manifest),
    # so a hook that reads a tuning number from it cannot infer: the documented one dynamic dispatch
    # per launch. `call_binary` binds `extras` apart from `args`. The launch size reads only `tile`.
    @test extras(op, kern, b) === (Val(64),)
    @test_broken (@inferred extras(op, kern, b); true)
    x, y = rand(Float32, 8), rand(Float32, 8)
    l = @inferred KOI.prepare_launch(b, OutArray(Float32, 8), x, y, 2.0f0; extent=8)
    @test l.groups === l.grid === (1, 1, 1)
    @test isconcretetype(typeof(l))
end

@testset "host path of an op" begin
    q, k, v = rand(Float32, 16, 8), rand(Float32, 16, 6), rand(Float32, 4, 6)
    @test_nowarn @inferred SDPA()(q, k, v)
end

@testset "backends without an implementation fail loudly" begin
    # Metal (jax-mps) and CUDA (XLA's kernel-call handler) have a custom-call convention; a traced launch
    # under a backend without one (ROCm) must say so.
    @test_throws ArgumentError KOI.custom_call(KOI.ROCmBackendTag(), "entry", "path", (), (), [])
end
