# What a backend can launch, asked before it is: formats, target architectures, traced support. Static, so
# they run without a device; a test-only array type stands in for the device arrays whose backend is asked.

using Test, KernelDispatch

fake_file(ext) = (p = joinpath(mktempdir(), "k" * ext); write(p, UInt8[1, 2, 3]); p)

# An array that says it lives on a CUDA device, so `prepare_launch` takes its device branch.
struct FakeCuArray <: AbstractVector{Float32}; n::Int; end
Base.size(a::FakeCuArray) = (a.n,)
Base.getindex(::FakeCuArray, i::Int) = 0.0f0
KD.backend_of(::FakeCuArray) = KD.CUDABackendTag()

@testset "binary formats" begin
    @test KD.binary_format(KernelBinary(fake_file(".cubin"); threadgroup=32)) === :cubin
    @test KD.binary_format(KernelBinary(fake_file(".PTX"); threadgroup=32)) === :ptx
    @test KD.supported_formats(KD.MetalBackendTag()) == (:metallib,)
    @test KD.supported_formats(KD.CUDABackendTag()) == (:cubin, :ptx)
    @test KD.supported_formats(KD.ROCmBackendTag()) == (:hsaco,)
    @test KD.supported_formats(KD.CPUBackend()) == ()                          # the host is unchecked

    cubin, ptx, metallib = (KernelBinary(fake_file(e); threadgroup=32) for e in (".cubin", ".ptx", ".metallib"))
    @test KD.check_binary(KD.CUDABackendTag(), cubin) === nothing
    @test KD.check_binary(KD.CUDABackendTag(), ptx) === nothing
    @test KD.check_binary(KD.MetalBackendTag(), metallib) === nothing
    e = try KD.check_binary(KD.CUDABackendTag(), metallib); nothing catch e; e end
    @test e isa ArgumentError && occursin(".cubin", e.msg) && occursin(".metallib", e.msg)
    @test_throws ArgumentError KD.check_binary(KD.MetalBackendTag(), cubin)
    @test KD.check_binary(KD.CPUBackend(), metallib) === nothing

    # `prepare_launch` runs the check for device arrays, eager or traced, and only for them.
    @test_throws ArgumentError KD.prepare_launch(metallib, FakeCuArray(4); extent=4)
    @test KD.prepare_launch(cubin, FakeCuArray(4); extent=4) isa KD.Launch
    @test KD.prepare_launch(metallib, rand(Float32, 4); extent=4) isa KD.Launch      # host arrays: unchecked
end

@testset "target architectures" begin
    @test KD.target_id(KD.CUDABackendTag()) == ""        # no extension loaded here: nothing to ask
    @test KD.cuda_sm_compatible("", "sm_90")              # portable
    @test KD.cuda_sm_compatible("sm_80", "sm_80")
    @test KD.cuda_sm_compatible("sm_80", "sm_86")         # same major, newer minor
    @test !KD.cuda_sm_compatible("sm_86", "sm_80")        # older minor
    @test !KD.cuda_sm_compatible("sm_80", "sm_90")        # another major
    @test KD.cuda_sm_compatible("sm_100", "sm_103") && !KD.cuda_sm_compatible("sm_100", "sm_90")   # two-digit major
    @test KD.cuda_sm_compatible("sm_90a", "sm_90")
    @test KD.cuda_sm_compatible("weird", "weird") && !KD.cuda_sm_compatible("weird", "sm_80")
    @test KD.compatible_target(KD.CPUBackend(), KernelBinary(fake_file(".cubin"); threadgroup=32, target="sm_50")) === nothing
end

@testset "traced support defaults" begin
    b = KernelBinary(fake_file(".metallib"); threadgroup=32, abi=:ka)
    @test KD.traced_support(KD.MetalBackendTag(), b) === nothing
    @test KD.traced_support(KD.CPUBackend(), b) === nothing
end
