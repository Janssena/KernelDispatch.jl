# What a backend can launch, asked before it is: formats, target architectures, traced support. Static, so
# they run without a device; a test-only array type stands in for the device arrays whose backend is asked.

using Test, KernelDispatch
const KB = KernelDispatch

fake_file(ext) = (p = joinpath(mktempdir(), "k" * ext); write(p, UInt8[1, 2, 3]); p)

# An array that says it lives on a CUDA device, so `prepare_launch` takes its device branch.
struct FakeCuArray <: AbstractVector{Float32}; n::Int; end
Base.size(a::FakeCuArray) = (a.n,)
Base.getindex(::FakeCuArray, i::Int) = 0.0f0
KB.backend_of(::FakeCuArray) = KB.CUDABackendTag()

@testset "binary formats" begin
    @test KB.binary_format(KernelBinary(fake_file(".cubin"); threadgroup=32)) === :cubin
    @test KB.binary_format(KernelBinary(fake_file(".PTX"); threadgroup=32)) === :ptx
    @test KB.supported_formats(KB.MetalBackendTag()) == (:metallib,)
    @test KB.supported_formats(KB.CUDABackendTag()) == (:cubin, :ptx)
    @test KB.supported_formats(KB.ROCmBackendTag()) == (:hsaco,)
    @test KB.supported_formats(KB.CPUBackend()) == ()                          # the host is unchecked

    cubin, ptx, metallib = (KernelBinary(fake_file(e); threadgroup=32) for e in (".cubin", ".ptx", ".metallib"))
    @test KB.check_binary(KB.CUDABackendTag(), cubin) === nothing
    @test KB.check_binary(KB.CUDABackendTag(), ptx) === nothing
    @test KB.check_binary(KB.MetalBackendTag(), metallib) === nothing
    e = try KB.check_binary(KB.CUDABackendTag(), metallib); nothing catch e; e end
    @test e isa ArgumentError && occursin(".cubin", e.msg) && occursin(".metallib", e.msg)
    @test_throws ArgumentError KB.check_binary(KB.MetalBackendTag(), cubin)
    @test KB.check_binary(KB.CPUBackend(), metallib) === nothing

    # `prepare_launch` runs the check for device arrays, eager or traced, and only for them.
    @test_throws ArgumentError KB.prepare_launch(metallib, FakeCuArray(4); extent=4)
    @test KB.prepare_launch(cubin, FakeCuArray(4); extent=4) isa KB.Launch
    @test KB.prepare_launch(metallib, rand(Float32, 4); extent=4) isa KB.Launch      # host arrays: unchecked
end

@testset "target architectures" begin
    @test KB.target_id(KB.CUDABackendTag()) == ""        # no extension loaded here: nothing to ask
    @test KB.cuda_sm_compatible("", "sm_90")              # portable
    @test KB.cuda_sm_compatible("sm_80", "sm_80")
    @test KB.cuda_sm_compatible("sm_80", "sm_86")         # same major, newer minor
    @test !KB.cuda_sm_compatible("sm_86", "sm_80")        # older minor
    @test !KB.cuda_sm_compatible("sm_80", "sm_90")        # another major
    @test KB.cuda_sm_compatible("sm_100", "sm_103") && !KB.cuda_sm_compatible("sm_100", "sm_90")   # two-digit major
    @test KB.cuda_sm_compatible("sm_90a", "sm_90")
    @test KB.cuda_sm_compatible("weird", "weird") && !KB.cuda_sm_compatible("weird", "sm_80")
    @test KB.compatible_target(KB.CPUBackend(), KernelBinary(fake_file(".cubin"); threadgroup=32, target="sm_50")) === nothing
end

@testset "traced support defaults" begin
    b = KernelBinary(fake_file(".metallib"); threadgroup=32, abi=:ka)
    @test KB.traced_support(KB.MetalBackendTag(), b) === nothing
    @test KB.traced_support(KB.CPUBackend(), b) === nothing
end
