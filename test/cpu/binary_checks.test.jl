# What a backend can launch, asked before it is: formats, target architectures, traced support. Static, so
# they run without a device; a test-only array type stands in for the device arrays whose backend is asked.

using Test, KernelDispatch, BFloat16s

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

    cubin, ptx, metallib = (KernelBinary(fake_file(e); threadgroup=32, signature=false) for e in (".cubin", ".ptx", ".metallib"))
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
    # Without the CUDA extension there is no device to ask; with it (a CUDA machine running the GPU section) it is the device's.
    @test KD.target_id(KD.CUDABackendTag()) == (:cuda in KD.LOADED_BACKENDS ? KD.current_target() : "")
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

@testset "ka_compile names a binary after its signature" begin
    tag = KD.ka_signature_tag
    @test tag((Vector{Float32}, Vector{Float32}, Float32, Int32, Val{64})) == "f32x2_f32s_i32s_v64"
    @test tag((Vector{Float16}, Vector{Float16}, Float32, Int32, Val{64})) == "f16x2_f32s_i32s_v64"
    @test tag((Vector{BFloat16}, Vector{Float32}, Vector{BFloat16}, Int32)) == "bf16_f32_bf16_i32s"
    @test tag((AbstractArray{Float16}, AbstractArray{Float16}, AbstractArray{Float16}, Int32, Int32)) == "f16x3_i32sx2"
    @test tag((Float64, UInt32, Bool, Val{:sym})) == "f64s_u32s_bools_vsym"
    # one kernel at different element types must never share a file name
    names = Set(KD.ka_binary_name("scale", (Vector{T}, Vector{T}, Float32)) for T in (Float32, Float16, BFloat16))
    @test length(names) == 3 && "scale_bf16x2_f32s" in names
    @test KD.ka_binary_name("k", (Vector{Float32}, Int32)) != KD.ka_binary_name("k", (Vector{Float32}, Int64))
    @test KD.ka_binary_name("k", (Vector{Float32}, Float32)) != KD.ka_binary_name("k", (Vector{Float32}, Vector{Float32}))
end
