# What is specific to CUDA's explicit binaries, on a GPU. The generic contract (eager and traced launches of the
# `scale` fixture as a cubin and as PTX, limits and timing, refusals, KA kernels, ops) is in test/gpu/ and runs on every
# backend; here are the CUDA-only mechanisms that a failure there would not pin down:
#
#   - the `:triton` ABI's launch parameters on a kernel that does not use them: dynamic shared memory above the default
#     48 KB (the function attribute must be raised), hidden trailing null pointers, eager and through XLA's handler;
#   - the architecture a cubin was built for (`sm_XY`), checked against the device;
#   - a flat KA binary with its input before its output, through XLA's handler.
#
# (Real Triton kernels are test/ext/cuda/triton.test.jl.) If every traced case fails, the XLA client may expect another
# calling convention: KERNELDISPATCH_CUDA_CUSTOM_CALL=<target>@<api_version> selects it.

using Test, Random, KernelDispatch, Reactant
isdefined(@__MODULE__, :Residual) || include(joinpath(@__DIR__, "..", "..", "fixtures.jl"))

if GPU === :cuda
    bins = scale_binaries(mktempdir())            # from ptx.jl, loaded by this backend's adapter
    rng = Xoshiro(5)

    @testset "Triton ABI (shared memory, trailing pointers), eager" begin
        # The scale kernel dressed as a Triton one; it declares no scratch parameters, so this shows that the extra
        # null pointers and the dynamic shared memory (60 KB: above the 48 KB default) are accepted.
        x = randn(rng, Float32, 100)
        for shared in (0, 1024, 60 * 1024), (kind, bin) in pairs(bins)
            tb = KernelBinary(bin; abi=:triton, shared_mem=shared, params=(; scratch_pointers=2), tile=64)
            (y,) = run_binary(tb, DeviceArray(x), OutArray(Float32, 100), 2.0f0, Int32(100); extent=100)
            @test Array(y) ≈ 2.0f0 .* x
        end
    end

    @testset "target architecture" begin
        dev = KernelDispatch.target_id(GPU_TAG)
        @test occursin(r"^sm_\d+$", dev) && bins.cubin.target == dev && isempty(bins.ptx.target)
        x = randn(rng, Float32, 16)
        run(b) = Array(only(run_binary(b, DeviceArray(x), OutArray(Float32, 16), 2.0f0, Int32(16); extent=16)))
        @test run(bins.cubin) ≈ 2.0f0 .* x
        # A cubin for another architecture is refused with the reason; PTX is portable whatever it names.
        wrong = KernelBinary(bins.cubin; target="sm_20")
        err = try run(wrong); nothing catch e; e end
        @test err isa ArgumentError && occursin("built for sm_20", err.msg)
        @test run(KernelBinary(bins.ptx; target="sm_20")) ≈ 2.0f0 .* x
    end

    prev = Reactant.XLA.default_backend()
    Reactant.set_default_backend(traced_client())
    try
        @test KernelDispatch.backend_of(Reactant.to_rarray(rand(Float32, 4))) isa KernelDispatch.CUDABackendTag

        @testset "Triton ABI, traced" begin
            file, target, sig = bins.cubin.file, bins.cubin.target, bins.cubin.signature
            loose() = KernelBinary(file; entry="scale", threadgroup=64, abi=:triton, target,
                shared_mem=1024, params=(; scratch_pointers=2), tile=64, signature=sig)
            x = randn(rng, Float32, 100)
            f(x) = only(run_binary(loose(), x, OutArray(Float32, 100), 2.0f0, Int32(100); extent=100))
            @test Array(@jit f(Reactant.to_rarray(x))) ≈ 2.0f0 .* x
        end

        @testset "a KernelAbstractions binary, traced (built flat, inputs before the output)" begin
            ka = compile_scale_ins_first()
            @test ka.abi === :ka && ka.params.ka_layout == "flat"
            file, entry, tg, sig = ka.file, ka.entry, ka.threadgroup, ka.signature
            # Built inside the traced function: a captured `KernelBinary` (abstract `params`) is not traceable.
            loose() = KernelBinary(file; entry, threadgroup=tg, abi=:ka, params=(; ka_layout="flat", tg), signature=sig)
            x = randn(rng, Float32, 100)
            f(x) = only(run_binary(loose(), x, OutArray(Float32, 100), 2.0f0, Int32(100), Val(tg); extent=100))
            @test Array(@jit f(Reactant.to_rarray(x))) ≈ 2.0f0 .* x
        end
    finally
        Reactant.set_default_backend(prev)
    end
end
