# Plain binaries on CUDA: a pointer-and-scalar kernel (hand-written PTX, assembled to a cubin), launched
# eagerly and, through XLA's kernel-call handler, from a Reactant trace. Plain binaries are the ones that
# can be traced: a KernelAbstractions binary takes struct arguments the handler cannot pass (asserted
# below). CUDA only; the Metal equivalent is gpu/reactant.test.jl.
#
# The traced half is the part not yet verified on hardware: if it fails, `julia --project=test
# test/cuda_probe.jl` finds which target/convention (KERNELDISPATCH_CUDA_CUSTOM_CALL) the XLA client accepts.

using Test, Random, KernelDispatch, Reactant
isdefined(@__MODULE__, :Residual) || include(joinpath(@__DIR__, "..", "fixtures.jl"))
isdefined(@__MODULE__, :scale_binaries) || include(joinpath(@__DIR__, "..", "cuda_ptx.jl"))

if GPU === :cuda
    bins = scale_binaries(mktempdir())
    rng = Xoshiro(5)

    @testset "eager" begin
        for (kind, bin) in pairs(bins), n in (1, 64, 100, 1000)
            x = randn(rng, Float32, n)
            (y,) = run_binary(bin, DeviceArray(x), OutArray(Float32, n), 3.0f0, Int32(n); extent=n)
            @test Array(y) ≈ 3.0f0 .* x
        end
        @test KernelDispatch.max_threads(bins.cubin) >= 64
    end

    @testset "traced, through XLA's kernel-call handler" begin
        prev = Reactant.XLA.default_backend()
        Reactant.set_default_backend("gpu")
        try
            @test KernelDispatch.backend_of(Reactant.to_rarray(rand(Float32, 4))) isa KernelDispatch.CUDABackendTag
            for (kind, bin) in pairs(bins)
                file, entry, tg = bin.file, bin.entry, bin.threadgroup
                # Built inside the traced function: a captured `KernelBinary` (abstract `params`) is not traceable.
                loose() = KernelBinary(file; entry, threadgroup=tg)
                x = randn(rng, Float32, 100)
                f(x) = only(run_binary(loose(), x, OutArray(Float32, 100), 2.0f0, Int32(100); extent=100))
                @testset "$kind" begin
                    @test Array(@jit f(Reactant.to_rarray(x))) ≈ 2.0f0 .* x
                end
            end

            @testset "a KernelAbstractions binary says it cannot be traced" begin
                register_logit_scale!()
                x = Reactant.to_rarray(randn(rng, Float32, 100))
                f(x) = logit_scale(x, 2.0f0)
                err = try
                    Reactant.@compile f(x)
                    nothing
                catch e
                    e
                end
                @test err !== nothing && occursin("KernelAbstractions", sprint(showerror, err))
            end
        finally
            Reactant.set_default_backend(prev)
        end
    end
end
