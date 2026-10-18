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
        # Only .cubin and .ptx are loaded: anything else is refused up front.
        junk = joinpath(mktempdir(), "scale.metallib"); write(junk, UInt8[1, 2, 3])
        bad = KernelBinary(junk; entry="scale", threadgroup=64)
        err = try run_binary(bad, DeviceArray(rand(Float32, 4)), OutArray(Float32, 4), 1.0f0, Int32(4); extent=4); nothing catch e; e end
        @test err isa ArgumentError && occursin(".cubin", err.msg)
    end

    @testset "Triton ABI (shared memory, trailing pointers)" begin
        # The scale kernel dressed as a Triton one; it declares no scratch parameters, so this shows that the extra
        # null pointers and the dynamic shared memory (60 KB: above the 48 KB default) are accepted.
        x = randn(rng, Float32, 100)
        for shared in (0, 1024, 60 * 1024), (kind, bin) in pairs(bins)
            tb = KernelBinary(bin.file; entry="scale", threadgroup=64, abi=:triton, target=bin.target,
                shared_mem=shared, params=(; scratch_pointers=2), tile=64)
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
        wrong = KernelBinary(bins.cubin.file; entry="scale", threadgroup=64, target="sm_20")
        err = try run(wrong); nothing catch e; e end
        @test err isa ArgumentError && occursin("built for sm_20", err.msg)
        @test run(KernelBinary(bins.ptx.file; entry="scale", threadgroup=64, target="sm_20")) ≈ 2.0f0 .* x
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

            @testset "Triton ABI, traced" begin
                file, target = bins.cubin.file, bins.cubin.target
                loose() = KernelBinary(file; entry="scale", threadgroup=64, abi=:triton, target,
                    shared_mem=1024, params=(; scratch_pointers=2), tile=64)
                x = randn(rng, Float32, 100)
                f(x) = only(run_binary(loose(), x, OutArray(Float32, 100), 2.0f0, Int32(100); extent=100))
                @test Array(@jit f(Reactant.to_rarray(x))) ≈ 2.0f0 .* x
            end

            @testset "a KernelAbstractions binary, traced (built flat, inputs before the output)" begin
                ka = compile_scale_ins_first()
                @test ka.abi === :ka && ka.params.ka_layout == "flat"
                file, entry, tg = ka.file, ka.entry, ka.threadgroup
                # Built inside the traced function: a captured `KernelBinary` (abstract `params`) is not traceable.
                loose() = KernelBinary(file; entry, threadgroup=tg, abi=:ka, params=(; ka_layout="flat", tg))
                x = randn(rng, Float32, 100)
                f(x) = only(run_binary(loose(), x, OutArray(Float32, 100), 2.0f0, Int32(100), Val(tg); extent=100))
                @test Array(@jit f(Reactant.to_rarray(x))) ≈ 2.0f0 .* x
            end

            # The kernels below put their OUTPUT FIRST (`f!(out, in...)`): each output is passed to XLA's handler as an
            # aliased operand in the kernel's own position, so the order of a kernel's buffers is unrestricted.
            @testset "output-first KA kernel, traced (logit_scale: y, x, ...; zero-filled tail)" begin
                register_logit_scale!()
                x = randn(rng, Float32, 100)
                f(x) = logit_scale(x, 2.0f0; out_len=128)
                y = Array(@jit f(Reactant.to_rarray(x)))
                @test y[1:100] ≈ 2.0f0 .* x
                @test all(iszero, y[101:128])
            end

            @testset "two outputs, traced (Residual: forward, and backward -> dx, dy)" begin
                isempty(KernelDispatch.variants(Residual(), ResidualKA())) && register_residual!()
                n = 64
                x, y, dz = randn(rng, Float32, n), randn(rng, Float32, n), randn(rng, Float32, n)
                fwd(x, y) = KernelDispatch.forward(Residual(), ResidualKA(), x, y, 3.0f0)
                @test Array(@jit fwd(Reactant.to_rarray(x), Reactant.to_rarray(y))) ≈ 3.0f0 .* x .+ y
                g(x, y, dz) = KernelDispatch.backward(Residual(), ResidualKA(), (x, y, 3.0f0), x, dz)[1:2]
                gx, gy = @jit g(Reactant.to_rarray(x), Reactant.to_rarray(y), Reactant.to_rarray(dz))
                @test Array(gx) ≈ 3.0f0 .* dz
                @test Array(gy) ≈ dz
            end
        finally
            Reactant.set_default_backend(prev)
        end
    end
end
