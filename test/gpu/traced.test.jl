# Traced, on the backend's own Reactant client (`traced_client()`: jax-mps on Metal when the plugin is found, Reactant's
# CUDA client on CUDA): the SAME binaries the eager path launches, embedded as the backend's custom call. The forward of a
# real op (SDPAOps) traced is in gpu/sdpa.test.jl; here, what only the fixtures have: a backward binary with two outputs,
# autodiff refusal, a zero-filled output after the output-first KA kernels, and the binary-level launch features.

using Test, Random, KernelDispatch, Reactant, Enzyme
isdefined(@__MODULE__, :Residual) || include(joinpath(@__DIR__, "..", "fixtures.jl"))

const _CLIENT = traced_client()

if _CLIENT === nothing
    @info "no traced client for $GPU (Metal needs the jax-mps plugin: JAX_MPS_PLUGIN): skipping traced tests"
else
    prev = Reactant.XLA.default_backend()
    Reactant.set_default_backend(_CLIENT)
    try
        local rng = Random.Xoshiro(2)
        isempty(KernelDispatch.variants(Residual(), ResidualKA())) && register_residual!()
        @assert current_kernel(:residual) === :ka

        @testset "traced forward" begin
            n = 64
            x, y = randn(rng, Float32, n), randn(rng, Float32, n)
            fwd(x, y) = KernelDispatch.forward(Residual(), ResidualKA(), x, y, 3.0f0)
            @test Array(@jit fwd(Reactant.to_rarray(x), Reactant.to_rarray(y))) ≈ 3.0f0 .* x .+ y
        end

        @testset "traced backward (two outputs)" begin
            n = 64
            x, y, dz = randn(rng, Float32, n), randn(rng, Float32, n), randn(rng, Float32, n)
            g(x, y, dz) = KernelDispatch.backward(Residual(), ResidualKA(), (x, y, 3.0f0), x, dz)[1:2]
            gx, gy = @jit g(Reactant.to_rarray(x), Reactant.to_rarray(y), Reactant.to_rarray(dz))
            @test Array(gx) ≈ 3 .* dz
            @test Array(gy) ≈ dz
        end

        @testset "traced autodiff refuses" begin
            x = Reactant.to_rarray(randn(rng, Float32, 16))
            h(x) = Enzyme.gradient(Enzyme.Reverse, x -> sum(Residual()(x, x, 1.0f0)), x)
            err = try
                Reactant.@compile h(x)
                nothing
            catch e
                e
            end
            @test err !== nothing && occursin("cannot be differentiated under Reactant", sprint(showerror, err))
        end

        @testset "ka_compile + call_binary, traced" begin
            register_logit_scale!()
            x = randn(rng, Float32, 100)
            xr = Reactant.to_rarray(x)
            f(x) = logit_scale(x, 2.0f0; out_len=128)
            y = Array(@jit f(xr))
            @test y[1:100] ≈ 2.0f0 .* x
            @test all(iszero, y[101:128])
        end

        @testset "run_binary of an unregistered binary, traced" begin
            bin = register_logit_scale!()[:main]
            # Built inside the traced function: a captured `KernelBinary` (abstract `params`) is not traceable.
            file, entry = bin.file, bin.entry
            layout = get(bin.params, :ka_layout, "")      # a backend that records one (CUDA: "flat") refuses a binary without it
            loose() = KernelBinary(file; entry, threadgroup=LS_TG, abi=:ka, params=(; ka_layout=layout))
            x = randn(rng, Float32, 100)
            f(x) = only(run_binary(loose(), OutArray(Float32, 100), x, 2.0f0, Int32(100), Val(LS_TG);
                extent=100))
            @test Array(@jit f(Reactant.to_rarray(x))) ≈ 2.0f0 .* x
        end
    finally
        Reactant.set_default_backend(prev)
    end
end
