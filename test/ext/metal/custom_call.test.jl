# Metal's own custom-call embedding: `mps.metal_kernel_lib`, the dispatcher jax-mps provides for a named kernel in a
# `.metallib`. Only what is specific to it is here: the buffer table it emits and what it refuses. The generic seam it is
# built on (reverse rules, aliases, result types) is tested for every backend in test/ext/reactant.
#
# Nothing runs a kernel: it traces on Reactant's CPU client and reads the emitted MLIR, so it needs no Metal device.
# (A real traced Metal launch is in gpu/traced.test.jl and gpu/conformance.test.jl.)
#
# Runnable on its own:
#     julia --project=test test/ext/metal/custom_call.test.jl

using Test, KernelDispatch, Reactant

const RX = Base.get_extension(KernelDispatch, :KernelDispatchReactantExt)

emit(f, args...) = repr(Reactant.@code_hlo optimize = false donated_args = :none f(args...))

@testset "mps.metal_kernel_lib" begin
    x = Reactant.to_rarray(rand(Float32, 4))
    lib(layout; kw...) = x -> first(RX._metal_kernel_lib("kern", "/tmp/none.metallib", (x,), [(4,)], layout;
        grid=(2, 1, 1), threadgroup=64, result_eltypes=(Float32,), kw...))

    @testset "the buffer table and the dispatch" begin
        ir = emit(lib(Any[(:in, 0), (:out, 0), (:bytes, "[1,2,3]")]), x)
        @test occursin("stablehlo.custom_call @mps.metal_kernel_lib", ir)
        @test occursin("\\22name\\22:\\22kern\\22", ir) && occursin("\\22metallib_path\\22:\\22/tmp/none.metallib\\22", ir)
        @test occursin("\\22grid\\22:[2,1,1]", ir) && occursin("\\22threadgroup\\22:[64,1,1]", ir)
        @test occursin("\\22dispatch\\22:\\22threadgroups\\22", ir)             # by threadgroups, as the kernel indexes
        @test occursin("\\22kind\\22:\\22input\\22", ir) && occursin("\\22kind\\22:\\22output\\22", ir)
        @test occursin("\\22kind\\22:\\22bytes\\22,\\22bytes\\22:[1,2,3]", ir)
        @test !occursin("zero_init", ir)
    end

    @testset "an output that accumulates is zero-filled by jax-mps" begin
        ir = emit(lib(Any[(:out_zero, 0), (:in, 0)]), x)
        @test occursin("\\22zero_init\\22:true", ir)
    end

    @testset "an InOutArray cannot be traced on Metal" begin
        # jax-mps cannot alias a result to an operand: refused with the way around it, before anything is emitted.
        e = try; RX._metal_kernel_lib("kern", "/tmp/none.metallib", (), [(4,)], Any[(:inout, (0, 0))];
            grid=(1, 1, 1), result_eltypes=(Float32,)); nothing; catch e; e; end
        @test e isa ArgumentError && occursin("InOutArray", e.msg)
    end

    @testset "the Metal embedding is what custom_call(::MetalBackendTag) emits" begin
        fwd(x) = KD.custom_call(KD.MetalBackendTag(), "kern", "/tmp/none.metallib", (x,), [(4,)],
            Any[(:in, 0), (:out, 0)]; grid=(1, 1, 1), threadgroup=4, result_eltypes=(Float32,))[1]
        @test occursin("mps.metal_kernel_lib", emit(fwd, x))
    end
end
