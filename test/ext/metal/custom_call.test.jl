# The reverse rule attached to a `stablehlo.custom_call` call site.
#
# These tests never run a kernel: they trace, and assert on the emitted MLIR. That is deliberate --
# the part that can silently go wrong is the *shape* of what gets emitted, and it is checkable
# without a GPU, without the metallibs, on the CPU client. Enzyme verifies this same signature
# before it fires, so a mismatch here is what would otherwise become a compile error at gradient
# time (or, if the check did not exist, a wrong gradient).
#
# Runnable on its own:
#     julia --project=test test/ext/metal/custom_call.test.jl

using Test, KernelDispatch, Reactant

const RX = Base.get_extension(KernelDispatch, :KernelDispatchReactantExt)

isdefined(@__MODULE__, :reverse_symbol) || include(joinpath(@__DIR__, "..", "..", "ir_utils.jl"))   # reverse_symbol, reverse_signature

emit(f, args...) = repr(Reactant.@code_hlo optimize = false donated_args = :none f(args...))

@testset "custom_call reverse rule" begin
    @testset "attributes and signature" begin
        vjp(x, o, dout) = (Reactant.Ops.multiply(dout, Reactant.Ops.fill(3.0f0, (4,))),)

        function fwd(x)
            o, = RX._metal_kernel_lib("_fwd", "/tmp/none.metallib", (x,), [(4,)],
                [(:out, 0), (:in, 0)]; grid=(1, 1, 1),
                reverse=vjp, active_operands=(0,))
            return o
        end

        ir = emit(fwd, Reactant.to_rarray(rand(Float32, 4)))

        @test occursin("enzyme.reverse = @", ir)
        @test occursin("enzyme.active_operands = array<i64: 0>", ir)
        # One primal operand, one primal result, one result cotangent -> one cotangent back.
        @test reverse_signature(ir) == (3, 1)
        # The rule is only ever referenced from an attribute; it has to survive anyway.
        @test occursin(string("@\"", reverse_symbol(ir), "\"("), ir)
    end

    @testset "multi-result kernel: boolean operand excluded" begin
        B, H, Nj, Nq, hd = 1, 1, 2, 2, 2
        S4, S3 = (B, H, Nj, Nq, hd), (B, H, Nj, Nq)

        # (q, k, v, bias, mask, o, lse, dout, dlse) -> (dq, dk, dv, dbias). `dlse` is accepted and
        # ignored, exactly as the real rule ignores the residual's cotangent.
        function vjp(q, k, v, b, m, o, lse, dout, dlse)
            dq = Reactant.Ops.multiply(dout, dout)
            dk = Reactant.Ops.add(dout, o)
            dv = Reactant.Ops.add(dout, dout)
            db = Reactant.Ops.multiply(lse, lse)
            return (dq, dk, dv, db)
        end

        function fwd(q, k, v, b)
            mask = Reactant.Ops.fill(false, S3)
            o, lse = RX._metal_kernel_lib("_fwd", "/tmp/none.metallib",
                (q, k, v, b, mask), [S4, S3],
                [(:out, 0), (:out, 1), (:in, 0), (:in, 1), (:in, 2), (:in, 3), (:in, 4)];
                grid=(1, 1, 1), reverse=vjp, active_operands=(0, 1, 2, 3))
            return o
        end

        ir = emit(fwd, Reactant.to_rarray(rand(Float32, S4...)),
            Reactant.to_rarray(rand(Float32, S4...)),
            Reactant.to_rarray(rand(Float32, S4...)),
            Reactant.to_rarray(rand(Float32, S3...)))

        # The mask is operand 4 and is left out: booleans carry no cotangent.
        @test occursin("enzyme.active_operands = array<i64: 0, 1, 2, 3>", ir)
        # 5 primal operands + 2 primal results + 2 result cotangents -> 4 cotangents.
        @test reverse_signature(ir) == (9, 4)
        # Both results really are results of one call, not two separate calls.
        @test occursin(r"custom_call @mps\.metal_kernel_lib\([^)]*\).*-> \(tensor", ir)

        # Kept for the enzymexlamlir-opt half of the loop: run this module through the rule itself.
        out = get(ENV, "KERNELDISPATCH_DUMP_MLIR", "")
        isempty(out) || (write(out, ir); @info "wrote module" out)
    end

    @testset "no rule attached when none is asked for" begin
        function fwd(x)
            o, = RX._metal_kernel_lib("_fwd", "/tmp/none.metallib", (x,), [(4,)],
                [(:out, 0), (:in, 0)]; grid=(1, 1, 1))
            return o
        end

        ir = emit(fwd, Reactant.to_rarray(rand(Float32, 4)))
        @test !occursin("enzyme.reverse", ir)
        @test !occursin("enzyme.active_operands", ir)
    end

    @testset "custom_call dispatches on the backend" begin
        # A backend without a method (ROCm) says so; CUDA has one (test/ext/cuda/launch_encoding.test.jl).
        @test_throws ArgumentError KernelDispatch.custom_call(KernelDispatch.ROCmBackendTag(),
            "_fwd", "/tmp/none.metallib", (), [(4,)], Any[]; grid=(1, 1, 1))

        fwd(x) = KernelDispatch.custom_call(KernelDispatch.MetalBackendTag(), "_fwd", "/tmp/none.metallib",
            (x,), [(4,)], Any[(:in, 0), (:out, 0)]; grid=(1, 1, 1), threadgroup=4)[1]
        @test occursin("mps.metal_kernel_lib", emit(fwd, Reactant.to_rarray(rand(Float32, 4))))
    end

    @testset "_embed_custom_call is not Metal-specific" begin
        # `_metal_kernel_lib` is one caller of `_embed_custom_call`, not the only one it's built
        # for — any `call_target_name` and hand-built `backend_config` string should work the same
        # way, with no jax-mps convention baked into the generic seam itself.
        vjp(x, o, dout) = (Reactant.Ops.multiply(dout, Reactant.Ops.fill(2.0f0, (4,))),)

        function fwd(x)
            o, = RX._embed_custom_call("my.custom_target", (x,), [(4,)];
                backend_config="{\"anything\":\"goes\"}", reverse=vjp, active_operands=(0,))
            return o
        end

        ir = emit(fwd, Reactant.to_rarray(rand(Float32, 4)))

        @test occursin("stablehlo.custom_call @my.custom_target", ir)
        @test occursin("anything", ir) && occursin("goes", ir)
        @test occursin("enzyme.reverse = @", ir)
        @test reverse_signature(ir) == (3, 1)
    end
end
