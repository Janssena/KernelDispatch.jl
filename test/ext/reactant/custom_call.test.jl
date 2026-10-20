# The Reactant extension's custom-call seam, backend-agnostic, on the GPU backend's own Reactant client
# (`traced_client()`: jax-mps on Metal, XLA's CUDA client on CUDA). Embedding a binary in a trace is a GPU operation, so
# these run in the GPU suite; the adapter (test/gpu_backend.jl) says which client, which element types and which
# binary the loaded backend has.
#
# Nothing here runs a kernel: the tests trace and assert on the emitted MLIR. That is deliberate, the part that can go
# silently wrong is the SHAPE of what is emitted, and it is checkable without launching anything. Real traced launches, and
# their numbers, are in gpu/traced.test.jl and gpu/conformance.test.jl.
#
# The tests use `_embed_custom_call` with a made-up target, the seam every backend's `custom_call` builds on, so no vendor
# convention is assumed. What a vendor's own embedding looks like is in test/ext/metal and test/ext/cuda.

using Test, KernelDispatch, Reactant, BFloat16s

const RX = Base.get_extension(KernelDispatch, :KernelDispatchReactantExt)

isdefined(@__MODULE__, :reverse_symbol) || include(joinpath(@__DIR__, "..", "..", "ir_utils.jl"))   # reverse_symbol, reverse_signature
isdefined(@__MODULE__, :ALL_ELTYPES) || include(joinpath(@__DIR__, "..", "..", "precision.jl"))

emit(f, args...) = repr(Reactant.@code_hlo optimize = false donated_args = :none f(args...))

const SEAM_TARGET = "kd.test_target"
embed(operands, shapes; kw...) = RX._embed_custom_call(SEAM_TARGET, operands, shapes; backend_config="{}", kw...)

seam_client = traced_client()
if seam_client === nothing
    @info "no traced client for $GPU: skipping the Reactant custom-call seam tests"
else
    seam_prev = Reactant.XLA.default_backend()
    Reactant.set_default_backend(seam_client)
    try
        @testset "the custom-call seam on $GPU's client" begin
            @testset "a reverse rule is attached to the call site" begin
                vjp(x, o, dout) = (Reactant.Ops.multiply(dout, Reactant.Ops.fill(3.0f0, (4,))),)

                fwd(x) = only(embed((x,), [(4,)]; result_eltypes=(Float32,), reverse=vjp, active_operands=(0,)))
                ir = emit(fwd, Reactant.to_rarray(rand(Float32, 4)))

                @test occursin("stablehlo.custom_call @$SEAM_TARGET", ir)
                @test occursin("enzyme.reverse = @", ir)
                @test occursin("enzyme.active_operands = array<i64: 0>", ir)
                # One primal operand, one primal result, one result cotangent -> one cotangent back.
                @test reverse_signature(ir) == (3, 1)
                # The rule is only ever referenced from an attribute; it has to survive anyway.
                @test occursin(string("@\"", reverse_symbol(ir), "\"("), ir)
            end

            @testset "multi-result call: a boolean operand is excluded" begin
                B, H, Nj, Nq, hd = 1, 1, 2, 2, 2
                S4, S3 = (B, H, Nj, Nq, hd), (B, H, Nj, Nq)

                # (q, k, v, bias, mask, o, lse, dout, dlse) -> (dq, dk, dv, dbias). `dlse` is accepted and ignored, exactly as
                # the real rule ignores the residual's cotangent.
                function vjp(q, k, v, b, m, o, lse, dout, dlse)
                    return (Reactant.Ops.multiply(dout, dout), Reactant.Ops.add(dout, o), Reactant.Ops.add(dout, dout),
                        Reactant.Ops.multiply(lse, lse))
                end

                function fwd(q, k, v, b)
                    mask = Reactant.Ops.fill(false, S3)
                    o, lse = embed((q, k, v, b, mask), [S4, S3]; result_eltypes=(Float32, Float32), reverse=vjp,
                        active_operands=(0, 1, 2, 3))
                    return o
                end

                ir = emit(fwd, (Reactant.to_rarray(rand(Float32, s...)) for s in (S4, S4, S4, S3))...)

                # The mask is operand 4 and is left out: booleans carry no cotangent.
                @test occursin("enzyme.active_operands = array<i64: 0, 1, 2, 3>", ir)
                # 5 primal operands + 2 primal results + 2 result cotangents -> 4 cotangents.
                @test reverse_signature(ir) == (9, 4)
                # Both results really are results of one call, not two separate calls.
                @test occursin(Regex("custom_call @" * replace(SEAM_TARGET, "." => "\\.") * "\\([^)]*\\).*-> \\(tensor"), ir)

                # Kept for the enzymexlamlir-opt half of the loop: run this module through the rule itself.
                out = get(ENV, "KERNELDISPATCH_DUMP_MLIR", "")
                isempty(out) || (write(out, ir); @info "wrote module" out)
            end

            @testset "no rule attached when none is asked for" begin
                fwd(x) = only(embed((x,), [(4,)]; result_eltypes=(Float32,)))
                ir = emit(fwd, Reactant.to_rarray(rand(Float32, 4)))
                @test !occursin("enzyme.reverse", ir)
                @test !occursin("enzyme.active_operands", ir)
            end

            @testset "output_operand_aliases" begin
                f(x) = begin
                    y0 = Reactant.Ops.fill(0.0f0, Int64[4])
                    only(embed((y0, x), [(4,)]; result_eltypes=(Float32,), output_operand_aliases=[(0, 0)]))
                end
                x = Reactant.to_rarray(rand(Float32, 4))
                ir = emit(f, x)
                @test occursin("output_operand_alias", ir) && occursin("operand_index = 0", ir)
                g(x) = only(embed((x,), [(4,)]; result_eltypes=(Float32,)))
                @test !occursin("output_operand_alias", emit(g, x))
            end

            @testset "result_eltypes has no default" begin
                # Omitting it used to declare Float32 results silently: a Float16 kernel's output was read back as garbage.
                @test_throws UndefKeywordError embed((Reactant.to_rarray(rand(Float32, 4)),), [(4,)])
                @test_throws ArgumentError embed((Reactant.to_rarray(rand(Float32, 4)),), [(4,)]; result_eltypes=(Float32, Float32))
            end

            @testset "a result is declared in its own type: $T" for T in ALL_ELTYPES
                mlir = T === Float32 ? "f32" : T === Float16 ? "f16" : "bf16"
                x = Reactant.to_rarray(rand(Float32, 4))                       # the operand is f32, so a `T` tensor is the result
                ir = emit(x -> only(embed((x,), [(4,)]; result_eltypes=(T,))), x)
                @test occursin("-> tensor<4x$mlir>", ir)
            end

            @testset "scalars cross as little-endian bytes" begin
                @test RX._le_bytes(2 * 64) == "[128,0,0,0,0,0,0,0]"
                @test RX._le_bytes(2.0f0) == "[0,0,0,64]"
                @test RX._le_bytes(Int32(7)) == "[7,0,0,0]"
            end

            @testset "KernelDispatch.custom_call dispatches on the backend" begin
                # A backend without a method says so (ROCm has no extension).
                @test_throws ArgumentError KD.custom_call(KD.ROCmBackendTag(), "_fwd", "/none", (), [(4,)], Any[]; grid=(1, 1, 1))

                # The loaded one embeds a real binary of its own as ITS custom call (its adapter names it).
                b = last(first(explicit_binaries(mktempdir())))
                file, entry, tg = b.file, b.entry, b.threadgroup           # a captured `KernelBinary` is not traceable
                layout = Any[(:in, 0), (:out, 0)]
                call(x; kw...) = KD.custom_call(GPU_TAG, entry, file, (x,), [(4,)], layout;
                    grid=(1, 1, 1), threadgroup=tg, kw...)[1]
                x = Reactant.to_rarray(rand(Float32, 4))
                @test occursin(TRACED_CALL_TARGET, emit(x -> call(x; result_eltypes=(Float32,)), x))
                @test_throws UndefKeywordError call(x)                         # no result type, no call
                for T in (Float16, BFloat16)
                    mlir = T === Float16 ? "f16" : "bf16"
                    @test occursin("tensor<4x$mlir>", emit(x -> call(x; result_eltypes=(T,)), x))
                end
            end
        end
    finally
        Reactant.set_default_backend(seam_prev)
    end
end
