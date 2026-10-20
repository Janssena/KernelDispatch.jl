# trifast's real triangle-attention kernels (test/fixtures/trifast/, made by test/fixtures/make_trifast_fixture.py: Triton
# 3.8.0 compiled `_fwd`, `_bwd_q`, `_bwd_kv`, `_bwd_b` at several autotune configs), launched through KernelDispatch on CUDA,
# eagerly and from a Reactant trace. Each is compared with Triton's own output for the same inputs (the same binary run by
# torch) and with a DENSE float64 torch reference, and the four kernels are chained (the backward kernels read what the
# forward and `_bwd_q` wrote) in one launch sequence and in one trace.
#
# What this exercises beyond test/ext/cuda/triton.test.jl: 29-40 arguments per kernel (a dozen strides each, the ones
# equal to 1 specialised away), a 3-D grid, `tl.dot` (shared memory from the pipeline stages), a Bool mask, outputs
# first or last, a mask indexed per batch item (`pid // H`) and a sequence length (40) that no block size divides.

using Test, KernelDispatch, Reactant, LinearAlgebra
isdefined(@__MODULE__, :TRITON_SETS) || include(joinpath(@__DIR__, "..", "..", "triton_fixtures.jl"))
isdefined(@__MODULE__, :TRIFAST_FIXTURES) || include(joinpath(@__DIR__, "..", "..", "trifast_fixtures.jl"))

if GPU === :cuda && have_trifast_fixtures()
    let
        case = trifast_case()
        insts = case["instances"]
        plans = Dict(i["name"] => trifast_plan(case, i) for i in insts)
        specs(inst, image) = triton_spec(triton_binary(trifast_json(inst); image=trifast_image(inst, image), signature=false))
        # The inputs every kernel may read (the forward's and `_bwd_q`'s outputs are saved as data too).
        base = ("d", "do", "k", "l", "mask", "o", "q", "v", "b")

        @info "trifast configs" shared_memory = Dict(i["name"] => i["shared"] for i in insts)

        @testset "eager: $(inst["name"]), $image" for inst in insts, image in (:cubin, :ptx)
            spec = specs(inst, image)
            @test spec.shared == inst["shared"] && spec.threadgroup == 32 * inst["num_warps"]
            tensors = Dict(n => DeviceArray(trifast_data(case, n)) for n in base)
            outs = trifast_launch(spec, plans[inst["name"]], tensors)
            for (b, info) in inst["outputs"]
                y = Array(outs[info["tensor"]])
                @test trifast_relerr(y, trifast_expected(case, inst, b)) < 1e-5     # Triton's own output
                # the dense reference, except `d`, which is the delta of Triton's own forward output
                @test trifast_relerr(y, trifast_data(case, info["reference"])) < 1e-4
            end
        end

        # The primary (first) config of each kernel, chained: forward -> _bwd_q -> _bwd_kv, _bwd_b.
        chain = [first(filter(i -> i["kernel"] == k, insts)) for k in ("_fwd", "_bwd_q", "_bwd_kv", "_bwd_b")]
        final = ("o", "l", "dq", "dk", "dv", "db")
        # Binaries are described outside the trace (file reads) and turned into KernelBinarys inside it.
        chain_specs = Dict(image => Tuple(specs(i, image) for i in chain) for image in (:cubin, :ptx))
        chain_plans = Tuple(plans[i["name"]] for i in chain)
        function run_chain(image, q, k, v, b, mask, dout)
            t = Dict{String,Any}("q" => q, "k" => k, "v" => v, "b" => b, "mask" => mask, "do" => dout)
            for (spec, plan) in zip(chain_specs[image], chain_plans)
                merge!(t, trifast_launch(spec, plan, t))
            end
            return map(n -> t[n], final)
        end
        reference = (trifast_data(case, "ref_o"), trifast_data(case, "ref_l"), trifast_data(case, "ref_dq"),
                     trifast_data(case, "ref_dk"), trifast_data(case, "ref_dv"), trifast_data(case, "ref_db"))
        raw = ntuple(i -> trifast_data(case, ("q", "k", "v", "b", "mask", "do")[i]), 6)

        @testset "eager chain: forward then the backward kernels" begin
            res = run_chain(:cubin, map(DeviceArray, raw)...)
            for (n, y, r) in zip(final, res, reference)
                @test trifast_relerr(Array(y), r) < 1e-4
            end
        end

        prev = Reactant.XLA.default_backend()
        Reactant.set_default_backend(traced_client())
        try
            @testset "traced: $(inst["name"]), $image" for inst in insts, image in (:cubin, :ptx)
                spec = specs(inst, image)
                args = map(n -> Reactant.to_rarray(trifast_data(case, n)), base)
                plan = plans[inst["name"]]
                function f(args...)
                    outs = trifast_launch(spec, plan, Dict(zip(base, args)))
                    return map(t -> outs[t], plan.out_tensors)
                end
                res = @jit f(args...)
                for (b, info) in inst["outputs"]
                    y = Array(res[findfirst(==(info["tensor"]), plan.out_tensors)])
                    @test trifast_relerr(y, trifast_expected(case, inst, b)) < 1e-5
                    @test trifast_relerr(y, trifast_data(case, info["reference"])) < 1e-4
                end
            end

            @testset "traced chain: one program, four launches" begin
                for image in (:cubin, :ptx)
                    args = map(a -> Reactant.to_rarray(a), raw)
                    f(args...) = run_chain(image, args...)
                    res = @jit f(args...)
                    for (n, y, r) in zip(final, res, reference)
                        @test trifast_relerr(Array(y), r) < 1e-4
                    end
                end
            end
        finally
            Reactant.set_default_backend(prev)
        end
    end
end
