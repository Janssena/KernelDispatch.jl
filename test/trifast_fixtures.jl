# The trifast fixtures (test/fixtures/trifast/, made by test/fixtures/make_trifast_fixture.py with real Triton on an NVIDIA
# machine and kept in the fixtures artifact, see fixture_root.jl): trifast's triangle-attention kernels (`_fwd`, `_bwd_q`, `_bwd_kv`, `_bwd_b`) compiled at several
# (BLOCK_J, BLOCK_K, num_warps, num_stages) configs, the inputs, Triton's own outputs and a dense float64 torch reference.
# `case.json` describes every instance: its grid and the arguments the COMPILED kernel takes, in order (so the tests need
# no knowledge of trifast). Tensors are raw row-major files; Julia reads them with the dimensions reversed, which is the
# same memory the kernel addresses with torch's strides.

isdefined(@__MODULE__, :fixture_dir) || include(joinpath(@__DIR__, "fixture_root.jl"))
const TRIFAST_FIXTURES = fixture_dir("trifast")        # the `trifast/` directory, or "" if there is none (see fixture_root.jl)
have_trifast_fixtures() = !isempty(TRIFAST_FIXTURES) && isfile(joinpath(TRIFAST_FIXTURES, "case.json"))
trifast_case() = KernelDispatch._json_parse(read(joinpath(TRIFAST_FIXTURES, "case.json"), String))

# A tensor file as a Julia array (dimensions reversed from torch's).
function trifast_read(file::AbstractString, info)
    bytes = read(file)
    v = info["dtype"] == "bool" ? collect(reinterpret(Bool, bytes)) : collect(reinterpret(Float32, bytes))
    return reshape(v, reverse(Int.(info["shape"]))...)
end
trifast_data(case, name) = trifast_read(joinpath(TRIFAST_FIXTURES, "data", name * ".bin"), case["tensors"][name])
trifast_expected(case, inst, base) =
    trifast_read(joinpath(TRIFAST_FIXTURES, inst["dir"], inst["outputs"][base]["file"]),
        case["tensors"][inst["outputs"][base]["reference"]])
trifast_json(inst) = joinpath(TRIFAST_FIXTURES, inst["dir"], inst["entry"] * ".json")
trifast_image(inst, image) = joinpath(TRIFAST_FIXTURES, inst["dir"], inst["entry"] * "." * String(image))

# What a launch of `inst` needs, as plain immutable values (it is captured by a traced function, where a Vector of mixed
# tuples or Symbols does not trace): `(args, out_tensors, grid)`. Each of `args` is `(code, tensor, number, dims)` with
# `code` 1 = an array `tensor`, 2 = an output `tensor` of size `dims`, 3 = an Int32 `number`, 4 = a Float32 `number`.
# `out_tensors` are the outputs' names, in the order `run_binary` returns them.
function trifast_plan(case, inst)
    outs = inst["outputs"]
    args = Tuple{Int,String,Float64,Tuple{Vararg{Int}}}[]
    out_tensors = String[]
    for a in inst["args"]
        k = a["kind"]
        if k == "i32"
            push!(args, (3, "", Float64(a["value"]), ()))
        elseif k == "f32"
            push!(args, (4, "", Float64(a["value"]), ()))
        elseif haskey(outs, a["base"])
            ref = case["tensors"][outs[a["base"]]["reference"]]
            push!(args, (2, a["tensor"], 0.0, Tuple(reverse(Int.(ref["shape"])))))
            push!(out_tensors, a["tensor"])
        else
            push!(args, (1, a["tensor"], 0.0, ()))
        end
    end
    return (; args=Tuple(args), out_tensors=Tuple(out_tensors), grid=Tuple(Int.(inst["grid"])))
end

# Launch `plan` on `tensors` (name => array, device or traced): a Dict of the outputs by tensor name.
function trifast_launch(spec, plan, tensors)
    bound = map(plan.args) do (code, name, num, dims)
        code == 1 ? tensors[name] :
        code == 2 ? OutArray(Float32, dims; zero=true) :
        code == 3 ? Int32(round(Int, num)) : Float32(num)
    end
    outs = run_binary(triton_mkbin(spec), bound...; extent=plan.grid)
    return Dict(zip(plan.out_tensors, outs))
end

trifast_relerr(y, r) = sqrt(sum(abs2, Float64.(y) .- Float64.(r)) / sum(abs2, Float64.(r)))
