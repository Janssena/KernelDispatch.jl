# Debugging the traced launch of a Triton kernel in half precision: runs one fixture kernel from the cubin and from the PTX,
# several times and in a fixed order, in a fresh process, and prints how each result differs from Triton's own output.
#
#     julia --project=test test/ext/cuda/triton_half_debug.jl float16 scale_kernel [3.2.0]
#     julia --project=test test/ext/cuda/triton_half_debug.jl float32 scale_kernel      # the same for the dtype that passes
#
# Paste the output. What it separates: does the first traced launch go wrong whatever the image, or only the cubin; does a
# second launch of the same binary differ; does it matter that nothing else ran before in this process.

using Test, KernelDispatch, Reactant, CUDA, BFloat16s
include(joinpath(@__DIR__, "..", "..", "triton_fixtures.jl"))
include(joinpath(@__DIR__, "..", "..", "precision.jl"))

dtype = get(ARGS, 1, "float16")
kname = get(ARGS, 2, "scale_kernel")
T = Dict("float32" => Float32, "float16" => Float16, "bfloat16" => BFloat16)[dtype]
version = get(ARGS, 3, nothing)
set = first(filter(s -> s.T === T && (version === nothing || s.version == version), TRITON_SETS))
k = only(filter(k -> k.name == kname, triton_kernels(set)))
files = Dict("scale_kernel" => ("x",), "block_sum_kernel" => ("x",), "strided_axpy" => ("x", "y"),
    "accumulate_kernel" => ("y_init", "x"), "atomic_sum_kernel" => ("x",))
p = triton_paths(kname, set)
data = Tuple(load_words(p.dir, f, T) for f in files[kname])
expected = load_words(p.dir, "expected", T)

calls = Dict(
    "scale_kernel" => (b, x) -> only(run_binary(b, x, OutArray(eltype(x), 100), 2.0f0, Int32(100); extent=cld(100, 64))),
    "block_sum_kernel" => (b, x) -> only(run_binary(b, x, OutArray(eltype(x), cld(1000, 128)), Int32(1000); extent=cld(1000, 128))),
    "strided_axpy" => (b, x, y) -> only(run_binary(b, OutArray(eltype(x), 150; zero=true), Int32(3), x, y, Int32(2), 3.0f0,
        Int32(50); extent=cld(50, 32))),
    "accumulate_kernel" => (b, y, x) -> only(run_binary(b, InOutArray(y), x, 0.5f0, Int32(100); extent=cld(100, 64))))
call = calls[kname]

Reactant.set_default_backend("gpu")
println("kernel $kname, ", set.label)
for (label, image) in (("cubin #1", :cubin), ("ptx   #1", :ptx), ("cubin #2", :cubin), ("ptx   #2", :ptx), ("cubin #3", :cubin))
    s = triton_spec(triton_binary(p.json; image=getfield(p, image), signature=triton_signature(k, T)))
    args = map(Reactant.to_rarray, data)
    f(args...) = call(triton_mkbin(s), args...)
    y = Array(@jit f(args...))
    wrong = findall(i -> !(y[i] === expected[i] || isapprox(Float64(y[i]), Float64(expected[i]); rtol=1e-2, atol=1e-2)), eachindex(y))
    println(label, ": maxrelerr ", round(maxrelerr(y, expected); sigdigits=3), ", ", length(wrong), " of ", length(y), " elements differ")
    if !isempty(wrong)
        w = first(wrong, 6)
        println("   first wrong indices ", w)
        println("   got      ", Float32.(y[w]))
        println("   expected ", Float32.(expected[w]))
    end
end
