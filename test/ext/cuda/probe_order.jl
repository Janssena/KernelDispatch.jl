# Does XLA's kernel-call handler launch a kernel whose OUTPUT buffer comes before an input (or between scalars)?
# Run on an NVIDIA machine and paste the output back:
#
#     julia --project=test test/ext/cuda/probe_order.jl
#
# Background: the handler is given a launch's buffers as operands, then results, and binds the kernel's buffer
# parameters to them in that order, so today a traced CUDA launch needs every input before every output
# (`_cuda_params` refuses anything else). Kernels cannot be asked to follow that: KA kernels are `f!(out, in...)`,
# trifast's Triton kernels mix both orders, and prebuilt binaries cannot be reordered. The idea tested here: pass the
# output buffer to the handler as an OPERAND too (a zero-filled array), with the result aliased to it
# (`output_operand_aliases`), so the buffers arrive in the kernel's own order. What is unknown is whether the handler
# accepts more buffers (operands + results) than the kernel has array parameters.
#
# The hand-written `scale` PTX (test/ext/cuda/ptx.jl) in three parameter orders, each launched from `@jit` with and without
# the aliasing. Each case runs in a child process (an XLA fatal error then costs only that case):
#
#   order  xyan   inputs first (today's contract)       plain: expected PASS (the control)
#   order  yxan   output first (`f!(out, in...)`)       plain: expected to fail; alias: the question
#   order  xayn   a scalar between the two buffers      plain: expected PASS; alias: the question
#
# RESULT (A40, probe_order.log): the aliased-operand layout works in every order; plain output-first gives silently wrong
# values. It is now what `custom_call(::CUDABackendTag)` emits.
#
# The launches use the `triton_kernel_call_ffi@4` convention, PTX images, built by the package's own encoder.

using Random, KernelDispatch, CUDA
const GPU_PKG = CUDA
include(joinpath(@__DIR__, "ptx.jl"))     # SCALE_PTX

# The scale kernel with its parameters in another order (the body reads them by name).
function variant_ptx(order::Vector{String}, entry::String)
    i = first(findfirst(".visible .entry scale(", SCALE_PTX))
    j = first(findfirst("\n)", SCALE_PTX))
    decl(p) = "    .param " * (p == "a" ? ".f32" : p == "n" ? ".u32" : ".u64") * " " * p
    return SCALE_PTX[1:i-1] * ".visible .entry $entry(\n" * join(decl.(order), ",\n") * SCALE_PTX[j:end]
end

const ORDERS = ["xyan" => ["x", "y", "a", "n"], "yxan" => ["y", "x", "a", "n"], "xayn" => ["x", "a", "y", "n"]]

function run_case(order_name, order, mode)
    entry = "scale_" * order_name
    ptx_file = joinpath(mktempdir(), entry * ".ptx")
    write(ptx_file, variant_ptx(order, entry))
    code = quote
        using KernelDispatch, CUDA, Reactant
        Reactant.set_default_backend("gpu")
        const RXC = Base.get_extension(KernelDispatch, :KernelDispatchReactantExt)
        ptx_file, entry, order, mode = $(ptx_file), $(entry), $(order), $(QuoteNode(mode))
        n = 100
        spec = Dict("x" => (:array, 0), "y" => (:array, 0),
            "a" => (:scalar, collect(reinterpret(UInt8, [2.0f0]))),
            "n" => (:scalar, collect(reinterpret(UInt8, [Int32(n)]))))
        proto = RXC._kernel_call_proto(; entry, image=read(ptx_file), is_ptx=true, num_warps=2, shared_mem=0,
            grid=(cld(n, 64), 1, 1), params=[spec[p] for p in order], name=entry)
        config = Dict("opaque" => String(RXC._zlib_stored(proto)))
        function f(x)
            if mode === :plain          # the output is only a result: the handler gets [x] then [y]
                operands, aliases = (x,), Tuple{Int,Int}[]
            else                         # the output is an operand too, in the kernel's buffer order, aliased
                y0 = Reactant.Ops.fill(0.0f0, (n,))
                buffers = [p == "y" ? y0 : x for p in order if p in ("x", "y")]
                operands, aliases = Tuple(buffers), [(0, findfirst(==("y"), filter(p -> p in ("x", "y"), order)) - 1)]
            end
            only(RXC._embed_custom_call("triton_kernel_call_ffi", operands, [(n,)]; backend_config=config,
                api_version=4, output_operand_aliases=aliases, label=entry, result_eltypes=(Float32,)))
        end
        off = Reactant.CompileOptions(; raise_triton_custom_call=false)
        x = randn(Float32, n)
        y = Array(Reactant.@jit compile_options = off f(Reactant.to_rarray(x)))
        println(y ≈ 2.0f0 .* x ? "RESULT ok" : "RESULT wrong values (max error $(maximum(abs, y .- 2.0f0 .* x)))")
    end
    src = join(string.(filter(a -> !(a isa LineNumberNode), code.args)), "\n")
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e $src`
    out = IOBuffer()
    proc = run(pipeline(ignorestatus(cmd); stdout=out, stderr=out))
    text = String(take!(out))
    return success(proc) && occursin("RESULT ok", text), text
end

println("Julia ", VERSION, " on ", Sys.MACHINE)
CUDA.functional() || (println("CUDA.functional() is false: no usable NVIDIA device/driver found; stopping."); exit(1))
println("device: ", CUDA.name(CUDA.device()), ", compute capability ", CUDA.capability(CUDA.device()))

rows = Tuple{String,Symbol,Bool}[]
for (name, order) in ORDERS, mode in (:plain, :alias)
    print(rpad("order $name / $mode", 28))
    ok, text = run_case(name, order, mode)
    println(ok ? "PASS" : "FAIL")
    if !ok
        lines = filter(l -> occursin(r"error|Error|ERROR|No registered|custom call|FATAL|RESULT|buffer|operand|parameter", l), split(text, "\n"))
        for l in first(lines, 6)
            println("    ", first(l, 220))
        end
    end
    push!(rows, (name, mode, ok))
end

println("\n════════ summary ════════")
for (name, mode, ok) in rows
    println(ok ? "PASS  " : "FAIL  ", "order ", name, " / ", mode)
end
println("""

Reading it: 'xyan / plain' is the control and should PASS. If 'yxan / alias' and 'xayn / alias' PASS, the handler
accepts aliased operands and the ordering restriction can be lifted for every binary (the launcher passes each output
as an aliased operand). If they FAIL, the first error lines above say why (a buffer-count check, most likely), and the
fallback is the KA-only rotation in the plan. Please paste the whole output.""")
