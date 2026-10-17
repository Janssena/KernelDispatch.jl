# Which custom-call convention does the XLA behind Reactant's CUDA client accept for launching a cubin?
# Run on an NVIDIA machine and paste the table back:
#
#     julia --project=test test/cuda_probe.jl
#
# Each case runs in a child Julia process (an XLA fatal error then costs only that case) and launches the
# hand-written `scale` kernel (test/cuda_ptx.jl) from a Reactant trace, once per combination of
#
#     calling convention   triton_kernel_call_ffi@4   (jax-triton's typed-FFI form; the default)
#                          triton_kernel_call@2       (legacy, status-returning)
#                          triton_kernel_call@4       (typed FFI under the legacy name)
#     kernel image         cubin  (the handler's `module_image`)    ptx  (assembled by the handler)
#
# and checks the result against the host. A PASS row names the setting to use:
#
#     KERNELDISPATCH_CUDA_CUSTOM_CALL=<convention>
#
# If every row fails, the XLA in this Reactant build has no handler that launches a kernel image, and traced
# CUDA launches need another route (see the notes printed at the end).

using Random

const CONVENTIONS = ["triton_kernel_call_ffi@4", "triton_kernel_call@2", "triton_kernel_call@4"]
const IMAGES = ["cubin", "ptx"]

function run_case(convention, image)
    using_code = quote
        using KernelDispatch, CUDA, Reactant
        const GPU_PKG = CUDA
        include($(joinpath(@__DIR__, "cuda_ptx.jl")))
        bins = scale_binaries(mktempdir())
        bin = bins[$(QuoteNode(Symbol(image)))]
        file, entry, tg = bin.file, bin.entry, bin.threadgroup
        Reactant.set_default_backend("gpu")
        println("platform: ", Reactant.XLA.platform_name(Reactant.XLA.default_backend()))
        loose() = KernelBinary(file; entry, threadgroup=tg)
        f(x) = only(run_binary(loose(), x, OutArray(Float32, 100), 2.0f0, Int32(100); extent=100))
        x = randn(Float32, 100)
        # Reactant rewrites a custom call named `triton_kernel_call` into Triton IR unless told not to.
        off = Reactant.CompileOptions(; raise_triton_custom_call=false)
        y = Array(Reactant.@jit compile_options = off f(Reactant.to_rarray(x)))
        println(y ≈ 2f0 .* x ? "RESULT ok" : "RESULT wrong values (max error $(maximum(abs, y .- 2f0 .* x)))")
    end
    code = join(string.(filter(a -> !(a isa LineNumberNode), using_code.args)), "\n")
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e $code`
    cmd = addenv(cmd, "KERNELDISPATCH_CUDA_CUSTOM_CALL" => convention)
    out = IOBuffer()
    proc = run(pipeline(ignorestatus(cmd); stdout=out, stderr=out))
    text = String(take!(out))
    ok = success(proc) && occursin("RESULT ok", text)
    return ok, text
end

println("Julia ", VERSION, " ", Sys.MACHINE)
rows = Tuple{String,String,Bool,String}[]
for conv in CONVENTIONS, image in IMAGES
    print(rpad("$conv / $image", 42))
    ok, text = run_case(conv, image)
    println(ok ? "PASS" : "FAIL")
    if !ok
        # the first informative lines: Julia/XLA/MLIR errors
        lines = filter(l -> occursin(r"error|Error|ERROR|No registered|custom call|FATAL|RESULT", l), split(text, "\n"))
        for l in first(lines, 6)
            println("    ", first(l, 200))
        end
    end
    push!(rows, (conv, image, ok, text))
end

println("\n════════ summary ════════")
for (conv, image, ok, _) in rows
    println(ok ? "PASS  " : "FAIL  ", conv, " / ", image)
end
winners = unique([c for (c, _, ok, _) in rows if ok])
if isempty(winners)
    println("""

No convention launched the kernel. Please paste the output above, plus the result of
    strings -a \$(julia --project=test -e 'using Reactant_jll; print(Reactant_jll.libReactantExtra)') | grep -i triton
so the next step can be chosen (a handler of our own, or Reactant's `enzymexla.kernel_call` with kernel IR).""")
else
    println("\nUse: KERNELDISPATCH_CUDA_CUSTOM_CALL=", first(winners))
end
