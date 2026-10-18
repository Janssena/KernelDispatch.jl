# Can a KernelAbstractions kernel be launched from a Reactant trace on CUDA as a *binary*? Run on an NVIDIA
# machine and paste the whole output back:
#
#     julia --project=test test/cuda_probe_ka.jl
#
# Background: XLA's `triton_kernel_call` handler passes a kernel only device pointers and 1/4/8-byte scalars. A
# KA binary takes CUDA.jl's `KernelState` and KA's launch context as struct arguments, so it cannot go through
# it as is (`traced_support` refuses it). The question here is whether the parameter list can be made flat
# (the `:ka_flat` idea), as opposed to tracing the kernel's Julia source with Reactant's own CUDA pipeline.
# This script only observes; it changes nothing in the package.
#
#   1  the compiled signature of a KA kernel as `ka_compile` builds it (what the handler would have to pass)
#   2  a wrapper entry taking only pointers and scalars; it rebuilds KA's context (six integers) on the device:
#        2.0 the rebuilt context equals KA's (host only)
#        2a  compiled by CUDA.jl as is (its hidden KernelState first parameter should survive)
#        2b  compiled with the kernel state switched off for that job
#   3  launch the best flat variant: eagerly (ground truth), then from a Reactant trace through the handler
#   4  what Reactant's own KA path lowers to on CUDA (which custom call, if any, it ends in)
#
# Each stage prints PASS or FAIL with the full error. The PTX parameter lists and the custom-call target are
# the findings; they are printed, not asserted.

using Random, KernelDispatch, KernelAbstractions, CUDA

const KA = KernelAbstractions
const KO = KernelDispatch
const CUDACore = CUDA.CUDACore
const GPUC = CUDACore.GPUCompiler
const EXT = Base.get_extension(KO, :KernelDispatchCUDAExt)
const TAG = KO.CUDABackendTag()

const RESULTS = Pair{String,Bool}[]
function stage(f, name)
    print("\n── ", name, " ")
    try
        f()
        println("PASS")
        push!(RESULTS, name => true)
    catch err
        println("FAIL")
        showerror(stdout, err, catch_backtrace())
        println()
        push!(RESULTS, name => false)
    end
end
check(cond, msg="check failed") = cond || error(msg)

println("Julia ", VERSION, " on ", Sys.MACHINE)
println("CUDA.jl ", pkgversion(CUDA), ", GPUCompiler ", pkgversion(GPUC), ", KernelDispatch ", pkgversion(KO))
if CUDA.functional()
    dev = CUDA.device()
    println("device: ", CUDA.name(dev), ", compute capability ", CUDA.capability(dev))
else
    println("CUDA.functional() is false: no usable NVIDIA device/driver found; stopping.")
    exit(1)
end

# --- the kernel: written to the binary contract (raw device pointers), like the package's own fixtures --------

@kernel unsafe_indices = true function probe_scale!(y, x, a::Float32, n::Int32, ::Val{TG}) where {TG}
    i = (@index(Group, Linear) - 1) * TG + @index(Local, Linear)
    if i <= n
        unsafe_store!(y, a * unsafe_load(x, i), i)
    end
end

const TG = 64
const ARGT = (Vector{Float32}, Vector{Float32}, Float32, Int32, Val{TG})   # y, x, a, n, Val

# --- helpers ----------------------------------------------------------------------------------------------------

# The PTX text of one kernel, built exactly like `CUDACore.compile` does before it runs `ptxas`.
function ptx_of(ftype::Type, sig::Type; name, config=nothing)
    source = GPUC.methodinstance(ftype, sig)
    config === nothing && (config = CUDACore.compiler_config(CUDACore.device(); kernel=true, name))
    job = GPUC.CompilerJob(source, config)
    return GPUC.JuliaContext() do ctx
        asm, _ = Base.invokelatest(GPUC.compile, :asm, job)
        asm
    end
end

# A cubin from the PTX of the same job, by assembling it with the `ptxas` CUDA.jl ships. (Not `CUDACore.compile`:
# it runs in a frozen world, in which this script's `NoStateParams` methods do not exist yet.)
function cubin_of(ftype::Type, sig::Type; name, config=nothing)
    ptx = ptx_of(ftype, sig; name, config)
    entry = match(r"\.entry\s+(\S+?)\s*\(", ptx)[1]
    dir = mktempdir()
    ptx_file, cubin_file = joinpath(dir, name * ".ptx"), joinpath(dir, name * ".cubin")
    write(ptx_file, ptx)
    run(`$(CUDACore.CUDA_Compiler.ptxas()) --gpu-name $(KO.target_id(TAG)) --output-file $cubin_file $ptx_file`)
    return read(cubin_file), String(entry)
end

# The `.param` lines of the first `.entry`: the kernel's real parameter list.
function ptx_params(ptx::AbstractString)
    i = findfirst(".entry", ptx)
    i === nothing && return String[]
    j = findnext("{", ptx, first(i))
    j === nothing && return String[]
    return [strip(l) for l in split(ptx[first(i):first(j)], '\n') if occursin(".param", l)]
end

function show_params(title, ptx)
    ps = ptx_params(ptx)
    println("\n   ", title, ": ", length(ps), " parameter(s)")
    for p in ps
        println("      ", p)
    end
    return ps
end

# The real KA kernel body, as a type: the wrapper calls its singleton instance.
const KOBJ = probe_scale!(CUDABackend(), TG)
const F = typeof(KOBJ.f)

# KA's launch context for a static workgroup size is six integers (stage 1: 48 bytes): the ND-range and the
# block counts, as two `CartesianIndices`. Rebuilt here from them directly, with none of `KA.launch_config` /
# `partition` (those have error paths, which a device cannot compile). Equal to `KO.ka_context`'s (stage 2.0).
@inline function rebuild_ctx(::Val{TG}, nd1::Int, nd2::Int, nd3::Int, b1::Int, b2::Int, b3::Int) where {TG}
    ND = KA.NDIteration
    iterspace = ND.NDRange{3,ND.DynamicSize,ND.StaticSize{(TG, 1, 1)}}(CartesianIndices((b1, b2, b3)), nothing)
    return KA.CompilerMetadata{ND.DynamicSize,ND.DynamicCheck}(CartesianIndices((nd1, nd2, nd3)), iterspace)
end

# A kernel entry with a flat signature: the six context integers, the arrays as pointers, the scalars. It
# rebuilds the context on the device and calls the real kernel body. x before y: inputs before outputs, as XLA's
# handler wants.
struct Flat{F,TG} end
function (::Flat{F,TG})(nd1::Int, nd2::Int, nd3::Int, b1::Int, b2::Int, b3::Int,
        x, y, a::Float32, n::Int32) where {F,TG}
    ctx = rebuild_ctx(Val(TG), nd1, nd2, nd3, b1, b2, b3)
    F.instance(ctx, y, x, a, n, Val(TG))
    return nothing
end
const FLAT = Flat{F,TG}
const FLAT_SIG = Tuple{Int,Int,Int,Int,Int,Int,EXT.dptr(Float32),EXT.dptr(Float32),Float32,Int32}

# The six integers `KO.ka_context` would put in the context, for `groups` threadgroups.
flat_ints(groups::NTuple{3,Int}) = (groups[1] * TG, groups[2], groups[3], groups[1], groups[2], groups[3])

# A job whose compiler does not add CUDA.jl's hidden `KernelState` first parameter.
struct NoStateParams <: CUDACore.AbstractCUDACompilerParams
    inner::CUDACore.CUDACompilerParams
end
Base.getproperty(p::NoStateParams, s::Symbol) = getproperty(getfield(p, :inner), s)
GPUC.kernel_state_type(::GPUC.CompilerJob{GPUC.PTXCompilerTarget,NoStateParams}) = Nothing
function stateless_config(name)
    cfg = CUDACore.compiler_config(CUDACore.device(); kernel=true, name)
    return GPUC.CompilerConfig(cfg; params=NoStateParams(cfg.params))
end

# --- 1: what the handler would have to pass --------------------------------------------------------------------

stage("1 signature of a KA binary (what ka_compile builds)") do
    ctx, ghost = KO.ka_context(TAG, (2, 1, 1), TG)
    println("\n   KernelState: ", sizeof(CUDACore.KernelState), " bytes; KA context: ", sizeof(ctx),
        " bytes, ghost=", ghost, "\n   context type: ", typeof(ctx))
    sig = Tuple{typeof(ctx),map(EXT._ka_argtype, ARGT)...}
    ptx = ptx_of(F, sig; name="probe_ka")
    ps = show_params("probe_ka", ptx)
    check(!isempty(ps), "no .param lines found in the PTX")
end

# --- 2: a flat wrapper -------------------------------------------------------------------------------------------

# The errors of a failed device compile are long; their distinct "Reason:" lines are what matters.
function compact(err)
    text = sprint(showerror, err)
    reasons = unique([strip(l) for l in split(text, '\n') if startswith(l, "Reason:")])
    return isempty(reasons) ? first(text, 600) : join(reasons, "\n      ")
end

stage("2.0 the rebuilt context equals KA's (host only)") do
    for groups in ((2, 1, 1), (3, 2, 1), (1, 1, 4))
        ref, _ = KO.ka_context(TAG, groups, TG)
        new = rebuild_ctx(Val(TG), flat_ints(groups)...)
        check(typeof(new) === typeof(ref), "types differ:\n  $(typeof(new))\n  $(typeof(ref))")
        bits(c) = (r = Ref(c); GC.@preserve r unsafe_load(Ptr{NTuple{6,Int}}(Base.unsafe_convert(Ptr{Cvoid}, r))))
        println("\n   groups ", groups, ": KA ", bits(ref), "  rebuilt ", bits(new))
        check(bits(new) == bits(ref), "the six integers differ")
    end
end

flat_ptx = Ref("")
stage("2a flat wrapper, CUDA.jl's default job") do
    ptx = try
        ptx_of(FLAT, FLAT_SIG; name="probe_flat")
    catch err
        println("\n   compile failed: ", compact(err))
        rethrow()
    end
    flat_ptx[] = ptx
    ps = show_params("probe_flat", ptx)
    println("   -> a leading `.b8` blob here is CUDA.jl's KernelState; only `.u64`/`.u32`/`.f32` would be flat")
    check(!isempty(ps), "no .param lines found in the PTX")
end

stateless_ok = Ref(false)
stage("2b flat wrapper, kernel state switched off") do
    ptx = try
        ptx_of(FLAT, FLAT_SIG; name="probe_flat0", config=stateless_config("probe_flat0"))
    catch err
        println("\n   compile failed: ", compact(err))
        rethrow()
    end
    ps = show_params("probe_flat0", ptx)
    n_blob = count(p -> occursin(".b8", p), ps)
    println("   -> struct-like (.b8) parameters: ", n_blob)
    stateless_ok[] = n_blob == 0 && !isempty(ps)
    check(stateless_ok[], "a struct parameter is still there")
end

# --- 3: launch the best flat variant ----------------------------------------------------------------------------

flat_file = Ref("")
flat_entry = Ref("")
stage("3a flat wrapper, eager launch vs host (ground truth)") do
    name = stateless_ok[] ? "probe_flat0" : "probe_flat"
    cfg = stateless_ok[] ? stateless_config(name) : nothing
    image, entry = cubin_of(FLAT, FLAT_SIG; name, config=cfg)
    file = joinpath(mktempdir(), name * ".cubin")
    write(file, image)
    flat_file[], flat_entry[] = file, entry
    println("\n   built ", stateless_ok[] ? "state-free" : "WITH the hidden KernelState", " cubin: ", file)
    bin = KernelBinary(file; entry, threadgroup=TG, tile=TG)
    n = 100
    x = randn(Float32, n)
    groups = cld(n, TG)
    args = (flat_ints((groups, 1, 1))..., CuArray(x), OutArray(Float32, n), 2.0f0, Int32(n))
    # With the state present, eager CUDA can still supply it: it is just a leading by-value argument.
    (y,) = stateless_ok[] ? run_binary(bin, args...; extent=n) : run_binary(bin, EXT._kernel_state(), args...; extent=n)
    check(Array(y) ≈ 2.0f0 .* x, "wrong values (max error $(maximum(abs, Array(y) .- 2.0f0 .* x)))")
end

# The traced launch runs in a child process: an XLA fatal error then costs only this case.
stage("3b flat wrapper, traced through XLA's kernel-call handler") do
    check(isfile(flat_file[]), "no cubin from stage 3a")
    stand_in = !stateless_ok[]
    code = quote
        using KernelDispatch, CUDA, Reactant
        Reactant.set_default_backend("gpu")
        file, entry, tg, stand_in, ints = $(flat_file[]), $(flat_entry[]), $(TG), $(stand_in), $(flat_ints((cld(100, TG), 1, 1)))
        loose() = KernelBinary(file; entry, threadgroup=tg, tile=tg)
        n = 100
        x = randn(Float32, n)
        # The state word, when the kernel still has one: a zero 8-byte scalar in the slot of its first parameter.
        function f(x)
            y = if stand_in
                run_binary(loose(), UInt64(0), ints..., x, OutArray(Float32, n), 2.0f0, Int32(n); extent=n)
            else
                run_binary(loose(), ints..., x, OutArray(Float32, n), 2.0f0, Int32(n); extent=n)
            end
            only(y)
        end
        off = Reactant.CompileOptions(; raise_triton_custom_call=false)
        y = Array(Reactant.@jit compile_options = off f(Reactant.to_rarray(x)))
        println(y ≈ 2.0f0 .* x ? "RESULT ok" : "RESULT wrong values (max error $(maximum(abs, y .- 2.0f0 .* x)))")
    end
    src = join(string.(filter(a -> !(a isa LineNumberNode), code.args)), "\n")
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e $src`
    out = IOBuffer()
    proc = run(pipeline(ignorestatus(cmd); stdout=out, stderr=out))
    text = String(take!(out))
    println("\n   (state ", stand_in ? "still present: passed as one 8-byte zero scalar, a stand-in" : "absent", ")")
    ok = success(proc) && occursin("RESULT ok", text)
    if !ok
        lines = filter(l -> occursin(r"error|Error|ERROR|No registered|custom call|FATAL|RESULT", l), split(text, "\n"))
        for l in first(lines, 8)
            println("      ", first(l, 220))
        end
    end
    check(ok, "the traced launch did not produce the host result")
end

# --- 4: what Reactant's own KA path lowers to --------------------------------------------------------------------

@kernel function probe_square!(y, @Const(x))
    i = @index(Global)
    @inbounds y[i] = x[i] * x[i]
end

using Reactant       # loaded only now: stages 1-3 are plain CUDA.jl

stage("4 what Reactant lowers a KA kernel to on CUDA") do
    Reactant.set_default_backend("gpu")
    x = Reactant.to_rarray(Float32.(1:64) ./ 64)
    square(x) = (y = similar(x); probe_square!(KA.get_backend(x))(y, x; ndrange=length(x)); y)
    mlir = repr(Reactant.@code_hlo square(x))
    println("\n   custom calls in the MLIR after Reactant's pipeline (first lines):")
    for l in first(filter(l -> occursin(r"custom_call|kernel_call|call_target_name", l), split(mlir, "\n")), 6)
        println("      ", first(strip(l), 260))
    end
    hlo = try
        repr(Reactant.@code_xla square(x))
    catch err
        "(@code_xla failed: $(sprint(showerror, err)))"
    end
    println("   custom-call targets in XLA's HLO (first lines):")
    for l in first(filter(l -> occursin(r"custom-call|custom_call_target", l), split(hlo, "\n")), 6)
        println("      ", first(strip(l), 260))
    end
end

println("\n════════ summary ════════")
for (name, ok) in RESULTS
    println(ok ? "PASS  " : "FAIL  ", name)
end
println("""

Please paste the whole output. The parameter lists of stages 1, 2a and 2b, the line saying whether 3a/3b used the
state-free variant, and the custom-call lines of stage 4 are what decide between tracing the KA source with Reactant
and launching a flattened binary.""")
