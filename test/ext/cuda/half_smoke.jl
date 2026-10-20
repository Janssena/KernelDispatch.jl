# A staged smoke test of Float16 and BFloat16 on the CUDA backend, for running on an NVIDIA machine and pasting the
# output back. The CUDA half of docs/half-precision.md was written without hardware; this is what checks it.
#
#     julia --project=test test/ext/cuda/half_smoke.jl
#
# Same shape as smoke.jl: every stage runs in its own `try` and prints PASS or FAIL with the error, later stages still
# run, and the traced ones run in a child process (an XLA fatal error then costs one stage, not the script). BFloat16
# needs an sm_80 or newer GPU for its native conversions: on an older one its stages are SKIPPED, loudly, not passed.
#
#   1 device                          5 explicit PTX, eager         9  SDPAOps (KA kernel), eager
#   2 ka_compile names and builds     6 explicit PTX, traced        10 SDPAOps, traced
#   3 a stale same-name build         7 flat KA binary, traced      11 Residual forward/backward + Enzyme
#   4 mixed outputs (T + Float32)     8 a 2-byte scalar, traced     10b mixed-type outputs, traced     10c = 10 with the builds done first
#                                     12 mismatched types: refused, and wrong when unchecked
#                                     (the scalar stays Float32 everywhere else)
#
# `HALF_SMOKE_ONLY=10,10b julia ... half_smoke.jl` runs just those stages.
#
# What to report back: the output from "summary" down, and for any FAIL the lines above it.

using Test, Random, InteractiveUtils, KernelDispatch, KernelAbstractions, CUDA, SDPAOps, Enzyme, BFloat16s
const GPU_PKG = CUDA
include(joinpath(@__DIR__, "..", "..", "fixtures.jl"))
include(joinpath(@__DIR__, "..", "..", "sdpa_helpers.jl"))
include(joinpath(@__DIR__, "..", "..", "precision.jl"))

const KD = KernelDispatch
const RESULTS = Pair{String,Symbol}[]       # :pass, :fail or :skip

check(cond, msg="check failed") = cond || error(msg)

# `HALF_SMOKE_ONLY=10` (or `6,7`) runs just those stages.
const ONLY = filter(!isempty, split(get(ENV, "HALF_SMOKE_ONLY", ""), ','))

function stage(f, name)
    isempty(ONLY) || first(split(name)) in ONLY || return
    print("\n── ", name, " ")
    try
        r = f()
        if r === :skip
            println("SKIP")
            push!(RESULTS, name => :skip)
        else
            println("PASS")
            push!(RESULTS, name => :pass)
        end
    catch err
        println("FAIL")
        showerror(stdout, err, catch_backtrace())
        println()
        push!(RESULTS, name => :fail)
    end
end

# A traced stage runs `code` in a child process; it must print "RESULT ok" to pass.
function child(code::Expr, what)
    src = join(string.(filter(a -> !(a isa LineNumberNode), code.args)), "\n")
    out = IOBuffer()
    proc = run(pipeline(ignorestatus(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $src`); stdout=out, stderr=out))
    text = String(take!(out))
    println("\n", join(filter(l -> startswith(l, "  "), split(text, "\n")), "\n"))
    ok = success(proc) && occursin("RESULT ok", text)
    if !ok
        # Say how the child ended and show the tail of everything it wrote: a crash inside XLA prints no "error" line.
        println("      child exit code ", proc.exitcode, ", signal ", proc.termsignal)
        log = joinpath(tempdir(), "half_smoke_child.log")
        write(log, text)
        println("      full child output: ", log)
        foreach(l -> println("      ", first(l, 220)), last(split(text, "\n"), 40))
    end
    check(ok, what)
end

println("Julia ", VERSION, " on ", Sys.MACHINE)
CUDA.functional() || (println("CUDA.functional() is false: no usable NVIDIA device/driver found; stopping."); exit(1))
const CAP = CUDA.capability(CUDA.device())
const NATIVE_BF16 = CAP >= v"8.0"
const HALF_TYPES = NATIVE_BF16 ? (Float16, BFloat16) : (Float16,)
rng = Xoshiro(1)

stage("1 device") do
    println("\n   ", CUDA.name(CUDA.device()), ", compute capability ", CAP, ", driver ", CUDA.driver_version(),
        ", runtime ", CUDA.runtime_version())
    NATIVE_BF16 || println("   BFloat16 stages are skipped: they need sm_80 or newer")
    check(KD.device_backend() isa KD.CUDABackendTag)
    check(Float16 in HALF_TYPES)
end

# ── hand-written PTX: y[i] = a * x[i] with 16-bit buffers and an fp32 scalar (the Triton-style convention) ────────────
# f16: ld.b16 + cvt.f32.f16 / cvt.rn.f16.f32 (sm_53+). bf16: widening is a shift (any arch); narrowing is
# cvt.rn.bf16.f32 (sm_80+, PTX 7.0).
function half_ptx(T)
    name = T === Float16 ? "scale_f16" : "scale_bf16"
    target, version = T === Float16 ? ("sm_75", "6.3") : ("sm_80", "7.0")
    widen = T === Float16 ?
        "    cvt.f32.f16 %f2, %h1;" :
        "    cvt.u32.u16 %r6, %h1;\n    shl.b32 %r6, %r6, 16;\n    mov.b32 %f2, %r6;"
    narrow = T === Float16 ? "    cvt.rn.f16.f32 %h2, %f3;" : "    cvt.rn.bf16.f32 %h2, %f3;"
    return name, """
.version $version
.target $target
.address_size 64

.visible .entry $name(
    .param .u64 x,
    .param .u64 y,
    .param .f32 a,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b16 %h<3>;
    .reg .b32 %r<8>;
    .reg .f32 %f<4>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [x];
    ld.param.u64 %rd2, [y];
    ld.param.f32 %f1, [a];
    ld.param.u32 %r1, [n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;
    cvta.to.global.u64 %rd3, %rd1;
    cvta.to.global.u64 %rd4, %rd2;
    mul.wide.u32 %rd5, %r5, 2;
    add.s64 %rd6, %rd3, %rd5;
    add.s64 %rd7, %rd4, %rd5;
    ld.global.b16 %h1, [%rd6];
$widen
    mul.f32 %f3, %f2, %f1;
$narrow
    st.global.b16 [%rd7], %h2;
DONE:
    ret;
}
"""
end

# -> (ptx, cubin) `:explicit` binaries of the kernel for `T`, written into `dir`
function half_binaries(dir, T)
    name, src = half_ptx(T)
    ptx, cubin = joinpath(dir, "$name.ptx"), joinpath(dir, "$name.cubin")
    write(ptx, src)
    arch = "sm_$(CAP.major)$(CAP.minor)"
    run(`$(CUDA.CUDACore.CUDA_Compiler.ptxas()) --gpu-name $arch --output-file $cubin $ptx`)
    sig = (Vector{T}, Vector{T}, Float32, Int32)
    return (ptx=KernelBinary(ptx; entry=name, threadgroup=64, signature=sig),
            cubin=KernelBinary(cubin; entry=name, threadgroup=64, target=arch, signature=sig))
end

stage("2 ka_compile: element types in the names, builds, eager launches") do
    dir = mktempdir()
    for T in (Float32, HALF_TYPES...)
        argtypes = (Vector{T}, Vector{T}, Float32, Int32, Val{64})
        b = ka_compile(KD.device_backend(), logit_scale_kernel!, argtypes; tg=64, name="half_scale", dir)
        println("\n   $T: ", basename(b.file))
        check(occursin(KD.ka_signature_tag(argtypes), b.file), "the signature is not in the name")
        check(read(b.file, 4) == UInt8[0x7f, 0x45, 0x4c, 0x46], "not an ELF image (expected a cubin)")
        n = 100
        x = rounded(T, randn(rng, Float32, n))
        y = only(run_binary(b, OutArray(T, n), CuArray(x), 2.0f0, Int32(n), Val(64); extent=n))
        check(eltype(y) === T, "output is $(eltype(y)), expected $T")
        err = maxrelerr(Array(y), 2.0 .* Float64.(x))
        println("   max relative error ", err)
        check(err <= ELEMENTWISE_RTOL[T], "error $err above $(ELEMENTWISE_RTOL[T])")
    end
end

stage("3 a same-name build at another type is its own file") do
    dir = mktempdir()
    files = String[]
    for T in (Float32, HALF_TYPES...)
        b = ka_compile(KD.device_backend(), logit_scale_kernel!, (Vector{T}, Vector{T}, Float32, Int32, Val{64});
            tg=64, name="collide", dir)
        push!(files, b.file)
    end
    check(length(unique(files)) == length(files), "two element types share a file: $files")
end

# One output in the buffers' type and one in Float32, as trifast's `o` and `lse` are.
@kernel unsafe_indices = true function mixed!(o, l, x, a::Float32, n::Int32, ::Val{TG}) where {TG}
    i = (@index(Group, Linear) - 1) * TG + @index(Local, Linear)
    if i <= n
        v = a * Float32(unsafe_load(x, i))
        unsafe_store!(o, v, i)
        unsafe_store!(l, v * v, i)
    end
end

stage("4 a mixed launch: a T output next to a Float32 one") do
    dir = mktempdir()
    for T in HALF_TYPES
        b = ka_compile(KD.device_backend(), mixed!, (Vector{T}, Vector{Float32}, Vector{T}, Float32, Int32, Val{64});
            tg=64, name="mixed", dir)
        n = 64
        x = rounded(T, randn(rng, Float32, n))
        o, l = run_binary(b, OutArray(T, n), OutArray(Float32, n), CuArray(x), 2.0f0, Int32(n), Val(64); extent=n)
        check(eltype(o) === T && eltype(l) === Float32, "got $(eltype(o)) and $(eltype(l))")
        check(maxrelerr(Array(o), 2.0 .* Float64.(x)) <= ELEMENTWISE_RTOL[T], "wrong $T output")
        check(maxrelerr(Array(l), (2.0 .* Float64.(x)) .^ 2) <= 1e-5, "wrong Float32 output")
    end
end

stage("5 explicit PTX and cubin at 16 bits, eager") do
    for T in HALF_TYPES, (label, b) in pairs(half_binaries(mktempdir(), T))
        n = 100
        x = rounded(T, randn(rng, Float32, n))
        y = only(run_binary(b, CuArray(x), OutArray(T, n), 3.0f0, Int32(n); extent=n))
        check(eltype(y) === T)
        err = maxrelerr(Array(y), 3.0 .* Float64.(x))
        println("\n   $T $label: max relative error ", err)
        check(err <= ELEMENTWISE_RTOL[T], "error above $(ELEMENTWISE_RTOL[T])")
    end
end

stage("6 explicit PTX at 16 bits, traced") do
    dir = mktempdir()
    files = [(T, half_binaries(dir, T).ptx) for T in HALF_TYPES]
    spec = [(string(nameof(T)), b.file, b.entry, b.signature) for (T, b) in files]
    child(quote
        using KernelDispatch, CUDA, Reactant, BFloat16s
        Reactant.set_default_backend("gpu")
        off = Reactant.CompileOptions(; raise_triton_custom_call=false)
        oks = Bool[]
        for (tname, file, entry, sig) in $spec
            T = tname == "Float16" ? Float16 : BFloat16
            x = T.(randn(Float32, 100))
            f(x) = only(run_binary(KernelBinary(file; entry, threadgroup=64, signature=sig), x, OutArray(eltype(x), 100), 2.0f0,
                Int32(100); extent=100))
            y = Array(Reactant.@jit compile_options = off f(Reactant.to_rarray(x)))
            good = eltype(y) === T && y == T.(2.0f0 .* Float32.(x))
            println("  ", T, ": ", good ? "ok" : "WRONG (eltype $(eltype(y)))")
            push!(oks, good)
        end
        println(all(oks) ? "RESULT ok" : "RESULT wrong values")
    end, "a traced 16-bit PTX launch did not produce the host result")
end

stage("7 flat KA binary at 16 bits, traced (incl. mixed outputs)") do
    child(quote
        using Random, KernelDispatch, KernelAbstractions, CUDA, Reactant, BFloat16s
        include($(joinpath(@__DIR__, "..", "..", "fixtures.jl")))
        Reactant.set_default_backend("gpu")
        off = Reactant.CompileOptions(; raise_triton_custom_call=false)
        be = KernelDispatch.device_backend()
        dir = mktempdir()
        oks = Bool[]
        for T in $(collect(HALF_TYPES))
            n = 64
            argt = (Vector{T}, Vector{T}, Vector{T}, Float32, Int32, Val{64})
            fwd = ka_compile(be, residual_fwd!, argt; tg=64, name="res_fwd", params=(; tg=64), dir)
            bwd = ka_compile(be, residual_bwd!, argt; tg=64, name="res_bwd", params=(; tg=64), dir)
            for k in (ResidualKA(), ResidualPlain())
                register_kernel!(Residual(), k); add_variant!(Residual(), k, (nameof(T),); fwd, bwd)
            end
            x, y, dz = (T.(randn(Float32, n)) for _ in 1:3)
            f(x, y) = KernelDispatch.forward(Residual(), ResidualKA(), x, y, 3.0f0)
            z = Array(Reactant.@jit compile_options = off f(Reactant.to_rarray(x), Reactant.to_rarray(y)))
            ok1 = eltype(z) === T && maximum(abs.(Float32.(z) .- (3 .* Float32.(x) .+ Float32.(y)))) <= 0.05
            g(x, y, dz) = KernelDispatch.backward(Residual(), ResidualKA(), (x, y, 3.0f0), x, dz)[1:2]
            gx, gy = Reactant.@jit compile_options = off g(Reactant.to_rarray(x), Reactant.to_rarray(y), Reactant.to_rarray(dz))
            ok2 = eltype(gx) === T && eltype(gy) === T && maximum(abs.(Float32.(Array(gx)) .- 3 .* Float32.(dz))) <= 0.05
            println("  ", T, ": forward ", ok1, ", backward (two outputs) ", ok2)
            push!(oks, ok1 && ok2)
        end
        println(all(oks) ? "RESULT ok" : "RESULT wrong values")
    end, "a traced half-precision KA launch did not produce the host result")
end

stage("8 a 2-byte scalar is refused when traced, with its reason") do
    child(quote
        using KernelDispatch, KernelAbstractions, CUDA, Reactant
        include($(joinpath(@__DIR__, "..", "..", "fixtures.jl")))
        Reactant.set_default_backend("gpu")
        off = Reactant.CompileOptions(; raise_triton_custom_call=false)
        @kernel unsafe_indices = true function k16!(y, x, a::Float16, n::Int32, ::Val{TG}) where {TG}
            i = (@index(Group, Linear) - 1) * TG + @index(Local, Linear)
            i <= n && unsafe_store!(y, a * unsafe_load(x, i), i)
        end
        b = ka_compile(KernelDispatch.device_backend(), k16!, (Vector{Float16}, Vector{Float16}, Float16, Int32, Val{64});
            tg=64, name="k16", dir=mktempdir())
        x = CuArray(rand(Float16, 8))
        y = only(run_binary(b, OutArray(Float16, 8), x, Float16(2), Int32(8), Val(64); extent=8))
        println("  eager with a Float16 scalar: ", Array(y) == Float16(2) .* Array(x))
        file, entry = b.file, b.entry
        f(x) = only(run_binary(KernelBinary(file; entry, threadgroup=64, abi=:ka, params=(; ka_layout="flat"), signature=false),
            OutArray(Float16, 8), x, Float16(2), Int32(8), Val(64); extent=8))
        msg = try
            Reactant.@jit compile_options = off f(Reactant.to_rarray(rand(Float16, 8)))
            "no error"
        catch e
            sprint(showerror, e)
        end
        println("  traced: ", first(msg, 160))
        println(occursin("2-byte", msg) ? "RESULT ok" : "RESULT the refusal did not name the 2-byte scalar")
    end, "the traced launch with a 2-byte scalar was not refused with its reason")
end

stage("9 SDPAOps (KA kernel) at 16 bits, eager") do
    for T in HALF_TYPES, (d, n, m, dv) in SDPA_SIZES
        q, k, v = inputs(rng, d, n, m, dv, T)
        o = KD.forward(SDPA(), SDPAKA(), CuArray(q), CuArray(k), CuArray(v))
        check(eltype(o) === T)
        err = maxrelerr(Array(o), sdpa_reference(q, k, v))
        check(err <= REDUCTION_RTOL[T], "$T d=$d n=$n m=$m dv=$dv: error $err above $(REDUCTION_RTOL[T])")
    end
end

# `warm`: build the variants with an eager call first. Without it the trace itself asks for the missing variant, whose build
# (a GPU compile) must run outside the tracer: stage 10 checks that, 10c is the same with the builds done beforehand.
function sdpa_traced_child(warm::Bool)
    child(quote
        using Random, KernelDispatch, SDPAOps, CUDA, Reactant, BFloat16s
        const KD = KernelDispatch
        include($(joinpath(@__DIR__, "..", "..", "fixtures.jl")))
        include($(joinpath(@__DIR__, "..", "..", "sdpa_helpers.jl")))
        include($(joinpath(@__DIR__, "..", "..", "precision.jl")))
        Reactant.set_default_backend("gpu")
        off = Reactant.CompileOptions(; raise_triton_custom_call=false)
        rng = Xoshiro(3)
        oks = Bool[]
        for T in $(collect(HALF_TYPES))
            q, k, v = inputs(rng, 16, 100, 70, 8, T)
            if $warm
                KernelDispatch.forward(SDPA(), SDPAKA(), CuArray(q), CuArray(k), CuArray(v))
                println("  ", T, ": variant built eagerly")
            end
            f(q, k, v) = KernelDispatch.forward(SDPA(), SDPAKA(), q, k, v)
            println("  ", T, ": tracing")
            o = Array(Reactant.@jit compile_options = off f(map(Reactant.to_rarray, (q, k, v))...))
            err = maxrelerr(o, sdpa_reference(q, k, v))
            good = eltype(o) === T && err <= REDUCTION_RTOL[T]
            println("  ", T, ": error ", err, good ? " ok" : " WRONG")
            push!(oks, good)
        end
        println(all(oks) ? "RESULT ok" : "RESULT wrong values")
    end, "a traced half-precision SDPAOps launch did not produce the host result")
end

stage("10 SDPAOps at 16 bits, traced (the variants are built from inside the trace)") do
    sdpa_traced_child(false)
end

stage("10c SDPAOps at 16 bits, traced (the variants are built eagerly first)") do
    sdpa_traced_child(true)
end

# Isolates stage 10: a traced launch with two outputs of different types (T and Float32), the second used or dropped.
stage("10b mixed-type outputs, traced (a T output next to a Float32 one)") do
    child(quote
        using Random, KernelDispatch, KernelAbstractions, CUDA, Reactant, BFloat16s
        Reactant.set_default_backend("gpu")
        off = Reactant.CompileOptions(; raise_triton_custom_call=false)
        @kernel unsafe_indices = true function mixed!(o, l, x, a::Float32, n::Int32, ::Val{TG}) where {TG}
            i = (@index(Group, Linear) - 1) * TG + @index(Local, Linear)
            if i <= n
                v = a * Float32(unsafe_load(x, i))
                unsafe_store!(o, v, i)
                unsafe_store!(l, v * v, i)
            end
        end
        be = KernelDispatch.device_backend()
        oks = Bool[]
        for T in $(collect(HALF_TYPES))
            n = 64
            b = ka_compile(be, mixed!, (Vector{T}, Vector{Float32}, Vector{T}, Float32, Int32, Val{64}); tg=64, name="m10b", dir=mktempdir())
            file, entry, sig = b.file, b.entry, b.signature
            mk() = KernelBinary(file; entry, threadgroup=64, abi=:ka, params=(; ka_layout="flat"), signature=sig)
            both(x) = run_binary(mk(), OutArray(eltype(x), 64), OutArray(Float32, 64), x, 2.0f0, Int32(64), Val(64); extent=64)
            first_only(x) = first(both(x))
            x = T.(randn(Float32, n))
            println("  ", T, ": tracing with both outputs used")
            o, l = Reactant.@jit compile_options = off both(Reactant.to_rarray(x))
            good1 = eltype(o) === T && eltype(l) === Float32 && Float32.(Array(o)) == 2 .* Float32.(x)
            println("  ", T, ": both used ", good1)
            println("  ", T, ": tracing with the second output dropped")
            o2 = Reactant.@jit compile_options = off first_only(Reactant.to_rarray(x))
            good2 = eltype(o2) === T && Float32.(Array(o2)) == 2 .* Float32.(x)
            println("  ", T, ": second dropped ", good2)
            push!(oks, good1 && good2)
        end
        println(all(oks) ? "RESULT ok" : "RESULT wrong values")
    end, "a traced launch with outputs of two types failed")
end

stage("11 Residual forward and backward at 16 bits, eager, with the Enzyme rule") do
    be = KD.device_backend()
    dir = mktempdir()
    for T in HALF_TYPES
        argt = (Vector{T}, Vector{T}, Vector{T}, Float32, Int32, Val{64})
        fwd = ka_compile(be, residual_fwd!, argt; tg=64, name="res_fwd", params=(; tg=64), dir)
        bwd = ka_compile(be, residual_bwd!, argt; tg=64, name="res_bwd", params=(; tg=64), dir)
        for k in (ResidualKA(), ResidualPlain())
            register_kernel!(Residual(), k); add_variant!(Residual(), k, (nameof(T),); fwd, bwd)
        end
        n = 50
        x, y, dz = (rounded(T, randn(rng, Float32, n)) for _ in 1:3)
        xd, yd, dzd = CuArray(x), CuArray(y), CuArray(dz)
        z = Residual()(xd, yd, 3.0f0)
        check(eltype(z) === T)
        check(maxrelerr(Array(z), 3.0 .* Float64.(x) .+ Float64.(y)) <= ELEMENTWISE_RTOL[T], "forward")
        gx, gy = zero(xd), zero(yd)
        fwdt, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(KD.forward)}, Duplicated,
            Const{Residual}, Const{ResidualKA}, Duplicated{typeof(xd)}, Duplicated{typeof(yd)}, Const{Float32})
        args = (Const(KD.forward), Const(Residual()), Const(ResidualKA()), Duplicated(xd, gx), Duplicated(yd, gy), Const(3.0f0))
        tape, primal, shadow = fwdt(args...)
        shadow .= dzd
        rev(args..., tape)
        check(eltype(gx) === T && eltype(gy) === T, "gradient types $(eltype(gx)), $(eltype(gy))")
        check(maxrelerr(Array(gx), 3.0 .* Float64.(dz)) <= ELEMENTWISE_RTOL[T], "dx")
        check(maxrelerr(Array(gy), Float64.(dz)) <= ELEMENTWISE_RTOL[T], "dy")
    end
end

stage("12 a mismatched launch is refused (eager and traced), and runs wrong without the check") do
    dir = mktempdir()
    b32 = ka_compile(KD.device_backend(), logit_scale_kernel!, (Vector{Float32}, Vector{Float32}, Float32, Int32, Val{64});
        tg=64, name="mismatch", dir)
    n = 64
    x = rounded(Float16, randn(rng, Float32, 2n))                         # 2n halves: n floats' worth, so nothing is out of bounds
    y = CuArray(fill(Float16(7), 2n))
    e = try; run_binary(b32, InOutArray(y), CuArray(x), 2.0f0, Int32(n), Val(64); extent=n); nothing; catch err; err; end
    check(e isa ArgumentError && occursin("argument 1", e.msg), "the Float16 buffer was not refused: $e")
    check(Array(y) == fill(Float16(7), 2n), "a refused launch still wrote the output")
    run_binary(b32, InOutArray(y), CuArray(x), 2.0f0, Int32(n), Val(64); extent=n, check_types=false)
    check(Array(y)[1:n] != Float16.(2 .* Float32.(x[1:n])), "the unchecked mismatched launch gave the right answer?")
    spec = (b32.file, b32.entry, b32.signature)
    child(quote
        using KernelDispatch, CUDA, Reactant, BFloat16s
        Reactant.set_default_backend("gpu")
        off = Reactant.CompileOptions(; raise_triton_custom_call=false)
        file, entry, sig = $spec
        f(x) = only(run_binary(KernelBinary(file; entry, threadgroup=64, abi=:ka, params=(; ka_layout="flat"), signature=sig),
            OutArray(eltype(x), 64), x, 2.0f0, Int32(64), Val(64); extent=64))
        msg = try
            Reactant.@jit compile_options = off f(Reactant.to_rarray(Float16.(randn(Float32, 64))))
            "no error"
        catch e
            sprint(showerror, e)
        end
        println("  traced: ", first(msg, 160))
        println(occursin("compiled for", msg) ? "RESULT ok" : "RESULT the traced mismatch was not refused")
    end, "a traced mismatched launch was not refused")
end

NATIVE_BF16 || push!(RESULTS, "BFloat16 (everything above) — needs sm_80, this device is sm_$(CAP.major)$(CAP.minor)" => :skip)

println("\n════════ summary ════════")
for (name, r) in RESULTS
    println(r === :pass ? "PASS  " : r === :skip ? "SKIP  " : "FAIL  ", name)
end
any(==(:fail), last.(RESULTS)) && exit(1)
