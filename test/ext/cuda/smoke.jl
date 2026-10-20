# A staged smoke test of the CUDA backend, for running on an NVIDIA machine and pasting the output back:
#
#     julia --project=test test/ext/cuda/smoke.jl
#
# Every stage runs in its own `try`, prints PASS or FAIL with the full error and backtrace, and later
# stages still run where they can. Nothing here is specific to this repository's test harness: it loads
# CUDA directly. The stages go from "is the extension registered" up to the real op (SDPAOps' KA kernel)
# and Enzyme, so the first FAIL says where the CUDA path breaks:
#
#   1 backend registered      4 ka_compile -> .cubin        7 views and offsets      10 SDPAOps (KA kernel)
#   2 array routing           5 load + thread limit         8 timing                 11 launch-size checks
#   3 KA context and prelude  6 run_binary vs host          9 Residual fwd/bwd + Enzyme  12 .cubin and .ptx binaries  13 target architecture
#   14 Triton ABI                 15 flat KA binary, traced   16 output-first KA kernels, traced
#
# `JULIA_DEBUG=KernelDispatchCUDAExt` additionally logs the entry point, image size and launch geometry.

using Test, Random, InteractiveUtils, KernelDispatch, KernelAbstractions, CUDA, SDPAOps, Enzyme
const GPU_PKG = CUDA
include(joinpath(@__DIR__, "..", "..", "fixtures.jl"))
include(joinpath(@__DIR__, "..", "..", "sdpa_helpers.jl"))
include(joinpath(@__DIR__, "ptx.jl"))

const KD = KernelDispatch
const EXT = Base.get_extension(KernelDispatch, :KernelDispatchCUDAExt)

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
println("CUDA.jl ", pkgversion(CUDA), ", GPUCompiler ", pkgversion(CUDA.CUDACore.GPUCompiler),
    ", KernelDispatch ", pkgversion(KernelDispatch))
if CUDA.functional()
    dev = CUDA.device()
    println("device: ", CUDA.name(dev), ", compute capability ", CUDA.capability(dev),
        ", driver ", CUDA.driver_version(), ", runtime ", CUDA.runtime_version())
else
    println("CUDA.functional() is false: no usable NVIDIA device/driver found; stopping.")
    exit(1)
end

stage("1 backend registered") do
    check(EXT !== nothing, "the KernelDispatchCUDAExt extension did not load")
    check(KD.device_available(), "device_available() is false although CUDA is functional")
    check(KD.device_backend() isa KD.CUDABackendTag, "device_backend() is $(KD.device_backend())")
end

const TAG = KD.CUDABackendTag()
rng = Xoshiro(1)

stage("2 array routing") do
    x = CuArray(rand(rng, Float32, 4, 6))
    check(KD.backend_of(x) === TAG)
    check(KD.device_dense(TAG, x) === x)
    contiguous = view(x, :, 2:3)                               # CUDA.jl returns a CuArray here
    check(KD.backend_of(contiguous) === TAG)
    for w in (view(x, 2:3, :), x', transpose(x))              # these stay wrappers
        check(w isa EXT.WrappedCuArray, "$(typeof(w)) is not a WrappedCuArray")
        d = KD.device_dense(TAG, w)
        check(d isa CuArray && Array(d) == Array(w), "device_dense of $(typeof(w)) is wrong")
    end
end

stage("3 KA context and prelude") do
    words = KD.ka_state_words(TAG)
    println("\n   state words: ", map(typeof, words))
    check(words === (), "a flat KA binary takes no kernel state, got $words")
    ctx, ghost = KD.ka_context(TAG, (2, 1, 1), 64)
    println("   ctx: ", sizeof(ctx), " bytes, ghost=", ghost)
    prelude = KD.ka_prelude(TAG, ctx, ghost)
    println("   prelude: ", prelude)
    check(prelude == Any[128, 1, 1, 2, 1, 1], "the flat prelude is the ND-range and the block counts, got $prelude")
    check(KD.ka_grid(TAG, (2, 3, 4)) === (24, 1, 1))
    check(KD.ka_backend_object(TAG) isa CUDABackend)
end

LS = nothing
stage("4 ka_compile -> .cubin") do
    global LS = register_logit_scale!()[:main]
    println("\n   ", LS.file, " (", filesize(LS.file), " bytes), entry ", LS.entry)
    check(isfile(LS.file) && endswith(LS.file, ".cubin"))
    check(read(LS.file, 4) == UInt8[0x7f, 0x45, 0x4c, 0x46], "not an ELF image (expected a cubin)")
    check(LS.is_ka && LS.tile == (LS_TG, 1, 1))
    check(LS.params.ka_layout == "flat", "ka_compile did not record the flat layout: $(LS.params)")
end

stage("5 load + thread limit") do
    lim = KD.max_threads(LS)
    println("\n   max threads for the function: ", lim)
    check(LS_TG <= lim <= 1024)
    check(lim == KD.max_threads(TAG, LS) == KD.max_threads(LS; backend=TAG))
end

stage("6 run_binary vs host") do
    for n in (1, 64, 100, 1000)
        x = randn(rng, Float32, n)
        y = logit_scale(CuArray(x), 2.0f0; out_len=n + 28)
        check(y isa CuArray && length(y) == n + 28)
        check(Array(y)[1:n] ≈ 2.0f0 .* x, "wrong values for n=$n")
        check(all(iszero, Array(y)[n+1:end]), "the zero-filled tail was written for n=$n")
    end
end

stage("7 views and offsets") do
    xm = CuArray(randn(rng, Float32, 8, 10))
    xv = view(xm, :, 2:2:10)                                   # strided: densified once
    check(Array(logit_scale(xv, 3.0f0))[1:length(xv)] ≈ 3.0f0 .* vec(Array(xv)), "strided view")
    xs = CuArray(randn(rng, Float32, 100))
    xo = view(xs, 11:60)                                       # contiguous view: a pointer offset
    check(Array(logit_scale(xo, 3.0f0))[1:50] ≈ 3.0f0 .* Array(xs)[11:60], "offset view")
end

stage("8 timing") do
    x = CuArray(randn(rng, Float32, 2^20))
    args = (OutArray(Float32, length(x)), x, 2.0f0, Int32(length(x)), Val(LS_TG))
    t = time_binary(LS, args...; extent=length(x))
    println("\n   time_binary: ", t, " s")
    check(t isa Float64 && t > 0)
    l = prepare_launch(LS, args...; extent=length(x))
    check(time_binary([l, l]; reps=2) > 0)
end

stage("9 Residual forward/backward binaries and the Enzyme rule") do
    register_residual!()
    n = 50
    x, y, dz = randn(rng, Float32, n), randn(rng, Float32, n), randn(rng, Float32, n)
    xd, yd = CuArray(x), CuArray(y)
    z = Residual()(xd, yd, 3.0f0)
    check(z isa CuArray && Array(z) ≈ 3 .* x .+ y, "forward")
    dx, dy, da = KD.backward(Residual(), ResidualKA(), (xd, yd, 3.0f0), z, CuArray(dz))
    check(Array(dx) ≈ 3 .* dz && Array(dy) ≈ dz && da === nothing, "backward binary")
    f(x, y, a) = Residual()(x, y, a)
    gx, gy = zero(xd), zero(yd)
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f)}, Duplicated,
        Duplicated{typeof(xd)}, Duplicated{typeof(yd)}, Const{Float32})
    args = (Const(f), Duplicated(xd, gx), Duplicated(yd, gy), Const(3.0f0))
    tape, primal, shadow = fwd(args...)
    shadow .= CuArray(dz)
    rev(args..., tape)
    check(Array(primal) ≈ 3 .* x .+ y && Array(gx) ≈ 3 .* dz && Array(gy) ≈ dz, "Enzyme gradients")
end

stage("10 SDPAOps, KernelAbstractions kernel") do
    for (d, n, m, dv) in SDPA_SIZES
        q, k, v = inputs(rng, d, n, m, dv)
        o = KD.forward(SDPA(), SDPAKA(), CuArray(q), CuArray(k), CuArray(v))
        check(o isa CuArray && size(o) == (dv, n))
        check(isapprox(Array(o), sdpa_reference(q, k, v); rtol=1e-3), "wrong values for (d,n,m,dv)=$((d, n, m, dv))")
    end
end

stage("11 launch-size checks") do
    big = KernelBinary(LS; threadgroup=4096)
    x = CuArray(rand(rng, Float32, 64))
    a() = (OutArray(Float32, 64), x, 2.0f0, Int32(64), Val(LS_TG))
    check(try; time_binary(big, a()...; extent=64); false; catch e; e isa ArgumentError; end,
        "a threadgroup over the function's limit was not refused with an ArgumentError")
    check(try; prepare_launch(LS, a()...); false; catch e; e isa ArgumentError; end, "no extent")
end

stage("12 explicit binaries: .cubin and .ptx, eager") do
    bins = scale_binaries(mktempdir())
    for (kind, bin) in pairs(bins), n in (1, 100, 1000)
        x = randn(rng, Float32, n)
        (y,) = run_binary(bin, CuArray(x), OutArray(Float32, n), 3.0f0, Int32(n); extent=n)
        check(Array(y) ≈ 3.0f0 .* x, "$kind: wrong values for n=$n")
    end
    junk = joinpath(mktempdir(), "scale.metallib")
    write(junk, UInt8[1, 2, 3])
    bad = KernelBinary(junk; entry="scale", threadgroup=64)
    check(try; run_binary(bad, CuArray(rand(Float32, 4)), OutArray(Float32, 4), 1f0, Int32(4); extent=4); false
          catch e; e isa ArgumentError && occursin(".cubin", e.msg); end, "a .metallib was not refused with an ArgumentError")
end

stage("13 target architecture") do
    dev = KD.target_id(TAG)
    println("\n   device target: ", dev, "; ka_compile recorded: ", LS.target)
    check(occursin(r"^sm_\d+$", dev) && LS.target == dev, "ka_compile did not record the device's target")
    bins = scale_binaries(mktempdir())
    x = randn(rng, Float32, 16)
    run(b) = Array(only(run_binary(b, CuArray(x), OutArray(Float32, 16), 2.0f0, Int32(16); extent=16)))
    check(run(bins.cubin) ≈ 2.0f0 .* x, "the device's own cubin")
    wrong = KernelBinary(bins.cubin; target="sm_20")
    check(try; run(wrong); false; catch e; e isa ArgumentError && occursin("built for sm_20", e.msg); end,
        "a cubin for another architecture was not refused with an ArgumentError")
    check(run(KernelBinary(bins.ptx; target="sm_20")) ≈ 2.0f0 .* x, "portable PTX")
end

stage("14 Triton ABI: shared memory and trailing pointers") do
    # The scale kernel dressed as a Triton one: it declares no scratch parameters, so the two trailing null
    # pointers and the dynamic shared memory only show that a launch with them is accepted and still correct.
    # (A real Triton kernel needs Python: its output is test/fixtures/triton/<version>/float32/, run by test/ext/cuda/triton.test.jl.)
    bins = scale_binaries(mktempdir())
    x = randn(rng, Float32, 100)
    for shared in (0, 1024, 60 * 1024), (kind, bin) in pairs(bins)      # 60 KB is above the 48 KB default
        tb = KernelBinary(bin; abi=:triton, shared_mem=shared, params=(; scratch_pointers=2))
        (y,) = run_binary(tb, CuArray(x), OutArray(Float32, 100), 2.0f0, Int32(100); extent=100)   # tile = 64: 2 blocks
        check(Array(y) ≈ 2.0f0 .* x, "$kind, shared_mem=$shared: wrong values")
    end
end

stage("15 flat KA binary, traced through XLA's kernel-call handler") do
    # `scale_ins_first!` (x, y, ...): a KA kernel with its input before its output, built flat by `ka_compile`.
    ka = compile_scale_ins_first()
    println("\n   ", ka.file, "; params ", ka.params)
    check(ka.abi === :ka && ka.params.ka_layout == "flat")
    n = 100
    x = randn(rng, Float32, n)
    (y,) = run_binary(ka, CuArray(x), OutArray(Float32, n), 2.0f0, Int32(n), Val(ka.threadgroup); extent=n)
    check(Array(y) ≈ 2.0f0 .* x, "eager launch of the flat binary is wrong")
    code = quote
        using KernelDispatch, CUDA, Reactant
        Reactant.set_default_backend("gpu")
        file, entry, tg = $(ka.file), $(ka.entry), $(ka.threadgroup)
        loose() = KernelBinary(file; entry, threadgroup=tg, abi=:ka, params=(; ka_layout="flat", tg), signature=$(ka.signature))
        n = 100
        x = randn(Float32, n)
        f(x) = only(run_binary(loose(), x, OutArray(Float32, n), 2.0f0, Int32(n), Val(tg); extent=n))
        off = Reactant.CompileOptions(; raise_triton_custom_call=false)
        y = Array(Reactant.@jit compile_options = off f(Reactant.to_rarray(x)))
        println(y ≈ 2.0f0 .* x ? "RESULT ok" : "RESULT wrong values (max error $(maximum(abs, y .- 2.0f0 .* x)))")
    end
    src = join(string.(filter(a -> !(a isa LineNumberNode), code.args)), "\n")
    out = IOBuffer()
    proc = run(pipeline(ignorestatus(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $src`); stdout=out, stderr=out))
    text = String(take!(out))
    ok = success(proc) && occursin("RESULT ok", text)
    ok || foreach(l -> println("      ", first(l, 220)),
        first(filter(l -> occursin(r"error|Error|ERROR|No registered|custom call|FATAL|RESULT", l), split(text, "\n")), 8))
    check(ok, "the traced launch of a flat KA binary did not produce the host result")
end

stage("16 output-first KA kernels, traced (aliased outputs)") do
    # The fixtures' kernels put their output first (`f!(out, in...)`); one has two outputs. XLA's handler gets each
    # output as an aliased operand in the kernel's position. In a child process: an XLA fatal error costs one stage.
    code = quote
        using Random, KernelDispatch, KernelAbstractions, CUDA, Reactant
        include($(joinpath(@__DIR__, "..", "..", "fixtures.jl")))
        Reactant.set_default_backend("gpu")
        rng = Xoshiro(7)
        register_logit_scale!(); register_residual!()
        x = randn(rng, Float32, 100)
        y = Array(Reactant.@jit logit_scale(Reactant.to_rarray(x), 2.0f0; out_len=128))
        ok1 = y[1:100] ≈ 2.0f0 .* x && all(iszero, y[101:128])
        n = 64
        a, b, dz = randn(rng, Float32, n), randn(rng, Float32, n), randn(rng, Float32, n)
        fwd(a, b) = KernelDispatch.forward(Residual(), ResidualKA(), a, b, 3.0f0)
        ok2 = Array(Reactant.@jit fwd(Reactant.to_rarray(a), Reactant.to_rarray(b))) ≈ 3.0f0 .* a .+ b
        g(a, b, dz) = KernelDispatch.backward(Residual(), ResidualKA(), (a, b, 3.0f0), a, dz)[1:2]
        gx, gy = Reactant.@jit g(Reactant.to_rarray(a), Reactant.to_rarray(b), Reactant.to_rarray(dz))
        ok3 = Array(gx) ≈ 3.0f0 .* dz && Array(gy) ≈ dz
        println("one output first: ", ok1, "; forward: ", ok2, "; two outputs (backward): ", ok3)
        println(ok1 && ok2 && ok3 ? "RESULT ok" : "RESULT wrong values")
    end
    src = join(string.(filter(a -> !(a isa LineNumberNode), code.args)), "\n")
    out = IOBuffer()
    proc = run(pipeline(ignorestatus(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $src`); stdout=out, stderr=out))
    text = String(take!(out))
    println("\n", join(filter(l -> occursin("one output first", l), split(text, "\n")), "\n"))
    ok = success(proc) && occursin("RESULT ok", text)
    ok || foreach(l -> println("      ", first(l, 220)),
        first(filter(l -> occursin(r"error|Error|ERROR|No registered|custom call|FATAL|RESULT", l), split(text, "\n")), 8))
    check(ok, "a traced output-first KA launch did not produce the host result")
end

println("\n════════ summary ════════")
for (name, ok) in RESULTS
    println(ok ? "PASS  " : "FAIL  ", name)
end
all(last, RESULTS) || exit(1)
