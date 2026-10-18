# A staged smoke test of the CUDA backend, for running on an NVIDIA machine and pasting the output back:
#
#     julia --project=test test/cuda_smoke.jl
#
# Every stage runs in its own `try`, prints PASS or FAIL with the full error and backtrace, and later
# stages still run where they can. Nothing here is specific to this repository's test harness: it loads
# CUDA directly. The stages go from "is the extension registered" up to the real op (SDPAOps' KA kernel)
# and Enzyme, so the first FAIL says where the CUDA path breaks:
#
#   1 backend registered      4 ka_compile -> .cubin        7 views and offsets      10 SDPAOps (KA kernel)
#   2 array routing           5 load + thread limit         8 timing                 11 launch-size checks
#   3 KA context and prelude  6 run_binary vs host          9 Residual fwd/bwd + Enzyme  12 .cubin and .ptx binaries  13 target architecture
#
# `JULIA_DEBUG=KernelDispatchCUDAExt` additionally logs the entry point, image size and launch geometry.

using Test, Random, InteractiveUtils, KernelDispatch, KernelAbstractions, CUDA, SDPAOps, Enzyme
const GPU_PKG = CUDA
include(joinpath(@__DIR__, "fixtures.jl"))
include(joinpath(@__DIR__, "sdpa_helpers.jl"))
include(joinpath(@__DIR__, "cuda_ptx.jl"))

const KO = KernelDispatch
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
    check(KO.device_available(), "device_available() is false although CUDA is functional")
    check(KO.device_backend() isa KO.CUDABackendTag, "device_backend() is $(KO.device_backend())")
end

const TAG = KO.CUDABackendTag()
rng = Xoshiro(1)

stage("2 array routing") do
    x = CuArray(rand(rng, Float32, 4, 6))
    check(KO.backend_of(x) === TAG)
    check(KO.device_dense(TAG, x) === x)
    contiguous = view(x, :, 2:3)                               # CUDA.jl returns a CuArray here
    check(KO.backend_of(contiguous) === TAG)
    for w in (view(x, 2:3, :), x', transpose(x))              # these stay wrappers
        check(w isa EXT.WrappedCuArray, "$(typeof(w)) is not a WrappedCuArray")
        d = KO.device_dense(TAG, w)
        check(d isa CuArray && Array(d) == Array(w), "device_dense of $(typeof(w)) is wrong")
    end
end

stage("3 KA context and prelude") do
    words = KO.ka_state_words(TAG)
    println("\n   state words: ", map(typeof, words))
    ctx, ghost = KO.ka_context(TAG, (2, 1, 1), 64)
    println("   ctx: ", sizeof(ctx), " bytes, ghost=", ghost)
    prelude = KO.ka_prelude(TAG, ctx, ghost)
    check(length(prelude) == length(words) + !ghost, "prelude has $(length(prelude)) entries")
    check(KO.ka_grid(TAG, (2, 3, 4)) === (24, 1, 1))
    check(KO.ka_backend_object(TAG) isa CUDABackend)
end

LS = nothing
stage("4 ka_compile -> .cubin") do
    global LS = register_logit_scale!()[:main]
    println("\n   ", LS.file, " (", filesize(LS.file), " bytes), entry ", LS.entry)
    check(isfile(LS.file) && endswith(LS.file, ".cubin"))
    check(read(LS.file, 4) == UInt8[0x7f, 0x45, 0x4c, 0x46], "not an ELF image (expected a cubin)")
    check(LS.is_ka && LS.tile == (LS_TG, 1, 1))
end

stage("5 load + thread limit") do
    lim = KO.max_threads(LS)
    println("\n   max threads for the function: ", lim)
    check(LS_TG <= lim <= 1024)
    check(lim == KO.max_threads(TAG, LS) == KO.max_threads(LS; backend=TAG))
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
    dx, dy, da = KO.backward(Residual(), ResidualKA(), (xd, yd, 3.0f0), z, CuArray(dz))
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
        o = KO.forward(SDPA(), SDPAKA(), CuArray(q), CuArray(k), CuArray(v))
        check(o isa CuArray && size(o) == (dv, n))
        check(isapprox(Array(o), sdpa_reference(q, k, v); rtol=1e-3), "wrong values for (d,n,m,dv)=$((d, n, m, dv))")
    end
end

stage("11 launch-size checks") do
    big = KernelBinary(LS.file; entry=LS.entry, threadgroup=4096, is_ka=true)
    x = CuArray(rand(rng, Float32, 64))
    a() = (OutArray(Float32, 64), x, 2.0f0, Int32(64), Val(LS_TG))
    check(try; time_binary(big, a()...; extent=64); false; catch e; e isa ArgumentError; end,
        "a threadgroup over the function's limit was not refused with an ArgumentError")
    check(try; prepare_launch(LS, a()...); false; catch e; e isa ArgumentError; end, "no extent")
end

stage("12 plain binaries: .cubin and .ptx, eager") do
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
    dev = KO.target_id(TAG)
    println("\n   device target: ", dev, "; ka_compile recorded: ", LS.target)
    check(occursin(r"^sm_\d+$", dev) && LS.target == dev, "ka_compile did not record the device's target")
    bins = scale_binaries(mktempdir())
    x = randn(rng, Float32, 16)
    run(b) = Array(only(run_binary(b, CuArray(x), OutArray(Float32, 16), 2.0f0, Int32(16); extent=16)))
    check(run(bins.cubin) ≈ 2.0f0 .* x, "the device's own cubin")
    wrong = KernelBinary(bins.cubin.file; entry="scale", threadgroup=64, target="sm_20")
    check(try; run(wrong); false; catch e; e isa ArgumentError && occursin("built for sm_20", e.msg); end,
        "a cubin for another architecture was not refused with an ArgumentError")
    check(run(KernelBinary(bins.ptx.file; entry="scale", threadgroup=64, target="sm_20")) ≈ 2.0f0 .* x, "portable PTX")
end

stage("14 Triton ABI: shared memory and trailing pointers") do
    # The scale kernel dressed as a Triton one: it declares no scratch parameters, so the two trailing null
    # pointers and the dynamic shared memory only show that a launch with them is accepted and still correct.
    # (A real Triton kernel needs Python: see docs/triton_kernels.md.)
    bins = scale_binaries(mktempdir())
    x = randn(rng, Float32, 100)
    for shared in (0, 1024, 60 * 1024), (kind, bin) in pairs(bins)      # 60 KB is above the 48 KB default
        tb = KernelBinary(bin.file; entry="scale", threadgroup=64, abi=:triton, target=bin.target,
            shared_mem=shared, params=(; scratch_pointers=2))
        (y,) = run_binary(tb, CuArray(x), OutArray(Float32, 100), 2.0f0, Int32(100); extent=2)    # tile = threadgroup
        check(Array(y) ≈ 2.0f0 .* x, "$kind, shared_mem=$shared: wrong values")
    end
end

println("\n════════ summary ════════")
for (name, ok) in RESULTS
    println(ok ? "PASS  " : "FAIL  ", name)
end
all(last, RESULTS) || exit(1)
