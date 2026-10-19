# Eager launches on the loaded GPU backend, for what SDPAOps (gpu/sdpa.test.jl) cannot show: a backward
# that is itself a launched binary and the Enzyme rule over it (`Residual`), and the binary-level launch
# features: sizing, `ka_compile`, unregistered binaries, timing (`LogitScale`). The forward path of a real
# op is in gpu/sdpa.test.jl. Backend-neutral: the array type, the backend tag and the backend's own limits
# come from gpu_backend.jl.

using Test, Random, KernelDispatch, SDPAOps, Enzyme
isdefined(@__MODULE__, :Residual) || include(joinpath(@__DIR__, "..", "fixtures.jl"))


begin
    rng = Random.Xoshiro(1)
    register_residual!()

    @testset "Residual: forward and binary backward" begin
        for n in (1, 64, 100, 1000)
            x, y, dz = randn(rng, Float32, n), randn(rng, Float32, n), randn(rng, Float32, n)
            xd, yd = DeviceArray(x), DeviceArray(y)
            z = Residual()(xd, yd, 3.0f0)
            @test z isa DeviceArray
            @test Array(z) ≈ 3 .* x .+ y
            dx, dy, da = KD.backward(Residual(), ResidualKA(), (xd, yd, 3.0f0), z, DeviceArray(dz))
            @test Array(dx) ≈ 3 .* dz
            @test Array(dy) ≈ dz
            @test da === nothing
        end
    end

    @testset "record_bindings" begin
        # A registered op's launches: the binary's label, `cld(extent, tile)` groups, and for a KA binary
        # the backend's kernel-state word(s) first. (Which slots each SDPAOps kernel binds: examples/SDPAOps/test.)
        n = 100
        q, k, v = (DeviceArray(rand(Float32, r, c)) for (r, c) in ((16, n), (16, 70), (8, 70)))
        for kern in prebuilt_sdpa_kernels()         # a binary shipped for this backend, if any (Metal: FlashAttn)
            label, args, groups = only(KD.record_bindings(() -> KD.forward(SDPA(), kern, q, k, v)))
            @test label === :fwd
            @test groups == (cld(n, 64), 1, 1)
        end
        label, args, groups = only(KD.record_bindings(() -> KD.forward(SDPA(), SDPAKA(), q, k, v)))
        @test groups == (cld(n, 64), 1, 1)
        state = KD.ka_state_words(GPU_TAG)                  # the backend's hidden leading kernel argument(s)
        @test collect(args[1:length(state)]) == [:bytes => w for w in state]
        @test last(args[end]) === 0.25f0 && last(args[end-1]) === Int32(8)    # scale = 1/√16, then dv

        # The recorded `groups` are `cld.(extent, tile)`, before the backend's KA grid.
        bin = register_logit_scale!()[:main]
        x = DeviceArray(rand(Float32, 300))
        rec = KD.record_bindings(() -> run_binary(bin, OutArray(Float32, 300), x, 2.0f0, Int32(300),
            Val(LS_TG); extent=(300, 2, 3), label=:scale))
        @test only(rec).label === :scale
        @test only(rec).groups == (cld(300, LS_TG), 2, 3)
        @test only(rec).launch isa KD.Launch && only(rec).launch.binary === bin
    end

    @testset "run_binary and prepare_launch" begin
        bin = register_logit_scale!()[:main]
        x = randn(rng, Float32, 300)
        # An unregistered binary: no op, no kernel, no variant.
        loose = KernelBinary(bin.file; entry=bin.entry, threadgroup=bin.threadgroup, abi=:ka, params=bin.params)
        y = only(@inferred run_binary(loose, OutArray(Float32, 300), DeviceArray(x), 2.0f0, Int32(300),
            Val(LS_TG); extent=300))
        @test Array(y) ≈ 2.0f0 .* x
        l = prepare_launch(loose, OutArray(Float32, 300), DeviceArray(x), 2.0f0, Int32(300), Val(LS_TG);
            extent=300)
        @test l.groups == (cld(300, LS_TG), 1, 1) && l.grid == (cld(300, LS_TG), 1, 1)
        @test Array(only(KD.execute(l))) ≈ 2.0f0 .* x
        @test time_binary(l) isa Float64
        @test_throws ArgumentError prepare_launch(loose, DeviceArray(x))       # no extent
    end

    # Reverse through `f` on device arrays: the forward's result and the gradients for cotangent `dz`.
    function residual_grads(f, x, y, a, dz)
        xd, yd = DeviceArray(x), DeviceArray(y)
        gx, gy = zero(xd), zero(yd)
        fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f)}, Duplicated,
            Duplicated{typeof(xd)}, Duplicated{typeof(yd)}, Const{Float32})
        args = (Const(f), Duplicated(xd, gx), Duplicated(yd, gy), Const(a))
        tape, primal, shadow = fwd(args...)
        shadow .= DeviceArray(dz)
        rev(args..., tape)
        return Array(primal), Array(gx), Array(gy)
    end

    @testset "Enzyme rule" begin
        n = 50
        x, y, dz = randn(rng, Float32, n), randn(rng, Float32, n), randn(rng, Float32, n)
        # Through the op call, and through a direct `forward(op, kernel, ...)`: both reach the rule.
        for f in ((x, y, a) -> Residual()(x, y, a), (x, y, a) -> KD.forward(Residual(), ResidualKA(), x, y, a))
            primal, gx, gy = residual_grads(f, x, y, 3.0f0, dz)
            @test primal ≈ 3 .* x .+ y
            @test gx ≈ 3 .* dz
            @test gy ≈ dz
        end
    end

    @testset "ka_compile + call_binary, eager" begin
        register_logit_scale!()
        x = randn(rng, Float32, 100)
        y = logit_scale(DeviceArray(x), 2.0f0; out_len=128)
        @test y isa DeviceArray && size(y) == (128,)
        @test Array(y)[1:100] ≈ 2.0f0 .* x
        @test all(iszero, Array(y)[101:128])
        xm = DeviceArray(randn(rng, Float32, 8, 10))
        xv = view(xm, :, 2:2:10)
        @test Array(logit_scale(xv, 3.0f0))[1:length(xv)] ≈ 3.0f0 .* vec(Array(xv))
        @test_throws Exception logit_scale(randn(rng, Float32, 16), 2.0f0)
    end

    @testset "ka_compile" begin
        be = GPU_TAG
        argtypes = (Vector{Float32}, Vector{Float32}, Float32, Int32, Val{64})
        b1 = ka_compile(be, logit_scale_kernel!, argtypes; tg=64, name="scale_t64", params=(; tg=64))
        p1 = b1.file
        @test b1 isa KernelBinary && b1.is_ka && b1.threadgroup == 64 && b1.params.tg == 64
        # CUDA records the flat layout it was built with (the hook is `nothing`, and nothing recorded, elsewhere).
        @test get(b1.params, :ka_layout, nothing) == (isnothing(KD.ka_layout(GPU_TAG)) ? nothing : String(KD.ka_layout(GPU_TAG)))
        @test b1.tile == (64, 1, 1)
        @test b1.abi === :ka && b1.target == KD.target_id(GPU_TAG)
        @test isfile(p1) && dirname(p1) == KernelDispatch.ka_binary_cache_dir()
        # Found, not rebuilt; the registry copy is what a variant uses.
        t = mtime(p1)
        @test ka_compile(be, logit_scale_kernel!, argtypes; tg=64, name="scale_t64").file == p1 && mtime(p1) == t
        v = add_variant!(LogitScale(), LogitScaleKA(), (:copied,); main=b1)
        @test dirname(v[:main].file) == KernelDispatch.kernel_dir(LogitScale(), LogitScaleKA()) && v[:main].is_ka
    end

    @testset "launch size: extent and tile" begin
        register_logit_scale!()
        x = randn(rng, Float32, 100)
        xd = DeviceArray(x)
        args(out_len=128) = (OutArray(Float32, out_len; zero=true), xd, 2.0f0, Int32(100), Val(LS_TG))
        ok(y) = Array(y)[1:100] ≈ 2.0f0 .* x && all(iszero, Array(y)[101:end])
        main(; kw...) = only(call_binary(LogitScale(), LogitScaleKA(), :main, args()...; kw...))

        @test ok(main(; extent=100))
        @test ok(main(; extent=(100,)))                         # a tuple, padded
        err = try main(); nothing catch e; e end                # no extent
        @test err isa ArgumentError && occursin("no launch size", err.msg)

        # A kernel without a `forward` method fails loudly instead of passing the op's arguments through.
        err = try KernelDispatch.forward(LogitScale(), LogitScaleKA(), args()...); nothing catch e; e end
        @test err isa ArgumentError && occursin("has no `forward`", err.msg)

        # Two items per thread: `tile` says so, and `extent` still counts items.
        bin2 = ka_compile(GPU_TAG, logit_scale2_kernel!,
            (Vector{Float32}, Vector{Float32}, Float32, Int32, Val{LS_TG}); tg=LS_TG,
            name="test_scale2_t$LS_TG", tile=2LS_TG)
        @test bin2.tile == (2LS_TG, 1, 1)
        register_kernel!(LogitScale(), LogitScaleTwo())
        add_variant!(LogitScale(), LogitScaleTwo(); main=bin2)
        x2 = randn(rng, Float32, 300)
        y2 = only(call_binary(LogitScale(), LogitScaleTwo(), :main, OutArray(Float32, 300),
            DeviceArray(x2), 2.0f0, Int32(300), Val(LS_TG); extent=300))
        @test Array(y2) ≈ 2.0f0 .* x2                           # 3 threadgroups of 64 threads
    end
    @testset "max_threads and time_binary" begin
        be = GPU_TAG
        bin = register_logit_scale!()[:main]
        ls_args(n) = (OutArray(Float32, n), DeviceArray(randn(rng, Float32, n)), 2.0f0, Int32(n), Val(LS_TG))
        tb(n; kw...) = minimum(time_binary(bin, ls_args(n)...; extent=n, kw...) for _ in 1:3)

        # The pipeline's own limit, at most the device's; over it, both launch paths refuse.
        lim = KD.max_threads(bin)
        @test lim == KD.max_threads(be, bin) == KD.max_threads(bin; backend=be)
        @test LS_TG <= lim <= device_max_threads()
        big = KernelBinary(bin.file; entry=bin.entry, threadgroup=4096, abi=:ka, params=bin.params)
        @test_throws ArgumentError KD.encode_launch(be, big, Any[], (1, 1, 1))
        @test_throws ArgumentError time_binary(big, ls_args(64)...; extent=64)
        @test_throws ArgumentError time_binary(bin, ls_args(64)...; extent=64, reps=0)
        @test_throws ArgumentError time_binary(bin, OutArray(Float32, 8), randn(Float32, 8), 2.0f0,
            Int32(8), Val(LS_TG); extent=8)                       # host arrays

        # A positive time. (How it scales with the work is a benchmark, not a test: it flakes.)
        t = tb(2^20)
        @test t isa Float64 && t > 0

        # Inputs are untouched; outputs are not returned.
        args = ls_args(1000)
        x0 = Array(args[2])
        @test time_binary(bin, args...; extent=1000) isa Float64
        @test Array(args[2]) == x0

        # The registered variant binds the same as the binary itself.
        @test time_binary(LogitScale(), LogitScaleKA(), :main, ls_args(1000)...; extent=1000) isa Float64

        # A sequence of launches, timed as one: each repetition dispatches all of them in order.
        l1 = prepare_launch(bin, ls_args(1000)...; extent=1000)
        l2 = prepare_launch(bin, ls_args(4000)...; extent=4000)
        ts = @inferred time_binary([l1, l2]; reps=2)
        @test ts isa Float64 && ts > 0
        @test time_binary([l1]) > 0 && time_binary(l1) > 0       # one launch: the single form
        @test_throws ArgumentError time_binary(KD.Launch[])
        @test_throws ArgumentError time_binary([l1, l2]; reps=0)
        lbig = prepare_launch(big, ls_args(64)...; extent=64)
        @test_throws ArgumentError time_binary([l1, lbig])        # over a pipeline's limit
    end
end
