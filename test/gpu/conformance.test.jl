# What any backend must do for an `:explicit`-ABI binary: the same checks against whichever backend is loaded
# (its `scale` fixture, from test/ext/<backend>/backend.jl), through the public launch API. A new backend is complete
# when it passes this file. The kernel is `scale(x, y, a, n)`: y[i] = a * x[i]. KA binaries, Enzyme and the ops have
# their own tests; this is the contract a backend extension implements.

using Test, Random, KernelDispatch, Reactant, BFloat16s
isdefined(@__MODULE__, :ELEMENTWISE_RTOL) || include(joinpath(@__DIR__, "..", "precision.jl"))


scale(b, x, a=3.0f0) = Array(only(run_binary(b, x, OutArray(Float32, length(x)), a, Int32(length(x)); extent=length(x))))

@testset "backend conformance: $GPU" begin
    dir = mktempdir()
    bins = explicit_binaries(dir)
    rng = Xoshiro(11)

    @testset "the backend is wired" begin
        @test KD.device_available() && KD.device_backend() === GPU_TAG
        @test KD.backend_of(DeviceArray(rand(Float32, 4))) === GPU_TAG
        @test GPU_TAG isa KD.GPUBackend
        @test !isempty(KD.supported_formats(GPU_TAG))
        @test KD.target_id(GPU_TAG) isa String && KD.current_target() == KD.target_id(GPU_TAG)
    end

    @testset "$label: eager launches" for (label, b) in bins
        @test KD.binary_format(b) in KD.supported_formats(GPU_TAG)
        @test KD.check_binary(GPU_TAG, b) === nothing
        for n in (1, 63, 64, 65, 1000)
            x = randn(rng, Float32, n)
            y = only(run_binary(b, DeviceArray(x), OutArray(Float32, n), 3.0f0, Int32(n); extent=n))
            @test y isa DeviceArray && Array(y) ≈ 3.0f0 .* x
        end
        x = randn(rng, Float32, 100)
        @test scale(b, view(DeviceArray(x), 11:60)) ≈ 3.0f0 .* x[11:60]               # a pointer offset
        xm = randn(rng, Float32, 8, 10)
        xv = view(DeviceArray(xm), :, 2:2:10)                                        # a strided wrapper, densified
        @test scale(b, xv) ≈ 3.0f0 .* vec(xm[:, 2:2:10])
        @test_throws ArgumentError prepare_launch(b, DeviceArray(x))                  # no extent
        @test_throws ArgumentError run_binary(b, rand(Float32, 4), OutArray(Float32, 4), 1.0f0, Int32(4); extent=4)
    end

    @testset "$label: limits and timing" for (label, b) in bins
        @test b.threadgroup <= KD.max_threads(b) <= device_max_threads()
        big = KernelBinary(b; threadgroup=4096)
        x = DeviceArray(rand(Float32, 64))
        args() = (OutArray(Float32, 64), x, 2.0f0, Int32(64))
        @test_throws ArgumentError time_binary(big, args()...; extent=64)
        x = DeviceArray(rand(Float32, 2^16))
        t = time_binary(b, x, OutArray(Float32, length(x)), 2.0f0, Int32(length(x)); extent=length(x))
        @test t isa Float64 && t > 0
    end

    @testset "what is refused, and why" begin
        e = try run_binary(foreign_binary(dir), DeviceArray(rand(Float32, 4)), OutArray(Float32, 4), 1.0f0,
            Int32(4); extent=4); nothing catch e; e end
        @test e isa ArgumentError && occursin("launches", e.msg)                      # the wrong format
        exotic = KernelBinary(first(bins).second; abi=:no_such_abi, signature=false)    # the ABI is refused whatever the arguments
        e = try prepare_launch(exotic, DeviceArray(rand(Float32, 4)); extent=4); nothing catch e; e end
        @test e isa ArgumentError && occursin(":no_such_abi", e.msg)                  # an ABI nothing implements
        if !isempty(KD.target_id(GPU_TAG))                                            # architecture-specific devices
            for (label, b) in bins
                wrong = KernelBinary(b; target="sm_20", signature=false)     # the architecture is what is asked, not the arguments
                ok = KD.binary_format(b) === :ptx                                    # PTX is portable
                e = try prepare_launch(wrong, DeviceArray(rand(Float32, 4)); extent=4); nothing catch e; e end
                @test ok ? e === nothing : (e isa ArgumentError && occursin("built for sm_20", e.msg))
            end
        end
    end

    @testset "traced launches" begin
        client = traced_client()
        if client === nothing
            @info "no traced client for $GPU (Metal needs the jax-mps plugin: JAX_MPS_PLUGIN): skipping traced conformance"
        else
            prev = Reactant.XLA.default_backend()
            Reactant.set_default_backend(client)
            try
                for (label, b) in bins
                    file, entry, tg, sig = b.file, b.entry, b.threadgroup, b.signature
                    loose() = KernelBinary(file; entry, threadgroup=tg, signature=sig)    # built inside the trace
                    x = randn(rng, Float32, 100)
                    f(x) = only(run_binary(loose(), x, OutArray(Float32, 100), 2.0f0, Int32(100); extent=100))
                    @test KD.traced_support(GPU_TAG, b) === nothing
                    @test Array(@jit f(Reactant.to_rarray(x))) ≈ 2.0f0 .* x
                    @test occursin(TRACED_CALL_TARGET, repr(Reactant.@code_hlo optimize = false f(Reactant.to_rarray(x))))
                end
            finally
                Reactant.set_default_backend(prev)
            end
        end
    end
    @testset "element type $T" for T in supported_eltypes()
        tbins = explicit_binaries(dir, T)
        tol = ELEMENTWISE_RTOL[T]

        @testset "$label: eager" for (label, b) in tbins
            @test KD.check_binary(GPU_TAG, b) === nothing
            for n in (1, 63, 64, 65, 1000)
                x = rounded(T, randn(rng, Float32, n))
                y = only(run_binary(b, DeviceArray(x), OutArray(T, n), 3.0f0, Int32(n); extent=n))
                @test y isa DeviceArray && eltype(y) === T          # the output is allocated in its own type
                @test maxrelerr(Array(y), 3.0 .* Float64.(x)) <= tol
            end
            # a view, a strided wrapper and a zero-filled longer output keep their element type too
            x = rounded(T, randn(rng, Float32, 100))
            @test eltype(only(run_binary(b, view(DeviceArray(x), 11:60), OutArray(T, 50; zero=true), 3.0f0,
                Int32(50); extent=50))) === T
        end

        @testset "$label: the bindings keep each buffer's type" for (label, b) in tbins
            x = DeviceArray(rounded(T, randn(rng, Float32, 8)))
            rec = only(KD.record_bindings(() -> run_binary(b, x, OutArray(T, 8), 2.0f0, Int32(8); extent=8)))
            slots = Dict(rec.slots)
            @test eltype(slots[:out]) === T && eltype(slots[:in]) === T
            @test slots[:scalar] isa Int32 || slots[:scalar] === 2.0f0
        end

        @testset "$label: traced" for (label, b) in tbins
            client = traced_client()
            client === nothing && (@info "no traced client for $GPU: skipping traced $T"; continue)
            prev = Reactant.XLA.default_backend()
            Reactant.set_default_backend(client)
            try
                file, entry, tg, sig = b.file, b.entry, b.threadgroup, b.signature
                x = rounded(T, randn(rng, Float32, 100))
                # the binary is built inside the trace: Reactant cannot trace a `KernelBinary` captured by a closure
                f(x) = only(run_binary(KernelBinary(file; entry, threadgroup=tg, signature=sig), x, OutArray(eltype(x), 100), 2.0f0,
                    Int32(100); extent=100))
                y = Array(@jit f(Reactant.to_rarray(x)))
                @test eltype(y) === T
                @test maxrelerr(y, 2.0 .* Float64.(x)) <= tol
            finally
                Reactant.set_default_backend(prev)
            end
        end
    end
end
