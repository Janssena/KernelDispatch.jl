# Eager launches are asynchronous: `encode_launch` returns once the dispatch is enqueued. What must still hold is the
# ordering with the backend's own work on the same task (Metal.jl's and CUDA.jl's fills, broadcasts, GEMVs), that a host
# read waits, and that an array freed right after a launch stays valid until the kernel is done. Every check here
# interleaves the two kinds of work with NO synchronize in between and reads once at the end, so a launch that ran too
# early or late, or a buffer recycled under it, shows up as a wrong number. Backend-neutral: see gpu_backend.jl.
#
# Launches wait by default; this file opts in to `KD.ASYNC_LAUNCHES` for its duration and restores it afterwards.

using Test, Random, LinearAlgebra, KernelDispatch
isdefined(@__MODULE__, :Residual) || include(joinpath(@__DIR__, "..", "fixtures.jl"))

async_was = KD.ASYNC_LAUNCHES[]
KD.ASYNC_LAUNCHES[] = true
try
begin
    rng = Random.Xoshiro(7)
    bin = register_logit_scale!()[:main]
    unsafe_free!(x) = GPU_PKG.unsafe_free!(x)

    # `y = a * x` through the binary, into a fresh output; `a` is a new scalar value on every call.
    scale(x, a) = only(run_binary(bin, OutArray(Float32, length(x)), x, Float32(a), Int32(length(x)),
        Val(LS_TG); extent=length(x)))

    # An in-place launch: `y = a * x` into the existing `y`. No allocation and, once a scalar value has been bound once,
    # no upload either: Metal.jl's allocations and uploads flush its batch, which would hide an ordering gap.
    scale!(y, x, a) = only(run_binary(bin, InOutArray(y), x, Float32(a), Int32(length(x)), Val(LS_TG);
        extent=length(x)))

    @testset "launches ordered with the backend's own work" begin
        n, K = 256, 200
        x0 = rand(rng, Float32, n)
        M0 = rand(rng, Float32, n, n) ./ n
        scalars = (0.9f0, 1.0f0, 1.1f0)                   # a few values, cached after the first launch
        a(i) = scalars[mod1(i, length(scalars))]

        # Reference: broadcast, launch, GEMV, K times, on the host.
        xr = copy(x0)
        for i in 1:K
            xr = M0 * (a(i) .* (xr .+ 1.0f0))
        end

        x, M = DeviceArray(x0), DeviceArray(M0)
        b, y, w = (DeviceArray(zeros(Float32, n)) for _ in 1:3)
        for a_ in scalars                                  # bind every scalar once, outside the checked loop
            scale!(y, b, a_)
        end
        Array(y)

        for i in 1:K
            b .= x .+ 1.0f0                                # the backend's broadcast writes b ...
            scale!(y, b, a(i))                             # ... the launch reads it and writes y ...
            fill!(w, 0.0f0)
            mul!(w, M, y)                                  # ... the backend's GEMV reads y
            x, w = w, x
        end
        @test Array(x) ≈ xr rtol = 1.0f-3                  # one host read at the very end
    end

    @testset "fresh scalar values, fresh outputs" begin
        # Each new scalar value is a small device upload, each output an allocation, both on the backend's queue.
        n, K = 256, 100
        x0 = rand(rng, Float32, n)
        x = DeviceArray(x0)
        xr = copy(x0)
        for i in 1:K
            a = 1.0f0 + 1.0f-4 * i
            x = scale(x .+ 1.0f0, a)
            xr = a .* (xr .+ 1.0f0)
        end
        @test Array(x) ≈ xr rtol = 1.0f-4
    end

    @testset "a read right after one launch needs no synchronize" begin
        for _ in 1:20
            x = rand(rng, Float32, 1000)
            @test Array(scale(DeviceArray(x), 3.0f0)) ≈ 3.0f0 .* x
        end
    end

    @testset "an input freed right after the launch that reads it" begin
        n, K = 512, 100
        xs = [rand(rng, Float32, n) for _ in 1:K]
        ys = map(1:K) do i
            xd = DeviceArray(xs[i])
            y = scale(xd, 2.0f0)
            unsafe_free!(xd)                             # the kernel has probably not run yet
            DeviceArray(fill(Float32(-i), n))            # a new allocation that may take the freed memory
            y
        end
        for i in 1:K
            @test Array(ys[i]) ≈ 2.0f0 .* xs[i]
        end
    end

    @testset "a temporary the launch owns, collected by the GC right after" begin
        # A strided `view` is copied on the device (`device_dense`) and the copy is the launch's own temporary: nothing
        # else references it, so the GC may free it while the kernel is still queued.
        n, K = 512, 50
        xs = [rand(rng, Float32, 2n) for _ in 1:K]
        ys = map(1:K) do i
            xd = DeviceArray(xs[i])
            y = scale(view(xd, 1:2:2n), 2.0f0)           # strided: the temporary copy
            xd = nothing
            GC.gc()                                      # frees the copy (and `xd`) now
            DeviceArray(fill(Float32(-i), 4n))           # an allocation that may take the freed memory
            y
        end
        for i in 1:K
            @test Array(ys[i]) ≈ 2.0f0 .* xs[i][1:2:2n]
        end
    end

    @testset "the documented cross-task pattern: synchronize on the launching task" begin
        # Ordering is per task. A task that launches and then hands the array to another must synchronize first; with
        # that, the reader sees every launch. (Without it the read is a race: test/probes/async_hazards.jl measures it.)
        n, K = 1 << 16, 100
        x0 = rand(rng, Float32, n)
        y = fetch(@async begin
            y = DeviceArray(copy(x0))
            for _ in 1:K
                scale!(y, y, 1.001f0)
            end
            GPU_PKG.synchronize()
            y
        end)
        @test Array(y) ≈ x0 .* 1.001f0^K rtol = 1.0f-3
    end

    @testset "many queued launches" begin
        # Thousands of launches with no synchronize in between must neither stall nor corrupt, and the device memory
        # they hold (outputs, temporaries) comes back once the queue drains.
        n, K = 4096, 5000
        x = DeviceArray(rand(rng, Float32, n))
        y = DeviceArray(zeros(Float32, n))
        t = @elapsed begin
            for _ in 1:K
                scale!(y, x, 2.0f0)
            end
            for _ in 1:K
                scale(x, 3.0f0)                          # the output is dropped at once
            end
            GPU_PKG.synchronize()
        end
        @test Array(y) ≈ 2.0f0 .* Array(x)
        @test t < 60                                     # seconds; milliseconds in practice. A hang is the failure
    end

    @testset "waiting launches (the default) give the same results" begin
        x = rand(rng, Float32, 1000)
        try
            KD.ASYNC_LAUNCHES[] = false
            @test Array(scale(DeviceArray(x), 5.0f0)) ≈ 5.0f0 .* x
            y = DeviceArray(zeros(Float32, 1000))
            for _ in 1:20
                scale!(y, DeviceArray(x), 2.0f0)
            end
            @test Array(y) ≈ 2.0f0 .* x
        finally
            KD.ASYNC_LAUNCHES[] = true
        end
    end

    @testset "timing still waits, after asynchronous launches" begin
        n = 1000
        x = DeviceArray(rand(rng, Float32, n))
        for _ in 1:50
            scale(x, 2.0f0)
        end
        t = time_binary(bin, OutArray(Float32, n), x, 2.0f0, Int32(n), Val(LS_TG); extent=n)
        @test t isa Float64 && t > 0
    end
end
finally
    KD.ASYNC_LAUNCHES[] = async_was
end
