# Probes for the hazards of asynchronous eager launches (docs/src/backends.md, "Asynchronous launches"). NOT part of the
# suite: each measures a race, whose outcome depends on timing, so it reports instead of asserting.
#
#     KD_TEST_GPU=metal julia --project=test test/probes/async_hazards.jl          (or cuda)
#
# 1. CROSS-TASK READ WITHOUT SYNCHRONIZE. One task launches a long chain; another reads at once. Ordering is per task, so the
#    reader may see the array before the chain finished. Prints how many of the trials read a stale (incomplete) result.
#    Measured: Metal (M1 Pro) reads stale in most trials (29 of 30). CUDA (A40) reads 0 stale, with or without
#    `SYNC_LAUNCHES`: CUDA.jl makes a reader on another stream wait for the memory's owner. With `SYNC_LAUNCHES` set
#    Metal also reads 0 stale.
# 2. WALL-CLOCK TIMING OF AN EAGER LAUNCH. `@elapsed` around launches without a synchronize now measures enqueueing only.
# 3. A LATE ERROR. A launch that fails device-side does not throw at the launch (not probed: it needs a faulting kernel).

using Test, Random, KernelDispatch
const KD = KernelDispatch
include(joinpath(@__DIR__, "..", "gpu_backend.jl"))
GPU === nothing && (println("no GPU backend"); exit(0))
include(joinpath(@__DIR__, "..", "fixtures.jl"))

bin = register_logit_scale!()[:main]
scale!(y, x, a) = only(run_binary(bin, InOutArray(y), x, Float32(a), Int32(length(x)), Val(LS_TG); extent=length(x)))

function chain_then_read_elsewhere(; n=1 << 20, K=300, trials=30)
    x0 = ones(Float32, n)
    expect = 1.0001f0^K
    stale = 0
    for _ in 1:trials
        y = fetch(@async begin                               # a new task: its own queue / stream
            y = DeviceArray(copy(x0))
            for _ in 1:K
                scale!(y, y, 1.0001f0)
            end
            y                                                # no synchronize: this is the hazard
        end)
        a = Array(y)                                         # read on THIS task
        stale += !(all(isapprox.(a, expect; rtol=1f-3)))
        GPU_PKG.synchronize()
    end
    return stale, trials
end

for sync in (false, true)
    KD.SYNC_LAUNCHES[] = sync
    stale, trials = chain_then_read_elsewhere()
    println("SYNC_LAUNCHES=$sync: $stale of $trials cross-task reads were stale")
end
KD.SYNC_LAUNCHES[] = false

let n = 1 << 20, K = 300
    x, y = DeviceArray(rand(Float32, n)), DeviceArray(zeros(Float32, n))
    scale!(y, x, 2f0); GPU_PKG.synchronize()
    enq = @elapsed for _ in 1:K scale!(y, x, 2f0) end
    GPU_PKG.synchronize()
    full = @elapsed begin
        for _ in 1:K scale!(y, x, 2f0) end
        GPU_PKG.synchronize()
    end
    println("$K launches: @elapsed without synchronize = $(round(1e3enq; digits=1)) ms (enqueue only), with = $(round(1e3full; digits=1)) ms")
end
