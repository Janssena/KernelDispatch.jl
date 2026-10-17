# The tutorial example, checked on the loaded GPU backend: its kernels against the host implementation,
# switching, the bindings each binary gets, and Enzyme gradients against Enzyme through the host path.
# `FlashAttn` is a prebuilt `.metallib`, so on any other backend (CUDA) only the KernelAbstractions
# kernel, `SDPAKA`, is checked.
#
# The backend is chosen by KernelDispatch's `test/gpu_backend.jl` (Metal, else CUDA; `KD_TEST_GPU` forces
# one). Run from KernelDispatch's own suite, that choice has been made already.

using SDPAOps, KernelDispatch, Enzyme, Random, Test

isdefined(Main, :GPU) || include(joinpath(pkgdir(KernelDispatch), "test", "gpu_backend.jl"))
const GPU = Main.GPU
GPU === nothing && error("no GPU backend: ", Main.GPU_WHY)
const DeviceArray = Main.DeviceArray
const DEFAULT_KERNEL = GPU === :metal ? :flash : :ka       # `@default_kernel` lists `FlashAttn` first
const KERNELS = GPU === :metal ? (:flash, :ka) : (:ka,)

rng = Xoshiro(1)
d, n, m, dv = 16, 100, 70, 8
q, k, v = randn(rng, Float32, d, n), randn(rng, Float32, d, m), randn(rng, Float32, dv, m)
qd, kd, vd = DeviceArray(q), DeviceArray(k), DeviceArray(v)
ref = SDPA()(q, k, v)                                    # host arrays: the host path
GPU === :metal || use_kernel!(:sdpa, :ka)                # the default kernel has no binary here

@testset "kernels" begin
    @test current_kernel(:sdpa) === DEFAULT_KERNEL        # the `@default_kernel` (Metal), else the switch above
    @test Array(SDPA()(qd, kd, vd)) ≈ ref
    # One call with a particular kernel: its `forward`, no switch.
    @test Array(KernelDispatch.forward(SDPA(), SDPAOps.SDPAKA(), qd, kd, vd)) ≈ ref
    @test current_kernel(SDPA()) === DEFAULT_KERNEL       # nothing switched
    rec(kern) = map(first, only(KernelDispatch.record_bindings(() -> KernelDispatch.forward(SDPA(), kern, qd, kd, vd))).slots)
    GPU === :metal &&
        @test rec(SDPAOps.FlashAttn()) == (:in, :in, :in, :out, :scalar, :scalar, :scalar, :scalar, :scalar)
    # A KA binary takes the backend's state word(s) and KA's context first.
    r = rec(SDPAOps.SDPAKA())
    @test r[end-8:end] == (:out, :in, :in, :in, :scalar, :scalar, :scalar, :scalar, :scalar)
    @test length(r) > 9 && all(==(:bytes), r[1:end-9])
end

w = randn(rng, Float32, dv, n)
loss(q, k, v, w) = sum(SDPA()(q, k, v) .* w)
dq, dk, dv_ = zero(q), zero(k), zero(v)
Enzyme.autodiff(Reverse, loss, Active, Duplicated(q, dq), Duplicated(k, dk), Duplicated(v, dv_), Const(w))

f(q, k, v) = SDPA()(q, k, v)
function gpu_grads(qd, kd, vd, w)
    gq, gk, gv = zero(qd), zero(kd), zero(vd)
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f)}, Duplicated,
        Duplicated{typeof(qd)}, Duplicated{typeof(kd)}, Duplicated{typeof(vd)})
    args = (Const(f), Duplicated(qd, gq), Duplicated(kd, gk), Duplicated(vd, gv))
    tape, o, dout = fwd(args...)
    dout .= DeviceArray(w)
    rev(args..., tape)
    return o, gq, gk, gv
end

# Switched at top level: a thunk holds the kernel selected when it was built.
for kname in KERNELS
    use_kernel!(:sdpa, kname)
    @eval @testset "Enzyme, $($(QuoteNode(kname)))" begin
        o, gq, gk, gv = gpu_grads(qd, kd, vd, w)
        @test Array(o) ≈ ref
        @test Array(gq) ≈ dq rtol = 1e-4
        @test Array(gk) ≈ dk rtol = 1e-4
        @test Array(gv) ≈ dv_ rtol = 1e-4
    end
end
reset_kernel!(:sdpa)
