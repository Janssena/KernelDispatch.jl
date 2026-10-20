# Building a variant on demand from inside a Reactant trace. Backend-agnostic: it traces on Reactant's own CPU client and
# registers a fake file, so no device, no kernel and no vendor format is involved.
#
# The build (`build!`) compiles a GPU kernel and writes a file. Inside a trace the tracer would interpret that compilation;
# on CUDA it crashed the process. `KernelDispatch.untraced` runs it natively instead.

using Test, KernelDispatch, Reactant

# A variant a trace asks for and that does not exist yet is built by `build!`, which compiles a kernel: that must run outside
# the tracer (inside it, CUDA.jl's compiler crashed the process). No kernel runs here: the build only registers a fake file.
struct UntracedOp <: AbstractKernelOp end
struct UntracedKernel <: AbstractKernel end
const UNTRACED_DIR = mktempdir()
const UNTRACED_BUILDS = Ref(0)
const UNTRACED_BUILD_TRACING = Ref(true)
KD.cache_dir(::UntracedOp) = UNTRACED_DIR
function KD.build!(op::UntracedOp, k::UntracedKernel, key::Tuple)
    UNTRACED_BUILDS[] += 1
    UNTRACED_BUILD_TRACING[] = KD.tracing()      # false: the tracer does not interpret this call
    f = joinpath(mktempdir(), "k.bin"); write(f, UInt8[1])      # never launched, so any file will do: no backend's format
    add_variant!(op, k, key; fwd=KernelBinary(f; threadgroup=32, signature=false))
end

@testset "a variant built from inside a trace is built outside the tracer" begin
    @test KD.untraced(() -> 7) == 7                           # not tracing: just the call
    f(x) = (KD.variant(UntracedOp(), UntracedKernel(), (:K,)); x .+ 1.0f0)
    x = Reactant.to_rarray(rand(Float32, 4))
    @test occursin("stablehlo.add", repr(Reactant.@code_hlo optimize = false f(x)))
    @test UNTRACED_BUILDS[] == 1
    @test !UNTRACED_BUILD_TRACING[]                                       # `build!` ran natively, not under the tracer
    @test haskey(KD.variants(UntracedOp(), UntracedKernel()), (:K,))
    @test !KD.tracing()
end
