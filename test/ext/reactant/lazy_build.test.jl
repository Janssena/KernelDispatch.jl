# A variant that a trace asks for and that does not exist yet is built on demand (`build!`). Building compiles a GPU kernel
# and writes a file, which the tracer must not interpret: on CUDA, doing it from inside the trace crashed the process (signal
# 11 in Reactant's `call_with_reactant`). `KernelDispatch.variant` therefore builds through `KernelDispatch.untraced`.
#
# Backend-agnostic: the op builds its kernel with `ka_compile` for whatever GPU backend is loaded, and every launch below is
# traced on that backend's Reactant client and checked against the numbers. The first launch of each element type is the
# trace that triggers the build, nothing is built beforehand.

using Test, Random, KernelDispatch, Reactant, BFloat16s
isdefined(@__MODULE__, :LS_TG) || include(joinpath(@__DIR__, "..", "..", "fixtures.jl"))            # logit_scale_kernel!, LS_TG
isdefined(@__MODULE__, :ELEMENTWISE_RTOL) || include(joinpath(@__DIR__, "..", "..", "precision.jl"))

struct LazyScale <: AbstractKernelOp end
struct LazyScaleKA <: AbstractKernel end
const LAZY_DIR = mktempdir()
const LAZY_BUILDS = Ref(0)
const LAZY_BUILD_TRACING = Ref(true)
KD.cache_dir(::LazyScale) = LAZY_DIR
KD.opname(::LazyScale) = :lazy_scale
KD.kernelname(::LazyScaleKA) = :ka
KD.variant_key(::LazyScale, ::LazyScaleKA, args...) = (nameof(KD.realtype(eltype(KD._first_array(args)))),)
KD.extras(::LazyScale, ::LazyScaleKA, b::KernelBinary) = (Val(b.params.tg),)
KD.host(::LazyScale, x) = 2.0f0 .* x

const LAZY_TYPES = Dict(:Float32 => Float32, :Float16 => Float16, :BFloat16 => BFloat16)

# y = 2x with the fixtures' KA kernel, compiled when first asked for.
function KD.build!(op::LazyScale, k::LazyScaleKA, key::Tuple)
    T = LAZY_TYPES[only(key)]
    LAZY_BUILDS[] += 1
    LAZY_BUILD_TRACING[] = KD.tracing()          # false: the tracer does not interpret the build
    fwd = ka_compile(GPU_TAG, logit_scale_kernel!, (Vector{T}, Vector{T}, Float32, Int32, Val{LS_TG});
        tg=LS_TG, name="lazy_scale", params=(; tg=LS_TG), dir=mktempdir())
    add_variant!(op, k, key; fwd)
end

KD.forward(op::LazyScale, k::LazyScaleKA, x) =
    only(call_binary(op, k, :fwd, OutArray(eltype(x), length(x)), x, 2.0f0, Int32(length(x)); extent=length(x)))

lazy_client = traced_client()
if lazy_client === nothing
    @info "no traced client for $GPU: skipping the lazy-build tests"
else
    lazy_prev = Reactant.XLA.default_backend()
    Reactant.set_default_backend(lazy_client)
    try
        @testset "a variant a trace asks for is built outside the tracer" begin
            rng = Xoshiro(41)
            @test KD.untraced(() -> 7) == 7                              # not tracing: just the call
            @test isempty(KD.variants(LazyScale(), LazyScaleKA()))
            for (i, T) in enumerate(supported_eltypes())
                x = rounded(T, randn(rng, Float32, 100))
                f(x) = KD.forward(LazyScale(), LazyScaleKA(), x)
                y = Array(Reactant.@jit f(Reactant.to_rarray(x)))
                @test eltype(y) === T
                @test maxrelerr(y, 2.0 .* Float64.(x)) <= ELEMENTWISE_RTOL[T]
                @test LAZY_BUILDS[] == i                                 # built by that trace, once
                @test !LAZY_BUILD_TRACING[]                              # and natively, not under the tracer
                @test haskey(KD.variants(LazyScale(), LazyScaleKA()), (nameof(T),))
                # asked again: found, not rebuilt
                Array(Reactant.@jit f(Reactant.to_rarray(x)))
                @test LAZY_BUILDS[] == i
            end
            @test !KD.tracing()
        end
    finally
        Reactant.set_default_backend(lazy_prev)
    end
end
