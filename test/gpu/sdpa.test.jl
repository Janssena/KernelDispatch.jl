# The tutorial example package (examples/SDPAOps) on the loaded GPU backend: its kernels, in each element type the
# backend supports, against a Float64 reference computed from the same rounded inputs, the lazy variant build and its manifest, the example's own test file, and the kernels traced as
# custom calls (on the backend's Reactant client). The device-free half is cpu/sdpa.test.jl.
#
# `FlashAttn` is a prebuilt `.metallib`: on any other backend only `SDPAKA` has a binary.

using Test, Random, KernelDispatch, SDPAOps, Enzyme, Reactant, BFloat16s
isdefined(@__MODULE__, :ELEMENTWISE_RTOL) || include(joinpath(@__DIR__, "..", "precision.jl"))
isdefined(@__MODULE__, :Residual) || include(joinpath(@__DIR__, "..", "fixtures.jl"))     # also keeps SDPAOps' manifests out of the user's cache
isdefined(@__MODULE__, :sdpa_reference) || include(joinpath(@__DIR__, "..", "sdpa_helpers.jl"))

const SDPA_GPU_KERNELS = (prebuilt_sdpa_kernels()..., SDPAKA())

@testset "device" begin
    rng = Xoshiro(3)
    # The SDPA kernels are shipped for each type the backend reports it supports (`supported_eltypes()`).
    @testset "$kern, $T, d=$d n=$n m=$m dv=$dv" for kern in SDPA_GPU_KERNELS, T in supported_eltypes(),
            (d, n, m, dv) in SDPA_SIZES
        q, k, v = inputs(rng, d, n, m, dv, T)
        o = KernelDispatch.forward(SDPA(), kern, DeviceArray(q), DeviceArray(k), DeviceArray(v))
        @test o isa DeviceArray && size(o) == (dv, n)
        @test eltype(o) === T                                # in, and out, in the inputs' type
        @test maxrelerr(Array(o), sdpa_reference(q, k, v)) <= REDUCTION_RTOL[T]
    end

    @testset "variants are built on first use and persisted" begin
        for kern in SDPA_GPU_KERNELS, T in supported_eltypes()
            q, k, v = inputs(rng, 16, 10, 7, 8, T)
            qd, kd, vd = DeviceArray(q), DeviceArray(k), DeviceArray(v)
            KernelDispatch.forward(SDPA(), kern, qd, kd, vd)
            vs = KernelDispatch.variants(SDPA(), kern)
            key = (nameof(T),)
            @test haskey(vs, key)
            bin = vs[key][:fwd]
            @test isfile(bin.file)
            @test isfile(joinpath(KernelDispatch.kernel_dir(SDPA(), kern), "manifest.toml"))
            t = mtime(bin.file)
            KernelDispatch.forward(SDPA(), kern, qd, kd, vd)             # found, not rebuilt
            @test KernelDispatch.variants(SDPA(), kern) === vs && mtime(bin.file) == t
        end
        for kern in SDPA_GPU_KERNELS                                      # one variant per type, never shared
            vs = KernelDispatch.variants(SDPA(), kern)
            files = [vs[(nameof(T),)][:fwd].file for T in supported_eltypes()]
            @test length(unique(files)) == length(files)
        end
    end

    @testset "mixed element types are refused, not computed" begin
        q, k, v = inputs(rng, 16, 10, 7, 8)
        for kern in SDPA_GPU_KERNELS
            @test_throws ArgumentError KernelDispatch.forward(SDPA(), kern, DeviceArray(q),
                DeviceArray(Float16.(k)), DeviceArray(v))
        end
    end

    @testset "no variant for an element type the kernel lacks" begin
        # `build!` declines any key but the three float types: Float64 is not in scope.
        for kern in SDPA_GPU_KERNELS
            @test_throws ArgumentError KernelDispatch.variant(SDPA(), kern, (:Float64,))
        end
    end

    @testset "gradients come back in the inputs' type" begin
        for T in supported_eltypes()
            q, k, v = inputs(rng, 8, 6, 5, 4, T)
            dout = T.(randn(rng, Float32, 4, 6))
            args = map(DeviceArray, (q, k, v))
            dq, dk, dv_ = KernelDispatch.backward(SDPA(), SDPAKA(), args, nothing, DeviceArray(dout))
            @test eltype(dq) === T && eltype(dk) === T && eltype(dv_) === T
            ref = KernelDispatch.backward(SDPA(), SDPAKA(), (Float64.(q), Float64.(k), Float64.(v)), nothing, Float64.(dout))
            for (g, r) in zip((dq, dk, dv_), ref)
                @test maxrelerr(Array(g), r) <= REDUCTION_RTOL[T]
            end
        end
    end

    isempty(prebuilt_sdpa_kernels()) && @testset "FlashAttn has no binary on this backend" begin
        @test_throws ArgumentError KernelDispatch.variant(SDPA(), FlashAttn(), (:Float32,))
    end
end

# The example's own test file, in a module of its own: its top-level helpers stay out of this one,
# and the file stays the single source of truth for what the tutorial claims.
@testset "SDPAOps example tests" begin
    Base.include(Module(:SDPAExampleTests), joinpath(pkgdir(SDPAOps), "test", "runtests.jl"))
end

# Traced: the SAME binaries the eager path launches, embedded as the backend's custom call (`TRACED_CALL_TARGET`), on the
# backend's own Reactant client (`traced_client()`: jax-mps on Metal when the plugin is found, Reactant's CUDA client on
# CUDA). `SDPAKA` puts its output first (`o, q, k, v, ...`), which the CUDA embedding supports by aliasing the output.
sdpa_client = traced_client()
if sdpa_client === nothing
    @info "no traced client for $GPU (Metal needs the jax-mps plugin: JAX_MPS_PLUGIN): skipping traced SDPAOps tests"
else
    prev = Reactant.XLA.default_backend()
    Reactant.set_default_backend(sdpa_client)
    try
        @testset "Reactant" begin
            rng = Xoshiro(4)
            # The kernel is named in the call, not switched: nothing global changes under the trace.
            for kern in SDPA_GPU_KERNELS
                @testset "$kern, $T" for T in supported_eltypes(), (d, n, m, dv) in SDPA_SIZES
                    q, k, v = inputs(rng, d, n, m, dv, T)
                    args = map(Reactant.to_rarray, (q, k, v))
                    f(q, k, v) = KernelDispatch.forward(SDPA(), kern, q, k, v)
                    o = Array(@jit f(args...))
                    @test eltype(o) === T
                    @test maxrelerr(o, sdpa_reference(q, k, v)) <= REDUCTION_RTOL[T]
                    @test occursin(TRACED_CALL_TARGET, repr(Reactant.@code_hlo optimize = false f(args...)))
                end
            end
            @test current_kernel(SDPA()) === :flash
        end
    finally
        Reactant.set_default_backend(prev)
    end
end
