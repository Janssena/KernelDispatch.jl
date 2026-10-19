# Run everything:        julia --project=test test/runtests.jl
# CPU section only:      julia --project=test test/runtests.jl cpu
# GPU section only:      julia --project=test test/runtests.jl gpu
# Backend-specific only: julia --project=test test/runtests.jl ext        (test/ext/<backend>/)
# Some sections:         julia --project=test test/runtests.jl kernels sdpa launch
#
# Three kinds of tests:
#   cpu/   needs no GPU package and runs anywhere.
#   gpu/   GENERIC: runs on whichever backend `gpu_backend.jl` finds usable (Metal on Apple, CUDA on NVIDIA) and is
#          skipped, not failed, when there is none. It uses only the names its adapter, test/ext/<backend>/backend.jl,
#          provides (see gpu_backend.jl), so one test base covers every backend.
#   ext/   specific to one backend extension, in test/ext/<backend>/ next to its adapter: the mechanisms a generic
#          failure would not pin down (and, for CUDA, the diagnostic scripts smoke.jl and probe*.jl). Some need no device
#          (`custom_call`, `cuda_launch`, `cuda_flat`: they run with the CPU group), the rest need that backend's GPU.
# `KD_TEST_GPU=none|metal|cuda` overrides the GPU backend choice.
# `JULIA_TEST_VERBOSE=true` prints the counts of the nested testsets too, also when they pass (Julia's Test hides them;
# the variable is read here, not by Julia). Example:  JULIA_TEST_VERBOSE=true julia --project=test test/runtests.jl gpu

using Test
using KernelDispatch

# The one alias every test file uses for the package (the files are `include`d here; standalone scripts such as
# ext/cuda/smoke.jl and flat_layout_host.jl define their own).
const KD = KernelDispatch

const VERBOSE = lowercase(get(ENV, "JULIA_TEST_VERBOSE", "false")) in ("true", "1", "yes")

const CPU_SECTIONS = ["kernels", "inference", "preferences", "sdpa", "binary_checks", "triton"]
const EXT_HOST_SECTIONS = ["custom_call", "cuda_launch", "cuda_flat"]      # ext/, no device needed
const GPU_SECTIONS = ["conformance", "launch", "gpu_inference", "gpu_sdpa", "traced"]
const EXT_GPU_SECTIONS = ["ext_cuda"]                                       # ext/, needs that backend's device
const GROUPS = ["cpu", "gpu", "ext"]
known = [GROUPS; CPU_SECTIONS; EXT_HOST_SECTIONS; GPU_SECTIONS; EXT_GPU_SECTIONS]
unknown = setdiff(ARGS, known)
isempty(unknown) || error("unknown test section(s) $unknown; known: $known")
selected(name, groups...) = isempty(ARGS) || name in ARGS || any(in(ARGS), groups)
cpu(name) = selected(name, "cpu")
ext_host(name) = selected(name, "cpu", "ext")
gpu(name) = selected(name, "gpu")
ext_gpu(name) = selected(name, "gpu", "ext")

# Choose the GPU backend before the testsets, at top level, so the package it loads is visible to the
# files included below. A CPU-only run never loads one.
needs_gpu = any(gpu, GPU_SECTIONS) || any(ext_gpu, EXT_GPU_SECTIONS)
if needs_gpu
    include("gpu_backend.jl")
else
    const GPU = nothing
end
has_gpu = GPU !== nothing
has_gpu || !needs_gpu || @warn "the GPU section is skipped: no usable GPU backend"

@testset verbose = VERBOSE "KernelDispatch" begin
    include("fixtures.jl")

    @testset verbose = VERBOSE "CPU" begin
        cpu("kernels") && @testset verbose = VERBOSE "kernels and registry" begin
            include("cpu/kernel.test.jl")
        end

        cpu("inference") && @testset verbose = VERBOSE "type stability" begin
            include("cpu/inference.test.jl")
        end

        cpu("preferences") && @testset verbose = VERBOSE "preferences" begin
            include("cpu/preferences.test.jl")
        end

        cpu("sdpa") && @testset verbose = VERBOSE "SDPAOps" begin
            include("cpu/sdpa.test.jl")
        end

        cpu("binary_checks") && @testset verbose = VERBOSE "binary formats, targets, traced support" begin
            include("cpu/binary_checks.test.jl")
        end

        cpu("triton") && @testset verbose = VERBOSE "Triton binaries (reader, ABI, real-Triton fixtures)" begin
            include("cpu/triton.test.jl")
        end

    end

    # Backend-specific tests that need no device: Reactant on the CPU client, or a child process that loads the
    # extension. Run with the CPU group, or on their own with `ext`.
    @testset verbose = VERBOSE "Backend extensions (host only)" begin
        ext_host("custom_call") && @testset verbose = VERBOSE "Metal: custom_call (traced, CPU client)" begin
            include("ext/metal/custom_call.test.jl")      # Reactant, but never runs a kernel
        end

        ext_host("cuda_launch") && @testset verbose = VERBOSE "CUDA: traced launch (encoding, no GPU)" begin
            include("ext/cuda/launch_encoding.test.jl")
        end

        ext_host("cuda_flat") && @testset verbose = VERBOSE "CUDA: flat KA layout (host only)" begin
            include("ext/cuda/flat_layout.test.jl")
        end
    end

    if has_gpu
        @testset verbose = VERBOSE "GPU ($GPU)" begin
            gpu("conformance") && @testset verbose = VERBOSE "backend conformance" begin
                include("gpu/conformance.test.jl")
            end

            gpu("launch") && @testset verbose = VERBOSE "eager launches" begin
                include("gpu/launch.test.jl")
            end

            gpu("gpu_inference") && @testset verbose = VERBOSE "type stability on the device" begin
                include("gpu/inference.test.jl")
            end

            gpu("gpu_sdpa") && @testset verbose = VERBOSE "SDPAOps on the device" begin
                include("gpu/sdpa.test.jl")
            end

            gpu("traced") && @testset verbose = VERBOSE "Traced (Reactant)" begin
                include("gpu/traced.test.jl")
            end

            ext_gpu("ext_cuda") && GPU === :cuda && @testset verbose = VERBOSE "CUDA extension" begin
                @testset verbose = VERBOSE "explicit binaries, Triton ABI, flat KA" begin
                    include("ext/cuda/explicit.test.jl")
                end
                @testset verbose = VERBOSE "Triton kernels, eager and traced" begin
                    include("ext/cuda/triton.test.jl")
                end
                @testset verbose = VERBOSE "trifast's Triton kernels, eager and traced" begin
                    include("ext/cuda/trifast.test.jl")
                end
            end
        end
    end
end
