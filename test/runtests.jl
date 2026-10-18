# Run everything:        julia --project=test test/runtests.jl
# CPU section only:      julia --project=test test/runtests.jl cpu
# GPU section only:      julia --project=test test/runtests.jl gpu
# Some sections:         julia --project=test test/runtests.jl kernels sdpa launch
#
# The CPU section needs no GPU package and runs anywhere. The GPU section runs on whichever backend
# `gpu_backend.jl` finds usable (Metal on Apple, CUDA on NVIDIA) and is skipped, not failed, when there
# is none. `KD_TEST_GPU=none|metal|cuda` overrides the choice.

using Test

const CPU_SECTIONS = ["kernels", "inference", "preferences", "sdpa", "custom_call", "cuda_launch", "binary_checks", "triton", "cuda_flat"]
const GPU_SECTIONS = ["conformance", "launch", "gpu_inference", "gpu_sdpa", "reactant", "cuda_plain"]
const GROUPS = ["cpu", "gpu"]
unknown = setdiff(ARGS, [CPU_SECTIONS; GPU_SECTIONS; GROUPS])
isempty(unknown) || error("unknown test section(s) $unknown; known: $([GROUPS; CPU_SECTIONS; GPU_SECTIONS])")
selected(name, group) = isempty(ARGS) || name in ARGS || group in ARGS
cpu(name) = selected(name, "cpu")
gpu(name) = selected(name, "gpu")

# Choose the GPU backend before the testsets, at top level, so the package it loads is visible to the
# files included below. A CPU-only run never loads one.
if any(gpu, GPU_SECTIONS)
    include("gpu_backend.jl")
else
    const GPU = nothing
end
has_gpu = GPU !== nothing
has_gpu || !any(gpu, GPU_SECTIONS) || @warn "the GPU section is skipped: no usable GPU backend"

@testset "KernelDispatch" begin
    include("fixtures.jl")

    @testset "CPU" begin
        cpu("kernels") && @testset "kernels and registry" begin
            include("cpu/kernel.test.jl")
        end

        cpu("inference") && @testset "type stability" begin
            include("cpu/inference.test.jl")
        end

        cpu("preferences") && @testset "preferences" begin
            include("cpu/preferences.test.jl")
        end

        cpu("sdpa") && @testset "SDPAOps" begin
            include("cpu/sdpa.test.jl")
        end

        cpu("custom_call") && @testset "custom_call (traced, CPU client)" begin
            include("cpu/custom_call.test.jl")      # Reactant, but never runs a kernel
        end

        cpu("binary_checks") && @testset "binary formats, targets, traced support" begin
            include("cpu/binary_checks.test.jl")
        end

        cpu("triton") && @testset "Triton binaries (metadata, ABI, shared memory)" begin
            include("cpu/triton.test.jl")
        end

        cpu("cuda_flat") && @testset "CUDA flat KA layout (host only)" begin
            include("cpu/cuda_flat.test.jl")
        end

        cpu("cuda_launch") && @testset "traced CUDA launch (encoding, no GPU)" begin
            include("cpu/cuda_launch.test.jl")
        end
    end

    if has_gpu
        @testset "GPU ($GPU)" begin
            gpu("conformance") && @testset "backend conformance" begin
                include("gpu/conformance.test.jl")
            end

            gpu("launch") && @testset "eager launches" begin
                include("gpu/launch.test.jl")
            end

            gpu("gpu_inference") && @testset "type stability on the device" begin
                include("gpu/inference.test.jl")
            end

            gpu("gpu_sdpa") && @testset "SDPAOps on the device" begin
                include("gpu/sdpa.test.jl")
            end

            gpu("reactant") && @testset "Reactant" begin
                include("gpu/reactant.test.jl")
            end

            gpu("cuda_plain") && GPU === :cuda && @testset "plain CUDA binaries, eager and traced" begin
                include("gpu/cuda_plain.test.jl")
            end
        end
    end
end
