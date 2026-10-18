# CUDA's flat KA layout needs the CUDA extension, which the CPU section otherwise never loads (other tests assert
# "no GPU extension loaded"). So the host-only checks run in a child process: test/cuda_flat_host.jl.

using Test

@testset "CUDA flat KA layout, host only (own process)" begin
    if Base.find_package("CUDA") === nothing
        @info "CUDA.jl is not installed: skipping the host-only flat-layout checks"
    else
        script = joinpath(@__DIR__, "..", "cuda_flat_host.jl")
        proc = run(ignorestatus(`$(Base.julia_cmd()) --project=$(Base.active_project()) $script`))
        @test success(proc)
    end
end
