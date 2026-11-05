# What is specific to CUDA in `device_caps`: warp 32, static shared memory (KernelAbstractions' `@localmem` is static, and
# CUDA caps that at 48 KiB per block), and a name that carries the compute capability (it keys persisted calibrations).

using Test, KernelDispatch

@testset "CUDA device_caps" begin
    caps = KernelDispatch.device_caps(GPU_TAG)
    @test caps.warp == 32
    @test 0 < caps.max_shmem <= 48 * 1024
    @test occursin(r"sm_\d+", caps.name)
end
