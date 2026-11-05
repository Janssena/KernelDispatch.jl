# `to_device`, `device_caps` and `release!` on a backend with no extension behind its tag (the host, or a device no
# kernel exists for): the first two refuse, naming the backend; `release!` has nothing to free and does nothing.
# What the Metal and CUDA extensions do is tested on the device, in test/gpu/device_utils.test.jl.

using Test, KernelDispatch

@testset "no device behind the tag" begin
    for be in (KernelDispatch.CPUBackend(), KernelDispatch.UnknownBackend())
        e = try; KernelDispatch.to_device(be, rand(Float32, 2, 2)); nothing; catch err; err; end
        @test e isa ArgumentError
        @test occursin(string(typeof(be)), e.msg)

        e = try; KernelDispatch.device_caps(be); nothing; catch err; err; end
        @test e isa ArgumentError
        @test occursin(string(typeof(be)), e.msg)

        x = rand(Float32, 3)
        @test KernelDispatch.release!(be, x) === nothing
        @test x isa Vector{Float32}                      # untouched: still a usable host array
    end
end

@testset "DeviceCaps" begin
    caps = KernelDispatch.DeviceCaps(32, 1024, 32768, "dev")
    @test (caps.warp, caps.max_threads, caps.max_shmem, caps.name) == (32, 1024, 32768, "dev")
end
