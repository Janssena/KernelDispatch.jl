# The device utilities on the backend `gpu_backend.jl` found: `to_device`, `device_caps`, `release!`. Uses only the
# names the adapter provides (`GPU_TAG`, `DeviceArray`, `device_max_threads`).

using Test, KernelDispatch

@testset "to_device" begin
    x = rand(Float32, 3, 4)
    d = KernelDispatch.to_device(GPU_TAG, x)
    @test d isa DeviceArray
    @test Array(d) == x

    # Any `AbstractArray` goes over, wrappers included, as a plain device array of the same values.
    v = view(x, 1:2, 2:3)
    dv = KernelDispatch.to_device(GPU_TAG, v)
    @test dv isa DeviceArray
    @test Array(dv) == x[1:2, 2:3]
    dt = KernelDispatch.to_device(GPU_TAG, x')
    @test dt isa DeviceArray
    @test Array(dt) == x'
end

@testset "device_caps" begin
    caps = KernelDispatch.device_caps(GPU_TAG)
    @test caps isa KernelDispatch.DeviceCaps
    @test caps.warp >= 1
    @test caps.max_threads >= caps.warp
    @test caps.max_threads == device_max_threads()
    @test caps.max_shmem > 0
    @test !isempty(caps.name)
end

@testset "release!" begin
    x = KernelDispatch.to_device(GPU_TAG, rand(Float32, 1024))
    @test KernelDispatch.release!(GPU_TAG, x) === nothing
    # The device still works after a free, and a fresh array is not the freed one.
    y = KernelDispatch.to_device(GPU_TAG, ones(Float32, 1024))
    @test Array(y) == ones(Float32, 1024)
    KernelDispatch.release!(GPU_TAG, y)
end
