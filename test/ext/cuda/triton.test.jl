# Kernels compiled by real Triton (test/fixtures/triton/, Triton 3.8.0, made by test/fixtures/make_triton_fixture.py),
# launched through KernelDispatch on CUDA, eagerly and from a Reactant trace, and compared with Triton's OWN output for
# the same inputs (saved by the script) and with a host formula. They exercise what a stand-in kernel cannot: the real
# metadata, dynamic shared memory (`block_sum_kernel`: `shared > 0`), the two hidden trailing pointers, arguments Triton
# dropped from the compiled signature (`strided_axpy`: `stride_x == 1`), and an output FIRST with strides between the
# pointers. The arguments each call binds are the compiled kernel's parameters, as test/cpu/triton.test.jl pins.

using Test, KernelDispatch, Reactant
isdefined(@__MODULE__, :TRITON_KERNELS) || include(joinpath(@__DIR__, "..", "..", "triton_fixtures.jl"))

# One binary from plain values: built INSIDE a traced function (a captured `KernelBinary` is not traceable).
mkbin(s) = KernelBinary(s.file; entry=s.entry, threadgroup=s.threadgroup, tile=1, abi=:triton, target=s.target,
    shared_mem=s.shared, params=(; scratch_pointers=s.nptr))
spec(b) = (; file=b.file, entry=b.entry, threadgroup=b.threadgroup, target=b.target, shared=b.shared_mem,
    nptr=b.params.scratch_pointers)

# The calls: the compiled signature, in order. `extent` is the grid (tile = 1).
scale_call(b, x) = only(run_binary(b, x, OutArray(Float32, 100), 2.0f0, Int32(100); extent=cld(100, 64)))
blocksum_call(b, x) = only(run_binary(b, x, OutArray(Float32, cld(1000, 128)), Int32(1000); extent=cld(1000, 128)))
axpy_call(b, x, y) = only(run_binary(b, OutArray(Float32, 150; zero=true), Int32(3), x, y, Int32(2), 3.0f0,
    Int32(50); extent=cld(50, 32)))

if GPU === :cuda && have_triton_fixtures()
    inputs = Dict(
        "scale_kernel" => (load_f32(triton_paths("scale_kernel").dir, "x"),),
        "block_sum_kernel" => (load_f32(triton_paths("block_sum_kernel").dir, "x"),),
        "strided_axpy" => (load_f32(triton_paths("strided_axpy").dir, "x"), load_f32(triton_paths("strided_axpy").dir, "y")))
    expected(name) = load_f32(triton_paths(name).dir, "expected")
    calls = Dict("scale_kernel" => scale_call, "block_sum_kernel" => blocksum_call, "strided_axpy" => axpy_call)
    # The host formulas, independent of Triton.
    x1, = inputs["scale_kernel"]; xs, = inputs["block_sum_kernel"]; xa, ya = inputs["strided_axpy"]
    host = Dict("scale_kernel" => 2.0f0 .* x1,
        "block_sum_kernel" => Float32[sum(xs[(i-1)*128+1:min(i * 128, 1000)]) for i in 1:cld(1000, 128)],
        "strided_axpy" => (r = zeros(Float32, 150); r[1:3:end] = 3.0f0 .* xa .+ ya[1:2:end]; r))

    @testset "eager: $(k.name), $image" for k in TRITON_KERNELS, image in (:cubin, :ptx)
        p = triton_paths(k.name)
        b = triton_binary(p.json; image=getfield(p, image))
        args = map(DeviceArray, inputs[k.name])
        y = Array(calls[k.name](b, args...))
        @test y ≈ expected(k.name) rtol = 1e-6 atol = 1e-6          # Triton's own output
        @test y ≈ host[k.name] rtol = 1e-4 atol = 1e-5               # the host formula
    end

    prev = Reactant.XLA.default_backend()
    Reactant.set_default_backend("gpu")
    try
        @testset "traced: $(k.name), $image" for k in TRITON_KERNELS, image in (:cubin, :ptx)
            p = triton_paths(k.name)
            s = spec(triton_binary(p.json; image=getfield(p, image)))
            args = map(a -> Reactant.to_rarray(a), inputs[k.name])
            call = calls[k.name]
            f(args...) = call(mkbin(s), args...)
            y = Array(@jit f(args...))
            @test y ≈ expected(k.name) rtol = 1e-6 atol = 1e-6
            @test y ≈ host[k.name] rtol = 1e-4 atol = 1e-5
        end
    finally
        Reactant.set_default_backend(prev)
    end
end
