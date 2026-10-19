# Kernels compiled by real Triton (test/fixtures/triton/, Triton 3.8.0, made by test/fixtures/make_triton_fixture.py),
# launched through KernelDispatch on CUDA, eagerly and from a Reactant trace, and compared with Triton's OWN output for
# the same inputs (saved by the script) and with a host formula. They exercise what a stand-in kernel cannot: the real
# metadata, dynamic shared memory (`block_sum_kernel`: `shared > 0`), the two hidden trailing pointers, arguments Triton
# dropped from the compiled signature (`strided_axpy`: `stride_x == 1`), and an output FIRST with strides between the
# pointers. The arguments each call binds are the compiled kernel's parameters, as test/cpu/triton.test.jl pins.
# Two kernels write what they read: `accumulate_kernel` updates its output (an `InOutArray`: the kernel must see the array's
# old values, eagerly and in a trace) and `atomic_sum_kernel` adds atomically into one slot of a zero-filled output.

using Test, KernelDispatch, Reactant
isdefined(@__MODULE__, :TRITON_KERNELS) || include(joinpath(@__DIR__, "..", "..", "triton_fixtures.jl"))

# The calls: the compiled signature, in order. `extent` is the grid (tile = 1).
triton_scale_call(b, x) = only(run_binary(b, x, OutArray(Float32, 100), 2.0f0, Int32(100); extent=cld(100, 64)))
triton_blocksum_call(b, x) = only(run_binary(b, x, OutArray(Float32, cld(1000, 128)), Int32(1000);
    extent=cld(1000, 128)))
triton_axpy_call(b, x, y) = only(run_binary(b, OutArray(Float32, 150; zero=true), Int32(3), x, y, Int32(2), 3.0f0,
    Int32(50); extent=cld(50, 32)))

triton_acc_call(b, y, x) = only(run_binary(b, InOutArray(y), x, 0.5f0, Int32(100); extent=cld(100, 64)))
triton_atomic_call(b, x) = only(run_binary(b, x, OutArray(Float32, 1; zero=true), Int32(1000); extent=cld(1000, 128)))

# Everything below is local to a `let`: this file is included into `Main`, where names such as `inputs` (a helper of
# sdpa_helpers.jl) and `host` (imported from KernelDispatch) already exist.
if GPU === :cuda && have_triton_fixtures()
    let
        files = Dict("scale_kernel" => ("x",), "block_sum_kernel" => ("x",), "strided_axpy" => ("x", "y"),
            "accumulate_kernel" => ("y_init", "x"), "atomic_sum_kernel" => ("x",))
        data = Dict(k.name => Tuple(load_f32(triton_paths(k.name).dir, f) for f in files[k.name]) for k in TRITON_KERNELS)
        expected_of(name) = load_f32(triton_paths(name).dir, "expected")
        call_of = Dict("scale_kernel" => triton_scale_call, "block_sum_kernel" => triton_blocksum_call,
            "strided_axpy" => triton_axpy_call, "accumulate_kernel" => triton_acc_call,
            "atomic_sum_kernel" => triton_atomic_call)
        # The host formulas, independent of Triton, on the kernel's inputs.
        formula = Dict(
            "scale_kernel" => x -> 2.0f0 .* x,
            "block_sum_kernel" => x -> Float32[sum(x[(i-1)*128+1:min(i * 128, 1000)]) for i in 1:cld(1000, 128)],
            "strided_axpy" => (x, y) -> (r = zeros(Float32, 150); r[1:3:end] = 3.0f0 .* x .+ y[1:2:end]; r),
            "accumulate_kernel" => (y, x) -> y .+ 0.5f0 .* x,
            "atomic_sum_kernel" => x -> Float32[sum(Float64, x)])
        host_ref(name) = formula[name](data[name]...)
        # Atomic adds land in any order: the sum differs from a sequential one in the last bits.
        rtol_of(name) = name == "atomic_sum_kernel" ? 1e-4 : 1e-6

        @testset "eager: $(k.name), $image" for k in TRITON_KERNELS, image in (:cubin, :ptx)
            p = triton_paths(k.name)
            b = triton_binary(p.json; image=getfield(p, image))
            args = map(DeviceArray, data[k.name])
            y = Array(call_of[k.name](b, args...))
            @test y ≈ expected_of(k.name) rtol = rtol_of(k.name) atol = 1e-6      # Triton's own output
            @test y ≈ host_ref(k.name) rtol = 1e-4 atol = 1e-5                     # the host formula
            if k.name == "accumulate_kernel"
                @test Array(first(args)) == y                # eagerly the InOutArray's array itself was written
            end
        end

        prev = Reactant.XLA.default_backend()
        Reactant.set_default_backend(traced_client())
        try
            @testset "traced: $(k.name), $image" for k in TRITON_KERNELS, image in (:cubin, :ptx)
                p = triton_paths(k.name)
                s = triton_spec(triton_binary(p.json; image=getfield(p, image)))
                args = map(a -> Reactant.to_rarray(a), data[k.name])
                call = call_of[k.name]
                f(args...) = call(triton_mkbin(s), args...)
                y = Array(@jit f(args...))
                @test y ≈ expected_of(k.name) rtol = rtol_of(k.name) atol = 1e-6
                @test y ≈ host_ref(k.name) rtol = 1e-4 atol = 1e-5
                if k.name == "accumulate_kernel"
                    # The result started from the array's values (an aliased operand, not a zero buffer) and the
                    # program is functional: the argument itself is unchanged.
                    @test Array(first(args)) == data[k.name][1]
                end
            end
        finally
            Reactant.set_default_backend(prev)
        end
    end
end
