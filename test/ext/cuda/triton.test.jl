# Kernels compiled by real Triton (test/fixtures/triton/, Triton 3.8.0, made by test/fixtures/make_triton_fixture.py),
# launched through KernelDispatch on CUDA, eagerly and from a Reactant trace, and compared with Triton's OWN output for
# the same inputs (saved by the script) and with a host formula. They exercise what a stand-in kernel cannot: the real
# metadata, dynamic shared memory (`block_sum_kernel`: `shared > 0`), the two hidden trailing pointers, arguments Triton
# dropped from the compiled signature (`strided_axpy`: `stride_x == 1`), and an output FIRST with strides between the
# pointers. The arguments each call binds are the compiled kernel's parameters, as test/cpu/triton.test.jl pins.
# Two kernels write what they read: `accumulate_kernel` updates its output (an `InOutArray`: the kernel must see the array's
# old values, eagerly and in a trace) and `atomic_sum_kernel` adds atomically into one slot of a zero-filled output.

using Test, KernelDispatch, Reactant, BFloat16s
isdefined(@__MODULE__, :TRITON_KERNELS) || include(joinpath(@__DIR__, "..", "..", "triton_fixtures.jl"))
isdefined(@__MODULE__, :ELEMENTWISE_RTOL) || include(joinpath(@__DIR__, "..", "..", "precision.jl"))

# The calls: the compiled signature, in order. `extent` is the grid (tile = 1). The output is allocated in the inputs' type;
# the scalars stay Float32 and Int32 (Triton's Python floats are fp32 whatever the tensors are).
triton_scale_call(b, x) = only(run_binary(b, x, OutArray(eltype(x), 100), 2.0f0, Int32(100); extent=cld(100, 64)))
triton_blocksum_call(b, x) = only(run_binary(b, x, OutArray(eltype(x), cld(1000, 128)), Int32(1000);
    extent=cld(1000, 128)))
triton_axpy_call(b, x, y) = only(run_binary(b, OutArray(eltype(x), 150; zero=true), Int32(3), x, y, Int32(2), 3.0f0,
    Int32(50); extent=cld(50, 32)))

triton_acc_call(b, y, x) = only(run_binary(b, InOutArray(y), x, 0.5f0, Int32(100); extent=cld(100, 64)))
triton_atomic_call(b, x) = only(run_binary(b, x, OutArray(eltype(x), 1; zero=true), Int32(1000); extent=cld(1000, 128)))

# Everything below is local to a `let`: this file is included into `Main`, where names such as `inputs` (a helper of
# sdpa_helpers.jl) and `host` (imported from KernelDispatch) already exist.
if GPU === :cuda
    # Every element type whose fixtures exist: `triton/` (Float32), `triton_float16/`, `triton_bfloat16/`.
    for T in TRITON_DTYPES
        have_triton_fixtures(T) || (@info "no Triton fixtures for $T in $(triton_root(T)) (make_triton_fixture.py $(lowercase(string(nameof(T))))): skipping"; continue)
        let
            kernels = triton_kernels(T)
            files = Dict("scale_kernel" => ("x",), "block_sum_kernel" => ("x",), "strided_axpy" => ("x", "y"),
                "accumulate_kernel" => ("y_init", "x"), "atomic_sum_kernel" => ("x",))
            data = Dict(k.name => Tuple(load_words(triton_paths(k.name, T).dir, f, T) for f in files[k.name]) for k in kernels)
            expected_of(name) = load_words(triton_paths(name, T).dir, "expected", T)
            call_of = Dict("scale_kernel" => triton_scale_call, "block_sum_kernel" => triton_blocksum_call,
                "strided_axpy" => triton_axpy_call, "accumulate_kernel" => triton_acc_call,
                "atomic_sum_kernel" => triton_atomic_call)
            # The host formulas, independent of Triton, in Float64 on the (already rounded) inputs of the kernel.
            formula = Dict(
                "scale_kernel" => x -> 2 .* Float64.(x),
                "block_sum_kernel" => x -> Float64[sum(Float64, x[(i-1)*128+1:min(i * 128, 1000)]) for i in 1:cld(1000, 128)],
                "strided_axpy" => (x, y) -> (r = zeros(150); r[1:3:end] = 3 .* Float64.(x) .+ Float64.(y[1:2:end]); r),
                "accumulate_kernel" => (y, x) -> Float64.(y) .+ 0.5 .* Float64.(x),
                "atomic_sum_kernel" => x -> Float64[sum(Float64, x)])
            host_ref(name) = formula[name](data[name]...)
            # Against Triton's own output of the same kernel (same bits, up to atomics landing in any order), and the host.
            tol_expected(name) = name == "atomic_sum_kernel" ? 1e-4 : ELEMENTWISE_RTOL[T]
            tol_host(name) = REDUCTION_RTOL[T]
            sig(k) = triton_signature(k, T)

            @testset "eager, $T: $(k.name), $image" for k in kernels, image in (:cubin, :ptx)
                p = triton_paths(k.name, T)
                b = triton_binary(p.json; image=getfield(p, image), signature=sig(k))
                @test b.signature == KD.bound_tokens(sig(k))
                args = map(DeviceArray, data[k.name])
                y = Array(call_of[k.name](b, args...))
                @test eltype(y) === T
                @test maxrelerr(y, expected_of(k.name)) <= tol_expected(k.name)    # Triton's own output
                @test maxrelerr(y, host_ref(k.name)) <= tol_host(k.name)            # the host formula
                if k.name == "accumulate_kernel"
                    @test Array(first(args)) == y                # eagerly the InOutArray's array itself was written
                end
                # a launch with the wrong element type is refused (the binary was compiled for `T`)
                if T !== Float32
                    wrong = map(a -> DeviceArray(Float32.(a)), data[k.name])
                    e = try; call_of[k.name](b, wrong...); nothing; catch e; e; end
                    @test e isa ArgumentError && occursin("compiled for", e.msg)
                end
            end

            prev = Reactant.XLA.default_backend()
            Reactant.set_default_backend(traced_client())
            try
                @testset "traced, $T: $(k.name), $image" for k in kernels, image in (:cubin, :ptx)
                    p = triton_paths(k.name, T)
                    s = triton_spec(triton_binary(p.json; image=getfield(p, image), signature=sig(k)))
                    args = map(a -> Reactant.to_rarray(a), data[k.name])
                    call = call_of[k.name]
                    f(args...) = call(triton_mkbin(s), args...)
                    y = Array(@jit f(args...))
                    @test eltype(y) === T
                    @test maxrelerr(y, expected_of(k.name)) <= tol_expected(k.name)
                    @test maxrelerr(y, host_ref(k.name)) <= tol_host(k.name)
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
end
