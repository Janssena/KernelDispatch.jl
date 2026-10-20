# What a launch with the wrong types does, and what stops it. A binary takes raw pointers and bits scalars, so one compiled
# for `float*` and handed `Float16` data runs happily and computes something else. For each mismatch:
#
#   hazard   with `check_types=false` the launch goes through and the result is WRONG. This pins down why the check exists.
#   guard    by default the launch is refused with an `ArgumentError` naming the argument, and nothing runs: the output
#            array, given as an `InOutArray` so it can be inspected, is untouched.
#
# The hazards are the ones that cannot write out of bounds: the buffers are sized for the wider of the two types. (A wider
# kernel on a narrower output buffer would write past its end; that is documented, not tested.) The kernel is the
# fixtures' `logit_scale_kernel!` (y[i] = a * x[i]) built by `ka_compile` for each element type.

using Test, Random, KernelDispatch, Reactant, BFloat16s
isdefined(@__MODULE__, :LS_TG) || include(joinpath(@__DIR__, "..", "fixtures.jl"))
isdefined(@__MODULE__, :ELEMENTWISE_RTOL) || include(joinpath(@__DIR__, "..", "precision.jl"))

const MM_TG = 64
mm_binary(T) = ka_compile(GPU_TAG, logit_scale_kernel!, (Vector{T}, Vector{T}, Float32, Int32, Val{MM_TG});
    tg=MM_TG, name="mismatch_scale", dir=mktempdir())

refusal(f) = try; f(); nothing; catch e; e; end

# y[i] = a * x[i] for i <= n: `x` and `y` are device arrays (whatever their element type), `y` filled with a sentinel.
function mm_launch(b, x, y, a, n; kw...)
    run_binary(b, InOutArray(y), x, a, n, Val(MM_TG); extent=Int(n), kw...)
    return Array(y)
end

@testset "mismatched types: the hazard, and the guard" begin
    rng = Xoshiro(31)
    n = 64
    types = collect(supported_eltypes())
    bins = Dict(T => mm_binary(T) for T in types)

    @testset "a $B buffer for a kernel built for $A" for A in types, B in types
        A === B && continue
        # sized for the wider type, so even the kernel built for the wider one stays in bounds
        len = n * max(1, sizeof(A) ÷ sizeof(B))
        x = rounded(B, randn(rng, Float32, len))
        expected = B.(2.0f0 .* Float32.(x[1:n]))
        sentinel = B(7)

        # hazard: it runs, and the answer is not 2x
        y = DeviceArray(fill(sentinel, len))
        got = mm_launch(bins[A], DeviceArray(x), y, 2.0f0, Int32(n); check_types=false)
        @test got[1:n] != expected

        # guard: refused, naming the argument and both types, and nothing was written
        y = DeviceArray(fill(sentinel, len))
        e = refusal(() -> mm_launch(bins[A], DeviceArray(x), y, 2.0f0, Int32(n)))
        @test e isa ArgumentError && occursin("argument 1", e.msg) && occursin("check_types=false", e.msg)
        @test occursin(KD._type_token(A), e.msg) && occursin(KD._type_token(B), e.msg)
        @test Array(y) == fill(sentinel, len)
    end

    @testset "a $A kernel's scalar of the wrong type" for A in types
        b = bins[A]
        x = rounded(A, randn(rng, Float32, n))
        expected = A.(3.0f0 .* Float32.(x))

        # hazard: an Int32 3 read as a Float32 is a denormal, not 3
        y = DeviceArray(fill(A(7), n))
        got = mm_launch(b, DeviceArray(x), y, Int32(3), Int32(n); check_types=false)
        @test maxrelerr(got, expected) > ELEMENTWISE_RTOL[A]

        # guard: Int32 for the Float32, Float64 for the Float32, Int64 for the Int32 (this last one happens to work on
        # little-endian paths, which is exactly why nothing else would have caught it)
        for (what, a, n_, text) in (("an Int32 for a Float32", Int32(3), Int32(n), "argument 3"),
                                    ("a Float64 for a Float32", 3.0, Int32(n), "argument 3"),
                                    ("an Int64 for an Int32", 3.0f0, n, "argument 4"))
            y = DeviceArray(fill(A(7), n))
            e = refusal(() -> mm_launch(b, DeviceArray(x), y, a, n_))
            @test e isa ArgumentError && occursin(text, e.msg)
            @test Array(y) == fill(A(7), n)
        end
    end

    @testset "right types are not refused" begin
        for A in types
            x = rounded(A, randn(rng, Float32, n))
            y = DeviceArray(fill(A(7), n))
            got = mm_launch(bins[A], DeviceArray(x), y, 2.0f0, Int32(n))
            @test maxrelerr(got, 2.0 .* Float64.(x)) <= ELEMENTWISE_RTOL[A]
        end
    end

    # The same refusal when traced: at trace time, before any custom call is built.
    client = traced_client()
    if client === nothing
        @info "no traced client for $GPU: skipping the traced mismatch guard"
    else
        prev = Reactant.XLA.default_backend()
        Reactant.set_default_backend(client)
        try
            @testset "traced: a $B buffer for a kernel built for $A" for A in types, B in types
                A === B && continue
                b = bins[A]
                # built inside the trace (a captured `KernelBinary` is not traceable), with the signature it was compiled with
                file, entry, sig = b.file, b.entry, b.signature
                layout = get(b.params, :ka_layout, "")
                f(x) = only(run_binary(KernelBinary(file; entry, threadgroup=MM_TG, abi=:ka, params=(; ka_layout=layout),
                        signature=sig), OutArray(eltype(x), n), x, 2.0f0, Int32(n), Val(MM_TG); extent=n))
                x = rounded(B, randn(rng, Float32, n))
                e = refusal(() -> Reactant.@jit f(Reactant.to_rarray(x)))
                @test e !== nothing && occursin("compiled for", sprint(showerror, e))
            end
        finally
            Reactant.set_default_backend(prev)
        end
    end
end
