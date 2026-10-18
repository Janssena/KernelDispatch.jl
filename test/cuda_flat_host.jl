# The CUDA backend's flat KernelAbstractions layout, as far as it can be checked without a GPU: the hooks, the six
# integers of KA's launch context, and the guard against binaries built before the layout was flat. Needs CUDA.jl
# to be loadable (it is in the test environment), not a device. Run in its own process by test/cpu/cuda_flat.test.jl,
# because loading CUDA.jl here would change what the CPU section's other tests see ("no GPU extension loaded").

using Test, KernelDispatch, KernelAbstractions, CUDA
const KO = KernelDispatch
const EXT = Base.get_extension(KO, :KernelDispatchCUDAExt)
const TAG = KO.CUDABackendTag()

# The six `Int`s a flat context consists of, read from its bits.
bits(c) = (r = Ref(c); GC.@preserve r unsafe_load(Ptr{NTuple{6,Int}}(Base.unsafe_convert(Ptr{Cvoid}, r))))

@testset "flat KA layout (no GPU needed)" begin
    @test EXT !== nothing
    @test KO.ka_state_words(TAG) === ()                      # CUDA.jl's KernelState is not passed
    @test KO.ka_layout(TAG) === :flat
    @test KO.ka_layout(KO.MetalBackendTag()) === nothing

    # The context a binary is launched with has six integers, equal to what the device side rebuilds from them.
    for groups in ((2, 1, 1), (3, 2, 1), (1, 1, 4)), tg in (32, 64, 128)
        ctx, ghost = KO.ka_context(TAG, groups, tg)
        @test !ghost
        ints = (groups[1] * tg, groups[2], groups[3], groups...)
        rebuilt = EXT._rebuild_ctx(Val(tg), ints...)
        @test typeof(rebuilt) === typeof(ctx)
        @test bits(rebuilt) == bits(ctx) == ints
        @test KO.ka_prelude(TAG, ctx, false) == Any[ints...]
    end
    @test_throws ArgumentError KO.ka_prelude(TAG, nothing, true)

    # A KA binary built before the layout was flat is refused with the reason, whatever its format or target.
    file = tempname() * ".cubin"
    write(file, UInt8[1, 2, 3])
    stale = KernelBinary(file; entry="k", threadgroup=64, abi=:ka)
    reason = KO.compatible_target(TAG, stale)
    @test reason isa String && occursin("older KernelDispatch", reason)
    @test_throws ArgumentError KO.check_binary(TAG, stale)
    @test KO.compatible_target(TAG, KernelBinary(file; entry="k", threadgroup=64, abi=:ka, params=(; ka_layout="flat"))) === nothing
    @test KO.compatible_target(TAG, KernelBinary(file; entry="k", threadgroup=64)) === nothing     # plain: not a KA layout
end
