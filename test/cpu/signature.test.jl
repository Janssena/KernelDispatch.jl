# The type check on a launch: a binary's `signature` (the types it was compiled for) against the arguments `call_binary`
# binds. Needs no device: a fake file stands in for the binary and a test-only array type says it lives on a CUDA device,
# so `prepare_launch` takes its device branch (the pattern of binary_checks.test.jl). The GPU half, where a mismatched
# binary really runs and computes the wrong thing, is gpu/mismatch.test.jl.

using Test, KernelDispatch, BFloat16s
isdefined(@__MODULE__, :ALL_ELTYPES) || include(joinpath(@__DIR__, "..", "precision.jl"))

# A fresh file per binary: the unknown-signature warning is once per (file, entry).
sig_file(ext=".cubin") = (p = joinpath(mktempdir(), "k" * ext); write(p, UInt8[1, 2, 3]); p)

struct DevArray{T} <: AbstractVector{T}; n::Int; end
Base.size(a::DevArray) = (a.n,)
Base.getindex(::DevArray{T}, i::Int) where {T} = zero(T)
KD.backend_of(::DevArray) = KD.CUDABackendTag()

launch(b, args...; kw...) = KD.prepare_launch(b, args...; extent=4, kw...)
refusal(f) = try; f(); nothing; catch e; e; end

@testset "tokens and declarations" begin
    @test KD.bound_tokens((Vector{Float16}, Vector{Float16}, Float32, Int32, Val{64})) == ("f16", "f16", "s:f32", "s:i32")
    @test KD.bound_tokens((AbstractArray{BFloat16}, UInt32, Bool)) == ("bf16", "s:u32", "s:bool")
    @test KD.bound_tokens(()) == ()
    @test KD._as_signature(nothing) === nothing
    @test KD._as_signature(false) === false
    @test KD._as_signature((Vector{Float32}, Float32)) == ("f32", "s:f32")
    @test KD._as_signature(("f32", "s:f32")) == ("f32", "s:f32")
    @test KD._as_signature(()) == ()
    @test_throws ArgumentError KD._as_signature(true)
    @test_throws ArgumentError KD._as_signature((Vector{Float32}, "f32"))
    @test_throws ArgumentError KD._as_signature(:f32)
    b = KernelBinary(sig_file(); threadgroup=32, signature=(Vector{Float16}, Int32, Val{8}))
    @test b.signature == ("f16", "s:i32")                                # a Val takes no slot
    @test KernelBinary(b; signature=false).signature === false
    @test KernelBinary(b; threadgroup=64).signature == b.signature && KernelBinary(b).threadgroup == 32
    @test KernelBinary(sig_file(); threadgroup=32).signature === nothing
end

@testset "a matching launch passes, $T" for T in ALL_ELTYPES
    other = first(filter(!=(T), ALL_ELTYPES))
    sig = (Vector{T}, Vector{T}, Float32, Int32)
    b = KernelBinary(sig_file(); threadgroup=32, signature=sig)
    x, y = DevArray{T}(4), DevArray{T}(4)

    @test (@inferred launch(b, x, OutArray(T, 4), 2.0f0, Int32(4))) isa KD.Launch
    @test launch(b, x, InOutArray(y), 2.0f0, Int32(4)) isa KD.Launch                      # an InOutArray is its array
    @test launch(b, x, OutArray(T, 4), 2.0f0; extras=(Int32(4), Val(64))) isa KD.Launch   # extras count, a Val does not
    @test launch(b, x, Val(64), OutArray(T, 4), 2.0f0, Int32(4)) isa KD.Launch            # nor does one among the arguments

    @testset "refused: $what" for (what, args, text) in (
        ("a buffer of another element type", (DevArray{other}(4), OutArray(T, 4), 2.0f0, Int32(4)), "argument 1"),
        ("an output of another element type", (x, OutArray(other, 4), 2.0f0, Int32(4)), "argument 2"),
        ("an in-out array of another element type", (x, InOutArray(DevArray{other}(4)), 2.0f0, Int32(4)), "argument 2"),
        ("an Int64 for an Int32", (x, OutArray(T, 4), 2.0f0, 4), "argument 4"),
        ("a Float64 for a Float32", (x, OutArray(T, 4), 2.0, Int32(4)), "argument 3"),
        ("a scalar for a buffer", (x, 2.0f0, 2.0f0, Int32(4)), "argument 2"),
        ("a buffer for a scalar", (x, OutArray(T, 4), x, Int32(4)), "argument 3"),
        ("a missing argument", (x, OutArray(T, 4), 2.0f0), "takes 4 arguments"),
        ("an extra argument", (x, OutArray(T, 4), 2.0f0, Int32(4), Int32(1)), "takes 4 arguments"),
    )
        e = refusal(() -> launch(b, args...))
        @test e isa ArgumentError
        e isa ArgumentError && (@test occursin(text, e.msg) && occursin("check_types=false", e.msg))
        @test launch(b, args...; check_types=false) isa KD.Launch         # the way out
    end

    e = refusal(() -> launch(b, DevArray{other}(4), OutArray(T, 4), 2.0f0, Int32(4)))
    @test occursin(KD._type_token(T), e.msg) && occursin(KD._type_token(other), e.msg)    # both types are named
end

module SigModA; struct Params; x::Float32; end; end      # two types with one name
module SigModB; struct Params; x::Float32; end; end
module SigModC; struct Float32; end; end                  # a user's own `Float32`
struct Pair2{A,B}; a::A; b::B; end

@testset "tokens are lossless for types outside the table" begin
    tok = KD._type_token
    @test tok(Float32) == "f32" && tok(BFloat16) == "bf16" && tok(Core.BFloat16) == "bf16" && tok(Bool) == "bool"
    # parametric types differ by their parameters
    @test tok(ComplexF32) != tok(ComplexF64)
    @test tok(Pair2{Float32,Int32}) != tok(Pair2{Float32,Int64})
    @test occursin("Complex", tok(ComplexF32)) && occursin("f32", tok(ComplexF32))
    # two modules' types of one name differ
    @test tok(SigModA.Params) != tok(SigModB.Params)
    @test occursin("SigModA", tok(SigModA.Params))
    # a user's own type called `Float32` is not the number
    @test tok(SigModC.Float32) != "f32" && occursin("SigModC", tok(SigModC.Float32))
    @test KD.bound_tokens((Vector{ComplexF32}, ComplexF32)) == (tok(ComplexF32), "s:" * tok(ComplexF32))

    # the guard catches what a name would not
    for (A, B) in ((ComplexF32, ComplexF64), (SigModA.Params, SigModB.Params), (Pair2{Float32,Int32}, Pair2{Float32,Int64}))
        b = KernelBinary(sig_file(); threadgroup=32, signature=(Vector{A}, Vector{A}))
        @test launch(b, DevArray{A}(4), OutArray(A, 4)) isa KD.Launch
        e = refusal(() -> launch(b, DevArray{B}(4), OutArray(A, 4)))
        @test e isa ArgumentError && occursin("argument 1", e.msg)
    end
    # and so do the file names of builds of such types
    @test KD.ka_binary_name("k", (Vector{ComplexF32},)) != KD.ka_binary_name("k", (Vector{ComplexF64},))
    @test KD.ka_binary_name("k", (Vector{SigModA.Params},)) != KD.ka_binary_name("k", (Vector{SigModB.Params},))
    @test KD.ka_binary_name("k", (Vector{ComplexF32},)) == KD.ka_binary_name("k", (Vector{ComplexF32},))   # and are stable
    @test KD.ka_binary_name("k", (Vector{Float32}, Float32)) == "k_f32_f32s"                              # numeric ones stay short
    # a signature survives the manifest with such tokens in it (commas, braces, dots)
    sig = KD.bound_tokens((Vector{Pair2{Float32,Int32}}, Float32))
    @test KD._loaded_signature(collect(sig)) == sig
end

@testset "buffers and scalars of one type are different arguments" begin
    b = KernelBinary(sig_file(); threadgroup=32, signature=(Vector{Float32}, Float32))
    @test launch(b, DevArray{Float32}(4), 2.0f0) isa KD.Launch
    @test refusal(() -> launch(b, DevArray{Float32}(4), DevArray{Float32}(4))) isa ArgumentError
end

@testset "unknown and unchecked signatures" begin
    x, o = DevArray{Float16}(4), OutArray(Float16, 4)
    # Unknown, on a device: warns once per binary, with the signature that fits this launch.
    b = KernelBinary(sig_file(); entry="scale", threadgroup=32)
    @test_logs (:warn, r"`scale`.*no recorded signature.*signature=\(Vector\{Float16\}, Vector\{Float16\}, Float32, Int32\)") launch(b, x, o, 2.0f0, Int32(4))
    @test_logs launch(b, x, o, 2.0f0, Int32(4))                                       # not again
    @test_logs launch(b, x, o, 2.0f0, Int32(4), Val(64))
    # A `Val` is not part of the suggestion, and a one-argument tuple is written as a tuple.
    b1 = KernelBinary(sig_file(); entry="one", threadgroup=32)
    @test_logs (:warn, r"signature=\(Vector\{Float16\},\)") launch(b1, x, Val(8))
    # Silent: declared unchecked, `check_types=false`, host arrays (no device), a signature.
    @test_logs launch(KernelBinary(sig_file(); threadgroup=32, signature=false), x, o, 2.0f0, Int32(4))
    @test_logs launch(KernelBinary(sig_file(); threadgroup=32), x, o, 2.0f0, Int32(4); check_types=false)
    @test_logs KD.prepare_launch(KernelBinary(sig_file(); threadgroup=32), rand(Float32, 4); extent=4)
    @test_logs launch(KernelBinary(sig_file(); threadgroup=32, signature=(Vector{Float16}, Vector{Float16}, Float32, Int32)),
        x, o, 2.0f0, Int32(4))
    # A declared signature is checked on the host's fake device too, and the suggestion pasted back is accepted.
    sugg = KernelBinary(sig_file(); threadgroup=32, signature=(Vector{Float16}, Vector{Float16}, Float32, Int32))
    @test launch(sugg, x, o, 2.0f0, Int32(4)) isa KD.Launch
end

struct SigOp <: AbstractKernelOp end
struct SigKernel <: AbstractKernel end
const SIG_DIR = mktempdir()
KD.cache_dir(::SigOp) = SIG_DIR

@testset "the signature persists with the registry" begin
    sigs = Dict(:checked => (Vector{Float16}, Float32), :unchecked => false, :unknown => nothing)
    for (name, sig) in sigs
        add_variant!(SigOp(), SigKernel(), (name,); fwd=KernelBinary(sig_file(); threadgroup=32, signature=sig))
    end
    vs = KD.variants(SigOp(), SigKernel())
    loaded = KD.load_manifest(SigOp(), SigKernel())
    @test loaded == vs                                                    # equal after a round trip
    @test loaded[(:checked,)][:fwd].signature == ("f16", "s:f32")
    @test loaded[(:unchecked,)][:fwd].signature === false
    @test loaded[(:unknown,)][:fwd].signature === nothing
    path = KD.manifest_path(SigOp(), SigKernel())
    @test occursin("format_version = 3", read(path, String))
    # A manifest from before signatures (no key) loads as unknown.
    write(path, join(filter(l -> !occursin("signature", l), readlines(path)), "\n"))
    old = KD.load_manifest(SigOp(), SigKernel())
    @test all(v -> v[:fwd].signature === nothing, values(old)) && length(old) == 3
end
