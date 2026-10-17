# Registration, selection, variants and the manifest: no device needed. Binaries are stand-in files,
# copied and hashed but never opened.

using Test, KernelDispatch
isdefined(@__MODULE__, :Residual) || include(joinpath(@__DIR__, "..", "fixtures.jl"))

const KO = KernelDispatch

struct Other <: AbstractKernel end
KernelDispatch.kernelname(::Other) = :other
KernelDispatch.source_tag(::Other) = "v1"
# Fall back to the largest registered size below the requested one.
function KernelDispatch.nearest(::Residual, ::Other, keys, key)
    below = [k for k in keys if k[1] == key[1] && k[2] <= key[2]]
    isempty(below) ? nothing : argmax(k -> k[2], below)
end

fake_bin(name) = (p = joinpath(mktempdir(), name * ".metallib"); write(p, rand(UInt8, 16)); p)

# A selection is a method, visible from the next top-level statement: switches and the checks that
# observe them through an op call sit in separate statements. (`current_kernel` reads the newest world.)
register_kernel!(Residual(), Other())
for k in (ResidualKA(), ResidualPlain())
    register_kernel!(Residual(), k)
end

# A kernel from elsewhere under a name the op already uses.
struct Impostor <: AbstractKernel end
KernelDispatch.kernelname(::Impostor) = :other
struct Unregistered <: AbstractKernel end
KernelDispatch.kernelname(::Unregistered) = :unregistered

@testset "kernel names are unique per op" begin
    @test register_kernel!(Residual(), Other()) === Other()   # the same kernel again: fine
    err = try register_kernel!(Residual(), Impostor()); nothing catch e; e end
    @test err isa ArgumentError && occursin("already has a kernel named `:other`", err.msg)
    @test KO._kernel(Residual(), :other) === Other()          # the original is untouched
end

@testset "selecting by kernel instance" begin
    @test use_kernel!(Residual(), ResidualPlain()) === :plain
    @test current_kernel(Residual()) === :plain
    # Sharing a registered name does not make a kernel the registered one.
    err = try use_kernel!(Residual(), Impostor()); nothing catch e; e end
    @test err isa ArgumentError && occursin("belongs to", err.msg)
    # A kernel new to the op is registered on the way.
    @test use_kernel!(Residual(), Unregistered()) === :unregistered
    @test current_kernel(Residual()) === :unregistered && haskey(KO.KERNELS[:residual], :unregistered)
    @test_throws MethodError register_kernel!(Residual(), Other(); select=true)   # `select` is gone
    reset_kernel!(Residual())
end

@testset "selection is explicit and per op" begin
    @test current_kernel(:residual) === :ka                   # the `@default_kernel`
    @test KO.default_candidates(Residual()) === (ResidualKA(), ResidualPlain())
    @test_throws ArgumentError use_kernel!(:residual, :nope)
    @test_throws ArgumentError use_kernel!(:nope, :other)
    @test use_kernel!(:residual, :other) === :other
    @test current_kernel(:residual) === :other
    @test use_kernel!(Residual, :other) === :other            # by type
    @test current_kernel(Residual()) === :other               # by instance
    @test [k.name for k in list_kernels(:residual)] ⊇ [:other]
    @test only(k.selected for k in list_kernels(:residual) if k.name === :other)
end

@test KO.selected_kernel(Residual()) === Other()              # the switch, in a new statement
@test @inferred(KO.selected_kernel(Residual())) === Other()

@testset "a switch inside one block is caught, not stale" begin
    epoch = KO.SWITCH_EPOCH[]
    use_kernel!(:residual, :other)                            # unchanged: no redefinition
    @test KO.SWITCH_EPOCH[] == epoch
    # A `let` (like a function) runs in the world it started in; a `@testset` body would not.
    stale = let
        use_kernel!(:residual, :plain)
        try
            KO.selected_kernel(Residual())
        catch e
            e
        end
    end
    @test stale isa ArgumentError && occursin("forward(op, kernel", stale.msg)
    @test current_kernel(:residual) === :plain
    @test KO.selected_kernel(Residual()) === ResidualPlain()
end

reset_kernel!(:residual)

@testset "reset_kernel! falls back to the default" begin
    @test current_kernel(:residual) === :ka
end
@test @inferred(KO.selected_kernel(Residual())) === ResidualKA()

struct NoDefault <: AbstractKernelOp end
KO.opname(::NoDefault) = :nodefault

@testset "default candidates and preferences" begin
    @test_throws ArgumentError set_default_kernel!(:residual, :other)    # not a candidate
    @test_throws ArgumentError set_default_kernel!(:residual, :plain)    # `Main`: no package, no preferences
    @test KO.default_kernel(NoDefault()) === nothing
    @test_throws ArgumentError KO.selected_kernel(NoDefault())
    @test current_kernel(NoDefault()) === nothing
end

@testset "KernelBinary tile" begin
    f = fake_bin("t")
    @test KernelBinary(f; threadgroup=64).tile == (64, 1, 1)             # one item per thread
    @test KernelBinary(f; threadgroup=1024, tile=1).tile == (1, 1, 1)    # one program per item
    @test KernelBinary(f; threadgroup=1024, tile=(32, 32)).tile == (32, 32, 1)
    @test KernelBinary(f; threadgroup=64, tile=(2, 3, 4)).tile == (2, 3, 4)
    @test_throws ArgumentError KernelBinary(f; threadgroup=64, tile=(0, 1, 1))
    @test_throws ArgumentError KernelBinary(f; threadgroup=64, tile=(1, 1, 1, 1))
end

@testset "variants and the manifest" begin
    for n in (64, 256)
        add_variant!(Residual(), Other(), (:Float32, n);
            fwd=KernelBinary(fake_bin("f$n"); threadgroup=32, params=(; tg=32, rows=n), is_ka=n == 256,
                tile=(n == 256 ? (8, 4) : 32)))
    end
    vs = KO.variants(Residual(), Other())
    @test length(vs) == 2
    @test KO.variant(Residual(), Other(), (:Float32, 64))[:fwd].params.rows == 64
    @test KO.variant(Residual(), Other(), (:Float32, 100))[:fwd].params.rows == 64     # `nearest`
    @test_throws ArgumentError KO.variant(Residual(), Other(), (:Float32, 8))           # nothing below
    @test_throws ArgumentError KO.variant(Residual(), Other(), (:Float16, 64))
    # Binaries were copied into the kernel's directory.
    @test all(b -> startswith(b.file, KO.cache_dir(Residual())), [v[:fwd] for v in values(vs)])
    # A package's own subdirectory beside the binaries survives re-registration (only files are pruned).
    own = joinpath(KO.cache_dir(Residual()), "other", "calibration")
    mkpath(own); write(joinpath(own, "x.toml"), "a = 1")
    add_variant!(Residual(), Other(), (:Float32, 64);
        fwd=KernelBinary(fake_bin("f64b"); threadgroup=32, params=(; tg=32, rows=64), tile=32))
    @test isfile(joinpath(own, "x.toml"))

    # A fresh session: the manifest reloads, params and all.
    snapshot = copy(vs)
    empty!(vs)
    register_kernel!(Residual(), Other())
    @test KO.variants(Residual(), Other()) == snapshot
    # `is_ka` and `tile` are properties of the binary and survive the round trip, set or not.
    @test KO.variant(Residual(), Other(), (:Float32, 256))[:fwd].is_ka
    @test !KO.variant(Residual(), Other(), (:Float32, 64))[:fwd].is_ka
    @test KO.variant(Residual(), Other(), (:Float32, 256))[:fwd].tile == (8, 4, 1)
    @test KO.variant(Residual(), Other(), (:Float32, 64))[:fwd].tile == (32, 1, 1)

    # A manifest from before `tile` existed loads with the default, one item per thread.
    path = joinpath(KO.cache_dir(Residual()), "other", "manifest.toml")
    write(path, join(filter(l -> !occursin(r"^\s*tile\s*=", l), readlines(path)), "\n"))
    empty!(KO.variants(Residual(), Other()))
    register_kernel!(Residual(), Other())
    @test KO.variant(Residual(), Other(), (:Float32, 256))[:fwd].tile == (32, 1, 1)

    # A manifest written from other sources is ignored.
    write(path, replace(read(path, String), "source_tag = \"v1\"" => "source_tag = \"old\""))
    empty!(KO.variants(Residual(), Other()))
    register_kernel!(Residual(), Other())
    @test isempty(KO.variants(Residual(), Other()))
end

@testset "host path" begin
    x, y = rand(Float32, 10), rand(Float32, 10)
    @test Residual()(x, y, 2.0f0) ≈ 2 .* x .+ y
end

@testset "no attention in the generic code" begin
    # KernelDispatch is op-generic: attention is ONE op, declared by its user (LuxTriangleAttention's
    # `SDPA`). Nothing here may know its roles or name it.
    root = joinpath(@__DIR__, "..", "..")
    files = [joinpath(d, f) for d in (joinpath(root, "src"), joinpath(root, "ext"), joinpath(root, "ext", "reactant"))
             if isdir(d) for f in readdir(d) if endswith(f, ".jl")]
    @test !isempty(files)
    for f in files
        @test !occursin(r"sdpa|attention|LuxTriangleAttention|:lse\b|:bias\b|:dO\b"i, read(f, String))
    end
end

# No `backward` overload: the error lists the variant's launches, one of which may be the backward.
struct BwdLess <: AbstractKernel end
KernelDispatch.kernelname(::BwdLess) = :bwdless
register_kernel!(Residual(), BwdLess())
add_variant!(Residual(), BwdLess(), (); fwd=KernelBinary(fake_bin("f"); threadgroup=32),
    bwd=KernelBinary(fake_bin("b"); threadgroup=32))

@testset "backward without an overload" begin
    x, y = rand(Float32, 4), rand(Float32, 4)
    msg(k) = try KO.backward(Residual(), k, (x, y, 1.0f0), nothing, nothing); "" catch e; e.msg end
    @test occursin("has no `backward`", msg(BwdLess()))
    @test occursin("has the launches :bwd, :fwd", msg(BwdLess()))
    @test occursin("has no `backward`", msg(Other()))
    @test !occursin("launches", msg(Other()))             # no variant here, so no hint
end

# Names default to the types'; an op declaring `@default_kernel` needs no registration.
struct Plainly <: AbstractKernelOp end
struct PlainK <: AbstractKernel end
struct OtherResidual <: AbstractKernelOp end
KernelDispatch.opname(::OtherResidual) = :residual                      # clashes with `Residual`'s name
struct DeclOp <: AbstractKernelOp end
struct DeclA <: AbstractKernel end
struct DeclB <: AbstractKernel end
KernelDispatch.@default_kernel DeclOp DeclA() DeclB()
struct LazyK <: AbstractKernel end
const LAZY_CACHE = mktempdir()
KernelDispatch.cache_dir(::Plainly) = joinpath(LAZY_CACHE, "plainly")

@testset "names and lazy registration" begin
    @test KO.opname(Plainly()) === :Plainly && KO.kernelname(PlainK()) === :PlainK
    @test KO.opname(Residual()) === :residual                      # an explicit name still wins
    @test basename(invoke(KO.cache_dir, Tuple{AbstractKernelOp}, Plainly())) == "Main.Plainly"
    err = try register_kernel!(OtherResidual(), PlainK()); nothing catch e; e end
    @test err isa ArgumentError && occursin("an op named `:residual`", err.msg)

    # Variants persisted in an earlier "session" are found without registering.
    add_variant!(Plainly(), LazyK(), (:a,); fwd=KernelBinary(fake_bin("lazy"); threadgroup=8))
    delete!(KO.VARIANTS, (Plainly, LazyK))
    @test !haskey(KO.KERNELS, :Plainly)
    @test KO.variant(Plainly(), LazyK(), (:a,))[:fwd].threadgroup == 8

    # Names resolve to declared candidates: by op name and kernel name, nothing registered.
    @test !haskey(KO.OPS, :DeclOp)
    @test current_kernel(:DeclOp) === :DeclA
    @test [k.name for k in list_kernels(:DeclOp)] == [:DeclA, :DeclB]
    @test use_kernel!(:DeclOp, :DeclB) === :DeclB
    @test current_kernel(:DeclOp) === :DeclB
    reset_kernel!(:DeclOp)
end
