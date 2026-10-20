# Registration, selection, variants and the manifest: no device needed. Binaries are stand-in files,
# copied and hashed but never opened.

using Test, KernelDispatch
isdefined(@__MODULE__, :Residual) || include(joinpath(@__DIR__, "..", "fixtures.jl"))


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
    @test KD._kernel(Residual(), :other) === Other()          # the original is untouched
end

@testset "selecting by kernel instance" begin
    @test use_kernel!(Residual(), ResidualPlain()) === :plain
    @test current_kernel(Residual()) === :plain
    # Sharing a registered name does not make a kernel the registered one.
    err = try use_kernel!(Residual(), Impostor()); nothing catch e; e end
    @test err isa ArgumentError && occursin("belongs to", err.msg)
    # A kernel new to the op is registered on the way.
    @test use_kernel!(Residual(), Unregistered()) === :unregistered
    @test current_kernel(Residual()) === :unregistered && haskey(KD.KERNELS[:residual], :unregistered)
    @test_throws MethodError register_kernel!(Residual(), Other(); select=true)   # `select` is gone
    reset_kernel!(Residual())
end

@testset "selection is explicit and per op" begin
    @test current_kernel(:residual) === :ka                   # the `@default_kernel`
    @test KD.default_candidates(Residual()) === (ResidualKA(), ResidualPlain())
    @test_throws ArgumentError use_kernel!(:residual, :nope)
    @test_throws ArgumentError use_kernel!(:nope, :other)
    @test use_kernel!(:residual, :other) === :other
    @test current_kernel(:residual) === :other
    @test use_kernel!(Residual, :other) === :other            # by type
    @test current_kernel(Residual()) === :other               # by instance
    @test [k.name for k in list_kernels(:residual)] ⊇ [:other]
    @test only(k.selected for k in list_kernels(:residual) if k.name === :other)
end

@test KD.selected_kernel(Residual()) === Other()              # the switch, in a new statement
@test @inferred(KD.selected_kernel(Residual())) === Other()

@testset "a switch inside one block is caught, not stale" begin
    epoch = KD.SWITCH_EPOCH[]
    use_kernel!(:residual, :other)                            # unchanged: no redefinition
    @test KD.SWITCH_EPOCH[] == epoch
    # A `let` (like a function) runs in the world it started in; a `@testset` body would not.
    stale = let
        use_kernel!(:residual, :plain)
        try
            KD.selected_kernel(Residual())
        catch e
            e
        end
    end
    @test stale isa ArgumentError && occursin("forward(op, kernel", stale.msg)
    @test current_kernel(:residual) === :plain
    @test KD.selected_kernel(Residual()) === ResidualPlain()
end

reset_kernel!(:residual)

@testset "reset_kernel! falls back to the default" begin
    @test current_kernel(:residual) === :ka
end
@test @inferred(KD.selected_kernel(Residual())) === ResidualKA()

struct NoDefault <: AbstractKernelOp end
KD.opname(::NoDefault) = :nodefault

@testset "default candidates and preferences" begin
    @test_throws ArgumentError set_default_kernel!(:residual, :other)    # not a candidate
    @test_throws ArgumentError set_default_kernel!(:residual, :plain)    # `Main`: no package, no preferences
    @test KD.default_kernel(NoDefault()) === nothing
    @test_throws ArgumentError KD.selected_kernel(NoDefault())
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

@testset "binary ABI and target" begin
    f = fake_bin("abi")
    b = KernelBinary(f; threadgroup=64)
    @test b.abi === :explicit && !b.is_ka && b.target == ""
    # `:plain`, the default's earlier name, is still accepted and means the same
    @test KernelBinary(f; threadgroup=64, abi=:plain).abi === :explicit && KernelBinary(f; threadgroup=64, abi=:plain) == b
    ka = KernelBinary(f; threadgroup=64, abi=:ka, target="sm_80")
    @test ka.abi === :ka && ka.is_ka && ka.target == "sm_80"
    # `is_ka` is a deprecated alias, as keyword and as property.
    @test KernelBinary(f; threadgroup=64, is_ka=true) == KernelBinary(f; threadgroup=64, abi=:ka)
    @test :is_ka in propertynames(b)
    # Field access stays inferred through the `getproperty` shim (the symbol must be a literal for that).
    is_ka_of(b) = b.is_ka; abi_of(b) = b.abi
    @test (@inferred is_ka_of(b)) isa Bool
    @test (@inferred abi_of(b)) isa Symbol

    # The two built-in ABIs are known to the core; another is an extension's, and refused until one provides it.
    x = rand(Float32, 4)
    exotic = KernelBinary(f; threadgroup=64, abi=:exotic)
    e = try KD.prepare_launch(exotic, x; extent=4); nothing catch e; e end
    @test e isa ArgumentError && occursin(":exotic", e.msg)
    KD.abi_prelude_and_grid(::Val{:exotic}, be::KD.KernelBackend, b::KernelBinary, groups) =
        (Any[UInt32(7)], (groups[1], 1, 1))
    l = @inferred KD.prepare_launch(exotic, x; extent=4)
    @test l.prelude == Any[UInt32(7)] && l.grid == (1, 1, 1)
    @test isempty(KD.prepare_launch(b, x; extent=4).prelude)                 # :explicit takes nothing
end

@testset "variants and the manifest" begin
    for n in (64, 256)
        add_variant!(Residual(), Other(), (:Float32, n);
            fwd=KernelBinary(fake_bin("f$n"); threadgroup=32, params=(; tg=32, rows=n), is_ka=n == 256,
                tile=(n == 256 ? (8, 4) : 32), target=(n == 256 ? "sm_80" : "")))
    end
    vs = KD.variants(Residual(), Other())
    @test length(vs) == 2
    @test KD.variant(Residual(), Other(), (:Float32, 64))[:fwd].params.rows == 64
    @test KD.variant(Residual(), Other(), (:Float32, 100))[:fwd].params.rows == 64     # `nearest`
    @test_throws ArgumentError KD.variant(Residual(), Other(), (:Float32, 8))           # nothing below
    @test_throws ArgumentError KD.variant(Residual(), Other(), (:Float16, 64))
    # Binaries were copied into the kernel's directory.
    @test all(b -> startswith(b.file, KD.cache_dir(Residual())), [v[:fwd] for v in values(vs)])
    # A package's own subdirectory beside the binaries survives re-registration (only files are pruned).
    own = joinpath(KD.cache_dir(Residual()), "other", "calibration")
    mkpath(own); write(joinpath(own, "x.toml"), "a = 1")
    add_variant!(Residual(), Other(), (:Float32, 64);
        fwd=KernelBinary(fake_bin("f64b"); threadgroup=32, params=(; tg=32, rows=64), tile=32))
    @test isfile(joinpath(own, "x.toml"))

    # A fresh session: the manifest reloads, params and all.
    snapshot = copy(vs)
    empty!(vs)
    register_kernel!(Residual(), Other())
    @test KD.variants(Residual(), Other()) == snapshot
    # `is_ka` and `tile` are properties of the binary and survive the round trip, set or not.
    @test KD.variant(Residual(), Other(), (:Float32, 256))[:fwd].is_ka
    @test !KD.variant(Residual(), Other(), (:Float32, 64))[:fwd].is_ka
    @test KD.variant(Residual(), Other(), (:Float32, 256))[:fwd].tile == (8, 4, 1)
    @test KD.variant(Residual(), Other(), (:Float32, 64))[:fwd].tile == (32, 1, 1)
    # So do the ABI and the target it was built for.
    @test KD.variant(Residual(), Other(), (:Float32, 256))[:fwd].abi === :ka
    @test KD.variant(Residual(), Other(), (:Float32, 64))[:fwd].abi === :explicit
    @test KD.variant(Residual(), Other(), (:Float32, 256))[:fwd].target == "sm_80"
    @test KD.variant(Residual(), Other(), (:Float32, 64))[:fwd].target == ""
    mtext = read(joinpath(KD.cache_dir(Residual()), "other", "manifest.toml"), String)
    @test occursin("format_version = 3", mtext) && occursin("abi = \"ka\"", mtext) && !occursin("is_ka", mtext)

    # A manifest from before `tile` existed loads with the default, one item per thread.
    path = joinpath(KD.cache_dir(Residual()), "other", "manifest.toml")
    write(path, join(filter(l -> !occursin(r"^\s*tile\s*=", l), readlines(path)), "\n"))
    empty!(KD.variants(Residual(), Other()))
    register_kernel!(Residual(), Other())
    @test KD.variant(Residual(), Other(), (:Float32, 256))[:fwd].tile == (32, 1, 1)

    # A version-1 manifest (`is_ka` instead of `abi`, no `target`, no `format_version`) loads too.
    old = filter(l -> !occursin(r"^\s*(target|format_version)\s*=", l), readlines(path))
    old = [replace(l, "abi = \"ka\"" => "is_ka = true", "abi = \"explicit\"" => "is_ka = false") for l in old]
    write(path, join(old, "\n"))
    empty!(KD.variants(Residual(), Other()))
    register_kernel!(Residual(), Other())
    @test KD.variant(Residual(), Other(), (:Float32, 256))[:fwd].abi === :ka
    @test KD.variant(Residual(), Other(), (:Float32, 64))[:fwd].abi === :explicit
    @test KD.variant(Residual(), Other(), (:Float32, 64))[:fwd].target == ""

    # A version-2 manifest written while the default ABI was still called `plain` loads as `:explicit`.
    v2 = [replace(l, "is_ka = false" => "abi = \"plain\"", "is_ka = true" => "abi = \"ka\"") for l in old]
    write(path, join(v2, "\n"))
    empty!(KD.variants(Residual(), Other()))
    register_kernel!(Residual(), Other())
    @test KD.variant(Residual(), Other(), (:Float32, 64))[:fwd].abi === :explicit
    @test KD.variant(Residual(), Other(), (:Float32, 256))[:fwd].abi === :ka

    # A manifest written from other sources is ignored.
    write(path, replace(read(path, String), "source_tag = \"v1\"" => "source_tag = \"old\""))
    empty!(KD.variants(Residual(), Other()))
    register_kernel!(Residual(), Other())
    @test isempty(KD.variants(Residual(), Other()))
end

@testset "per-target directories" begin
    @test KD.current_target() == ""                       # no device backend here
    r, o = Residual(), Other()
    @test KD.kernel_dir(r, o) == KD.kernel_dir(r, o; target="") == joinpath(KD.cache_dir(r), "other")
    @test KD.kernel_dir(r, o; target="sm_80") == joinpath(KD.cache_dir(r), "sm_80", "other")
    @test KD.ka_binary_cache_dir() == joinpath(KD.cache_root(), "ka")
    @test KD.ka_binary_cache_dir("sm_80") == joinpath(KD.cache_root(), "ka", "sm_80")

    # Variants for two architectures live apart: separate tables, directories and manifests.
    key = (:Float32, 4096)
    b80 = KernelBinary(fake_bin("t80"); threadgroup=32, target="sm_80")
    b90 = KernelBinary(fake_bin("t90"); threadgroup=32, target="sm_90")
    KD._add_variant!(r, o, key, (; fwd=b80), "sm_80")
    KD._add_variant!(r, o, key, (; fwd=b90), "sm_90")
    v80, v90 = KD.variant(r, o, key; target="sm_80"), KD.variant(r, o, key; target="sm_90")
    @test v80[:fwd].target == "sm_80" && v90[:fwd].target == "sm_90"
    @test dirname(v80[:fwd].file) == KD.kernel_dir(r, o; target="sm_80")
    @test dirname(v90[:fwd].file) == KD.kernel_dir(r, o; target="sm_90")
    @test isfile(KD.manifest_path(r, o; target="sm_80")) && isfile(KD.manifest_path(r, o; target="sm_90"))
    @test !haskey(KD.variants(r, o), key)                  # the default target is untouched

    # And reload separately from their manifests.
    empty!(KD.variants(r, o; target="sm_80")); empty!(KD.variants(r, o; target="sm_90"))
    @test KD.load_manifest(r, o; target="sm_80")[key][:fwd].file == v80[:fwd].file
    @test KD.load_manifest(r, o; target="sm_90")[key][:fwd].file == v90[:fwd].file

    # Re-registering under one target prunes only that target's files.
    KD._add_variant!(r, o, key, (; fwd=KernelBinary(fake_bin("t80b"); threadgroup=32, target="sm_80")), "sm_80")
    @test isfile(v90[:fwd].file)
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
    msg(k) = try KD.backward(Residual(), k, (x, y, 1.0f0), nothing, nothing); "" catch e; e.msg end
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
    @test KD.opname(Plainly()) === :Plainly && KD.kernelname(PlainK()) === :PlainK
    @test KD.opname(Residual()) === :residual                      # an explicit name still wins
    @test basename(invoke(KD.cache_dir, Tuple{AbstractKernelOp}, Plainly())) == "Main.Plainly"
    err = try register_kernel!(OtherResidual(), PlainK()); nothing catch e; e end
    @test err isa ArgumentError && occursin("an op named `:residual`", err.msg)

    # Variants persisted in an earlier "session" are found without registering.
    add_variant!(Plainly(), LazyK(), (:a,); fwd=KernelBinary(fake_bin("lazy"); threadgroup=8))
    delete!(KD.VARIANTS, (Plainly, LazyK, ""))
    @test !haskey(KD.KERNELS, :Plainly)
    @test KD.variant(Plainly(), LazyK(), (:a,))[:fwd].threadgroup == 8

    # Names resolve to declared candidates: by op name and kernel name, nothing registered.
    @test !haskey(KD.OPS, :DeclOp)
    @test current_kernel(:DeclOp) === :DeclA
    @test [k.name for k in list_kernels(:DeclOp)] == [:DeclA, :DeclB]
    @test use_kernel!(:DeclOp, :DeclB) === :DeclB
    @test current_kernel(:DeclOp) === :DeclB
    reset_kernel!(:DeclOp)
end
