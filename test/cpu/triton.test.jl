# Triton-compiled kernels as binaries, without a GPU and without Python. Two halves:
#
#   - what needs no Triton at all: the JSON reader (generic JSON, no claim about Triton's schema), the `shared_mem`
#     field, and the `:triton` ABI's prelude and grid;
#   - what is checked against kernels REAL Triton compiled (test/fixtures/triton/, Triton 3.8.0 on an NVIDIA A40, made by
#     test/fixtures/make_triton_fixture.py and committed): `triton_binary` on the real metadata, the compiled signature
#     against what a `forward` binds, the refusals (made by editing the real metadata), and the manifest round trip. This
#     half is skipped, with a warning, when the fixtures are absent.
#
# The launches themselves are ext/cuda/triton.test.jl (eager and traced, against Triton's own output).

using Test, KernelDispatch, TOML
const KT = KernelDispatch
isdefined(@__MODULE__, :TRITON_KERNELS) || include(joinpath(@__DIR__, "..", "triton_fixtures.jl"))

struct TritonOp <: AbstractKernelOp end
struct TritonK <: AbstractKernel end
const TRITON_ROOT = mktempdir()
KT.cache_dir(::TritonOp) = joinpath(TRITON_ROOT, "triton_op")

err(f) = try f(); nothing catch e; e end

# --- no Triton needed --------------------------------------------------------------------------------------------

@testset "JSON reader" begin
    doc = """{"name": "k", "n": 4, "x": -1.5e-3, "yes": true, "no": false, "none": null,
             "list": [1, [2, 3], {"a": []}], "nested": {"arch": 86}, "esc": "a\\"b\\n\\u0041\\\\"}"""
    j = KT._json_parse(doc)
    @test j["name"] == "k" && j["n"] == 4 && j["x"] ≈ -1.5e-3
    @test j["yes"] === true && j["no"] === false && j["none"] === nothing
    @test j["list"] == [1, [2, 3], Dict("a" => []) ] && j["nested"]["arch"] == 86
    @test j["esc"] == "a\"b\nA\\"
    @test KT._json_parse("[]") == [] && KT._json_parse(" {} ") == Dict{String,Any}() && KT._json_parse("-3") == -3
    for bad in ("", "{", "{\"a\":}", "[1,]x", "{\"a\" 1}", "tru", "1 2", "\"open")
        @test_throws Exception KT._json_parse(bad)
    end
end

@testset "shared_mem" begin
    f = tempname() * ".cubin"; write(f, UInt8[1])
    @test KernelBinary(f; threadgroup=32).shared_mem == 0
    @test KernelBinary(f; threadgroup=32, shared_mem=4096).shared_mem == 4096
    @test_throws ArgumentError KernelBinary(f; threadgroup=32, shared_mem=-1)
end

@testset "the :triton ABI" begin
    f = tempname() * ".cubin"; write(f, UInt8[1])
    b = KernelBinary(f; entry="k", threadgroup=64, tile=1, abi=:triton, params=(; scratch_pointers=2))   # tile 1: extent is the grid
    # Nothing ahead of the arguments, and the grid as given: the trailing pointers are the launcher's.
    prelude, grid = KT.abi_prelude_and_grid(Val(:triton), KT.CUDABackendTag(), b, (3, 2, 1))
    @test isempty(prelude) && grid == (3, 2, 1)
    l = KT.prepare_launch(b, rand(Float32, 4); extent=(3, 2), extras=(Int32(4),))
    @test l.groups == (3, 2, 1) && l.grid == (3, 2, 1) && isempty(l.prelude)
    @test KT.trailing_pointers(b) == 2
    @test KT.trailing_pointers(KernelBinary(f; threadgroup=32)) == 0          # only a `:triton` binary has them
end

# --- against what real Triton compiled -----------------------------------------------------------------------------

if !have_triton_fixtures()
    @warn "no Triton fixtures in $TRITON_FIXTURES (run test/fixtures/make_triton_fixture.py on an NVIDIA machine): " *
          "skipping the real-Triton tests"
else
    @testset "triton_binary and the compiled signature: $(k.name)" for k in TRITON_KERNELS
        p = triton_paths(k.name)
        meta = KT._json_parse(read(p.json, String))
        kinds = ptx_param_kinds(read(p.ptx, String))

        for image in (:cubin, :ptx)
            b = triton_binary(p.json; image=getfield(p, image))
            @test b.abi === :triton && b.entry == k.name && b.tile == (1, 1, 1)    # tile 1: `extent` is the grid
            @test b.threadgroup == 32 * k.num_warps == 32 * meta["num_warps"]
            @test b.shared_mem == meta["shared"]
            @test KT.binary_format(b) === image
            @test b.target == (image === :cubin ? "sm_" * string(meta["target"]["arch"]) : "")   # PTX is portable
            @test b.params.triton_version == meta["triton_version"]
        end

        # The hidden trailing pointers: the metadata has both scratch entries, and the compiled kernel ends in exactly
        # that many pointer parameters after the arguments Triton kept.
        b = triton_binary(p.json)
        nptr = KT.trailing_pointers(b)
        @test nptr == b.params.scratch_pointers == 2
        @test length(kinds) == length(k.args) + nptr
        @test kinds[1:length(k.args)] == k.args                    # what `forward` binds, in this order
        @test all(==(:ptr), kinds[end-nptr+1:end])                 # what the launcher appends
    end

    @testset "overrides and the choice of image" begin
        p = triton_paths("scale_kernel")
        @test endswith(triton_binary(p.json).file, ".cubin")                       # the cubin beside the JSON, by default
        @test endswith(triton_binary(p.json; image=p.ptx).file, ".ptx")
        @test triton_binary(p.json; tile=(8, 1)).tile == (8, 1, 1)
        @test KT.trailing_pointers(triton_binary(p.json; scratch_pointers=0)) == 0
        @test triton_binary(p.json; params=(; mine=1)).params.mine == 1
        # With only the PTX beside the JSON, the PTX is used.
        dir = mktempdir()
        cp(p.json, joinpath(dir, "scale_kernel.json")); cp(p.ptx, joinpath(dir, "scale_kernel.ptx"))
        @test endswith(triton_binary(joinpath(dir, "scale_kernel.json")).file, ".ptx")
    end

    @testset "what is refused, from the real metadata edited" begin
        p = triton_paths("scale_kernel")
        meta = read(p.json, String)
        function refused(edit)
            dir = mktempdir()
            write(joinpath(dir, "scale_kernel.json"), edit(meta))
            cp(p.cubin, joinpath(dir, "scale_kernel.cubin"))
            return err(() -> triton_binary(joinpath(dir, "scale_kernel.json")))
        end
        says(e, needle) = e isa ArgumentError && occursin(needle, e.msg)
        @test says(refused(m -> replace(m, r"\"global_scratch_size\":\s*0" => "\"global_scratch_size\": 64")), "scratch")
        @test says(refused(m -> replace(m, r"\"profile_scratch_size\":\s*0" => "\"profile_scratch_size\": 64")), "scratch")
        @test says(refused(m -> replace(m, r"\"num_ctas\":\s*1" => "\"num_ctas\": 2")), "clusters")
        @test says(refused(m -> replace(m, "\"cuda\"" => "\"hip\"")), "hip")
        @test says(refused(m -> replace(m, r"\"num_warps\":\s*\d+" => "\"num_warps\": null")), "num_warps")
        @test says(refused(m -> "[1, 2]"), "metadata")
        @test says(refused(m -> "{not json"), "JSON")
        @test err(() -> triton_binary(joinpath(mktempdir(), "missing.json"))) isa ArgumentError
        dir = mktempdir(); cp(p.json, joinpath(dir, "scale_kernel.json"))          # a JSON with no image beside it
        @test err(() -> triton_binary(joinpath(dir, "scale_kernel.json"))) isa ArgumentError
    end

    @testset "persisted with the binary: shared_mem, abi, scratch pointers" begin
        b = triton_binary(triton_paths("block_sum_kernel").json)
        @test b.shared_mem > 0                                         # a real, nonzero case
        v = add_variant!(TritonOp(), TritonK(), (:f32,); fwd=b)
        @test v[:fwd].shared_mem == b.shared_mem && v[:fwd].abi === :triton && v[:fwd].params.scratch_pointers == 2
        delete!(KT.VARIANTS, (TritonOp, TritonK, KT.current_target()))
        again = KT.load_manifest(TritonOp(), TritonK())[(:f32,)][:fwd]
        @test again == v[:fwd]

        # A manifest written before `shared_mem` existed loads with 0.
        path = KT.manifest_path(TritonOp(), TritonK())
        raw = TOML.parsefile(path)
        delete!(only(only(raw["variant"])["binary"]), "shared_mem")
        open(io -> TOML.print(io, raw), path, "w")
        @test KT.load_manifest(TritonOp(), TritonK())[(:f32,)][:fwd].shared_mem == 0
    end
end
