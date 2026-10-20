# Triton-compiled kernels as binaries, without a GPU and without Python. Two halves:
#
#   - what needs no Triton at all: the JSON reader (generic JSON, no claim about Triton's schema), the `shared_mem`
#     field, and the `:triton` ABI's prelude and grid;
#   - what is checked against kernels REAL Triton compiled (test/fixtures/triton/, Triton 3.8.0 on an NVIDIA A40, made by
#     test/fixtures/make_triton_fixture.py and committed): `triton_binary` on the real metadata, the compiled signature
#     against what a `forward` binds, the refusals (made by editing the real metadata), and the manifest round trip. This
#     half is skipped, with a warning, when the fixtures are absent.
#
# The launches themselves are test/ext/cuda/triton.test.jl (eager and traced, against Triton's own output).

using Test, KernelDispatch, TOML, BFloat16s
isdefined(@__MODULE__, :TRITON_KERNELS) || include(joinpath(@__DIR__, "..", "triton_fixtures.jl"))
isdefined(@__MODULE__, :TRIFAST_FIXTURES) || include(joinpath(@__DIR__, "..", "trifast_fixtures.jl"))

struct TritonOp <: AbstractKernelOp end
struct TritonK <: AbstractKernel end
const TRITON_ROOT = mktempdir()
KD.cache_dir(::TritonOp) = joinpath(TRITON_ROOT, "triton_op")

err(f) = try f(); nothing catch e; e end

# --- no Triton needed --------------------------------------------------------------------------------------------

@testset "JSON reader" begin
    doc = """{"name": "k", "n": 4, "x": -1.5e-3, "yes": true, "no": false, "none": null,
             "list": [1, [2, 3], {"a": []}], "nested": {"arch": 86}, "esc": "a\\"b\\n\\u0041\\\\"}"""
    j = KD._json_parse(doc)
    @test j["name"] == "k" && j["n"] == 4 && j["x"] ≈ -1.5e-3
    @test j["yes"] === true && j["no"] === false && j["none"] === nothing
    @test j["list"] == [1, [2, 3], Dict("a" => []) ] && j["nested"]["arch"] == 86
    @test j["esc"] == "a\"b\nA\\"
    @test KD._json_parse("[]") == [] && KD._json_parse(" {} ") == Dict{String,Any}() && KD._json_parse("-3") == -3
    for bad in ("", "{", "{\"a\":}", "[1,]x", "{\"a\" 1}", "tru", "1 2", "\"open")
        @test_throws Exception KD._json_parse(bad)
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
    prelude, grid = KD.abi_prelude_and_grid(Val(:triton), KD.CUDABackendTag(), b, (3, 2, 1))
    @test isempty(prelude) && grid == (3, 2, 1)
    l = KD.prepare_launch(b, rand(Float32, 4); extent=(3, 2), extras=(Int32(4),))
    @test l.groups == (3, 2, 1) && l.grid == (3, 2, 1) && isempty(l.prelude)
    @test KD.trailing_pointers(b) == 2
    @test KD.trailing_pointers(KernelBinary(f; threadgroup=32)) == 0          # only a `:triton` binary has them
end

# --- against what real Triton compiled -----------------------------------------------------------------------------

if !have_triton_fixtures()
    @warn "no Triton fixtures in $TRITON_FIXTURES (run test/fixtures/make_triton_fixture.py on an NVIDIA machine): " *
          "skipping the real-Triton tests"
else
    @testset "triton_binary and the compiled signature: $(k.name)" for k in TRITON_KERNELS
        p = triton_paths(k.name)
        meta = KD._json_parse(read(p.json, String))
        kinds = ptx_param_kinds(read(p.ptx, String))

        for image in (:cubin, :ptx)
            b = triton_binary(p.json; image=getfield(p, image))
            @test b.abi === :triton && b.entry == k.name && b.tile == (1, 1, 1)    # tile 1: `extent` is the grid
            @test b.threadgroup == 32 * k.num_warps == 32 * meta["num_warps"]
            @test b.shared_mem == meta["shared"]
            @test KD.binary_format(b) === image
            @test b.target == (image === :cubin ? "sm_" * string(meta["target"]["arch"]) : "")   # PTX is portable
            # Triton 3.8.0 writes its version into the metadata; 3.2.0 does not (and has no scratch-size entries either).
            @test haskey(meta, "triton_version") ? b.params.triton_version == meta["triton_version"] : !haskey(b.params, :triton_version)
        end

        # The hidden trailing pointers: one for each scratch entry the metadata has (both in Triton 3.8.0, none in 3.2.0), and
        # the compiled kernel ends in exactly that many pointer parameters after the arguments Triton kept.
        b = triton_binary(p.json)
        nptr = KD.trailing_pointers(b)
        @test nptr == b.params.scratch_pointers == count(key -> haskey(meta, key), ("global_scratch_size", "profile_scratch_size"))
        @test length(kinds) == length(k.args) + nptr
        @test kinds[1:length(k.args)] == k.args                    # what `forward` binds, in this order
        @test all(==(:ptr), kinds[end-nptr+1:end])                 # what the launcher appends
    end

    @testset "overrides and the choice of image" begin
        p = triton_paths("scale_kernel")
        @test endswith(triton_binary(p.json).file, ".cubin")                       # the cubin beside the JSON, by default
        @test endswith(triton_binary(p.json; image=p.ptx).file, ".ptx")
        @test triton_binary(p.json; tile=(8, 1)).tile == (8, 1, 1)
        @test KD.trailing_pointers(triton_binary(p.json; scratch_pointers=0)) == 0
        @test triton_binary(p.json; params=(; mine=1)).params.mine == 1
        # With only the PTX beside the JSON, the PTX is used.
        dir = mktempdir()
        cp(p.json, joinpath(dir, "scale_kernel.json")); cp(p.ptx, joinpath(dir, "scale_kernel.ptx"))
        @test endswith(triton_binary(joinpath(dir, "scale_kernel.json")).file, ".ptx")
    end

    @testset "what is refused, from the real metadata edited" begin
        p = triton_paths("scale_kernel")
        meta = read(p.json, String)
        # The metadata edited: a key set to a value, added if this Triton's metadata lacks it (3.2.0 has no scratch entries).
        setkey(m, key, val) = occursin("\"$key\"", m) ? replace(m, Regex("\"$key\":\\s*[^,}\\s]+") => "\"$key\": $val") :
                              replace(m, "{" => "{\"$key\": $val,"; count=1)
        function refused(edit)
            dir = mktempdir()
            write(joinpath(dir, "scale_kernel.json"), edit(meta))
            cp(p.cubin, joinpath(dir, "scale_kernel.cubin"))
            return err(() -> triton_binary(joinpath(dir, "scale_kernel.json")))
        end
        says(e, needle) = e isa ArgumentError && occursin(needle, e.msg)
        @test says(refused(m -> setkey(m, "global_scratch_size", 64)), "scratch")
        @test says(refused(m -> setkey(m, "profile_scratch_size", 64)), "scratch")
        @test says(refused(m -> setkey(m, "num_ctas", 2)), "clusters")
        @test says(refused(m -> replace(m, "\"cuda\"" => "\"hip\"")), "hip")
        @test says(refused(m -> setkey(m, "num_warps", "null")), "num_warps")
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
        @test v[:fwd].shared_mem == b.shared_mem && v[:fwd].abi === :triton && v[:fwd].params.scratch_pointers == b.params.scratch_pointers
        delete!(KD.VARIANTS, (TritonOp, TritonK, KD.current_target()))
        again = KD.load_manifest(TritonOp(), TritonK())[(:f32,)][:fwd]
        @test again == v[:fwd]

        # A manifest written before `shared_mem` existed loads with 0.
        path = KD.manifest_path(TritonOp(), TritonK())
        raw = TOML.parsefile(path)
        delete!(only(only(raw["variant"])["binary"]), "shared_mem")
        open(io -> TOML.print(io, raw), path, "w")
        @test KD.load_manifest(TritonOp(), TritonK())[(:f32,)][:fwd].shared_mem == 0
    end
end

# --- every element type the fixtures were made for ----------------------------------------------------------------------
# `triton/` is Float32; `triton_float16/` and `triton_bfloat16/` come from `make_triton_fixture.py float16|bfloat16` (the atomic
# kernel is Float32 only). Triton compiles a different binary per dtype, so each is read like the Float32 ones: the compiled
# signature has the same shape (a `*bf16` is still a `.ptr`), and a `signature=` declared from it is what a launch is checked against.

for T in TRITON_DTYPES
    if !have_triton_fixtures(T)
        T === Float32 || @info "no Triton fixtures for $T in $(triton_root(T)): skipping them"
        continue
    end
    @testset "$T: the compiled signature and a declared type signature, $(k.name)" for k in triton_kernels(T)
        p = triton_paths(k.name, T)
        kinds = ptx_param_kinds(read(p.ptx, String))
        b = triton_binary(p.json)
        nptr = KD.trailing_pointers(b)
        @test length(kinds) == length(k.args) + nptr
        @test kinds[1:length(k.args)] == k.args
        @test all(==(:ptr), kinds[end-nptr+1:end])

        # Declared from those kinds, the signature is what a launch is checked against (host arrays: the check does not need a device).
        sig = triton_signature(k, T)
        typed = triton_binary(p.json; signature=sig)
        @test typed.signature == KD.bound_tokens(sig)
        arg(kind, E) = kind === :ptr ? zeros(E, 8) : kind === :f32 ? 1.0f0 : kind === :u32 ? Int32(1) : Int64(1)
        good = [arg(a, T) for a in k.args]
        @test KD.prepare_launch(typed, good...; extent=1) isa KD.Launch
        other = T === Float32 ? Float16 : Float32
        i = findfirst(==(:ptr), k.args)
        bad = [arg(a, a === :ptr ? other : T) for a in k.args]
        e = err(() -> KD.prepare_launch(typed, bad...; extent=1))
        @test e isa ArgumentError && occursin("compiled for", e.msg) && occursin("argument $i", e.msg)
        # a scalar of the wrong width is refused too (the PTX `.u32` is an Int32 here, not an Int64)
        j = findfirst(a -> a === :u32, k.args)
        if j !== nothing
            wide = copy(good); wide[j] = Int64(1)
            @test err(() -> KD.prepare_launch(typed, wide...; extent=1)) isa ArgumentError
        end
    end
end

# --- trifast's kernels (test/fixtures/trifast/): the described call against the compiled kernels -------------------

if !have_trifast_fixtures()
    @warn "no trifast fixtures in $TRIFAST_FIXTURES (run test/fixtures/make_trifast_fixture.py on an NVIDIA machine): " *
          "skipping them"
else
    case = trifast_case()
    @testset "trifast: $(inst["name"])" for inst in case["instances"]
        b = triton_binary(trifast_json(inst))
        kinds = ptx_param_kinds(read(trifast_image(inst, :ptx), String))
        want = [a["kind"] == "i32" ? :u32 : a["kind"] == "f32" ? :f32 : :ptr for a in inst["args"]]
        # What the script recorded is what the compiled kernel declares, then the hidden pointers.
        @test kinds[1:length(want)] == want
        @test length(kinds) == length(want) + KD.trailing_pointers(b)
        @test b.shared_mem == inst["shared"] && b.threadgroup == 32 * inst["num_warps"] && b.abi === :triton
        plan = trifast_plan(case, inst)
        @test length(plan.args) == length(want) && length(plan.grid) == 3
        # Every array the call names exists with the shape the files say.
        for (code, name, _, dims) in plan.args
            code == 1 && @test size(trifast_data(case, name)) == Tuple(reverse(Int.(case["tensors"][name]["shape"])))
            code == 2 && @test !isempty(dims)
        end
    end
end
