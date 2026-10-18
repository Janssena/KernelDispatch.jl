# Triton-compiled kernels as binaries, as far as they can be checked without Triton or a GPU: the metadata reader,
# `triton_binary`'s refusals, the trailing scratch pointers, `shared_mem` and its manifest round trip. That the
# driver and XLA launch such a kernel is checked on a CUDA machine (gpu/triton.test.jl, cuda_smoke.jl).

using Test, KernelDispatch, TOML
const KT = KernelDispatch

struct TritonOp <: AbstractKernelOp end
struct TritonK <: AbstractKernel end
const TRITON_ROOT = mktempdir()
KT.cache_dir(::TritonOp) = joinpath(TRITON_ROOT, "triton_op")

# Metadata as Triton 3.x writes it (trimmed to what is read, plus entries that must be ignored).
const META = """
{"hash": "d41d8cd98f", "target": {"backend": "cuda", "arch": 80, "warp_size": 32}, "num_warps": 4,
 "num_ctas": 1, "num_stages": 3, "maxnreg": null, "cluster_dims": [1, 1, 1], "enable_fp_fusion": true,
 "extern_libs": [["libdevice", "/x/libdevice.10.bc"]], "ratio": 1.5e-3, "label": "a\\"b\\n\\u0041",
 "triton_version": "3.3.0", "shared": 2048, "global_scratch_size": 0, "global_scratch_align": 1,
 "name": "scale_kernel"}
"""

function triton_dir(; meta=META, files=(".cubin",))
    d = mktempdir()
    write(joinpath(d, "scale_kernel.json"), meta)
    for e in files
        write(joinpath(d, "scale_kernel" * e), UInt8[0x7f, 0x45, 0x4c, 0x46])
    end
    return d
end
meta_with(; kw...) = replace(META, kw...)

@testset "JSON reader" begin
    j = KT._json_parse(META)
    @test j["num_warps"] == 4 && j["shared"] == 2048 && j["name"] == "scale_kernel"
    @test j["target"]["arch"] == 80 && j["maxnreg"] === nothing && j["enable_fp_fusion"] === true
    @test j["ratio"] ≈ 1.5e-3 && j["label"] == "a\"b\nA"
    @test j["extern_libs"] == [["libdevice", "/x/libdevice.10.bc"]] && j["cluster_dims"] == [1, 1, 1]
    @test KT._json_parse("[]") == [] && KT._json_parse(" {} ") == Dict{String,Any}() && KT._json_parse("-3") == -3
    for bad in ("", "{", "{\"a\":}", "[1,]x", "{\"a\" 1}", "tru", "1 2", "\"open")
        @test_throws Exception KT._json_parse(bad)
    end
end

@testset "triton_binary" begin
    d = triton_dir()
    b = triton_binary(joinpath(d, "scale_kernel.json"))
    @test b.abi === :triton && b.entry == "scale_kernel" && b.threadgroup == 128 && b.shared_mem == 2048
    @test b.file == joinpath(d, "scale_kernel.cubin") && b.target == "sm_80"
    @test b.tile == (1, 1, 1)                                   # extent is the grid of programs
    @test b.params.scratch_pointers == 1 && b.params.triton_version == "3.3.0"
    @test KT.trailing_pointers(b) == 1
    @test KT.trailing_pointers(KernelBinary(b.file; threadgroup=32)) == 0   # only a :triton binary has them

    # Overrides, and the choice of image.
    @test triton_binary(joinpath(d, "scale_kernel.json"); tile=(8, 1), scratch_pointers=2).tile == (8, 1, 1)
    @test KT.trailing_pointers(triton_binary(joinpath(d, "scale_kernel.json"); scratch_pointers=0)) == 0
    both = triton_dir(files=(".cubin", ".ptx"))
    @test endswith(triton_binary(joinpath(both, "scale_kernel.json")).file, ".cubin")
    p = triton_binary(joinpath(both, "scale_kernel.json"); image=joinpath(both, "scale_kernel.ptx"))
    @test endswith(p.file, ".ptx") && p.target == ""            # PTX is portable
    only_ptx = triton_dir(files=(".ptx",))
    @test endswith(triton_binary(joinpath(only_ptx, "scale_kernel.json")).file, ".ptx")

    # Triton versions without the scratch entries take no trailing pointers; older metadata spells the target as a list.
    old = replace(META, "\"global_scratch_size\": 0, \"global_scratch_align\": 1," => "",
        "{\"backend\": \"cuda\", \"arch\": 80, \"warp_size\": 32}" => "[\"cuda\", 86]")
    ob = triton_binary(joinpath(triton_dir(meta=old), "scale_kernel.json"))
    @test ob.params.scratch_pointers == 0 && ob.target == "sm_86"

    # What cannot be launched is refused with the reason.
    err(f) = try f(); nothing catch e; e end
    refuses(meta, needle) = (e = err(() -> triton_binary(joinpath(triton_dir(; meta), "scale_kernel.json")));
        e isa ArgumentError && occursin(needle, e.msg))
    @test refuses(replace(META, "\"cuda\"" => "\"hip\""), "hip")
    @test refuses(replace(META, "\"num_ctas\": 1" => "\"num_ctas\": 2"), "clusters")
    @test refuses(replace(META, "\"cluster_dims\": [1, 1, 1]" => "\"cluster_dims\": [2, 1, 1]"), "clusters")
    @test refuses(replace(META, "\"global_scratch_size\": 0" => "\"global_scratch_size\": 128"), "scratch")
    @test refuses(replace(META, "\"num_warps\": 4," => ""), "num_warps")
    @test refuses("[1, 2]", "metadata")
    @test refuses("{not json", "JSON")
    @test err(() -> triton_binary(joinpath(d, "missing.json"))) isa ArgumentError
    @test err(() -> triton_binary(joinpath(triton_dir(files=()), "scale_kernel.json"))) isa ArgumentError   # no image
end

@testset "the :triton ABI" begin
    b = triton_binary(joinpath(triton_dir(), "scale_kernel.json"))
    # Nothing ahead of the arguments, and the grid as given: the trailing pointers are the launcher's.
    prelude, grid = KT.abi_prelude_and_grid(Val(:triton), KT.CUDABackendTag(), b, (3, 2, 1))
    @test isempty(prelude) && grid == (3, 2, 1)
    x = rand(Float32, 4)
    l = KT.prepare_launch(b, x; extent=(3, 2), extras=(Int32(4),))
    @test l.groups == (3, 2, 1) && l.grid == (3, 2, 1) && isempty(l.prelude)
end

@testset "shared_mem" begin
    f = tempname() * ".cubin"; write(f, UInt8[1])
    @test KernelBinary(f; threadgroup=32).shared_mem == 0
    @test KernelBinary(f; threadgroup=32, shared_mem=4096).shared_mem == 4096
    @test_throws ArgumentError KernelBinary(f; threadgroup=32, shared_mem=-1)

    # Persisted with the binary, and an older manifest without the entry loads as 0.
    begin
        b = triton_binary(joinpath(triton_dir(), "scale_kernel.json"))
        v = add_variant!(TritonOp(), TritonK(), (:f32,); fwd=b)
        @test v[:fwd].shared_mem == 2048 && v[:fwd].abi === :triton && v[:fwd].params.scratch_pointers == 1
        delete!(KT.VARIANTS, (TritonOp, TritonK, KT.current_target()))
        again = KT.load_manifest(TritonOp(), TritonK())[(:f32,)][:fwd]
        @test again == v[:fwd] && again.shared_mem == 2048

        path = KT.manifest_path(TritonOp(), TritonK())
        raw = TOML.parsefile(path)
        delete!(only(only(raw["variant"])["binary"]), "shared_mem")
        open(io -> TOML.print(io, raw), path, "w")
        @test KT.load_manifest(TritonOp(), TritonK())[(:f32,)][:fwd].shared_mem == 0
    end
end
