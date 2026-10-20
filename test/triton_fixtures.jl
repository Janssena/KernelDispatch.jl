# The Triton fixtures (test/fixtures/triton/<triton version>/<precision>/, made by test/fixtures/make_triton_fixture.py with real Triton
# on an NVIDIA machine and committed, so no test needs Python) and what the tests know about them.
#
# One SET per Triton version and element type: `triton/3.2.0/float32/`, `triton/3.2.0/float16/`, `triton/3.8.0/float32/`, ... Triton
# compiles a different binary per dtype and writes different metadata (and a different number of hidden trailing pointers) per
# version, so each set is read and launched on its own. A set is a directory `triton/<version>/<precision>/` with a `versions.txt`;
# its Triton version and dtype come from that file (no `dtype` line means Float32), the names only place it.
#
# Each kernel's directory has `<name>.cubin`, `<name>.ptx`, `<name>.json` (what Triton wrote) and the raw inputs and `expected.bin`,
# Triton's own output for them (float32, or the raw 16-bit patterns of a Float16 / BFloat16). The compiled signature of each kernel
# (the PTX `.entry`'s parameters) is what a `forward` has to bind: Triton drops `constexpr` arguments and integers equal to 1, and
# (depending on its version) adds hidden trailing pointers.
#
# Where the `triton/` directory comes from (a local checkout of generated fixtures, `KD_FIXTURES`, or the downloaded artifact) is
# decided in fixture_root.jl. `KD_TRITON_DTYPES=Float16` and `KD_TRITON_VERSIONS=3.8.0` (comma-separated) run only those: a failure that depends on which sets ran before it is a different bug
# from one that does not.

using BFloat16s: BFloat16

isdefined(@__MODULE__, :fixture_dir) || include(joinpath(@__DIR__, "fixture_root.jl"))
const TRITON_FIXTURES = fixture_dir("triton")          # the `triton/` directory, or "" if there is none
const TRITON_DTYPES = (Float32, Float16, BFloat16)

# What each Triton version we have fixtures for is KNOWN to do: the hidden trailing pointers it adds and whether its metadata
# names its version. A version not in the table is only checked against its own metadata.
const TRITON_KNOWN = Dict(
    "3.2.0" => (; hidden_pointers=0, version_in_metadata=false),
    "3.8.0" => (; hidden_pointers=2, version_in_metadata=true),
)

function _triton_versions_file(dir)
    kv = Dict{String,String}()
    for l in eachline(joinpath(dir, "versions.txt"))
        m = match(r"^(\w+)\s+(.*)$", strip(l))
        m === nothing || (kv[m.captures[1]] = strip(m.captures[2]))
    end
    return kv
end

function triton_sets()
    names = Dict("float32" => Float32, "float16" => Float16, "bfloat16" => BFloat16)
    want_dtypes = filter(!isempty, split(get(ENV, "KD_TRITON_DTYPES", ""), ','))
    want_versions = filter(!isempty, split(get(ENV, "KD_TRITON_VERSIONS", ""), ','))
    sets = NamedTuple[]
    isempty(TRITON_FIXTURES) && return sets
    top = TRITON_FIXTURES
    dirs = [joinpath(top, v, p) for v in sort(readdir(top)) if isdir(joinpath(top, v)) for p in sort(readdir(joinpath(top, v)))]
    for d in dirs
        (isdir(d) && isfile(joinpath(d, "versions.txt"))) || continue
        kv = _triton_versions_file(d)
        T = names[get(kv, "dtype", "float32")]
        isempty(want_dtypes) || string(nameof(T)) in want_dtypes || continue
        isempty(want_versions) || get(kv, "triton", "") in want_versions || continue
        push!(sets, (; root=d, version=get(kv, "triton", "?"), T, label="Triton $(get(kv, "triton", "?")), $(nameof(T))"))
    end
    return sets
end
const TRITON_SETS = triton_sets()
have_triton_fixtures() = !isempty(TRITON_SETS)

triton_paths(name, set) = (; dir=joinpath(set.root, name),
    json=joinpath(set.root, name, name * ".json"),
    cubin=joinpath(set.root, name, name * ".cubin"),
    ptx=joinpath(set.root, name, name * ".ptx"))

# The raw words a fixture holds, as `T`: float32, or the 16-bit patterns of a Float16 / BFloat16.
load_words(dir, name, ::Type{T}) where {T} = collect(reinterpret(T, read(joinpath(dir, name * ".bin"))))
load_words(dir, name, set::NamedTuple) = load_words(dir, name, set.T)

# The kinds of a PTX entry's parameters, in order: `:ptr`, `:f32`, `:u32`, `:u64`.
function ptx_param_kinds(ptx::AbstractString)
    i = findfirst(".entry", ptx)
    j = findnext("{", ptx, first(i))
    kinds = Symbol[]
    for l in split(ptx[first(i):first(j)], '\n')
        occursin(".param", l) || continue
        push!(kinds, occursin(".ptr", l) ? :ptr : occursin(".f32", l) ? :f32 :
                     occursin(".u32", l) ? :u32 : occursin(".u64", l) ? :u64 : :other)
    end
    return kinds
end

# `args`: the parameters the compiled kernel declares of its own (before the hidden trailing pointers), as the PTX
# shows them. `num_warps`: how it was launched.
const ALL_TRITON_KERNELS = [
    (name="scale_kernel", args=[:ptr, :ptr, :f32, :u32], num_warps=2),                  # x, y, a, n   (BLOCK dropped)
    (name="block_sum_kernel", args=[:ptr, :ptr, :u32], num_warps=4),                    # x, out, n    (BLOCK dropped)
    # out, stride_o, x, [stride_x == 1: dropped], y, stride_y, a, n   (BLOCK dropped)
    (name="strided_axpy", args=[:ptr, :u32, :ptr, :ptr, :u32, :f32, :u32], num_warps=1),
    (name="accumulate_kernel", args=[:ptr, :ptr, :f32, :u32], num_warps=2),             # y (in/out), x, a, n
    (name="atomic_sum_kernel", args=[:ptr, :ptr, :u32], num_warps=4),                   # x, out (atomic), n
]
# Only the kernels whose directory exists: fixtures generated before a kernel was added lack it.
triton_kernels(set) = filter(k -> isdir(joinpath(set.root, k.name)), ALL_TRITON_KERNELS)

# The types a `forward` binds for kernel `k` at element type `T`: its pointers are `T` buffers, its scalars what the PTX declares
# (the hidden trailing pointers are the launcher's, not in `k.args`). What `triton_binary(...; signature=)` takes.
triton_signature(k, ::Type{T}) where {T} =
    Tuple(a === :ptr ? Vector{T} : a === :f32 ? Float32 : a === :u32 ? Int32 : Int64 for a in k.args)

# One binary from plain values: built INSIDE a traced function (a captured `KernelBinary` is not traceable).
triton_mkbin(s) = KernelBinary(s.file; entry=s.entry, threadgroup=s.threadgroup, tile=1, abi=:triton, target=s.target,
    shared_mem=s.shared, params=(; scratch_pointers=s.nptr), signature=s.signature)
triton_spec(b) = (; file=b.file, entry=b.entry, threadgroup=b.threadgroup, target=b.target, shared=b.shared_mem,
    nptr=b.params.scratch_pointers, signature=b.signature)
