# The Triton fixtures (test/fixtures/triton/, made by test/fixtures/make_triton_fixture.py with real Triton on an NVIDIA
# machine and committed, so no test needs Python) and what the tests know about them.
#
# Each kernel's directory has `<name>.cubin`, `<name>.ptx`, `<name>.json` (what Triton wrote) and the raw
# inputs and `expected.bin`, Triton's own output for them. Float32 is `triton/`; `make_triton_fixture.py float16` and
# `bfloat16` make `triton_float16/` and `triton_bfloat16/` (raw 16-bit words, the atomic kernel left out): Triton compiles a
# different binary for each dtype, and the tests run every root that exists. The compiled signature of each kernel (the PTX `.entry`'s
# parameters) is what a `forward` has to bind: Triton drops `constexpr` arguments and integers equal to 1, and adds two
# hidden trailing pointers.

using BFloat16s: BFloat16

const TRITON_FIXTURES = get(ENV, "KD_TRITON_FIXTURES", joinpath(@__DIR__, "fixtures", "triton"))
const TRITON_DTYPES = (Float32, Float16, BFloat16)

# The directory of the fixtures made for element type `T`: `triton/`, `triton_float16/`, `triton_bfloat16/`.
triton_root(::Type{T}=Float32) where {T} =
    T === Float32 ? TRITON_FIXTURES : TRITON_FIXTURES * "_" * lowercase(string(nameof(T)))
have_triton_fixtures(::Type{T}=Float32) where {T} = isfile(joinpath(triton_root(T), "versions.txt"))

triton_paths(name, ::Type{T}=Float32) where {T} = (; dir=joinpath(triton_root(T), name),
    json=joinpath(triton_root(T), name, name * ".json"),
    cubin=joinpath(triton_root(T), name, name * ".cubin"),
    ptx=joinpath(triton_root(T), name, name * ".ptx"))

load_f32(dir, name) = collect(reinterpret(Float32, read(joinpath(dir, name * ".bin"))))
# The raw words a fixture holds, as `T`: float32, or the 16-bit patterns of a Float16 / BFloat16.
load_words(dir, name, ::Type{T}) where {T} = collect(reinterpret(T, read(joinpath(dir, name * ".bin"))))

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
triton_kernels(::Type{T}=Float32) where {T} = filter(k -> isdir(joinpath(triton_root(T), k.name)), ALL_TRITON_KERNELS)
const TRITON_KERNELS = triton_kernels(Float32)

# The types a `forward` binds for kernel `k` at element type `T`: its pointers are `T` buffers, its scalars what the PTX declares
# (the hidden trailing pointers are the launcher's, not in `k.args`). What `triton_binary(...; signature=)` takes.
triton_signature(k, ::Type{T}) where {T} =
    Tuple(a === :ptr ? Vector{T} : a === :f32 ? Float32 : a === :u32 ? Int32 : Int64 for a in k.args)

# One binary from plain values: built INSIDE a traced function (a captured `KernelBinary` is not traceable).
triton_mkbin(s) = KernelBinary(s.file; entry=s.entry, threadgroup=s.threadgroup, tile=1, abi=:triton, target=s.target,
    shared_mem=s.shared, params=(; scratch_pointers=s.nptr), signature=s.signature)
triton_spec(b) = (; file=b.file, entry=b.entry, threadgroup=b.threadgroup, target=b.target, shared=b.shared_mem,
    nptr=b.params.scratch_pointers, signature=b.signature)
