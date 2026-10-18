# Triton-compiled kernels as binaries. Triton (Python) compiles a kernel to a `.cubin` (and `.ptx`) and writes a
# metadata JSON beside it; `triton_binary` reads that JSON and describes the image as a `KernelBinary`
# with `abi = :triton`. Nothing here compiles Triton: the user does that, KernelDispatch only launches the output.
#
# What distinguishes a Triton binary from a plain one:
#   - its dynamic shared memory (`shared` in the metadata) is a launch parameter: `KernelBinary.shared_mem`;
#   - recent Triton versions declare hidden trailing pointer parameters after the kernel's own (a global scratch
#     and a profiling pointer, both unused by kernels that need no scratch): the launcher passes that many null
#     pointers after the arguments (`trailing_pointers`, kept in `params.scratch_pointers`).

"""
    triton_binary(metadata_json; image=nothing, tile=1, params=(;), scratch_pointers=nothing) -> KernelBinary

Describe a kernel Triton compiled for NVIDIA as a [`KernelBinary`](@ref) (`abi = :triton`), from the metadata
JSON Triton wrote beside its output (in `~/.triton/cache/<hash>/`, or `TRITON_CACHE_DIR`: `<name>.json`
next to `<name>.cubin` and `<name>.ptx`). Register it like any binary:

    add_variant!(op, kernel, key; fwd=triton_binary("scale_kernel.json"))

- `image`: the `.cubin` or `.ptx` to launch. Default: the `.cubin` beside the JSON, else the `.ptx`. A cubin is
  machine code for the architecture the metadata names (recorded as the binary's `target`); PTX is portable.
- `tile`: how much of the problem one Triton program covers (see [`KernelBinary`](@ref)). The default `1` makes
  `extent` the grid of programs, which is how Triton is launched.
- `scratch_pointers`: how many hidden trailing pointer parameters the compiled kernel takes after its own.
  Read from the metadata (one for a `global_scratch_size` entry, one more for `profile_scratch_size`), which is a
  heuristic over Triton versions: pass it to override. A kernel compiled without them takes none.

The kernel's entry, `threadgroup` (`num_warps * 32`) and `shared_mem` come from the metadata. What cannot be
launched is refused here, with an explanation: a non-NVIDIA target, `num_ctas` or cluster dimensions above 1, and
a kernel that needs global scratch memory (`global_scratch_size > 0`).

The arguments `call_binary` binds must be exactly the ones the *compiled* kernel declares: Triton drops
`constexpr` arguments from the signature, and (depending on its version) integers it specialized to 1. Buffers
must also list every input before every output when launched from a Reactant trace.
"""
function triton_binary(json::AbstractString; image::Union{Nothing,AbstractString}=nothing, tile=1,
        params::NamedTuple=NamedTuple(), scratch_pointers::Union{Nothing,Integer}=nothing)
    isfile(json) || throw(ArgumentError("Triton metadata `$json` not found"))
    meta = try
        _json_parse(read(json, String))
    catch e
        throw(ArgumentError("`$json` is not valid JSON: $(sprint(showerror, e))"))
    end
    meta isa AbstractDict || throw(ArgumentError("`$json` is not a Triton metadata object"))

    name = get(meta, "name", nothing)
    nw = get(meta, "num_warps", nothing)
    (name isa AbstractString && !isempty(name) && nw isa Integer && nw >= 1) || throw(ArgumentError(
        "`$json` lacks the Triton metadata this needs (`name`, `num_warps`): is it the `<kernel>.json` " *
        "Triton wrote beside the cubin?"))

    backend = _triton_backend(meta)
    isempty(backend) || backend == "cuda" || throw(ArgumentError(
        "`$json` is a Triton `$backend` kernel; only NVIDIA (`cuda`) kernels can be launched"))
    nctas = get(meta, "num_ctas", 1)
    cluster = get(meta, "cluster_dims", [1, 1, 1])
    (nctas == 1 && all(==(1), cluster)) || throw(ArgumentError(
        "kernel `$name` was compiled for thread-block clusters (num_ctas=$nctas, cluster_dims=$cluster), " *
        "which are not supported; recompile with num_ctas=1"))
    for key in ("global_scratch_size", "profile_scratch_size")
        get(meta, key, 0) == 0 || throw(ArgumentError(
            "kernel `$name` needs `$key` = $(meta[key]) bytes of scratch memory, which is not supported yet"))
    end

    stem = splitext(json)[1]
    file = if image !== nothing
        String(image)
    else
        i = findfirst(isfile, stem .* (".cubin", ".ptx"))
        i === nothing && throw(ArgumentError(
            "no kernel image beside `$json`: expected `$(basename(stem)).cubin` or `.ptx`; pass `image=`"))
        stem * (".cubin", ".ptx")[i]
    end
    isfile(file) || throw(ArgumentError("kernel image `$file` not found"))
    target = lowercase(splitext(file)[2]) == ".cubin" ? _triton_sm(meta) : ""

    nptr = scratch_pointers !== nothing ? Int(scratch_pointers) :
           count(k -> haskey(meta, k), ("global_scratch_size", "profile_scratch_size"))
    nptr >= 0 || throw(ArgumentError("`scratch_pointers` must be non-negative, got $nptr"))
    warp = _triton_warp_size(meta)
    p = merge(params, (; scratch_pointers=nptr))
    haskey(meta, "triton_version") && meta["triton_version"] isa AbstractString &&
        (p = merge(p, (; triton_version=String(meta["triton_version"]))))

    return KernelBinary(file; entry=String(name), threadgroup=warp * nw, tile, params=p, abi=:triton, target,
        shared_mem=Int(get(meta, "shared", 0)))
end

# `"cuda"` / `"hip"` from the `target` entry (an object `{backend, arch, warp_size}` in recent Triton, a list
# `[backend, arch]` in older ones) or `backend_name`; `""` when the metadata does not say.
function _triton_backend(meta)
    t = get(meta, "target", nothing)
    t isa AbstractDict && haskey(t, "backend") && return String(t["backend"])
    t isa AbstractVector && !isempty(t) && t[1] isa AbstractString && return String(t[1])
    b = get(meta, "backend_name", "")
    return b isa AbstractString ? String(b) : ""
end

# "sm_80" from `target.arch` (80), or the `arch` string ("sm80"); `""` if neither is there.
function _triton_sm(meta)
    t = get(meta, "target", nothing)
    arch = t isa AbstractDict ? get(t, "arch", nothing) :
           t isa AbstractVector && length(t) >= 2 ? t[2] : nothing
    arch isa Integer && return "sm_$arch"
    a = get(meta, "arch", nothing)
    if a isa AbstractString
        m = match(r"^sm_?(\d+[af]?)$", a)
        m === nothing || return "sm_" * m[1]
    end
    return ""
end

function _triton_warp_size(meta)
    t = get(meta, "target", nothing)
    w = t isa AbstractDict ? get(t, "warp_size", 32) : 32
    return w isa Integer && w >= 1 ? Int(w) : 32
end

# The null pointers a binary takes after its arguments: Triton's hidden scratch parameters. Passed as zero
# words, by the eager launcher and in a traced program alike, so a kernel that declares them gets them whether or
# not the launcher appends any itself (extra parameters beyond the kernel's own are ignored by the driver).
trailing_pointers(b::KernelBinary) = getfield(b, :abi) === :triton ? Int(get(b.params, :scratch_pointers, 0)) : 0

# A Triton binary takes nothing ahead of its arguments, and is dispatched on its programs' grid as is.
abi_prelude_and_grid(::Val{:triton}, ::KernelBackend, ::KernelBinary, groups) = (Any[], groups)

# --- a JSON reader, just enough for Triton's metadata -----------------------------------------------------------

# Objects become `Dict{String,Any}`, arrays `Vector{Any}`, numbers `Int` or `Float64`, plus strings, `Bool` and
# `nothing`. Strict enough to reject garbage, small enough not to need a dependency.
function _json_parse(s::AbstractString)
    str = String(s)
    n = ncodeunits(str)
    i = 1
    peek() = i <= n ? str[i] : '\0'
    function skipws()
        while i <= n && isspace(str[i])
            i = nextind(str, i)
        end
    end
    function expect(c::Char)
        peek() == c || error("expected '$c' at byte $i")
        i = nextind(str, i)
    end
    function literal(word, val)
        stop = i + ncodeunits(word) - 1
        (stop <= n && str[i:stop] == word) || error("unexpected token at byte $i")
        i = stop + 1
        return val
    end
    function string_()
        expect('"')
        io = IOBuffer()
        while true
            i <= n || error("unterminated string")
            c = str[i]; i = nextind(str, i)
            c == '"' && break
            if c == '\\'
                e = str[i]; i = nextind(str, i)
                if e == 'u'
                    cp = parse(UInt32, str[i:i+3]; base=16); i += 4
                    write(io, Char(cp))
                else
                    write(io, get(Dict('n' => '\n', 't' => '\t', 'r' => '\r', 'b' => '\b', 'f' => '\f'), e, e))
                end
            else
                write(io, c)
            end
        end
        return String(take!(io))
    end
    function value()
        skipws()
        c = peek()
        if c == '{'
            expect('{'); d = Dict{String,Any}(); skipws()
            if peek() == '}'
                expect('}'); return d
            end
            while true
                skipws(); k = string_(); skipws(); expect(':')
                d[k] = value(); skipws()
                peek() == ',' ? expect(',') : (expect('}'); break)
            end
            return d
        elseif c == '['
            expect('['); a = Any[]; skipws()
            if peek() == ']'
                expect(']'); return a
            end
            while true
                push!(a, value()); skipws()
                peek() == ',' ? expect(',') : (expect(']'); break)
            end
            return a
        elseif c == '"'
            return string_()
        elseif c == 't'
            return literal("true", true)
        elseif c == 'f'
            return literal("false", false)
        elseif c == 'n'
            return literal("null", nothing)
        else
            j = i
            while j <= n && str[j] in "+-0123456789.eE"
                j += 1
            end
            j > i || error("unexpected character '$c' at byte $i")
            tok = str[i:j-1]; i = j
            x = something(tryparse(Int, tok), tryparse(Float64, tok), Some(nothing))
            x === nothing && error("bad number `$tok`")
            return x
        end
    end
    v = value()
    skipws()
    i > n || error("trailing characters at byte $i")
    return v
end
