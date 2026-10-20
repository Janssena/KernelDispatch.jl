# Triton-compiled kernels as binaries. Triton (Python) compiles a kernel to a `.cubin` (and `.ptx`) and writes a
# metadata JSON beside it; `triton_binary` reads that JSON and describes the image as a `KernelBinary`
# with `abi = :triton`. Nothing here compiles Triton: the user does that, KernelDispatch only launches the output.
#
# What distinguishes a Triton binary from an `:explicit` one:
#   - its dynamic shared memory (`shared` in the metadata) is a launch parameter: `KernelBinary.shared_mem`;
#   - recent Triton versions declare hidden trailing pointer parameters after the kernel's own (a global scratch
#     and a profiling pointer, both unused by kernels that need no scratch): the launcher passes that many null
#     pointers after the arguments (`trailing_pointers`, kept in `params.scratch_pointers`).

"""
    triton_binary(metadata_json; image=nothing, tile=1, params=(;), scratch_pointers=nothing, signature=nothing) -> KernelBinary

Describe a kernel Triton compiled for NVIDIA as a [`KernelBinary`](@ref) (`abi = :triton`), from the metadata
JSON Triton wrote beside its output (`<name>.json` next to `<name>.cubin` and `<name>.ptx`). Where those files are is
your build step's choice: compile ahead into a directory you control. (Triton's own JIT cache, `<hash>/` directories
under its cache directory, `TRITON_CACHE_DIR` to move it, also holds them, but the hash names are opaque and Triton
may rewrite or clear them.) Register it like any binary:

    add_variant!(op, kernel, key; fwd=triton_binary("scale_kernel.json"))

- `image`: the `.cubin` or `.ptx` to launch. Default: the `.cubin` beside the JSON, else the `.ptx`. A cubin is
  machine code for the architecture the metadata names (recorded as the binary's `target`); PTX is portable.
- `tile`: how much of the problem one Triton program covers (see [`KernelBinary`](@ref)). The default `1` makes
  `extent` the grid of programs, which is how Triton is launched.
- `scratch_pointers`: how many hidden trailing pointer parameters the compiled kernel takes after its own.
  Read from the metadata: one for each of the `global_scratch_size` and `profile_scratch_size` entries it has (both
  are present in Triton 3.8.0, whose kernels end in exactly two pointers; Triton 3.2.0 has neither and none). Pass it to override for another version.

The kernel's entry, `threadgroup` (`num_warps` times the target's warp size) and `shared_mem` come from the metadata,
which must have the keys Triton 3.2.0 and 3.8.0 write (`name`, `num_warps`, `target` with `backend`, `arch` and `warp_size`).
What cannot be launched is refused here, with an explanation: a non-NVIDIA target, `num_ctas` above 1 (thread-block
clusters), and a kernel that needs global scratch memory (`global_scratch_size > 0`).

- `signature`: the types of the arguments `call_binary` binds, as for [`KernelBinary`](@ref) (`signature=(Vector{Float32},
  Vector{Float32}, Float32, Int32)`, the hidden trailing pointers and `constexpr` arguments not among them), or `false`
  to leave the binary unchecked. Triton's metadata does not say, so without it the first launch warns once and suggests
  the one that fits.

The arguments `call_binary` binds must be exactly the ones the *compiled* kernel declares: Triton drops
`constexpr` arguments from the signature, and (depending on its version) integers it specialized to 1. The order
of the buffers is the kernel's own: a Reactant trace passes each output as an aliased operand in its position.
"""
function triton_binary(json::AbstractString; image::Union{Nothing,AbstractString}=nothing, tile=1,
        params::NamedTuple=NamedTuple(), scratch_pointers::Union{Nothing,Integer}=nothing, signature=nothing)
    isfile(json) || throw(ArgumentError("Triton metadata `$json` not found"))
    meta = try
        _json_parse(read(json, String))
    catch e
        throw(ArgumentError("`$json` is not valid JSON: $(sprint(showerror, e))"))
    end
    meta isa AbstractDict || throw(ArgumentError("`$json` is not a Triton metadata object"))

    name = get(meta, "name", nothing)
    nw = get(meta, "num_warps", nothing)
    target = get(meta, "target", nothing)
    (name isa AbstractString && !isempty(name) && nw isa Integer && nw >= 1 && target isa AbstractDict &&
     get(target, "backend", nothing) isa AbstractString && get(target, "arch", nothing) isa Integer &&
     get(target, "warp_size", nothing) isa Integer) || throw(ArgumentError(
        "`$json` lacks the Triton metadata this needs (`name`, `num_warps` and a `target` with `backend`, `arch` " *
        "and `warp_size`): is it the `<kernel>.json` Triton 3.8 wrote beside the cubin?"))

    backend = target["backend"]
    backend == "cuda" || throw(ArgumentError(
        "`$json` is a Triton `$backend` kernel; only NVIDIA (`cuda`) kernels can be launched"))
    nctas = get(meta, "num_ctas", 1)
    nctas == 1 || throw(ArgumentError(
        "kernel `$name` was compiled with num_ctas=$nctas (thread-block clusters), which are not supported; " *
        "recompile with num_ctas=1"))
    for key in ("global_scratch_size", "profile_scratch_size")
        get(meta, key, 0) == 0 || throw(ArgumentError(
            "kernel `$name` needs `$key` = $(meta[key]) bytes of scratch memory, which is not supported yet"))
    end

    stem = splitext(json)[1]
    file = if image !== nothing
        String(image)
    else
        i = findfirst(isfile, stem .* (".cubin", ".ptx"))
        isnothing(i) && throw(ArgumentError(
            "no kernel image beside `$json`: expected `$(basename(stem)).cubin` or `.ptx`; pass `image=`"))
        stem * (".cubin", ".ptx")[i]
    end
    isfile(file) || throw(ArgumentError("kernel image `$file` not found"))
    sm = lowercase(splitext(file)[2]) == ".cubin" ? "sm_$(target["arch"])" : ""     # PTX is portable

    nptr = scratch_pointers !== nothing ? Int(scratch_pointers) :
           count(k -> haskey(meta, k), ("global_scratch_size", "profile_scratch_size"))
    nptr >= 0 || throw(ArgumentError("`scratch_pointers` must be non-negative, got $nptr"))
    p = merge(params, (; scratch_pointers=nptr))
    haskey(meta, "triton_version") && meta["triton_version"] isa AbstractString &&
        (p = merge(p, (; triton_version=String(meta["triton_version"]))))

    return KernelBinary(file; entry=String(name), threadgroup=target["warp_size"] * nw, tile, params=p, abi=:triton,
        target=sm, shared_mem=Int(get(meta, "shared", 0)), signature)
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
            isnothing(x) && error("bad number `$tok`")
            return x
        end
    end
    v = value()
    skipws()
    i > n || error("trailing characters at byte $i")
    return v
end
