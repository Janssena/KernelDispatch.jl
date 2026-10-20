# The types a binary was compiled for, and the check that a launch passes them.
#
# A binary takes raw pointers and bits scalars, so nothing in it notices a `Float16` array bound to a `float*`, or an
# `Int64` where the kernel reads 4 bytes: the kernel runs and computes something else. A binary therefore carries a
# `signature`: one token per argument `call_binary` binds, in order (`Val`s take no slot and are left out). A buffer is
# the token of its element type, a scalar the token of its type after `s:`:
#
#     (Vector{Float16}, Vector{Float16}, Float32, Int32)   ->   ("f16", "f16", "s:f32", "s:i32")
#
# A token is lossless: the common numeric types have a short one (`f32`, `bf16`, `i32`), every other type is spelled out
# with its module and parameters (`Base.Complex{f32}`), so `ComplexF32` and `ComplexF64`, or two modules' `Params`, differ.
#
# `ka_compile` records it from its `argtypes`; for any other binary the user may declare it (`signature=`).

# --- tokens ----------------------------------------------------------------------------------------------

# The token of an argument type in a signature (`Val`s never reach it).
_arg_token(::Type{A}) where {A<:AbstractArray} = _type_token(eltype(A))
_arg_token(::Type{T}) where {T} = "s:" * _type_token(T)          # a scalar: `s:` apart from a buffer of the same type

const _TYPE_TOKENS = Dict{Symbol,String}(
    :Float32 => "f32", :Float16 => "f16", :BFloat16 => "bf16", :Float64 => "f64",
    :Int8 => "i8", :Int16 => "i16", :Int32 => "i32", :Int64 => "i64",
    :UInt8 => "u8", :UInt16 => "u16", :UInt32 => "u32", :UInt64 => "u64", :Bool => "bool")

# The short token of a type, if it is the numeric type the table means by that name: from Core or Base, or BFloat16s'
# `BFloat16` (Julia's own `Core.BFloat16` has the same layout and the same token). A user's own `Float32` is not.
function _known_token(T)
    T isa DataType || return nothing
    t = get(_TYPE_TOKENS, nameof(T), nothing)
    isnothing(t) && return nothing
    return nameof(parentmodule(T)) in (:Core, :Base, :BFloat16s) ? t : nothing
end

_type_token(::Type{T}) where {T} = something(_known_token(T), _canonical(T))

# A type spelled out: module path, name, parameters (types by their own token, other values as they print).
function _canonical(T)
    T isa DataType || return string(T)
    base = join(fullname(parentmodule(T)), ".") * "." * String(nameof(T))
    isempty(T.parameters) && return base
    return base * "{" * join((p isa Type ? _type_token(p) : string(p) for p in T.parameters), ",") * "}"
end

# Characters that are safe in a file name and a kernel's entry name.
_sanitize(s::AbstractString) = replace(s, r"[^A-Za-z0-9]" => "")

# The part of a file name that stands for a type: the short token for a numeric type, else its name and a hash of its full
# spelling (so `Complex{Float32}` and `Complex{Float64}` never share a file).
function _file_type(T)
    t = _known_token(T)
    isnothing(t) || return t
    return _sanitize(lowercase(String(nameof(T)))) * string(hash(_canonical(T)); base=16, pad=16)[1:8]
end

# One argument's part of a `ka_compile` file name; `Val`s are in the name (they change the compiled kernel), not the signature.
_file_token(::Type{A}) where {A<:AbstractArray} = _file_type(eltype(A))
_file_token(::Type{Val{x}}) where {x} = "v" * _sanitize(string(x))
_file_token(::Type{T}) where {T} = _file_type(T) * "s"          # a scalar: `s` apart from a buffer of the same type

# The signature of an argument-type list as `ka_compile` takes it: a token per argument, `Val{x}` (which takes no slot at
# launch) left out.
bound_tokens(argtypes::Tuple) = Tuple(_arg_token(T) for T in argtypes if !(T <: Val))

# What `KernelBinary(...; signature=)` accepts: `nothing` (unknown), `false` (unchecked), a tuple of tokens, or a tuple
# of the types the caller passes.
_as_signature(::Nothing) = nothing
_as_signature(x::Bool) = x ? throw(ArgumentError("`signature=true` says nothing; pass the argument types, " *
    "`false` to leave the binary unchecked, or `nothing` if they are unknown")) : false
_as_signature(t::Tuple{Vararg{String}}) = t
_as_signature(t::Union{Tuple,AbstractVector}) = all(x -> x isa Type, t) ? bound_tokens(Tuple(t)) :
    throw(ArgumentError("a `signature` is a tuple of argument types, like `(Vector{Float32}, Float32, Int32)`; got $t"))
_as_signature(x) = throw(ArgumentError(
    "a `signature` is a tuple of argument types, like `(Vector{Float32}, Float32, Int32)`, or `false`; got $x"))

# From a manifest: a list of strings, `false`, or absent.
_loaded_signature(x::AbstractVector) = Tuple(String.(x))
_loaded_signature(x) = x === false ? false : nothing

# --- the arguments of a launch -----------------------------------------------------------------------------

# The token of an argument as `call_binary` binds it, or `nothing` for one that takes no slot. A traced array's
# element type is Reactant's `TracedRNumber{T}`: `realtype` takes the `T` out.
_actual_token(::OutArray{T}) where {T} = _type_token(T)
_actual_token(x::InOutArray) = _type_token(realtype(eltype(x.array)))
_actual_token(x::AbstractArray) = _type_token(realtype(eltype(x)))
_actual_token(::Val) = nothing
_actual_token(x) = "s:" * _type_token(realtype(typeof(x)))

# The same argument as the type a signature would be declared with, for the message that suggests one.
_declared_type(::OutArray{T}) where {T} = Vector{T}
_declared_type(x::InOutArray) = Vector{realtype(eltype(x.array))}
_declared_type(x::AbstractArray) = Vector{realtype(eltype(x))}
_declared_type(x) = realtype(typeof(x))

_actual_tokens(args::Tuple, extras::Tuple) =
    String[t for t in (_actual_token(a) for a in (args..., extras...)) if t !== nothing]

# --- the check ---------------------------------------------------------------------------------------------

# Refuse, with the position and both types, a launch whose arguments are not the ones `b` was compiled for (`b.signature`).
# A binary whose signature is unknown (`nothing`) is not checked; on a device, the first launch of it warns once and
# suggests the `signature=` that fits. Done once per `prepare_launch`, eager or traced.
function check_signature(be::KernelBackend, b::KernelBinary, args::Tuple, extras::Tuple)::Nothing
    sig = getfield(b, :signature)
    sig === false && return nothing
    if isnothing(sig)
        be isa GPUBackend && warn_unknown_signature(b, args, extras)
        return nothing
    end
    actual = _actual_tokens(args, extras)
    length(actual) == length(sig) && all(i -> actual[i] == sig[i], eachindex(sig)) && return nothing
    throw_signature_mismatch(b, sig, actual)
end

@noinline function throw_signature_mismatch(b::KernelBinary, sig::Tuple, actual::Vector{String})
    what(t) = startswith(t, "s:") ? "scalar of type $(t[3:end])" : "buffer of $t"       # a scalar's token starts with `s:`
    expected = join(sig, ", ")
    msg = if length(actual) != length(sig)
        "binary `$(b.entry)` takes $(length(sig)) arguments ($expected) but this launch binds $(length(actual)) " *
        "($(join(actual, ", ")))"
    else
        i = findfirst(i -> actual[i] != sig[i], eachindex(sig))
        "argument $i of binary `$(b.entry)` is a $(what(actual[i])), but the binary was compiled for a $(what(sig[i])); " *
        "it takes ($expected) and this launch binds ($(join(actual, ", ")))"
    end
    throw(ArgumentError(msg * ". Pass `check_types=false` to launch anyway."))
end

const _WARNED_UNKNOWN = Set{Tuple{String,String}}()
const _WARNED_LOCK = ReentrantLock()

@noinline function warn_unknown_signature(b::KernelBinary, args::Tuple, extras::Tuple)
    id = (b.file, b.entry)
    first = lock(_WARNED_LOCK) do
        id in _WARNED_UNKNOWN ? false : (push!(_WARNED_UNKNOWN, id); true)
    end
    first || return nothing
    types = [_declared_type(a) for a in (args..., extras...) if !(a isa Val)]
    suggestion = "(" * join(types, ", ") * (length(types) == 1 ? ",)" : ")")
    @warn "Binary `$(b.entry)` ($(basename(b.file)), abi `:$(b.abi)`) has no recorded signature, so the types of its " *
        "arguments are not checked. This launch binds $suggestion; if that is what the binary was compiled for, pass " *
        "`signature=$suggestion` to `KernelBinary` (or `triton_binary`), or `signature=false` to silence this. " *
        "`Val`s take no slot and are left out."
    return nothing
end
