@kernel function _geometry_probe() end

"""
    ka_context(backend, groups::NTuple{3,Int}, tg) -> (ctx, is_ghost)

KA's launch context for a binary dispatched over `groups` threadgroups of `tg` threads: the ND-range
`(groups[1]·tg, groups[2], groups[3])` with workgroup `(tg, 1, 1)`, built on the backend's
[`ka_backend_object`](@ref). `is_ghost`: the context occupies no buffer slot.

The context is built from `_geometry_probe`, an empty KA kernel, not from the kernel the binary was
compiled from. A KA context depends only on the launch geometry (ND-range and static workgroup size),
never on which kernel it serves, so the probe yields exactly the context the binary was compiled
against. That matters because the kernel object is often gone by launch time: a binary loaded from a
manifest has no kernel in this session. The context itself comes from KA's own `launch_config` and
`mkcontext`, never built by hand, so its layout stays KA's business.
"""
function ka_context(be::KernelBackend, groups::NTuple{3,Int}, tg::Integer)
    obj = _geometry_probe(ka_backend_object(be), (Int(tg), 1, 1))
    ndrange = (groups[1] * Int(tg), groups[2], groups[3])
    nd, _, iterspace, _ = KernelAbstractions.launch_config(obj, ndrange, (Int(tg), 1, 1))
    ctx = KernelAbstractions.mkcontext(obj, nd, iterspace)
    return (ctx, sizeof(typeof(ctx)) == 0)
end

"""
    ka_grid(backend, groups::NTuple{3,Int}) -> NTuple{3,Int}

The 3-D grid of threadgroups a KernelAbstractions binary is dispatched on, given the `groups` it
covers (the ones its [`ka_context`](@ref) was built from). Implemented by the backend extension: how a
backend lays out threadgroup positions is its own business (Metal's KA kernels read only the x
component, so it flattens). No default, so a backend that forgets fails here instead of launching a
grid its kernels misread.
"""
ka_grid(be::KernelBackend, ::NTuple{3,Int}) = throw(ArgumentError(
    "no `ka_grid` for backend $(typeof(be)); load the package for that backend"))

"""
    ka_state_words(backend) -> Tuple

The bits values a KernelAbstractions-compiled binary takes ahead of KA's launch context, in order:
whatever the backend's KA kernels pass as hidden leading arguments (Metal.jl's kernel-state word, say).
Implemented by the backend extension for its own tag; `()` if its kernels take none (CUDA's are built without one). There is no
default, so a backend that forgets fails here instead of launching with a shifted argument list.
"""
ka_state_words(be::KernelBackend) = throw(ArgumentError(
    "no `ka_state_words` for backend $(typeof(be)); load the package for that backend"))

"""
    ka_prelude(backend, ctx, ghost) -> Vector{Any}

What a KernelAbstractions-compiled binary takes ahead of its arguments: the backend's
[`ka_state_words`](@ref), then KA's launch context unless it is a ghost type. (A backend may pass the context in
another form: CUDA passes its six integers, the ND-range and the block counts, as separate scalars.) A `Vector{Any}`: the
context's type depends on the threadgroup VALUE, and keeping that out of the binder's signature keeps
`call_binary` inferable.
"""
ka_prelude(be::KernelBackend, ctx, ghost::Bool) =
    (p = Any[ka_state_words(be)...]; ghost || push!(p, ctx); p)

"""
    ka_layout(backend) -> Union{Nothing,Symbol}

How `backend` lays out what a KernelAbstractions binary takes ahead of its arguments, when that is a layout it has
changed over time: `nothing` (the default, and Metal) records nothing. A backend that returns a symbol has
[`ka_compile`](@ref) store it in the binary's `params` as `ka_layout`, and can refuse (through
[`compatible_target`](@ref)) a binary built with another layout instead of launching it with shifted arguments.
CUDA returns `:flat`: no kernel-state argument, and KA's context as six plain integers.
"""
ka_layout(::KernelBackend) = nothing

"""
    ka_binary_cache_dir(target=current_target()) -> String

Where [`ka_compile`](@ref) builds binaries by default: `ka` under [`cache_root`](@ref), in a subdirectory named
for the device architecture (`ka/sm_86`) when there is one, so a cache shared by machines with different GPUs
never serves one's binary to another. Output is only
staged here — [`add_variant!`](@ref) copies a registered binary into the registry, which is where
kernels live — but it is kept between sessions, so a `name` already built is found, not rebuilt.
"""
ka_binary_cache_dir(target::AbstractString=current_target()) =
    isempty(target) ? joinpath(cache_root(), "ka") : joinpath(cache_root(), "ka", String(target))

"""
    ka_compile(backend, kernel, argtypes::Tuple; tg, name, params=(;), tile=(tg, 1, 1), dir=ka_binary_cache_dir(target_id(backend))) -> KernelBinary

Ahead-of-time compile the KernelAbstractions `kernel` at workgroup size `tg` to a binary at
`joinpath(dir, "<name>_<types>" * ext)` whose entry point is `<name>_<types>`, and return it as a
[`KernelBinary`](@ref) with `abi = :ka` (and the given tuning `params`), ready for [`add_variant!`](@ref).
`<types>` is the signature of `argtypes` (see [`ka_signature_tag`](@ref): `f16x3_f32s_i32s_v64` for three
`Float16` buffers, a `Float32` and an `Int32` scalar and a `Val{64}`), so builds of one kernel at different
element types never share a file. Rebuilt only when absent — put a source hash in `name`. `tile` defaults to one item per thread; a kernel whose threads each cover
several items says so here.

This is the KernelAbstractions producer of binaries and it belongs to the launch side: the binary is 
built against the launch context ([`ka_context`](@ref)) and argument pointer types that `call_binary` 
binds for an `abi = :ka` binary, so compiling by other means binds wrongly. `dir` is where the 
file is built, by default [`ka_binary_cache_dir`](@ref): not part of the registry, which is where kernels 
live. [`add_variant!`](@ref) copies what it registers into it.

On CUDA the binary is built without CUDA.jl's kernel state (and with KA's context as six plain integers): its
parameters are only device pointers and scalars, so the same file launches eagerly and from a Reactant trace. The
price is that a kernel whose compiled code can still throw (a bounds check, a `DomainError`, `error`) is refused here,
at build time, because a device cannot report an exception without that state; index with `unsafe_load` /
`unsafe_store!` as the binary contract already says. `params` gains a reserved `ka_layout` entry ([`ka_layout`](@ref)).

The binary records its `signature` (the `argtypes` without the `Val`s), and a launch with other types is refused: a
`Float16` array for a `Float32` buffer, an `Int64` for an `Int32`. See [`KernelBinary`](@ref).

`argtypes` lists the kernel's arguments after KA's context, in signature order: `AbstractArray{T}`
for a buffer (bound as a raw device pointer to `T`; the kernel indexes it with `unsafe_load` /
`unsafe_store!`), a bits type for a scalar, `Val{x}` for a compile-time parameter — the order
[`call_binary`](@ref) will bind them in.
"""
function ka_compile(be::KernelBackend, kernel, argtypes::Tuple;
    tg::Integer,
    name::AbstractString,
    params::NamedTuple=NamedTuple(),
    tile=(tg, 1, 1),
    dir::AbstractString=ka_binary_cache_dir(target_id(be))
)
    path = _ka_compile(be, kernel, argtypes, Int(tg), ka_binary_name(name, argtypes), String(dir))
    layout = ka_layout(be)
    isnothing(layout) || (params = merge(params, (; ka_layout=String(layout))))
    return KernelBinary(path; threadgroup=tg, tile, params, abi=:ka, target=target_id(be),
        signature=bound_tokens(argtypes))
end

"""
    ka_signature_tag(argtypes::Tuple) -> String

The part of a [`ka_compile`](@ref) binary's name that says which types it was built for: one token per
argument, in order, runs of equal tokens written `tokenxcount`. A buffer (`AbstractArray{T}`) is the token of its
element type: `Float32` is `f32`, `Float16` `f16`, `BFloat16` `bf16`, `Float64` `f64`, `Int32` `i32`, `Int64`
`i64`, `UInt32` `u32`, `Bool` `bool`; any other type its lowercased name and eight hex digits of a hash of its full
spelling, so `Complex{Float32}` and `Complex{Float64}` differ. A scalar of that type has an `s` after it
(`f32s`), and `Val{x}` is `v<x>`. For instance `(Vector{Float16}, Vector{Float16}, Float32, Int32, Val{64})` gives
`"f16x2_f32s_i32s_v64"`.
"""
function ka_signature_tag(argtypes::Tuple)
    toks = String[_file_token(T) for T in argtypes]
    parts = String[]
    i = 1
    while i <= length(toks)
        j = i
        while j < length(toks) && toks[j+1] == toks[i]
            j += 1
        end
        push!(parts, j > i ? "$(toks[i])x$(j - i + 1)" : toks[i])
        i = j + 1
    end
    return join(parts, "_")
end

ka_binary_name(name::AbstractString, argtypes::Tuple) = String(name) * "_" * ka_signature_tag(argtypes)

function _ka_compile end
