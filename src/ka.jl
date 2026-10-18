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
Implemented by the backend extension for its own tag; `()` if its kernels take none. There is no
default, so a backend that forgets fails here instead of launching with a shifted argument list.
"""
ka_state_words(be::KernelBackend) = throw(ArgumentError(
    "no `ka_state_words` for backend $(typeof(be)); load the package for that backend"))

"""
    ka_prelude(backend, ctx, ghost) -> Vector{Any}

What a KernelAbstractions-compiled binary takes ahead of its arguments: the backend's
[`ka_state_words`](@ref), then KA's launch context unless it is a ghost type. A `Vector{Any}`: the
context's type depends on the threadgroup VALUE, and keeping that out of the binder's signature keeps
`call_binary` inferable.
"""
ka_prelude(be::KernelBackend, ctx, ghost::Bool) =
    (p = Any[ka_state_words(be)...]; ghost || push!(p, ctx); p)

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
`joinpath(dir, name * ext)` whose entry point is `name`, and return it as a [`KernelBinary`](@ref) with
`abi = :ka` (and the given tuning `params`), ready for [`add_variant!`](@ref). Rebuilt only when absent —
put a source hash in `name`. `tile` defaults to one item per thread; a kernel whose threads each cover
several items says so here.

This is the KernelAbstractions producer of binaries and it belongs to the launch side: the binary is 
built against the launch context ([`ka_context`](@ref)) and argument pointer types that `call_binary` 
binds for an `abi = :ka` binary, so compiling by other means binds wrongly, silently. `dir` is where the 
file is built, by default [`ka_binary_cache_dir`](@ref): not part of the registry, which is where kernels 
live. [`add_variant!`](@ref) copies what it registers into it.

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
    path = _ka_compile(be, kernel, argtypes, Int(tg), String(name), String(dir))
    return KernelBinary(path; threadgroup=tg, tile, params, abi=:ka, target=target_id(be))
end

function _ka_compile end
