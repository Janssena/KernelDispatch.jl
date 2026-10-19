"""
    AbstractKernelOp

An operation that runs as compiled kernels. Subtype it with a singleton:

    struct Axpy <: AbstractKernelOp end

Its name ([`opname`](@ref)) defaults to the type's; define it for an identifier that survives renaming.

Calling it (`Axpy()(args...)`) runs [`host`](@ref) on host arrays and [`forward`](@ref) on device
arrays, with its [`default_kernel`](@ref) or the one selected by [`use_kernel!`](@ref).
"""
abstract type AbstractKernelOp end

"""
    AbstractKernel

One implementation of an op, switchable with [`use_kernel!`](@ref):

    struct AxpyKA <: AbstractKernel end

Its name ([`kernelname`](@ref)) defaults to the type's; define it for an identifier that survives
renaming. Nothing needs registering: an op's [`@default_kernel`](@ref) candidates and any kernel
passed to `use_kernel!` are registered on first use.

It needs a [`forward`](@ref) method (and a [`backward`](@ref) one to differentiate), which build its
binary's arguments and launch it with [`call_binary`](@ref). Its other methods ([`variant_key`](@ref),
[`extras`](@ref), [`nearest`](@ref), [`build!`](@ref), [`source_tag`](@ref)) say how its binaries are
chosen and launched.
"""
abstract type AbstractKernel end

"""
    KernelBinary(file; entry=basename without extension, threadgroup, tile=(threadgroup, 1, 1),
        params=(;), abi=:explicit, target="")

One compiled launch, everything about it persisted with it in the manifest.

- `file` is the compiled binary: a `.metallib` on Metal; a `.cubin` or `.ptx` on CUDA (a cubin is machine
  code for one architecture, PTX is text the driver or XLA compiles at load). Any other extension is refused
  on CUDA.

- `entry` is the name of the kernel function inside `file`, the one the runtime launches. It defaults
  to the file name without its extension, which is right whenever the file is named after its
  function (trifast's `_fwd.metallib`, everything from [`ka_compile`](@ref)). Set it only when the
  function is named otherwise, or the file holds several functions. It is fixed here, from the
  original path, because [`add_variant!`](@ref) copies the file into the registry under a new name.
  It is unrelated to the launch name a variant files the binary under (`fwd=` in `add_variant!`, the
  name `call_binary` takes): that is your label for its role, this is the toolchain's name for it.
- `threadgroup` is the number of threads per threadgroup the binary is dispatched with.
- `tile` is how much of the problem one threadgroup covers along each axis: [`call_binary`](@ref)
  launches `cld.(extent, tile)` threadgroups for the `extent` a call passes it. The default,
  `(threadgroup, 1, 1)`, is one item per thread; a Triton kernel running one program per item has
  `tile=(1, 1, 1)`, a kernel tiling a matrix in 32×32 blocks `(32, 32, 1)`. An integer or a shorter
  tuple is padded with 1s. It is a property of the build, so it changes with the binary when retuned.
  Any grid can be launched this way: with `tile=1`, the extent IS the grid.
- `params` are the tuning numbers it was built with, for [`extras`](@ref) to read.
- `abi` says what the binary takes besides the arguments `call_binary` binds, so the launcher can supply
  it. `:explicit` (the default) takes nothing: exactly the buffers and scalars you bind, in the order given, with
  nothing added or implied (`:plain`, its earlier name, is still accepted). `:ka` marks a binary
  compiled from a KernelAbstractions kernel ([`ka_compile`](@ref) sets it): it takes the backend's
  [`ka_state_words`](@ref) and KA's launch context ahead of its arguments (in the form the backend's
  [`ka_prelude`](@ref) gives; CUDA passes the context as six integers), and is dispatched flat. An
  extension can define another ABI (see `abi_prelude_and_grid`). It is a property of the binary, not of
  the kernel type that holds it: one kernel can register a KA-compiled binary for one launch and a Triton
  one for another. (`is_ka=true` and `b.is_ka` remain as deprecated aliases for `abi=:ka`.)
- `target` names the device architecture the binary was built for (`"sm_86"` for a CUDA cubin), `""` when it
  is portable (a `.metallib`, PTX). [`ka_compile`](@ref) sets it; a prebuilt file can say it, so a launch on
  a device it cannot run on is refused with an explanation instead of failing in the driver.
- `shared_mem` is the dynamic shared memory, in bytes, each threadgroup is launched with (a Triton kernel's
  `shared`; see [`triton_binary`](@ref)). `0` for a kernel that declares its threadgroup memory itself, as
  every Metal and KernelAbstractions kernel does. Metal ignores it.
"""
struct KernelBinary
    file::String
    entry::String
    threadgroup::Int
    params::NamedTuple
    abi::Symbol
    tile::NTuple{3,Int}
    target::String
    shared_mem::Int
    # Fields of `params` kept sorted, so a binary compares equal to itself after a manifest round-trip.
    function KernelBinary(file, entry, threadgroup, params::NamedTuple, abi::Symbol, tile,
            target::AbstractString="", shared_mem::Integer=0)
        abi === :plain && (abi = :explicit)          # the earlier name of the default ABI
        t = _pad3(tile)
        all(>=(1), t) || throw(ArgumentError("`tile` must be positive along every axis, got $t"))
        shared_mem >= 0 || throw(ArgumentError("`shared_mem` must be non-negative, got $shared_mem"))
        return new(String(file), String(entry), Int(threadgroup), _sorted(params), abi, t, String(target),
            Int(shared_mem))
    end
end

function KernelBinary(file::AbstractString;
    entry::AbstractString=splitext(basename(file))[1],
    threadgroup::Integer,
    tile=(threadgroup, 1, 1),
    params::NamedTuple=NamedTuple(),
    abi::Symbol=:explicit,
    target::AbstractString="",
    shared_mem::Integer=0,
    is_ka::Union{Nothing,Bool}=nothing)       # deprecated alias for `abi=:ka`
    is_ka === true && (abi = :ka)
    return KernelBinary(file, entry, threadgroup, params, abi, tile, target, shared_mem)
end

# `b.is_ka`, the pre-`abi` spelling: derived, so there is one source of truth.
function Base.getproperty(b::KernelBinary, s::Symbol)
    s === :is_ka && return getfield(b, :abi) === :ka
    return getfield(b, s)
end
Base.propertynames(::KernelBinary, private::Bool=false) = (fieldnames(KernelBinary)..., :is_ka)

# An extent or tile as three axes: an integer or a tuple of up to three, padded with 1s.
_pad3(x::Integer) = (Int(x), 1, 1)
function _pad3(x::Tuple)
    length(x) <= 3 || throw(ArgumentError("at most three axes, got $(length(x)): $x"))
    return ntuple(i -> i <= length(x) ? Int(x[i]) : 1, 3)
end
_pad3(x::AbstractVector) = _pad3(Tuple(x))

_sorted(nt::NamedTuple) = NamedTuple{Tuple(sort(collect(keys(nt))))}(nt)

# --- the interface ---------------------------------------------------------------------------------

"""
`opname(op) -> Symbol`: the op's name, its identifier outside the running code: the REPL shorthand
(`use_kernel!(:Axpy, …)`), its preference key (`"kernel.<opname>"`) and its registry directory.
Default: the op type's name. Define it to keep those stable when the type is renamed.
"""
opname(op::AbstractKernelOp) = nameof(typeof(op))

"""
`kernelname(kernel) -> Symbol`: the kernel's name within its op, its identifier outside the running
code: the REPL shorthand, the value a default-kernel preference holds, its registry directory.
Default: the kernel type's name. Define it to keep those stable when the type is renamed.
"""
kernelname(k::AbstractKernel) = nameof(typeof(k))

"""
    host(op, args...)

The op on host arrays. Runs whatever kernel is selected: the selection applies on a device.
"""
host(op::AbstractKernelOp, args...) =
    throw(ArgumentError("op `:$(opname(op))` has no host implementation; move the arguments to a device"))

"""
    forward(op, args...)
    forward(op, kernel, args...)

The op on device arrays. The first form runs the selected kernel. The second is defined per op and
kernel, and is required: it builds the binary's arguments from the op's (outputs as
[`OutArray`](@ref), sizes and scalars in the binary's types), launches it with
[`call_binary`](@ref) (`extent=` the problem size), and returns what the op's callers should get.

There is no working default: a binary's calling convention can't be guessed, and passing the op's
arguments through unchanged would bind them wrongly, silently. A kernel without a `forward` throws.
"""
forward(op::AbstractKernelOp, args...) = forward(op, selected_kernel(op), args...)
forward(op::AbstractKernelOp, k::AbstractKernel, args...) = throw(ArgumentError(
    "kernel `:$(kernelname(k))` of op `:$(opname(op))` has no `forward`: define " *
    "`KernelDispatch.forward(op, kernel, args...)` to build its binary's arguments (outputs as `OutArray`, " *
    "sizes and scalars in the binary's types) and launch it with `call_binary(...; extent=...)`"
))

"""
    backward(op, kernel, args, outs, cotangents) -> Tuple

Gradients of `forward(op, kernel, args...)`, one per argument (`nothing` for one with none), given
its outputs `outs` and their `cotangents`. No default: overload it to make an op differentiable.
"""
function backward(op::AbstractKernelOp, k::AbstractKernel, args, outs, cots)
    msg = "op `:$(opname(op))` kernel `:$(kernelname(k))` has no `backward`; overload " *
        "`KernelDispatch.backward(op, kernel, args, outs, cotangents)` to differentiate it"
    names = _launch_names(op, k, args)
    length(names) > 1 && (msg *= ". The variant for these arguments has the launches " *
        join((":$n" for n in names), ", ") * ": call the backward one there with " *
        "`call_binary(op, kernel, name, ...)` and return one gradient per argument (`nothing` for one " *
        "with none)")
    throw(ArgumentError(msg * "."))
end

# The launch names of the variant serving `args`, for the error above. Looks only at the registered
# variants: it never builds one (`build!`) just to word an error.
function _launch_names(op, k, args)
    v = try
        _lookup(op, k, variants(op, k), variant_key(op, k, args...))
    catch
        nothing
    end
    return isnothing(v) ? Symbol[] : sort(collect(keys(v)))
end

"""`variant_key(op, kernel, args...) -> Tuple`: which tuned variant serves these arguments. Default `()`."""
variant_key(::AbstractKernelOp, ::AbstractKernel, args...) = ()

"""`extras(op, kernel, binary) -> Tuple`: arguments appended to the caller's (tuning values). Default `()`."""
extras(::AbstractKernelOp, ::AbstractKernel, ::KernelBinary) = ()

"""`nearest(op, kernel, keys, key)`: a registered key to use when `key` has no exact variant, or `nothing`."""
nearest(::AbstractKernelOp, ::AbstractKernel, keys, key) = nothing

"""`build!(op, kernel, key)`: create the variant for `key` (a lazy tune) via [`add_variant!`](@ref). Default: nothing."""
build!(::AbstractKernelOp, ::AbstractKernel, key) = nothing

"""`source_tag(kernel) -> String`: a manifest recorded with another tag is ignored. Default `""`."""
source_tag(::AbstractKernel) = ""

"""
`cache_dir(op) -> String`: where the op's kernels keep their manifests and binaries. Default:
`ops/<Package>.<opname>` under [`cache_root`](@ref), so equally named ops from different packages
(or a script, `Main`) never share a registry.
"""
cache_dir(op::AbstractKernelOp) = joinpath(cache_root(), "ops",
    "$(nameof(Base.moduleroot(parentmodule(typeof(op))))).$(opname(op))")

# --- tables ----------------------------------------------------------------------------------------

# Registration is lazy: nothing here defines methods, so it can all happen at first use. Variants are
# keyed by the op and kernel TYPES, so they never mix across equally named ops; the name tables (`OPS`,
# `KERNELS`) only resolve names, and refuse a second type under a taken name.
const Variant = Dict{Symbol,KernelBinary}
const OPS = Dict{Symbol,AbstractKernelOp}()
const KERNELS = Dict{Symbol,Dict{Symbol,AbstractKernel}}()
const VARIANTS = Dict{Tuple{DataType,DataType,String},Dict{Tuple,Variant}}()

"""
`variants(op, kernel; target=current_target()) -> Dict{Tuple,Variant}`: the kernel's variants, by key, for
the device architecture `target` (see [`current_target`](@ref)): binaries built for one architecture are kept
apart from another's. The first access in a session loads them from the kernel's manifest, so persisted
variants need no registration.
"""
variants(op::AbstractKernelOp, k::AbstractKernel; target::AbstractString=current_target()) =
    get!(() -> load_manifest(op, k; target), VARIANTS, (typeof(op), typeof(k), String(target)))

"""
    register_kernel!(op, kernel) -> kernel

Make `kernel` known to `op` by name, and `op` by its name: for [`use_kernel!`](@ref)`(op, name)` and
[`list_kernels`](@ref). (Re)loads the kernel's persisted variants.

Rarely needed: an op's [`@default_kernel`](@ref) candidates are registered on first use, and
`use_kernel!(op, kernel)` registers the kernel it selects. Call it to have another kernel (say, one
from another package) selectable by name and listed before its first use. Registering the same kernel
again only reloads its manifest; a DIFFERENT kernel under a name the op already has throws, as does a
different op under a name already taken: they would share one entry, directory and manifest.
"""
function register_kernel!(op::AbstractKernelOp, k::AbstractKernel)
    on, kn = _register_op!(op), kernelname(k)
    ks = get!(() -> Dict{Symbol,AbstractKernel}(), KERNELS, on)
    prev = get(ks, kn, nothing)
    (isnothing(prev) || typeof(prev) === typeof(k)) || throw(ArgumentError(
        "op `:$on` already has a kernel named `:$kn` ($(typeof(prev)), from " *
        "$(parentmodule(typeof(prev)))); give $(typeof(k)) another `kernelname`"))
    ks[kn] = k
    merge!(variants(op, k), load_manifest(op, k))
    return k
end

# Make `op` known by its name, refusing a name another op type has; returns the name.
function _register_op!(op::AbstractKernelOp)
    on = opname(op)
    prev = get(OPS, on, nothing)
    (isnothing(prev) || typeof(prev) === typeof(op)) || throw(ArgumentError(
        "an op named `:$on` is already registered ($(typeof(prev)), from $(parentmodule(typeof(prev)))); " *
        "give $(typeof(op)) another `opname`"))
    OPS[on] = op
    return on
end

_op(op::AbstractKernelOp) = op
_op(::Type{O}) where {O<:AbstractKernelOp} = O()
function _op(name::Symbol)
    haskey(OPS, name) && return OPS[name]
    found = filter(o -> opname(o) === name, _declared_ops())
    length(found) == 1 && (_register_op!(only(found)); return only(found))
    isempty(found) || throw(ArgumentError(
        "several ops are named `:$name` ($(join(map(typeof, found), ", "))); pass the op itself"))
    throw(ArgumentError("no op named `:$name`. Ops: " *
        _names(union(keys(OPS), map(opname, _declared_ops())))))
end

# The ops with a `@default_kernel` in the loaded code: the singleton op types of
# `default_candidates`' methods. How a name finds an op that nothing has registered yet.
function _declared_ops()
    ops = AbstractKernelOp[]
    for m in methods(default_candidates)
        sig = Base.unwrap_unionall(m.sig)
        length(sig.parameters) == 2 || continue
        T = sig.parameters[2]
        (T isa DataType && T <: AbstractKernelOp && Base.issingletontype(T)) && push!(ops, T.instance)
    end
    return ops
end

_names(xs) = isempty(xs) ? "(none)" : join(sort([":$x" for x in xs]), ", ")

# --- selection -------------------------------------------------------------------------------------
#
# Which kernel an op runs is a METHOD, not a table entry, so it is a compile-time constant and an op
# call infers. Two layers: the op's default (`default_kernel`, defined by `@default_kernel` in the op's
# own package from a compile-time preference, so it is baked into precompile caches) and a session
# override (`_override`, redefined by `use_kernel!`, which invalidates and recompiles the callers).
# A redefinition is visible from the next top-level expression on: see `throw_stale_switch`.

"""
    default_kernel(op) -> kernel or nothing

The kernel `op` runs unless [`use_kernel!`](@ref) overrides it. Defined by [`@default_kernel`](@ref)
in the op's package; `nothing` (no default) otherwise.
"""
default_kernel(::AbstractKernelOp) = nothing

# The session override per op type, redefined by `use_kernel!`. Owned by KernelDispatch, so a switch
# never overwrites a method of the op's package.
_override(::AbstractKernelOp) = nothing

# Bumped on every switch; `_switch_epoch()` is redefined to match. They disagree exactly when the code
# running was compiled in a world from before the latest switch.
const SWITCH_EPOCH = Ref(0)
_switch_epoch() = 0

_generating_output() = ccall(:jl_generating_output, Cint, ()) == 1

function _set_override!(o::AbstractKernelOp, k::Union{Nothing,AbstractKernel})
    Base.invokelatest(_override, o) === k && return nothing      # unchanged: invalidate nothing
    _generating_output() && throw(ArgumentError(
        "kernels cannot be switched while precompiling; set the default of op `:$(opname(o))` with " *
        "`KernelDispatch.@default_kernel` (and its preference with `set_default_kernel!`) instead"))
    n = SWITCH_EPOCH[] + 1
    @eval begin
        _override(::$(typeof(o))) = $k
        _switch_epoch() = $n
    end
    SWITCH_EPOCH[] = n
    return nothing
end

"""
    use_kernel!(op, kernel_name) -> Symbol
    use_kernel!(op, kernel) -> Symbol

Select which kernel `op` runs for the rest of the session, overriding its [`default_kernel`](@ref).
`op` is the op (`Axpy()`), its type (`Axpy`) or its name (`:axpy`); the kernel is the kernel itself
(`AxpyPlain()`), registered on the way if it is new to the op, or its name (`:plain`), for a kernel
already registered or listed in the op's `@default_kernel`. Not persisted: see
[`set_default_kernel!`](@ref).

Selection is the application's decision: global per op, it affects every caller in the session, so a
library never switches. A library that needs one particular kernel calls `forward(op, kernel, args...)`.

The selection is a method, so op calls stay inferred; code calling the op recompiles on its next call.
It takes effect from the NEXT top-level expression (REPL input, script statement): later in the same
function or `begin`/`let` block the op call throws, rather than run the previous kernel. (A
`@testset` body sees it at once.) To run one call with a particular kernel, call
`forward(op, kernel, args...)` instead: no switch, no recompilation.
"""
function use_kernel!(op, k::AbstractKernel)
    o = _op(op)
    _set_override!(o, _registered(o, k))
    return kernelname(k)
end

function use_kernel!(op, kname::Symbol)
    o = _op(op)
    return use_kernel!(o, _kernel(o, kname))
end

# `k` as known to `o`, registering it if its name is free. A kernel that merely shares a known
# kernel's name must not select it.
function _registered(o::AbstractKernelOp, k::AbstractKernel)
    reg = _find_kernel(o, kernelname(k))
    isnothing(reg) && return register_kernel!(o, k)
    typeof(reg) === typeof(k) || throw(ArgumentError(
        "$(typeof(k)) can't be selected for op `:$(opname(o))`: its name `:$(kernelname(k))` belongs " *
        "to $(typeof(reg)). Give it another `kernelname`."))
    return reg
end

# The kernel named `kname`: a registered one, else one of the op's `@default_kernel` candidates
# (registered on the way); `nothing` if neither.
function _find_kernel(o::AbstractKernelOp, kname::Symbol)
    ks = get(KERNELS, opname(o), nothing)
    (isnothing(ks) || !haskey(ks, kname)) || return ks[kname]
    i = findfirst(k -> kernelname(k) === kname, default_candidates(o))
    return isnothing(i) ? nothing : register_kernel!(o, default_candidates(o)[i])
end

function _kernel(o::AbstractKernelOp, kname::Symbol)
    k = _find_kernel(o, kname)
    isnothing(k) || return k
    known = union(keys(get(KERNELS, opname(o), Dict{Symbol,AbstractKernel}())),
        map(kernelname, default_candidates(o)))
    throw(ArgumentError("op `:$(opname(o))` has no kernel `:$kname`. Kernels: " * _names(known)))
end

"""`reset_kernel!(op)`: drop the [`use_kernel!`](@ref) override; `op` runs its default again."""
reset_kernel!(op) = (_set_override!(_op(op), nothing); nothing)

"""`current_kernel(op) -> Symbol or nothing`: the kernel `op` runs (as of the newest world)."""
function current_kernel(op)
    k = Base.invokelatest(_selected, _op(op))
    return isnothing(k) ? nothing : kernelname(k)
end

"""`list_kernels(op)`: every kernel of `op` (registered, or one of its `@default_kernel` candidates), its variant count, and which is selected."""
function list_kernels(op)
    o = _op(op)
    foreach(k -> _find_kernel(o, kernelname(k)), default_candidates(o))
    sel = current_kernel(o)
    return [(; name=kn, nvariants=length(variants(o, k)), selected=(kn === sel))
            for (kn, k) in sort(collect(get(KERNELS, opname(o), Dict{Symbol,AbstractKernel}())); by=first)]
end

_selected(op::AbstractKernelOp) = (k = _override(op); k === nothing ? default_kernel(op) : k)

"""
    selected_kernel(op) -> kernel

The kernel `op` runs now: the [`use_kernel!`](@ref) override, else its [`default_kernel`](@ref). Unlike
[`current_kernel`](@ref), which returns the NAME as of the newest world, this returns the kernel itself and
is inferred, so it is what to call (and dispatch on) from code that runs the op. Throws when none is
selected, and on a call from code compiled before the latest switch (see [`use_kernel!`](@ref)).
"""
function selected_kernel(op::AbstractKernelOp)
    _switch_epoch() == SWITCH_EPOCH[] || throw_stale_switch(op)
    k = _selected(op)
    isnothing(k) && throw_no_kernel(op)
    return k
end

@noinline throw_no_kernel(op) = throw(ArgumentError(
    "no kernel selected for op `:$(opname(op))`; call `use_kernel!(:$(opname(op)), name)`, or give " *
    "the op a default with `KernelDispatch.@default_kernel`. Kernels: " *
    _names(keys(get(KERNELS, opname(op), Dict())))))

@noinline throw_stale_switch(op) = throw(ArgumentError(
    "op `:$(opname(op))` was called from code compiled before the latest kernel switch: `use_kernel!` " *
    "(or `reset_kernel!`) takes effect from the next top-level expression, not later in the same " *
    "function or block. Switch at top level, or call `forward(op, kernel, args...)` to run one call " *
    "with a particular kernel."))

"""
    variant(op, kernel, key; target=current_target()) -> Variant

The variant serving `key`: an exact match, else the kernel's [`nearest`](@ref) choice, else what
its [`build!`](@ref) creates. Throws when none exists. `target` selects the architecture's variants;
`build!` always builds for the current device, so a lookup for another target only finds what is registered.
"""
function variant(op::AbstractKernelOp, k::AbstractKernel, key::Tuple; target::AbstractString=current_target())
    vs = variants(op, k; target)
    v = _lookup(op, k, vs, key)
    isnothing(v) || return v
    build!(op, k, key)
    v = _lookup(op, k, vs, key)
    isnothing(v) || return v
    throw(ArgumentError("kernel `:$(kernelname(k))` of op `:$(opname(op))` has no variant for key " *
        "$key. Registered: $(sort(collect(keys(vs)); by=string))"))
end

function _lookup(op, k, vs, key)
    haskey(vs, key) && return vs[key]
    isempty(vs) && return nothing
    nk = nearest(op, k, keys(vs), key)
    return isnothing(nk) ? nothing : vs[nk]
end

# --- manifest: one TOML per (op, kernel) -------------------------------------------------------------

# `<cache_dir(op)>/[<target>/]<kernel>`: a device architecture's binaries have a directory of their own, so
# machines with different GPUs can share a cache; no target (the host, Metal, portable) keeps the plain path.
kernel_dir(op::AbstractKernelOp, k::AbstractKernel; target::AbstractString=current_target()) =
    isempty(target) ? joinpath(cache_dir(op), String(kernelname(k))) :
    joinpath(cache_dir(op), String(target), String(kernelname(k)))
manifest_path(op, k; target::AbstractString=current_target()) = joinpath(kernel_dir(op, k; target), "manifest.toml")

# Key elements are Symbols, integers or Bools; `repr` round-trips them through `Meta.parse`.
_key_str(key::Tuple) = [repr(x) for x in key]
_key_parse(strs) = Tuple(begin
    x = Meta.parse(s)
    x isa QuoteNode ? x.value : x
end for s in strs)

"""
    add_variant!(op, kernel, key; binaries...) -> Variant

Register the variant for `key`: one [`KernelBinary`](@ref) per launch name (`fwd=…, dq=…`). Each binary is
COPIED into the kernel's cache directory (so it survives its original moving) and the manifest is
rewritten. The copy is named `<launch name>_<id>.<ext>`, `id` being the first 8 hex digits of a hash
of the file's contents, `key` and the launch name: a different binary never takes a registered one's
file. Binaries no variant names any more are removed.
"""
function add_variant!(op::AbstractKernelOp, k::AbstractKernel, key::Tuple; binaries...)
    isempty(binaries) && throw(ArgumentError("a variant needs at least one binary"))
    return _add_variant!(op, k, key, binaries, current_target())
end

function _add_variant!(op::AbstractKernelOp, k::AbstractKernel, key::Tuple, binaries, target::AbstractString)
    dir = kernel_dir(op, k; target)
    mkpath(dir)
    v = Variant()
    for (name, b) in pairs(binaries)
        isfile(b.file) || throw(ArgumentError("binary `$(b.file)` does not exist"))
        id = first(string(hash(read(b.file), hash((key, name))); base=16, pad=16), 8)
        dst = joinpath(dir, "$(name)_$(id)$(lowercase(splitext(b.file)[2]))")
        abspath(b.file) == abspath(dst) || cp(b.file, dst; force=true)
        v[name] = KernelBinary(dst, b.entry, b.threadgroup, b.params, b.abi, b.tile, b.target, b.shared_mem)
    end
    variants(op, k; target)[key] = v
    save_manifest(op, k; target)
    return v
end

# No key means empty Tuple:
add_variant!(op::AbstractKernelOp, k::AbstractKernel; kwargs...) = 
    add_variant!(op, k, (); kwargs...)

# 2: `abi` and `target` per binary replace `is_ka` (a version-1 manifest, with `is_ka`, still loads).
const MANIFEST_VERSION = 2

function save_manifest(op, k; target::AbstractString=current_target())
    vs = variants(op, k; target)
    tbl = Dict{String,Any}("format_version" => MANIFEST_VERSION, "source_tag" => source_tag(k), "variant" => [
        Dict{String,Any}("key" => _key_str(key), "binary" => [
            Dict{String,Any}("name" => String(n), "file" => basename(b.file), "entry" => b.entry,
                "threadgroup" => b.threadgroup, "tile" => collect(b.tile), "abi" => String(b.abi),
                "target" => b.target, "shared_mem" => b.shared_mem, "params" => Dict(String(p) => x for (p, x) in pairs(b.params)))
            for (n, b) in sort(collect(v); by=first)])
        for (key, v) in sort(collect(vs); by=string ∘ first)])
    open(io -> TOML.print(io, tbl), manifest_path(op, k; target), "w")
    # Binaries no variant names any more (a re-registered variant's predecessor) are removed. Only
    # regular files: a package may keep its own data beside a kernel's binaries in a subdirectory
    # (a tuning calibration, say).
    keep = Set(basename(b.file) for v in values(vs) for b in values(v))
    kdir = kernel_dir(op, k; target)
    for f in readdir(kdir)
        path = joinpath(kdir, f)
        (f == "manifest.toml" || f in keep || !isfile(path)) || rm(path; force=true)
    end
end

# Never throws: a missing, corrupt or stale manifest is simply no variants — a kernel that can build
# its own (`build!`) rebuilds, any other is registered again.
function load_manifest(op, k; target::AbstractString=current_target())
    out = Dict{Tuple,Variant}()
    path = manifest_path(op, k; target)
    isfile(path) || return out
    try
        raw = TOML.parsefile(path)
        get(raw, "source_tag", "") == source_tag(k) || return out
        dir = dirname(path)
        for vd in get(raw, "variant", Any[])
            v = Variant()
            for bd in vd["binary"]
                file = joinpath(dir, bd["file"])
                isfile(file) || (v = nothing; break)
                abi = haskey(bd, "abi") ? Symbol(bd["abi"]) : (get(bd, "is_ka", false) ? :ka : :explicit)
                v[Symbol(bd["name"])] = KernelBinary(file, bd["entry"], bd["threadgroup"],
                    NamedTuple(Symbol(p) => x for (p, x) in bd["params"]), abi,
                    get(bd, "tile", (bd["threadgroup"], 1, 1)), get(bd, "target", ""), get(bd, "shared_mem", 0))
            end
            isnothing(v) || (out[_key_parse(vd["key"])] = v)
        end
    catch
    end
    return out
end
