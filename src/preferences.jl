"""
    @default_kernel Op kernel₁ kernel₂ ...

Give op type `Op` its default kernel: the one its package's preference `"kernel.<opname>"` names (set
with [`set_default_kernel!`](@ref)), else `kernel₁`. The kernels listed are the candidates; their
[`kernelname`](@ref)s are what the preference holds.

Use it at the top level of the package declaring `Op`, after the kernels' `kernelname` methods:

    KernelDispatch.@default_kernel Softmax OnlineSoftmax() TwoPassSoftmax()

Outside a package (a script, `Main`) there are no preferences, and the default is `kernel₁`.
"""
macro default_kernel(optype, kernels...)
    isempty(kernels) && throw(ArgumentError("`@default_kernel` needs at least one kernel"))
    T = esc(optype)
    ks = Expr(:tuple, map(esc, kernels)...)
    choice = esc(gensym(:default_kernel))
    cands = esc(gensym(:default_candidates))
    return quote
        const $cands = $ks
        const $choice = $(GlobalRef(@__MODULE__, :_pick_default))($__module__, $T(), $cands)
        $(GlobalRef(@__MODULE__, :default_candidates))(::$T) = $cands
        $(GlobalRef(@__MODULE__, :default_kernel))(::$T) = $choice
    end
end

"""`default_candidates(op) -> Tuple`: the kernels [`@default_kernel`](@ref) lists for `op`. Default `()`."""
default_candidates(::AbstractKernelOp) = ()

pref_key(op::AbstractKernelOp) = "kernel.$(opname(op))"

# The package `m` belongs to, if any: preferences are keyed by its UUID.
function _package_uuid(m::Module)
    root = Base.moduleroot(m)
    root === Main && return nothing
    return Base.PkgId(root).uuid
end

function _pick_default(m::Module, op::AbstractKernelOp, cands::Tuple)
    isnothing(_package_uuid(m)) && return first(cands)
    
    name = Preferences.load_preference(m, pref_key(op), nothing)    
    isnothing(name) && return first(cands)
    
    i = findfirst(k -> kernelname(k) === Symbol(name), cands)
    isnothing(i) || return cands[i]
    
    @warn "Preference `$(pref_key(op))` of $(Base.moduleroot(m)) names kernel `:$name`, which is not " *
        "a default candidate of op `:$(opname(op))` ($(_names(map(kernelname, cands)))); using " *
        "`:$(kernelname(first(cands)))`."

    return first(cands)
end

function _pref_owner(o::AbstractKernelOp)
    m = parentmodule(typeof(o))
    isnothing(_package_uuid(m)) && throw(ArgumentError(
        "op `:$(opname(o))` is not declared in a package, so it has no preferences"
    ))
    return Base.moduleroot(m)
end

"""
    set_default_kernel!(op, kernel_name)

Persist `kernel_name` as `op`'s default: a compile-time preference of the package declaring `op`,
written to the active project's `LocalPreferences.toml`. It takes effect in the next session (the
package and its dependents re-precompile once, with the kernel baked in); [`use_kernel!`](@ref)
switches the current one. `kernel_name` must be one of the op's [`@default_kernel`](@ref) candidates.
"""
function set_default_kernel!(op, kname::Symbol)
    o = _op(op)
    names = map(kernelname, default_candidates(o))
    kname in names || throw(ArgumentError(
        "kernel `:$kname` is not a default candidate of op `:$(opname(o))`. Candidates: " *
        _names(names) * " (listed in its `KernelDispatch.@default_kernel`)"
    ))
    owner = _pref_owner(o)
    Preferences.set_preferences!(owner, pref_key(o) => String(kname); force=true)
    
    @info "Default kernel of op `:$(opname(o))` set to `:$kname` in $(owner)'s preferences. It takes " *
        "effect after a restart (re-precompiling $owner and its dependents); `use_kernel!` switches " *
        "this session."
    
    return kname
end

"""`clear_default_kernel!(op)`: remove the preference [`set_default_kernel!`](@ref) wrote."""
function clear_default_kernel!(op)
    o = _op(op)
    Preferences.delete_preferences!(_pref_owner(o), pref_key(o); force=true)
    return nothing
end
