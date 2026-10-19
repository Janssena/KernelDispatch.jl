# A string is a StringAttr (the legacy calling conventions); a dictionary is a DictionaryAttr, the form the
# typed-FFI convention (`api_version = 4`) takes its attributes in.
_config_attr(s::AbstractString) = Reactant.MLIR.IR.Attribute(s)
_config_attr(d::AbstractDict) = Reactant.MLIR.IR.Attribute(d)
_config_attr(a::Reactant.MLIR.IR.Attribute) = a

# `output_operand_aliases`: pairs `(result, operand)`, both 0-based. The result then shares the operand's buffer
# (an in-place update, or an output that is handed to the kernel as an input too); `nothing` when there are none.
function _aliases_attr(aliases, nresults::Integer)
    isempty(aliases) && return nothing
    ctx = Reactant.MLIR.IR.current_context()
    attrs = map(aliases) do (result, operand)
        out_tup = Ref{Int64}(result)
        Reactant.MLIR.IR.Attribute(
            GC.@preserve ctx out_tup Reactant.MLIR.API.stablehloOutputOperandAliasGet(
                ctx, nresults == 1 ? 0 : 1, pointer_from_objref(out_tup), operand, 0, C_NULL))
    end
    return Reactant.MLIR.IR.Attribute(attrs)
end

"""
    _embed_custom_call(call_target_name, operands, result_shapes; backend_config,
        api_version=1, reverse=nothing, active_operands=nothing, label=call_target_name) -> results

Emit a `stablehlo.custom_call` naming `call_target_name`, with the given `backend_config` string
attribute. The op itself carries no meaning tied to any one backend or dispatcher —
`"mps.metal_kernel_lib"` (below) is one Metal convention built on it, but a CUDA or ROCm custom
call, or a user's own, attaches through the same seam by passing a different `call_target_name`.

`backend_config` is a string (a StringAttr; XLA's custom-call plugins that key off JSON, as jax-mps' does,
expect it pre-serialised, since the JSON schema is entirely the target's own convention to define) or, for
the typed-FFI convention (`api_version = 4`), a dictionary of attributes.

`result_eltypes` is the element type of each result, one per entry of `result_shapes`; it
defaults to all `Float32`. The declared type IS the buffer the runtime allocates, so it must match
what the kernel writes: a kernel storing `Float16` into a result declared `Float32` fills half the
bytes and is read back as garbage, with no error.

`output_operand_aliases` is a list of `(result, operand)` pairs (0-based) whose buffers are shared.

`reverse`, given, is a VJP attached via [`_attach_reverse_rule!`](@ref) — see its docstring for the
calling convention. `label` only affects the generated reverse function's (gensym'd) name, for
readability in an MLIR dump; it defaults to `call_target_name` but is worth overriding when one
target name is shared by several logically distinct calls (`"mps.metal_kernel_lib"` dispatches
every kernel in a `.metallib` under that one target, for instance).
"""
function _embed_custom_call(call_target_name::AbstractString, operands, result_shapes;
        backend_config, api_version::Integer=1, reverse=nothing,
        active_operands=nothing, label::AbstractString=call_target_name,
        result_eltypes=ntuple(Returns(Float32), length(result_shapes)),
        output_operand_aliases=Tuple{Int,Int}[]
)
    length(result_eltypes) == length(result_shapes) || throw(ArgumentError(
        "$(length(result_eltypes)) result eltypes for $(length(result_shapes)) result shapes"
    ))

    rts = [
        Reactant.Ops.mlir_type(Reactant.TracedRArray{E,length(sh)}, Int64[sh...])
        for (E, sh) in zip(result_eltypes, result_shapes)
    ]
    
    op = Reactant.MLIR.Dialects.stablehlo.custom_call([x.mlir_data for x in operands];
        result_0=rts,
        call_target_name=Reactant.MLIR.IR.Attribute(call_target_name),
        backend_config=_config_attr(backend_config),
        api_version=Reactant.MLIR.IR.Attribute(Int32(api_version)),
        output_operand_aliases=_aliases_attr(output_operand_aliases, length(result_shapes))
    )
    
    results = ntuple(length(result_shapes)) do i
        Reactant.TracedRArray{result_eltypes[i],length(result_shapes[i])}(
            (), Reactant.MLIR.IR.result(op, i), result_shapes[i])
    end
    
    isnothing(reverse) ||
        _attach_reverse_rule!(op, label, operands, results, reverse, active_operands)
    
    return results
end

"""
    _attach_reverse_rule!(op, name, operands, results, vjp, active_operands)

Give a `stablehlo.custom_call` a reverse-mode derivative, by tracing `vjp` into its own
`func.func` and pointing the call site at it with `enzyme.reverse`.

Enzyme cannot see inside a custom call, so the rule has to be supplied from here. It lives on the
call site rather than in a registry keyed on `call_target_name`, because one target name can
dispatch several logically distinct kernels — `"mps.metal_kernel_lib"` names whichever kernel a
call's `backend_config` JSON points at, so multiple calls sharing that target name can have
completely different adjoints. `label` in [`_embed_custom_call`](@ref) exists to keep those apart
in the generated function name.

`vjp` is called as

    vjp(primal_operands..., primal_results..., result_cotangents...)
        -> one cotangent per entry of `active_operands`, in that order

`active_operands` is 0-based, and names the operands that can carry a gradient; anything left out
is a constant, which is what the boolean mask needs. The cotangent of every result is passed even
when it is meaningless — the `lse` residual's is a zero Enzyme materialises — so the argument list
stays positional.

Enzyme verifies this signature against the call site before it fires, so a mismatch is a compile
error rather than a wrong gradient. That check is the reason this is safe to write by hand.
"""
function _attach_reverse_rule!(op, name, operands, results, vjp, active_operands)
    # Types are what matter for the traced signature; the values are placeholders. Each slot needs
    # its OWN wrapper object, though: the tracer keys on object identity, so passing a value twice
    # -- the primal result and again as its cotangent -- collapses the two into a single block
    # argument and silently drops an argument from the signature.
    args = (_fresh.(operands)..., _fresh.(results)..., _fresh.(results)...)
    revfn = Reactant.TracedUtils.make_mlir_fn(
        vjp, args, (), string(gensym(name * "_reverse")), false;
        do_transpose=false,          # the signature has to match the call site exactly
        args_in_result=:result,      # return the cotangents only, not the arguments back
        argprefix=gensym("revarg"),
        resprefix=gensym("revres"),
        resargprefix=gensym("revresarg"),
    ).f
    # Read the symbol back rather than reusing the name we asked for: inserting into the module's
    # symbol table can uniquify it, and a stale reference here would be a dangling symbol.
    sym = String(Reactant.MLIR.IR.getattr(revfn, "sym_name"))
    
    Reactant.MLIR.IR.setattr!(op, "enzyme.reverse",
        Reactant.MLIR.IR.FlatSymbolRefAttribute(sym))

    Reactant.MLIR.IR.setattr!(op, "enzyme.active_operands",
        Reactant.MLIR.IR.Attribute(Int64[active_operands...]))
        
    return nothing
end

# A new wrapper around the same MLIR value. Note `eltype` of a TracedRArray is a TracedRNumber, so
# the scalar type has to come off the type parameters rather than from `eltype`.
_fresh(x::Reactant.TracedRArray{T,N}) where {T,N} =
    Reactant.TracedRArray{T,N}((), x.mlir_data, size(x))
