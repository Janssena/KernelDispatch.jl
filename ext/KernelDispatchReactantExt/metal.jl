_le_bytes(x) = "[" * join(Int.(reinterpret(UInt8, [x])), ",") * "]"

"""
    _metal_kernel_lib(name, libpath, operands, result_shapes, layout; grid,
        result_eltypes=all Float32) -> results

[`_embed_custom_call`](@ref) targeting `"mps.metal_kernel_lib"`, jax-mps' dispatcher for a named
kernel inside a pre-compiled `.metallib`. `layout` is the kernel's argument table in order, as
`(:in, i)`, `(:out, i)`, `(:out_zero, i)` or `(:bytes, payload)`. `:out_zero` is an output jax-mps
zero-fills before dispatch (`"zero_init"`), for a kernel that accumulates into it.

Two things are easy to get wrong and are silent when wrong: the `backend_config` JSON must include
`api_version = 1`, and dispatch must be by *threadgroups*, matching how the compiled kernel indexes
by `threadgroup_position_in_grid`.
"""
function _metal_kernel_lib(name, libpath, operands, result_shapes, layout; grid,
        threadgroup=1024, reverse=nothing, active_operands=nothing, shared_mem::Integer=0,
        result_eltypes=ntuple(Returns(Float32), length(result_shapes))
)
    any(e -> first(e) === :inout, layout) && throw(ArgumentError(
        "an `InOutArray` argument cannot be traced on Metal: jax-mps' kernel call cannot alias a result to an operand " *
        "(it starts every output from zeros or leaves it uninitialised); pass the array as an input and the " *
        "updated one as an `OutArray`, with the kernel copying the old values"))
    slots = map(enumerate(layout)) do (i, entry)
        kind, val = entry
        s = i - 1
        kind === :bytes ? "{\"slot\":$s,\"kind\":\"bytes\",\"bytes\":$val}" :
        kind === :out_zero ? "{\"slot\":$s,\"kind\":\"output\",\"arg\":$val,\"zero_init\":true}" :
        "{\"slot\":$s,\"kind\":\"$(kind === :in ? "input" : "output")\",\"arg\":$val}"
    end

    cfg = "{\"name\":\"$name\",\"metallib_path\":\"$libpath\"," *
          "\"grid\":[$(grid[1]),$(grid[2]),$(grid[3])]," *
          "\"threadgroup\":[$(threadgroup),1,1]," *
          "\"dispatch\":\"threadgroups\",\"buffers\":[" * join(slots, ",") * "]}"

    return _embed_custom_call("mps.metal_kernel_lib", operands, result_shapes;
        backend_config=cfg, reverse, active_operands, label=name, result_eltypes)
end

"""
    KernelDispatch.custom_call(::MetalBackendTag, entry, path, operands, result_shapes, layout; kw...)

The Metal embedding: [`_metal_kernel_lib`](@ref) (`mps.metal_kernel_lib`).
"""
KD.custom_call(::KD.MetalBackendTag, entry, path, operands, result_shapes, layout; kw...) =
    _metal_kernel_lib(entry, path, operands, result_shapes, layout; kw...)