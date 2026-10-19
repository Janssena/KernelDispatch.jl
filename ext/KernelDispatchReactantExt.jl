module KernelDispatchReactantExt

# Note: A gradient traced THROUGH such a call throws (`traced_autodiff`): Enzyme-JAX cannot differentiate
# a custom call yet (EnzymeAD/Enzyme#2516). `_attach_reverse_rule!` (custom_call.jl) is kept for when
# it can. Calling an op's `backward` on traced arrays works today: it is ordinary traced code.

using KernelDispatch
using Reactant: Reactant, AnyTracedRArray, TracedRNumber

const KD = KernelDispatch

KD.realtype(::Type{TracedRNumber{T}}) where T = T
KD.traced_autodiff(::AnyTracedRArray) = Reactant.WITHIN_AUTODIFF[]

"""
    backend_of(::AnyTracedRArray)

A traced array is the same Julia type whether it will run on CPU or GPU — only the compiling client
knows, so this asks it. An unrecognised platform maps to `UnknownBackend`, which falls back.
"""
KD.backend_of(::Union{Reactant.AnyConcreteRArray,AnyTracedRArray}) = reactant_backend()

function reactant_backend()
    platform = try
        Reactant.XLA.platform_name(Reactant.XLA.default_backend())
    catch
        return KD.UnknownBackend()
    end
    platform = lowercase(platform)
    
    platform == "mps" && return KD.MetalBackendTag()
    platform in ("cuda", "gpu") && return KD.CUDABackendTag()
    platform == "rocm" && return KD.ROCmBackendTag()
    platform == "cpu" && return KD.CPUBackend()
    return KD.UnknownBackend()
end

KD.device_zeros(::AnyTracedRArray, ::Type{T}, dims::Integer...) where {T} =
    _mat(Reactant.Ops.fill(zero(T), Int.(dims)))

include("KernelDispatchReactantExt/custom_call.jl")
include("KernelDispatchReactantExt/metal.jl")
include("KernelDispatchReactantExt/cuda.jl")

_mat(x) = Reactant.TracedUtils.materialize_traced_array(x)

"""
    _rev(x)

Reverse a traced array's dimensions — the ONLY layout conversion the traced path performs, and a
free one. Reactant lays a traced value out row-major over its Julia size; a kernel addresses the
column-major Julia layout. The two disagree only about which END of the dimension list is written
first, so reversing is a pure relabelling XLA folds into the parameter or result layout.
"""
_rev(x) = _mat(permutedims(x, ntuple(i -> ndims(x) - i + 1, ndims(x))))

"""
    execute(::AnyTracedRArray, l::Launch) -> Tuple

The traced binder: one `KD.custom_call` (dispatched on the compiling client's backend; on Metal,
`mps.metal_kernel_lib`). Its buffer table follows
`(prelude..., args..., extras...)` in order: a bits value (the prelude, a scalar) is a `bytes` slot, an array an
operand, an `OutArray` a result (`zero` ones zero-filled by jax-mps), an `InOutArray` an operand with a result aliased to it,
a `Val` nothing. Arrays cross
through `_rev`.
"""
function KD.execute(::AnyTracedRArray, l::KD.Launch)
    be = reactant_backend()
    # Before any argument is converted: a binary the backend cannot launch from a trace should say so, not fail
    # on the first value that has no byte form.
    reason = KD.traced_support(be, l.binary)
    reason === nothing || throw(ArgumentError(reason))
    layout = Any[(:bytes, _le_bytes(p)) for p in l.prelude]
    operands, shapes, eltypes = Any[], Any[], DataType[]
    for a in (l.args..., l.extras...)
        if a isa KD.OutArray
            push!(layout, (a.zero ? :out_zero : :out, length(shapes)))
            push!(shapes, Base.reverse(a.dims))
            push!(eltypes, eltype(a))
        elseif a isa KD.InOutArray
            # An array the kernel updates: the operand, and a result aliased to it (the updated array).
            x = _rev(_mat(a.array))
            push!(layout, (:inout, (length(operands), length(shapes))))
            push!(operands, x)
            push!(shapes, size(x))
            push!(eltypes, Reactant.unwrapped_eltype(a.array))   # not `eltype`: that is a TracedRNumber
        elseif a isa AbstractArray
            push!(layout, (:in, length(operands)))
            push!(operands, _rev(_mat(a)))
        elseif !(a isa Val)
            push!(layout, (:bytes, _le_bytes(a)))
        end
    end
    b = l.binary
    # A Triton binary's hidden trailing pointers: null words after the arguments (see `triton_binary`).
    append!(layout, ((:bytes, _le_bytes(UInt64(0))) for _ in 1:KD.trailing_pointers(b)))
    results = KD.custom_call(be, b.entry, b.file, Tuple(operands), shapes, layout;
        grid=l.grid, threadgroup=b.threadgroup, shared_mem=b.shared_mem, result_eltypes=Tuple(eltypes))
    return Tuple(map(_rev, results))
end

function __init__()
    KD._TRACING[] = () -> Reactant.within_compile()
    return nothing
end

end # module
