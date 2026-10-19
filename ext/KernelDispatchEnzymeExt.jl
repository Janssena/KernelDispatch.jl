module KernelDispatchEnzymeExt

# One reverse rule for every op: on `forward(op, kernel, args...)`, the call `op(args...)` makes on a
# device and the one a caller may make directly. Enzyme therefore never enters a `forward` overload
# (whose plain Julia it could not differentiate on Metal) or a binary; the reverse is the op's own
# `backward`.
#
# Enzyme + Metal is not a training stack: the rule fires over `MtlArray`s and its gradients are
# exact, but Enzyme cannot differentiate ordinary Metal.jl code around it. Its payoff is CUDA; on
# Apple hardware, train through Reactant.

using KernelDispatch
using Enzyme
using Enzyme: EnzymeRules

const KD = KernelDispatch

# Which kernel runs is a table lookup, not something to differentiate.
EnzymeRules.inactive(::typeof(KD.selected_kernel), args...) = nothing

_shadow(x::AbstractArray) = zero(x)
_shadow(x) = nothing
_shadows(out::Tuple) = map(_shadow, out)
_shadows(out) = _shadow(out)

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig,
        ::Enzyme.Const{typeof(KD.forward)}, ::Type{RT},
        op::Enzyme.Const{<:KD.AbstractKernelOp}, k::Enzyme.Const{<:KD.AbstractKernel},
        args::Vararg{Enzyme.Annotation,N}) where {RT,N}
    vals = map(a -> a.val, args)
    out = KD.forward(op.val, k.val, vals...)
    primal = EnzymeRules.needs_primal(config) ? out : nothing
    shadow = EnzymeRules.needs_shadow(config) ? _shadows(out) : nothing
    return EnzymeRules.augmented_rule_return_type(config, RT)(primal, shadow, (vals, out, shadow))
end

# Gradients ACCUMULATE into the shadows (one argument can be reached by several paths). A
# `Duplicated` argument's gradient goes into its shadow (`nothing` returned); an `Active` one is a
# scalar parameter, which is not differentiated through a kernel (zero).
function EnzymeRules.reverse(config::EnzymeRules.RevConfig,
        ::Enzyme.Const{typeof(KD.forward)}, dret, tape,
        op::Enzyme.Const{<:KD.AbstractKernelOp}, k::Enzyme.Const{<:KD.AbstractKernel},
        args::Vararg{Enzyme.Annotation,N}) where {N}
    vals, out, shadow = tape
    cots = dret isa Union{Tuple,AbstractArray} ? dret : shadow
    isnothing(cots) && error("no cotangent available for op `:$(KD.opname(op.val))`")
    grads = KD.backward(op.val, k.val, vals, out, cots)
    for (a, g) in zip(args, grads)
        (a isa Enzyme.Duplicated && !isnothing(g)) && (a.dval .+= g)
    end
    return (nothing, nothing, map(a -> a isa Enzyme.Active ? zero(a.val) : nothing, args)...)
end

end # module
