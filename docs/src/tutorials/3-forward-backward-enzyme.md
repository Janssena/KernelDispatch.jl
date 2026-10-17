# 3. Custom forward and backward, and gradients with Enzyme

## What calling an op does

```julia
SDPA()(q, k, v)
```

1. On **host arrays** it runs `host(op, q, k, v)`, plain Julia.
2. On **device arrays** (or traced ones) it runs `forward(op, q, k, v)`, which calls
   `forward(op, kernel, q, k, v)` with the selected kernel.
3. `forward(op, kernel, args...)` is yours to define, per kernel. There is no default: a binary's
   calling convention can't be guessed, and passing the op's arguments through unchanged would bind
   them wrongly. A kernel without one throws, saying what to define.

`forward(op, kernel, args...)` is also where gradients attach: Enzyme sees it as one opaque call and
runs your `backward` for its reverse (below).

## Custom forward: building a binary's argument list

`forward(op, kernel, args...)` is plain Julia that turns the op's arguments into the binary's. In
`SDPAOps` the two kernels take different argument lists, so each kernel gets its own method, sharing
the preparation:

```julia
_sizes(q, k, v) = (Int32(size(q, 1)), Int32(size(q, 2)), Int32(size(k, 2)), Int32(size(v, 1)))
_scale(q) = inv(sqrt(Float32(size(q, 1))))

# The Metal binary: inputs, the output, n, m, d, dv, scale.
function forward(op::SDPA, kern::FlashAttn, q, k, v)
    d, n, m, dv = _sizes(q, k, v)
    return only(call_binary(op, kern, :fwd, q, k, v, OutArray(Float32, dv, n), n, m, d, dv, _scale(q);
        extent=n))
end

# The KA binary: the output first, then d, n, m, dv, scale.
function forward(op::SDPA, kern::SDPAKA, q, k, v)
    d, n, m, dv = _sizes(q, k, v)
    return only(call_binary(op, kern, :fwd, OutArray(Float32, dv, n), q, k, v, d, n, m, dv, _scale(q);
        extent=n))
end
```

What a forward is for:

- **Converting scalars** to the exact types the binary reads (`Int32`, `Float32`).
- **Deriving values** the binary needs but the caller shouldn't pass (sizes, the scale).
- **Ordering** the arguments as the binary takes them, outputs marked with `OutArray`.
- **Sizing the launch**: `extent=` is the problem size, which each binary divides by its own `tile`.
- **Reshaping** (`reshape`, `permutedims`, views) and computing anything else in ordinary Julia. A
  device view or wrapper the launcher can't bind directly is copied to a plain device array once.
- **Shaping the result**: `call_binary` returns the outputs in `OutArray` order as a tuple. Return
  what the op's callers should see. Here that is the one matrix, hence `only(…)`.

A forward can launch several binaries in turn (by launch name: `call_binary(op, kern, :partial, …)`,
then `:combine`), as long as each is registered in the variant.

## Custom backward

```julia
backward(op, kernel, args, outs, cotangents) -> one gradient per argument
```

- `args` is the tuple of the op's arguments, exactly as `forward` received them.
- `outs` is what `forward` returned, and `cotangents` has the same shape (here: a matrix each).
- Return one gradient per argument, in order: an array like the argument, or `nothing` for one with
  no gradient (an integer, a flag).

There is no default. Without one, differentiating the op throws, and the error lists the variant's
launches when there are several, one of which may be the backward binary to call.

### A. In plain Julia

`SDPAOps` computes the backward with array operations, which run on the device for device arrays.
One method serves both kernels:

```julia
const SDPAKernels = Union{FlashAttn,SDPAKA}

function backward(::SDPA, ::SDPAKernels, (q, k, v), o, dout)
    scale = _scale(q)
    p = _softmax((k' * q) .* scale)            # recomputed: o alone doesn't determine it
    dp = v' * dout
    ds = p .* (dp .- sum(p .* dp; dims=1))
    return ((k * ds) .* scale, (q * ds') .* scale, dout * p')    # dq, dk, dv
end
```

### B. With a backward binary

When the gradient is a kernel too, register it in the same variant under its own launch name and
call it from `backward` like any other binary. A sketch, for a hypothetical backward binary with the
argument list below:

```julia
add_variant!(SDPA(), FlashAttn(), (:Float32,); fwd=fwd_binary, bwd=bwd_binary)

function backward(op::SDPA, kern::FlashAttn, (q, k, v), o, dout)
    d, n, m, dv = _sizes(q, k, v)
    dq, dk, dv_ = call_binary(op, kern, :bwd, q, k, v, o, dout,
        OutArray(Float32, d, n), OutArray(Float32, d, m), OutArray(Float32, dv, m),
        n, m, d, dv, _scale(q); key=(nameof(KernelDispatch.realtype(eltype(q))),), extent=m)
    return (dq, dk, dv_)
end
```

`variant_key` sees each launch's own argument list, so the `:bwd` call passes `key=` rather than adding
a second `variant_key` method for the longer list. `extent=m` (one item per key, say) is divided by the
`:bwd` binary's own `tile`. `KernelBinary.is_ka` is per binary, so a KA-compiled forward and a Triton
backward can share a variant.

## Gradients with Enzyme

KernelDispatch' Enzyme extension adds a single reverse rule on `forward(op, kernel, args...)`, for every op / 
kernel. Enzyme never looks inside your forward or the binary. Its reverse calls your `backward` and
accumulates the gradients into the `Duplicated` shadows of the arguments.

### On host arrays

The host path is plain Julia, so Enzyme differentiates it directly:

```julia
using Enzyme
loss(q, k, v, w) = sum(SDPA()(q, k, v) .* w)

dq, dk, dv = zero(q), zero(k), zero(v)
Enzyme.autodiff(Reverse, loss, Active, Duplicated(q, dq), Duplicated(k, dk), Duplicated(v, dv), Const(w))
```

Pass data the loss uses as arguments (`Const(w)`) rather than capturing it in a closure, which
Enzyme may refuse to treat as constant.

### On GPU arrays

Enzyme can differentiate the op call on `MtlArray`s through the rule, but not ordinary Metal.jl code
around it (a broadcast, a `sum`). So differentiate the op call alone, in split mode, and seed the
output's cotangent yourself:

```julia
f(q, k, v) = SDPA()(q, k, v)

function gpu_grads(qd, kd, vd, w)
    gq, gk, gv = zero(qd), zero(kd), zero(vd)
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f)}, Duplicated,
        Duplicated{typeof(qd)}, Duplicated{typeof(kd)}, Duplicated{typeof(vd)})
    args = (Const(f), Duplicated(qd, gq), Duplicated(kd, gk), Duplicated(vd, gv))
    tape, o, dout = fwd(args...)      # forward: o, and a zero shadow of it
    dout .= MtlArray(w)               # the cotangent of o, e.g. ∂loss/∂o for loss = sum(o .* w)
    rev(args..., tape)                # reverse: your backward, accumulated into gq, gk, gv
    return o, gq, gk, gv
end
```

With the same `w`, `gq`, `gk`, `gv` match the host gradients above. That comparison is a good test of
a backward: `SDPAOps`' test suite does it for both kernels.

**Switch kernels before building the thunk, at top level.** An Enzyme thunk is compiled once and
keeps the kernel selected when it was built. Calling it after a switch throws the stale-switch error
from [tutorial 1](1-switching-kernels.md), so switch at top level and then build it:

```julia
use_kernel!(:sdpa, :ka)               # top level
o, gq, gk, gv = gpu_grads(qd, kd, vd, w)
```

### Under Reactant

In a Reactant-compiled program the forward is a custom call that names the binary. Enzyme-JAX can't
differentiate a custom call yet (EnzymeAD/Enzyme#2516), so a gradient traced *through* the op throws
instead of silently giving a wrong answer. Call `backward` directly on the traced arrays instead; it is
ordinary traced code, and a `:bwd` binary becomes a custom call of its own. Once Enzyme-JAX supports
it, `KernelDispatch.KERNEL_IN_TRACED_AUTODIFF[] = true` lifts the check.
