"""
    OutArray(T, dims...; zero=false)

An output argument: a fresh `T` array of size `dims`, allocated eagerly or declared as a result in a
trace, in the argument's place. `zero = true` zero-fills it first, for a kernel that accumulates.
"""
struct OutArray{T,N}
    dims::NTuple{N,Int}
    zero::Bool
end

OutArray(::Type{T}, dims::Integer...; zero::Bool=false) where {T} =
    OutArray{realtype(T),length(dims)}(Int.(dims), zero)

OutArray(::Type{T}, dims::Tuple; zero::Bool=false) where {T} = OutArray(T, dims...; zero)

Base.eltype(::OutArray{T}) where {T} = T

"""
    InOutArray(x::AbstractArray)

An argument the kernel both reads and updates in place (`y += a * x`, a read-modify-write output), bound where the
array would be. It is returned among the outputs, in [`OutArray`](@ref) order, as the UPDATED array:

- **eagerly** it is `x` itself, written by the kernel (so `x` must be a dense device array, not a view or wrapper that
  would have to be copied: that is refused);
- **in a trace** the array is the call's operand and its result is aliased to it (`output_operand_aliases`), so the kernel
  sees `x`'s values, and the returned array is the updated one. XLA copies `x` first if it is still used afterwards, so
  the program stays functional: `x` itself is not changed.

An [`OutArray`](@ref) starts as a fresh (zero) buffer, an `InOutArray` starts as `x`. Not supported on Metal under a trace
(jax-mps cannot alias a result to an operand).
"""
struct InOutArray{A<:AbstractArray}
    array::A
end

# The first argument that is an array (an `InOutArray` counts as its array): it decides the backend. Recursive over the
# argument tuple, so the result type infers (a loop over a heterogeneous tuple would not).
_first_array(xs::Tuple) = _first_array_(xs...)
_first_array_() = nothing
_first_array_(x::AbstractArray, rest...) = x
_first_array_(x::InOutArray, rest...) = x.array
_first_array_(x, rest...) = _first_array_(rest...)

"""
    (op::AbstractKernelOp)(args...)

Run `op`: [`host`](@ref) on host arrays (and, with a warning, on a device no kernel exists for);
[`forward`](@ref) with the selected kernel on device arrays. On a device there is no silent fallback:
a missing kernel is an error, not a 10–100× slowdown behind a correct answer.
"""
function (op::AbstractKernelOp)(args...)
    a = _first_array(args)
    isnothing(a) && throw(ArgumentError("op `:$(opname(op))` was called without any array argument"))
    be = backend_of(a)
    be isa CPUBackend && return host(op, args...)
    if be isa UnknownBackend
        @warn "Unrecognised device backend for op `:$(opname(op))`; falling back to its host path." maxlog = 1
        return host(op, args...)
    end

    check_traced_autodiff(a, "op `:$(opname(op))`")

    return forward(op, args...)
end

"""
    call_binary(op, kernel, name, args...; key=variant_key(op, kernel, args...), extent) -> Tuple

Launch binary `name` of the variant for `key`, binding `(args..., extras(op, kernel, binary)...)`
IN THE ORDER GIVEN: an [`OutArray`](@ref) is allocated (or a traced result), an array is bound in place,
an [`InOutArray`](@ref) is an array the kernel updates (returned too), a number by value, a `Val` not at all. Returns the
outputs, in the order of the `OutArray`s and `InOutArray`s. Getting the order
right is the caller's job.

It launches `cld.(extent, tile)` threadgroups: the problem size `extent` (an integer or a tuple of up to
three axes, required) divided by what one threadgroup of the binary covers, its `tile`.

The op-level layer: it finds the binary and its [`extras`](@ref), then [`run_binary`](@ref) does the
rest. The outputs are picked from `args` alone: `extras` (which may read `KernelBinary.params`, typed
only at run time) are only bound, so whatever they return never reaches the result's type. They must
therefore not contain `OutArray`s.
"""
function call_binary(
    op::AbstractKernelOp, k::AbstractKernel, name::Symbol, args...;
    key::Tuple=variant_key(op, k, args...), extent=nothing
)
    _check_extent(op, k, extent)
    b = _binary(op, k, name, key)
    
    return run_binary(
        b, args...; extent, extras=extras(op, k, b), label=name
    )
end

# `extent` is required: a call without one throws rather than guess.
_check_extent(op, k, extent) = isnothing(extent) && throw(ArgumentError(
    "no launch size for kernel `:$(kernelname(k))` of op `:$(opname(op))`: pass `extent=` (the " *
    "problem size, divided by the binary's `tile`) to `call_binary`"))

# The binary `name` of the variant for `key`.
function _binary(op::AbstractKernelOp, k::AbstractKernel, name::Symbol, key::Tuple)
    v = variant(op, k, key)
    return get(v, name) do
        throw(
            ArgumentError(
                "variant $key of kernel `:$(kernelname(k))` has no binary `:$name`; " *
                "it has $(sort(collect(keys(v))))"
            )
        )
    end
end

"""
    run_binary(b::KernelBinary, args...; extent, extras=(), label=Symbol(b.entry)) -> Tuple

Launch `b` on `args` and return the outputs: `execute(prepare_launch(b, args...; …))`. The
binary-level layer under [`call_binary`](@ref), which adds the registry lookup; call it directly to run
a binary that is not registered (a tuner's candidate). Eager or traced, like `call_binary`.
"""
run_binary(b::KernelBinary, args...; kwargs...) = execute(prepare_launch(b, args...; kwargs...))


"""
    Launch

One concrete launch of a [`KernelBinary`](@ref), made by [`prepare_launch`](@ref) and run by
[`execute`](@ref) or timed by [`time_binary`](@ref):

- `binary`: what is launched;
- `args`, `extras`: the arguments as they will be bound (device wrappers already densified),
  `OutArray`s still unallocated. `args` is a typed tuple, so the outputs' types infer;
- `prelude`: what the binary takes ahead of its arguments (for an `abi = :ka` binary the backend's state
  words and KA's context, else nothing);
- `groups`: the threadgroups the problem needs, `cld.(extent, tile)`;
- `grid`: the grid actually dispatched (`groups`, flattened by [`ka_grid`](@ref) for an `abi = :ka` binary);
- `label`: its name in [`record_bindings`](@ref).
"""
struct Launch{A<:Tuple,E<:Tuple,P<:AbstractVector}
    binary::KernelBinary
    args::A
    extras::E
    prelude::P
    groups::NTuple{3,Int}
    grid::NTuple{3,Int}
    label::Symbol
end

"""
    prepare_launch(b::KernelBinary, args...; extent, extras=(), label=Symbol(b.entry)) -> Launch

Turn a problem size into a concrete [`Launch`](@ref) of `b`, without running it: `cld.(extent, b.tile)`
threadgroups, the prelude and grid of the binary's ABI (KA's context and flat grid for `:ka`), device wrappers densified
([`device_dense`](@ref)). The binary need not be registered. Binding follows [`call_binary`](@ref):
`(args..., extras...)` in the order given.
"""
function prepare_launch(b::KernelBinary, args...; 
    extent=nothing, extras::Tuple=(), label::Symbol=Symbol(b.entry)
)
    isnothing(extent) && throw(ArgumentError(
        "no launch size for binary `$(b.entry)`: pass `extent=` (the problem size, divided by the " *
        "binary's `tile`)"
    ))

    proto = _first_array(args)
    isnothing(proto) && throw(ArgumentError("binary `$(b.entry)` was launched without any array argument"))
    
    be = backend_of(proto)
    be isa GPUBackend && check_binary(be, b)
    groups = cld.(_pad3(extent), b.tile)
    prelude, grid = _prelude_and_grid(be, b, groups)
    _make_dense = Base.Fix1(_dense_arg, be)

    return Launch(
        b, 
        map(_make_dense, args), 
        map(_make_dense, extras), 
        prelude, groups, grid, label
    )
end

# An argument as it will be bound: device wrappers densified. An `InOutArray` is written by the kernel, so its array must
# BE the dense one: a copy would take the update and lose it.
_dense_arg(be, x) = device_dense(be, x)
function _dense_arg(be, x::InOutArray)
    y = device_dense(be, x.array)
    y === x.array || throw(ArgumentError(
        "an `InOutArray` argument must be a dense array the kernel can write in place, not a $(typeof(x.array)) " *
        "(which would be copied first, losing the update): pass `parent(x)` or `collect` it into an array"))
    return x
end

# What `b` takes ahead of its arguments, and the grid it is dispatched on, for `groups` threadgroups, by its
# ABI. The two built-in ABIs are branched on, not dispatched, so the result infers; any other ABI is an
# extension's (`abi_prelude_and_grid`), asserted to the same type.
function _prelude_and_grid(be::KernelBackend, b::KernelBinary, groups::NTuple{3,Int})
    abi = getfield(b, :abi)
    abi === :explicit && return (Any[], groups)
    if abi === :ka     # KA's context (built by KA from this grid) ahead of the arguments, the backend's own grid
        ctx, ghost = ka_context(be, groups, b.threadgroup)
        return (ka_prelude(be, ctx, ghost), ka_grid(be, groups))
    end
    return abi_prelude_and_grid(Val(abi), be, b, groups)::Tuple{Vector{Any},NTuple{3,Int}}
end

"""
    abi_prelude_and_grid(::Val{abi}, backend, b::KernelBinary, groups) -> (prelude::Vector{Any}, grid::NTuple{3,Int})

What a binary of a non-built-in ABI (`b.abi`, neither `:explicit` nor `:ka`) takes ahead of its arguments, and the
grid it is dispatched on. An extension that supports such a binary (a Triton one, say) defines a method for its
`Val{:name}`. No default: a binary whose ABI nothing implements is refused here, not launched with a shifted
argument list.
"""
abi_prelude_and_grid(::Val{A}, be::KernelBackend, b::KernelBinary, groups) where {A} = throw(ArgumentError(
    "binary `$(b.entry)` declares ABI `:$A`, which no loaded extension implements for $(typeof(be))"))

"""
    execute(l::Launch) -> Tuple

Run a prepared [`Launch`](@ref) and return its outputs, in `OutArray` order. What running means is
the arrays' business: eager device arrays are launched (the backend extension's `execute(proto, l)`
method), traced arrays become a custom call, and inside [`record_bindings`](@ref) nothing runs.
"""
execute(l::Launch) = execute(_first_array(l.args), l)

execute(proto, ::Launch) = throw(ArgumentError(
    "no backend can execute a launch on a $(typeof(proto)); load the package for that backend"
))

"""
    bind_slots(proto, prelude, args, extras) -> (bound, slots)

The one binding rule, shared by every eager launch ([`execute`](@ref), [`time_binary`](@ref)):
each [`OutArray`](@ref) of `args` is allocated like `proto` (zero-filled if it asks) and each [`InOutArray`](@ref) is
its array, giving `bound`; `slots` are `(prelude..., bound..., extras...)` in order, a `Val` taking no slot.
"""
function bind_slots(proto, prelude, args::Tuple, extras::Tuple)
    bound = map(args) do a
        return a isa OutArray ? _alloc_out(proto, a) : a isa InOutArray ? a.array : a
    end
    
    slots = Any[prelude...]
    for a in (bound..., extras...)
        a isa Val || push!(slots, a)
    end
    return (bound, slots)
end

function _alloc_out(proto, o::OutArray{T}) where {T}
    a = similar(proto, T, o.dims)
    o.zero && fill!(a, zero(T))
    return a
end

"""
    max_threads(binary::KernelBinary; backend=device_backend()) -> Int
    max_threads(backend, binary::KernelBinary) -> Int

The compiled binary's own threads-per-threadgroup ceiling: what its register use leaves of the
device's. Dispatching over it does not raise but returns garbage AND an impossibly fast time, so
eager launches and [`time_binary`](@ref) refuse it; a tuner asks this first to skip the
candidate. The two-argument form is implemented by the backend extension.
"""
max_threads(b::KernelBinary; backend::KernelBackend=device_backend()) = max_threads(backend, b)

max_threads(be::KernelBackend, ::KernelBinary) = throw(ArgumentError(
    "no `max_threads` for backend $(typeof(be)); load the package for that backend"
))

"""
    device_time(backend, items; reps) -> Float64
    device_time(backend, binary::KernelBinary, slots, grid::NTuple{3,Int}; reps) -> Float64

Seconds per repetition of `reps` repetitions of a sequence of dispatches, all back to back in ONE
command buffer (or stream segment), from the device's own timestamps. `items` are
`(binary, slots, grid)` triples in launch order, `slots` the bound arguments in slot order as
[`encode_launch`](@ref) takes them; the second form times one dispatch. No warm-up. Implemented by the
backend extension (the sequence form); most callers want [`time_binary`](@ref), which binds the slots.
"""
device_time(be::KernelBackend, items::AbstractVector; reps) = throw(ArgumentError(
    "no `device_time` for backend $(typeof(be)); load the package for that backend"
))
device_time(be::KernelBackend, b::KernelBinary, slots, grid; reps) =
    device_time(be, [(b, slots, grid)]; reps)

"""
    time_binary(l::Launch; reps=3, warmup=1) -> Float64
    time_binary(ls::AbstractVector{<:Launch}; reps=3, warmup=1) -> Float64
    time_binary(b::KernelBinary, args...; extent, extras=(), reps=3, warmup=1)
    time_binary(op, kernel, name, args...; key=variant_key(op, kernel, args...), extent, reps=3, warmup=1)

Seconds per dispatch of a binary on the device: `warmup` untimed launches, then ONE
[`device_time`](@ref) of `reps` dispatches back to back, from the device's own timestamps. Host-side
launch latency is excluded. Repeating the call and taking a minimum is the caller's policy.

The core form times a prepared [`Launch`](@ref). A vector of launches is timed as a sequence: each
repetition dispatches all of them in order, in the same command buffer, and the result is seconds
per repetition — a whole forward or backward of several binaries (`record_bindings` hands over the
launches an op call makes). The third form prepares a launch for a binary that need not be
registered (a tuner's candidate); the fourth resolves the registered variant and its
[`extras`](@ref) as [`call_binary`](@ref) does. All bind with [`bind_slots`](@ref), exactly as an
eager [`execute`](@ref).

`OutArray`s are allocated once and reused by every dispatch (a `zero=true` one is zeroed once only),
so their contents afterwards are meaningless and are not returned. The `reps` dispatches share cache
state, which flatters a bandwidth-bound kernel slightly against a cold call.

Throws if a binary's threadgroup exceeds [`max_threads`](@ref), on host arrays, for launches on
different backends, and while tracing (timing is eager device work, which the tracer would capture).
"""
time_binary(l::Launch; reps::Integer=3, warmup::Integer=1) = time_binary([l]; reps, warmup)

function time_binary(ls::AbstractVector{<:Launch}; reps::Integer=3, warmup::Integer=1)
    isempty(ls) && throw(ArgumentError("no launches to time"))
    be = _timing_backend(first(ls).args)
    reps >= 1 || throw(ArgumentError("`reps` must be at least 1, got $reps"))
    items = map(ls) do l
        _timing_backend(l.args) == be || throw(ArgumentError(
            "launches on different backends cannot be timed as one sequence"))
        b = l.binary
        lim = max_threads(be, b)
        b.threadgroup <= lim || throw(ArgumentError(
            "threadgroup $(b.threadgroup) of `$(b.entry)` exceeds this pipeline's limit of $lim; " *
            "dispatching over it returns garbage AND a fast time"
        ))
        _, slots = bind_slots(_first_array(l.args), l.prelude, l.args, l.extras)
        (b, slots, l.grid)
    end
    for _ in 1:warmup, (b, slots, grid) in items
        encode_launch(be, b, slots, grid)
    end

    return Float64(device_time(be, items; reps=Int(reps)))
end

function time_binary(b::KernelBinary, args...; 
    extent=nothing, extras::Tuple=(), reps::Integer=3, warmup::Integer=1
)
    _timing_backend(args)
    return time_binary(prepare_launch(b, args...; extent, extras); reps, warmup)
end

function time_binary(op::AbstractKernelOp, k::AbstractKernel, name::Symbol, args...; 
    key::Tuple=variant_key(op, k, args...), extent=nothing, reps::Integer=3, warmup::Integer=1
)
    _check_extent(op, k, extent)
    b = _binary(op, k, name, key)
    return time_binary(b, args...; extent, extras=extras(op, k, b), reps, warmup)
end

# The device backend timing runs on: refuses tracing and host arrays.
function _timing_backend(args)
    tracing() && throw(ArgumentError("`time_binary` cannot run inside a trace: time the compiled program"))
    proto = _first_array(args)
    be = isnothing(proto) ? UnknownBackend() : backend_of(proto)
    be isa GPUBackend || throw(ArgumentError(
        "`time_binary` needs device arrays; got a $(typeof(be)) argument"
    ))
    return be
end

# --- inspecting what would be bound -----------------------------------------------------------------

const _RECORD = Ref{Union{Nothing,Vector{Any}}}(nothing)

"""
    record_bindings(f) -> Vector{(; label, slots, groups, launch)}

Run `f()` with every eager launch replaced by a record of what it would bind: the launch's `label`
(the name `call_binary` was given), its `slots` in order, each `kind => value` with `kind` one of
`:bytes` (the KA prelude), `:out` (an allocated output), `:inout` (an array updated in place), `:in` (an array), `:scalar`, and the
threadgroups it would launch, `groups` (`cld.(extent, tile)`, before any flattening by
[`ka_grid`](@ref)), and the [`Launch`](@ref) itself, which can be run or timed later
([`execute`](@ref), [`time_binary`](@ref)) to replay what `f` launches. Nothing is dispatched. A record
destructures as `label, slots, groups = rec`.
"""
function record_bindings(f)
    prev = _RECORD[]
    rec = Any[]
    _RECORD[] = rec
    try
        f()
        return rec
    finally
        _RECORD[] = prev
    end
end

# Called by an eager executor in place of launching while recording.
recording() = _RECORD[] !== nothing
record!(l::Launch, slots) = 
    push!(_RECORD[], (; label=l.label, slots=Tuple(slots), groups=l.groups, launch=l))

# The allocated outputs (and updated `InOutArray`s), picked out of the bound arguments in argument order. Recursive over the
# tuple, so the result type is inferred.
_pick_outs(::Tuple{}, ::Tuple{}) = ()
_pick_outs(a::Tuple, b::Tuple) = first(a) isa Union{OutArray,InOutArray} ?
    (first(b), _pick_outs(Base.tail(a), Base.tail(b))...) : _pick_outs(Base.tail(a), Base.tail(b))
