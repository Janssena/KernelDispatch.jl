# 2. Registering kernels

A kernel is a type with a few methods, plus compiled binaries registered as its **variants**. This
tutorial does the whole process twice for the [`SDPAOps`](https://github.com/Janssena/KernelDispatch.jl/blob/main/examples/SDPAOps/src/SDPAOps.jl) op:
once for a prebuilt GPU binary (e.g. `.fatbin`, `.cubin`, `.metallib`, etc.), once for a KernelAbstractions kernel. 
Both end up as the same thing; a `KernelBinary` in the registry, launched by `call_binary`.

What every kernel needs:

| method | says | |
|:--|:--|:--|
| `kernelname(k)` | an unique name within the op | optional, default is the type's name |
| `forward(op, k, args...)` | builds the binary's arguments from the op's and launches it | required |
| `variant_key(op, k, args...)` | which variant serves these arguments | default `()` |
| `extras(op, k, binary)` | values appended to the arguments (tuning numbers) | default `()` |
| `build!(op, k, key)` | create a missing variant on first use | optional |
| `source_tag(k)` | a manifest saved under another tag is ignored | default `""` |

> `variant_key` receives the arguments **as `call_binary` got them**, in the binary's order, not the
> arguments the op was called with. Write its signature against the binary's argument list.

## A. From a GPU binary (in this tutorial a `.metallib`)

### The binary and its calling convention

Below is a is hand-written `spda.metal` kernel per example, but these binaries can also be obtained from 
another toolchain such as Triton by directly compiling triton code to file. Built with 
`xcrun -sdk macosx metal sdpa.metal -o sdpa.metallib`, its entry point is:

```metal
kernel void sdpa_fwd(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* o       [[buffer(3)]],
    constant int& n       [[buffer(4)]],
    constant int& m       [[buffer(5)]],
    constant int& d       [[buffer(6)]],
    constant int& dv      [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint group  [[threadgroup_position_in_grid]],
    uint lane   [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]
)
```

What KernelDispatch does with a binary like this:

- It binds the arguments **in the order you give them**, from buffer index 0: an array as its device
  buffer, a scalar as a small buffer of its own (read it as `constant T&`). Under Reactant
  a scalar is passed as bytes, which `constant T&` reads the same way.
- It dispatches **threadgroups** of `threadgroup` threads each (how many: see
  [How many threadgroups](#how-many-threadgroups)), so the kernel computes its position from
  `threadgroup_position_in_grid` and `thread_position_in_threadgroup`.
- Arrays follow Julia conventions: column-major, so column `i` of a `d×n` matrix starts at `i * d`.

Nothing checks the argument list of `call_binary` against the actual binary. A wrong order or a wrong 
scalar type (`Int64` for an `int`) binds without error and computes garbage. Pin it with `record_bindings`
(see below).

### The kernel type and its methods

```julia
struct FlashAttn <: AbstractKernel end
kernelname(::FlashAttn) = :flash

_sizes(q, k, v) = (Int32(size(q, 1)), Int32(size(q, 2)), Int32(size(k, 2)), Int32(size(v, 1)))
_scale(q) = inv(sqrt(Float32(size(q, 1))))

# The binary's arguments, in its order: inputs, the output, sizes, scale. One item per query.
function forward(op::SDPA, kern::FlashAttn, q, k, v)
    d, n, m, dv = _sizes(q, k, v)
    out = OutArray(Float32, dv, n)
    scale = inv(sqrt(Float32(size(q, 1))))
    result = call_binary(op, kern, :fwd, q, k, v, out, n, m, d, dv, scale; extent=n) # returns a tuple
    return only(result)
end

# Against the same argument list: one variant per element type.
variant_key(::SDPA, ::FlashAttn, q, k, v, o, n, m, d, dv, scale) = (
  nameof(KernelDispatch.realtype(eltype(q))),
)
```

- `OutArray(Float32, dv, n)` marks an output. `call_binary` allocates it (or declares it a result in a
  trace) and returns the outputs in `OutArray` order, hence `only(result)`.
- `realtype` makes the key the same for a traced array under Reactant, whose element type is a
  `TracedRNumber{Float32}`.
- `extent=n` is the problem size: one item per query. The binary decides how many items one
  threadgroup covers (below), so the same `forward` would serve a binary built with another
  threadgroup size.

### Registering the binary

A `KernelBinary` describes one compiled launch:

```julia
KernelBinary(path; entry="sdpa_fwd", threadgroup=64, tile=(64, 1, 1), params=(;))
```

- `entry` is the name of the kernel function inside the file, the one the runtime launches. It
  **defaults to the file name without its extension**, so a file named after its function needs none:
  `KernelBinary("_fwd.metallib"; threadgroup=1024)` launches `_fwd`. Set it only when the function is
  named otherwise, as here (`sdpa.metallib` holds `sdpa_fwd`), or when one file holds several
  functions, each registered with its own `entry`.
- `threadgroup` is the number of threads per threadgroup.
- `tile` is how many items one threadgroup covers along each axis. The default, `(threadgroup, 1, 1)`,
  is one item per thread, as this kernel does, so it can be left out.
- `params` holds any tuning numbers `extras` read, and is saved with the binary.

### How many threadgroups

`call_binary` launches `cld.(extent, tile)` threadgroups: the call's problem size divided by what one
threadgroup of this binary covers. `extent` and `tile` are integers or tuples of up to three axes.
The two halves live where they are known: the extent at the call, the tile with the build, so a
retuned binary needs no change to the code that calls it.

| kernel | `tile` | `extent` | threadgroups |
|:--|:--|:--|:--|
| one thread per query, 64 threads | `(64, 1, 1)` (default) | `n` | `cld(n, 64)` |
| one Triton program per query | `1` | `n` | `n` |
| 32×32 tiles over an `n×n` matrix, per head | `(32, 32, 1)` | `(n, n, H)` | `(n/32, n/32, H)` |

Any grid can be launched this way: with `tile=1`, the extent is the grid. `extent=` is required:
`forward` computes the sizes anyway, so it passes the problem size next to them, and a call without
one throws rather than guess.

Register it as a variant for a key, under a **launch name**: `fwd` here, the name `forward` passes to
`call_binary`. The launch name is your label for the binary's role in the variant (`fwd`, `bwd`, or
trifast's `dq` and `dkv`); `entry` is the toolchain's name for the function in the file. They are
independent, and often differ: trifast's `_bwd_q.metallib` launches `_bwd_q` and is filed as `dq`.

```julia
add_variant!(SDPA(), FlashAttn(), (:Float32,); fwd=KernelBinary(METALLIB; entry="sdpa_fwd", threadgroup=64))
```

No `register_kernel!` is needed first: KernelDispatch registers kernels on first use.

`add_variant!` **copies** the file into the registry and writes a manifest:

```
~/.cache/KernelDispatch/ops/SDPAOps.sdpa/flash/
├── fwd_14162b5b.metallib
└── manifest.toml
```

```toml
source_tag = "27d7pc0r96syx"

[[variant]]
key = [":Float32"]

    [[variant.binary]]
    entry = "sdpa_fwd"
    name = "fwd"
    tile = [64, 1, 1]
    threadgroup = 64
    file = "fwd_14162b5b.metallib"
    is_ka = false
```

The copy is named `<launch name>_<id>.<ext>`: `fwd` from the `fwd=` keyword, and `14162b5b`, the
first 8 hex digits of a hash of the file's contents, the variant key and the launch name. So a rebuilt
binary (other contents), or the same binary filed under another key or launch, gets its own file, and
the file a manifest names is never overwritten by a different binary. The copy's name no longer says
which function to launch: that is why `entry` is fixed when the `KernelBinary` is made, from the
original path, and kept in the manifest.

The manifest is read back the first time the kernel is used in a later session, so a variant is
registered **once**, not every session. The directory is `<cache_dir>/ops/<Package>.<opname>/<kernelname>`: the
package naturally separates ops with the same name coming from different packages. A variant can hold several 
launches (`fwd=…, bwd=…`); registering a key again replaces its variant and removes the binaries no variant uses 
any more.

### Registering on first use: `build!`

Rather than registering in a setup script, `SDPAOps` builds a missing variant the first time it's
needed. `call_binary` looks the key up, then tries your `nearest`, then calls your `build!`:

```julia
const METALLIB = joinpath(@__DIR__, "..", "kernels", "sdpa.metallib")   # src/../kernels
include_dependency(METALLIB)        # a new binary re-precompiles the package, and so its tag

source_tag(::FlashAttn) = string(hash(read(METALLIB)); base=36)

function build!(op::SDPA, kern::FlashAttn, key::Tuple)
    key == (:Float32,) || return nothing                # the binary is Float32 only
    add_variant!(op, kern, key; fwd=KernelBinary(METALLIB; entry="sdpa_fwd", threadgroup=64))
    return nothing
end
```

`source_tag` makes a stale registry harmless: a manifest saved under another tag is ignored, so a new
binary is registered afresh instead of the old one being reused. If `build!` doesn't create the
variant (a `Float16` call here), `call_binary` throws, naming the key and the registered ones.

## B. From a KernelAbstractions kernel

A KernelAbstractions kernel is one more source of binaries: `ka_compile` compiles it, and from there 
it is registered and launched exactly like any other binary.

### Writing the kernel

```julia
@kernel unsafe_indices = true function sdpa_ka!(o, q, k, v, d::Int32, n::Int32, m::Int32, dv::Int32,
        scale::Float32, ::Val{TG}) where {TG}
    i = (@index(Group, Linear) - 1) * TG + @index(Local, Linear)
    if i <= n
        qi, oi = (i - 1) * d, (i - 1) * dv              # where column `i` starts
        # … scores with unsafe_load(q, qi + t) * unsafe_load(k, (j - 1) * d + t), a stable softmax,
        #   and unsafe_store!(o, …, oi + c): see SDPAOps.jl for the whole kernel
    end
end
```

The contract a KernelAbstractions kernel compiled to a binary follows:

- **Arrays arrive as raw device pointers**: index them with `unsafe_load(x, i)` and
  `unsafe_store!(x, val, i)`, 1-based and linear (column-major).
- **Scalars are bits values** of exactly the type you compile for (`Int32` here).
- **Compile-time parameters are `Val`s**: the tile size `TG` is baked into the binary.
- **Bounds-check yourself** (`unsafe_indices = true`, `if i <= n`): the launch covers whole
  threadgroups, so the last one usually runs past `n`.

### The kernel type and its methods

The same methods as for the `.metallib`, against this kernel's argument order (output first):

```julia
struct SDPAKA <: AbstractKernel end
kernelname(::SDPAKA) = :ka

# Changes with the kernel's source: a manifest under another tag is ignored, and the binary's name
# below includes it, so an edited kernel is rebuilt instead of reused.
source_tag(::SDPAKA) = string(hash(read(@__FILE__)); base=36)

function forward(op::SDPA, kern::SDPAKA, q, k, v)
    d, n, m, dv = _sizes(q, k, v)
    out = OutArray(Float32, dv, n)
    scale = inv(sqrt(Float32(size(q, 1))))
    result = call_binary(op, kern, :fwd, out, q, k, v, d, n, m, dv, scale; extent=n)
    return only(result)
end
variant_key(::SDPA, ::SDPAKA, o, q, k, v, d, n, m, dv, scale) = (
  nameof(KernelDispatch.realtype(eltype(q))),
)

# The tile size is compiled in: appended as a `Val`, it takes no argument slot.
extras(::SDPA, ::SDPAKA, b::KernelBinary) = (Val(b.params.tg),)
```

KernelDispatch builds KA's context for the threadgroups it launches (`groups × tg` work items) and
dispatches them the way the backend's KA kernels expect. Define the type and these methods first:
compiling below uses its `source_tag`.

### Compiling it

```julia
argtypes = (
  Vector{Float32}, Vector{Float32}, Vector{Float32}, Vector{Float32},   # o, q, k, v
  Int32, Int32, Int32, Int32, Float32, Val{64}                          # d, n, m, dv, scale, TG
)

fwd = ka_compile(KernelDispatch.MetalBackendTag(), sdpa_ka!, argtypes;
  tg=64, name="sdpa_fwd_$(source_tag(SDPAKA()))", params=(; tg=64)
)
```

- `argtypes` are the kernel's arguments after KernelAbstractions's own launch context, in order: 
  `Vector{T}` for a buffer, the bits type for a scalar, `Val{x}` for a compile-time value. For a 
  buffer only the element type `T` matters: the kernel is compiled for a raw device pointer to `T`, 
  not for an array, so `Vector{Float32}`, `MtlVector{Float32}` or any other `AbstractArray{Float32}` 
  give the same binary. Whatever device array the call passes (an `MtlArray`, a traced array) is 
  bound as that pointer.
- `tg` is the workgroup size the binary is compiled for. `name` is the binary's file name and entry
  point. A file already built under that name is reused, so put a source hash in it.
- The result is a `KernelBinary` with **`is_ka = true`**. That flag tells `call_binary` to pass the two
  hidden leading arguments every KA-compiled binary takes, the backend's state word and KA's launch
  context, and to dispatch the grid flat. A KA kernel compiled by other means would bind wrongly
  without error, so compile KA kernels with `ka_compile`.
- Its `tile` is `(tg, 1, 1)`, one item per thread, as `sdpa_ka!` indexes (`i` from the group and
  lane). A kernel whose threads each cover several items passes `tile=` to `ka_compile`.
- It is built in `ka_binary_cache_dir()` (`~/.cache/KernelDispatch/ka`) and copied into the registry by
  `add_variant!`.

### Registering it

As before, once:

```julia
# Note that compiled KA kernels are type specific, so we add a key as the third argument:
add_variant!(SDPA(), SDPAKA(), (:Float32,); fwd)
```

or on first use, in `build!`. That is the robust form: it compiles only when a call needs the variant,
by which time the kernel, its type and its `source_tag` are all defined, whatever order the file
defines them in.

```julia
function build!(op::SDPA, kern::SDPAKA, key::Tuple)
    key == (:Float32,) || return nothing
    tg = 64
    argtypes = (Vector{Float32}, Vector{Float32}, Vector{Float32}, Vector{Float32},
        Int32, Int32, Int32, Int32, Float32, Val{tg})
    fwd = ka_compile(KernelDispatch.MetalBackendTag(), sdpa_ka!, argtypes;
        tg, name="sdpa_fwd_$(source_tag(kern))", params=(; tg))
    add_variant!(op, kern, key; fwd)
    return nothing
end
```

Its manifest records `is_ka = true`, the tile and the tuning numbers:

```toml
    [[variant.binary]]
    entry = "sdpa_fwd_37wlduvo9jvu0"
    name = "fwd"
    tile = [64, 1, 1]
    threadgroup = 64
    file = "fwd_99627d4e.metallib"
    is_ka = true

        [variant.binary.params]
        tg = 64
```

## Checking what a binary gets

`record_bindings(f)` runs `f` with every launch replaced by a record of what it would bind:

```julia
julia> label, slots, groups = only(KernelDispatch.record_bindings(() -> SDPA()(q, k, v)));

julia> label => map(first, slots)          # with `:flash` selected
:fwd => (:in, :in, :in, :out, :scalar, :scalar, :scalar, :scalar, :scalar)

julia> label => map(first, slots)          # with `:ka` selected
:fwd => (:bytes, :bytes, :out, :in, :in, :in, :scalar, :scalar, :scalar, :scalar, :scalar)
```

The KA binary's two leading `:bytes` slots are the state word and KA's context; its `Val` takes no
slot. Each slot's second element is the value bound, so a test can pin types and sizes too. `groups`
is the number of threadgroups along each axis, `cld.(extent, tile)`: with 100 queries and a tile of 64,
`(2, 1, 1)`. This is the cheapest guard against a mis-ordered argument list or a wrong launch size.

## Running a binary that is not registered

`call_binary` is a lookup on top of `run_binary`, which runs one given `KernelBinary` with the same
binding rules. A tuner compiling candidates calls it directly, without registering each one:

```julia
cand = ka_compile(KernelDispatch.MetalBackendTag(), sdpa_ka!, argtypes; tg=128, name="sdpa_fwd_t128")
o, = run_binary(cand, OutArray(Float32, dv, n), q, k, v, d, n, m, dv, scale; extent=n)
```

`prepare_launch` (same arguments) returns the launch as a `KernelDispatch.Launch` without running it, so one
prepared launch can be both checked (`KernelDispatch.execute(l)`) and timed (`time_binary(l)`).

`time_binary` reports device time: `reps` dispatches back to back in one command buffer, timed by the
device's own timestamps, so the host's launch latency is excluded. A vector of launches is timed as a
sequence, every repetition dispatching all of them in order. Together with `record_bindings`, whose
records carry each `Launch`, that times everything one op call launches:

```julia
launches = [r.launch for r in KernelDispatch.record_bindings(() -> SDPA()(q, k, v))]
t = time_binary(launches; reps=10)      # seconds per call, kernels only
```

## Kernels from another package

The kernel doesn't have to live in the op's package. Anyone can define a kernel type for `SDPA`, give
it these methods and binaries, and select it with `use_kernel!(SDPA(), MyKernel())`, which registers it
under its `kernelname` (a name the op already uses for another kernel throws). `register_kernel!`
registers it without selecting it, to have it listed and selectable by name upfront. Only `SDPAOps`
can make a kernel the persisted default ([tutorial 1](1-switching-kernels.md)).
