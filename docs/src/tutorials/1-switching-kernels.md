# 1. Switching kernels and persisting defaults

An op is *what* is computed; a kernel is *one implementation* of it. An op can have several kernels,
and which one runs is chosen in two ways:

| | set with | takes effect | lasts |
|:--|:--|:--|:--|
| **default** | `@default_kernel` in the op's package, `set_default_kernel!` | next session | until changed |
| **session override** | `use_kernel!` | next top-level statement | this session |

The selection is a *method*, not a table lookup, so the compiler sees it as a constant and op calls
stay type-stable. Both layers follow from that.

## The op and its kernels
 
The below is an example from a hypothetical SDPAOps package implementing a SPDA operation and two kernels:

```julia
struct SDPA <: AbstractKernelOp end
opname(::SDPA) = :sdpa                       # optional: a stable name (default = struct name, i.e. `:SDPA`)

struct FlashAttn <: AbstractKernel end       # a prebuilt Metal binary
struct SDPAKA <: AbstractKernel end          # a KernelAbstractions kernel, compiled to a binary
kernelname(::FlashAttn) = :flash             # optional (default = struct name, i.e. `:FlashAttn`)
kernelname(::SDPAKA) = :ka

KernelDispatch.@default_kernel SDPA FlashAttn() SDPAKA()
```

- `@default_kernel` lists the kernels that can be the op's **default**. The first, `FlashAttn()`, is the
  default unless a preference names another (below).
- **Nothing is registered by hand.** KernelDispatch' tables are filled at run time, on first use: the
  `@default_kernel` candidates when a name or `list_kernels` needs them, any other kernel when
  `use_kernel!` selects it.
- **Names are optional.** An op and a kernel are named after their types unless they define `opname`
  and `kernelname`. A name is the kernel's identifier outside the running code: what a preference
  stores, its registry directory, a REPL shorthand. `SDPAOps` defines them so those stay stable if its
  types are renamed. Everything else works with the types themselves.

What a kernel needs beyond this is the subject of [tutorial 2](2-registering-kernels.md).

## Inspecting and switching for a session

```julia
julia> using SDPAOps, KernelDispatch

julia> current_kernel(:sdpa)
:flash

julia> o = SDPA()(q, k, v);                   # runs the `:flash` kernel for device arrays

julia> list_kernels(:sdpa)
2-element Vector{@NamedTuple{name::Symbol, nvariants::Int64, selected::Bool}}:
 (name = :flash, nvariants = 1, selected = 1)
 (name = :ka, nvariants = 0, selected = 0)

julia> use_kernel!(SDPA(), SDPAKA())          # by kernel; the op by instance, type or name…
:ka

julia> use_kernel!(:sdpa, :ka)                # …or both by name, the REPL shorthand
:ka

julia> SDPA()(q, k, v)                        # now runs the `:ka` kernel

julia> reset_kernel!(:sdpa)                   # back to the default (FlashAttn())
```

`use_kernel!` redefines a method, and Julia makes a new method visible from the **next top-level
statement** (REPL input, script line). Calling the op later in the same function or `begin`/`let`
block throws instead of silently running the previous kernel:

```
ArgumentError: op `:sdpa` was called from code compiled before the latest kernel switch: `use_kernel!`
(or `reset_kernel!`) takes effect from the next top-level expression, not later in the same function or
block. Switch at top level, or call `forward(op, kernel, args...)` to run one call with a particular
kernel.
```

To run one call with a particular kernel, inside a function or anywhere else, call that kernel's
`forward` directly. Nothing is switched or recompiled, and the call is inferred:

```julia
for kern in (FlashAttn(), SDPAKA())
    o = KernelDispatch.forward(SDPA(), kern, q, k, v)    # a benchmark comparing kernels
end
```

`forward` takes the device path, so pass it device (or traced) arrays. The selection only matters for
op calls you can't reach directly, inside a model say: switch those with `use_kernel!` at top level.

Anything compiled *once* for later use keeps the kernel selected when it was built: a Reactant thunk
(`@compile`) and an Enzyme thunk (`autodiff_thunk`, see [tutorial 3](3-forward-backward-enzyme.md)).
Switch at top level first, then build it, and rebuild it after switching.

## Persisting a default

The session override is gone after a restart. To make `:ka` the default from now on:

```julia
julia> set_default_kernel!(:sdpa, :ka)
[ Info: Default kernel of op `:sdpa` set to `:ka` in SDPAOps's preferences. It takes effect after a
restart (re-precompiling SDPAOps and its dependents); `use_kernel!` switches this session.
```

This writes a preference of the package that declares the op (`SDPAOps`) to the active project's
`LocalPreferences.toml`:

```toml
[SDPAOps]
"kernel.sdpa" = "ka"
```

In the next session `SDPAOps` re-precompiles once, with `:ka` baked in as the default:

```julia
julia> using SDPAOps, KernelDispatch        # precompiles SDPAOps

julia> current_kernel(:sdpa)
:ka

julia> KernelDispatch.default_kernel(SDPA())
SDPAKA()
```

`clear_default_kernel!(:sdpa)` removes the preference, and the first candidate is the default again
from the next session. A name that isn't one of the `@default_kernel` candidates is refused:

```
julia> set_default_kernel!(:sdpa, :nope)
ERROR: ArgumentError: kernel `:nope` is not a default candidate of op `:sdpa`. Candidates: :flash, :ka
(listed in its `KernelDispatch.@default_kernel`)
```

### How this works

`@default_kernel` runs at the top level of the op's package, so while the package precompiles. There
it reads the preference `"kernel.<opname>"` with Preferences.jl, picks the matching candidate (the
first if there is no preference), and defines `default_kernel(::SDPA)` to return it as a constant.
Preferences.jl records that the package read the preference during precompilation, so changing it
invalidates the package's cache: the next `using` re-precompiles with the new default. That is why a
default costs one re-precompile but no run-time lookup.

Some consequences:

- **Preferences belong to the op's package.** `set_default_kernel!` writes to the package declaring
  the op type, in the *active project's* `LocalPreferences.toml`. A different project (another
  environment, a test environment) has its own preferences.
- **Scripts have none.** An op defined in a script (`Main`) has no package, so its default is always
  the first candidate, and `set_default_kernel!` throws.
- **A stale preference falls back.** If `LocalPreferences.toml` names a kernel that isn't a candidate
  (a renamed kernel, say), precompilation warns and uses the first candidate.
- **Only the op's package controls the default.** Anyone can add a kernel to `SDPA` and select it
  with `use_kernel!(SDPA(), MyKernel())`, which registers it on the way, but can't make it the
  default: the candidates are listed in `SDPAOps`.

## Selection belongs to the application

The selection is global per op: one method, shared by everything in the session. Whoever switches
switches it for every caller, including libraries that were tested with another kernel. So:

- **An application or script decides.** `use_kernel!` in your session or script,
  and preferences in your project's `LocalPreferences.toml`, are choices you make explicitly.
- **A library never switches.** It can't anyway: `use_kernel!` refuses to run while precompiling,
  which includes a package's top level and its `__init__` while dependents precompile.
- **A library that needs one particular kernel calls it directly**, past the selection:
  `KernelDispatch.forward(SDPA(), FlashAttn(), q, k, v)`. Normally a library shouldn't care: all of an op's kernels
  compute the same thing, and which one runs is the application's performance decision.
