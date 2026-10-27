# Backend interface

```@meta
CurrentModule = KernelDispatch
```

Everything here is for people **adding a backend** (a GPU vendor) or **a binary ABI** (a producer whose kernels take
hidden arguments). Users and kernel authors need the [API reference](api.md) only. These names are documented
but not `public`: they will change as backends are added, so pin KernelDispatch's version if you implement them.

## The shape of it

A backend is a package extension. It owns a tag (`MetalBackendTag`, `CUDABackendTag`, ...), registers itself in
`LOADED_BACKENDS` when its device is usable, and implements the hooks below by dispatching on its tag or its
array type. The core never asks "which vendor": it asks the hook.

A binary declares **what it is** with three facts: its file's *format* (the extension), its `abi` (what it takes
besides the arguments the caller binds) and its `target` (the device architecture it was built for). The
backend decides what it can launch from those.

| hook | says | default | Metal | CUDA |
|:--|:--|:--|:--|:--|
| `backend_of(::MyArray)` | the tag of an array | `CPUBackend` | `MtlArray` | `CuArray` |
| `device_dense` | densify a wrapper (view, adjoint) | identity | copy | `CuArray(x)` |
| `execute(::MyArray, ::Launch)` | run a prepared launch | error | yes | yes |
| `encode_launch`, `device_time`, `max_threads` | the launch, its timing, its thread limit | error | yes | yes |
| `supported_formats` | formats it launches | `()` (unchecked) | `.metallib` | `.cubin`, `.ptx` |
| `target_id`, `compatible_target` | architecture, and whether a binary fits it | `""`, accept | portable | `sm_XY` |
| `ka_backend_object`, `ka_state_words`, `ka_prelude`, `ka_grid`, `ka_layout`, `_ka_compile` | KernelAbstractions binaries (`abi = :ka`) | error (`nothing` for `ka_layout`) | yes (state word, context bytes) | yes (flat: no state, six integers) |
| `abi_prelude_and_grid` | another ABI's hidden arguments | error | none | none |
| `custom_call`, `traced_support` | embedding in a Reactant trace, and what it cannot carry | error, accept | `mps.metal_kernel_lib` | `triton_kernel_call_ffi` (any binary; `reverse` rules refused) |

**Traced CUDA and buffer order.** XLA's kernel-call handler hands a launch's buffers over as operands, then results, and
binds a kernel's buffer parameters to them in that order. Kernels cannot be asked to list inputs before outputs (KA
kernels are `f!(out, in...)`, Triton kernels differ, prebuilt binaries cannot be reordered), so the CUDA embedding
passes every output as an extra, zero-filled operand in the kernel's own parameter position and aliases the result to
it (`output_operand_aliases`). The buffers then arrive in the kernel's order, whatever it is, and the repeated result
buffers after them go unused. This was measured on an NVIDIA machine (outputs first, last or
between scalars); without the aliasing an output-first kernel is bound wrongly and silently. The same mechanism carries an
[`InOutArray`](@ref): its operand is bound in the kernel's position with the result aliased to it. Metal has no such limit and
refuses an `InOutArray` in a trace.

**Traced launches declare their layouts.** The custom call gives every operand and result the default dense layout
(`operand_layouts` / `result_layouts`, last dimension fastest, i.e. Julia's column-major order). A backend's `custom_call`
inherits it from `_embed_custom_call`; it has nothing to pass. It matters wherever the XLA compiler is free to choose the
layout of an operand it computes (its GPU compiler is): see [Traced launches](backends.md#Traced-launches).

A backend is complete when it passes `test/gpu/conformance.test.jl`, which runs the same checks against any loaded
backend using an `:explicit`-ABI kernel.

## Launching

```@docs
encode_launch
device_time
bind_slots
device_dense
LOADED_BACKENDS
```

## Formats and targets

```@docs
supported_formats
binary_format
target_id
compatible_target
check_binary
```

## KernelAbstractions binaries

What a KA binary takes ahead of its arguments is the backend's business: `ka_state_words` (hidden leading
arguments), then `ka_prelude` (KA's launch context, in whatever form the backend passes it), on a grid `ka_grid`
lays out. Metal passes a state word and the context's bytes. **CUDA passes nothing for the state and the context as
six integers** (the ND-range, then the block counts), because the binary is built by `_ka_compile` as a *flat*
wrapper entry compiled with CUDA.jl's kernel state switched off, so that XLA's handler can launch it; the wrapper
rebuilds the context on the device. The layout is recorded as `params.ka_layout` ([`ka_layout`](@ref)), and the CUDA
`compatible_target` refuses a `:ka` binary without it. The details, and the consequence that a CUDA KA kernel
must be free of code that can throw, are on the [Backends](backends.md#KernelAbstractions-kernels:-built-flat) page.

```@docs
ka_backend_object
ka_state_words
ka_context
ka_grid
ka_prelude
ka_layout
```

## ABIs

```@docs
abi_prelude_and_grid
```

## Reactant traces

```@docs
custom_call
traced_support
```

## Adding a backend

The checklist, in the order that finds problems fastest. ROCm is the next vendor, and the notes say what is likely
to differ.

1. **A tag and its formats.** `MetalBackendTag`, `CUDABackendTag` and `ROCmBackendTag` already exist in `src/backend.jl`
   with their [`supported_formats`](@ref) (`.hsaco` for ROCm). A new vendor adds a tag and a method.
2. **An extension** that loads on the vendor's package, pushes the tag onto [`LOADED_BACKENDS`](@ref) in `__init__`
   only when the device is usable, and defines [`backend_of`](@ref) for its arrays and [`device_dense`](@ref) for the
   wrappers (views, adjoints) that are not plain device arrays.
3. **Launching.** [`execute`](@ref)`(::MyArray, ::Launch)` using [`bind_slots`](@ref), [`encode_launch`](@ref),
   [`device_time`](@ref) and [`max_threads`](@ref). The conformance suite exercises exactly these.
4. **Architecture.** [`target_id`](@ref) and [`compatible_target`](@ref). CUDA accepts a cubin from the same major
   version when the device's minor is at least its own; for ROCm expect an exact `gfx` match (with a few family rules),
   not CUDA's rule.
5. **KernelAbstractions binaries.** [`ka_backend_object`](@ref), [`ka_state_words`](@ref), [`ka_prelude`](@ref),
   [`ka_grid`](@ref), optionally [`ka_layout`](@ref), and `_ka_compile`. What the vendor's KA kernels take as hidden
   leading arguments must be read from the vendor's package, not assumed from CUDA or Metal.
6. **Traced launches.** [`custom_call`](@ref) and [`traced_support`](@ref). First find out how the XLA client
   behind Reactant launches a kernel image at all (which custom-call target and API version it accepts; CUDA's is
   selectable with `KERNELDISPATCH_CUDA_CUSTOM_CALL`, so a candidate can be tried per process). Until something works, make `traced_support` say so and leave
   `custom_call` unimplemented, so a traced launch fails with an explanation. Whether the vendor's handler binds
   buffers in a fixed order, and whether aliased outputs lift that, is measured with a kernel that writes a different value to each buffer, run in every order.
7. **Tests.** Add `test/ext/<backend>/backend.jl`, the adapter the generic tests in `test/gpu/` run through, and a
   candidate in `test/gpu_backend.jl`. The adapter defines the device array type, the backend's thread limit, its
   array-wrapper union, an `:explicit`-ABI `scale` fixture (hand-written PTX for CUDA, a committed `.metallib` for Metal),
   a binary in a format it must refuse, its traced Reactant client and the custom call's name, the KA state words it
   expects, and any prebuilt SDPA kernels (the full list is in `test/gpu_backend.jl`). The backend is complete when the
   generic tests pass, `test/gpu/conformance.test.jl` first. Tests for what only this backend does go beside the adapter,
   and for a machine you can only reach remotely a staged script in the style of `test/ext/cuda/smoke.jl`.

For ROCm specifically: the code object is a `.hsaco`; wavefronts are usually 64 wide (so threadgroup multiples and a
Triton `threadgroup` of `num_warps * warp_size` differ from CUDA's 32); and `triton_binary` keeps NVIDIA-only logic
(`sm_*`, `.cubin`/`.ptx`, the CUDA scratch-pointer convention) that would move behind a per-vendor hook.
