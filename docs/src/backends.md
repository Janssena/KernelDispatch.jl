# Backends: Metal and CUDA

```@meta
CurrentModule = KernelDispatch
```

KernelDispatch launches compiled binaries. This page says **which binaries run where, how each one gets
launched, and what is and is not verified**. It is the user's view; the hooks a new backend implements are on the
[backend interface](backend.md) page.

## What runs where

| binary source | how you get a [`KernelBinary`](@ref) | Metal | CUDA |
|:--|:--|:--|:--|
| a prebuilt file | `KernelBinary("x.metallib"; ...)` or `"x.cubin"` / `"x.ptx"` | `.metallib` | `.cubin`, `.ptx` |
| a KernelAbstractions kernel | [`ka_compile`](@ref) | yes | yes |
| a Triton kernel | [`triton_binary`](@ref) | not applicable | yes (see [Triton on CUDA](#Triton-on-CUDA)) |

and how each is launched:

| | eager (device arrays) | traced (Reactant `@jit`) |
|:--|:--|:--|
| Metal | Metal.jl | `mps.metal_kernel_lib` custom call (jax-mps) |
| CUDA | CUDA.jl driver launch | `triton_kernel_call_ffi` custom call (XLA's kernel-call handler) |
| Float16 and BFloat16, Metal: prebuilt `.metallib` (half and bfloat), KA kernels, `SDPAOps` (FlashAttn and KA), mixed outputs, eager and traced, gradients through Enzyme | tested on Apple silicon |
| Float16 and BFloat16, CUDA: hand-written PTX and cubin, flat KA binaries, mixed outputs, `SDPAOps`, gradients through Enzyme, eager and traced (BFloat16 needs sm_80 for the native conversions) | tested on an NVIDIA A40 (`test/ext/cuda/half_smoke.jl`) |
| Argument-type check (signature recorded by `ka_compile`, declared for prebuilt binaries): refused eagerly and from a trace | tested on Metal and on an NVIDIA A40 |
| Triton kernels in half precision | not tested (the fixtures are `float32`) |
| ROCm | not implemented | not implemented |

Which backend a call runs on is decided by the arrays: a `MtlArray` or a traced array under a Metal client goes to
Metal, a `CuArray` or a traced array under a CUDA client goes to CUDA, and a host array runs the op's
[`host`](@ref) method. A prebuilt `.metallib` is Metal-only, so an op that ships one needs a KA kernel (or a
`.cubin`) to run on CUDA.

## How a launch goes

Every source of binaries ends up in the same path:

1. [`add_variant!`](@ref) copies the file into the registry (one directory per kernel and, for a device with an
   architecture, per target: `ops/<Package>.<op>/sm_86/<kernel>/`) and writes a manifest recording each binary's
   file, entry, `threadgroup`, `tile`, `abi`, `target`, `shared_mem` and tuning `params`.
2. Your `forward` calls [`call_binary`](@ref)`(op, kernel, name, args...; extent)`, which looks up the variant and
   the binary, and [`prepare_launch`](@ref) turns it into a [`Launch`](@ref).
3. `prepare_launch` runs [`check_binary`](@ref) (the file's format and the target architecture are
   checked against the device, with an explanation instead of a driver error), works out the threadgroups
   (`cld.(extent, tile)`), and asks the binary's `abi` what to bind ahead of the arguments and which grid to use.
4. [`execute`](@ref) takes one of two routes. On eager device arrays the backend extension launches the binary. On
   traced arrays [`traced_support`](@ref) is asked first, and then the binary is embedded as a
   `stablehlo.custom_call` ([`custom_call`](@ref)).

A binary declares what it is with three facts, and the backend decides what it can launch from them:

- its **format**, from the file extension (`:metallib`, `:cubin`, `:ptx`, ...);
- its **`abi`**: what it takes besides the arguments the caller binds. `:explicit` takes nothing (exactly the buffers and scalars,
  in the order given); `:ka` is a KernelAbstractions binary; `:triton` is a Triton binary (hidden trailing pointers);
  an extension may define others ([`abi_prelude_and_grid`](@ref));
- its **`target`**: the device architecture it was built for (`"sm_86"` for a cubin), `""` when portable.

## CUDA

### Eager launches

A `.cubin` (machine code for one architecture) or `.ptx` (text the driver compiles at load, so it runs on any
newer device) is loaded into a module, cached per context, and launched on the current stream. A cubin's `target`
must be compatible with the device: the same major compute capability and a device minor version at least the
cubin's. PTX is accepted on any device. A threadgroup above the function's own limit is refused before launching
(over the limit a launch otherwise fails or, worse, silently misbehaves).

Dynamic shared memory comes from [`KernelBinary`](@ref)'s `shared_mem`; above the default 48 KB it is raised on the
function once. Every eager launch synchronizes.

### KernelAbstractions kernels: built flat

A KA kernel compiled by CUDA.jl normally takes two struct arguments ahead of its own: CUDA.jl's hidden
`KernelState` (an exception flag and RNG seed, 16 bytes) and KA's launch context (48 bytes). XLA's kernel-call
handler passes a kernel only device pointers and 1-, 4- or 8-byte scalars, so such a binary could not be traced.

[`ka_compile`](@ref) therefore builds CUDA binaries **flat**:

- The kernel state is switched off for the build.
- KA's launch context for a dynamic ND-range and a static workgroup size is six integers, the ND-range and the block
  counts. They are passed as six scalars, and a small wrapper entry rebuilds the context on the device from them,
  then calls your kernel body.
- The result takes only device pointers and scalars (and compile-time `Val`s, which leave no parameter), so **the same
  file launches eagerly and from a Reactant trace**.

Two consequences:

!!! warning "A CUDA KA kernel must not be able to throw"
    Reporting a device exception needs the kernel state, which a flat build does not have. A kernel whose compiled
    code can still throw (a bounds check, a `DomainError` from `sqrt` of a negative, `error`) is refused by
    `ka_compile` with the remaining reasons listed. Index with `unsafe_load` / `unsafe_store!`, as the binary
    contract already says. A running kernel can no longer raise a `KernelException` host-side.

!!! note "Old CUDA KA binaries are refused"
    A CUDA `:ka` binary records `params.ka_layout = "flat"` ([`ka_layout`](@ref)). One built before this layout
    existed has no such entry and is refused with a message to delete the stale `ka/<target>` build directory and
    the registry entry and rebuild it with `ka_compile`.

Metal keeps its own layout (a kernel-state word and the context bytes); that difference is in the backend's
hooks ([`ka_state_words`](@ref), [`ka_prelude`](@ref)), not in the ABI name.

### Traced launches

Under Reactant, a CUDA `call_binary` becomes a `stablehlo.custom_call` to XLA's `triton_kernel_call_ffi` handler,
whatever the binary: despite its name the handler launches any kernel image (a cubin, or PTX it assembles) given a
kernel name, a grid, a thread count and a parameter list. KernelDispatch encodes jaxlib's `TritonAnyKernelCall`
message itself, so there is no Python and no native code of its own.

What follows from the handler:

- **Parameters** are device buffers and 1-, 4- or 8-byte scalars, no structs. That is why CUDA's KA binaries are flat.
- **Threads per block** are `num_warps * 32`, so a threadgroup is a multiple of 32.
- **Buffer order is free.** The handler binds kernel buffer parameters to the call's operands and then its results,
  in that order, which would force a kernel to list inputs before outputs (a KA kernel is `f!(out, in...)`; Triton and
  prebuilt kernels differ). The CUDA embedding instead passes **every output as an extra zero-filled operand in the
  kernel's own parameter position, with its result aliased to it** (`output_operand_aliases`). The buffers arrive in
  the kernel's order whatever it is, and the repeated result buffers after them go unused. Outputs therefore always
  start as zeros, `OutArray(...; zero=true)` or not. Without the aliasing an output-first kernel is bound wrongly
  *and silently*.
- **Arrays a kernel updates** (`y += a * x`) are passed as an [`InOutArray`](@ref): its operand is bound in the kernel's
  position and the result aliased to it, so the kernel starts from the array's values (an [`OutArray`](@ref) starts from
  zeros) and the updated array is returned. XLA copies the operand first if it is used afterwards, so the traced program
  stays functional. Not available on Metal under a trace.
- **Autodiff** through a traced custom call is not supported (Enzyme cannot differentiate one yet); an op's
  `backward` can be called explicitly.
- **Layouts are declared.** Every operand and result of the call is declared dense with the last dimension fastest
  (`operand_layouts` / `result_layouts`; Reactant reverses a Julia array's dimensions, so this is Julia's column-major
  order, the one kernels read). Without the declaration XLA's GPU compiler may hand the kernel an operand it computed
  itself in another physical layout: a `permutedims` feeding the call can become a relabelling instead of a copy, and the
  kernel then reads the buffer in the wrong order, silently (measured on an NVIDIA A40, where the transposed bias of an
  attention kernel gave a relative error of about 0.8 while operands that were plain parameters were fine). With the
  declaration XLA has to produce the operand in memory in that order, so a transpose in an op's `forward` is a real
  copy and nothing needs to be passed by the user. A runtime that executes the HLO as written (jax-mps) keeps every
  tensor in this layout already and accepts the attribute, so the results do not change there.

The calling convention is `triton_kernel_call_ffi@4` (jax-triton's typed-FFI form, the payload in an `opaque`
attribute). Reactant's compile pipeline rewrites a custom call named exactly `triton_kernel_call` into Triton IR,
which is why the `_ffi` name is the default. For another XLA build,
`KERNELDISPATCH_CUDA_CUSTOM_CALL=<target>@<api_version>` selects another convention.

### Triton on CUDA

[`triton_binary`](@ref) turns the metadata JSON Triton writes beside its `.cubin` / `.ptx` into a
[`KernelBinary`](@ref) with `abi = :triton`. It reads the kernel's entry, `threadgroup` (`num_warps * 32`),
`shared_mem`, the architecture a cubin was built for and how many hidden trailing pointer parameters the compiled
kernel takes (recent Triton versions declare scratch pointers after the kernel's own parameters; a launch passes that
many null pointers, eager or traced). It refuses what it cannot launch, with the reason: a non-NVIDIA target,
`num_ctas` above 1 (thread-block clusters), and a kernel needing global scratch memory.

!!! warning "A Triton binary is compiled for the arguments it was first launched with"
    Triton specialises a kernel on its arguments: an integer equal to 1 is dropped from the signature (a stride of a
    contiguous array; confirmed). Triton also specialises integers and pointers divisible by 16 (vectorised loads); the
    measurement below confirms it. A registered binary is therefore valid only for launches that share those properties (a tensor with a
    different dimension size that changes which strides or sizes are 1 or divisible by 16 needs its own binary, or at
    least one compiled with the same properties). Measured with trifast's `_fwd` on Triton 3.8.0: sequence lengths 33, 40
    and 41 share one binary, 48 and 64 another (divisible by 16).

The arguments `call_binary` binds must be exactly the ones the *compiled* kernel declares: Triton drops
`constexpr` arguments from the signature and, depending on its version, integers it specialized to 1. See
[tutorial 4](tutorials/4-triton-kernels.md#B.-NVIDIA:-a-Triton-.cubin).

### Building without a GPU

On some clusters a GPU is only available inside a job, so binaries have to be built on a node that has none. What that
allows depends on where the binary comes from:

| source | without a GPU | note |
|:--|:--|:--|
| `nvcc` (`.cubin`, `.ptx`) | **yes** | `nvcc -cubin -arch=sm_86 …` and `nvcc -ptx …` need the CUDA toolkit, not a device. The architecture is the `-arch` you give |
| Triton | not tested | the scripts in this repository (`test/fixtures/make_*_fixture.py`) launch the kernel once and so need a GPU. Triton's ahead-of-time `triton.compile(ASTSource(...), target=GPUTarget("cuda", 86, 32))`, the call [tutorial 4](tutorials/4-triton-kernels.md) uses for Metal, should not, but this has not been tried |
| [`ka_compile`](@ref) | **no, currently** | the target is read from the device, the compiler is configured from the current device, and the CUDA backend only registers itself when `CUDA.jl` reports a working device. Build KA kernels inside the job (the first call does, and the result is cached per architecture) |

Registering is the other half. [`add_variant!`](@ref) files a variant under the *current device's* architecture
(`ops/<Package>.<op>/sm_86/<kernel>/`), and without a device that architecture is `""`, so a variant registered on the
GPU-less node would be looked up in the wrong place on the GPU node. So:

1. Build the binary files (`.cubin`, `.ptx`, and Triton's `.json`) wherever you like, and keep them in a directory the job
   can read.
2. **Register them at run time, on the GPU node**, ideally from `build!` ([tutorial 2](tutorials/2-registering-kernels.md)),
   which does it on first use. Registering only copies the file and writes a manifest, so it is cheap.
3. **Give prebuilt cubins their `target`** (`KernelBinary("x.cubin"; target="sm_86", …)`, or `triton_binary`, which reads it
   from Triton's metadata). The launch then compares it with the device and refuses a binary built for another architecture
   with an explanation. Without a `target` the file is launched unchecked and a mismatch surfaces as a driver error.
   A `.ptx` is portable and needs none.
4. Build for the architecture of the GPUs the job will get, not the login node's. A cubin runs on a device of the same major
   compute capability whose minor version is at least the cubin's; PTX runs on any newer device.

## Metal

Metal launches `.metallib` binaries (prebuilt, or compiled from a KA kernel by [`ka_compile`](@ref)) through Metal.jl.
Its KA binaries take the kernel-state word and KA's context bytes ahead of their arguments. A traced launch is a
`mps.metal_kernel_lib` custom call for the [jax-mps](https://github.com/tillahoffmann/jax-mps) plugin, which takes
arbitrary byte slots, so every binary can be traced and no buffer order is required. The `mps.metal_kernel_lib` handler is a proposal in a jax-mps pull request, not yet in a release. The traced tests
run only when a plugin that has it is found (`JAX_MPS_PLUGIN=/path/to/libpjrt_plugin_mps.dylib`).

## What is verified

| | status |
|:--|:--|
| Metal, eager and traced (CPU + Metal test suite on Apple silicon, jax-mps plugin) | tested |
| CUDA, eager: prebuilt cubin and PTX, KA binaries, target checks, timing | tested on an NVIDIA A40 |
| CUDA, traced: `:explicit` binaries in every buffer order, KA binaries (flat), one and two outputs, a real op (`SDPAOps`) | tested on an NVIDIA A40 |
| CUDA, Triton ABI (shared memory above 48 KB, trailing null pointers) | tested with the scale kernel dressed as a Triton one |
| CUDA, kernels compiled by Triton 3.2.0 and 3.8.0 (scale, block sum with shared memory, strided axpy, accumulate), eager and traced; also Float16 and BFloat16 with 3.2.0 | tested on an NVIDIA A40 against Triton's own output and host formulas |
| CUDA, a read-modify-write kernel on an `InOutArray` (eager) and an atomic accumulation (eager and traced) | tested on an NVIDIA A40 |
| CUDA, an `InOutArray` in a trace (the operand is aliased to the result; the argument itself stays unchanged) | tested on an NVIDIA A40 |
| CUDA, trifast's triangle-attention kernels (`_fwd`, `_bwd_q`, `_bwd_kv`, `_bwd_b`; `tl.dot`, 3-D grids, 29-40 arguments, up to 57 KB shared memory), cubin and PTX, eager, traced and chained | tested on an NVIDIA A40 against Triton's output and a dense float64 torch reference (float32 only) |
| ROCm | not implemented |

The tests are organised so one test base covers every backend. `test/gpu/` is generic: it runs on whichever backend
is found (Metal on Apple, CUDA on NVIDIA) and uses only what that backend's **adapter**, `test/ext/<backend>/backend.jl`,
provides (the device array type, the `scale` fixture, the traced client, ...). What is specific to one backend lives in
`test/ext/<backend>/` next to its adapter: for CUDA the encoding of the XLA handler's message, the flat KA layout, the
Triton ABI's launch parameters, real Triton kernels and trifast's triangle-attention kernels (from the committed
fixtures in `test/fixtures/`, no Python needed); for Metal the jax-mps custom call's buffer table. Two directories hold what every backend shares, and run on the
loaded backend like `test/gpu/` does: `test/ext/reactant/` (the custom-call seam on the backend's own Reactant client, and
variants built from inside a trace) and `test/ext/enzyme/` (the gradient of an op on the device against the host's, and
which kernels the gradient launched). The tests loop over `Float32`, `Float16` and `BFloat16` wherever the backend supports
them. Run them with `julia --project=test test/runtests.jl ext`. For a CUDA machine you can only reach remotely there
are also staged scripts, `test/ext/cuda/smoke.jl` (stages 1-16) and `test/ext/cuda/half_smoke.jl` (Float16 and BFloat16,
stages 1-12; `HALF_SMOKE_ONLY=10,10c` runs some), each printing PASS or FAIL.

## Limits and what is left

- `ka_compile` on CUDA, and registering variants, need the GPU to be present (see [Building without a GPU](@ref)).
- A single global device backend per session: [`current_target`](@ref) is `""` when more than one is loaded.
- Eager launches are synchronous; there are no asynchronous launches.
- Float32, Float16 and BFloat16 are supported. A prebuilt `bfloat` kernel on Metal needs Metal 3.1 (macOS 14); BFloat16 on CUDA
  needs sm_80 for the native conversions. Triton kernels have been tested in Float32 only. Float64 is out of scope.
  A 2-byte scalar cannot be passed to XLA's handler, so it is refused in a CUDA trace: pass a `Float32` scalar.
- A launch's argument types are checked only against a known signature: one `ka_compile` recorded, or one given as
  `signature=`. A prebuilt or Triton binary without one warns once and is not checked; the order and meaning of
  arguments of one type, shapes and strides are never checked.
- Variants and device queries that a trace triggers (building a missing variant, asking the device its architecture) run outside
  the tracer ([`untraced`](@ref)): on CUDA, the first query of a fresh process runs a subprocess that crashed when
  interpreted by Reactant.
- `InOutArray` is not available in a Metal trace (jax-mps cannot alias a result to an operand).
- Triton scratch memory (`global_scratch_size > 0`) and thread-block clusters are refused. Other metadata that changes a
  kernel's parameters or launch (`launch_cooperative_grid`, `launch_pdl`, tensor descriptors, `maxnreg`) is currently
  ignored, not refused; the arguments bound are checked against the compiled signature only if `triton_binary` is given a `signature=` (Triton's metadata has none); only Triton 3.8.0 and
  sm_86 have been tested.
- Each traced CUDA launch makes a zero buffer per output; its cost has not been measured.
- XLA's kernel-call handler keeps one loaded module per kernel name and dynamic shared-memory size for the life of the process, so two
  different binaries with the same entry name and size would run the first one's code (seen with a Triton kernel compiled for
  Float32, Float16 and BFloat16). KernelDispatch gives each distinct image under one name and size its own key by asking for 4 more
  bytes of dynamic shared memory per extra image (the first is unchanged; the kernel never uses them). Which image runs does not
  depend on the order they are traced in, but a process's second binary of a name does launch with a slightly larger shared allocation.
- ROCm needs an extension of its own and, most likely, its own traced story (see the checklist on the
  [backend interface](backend.md#Adding-a-backend)).
- Reactant's own source-tracing route for KA (`enzymexla_compile_gpu`) is not used: it compiles from Julia source
  that must be in the running process and embeds in-process pointers, so it cannot work from a binary.
