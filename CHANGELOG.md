# Changelog

## 0.2.2 - 2026-11-05

### Added

- **Device utilities, `public`:** `to_device(backend, x)` (any `AbstractArray` to a device array), `device_caps(backend)`
  with its `DeviceCaps(warp, max_threads, max_shmem, name)` (the device's limits; `DeviceCaps.max_threads` is the device
  ceiling, `max_threads(binary)` a compiled binary's own), and `release!(backend, x)` (free a device array now; a no-op
  where the backend has nothing to free). Moved here from LuxTriangleAttention, where they were methods on these same
  backend tags. Without an extension behind the tag `to_device` and `device_caps` throw `ArgumentError`. On CUDA
  `DeviceCaps.max_shmem` is the static shared memory a block may use (48 KiB, which is what KernelAbstractions'
  `@localmem` can get), not the device's opt-in maximum, and `name` includes the compute capability (`"... sm_86"`).
  Documented under "Device utilities" in the API reference and in the backend page's table. Tests:
  `test/cpu/device_utils.test.jl`, `test/gpu/device_utils.test.jl`, `test/ext/cuda/device_caps.test.jl`.
- **Opt-in asynchronous eager launches.** With `KERNELDISPATCH_ASYNC_LAUNCHES=1` in the environment (read when the package
  loads), or `KernelDispatch.ASYNC_LAUNCHES[] = true`, `encode_launch` (and so `call_binary`, `run_binary` and `execute`)
  returns once the dispatch is enqueued instead of waiting for the device. **Off by default: launches still wait, as before.**
  - Metal: the dispatch is encoded into a command buffer derived from Metal.jl's task-local queue (`Metal.global_queue`)
    and committed. Committing flushes Metal.jl's open batch first, so the kernel is ordered after everything the task
    enqueued before it. 1000 launches of a small kernel took 225 ms waiting and 26 ms asynchronous on an M1 Pro.
  - CUDA: the per-launch `CUDACore.synchronize()` is skipped; the launch is ordered on the task's stream. Tested on an
    NVIDIA A40 (sm_86).
  - Ordering is task-local, as in Metal.jl and CUDA.jl: work launched on one task and consumed on another needs a
    `Metal.synchronize()` / `CUDA.synchronize()` first (on Metal a reader that skips it mostly sees stale data). A device-side
    error surfaces at the next synchronization point, not at the launch.
  - `device_time`, `time_binary` and `max_threads` always wait. Traced launches are unchanged.
- `test/gpu/async_launch.test.jl` (section `async`): launches interleaved with the backend's own work, no
  synchronization in between; an input freed straight after the launch that reads it; a temporary collected by the GC.
  `test/probes/async_hazards.jl` measures the cross-task read race (not part of the suite).

### Changed

- The default eager Metal launch is encoded on Metal.jl's queue instead of a queue of KernelDispatch's own, and still
  waits. Results are unchanged.

### Fixed

- **Enzyme: gradients for ops whose arguments or results are `NamedTuple`s or `Tuple`s.** An op that takes a layer's
  parameters as a `NamedTuple` failed in the reverse rule, which added the gradient with `a.dval .+= g`, and an op that
  returns a layer's updated state had no shadow of the right type. The gradient is now accumulated field by field
  (arrays in place, a `nothing` gradient skipped, `g` may name fewer fields than the shadow), and the shadow of a returned
  `NamedTuple` / `Tuple` is built field by field: arrays zeroed, anything else (an `Int`, a `Symbol`) kept as is.


## 0.2.1 - 2026-10-27

### Fixed

- **Traced kernel calls declare their layouts.** Every operand and result of the `stablehlo.custom_call` now carries
  `operand_layouts` / `result_layouts` set to Julia's column-major memory order. Without them XLA could pick another layout
  for an operand it computes itself (its GPU compiler does), and the kernel then read the data in a different order with no
  error: a silent wrong result on CUDA. Documented in the backend pages, with a traced test.

## 0.2.0 - 2026-10-20

### Added

- **Float16 and BFloat16.** Supported on Metal and CUDA, eager and traced. `SDPAOps` (the example op) works in any
  precision, and traced custom calls take `result_eltypes`, so an output keeps its element type through the trace instead of
  being given the operands'. New generic tests cover the precisions, and `test/ext/cuda/half_smoke.jl` is a staged check on
  CUDA hardware.
- **Signatures: a binary says which argument types it was compiled for.** `KernelBinary` has a `signature` (one token per bound
  argument, such as `("f16", "f16", "s:f32", "s:i32")`; `Val`s take no slot), declared as the types you pass
  (`signature=(Vector{Float16}, Vector{Float16}, Float32, Int32)`). `call_binary`, `run_binary`, `prepare_launch` and
  `time_binary` check the arguments against it and throw, naming the argument, instead of running a kernel on data of the
  wrong type (which computes something else without an error). `check_types=false` skips the check for one launch;
  `signature=false` declares a binary unchecked; a binary with no signature warns once on its first launch and is not
  checked. `ka_compile` records the signature itself, and `ka_signature_tag(argtypes)` names the tag that goes into its file
  names. A new `docs/src/signatures.md` page describes it.
- `untraced(f)` (public): runs `f()` outside the Reactant tracer. Building a variant (`build!`) and checking a binary against
  the device's architecture no longer run inside a trace, where compiling a GPU kernel could crash the process.
- `KernelBinary(b; ...)`, a copy of a binary with some fields changed, which cannot forget a field such as `signature`.
- Docs on precision and signatures (README, `backends.md`) and new generic tests: type mismatches
  (`test/gpu/mismatch.test.jl`), the earlier Metal and CUDA traced and gradient tests re-organised into backend-neutral
  Reactant and Enzyme extension tests.

### Changed

- `ka_compile` file names and registry entries include the signature tag, so a binary built for one set of types is never
  picked up for another. Binaries built by 0.1.0 are named differently and are rebuilt.
- The Triton test fixtures moved out of the repository into a versioned Artifact (`kernel_dispatch_fixtures`, Triton 3.8.0,
  with a folder per version and element type), which keeps the repository small. The README was simplified and the docs got a
  stable link.

## 0.1.0 - 2026-10-19

First release.

### Added

- **Ops, kernels and variants.** An op (`AbstractKernelOp`) has several kernels (`AbstractKernel`) you switch between at
  runtime with `use_kernel!`, each with tuned variants (`add_variant!`) picked from the inputs. `@default_kernel` gives an op
  its default, and `set_default_kernel!` persists the choice as a package preference. Calling the op picks the CPU
  (`host`) or the device path from its arguments.
- **Launching compiled binaries.** `KernelBinary` and `call_binary`, `run_binary`, `prepare_launch` and `time_binary`, with
  `OutArray` for allocated outputs and `InOutArray` for an array updated in place. `time_binary` reports the device's own
  timestamps, and a threadgroup over the pipeline's limit is refused.
- **Backends.** Metal runs `.metallib` binaries (prebuilt, or compiled from a KernelAbstractions kernel); CUDA runs `.cubin`
  and `.ptx` files, KernelAbstractions kernels and kernels compiled by Triton (`triton_binary`). A binary carries a `target`
  (the architecture it was built for), which is checked against the device.
- **KernelAbstractions kernels as a source of binaries** (`ka_compile`). On CUDA they are built flat, without CUDA.jl's kernel
  state, so one file launches eagerly and from a trace; the price is that such a kernel must not be able to throw.
- **Reactant.** A binary embeds in a compiled program as a `stablehlo.custom_call`: on Metal through jax-mps
  (`mps.metal_kernel_lib`), on CUDA through XLA's `triton_kernel_call_ffi` handler. A kernel's buffers may come in any order.
- **Enzyme.** Custom `forward` and `backward` for an op, with gradients through Enzyme.
- **Docs and CI.** Documentation with four tutorials (switching kernels, registering kernels, forward and backward with
  Enzyme, Triton kernels), a backend-author guide, a "building without a GPU" section, an API reference, and the CPU tests
  in GitHub Actions. Metal was tested on Apple silicon and CUDA on an NVIDIA A40 (sm_86).
