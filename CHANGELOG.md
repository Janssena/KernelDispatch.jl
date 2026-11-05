# Changelog

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
