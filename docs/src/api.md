# API reference

```@meta
CurrentModule = KernelDispatch
```

This page is the API: what a user calls and what a kernel author defines. What a *backend* implements (a new
GPU vendor, a new binary ABI) is on the [backend interface](backend.md) page and may change.

```@docs
KernelDispatch
```

## Ops and kernels

```@docs
AbstractKernelOp
AbstractKernel
KernelBinary
OutArray
InOutArray
```

The methods a kernel or op defines:

```@docs
opname
kernelname
host
forward
backward
variant_key
extras
nearest
build!
source_tag
cache_dir
```

## Selecting a kernel

```@docs
@default_kernel
default_kernel
default_candidates
use_kernel!
reset_kernel!
current_kernel
selected_kernel
list_kernels
set_default_kernel!
clear_default_kernel!
```

## Registering variants

```@docs
register_kernel!
add_variant!
variants
variant
```

## Launching binaries

```@docs
call_binary
run_binary
prepare_launch
Launch
execute
time_binary
record_bindings
max_threads
```

## KernelAbstractions kernels

```@docs
ka_compile
ka_binary_cache_dir
```

## Triton kernels

```@docs
triton_binary
```

## Backends and devices

```@docs
KernelBackend
CPUBackend
MetalBackendTag
CUDABackendTag
ROCmBackendTag
UnknownBackend
GPUBackend
backend_of
device_backend
device_available
current_target
device_zeros
cache_root
realtype
```

## Autodiff and tracing

```@docs
tracing
traced_autodiff
KERNEL_IN_TRACED_AUTODIFF
```
