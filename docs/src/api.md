# API reference

```@meta
CurrentModule = KernelDispatch
```

```@docs
KernelDispatch
```

## Ops and kernels

```@docs
AbstractKernelOp
AbstractKernel
KernelBinary
OutArray
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
bind_slots
max_threads
device_time
```

## KernelAbstractions kernels

```@docs
ka_compile
ka_binary_cache_dir
ka_context
ka_grid
ka_backend_object
ka_state_words
ka_prelude
```

## Backends

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
device_dense
device_zeros
custom_call
cache_root
realtype
LOADED_BACKENDS
encode_launch
```

## Autodiff and tracing

```@docs
tracing
traced_autodiff
KERNEL_IN_TRACED_AUTODIFF
```
