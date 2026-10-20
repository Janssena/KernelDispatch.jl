"""
    KernelDispatch

Run operations as compiled kernels: eagerly on device arrays, embedded as custom calls in a
Reactant-compiled program, or as plain Julia on cpu. Kernels are switchable per op and tuned
variants picked from the inputs. See the documentation for tutorials and the API reference.
"""
module KernelDispatch

import Preferences

using TOML
using KernelAbstractions

include("backend.jl")
include("kernel.jl")
include("formats.jl")
include("preferences.jl")
include("call.jl")
include("signature.jl")
include("ka.jl")
include("triton.jl")

export AbstractKernelOp, AbstractKernel, KernelBinary, OutArray, InOutArray
export register_kernel!, add_variant!, use_kernel!, current_kernel, list_kernels
export reset_kernel!, set_default_kernel!, clear_default_kernel!, selected_kernel
export call_binary, run_binary, prepare_launch, time_binary, ka_compile, triton_binary, backend_of

public opname, default_kernel, @default_kernel, default_candidates, kernelname, host, forward, backward,
    variant_key, extras, nearest, build!, source_tag, cache_dir, variants, variant, record_bindings, execute,
    Launch, KernelBackend, CPUBackend, MetalBackendTag, CUDABackendTag, ROCmBackendTag, UnknownBackend,
    GPUBackend, realtype, traced_autodiff, KERNEL_IN_TRACED_AUTODIFF, tracing, untraced, cache_root, ka_binary_cache_dir,
    current_target, device_available, device_zeros, max_threads, device_backend

"""
    device_zeros(proto, T, dims...) -> array

A zero-filled array of `T` on the same device (or in the same trace) as `proto`: for `forward`
overloads that need a placeholder. The Reactant extension specialises it for traced arrays.
"""
device_zeros(proto, ::Type{T}, dims::Integer...) where {T} = 
    fill!(similar(proto, T, Int.(dims)), zero(T))

end
