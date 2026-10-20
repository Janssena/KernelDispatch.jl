# The CUDA adapter for the generic GPU tests (test/gpu/*): everything they need that differs by backend. The
# contract is listed in test/gpu_backend.jl; test/ext/metal/backend.jl is the same for Metal. Included by gpu_backend.jl,
# after `GPU`, `GPU_PKG` (CUDA) and `GPU_TAG` are defined. CUDA-only tests live next to this file.

include(joinpath(@__DIR__, "ptx.jl"))               # SCALE_PTX, `scale_binaries`: the hand-written explicit kernel

const DeviceArray = GPU_PKG.CuArray

device_max_threads() = Int(GPU_PKG.attribute(GPU_PKG.device(), GPU_PKG.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))

wrapped_array_type() = Base.get_extension(KernelDispatch, :KernelDispatchCUDAExt).WrappedCuArray

"""
The element types the backend's `:explicit` `scale` binaries exist for. Float32 only: the half-precision PTX kernels are
written in test/ext/cuda/half_smoke.jl, to be run on a GPU before they join the generic suite.
"""
supported_eltypes() = (Float32,)

"The backend's `:explicit` `scale` binaries (`y[i] = a * x[i]`), as `label => KernelBinary`, built in `dir`."
function explicit_binaries(dir, ::Type{T}=Float32) where {T}
    T === Float32 || throw(ArgumentError("no explicit CUDA `scale` binary for $T"))
    b = scale_binaries(dir)
    return ["cubin" => b.cubin, "ptx" => b.ptx]
end

"A file in a format this backend does not launch."
foreign_binary(dir) = (f = joinpath(dir, "scale.metallib"); write(f, UInt8[1, 2, 3]);
    KernelBinary(f; entry="scale", threadgroup=64))

"The Reactant client a traced launch runs on: Reactant's own CUDA client."
traced_client() = "gpu"

"What a traced launch's custom call is named in the emitted MLIR."
const TRACED_CALL_TARGET = "triton_kernel_call_ffi"

"The hidden leading arguments of a KernelAbstractions binary (`ka_state_words`): none, CUDA binaries are built flat."
const EXPECTED_KA_STATE_WORDS = ()

"SDPAOps kernels shipped as prebuilt binaries for this backend: none (`FlashAttn` is a `.metallib`)."
prebuilt_sdpa_kernels() = ()
