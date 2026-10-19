# The Metal adapter for the generic GPU tests (test/gpu/*): everything they need that differs by backend. The
# contract is listed in test/gpu_backend.jl; test/ext/cuda/backend.jl is the same for CUDA. Included by gpu_backend.jl,
# after `GPU`, `GPU_PKG` (Metal) and `GPU_TAG` are defined. Metal-only tests live next to this file.

include(joinpath(@__DIR__, "mps_client.jl"))       # `mps_client()`: the jax-mps PJRT client, or nothing

const DeviceArray = GPU_PKG.MtlArray

device_max_threads() = Int(GPU_PKG.device().maxThreadsPerThreadgroup.width)

wrapped_array_type() = Base.get_extension(KernelDispatch, :KernelDispatchMetalExt).WrappedMtlArray

"The backend's `:explicit` `scale` binaries (`y[i] = a * x[i]`), as `label => KernelBinary`, copied into `dir`."
function explicit_binaries(dir)
    file = joinpath(dir, "scale.metallib")
    cp(joinpath(@__DIR__, "fixtures", "scale.metallib"), file)
    return ["metallib" => KernelBinary(file; entry="scale", threadgroup=64)]
end

"A file in a format this backend does not launch."
foreign_binary(dir) = (f = joinpath(dir, "scale.cubin"); write(f, UInt8[1, 2, 3]);
    KernelBinary(f; entry="scale", threadgroup=64))

"The Reactant client a traced launch runs on, or `nothing` when this machine has none (no jax-mps plugin)."
traced_client() = mps_client()

"What a traced launch's custom call is named in the emitted MLIR."
const TRACED_CALL_TARGET = "mps.metal_kernel_lib"

"The hidden leading arguments of a KernelAbstractions binary (`ka_state_words`): Metal.jl's kernel-state word."
const EXPECTED_KA_STATE_WORDS = (UInt32(0),)

"SDPAOps kernels shipped as prebuilt binaries for this backend (`FlashAttn` is a `.metallib`)."
prebuilt_sdpa_kernels() = (FlashAttn(),)
