# Which GPU backend, if any, the GPU section of the suite runs on.
#
# Candidates are tried in order -- Metal (Apple only), then CUDA -- and the first whose package is in the
# test environment, loads, reports `functional()` AND has its KernelDispatch extension register the device
# (`device_available()`) wins. Nothing else is loaded, so a CPU-only run never touches a GPU package.
#
#     KD_TEST_GPU=none    skip the GPU section even on a GPU machine
#     KD_TEST_GPU=metal   require Metal (error if it cannot be used);  likewise `cuda`
#
# Include this at top level, before any testset: the package it loads has to be visible (world age) to
# the files included afterwards. It defines `GPU` (`:metal`, `:cuda` or `nothing`) and, when there is one, `GPU_PKG`
# (the Metal/CUDA module), `GPU_TAG` (KernelDispatch's tag for it), and includes the backend's ADAPTER,
# test/ext/<backend>/backend.jl, which defines the rest of what the generic tests in test/gpu/ need:
#
#     DeviceArray                the device array type (`MtlArray`/`CuArray`)
#     device_max_threads()       the device's per-threadgroup (per-block) thread limit
#     wrapped_array_type()       the extension's union of array wrappers that are not a plain device array
#     explicit_binaries(dir)     `label => KernelBinary` of the `:explicit` `scale` fixture (y[i] = a * x[i]), built in `dir`
#     foreign_binary(dir)        a binary in a format the OTHER vendors use, which this backend must refuse
#     traced_client()            the Reactant client for traced launches, or `nothing` when the machine has none
#     TRACED_CALL_TARGET         the custom call's name in emitted MLIR ("mps.metal_kernel_lib", "triton_kernel_call_ffi")
#     EXPECTED_KA_STATE_WORDS    what `ka_state_words(GPU_TAG)` must be
#     prebuilt_sdpa_kernels()    SDPAOps kernels shipped as prebuilt binaries for this backend (FlashAttn on Metal)
#
# The generic tests use only these names, never a backend's own. What is specific to one backend is tested in
# test/ext/<backend>/, next to its adapter; a new backend adds a directory there with a `backend.jl`.

using KernelDispatch

const _GPU_PACKAGES = (metal=:Metal, cuda=:CUDA)

# -> (backend::Union{Symbol,Nothing}, module or nothing, why-not::String)
function _detect_gpu()
    want = lowercase(get(ENV, "KD_TEST_GPU", ""))
    want == "none" && return (nothing, nothing, "KD_TEST_GPU=none")
    want in ("", "metal", "cuda") || error("KD_TEST_GPU must be none, metal or cuda; got `$want`")
    names = isempty(want) ? collect(keys(_GPU_PACKAGES)) : [Symbol(want)]
    why = String[]
    for name in names
        pkg = _GPU_PACKAGES[name]
        if name === :metal && !Sys.isapple()
            push!(why, "Metal: not an Apple machine")
        elseif Base.find_package(String(pkg)) === nothing
            push!(why, "$pkg: not in the test environment")
        else
            mod = try
                Base.require(Main, pkg)
            catch err
                push!(why, "$pkg: failed to load ($(sprint(showerror, err)))")
                continue
            end
            # `invokelatest`: `mod` was loaded a moment ago, in a newer world than this function's.
            ok = try
                Base.invokelatest(() -> mod.functional())
            catch
                false
            end
            if !ok
                push!(why, "$pkg: no functional device")
            elseif !Base.invokelatest(KernelDispatch.device_available)
                push!(why, "$pkg: functional, but KernelDispatch's extension did not register the device")
            else
                return (name, mod, "")
            end
        end
    end
    isempty(want) || error("KD_TEST_GPU=$want requested, but: $(join(why, "; "))")
    return (nothing, nothing, join(why, "; "))
end

const GPU, GPU_PKG, GPU_WHY = _detect_gpu()

if GPU === nothing
    @info "no GPU backend: the GPU section is skipped" reason = GPU_WHY
else
    @info "GPU backend for the GPU section" backend = GPU
    const GPU_TAG = KernelDispatch.device_backend()
    include(joinpath(@__DIR__, "ext", String(GPU), "backend.jl"))
end
