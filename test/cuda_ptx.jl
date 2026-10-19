# An `:explicit` CUDA kernel binary for the tests and the probe: hand-written PTX with a pointer-and-scalar
# signature, assembled to a cubin with the `ptxas` that CUDA.jl ships. "Explicit" is the point: unlike a
# KernelAbstractions kernel (CUDA.jl's `KernelState` and KA's context are struct arguments) it can be
# launched from a trace, and unlike a Julia kernel it needs no GPUCompiler. Needs the CUDA module as `GPU_PKG`
# (gpu_backend.jl defines it; a script that does `using CUDA` sets `const GPU_PKG = CUDA`).
# The PTX targets sm_75, the oldest architecture CUDA 13 still builds for: older devices are not supported.
#
#     scale(x, y, a, n):   y[i] = a * x[i]  for i < n          (inputs before outputs: XLA's buffer order)

const SCALE_PTX = """
.version 6.3
.target sm_75
.address_size 64

.visible .entry scale(
    .param .u64 x,
    .param .u64 y,
    .param .f32 a,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .f32 %f<4>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [x];
    ld.param.u64 %rd2, [y];
    ld.param.f32 %f1, [a];
    ld.param.u32 %r1, [n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;
    cvta.to.global.u64 %rd3, %rd1;
    cvta.to.global.u64 %rd4, %rd2;
    mul.wide.u32 %rd5, %r5, 4;
    add.s64 %rd6, %rd3, %rd5;
    add.s64 %rd7, %rd4, %rd5;
    ld.global.f32 %f2, [%rd6];
    mul.f32 %f3, %f2, %f1;
    st.global.f32 [%rd7], %f3;
DONE:
    ret;
}
"""

"""
    scale_binaries(dir) -> (ptx=KernelBinary, cubin=KernelBinary)

Write `SCALE_PTX` into `dir` and assemble it for the current device. Both are `:explicit` binaries
for the kernel `scale`, 64 threads per block, one item per thread.
"""
function scale_binaries(dir::AbstractString; threadgroup=64)
    CUDACore = GPU_PKG.CUDACore
    ptx = joinpath(dir, "scale.ptx")
    cubin = joinpath(dir, "scale.cubin")
    write(ptx, SCALE_PTX)
    cap = GPU_PKG.capability(GPU_PKG.device())
    arch = "sm_$(cap.major)$(cap.minor)"
    run(`$(CUDACore.CUDA_Compiler.ptxas()) --gpu-name $arch --output-file $cubin $ptx`)
    kw = (; entry="scale", threadgroup)
    # The cubin is for this device's architecture; the PTX is portable, so names none.
    return (ptx=KernelBinary(ptx; kw...), cubin=KernelBinary(cubin; kw..., target=arch))
end
