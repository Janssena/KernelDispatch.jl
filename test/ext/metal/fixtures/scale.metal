// The `:explicit`-ABI kernel of the conformance suite (test/gpu/conformance.test.jl): y[i] = a * x[i] for i < n.
// Buffers, then scalars, in the order the caller binds them (a scalar is a one-element buffer). Compile with
//
//     xcrun -sdk macosx metal -std=metal3.1 scale.metal -o scale.metallib
//
// `scale` is float; `scale_f16` and `scale_bf16` take half and bfloat buffers and still a float scalar `a` (a 2-byte
// scalar cannot cross every path), computing in float and converting once on the store.

#include <metal_stdlib>
using namespace metal;

kernel void scale(device const float* x [[buffer(0)]],
                  device float* y [[buffer(1)]],
                  constant float& a [[buffer(2)]],
                  constant uint& n [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = a * x[i];
}

kernel void scale_f16(device const half* x [[buffer(0)]],
                      device half* y [[buffer(1)]],
                      constant float& a [[buffer(2)]],
                      constant uint& n [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = half(a * float(x[i]));
}

#if __METAL_VERSION__ >= 310
kernel void scale_bf16(device const bfloat* x [[buffer(0)]],
                       device bfloat* y [[buffer(1)]],
                       constant float& a [[buffer(2)]],
                       constant uint& n [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = bfloat(a * float(x[i]));
}
#endif
