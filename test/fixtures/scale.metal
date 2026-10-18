// The plain-ABI kernel of the conformance suite (test/gpu/conformance.test.jl): y[i] = a * x[i] for i < n.
// Buffers, then scalars, in the order the caller binds them (a scalar is a one-element buffer). Compile with
//
//     xcrun -sdk macosx metal scale.metal -o scale.metallib

#include <metal_stdlib>
using namespace metal;

kernel void scale(device const float* x [[buffer(0)]],
                  device float* y [[buffer(1)]],
                  constant float& a [[buffer(2)]],
                  constant uint& n [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = a * x[i];
}
