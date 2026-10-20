// Scaled dot-product attention, one thread per query: a hand-written Metal kernel standing in for a
// binary from another toolchain (Triton, say). Compile with
//
//     xcrun -sdk macosx metal -std=metal3.1 sdpa.metal -o sdpa.metallib
//
// (`-std=metal3.1` for the `bfloat` kernel; it needs macOS 14 to load.)
//
// Three entry points, one per element type: `sdpa_fwd` (float, accumulating in the output), and `sdpa_fwd_f16`
// and `sdpa_fwd_bf16` (half and bfloat buffers, every sum in float; see the end of the file).
//
// Column-major (Julia) layout: q is d×n, k is d×m, v is dv×m, o is dv×n; each column is one
// query/key/value. Argument order, Triton style: inputs, the output, then sizes and the scale.
// KernelDispatch binds every argument as a buffer (a scalar too), in order, from index 0.

#include <metal_stdlib>
using namespace metal;

kernel void sdpa_fwd(device const float* q [[buffer(0)]],
                     device const float* k [[buffer(1)]],
                     device const float* v [[buffer(2)]],
                     device float* o       [[buffer(3)]],
                     constant int& n       [[buffer(4)]],
                     constant int& m       [[buffer(5)]],
                     constant int& d       [[buffer(6)]],
                     constant int& dv      [[buffer(7)]],
                     constant float& scale [[buffer(8)]],
                     uint group  [[threadgroup_position_in_grid]],
                     uint lane   [[thread_position_in_threadgroup]],
                     uint tgsize [[threads_per_threadgroup]]) {
    int i = group * tgsize + lane;           // this thread's query, 0-based
    if (i >= n) return;
    device const float* qi = q + i * d;
    device float* oi = o + i * dv;

    float mx = -INFINITY;                    // pass 1: the largest score, for a stable softmax
    for (int j = 0; j < m; ++j) {
        float s = 0.0f;
        for (int t = 0; t < d; ++t) s += qi[t] * k[j * d + t];
        mx = max(mx, s * scale);
    }
    float denom = 0.0f;                      // pass 2: weights and the weighted sum of values
    for (int c = 0; c < dv; ++c) oi[c] = 0.0f;
    for (int j = 0; j < m; ++j) {
        float s = 0.0f;
        for (int t = 0; t < d; ++t) s += qi[t] * k[j * d + t];
        float p = exp(s * scale - mx);
        denom += p;
        for (int c = 0; c < dv; ++c) oi[c] += p * v[j * dv + c];
    }
    for (int c = 0; c < dv; ++c) oi[c] /= denom;
}


// --- half and bfloat -------------------------------------------------------------------------------
//
// The same algorithm on `T` buffers (half or bfloat), every sum in float: the scores, the softmax denominator and the
// weighted sum of values. The sum lives in `acc`, a float workspace of the output's shape (one more output, bound after
// `o`); `o` is written once, converted, at the end. Argument order: q, k, v, o, acc, then n, m, d, dv, scale.

template <typename T>
inline void sdpa_fwd_acc(device const T* q, device const T* k, device const T* v, device T* o, device float* acc,
                         int n, int m, int d, int dv, float scale, int i) {
    if (i >= n) return;
    device const T* qi = q + i * d;
    device float* ai = acc + i * dv;
    device T* oi = o + i * dv;

    float mx = -INFINITY;
    for (int j = 0; j < m; ++j) {
        float s = 0.0f;
        for (int t = 0; t < d; ++t) s += float(qi[t]) * float(k[j * d + t]);
        mx = max(mx, s * scale);
    }
    float denom = 0.0f;
    for (int c = 0; c < dv; ++c) ai[c] = 0.0f;
    for (int j = 0; j < m; ++j) {
        float s = 0.0f;
        for (int t = 0; t < d; ++t) s += float(qi[t]) * float(k[j * d + t]);
        float p = exp(s * scale - mx);
        denom += p;
        for (int c = 0; c < dv; ++c) ai[c] += p * float(v[j * dv + c]);
    }
    for (int c = 0; c < dv; ++c) oi[c] = T(ai[c] / denom);
}

kernel void sdpa_fwd_f16(device const half* q [[buffer(0)]],
                         device const half* k [[buffer(1)]],
                         device const half* v [[buffer(2)]],
                         device half* o       [[buffer(3)]],
                         device float* acc    [[buffer(4)]],
                         constant int& n      [[buffer(5)]],
                         constant int& m      [[buffer(6)]],
                         constant int& d      [[buffer(7)]],
                         constant int& dv     [[buffer(8)]],
                         constant float& scale [[buffer(9)]],
                         uint i [[thread_position_in_grid]]) {
    sdpa_fwd_acc<half>(q, k, v, o, acc, n, m, d, dv, scale, i);
}

#if __METAL_VERSION__ >= 310
kernel void sdpa_fwd_bf16(device const bfloat* q [[buffer(0)]],
                          device const bfloat* k [[buffer(1)]],
                          device const bfloat* v [[buffer(2)]],
                          device bfloat* o       [[buffer(3)]],
                          device float* acc      [[buffer(4)]],
                          constant int& n        [[buffer(5)]],
                          constant int& m        [[buffer(6)]],
                          constant int& d        [[buffer(7)]],
                          constant int& dv       [[buffer(8)]],
                          constant float& scale  [[buffer(9)]],
                          uint i [[thread_position_in_grid]]) {
    sdpa_fwd_acc<bfloat>(q, k, v, o, acc, n, m, d, dv, scale, i);
}
#endif
