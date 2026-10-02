// Small elementwise / reduction kernels (M1, unfused).
#include <cfloat>

#include "kernels.h"

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// block-wide sum, blockDim.x multiple of 32 and <= 1024
__device__ float block_sum(float v) {
    __shared__ float sh[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = blockDim.x >> 5;
    float t = lane < nw ? sh[lane] : 0.f;
    t = warp_sum(t);
    return t;  // valid in every thread
}

__global__ void k_rmsnorm(const float* x, const float* w, float* y, int n, float eps) {
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss);
    const float scale = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = (x[i] * scale) * w[i];
}

void launch_rmsnorm(const float* x, const float* w, float* y, int n, float eps, cudaStream_t s) {
    k_rmsnorm<<<1, 1024, 0, s>>>(x, w, y, n, eps);
}

__global__ void k_rmsnorm_heads(const float* x, const float* w, float* y, int hdim, int in_stride, int out_stride,
                                float eps) {
    const float* xh = x + (int64_t)blockIdx.x * in_stride;
    float* yh = y + (int64_t)blockIdx.x * out_stride;
    float ss = 0.f;
    for (int i = threadIdx.x; i < hdim; i += blockDim.x) ss += xh[i] * xh[i];
    ss = block_sum(ss);
    const float scale = rsqrtf(ss / hdim + eps);
    for (int i = threadIdx.x; i < hdim; i += blockDim.x) yh[i] = (xh[i] * scale) * w[i];
}

void launch_rmsnorm_heads(const float* x, const float* w, float* y, int nheads, int hdim, int in_stride,
                          int out_stride, float eps, cudaStream_t s) {
    k_rmsnorm_heads<<<nheads, 128, 0, s>>>(x, w, y, hdim, in_stride, out_stride, eps);
}

__global__ void k_add(float* h, const float* a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = h[i] + a[i];
}

void launch_add(float* h, const float* a, int n, cudaStream_t s) { k_add<<<(n + 255) / 256, 256, 0, s>>>(h, a, n); }

__global__ void k_silu_mul(const float* g, const float* u, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float x = g[i];
        out[i] = (x / (1.0f + expf(-x))) * u[i];
    }
}

void launch_silu_mul(const float* g, const float* u, float* out, int n, cudaStream_t s) {
    k_silu_mul<<<(n + 255) / 256, 256, 0, s>>>(g, u, out, n);
}

__global__ void k_argmax(const float* x, int n, int* out_idx, float* out_val) {
    __shared__ float sv[1024];
    __shared__ int si[1024];
    float best = -FLT_MAX;
    int bi = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (x[i] > best) { best = x[i]; bi = i; }
    sv[threadIdx.x] = best;
    si[threadIdx.x] = bi;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            const float ov = sv[threadIdx.x + st];
            const int oi = si[threadIdx.x + st];
            if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) {
                sv[threadIdx.x] = ov;
                si[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { *out_idx = si[0]; if (out_val) *out_val = sv[0]; }
}

void launch_argmax(const float* x, int n, int* out_idx, float* out_val, cudaStream_t s) {
    k_argmax<<<1, 1024, 0, s>>>(x, n, out_idx, out_val);
}
