// Full-attention decode kernels (M1, unfused): per-head q/k RMSNorm + partial NeoX RoPE, fp16 KV append,
// naive softmax attention with GQA (q head h uses kv head h / 6), sigmoid output gate.
#include <cuda_fp16.h>

#include <cfloat>

#include "kernels.h"

namespace {
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// blockDim.x == 256
__device__ float bsum256(float v, float* sh) {
    v = wsum(v);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) t += sh[i];
    return t;
}
__device__ float bmax256(float v, float* sh) {
    v = wmax(v);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = -FLT_MAX;
#pragma unroll
    for (int i = 0; i < 8; i++) t = fmaxf(t, sh[i]);
    return t;
}
}  // namespace

// blocks 0..23: q heads, 24..27: k heads. blockDim 256 (= head_dim)
__global__ void k_qk_norm_rope(const float* qfull, const float* k, const float* qw, const float* kw, float* qn,
                               float* kn, int pos, float eps, float theta_scale, int n_rot) {
    __shared__ float red[8];
    __shared__ float y[256];
    const int b = blockIdx.x, d = threadIdx.x;
    const bool isq = b < 24;
    const float* src = isq ? qfull + b * 512 : k + (b - 24) * 256;
    const float* w = isq ? qw : kw;
    float* dst = isq ? qn + b * 256 : kn + (b - 24) * 256;
    const float x = src[d];
    const float ss = bsum256(x * x, red);
    const float scale = rsqrtf(ss / 256.0f + eps);
    y[d] = (x * scale) * w[d];
    __syncthreads();
    const int half_rot = n_rot / 2;
    if (d < half_rot) {
        const float theta = (float)pos * powf(theta_scale, (float)d);
        const float c = cosf(theta), s = sinf(theta);
        const float x0 = y[d], x1 = y[d + half_rot];
        dst[d] = x0 * c - x1 * s;
        dst[d + half_rot] = x0 * s + x1 * c;
    } else if (d >= n_rot) {
        dst[d] = y[d];
    }
}

void launch_qk_norm_rope(const float* qfull, const float* k, const float* qw, const float* kw, float* qn, float* kn,
                         int pos, float eps, float freq_base, int n_rot, cudaStream_t s) {
    const float theta_scale = powf(freq_base, -2.0f / n_rot);
    k_qk_norm_rope<<<28, 256, 0, s>>>(qfull, k, qw, kw, qn, kn, pos, eps, theta_scale, n_rot);
}

__global__ void k_kv_store(const float* k, const float* v, __half* kc, __half* vc, int pos, int max_ctx) {
    const int j = blockIdx.x, d = threadIdx.x;
    const int64_t o = ((int64_t)j * max_ctx + pos) * 256 + d;
    kc[o] = __float2half_rn(k[j * 256 + d]);
    vc[o] = __float2half_rn(v[j * 256 + d]);
}

void launch_kv_store(const float* k, const float* v, uint16_t* kc, uint16_t* vc, int pos, int max_ctx, cudaStream_t s) {
    k_kv_store<<<4, 256, 0, s>>>(k, v, (__half*)kc, (__half*)vc, pos, max_ctx);
}

__global__ void k_attn_decode(const float* q, const __half* kc, const __half* vc, float* out, float* scores, int n_kv,
                              int max_ctx, float scale) {
    __shared__ float qs[256];
    __shared__ float red[8];
    const int h = blockIdx.x, j = h / 6;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    qs[tid] = q[h * 256 + tid];
    __syncthreads();
    const __half* K = kc + (int64_t)j * max_ctx * 256;
    const __half* V = vc + (int64_t)j * max_ctx * 256;
    float* sc = scores + (int64_t)h * max_ctx;
    for (int t = warp; t < n_kv; t += 8) {
        const __half* kr = K + (int64_t)t * 256;
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) a += qs[lane + 32 * i] * __half2float(kr[lane + 32 * i]);
        a = wsum(a);
        if (lane == 0) sc[t] = a * scale;
    }
    __syncthreads();
    float m = -FLT_MAX;
    for (int t = tid; t < n_kv; t += 256) m = fmaxf(m, sc[t]);
    m = bmax256(m, red);
    float sum = 0.f;
    for (int t = tid; t < n_kv; t += 256) {
        const float e = expf(sc[t] - m);
        sc[t] = e;
        sum += e;
    }
    sum = bsum256(sum, red);
    __syncthreads();
    float acc = 0.f;
    for (int t = 0; t < n_kv; t++) acc += sc[t] * __half2float(V[(int64_t)t * 256 + tid]);
    out[h * 256 + tid] = acc / sum;
}

void launch_attn_decode(const float* q, const uint16_t* kc, const uint16_t* vc, float* out, float* scores, int n_kv,
                        int max_ctx, float scale, cudaStream_t s) {
    k_attn_decode<<<24, 256, 0, s>>>(q, (const __half*)kc, (const __half*)vc, out, scores, n_kv, max_ctx, scale);
}

__global__ void k_gate_sigmoid(const float* att, const float* qfull, float* out) {
    const int h = blockIdx.x, d = threadIdx.x;
    const float g = qfull[h * 512 + 256 + d];
    out[h * 256 + d] = att[h * 256 + d] * (1.0f / (1.0f + expf(-g)));
}

void launch_gate_sigmoid(const float* att, const float* qfull, float* out, cudaStream_t s) {
    k_gate_sigmoid<<<24, 256, 0, s>>>(att, qfull, out);
}
