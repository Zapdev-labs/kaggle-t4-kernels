// M1 reference GEMV: one warp per output row, each lane dequantizes 32-weight groups (deq32) and dots them with
// fp32 activations. Correctness first; the fast dp4a kernels live in gemv.cuh (M2).
#include "deq.cuh"
#include "kernels.h"

template <int FMT>
__global__ void __launch_bounds__(256) k_gemv(PackedW W, const float* __restrict__ x, float* __restrict__ y) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t row = (int64_t)blockIdx.x * 8 + warp;
    if (row >= W.rows) return;
    const int64_t ng = W.cols / 32;
    float acc = 0.f;
    for (int64_t g = lane; g < ng; g += 32) {
        float w[32];
        deq32<FMT>(W, row, g, w);
        const float4* xv = (const float4*)(x + g * 32);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float4 a = __ldg(xv + j);
            acc += w[4 * j] * a.x + w[4 * j + 1] * a.y + w[4 * j + 2] * a.z + w[4 * j + 3] * a.w;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = acc;
}

void launch_gemv(const PackedW& W, const float* x, float* y, cudaStream_t s) {
    const unsigned G = (unsigned)((W.rows + 7) / 8);
    switch (W.fmt) {
        case FMT_F32: k_gemv<FMT_F32><<<G, 256, 0, s>>>(W, x, y); break;
        case FMT_P4: k_gemv<FMT_P4><<<G, 256, 0, s>>>(W, x, y); break;
        case FMT_P4M: k_gemv<FMT_P4M><<<G, 256, 0, s>>>(W, x, y); break;
        case FMT_Q8: k_gemv<FMT_Q8><<<G, 256, 0, s>>>(W, x, y); break;
        case FMT_K5: k_gemv<FMT_K5><<<G, 256, 0, s>>>(W, x, y); break;
        case FMT_K6: k_gemv<FMT_K6><<<G, 256, 0, s>>>(W, x, y); break;
    }
}

// ------------------------------------------------------------------------------------------- q8_1 activations
__global__ void k_quantize_q8_1(const float* __restrict__ x, int K, int8_t* xq, float* xd, float* xs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // K % 32 == 0, blockDim multiple of 32
    if (i >= K) return;
    const float xi = x[i];
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = amax / 127.0f;
    xq[i] = amax == 0.0f ? 0 : (int8_t)roundf(xi / d);
    if ((i & 31) == 0) {
        xd[i >> 5] = __half2float(__float2half_rn(d));
        xs[i >> 5] = __half2float(__float2half_rn(sum));
    }
}

void launch_quantize_q8_1(const float* x, int K, int8_t* xq, float* xd, float* xs, cudaStream_t s) {
    k_quantize_q8_1<<<(K + 255) / 256, 256, 0, s>>>(x, K, xq, xd, xs);
}

template <int FMT>
__device__ __forceinline__ float dot_q8(const PackedW& W, int64_t row, int64_t g, const int8_t* x, float d8, float s8) {
    if constexpr (FMT == FMT_P4 || FMT == FMT_P4M) {
        const int64_t b = row * (W.cols / 32) + g;
        const uint4 c4 = *(const uint4*)(W.codes + b * 16);
        const uint8_t* c = (const uint8_t*)&c4;
        int s0 = 0, s1 = 0;
#pragma unroll
        for (int j = 0; j < 8; j++) s0 += (c[j] & 15) * x[j] + (c[j] >> 4) * x[j + 16];
#pragma unroll
        for (int j = 8; j < 16; j++) s1 += (c[j] & 15) * x[j] + (c[j] >> 4) * x[j + 16];
        if constexpr (FMT == FMT_P4) {
            const float d4 = h2f(W.d[b]);
            return d4 * (s0 * d8 - 4 * s8) + d4 * (s1 * d8 - 4 * s8);
        } else {
            const __half d4 = __ushort_as_half(W.d[b]), m4 = __ushort_as_half(W.m[b]);
            const float d4d8 = __half2float(__hmul(d4, __float2half_rn(d8)));
            const float m4s8 = __half2float(__hmul(m4, __float2half_rn(s8)));
            return (s0 * d4d8 + m4s8 / 2) + (s1 * d4d8 + m4s8 / 2);
        }
    } else if constexpr (FMT == FMT_Q8) {
        const int64_t b = row * (W.cols / 32) + g;
        const int8_t* c = (const int8_t*)W.codes + b * 32;
        int si = 0;
#pragma unroll
        for (int j = 0; j < 32; j++) si += c[j] * x[j];
        return h2f(W.d[b]) * (si * d8);
    } else if constexpr (FMT == FMT_K5) {
        const int64_t nb = W.cols / 256;
        const int64_t blk = row * nb + (g >> 3);
        const int s = (int)(g & 7);
        const uint8_t* meta = W.meta + blk * 16;
        const float d = h2f((uint16_t)(meta[0] | (meta[1] << 8)));
        const float dmin = h2f((uint16_t)(meta[2] | (meta[3] << 8)));
        int sc, mm;
        dev_scale_min_k4(s, meta + 4, sc, mm);
        const uint8_t* qs = W.codes + blk * 128 + 32 * (s >> 1);
        const uint8_t* qh = W.hi + blk * 32;
        const int hin = s & 1;
        float td = 0.f, tm = 0.f;
#pragma unroll
        for (int p = 0; p < 4; p++) {
            int dot = 0, sum = 0;
#pragma unroll
            for (int l = 8 * p; l < 8 * p + 8; l++) {
                const int lo = hin ? (qs[l] >> 4) : (qs[l] & 15);
                const int q = lo + (((qh[l] >> s) & 1) ? 16 : 0);
                dot += q * x[l];
                sum += x[l];
            }
            td += d8 * (dot * sc);
            tm += d8 * (sum * mm);
        }
        return d * td - dmin * tm;
    } else if constexpr (FMT == FMT_K6) {
        const int64_t nb = W.cols / 256;
        const int64_t blk = row * nb + (g >> 3);
        const int gg = (int)(g & 7);
        const int n = gg >> 2, qd = gg & 3;
        const uint8_t* ql = W.codes + blk * 128 + 64 * n + (qd & 1) * 32;
        const uint8_t* qh = W.hi + blk * 64 + 32 * n;
        const int8_t* sc = (const int8_t*)W.meta + blk * 16 + 8 * n + 2 * qd;
        float acc = 0.f;
#pragma unroll
        for (int p = 0; p < 8; p++) {
            int dot = 0;
#pragma unroll
            for (int l = 4 * p; l < 4 * p + 4; l++) {
                const int lo = (qd >= 2) ? (ql[l] >> 4) : (ql[l] & 15);
                const int hb = (qh[l] >> (2 * qd)) & 3;
                dot += ((lo | (hb << 4)) - 32) * x[l];
            }
            acc += d8 * (dot * sc[p >> 2]);
        }
        return h2f(W.d[blk]) * acc;
    } else {
        return 0.f;
    }
}

template <int FMT>
__global__ void __launch_bounds__(256) k_gemv_q8(PackedW W, const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                                 const float* __restrict__ xs, float* __restrict__ y) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t row = (int64_t)blockIdx.x * 8 + warp;
    if (row >= W.rows) return;
    const int64_t ng = W.cols / 32;
    float acc = 0.f;
    for (int64_t g = lane; g < ng; g += 32) {
        int8_t xv[32];
        *(int4*)xv = __ldg((const int4*)(xq + g * 32));
        *(int4*)(xv + 16) = __ldg((const int4*)(xq + g * 32 + 16));
        acc += dot_q8<FMT>(W, row, g, xv, xd[g], xs[g]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = acc;
}

void launch_gemv_q8(const PackedW& W, const int8_t* xq, const float* xd, const float* xs, float* y, cudaStream_t s) {
    const unsigned G = (unsigned)((W.rows + 7) / 8);
    switch (W.fmt) {
        case FMT_P4: k_gemv_q8<FMT_P4><<<G, 256, 0, s>>>(W, xq, xd, xs, y); break;
        case FMT_P4M: k_gemv_q8<FMT_P4M><<<G, 256, 0, s>>>(W, xq, xd, xs, y); break;
        case FMT_Q8: k_gemv_q8<FMT_Q8><<<G, 256, 0, s>>>(W, xq, xd, xs, y); break;
        case FMT_K5: k_gemv_q8<FMT_K5><<<G, 256, 0, s>>>(W, xq, xd, xs, y); break;
        case FMT_K6: k_gemv_q8<FMT_K6><<<G, 256, 0, s>>>(W, xq, xd, xs, y); break;
        default: break;
    }
}
