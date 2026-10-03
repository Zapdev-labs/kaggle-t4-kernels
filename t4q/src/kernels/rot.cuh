// t4q/src/kernels/rot.cuh -- milestone P round 3: randomized block-Hadamard rotation for the prefill GEMMs ("R512").
//
// y = W x = (W D H) (H D x) for H the normalized 512-point Walsh-Hadamard matrix (block diagonal over K, symmetric,
// H H = I) and D a fixed random +-1 diagonal (sgn(k), k = the GEMM's local K index). The same transform
// T(v) = H D v is applied to every activation row (quant_rot_kernel, before per-token int8 quantization) and to every
// weight row (w8r_convert_kernel, before per-row int8 requantization). Rotation spreads outlier channels over 512
// channels, so one int8 scale per token is about as accurate as per-64-group scales (t4q-p v19 / v20 study), and the
// GEMM main loop is plain int8 (gemm17 WSRC 0, GSH 0).
//
// Weight rows are rotated from the decode layout (P4 / P4M / K5, read in place) into a per-GEMM int8 scratch once per
// ubatch; invr[n] = 127 / max_k |T(w_n)[k]| is computed once per context (rot_invr_kernel).
// Work split: one warp per row; lanes 0..15 hold one 512-block (32 consecutive values each = one Q4 32-block), lanes
// 16..31 the next one. Within-lane butterflies cover strides 1..16, shfl_xor over lane bits 0..3 strides 32..256.
#pragma once
#include "gemm8.cuh"

namespace t4q {
namespace rot {

constexpr int RB = 512;

__host__ __device__ __forceinline__ float sgn(int k) { return (((unsigned)k * 2654435761u) & 0x80000000u) ? -1.f : 1.f; }

// one (row, 32-block) of the decode layout -> 32 floats in natural order (same codes as gemm16::convert32)
template <int FMT>
__host__ __device__ __forceinline__ void deq32f(const uint32_t q[4], float A, float B, uint32_t hb, float v[32]) {
    for (int w = 0; w < 4; ++w) {
        const uint32_t x = q[w];
        const uint32_t H = FMT == gemv::FAST_K5 ? (hb >> w) : 0u;
        const int c[8] = {
            (int)((x & 0xF) | ((H & 1) << 4)),                  (int)(((x >> 8) & 0xF) | (((H >> 8) & 1) << 4)),
            (int)(((x >> 16) & 0xF) | (((H >> 16) & 1) << 4)),  (int)(((x >> 24) & 0xF) | (((H >> 24) & 1) << 4)),
            (int)(((x >> 4) & 0xF) | (((H >> 4) & 1) << 4)),    (int)(((x >> 12) & 0xF) | (((H >> 12) & 1) << 4)),
            (int)(((x >> 20) & 0xF) | (((H >> 20) & 1) << 4)),  (int)(((x >> 28) & 0xF) | (((H >> 28) & 1) << 4))};
        for (int e = 0; e < 4; ++e) {
            v[4 * w + e] = FMT == gemv::FAST_P4 ? A * (float)(c[e] - 8) : A * (float)c[e] + B;
            v[16 + 4 * w + e] = FMT == gemv::FAST_P4 ? A * (float)(c[4 + e] - 8) : A * (float)c[4 + e] + B;
        }
    }
}

}  // namespace rot
}  // namespace t4q

#ifdef __CUDACC__
namespace t4q {
namespace rot {

// (A, B) of a block from the staged scale words (P4: d; P4M: d, m; K5: {sc, mn} bytes, {d, dmin})
template <int FMT>
__device__ __forceinline__ void block_ab(uint32_t s0, uint32_t s1, float& A, float& B) {
    if (FMT == gemv::FAST_P4) { A = gemm8::h2f(s0); B = 0.f; }
    else if (FMT == gemv::FAST_P4M) { A = gemm8::h2f(s0); B = gemm8::h2f(s1); }
    else { A = gemm8::h2f(s1) * (float)(s0 & 0xff); B = -gemm8::h2f(s1 >> 16) * (float)((s0 >> 8) & 0xff); }
}

// load (row R, 32-block kb) of the decode layout and dequantize
template <int FMT, int RPL>
__device__ __forceinline__ void load_deq(const gemm8::Args& a, int R, int kb, float v[32]) {
    const gemm8::WPos<RPL> p(R, kb, a.K >> 9, a.ntiles, a.cm);
    const int4 wq = gemv::ldg_nc_v4(a.codes + (p.tc * RPL + p.r) * 512 + p.lane * 16);
    uint32_t s0 = 0, s1 = 0, hb = 0;
    if (FMT == gemv::FAST_K5) {
        s0 = (uint32_t)__ldg((const unsigned short*)(a.sc + ((p.tc * 32 + p.lane) * RPL + p.r) * 2));
        s1 = __ldg((const unsigned int*)a.d + ((p.tc * 2 + p.h) * 2 + (p.j >> 3)) * RPL + p.r);
        hb = __ldg((const unsigned int*)(a.qh + (p.tc * RPL + p.r) * 128 + p.lane * 4));
    } else {
        s0 = (uint32_t)__ldg((const unsigned short*)a.d + (p.tc * 32 + p.lane) * RPL + p.r);
        if (FMT == gemv::FAST_P4M) s1 = (uint32_t)__ldg((const unsigned short*)a.sc + (p.tc * 32 + p.lane) * RPL + p.r);
    }
    float A, B;
    block_ab<FMT>(s0, s1, A, B);
    const uint32_t q[4] = {(uint32_t)wq.x, (uint32_t)wq.y, (uint32_t)wq.z, (uint32_t)wq.w};
    deq32f<FMT>(q, A, B, hb, v);
}

// T(v) on the lane's 32 values: signs (global index k0 + i), 5 in-lane levels, 4 cross-lane levels (lanes of one
// 16-lane half), normalization
__device__ __forceinline__ void fwht512(float v[32], int k0, int lane) {
#pragma unroll
    for (int i = 0; i < 32; ++i) v[i] *= sgn(k0 + i);
#pragma unroll
    for (int h = 1; h < 32; h <<= 1)
#pragma unroll
        for (int i = 0; i < 32; ++i)
            if (!(i & h)) { const float a = v[i], b = v[i + h]; v[i] = a + b; v[i + h] = a - b; }
#pragma unroll
    for (int m = 1; m < 16; m <<= 1)
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const float o = __shfl_xor_sync(0xffffffffu, v[i], m);
            v[i] = (lane & m) ? o - v[i] : v[i] + o;
        }
    const float nrm = 0.04419417382415922f;  // 1/sqrt(512)
#pragma unroll
    for (int i = 0; i < 32; ++i) v[i] *= nrm;
}

// invr[n] = 127 / max_k |T(w_n)[k]|; one warp per row (grid N/8 x 256)
template <int FMT, int RPL>
__global__ void __launch_bounds__(256) rot_invr_kernel(const gemm8::Args a, float* invr) {
    const int lane = threadIdx.x & 31, R = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (R >= a.N) return;
    const int nb = a.K / RB;
    float m = 0.f;
    for (int b0 = 0; b0 < nb; b0 += 2) {
        const int b = b0 + (lane >> 4), ch = lane & 15;
        float v[32];
        if (b < nb) load_deq<FMT, RPL>(a, R, b * 16 + ch, v);
        else
#pragma unroll
            for (int i = 0; i < 32; ++i) v[i] = 0.f;
        fwht512(v, b * RB + ch * 32, lane);
#pragma unroll
        for (int i = 0; i < 32; ++i) m = fmaxf(m, fabsf(v[i]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) invr[R] = m > 0.f ? 127.f / m : 1.f;
}

// w8[n][k] = rint(T(w_n)[k] * invr[n]); one warp per row
template <int FMT, int RPL>
__global__ void __launch_bounds__(256) w8r_convert_kernel(const gemm8::Args a, const float* __restrict__ invr,
                                                          int8_t* __restrict__ out) {
    const int lane = threadIdx.x & 31, R = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (R >= a.N) return;
    const int nb = a.K / RB;
    const float sc = invr[R];
    for (int b0 = 0; b0 < nb; b0 += 2) {
        const int b = b0 + (lane >> 4), ch = lane & 15;
        float v[32];
        if (b < nb) load_deq<FMT, RPL>(a, R, b * 16 + ch, v);
        else
#pragma unroll
            for (int i = 0; i < 32; ++i) v[i] = 0.f;
        fwht512(v, b * RB + ch * 32, lane);
        if (b < nb) {
            uint32_t pk[8];
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                uint32_t p = 0;
#pragma unroll
                for (int e = 0; e < 4; ++e) p |= ((uint32_t)__float2int_rn(v[4 * w + e] * sc) & 0xffu) << (8 * e);
                pk[w] = p;
            }
            int4* o = (int4*)(out + (size_t)R * a.K + b * RB + ch * 32);
            o[0] = make_int4(pk[0], pk[1], pk[2], pk[3]);
            o[1] = make_int4(pk[4], pk[5], pk[6], pk[7]);
        }
    }
}

// activations: x (rows of ldx, fp32) -> T(x) quantized with one scale per token: xq [Tp][K], dx[t] = amax / 127.
// One 256-thread block per token; K / 512 <= 17 blocks; rotated row staged in smem (dynamic, K floats).
template <class XT>
__global__ void __launch_bounds__(256) quant_rot_kernel(const XT* __restrict__ x, int ldx, int T, int K,
                                                        int8_t* __restrict__ xq, float* __restrict__ dx) {
    extern __shared__ float xs[];
    __shared__ float red[8];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int8_t* dst = xq + (size_t)t * K;
    if (t >= T) {
        for (int k = tid * 4; k < K; k += 1024) *(int*)(dst + k) = 0;
        if (tid == 0) dx[t] = 0.f;
        return;
    }
    const XT* src = x + (size_t)t * ldx;
    const int nb = K / RB;
    float m = 0.f;
    for (int b0 = warp * 2; b0 < nb; b0 += 16) {
        const int b = b0 + (lane >> 4), ch = lane & 15;
        float v[32];
        if (b < nb) {
#pragma unroll
            for (int i = 0; i < 32; ++i) v[i] = (float)src[b * RB + ch * 32 + i];
        } else {
#pragma unroll
            for (int i = 0; i < 32; ++i) v[i] = 0.f;
        }
        fwht512(v, b * RB + ch * 32, lane);
        if (b < nb)
#pragma unroll
            for (int i = 0; i < 32; ++i) { xs[b * RB + ch * 32 + i] = v[i]; m = fmaxf(m, fabsf(v[i])); }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    float am = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) am = fmaxf(am, red[w]);
    const float d = am / 127.f;
    for (int k = tid * 4; k < K; k += 1024) {
        uint32_t pk = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int q = am == 0.f ? 0 : __float2int_rn(xs[k + e] / d);
            pk |= (uint32_t)(q & 0xff) << (8 * e);
        }
        *(uint32_t*)(dst + k) = pk;
    }
    if (tid == 0) dx[t] = d;
}

template <class XT>
static inline void quant_rot(const XT* x, int ldx, int T, int Tp, int K, int8_t* xq, float* dx, cudaStream_t s) {
    quant_rot_kernel<XT><<<Tp, 256, K * 4, s>>>(x, ldx, T, K, xq, dx);
}

static inline cudaError_t invr(int fmt, int rpl, const gemm8::Args& a, float* out, cudaStream_t s) {
    const int grid = (a.N + 7) / 8;
#define T4Q_RI(F, R) \
    if (fmt == F && rpl == R) { rot_invr_kernel<F, R><<<grid, 256, 0, s>>>(a, out); return cudaGetLastError(); }
    T4Q_RI(gemv::FAST_P4, 4) T4Q_RI(gemv::FAST_P4, 2)
    T4Q_RI(gemv::FAST_P4M, 4) T4Q_RI(gemv::FAST_P4M, 2)
    T4Q_RI(gemv::FAST_K5, 4) T4Q_RI(gemv::FAST_K5, 2)
#undef T4Q_RI
    return cudaErrorInvalidValue;
}

static inline cudaError_t convert(int fmt, int rpl, const gemm8::Args& a, const float* invr, int8_t* out, cudaStream_t s) {
    const int grid = (a.N + 7) / 8;
#define T4Q_RC(F, R) \
    if (fmt == F && rpl == R) { w8r_convert_kernel<F, R><<<grid, 256, 0, s>>>(a, invr, out); return cudaGetLastError(); }
    T4Q_RC(gemv::FAST_P4, 4) T4Q_RC(gemv::FAST_P4, 2)
    T4Q_RC(gemv::FAST_P4M, 4) T4Q_RC(gemv::FAST_P4M, 2)
    T4Q_RC(gemv::FAST_K5, 4) T4Q_RC(gemv::FAST_K5, 2)
#undef T4Q_RC
    return cudaErrorInvalidValue;
}

}  // namespace rot
}  // namespace t4q
#endif
