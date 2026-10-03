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

// D = per-32-chunk random sign (hash of k / 32) times a fixed 32-periodic element pattern (ESGN bit k % 32), so the
// fp16x2 weight converter can fold the chunk sign into its block scale and the element signs into constants.
constexpr uint32_t ESGN = 0x6B8B4567u;
__host__ __device__ __forceinline__ float csgn(int c) { return (((unsigned)c * 2654435761u) & 0x80000000u) ? -1.f : 1.f; }
__host__ __device__ __forceinline__ float sgn(int k) { return csgn(k >> 5) * (((ESGN >> (k & 31)) & 1u) ? -1.f : 1.f); }
// weight rows are scaled to +-126 (not 127) so fp16 rounding in the converter can never reach 128
constexpr float QMAX = 126.f;

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
    if (lane == 0) invr[R] = m > 0.f ? QMAX / m : 1.f;
}

// fp16x2 converter (P4 / P4M): lane = one 32-chunk as 16 half2 registers (element r, element r + 16), the layout of a
// Q4 code byte (low nibble r, high nibble r + 16). Normalization, invr and the chunk sign are folded into the block
// scale; element signs are compile-time constants; in-register levels cover element bits 0..4, shfl_xor bits 5..8.
// About 350 instructions per lane per 32 weights (the fp32 converter: ~1000), so the conversion is memory-bound.
__device__ __forceinline__ __half2 hsel(uint32_t v) { return *reinterpret_cast<__half2*>(&v); }
__device__ __forceinline__ uint32_t hbits(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
template <int FMT, int RPL>
__global__ void __launch_bounds__(256) w8r_convert_h2_kernel(const gemm8::Args a, const float* __restrict__ invr,
                                                             int8_t* __restrict__ out) {
    const int lane = threadIdx.x & 31, R = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (R >= a.N) return;
    const int nb = a.K / RB;
    const float scR = invr[R] * 0.04419417382415922f;
    const __half2 one_m = __floats2half2_rn(1.f, -1.f);
    for (int b0 = 0; b0 < nb; b0 += 2) {
        const int b = b0 + (lane >> 4), ch = lane & 15;
        const bool ok = b < nb;
        uint32_t q[4] = {0x88888888u, 0x88888888u, 0x88888888u, 0x88888888u};
        uint32_t s0 = 0, s1 = 0;
        if (ok) {
            const gemm8::WPos<RPL> p(R, b * 16 + ch, a.K >> 9, a.ntiles, a.cm);
            const int4 wq = gemv::ldg_nc_v4(a.codes + (p.tc * RPL + p.r) * 512 + p.lane * 16);
            q[0] = wq.x; q[1] = wq.y; q[2] = wq.z; q[3] = wq.w;
            s0 = (uint32_t)__ldg((const unsigned short*)a.d + (p.tc * 32 + p.lane) * RPL + p.r);
            if (FMT == gemv::FAST_P4M) s1 = (uint32_t)__ldg((const unsigned short*)a.sc + (p.tc * 32 + p.lane) * RPL + p.r);
        }
        const float A = (ok ? gemm8::h2f(s0) : 0.f) * scR * csgn(b * 16 + ch);
        const float Bm = FMT == gemv::FAST_P4M && ok ? gemm8::h2f(s1) * scR * csgn(b * 16 + ch) : 0.f;
        // sign combos (element r, element r + 16): ++, +-, -+, --
        __half2 sc[4], bi[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float sl = (c & 2) ? -1.f : 1.f, sh = (c & 1) ? -1.f : 1.f;
            sc[c] = __floats2half2_rn(A * sl, A * 0.0625f * sh);
            bi[c] = __floats2half2_rn(Bm * sl, Bm * sh);
        }
        __half2 v[16];
        const __half2 kofs = FMT == gemv::FAST_P4 ? __floats2half2_rn(1032.f, 1152.f) : __floats2half2_rn(1024.f, 1024.f);
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            const uint32_t sel = (r & 3) | 0x40u | ((r & 3) << 8) | 0x4000u;
            const uint32_t t = (__byte_perm(q[r >> 2], 0u, sel) & 0x00F0000Fu) | 0x64006400u;
            const int c = (((ESGN >> r) & 1u) ? 2 : 0) | (((ESGN >> (r + 16)) & 1u) ? 1 : 0);
            const __half2 d = __hsub2(hsel(t), kofs);
            v[r] = FMT == gemv::FAST_P4 ? __hmul2(d, sc[c]) : __hfma2(d, sc[c], bi[c]);
        }
        // element bits 0..3: between registers
#pragma unroll
        for (int h = 1; h < 16; h <<= 1)
#pragma unroll
            for (int r = 0; r < 16; ++r)
                if (!(r & h)) { const __half2 x = v[r], y = v[r + h]; v[r] = __hadd2(x, y); v[r + h] = __hsub2(x, y); }
        // element bit 4: inside each half2: (a, b) -> (a + b, a - b)
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            const uint32_t sw = __byte_perm(hbits(v[r]), 0u, 0x1032u);
            v[r] = __hfma2(v[r], one_m, hsel(sw));
        }
        // element bits 5..8: lanes
#pragma unroll
        for (int m = 1; m < 16; m <<= 1) {
            const __half2 sg = (lane & m) ? __floats2half2_rn(-1.f, -1.f) : __floats2half2_rn(1.f, 1.f);
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                const uint32_t o = __shfl_xor_sync(0xffffffffu, hbits(v[r]), m);
                v[r] = __hfma2(v[r], sg, hsel(o));
            }
        }
        // round: 1536 + v has the int8 in its low byte (|v| <= 126.x)
        const __half2 k1536 = __floats2half2_rn(1536.f, 1536.f);
        uint32_t u[16];
#pragma unroll
        for (int r = 0; r < 16; ++r) u[r] = hbits(__hadd2(v[r], k1536));
        uint32_t w[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            w[j] = __byte_perm(__byte_perm(u[4 * j], u[4 * j + 1], 0x0040u), __byte_perm(u[4 * j + 2], u[4 * j + 3], 0x0040u), 0x5410u);
            w[4 + j] = __byte_perm(__byte_perm(u[4 * j], u[4 * j + 1], 0x0062u), __byte_perm(u[4 * j + 2], u[4 * j + 3], 0x0062u), 0x5410u);
        }
        if (ok) {
            int4* o = (int4*)(out + (size_t)R * a.K + b * RB + ch * 32);
            o[0] = make_int4(w[0], w[1], w[2], w[3]);
            o[1] = make_int4(w[4], w[5], w[6], w[7]);
        }
    }
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

// Activation-side layout: lane l (of a 16-lane half) holds elements l + 16 i (i = 0..31) of a 512-block, so global
// and smem accesses are contiguous per half-warp; cross-lane levels cover strides 1..8, in-register levels 16..256.
// Same transform T = H D as fwht512 (only the data layout differs).
__device__ __forceinline__ void fwht512s(float v[32], int k0, int l, int lane) {
#pragma unroll
    for (int i = 0; i < 32; ++i) v[i] *= sgn(k0 + l + 16 * i);
#pragma unroll
    for (int m = 1; m < 16; m <<= 1)
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const float o = __shfl_xor_sync(0xffffffffu, v[i], m);
            v[i] = (lane & m) ? o - v[i] : v[i] + o;
        }
#pragma unroll
    for (int h = 1; h < 32; h <<= 1)
#pragma unroll
        for (int i = 0; i < 32; ++i)
            if (!(i & h)) { const float a = v[i], b = v[i + h]; v[i] = a + b; v[i + h] = a - b; }
    const float nrm = 0.04419417382415922f;
#pragma unroll
    for (int i = 0; i < 32; ++i) v[i] *= nrm;
}

// Rotate one token row (source: global or smem, element k at src[k]) into xs (smem, may alias src), then quantize
// with one scale per token into dst; returns nothing, writes dx (thread 0). 256 threads; K % 512 == 0, K <= 8704.
template <class SRC>
__device__ __forceinline__ void rot_quant_row(const SRC* src, float* xs, int K, int8_t* dst, float* dxp, float* red) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, l = lane & 15;
    const int nb = K / RB;
    float m = 0.f;
    for (int b0 = warp * 2; b0 < nb; b0 += 16) {
        const int b = b0 + (lane >> 4);
        float v[32];
        if (b < nb) {
#pragma unroll
            for (int i = 0; i < 32; ++i) v[i] = (float)src[b * RB + l + 16 * i];
        } else {
#pragma unroll
            for (int i = 0; i < 32; ++i) v[i] = 0.f;
        }
        fwht512s(v, b * RB, l, lane);
        if (b < nb)
#pragma unroll
            for (int i = 0; i < 32; ++i) { xs[b * RB + l + 16 * i] = v[i]; m = fmaxf(m, fabsf(v[i])); }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    __syncthreads();  // red reuse + all xs writes visible
    if (lane == 0) red[warp] = m;
    __syncthreads();
    float am = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) am = fmaxf(am, red[w]);
    const float inv = am > 0.f ? 127.f / am : 0.f;
    for (int k = tid * 4; k < K; k += 1024) {
        const float4 f = *(const float4*)(xs + k);
        const uint32_t pk = ((uint32_t)__float2int_rn(f.x * inv) & 0xffu) | (((uint32_t)__float2int_rn(f.y * inv) & 0xffu) << 8) |
                            (((uint32_t)__float2int_rn(f.z * inv) & 0xffu) << 16) | (((uint32_t)__float2int_rn(f.w * inv) & 0xffu) << 24);
        *(uint32_t*)(dst + k) = pk;
    }
    if (tid == 0) *dxp = am / 127.f;
}

// activations: x (rows of ldx, fp32 or fp16) -> T(x) quantized with one scale per token: xq [Tp][K], dx[t] = amax/127.
// One 256-thread block per token; dynamic smem K floats.
template <class XT>
__global__ void __launch_bounds__(256) quant_rot_kernel(const XT* __restrict__ x, int ldx, int T, int K,
                                                        int8_t* __restrict__ xq, float* __restrict__ dx) {
    extern __shared__ __align__(16) float xs[];
    __shared__ float red[8];
    const int t = blockIdx.x, tid = threadIdx.x;
    int8_t* dst = xq + (size_t)t * K;
    if (t >= T) {
        for (int k = tid * 4; k < K; k += 1024) *(int*)(dst + k) = 0;
        if (tid == 0) dx[t] = 0.f;
        return;
    }
    rot_quant_row(x + (size_t)t * ldx, xs, K, dst, dx + t, red);
}

// fp32 weight rows w[N][K] -> T(w) int8 rows out[Npad][K] with invr (rows N..Npad-1 zero, invr 1); one warp per row,
// two passes (amax, then quantize). Used for the DeltaNet alpha/beta rows (once per context).
__global__ void __launch_bounds__(256) w8r_f32_kernel(const float* __restrict__ w, int N, int Npad, int K,
                                                      float* invr, int8_t* __restrict__ out) {
    const int lane = threadIdx.x & 31, R = blockIdx.x * 8 + (threadIdx.x >> 5), l = lane & 15;
    if (R >= Npad) return;
    const int nb = K / RB;
    if (R >= N) {
        for (int k = lane * 16; k < K; k += 512) *(int4*)(out + (size_t)R * K + k) = make_int4(0, 0, 0, 0);
        if (lane == 0) invr[R] = 1.f;
        return;
    }
    float m = 0.f;
    for (int pass = 0; pass < 2; ++pass) {
        const float sc = pass ? QMAX / m : 0.f;
        for (int b0 = 0; b0 < nb; b0 += 2) {
            const int b = b0 + (lane >> 4);
            float v[32];
#pragma unroll
            for (int i = 0; i < 32; ++i) v[i] = b < nb ? w[(size_t)R * K + b * RB + l + 16 * i] : 0.f;
            fwht512s(v, b * RB, l, lane);
            if (!pass) {
#pragma unroll
                for (int i = 0; i < 32; ++i) m = fmaxf(m, fabsf(v[i]));
            } else if (b < nb) {
#pragma unroll
                for (int i = 0; i < 32; ++i) out[(size_t)R * K + b * RB + l + 16 * i] = (int8_t)__float2int_rn(v[i] * sc);
            }
        }
        if (!pass) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
            m = m > 0.f ? m : 1.f;
        }
    }
    if (lane == 0) invr[R] = QMAX / m;
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

static inline cudaError_t convert(int fmt, int rpl, const gemm8::Args& a, const float* invr, int8_t* out, cudaStream_t s,
                                  bool fp16 = true) {
    const int grid = (a.N + 7) / 8;
#define T4Q_RC(F, R) \
    if (fmt == F && rpl == R) { w8r_convert_kernel<F, R><<<grid, 256, 0, s>>>(a, invr, out); return cudaGetLastError(); }
#define T4Q_RH(F, R) \
    if (fmt == F && rpl == R) { w8r_convert_h2_kernel<F, R><<<grid, 256, 0, s>>>(a, invr, out); return cudaGetLastError(); }
    if (fp16) { T4Q_RH(gemv::FAST_P4, 4) T4Q_RH(gemv::FAST_P4, 2) T4Q_RH(gemv::FAST_P4M, 4) T4Q_RH(gemv::FAST_P4M, 2) }
    T4Q_RC(gemv::FAST_P4, 4) T4Q_RC(gemv::FAST_P4, 2)
    T4Q_RC(gemv::FAST_P4M, 4) T4Q_RC(gemv::FAST_P4M, 2)
    T4Q_RC(gemv::FAST_K5, 4) T4Q_RC(gemv::FAST_K5, 2)
#undef T4Q_RC
#undef T4Q_RH
    return cudaErrorInvalidValue;
}

}  // namespace rot
}  // namespace t4q
#endif
