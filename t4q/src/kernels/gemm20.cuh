// t4q/src/kernels/gemm20.cuh -- prefill GEMM v4 (milestone P round 4): W4A8 on the INT4 tensor cores.
//
// Why: prefill is energy-bound at the 70 W cap (round 1-3). CUTLASS's int4 GEMM sustains 122-126 TOPS at 55-59 W
// (not capped) where int8 sustains 45-55 TOPS at 70 W, so an int4 MAC costs about 1/3 of an int8 MAC. This kernel
// spends two int4 MACs per W4A8 MAC and keeps the weights exactly as Q4_0 stores them:
//
//   * weights: the decode layout's Q4_0 codes, read in place (no requantization, no second copy). A 16-B code group is
//     one k32 row of the mma B operand; c ^ 8 per nibble turns the unsigned code into the signed value c - 8 (s4).
//     Byte i holds elements i (low nibble) and i + 16 (high nibble); the activations use the same nibble order, so the
//     block dot product is unchanged.
//   * activations: int8 with one scale per 64 (GA64, the production format), split into two nibble planes
//     x = 16 hi + lo (hi s4 in [-8, 7], lo u4 in [0, 15]), stored per token per 32-block as [lo 16 B][hi 16 B]
//     (d4_from_q8 builds it from the q8 GA64 layout; producers can write it directly).
//   * math per (token, row) and 64-group (two Q4 blocks b0, b1):
//       H_b = mma.s4.s4(hi, w_b, 0), L_b = mma.u4.s4(lo, w_b, MAGIC) -> P_b' = 16 H_b + L_b is the float MAGIC + P_b;
//       v = fma(d0, P_0', cm); v = fma(d1, P_1', v)  with cm = -(d0 + d1) MAGIC (exact for fp16 d);
//       F = fma(a_g, v, F).
//     Per output per 64 k: 4 int4 mmas (shared by 2 outputs each), 2 LEA + 3 FFMA.
//
// Tiling: CUTLASS orientation as gemm17: A (mma m) = 128 tokens, B (mma n) = BR weight rows (256 or 128), 8 warps
// 2 (tokens) x 4 (rows), warp tile 64 x BR/4, k stage 64 = two Q4 blocks, register-staged double buffer, one barrier
// per stage. Smem per stage: X [128][64 B] (units lo0 hi0 lo1 hi1, swizzled), W [BR][32 B], WS [BR] float4
// {d0, d1, cm, 0}, XS [128] float.
// FOLD 0 is a timing probe (all mmas chained into one int accumulator, no float math, wrong numbers).
#pragma once
#include "gemm8.cuh"

#ifdef __CUDACC__
namespace t4q {
namespace g20 {

struct Args {
    gemm8::Args q;              // decode-layout weights (P4): q.codes / q.d, N, K, ntiles, cm
    const uint8_t* xd = nullptr;  // [Tp][K/32][32 B] activation digit planes
    const float* dx = nullptr;    // [K/64][Tp] activation scale per 64-group (0 for padding tokens)
    float* y = nullptr;           // y[t * ldy + n]
    __half* yh = nullptr;         // fp16 output instead (no accumulate)
    int ldy = 0, N = 0, K = 0, T = 0, Tp = 0, accumulate = 0;
    int nvalid = 1 << 30;
};

constexpr int NT = 256;
template <int BR>
struct Cfg {
    static constexpr int O_X = 0;                 // [128][64 B]
    static constexpr int O_W = O_X + 128 * 64;    // [BR][32 B]
    static constexpr int O_WS = O_W + BR * 32;    // [BR] float4
    static constexpr int O_XS = O_WS + BR * 16;   // [128] float
    static constexpr int STAGE = O_XS + 128 * 4;
    static constexpr int SMEM = 2 * STAGE;
    static constexpr int WN = BR / 4;             // rows per warp
    static constexpr int NG = WN / 8;             // n tiles per warp
};

__device__ __forceinline__ void mma_ss(int& d0, int& d1, uint32_t a, uint32_t b, int c0, int c1) {
    asm("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1}, {%2}, {%3}, {%4,%5};"
        : "=r"(d0), "=r"(d1) : "r"(a), "r"(b), "r"(c0), "r"(c1));
}
__device__ __forceinline__ void mma_us(int& d0, int& d1, uint32_t a, uint32_t b, int c0, int c1) {
    asm("mma.sync.aligned.m8n8k32.row.col.s32.u4.s4.s32 {%0,%1}, {%2}, {%3}, {%4,%5};"
        : "=r"(d0), "=r"(d1) : "r"(a), "r"(b), "r"(c0), "r"(c1));
}
__device__ __forceinline__ int4 ldg_l2(const void* p) {
    int4 v;
    asm volatile("ld.global.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
// 32-B weight rows: 16-B unit swizzle so 8 consecutive rows hit distinct banks
__device__ __forceinline__ int wswz(int row, int u) { return u ^ ((row >> 2) & 1); }

template <int FMT, int RPL, int BR, int FOLD, int OUT>
__global__ void __launch_bounds__(NT, 1) gemm20_kernel(const Args a) {
    static_assert(FMT == gemv::FAST_P4, "gemm20: Q4_0 weights only");
    using C = Cfg<BR>;
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp & 1, wn = warp >> 1;
    const int tok0 = blockIdx.x * 128, row0 = blockIdx.y * BR;
    const int K = a.K, nst = K >> 6;
    const int t4 = lane & 3;
    // staging: X row tid/2 half tid&1 (32 B); W row tid (BR 256) or tid/2 half tid&1 (BR 128); XS tid < 128
    const uint8_t* xp = a.xd + (size_t)(tok0 + (tid >> 1)) * K + (tid & 1) * 32;
    const int wrow = BR == 256 ? tid : (tid >> 1);
    const int wh = BR == 256 ? 0 : (tid & 1);  // BR 128: which 16-B block of the stage
    const gemm8::WPtr<FMT, RPL> P(a.q, row0 + wrow, wh);
    const float* dxp = a.dx + tok0 + (tid & 127);

    float F[8][C::NG][2];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int g = 0; g < C::NG; ++g) F[i][g][0] = F[i][g][1] = 0.f;

    int4 sx[2], sw[2];
    uint32_t sd0 = 0, sd1 = 0;
    float sxs = 0.f;
    auto load = [&](int st) {
        sx[0] = ldg_l2(xp + st * 64);
        sx[1] = ldg_l2(xp + st * 64 + 16);
        const int c = (st * 2) >> 4, jj = (st * 2) & 15;
        const uint8_t* cp = P.code + c * P.cs_code + jj * 16;
        const uint8_t* dp = P.sa + c * P.cs_s + jj * RPL * 2;
        sw[0] = gemv::ldg_nc_v4(cp);
        sd0 = (uint32_t)__ldg((const unsigned short*)dp);
        if (BR == 256) {
            sw[1] = gemv::ldg_nc_v4(cp + 16);
            sd1 = (uint32_t)__ldg((const unsigned short*)(dp + RPL * 2));
        }
        if (tid < 128) sxs = __ldg(dxp + (size_t)st * a.Tp);
    };
    auto store = [&](int b) {
        unsigned char* B = smem + b * C::STAGE;
        const int xr = tid >> 1, xu = (tid & 1) * 2;
        *(int4*)(B + C::O_X + xr * 64 + gemm8::swz(xr, xu) * 16) = sx[0];
        *(int4*)(B + C::O_X + xr * 64 + gemm8::swz(xr, xu + 1) * 16) = sx[1];
        auto flip = [](int4 v) {
            return make_int4(v.x ^ 0x88888888, v.y ^ 0x88888888, v.z ^ 0x88888888, v.w ^ 0x88888888);
        };
        if (BR == 256) {
            *(int4*)(B + C::O_W + wrow * 32 + wswz(wrow, 0) * 16) = flip(sw[0]);
            *(int4*)(B + C::O_W + wrow * 32 + wswz(wrow, 1) * 16) = flip(sw[1]);
            const float d0 = gemm8::h2f(sd0), d1 = gemm8::h2f(sd1);
            *(float4*)(B + C::O_WS + wrow * 16) = make_float4(d0, d1, -(d0 + d1) * gemm8::MAGIC_F, 0.f);
        } else {
            *(int4*)(B + C::O_W + wrow * 32 + wswz(wrow, wh) * 16) = flip(sw[0]);
            // the pair partner (tid ^ 1) holds the other block's d
            const float dm = gemm8::h2f(sd0), dp = __shfl_xor_sync(0xffffffffu, dm, 1);
            if (wh == 0) *(float4*)(B + C::O_WS + wrow * 16) = make_float4(dm, dp, -(dm + dp) * gemm8::MAGIC_F, 0.f);
        }
        if (tid < 128) *(float*)(B + C::O_XS + tid * 4) = sxs;
    };
    int arow[2];
#pragma unroll
    for (int q = 0; q < 2; ++q) arow[q] = wm * 64 + (q * 4 + (lane >> 3)) * 8 + (lane & 7);
    // B rows for ldmatrix: n tile j of the warp = rows wn*WN + 8 j + (lane & 7)
    const int brow_l = wn * C::WN + (lane & 7);

    load(0);
    store(0);
    __syncthreads();
    int buf = 0;
    for (int s = 0; s < nst; ++s) {
        if (s + 1 < nst) load(s + 1);
        const unsigned char* Bp = smem + buf * C::STAGE;
        // fragments: A[dg][bb][i] (digit, block, m tile), B[bb][g]
        uint32_t fa[2][2][8], fb[2][C::NG];
#pragma unroll
        for (int bb = 0; bb < 2; ++bb)
#pragma unroll
            for (int dg = 0; dg < 2; ++dg)
#pragma unroll
                for (int q = 0; q < 2; ++q)
                    gemm8::ldsm_x4(fa[dg][bb][q * 4], fa[dg][bb][q * 4 + 1], fa[dg][bb][q * 4 + 2], fa[dg][bb][q * 4 + 3],
                                   Bp + C::O_X + arow[q] * 64 + gemm8::swz(arow[q], bb * 2 + dg) * 16);
#pragma unroll
        for (int bb = 0; bb < 2; ++bb)
#pragma unroll
            for (int q = 0; q < C::NG / 4; ++q) {
                const int r = brow_l + (q * 4 + (lane >> 3)) * 8;
                gemm8::ldsm_x4(fb[bb][q * 4], fb[bb][q * 4 + 1], fb[bb][q * 4 + 2], fb[bb][q * 4 + 3],
                               Bp + C::O_W + r * 32 + wswz(r, bb) * 16);
            }
        float xa[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) xa[i] = *(const float*)(Bp + C::O_XS + (wm * 64 + i * 8 + (lane >> 2)) * 4);
#pragma unroll
        for (int g = 0; g < C::NG; ++g) {
            float4 ws[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) ws[e] = *(const float4*)(Bp + C::O_WS + (wn * C::WN + 8 * g + 2 * t4 + e) * 16);
#pragma unroll
            for (int ii = 0; ii < 8; ++ii) {
                const int i = (g & 1) ? 7 - ii : ii;
                if (FOLD == 0) {
                    int c0 = __float_as_int(F[i][g][0]), c1 = __float_as_int(F[i][g][1]);
                    mma_ss(c0, c1, fa[1][0][i], fb[0][g], c0, c1);
                    mma_us(c0, c1, fa[0][0][i], fb[0][g], c0, c1);
                    mma_ss(c0, c1, fa[1][1][i], fb[1][g], c0, c1);
                    mma_us(c0, c1, fa[0][1][i], fb[1][g], c0, c1);
                    F[i][g][0] = __int_as_float(c0); F[i][g][1] = __int_as_float(c1);
                } else {
                    int h0[2], l0[2], h1[2], l1[2];
                    mma_ss(h0[0], h0[1], fa[1][0][i], fb[0][g], 0, 0);
                    mma_us(l0[0], l0[1], fa[0][0][i], fb[0][g], gemm8::MAGIC_I, gemm8::MAGIC_I);
                    mma_ss(h1[0], h1[1], fa[1][1][i], fb[1][g], 0, 0);
                    mma_us(l1[0], l1[1], fa[0][1][i], fb[1][g], gemm8::MAGIC_I, gemm8::MAGIC_I);
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const float p0 = __int_as_float((h0[e] << 4) + l0[e]);
                        const float p1 = __int_as_float((h1[e] << 4) + l1[e]);
                        float v = fmaf(ws[e].x, p0, ws[e].z);
                        v = fmaf(ws[e].y, p1, v);
                        F[i][g][e] = fmaf(xa[i], v, F[i][g][e]);
                    }
                }
            }
        }
        if (s + 1 < nst) store(buf ^ 1);
        __syncthreads();
        buf ^= 1;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int tok = tok0 + wm * 64 + i * 8 + (lane >> 2);
        if (tok >= a.T) continue;
#pragma unroll
        for (int g = 0; g < C::NG; ++g) {
            const int row = row0 + wn * C::WN + 8 * g + 2 * t4;
            if (row >= a.nvalid) continue;
            const float v0 = F[i][g][0], v1 = F[i][g][1];
            if (OUT == 1) {
                *(__half2*)(a.yh + (size_t)tok * a.ldy + row) = __floats2half2_rn(v0, v1);
            } else {
                float2* p = (float2*)(a.y + (size_t)tok * a.ldy + row);
                if (a.accumulate) { const float2 o = *p; *p = make_float2(o.x + v0, o.y + v1); }
                else *p = make_float2(v0, v1);
            }
        }
    }
}

template <int FMT, int RPL, int BR, int FOLD, int OUT>
static cudaError_t launch_t(const Args& a, cudaStream_t s) {
    auto k = gemm20_kernel<FMT, RPL, BR, FOLD, OUT>;
    static int attr_dev_mask = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    if (!(attr_dev_mask & (1 << dev))) {
        cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<BR>::SMEM);
        if (e != cudaSuccess) return e;
        attr_dev_mask |= 1 << dev;
    }
    if (a.N % BR || a.K % 64 || a.Tp % 128) return cudaErrorInvalidValue;
    dim3 grid(a.Tp / 128, a.N / BR);
    k<<<grid, NT, Cfg<BR>::SMEM, s>>>(a);
    return cudaGetLastError();
}

// GA64 q8 ([Tp][K] int8, natural order) -> digit planes [Tp][K/32][lo 16 B | hi 16 B]; one thread per (token, block)
__global__ void d4_from_q8_kernel(const int8_t* __restrict__ xq, int K, int n, uint8_t* __restrict__ xd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int4* src = (const int4*)(xq + (size_t)i * 32);
    const int4 a = src[0], b = src[1];
    const uint32_t A[4] = {(uint32_t)a.x, (uint32_t)a.y, (uint32_t)a.z, (uint32_t)a.w};
    const uint32_t B[4] = {(uint32_t)b.x, (uint32_t)b.y, (uint32_t)b.z, (uint32_t)b.w};
    uint32_t lo[4], hi[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        // byte j of the output word: element 4w + j (low nibble) and 16 + 4w + j (high nibble)
        const uint32_t x = A[w], y = B[w];
        lo[w] = (x & 0x0F0F0F0Fu) | ((y & 0x0F0F0F0Fu) << 4);
        hi[w] = ((x >> 4) & 0x0F0F0F0Fu) | (y & 0xF0F0F0F0u);
    }
    int4* dst = (int4*)(xd + (size_t)i * 32);
    dst[0] = make_int4(lo[0], lo[1], lo[2], lo[3]);
    dst[1] = make_int4(hi[0], hi[1], hi[2], hi[3]);
}
static inline cudaError_t d4_from_q8(const int8_t* xq, int Tp, int K, uint8_t* xd, cudaStream_t s) {
    const int n = Tp * (K / 32);
    d4_from_q8_kernel<<<(n + 255) / 256, 256, 0, s>>>(xq, K, n, xd);
    return cudaGetLastError();
}

}  // namespace g20
}  // namespace t4q
#endif
