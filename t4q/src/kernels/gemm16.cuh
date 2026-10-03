// t4q/src/kernels/gemm16.cuh -- prefill GEMM v3 (milestone P round 3): plain int8 x int8 tensor-core GEMM with
// per-ROW weight scales and per-TOKEN activation scales, so the main loop is nothing but ldmatrix + mma (int32
// accumulation in the mma C operand) and the only float math is the epilogue y = acc * dx[t] / invs[n].
//
// Why: round 2 measured CUTLASS's plain int8 GEMM at ~1400 ops/clk/SM under the 70 W cap against ~775 for gemm9
// (per-64 activation scales, in-kernel Q4 -> int8 conversion right before the barrier) and ~1010 for gemm13 (per-token,
// same conversion). This kernel removes both differences:
//   * WSRC 0: weights come pre-converted as int8 rows w8[N][K] (w8_convert_kernel, same numerics as gemm8/9: one scale
//     per row, invs = 127 / max |w[n]|), loaded and stored like the activations (2 x 16 B per thread per stage);
//   * WSRC 1: weights read in place from the decode layout (P4/P4M/K5) and converted to int8 in registers two k16
//     steps after the loads were issued (off the barrier's critical path), then stored like WSRC 0.
// Activations: int8 with one scale per token (dx[t] = amax / 127; outlier handling is the producer's business).
//
// Tiling (same as gemm13 / CUTLASS 128x256x64): block 128 weight rows x 256 tokens, 8 warps as 2 (rows) x 4 (tokens),
// warp tile 64 x 64 (8 x 8 m8n8k16 tiles), k stage 64, register-staged double buffer (sm_75 has no cp.async), fragment
// double buffering per k16 step, the next stage's smem stores + the one barrier per stage sit before the last k16
// step's mmas, mmas in serpentine order (A-operand reuse). Smem: [row|token][64 B], 16-B unit swizzle, 48 KB.
#pragma once
#include "gemm8.cuh"

#ifdef __CUDACC__
namespace t4q {
namespace g16 {

constexpr int BM = 128, BN = 256, NT = 256;
constexpr int O_W = 0, O_X = BM * 64, STAGE_BYTES = (BM + BN) * 64, SMEM = 2 * STAGE_BYTES;

struct Args {
    const int8_t* w8 = nullptr;  // WSRC 0: [N][K]
    gemm8::Args q;               // WSRC 1: decode-layout weights (q.codes / d / sc / qh, N, K, ntiles, cm)
    const float* invs = nullptr; // [N]
    const int8_t* xq = nullptr;  // [Tp][K]
    const float* dx = nullptr;   // [Tp] per-token scale (0 for padding tokens)
    float* y = nullptr;          // y[t * ldy + n]
    __half* yh = nullptr;        // fp16 output instead of y (no accumulate)
    int ldy = 0, N = 0, K = 0, T = 0, Tp = 0, accumulate = 0;
    int nvalid = 1 << 30;        // gemm17 OUT 0/1: rows >= nvalid are not written (padded weight rows)
    int kstride = 0;             // gemm17 WSRC 0: row stride of w8 and xq (0 = K; K-slices of a wider matrix)
    const float* dsr = nullptr;  // gemm17 GSH 2: per-512-block rescale ratios [K/512][Tp] (kernels/rot.cuh gb_ratios)
    const int8_t* dsh = nullptr; // gemm17 GSH: shift deltas [K/64][Tp] (permuted token order)
};

using gemm8::ldsm_x4;
using gemm8::swz;
using gemm8::as_h2;
using gemm8::h2_bits;
using gemm8::h2f;

// Q4 -> int8 for one (row, 32-block): 16 B of codes + scales -> lo (elements 0..15), hi (16..31); same arithmetic as
// gemm8::stage_store
template <int FMT>
__device__ __forceinline__ void convert32(const int4 wq, uint32_t s0, uint32_t s1, uint32_t hb, float invs,
                                          uint32_t lo[4], uint32_t hi[4]) {
    const uint32_t q[4] = {(uint32_t)wq.x, (uint32_t)wq.y, (uint32_t)wq.z, (uint32_t)wq.w};
    const __half2 k1536 = __float2half2_rn(1536.f);
    if (FMT == gemv::FAST_P4) {
        const float av = h2f(s0) * invs;
        const __half2 r = __float2half2_rn(av), r16 = __float2half2_rn(av * 0.0625f);
        const __half2 k1032 = __float2half2_rn(1032.f), k1152 = __float2half2_rn(1152.f);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t x = q[w], x8 = x >> 8;
            const __half2 v0 = as_h2((x & 0x000F000Fu) | 0x64006400u), v1 = as_h2((x8 & 0x000F000Fu) | 0x64006400u);
            const __half2 v2 = as_h2((x & 0x00F000F0u) | 0x64006400u), v3 = as_h2((x8 & 0x00F000F0u) | 0x64006400u);
            const uint32_t t0 = h2_bits(__hfma2(__hsub2(v0, k1032), r, k1536));
            const uint32_t t1 = h2_bits(__hfma2(__hsub2(v1, k1032), r, k1536));
            const uint32_t t2 = h2_bits(__hfma2(__hsub2(v2, k1152), r16, k1536));
            const uint32_t t3 = h2_bits(__hfma2(__hsub2(v3, k1152), r16, k1536));
            lo[w] = __byte_perm(t0, t1, 0x6240);
            hi[w] = __byte_perm(t2, t3, 0x6240);
        }
    } else {
        float A, B;  // w = A q + B
        if (FMT == gemv::FAST_P4M) { A = h2f(s0); B = h2f(s1); }
        else {
            A = h2f(s1) * (float)(s0 & 0xff);
            B = -h2f(s1 >> 16) * (float)((s0 >> 8) & 0xff);
        }
        const __half2 a = __float2half2_rn(A * invs), a16 = __float2half2_rn(A * invs * 0.0625f);
        const __half2 b = __float2half2_rn(B * invs);
        const __half2 k1024 = __float2half2_rn(1024.f);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t x = q[w], x8 = x >> 8;
            uint32_t m0 = 0, m1 = 0, m2 = 0, m3 = 0;
            if (FMT == gemv::FAST_K5) {
                const uint32_t H = hb >> w;
                m0 = (H << 4) & 0x00100010u; m1 = (H >> 4) & 0x00100010u;
                m2 = (H << 4) & 0x01000100u; m3 = (H >> 4) & 0x01000100u;
            }
            const __half2 v0 = as_h2((x & 0x000F000Fu) | m0 | 0x64006400u);
            const __half2 v1 = as_h2((x8 & 0x000F000Fu) | m1 | 0x64006400u);
            const __half2 v2 = as_h2((x & 0x00F000F0u) | m2 | 0x64006400u);
            const __half2 v3 = as_h2((x8 & 0x00F000F0u) | m3 | 0x64006400u);
            const uint32_t t0 = h2_bits(__hadd2(__hfma2(__hsub2(v0, k1024), a, b), k1536));
            const uint32_t t1 = h2_bits(__hadd2(__hfma2(__hsub2(v1, k1024), a, b), k1536));
            const uint32_t t2 = h2_bits(__hadd2(__hfma2(__hsub2(v2, k1024), a16, b), k1536));
            const uint32_t t3 = h2_bits(__hadd2(__hfma2(__hsub2(v3, k1024), a16, b), k1536));
            lo[w] = __byte_perm(t0, t1, 0x6240);
            hi[w] = __byte_perm(t2, t3, 0x6240);
        }
    }
}

// decode layout -> w8[N][K] (one thread per (row, 32-block); rows of consecutive threads are consecutive blocks)
template <int FMT, int RPL>
__global__ void __launch_bounds__(256) w8_convert_kernel(const gemm8::Args a, const float* __restrict__ invs,
                                                         int8_t* __restrict__ out) {
    const int nkb = a.K >> 5;
    const long long idx = (long long)blockIdx.x * 256 + threadIdx.x;
    if (idx >= (long long)a.N * nkb) return;
    const int R = (int)(idx / nkb), kb = (int)(idx % nkb);
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
    uint32_t lo[4], hi[4];
    convert32<FMT>(wq, s0, s1, hb, invs[R], lo, hi);
    int4* o = (int4*)(out + (size_t)R * a.K + kb * 32);
    o[0] = make_int4(lo[0], lo[1], lo[2], lo[3]);
    o[1] = make_int4(hi[0], hi[1], hi[2], hi[3]);
}

// 16-B global load: LDH 0 = ld.global.nc (texture path), 1 = ld.global.L2::128B (CUTLASS's sm_75 default)
template <int LDH>
__device__ __forceinline__ int4 ldg16(const void* p) {
    if (LDH == 0) return __ldg((const int4*)p);
    int4 v;
    asm volatile("ld.global.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

struct Stg {
    int4 a[2];        // A tile: 2 x 16 B (SWAP 0: weights, WSRC 1: a[0] = 16 B of codes converted into a[0..1])
    int4 b[4];        // B tile: 4 x 16 B
    uint32_t s0, s1, hb;
};

// SWAP 0: A (mma m side, 128 rows of the block) = weight rows, B (n side, 256) = tokens; grid (Tp/256, N/128).
// SWAP 1: A = 128 tokens, B = 256 weight rows (CUTLASS's orientation for this problem); grid (Tp/128, N/256).
template <int WSRC, int FMT, int RPL, int OUT, int SWAP, int LDH>
__global__ void __launch_bounds__(NT, 1) gemm16_kernel(const Args a) {
    static_assert(!(SWAP && WSRC), "in-place weights only with SWAP 0");
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr int MA = 128, NB = 256;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp & 1, wn = warp >> 1;
    const int tok0 = blockIdx.x * (SWAP ? MA : NB), row0 = blockIdx.y * (SWAP ? NB : MA);
    const int K = a.K, nst = K >> 6;
    const int t4 = lane & 3;
    // staging: A row tid/2, units (tid&1)*2 + {0,1}; B row tid/4 + 64 i, unit tid&3
    const int arw = tid >> 1, ablk = tid & 1;
    const int brw = tid >> 2, bu = tid & 3;
    const int8_t* ap = SWAP ? a.xq + (size_t)(tok0 + arw) * K + ablk * 32 : a.w8 + (size_t)(row0 + arw) * K + ablk * 32;
    const int8_t* bp = SWAP ? a.w8 + (size_t)(row0 + brw) * K + bu * 16 : a.xq + (size_t)(tok0 + brw) * K + bu * 16;
    const size_t bstep = (size_t)64 * K;
    const float invs_st = WSRC == 1 ? a.invs[row0 + arw] : 0.f;
    const gemm8::WPtr<FMT, RPL> P(a.q, row0 + arw, ablk);

    int acc[8][8][2];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int g = 0; g < 8; ++g) acc[i][g][0] = acc[i][g][1] = 0;

    Stg S;
    auto load = [&](int st) {
        const int k0 = st * 64;
        if (WSRC == 0) {
            S.a[0] = ldg16<LDH>(ap + k0);
            S.a[1] = ldg16<LDH>(ap + k0 + 16);
        } else {
            const int c = (st * 2) >> 4, jj = (st * 2) & 15;
            S.a[0] = gemv::ldg_nc_v4(P.code + c * P.cs_code + jj * 16);
            if (FMT == gemv::FAST_K5) {
                S.s0 = (uint32_t)__ldg((const unsigned short*)(P.sa + c * P.cs_s + jj * RPL * 2));
                S.s1 = __ldg((const unsigned int*)(P.sb + c * P.cs_d + (jj >> 3) * RPL * 4));
                S.hb = __ldg((const unsigned int*)(P.hq + c * P.cs_h + jj * 4));
            } else {
                S.s0 = (uint32_t)__ldg((const unsigned short*)(P.sa + c * P.cs_s + jj * RPL * 2));
                if (FMT == gemv::FAST_P4M) S.s1 = (uint32_t)__ldg((const unsigned short*)(P.sb + c * P.cs_s + jj * RPL * 2));
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) S.b[i] = ldg16<LDH>(bp + i * bstep + k0);
    };
    auto convert = [&]() {
        if (WSRC == 1) {
            uint32_t lo[4], hi[4];
            convert32<FMT>(S.a[0], S.s0, S.s1, S.hb, invs_st, lo, hi);
            S.a[0] = make_int4(lo[0], lo[1], lo[2], lo[3]);
            S.a[1] = make_int4(hi[0], hi[1], hi[2], hi[3]);
        }
    };
    auto store = [&](int b) {
        unsigned char* B = smem + b * STAGE_BYTES;
        unsigned char* wr = B + O_W + arw * 64;
        *(int4*)(wr + swz(arw, ablk * 2) * 16) = S.a[0];
        *(int4*)(wr + swz(arw, ablk * 2 + 1) * 16) = S.a[1];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = brw + 64 * i;
            *(int4*)(B + O_X + r * 64 + swz(r, bu) * 16) = S.b[i];
        }
    };
    int arow[2], brow[2];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        arow[q] = wm * 64 + (q * 4 + (lane >> 3)) * 8 + (lane & 7);
        brow[q] = wn * 64 + (q * 4 + (lane >> 3)) * 8 + (lane & 7);
    }
    uint32_t fa[2][8], fb[2][8];
    auto frag = [&](int slot, int b, int u) {
        const unsigned char* Bp = smem + b * STAGE_BYTES;
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            ldsm_x4(fa[slot][q * 4], fa[slot][q * 4 + 1], fa[slot][q * 4 + 2], fa[slot][q * 4 + 3],
                    Bp + O_W + arow[q] * 64 + swz(arow[q], u) * 16);
            ldsm_x4(fb[slot][q * 4], fb[slot][q * 4 + 1], fb[slot][q * 4 + 2], fb[slot][q * 4 + 3],
                    Bp + O_X + brow[q] * 64 + swz(brow[q], u) * 16);
        }
    };
    load(0);
    convert();
    store(0);
    __syncthreads();
    if (nst > 1) load(1);
    frag(0, 0, 0);
    int buf = 0;
    for (int s = 0; s < nst; ++s) {
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            if (u == 2 && s + 1 < nst) convert();
            if (u == 3) {
                if (s + 1 < nst) store(buf ^ 1);
                __syncthreads();
                buf ^= 1;
                if (s + 2 < nst) load(s + 2);
            }
            if (!(u == 3 && s + 1 == nst)) frag((u + 1) & 1, buf, (u + 1) & 3);
#pragma unroll
            for (int g = 0; g < 8; ++g)
#pragma unroll
                for (int ii = 0; ii < 8; ++ii) {
                    const int i = (g & 1) ? 7 - ii : ii;  // serpentine: A-operand reuse across g
                    gemm8::mma_s8p(acc[i][g][0], acc[i][g][1], fa[u & 1][i], fb[u & 1][g], acc[i][g][0], acc[i][g][1]);
                }
        }
    }
    if (SWAP == 0) {  // m = weight rows, n = tokens
        float dxv[8][2];
#pragma unroll
        for (int g = 0; g < 8; ++g)
#pragma unroll
            for (int e = 0; e < 2; ++e) dxv[g][e] = a.dx[tok0 + wn * 64 + 8 * g + 2 * t4 + e];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = row0 + wm * 64 + i * 8 + (lane >> 2);
            const float sr = __frcp_rn(a.invs[row]);
#pragma unroll
            for (int g = 0; g < 8; ++g)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int tok = tok0 + wn * 64 + 8 * g + 2 * t4 + e;
                    if (tok < a.T) {
                        const float v = (float)acc[i][g][e] * (dxv[g][e] * sr);
                        if (OUT == 1) {
                            a.yh[(size_t)tok * a.ldy + row] = __float2half_rn(v);
                        } else {
                            float* p = a.y + (size_t)tok * a.ldy + row;
                            *p = a.accumulate ? *p + v : v;
                        }
                    }
                }
        }
    } else {  // m = tokens, n = weight rows: adjacent e are adjacent rows (8-B stores)
        float srv[8][2];
#pragma unroll
        for (int g = 0; g < 8; ++g)
#pragma unroll
            for (int e = 0; e < 2; ++e) srv[g][e] = __frcp_rn(a.invs[row0 + wn * 64 + 8 * g + 2 * t4 + e]);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int tok = tok0 + wm * 64 + i * 8 + (lane >> 2);
            if (tok >= a.T) continue;
            const float d = a.dx[tok];
#pragma unroll
            for (int g = 0; g < 8; ++g) {
                const int row = row0 + wn * 64 + 8 * g + 2 * t4;
                const float v0 = (float)acc[i][g][0] * (d * srv[g][0]), v1 = (float)acc[i][g][1] * (d * srv[g][1]);
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
}

template <int WSRC, int FMT, int RPL, int OUT, int SWAP = 0, int LDH = 0>
static cudaError_t launch_t(const Args& a, cudaStream_t s) {
    auto k = gemm16_kernel<WSRC, FMT, RPL, OUT, SWAP, LDH>;
    static int attr_dev_mask = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    if (!(attr_dev_mask & (1 << dev))) {
        cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        if (e != cudaSuccess) return e;
        attr_dev_mask |= 1 << dev;
    }
    const int TT = SWAP ? 128 : 256, TR = SWAP ? 256 : 128;
    if (a.N % TR || a.K % 64 || a.Tp % TT) return cudaErrorInvalidValue;
    dim3 grid(a.Tp / TT, a.N / TR);
    k<<<grid, NT, SMEM, s>>>(a);
    return cudaGetLastError();
}

// wsrc 0: a.w8 (any weight format, pre-converted); wsrc 1: decode layout a.q (fmt / rpl)
static inline cudaError_t launch(int wsrc, int fmt, int rpl, const Args& a, cudaStream_t s) {
    const bool h = a.yh != nullptr;
    if (wsrc == 0) return h ? launch_t<0, gemv::FAST_P4, 4, 1>(a, s) : launch_t<0, gemv::FAST_P4, 4, 0>(a, s);
#define T4Q_G16(F, R) \
    if (fmt == F && rpl == R) return h ? launch_t<1, F, R, 1>(a, s) : launch_t<1, F, R, 0>(a, s);
    T4Q_G16(gemv::FAST_P4, 4) T4Q_G16(gemv::FAST_P4, 2)
    T4Q_G16(gemv::FAST_P4M, 4) T4Q_G16(gemv::FAST_P4M, 2)
    T4Q_G16(gemv::FAST_K5, 4) T4Q_G16(gemv::FAST_K5, 2)
#undef T4Q_G16
    return cudaErrorInvalidValue;
}

// ================================================================================================ gemm17
// CUTLASS orientation (A = 128 tokens, B = 256 weight rows, grid (Tp/128, N/256)), weights either pre-converted int8
// rows (WSRC 0) or the decode layout converted in registers (WSRC 1, 2 (row, 32-block) units per thread per stage,
// rows tid/2 and tid/2 + 128), L2::128B activation loads.
// GSH 0: one activation scale per token (a.dx[t]); the k loop is pure ldmatrix + mma.
// GSH 1: "shift-folded" groups: activations are int8 with step D_t 2^-e[g][t] / 127 per 64-k group g (e in 0..7);
// the int32 accumulator T is kept in units of the current group's step (Horner): before group g's mmas
// T <<= d or T >>= -d with d = e[g] - e[g-1] (a.dsh, int8 [K/64][Tp], token order permuted per 64-token chunk as
// [tok % 8][(tok / 8) % 8] so each thread's 8 tokens are one 8-byte smem word); a.dx[t] = D_t 2^-e[last] / 127.
// Shifts cost 2 integer ops per accumulator per group (no extra registers, unlike a float fold).
struct Stg17 {
    int4 a[2];     // tokens: 2 x 16 B
    int4 b[4];     // weights: 4 x 16 B (WSRC 1: b[0], b[2] = codes of the two (row, block) units before conversion)
    uint32_t s0[2], s1[2], hb[2];
    int4 dsh;      // GSH: 16 B of shift deltas (threads < 8)
};

// KNOB (bench A/B): bit 0 global loads of stage s+1 issued at k-group 0 of stage s (CUTLASS MmaPipelined order) instead
// of right after the barrier; bit 1 mma .satfinite; bit 2 plain (non-serpentine) mma order
__device__ __forceinline__ void mma_s8sat(int& d0, int& d1, uint32_t a, uint32_t b, int c0, int c1) {
    asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};"
        : "=r"(d0), "=r"(d1) : "r"(a), "r"(b), "r"(c0), "r"(c1));
}
template <int WSRC, int FMT, int RPL, int OUT, int GSH, int KNOB = 0>
__global__ void __launch_bounds__(NT, 1) gemm17_kernel(const Args a) {
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr int ST = STAGE_BYTES + (GSH == 1 ? 128 : 0);  // + 128 B of shift deltas per stage
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp & 1, wn = warp >> 1;
    const int tok0 = blockIdx.x * 128, row0 = blockIdx.y * 256;
    const int K = a.K, nst = K >> 6;
    const int t4 = lane & 3;
    const int arw = tid >> 1, ablk = tid & 1;  // A (tokens): row tid/2, 32-B half tid&1
    const int brw = tid >> 2, bu = tid & 3;    // B (WSRC 0): rows tid/4 + 64 i, unit tid&3
    const int KS = a.kstride ? a.kstride : K;
    const int8_t* ap = a.xq + (size_t)(tok0 + arw) * KS + ablk * 32;
    const int8_t* bp = a.w8 + (size_t)(row0 + brw) * KS + bu * 16;
    const size_t bstep = (size_t)64 * KS;
    const int8_t* dp = (GSH == 1 && a.dsh) ? (const int8_t*)a.dsh + tok0 + tid * 16 : nullptr;
    // WSRC 1: units (row tid/2, block tid&1) and (row tid/2 + 128, block tid&1)
    const gemm8::WPtr<FMT, RPL> P0(a.q, row0 + arw, ablk), P1(a.q, row0 + arw + 128, ablk);
    const float inv0 = WSRC == 1 ? a.invs[row0 + arw] : 0.f, inv1 = WSRC == 1 ? a.invs[row0 + arw + 128] : 0.f;

    int acc[8][8][2];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int g = 0; g < 8; ++g) acc[i][g][0] = acc[i][g][1] = 0;

    Stg17 S;
    auto ldq = [&](const gemm8::WPtr<FMT, RPL>& P, int c, int jj, int4& code, uint32_t& s0, uint32_t& s1, uint32_t& hb) {
        code = gemv::ldg_nc_v4(P.code + c * P.cs_code + jj * 16);
        if (FMT == gemv::FAST_K5) {
            s0 = (uint32_t)__ldg((const unsigned short*)(P.sa + c * P.cs_s + jj * RPL * 2));
            s1 = __ldg((const unsigned int*)(P.sb + c * P.cs_d + (jj >> 3) * RPL * 4));
            hb = __ldg((const unsigned int*)(P.hq + c * P.cs_h + jj * 4));
        } else {
            s0 = (uint32_t)__ldg((const unsigned short*)(P.sa + c * P.cs_s + jj * RPL * 2));
            if (FMT == gemv::FAST_P4M) s1 = (uint32_t)__ldg((const unsigned short*)(P.sb + c * P.cs_s + jj * RPL * 2));
        }
    };
    auto load = [&](int st) {
        const int k0 = st * 64;
        S.a[0] = ldg16<1>(ap + k0);
        S.a[1] = ldg16<1>(ap + k0 + 16);
        if (WSRC == 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) S.b[i] = ldg16<1>(bp + i * bstep + k0);
        } else {
            const int c = (st * 2) >> 4, jj = (st * 2) & 15;
            ldq(P0, c, jj, S.b[0], S.s0[0], S.s1[0], S.hb[0]);
            ldq(P1, c, jj, S.b[2], S.s0[1], S.s1[1], S.hb[1]);
        }
        if (GSH == 1 && tid < 8) S.dsh = __ldg((const int4*)(dp + (size_t)st * a.Tp));
    };
    auto convert = [&]() {
        if (WSRC == 1) {
            uint32_t lo[4], hi[4];
            convert32<FMT>(S.b[0], S.s0[0], S.s1[0], S.hb[0], inv0, lo, hi);
            S.b[0] = make_int4(lo[0], lo[1], lo[2], lo[3]);
            S.b[1] = make_int4(hi[0], hi[1], hi[2], hi[3]);
            convert32<FMT>(S.b[2], S.s0[1], S.s1[1], S.hb[1], inv1, lo, hi);
            S.b[2] = make_int4(lo[0], lo[1], lo[2], lo[3]);
            S.b[3] = make_int4(hi[0], hi[1], hi[2], hi[3]);
        }
    };
    auto store = [&](int b) {
        unsigned char* B = smem + b * ST;
        unsigned char* ar = B + O_W + arw * 64;
        *(int4*)(ar + swz(arw, ablk * 2) * 16) = S.a[0];
        *(int4*)(ar + swz(arw, ablk * 2 + 1) * 16) = S.a[1];
        if (WSRC == 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = brw + 64 * i;
                *(int4*)(B + O_X + r * 64 + swz(r, bu) * 16) = S.b[i];
            }
        } else {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = arw + 128 * h;
                *(int4*)(B + O_X + r * 64 + swz(r, ablk * 2) * 16) = S.b[2 * h];
                *(int4*)(B + O_X + r * 64 + swz(r, ablk * 2 + 1) * 16) = S.b[2 * h + 1];
            }
        }
        if (GSH == 1 && tid < 8) *(int4*)(B + STAGE_BYTES + tid * 16) = S.dsh;
    };
    int arow[2], brow[2];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        arow[q] = wm * 64 + (q * 4 + (lane >> 3)) * 8 + (lane & 7);
        brow[q] = wn * 64 + (q * 4 + (lane >> 3)) * 8 + (lane & 7);
    }
    uint32_t fa[2][8], fb[2][8];
    auto frag = [&](int slot, int b, int u) {
        const unsigned char* Bp = smem + b * ST;
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            ldsm_x4(fa[slot][q * 4], fa[slot][q * 4 + 1], fa[slot][q * 4 + 2], fa[slot][q * 4 + 3],
                    Bp + O_W + arow[q] * 64 + swz(arow[q], u) * 16);
            ldsm_x4(fb[slot][q * 4], fb[slot][q * 4 + 1], fb[slot][q * 4 + 2], fb[slot][q * 4 + 3],
                    Bp + O_X + brow[q] * 64 + swz(brow[q], u) * 16);
        }
    };
    load(0);
    convert();
    store(0);
    __syncthreads();
    if (!(KNOB & 1) && nst > 1) load(1);
    frag(0, 0, 0);
    int buf = 0;
    for (int s = 0; s < nst; ++s) {
        if (GSH == 2 && s > 0 && (s & 7) == 0) {  // 512-k block boundary: T *= es_{b-1} / es_b (exact up to rounding)
            const float* rp = a.dsr + (size_t)(s >> 3) * a.Tp + tok0 + wm * 64 + (lane >> 2);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float r = __ldg(rp + 8 * i);
#pragma unroll
                for (int g = 0; g < 8; ++g) {
                    acc[i][g][0] = __float2int_rn(__int2float_rn(acc[i][g][0]) * r);
                    acc[i][g][1] = __float2int_rn(__int2float_rn(acc[i][g][1]) * r);
                }
            }
        }
        if (GSH == 1 && s > 0) {  // Horner step: rescale T to this group's step (the deltas of stage s are in buf)
            const uint2 dd = *(const uint2*)(smem + buf * ST + STAGE_BYTES + wm * 64 + (lane >> 2) * 8);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int d = (int)(signed char)(((i < 4 ? dd.x : dd.y) >> (8 * (i & 3))) & 0xff);
                const int dl = max(d, 0), dr = max(-d, 0);
#pragma unroll
                for (int g = 0; g < 8; ++g) {
                    acc[i][g][0] = (acc[i][g][0] << dl) >> dr;
                    acc[i][g][1] = (acc[i][g][1] << dl) >> dr;
                }
            }
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            if (u == 2 && s + 1 < nst) convert();
            if (u == 3) {
                if (s + 1 < nst) store(buf ^ 1);
                __syncthreads();
                buf ^= 1;
                if (!(KNOB & 1) && s + 2 < nst) load(s + 2);
            }
            if (!(u == 3 && s + 1 == nst)) frag((u + 1) & 1, buf, (u + 1) & 3);
            if ((KNOB & 1) && u == 0 && s + 1 < nst) load(s + 1);
#pragma unroll
            for (int g = 0; g < 8; ++g)
#pragma unroll
                for (int ii = 0; ii < 8; ++ii) {
                    const int i = ((g & 1) && !(KNOB & 4)) ? 7 - ii : ii;
                    if (KNOB & 2) mma_s8sat(acc[i][g][0], acc[i][g][1], fa[u & 1][i], fb[u & 1][g], acc[i][g][0], acc[i][g][1]);
                    else gemm8::mma_s8p(acc[i][g][0], acc[i][g][1], fa[u & 1][i], fb[u & 1][g], acc[i][g][0], acc[i][g][1]);
                }
        }
    }
    float srv[8][2];
#pragma unroll
    for (int g = 0; g < 8; ++g)
#pragma unroll
        for (int e = 0; e < 2; ++e) srv[g][e] = __frcp_rn(a.invs[row0 + wn * 64 + 8 * g + 2 * t4 + e]);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int tok = tok0 + wm * 64 + i * 8 + (lane >> 2);
        if (OUT < 2 && tok >= a.T) continue;  // OUT 2/3 shuffle below: all lanes stay (dx of padding tokens is 0)
        const float d = a.dx[tok];
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int row = row0 + wn * 64 + 8 * g + 2 * t4;
            const float v0 = (float)acc[i][g][0] * (d * srv[g][0]), v1 = (float)acc[i][g][1] * (d * srv[g][1]);
            if (OUT < 2 && row >= a.nvalid) continue;
            if (OUT == 2 || OUT == 3) {  // gate|up rows interleaved by 4: lanes t4 0,1 gate rows, lane ^ 2 the up rows
                const float u0 = __shfl_xor_sync(0xffffffffu, v0, 2), u1 = __shfl_xor_sync(0xffffffffu, v1, 2);
                if (t4 < 2 && tok < a.T) {
                    const int f = ((row0 + wn * 64 + 8 * g) >> 1) + 2 * t4;
                    const float h0 = (v0 / (1.0f + expf(-v0))) * u0, h1 = (v1 / (1.0f + expf(-v1))) * u1;
                    if (OUT == 2) *(__half2*)(a.yh + (size_t)tok * a.ldy + f) = __floats2half2_rn(h0, h1);
                    else *(float2*)(a.y + (size_t)tok * a.ldy + f) = make_float2(h0, h1);  // fp32: no fp16 overflow
                }
            } else if (OUT == 1) {
                *(__half2*)(a.yh + (size_t)tok * a.ldy + row) = __floats2half2_rn(v0, v1);
            } else {
                float2* p = (float2*)(a.y + (size_t)tok * a.ldy + row);
                if (a.accumulate) { const float2 o = *p; *p = make_float2(o.x + v0, o.y + v1); }
                else *p = make_float2(v0, v1);
            }
        }
    }
}

template <int WSRC, int FMT, int RPL, int OUT, int GSH, int KNOB = 0>
static cudaError_t launch17_t(const Args& a, cudaStream_t s) {
    auto k = gemm17_kernel<WSRC, FMT, RPL, OUT, GSH, KNOB>;
    const int smem = 2 * (STAGE_BYTES + (GSH == 1 ? 128 : 0));
    static int attr_dev_mask = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    if (!(attr_dev_mask & (1 << dev))) {
        cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return e;
        attr_dev_mask |= 1 << dev;
    }
    if (a.N % 256 || a.K % 64 || a.Tp % 128 || (GSH == 1 && !a.dsh) || (GSH == 2 && (!a.dsr || a.K % 512))) return cudaErrorInvalidValue;
    dim3 grid(a.Tp / 128, a.N / 256);
    k<<<grid, NT, smem, s>>>(a);
    return cudaGetLastError();
}

// gsh: 0 per-token scales, 1 shift-folded 64-groups; weights from the decode layout (fmt / rpl)
static inline cudaError_t launch17(int fmt, int rpl, int gsh, const Args& a, cudaStream_t s) {
    const bool h = a.yh != nullptr;
#define T4Q_G17(F, R)                                                                              \
    if (fmt == F && rpl == R) {                                                                    \
        if (gsh) return h ? launch17_t<1, F, R, 1, 1>(a, s) : launch17_t<1, F, R, 0, 1>(a, s);     \
        return h ? launch17_t<1, F, R, 1, 0>(a, s) : launch17_t<1, F, R, 0, 0>(a, s);              \
    }
    T4Q_G17(gemv::FAST_P4, 4) T4Q_G17(gemv::FAST_P4, 2)
    T4Q_G17(gemv::FAST_P4M, 4) T4Q_G17(gemv::FAST_P4M, 2)
    T4Q_G17(gemv::FAST_K5, 4) T4Q_G17(gemv::FAST_K5, 2)
#undef T4Q_G17
    return cudaErrorInvalidValue;
}

// ------------------------------------------------------------------------------------------------ GSH activations
// x (fp32 rows, stride ldx) -> xq [Tp][K] int8, dsh [K/64][Tp] (permuted token order), dx [Tp] (final factor).
// One 256-thread block per token, K <= 8704. e[g] = min(emax, floor(log2(D / amax_g))), q = rint(x * 127 2^e / D).
__device__ __forceinline__ int dsh_pos(int t) { return (t & ~63) | ((t & 7) << 3) | ((t >> 3) & 7); }
template <class XT>
__global__ void __launch_bounds__(256) quant_gsh_kernel(const XT* __restrict__ x, int ldx, int T, int K, int emax,
                                                        int8_t* __restrict__ xq, int8_t* __restrict__ dsh,
                                                        float* __restrict__ dx, int Tp) {
    __shared__ float red[8];
    __shared__ signed char es[8704 / 64];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int ng = K >> 6;
    int8_t* dst = xq + (size_t)t * K;
    if (t >= T) {
        for (int k = tid * 4; k < K; k += 1024) *(int*)(dst + k) = 0;
        for (int g = tid; g < ng; g += 256) dsh[(size_t)g * Tp + dsh_pos(t)] = 0;
        if (tid == 0) dx[t] = 0.f;
        return;
    }
    const XT* src = x + (size_t)t * ldx;
    // token amax D (one warp per 64-group, 2 elements per lane)
    float am = 0.f;
    for (int g = warp; g < ng; g += 8)
        am = fmaxf(am, fmaxf(fabsf((float)src[g * 64 + lane]), fabsf((float)src[g * 64 + 32 + lane])));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    if (lane == 0) red[warp] = am;
    __syncthreads();
    float D = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) D = fmaxf(D, red[w]);
    for (int g = warp; g < ng; g += 8) {
        const float v0 = (float)src[g * 64 + lane], v1 = (float)src[g * 64 + 32 + lane];
        float m = fmaxf(fabsf(v0), fabsf(v1));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        int e = emax;
        if (m > 0.f) e = min(emax, (int)floorf(log2f(D / m)));
        if (D == 0.f) e = 0;
        const float sc = D > 0.f ? ldexpf(127.f, e) / D : 0.f;
        dst[g * 64 + lane] = (int8_t)__float2int_rn(v0 * sc);
        dst[g * 64 + 32 + lane] = (int8_t)__float2int_rn(v1 * sc);
        if (lane == 0) es[g] = (signed char)e;
    }
    __syncthreads();
    for (int g = tid; g < ng; g += 256) dsh[(size_t)g * Tp + dsh_pos(t)] = (int8_t)(g ? es[g] - es[g - 1] : 0);
    if (tid == 0) dx[t] = D > 0.f ? ldexpf(D, -es[ng - 1]) / 127.f : 0.f;
}

static inline cudaError_t w8_convert(int fmt, int rpl, const gemm8::Args& q, const float* invs, int8_t* out,
                                     cudaStream_t s) {
    const long long n = (long long)q.N * (q.K >> 5);
    const int grid = (int)((n + 255) / 256);
#define T4Q_W8C(F, R) \
    if (fmt == F && rpl == R) { w8_convert_kernel<F, R><<<grid, 256, 0, s>>>(q, invs, out); return cudaGetLastError(); }
    T4Q_W8C(gemv::FAST_P4, 4) T4Q_W8C(gemv::FAST_P4, 2)
    T4Q_W8C(gemv::FAST_P4M, 4) T4Q_W8C(gemv::FAST_P4M, 2)
    T4Q_W8C(gemv::FAST_K5, 4) T4Q_W8C(gemv::FAST_K5, 2)
#undef T4Q_W8C
    return cudaErrorInvalidValue;
}

}  // namespace g16
}  // namespace t4q
#endif
