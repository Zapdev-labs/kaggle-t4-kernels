// t4q/src/kernels/gemm21.cuh -- prefill GEMM v5 (milestone P round 4): gemm9's numerics (in-register Q4 -> per-row
// int8 weights, GA64 int8 activations, one FFMA per output per 64 k) in gemm17's CUTLASS orientation.
//
// gemm17 (plain int8, per-token activations) sustains ~1300 ops/clk/SM at the 70 W cap against ~745 for gemm9, but a
// per-token activation scale is not accurate enough (round 3). gemm14/15 put the GA64 fold into gemm17's k-step-outer
// loop and either spilled (a second accumulator set) or reloaded A fragments. Here the loop is tile-outer within a
// 64-k stage: all A fragments of the stage (4 k16 steps x 8 m tiles = 32 registers) are loaded once, B fragments one
// n tile at a time (one ldmatrix.x4 = the 4 k16 steps of 8 rows), and each (m, n) tile runs its 4 mmas as a chain
// that starts from MAGIC, so the int32 result is the float MAGIC + P and is folded right away:
//   FB 0: F = fma(P' - MAGIC, a_g, F)          (exact, 2 ALU per output per 64 k)
//   FB 1: F = fma(P', a_g, F), and every 4 stages F -= MAGIC * sum(a_g)   (gemm9's bias trick, 1.25 ALU)
// The int32 accumulators never outlive the tile, so the persistent state is the 128 fp32 F values per thread.
// Weights: decode layout read in place (P4 / P4M / K5), converted to int8 with one scale per row (gemm8::row_invs,
// same arithmetic as gemm9 / gemm17 WSRC 1) right before the smem store. Output y = F / invs[n].
// Activations: q8 with one scale per 64 (gemm8::quant8 GA 64 layout: xq [Tp][K], dx [K/64][Tp]).
#pragma once
#include "gemm16.cuh"

#ifdef __CUDACC__
namespace t4q {
namespace g21 {

constexpr int NT = 256;
// BT tokens x (384 - BT) weight rows per block: BT 128 (CUTLASS orientation, 128 x 256) or 256 (256 x 128: each weight
// element is converted once per 256 tokens, as in gemm9)
template <int BT>
struct Cfg {
    static constexpr int BR = BT == 128 ? 256 : 128;
    static constexpr int O_A = 0, O_B = BT * 64, O_S = O_B + BR * 64, STAGE = O_S + BT * 4, SMEM = 2 * STAGE;
    static constexpr int WMN = BT / 64;  // warps along tokens (2 or 4); along rows: 8 / WMN
};

struct Args {
    gemm8::Args q;               // decode-layout weights (codes / d / sc / qh, N, K, ntiles, cm)
    const int8_t* w8 = nullptr;  // WSRC 0: pre-converted int8 rows [N][K] (g16::w8_convert, same numerics)
    const float* invs = nullptr; // [N] per-row int8 scale (127 / max |w|)
    const int8_t* xq = nullptr;  // [Tp][K]
    const float* dx = nullptr;   // [K/64][Tp]
    float* y = nullptr;          // y[t * ldy + n]
    __half* yh = nullptr;        // fp16 output instead (no accumulate)
    int ldy = 0, N = 0, K = 0, T = 0, Tp = 0, accumulate = 0;
};

template <int FMT, int RPL, int FB, int BT, int WSRC>
__global__ void __launch_bounds__(NT, 1) gemm21_kernel(const Args a) {
    using C = Cfg<BT>;
    constexpr int BR = C::BR, WMN = C::WMN;
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp % WMN, wn = warp / WMN;
    const int tok0 = blockIdx.x * BT, row0 = blockIdx.y * BR;
    const int K = a.K, nst = K >> 6;
    const int t4 = lane & 3;
    // A staging: BT 128: row tid/2, 32-B half tid&1; BT 256: row tid, all 64 B
    constexpr int NA = BT / 64;  // 16-B A units per thread
    const int arw = BT == 128 ? (tid >> 1) : tid, au0 = BT == 128 ? (tid & 1) * 2 : 0;
    const int8_t* ap = a.xq + (size_t)(tok0 + arw) * K + au0 * 16;
    const float* dxp = a.dx + tok0 + (tid % BT);
    // weight staging (WSRC 1): (row, 32-block) units: BT 128: rows tid/2 and tid/2 + 128, block tid&1; BT 256: row
    // tid/2, block tid&1. WSRC 0: the same rows, 32 B (one block) each
    constexpr int NW = BR / 128;  // weight units per thread
    const int wr = tid >> 1, wblk = tid & 1;
    const gemm8::WPtr<FMT, RPL> P0(a.q, row0 + wr, wblk), P1(a.q, row0 + wr + (NW > 1 ? 128 : 0), wblk);
    const float inv0 = a.invs[row0 + wr], inv1 = NW > 1 ? a.invs[row0 + wr + 128] : 0.f;

    float F[8][8][2];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int g = 0; g < 8; ++g) F[i][g][0] = F[i][g][1] = 0.f;

    int4 sa[NA], sw[2 * NW];
    uint32_t s0[NW], s1[NW], hb[NW];
    float sdx = 0.f;
    auto ldq = [&](const gemm8::WPtr<FMT, RPL>& P, int c, int jj, int4& code, uint32_t& x0, uint32_t& x1, uint32_t& xh) {
        code = gemv::ldg_nc_v4(P.code + c * P.cs_code + jj * 16);
        if (FMT == gemv::FAST_K5) {
            x0 = (uint32_t)__ldg((const unsigned short*)(P.sa + c * P.cs_s + jj * RPL * 2));
            x1 = __ldg((const unsigned int*)(P.sb + c * P.cs_d + (jj >> 3) * RPL * 4));
            xh = __ldg((const unsigned int*)(P.hq + c * P.cs_h + jj * 4));
        } else {
            x0 = (uint32_t)__ldg((const unsigned short*)(P.sa + c * P.cs_s + jj * RPL * 2));
            if (FMT == gemv::FAST_P4M) x1 = (uint32_t)__ldg((const unsigned short*)(P.sb + c * P.cs_s + jj * RPL * 2));
        }
    };
    auto load = [&](int st) {
        const int k0 = st * 64;
#pragma unroll
        for (int u = 0; u < NA; ++u) sa[u] = g16::ldg16<1>(ap + k0 + 16 * u);
        if (WSRC == 1) {
            const int c = (st * 2) >> 4, jj = (st * 2) & 15;
            ldq(P0, c, jj, sw[0], s0[0], s1[0], hb[0]);
            if (NW > 1) ldq(P1, c, jj, sw[2], s0[NW - 1], s1[NW - 1], hb[NW - 1]);
        } else {
#pragma unroll
            for (int h = 0; h < NW; ++h) {
                const int8_t* wp = a.w8 + (size_t)(row0 + wr + 128 * h) * K + k0 + wblk * 32;
                sw[2 * h] = g16::ldg16<1>(wp);
                sw[2 * h + 1] = g16::ldg16<1>(wp + 16);
            }
        }
        if (BT == 256 || tid < 128) sdx = __ldg(dxp + (size_t)st * a.Tp);
    };
    auto convert_store = [&](int b) {
        if (WSRC == 1) {
#pragma unroll
            for (int h = 0; h < NW; ++h) {
                uint32_t lo[4], hi[4];
                g16::convert32<FMT>(sw[2 * h], s0[h], s1[h], hb[h], h ? inv1 : inv0, lo, hi);
                sw[2 * h] = make_int4(lo[0], lo[1], lo[2], lo[3]);
                sw[2 * h + 1] = make_int4(hi[0], hi[1], hi[2], hi[3]);
            }
        }
        unsigned char* B = smem + b * C::STAGE;
#pragma unroll
        for (int u = 0; u < NA; ++u) *(int4*)(B + C::O_A + arw * 64 + gemm8::swz(arw, au0 + u) * 16) = sa[u];
#pragma unroll
        for (int h = 0; h < NW; ++h) {
            const int r = wr + 128 * h;
            *(int4*)(B + C::O_B + r * 64 + gemm8::swz(r, wblk * 2) * 16) = sw[2 * h];
            *(int4*)(B + C::O_B + r * 64 + gemm8::swz(r, wblk * 2 + 1) * 16) = sw[2 * h + 1];
        }
        if (BT == 256 || tid < 128) *(float*)(B + C::O_S + (tid % BT) * 4) = sdx;
    };
    int arow[2];
#pragma unroll
    for (int q = 0; q < 2; ++q) arow[q] = wm * 64 + (q * 4 + (lane >> 3)) * 8 + (lane & 7);
    const int brow = wn * 64 + (lane & 7);  // + 8 g; matrix j = lane >> 3 is k16 step j
    float bsum[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) bsum[i] = 0.f;

    load(0);
    convert_store(0);
    __syncthreads();
    int buf = 0;
    for (int s = 0; s < nst; ++s) {
        if (s + 1 < nst) load(s + 1);
        const unsigned char* Bp = smem + buf * C::STAGE;
        uint32_t fa[4][8];  // [k16 step][m tile]
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int q = 0; q < 2; ++q)
                gemm8::ldsm_x4(fa[u][q * 4], fa[u][q * 4 + 1], fa[u][q * 4 + 2], fa[u][q * 4 + 3],
                               Bp + C::O_A + arow[q] * 64 + gemm8::swz(arow[q], u) * 16);
        float xa[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) xa[i] = *(const float*)(Bp + C::O_S + (wm * 64 + i * 8 + (lane >> 2)) * 4);
        if (FB == 1) {
#pragma unroll
            for (int i = 0; i < 8; ++i) bsum[i] += xa[i];
        }
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            uint32_t fb[4];
            {
                const int r = brow + 8 * g;
                gemm8::ldsm_x4(fb[0], fb[1], fb[2], fb[3], Bp + C::O_B + r * 64 + gemm8::swz(r, lane >> 3) * 16);
            }
#pragma unroll
            for (int ii = 0; ii < 8; ++ii) {
                const int i = (g & 1) ? 7 - ii : ii;
                int c0, c1;
                gemm8::mma_s8p(c0, c1, fa[0][i], fb[0], gemm8::MAGIC_I, gemm8::MAGIC_I);
                gemm8::mma_s8p(c0, c1, fa[1][i], fb[1], c0, c1);
                gemm8::mma_s8p(c0, c1, fa[2][i], fb[2], c0, c1);
                gemm8::mma_s8p(c0, c1, fa[3][i], fb[3], c0, c1);
                if (FB == 0) {
                    F[i][g][0] = fmaf(__int_as_float(c0) - gemm8::MAGIC_F, xa[i], F[i][g][0]);
                    F[i][g][1] = fmaf(__int_as_float(c1) - gemm8::MAGIC_F, xa[i], F[i][g][1]);
                } else {
                    F[i][g][0] = fmaf(__int_as_float(c0), xa[i], F[i][g][0]);
                    F[i][g][1] = fmaf(__int_as_float(c1), xa[i], F[i][g][1]);
                }
            }
        }
        if (FB == 1 && ((s & 3) == 3 || s + 1 == nst)) {  // remove MAGIC * sum(a_g) of the last <= 4 stages
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float bv = bsum[i] * gemm8::MAGIC_F;
                bsum[i] = 0.f;
#pragma unroll
                for (int g = 0; g < 8; ++g) { F[i][g][0] -= bv; F[i][g][1] -= bv; }
            }
        }
        if (s + 1 < nst) convert_store(buf ^ 1);
        __syncthreads();
        buf ^= 1;
    }
    float srv[8][2];
#pragma unroll
    for (int g = 0; g < 8; ++g)
#pragma unroll
        for (int e = 0; e < 2; ++e) srv[g][e] = __frcp_rn(a.invs[row0 + wn * 64 + 8 * g + 2 * t4 + e]);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int tok = tok0 + wm * 64 + i * 8 + (lane >> 2);
        if (tok >= a.T) continue;
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int row = row0 + wn * 64 + 8 * g + 2 * t4;
            const float v0 = F[i][g][0] * srv[g][0], v1 = F[i][g][1] * srv[g][1];
            if (a.yh) {
                *(__half2*)(a.yh + (size_t)tok * a.ldy + row) = __floats2half2_rn(v0, v1);
            } else {
                float2* p = (float2*)(a.y + (size_t)tok * a.ldy + row);
                if (a.accumulate) { const float2 o = *p; *p = make_float2(o.x + v0, o.y + v1); }
                else *p = make_float2(v0, v1);
            }
        }
    }
}

template <int FMT, int RPL, int FB, int BT = 128, int WSRC = 1>
static cudaError_t launch_t(const Args& a, cudaStream_t s) {
    using C = Cfg<BT>;
    auto k = gemm21_kernel<FMT, RPL, FB, BT, WSRC>;
    static int attr_dev_mask = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    if (!(attr_dev_mask & (1 << dev))) {
        cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
        if (e != cudaSuccess) return e;
        attr_dev_mask |= 1 << dev;
    }
    if (a.N % C::BR || a.K % 64 || a.Tp % BT) return cudaErrorInvalidValue;
    dim3 grid(a.Tp / BT, a.N / C::BR);
    k<<<grid, NT, C::SMEM, s>>>(a);
    return cudaGetLastError();
}

static inline cudaError_t launch(int fmt, int rpl, int fb, const Args& a, cudaStream_t s) {
#define T4Q_G21(F, R) \
    if (fmt == F && rpl == R) return fb ? launch_t<F, R, 1>(a, s) : launch_t<F, R, 0>(a, s);
    T4Q_G21(gemv::FAST_P4, 4) T4Q_G21(gemv::FAST_P4, 2)
    T4Q_G21(gemv::FAST_P4M, 4) T4Q_G21(gemv::FAST_P4M, 2)
    T4Q_G21(gemv::FAST_K5, 4) T4Q_G21(gemv::FAST_K5, 2)
#undef T4Q_G21
    return cudaErrorInvalidValue;
}

}  // namespace g21
}  // namespace t4q
#endif
