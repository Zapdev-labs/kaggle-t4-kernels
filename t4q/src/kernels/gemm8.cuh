// t4q/src/kernels/gemm8.cuh -- prefill GEMM v2 (milestone P round 2): in-kernel W4 -> int8 requantization.
//
// Why: the round-1 W4A8 kernel (gemm.cuh) applies the per-32 weight scale and the per-32 activation scale with two
// FFMAs per output per 32-block. That keeps the FP32 pipe as busy as the int8 tensor pipe and the kernel reached only
// ~31% of tensor peak per clock (24-26 TOPS sustained at the 70 W cap), while CUTLASS's plain int8 GEMM sustains 45-55
// TOPS on the same box (ref_bench, t4q-p v7). This kernel removes the weight scale from the inner loop:
//
//   * Weights: each block converts its 128-row weight tile from the decode layout (P4/P4M/K5, read in place) to int8
//     with one scale per ROW (whole K), w8 = round(w * invs[n]), invs = 127 / max_k |w[n][k]| (row_invs_kernel, once
//     per weight). Conversion runs once per weight element per block (amortized over BN tokens) in fp16x2:
//     P4: (1024 + c) - 1032 = c - 8 exactly, fma(c - 8, r, 1536) rounds to the integer in the low byte; P4M / K5:
//     fma(q, a, b) + 1536 (affine). Measured on real Qwen3.8 tensors, per-row requantization adds ~1% of Q4_0's own
//     noise variance (rel error of y ~ 1e-2 vs the exact Q4 product; act-q8 noise is ~0.7e-2).
//   * Activations: exact prefill q8 (int8 per 32-block, d = amax/127), as in gemm.cuh.
//   * Math: per 32-block the two k16 mmas accumulate on top of MAGIC (0x4B400000), so the int32 result is the float
//     M + S exactly; acc = fma(M + S, d_x, acc) (ONE FFMA per output per block). Every 4 blocks the accumulated
//     M * sum(d_x) is subtracted (bias computed by the staging threads), keeping acc small. Output y = acc / invs[n].
//
// Tiling: block = 128 weight rows x BN tokens (BN 256 or 128), 8 warps as 2 (rows) x 4 (tokens), warp tile 64 x BN/4
// (8 x BN/32 mma tiles), K stage = 2 blocks (64 k), register-staged double buffer (sm_75 has no cp.async), one
// __syncthreads per stage. Both operands live in smem as [row|token][64 B] with a 16-B unit swizzle, read by ldmatrix.
// Grid: x = token tiles (fast, so concurrently running blocks share the weight tile in L2), y = 128-row tiles.
#pragma once
#include "gemv.cuh"

namespace t4q {
namespace gemm8 {

constexpr int BM = 128, NT = 256;
constexpr int MAGIC_I = 0x4B400000;
constexpr float MAGIC_F = 12582912.f;

template <int BN>
struct Cfg {
    static constexpr int WN = BN / 4;   // tokens per warp
    static constexpr int NG = WN / 8;   // 8-token groups per warp
    static constexpr int O_W = 0;                   // [128 rows][64 B]
    static constexpr int O_X = O_W + BM * 64;       // [BN tokens][64 B]
    static constexpr int O_DX = O_X + BN * 64;      // [2 blocks][BN] float
    static constexpr int O_BI = O_DX + 2 * BN * 4;  // [BN] float: M * sum d_x over the 4-block group ending here
    static constexpr int BYTES = O_BI + BN * 4;
    static constexpr int NXA = BN * 4 / NT;         // 16-B activation units per thread per stage
};

static inline int smem_bytes(int bn) { return bn == 256 ? 2 * Cfg<256>::BYTES : 2 * Cfg<128>::BYTES; }

// host mirror of the per-row requantization (one row, natural-order fp32 weights): w8 and the row scale invs
static inline float row_invs_host(const float* w, int K) {
    float m = 0.f;
    for (int k = 0; k < K; ++k) m = std::fmax(m, std::fabs(w[k]));
    return m > 0.f ? 127.f / m : 1.f;
}

}  // namespace gemm8
}  // namespace t4q

#ifdef __CUDACC__
#include <cuda_fp16.h>
namespace t4q {
namespace gemm8 {

struct Args {
    const uint8_t* codes = nullptr;
    const uint8_t* qh = nullptr;
    const uint8_t* sc = nullptr;
    const uint16_t* d = nullptr;
    int N = 0, K = 0, ntiles = 0, cm = 0;
    const float* invs = nullptr;  // [N]
    const int8_t* xq = nullptr;   // [Tp][K]
    const float* dx = nullptr;    // [K/32][Tp] activation block scale d = amax/127 (0 for padding tokens)
    float* y = nullptr;           // y[t * ldy + n]
    int ldy = 0, T = 0, Tp = 0, accumulate = 0;
};

static inline Args make_args(const gemv::Layout& L, const uint8_t* base, const float* invs, const int8_t* xq,
                             const float* dx, float* y, int ldy, int T, int Tp) {
    Args a;
    a.codes = base + L.off_codes; a.qh = base + L.off_qh; a.sc = base + L.off_sc;
    a.d = (const uint16_t*)(base + L.off_d);
    a.N = L.N; a.K = L.K; a.ntiles = L.ntiles; a.cm = L.cm;
    a.invs = invs; a.xq = xq; a.dx = dx; a.y = y; a.ldy = ldy; a.T = T; a.Tp = Tp;
    return a;
}

__device__ __forceinline__ void mma_s8(int& d0, int& d1, uint32_t a, uint32_t b, int c0, int c1) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};"
                 : "=r"(d0), "=r"(d1) : "r"(a), "r"(b), "r"(c0), "r"(c1));
}
// non-volatile: a pure function of its inputs, so ptxas may schedule it freely
__device__ __forceinline__ void mma_s8p(int& d0, int& d1, uint32_t a, uint32_t b, int c0, int c1) {
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};"
        : "=r"(d0), "=r"(d1) : "r"(a), "r"(b), "r"(c0), "r"(c1));
}
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const unsigned sp = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(sp));
}
__device__ __forceinline__ float h2f(uint32_t b) { return __half2float(__ushort_as_half((unsigned short)(b & 0xffff))); }
__device__ __forceinline__ __half2 as_h2(uint32_t v) { return *reinterpret_cast<const __half2*>(&v); }
__device__ __forceinline__ uint32_t h2_bits(__half2 h) { return *reinterpret_cast<const uint32_t*>(&h); }
// 16-B unit swizzle for 64-B rows: conflict-free ldmatrix over 8 consecutive rows
__device__ __forceinline__ int swz(int row, int u) { return u ^ ((row >> 1) & 3); }

// ------------------------------------------------------------------------------------------------ weight coordinates
// (row R, global 32-block kb) -> layout tile/plane indices (gemv.cuh repack_host)
template <int RPL>
struct WPos {
    size_t tc;
    int r, h, lane, j;
    __device__ __forceinline__ WPos(int R, int kb, int nch, int ntiles, int cm) {
        const int lt = R / (2 * RPL), w = R % (2 * RPL);
        h = w / RPL; r = w % RPL;
        const int c = kb >> 4;
        j = kb & 15;
        lane = h * 16 + j;
        tc = gemv::tc_index(lt, c, nch, ntiles, cm);
    }
};

// max |w| over one 32-block of row R (P4: 8|d|; P4M: max(|m|, |15d + m|); K5: max(|B|, |31A - B|))
template <int FMT, int RPL>
__device__ __forceinline__ float block_absmax(const Args& a, int R, int kb) {
    const WPos<RPL> p(R, kb, a.K >> 9, a.ntiles, a.cm);
    if (FMT == gemv::FAST_P4) return 8.f * fabsf(h2f(a.d[(p.tc * 32 + p.lane) * RPL + p.r]));
    if (FMT == gemv::FAST_P4M) {
        const float d = h2f(a.d[(p.tc * 32 + p.lane) * RPL + p.r]);
        const float m = h2f(((const uint16_t*)a.sc)[(p.tc * 32 + p.lane) * RPL + p.r]);
        return fmaxf(fabsf(m), fabsf(15.f * d + m));
    }
    const uint8_t* sp = a.sc + ((p.tc * 32 + p.lane) * RPL + p.r) * 2;
    const uint32_t dd = ((const uint32_t*)a.d)[((p.tc * 2 + p.h) * 2 + (p.j >> 3)) * RPL + p.r];
    const float A = h2f(dd) * (float)sp[0], B = h2f(dd >> 16) * (float)sp[1];
    return fmaxf(fabsf(B), fabsf(31.f * A - B));
}

// invs[n] = 127 / max_k |w[n][k]| (1 for an all-zero row). One warp per row.
template <int FMT, int RPL>
__global__ void row_invs_kernel(const Args a, float* invs) {
    const int R = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (R >= a.N) return;
    float m = 0.f;
    for (int kb = lane; kb < (a.K >> 5); kb += 32) m = fmaxf(m, block_absmax<FMT, RPL>(a, R, kb));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) invs[R] = m > 0.f ? 127.f / m : 1.f;
}

// ------------------------------------------------------------------------------------------------ staging
template <int BN>
struct Stage {
    int4 wq;          // 16 B of codes: one (row, block)
    uint32_t s0, s1;  // P4: d | P4M: d, m | K5: {sc, mn} bytes, {d, dmin}
    uint32_t hb;      // K5 high bits
    int4 xa[Cfg<BN>::NXA];
    float2 dx0, dx1;  // threads < BN/2: token pair p, blocks kb0 and kb0 + 1
};

template <int FMT, int RPL, int BN>
__device__ __forceinline__ void stage_load(Stage<BN>& S, const Args& a, int row0, int tok0, int kb0, int tid) {
    {
        const int R = row0 + (tid >> 1), kb = kb0 + (tid & 1);
        const WPos<RPL> p(R, kb, a.K >> 9, a.ntiles, a.cm);
        S.wq = gemv::ldg_nc_v4(a.codes + (p.tc * RPL + p.r) * 512 + p.lane * 16);
        if (FMT == gemv::FAST_K5) {
            const uint8_t* sp = a.sc + ((p.tc * 32 + p.lane) * RPL + p.r) * 2;
            S.s0 = (uint32_t)__ldg((const unsigned short*)sp);
            S.s1 = __ldg((const unsigned int*)a.d + ((p.tc * 2 + p.h) * 2 + (p.j >> 3)) * RPL + p.r);
            S.hb = __ldg((const unsigned int*)(a.qh + (p.tc * RPL + p.r) * 128 + p.lane * 4));
        } else {
            S.s0 = (uint32_t)__ldg((const unsigned short*)a.d + (p.tc * 32 + p.lane) * RPL + p.r);
            if (FMT == gemv::FAST_P4M) S.s1 = (uint32_t)__ldg((const unsigned short*)a.sc + (p.tc * 32 + p.lane) * RPL + p.r);
        }
    }
#pragma unroll
    for (int i = 0; i < Cfg<BN>::NXA; ++i) {
        const int U = tid + i * NT, tok = U >> 2, u = U & 3;
        S.xa[i] = __ldg((const int4*)(a.xq + (size_t)(tok0 + tok) * a.K + kb0 * 32 + u * 16));
    }
    if (tid < BN / 2) {
        S.dx0 = __ldg((const float2*)(a.dx + (size_t)kb0 * a.Tp + tok0) + tid);
        S.dx1 = __ldg((const float2*)(a.dx + (size_t)(kb0 + 1) * a.Tp + tok0) + tid);
    }
}

template <int FMT, int BN, int MODE = 0, int PART = 3>
// ph: bias group phase of this stage: 0 first (bkeep = partial), 1 middle (bkeep += partial), 2 last (write the sum)
// PART: bit 0 weights, bit 1 activations + scales
__device__ __forceinline__ void stage_store(const Stage<BN>& S, unsigned char* buf, float invs, float2& bkeep,
                                            int ph, int tid) {
    using C = Cfg<BN>;
    // ---- weights
    if (!(PART & 1)) {
    } else if (MODE & 1) {  // timing only: no conversion (raw codes into both units)
        const int row = tid >> 1, blk = tid & 1;
        unsigned char* wr = buf + C::O_W + row * 64;
        *(int4*)(wr + swz(row, blk * 2) * 16) = S.wq;
        *(int4*)(wr + swz(row, blk * 2 + 1) * 16) = S.wq;
    } else {
        const uint32_t q[4] = {(uint32_t)S.wq.x, (uint32_t)S.wq.y, (uint32_t)S.wq.z, (uint32_t)S.wq.w};
        uint32_t lo[4], hi[4];
        const __half2 k1536 = __float2half2_rn(1536.f);
        if (FMT == gemv::FAST_P4) {
            const float av = h2f(S.s0) * invs;
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
            if (FMT == gemv::FAST_P4M) { A = h2f(S.s0); B = h2f(S.s1); }
            else {
                A = h2f(S.s1) * (float)(S.s0 & 0xff);
                B = -h2f(S.s1 >> 16) * (float)((S.s0 >> 8) & 0xff);
            }
            const __half2 a = __float2half2_rn(A * invs), a16 = __float2half2_rn(A * invs * 0.0625f);
            const __half2 b = __float2half2_rn(B * invs);
            const __half2 k1024 = __float2half2_rn(1024.f);
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const uint32_t x = q[w], x8 = x >> 8;
                uint32_t m0 = 0, m1 = 0, m2 = 0, m3 = 0;
                if (FMT == gemv::FAST_K5) {
                    const uint32_t H = S.hb >> w;
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
        const int row = tid >> 1, blk = tid & 1;
        unsigned char* wr = buf + C::O_W + row * 64;
        *(int4*)(wr + swz(row, blk * 2) * 16) = make_int4(lo[0], lo[1], lo[2], lo[3]);
        *(int4*)(wr + swz(row, blk * 2 + 1) * 16) = make_int4(hi[0], hi[1], hi[2], hi[3]);
    }
    // ---- activations
    if (!(PART & 2)) return;
#pragma unroll
    for (int i = 0; i < C::NXA; ++i) {
        const int U = tid + i * NT, tok = U >> 2, u = U & 3;
        *(int4*)(buf + C::O_X + tok * 64 + swz(tok, u) * 16) = S.xa[i];
    }
    if (tid < BN / 2) {
        float2* dxs = (float2*)(buf + C::O_DX);
        dxs[tid] = S.dx0;
        dxs[BN / 2 + tid] = S.dx1;
        const float2 pb = make_float2(MAGIC_F * (S.dx0.x + S.dx1.x), MAGIC_F * (S.dx0.y + S.dx1.y));
        if (ph == 2) ((float2*)(buf + C::O_BI))[tid] = make_float2(bkeep.x + pb.x, bkeep.y + pb.y);
        else if (ph == 1) bkeep = make_float2(bkeep.x + pb.x, bkeep.y + pb.y);
        else bkeep = pb;
    }
}

// MODE (timing-only ablations, bench only): bit 0 no weight conversion, bit 1 no FFMA epilogue (int32 accumulate),
// bit 2 no global loads, bit 3 no smem stores / barriers (pure ldmatrix + mma loop).
// MODE bit 4 (real math): batched issue order -- per pair of token groups, the 16 first-half mmas, then the 16
// dependent second-half mmas, then the 32 FFMAs, so no instruction waits on the one just before it.
template <int FMT, int RPL, int BN, int MODE = 0>
__global__ void __launch_bounds__(NT, 1) gemm8_kernel(const Args a) {
    using C = Cfg<BN>;
    constexpr int NG = C::NG, WN = C::WN;
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp & 1, wn = warp >> 1;
    const int tok0 = blockIdx.x * BN, row0 = blockIdx.y * BM;
    const int nkb = a.K >> 5;
    const int t4 = lane & 3;
    const float invs_st = a.invs[row0 + (tid >> 1)];

    float acc[8][NG][2];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int g = 0; g < NG; ++g) acc[i][g][0] = acc[i][g][1] = 0.f;

    Stage<BN> S;
    float2 bkeep = make_float2(0.f, 0.f);
    stage_load<FMT, RPL, BN>(S, a, row0, tok0, 0, tid);
    stage_store<FMT, BN>(S, smem, invs_st, bkeep, 0, tid);
    __syncthreads();

    int buf = 0;
    for (int kb0 = 0; kb0 < nkb; kb0 += 2) {
        const bool more = kb0 + 2 < nkb;
        if (more && !(MODE & 4)) stage_load<FMT, RPL, BN>(S, a, row0, tok0, kb0 + 2, tid);
        const unsigned char* B = smem + buf * C::BYTES;
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
            uint32_t af[2][8], bf[2][NG];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int u = kb * 2 + h;
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const int row = wm * 64 + (q * 4 + (lane >> 3)) * 8 + (lane & 7);
                    ldsm_x4(af[h][q * 4], af[h][q * 4 + 1], af[h][q * 4 + 2], af[h][q * 4 + 3],
                            B + C::O_W + row * 64 + swz(row, u) * 16);
                }
#pragma unroll
                for (int q = 0; q < NG / 4; ++q) {
                    const int tok = wn * WN + (q * 4 + (lane >> 3)) * 8 + (lane & 7);
                    ldsm_x4(bf[h][q * 4], bf[h][q * 4 + 1], bf[h][q * 4 + 2], bf[h][q * 4 + 3],
                            B + C::O_X + tok * 64 + swz(tok, u) * 16);
                }
            }
            if (MODE & 16) {
#pragma unroll
                for (int gp = 0; gp < NG / 2; ++gp) {
                    int tq[2][8][2];
#pragma unroll
                    for (int h = 0; h < 2; ++h)
#pragma unroll
                        for (int gg = 0; gg < 2; ++gg)
#pragma unroll
                            for (int i = 0; i < 8; ++i)
                                mma_s8p(tq[gg][i][0], tq[gg][i][1], af[h][i], bf[h][gp * 2 + gg], h ? tq[gg][i][0] : MAGIC_I,
                                        h ? tq[gg][i][1] : MAGIC_I);
#pragma unroll
                    for (int gg = 0; gg < 2; ++gg) {
                        const int g = gp * 2 + gg;
                        const float2 dv = *(const float2*)(B + C::O_DX + (kb * BN + wn * WN + 8 * g + 2 * t4) * 4);
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            acc[i][g][0] = fmaf(__int_as_float(tq[gg][i][0]), dv.x, acc[i][g][0]);
                            acc[i][g][1] = fmaf(__int_as_float(tq[gg][i][1]), dv.y, acc[i][g][1]);
                        }
                    }
                }
                continue;
            }
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                const float2 dv = *(const float2*)(B + C::O_DX + (kb * BN + wn * WN + 8 * g + 2 * t4) * 4);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int d0, d1;
                    if (MODE & 2) {
                        mma_s8(d0, d1, af[0][i], bf[0][g], __float_as_int(acc[i][g][0]), __float_as_int(acc[i][g][1]));
                        mma_s8(d0, d1, af[1][i], bf[1][g], d0, d1);
                        acc[i][g][0] = __int_as_float(d0); acc[i][g][1] = __int_as_float(d1);
                        continue;
                    }
                    mma_s8(d0, d1, af[0][i], bf[0][g], MAGIC_I, MAGIC_I);
                    mma_s8(d0, d1, af[1][i], bf[1][g], d0, d1);
                    acc[i][g][0] = fmaf(__int_as_float(d0), dv.x, acc[i][g][0]);
                    acc[i][g][1] = fmaf(__int_as_float(d1), dv.y, acc[i][g][1]);
                }
            }
        }
        if ((kb0 & 2) && !(MODE & 2)) {  // odd stage: end of a 4-block group
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                const float2 bv = *(const float2*)(B + C::O_BI + (wn * WN + 8 * g + 2 * t4) * 4);
#pragma unroll
                for (int i = 0; i < 8; ++i) { acc[i][g][0] -= bv.x; acc[i][g][1] -= bv.y; }
            }
        }
        if (more && !(MODE & 8)) {
            stage_store<FMT, BN, MODE>(S, smem + (buf ^ 1) * C::BYTES, invs_st, bkeep, ((kb0 + 2) & 2) ? 2 : 0, tid);
            __syncthreads();
            buf ^= 1;
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = row0 + wm * 64 + i * 8 + (lane >> 2);
        const float sr = 1.f / a.invs[row];
#pragma unroll
        for (int g = 0; g < NG; ++g)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int tok = tok0 + wn * WN + 8 * g + 2 * t4 + e;
                if (tok < a.T) {
                    float* p = a.y + (size_t)tok * a.ldy + row;
                    const float v = acc[i][g][e] * sr;
                    *p = a.accumulate ? *p + v : v;
                }
            }
    }
}

// ================================================================================================ gemm9 (round 2, v2)
// Same math as gemm8_kernel, rebuilt for instruction economy (t4q-p v9 ablation: a pure ldmatrix + mma loop sustains
// 90 TOPS at the cap, the gemm8 loop with ~1100 instructions per 256 mmas only 27):
//   * per-thread weight / activation / scale pointers computed once (no per-stage index math);
//   * GA: activation scale group: 32 (exact q8 blocks, 1 FFMA per output per 32), 64 (1 FFMA per output per 64;
//     |S| <= 64*127*127 < 2^22 keeps the magic exact), or 0 = one scale per token (int32 accumulation over all of K,
//     no FFMA in the loop; |S| <= 8704*127^2 < 2^31);
//   * per 8-token group: one ldmatrix.x4 for B, then the mmas in dependency-free order, then the FFMAs.
// AB (timing-only ablations, bench): bit 0 no weight conversion, bit 1 no FFMA (int32 accumulate), bit 2 loads cycle
// over the first two stages (L1/L2-hot), bit 3 no global loads (staging registers perturbed per stage), bit 4 no
// smem stores / barriers.
template <int FMT, int RPL>
struct WPtr {
    const uint8_t* code;  // this thread's (row, blk) 16-B code group at kb0 = 0
    const uint8_t* sa;    // P4 / P4M: fp16 d; K5: {sc, mn} bytes
    const uint8_t* sb;    // P4M: fp16 m; K5: {d, dmin} (per 256)
    const uint8_t* hq;    // K5: high bits
    long long cs_code, cs_s, cs_d, cs_h;  // bytes per 512-chunk
    __device__ __forceinline__ WPtr(const Args& a, int R, int blk) {
        const int lt = R / (2 * RPL), w = R % (2 * RPL), h = w / RPL, r = w % RPL;
        const int lane0 = h * 16 + blk;
        const long long dtc = a.cm ? a.ntiles : 1;
        const long long tc0 = (long long)gemv::tc_index(lt, 0, a.K >> 9, a.ntiles, a.cm);
        code = a.codes + (tc0 * RPL + r) * 512 + lane0 * 16;
        cs_code = dtc * RPL * 512;
        if (FMT == gemv::FAST_K5) {
            sa = a.sc + ((tc0 * 32 + lane0) * RPL + r) * 2;
            sb = (const uint8_t*)a.d + (((tc0 * 2 + h) * 2) * RPL + r) * 4;
            hq = a.qh + (tc0 * RPL + r) * 128 + lane0 * 4;
            cs_d = dtc * 4 * RPL * 4;
            cs_h = dtc * RPL * 128;
        } else {
            sa = (const uint8_t*)(a.d + (tc0 * 32 + lane0) * RPL + r);
            sb = FMT == gemv::FAST_P4M ? (const uint8_t*)((const uint16_t*)a.sc + (tc0 * 32 + lane0) * RPL + r) : nullptr;
            hq = nullptr;
            cs_d = 0; cs_h = 0;
        }
        cs_s = dtc * 32 * RPL * 2;
    }
};

template <int FMT, int RPL, int BN, int GA, int AB>
__device__ __forceinline__ void stage_load9(Stage<BN>& S, const WPtr<FMT, RPL>& P, const int8_t* xb, long long xstep,
                                            const float* dxb, long long tps, int kb0, int tid) {
    if (AB & 8) {  // no loads: perturb the staged values so the data still changes per stage
        S.wq.x ^= kb0 * 0x01010101; S.wq.y += kb0; S.wq.z ^= kb0 << 3; S.wq.w += 7 * kb0;
#pragma unroll
        for (int i = 0; i < Cfg<BN>::NXA; ++i) { S.xa[i].x ^= kb0 * 0x01030507; S.xa[i].y += kb0; S.xa[i].z ^= kb0 << 9; S.xa[i].w += 3 * kb0; }
        return;
    }
    if (AB & 4) kb0 &= 2;
    const int c = kb0 >> 4, jj = kb0 & 15;
    S.wq = gemv::ldg_nc_v4(P.code + c * P.cs_code + jj * 16);
    if (FMT == gemv::FAST_K5) {
        S.s0 = (uint32_t)__ldg((const unsigned short*)(P.sa + c * P.cs_s + jj * RPL * 2));
        S.s1 = __ldg((const unsigned int*)(P.sb + c * P.cs_d + (jj >> 3) * RPL * 4));
        S.hb = __ldg((const unsigned int*)(P.hq + c * P.cs_h + jj * 4));
    } else {
        S.s0 = (uint32_t)__ldg((const unsigned short*)(P.sa + c * P.cs_s + jj * RPL * 2));
        if (FMT == gemv::FAST_P4M) S.s1 = (uint32_t)__ldg((const unsigned short*)(P.sb + c * P.cs_s + jj * RPL * 2));
    }
    const int8_t* xp = xb + kb0 * 32;
#pragma unroll
    for (int i = 0; i < Cfg<BN>::NXA; ++i) S.xa[i] = __ldg((const int4*)(xp + i * xstep));
    if (GA != 0 && tid < BN / 2) {
        if (GA == 32) {
            S.dx0 = __ldg((const float2*)(dxb + kb0 * tps));
            S.dx1 = __ldg((const float2*)(dxb + (kb0 + 1) * tps));
        } else {
            S.dx0 = __ldg((const float2*)(dxb + (kb0 >> 1) * tps));
            S.dx1 = make_float2(0.f, 0.f);
        }
    }
}

template <int FMT, int RPL, int BN, int GA, int AB = 0>
__global__ void __launch_bounds__(NT, 1) gemm9_kernel(const Args a) {
    using C = Cfg<BN>;
    constexpr int NG = C::NG, WN = C::WN;
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp & 1, wn = warp >> 1;
    const int tok0 = blockIdx.x * BN, row0 = blockIdx.y * BM;
    const int nkb = a.K >> 5;
    const int t4 = lane & 3;
    const float invs_st = a.invs[row0 + (tid >> 1)];
    const WPtr<FMT, RPL> P(a, row0 + (tid >> 1), tid & 1);
    const int8_t* xb = a.xq + (size_t)(tok0 + (tid >> 2)) * a.K + (tid & 3) * 16;
    const long long xstep = (long long)(NT / 4) * a.K;
    const float* dxb = a.dx + tok0 + 2 * (tid & (BN / 2 - 1));
    const long long tps = a.Tp;

    float acc[8][NG][2];  // GA 0: int32 bits
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int g = 0; g < NG; ++g) acc[i][g][0] = acc[i][g][1] = 0.f;

    Stage<BN> S;
    S.dx0 = S.dx1 = make_float2(0.f, 0.f);
    float2 bkeep = make_float2(0.f, 0.f);
    stage_load9<FMT, RPL, BN, GA, AB & 4>(S, P, xb, xstep, dxb, tps, 0, tid);
    stage_store<FMT, BN, AB & 1>(S, smem, invs_st, bkeep, 0, tid);
    __syncthreads();

    int arow[2], brow[NG];
#pragma unroll
    for (int q = 0; q < 2; ++q) arow[q] = wm * 64 + (q * 4 + (lane >> 3)) * 8 + (lane & 7);
#pragma unroll
    for (int g = 0; g < NG; ++g) brow[g] = wn * WN + 8 * g + (lane & 7);

    int buf = 0;
    for (int kb0 = 0; kb0 < nkb; kb0 += 2) {
        const bool more = kb0 + 2 < nkb;
        if (more) stage_load9<FMT, RPL, BN, GA, AB>(S, P, xb, xstep, dxb, tps, kb0 + 2, tid);
        const unsigned char* B = smem + ((AB & 16) ? 0 : buf) * C::BYTES;
        if (GA == 64 || GA == 0) {
            uint32_t af[4][8];
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int q = 0; q < 2; ++q)
                    ldsm_x4(af[u][q * 4], af[u][q * 4 + 1], af[u][q * 4 + 2], af[u][q * 4 + 3],
                            B + C::O_W + arow[q] * 64 + swz(arow[q], u) * 16);
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                if ((AB & 32) && more) {  // interleave the next stage's smem stores with this stage's mmas
                    const int sp = ((kb0 + 2) >> 1) & 3;
                    if (g == NG / 4)
                        stage_store<FMT, BN, AB & 1, 1>(S, smem + (buf ^ 1) * C::BYTES, invs_st, bkeep, 0, tid);
                    if (g == NG / 2)
                        stage_store<FMT, BN, AB & 1, 2>(S, smem + (buf ^ 1) * C::BYTES, invs_st, bkeep,
                                                        sp == 0 ? 0 : sp == 3 ? 2 : 1, tid);
                }
                uint32_t b[4];
                ldsm_x4(b[0], b[1], b[2], b[3], B + C::O_X + brow[g] * 64 + swz(brow[g], lane >> 3) * 16);
                if (GA == 0 || (AB & 2)) {
#pragma unroll
                    for (int u = 0; u < 4; ++u)
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            int d0, d1;
                            mma_s8p(d0, d1, af[u][i], b[u], __float_as_int(acc[i][g][0]), __float_as_int(acc[i][g][1]));
                            acc[i][g][0] = __int_as_float(d0); acc[i][g][1] = __int_as_float(d1);
                        }
                    continue;
                }
                int tq[8][2];
#pragma unroll
                for (int u = 0; u < 4; ++u)
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                        mma_s8p(tq[i][0], tq[i][1], af[u][i], b[u], u ? tq[i][0] : MAGIC_I, u ? tq[i][1] : MAGIC_I);
                const float2 dv = *(const float2*)(B + C::O_DX + (wn * WN + 8 * g + 2 * t4) * 4);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[i][g][0] = fmaf(__int_as_float(tq[i][0]), dv.x, acc[i][g][0]);
                    acc[i][g][1] = fmaf(__int_as_float(tq[i][1]), dv.y, acc[i][g][1]);
                }
            }
        } else {
#pragma unroll
            for (int kb = 0; kb < 2; ++kb) {
                uint32_t af[2][8];
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int q = 0; q < 2; ++q)
                        ldsm_x4(af[h][q * 4], af[h][q * 4 + 1], af[h][q * 4 + 2], af[h][q * 4 + 3],
                                B + C::O_W + arow[q] * 64 + swz(arow[q], kb * 2 + h) * 16);
#pragma unroll
                for (int gp = 0; gp < NG / 2; ++gp) {
                    uint32_t b[4];
                    {
                        const int m = lane >> 3, row = brow[2 * gp + (m & 1)];
                        ldsm_x4(b[0], b[1], b[2], b[3], B + C::O_X + row * 64 + swz(row, kb * 2 + (m >> 1)) * 16);
                    }
#pragma unroll
                    for (int gg = 0; gg < 2; ++gg) {
                        const int g = 2 * gp + gg;
                        if (AB & 2) {
#pragma unroll
                            for (int h = 0; h < 2; ++h)
#pragma unroll
                                for (int i = 0; i < 8; ++i) {
                                    int d0, d1;
                                    mma_s8p(d0, d1, af[h][i], b[h * 2 + gg], __float_as_int(acc[i][g][0]),
                                            __float_as_int(acc[i][g][1]));
                                    acc[i][g][0] = __int_as_float(d0); acc[i][g][1] = __int_as_float(d1);
                                }
                            continue;
                        }
                        int tq[8][2];
#pragma unroll
                        for (int h = 0; h < 2; ++h)
#pragma unroll
                            for (int i = 0; i < 8; ++i)
                                mma_s8p(tq[i][0], tq[i][1], af[h][i], b[h * 2 + gg], h ? tq[i][0] : MAGIC_I,
                                        h ? tq[i][1] : MAGIC_I);
                        const float2 dv = *(const float2*)(B + C::O_DX + (kb * BN + wn * WN + 8 * g + 2 * t4) * 4);
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            acc[i][g][0] = fmaf(__int_as_float(tq[i][0]), dv.x, acc[i][g][0]);
                            acc[i][g][1] = fmaf(__int_as_float(tq[i][1]), dv.y, acc[i][g][1]);
                        }
                    }
                }
            }
        }
        if (GA != 0 && !(AB & 2) && (kb0 & 6) == 6) {  // end of a 256-k bias group (every 4th stage)
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                const float2 bv = *(const float2*)(B + C::O_BI + (wn * WN + 8 * g + 2 * t4) * 4);
#pragma unroll
                for (int i = 0; i < 8; ++i) { acc[i][g][0] -= bv.x; acc[i][g][1] -= bv.y; }
            }
        }
        if (more && !(AB & 16)) {
            const int sp = ((kb0 + 2) >> 1) & 3;
            if (!((AB & 32) && (GA == 64 || GA == 0)))
                stage_store<FMT, BN, AB & 1>(S, smem + (buf ^ 1) * C::BYTES, invs_st, bkeep, sp == 0 ? 0 : sp == 3 ? 2 : 1, tid);
            __syncthreads();
            buf ^= 1;
        }
    }
    // GA 0: per-token scale dx[t] (dx is [Tp])
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = row0 + wm * 64 + i * 8 + (lane >> 2);
        const float sr = __frcp_rn(a.invs[row]);
#pragma unroll
        for (int g = 0; g < NG; ++g)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int tok = tok0 + wn * WN + 8 * g + 2 * t4 + e;
                if (tok < a.T) {
                    float* p = a.y + (size_t)tok * a.ldy + row;
                    const float v = GA == 0 ? (float)__float_as_int(acc[i][g][e]) * (a.dx[tok] * sr) : acc[i][g][e] * sr;
                    *p = a.accumulate ? *p + v : v;
                }
            }
    }
}

template <int FMT, int RPL, int BN, int GA, int AB = 0>
static cudaError_t launch9_t(const Args& a, cudaStream_t s) {
    auto k = gemm9_kernel<FMT, RPL, BN, GA, AB>;
    const int smem = 2 * Cfg<BN>::BYTES;
    static int attr_dev_mask = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    if (!(attr_dev_mask & (1 << dev))) {
        cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return e;
        attr_dev_mask |= 1 << dev;
    }
    if (a.N % BM || a.K % 256 || a.Tp % BN) return cudaErrorInvalidValue;
    dim3 grid(a.Tp / BN, a.N / BM);
    k<<<grid, NT, smem, s>>>(a);
    return cudaGetLastError();
}

// ga: activation group (32 exact q8 blocks, 64, or 0 = per token); bn 256 / 128
static inline cudaError_t launch9(int fmt, int rpl, int bn, int ga, const Args& a, cudaStream_t s) {
#define T4Q_G9(F, R)                                                                                     \
    if (fmt == F && rpl == R) {                                                                          \
        if (ga == 64) return bn == 256 ? launch9_t<F, R, 256, 64>(a, s) : launch9_t<F, R, 128, 64>(a, s); \
        if (ga == 0) return bn == 256 ? launch9_t<F, R, 256, 0>(a, s) : launch9_t<F, R, 128, 0>(a, s);    \
        return bn == 256 ? launch9_t<F, R, 256, 32>(a, s) : launch9_t<F, R, 128, 32>(a, s);               \
    }
    T4Q_G9(gemv::FAST_P4, 4) T4Q_G9(gemv::FAST_P4, 2)
    T4Q_G9(gemv::FAST_P4M, 4) T4Q_G9(gemv::FAST_P4M, 2)
    T4Q_G9(gemv::FAST_K5, 4) T4Q_G9(gemv::FAST_K5, 2)
#undef T4Q_G9
    return cudaErrorInvalidValue;
}

// ================================================================================================ gemm10
// CUTLASS-style software pipeline (MmaPipelined) on a 128-row x 128-token block, 8 warps of 64 x 32: per k16 step the
// fragments of the next step are loaded while this step's 32 mmas issue; the next stage's smem stores and the
// barrier sit just before the last k16 step's mmas (its fragments are already in registers), so the barrier wait and
// the first ldmatrix of the new stage overlap mma work. GA 64 or 128 (activation scale group); int32 chains per group
// in registers (64 + 64 accumulators), bias every 4 stages.
template <int FMT, int RPL, int GA, int AB = 0>
__global__ void __launch_bounds__(NT, 1) gemm10_kernel(const Args a) {
    constexpr int BN = 128;
    using C = Cfg<BN>;
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp & 1, wn = warp >> 1;
    const int tok0 = blockIdx.x * BN, row0 = blockIdx.y * BM;
    const int nst = a.K >> 6;
    const int t4 = lane & 3;
    const float invs_st = a.invs[row0 + (tid >> 1)];
    const WPtr<FMT, RPL> P(a, row0 + (tid >> 1), tid & 1);
    const int8_t* xb = a.xq + (size_t)(tok0 + (tid >> 2)) * a.K + (tid & 3) * 16;
    const long long xstep = (long long)(NT / 4) * a.K;
    const float* dxb = a.dx + tok0 + 2 * (tid & (BN / 2 - 1));
    const long long tps = a.Tp;

    float acc[8][4][2];
    int tq[8][4][2];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int g = 0; g < 4; ++g) { acc[i][g][0] = acc[i][g][1] = 0.f; tq[i][g][0] = tq[i][g][1] = 0; }

    Stage<BN> S;
    float2 bkeep = make_float2(0.f, 0.f);
    auto load = [&](int st) {  // stage st -> S (dx0 = this stage's group scale, dx1 = 0)
        const int kb0 = st * 2;
        stage_load9<FMT, RPL, BN, 0, AB & 12>(S, P, xb, xstep, dxb, tps, kb0, tid);
        if (tid < BN / 2) {
            S.dx0 = __ldg((const float2*)(dxb + (long long)(GA == 64 ? st : st >> 1) * tps));
            S.dx1 = make_float2(0.f, 0.f);
        }
    };
    auto store = [&](int st, int b) {
        Stage<BN> T = S;
        if (GA == 128 && (st & 1)) T.dx1 = make_float2(-T.dx0.x, -T.dx0.y);  // odd stage: no new magic -> no bias
        const int sp = st & 3;
        stage_store<FMT, BN, AB & 1>(T, smem + b * C::BYTES, invs_st, bkeep, sp == 0 ? 0 : sp == 3 ? 2 : 1, tid);
    };
    int arow[2];
#pragma unroll
    for (int q = 0; q < 2; ++q) arow[q] = wm * 64 + (q * 4 + (lane >> 3)) * 8 + (lane & 7);
    const int brow = wn * 32 + 8 * (lane >> 3) + (lane & 7);
    uint32_t fa[2][8], fb[2][4];
    auto frag = [&](int slot, int b, int u) {
        const unsigned char* Bp = smem + b * C::BYTES;
#pragma unroll
        for (int q = 0; q < 2; ++q)
            ldsm_x4(fa[slot][q * 4], fa[slot][q * 4 + 1], fa[slot][q * 4 + 2], fa[slot][q * 4 + 3],
                    Bp + C::O_W + arow[q] * 64 + swz(arow[q], u) * 16);
        ldsm_x4(fb[slot][0], fb[slot][1], fb[slot][2], fb[slot][3], Bp + C::O_X + brow * 64 + swz(brow, u) * 16);
    };

    load(0);
    store(0, 0);
    __syncthreads();
    if (nst > 1) load(1);
    frag(0, 0, 0);
    int buf = 0;
    float2 dxr[4], bvr[4];
    for (int s = 0; s < nst; ++s) {
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            if (u == 3) {
                const unsigned char* Bp = smem + buf * C::BYTES;
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    dxr[g] = *(const float2*)(Bp + C::O_DX + (wn * 32 + 8 * g + 2 * t4) * 4);
                    if ((s & 3) == 3) bvr[g] = *(const float2*)(Bp + C::O_BI + (wn * 32 + 8 * g + 2 * t4) * 4);
                }
                if (s + 1 < nst) store(s + 1, buf ^ 1);
                __syncthreads();
                buf ^= 1;
                if (s + 2 < nst) load(s + 2);
            }
            if (!(u == 3 && s + 1 == nst)) frag((u + 1) & 1, buf, (u + 1) & 3);
            const bool first = u == 0 && (GA == 64 || (s & 1) == 0);
#pragma unroll
            for (int g = 0; g < 4; ++g)
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    mma_s8p(tq[i][g][0], tq[i][g][1], fa[u & 1][i], fb[u & 1][g], first ? MAGIC_I : tq[i][g][0],
                            first ? MAGIC_I : tq[i][g][1]);
            if (u == 3 && (GA == 64 || (s & 1))) {
#pragma unroll
                for (int g = 0; g < 4; ++g)
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        acc[i][g][0] = fmaf(__int_as_float(tq[i][g][0]), dxr[g].x, acc[i][g][0]);
                        acc[i][g][1] = fmaf(__int_as_float(tq[i][g][1]), dxr[g].y, acc[i][g][1]);
                    }
            }
        }
        if ((s & 3) == 3) {
#pragma unroll
            for (int g = 0; g < 4; ++g)
#pragma unroll
                for (int i = 0; i < 8; ++i) { acc[i][g][0] -= bvr[g].x; acc[i][g][1] -= bvr[g].y; }
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = row0 + wm * 64 + i * 8 + (lane >> 2);
        const float sr = __frcp_rn(a.invs[row]);
#pragma unroll
        for (int g = 0; g < 4; ++g)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int tok = tok0 + wn * 32 + 8 * g + 2 * t4 + e;
                if (tok < a.T) {
                    float* p = a.y + (size_t)tok * a.ldy + row;
                    const float v = acc[i][g][e] * sr;
                    *p = a.accumulate ? *p + v : v;
                }
            }
    }
}

template <int FMT, int RPL, int GA, int AB = 0>
static cudaError_t launch10_t(const Args& a, cudaStream_t s) {
    auto k = gemm10_kernel<FMT, RPL, GA, AB>;
    const int smem = 2 * Cfg<128>::BYTES;
    static int attr_dev_mask = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    if (!(attr_dev_mask & (1 << dev))) {
        cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return e;
        attr_dev_mask |= 1 << dev;
    }
    if (a.N % BM || a.K % 256 || a.Tp % 128) return cudaErrorInvalidValue;
    dim3 grid(a.Tp / 128, a.N / BM);
    k<<<grid, NT, smem, s>>>(a);
    return cudaGetLastError();
}

static inline cudaError_t launch10(int fmt, int rpl, int ga, const Args& a, cudaStream_t s) {
#define T4Q_G10(F, R) \
    if (fmt == F && rpl == R) return ga == 128 ? launch10_t<F, R, 128>(a, s) : launch10_t<F, R, 64>(a, s);
    T4Q_G10(gemv::FAST_P4, 4) T4Q_G10(gemv::FAST_P4, 2)
    T4Q_G10(gemv::FAST_P4M, 4) T4Q_G10(gemv::FAST_P4M, 2)
    T4Q_G10(gemv::FAST_K5, 4) T4Q_G10(gemv::FAST_K5, 2)
#undef T4Q_G10
    return cudaErrorInvalidValue;
}

// ================================================================================================ gemm11
// gemm9's GA 64 loop on a 256-row x 128-token block (8 warps as 4 rows x 2 tokens, 64 x 64 warp tiles). Per stage the
// block moves 256 x 36 B of weights (Q4 + scales) + 128 x 64 B of activations instead of 128 x 36 + 256 x 64: 21% fewer
// L2 -> smem bytes per mma, which is what the 70 W cap charges for (t4q-p v11 ablation).
template <int BMX, int BN>
struct Cfg2 {
    static constexpr int WMW = BMX / 64, WNW = 8 / WMW;  // warps along rows / tokens
    static constexpr int WN = BN / WNW, NG = WN / 8;
    static constexpr int O_W = 0, O_X = BMX * 64, O_DX = O_X + BN * 64, O_BI = O_DX + BN * 4;
    static constexpr int BYTES = O_BI + BN * 4;
    static constexpr int NW = BMX / 128;      // (row, blk) weight units per thread
    static constexpr int NXA = BN * 4 / NT;   // 16-B activation units per thread
};

template <int FMT, int RPL, int BMX, int BN>
__global__ void __launch_bounds__(NT, 1) gemm11_kernel(const Args a) {
    using C = Cfg2<BMX, BN>;
    constexpr int NG = C::NG, WN = C::WN, NW = C::NW, NXA = C::NXA;
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp % C::WMW, wn = warp / C::WMW;
    const int tok0 = blockIdx.x * BN, row0 = blockIdx.y * BMX;
    const int nkb = a.K >> 5;
    const int t4 = lane & 3;
    float invs_st[NW];
    const WPtr<FMT, RPL>* Pp;
    WPtr<FMT, RPL> P0(a, row0 + (tid >> 1), tid & 1);
    WPtr<FMT, RPL> P1(a, row0 + (tid >> 1) + (NW > 1 ? 128 : 0), tid & 1);
    (void)Pp;
#pragma unroll
    for (int w = 0; w < NW; ++w) invs_st[w] = a.invs[row0 + (tid >> 1) + 128 * w];
    const int8_t* xb = a.xq + (size_t)(tok0 + (tid >> 2)) * a.K + (tid & 3) * 16;
    const long long xstep = (long long)(NT / 4) * a.K;
    const float* dxb = a.dx + tok0 + 2 * (tid & (BN / 2 - 1));
    const long long tps = a.Tp;

    float acc[8][NG][2];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int g = 0; g < NG; ++g) acc[i][g][0] = acc[i][g][1] = 0.f;

    int4 wq[NW], xa[NXA];
    uint32_t s0[NW], s1[NW], hb[NW];
    float2 dx0 = make_float2(0.f, 0.f), bkeep = make_float2(0.f, 0.f);
    auto load = [&](int kb0) {
        const int c = kb0 >> 4, jj = kb0 & 15;
#pragma unroll
        for (int w = 0; w < NW; ++w) {
            const WPtr<FMT, RPL>& P = w ? P1 : P0;
            wq[w] = gemv::ldg_nc_v4(P.code + c * P.cs_code + jj * 16);
            s0[w] = (uint32_t)__ldg((const unsigned short*)(P.sa + c * P.cs_s + jj * RPL * 2));
            if (FMT == gemv::FAST_K5) {
                s1[w] = __ldg((const unsigned int*)(P.sb + c * P.cs_d + (jj >> 3) * RPL * 4));
                hb[w] = __ldg((const unsigned int*)(P.hq + c * P.cs_h + jj * 4));
            } else if (FMT == gemv::FAST_P4M) {
                s1[w] = (uint32_t)__ldg((const unsigned short*)(P.sb + c * P.cs_s + jj * RPL * 2));
            }
        }
        const int8_t* xp = xb + kb0 * 32;
#pragma unroll
        for (int i = 0; i < NXA; ++i) xa[i] = __ldg((const int4*)(xp + i * xstep));
        if (tid < BN / 2) dx0 = __ldg((const float2*)(dxb + (kb0 >> 1) * tps));
    };
    auto store = [&](unsigned char* buf, int ph) {
#pragma unroll
        for (int w = 0; w < NW; ++w) {
            Stage<256> T;  // reuse gemm8's weight conversion (weights part only)
            T.wq = wq[w]; T.s0 = s0[w]; T.s1 = s1[w]; T.hb = hb[w];
            stage_store<FMT, 256, 0, 1>(T, buf + C::O_W + w * 128 * 64 - 0, invs_st[w], bkeep, 0, tid);
        }
#pragma unroll
        for (int i = 0; i < NXA; ++i) {
            const int U = tid + i * NT, tok = U >> 2, u = U & 3;
            *(int4*)(buf + C::O_X + tok * 64 + swz(tok, u) * 16) = xa[i];
        }
        if (tid < BN / 2) {
            ((float2*)(buf + C::O_DX))[tid] = dx0;
            const float2 pb = make_float2(MAGIC_F * dx0.x, MAGIC_F * dx0.y);
            if (ph == 2) ((float2*)(buf + C::O_BI))[tid] = make_float2(bkeep.x + pb.x, bkeep.y + pb.y);
            else if (ph == 1) bkeep = make_float2(bkeep.x + pb.x, bkeep.y + pb.y);
            else bkeep = pb;
        }
    };
    load(0);
    store(smem, 0);
    __syncthreads();

    int arow[2], brow[NG];
#pragma unroll
    for (int q = 0; q < 2; ++q) arow[q] = wm * 64 + (q * 4 + (lane >> 3)) * 8 + (lane & 7);
#pragma unroll
    for (int g = 0; g < NG; ++g) brow[g] = wn * WN + 8 * g + (lane & 7);

    int buf = 0;
    for (int kb0 = 0; kb0 < nkb; kb0 += 2) {
        const bool more = kb0 + 2 < nkb;
        if (more) load(kb0 + 2);
        const unsigned char* B = smem + buf * C::BYTES;
        uint32_t af[4][8];
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int q = 0; q < 2; ++q)
                ldsm_x4(af[u][q * 4], af[u][q * 4 + 1], af[u][q * 4 + 2], af[u][q * 4 + 3],
                        B + C::O_W + arow[q] * 64 + swz(arow[q], u) * 16);
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            uint32_t b[4];
            ldsm_x4(b[0], b[1], b[2], b[3], B + C::O_X + brow[g] * 64 + swz(brow[g], lane >> 3) * 16);
            int tq[8][2];
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    mma_s8p(tq[i][0], tq[i][1], af[u][i], b[u], u ? tq[i][0] : MAGIC_I, u ? tq[i][1] : MAGIC_I);
            const float2 dv = *(const float2*)(B + C::O_DX + (wn * WN + 8 * g + 2 * t4) * 4);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[i][g][0] = fmaf(__int_as_float(tq[i][0]), dv.x, acc[i][g][0]);
                acc[i][g][1] = fmaf(__int_as_float(tq[i][1]), dv.y, acc[i][g][1]);
            }
        }
        if ((kb0 & 6) == 6) {
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                const float2 bv = *(const float2*)(B + C::O_BI + (wn * WN + 8 * g + 2 * t4) * 4);
#pragma unroll
                for (int i = 0; i < 8; ++i) { acc[i][g][0] -= bv.x; acc[i][g][1] -= bv.y; }
            }
        }
        if (more) {
            const int sp = ((kb0 + 2) >> 1) & 3;
            store(smem + (buf ^ 1) * C::BYTES, sp == 0 ? 0 : sp == 3 ? 2 : 1);
            __syncthreads();
            buf ^= 1;
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = row0 + wm * 64 + i * 8 + (lane >> 2);
        const float sr = __frcp_rn(a.invs[row]);
#pragma unroll
        for (int g = 0; g < NG; ++g)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int tok = tok0 + wn * WN + 8 * g + 2 * t4 + e;
                if (tok < a.T) {
                    float* p = a.y + (size_t)tok * a.ldy + row;
                    const float v = acc[i][g][e] * sr;
                    *p = a.accumulate ? *p + v : v;
                }
            }
    }
}

template <int FMT, int RPL, int BMX, int BN>
static cudaError_t launch11_t(const Args& a, cudaStream_t s) {
    auto k = gemm11_kernel<FMT, RPL, BMX, BN>;
    const int smem = 2 * Cfg2<BMX, BN>::BYTES;
    static int attr_dev_mask = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    if (!(attr_dev_mask & (1 << dev))) {
        cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return e;
        attr_dev_mask |= 1 << dev;
    }
    if (a.N % BMX || a.K % 256 || a.Tp % BN) return cudaErrorInvalidValue;
    dim3 grid(a.Tp / BN, a.N / BMX);
    k<<<grid, NT, smem, s>>>(a);
    return cudaGetLastError();
}

// GA 64 only; bm 256 (bn 128) or 128 (bn 256)
static inline cudaError_t launch11(int fmt, int rpl, int bm, const Args& a, cudaStream_t s) {
#define T4Q_G11(F, R) \
    if (fmt == F && rpl == R) return bm == 256 ? launch11_t<F, R, 256, 128>(a, s) : launch11_t<F, R, 128, 256>(a, s);
    T4Q_G11(gemv::FAST_P4, 4) T4Q_G11(gemv::FAST_P4, 2)
    T4Q_G11(gemv::FAST_P4M, 4) T4Q_G11(gemv::FAST_P4M, 2)
    T4Q_G11(gemv::FAST_K5, 4) T4Q_G11(gemv::FAST_K5, 2)
#undef T4Q_G11
    return cudaErrorInvalidValue;
}

// ================================================================================================ gemm12
// Two co-resident blocks per SM (128 threads each, 4 warps of 64 x 64 on a 128 x 128 tile, ~18 KB smem): one block's
// smem-store / barrier phase overlaps the other block's mmas, which a single 8-warp block per SM cannot do. Stage =
// one 32-block (k32, 2 ldmatrix units per 32-B row), GA 32 (exact q8 blocks: one FFMA per output per 32).
constexpr int NT12 = 128;
struct Cfg12 {
    static constexpr int O_W = 0, O_X = 128 * 32, O_DX = O_X + 128 * 32, O_BI = O_DX + 128 * 4;
    static constexpr int BYTES = O_BI + 128 * 4;  // 9 KB per stage
};
__device__ __forceinline__ int swz32(int row, int u) { return u ^ ((row >> 2) & 1); }

template <int FMT, int RPL, int NSTG>
__global__ void __launch_bounds__(NT12, 2) gemm12_kernel(const Args a) {
    using C = Cfg12;
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp & 1, wn = warp >> 1;
    const int tok0 = blockIdx.x * 128, row0 = blockIdx.y * 128;
    const int nkb = a.K >> 5;
    const int t4 = lane & 3;
    // staging: weight unit = (row tid, this block); activations: units (tok = (tid + 128 i) >> 1, u = tid & 1), i < 2
    const float invs_st = a.invs[row0 + tid];
    const WPtr<FMT, RPL> P(a, row0 + tid, 0);
    const int8_t* xb = a.xq + (size_t)(tok0 + (tid >> 1)) * a.K + (tid & 1) * 16;
    const long long xstep = 64LL * a.K;
    const float* dxb = a.dx + tok0 + 2 * (tid & 63);
    const long long tps = a.Tp;

    float acc[8][8][2];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int g = 0; g < 8; ++g) acc[i][g][0] = acc[i][g][1] = 0.f;

    int4 wq, xa[2];
    uint32_t s0 = 0, s1 = 0, hb = 0;
    float2 dx0 = make_float2(0.f, 0.f), bkeep = make_float2(0.f, 0.f);
    auto load = [&](int kb) {
        const int c = kb >> 4, jj = kb & 15;
        wq = gemv::ldg_nc_v4(P.code + c * P.cs_code + jj * 16);
        s0 = (uint32_t)__ldg((const unsigned short*)(P.sa + c * P.cs_s + jj * RPL * 2));
        if (FMT == gemv::FAST_K5) {
            s1 = __ldg((const unsigned int*)(P.sb + c * P.cs_d + (jj >> 3) * RPL * 4));
            hb = __ldg((const unsigned int*)(P.hq + c * P.cs_h + jj * 4));
        } else if (FMT == gemv::FAST_P4M) {
            s1 = (uint32_t)__ldg((const unsigned short*)(P.sb + c * P.cs_s + jj * RPL * 2));
        }
        const int8_t* xp = xb + kb * 32;
        xa[0] = __ldg((const int4*)xp);
        xa[1] = __ldg((const int4*)(xp + xstep));
        if (tid < 64) dx0 = __ldg((const float2*)(dxb + kb * tps));
    };
    auto store = [&](unsigned char* buf, int ph) {
        // weights: convert 32 codes of (row tid) into units 0 (k 0..15) and 1 (k 16..31)
        {
            Stage<256> T;
            T.wq = wq; T.s0 = s0; T.s1 = s1; T.hb = hb;
            // reuse gemm8's conversion by storing into a scratch layout, then move: inline the conversion instead
            const uint32_t q[4] = {(uint32_t)wq.x, (uint32_t)wq.y, (uint32_t)wq.z, (uint32_t)wq.w};
            uint32_t lo[4], hi[4];
            const __half2 k1536 = __float2half2_rn(1536.f);
            if (FMT == gemv::FAST_P4) {
                const float av = h2f(s0) * invs_st;
                const __half2 r = __float2half2_rn(av), r16 = __float2half2_rn(av * 0.0625f);
                const __half2 k1032 = __float2half2_rn(1032.f), k1152 = __float2half2_rn(1152.f);
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const uint32_t x = q[w], x8 = x >> 8;
                    const __half2 v0 = as_h2((x & 0x000F000Fu) | 0x64006400u), v1 = as_h2((x8 & 0x000F000Fu) | 0x64006400u);
                    const __half2 v2 = as_h2((x & 0x00F000F0u) | 0x64006400u), v3 = as_h2((x8 & 0x00F000F0u) | 0x64006400u);
                    lo[w] = __byte_perm(h2_bits(__hfma2(__hsub2(v0, k1032), r, k1536)), h2_bits(__hfma2(__hsub2(v1, k1032), r, k1536)), 0x6240);
                    hi[w] = __byte_perm(h2_bits(__hfma2(__hsub2(v2, k1152), r16, k1536)), h2_bits(__hfma2(__hsub2(v3, k1152), r16, k1536)), 0x6240);
                }
            } else {
                float A, B;
                if (FMT == gemv::FAST_P4M) { A = h2f(s0); B = h2f(s1); }
                else { A = h2f(s1) * (float)(s0 & 0xff); B = -h2f(s1 >> 16) * (float)((s0 >> 8) & 0xff); }
                const __half2 ah = __float2half2_rn(A * invs_st), a16 = __float2half2_rn(A * invs_st * 0.0625f);
                const __half2 bh = __float2half2_rn(B * invs_st), k1024 = __float2half2_rn(1024.f);
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const uint32_t x = q[w], x8 = x >> 8;
                    uint32_t m0 = 0, m1 = 0, m2 = 0, m3 = 0;
                    if (FMT == gemv::FAST_K5) {
                        const uint32_t H = hb >> w;
                        m0 = (H << 4) & 0x00100010u; m1 = (H >> 4) & 0x00100010u;
                        m2 = (H << 4) & 0x01000100u; m3 = (H >> 4) & 0x01000100u;
                    }
                    const __half2 v0 = as_h2((x & 0x000F000Fu) | m0 | 0x64006400u), v1 = as_h2((x8 & 0x000F000Fu) | m1 | 0x64006400u);
                    const __half2 v2 = as_h2((x & 0x00F000F0u) | m2 | 0x64006400u), v3 = as_h2((x8 & 0x00F000F0u) | m3 | 0x64006400u);
                    lo[w] = __byte_perm(h2_bits(__hadd2(__hfma2(__hsub2(v0, k1024), ah, bh), k1536)),
                                        h2_bits(__hadd2(__hfma2(__hsub2(v1, k1024), ah, bh), k1536)), 0x6240);
                    hi[w] = __byte_perm(h2_bits(__hadd2(__hfma2(__hsub2(v2, k1024), a16, bh), k1536)),
                                        h2_bits(__hadd2(__hfma2(__hsub2(v3, k1024), a16, bh), k1536)), 0x6240);
                }
            }
            (void)T;
            unsigned char* wr = buf + C::O_W + tid * 32;
            *(int4*)(wr + swz32(tid, 0) * 16) = make_int4(lo[0], lo[1], lo[2], lo[3]);
            *(int4*)(wr + swz32(tid, 1) * 16) = make_int4(hi[0], hi[1], hi[2], hi[3]);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int tok = (tid >> 1) + 64 * i, u = tid & 1;
            *(int4*)(buf + C::O_X + tok * 32 + swz32(tok, u) * 16) = xa[i];
        }
        if (tid < 64) {
            ((float2*)(buf + C::O_DX))[tid] = dx0;
            const float2 pb = make_float2(MAGIC_F * dx0.x, MAGIC_F * dx0.y);
            if (ph == 2) ((float2*)(buf + C::O_BI))[tid] = make_float2(bkeep.x + pb.x, bkeep.y + pb.y);
            else if (ph == 1) bkeep = make_float2(bkeep.x + pb.x, bkeep.y + pb.y);
            else bkeep = pb;
        }
    };
    for (int st = 0; st < NSTG - 1 && st < nkb; ++st) {
        load(st);
        store(smem + st * C::BYTES, st == 0 ? 0 : (st & 3) == 3 ? 2 : 1);
    }
    __syncthreads();
    int arow[2], brow[2];
#pragma unroll
    for (int q = 0; q < 2; ++q) arow[q] = wm * 64 + (q * 4 + (lane >> 3)) * 8 + (lane & 7);
    // B: matrices m = lane >> 3 -> token group 2gp + (m & 1), k16 half m >> 1
#pragma unroll
    for (int q = 0; q < 2; ++q) brow[q] = wn * 64 + (lane >> 3 & 1) * 8 + (lane & 7);
    for (int kb = 0; kb < nkb; ++kb) {
        const int nxt = kb + NSTG - 1;
        const bool more = nxt < nkb;
        if (more) load(nxt);
        const unsigned char* B = smem + (kb % NSTG) * C::BYTES;
        uint32_t af[2][8];
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int q = 0; q < 2; ++q)
                ldsm_x4(af[h][q * 4], af[h][q * 4 + 1], af[h][q * 4 + 2], af[h][q * 4 + 3],
                        B + C::O_W + arow[q] * 32 + swz32(arow[q], h) * 16);
#pragma unroll
        for (int gp = 0; gp < 4; ++gp) {
            uint32_t b[4];
            {
                const int row = brow[0] + 16 * gp;
                ldsm_x4(b[0], b[1], b[2], b[3], B + C::O_X + row * 32 + swz32(row, lane >> 4) * 16);
            }
#pragma unroll
            for (int gg = 0; gg < 2; ++gg) {
                const int g = 2 * gp + gg;
                int tq[8][2];
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                        mma_s8p(tq[i][0], tq[i][1], af[h][i], b[h * 2 + gg], h ? tq[i][0] : MAGIC_I, h ? tq[i][1] : MAGIC_I);
                const float2 dv = *(const float2*)(B + C::O_DX + (wn * 64 + 8 * g + 2 * t4) * 4);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[i][g][0] = fmaf(__int_as_float(tq[i][0]), dv.x, acc[i][g][0]);
                    acc[i][g][1] = fmaf(__int_as_float(tq[i][1]), dv.y, acc[i][g][1]);
                }
            }
        }
        if ((kb & 3) == 3) {
#pragma unroll
            for (int g = 0; g < 8; ++g) {
                const float2 bv = *(const float2*)(B + C::O_BI + (wn * 64 + 8 * g + 2 * t4) * 4);
#pragma unroll
                for (int i = 0; i < 8; ++i) { acc[i][g][0] -= bv.x; acc[i][g][1] -= bv.y; }
            }
        }
        if (NSTG == 2) {
            if (more) store(smem + (nxt % NSTG) * C::BYTES, (nxt & 3) == 0 ? 0 : (nxt & 3) == 3 ? 2 : 1);
            __syncthreads();
        } else {
            __syncthreads();  // everyone is done reading stage kb's buffer before it is refilled below
            if (more) store(smem + (nxt % NSTG) * C::BYTES, (nxt & 3) == 0 ? 0 : (nxt & 3) == 3 ? 2 : 1);
        }
    }
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
                    float* p = a.y + (size_t)tok * a.ldy + row;
                    const float v = acc[i][g][e] * sr;
                    *p = a.accumulate ? *p + v : v;
                }
            }
    }
}

template <int FMT, int RPL, int NSTG>
static cudaError_t launch12_t(const Args& a, cudaStream_t s) {
    auto k = gemm12_kernel<FMT, RPL, NSTG>;
    const int smem = NSTG * Cfg12::BYTES;
    static int attr_dev_mask = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    if (!(attr_dev_mask & (1 << dev))) {
        cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return e;
        e = cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        if (e != cudaSuccess) return e;
        attr_dev_mask |= 1 << dev;
    }
    if (a.N % 128 || a.K % 128 || a.Tp % 128) return cudaErrorInvalidValue;
    dim3 grid(a.Tp / 128, a.N / 128);
    k<<<grid, NT12, smem, s>>>(a);
    return cudaGetLastError();
}

static inline cudaError_t launch12(int fmt, int rpl, int nstg, const Args& a, cudaStream_t s) {
#define T4Q_G12(F, R) \
    if (fmt == F && rpl == R) return nstg == 3 ? launch12_t<F, R, 3>(a, s) : launch12_t<F, R, 2>(a, s);
    T4Q_G12(gemv::FAST_P4, 4) T4Q_G12(gemv::FAST_P4, 2)
    T4Q_G12(gemv::FAST_P4M, 4) T4Q_G12(gemv::FAST_P4M, 2)
    T4Q_G12(gemv::FAST_K5, 4) T4Q_G12(gemv::FAST_K5, 2)
#undef T4Q_G12
    return cudaErrorInvalidValue;
}

// per-token quantizer (GA 0): one 256-thread block per token, q = round(x / d), d = amax(x[t]) / 127; dx is [Tp]
__global__ void __launch_bounds__(256) quant8_tok_kernel(const float* __restrict__ x, int ldx, int T, int K,
                                                         int8_t* __restrict__ xq, float* __restrict__ dx) {
    __shared__ float red[8];
    const int t = blockIdx.x, tid = threadIdx.x;
    int8_t* dst = xq + (size_t)t * K;
    if (t >= T) {
        for (int k = tid * 4; k < K; k += 1024) *(int*)(dst + k) = 0;
        if (tid == 0) dx[t] = 0.f;
        return;
    }
    const float* src = x + (size_t)t * ldx;
    float m = 0.f;
    for (int k = tid * 4; k < K; k += 1024) {
        const float4 v = *(const float4*)(src + k);
        m = fmaxf(m, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((tid & 31) == 0) red[tid >> 5] = m;
    __syncthreads();
    float amax = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) amax = fmaxf(amax, red[w]);
    const float d = amax / 127.f;
    for (int k = tid * 4; k < K; k += 1024) {
        const float4 v = *(const float4*)(src + k);
        const float f[4] = {v.x, v.y, v.z, v.w};
        uint32_t pk = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int q = amax == 0.f ? 0 : __float2int_rn(f[e] / d);
            pk |= (uint32_t)(q & 0xff) << (8 * e);
        }
        *(uint32_t*)(dst + k) = pk;
    }
    if (tid == 0) dx[t] = d;
}

template <int FMT, int RPL, int BN, int MODE = 0>
static cudaError_t launch_t(const Args& a, cudaStream_t s) {
    auto k = gemm8_kernel<FMT, RPL, BN, MODE>;
    const int smem = 2 * Cfg<BN>::BYTES;
    static int attr_dev_mask = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    if (!(attr_dev_mask & (1 << dev))) {
        cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return e;
        attr_dev_mask |= 1 << dev;
    }
    if (a.N % BM || a.K % 128 || a.Tp % BN) return cudaErrorInvalidValue;
    dim3 grid(a.Tp / BN, a.N / BM);
    k<<<grid, NT, smem, s>>>(a);
    return cudaGetLastError();
}

// bn: 256 or 128 (token tile); fmt / rpl from the weight's gemv::Layout
static inline cudaError_t launch(int fmt, int rpl, int bn, const Args& a, cudaStream_t s) {
#define T4Q_G8(F, R)                                                                  \
    if (fmt == F && rpl == R) return bn == 256 ? launch_t<F, R, 256>(a, s) : launch_t<F, R, 128>(a, s);
    T4Q_G8(gemv::FAST_P4, 4) T4Q_G8(gemv::FAST_P4, 2)
    T4Q_G8(gemv::FAST_P4M, 4) T4Q_G8(gemv::FAST_P4M, 2)
    T4Q_G8(gemv::FAST_K5, 4) T4Q_G8(gemv::FAST_K5, 2)
#undef T4Q_G8
    return cudaErrorInvalidValue;
}

static inline cudaError_t row_invs(int fmt, int rpl, const Args& a, float* invs, cudaStream_t s) {
    const int blocks = (a.N + 7) / 8;
#define T4Q_RI(F, R) \
    if (fmt == F && rpl == R) { row_invs_kernel<F, R><<<blocks, 256, 0, s>>>(a, invs); return cudaGetLastError(); }
    T4Q_RI(gemv::FAST_P4, 4) T4Q_RI(gemv::FAST_P4, 2)
    T4Q_RI(gemv::FAST_P4M, 4) T4Q_RI(gemv::FAST_P4M, 2)
    T4Q_RI(gemv::FAST_K5, 4) T4Q_RI(gemv::FAST_K5, 2)
#undef T4Q_RI
    return cudaErrorInvalidValue;
}

// prefill activation quantizer for this kernel: x [T][K] fp32 -> xq [Tp][K] int8, dx [K/32][Tp] (d = amax/127).
// One thread per (token, 32-block); padding tokens get zeros. Same q values as gemm::quant_rows_kernel.
template <int GA = 32>
__global__ void quant8_kernel(const float* __restrict__ x, int ldx, int T, int Tp, int K, int8_t* __restrict__ xq,
                              float* __restrict__ dx) {
    const int nb = K / GA;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Tp * nb) return;
    const int t = i / nb, b = i % nb;
    int4* dst = (int4*)(xq + (size_t)t * K + b * GA);
    if (t >= T) {
#pragma unroll
        for (int k = 0; k < GA / 16; ++k) dst[k] = make_int4(0, 0, 0, 0);
        dx[(size_t)b * Tp + t] = 0.f;
        return;
    }
    const float4* src = (const float4*)(x + (size_t)t * ldx + b * GA);
    float v[GA];
#pragma unroll
    for (int k = 0; k < GA / 4; ++k) {
        const float4 f = src[k];
        v[4 * k] = f.x; v[4 * k + 1] = f.y; v[4 * k + 2] = f.z; v[4 * k + 3] = f.w;
    }
    float amax = 0.f;
#pragma unroll
    for (int k = 0; k < GA; ++k) amax = fmaxf(amax, fabsf(v[k]));
    const float d = amax / 127.f;
    uint32_t w[GA / 4];
#pragma unroll
    for (int k = 0; k < GA / 4; ++k) {
        uint32_t pk = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int q = amax == 0.f ? 0 : __float2int_rn(v[4 * k + e] / d);
            pk |= (uint32_t)(q & 0xff) << (8 * e);
        }
        w[k] = pk;
    }
#pragma unroll
    for (int k = 0; k < GA / 16; ++k) dst[k] = make_int4(w[4 * k], w[4 * k + 1], w[4 * k + 2], w[4 * k + 3]);
    dx[(size_t)b * Tp + t] = d;
}

// GA 128: one thread per (token, 128-group), two passes over the inputs (no 128-float register array)
__global__ void quant8_g128_kernel(const float* __restrict__ x, int ldx, int T, int Tp, int K, int8_t* __restrict__ xq,
                                   float* __restrict__ dx) {
    const int nb = K / 128;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Tp * nb) return;
    const int t = i / nb, b = i % nb;
    int4* dst = (int4*)(xq + (size_t)t * K + b * 128);
    if (t >= T) {
#pragma unroll
        for (int k = 0; k < 8; ++k) dst[k] = make_int4(0, 0, 0, 0);
        dx[(size_t)b * Tp + t] = 0.f;
        return;
    }
    const float4* src = (const float4*)(x + (size_t)t * ldx + b * 128);
    float amax = 0.f;
    for (int k = 0; k < 32; ++k) {
        const float4 f = src[k];
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f.x), fabsf(f.y)), fmaxf(fabsf(f.z), fabsf(f.w))));
    }
    const float d = amax / 127.f;
    for (int k = 0; k < 8; ++k) {
        uint32_t w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 f = src[4 * k + j];
            const float v[4] = {f.x, f.y, f.z, f.w};
            uint32_t pk = 0;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int q = amax == 0.f ? 0 : __float2int_rn(v[e] / d);
                pk |= (uint32_t)(q & 0xff) << (8 * e);
            }
            w[j] = pk;
        }
        dst[k] = make_int4(w[0], w[1], w[2], w[3]);
    }
    dx[(size_t)b * Tp + t] = d;
}

// ga: activation scale group (32, 64, 128, or 0 = per token); dx is [K/ga][Tp]
static inline void quant8(const float* x, int ldx, int T, int Tp, int K, int8_t* xq, float* dx, cudaStream_t s,
                          int ga = 32) {
    if (ga == 0) { quant8_tok_kernel<<<Tp, 256, 0, s>>>(x, ldx, T, K, xq, dx); return; }
    if (ga == 128) { const int n = Tp * (K / 128); quant8_g128_kernel<<<(n + 127) / 128, 128, 0, s>>>(x, ldx, T, Tp, K, xq, dx); return; }
    const int n = Tp * (K / ga);
    if (ga == 64) quant8_kernel<64><<<(n + 127) / 128, 128, 0, s>>>(x, ldx, T, Tp, K, xq, dx);
    else quant8_kernel<32><<<(n + 127) / 128, 128, 0, s>>>(x, ldx, T, Tp, K, xq, dx);
}

}  // namespace gemm8
}  // namespace t4q
#endif
