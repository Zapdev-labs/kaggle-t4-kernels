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

template <int FMT, int BN>
__device__ __forceinline__ void stage_store(const Stage<BN>& S, unsigned char* buf, float invs, float2& bkeep,
                                            bool odd, int tid) {
    using C = Cfg<BN>;
    // ---- weights
    {
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
        if (odd) ((float2*)(buf + C::O_BI))[tid] = make_float2(bkeep.x + pb.x, bkeep.y + pb.y);
        else bkeep = pb;
    }
}

template <int FMT, int RPL, int BN>
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
    stage_store<FMT, BN>(S, smem, invs_st, bkeep, false, tid);
    __syncthreads();

    int buf = 0;
    for (int kb0 = 0; kb0 < nkb; kb0 += 2) {
        const bool more = kb0 + 2 < nkb;
        if (more) stage_load<FMT, RPL, BN>(S, a, row0, tok0, kb0 + 2, tid);
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
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                const float2 dv = *(const float2*)(B + C::O_DX + (kb * BN + wn * WN + 8 * g + 2 * t4) * 4);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int d0, d1;
                    mma_s8(d0, d1, af[0][i], bf[0][g], MAGIC_I, MAGIC_I);
                    mma_s8(d0, d1, af[1][i], bf[1][g], d0, d1);
                    acc[i][g][0] = fmaf(__int_as_float(d0), dv.x, acc[i][g][0]);
                    acc[i][g][1] = fmaf(__int_as_float(d1), dv.y, acc[i][g][1]);
                }
            }
        }
        if (kb0 & 2) {  // odd stage: end of a 4-block group
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                const float2 bv = *(const float2*)(B + C::O_BI + (wn * WN + 8 * g + 2 * t4) * 4);
#pragma unroll
                for (int i = 0; i < 8; ++i) { acc[i][g][0] -= bv.x; acc[i][g][1] -= bv.y; }
            }
        }
        if (more) {
            stage_store<FMT, BN>(S, smem + (buf ^ 1) * C::BYTES, invs_st, bkeep, ((kb0 + 2) & 2) != 0, tid);
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

template <int FMT, int RPL, int BN>
static cudaError_t launch_t(const Args& a, cudaStream_t s) {
    auto k = gemm8_kernel<FMT, RPL, BN>;
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
__global__ void quant8_kernel(const float* __restrict__ x, int ldx, int T, int Tp, int K, int8_t* __restrict__ xq,
                              float* __restrict__ dx) {
    const int nb = K >> 5;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Tp * nb) return;
    const int t = i / nb, b = i % nb;
    int4* dst = (int4*)(xq + (size_t)t * K + b * 32);
    if (t >= T) {
        dst[0] = make_int4(0, 0, 0, 0); dst[1] = make_int4(0, 0, 0, 0);
        dx[(size_t)b * Tp + t] = 0.f;
        return;
    }
    const float4* src = (const float4*)(x + (size_t)t * ldx + b * 32);
    float v[32];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float4 f = src[k];
        v[4 * k] = f.x; v[4 * k + 1] = f.y; v[4 * k + 2] = f.z; v[4 * k + 3] = f.w;
    }
    float amax = 0.f;
#pragma unroll
    for (int k = 0; k < 32; ++k) amax = fmaxf(amax, fabsf(v[k]));
    const float d = amax / 127.f;
    uint32_t w[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        uint32_t pk = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int q = amax == 0.f ? 0 : __float2int_rn(v[4 * k + e] / d);
            pk |= (uint32_t)(q & 0xff) << (8 * e);
        }
        w[k] = pk;
    }
    dst[0] = make_int4(w[0], w[1], w[2], w[3]);
    dst[1] = make_int4(w[4], w[5], w[6], w[7]);
    dx[(size_t)b * Tp + t] = d;
}

static inline void quant8(const float* x, int ldx, int T, int Tp, int K, int8_t* xq, float* dx, cudaStream_t s) {
    const int n = Tp * (K >> 5);
    quant8_kernel<<<(n + 127) / 128, 128, 0, s>>>(x, ldx, T, Tp, K, xq, dx);
}

}  // namespace gemm8
}  // namespace t4q
#endif
