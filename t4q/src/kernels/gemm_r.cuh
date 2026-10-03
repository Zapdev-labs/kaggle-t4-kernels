// t4q/src/kernels/gemm_r.cuh -- batched-decode GEMM (milestone B): register-direct weight fragments for P4 (Q4_0).
//
// Why: gemm9 stages every weight tile through shared memory (convert, store, barrier, ldmatrix). At 32 / 64-token
// tiles that staging and the barrier per 64 k dominate (t4q-b v6 ablations: no smem stores / barriers -> 2x faster;
// L2 prefetch, 3-stage load rings and line-batched loads changed nothing), and each warp re-reads the whole 128-row
// A tile for its 8-16 tokens. The step's GEMMs streamed weights at only ~110-125 GB/s (dp4a GEMV: 260).
//
// Here the Q4_0 nibble layout is used directly as mma.m8n8k16 A fragments: in a 16-B code group, byte b holds k = b
// (low nibble) and k = b + 16 (high nibble), so thread t of a warp loads the 4 code bytes 4*(t&3)..+3 of row (t >> 2)
// and converts them in registers into the A fragment of the low k16 half (low nibbles) and of the high half (high
// nibbles) -- exactly gemm9's per-row int8 requantization (same fp16 ops), so outputs are bit-identical to gemm9.
// Only the activations (BN tokens x 256 k per stage, shared by all warps) go through shared memory.
//
// Block: 8 warps x MT m-tiles of 8 rows (MT 2: 128 rows, MT 1: 64 rows); the warp owns all BN tokens. Per 256-k stage:
// weight codes / scales for the next stage are loaded into registers while the current stage computes (ping-pong),
// activations are register-staged into the other smem buffer. Accumulation per output: per 64 k four k16 mmas chained
// from MAGIC, one FFMA with the GA64 activation scale, bias subtracted every 256 k (gemm9's order and roundings).
#pragma once
#include "gemm8.cuh"

#ifdef __CUDACC__
namespace t4q {
namespace gemmr {

using gemm8::Args;
constexpr int KC = 256;  // k per stage
constexpr int NT = 256;

template <int BN>
struct SCfg {
    static constexpr int O_X = 0;                 // int8 activations [BN][256 B], 16-B units swizzled by token
    static constexpr int O_DX = BN * KC;          // float [4][BN] GA64 scales of the stage
    static constexpr int O_BI = O_DX + 4 * BN * 4;  // float [BN] MAGIC * sum of the stage's 4 scales
    static constexpr int BYTES = O_BI + BN * 4;
    static constexpr int NU = BN * KC / 16 / NT;  // 16-B activation units per thread per stage (BN 32: 2, 64: 4)
};
__device__ __forceinline__ int xswz(int tok, int u) { return u ^ (tok & 7); }

// gemm9's P4 conversion of one 4-byte code word (stage_store, FMT P4): lo = k 0..3 of the word's low nibbles,
// hi = the high nibbles; d16 = the group's fp16 scale bits, invs = the row's 127 / max |w|
__device__ __forceinline__ void p4_word(uint32_t x, uint32_t d16, float invs, uint32_t& lo, uint32_t& hi) {
    using namespace gemm8;
    const float av = h2f(d16) * invs;
    const __half2 r = __float2half2_rn(av), r16 = __float2half2_rn(av * 0.0625f);
    const __half2 k1032 = __float2half2_rn(1032.f), k1152 = __float2half2_rn(1152.f), k1536 = __float2half2_rn(1536.f);
    const uint32_t x8 = x >> 8;
    const __half2 v0 = as_h2((x & 0x000F000Fu) | 0x64006400u), v1 = as_h2((x8 & 0x000F000Fu) | 0x64006400u);
    const __half2 v2 = as_h2((x & 0x00F000F0u) | 0x64006400u), v3 = as_h2((x8 & 0x00F000F0u) | 0x64006400u);
    const uint32_t t0 = h2_bits(__hfma2(__hsub2(v0, k1032), r, k1536));
    const uint32_t t1 = h2_bits(__hfma2(__hsub2(v1, k1032), r, k1536));
    const uint32_t t2 = h2_bits(__hfma2(__hsub2(v2, k1152), r16, k1536));
    const uint32_t t3 = h2_bits(__hfma2(__hsub2(v3, k1152), r16, k1536));
    lo = __byte_perm(t0, t1, 0x6240);
    hi = __byte_perm(t2, t3, 0x6240);
}

template <int MT>
struct WReg {
    uint32_t q[MT][8];   // code word of (m-tile, group) for this thread
    uint32_t d[MT][8];   // fp16 scale bits
};

// EPI 0: y / yh (+ yh2) like gemm9 (no split-K, no accumulate); EPI 64: gate|up silu -> q8 GA64 (a.oq / a.odx)
template <int RPL, int BN, int MT, int EPI>
__global__ void __launch_bounds__(NT, MT == 1 ? 2 : 1) gemmr_kernel(const Args a) {
    using C = SCfg<BN>;
    constexpr int NTL = BN / 8;  // n-tiles (8 tokens) per warp
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, t4 = lane & 3;
    const int BMr = 64 * MT;
    const int row_blk = blockIdx.y * BMr;
    const int nst = a.K / KC;
    // per m-tile: this thread's row and its code / scale pointers at chunk 0, group 0
    const uint8_t* cp[MT];
    const uint16_t* dp[MT];
    float invs[MT];
    const long long dtc = a.cm ? a.ntiles : 1;
    const long long cs_code = dtc * RPL * 512, cs_d = dtc * 32 * RPL;  // bytes / u16 per 512-k chunk
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int R = row_blk + (warp * MT + m) * 8 + (lane >> 2);
        const int lt = R / (2 * RPL), w = R % (2 * RPL), h = w / RPL, r = w % RPL;
        const long long tc0 = (long long)gemv::tc_index(lt, 0, a.K >> 9, a.ntiles, a.cm);
        cp[m] = a.codes + (tc0 * RPL + r) * 512 + (h * 16) * 16 + t4 * 4;
        dp[m] = a.d + (tc0 * 32 + h * 16) * RPL + r;
        invs[m] = a.invs[R];
    }
    auto wload = [&](WReg<MT>& W, int st) {
        const int c = st >> 1, j0 = (st & 1) * 8;
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int g = 0; g < 8; ++g) {
                W.q[m][g] = __ldg((const unsigned int*)(cp[m] + c * cs_code + (j0 + g) * 16));
                W.d[m][g] = (uint32_t)__ldg((const unsigned short*)(dp[m] + c * cs_d + (j0 + g) * RPL));
            }
    };
    // activations: unit U = tid + i * NT -> token U >> 4, 16-B unit U & 15 of the stage
    int4 xr[C::NU];
    float dxr[4];
    auto xload = [&](int st) {
#pragma unroll
        for (int i = 0; i < C::NU; ++i) {
            const int U = tid + i * NT, tok = U >> 4, u = U & 15;
            xr[i] = __ldg((const int4*)(a.xq + (size_t)tok * a.K + st * KC + u * 16));
        }
        if (tid < BN) {
#pragma unroll
            for (int q = 0; q < 4; ++q) dxr[q] = __ldg(a.dx + (size_t)(st * 4 + q) * a.Tp + tid);
        }
    };
    auto xstore = [&](unsigned char* buf) {
#pragma unroll
        for (int i = 0; i < C::NU; ++i) {
            const int U = tid + i * NT, tok = U >> 4, u = U & 15;
            *(int4*)(buf + C::O_X + tok * KC + xswz(tok, u) * 16) = xr[i];
        }
        if (tid < BN) {
            float* dxs = (float*)(buf + C::O_DX);
#pragma unroll
            for (int q = 0; q < 4; ++q) dxs[q * BN + tid] = dxr[q];
            // gemm9's bias order: ((M dx0 + M dx1) + M dx2) + M dx3, explicit roundings
            float b = __fmul_rn(gemm8::MAGIC_F, __fadd_rn(dxr[0], 0.f));
            b = __fadd_rn(b, __fmul_rn(gemm8::MAGIC_F, __fadd_rn(dxr[1], 0.f)));
            b = __fadd_rn(b, __fmul_rn(gemm8::MAGIC_F, __fadd_rn(dxr[2], 0.f)));
            b = __fadd_rn(b, __fmul_rn(gemm8::MAGIC_F, __fadd_rn(dxr[3], 0.f)));
            ((float*)(buf + C::O_BI))[tid] = b;
        }
    };

    float acc[MT][NTL][2];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int n = 0; n < NTL; ++n) acc[m][n][0] = acc[m][n][1] = 0.f;

    WReg<MT> W0, W1;
    wload(W0, 0);
    xload(0);
    xstore(smem);
    __syncthreads();
    if (nst > 1) { xload(1); wload(W1, 1); }

    auto compute = [&](const WReg<MT>& W, const unsigned char* B) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {  // 64-k sub-step: groups 2q, 2q+1 of the stage
            uint32_t af[MT][4];
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                p4_word(W.q[m][2 * q], W.d[m][2 * q], invs[m], af[m][0], af[m][1]);
                p4_word(W.q[m][2 * q + 1], W.d[m][2 * q + 1], invs[m], af[m][2], af[m][3]);
            }
            const float* dxs = (const float*)(B + C::O_DX) + q * BN;
#pragma unroll
            for (int n = 0; n < NTL; ++n) {
                uint32_t b[4];
                const int tok = n * 8 + (lane & 7);
                gemm8::ldsm_x4(b[0], b[1], b[2], b[3], B + C::O_X + tok * KC + xswz(tok, q * 4 + (lane >> 3)) * 16);
                const float2 dv = *(const float2*)(dxs + n * 8 + 2 * t4);
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    int t0, t1;
                    gemm8::mma_s8p(t0, t1, af[m][0], b[0], gemm8::MAGIC_I, gemm8::MAGIC_I);
                    gemm8::mma_s8p(t0, t1, af[m][1], b[1], t0, t1);
                    gemm8::mma_s8p(t0, t1, af[m][2], b[2], t0, t1);
                    gemm8::mma_s8p(t0, t1, af[m][3], b[3], t0, t1);
                    acc[m][n][0] = fmaf(__int_as_float(t0), dv.x, acc[m][n][0]);
                    acc[m][n][1] = fmaf(__int_as_float(t1), dv.y, acc[m][n][1]);
                }
            }
        }
        const float* bi = (const float*)(B + C::O_BI);  // end of a 256-k bias group (every stage)
#pragma unroll
        for (int n = 0; n < NTL; ++n) {
            const float2 bv = *(const float2*)(bi + n * 8 + 2 * t4);
#pragma unroll
            for (int m = 0; m < MT; ++m) { acc[m][n][0] -= bv.x; acc[m][n][1] -= bv.y; }
        }
    };

    int buf = 0;
    for (int st = 0; st < nst; st += 2) {
        // stage st: weights in W0, activations in smem[buf]; stage st + 1 staged in W1 / xr
        compute(W0, smem + buf * C::BYTES);
        if (st + 1 < nst) {
            xstore(smem + (buf ^ 1) * C::BYTES);
            if (st + 2 < nst) wload(W0, st + 2);
            __syncthreads();
            buf ^= 1;
            if (st + 2 < nst) xload(st + 2);
            compute(W1, smem + buf * C::BYTES);
            if (st + 2 < nst) {
                xstore(smem + (buf ^ 1) * C::BYTES);
                if (st + 3 < nst) wload(W1, st + 3);
                __syncthreads();
                buf ^= 1;
                if (st + 3 < nst) xload(st + 3);
            }
        }
    }

    if (EPI == 64) {
        // gate|up rows interleaved by 4 in each 8-row m-tile: lanes 0..15 hold gate rows, lane ^ 16 the matching up
        // row. Feature f = row_blk / 2 + (warp * MT + m) * 4 + (lane >> 2); the 64-feature group is the block's 128 rows.
        __syncthreads();
        float* amx = (float*)smem;  // [16 warp-mtiles][BN]
        float hv[MT][NTL][2];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int R = row_blk + (warp * MT + m) * 8 + (lane >> 2);
            const float sr = __frcp_rn(a.invs[R]);
#pragma unroll
            for (int n = 0; n < NTL; ++n)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const float v = acc[m][n][e] * sr;
                    const float u = __shfl_xor_sync(0xffffffffu, v, 16);
                    hv[m][n][e] = lane < 16 ? (v / (1.0f + expf(-v))) * u : 0.f;
                }
        }
#pragma unroll
        for (int n = 0; n < NTL; ++n)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                float mx = 0.f;
#pragma unroll
                for (int m = 0; m < MT; ++m) mx = fmaxf(mx, fabsf(hv[m][n][e]));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
                if (lane < 4) amx[warp * BN + n * 8 + 2 * t4 + e] = mx;
            }
        __syncthreads();
        const int K2 = a.N >> 1;
#pragma unroll
        for (int n = 0; n < NTL; ++n)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int tok = n * 8 + 2 * t4 + e;
                float am = 0.f;
#pragma unroll
                for (int w = 0; w < 8; ++w) am = fmaxf(am, amx[w * BN + tok]);
                const float d = am / 127.f;
                if (lane < 16) {
#pragma unroll
                    for (int m = 0; m < MT; ++m) {
                        const int fb = (row_blk >> 1) + (warp * MT + m) * 4 + (lane >> 2);
                        a.oq[(size_t)tok * K2 + fb] = (int8_t)((tok < a.T && am != 0.f) ? __float2int_rn(hv[m][n][e] / d) : 0);
                    }
                }
                if (warp == 0 && lane < 4) a.odx[(size_t)(row_blk >> 7) * a.Tp + tok] = tok < a.T ? d : 0.f;
            }
        return;
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int row = row_blk + (warp * MT + m) * 8 + (lane >> 2);
        const float sr = __frcp_rn(a.invs[row]);
#pragma unroll
        for (int n = 0; n < NTL; ++n)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int tok = n * 8 + 2 * t4 + e;
                if (tok < a.T) {
                    const float v = acc[m][n][e] * sr;
                    if (a.yh) {
                        const __half hv2 = __float2half_rn(v);
                        a.yh[(size_t)tok * a.ldy + row] = hv2;
                        if (a.yh2) a.yh2[(size_t)tok * a.ldy + row] = hv2;
                    } else {
                        a.y[(size_t)tok * a.ldy + row] = v;
                    }
                }
            }
    }
}

template <int RPL, int BN, int MT, int EPI>
static cudaError_t launchr_t(const Args& a, cudaStream_t s) {
    auto k = gemmr_kernel<RPL, BN, MT, EPI>;
    const int smem = 2 * SCfg<BN>::BYTES;
    static int attr_dev_mask = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    if (!(attr_dev_mask & (1 << dev))) {
        cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return e;
        attr_dev_mask |= 1 << dev;
    }
    if (a.N % (64 * MT) || a.K % KC || a.Tp != BN || a.kz != 1) return cudaErrorInvalidValue;
    k<<<dim3(1, a.N / (64 * MT)), NT, smem, s>>>(a);
    return cudaGetLastError();
}

// P4 weights (rpl 2 / 4), Tp 32 or 64. silu: gate|up epilogue (needs 128-row blocks). mt: 1 (64-row blocks) or 2.
static inline cudaError_t launch(int fmt, int rpl, int Tp, int mt, bool silu, const Args& a, cudaStream_t s) {
    if (fmt != gemv::FAST_P4 || (Tp != 32 && Tp != 64)) return cudaErrorInvalidValue;
#define T4Q_GR(R, B)                                                                                   \
    if (rpl == R && Tp == B) {                                                                         \
        if (silu) return launchr_t<R, B, 2, 64>(a, s);                                                 \
        return mt == 1 ? launchr_t<R, B, 1, 0>(a, s) : launchr_t<R, B, 2, 0>(a, s);                     \
    }
    T4Q_GR(4, 32) T4Q_GR(4, 64) T4Q_GR(2, 32) T4Q_GR(2, 64)
#undef T4Q_GR
    return cudaErrorInvalidValue;
}

}  // namespace gemmr
}  // namespace t4q
#endif
