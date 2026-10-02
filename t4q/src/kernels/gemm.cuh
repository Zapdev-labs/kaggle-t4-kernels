// t4q/src/kernels/gemm.cuh -- W4A8 prefill GEMM on sm_75 int8 tensor cores (milestone P).
//
// y[t][n] = sum_k W[n][k] * x[t][k] for T tokens, reading the decode GEMV's packed weights in place (gemv.cuh
// Layout with rpl = 4, i.e. 8-row tiles; cm 0 or 1), so prefill needs no second weight copy.
//
// Math: mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32 with A = weight codes (unsigned, scaled so that the code
// sits in the high bits of each byte: P4/P4M 16*c, K5 8*c), B = q8 activations (int8 per 32, scale d per token per
// 32-block). Per 32-block the two k16 mmas accumulate on top of the magic constant 0x4B400000, so the int32 result
// is a float (M + S) exactly (|S| < 2^22) with no I2F. Epilogue per element per block (fp32):
//     t   = f * s + nq            s = d/16, nq = -(M + 128*sumq) * d/16   ->  t = d * (S/16 - 8*sumq)
//     acc = fma(wa, t, acc)       [+ fma(wb, sx, acc) for P4M/K5 with sx = d*sumq]
//   P4 (Q4_0):  wa = d_w                  -> d_w * d * sum (c-8) q
//   P4M (Q4_1): wa = d_w, wb = 8 d_w + m  -> d_w d sum c q + m sx
//   K5 (Q5_K):  A = 8c, wa = 2 d sc, wb = 16 d sc - dmin m
// Two FFMAs per output per 32-block against two mmas: the FP32 pipe (2 warp-instr/clk/SM) and the int8 tensor
// pipe (one m8n8k16/clk/SM) are balanced, so the kernel is built to keep every other instruction off the FP32 pipe.
//
// Layout rpl 2 (4-row tiles) and 4 (8-row tiles) are both read; smem is canonical 8-row groups.
// Tiling: block = 128 weight rows (16 tiles) x 128 tokens, 256 threads = 8 warps as 2 (rows) x 4 (tokens), warp tile
// 64 rows x 32 tokens = 8 x 4 mma tiles, 64 fp32 accumulators per thread. K stage = KS 32-blocks (P4: 4, P4M/K5: 2),
// register-staged double buffering in dynamic shared memory (sm_75 has no cp.async), one __syncthreads per stage.
// Fragments come from ldmatrix: a weight matrix is 8 rows x one Q4_0-ordered 16-B group (lo nibbles = k 0..15 of the
// block, hi nibbles = k 16..31), which is exactly the m8n8k16 A fragment after a mask.
//
// Activations (prefill q8): xq [Tp][K] int8 (natural order), xs [K/32][Tp] float2 {d/16, nq}, xsum [K/32][Tp] float
// (d*sumq). Tp = T rounded up to 128; padding tokens must be zero (q = 0, s = nq = 0) and are not stored.
#pragma once
#include "gemv.cuh"

namespace t4q {
namespace gemm {

constexpr int BM = 128, BN = 128, NT = 256;
constexpr int MAGIC_I = 0x4B400000;
constexpr float MAGIC_F = 12582912.f;

static inline int ks_of(int fmt) { return fmt == gemv::FAST_P4 ? 4 : 2; }

// shared memory bytes per stage buffer
static inline size_t stage_bytes(int fmt) {
    const int KS = ks_of(fmt);
    size_t b = (size_t)16 * KS * 128;       // wq
    b += (size_t)KS * 8 * 16 * 4;           // wa
    if (fmt != gemv::FAST_P4) b += (size_t)KS * 8 * 16 * 4;  // wb
    if (fmt == gemv::FAST_K5) b += (size_t)16 * KS * 8 * 4;   // wh
    b += (size_t)BN * KS * 32;              // xa
    b += (size_t)KS * BN * 8;               // meta
    if (fmt != gemv::FAST_P4) b += (size_t)KS * BN * 4;       // xsum
    return b;
}
static inline size_t smem_bytes(int fmt) { return 2 * stage_bytes(fmt); }

// host mirror of the prefill activation quantizer (one token row)
static inline void quant_row_host(const float* x, int K, int8_t* xq, float* s_out, float* nq_out, float* sx_out) {
    for (int b = 0; b < K / 32; ++b) {
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[32 * b + i]));
        const float d = amax / 127.f;
        int sq = 0;
        for (int i = 0; i < 32; ++i) {
            const int q = amax == 0.f ? 0 : (int)std::nearbyint(x[32 * b + i] / d);
            xq[32 * b + i] = (int8_t)q;
            sq += q;
        }
        const float s = d * 0.0625f;
        s_out[b] = s;
        nq_out[b] = -(float)(12582912 + 128 * sq) * s;
        sx_out[b] = d * (float)sq;
    }
}

}  // namespace gemm
}  // namespace t4q

#ifdef __CUDACC__
namespace t4q {
namespace gemm {

struct GemmArgs {
    const uint8_t* codes = nullptr;  // weight planes (gemv::Layout, rpl 4)
    const uint8_t* qh = nullptr;
    const uint8_t* sc = nullptr;
    const uint16_t* d = nullptr;
    int N = 0, K = 0, ntiles = 0, cm = 0;
    const int8_t* xq = nullptr;   // [Tp][K]
    const float2* xs = nullptr;   // [K/32][Tp]
    const float* xsum = nullptr;  // [K/32][Tp] (P4M, K5)
    float* y = nullptr;           // y[t * ldy + n]
    int ldy = 0;
    int T = 0, Tp = 0;
    int accumulate = 0;           // 1: y += result
};

static inline GemmArgs make_args(const gemv::Layout& L, const uint8_t* base, const int8_t* xq, const float2* xs,
                                 const float* xsum, float* y, int ldy, int T, int Tp) {
    GemmArgs a;
    a.codes = base + L.off_codes; a.qh = base + L.off_qh; a.sc = base + L.off_sc;
    a.d = (const uint16_t*)(base + L.off_d);
    a.N = L.N; a.K = L.K; a.ntiles = L.ntiles; a.cm = L.cm;
    a.xq = xq; a.xs = xs; a.xsum = xsum; a.y = y; a.ldy = ldy; a.T = T; a.Tp = Tp;
    return a;
}

__device__ __forceinline__ void mma_u8s8(int& d0, int& d1, uint32_t a, uint32_t b, int c0, int c1) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};"
                 : "=r"(d0), "=r"(d1) : "r"(a), "r"(b), "r"(c0), "r"(c1));
}
// int4 tensor path (m8n8k32): A = Q4_0 codes as u4 straight from ldmatrix (a Q4_0 16-B group is one row of the A
// fragment, k slots permuted: slot 8t+2j = element 4t+j, slot 8t+2j+1 = element 4t+16+j); B = int8 activations split
// into a signed high nibble and an unsigned low nibble, each packed in the same Q4_0 order (x = 16*xh + xl exactly).
__device__ __forceinline__ void mma_u4s4(int& d0, int& d1, uint32_t a, uint32_t b, int c0, int c1) {
    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.u4.s4.s32 {%0,%1}, {%2}, {%3}, {%4,%5};"
                 : "=r"(d0), "=r"(d1) : "r"(a), "r"(b), "r"(c0), "r"(c1));
}
__device__ __forceinline__ void mma_u4u4(int& d0, int& d1, uint32_t a, uint32_t b, int c0, int c1) {
    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.u4.u4.s32 {%0,%1}, {%2}, {%3}, {%4,%5};"
                 : "=r"(d0), "=r"(d1) : "r"(a), "r"(b), "r"(c0), "r"(c1));
}
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const unsigned sp = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(sp));
}
__device__ __forceinline__ float h2f_u16(uint32_t b) { return __half2float(__ushort_as_half((unsigned short)(b & 0xffff))); }

// activation 16-B unit swizzle (UPR units per token row): conflict-free ldmatrix over 8 consecutive tokens
template <int UPR>
__device__ __forceinline__ int xswz(int tok, int u) {
    return UPR == 8 ? (u ^ (tok & 7)) : (u ^ ((tok >> 1) & 3));
}

template <int FMT>
struct Cfg {
    static constexpr int KS = FMT == gemv::FAST_P4 ? 4 : 2;
    static constexpr int UPR = KS * 2;  // 16-B activation units per token row per stage
    static constexpr bool HASB = FMT != gemv::FAST_P4;
    static constexpr bool K5 = FMT == gemv::FAST_K5;
    // smem offsets within a stage buffer
    static constexpr int O_WQ = 0;
    static constexpr int O_WA = O_WQ + 16 * KS * 128;
    static constexpr int O_WB = O_WA + KS * 8 * 16 * 4;
    static constexpr int O_WH = O_WB + (HASB ? KS * 8 * 16 * 4 : 0);
    static constexpr int O_XA = O_WH + (K5 ? 16 * KS * 8 * 4 : 0);
    static constexpr int O_MT = O_XA + BN * KS * 32;
    static constexpr int O_XS = O_MT + KS * BN * 8;
    static constexpr int BYTES = O_XS + (HASB ? KS * BN * 4 : 0);
    // staging units per thread
    static constexpr int N_WQ = 16 * 4 * 2 * KS / NT * 1;     // 16-B code units: (128 KS) / 256 -> KS/2
    static constexpr int N_XA = BN * UPR / NT;                // 16-B activation units
};

template <int FMT, int RPL>
struct Stage {
    static constexpr int KS = Cfg<FMT>::KS;
    int4 wq[KS / 2];      // codes
    uint32_t ws[RPL / 2]; // RPL fp16 scales (P4/P4M d) or RPL x K5 {sc,m} bytes
    uint32_t wm[RPL / 2]; // P4M m (RPL fp16)
    uint32_t wd[RPL];     // K5 {d,dmin} per row r
    uint32_t wh;          // K5 high bits (one u32)
    int4 xa[Cfg<FMT>::N_XA];
    float4 mt;            // meta (threads < KS*BN/2)
    float4 xs;            // xsum (threads < KS*BN/4)
};

// code / high-bit unit U in [0, 128*KS): (kb, h, r, q, tile8) and its row within the 8-row group.
// RPL 4: layout tile = 8-row group; RPL 2: two 4-row layout tiles q = 0, 1 per group.
template <int KS, int RPL>
__device__ __forceinline__ void code_unit(int U, int& kb, int& h, int& r, int& ltile_off, int& row8, int& tile8) {
    kb = U % KS;
    const int idx = U / KS;
    h = idx & 1;
    if (RPL == 4) { r = (idx >> 1) & 3; tile8 = idx >> 3; ltile_off = tile8; row8 = h * 4 + r; }
    else { r = (idx >> 1) & 1; const int q = (idx >> 2) & 1; tile8 = idx >> 3; ltile_off = tile8 * 2 + q; row8 = q * 4 + h * 2 + r; }
}
// scale unit V in [0, 32*KS*(4/RPL)): one lane (kb, h) of one layout tile, RPL rows
template <int KS, int RPL>
__device__ __forceinline__ void scale_unit(int V, int& kb, int& h, int& ltile_off, int& row8base, int& tile8) {
    kb = V % KS;
    h = (V / KS) & 1;
    if (RPL == 4) { tile8 = V / (2 * KS); ltile_off = tile8; row8base = h * 4; }
    else { const int q = (V / (2 * KS)) & 1; tile8 = V / (4 * KS); ltile_off = tile8 * 2 + q; row8base = q * 4 + h * 2; }
}

template <int FMT, int RPL>
__device__ __forceinline__ void stage_load(Stage<FMT, RPL>& S, const GemmArgs& a, int tile0, int tok0, int kb0, int tid) {
    using C = Cfg<FMT>;
    constexpr int KS = C::KS;
    const int c = kb0 >> 4, j0 = kb0 & 15;
    const int nch = a.K >> 9;
    const int lt0 = tile0 * (8 / (2 * RPL));  // first layout tile of this block
#pragma unroll
    for (int i = 0; i < KS / 2; ++i) {
        int kb, h, r, lto, row8, tile8;
        code_unit<KS, RPL>(tid + i * NT, kb, h, r, lto, row8, tile8);
        const size_t tc = gemv::tc_index(lt0 + lto, c, nch, a.ntiles, a.cm);
        S.wq[i] = gemv::ldg_nc_v4(a.codes + (tc * RPL + r) * 512 + (h * 16 + j0 + kb) * 16);
    }
    if (tid < 32 * KS * (4 / RPL)) {
        int kb, h, lto, row8b, tile8;
        scale_unit<KS, RPL>(tid, kb, h, lto, row8b, tile8);
        const size_t tc = gemv::tc_index(lt0 + lto, c, nch, a.ntiles, a.cm);
        const int lane = h * 16 + j0 + kb;
        if (FMT == gemv::FAST_K5) {
            const uint8_t* sp = a.sc + ((tc * 32 + lane) * RPL) * 2;
            if (RPL == 4) { uint2 v = gemv::ldg_nc_v2(sp); S.ws[0] = v.x; S.ws[1 % (RPL / 2)] = v.y; }
            else S.ws[0] = gemv::ldg_nc_u32(sp);
            const uint32_t* dp = (const uint32_t*)a.d + ((tc * 2 + h) * 2 + ((j0 + kb) >> 3)) * RPL;
            if (RPL == 4) { int4 v = gemv::ldg_nc_v4(dp); S.wd[0] = v.x; S.wd[1] = v.y; S.wd[2 % RPL] = v.z; S.wd[3 % RPL] = v.w; }
            else { uint2 v = gemv::ldg_nc_v2(dp); S.wd[0] = v.x; S.wd[1] = v.y; }
        } else {
            if (RPL == 4) { uint2 v = gemv::ldg_nc_v2(a.d + (tc * 32 + lane) * 4); S.ws[0] = v.x; S.ws[1 % (RPL / 2)] = v.y; }
            else S.ws[0] = gemv::ldg_nc_u32(a.d + (tc * 32 + lane) * 2);
            if (FMT == gemv::FAST_P4M) {
                const uint16_t* mp = (const uint16_t*)a.sc + (tc * 32 + lane) * RPL;
                if (RPL == 4) { uint2 v = gemv::ldg_nc_v2(mp); S.wm[0] = v.x; S.wm[1 % (RPL / 2)] = v.y; }
                else S.wm[0] = gemv::ldg_nc_u32(mp);
            }
        }
    }
    if (FMT == gemv::FAST_K5) {  // KS = 2: 256 units, one per thread
        int kb, h, r, lto, row8, tile8;
        code_unit<KS, RPL>(tid, kb, h, r, lto, row8, tile8);
        const size_t tc = gemv::tc_index(lt0 + lto, c, nch, a.ntiles, a.cm);
        S.wh = gemv::ldg_nc_u32(a.qh + (tc * RPL + r) * 128 + (h * 16 + j0 + kb) * 4);
    }
#pragma unroll
    for (int i = 0; i < C::N_XA; ++i) {
        const int U = tid + i * NT, tok = U / C::UPR, u = U % C::UPR;
        S.xa[i] = __ldg((const int4*)(a.xq + (size_t)(tok0 + tok) * a.K + kb0 * 32 + u * 16));
    }
    if (tid < KS * BN / 2) {
        const int kb = tid / (BN / 2), p = tid % (BN / 2);
        S.mt = __ldg((const float4*)(a.xs + (size_t)(kb0 + kb) * a.Tp + tok0) + p);
    }
    if (C::HASB && tid < KS * BN / 4) {
        const int kb = tid / (BN / 4), p = tid % (BN / 4);
        S.xs = __ldg((const float4*)(a.xsum + (size_t)(kb0 + kb) * a.Tp + tok0) + p);
    }
}

template <int FMT, int RPL>
__device__ __forceinline__ void stage_store(const Stage<FMT, RPL>& S, unsigned char* buf, int tid) {
    using C = Cfg<FMT>;
    constexpr int KS = C::KS;
#pragma unroll
    for (int i = 0; i < KS / 2; ++i) {
        int kb, h, r, lto, row8, tile8;
        code_unit<KS, RPL>(tid + i * NT, kb, h, r, lto, row8, tile8);
        *(int4*)(buf + C::O_WQ + ((tile8 * KS + kb) * 8 + row8) * 16) = S.wq[i];
    }
    if (tid < 32 * KS * (4 / RPL)) {
        int kb, h, lto, row8b, tile8;
        scale_unit<KS, RPL>(tid, kb, h, lto, row8b, tile8);
        float* wa = (float*)(buf + C::O_WA);
        float* wb = (float*)(buf + C::O_WB);
#pragma unroll
        for (int r = 0; r < RPL; ++r) {
            const int idx = (kb * 8 + row8b + r) * 16 + tile8;
            const uint32_t w2 = S.ws[r >> 1];
            if (FMT == gemv::FAST_K5) {
                const uint32_t scm = w2 >> (16 * (r & 1));
                const float sc = (float)(scm & 0xff), mn = (float)((scm >> 8) & 0xff);
                const uint32_t dd = S.wd[r];
                const float d = h2f_u16(dd), dmin = h2f_u16(dd >> 16);
                const float A = d * sc;
                wa[idx] = 2.f * A;
                wb[idx] = 16.f * A - dmin * mn;
            } else {
                const float dw = h2f_u16(w2 >> (16 * (r & 1)));
                wa[idx] = dw;
                if (FMT == gemv::FAST_P4M) wb[idx] = 8.f * dw + h2f_u16(S.wm[r >> 1] >> (16 * (r & 1)));
            }
        }
    }
    if (FMT == gemv::FAST_K5) {
        int kb, h, r, lto, row8, tile8;
        code_unit<KS, RPL>(tid, kb, h, r, lto, row8, tile8);
        ((uint32_t*)(buf + C::O_WH))[(tile8 * KS + kb) * 8 + row8] = S.wh;
    }
#pragma unroll
    for (int i = 0; i < C::N_XA; ++i) {
        const int U = tid + i * NT, tok = U / C::UPR, u = U % C::UPR;
        *(int4*)(buf + C::O_XA + tok * (C::UPR * 16) + xswz<C::UPR>(tok, u) * 16) = S.xa[i];
    }
    if (tid < KS * BN / 2) ((float4*)(buf + C::O_MT))[tid] = S.mt;
    if (C::HASB && tid < KS * BN / 4) ((float4*)(buf + C::O_XS))[tid] = S.xs;
}

#ifndef T4Q_GEMM_KBU
#define T4Q_GEMM_KBU 2
#endif
constexpr int GEMM_KBU = T4Q_GEMM_KBU;  // kb-loop unroll inside a stage (register pressure vs smem latency)
// EPI 2: correct epilogue. EPI 1 (timing only): one FFMA per output per block. EPI 0 (timing only): int32
// accumulation over the whole K (what a per-row-scaled W8A8 GEMM would cost).
// EPI 3: int4 tensor path (P4/P4M only), activations from quant_rows_i4 (xq holds {XH, XL} 16 B each per block).
// EPI 4 (timing only): one int4 mma per block (the cost of a W4A4 GEMM).
template <int FMT, int RPL, int EPI = 2>
__global__ void __launch_bounds__(NT, 1) gemm_w4a8_kernel(const GemmArgs a) {
    using C = Cfg<FMT>;
    constexpr int KS = C::KS;
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp & 1, wn = warp >> 1;
    const int tile0 = blockIdx.x * 16;  // in 8-row groups
    const int tok0 = blockIdx.y * BN;
    const int nkb = a.K >> 5;
    const int t4 = lane & 3;

    float acc[8][4][2];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int g = 0; g < 4; ++g) acc[i][g][0] = acc[i][g][1] = 0.f;

    Stage<FMT, RPL> S;
    stage_load<FMT, RPL>(S, a, tile0, tok0, 0, tid);
    stage_store<FMT, RPL>(S, smem, tid);
    __syncthreads();

    int buf = 0;
    for (int kb0 = 0; kb0 < nkb; kb0 += KS) {
        const bool more = kb0 + KS < nkb;
        if (more) stage_load<FMT, RPL>(S, a, tile0, tok0, kb0 + KS, tid);
        const unsigned char* B = smem + buf * C::BYTES;
#pragma unroll GEMM_KBU
        for (int kb = 0; kb < KS; ++kb) {
            // B fragments: 4 token groups x {lo, hi}
            uint32_t bf[4][2];
#pragma unroll
            for (int gp = 0; gp < 2; ++gp) {
                const int j = lane >> 3;
                const int g = gp * 2 + (j >> 1);
                const int tok = wn * 32 + 8 * g + (lane & 7);
                const int u = 2 * kb + (j & 1);
                ldsm_x4(bf[gp * 2][0], bf[gp * 2][1], bf[gp * 2 + 1][0], bf[gp * 2 + 1][1],
                        B + C::O_XA + tok * (C::UPR * 16) + xswz<C::UPR>(tok, u) * 16);
            }
            float4 mt[4];
            float2 sx[4];
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                mt[g] = *(const float4*)(B + C::O_MT + (kb * BN + wn * 32 + 8 * g + 2 * t4) * 8);
                if (C::HASB) sx[g] = *(const float2*)(B + C::O_XS + (kb * BN + wn * 32 + 8 * g + 2 * t4) * 4);
            }
            // A fragments: 8 tiles of this warp (rows wm*64 ..), one 32-block
            uint32_t af[8];
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int tile = wm * 8 + q * 4 + (lane >> 3);
                ldsm_x4(af[q * 4], af[q * 4 + 1], af[q * 4 + 2], af[q * 4 + 3],
                        B + C::O_WQ + ((tile * KS + kb) * 8 + (lane & 7)) * 16);
            }
            float wa[8], wb[8];
            {
                const float4* p = (const float4*)(B + C::O_WA + ((kb * 8 + (lane >> 2)) * 16 + wm * 8) * 4);
                const float4 v0 = p[0], v1 = p[1];
                wa[0] = v0.x; wa[1] = v0.y; wa[2] = v0.z; wa[3] = v0.w;
                wa[4] = v1.x; wa[5] = v1.y; wa[6] = v1.z; wa[7] = v1.w;
                if (C::HASB) {
                    const float4* pb = (const float4*)(B + C::O_WB + ((kb * 8 + (lane >> 2)) * 16 + wm * 8) * 4);
                    const float4 u0 = pb[0], u1 = pb[1];
                    wb[0] = u0.x; wb[1] = u0.y; wb[2] = u0.z; wb[3] = u0.w;
                    wb[4] = u1.x; wb[5] = u1.y; wb[6] = u1.z; wb[7] = u1.w;
                }
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                uint32_t lo, hi;
                const uint32_t q = af[i];
                if (EPI >= 3) {
#pragma unroll
                    for (int g = 0; g < 4; ++g) {
                        int d0, d1;
                        if (EPI == 3) {
                            mma_u4s4(d0, d1, q, bf[g][0], 0, 0);
                            mma_u4u4(d0, d1, q, bf[g][1], (d0 << 4) + MAGIC_I, (d1 << 4) + MAGIC_I);
                        } else {
                            mma_u4s4(d0, d1, q, bf[g][0], MAGIC_I, MAGIC_I);
                        }
                        const float t0 = fmaf(__int_as_float(d0), mt[g].x, mt[g].y);
                        const float t1 = fmaf(__int_as_float(d1), mt[g].z, mt[g].w);
                        acc[i][g][0] = fmaf(wa[i], t0, acc[i][g][0]);
                        acc[i][g][1] = fmaf(wa[i], t1, acc[i][g][1]);
                        if (C::HASB) {
                            acc[i][g][0] = fmaf(wb[i], sx[g].x, acc[i][g][0]);
                            acc[i][g][1] = fmaf(wb[i], sx[g].y, acc[i][g][1]);
                        }
                    }
                    continue;
                }
                if (C::K5) {
                    const int tile = wm * 8 + i;
                    const uint32_t H = ((const uint32_t*)(B + C::O_WH))[(tile * KS + kb) * 8 + (lane >> 2)] >> t4;
                    lo = ((q << 3) & 0x78787878u) | ((H << 7) & 0x80808080u);
                    hi = ((q >> 1) & 0x78787878u) | ((H << 3) & 0x80808080u);
                } else {
                    lo = (q << 4) & 0xF0F0F0F0u;
                    hi = q & 0xF0F0F0F0u;
                }
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    int d0, d1;
                    if (EPI == 0) {
                        mma_u8s8(d0, d1, lo, bf[g][0], __float_as_int(acc[i][g][0]), __float_as_int(acc[i][g][1]));
                        mma_u8s8(d0, d1, hi, bf[g][1], d0, d1);
                        acc[i][g][0] = __int_as_float(d0); acc[i][g][1] = __int_as_float(d1);
                        continue;
                    }
                    mma_u8s8(d0, d1, lo, bf[g][0], MAGIC_I, MAGIC_I);
                    mma_u8s8(d0, d1, hi, bf[g][1], d0, d1);
                    if (EPI == 1) {
                        acc[i][g][0] = fmaf(__int_as_float(d0), mt[g].x, acc[i][g][0]);
                        acc[i][g][1] = fmaf(__int_as_float(d1), mt[g].z, acc[i][g][1]);
                        continue;
                    }
                    const float t0 = fmaf(__int_as_float(d0), mt[g].x, mt[g].y);
                    const float t1 = fmaf(__int_as_float(d1), mt[g].z, mt[g].w);
                    acc[i][g][0] = fmaf(wa[i], t0, acc[i][g][0]);
                    acc[i][g][1] = fmaf(wa[i], t1, acc[i][g][1]);
                    if (C::HASB) {
                        acc[i][g][0] = fmaf(wb[i], sx[g].x, acc[i][g][0]);
                        acc[i][g][1] = fmaf(wb[i], sx[g].y, acc[i][g][1]);
                    }
                }
            }
        }
        if (more) {
            stage_store<FMT, RPL>(S, smem + (buf ^ 1) * C::BYTES, tid);
            __syncthreads();
            buf ^= 1;
        }
    }
    // store: row = (tile0 + wm*8 + i)*8 + lane/4, token = tok0 + wn*32 + 8g + 2*t4 + {0,1}
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = (tile0 + wm * 8 + i) * 8 + (lane >> 2);
#pragma unroll
        for (int g = 0; g < 4; ++g)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int tok = tok0 + wn * 32 + 8 * g + 2 * t4 + e;
                if (tok < a.T && row < a.N) {
                    float* p = a.y + (size_t)tok * a.ldy + row;
                    *p = a.accumulate ? *p + acc[i][g][e] : acc[i][g][e];
                }
            }
    }
}

template <int FMT, int RPL, int EPI = 2>
static cudaError_t gemm_launch(const GemmArgs& a, cudaStream_t s) {
    auto k = gemm_w4a8_kernel<FMT, RPL, EPI>;
    const int smem = 2 * Cfg<FMT>::BYTES;
    static int attr_dev_mask = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    if (!(attr_dev_mask & (1 << dev))) {
        cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return e;
        attr_dev_mask |= 1 << dev;
    }
    if (a.N % BM || a.K % (32 * Cfg<FMT>::KS) || a.Tp % BN) return cudaErrorInvalidValue;
    dim3 grid(a.N / BM, a.Tp / BN);
    k<<<grid, NT, smem, s>>>(a);
    return cudaGetLastError();
}

// fmt / rpl from the weight's gemv::Layout (rpl 2 or 4)
static inline cudaError_t gemm_launch_fmt(int fmt, int rpl, const GemmArgs& a, cudaStream_t s) {
    if (rpl == 4) {
        if (fmt == gemv::FAST_P4) return gemm_launch<gemv::FAST_P4, 4>(a, s);
        if (fmt == gemv::FAST_P4M) return gemm_launch<gemv::FAST_P4M, 4>(a, s);
        if (fmt == gemv::FAST_K5) return gemm_launch<gemv::FAST_K5, 4>(a, s);
    } else if (rpl == 2) {
        if (fmt == gemv::FAST_P4) return gemm_launch<gemv::FAST_P4, 2>(a, s);
        if (fmt == gemv::FAST_P4M) return gemm_launch<gemv::FAST_P4M, 2>(a, s);
        if (fmt == gemv::FAST_K5) return gemm_launch<gemv::FAST_K5, 2>(a, s);
    }
    return cudaErrorInvalidValue;
}

// ================================================================================================ fp16 HMMA variant
// Same weight layout (P4 only), weights dequantized to fp16 in registers ((1024 + c) magic - 1032, times d: one
// rounding, rel <= 2^-11), activations fp16, mma.sync.m16n8k8 with fp32 accumulation, no per-block epilogue.
// Activations xh [Tp][K] fp16, permuted within each 32-block: slot p = 8j + 2t + e holds logical
// k = 16*(j>>1) + 4t + 2*(j&1) + e (so that a ldmatrix of slots 8j..8j+7 is the B fragment of the j-th k8 mma, and
// the j-th mma's A fragment comes from bytes 4t..4t+3 of the Q4_0 group that ldmatrix gives the same thread).
// Block 128 rows x 128 tokens, 8 warps = 4 (rows) x 2 (tokens), warp tile 32 x 64, KS = 2 blocks per stage.
struct F16Cfg {
    static constexpr int KS = 2;
    static constexpr int O_WQ = 0;                       // [16 tile8][KS][8 rows][16 B]
    static constexpr int O_WD = 16 * KS * 128;           // [KS][128 rows] fp16
    static constexpr int O_XA = O_WD + KS * 128 * 2;     // [128 tok][KS*32 halves], 16-B units swizzled
    static constexpr int BYTES = O_XA + BN * KS * 64;
};

template <int RPL>
struct F16Stage {
    int4 wq;
    uint32_t ws[RPL / 2];
    int4 xa[4];
};

__device__ __forceinline__ void mma_f16(float* c, uint32_t a0, uint32_t a1, uint32_t b) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b));
}

template <int RPL>
__device__ __forceinline__ void f16_stage_load(F16Stage<RPL>& S, const GemmArgs& a, const __half* xh, int tile0, int tok0,
                                               int kb0, int tid) {
    constexpr int KS = F16Cfg::KS;
    const int c = kb0 >> 4, j0 = kb0 & 15, nch = a.K >> 9;
    const int lt0 = tile0 * (8 / (2 * RPL));
    {
        int kb, h, r, lto, row8, tile8;
        code_unit<KS, RPL>(tid, kb, h, r, lto, row8, tile8);
        const size_t tc = gemv::tc_index(lt0 + lto, c, nch, a.ntiles, a.cm);
        S.wq = gemv::ldg_nc_v4(a.codes + (tc * RPL + r) * 512 + (h * 16 + j0 + kb) * 16);
    }
    if (tid < 32 * KS * (4 / RPL)) {
        int kb, h, lto, row8b, tile8;
        scale_unit<KS, RPL>(tid, kb, h, lto, row8b, tile8);
        const size_t tc = gemv::tc_index(lt0 + lto, c, nch, a.ntiles, a.cm);
        const int lane = h * 16 + j0 + kb;
        if (RPL == 4) { uint2 v = gemv::ldg_nc_v2(a.d + (tc * 32 + lane) * 4); S.ws[0] = v.x; S.ws[1 % (RPL / 2)] = v.y; }
        else S.ws[0] = gemv::ldg_nc_u32(a.d + (tc * 32 + lane) * 2);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int U = tid + i * NT, tok = U >> 3, u = U & 7;
        S.xa[i] = __ldg((const int4*)(xh + (size_t)(tok0 + tok) * a.K + kb0 * 32) + u);
    }
}

template <int RPL>
__device__ __forceinline__ void f16_stage_store(const F16Stage<RPL>& S, unsigned char* buf, int tid) {
    constexpr int KS = F16Cfg::KS;
    {
        int kb, h, r, lto, row8, tile8;
        code_unit<KS, RPL>(tid, kb, h, r, lto, row8, tile8);
        *(int4*)(buf + F16Cfg::O_WQ + ((tile8 * KS + kb) * 8 + row8) * 16) = S.wq;
    }
    if (tid < 32 * KS * (4 / RPL)) {
        int kb, h, lto, row8b, tile8;
        scale_unit<KS, RPL>(tid, kb, h, lto, row8b, tile8);
        uint16_t* wd = (uint16_t*)(buf + F16Cfg::O_WD);
#pragma unroll
        for (int r = 0; r < RPL; ++r) wd[kb * 128 + tile8 * 8 + row8b + r] = (uint16_t)(S.ws[r >> 1] >> (16 * (r & 1)));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int U = tid + i * NT, tok = U >> 3, u = U & 7;
        *(int4*)(buf + F16Cfg::O_XA + tok * 128 + (u ^ (tok & 7)) * 16) = S.xa[i];
    }
}

template <int RPL>
__global__ void __launch_bounds__(NT, 1) gemm_f16_kernel(const GemmArgs a, const __half* __restrict__ xh) {
    constexpr int KS = F16Cfg::KS;
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp & 3, wn = warp >> 2;
    const int tile0 = blockIdx.x * 16, tok0 = blockIdx.y * BN;
    const int nkb = a.K >> 5;
    const int g = lane >> 2, t4 = lane & 3;
    float acc[2][8][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int n = 0; n < 8; ++n) acc[m][n][0] = acc[m][n][1] = acc[m][n][2] = acc[m][n][3] = 0.f;
    F16Stage<RPL> S;
    f16_stage_load<RPL>(S, a, xh, tile0, tok0, 0, tid);
    f16_stage_store<RPL>(S, smem, tid);
    __syncthreads();
    int buf = 0;
    const __half2 k1032 = __float2half2_rn(1032.f);
    for (int kb0 = 0; kb0 < nkb; kb0 += KS) {
        const bool more = kb0 + KS < nkb;
        if (more) f16_stage_load<RPL>(S, a, xh, tile0, tok0, kb0 + KS, tid);
        const unsigned char* B = smem + buf * F16Cfg::BYTES;
#pragma unroll
        for (int kb = 0; kb < KS; ++kb) {
            uint32_t q[4];
            ldsm_x4(q[0], q[1], q[2], q[3], B + F16Cfg::O_WQ + (((wm * 4 + (lane >> 3)) * KS + kb) * 8 + (lane & 7)) * 16);
            uint32_t wv[4][4];
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const uint16_t db = ((const uint16_t*)(B + F16Cfg::O_WD))[kb * 128 + (wm * 4 + m) * 8 + g];
                const __half2 d2 = __halves2half2(__ushort_as_half(db), __ushort_as_half(db));
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t x = __byte_perm(q[m], 0, (j & 1) ? 0x4342 : 0x4140);
                    const uint32_t v = (j >= 2 ? ((x >> 4) & 0x000F000Fu) : (x & 0x000F000Fu)) | 0x64006400u;
                    __half2 hv = *(const __half2*)&v;
                    hv = __hmul2(__hsub2(hv, k1032), d2);
                    wv[m][j] = *(uint32_t*)&hv;
                }
            }
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int tok = wn * 64 + n * 8 + (lane & 7);
                uint32_t b[4];
                const int u = kb * 4 + (lane >> 3);
                ldsm_x4(b[0], b[1], b[2], b[3], B + F16Cfg::O_XA + tok * 128 + (u ^ (tok & 7)) * 16);
#pragma unroll
                for (int mg = 0; mg < 2; ++mg)
#pragma unroll
                    for (int j = 0; j < 4; ++j) mma_f16(acc[mg][n], wv[mg * 2][j], wv[mg * 2 + 1][j], b[j]);
            }
        }
        if (more) {
            f16_stage_store<RPL>(S, smem + (buf ^ 1) * F16Cfg::BYTES, tid);
            __syncthreads();
            buf ^= 1;
        }
    }
#pragma unroll
    for (int mg = 0; mg < 2; ++mg)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int row = tile0 * 8 + wm * 32 + mg * 16 + hh * 8 + g;
#pragma unroll
            for (int n = 0; n < 8; ++n)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int tok = tok0 + wn * 64 + n * 8 + 2 * t4 + e;
                    if (tok < a.T && row < a.N) {
                        float* p = a.y + (size_t)tok * a.ldy + row;
                        *p = a.accumulate ? *p + acc[mg][n][hh * 2 + e] : acc[mg][n][hh * 2 + e];
                    }
                }
        }
}

template <int RPL>
static cudaError_t gemm_f16_launch(const GemmArgs& a, const __half* xh, cudaStream_t s) {
    auto k = gemm_f16_kernel<RPL>;
    const int smem = 2 * F16Cfg::BYTES;
    if (a.N % BM || a.K % 64 || a.Tp % BN) return cudaErrorInvalidValue;
    dim3 grid(a.N / BM, a.Tp / BN);
    k<<<grid, NT, smem, s>>>(a, xh);
    return cudaGetLastError();
}

// fp32 x [T][K] -> fp16 xh [Tp][K] in the permuted per-32 order above (zeros for padding tokens)
__global__ void to_f16_perm_kernel(const float* __restrict__ x, int ldx, int T, int Tp, int K, __half* __restrict__ xh) {
    const int nb = K >> 5;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Tp * nb) return;
    const int t = i / nb, b = i % nb;
    uint32_t w[16];
    if (t >= T) {
#pragma unroll
        for (int k = 0; k < 16; ++k) w[k] = 0;
    } else {
        const float* src = x + (size_t)t * ldx + b * 32;
#pragma unroll
        for (int p = 0; p < 32; p += 2) {
            const int j = p >> 3, tt = (p >> 1) & 3;
            const int k = 16 * (j >> 1) + 4 * tt + 2 * (j & 1);
            const __half2 h = __floats2half2_rn(src[k], src[k + 1]);
            w[p >> 1] = *(const uint32_t*)&h;
        }
    }
    int4* dst = (int4*)(xh + (size_t)t * K + b * 32);
#pragma unroll
    for (int k = 0; k < 4; ++k) dst[k] = make_int4(w[4 * k], w[4 * k + 1], w[4 * k + 2], w[4 * k + 3]);
}

// prefill activation quantizer: x [T][K] fp32 (row stride ldx) -> xq [Tp][K], xs [K/32][Tp], xsum [K/32][Tp].
// One thread per (token, 32-block); tokens T..Tp-1 are written as zeros.
__global__ void quant_rows_kernel(const float* __restrict__ x, int ldx, int T, int Tp, int K, int8_t* __restrict__ xq,
                                  float2* __restrict__ xs, float* __restrict__ xsum) {
    const int nb = K >> 5;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Tp * nb) return;
    const int t = i / nb, b = i % nb;
    int4* dst = (int4*)(xq + (size_t)t * K + b * 32);
    if (t >= T) {
        dst[0] = make_int4(0, 0, 0, 0); dst[1] = make_int4(0, 0, 0, 0);
        xs[(size_t)b * Tp + t] = make_float2(0.f, 0.f);
        xsum[(size_t)b * Tp + t] = 0.f;
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
    int sq = 0;
    uint32_t w[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        uint32_t pk = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int q = amax == 0.f ? 0 : __float2int_rn(v[4 * k + e] / d);
            sq += q;
            pk |= (uint32_t)(q & 0xff) << (8 * e);
        }
        w[k] = pk;
    }
    dst[0] = make_int4(w[0], w[1], w[2], w[3]);
    dst[1] = make_int4(w[4], w[5], w[6], w[7]);
    const float s = d * 0.0625f;
    xs[(size_t)b * Tp + t] = make_float2(s, -(float)(12582912 + 128 * sq) * s);
    xsum[(size_t)b * Tp + t] = d * (float)sq;
}

// int4-path activation quantizer: same q8 values, stored per 32-block as {XH[16], XL[16]} (Q4_0-packed signed high
// nibbles, unsigned low nibbles); meta {d, -(M + 8*sumq)*d}, xsum d*sumq.
__global__ void quant_rows_i4_kernel(const float* __restrict__ x, int ldx, int T, int Tp, int K, int8_t* __restrict__ xq,
                                     float2* __restrict__ xs, float* __restrict__ xsum) {
    const int nb = K >> 5;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Tp * nb) return;
    const int t = i / nb, b = i % nb;
    int4* dst = (int4*)(xq + (size_t)t * K + b * 32);
    if (t >= T) {
        dst[0] = make_int4(0, 0, 0, 0); dst[1] = make_int4(0, 0, 0, 0);
        xs[(size_t)b * Tp + t] = make_float2(0.f, 0.f);
        xsum[(size_t)b * Tp + t] = 0.f;
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
    int sq = 0, q[32];
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        q[k] = amax == 0.f ? 0 : __float2int_rn(v[k] / d);
        sq += q[k];
    }
    uint32_t h[4], l[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        uint32_t hw = 0, lw = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int a = q[4 * w + e], c = q[4 * w + e + 16];
            hw |= (uint32_t)(((a >> 4) & 15) | (((c >> 4) & 15) << 4)) << (8 * e);
            lw |= (uint32_t)((a & 15) | ((c & 15) << 4)) << (8 * e);
        }
        h[w] = hw; l[w] = lw;
    }
    dst[0] = make_int4(h[0], h[1], h[2], h[3]);
    dst[1] = make_int4(l[0], l[1], l[2], l[3]);
    xs[(size_t)b * Tp + t] = make_float2(d, -(float)(12582912 + 8 * sq) * d);
    xsum[(size_t)b * Tp + t] = d * (float)sq;
}

static inline void quant_rows(const float* x, int ldx, int T, int Tp, int K, int8_t* xq, float2* xs, float* xsum,
                              cudaStream_t s) {
    const int n = Tp * (K >> 5);
    quant_rows_kernel<<<(n + 127) / 128, 128, 0, s>>>(x, ldx, T, Tp, K, xq, xs, xsum);
}

}  // namespace gemm
}  // namespace t4q
#endif
