// t4q/src/kernels/gemv.cuh -- fast decode GEMV for sm_75 (M0 owner: gemv fast path).
//
// Repacked, lossless SoA weight formats + q8_1 int8 activations + dp4a, m = 1..8 columns.
//   P4 : GGUF Q4_0   (codes 16 B / 32 w, fp16 d per 32)              18 B / 32 w  (same bytes as GGUF)
//   Q8 : GGUF Q8_0   (codes 32 B / 32 w, fp16 d per 32)              34 B / 32 w
//   K6 : GGUF Q6_K   (ql 16 B + qh 8 B / 32 w, int8 sc per 16, fp16 d per 256) 210 B / 256 w
//
// Layout (all formats). A warp owns a "tile" of 2*RPL consecutive output rows. Half-warp h (lane>>4)
// owns rows tile*2*RPL + h*RPL + r (r < RPL); lane j = lane&15 owns 32-weight group kb = c*16 + j of
// chunk c (one chunk = 16 groups = 512 weights, so K must be a multiple of 512: 3072, 5120, 8704 ok).
// Code planes are stored [tile][chunk][r][(part)][lane][16 B], so every warp load instruction reads
// 512 contiguous bytes. Scale planes are [tile][chunk][lane][r] (2*RPL bytes per lane per chunk).
// Accumulation order per output column is fixed and independent of m, RPL, D and XSM, so every
// variant returns bit-identical results (spec verify == plain decode).
//
// Activations: q8_1-like per 32: int8 xq[col][K] (natural order) + int2 xm[col][K/32] =
//   { float_as_int(d), (s0 & 0xffff) | (s1 << 16) } where s0 = sum q[0..15], s1 = sum q[16..31].
//
// Host-side helpers (fp16 conversion, GGUF dequant, host repack, CPU reference) compile with plain g++.
//
// Engine usage (measured in M0, research/m0_results.md): repack P4 tensors with RPL=4 (sustained m=4 +20-25% on a
// power-throttled T4, m=1 within 1-2%), K6 with RPL=2; launch gemv_fast_launch<FMT, RPL, M, K/512, 1, false> with
// 256 threads and grid = min(ceil(ntiles / 8), occupancy * numSMs) (persistent); XSM=true only for m >= 6.
// Burst on Kaggle T4: P4 m=1 ~258-263 GB/s, m=4 ~250-258, K6 lm_head m=1 265 GB/s.
#pragma once
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <cmath>

namespace t4q {
namespace gemv {

// FAST_P4M (GGUF Q4_1) and FAST_K5 (GGUF Q5_K) added in M2:
//   P4M: P4 codes/d planes + fp16 m plane (stored at off_sc, same shape as d). w = d*q + m, q in 0..15.
//   K5 : 16 B low nibbles per 32 w (Q4_0 order) + 4 B high bits per 32 w ([tc][r][lane][4 B]; element 4t+jj of
//        the low half at bit 8jj+t, of the high half at bit 8jj+4+t) + 2 B {sc, m} (decoded 6-bit) per 32 w at
//        off_sc (same addressing as K6 sc) + {d, dmin} fp16 pair per 256 w (u32, addressing as K6 d).
//        w = d*sc*q - dmin*m, q in 0..31. 180 B / 256 w (GGUF 176).
enum FastFmt { FAST_P4 = 0, FAST_Q8 = 1, FAST_K6 = 2, FAST_P4M = 3, FAST_K5 = 4 };

static inline const char* fmt_name(int f) {
    return f == FAST_P4 ? "P4" : f == FAST_Q8 ? "Q8" : f == FAST_K6 ? "K6" : f == FAST_P4M ? "P4M" : "K5";
}

// GGUF block geometry of the source type for each packed format
static inline int src_block_elems(int f) { return (f == FAST_K6 || f == FAST_K5) ? 256 : 32; }
static inline int src_block_bytes(int f) {
    return f == FAST_P4 ? 18 : f == FAST_Q8 ? 34 : f == FAST_K6 ? 210 : f == FAST_P4M ? 20 : 176;
}
static inline size_t src_row_bytes(int f, int K) { return (size_t)(K / src_block_elems(f)) * src_block_bytes(f); }

// ------------------------------------------------------------------------------------------------ layout
struct Layout {
    int fmt = 0, N = 0, K = 0, rpl = 1;
    int cm = 0;  // 1: chunk-major plane order ([chunk][tile] instead of [tile][chunk]; same bytes, other addresses)
    int ntiles = 0, nch = 0;
    size_t off_codes = 0, off_qh = 0, off_sc = 0, off_d = 0, bytes = 0;
};

static inline size_t align256(size_t x) { return (x + 255) & ~(size_t)255; }
#ifdef __CUDACC__
__host__ __device__
#endif
static inline size_t tc_index(int tile, int c, int nch, int ntiles, int cm) {
    return cm ? (size_t)c * ntiles + tile : (size_t)tile * nch + c;
}

static inline Layout make_layout(int fmt, int N, int K, int rpl, int cm = 0) {
    Layout L;
    L.fmt = fmt; L.N = N; L.K = K; L.rpl = rpl; L.cm = cm;
    L.ntiles = (N + 2 * rpl - 1) / (2 * rpl);
    L.nch = K / 512;
    size_t tc = (size_t)L.ntiles * L.nch;
    size_t codes = 0, qh = 0, sc = 0, d = 0;
    if (fmt == FAST_P4) { codes = tc * rpl * 512; d = tc * 32 * rpl * 2; }
    else if (fmt == FAST_Q8) { codes = tc * rpl * 1024; d = tc * 32 * rpl * 2; }
    else if (fmt == FAST_P4M) { codes = tc * rpl * 512; sc = tc * 32 * rpl * 2; d = tc * 32 * rpl * 2; }
    else if (fmt == FAST_K5) { codes = tc * rpl * 512; qh = tc * rpl * 128; sc = tc * 32 * rpl * 2; d = tc * 4 * rpl * 4; }
    else { codes = tc * rpl * 512; qh = tc * rpl * 256; sc = tc * 32 * rpl * 2; d = tc * 4 * rpl * 2; }
    L.off_codes = 0;
    L.off_qh = align256(codes);
    L.off_sc = L.off_qh + align256(qh);
    L.off_d = L.off_sc + align256(sc);
    L.bytes = L.off_d + align256(d);
    return L;
}

// ------------------------------------------------------------------------------------------------ fp16 (host)
static inline float h2f_host(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023, u;
    if (e == 0) {
        if (m == 0) u = s;
        else { float f = std::ldexp((float)m, -24); std::memcpy(&u, &f, 4); u |= s; }
    } else if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &u, 4); return f;
}
static inline uint16_t f2h_host(float f) {  // round to nearest even, normal range only (enough for scales)
    uint32_t u; std::memcpy(&u, &f, 4);
    uint32_t s = (u >> 16) & 0x8000; int e = (int)((u >> 23) & 255) - 127 + 15; uint32_t m = u & 0x7fffff;
    if (e <= 0) return (uint16_t)s;
    if (e >= 31) return (uint16_t)(s | 0x7c00);
    uint32_t r = (uint32_t)e << 10 | (m >> 13);
    uint32_t rem = m & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) r++;
    return (uint16_t)(s | r);
}

// ------------------------------------------------------------------------------------------------ GGUF dequant (host, natural order)
// Q6_K: natural 6-bit code q in 0..63 (w = d * sc[e/16] * (q - 32))
static inline void q6k_codes(const uint8_t* blk, uint8_t q[256]) {
    const uint8_t* ql = blk; const uint8_t* qh = blk + 128;
    for (int n = 0; n < 2; ++n)
        for (int l = 0; l < 32; ++l) {
            const uint8_t* L = ql + 64 * n; uint8_t H = qh[32 * n + l];
            q[128 * n + l]      = (L[l] & 15)      | (((H >> 0) & 3) << 4);
            q[128 * n + 32 + l] = (L[l + 32] & 15) | (((H >> 2) & 3) << 4);
            q[128 * n + 64 + l] = (L[l] >> 4)      | (((H >> 4) & 3) << 4);
            q[128 * n + 96 + l] = (L[l + 32] >> 4) | (((H >> 6) & 3) << 4);
        }
}

// Q5_K: natural 5-bit code q in 0..31 and decoded 6-bit sub-scales / mins (w = d*sc[e/32]*q - dmin*m[e/32])
static inline void q5k_scale_min(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
static inline void q5k_codes(const uint8_t* blk, uint8_t q[256], int sc[8], int mn[8]) {
    const uint8_t* scales = blk + 4; const uint8_t* qh = blk + 16; const uint8_t* qs = blk + 48;
    for (int s = 0; s < 8; ++s) {
        q5k_scale_min(s, scales, sc[s], mn[s]);
        for (int l = 0; l < 32; ++l) {
            const int lo = (s & 1) ? (qs[32 * (s >> 1) + l] >> 4) : (qs[32 * (s >> 1) + l] & 15);
            q[32 * s + l] = (uint8_t)(lo | (((qh[l] >> s) & 1) << 4));
        }
    }
}
// K5 high-bit word for one 32-weight group g[0..31] (5-bit codes)
static inline uint32_t k5_hbits(const uint8_t* g) {
    uint32_t H = 0;
    for (int t = 0; t < 4; ++t)
        for (int jj = 0; jj < 4; ++jj) {
            H |= (uint32_t)((g[4 * t + jj] >> 4) & 1) << (8 * jj + t);
            H |= (uint32_t)((g[16 + 4 * t + jj] >> 4) & 1) << (8 * jj + 4 + t);
        }
    return H;
}

static inline void dequant_row_host(int fmt, const uint8_t* row, int K, float* out) {
    if (fmt == FAST_P4) {
        for (int b = 0; b < K / 32; ++b) {
            const uint8_t* p = row + 18 * b; uint16_t dh; std::memcpy(&dh, p, 2); float d = h2f_host(dh);
            for (int j = 0; j < 16; ++j) {
                out[32 * b + j] = d * (float)((p[2 + j] & 15) - 8);
                out[32 * b + 16 + j] = d * (float)((p[2 + j] >> 4) - 8);
            }
        }
    } else if (fmt == FAST_Q8) {
        for (int b = 0; b < K / 32; ++b) {
            const uint8_t* p = row + 34 * b; uint16_t dh; std::memcpy(&dh, p, 2); float d = h2f_host(dh);
            for (int j = 0; j < 32; ++j) out[32 * b + j] = d * (float)(int8_t)p[2 + j];
        }
    } else if (fmt == FAST_P4M) {
        for (int b = 0; b < K / 32; ++b) {
            const uint8_t* p = row + 20 * b; uint16_t dh, mh; std::memcpy(&dh, p, 2); std::memcpy(&mh, p + 2, 2);
            const float d = h2f_host(dh), m = h2f_host(mh);
            for (int j = 0; j < 16; ++j) {
                out[32 * b + j] = d * (float)(p[4 + j] & 15) + m;
                out[32 * b + 16 + j] = d * (float)(p[4 + j] >> 4) + m;
            }
        }
    } else if (fmt == FAST_K5) {
        uint8_t q[256]; int sc[8], mn[8];
        for (int b = 0; b < K / 256; ++b) {
            const uint8_t* p = row + 176 * b; uint16_t dh, mh; std::memcpy(&dh, p, 2); std::memcpy(&mh, p + 2, 2);
            const float d = h2f_host(dh), dmin = h2f_host(mh);
            q5k_codes(p, q, sc, mn);
            for (int e = 0; e < 256; ++e) out[256 * b + e] = d * (float)sc[e / 32] * (float)q[e] - dmin * (float)mn[e / 32];
        }
    } else {
        uint8_t q[256];
        for (int b = 0; b < K / 256; ++b) {
            const uint8_t* p = row + 210 * b; uint16_t dh; std::memcpy(&dh, p + 208, 2); float d = h2f_host(dh);
            const int8_t* sc = (const int8_t*)(p + 192);
            q6k_codes(p, q);
            for (int e = 0; e < 256; ++e) out[256 * b + e] = d * (float)sc[e / 16] * (float)((int)q[e] - 32);
        }
    }
}

// ------------------------------------------------------------------------------------------------ host repack
// src: N rows of GGUF blocks (row stride src_row_bytes). dst: L.bytes (zeroed by this function).
static inline void repack_host(const Layout& L, const uint8_t* src, uint8_t* dst) {
    std::memset(dst, 0, L.bytes);
    const int rpl = L.rpl, nch = L.nch, K = L.K;
    const size_t rb = src_row_bytes(L.fmt, K);
    uint8_t* codes = dst + L.off_codes; uint8_t* qhp = dst + L.off_qh; uint8_t* scp = dst + L.off_sc;
    uint16_t* dp = (uint16_t*)(dst + L.off_d);
    for (int row = 0; row < L.N; ++row) {
        const uint8_t* s = src + (size_t)row * rb;
        const int tile = row / (2 * rpl), w = row % (2 * rpl), h = w / rpl, r = w % rpl;
        for (int kb = 0; kb < K / 32; ++kb) {
            const int c = kb / 16, j = kb % 16, lane = h * 16 + j;
            const size_t tc = tc_index(tile, c, nch, L.ntiles, L.cm);
            if (L.fmt == FAST_P4) {
                const uint8_t* b = s + 18 * kb;
                std::memcpy(codes + (tc * rpl + r) * 512 + lane * 16, b + 2, 16);
                std::memcpy(dp + (tc * 32 + lane) * rpl + r, b, 2);
            } else if (L.fmt == FAST_P4M) {
                const uint8_t* b = s + 20 * kb;
                std::memcpy(codes + (tc * rpl + r) * 512 + lane * 16, b + 4, 16);
                std::memcpy(dp + (tc * 32 + lane) * rpl + r, b, 2);
                std::memcpy((uint16_t*)scp + (tc * 32 + lane) * rpl + r, b + 2, 2);
            } else if (L.fmt == FAST_K5) {
                const int sb = kb / 8, sub = kb % 8;
                const uint8_t* b = s + 176 * sb;
                uint8_t q[256]; int sc[8], mn[8]; q5k_codes(b, q, sc, mn);
                const uint8_t* g = q + 32 * sub;
                uint8_t* ql = codes + (tc * rpl + r) * 512 + lane * 16;
                for (int i = 0; i < 16; ++i) ql[i] = (uint8_t)((g[i] & 15) | ((g[16 + i] & 15) << 4));
                const uint32_t H = k5_hbits(g);
                std::memcpy(qhp + (tc * rpl + r) * 128 + lane * 4, &H, 4);
                uint8_t* scd = scp + ((tc * 32 + lane) * rpl + r) * 2;
                scd[0] = (uint8_t)sc[sub]; scd[1] = (uint8_t)mn[sub];
                std::memcpy((uint8_t*)dp + ((((tc * 2 + h) * 2 + (j >> 3)) * rpl + r) * 4), b, 4);
            } else if (L.fmt == FAST_Q8) {
                const uint8_t* b = s + 34 * kb;
                for (int p = 0; p < 2; ++p)
                    std::memcpy(codes + ((tc * rpl + r) * 2 + p) * 512 + lane * 16, b + 2 + 16 * p, 16);
                std::memcpy(dp + (tc * 32 + lane) * rpl + r, b, 2);
            } else {
                const int sb = kb / 8, sub = kb % 8;
                const uint8_t* b = s + 210 * sb;
                uint8_t q[256]; q6k_codes(b, q);
                const uint8_t* g = q + 32 * sub;
                uint8_t* ql = codes + (tc * rpl + r) * 512 + lane * 16;
                for (int i = 0; i < 16; ++i) ql[i] = (uint8_t)((g[i] & 15) | ((g[16 + i] & 15) << 4));
                uint32_t H[2] = {0, 0};
                for (int half = 0; half < 2; ++half)
                    for (int t = 0; t < 4; ++t)
                        for (int jj = 0; jj < 4; ++jj)
                            H[half] |= (uint32_t)((g[16 * half + 4 * t + jj] >> 4) & 3) << (8 * jj + 2 * t);
                std::memcpy(qhp + (tc * rpl + r) * 256 + lane * 8, H, 8);
                uint8_t* scd = scp + ((tc * 32 + lane) * rpl + r) * 2;
                scd[0] = b[192 + 2 * sub]; scd[1] = b[192 + 2 * sub + 1];
                std::memcpy(dp + (((tc * 2 + h) * 2 + (j >> 3)) * rpl + r), b + 208, 2);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ q8_1 (host mirror)
// Mirrors quantize_q8_kernel exactly: d = amax/127, q = roundf(x/d).
static inline void quantize_q8_host(const float* x, int K, int8_t* xq, int32_t* xm /* 2 ints per block */) {
    for (int b = 0; b < K / 32; ++b) {
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[32 * b + i]));
        float d = amax / 127.f;
        int s0 = 0, s1 = 0;
        for (int i = 0; i < 32; ++i) {
            int q = amax == 0.f ? 0 : (int)std::roundf(x[32 * b + i] / d);
            xq[32 * b + i] = (int8_t)q;
            (i < 16 ? s0 : s1) += q;
        }
        std::memcpy(&xm[2 * b], &d, 4);
        xm[2 * b + 1] = (int32_t)((uint32_t)(s0 & 0xffff) | ((uint32_t)s1 << 16));
    }
}

}  // namespace gemv
}  // namespace t4q

// ================================================================================================ device
#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace t4q {
namespace gemv {

struct GemvArgs {
    const uint8_t* codes;   // L.off_codes
    const uint8_t* qh;      // K6 only
    const uint8_t* sc;      // K6 only
    const uint16_t* d;      // fp16 bits
    const int8_t* xq;       // [M][K]
    const int2* xm;         // [M][K/32]
    float* y;               // [M][ldy]
    int N, K, ntiles, ldy;
    int cm;                 // Layout::cm
};

static inline GemvArgs make_args(const Layout& L, const uint8_t* dev_base, const int8_t* xq, const int2* xm, float* y,
                                 int ldy) {
    GemvArgs a;
    a.codes = dev_base + L.off_codes; a.qh = dev_base + L.off_qh; a.sc = dev_base + L.off_sc;
    a.d = (const uint16_t*)(dev_base + L.off_d);
    a.xq = xq; a.xm = xm; a.y = y; a.N = L.N; a.K = L.K; a.ntiles = L.ntiles; a.ldy = ldy; a.cm = L.cm;
    return a;
}

__device__ __forceinline__ int4 ldg_nc_v4(const void* p) {
    int4 v;
    asm volatile("ld.global.nc.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ uint2 ldg_nc_v2(const void* p) {
    uint2 v;
    asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}
__device__ __forceinline__ uint32_t ldg_nc_u32(const void* p) {
    uint32_t v; asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v;
}
__device__ __forceinline__ uint32_t ldg_nc_u16(const void* p) {
    unsigned short v; asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p)); return v;
}
__device__ __forceinline__ float h2f_bits(uint32_t b) { return __half2float(__ushort_as_half((unsigned short)(b & 0xffff))); }

template <int FMT, int RPL>
struct WChunk {
    int4 q[RPL][FMT == FAST_Q8 ? 2 : 1];
    uint2 hb[RPL];                 // K6 high bits
    uint32_t dd[(RPL + 1) / 2];    // RPL fp16 scales (P4/Q8/P4M: per 32; K6: per 256), two per word
    uint32_t sc[(RPL + 1) / 2];    // K6: RPL x 2 int8, one row per 16-bit half; P4M: RPL fp16 m; K5: RPL x {sc, m} u8
    uint32_t h5[RPL];              // K5 high bits
    uint32_t d2[RPL];              // K5 {d, dmin} fp16 pair per 256
};

// RPL x 16-bit values at p (2*RPL bytes, naturally aligned) -> out[(RPL+1)/2] words
template <int RPL>
__device__ __forceinline__ void ldg_halves(uint32_t* out, const void* p) {
    if (RPL == 1) out[0] = ldg_nc_u16(p);
    else if (RPL == 2) out[0] = ldg_nc_u32(p);
    else { uint2 v = ldg_nc_v2(p); out[0] = v.x; out[1] = v.y; }
}

template <int FMT, int RPL, int NCH>
__device__ __forceinline__ void load_chunk(WChunk<FMT, RPL>& w, const GemvArgs& a, int tile, int c, int lane) {
    const size_t tc = tc_index(tile, c, NCH, a.ntiles, a.cm);
    constexpr int NP = FMT == FAST_Q8 ? 2 : 1;
#pragma unroll
    for (int r = 0; r < RPL; ++r)
#pragma unroll
        for (int p = 0; p < NP; ++p) w.q[r][p] = ldg_nc_v4(a.codes + ((tc * RPL + r) * NP + p) * 512 + lane * 16);
    if (FMT == FAST_P4M) {
        ldg_halves<RPL>(w.dd, a.d + (tc * 32 + lane) * RPL);
        ldg_halves<RPL>(w.sc, (const uint16_t*)a.sc + (tc * 32 + lane) * RPL);
    } else if (FMT == FAST_K5) {
#pragma unroll
        for (int r = 0; r < RPL; ++r) w.h5[r] = ldg_nc_u32(a.qh + (tc * RPL + r) * 128 + lane * 4);
        ldg_halves<RPL>(w.sc, a.sc + (tc * 32 + lane) * RPL * 2);
        const int h = lane >> 4, j = lane & 15;
        const uint32_t* dp = (const uint32_t*)a.d + ((tc * 2 + h) * 2 + (j >> 3)) * RPL;
        if (RPL == 1) w.d2[0] = ldg_nc_u32(dp);
        else if (RPL == 2) { uint2 v = ldg_nc_v2(dp); w.d2[0] = v.x; w.d2[1 % RPL] = v.y; }
        else {
            int4 v = ldg_nc_v4(dp);
            w.d2[0] = v.x; w.d2[1 % RPL] = v.y; w.d2[2 % RPL] = v.z; w.d2[3 % RPL] = v.w;
        }
    } else if (FMT == FAST_K6) {
#pragma unroll
        for (int r = 0; r < RPL; ++r) w.hb[r] = ldg_nc_v2(a.qh + (tc * RPL + r) * 256 + lane * 8);
        ldg_halves<RPL>(w.sc, a.sc + (tc * 32 + lane) * RPL * 2);
        const int h = lane >> 4, j = lane & 15;
        ldg_halves<RPL>(w.dd, a.d + ((tc * 2 + h) * 2 + (j >> 3)) * RPL);
    } else {
        ldg_halves<RPL>(w.dd, a.d + (tc * 32 + lane) * RPL);
    }
}

// one 32-weight group of one row against one activation column
// CVT 0: s -> float via I2F. CVT 1 (P4/Q8 only): exact magic-number conversion, |s| < 2^22 guaranteed for these
// formats, folds the -8*S offset into the integer add: __int_as_float(0x4B400000 + s + off) - 12582912 == (float)(s+off).
// Both give bit-identical results. moff = 0x4B400000 - 8*(s0+s1) (P4) or 0x4B400000 (Q8), precomputed per column.
template <int FMT, int CVT>
__device__ __forceinline__ float group_dot(const int4* q, uint2 hb, uint32_t dd16, uint32_t sc16, const int4& xl,
                                           const int4& xh, float xd, int s0, int s1, int moff) {
    if (FMT == FAST_P4) {
        const int m = 0x0F0F0F0F;
        int s = __dp4a(q[0].x & m, xl.x, 0);
        s = __dp4a(q[0].y & m, xl.y, s);
        s = __dp4a(q[0].z & m, xl.z, s);
        s = __dp4a(q[0].w & m, xl.w, s);
        s = __dp4a((q[0].x >> 4) & m, xh.x, s);
        s = __dp4a((q[0].y >> 4) & m, xh.y, s);
        s = __dp4a((q[0].z >> 4) & m, xh.z, s);
        s = __dp4a((q[0].w >> 4) & m, xh.w, s);
        if (CVT) return h2f_bits(dd16) * (xd * (__int_as_float(s + moff) - 12582912.f));
        s -= 8 * (s0 + s1);
        return h2f_bits(dd16) * (xd * (float)s);
    } else if (FMT == FAST_Q8) {
        int s = __dp4a(q[0].x, xl.x, 0);
        s = __dp4a(q[0].y, xl.y, s);
        s = __dp4a(q[0].z, xl.z, s);
        s = __dp4a(q[0].w, xl.w, s);
        s = __dp4a(q[1].x, xh.x, s);
        s = __dp4a(q[1].y, xh.y, s);
        s = __dp4a(q[1].z, xh.z, s);
        s = __dp4a(q[1].w, xh.w, s);
        if (CVT) return h2f_bits(dd16) * (xd * (__int_as_float(s + moff) - 12582912.f));
        return h2f_bits(dd16) * (xd * (float)s);
    } else {
        const int m = 0x0F0F0F0F, mh = 0x30303030;
        const int H0 = (int)hb.x, H1 = (int)hb.y;
        int sa = __dp4a((q[0].x & m) | ((H0 << 4) & mh), xl.x, 0);
        sa = __dp4a((q[0].y & m) | ((H0 << 2) & mh), xl.y, sa);
        sa = __dp4a((q[0].z & m) | (H0 & mh), xl.z, sa);
        sa = __dp4a((q[0].w & m) | ((int)((unsigned)H0 >> 2) & mh), xl.w, sa);
        int sb = __dp4a(((q[0].x >> 4) & m) | ((H1 << 4) & mh), xh.x, 0);
        sb = __dp4a(((q[0].y >> 4) & m) | ((H1 << 2) & mh), xh.y, sb);
        sb = __dp4a(((q[0].z >> 4) & m) | (H1 & mh), xh.z, sb);
        sb = __dp4a(((q[0].w >> 4) & m) | ((int)((unsigned)H1 >> 2) & mh), xh.w, sb);
        const int scA = (int)(int8_t)(sc16 & 0xff), scB = (int)(int8_t)((sc16 >> 8) & 0xff);
        const int si = scA * (sa - 32 * s0) + scB * (sb - 32 * s1);
        return h2f_bits(dd16) * (xd * (float)si);
    }
}

// P4M / K5 group dot (one row r of the chunk) and dispatch for every format
template <int FMT, int CVT, int RPL>
__device__ __forceinline__ float group_dot_r(const WChunk<FMT, RPL>& w, int r, const int4& xl, const int4& xh, float xd,
                                             int s0, int s1, int moff) {
    if (FMT == FAST_P4M) {
        const int m = 0x0F0F0F0F;
        const int4 q = w.q[r][0];
        int s = __dp4a(q.x & m, xl.x, 0);
        s = __dp4a(q.y & m, xl.y, s);
        s = __dp4a(q.z & m, xl.z, s);
        s = __dp4a(q.w & m, xl.w, s);
        s = __dp4a((q.x >> 4) & m, xh.x, s);
        s = __dp4a((q.y >> 4) & m, xh.y, s);
        s = __dp4a((q.z >> 4) & m, xh.z, s);
        s = __dp4a((q.w >> 4) & m, xh.w, s);
        const float d = h2f_bits(w.dd[r >> 1] >> (16 * (r & 1)));
        const float mm = h2f_bits(w.sc[r >> 1] >> (16 * (r & 1)));
        return d * (xd * (float)s) + mm * (xd * (float)(s0 + s1));
    } else if (FMT == FAST_K5) {
        const int m = 0x0F0F0F0F, mb = 0x10101010;
        const int4 q = w.q[r][0];
        const int H = (int)w.h5[r];
        int s = __dp4a((q.x & m) | ((H << 4) & mb), xl.x, 0);
        s = __dp4a((q.y & m) | ((H << 3) & mb), xl.y, s);
        s = __dp4a((q.z & m) | ((H << 2) & mb), xl.z, s);
        s = __dp4a((q.w & m) | ((H << 1) & mb), xl.w, s);
        s = __dp4a(((q.x >> 4) & m) | (H & mb), xh.x, s);
        s = __dp4a(((q.y >> 4) & m) | ((int)((unsigned)H >> 1) & mb), xh.y, s);
        s = __dp4a(((q.z >> 4) & m) | ((int)((unsigned)H >> 2) & mb), xh.z, s);
        s = __dp4a(((q.w >> 4) & m) | ((int)((unsigned)H >> 3) & mb), xh.w, s);
        const uint32_t scm = w.sc[r >> 1] >> (16 * (r & 1));
        const int sc = (int)(scm & 0xff), mn = (int)((scm >> 8) & 0xff);
        const float d = h2f_bits(w.d2[r]), dmin = h2f_bits(w.d2[r] >> 16);
        return xd * (d * (float)(sc * s) - dmin * (float)(mn * (s0 + s1)));
    } else {
        return group_dot<FMT, CVT>(w.q[r], w.hb[r], w.dd[r >> 1] >> (16 * (r & 1)),
                                   FMT == FAST_K6 ? (w.sc[r >> 1] >> (16 * (r & 1))) : 0u, xl, xh, xd, s0, s1, moff);
    }
}

// smem bytes for XSM variant: lo plane + hi plane (16 B each per group) + meta (8 B per group)
static inline size_t xsm_bytes(int M, int K) { return (size_t)M * (K / 32) * 40; }

// Template: FMT, RPL (rows per half-warp: 1, 2, 4), M columns (1..8), NCH = K/512, CVT (see group_dot; ignored for K6),
// XSM (stage x in shared memory once per block; persistent grids). Register cap 128 (__launch_bounds__(256, 2)); a
// 64-register cap measured worse on every shape in M0 round 1.
template <int FMT, int RPL, int M, int NCH, int CVT, bool XSM>
__global__ void __launch_bounds__(256, 2) gemv_fast_kernel(const GemvArgs a) {
    constexpr int D = 2;  // register ring depth (nvcc schedules the loads itself; D has no measurable effect)
    extern __shared__ int4 s_x[];
    const int lane = threadIdx.x & 31, h = lane >> 4, j = lane & 15;
    const int wpb = blockDim.x >> 5;
    const int warp = blockIdx.x * wpb + (threadIdx.x >> 5);
    const int nwarps = gridDim.x * wpb;
    constexpr int NB = NCH * 16;  // groups per row
    const int K = NCH * 512;

    if (XSM) {
        int4* lo = s_x; int4* hi = s_x + M * NB; int2* mt = (int2*)(s_x + 2 * M * NB);
        for (int i = threadIdx.x; i < M * NB; i += blockDim.x) {
            const int col = i / NB, kb = i % NB;
            const int4* src = (const int4*)(a.xq + (size_t)col * K + kb * 32);
            lo[i] = src[0]; hi[i] = src[1];
            mt[i] = a.xm[(size_t)col * NB + kb];
        }
        __syncthreads();
    }
    const int4* s_lo = s_x; const int4* s_hi = s_x + M * NB; const int2* s_mt = (const int2*)(s_x + 2 * M * NB);

    for (int tile = warp; tile < a.ntiles; tile += nwarps) {
        float acc[RPL][M];
#pragma unroll
        for (int r = 0; r < RPL; ++r)
#pragma unroll
            for (int c = 0; c < M; ++c) acc[r][c] = 0.f;

        WChunk<FMT, RPL> w[D];
#pragma unroll
        for (int c = 0; c < D; ++c)
            if (c < NCH) load_chunk<FMT, RPL, NCH>(w[c], a, tile, c, lane);

#pragma unroll
        for (int c = 0; c < NCH; ++c) {
            WChunk<FMT, RPL> cur = w[c % D];
            if (c + D < NCH) load_chunk<FMT, RPL, NCH>(w[c % D], a, tile, c + D, lane);
            const int kb = c * 16 + j;
#pragma unroll
            for (int col = 0; col < M; ++col) {
                int4 xl, xh; int2 mt;
                if (XSM) {
                    xl = s_lo[col * NB + kb]; xh = s_hi[col * NB + kb]; mt = s_mt[col * NB + kb];
                } else {
                    const int4* xp = (const int4*)(a.xq + (size_t)col * K + kb * 32);
                    xl = __ldg(xp); xh = __ldg(xp + 1); mt = __ldg(a.xm + (size_t)col * NB + kb);
                }
                const float xd = __int_as_float(mt.x);
                const int s0 = (int)(short)(mt.y & 0xffff), s1 = mt.y >> 16;
                const int moff = FMT == FAST_P4 ? 0x4B400000 - 8 * (s0 + s1) : 0x4B400000;
#pragma unroll
                for (int r = 0; r < RPL; ++r)
                    acc[r][col] += group_dot_r<FMT, CVT, RPL>(cur, r, xl, xh, xd, s0, s1, moff);
            }
        }
#pragma unroll
        for (int r = 0; r < RPL; ++r)
#pragma unroll
            for (int col = 0; col < M; ++col) {
                float v = acc[r][col];
                v += __shfl_xor_sync(0xffffffffu, v, 8);
                v += __shfl_xor_sync(0xffffffffu, v, 4);
                v += __shfl_xor_sync(0xffffffffu, v, 2);
                v += __shfl_xor_sync(0xffffffffu, v, 1);
                const int row = tile * 2 * RPL + h * RPL + r;
                if (j == 0 && row < a.N) a.y[(size_t)col * a.ldy + row] = v;
            }
    }
}

template <int FMT, int RPL, int M, int NCH, int CVT, bool XSM>
static cudaError_t gemv_fast_launch(const GemvArgs& a, int blocks, int threads, cudaStream_t s) {
    auto k = gemv_fast_kernel<FMT, RPL, M, NCH, CVT, XSM>;
    size_t smem = XSM ? xsm_bytes(M, a.K) : 0;
    static int attr_done = 0;  // per instantiation (and per device in practice: both T4s identical)
    if (smem > 48 * 1024 && !attr_done) {
        cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
        if (e != cudaSuccess) return e;
        attr_done = 1;
    }
    k<<<blocks, threads, smem, s>>>(a);
    return cudaGetLastError();
}

template <int FMT, int RPL, int M, int NCH, int CVT, bool XSM>
static int gemv_fast_occupancy(int threads) {
    int nb = 0;
    size_t smem = XSM ? (size_t)M * NCH * 16 * 40 : 0;
    auto k = gemv_fast_kernel<FMT, RPL, M, NCH, CVT, XSM>;
    if (smem > 48 * 1024) cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, threads, smem);
    return nb;
}

// ------------------------------------------------------------------------------------------------ q8_1 quantize
// x: [M][K] fp32 -> xq [M][K] int8, xm [M][K/32]. One warp per 32-group.
static __global__ void quantize_q8_kernel(const float* __restrict__ x, int K, int M, int8_t* __restrict__ xq,
                                   int2* __restrict__ xm) {
    const int g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int nb = K / 32;
    if (g >= M * nb) return;
    const float v = x[(size_t)g * 32 + lane];
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.f;
    const int q = amax == 0.f ? 0 : (int)roundf(v / d);
    xq[(size_t)g * 32 + lane] = (int8_t)q;
    int s = q;
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);  // sums within each half
    const int s1 = __shfl_sync(0xffffffffu, s, 16);
    if (lane == 0) xm[g] = make_int2(__float_as_int(d), (int)((unsigned)(s & 0xffff) | ((unsigned)s1 << 16)));
}

// ------------------------------------------------------------------------------------------------ device repack
// One thread per (row, 32-group). src rows are raw GGUF blocks (row stride src_row_bytes); dst zero-initialized.
// rows [row0, row0 + N) of the layout; src holds exactly those N rows
static __global__ void repack_kernel(int fmt, int N, int K, int rpl, int ntiles, size_t off_qh, size_t off_sc, size_t off_d,
                              const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int row0 = 0, int cm = 0) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = K / 32;
    if (idx >= (size_t)N * nb) return;
    const int lrow = (int)(idx / nb), kb = (int)(idx % nb);
    const int row = row0 + lrow;
    const int nch = K / 512;
    const int be = (fmt == FAST_K6 || fmt == FAST_K5) ? 256 : 32;
    const int bb = fmt == FAST_P4 ? 18 : fmt == FAST_Q8 ? 34 : fmt == FAST_K6 ? 210 : fmt == FAST_P4M ? 20 : 176;
    const size_t rb = (size_t)(K / be) * bb;
    const uint8_t* s = src + (size_t)lrow * rb;
    const int tile = row / (2 * rpl), w = row % (2 * rpl), h = w / rpl, r = w % rpl;
    const int c = kb / 16, j = kb % 16, lane = h * 16 + j;
    const size_t tc = tc_index(tile, c, nch, ntiles, cm);
    uint16_t* dp = (uint16_t*)(dst + off_d);
    if (fmt == FAST_P4) {
        const uint8_t* b = s + 18 * kb;
        uint8_t* o = dst + (tc * rpl + r) * 512 + lane * 16;
        for (int i = 0; i < 16; ++i) o[i] = b[2 + i];
        dp[(tc * 32 + lane) * rpl + r] = (uint16_t)(b[0] | (b[1] << 8));
    } else if (fmt == FAST_P4M) {
        const uint8_t* b = s + 20 * kb;
        uint8_t* o = dst + (tc * rpl + r) * 512 + lane * 16;
        for (int i = 0; i < 16; ++i) o[i] = b[4 + i];
        dp[(tc * 32 + lane) * rpl + r] = (uint16_t)(b[0] | (b[1] << 8));
        ((uint16_t*)(dst + off_sc))[(tc * 32 + lane) * rpl + r] = (uint16_t)(b[2] | (b[3] << 8));
    } else if (fmt == FAST_K5) {
        const int sb = kb / 8, sub = kb % 8;
        const uint8_t* b = s + 176 * sb;
        const uint8_t* qh = b + 16; const uint8_t* qs = b + 48;
        uint8_t g[32];
        for (int l = 0; l < 32; ++l) {
            const int lo = (sub & 1) ? (qs[32 * (sub >> 1) + l] >> 4) : (qs[32 * (sub >> 1) + l] & 15);
            g[l] = (uint8_t)(lo | (((qh[l] >> sub) & 1) << 4));
        }
        uint8_t* o = dst + (tc * rpl + r) * 512 + lane * 16;
        for (int i = 0; i < 16; ++i) o[i] = (uint8_t)((g[i] & 15) | ((g[16 + i] & 15) << 4));
        uint32_t H = 0;
        for (int t = 0; t < 4; ++t)
            for (int jj = 0; jj < 4; ++jj) {
                H |= (uint32_t)((g[4 * t + jj] >> 4) & 1) << (8 * jj + t);
                H |= (uint32_t)((g[16 + 4 * t + jj] >> 4) & 1) << (8 * jj + 4 + t);
            }
        *(uint32_t*)(dst + off_qh + (tc * rpl + r) * 128 + lane * 4) = H;
        const uint8_t* q = b + 4;
        int sc, mn;
        if (sub < 4) { sc = q[sub] & 63; mn = q[sub + 4] & 63; }
        else { sc = (q[sub + 4] & 0xF) | ((q[sub - 4] >> 6) << 4); mn = (q[sub + 4] >> 4) | ((q[sub] >> 6) << 4); }
        uint8_t* scd = dst + off_sc + ((tc * 32 + lane) * rpl + r) * 2;
        scd[0] = (uint8_t)sc; scd[1] = (uint8_t)mn;
        if ((j & 7) == 0)
            *(uint32_t*)(dst + off_d + (((tc * 2 + h) * 2 + (j >> 3)) * rpl + r) * 4) =
                (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    } else if (fmt == FAST_Q8) {
        const uint8_t* b = s + 34 * kb;
        for (int p = 0; p < 2; ++p) {
            uint8_t* o = dst + ((tc * rpl + r) * 2 + p) * 512 + lane * 16;
            for (int i = 0; i < 16; ++i) o[i] = b[2 + 16 * p + i];
        }
        dp[(tc * 32 + lane) * rpl + r] = (uint16_t)(b[0] | (b[1] << 8));
    } else {
        const int sb = kb / 8, sub = kb % 8;
        const uint8_t* b = s + 210 * sb;
        const uint8_t* ql = b; const uint8_t* qhs = b + 128;
        uint8_t g[32];
        for (int i = 0; i < 32; ++i) {
            const int e = 32 * sub + i, n = e / 128, ii = e % 128, gg = ii / 32, l = ii % 32;
            const uint8_t L = ql[64 * n + l + (gg & 1) * 32];
            const uint8_t nib = gg >= 2 ? (L >> 4) : (L & 15);
            g[i] = nib | (((qhs[32 * n + l] >> (2 * gg)) & 3) << 4);
        }
        uint8_t* o = dst + (tc * rpl + r) * 512 + lane * 16;
        for (int i = 0; i < 16; ++i) o[i] = (uint8_t)((g[i] & 15) | ((g[16 + i] & 15) << 4));
        uint32_t H0 = 0, H1 = 0;
        for (int t = 0; t < 4; ++t)
            for (int jj = 0; jj < 4; ++jj) {
                H0 |= (uint32_t)((g[4 * t + jj] >> 4) & 3) << (8 * jj + 2 * t);
                H1 |= (uint32_t)((g[16 + 4 * t + jj] >> 4) & 3) << (8 * jj + 2 * t);
            }
        uint32_t* qo = (uint32_t*)(dst + off_qh + (tc * rpl + r) * 256 + lane * 8);
        qo[0] = H0; qo[1] = H1;
        uint8_t* scd = dst + off_sc + ((tc * 32 + lane) * rpl + r) * 2;
        scd[0] = b[192 + 2 * sub]; scd[1] = b[192 + 2 * sub + 1];
        if ((j & 7) == 0) dp[((tc * 2 + h) * 2 + (j >> 3)) * rpl + r] = (uint16_t)(b[208] | (b[209] << 8));
    }
}

// repack rows [row0, row0 + n) (d_src holds just those rows); the caller zeroes d_dst once beforehand
static inline cudaError_t repack_device_rows(const Layout& L, const uint8_t* d_src, uint8_t* d_dst, int row0, int n,
                                             cudaStream_t s) {
    const size_t tot = (size_t)n * (L.K / 32);
    repack_kernel<<<(unsigned)((tot + 255) / 256), 256, 0, s>>>(L.fmt, n, L.K, L.rpl, L.ntiles, L.off_qh, L.off_sc,
                                                                 L.off_d, d_src, d_dst, row0, L.cm);
    return cudaGetLastError();
}

static inline cudaError_t repack_device(const Layout& L, const uint8_t* d_src, uint8_t* d_dst, cudaStream_t s) {
    cudaError_t e = cudaMemsetAsync(d_dst, 0, L.bytes, s);
    if (e != cudaSuccess) return e;
    const size_t n = (size_t)L.N * (L.K / 32);
    repack_kernel<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(L.fmt, L.N, L.K, L.rpl, L.ntiles, L.off_qh, L.off_sc,
                                                               L.off_d, d_src, d_dst, 0, L.cm);
    return cudaGetLastError();
}

}  // namespace gemv
}  // namespace t4q
#endif  // __CUDACC__
