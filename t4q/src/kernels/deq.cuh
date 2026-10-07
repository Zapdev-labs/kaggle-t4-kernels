// Device dequant of one 32-weight group from the M1 packed formats. Uses explicit _rn intrinsics so the
// result is bit-identical to the CPU port of ggml's dequantize_row_* (loader round-trip check).
#pragma once
#ifdef T4Q_HOST_SIM
// host simulation for local tests (tests/test_deq_sim.cpp): same code, plain fp ops (compile with -ffp-contract=off)
#include <cstdint>
#include <cstring>
float fp16_to_fp32(uint16_t h);
#define T4Q_HD inline
struct uint4 { uint32_t x, y, z, w; };
inline float __fmul_rn(float a, float b) { return a * b; }
inline float __fadd_rn(float a, float b) { return a + b; }
inline float __fsub_rn(float a, float b) { return a - b; }
T4Q_HD float h2f(uint16_t v) { return fp16_to_fp32(v); }
#else
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define T4Q_HD __device__ __forceinline__
__device__ __forceinline__ float h2f(uint16_t v) { return __half2float(__ushort_as_half(v)); }
#endif

#include "../packed.h"
#include "../iq1s_table.h"

T4Q_HD void dev_scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// w[0..31] = weights row `row`, elements 32g .. 32g+31
template <int FMT>
T4Q_HD void deq32(const PackedW& W, int64_t row, int64_t g, float* w) {
    if constexpr (FMT == FMT_F32) {
        const float* p = (const float*)W.codes + row * W.cols + g * 32;
#pragma unroll
        for (int j = 0; j < 32; j++) w[j] = p[j];
    } else if constexpr (FMT == FMT_P4 || FMT == FMT_P4M) {
        const int64_t b = row * (W.cols / 32) + g;
        const uint4 c4 = *(const uint4*)(W.codes + b * 16);
        const uint8_t* c = (const uint8_t*)&c4;
        const float d = h2f(W.d[b]);
        if constexpr (FMT == FMT_P4) {
#pragma unroll
            for (int j = 0; j < 16; j++) {
                w[j] = __fmul_rn((float)((c[j] & 15) - 8), d);
                w[j + 16] = __fmul_rn((float)((c[j] >> 4) - 8), d);
            }
        } else {
            const float m = h2f(W.m[b]);
#pragma unroll
            for (int j = 0; j < 16; j++) {
                w[j] = __fadd_rn(__fmul_rn((float)(c[j] & 15), d), m);
                w[j + 16] = __fadd_rn(__fmul_rn((float)(c[j] >> 4), d), m);
            }
        }
    } else if constexpr (FMT == FMT_Q8) {
        const int64_t b = row * (W.cols / 32) + g;
        const int8_t* c = (const int8_t*)W.codes + b * 32;
        const float d = h2f(W.d[b]);
#pragma unroll
        for (int j = 0; j < 32; j++) w[j] = __fmul_rn((float)c[j], d);
    } else if constexpr (FMT == FMT_K5) {
        const int64_t nb = W.cols / 256;
        const int64_t blk = row * nb + (g >> 3);
        const int s = (int)(g & 7);
        const uint8_t* meta = W.meta + blk * 16;
        const float d = h2f((uint16_t)(meta[0] | (meta[1] << 8)));
        const float dmin = h2f((uint16_t)(meta[2] | (meta[3] << 8)));
        int sc, mm;
        dev_scale_min_k4(s, meta + 4, sc, mm);
        const float d1 = __fmul_rn(d, (float)sc);
        const float m1 = __fmul_rn(dmin, (float)mm);
        const uint8_t* qs = W.codes + blk * 128 + 32 * (s >> 1);
        const uint8_t* qh = W.hi + blk * 32;
        const int hin = s & 1;
#pragma unroll
        for (int l = 0; l < 32; l++) {
            const int lo = hin ? (qs[l] >> 4) : (qs[l] & 15);
            const int q = lo + (((qh[l] >> s) & 1) ? 16 : 0);
            w[l] = __fsub_rn(__fmul_rn(d1, (float)q), m1);
        }
    } else if constexpr (FMT == FMT_K2) {
        // Q2_K (r17 CF): 84 B / 256. Sub s (16 elems [16s,+16)): bytes qs[32*(s>>3) + 16*(s&1) .. +16)
        // at ONE shared shift 2*((s&7)>>1); a = d*(scales[s]&15), m = dmin*(scales[s]>>4); w = a*q - m.
        // The 32-elem group g covers subs 2g, 2g+1 = the 32 bytes at qs[32*((g&7)>>2) .. +32), shift 2*(g&3).
        const int64_t nb = W.cols / 256;
        const int64_t blk = row * nb + (g >> 3);
        const uint8_t* meta = W.meta + blk * 20;
        const float d = h2f((uint16_t)(meta[0] | (meta[1] << 8)));
        const float dmin = h2f((uint16_t)(meta[2] | (meta[3] << 8)));
        const uint8_t* sc = meta + 4;
        const uint8_t* qs = W.codes + blk * 64 + 32 * ((g & 7) >> 2);
        const int sh = (g & 3) << 1;
        const int s0 = 2 * (g & 7), s1 = s0 + 1;
        const float a0 = __fmul_rn(d, (float)(sc[s0] & 15));
        const float m0 = __fmul_rn(dmin, (float)(sc[s0] >> 4));
        const float a1 = __fmul_rn(d, (float)(sc[s1] & 15));
        const float m1 = __fmul_rn(dmin, (float)(sc[s1] >> 4));
#pragma unroll
        for (int l = 0; l < 16; l++) {
            const int q0 = (qs[l] >> sh) & 3;
            const int q1 = (qs[l + 16] >> sh) & 3;
            w[l] = __fsub_rn(__fmul_rn(a0, (float)q0), m0);
            w[l + 16] = __fsub_rn(__fmul_rn(a1, (float)q1), m1);
        }
    } else if constexpr (FMT == FMT_K4) {
        // Q4_K (r17 CF): 144 B / 256. Sub s (32 elems [32s,+32)) = nibble plane s&1 of the 32 bytes
        // qs[32*((g&7)>>1) .. +32); sc/mi 6-bit from scales[12] (same decode as K5); w = d*sc*(nib) - dmin*mi.
        const int64_t nb = W.cols / 256;
        const int64_t blk = row * nb + (g >> 3);
        const uint8_t* meta = W.meta + blk * 16;
        const float d = h2f((uint16_t)(meta[0] | (meta[1] << 8)));
        const float dmin = h2f((uint16_t)(meta[2] | (meta[3] << 8)));
        int sc, mi;
        dev_scale_min_k4((int)(g & 7), meta + 4, sc, mi);
        const float d1 = __fmul_rn(d, (float)sc);
        const float m1 = __fmul_rn(dmin, (float)mi);
        const uint8_t* qs = W.codes + blk * 128 + 32 * ((g & 7) >> 1);
        const int hin = g & 1;
#pragma unroll
        for (int l = 0; l < 32; l++) {
            const int lo = hin ? (qs[l] >> 4) : (qs[l] & 15);
            w[l] = __fsub_rn(__fmul_rn(d1, (float)lo), m1);
        }
    } else if constexpr (FMT == FMT_Q51) {
        // Q5_1 (r17 CF): 24 B / 32. w = d*(nib | qhbit<<4) + m; lo nibble = elems 0..15, hi = 16..31;
        // qh[4] as a LE u32: bit e = elem e's 5th bit.
        const int64_t b = row * (W.cols / 32) + g;
        const uint8_t* c = W.codes + b * 16;
        const uint8_t* qh = W.hi + b * 4;
        const float d = h2f(W.d[b]);
        const float m = h2f(W.m[b]);
        const uint32_t qhw = (uint32_t)qh[0] | ((uint32_t)qh[1] << 8) | ((uint32_t)qh[2] << 16) | ((uint32_t)qh[3] << 24);
#pragma unroll
        for (int l = 0; l < 16; l++) {
            const int q0 = (c[l] & 15) + (int)(((qhw >> l) & 1) << 4);
            const int q1 = (c[l] >> 4) + (int)(((qhw >> (16 + l)) & 1) << 4);
            w[l] = __fadd_rn(__fmul_rn(d, (float)q0), m);
            w[l + 16] = __fadd_rn(__fmul_rn(d, (float)q1), m);
        }
    } else if constexpr (FMT == FMT_K6) {
        const int64_t nb = W.cols / 256;
        const int64_t blk = row * nb + (g >> 3);
        const int gg = (int)(g & 7);
        const int n = gg >> 2, qd = gg & 3;
        const uint8_t* ql = W.codes + blk * 128 + 64 * n + (qd & 1) * 32;
        const uint8_t* qh = W.hi + blk * 64 + 32 * n;
        const int8_t* sc = (const int8_t*)W.meta + blk * 16 + 8 * n + 2 * qd;
        const float d = h2f(W.d[blk]);
        const float ds0 = __fmul_rn(d, (float)sc[0]);
        const float ds1 = __fmul_rn(d, (float)sc[1]);
#pragma unroll
        for (int l = 0; l < 32; l++) {
            const int lo = (qd >= 2) ? (ql[l] >> 4) : (ql[l] & 15);
            const int hb = (qh[l] >> (2 * qd)) & 3;
            const int q = (lo | (hb << 4)) - 32;
            w[l] = __fmul_rn(l < 16 ? ds0 : ds1, (float)q);
        }
    } else if constexpr (FMT == FMT_IQ1S) {
        // IQ1_S (cf-m6 requant, CF_REQUANT.md section 2): 50 B / 256. Group g (32 elems) =
        // block (g>>3)'s qh[g&7] (the scale/shift/index-halves) + 4 grid indices at
        // qs[4*ib+l]; each index selects one 8-elem lattice vector (t4q_kgrid_1bit_2048,
        // L_j at bits 2j). The exact reference fp-op order (y = dl*(v + delta)):
        // dl = d*(2*sc + 1), v = L - 1 (in {-1,0,1}), delta = +-0.125 (qh bit 15).
        const int64_t nb = W.cols / 256;
        const int64_t blk = row * nb + (g >> 3);
        const int ib = (int)(g & 7);
        const uint8_t* qs = W.codes + blk * 32;
        const uint16_t qhi = ((const uint16_t*)W.hi + blk * 8)[ib];
        const float d = h2f(W.d[blk]);
        const float dl = __fmul_rn(d, (float)(2 * ((qhi >> 12) & 7) + 1));
        const float delta = (qhi & 0x8000u) ? -T4Q_IQ1S_DELTA : T4Q_IQ1S_DELTA;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int idx = qs[4 * ib + l] | (int)(((qhi >> (3 * l)) & 7) << 8);
            const uint16_t kv = t4q_kgrid_1bit_2048[idx];
#pragma unroll
            for (int j = 0; j < 8; ++j)
                w[8 * l + j] = __fmul_rn(dl, __fadd_rn((float)(((kv >> (2 * j)) & 3) - 1), delta));
        }
    }
}

// ---- repack of one source block (shared by the repack kernels and the host simulation) ----
// raw points at the GGUF block; dst = destination block index in the plane (row * blocks_per_row + b)
T4Q_HD void repack_q4_block(const PackedW& W, const uint8_t* src, int64_t dst, int has_m) {
    W.d[dst] = (uint16_t)(src[0] | (src[1] << 8));
    int off = 2;
    if (has_m) { W.m[dst] = (uint16_t)(src[2] | (src[3] << 8)); off = 4; }
    for (int j = 0; j < 16; j++) W.codes[dst * 16 + j] = src[off + j];
}
T4Q_HD void repack_q8_block(const PackedW& W, const uint8_t* src, int64_t dst) {
    W.d[dst] = (uint16_t)(src[0] | (src[1] << 8));
    for (int j = 0; j < 32; j++) W.codes[dst * 32 + j] = src[2 + j];
}
T4Q_HD void repack_q5k_block(const PackedW& W, const uint8_t* src, int64_t dst) {
    for (int j = 0; j < 16; j++) W.meta[dst * 16 + j] = src[j];
    for (int j = 0; j < 32; j++) W.hi[dst * 32 + j] = src[16 + j];
    for (int j = 0; j < 128; j++) W.codes[dst * 128 + j] = src[48 + j];
}
T4Q_HD void repack_q6k_block(const PackedW& W, const uint8_t* src, int64_t dst) {
    for (int j = 0; j < 128; j++) W.codes[dst * 128 + j] = src[j];
    for (int j = 0; j < 64; j++) W.hi[dst * 64 + j] = src[128 + j];
    for (int j = 0; j < 16; j++) W.meta[dst * 16 + j] = src[192 + j];
    W.d[dst] = (uint16_t)(src[208] | (src[209] << 8));
}
// ---- r17 CF blocks ----
// Q2_K source block: scales[16] (lo = d-index, hi = dmin-index), qs[64], d, dmin (fp16) = 84 B.
T4Q_HD void repack_q2k_block(const PackedW& W, const uint8_t* src, int64_t dst) {
    W.meta[dst * 20 + 0] = src[80];
    W.meta[dst * 20 + 1] = src[81];
    W.meta[dst * 20 + 2] = src[82];
    W.meta[dst * 20 + 3] = src[83];
    for (int j = 0; j < 16; j++) W.meta[dst * 20 + 4 + j] = src[j];
    for (int j = 0; j < 64; j++) W.codes[dst * 64 + j] = src[16 + j];
}
// Q4_K source block (real ggml order): d, dmin (fp16), scales[12] (6-bit), qs[128] = 144 B.
// meta = [d, dmin, scales[12]] (the K5-style plane).
T4Q_HD void repack_q4k_block(const PackedW& W, const uint8_t* src, int64_t dst) {
    W.meta[dst * 16 + 0] = src[0];
    W.meta[dst * 16 + 1] = src[1];
    W.meta[dst * 16 + 2] = src[2];
    W.meta[dst * 16 + 3] = src[3];
    for (int j = 0; j < 12; j++) W.meta[dst * 16 + 4 + j] = src[4 + j];
    for (int j = 0; j < 128; j++) W.codes[dst * 128 + j] = src[16 + j];
}
// Q5_1 source block: d, m (fp16), qh[4], qs[16] = 24 B.
T4Q_HD void repack_q51_block(const PackedW& W, const uint8_t* src, int64_t dst) {
    W.d[dst] = (uint16_t)(src[0] | (src[1] << 8));
    W.m[dst] = (uint16_t)(src[2] | (src[3] << 8));
    for (int j = 0; j < 4; j++) W.hi[dst * 4 + j] = src[4 + j];
    for (int j = 0; j < 16; j++) W.codes[dst * 16 + j] = src[8 + j];
}
