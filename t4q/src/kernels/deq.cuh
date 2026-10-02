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
