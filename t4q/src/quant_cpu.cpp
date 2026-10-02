#include "quant_cpu.h"

#include <cstring>

#include "gguf.h"

float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {  // subnormal: normalize
            int e = -1;
            do { e++; man <<= 1; } while (!(man & 0x400));
            man &= 0x3ff;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (man << 13);
        }
    } else if (exp == 31) bits = sign | 0x7f800000u | (man << 13);
    else bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static inline uint16_t rd16(const uint8_t* p) { uint16_t v; memcpy(&v, p, 2); return v; }

static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4); }
}

bool dequant_row_cpu(uint32_t type, const uint8_t* x, float* y, int64_t n) {
    switch (type) {
        case GT_F32: memcpy(y, x, n * 4); return true;
        case GT_F16: for (int64_t i = 0; i < n; i++) y[i] = fp16_to_fp32(rd16(x + 2 * i)); return true;
        case GT_Q4_0: {
            for (int64_t i = 0; i < n / 32; i++) {
                const uint8_t* b = x + 18 * i;
                const float d = fp16_to_fp32(rd16(b));
                for (int j = 0; j < 16; ++j) {
                    const int x0 = (b[2 + j] & 0x0F) - 8;
                    const int x1 = (b[2 + j] >> 4) - 8;
                    y[i * 32 + j] = x0 * d;
                    y[i * 32 + j + 16] = x1 * d;
                }
            }
            return true;
        }
        case GT_Q4_1: {
            for (int64_t i = 0; i < n / 32; i++) {
                const uint8_t* b = x + 20 * i;
                const float d = fp16_to_fp32(rd16(b));
                const float m = fp16_to_fp32(rd16(b + 2));
                for (int j = 0; j < 16; ++j) {
                    const int x0 = (b[4 + j] & 0x0F);
                    const int x1 = (b[4 + j] >> 4);
                    y[i * 32 + j] = x0 * d + m;
                    y[i * 32 + j + 16] = x1 * d + m;
                }
            }
            return true;
        }
        case GT_Q8_0: {
            for (int64_t i = 0; i < n / 32; i++) {
                const uint8_t* b = x + 34 * i;
                const float d = fp16_to_fp32(rd16(b));
                for (int j = 0; j < 32; ++j) y[i * 32 + j] = (int8_t)b[2 + j] * d;
            }
            return true;
        }
        case GT_Q5_K: {
            for (int64_t i = 0; i < n / 256; i++) {
                const uint8_t* b = x + 176 * i;
                const float d = fp16_to_fp32(rd16(b));
                const float min = fp16_to_fp32(rd16(b + 2));
                const uint8_t* scales = b + 4;
                const uint8_t* qh = b + 16;
                const uint8_t* ql = b + 48;
                float* yy = y + i * 256;
                int is = 0;
                uint8_t sc, m;
                uint8_t u1 = 1, u2 = 2;
                for (int j = 0; j < 256; j += 64) {
                    get_scale_min_k4(is + 0, scales, &sc, &m);
                    const float d1 = d * sc; const float m1 = min * m;
                    get_scale_min_k4(is + 1, scales, &sc, &m);
                    const float d2 = d * sc; const float m2 = min * m;
                    for (int l = 0; l < 32; ++l) *yy++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
                    for (int l = 0; l < 32; ++l) *yy++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
                    ql += 32; is += 2;
                    u1 <<= 2; u2 <<= 2;
                }
            }
            return true;
        }
        case GT_Q6_K: {
            for (int64_t i = 0; i < n / 256; i++) {
                const uint8_t* b = x + 210 * i;
                const float d = fp16_to_fp32(rd16(b + 208));
                const uint8_t* ql = b;
                const uint8_t* qh = b + 128;
                const int8_t* sc = (const int8_t*)(b + 192);
                float* yy = y + i * 256;
                for (int nn = 0; nn < 256; nn += 128) {
                    for (int l = 0; l < 32; ++l) {
                        int is = l / 16;
                        const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        yy[l + 0] = d * sc[is + 0] * q1;
                        yy[l + 32] = d * sc[is + 2] * q2;
                        yy[l + 64] = d * sc[is + 4] * q3;
                        yy[l + 96] = d * sc[is + 6] * q4;
                    }
                    yy += 128; ql += 64; qh += 32; sc += 8;
                }
            }
            return true;
        }
        default: return false;
    }
}
