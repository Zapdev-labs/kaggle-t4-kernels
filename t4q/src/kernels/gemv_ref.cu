// M1 reference GEMV: one warp per output row, each lane dequantizes 32-weight groups (deq32) and dots them with
// fp32 activations. Correctness first; the fast dp4a kernels live in gemv.cuh (M2).
#include "deq.cuh"
#include "kernels.h"

template <int FMT>
__global__ void __launch_bounds__(256) k_gemv(PackedW W, const float* __restrict__ x, float* __restrict__ y) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t row = (int64_t)blockIdx.x * 8 + warp;
    if (row >= W.rows) return;
    const int64_t ng = W.cols / 32;
    float acc = 0.f;
    for (int64_t g = lane; g < ng; g += 32) {
        float w[32];
        deq32<FMT>(W, row, g, w);
        const float4* xv = (const float4*)(x + g * 32);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float4 a = __ldg(xv + j);
            acc += w[4 * j] * a.x + w[4 * j + 1] * a.y + w[4 * j + 2] * a.z + w[4 * j + 3] * a.w;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = acc;
}

void launch_gemv(const PackedW& W, const float* x, float* y, cudaStream_t s) {
    const unsigned G = (unsigned)((W.rows + 7) / 8);
    switch (W.fmt) {
        case FMT_F32: k_gemv<FMT_F32><<<G, 256, 0, s>>>(W, x, y); break;
        case FMT_P4: k_gemv<FMT_P4><<<G, 256, 0, s>>>(W, x, y); break;
        case FMT_P4M: k_gemv<FMT_P4M><<<G, 256, 0, s>>>(W, x, y); break;
        case FMT_Q8: k_gemv<FMT_Q8><<<G, 256, 0, s>>>(W, x, y); break;
        case FMT_K5: k_gemv<FMT_K5><<<G, 256, 0, s>>>(W, x, y); break;
        case FMT_K6: k_gemv<FMT_K6><<<G, 256, 0, s>>>(W, x, y); break;
        case FMT_K2: k_gemv<FMT_K2><<<G, 256, 0, s>>>(W, x, y); break;
        case FMT_K4: k_gemv<FMT_K4><<<G, 256, 0, s>>>(W, x, y); break;
        case FMT_Q51: k_gemv<FMT_Q51><<<G, 256, 0, s>>>(W, x, y); break;
    }
}

// ------------------------------------------------------------------------------------------- q8_1 activations
// ggml quantize_row_q8_1_ref: q = roundf(x*id), d = amax/127 as FP16, s = FP16(sum(q) * d)
// (the sum of the QUANTIZED ints scaled back by d - NOT the raw float sum; the m-terms of the
// q4_0/q4_1/q5_1 dots consume exactly this dequantized-block-sum).
__global__ void k_quantize_q8_1(const float* __restrict__ x, int K, int8_t* xq, float* xd, float* xs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // K % 32 == 0, blockDim multiple of 32
    if (i >= K) return;
    const float xi = x[i];
    float amax = fabsf(xi);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    }
    const float d = amax / 127.0f;
    const float id = d ? 1.0f / d : 0.0f;
    const int8_t q = (int8_t)roundf(xi * id);
    xq[i] = q;
    float sum = (float)q;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    if ((i & 31) == 0) {
        xd[i >> 5] = __half2float(__float2half_rn(d));
        xs[i >> 5] = __half2float(__float2half_rn(d * sum));
    }
}

void launch_quantize_q8_1(const float* x, int K, int8_t* xq, float* xd, float* xs, cudaStream_t s) {
    k_quantize_q8_1<<<(K + 255) / 256, 256, 0, s>>>(x, K, xq, xd, xs);
}

template <int FMT>
__device__ __forceinline__ float dot_q8(const PackedW& W, int64_t row, int64_t g, const int8_t* x, float d8, float s8) {
    if constexpr (FMT == FMT_P4 || FMT == FMT_P4M) {
        const int64_t b = row * (W.cols / 32) + g;
        const uint4 c4 = *(const uint4*)(W.codes + b * 16);
        const uint8_t* c = (const uint8_t*)&c4;
        int s0 = 0, s1 = 0;
#pragma unroll
        for (int j = 0; j < 8; j++) s0 += (c[j] & 15) * x[j] + (c[j] >> 4) * x[j + 16];
#pragma unroll
        for (int j = 8; j < 16; j++) s1 += (c[j] & 15) * x[j] + (c[j] >> 4) * x[j + 16];
        if constexpr (FMT == FMT_P4) {
            const float d4 = h2f(W.d[b]);
            return d4 * (s0 * d8 - 4 * s8) + d4 * (s1 * d8 - 4 * s8);
        } else {
            const __half d4 = __ushort_as_half(W.d[b]), m4 = __ushort_as_half(W.m[b]);
            const float d4d8 = __half2float(__hmul(d4, __float2half_rn(d8)));
            const float m4s8 = __half2float(__hmul(m4, __float2half_rn(s8)));
            return (s0 * d4d8 + m4s8 / 2) + (s1 * d4d8 + m4s8 / 2);
        }
    } else if constexpr (FMT == FMT_Q8) {
        const int64_t b = row * (W.cols / 32) + g;
        const int8_t* c = (const int8_t*)W.codes + b * 32;
        int si = 0;
#pragma unroll
        for (int j = 0; j < 32; j++) si += c[j] * x[j];
        return h2f(W.d[b]) * (si * d8);
    } else if constexpr (FMT == FMT_K5) {
        const int64_t nb = W.cols / 256;
        const int64_t blk = row * nb + (g >> 3);
        const int s = (int)(g & 7);
        const uint8_t* meta = W.meta + blk * 16;
        const float d = h2f((uint16_t)(meta[0] | (meta[1] << 8)));
        const float dmin = h2f((uint16_t)(meta[2] | (meta[3] << 8)));
        int sc, mm;
        dev_scale_min_k4(s, meta + 4, sc, mm);
        const uint8_t* qs = W.codes + blk * 128 + 32 * (s >> 1);
        const uint8_t* qh = W.hi + blk * 32;
        const int hin = s & 1;
        float td = 0.f, tm = 0.f;
#pragma unroll
        for (int p = 0; p < 4; p++) {
            int dot = 0, sum = 0;
#pragma unroll
            for (int l = 8 * p; l < 8 * p + 8; l++) {
                const int lo = hin ? (qs[l] >> 4) : (qs[l] & 15);
                const int q = lo + (((qh[l] >> s) & 1) ? 16 : 0);
                dot += q * x[l];
                sum += x[l];
            }
            td += d8 * (dot * sc);
            tm += d8 * (sum * mm);
        }
        return d * td - dmin * tm;
    } else if constexpr (FMT == FMT_Q51) {
        // transcribed verbatim from ggml_vec_dot_q5_1_q8_1_generic: the qh bit fold to 0x10
        // ((qh>>j)<<4 & 0x10 for the low half, (qh>>(j+12)) & 0x10 for the high), the two
        // 16-elem int partials, then (dx*dy)*sumi + mx*sy. The group IS the 32-elem block,
        // so the integers are bit-identical; only the cross-block order differs (~1e-6).
        // The dp4a form: the packed SPLIT layout (byte j: lo nib = elem j, hi = elem 16+j)
        // packs the lanes cleanly, and the qh bits fold 4 at a time: the elems 4q..4q+3's
        // bits are the CONSECUTIVE qh bits (4q..4q+3) - one nibble - so they spread into the
        // 4 lane positions with the magic (n * 0x00204081 & 0x01010101) << 4 (and 16+4q for
        // the hi half; the 0x01010101 direct mask would wrongly pick bits 8 apart).
        const int64_t b = row * (W.cols / 32) + g;
        const int* ci = (const int*)(W.codes + b * 16);
        const int* xi = (const int*)x;
        const uint32_t qh = (uint32_t)W.hi[b * 4] | ((uint32_t)W.hi[b * 4 + 1] << 8) |
                            ((uint32_t)W.hi[b * 4 + 2] << 16) | ((uint32_t)W.hi[b * 4 + 3] << 24);
        const float dx = h2f(W.d[b]), mx = h2f(W.m[b]);
        const int m = 0x0F0F0F0F;
        int sumi = 0;
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const unsigned n0 = (qh >> (4 * q)) & 0xFu, n1 = (qh >> (16 + 4 * q)) & 0xFu;
            const int lo = (int)((ci[q] & m) | (((n0 * 0x00204081u) & 0x01010101u) << 4));
            const int hi = (int)(((ci[q] >> 4) & m) | (((n1 * 0x00204081u) & 0x01010101u) << 4));
            sumi = __dp4a(lo, xi[q], sumi);
            sumi = __dp4a(hi, xi[4 + q], sumi);
        }
        return (dx * d8) * (float)sumi + mx * s8;
    } else if constexpr (FMT == FMT_K6) {
        const int64_t nb = W.cols / 256;
        const int64_t blk = row * nb + (g >> 3);
        const int gg = (int)(g & 7);
        const int n = gg >> 2, qd = gg & 3;
        const uint8_t* ql = W.codes + blk * 128 + 64 * n + (qd & 1) * 32;
        const uint8_t* qh = W.hi + blk * 64 + 32 * n;
        const int8_t* sc = (const int8_t*)W.meta + blk * 16 + 8 * n + 2 * qd;
        float acc = 0.f;
#pragma unroll
        for (int p = 0; p < 8; p++) {
            int dot = 0;
#pragma unroll
            for (int l = 4 * p; l < 4 * p + 4; l++) {
                const int lo = (qd >= 2) ? (ql[l] >> 4) : (ql[l] & 15);
                const int hb = (qh[l] >> (2 * qd)) & 3;
                dot += ((lo | (hb << 4)) - 32) * x[l];
            }
            acc += d8 * (dot * sc[p >> 2]);
        }
        return h2f(W.d[blk]) * acc;
    } else {
        return 0.f;
    }
}

template <int FMT>
__global__ void __launch_bounds__(256) k_gemv_q8(PackedW W, const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                                 const float* __restrict__ xs, float* __restrict__ y) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t row = (int64_t)blockIdx.x * 8 + warp;
    if (row >= W.rows) return;
    const int64_t ng = W.cols / 32;
    float acc = 0.f;
    for (int64_t g = lane; g < ng; g += 32) {
        int8_t xv[32];
        *(int4*)xv = __ldg((const int4*)(xq + g * 32));
        *(int4*)(xv + 16) = __ldg((const int4*)(xq + g * 32 + 16));
        acc += dot_q8<FMT>(W, row, g, xv, xd[g], xs[g]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = acc;
}

void launch_gemv_q8(const PackedW& W, const int8_t* xq, const float* xd, const float* xs, float* y, cudaStream_t s) {
    const unsigned G = (unsigned)((W.rows + 7) / 8);
    switch (W.fmt) {
        case FMT_P4: k_gemv_q8<FMT_P4><<<G, 256, 0, s>>>(W, xq, xd, xs, y); break;
        case FMT_P4M: k_gemv_q8<FMT_P4M><<<G, 256, 0, s>>>(W, xq, xd, xs, y); break;
        case FMT_Q8: k_gemv_q8<FMT_Q8><<<G, 256, 0, s>>>(W, xq, xd, xs, y); break;
        case FMT_K5: k_gemv_q8<FMT_K5><<<G, 256, 0, s>>>(W, xq, xd, xs, y); break;
        case FMT_K6: k_gemv_q8<FMT_K6><<<G, 256, 0, s>>>(W, xq, xd, xs, y); break;
        case FMT_Q51: k_gemv_q8<FMT_Q51><<<G, 256, 0, s>>>(W, xq, xd, xs, y); break;
        default: break;
    }
}

// ------------------------------------------------------------------------- Q8_K activations (the K-quants)
// ggml's nearest_int verbatim (the 12582912 magic): the assert's range holds by construction
// (|iscale*x| <= 127 < 4194303).
__device__ __forceinline__ int nearest_int_ggml(float fval) {
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// quantize_row_q8_K_ref verbatim (the CPU from_float for every K-quant pairing): the signed
// max-abs 'max', iscale = -127/max (the negative quirk that serves IQ2_XXS), nearest_int's
// magic round, MIN(127, v), the 16 int16 bsums (16-elem partial sums), d = 1/iscale; the
// all-zero block: d = 0, qs = 0. One 256-thread block per super-block; the (amax, max) pair
// reduce carries the first-occurrence tie-break (the lowest index wins, like the sequential
// ggml loop).
__global__ void k_quantize_q8_K(const float* __restrict__ x, int K, int8_t* __restrict__ qs,
                                 int16_t* __restrict__ bsums, float* __restrict__ d) {
    const int sb = blockIdx.x, i = threadIdx.x;  // K % 256 == 0
    const float xv = x[(size_t)sb * 256 + i];
    float am = fabsf(xv), mx = xv;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float oam = __shfl_xor_sync(0xffffffffu, am, o);
        const float omx = __shfl_xor_sync(0xffffffffu, mx, o);
        if (oam > am || (oam == am && (threadIdx.x & o) != 0)) { am = oam; mx = omx; }  // the lower lane wins ties
    }
    __shared__ float2 sh[9];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) sh[wid] = make_float2(am, mx);
    __syncthreads();
    if (i == 0) {
        int w = 0;
#pragma unroll
        for (int k = 1; k < 8; k++)
            if (sh[k].x > sh[w].x) w = k;  // the lowest warp wins ties = the lowest element
        sh[8] = sh[w];  // the broadcast slot past the 8 per-warp pairs
    }
    __syncthreads();
    const float amax = sh[8].x, maxv = sh[8].y;
    if (amax == 0.f) {
        qs[(size_t)sb * 256 + i] = 0;
        if (i == 0) d[sb] = 0.f;
        return;
    }
    const float iscale = -127.f / maxv;
    int v = nearest_int_ggml(iscale * xv);
    if (127 < v) v = 127;  // ggml's MIN(127, v)
    qs[(size_t)sb * 256 + i] = (int8_t)v;
    int sm = v;
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) sm += __shfl_xor_sync(0xffffffffu, sm, o);  // within the 16-lane group
    if ((i & 15) == 0) bsums[(size_t)sb * 16 + (i >> 4)] = (int16_t)sm;
    if (i == 0) d[sb] = 1.f / iscale;
}

void launch_quantize_q8_K(const float* x, int K, int8_t* qs, int16_t* bsums, float* d, cudaStream_t s) {
    k_quantize_q8_K<<<K / 256, 256, 0, s>>>(x, K, qs, bsums, d);
}

// the Q8_K-paired dots for one 32-elem group (the lane holds the group's 32 x codes in xv,
// the group's 2 bsums and the super-block's d separately). Transcribed from the ggml
// vec_dot_*_q8_K_generic bodies; the integer work is exact and order-free, so the per-group
// split (the ggml's 8 aux32 lanes) only regroups the fp32 folds (~1e-6 class). The meta decode
// uses dev_scale_min_k4, the same 6-bit values the ggml utmp dance produces.
template <int FMT>
__device__ __forceinline__ float dot_q8k(const PackedW& W, int64_t row, int64_t g, const int8_t* xv, int16_t bs0,
                                         int16_t bs1, float yd) {
    if constexpr (FMT == FMT_K2) {
        // ggml_vec_dot_q2_K_q8_K_generic: dall*isum - dmin*summs; sub s = elems 16s..16s+15,
        // byte window 32*((g&7)>>2), shift (g&3)*2, the +16 half, a = sc&15, m = sc>>4.
        const int64_t nb = W.cols / 256;
        const int64_t blk = row * nb + (g >> 3);
        const uint8_t* meta = W.meta + blk * 20;
        const float dx = h2f((uint16_t)(meta[0] | (meta[1] << 8)));
        const float dminx = h2f((uint16_t)(meta[2] | (meta[3] << 8)));
        const uint8_t* sc = meta + 4;
        const uint8_t* qs = W.codes + blk * 64 + 32 * ((g & 7) >> 2);
        const int sh = (int)(g & 3) << 1;
        const int s0 = 2 * (int)(g & 7);
        const int a0 = sc[s0] & 15, m0 = sc[s0] >> 4;
        const int a1 = sc[s0 + 1] & 15, m1 = sc[s0 + 1] >> 4;
        int is0 = 0, is1 = 0;
#pragma unroll
        for (int l = 0; l < 16; l++) {
            is0 += xv[l] * ((qs[l] >> sh) & 3);
            is1 += xv[l + 16] * ((qs[l + 16] >> sh) & 3);
        }
        const float dall = yd * dx, dmin = yd * dminx;
        return dall * (float)(a0 * is0 + a1 * is1) - dmin * (float)((int)bs0 * m0 + (int)bs1 * m1);
    } else if constexpr (FMT == FMT_K4) {
        // ggml_vec_dot_q4_K_q8_K_generic: sums[l] += d*aux32[l], sumf -= dmin*sumi; the group's
        // 2 bsums share mi (mins[j/2], j = 2s, 2s+1); sc*sumi is the int-distributive same total.
        const int64_t nb = W.cols / 256;
        const int64_t blk = row * nb + (g >> 3);
        const uint8_t* meta = W.meta + blk * 16;
        const float dx = h2f((uint16_t)(meta[0] | (meta[1] << 8)));
        const float dminx = h2f((uint16_t)(meta[2] | (meta[3] << 8)));
        int sc, mi;
        dev_scale_min_k4((int)(g & 7), meta + 4, sc, mi);
        // the dp4a form: the group = one 32-elem sub = the 32 window bytes at ONE shared nibble
        // plane (byte l = elem l), so the lanes pack cleanly - no bias, the mins fold outside
        const int* qi = (const int*)(W.codes + blk * 128 + 32 * ((g & 7) >> 1));
        const int* xi = (const int*)xv;
        const int m = 0x0F0F0F0F;
        const int hin = (int)(g & 1);
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 8; l++) sumi = __dp4a(hin ? (qi[l] >> 4) & m : qi[l] & m, xi[l], sumi);
        return (dx * yd) * (float)(sc * sumi) - (dminx * yd) * (float)(((int)bs0 + (int)bs1) * mi);
    } else if constexpr (FMT == FMT_IQ1S) {
        // ggml vec_dot_iq1_s_q8_1 translated to the PackedW planes + the q8_K pairing:
        // the nibble grid (the halves-interleave, t4q_iq1s_grid_gpu) unpacks to 2 dp4a
        // byte-quads per 8 elems, 4 iterations per 32-group; the weight value = L +
        // (delta-1), delta = +-0.125 by the qh shift bit, so the group dot = d1q*yd*(sumi +
        // (delta-1)*(bs0+bs1)) - the SAME two bsums the K2/K4 branch reads (no new
        // activation format; the ggml q8_1 s-term becomes the bsums pair here).
        const int64_t nb = W.cols / 256;
        const int64_t blk = row * nb + (g >> 3);
        const int ib = (int)(g & 7);
        const int qs4 = *(const int*)(W.codes + blk * 32 + 4 * ib);
        const uint8_t* qs = (const uint8_t*)&qs4;
        const int qh = ((const uint16_t*)W.hi + blk * 8)[ib];
        int sumi = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int grid = t4q_iq1s_grid_gpu[qs[k] | (((qh >> (3 * k)) & 0x07) << 8)];
            const int grid0 = (grid >> 0) & 0x0F0F0F0F;  // bytes = L_0..L_3 (elems 8k+0..3)
            const int grid1 = (grid >> 4) & 0x0F0F0F0F;  // bytes = L_4..L_7 (elems 8k+4..7)
            const int* xw = (const int*)xv;
            sumi = __dp4a(grid0, xw[2 * k], sumi);
            sumi = __dp4a(grid1, xw[2 * k + 1], sumi);
        }
        const float d1q = h2f(W.d[blk]) * (float)(((qh >> 11) & 0x0E) + 1);
        const float delta = -1.f + T4Q_IQ1S_DELTA - (float)(qh & 0x8000) * (2.f * T4Q_IQ1S_DELTA / 0x8000);
        return d1q * yd * ((float)sumi + delta * (float)((int)bs0 + (int)bs1));
    } else {
        return 0.f;
    }
}

template <int FMT>
__global__ void __launch_bounds__(256) k_gemv_q8k(PackedW W, const int8_t* __restrict__ xq,
                                                  const int16_t* __restrict__ bsums, const float* __restrict__ yd,
                                                  float* __restrict__ y) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t row = (int64_t)blockIdx.x * 8 + warp;
    if (row >= W.rows) return;
    const int64_t ng = W.cols / 32;
    float acc = 0.f;
    for (int64_t g = lane; g < ng; g += 32) {
        int8_t xv[32];
        *(int4*)xv = __ldg((const int4*)(xq + g * 32));
        *(int4*)(xv + 16) = __ldg((const int4*)(xq + g * 32 + 16));
        const int64_t sb = g >> 3;
        const int sub = 2 * (int)(g & 7);  // K2: the subs s0, s0+1; K4/IQ1S: the bsums 2s, 2s+1
        acc += dot_q8k<FMT>(W, row, g, xv, bsums[sb * 16 + sub], bsums[sb * 16 + sub + 1], yd[sb]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = acc;
}

void launch_gemv_q8k(const PackedW& W, const int8_t* xq, const int16_t* bsums, const float* yd, float* y,
                      cudaStream_t s) {
    const unsigned G = (unsigned)((W.rows + 7) / 8);
    switch (W.fmt) {
        case FMT_K2: k_gemv_q8k<FMT_K2><<<G, 256, 0, s>>>(W, xq, bsums, yd, y); break;
        case FMT_K4: k_gemv_q8k<FMT_K4><<<G, 256, 0, s>>>(W, xq, bsums, yd, y); break;
        // cf-m6 r2 (CF_REQUANT.md section 4): the M=1 greedy path for the iq1_s experts.
        // NOTE: NOT the batched _b form (it re-decodes W per (row, batch) pair - the spec
        // freezes the amortized M=8 verify kernel as r5, a different kernel, not this
        // dot's drop-in).
        case FMT_IQ1S: k_gemv_q8k<FMT_IQ1S><<<G, 256, 0, s>>>(W, xq, bsums, yd, y); break;
        default: break;
    }
}

// The per-pick W-table batched variant (the cf-m3 tiering mechanism, adopted by the default
// path as its stepping stone): grid.y = the pick, and the pick's slab view is READ from
// wt[by] (a device table of PackedW views) instead of a uniform-stride advance. With the
// identity table (the pick's staged slab views) the pointers are exactly the ones the
// stride math produced, so the default path is bit-identical; the tiering later swaps in
// per-hit resident views with zero kernel change. The x/bsums/d strides are 0 when the
// activation is shared (the y_gu gate|up gemv quantizes the xn ONCE); y advances per pick.
template <int FMT>
__global__ void __launch_bounds__(256) k_gemv_q8k_b(const PackedW* __restrict__ wt, const int8_t* __restrict__ xq,
                                                   const int16_t* __restrict__ bsums, const float* __restrict__ yd,
                                                   float* __restrict__ y, int64_t x_stride, int64_t bs_stride,
                                                   int64_t d_stride, int64_t y_stride) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const PackedW W = wt[blockIdx.y];
    const int64_t row = (int64_t)blockIdx.x * 8 + warp;
    if (row >= W.rows) return;
    const int8_t* __restrict__ xb = xq + (int64_t)blockIdx.y * x_stride;
    const int16_t* __restrict__ bsb = bsums + (int64_t)blockIdx.y * bs_stride;
    const float* __restrict__ db = yd + (int64_t)blockIdx.y * d_stride;
    float* __restrict__ yb = y + (int64_t)blockIdx.y * y_stride;
    const int64_t ng = W.cols / 32;
    float acc = 0.f;
    for (int64_t g = lane; g < ng; g += 32) {
        int8_t xv[32];
        *(int4*)xv = __ldg((const int4*)(xb + g * 32));
        *(int4*)(xv + 16) = __ldg((const int4*)(xb + g * 32 + 16));
        const int64_t sb = g >> 3;
        const int sub = 2 * (int)(g & 7);  // K2: the subs s0, s0+1; K4: the bsums 2s, 2s+1
        acc += dot_q8k<FMT>(W, row, g, xv, bsb[sb * 16 + sub], bsb[sb * 16 + sub + 1], db[sb]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) yb[row] = acc;
}

void launch_gemv_q8k_b(const PackedW* wt, int fmt, int rows, const int8_t* xq, const int16_t* bsums,
                       const float* yd, float* y, int64_t x_stride, int64_t bs_stride, int64_t d_stride,
                       int64_t y_stride, int batch, cudaStream_t s) {
    // rows is the HOST-side row count (the grid covers it; wt[] is device memory)
    const unsigned G = (unsigned)((rows + 7) / 8);
    const dim3 grid(G, batch);
    switch (fmt) {
        case FMT_K2:
            k_gemv_q8k_b<FMT_K2><<<grid, 256, 0, s>>>(wt, xq, bsums, yd, y, x_stride, bs_stride, d_stride, y_stride);
            break;
        case FMT_K4:
            k_gemv_q8k_b<FMT_K4><<<grid, 256, 0, s>>>(wt, xq, bsums, yd, y, x_stride, bs_stride, d_stride, y_stride);
            break;
        default: break;
    }
}

// ------------------------------------------------------------------------- Q8_0 activations (the Q4_0 pairing)
// quantize_row_q8_0_ref verbatim: d = amax/127 rounded to fp16, id = d ? 1/d : 0, q = roundf(x*id);
// PLUS the per-32-block signed code sum (the dp4a factored bias below: sum (nib-8)*x =
// sum nib*x - 8*sum x, so the dot never pays the per-lane -8; the 27B group_dot's pattern).
__global__ void k_quantize_q8_0(const float* __restrict__ x, int K, int8_t* __restrict__ xq, float* __restrict__ xd,
                                int* __restrict__ xs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // K % 32 == 0
    if (i >= K) return;
    const float xi = x[i];
    float amax = fabsf(xi);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f;
    const float id = d ? 1.0f / d : 0.0f;
    const int q = (int)roundf(xi * id);
    xq[i] = (int8_t)q;
    int sum = q;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if ((i & 31) == 0) {
        xd[i >> 5] = __half2float(__float2half_rn(d));
        xs[i >> 5] = sum;
    }
}

void launch_quantize_q8_0(const float* x, int K, int8_t* xq, float* xd, int* xs, cudaStream_t s) {
    k_quantize_q8_0<<<(K + 255) / 256, 256, 0, s>>>(x, K, xq, xd, xs);
}

// ggml_vec_dot_q4_0_q8_0_generic verbatim (the PLAIN form - no sum fold): v = nib-8, the two
// 16-elem int partials, then sumi * dx * dy (the left-assoc chain). The group IS the 32-block,
// so the integers are bit-identical to the oracle's own Q4_0xQ8_0 arithmetic. The dp4a form
// (the 27B group_dot's measured 254 GB/s class vs this family's issue-bound ~122): the lo
// nibbles = elems 0..15, hi = 16..31 (the packed SPLIT layout), and the -8 bias factors out
// exactly in int32: sum (nib-8)*x = sum nib*x - 8*sum x, with the block sum carried by the
// quantize's xs plane - no overflow (|s| <= 32*15*127 + 8*32*127 << 2^31).
__device__ __forceinline__ float dot_q8_0_p4(const PackedW& W, int64_t row, int64_t g, const int8_t* x, float dy,
                                             int s32) {
    const int64_t b = row * (W.cols / 32) + g;
    const uint4 c4 = *(const uint4*)(W.codes + b * 16);
    const int* ci = (const int*)&c4;
    const int* xi = (const int*)x;
    const int m = 0x0F0F0F0F;
    int s = 0;
#pragma unroll
    for (int q = 0; q < 4; q++) s = __dp4a(ci[q] & m, xi[q], s);
#pragma unroll
    for (int q = 0; q < 4; q++) s = __dp4a((ci[q] >> 4) & m, xi[4 + q], s);
    s -= 8 * s32;  // the factored bias: identical int32 to the per-lane (nib-8)*x form
    return (float)s * h2f(W.d[b]) * dy;
}

template <int FMT>
__global__ void __launch_bounds__(256) k_gemv_q80(PackedW W, const int8_t* __restrict__ xq,
                                                  const float* __restrict__ xd, const int* __restrict__ xs,
                                                  float* __restrict__ y) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t row = (int64_t)blockIdx.x * 8 + warp;
    if (row >= W.rows) return;
    const int64_t ng = W.cols / 32;
    float acc = 0.f;
    for (int64_t g = lane; g < ng; g += 32) {
        int8_t xv[32];
        *(int4*)xv = __ldg((const int4*)(xq + g * 32));
        *(int4*)(xv + 16) = __ldg((const int4*)(xq + g * 32 + 16));
        acc += dot_q8_0_p4(W, row, g, xv, xd[g], xs[g]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = acc;
}

// the batched variant for the 10-expert SoA-staged down slabs, in the per-pick W-table form
// (the same mechanism as k_gemv_q8k_b): grid.y = the pick, the pick's slab view read from
// wt[by]; the identity table (the staged slab views) is bit-identical to the old
// uniform-stride advance. The xq/xd/xs planes advance by the per-pick strides (the ffa
// [TOPK][EE] is flat, the expert boundaries are the 32-group boundaries, so one flat
// quantize feeds it; the xs shares the xd stride - both per-32-block planes).
__global__ void __launch_bounds__(256) k_gemv_q80_b(const PackedW* __restrict__ wt, const int8_t* __restrict__ xq,
                                                    const float* __restrict__ xd, const int* __restrict__ xs,
                                                    float* __restrict__ y, int64_t x_stride, int64_t y_stride,
                                                    int64_t xd_stride) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const PackedW W = wt[blockIdx.y];
    const int64_t row = (int64_t)blockIdx.x * 8 + warp;
    if (row >= W.rows) return;
    const int8_t* __restrict__ xb = xq + (int64_t)blockIdx.y * x_stride;
    const float* __restrict__ db = xd + (int64_t)blockIdx.y * xd_stride;
    const int* __restrict__ sb = xs + (int64_t)blockIdx.y * xd_stride;
    float* __restrict__ yb = y + (int64_t)blockIdx.y * y_stride;
    const int64_t ng = W.cols / 32;
    float acc = 0.f;
    for (int64_t g = lane; g < ng; g += 32) {
        int8_t xv[32];
        *(int4*)xv = __ldg((const int4*)(xb + g * 32));
        *(int4*)(xv + 16) = __ldg((const int4*)(xb + g * 32 + 16));
        acc += dot_q8_0_p4(W, row, g, xv, db[g], sb[g]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) yb[row] = acc;
}

void launch_gemv_q8_0(const PackedW& W, const int8_t* xq, const float* xd, const int* xs, float* y,
                      cudaStream_t s) {
    const unsigned G = (unsigned)((W.rows + 7) / 8);
    k_gemv_q80<FMT_P4><<<G, 256, 0, s>>>(W, xq, xd, xs, y);
}

void launch_gemv_q8_0_b(const PackedW* wt, int rows, const int8_t* xq, const float* xd, const int* xs, float* y,
                        int64_t x_stride, int64_t y_stride, int64_t xd_stride, int batch, cudaStream_t s) {
    // rows is the HOST-side row count (the grid covers it; wt[] is device memory)
    const unsigned G = (unsigned)((rows + 7) / 8);
    k_gemv_q80_b<<<dim3(G, batch), 256, 0, s>>>(wt, xq, xd, xs, y, x_stride, y_stride, xd_stride);
}
