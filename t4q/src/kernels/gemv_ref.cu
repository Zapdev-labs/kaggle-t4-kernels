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

// batched GEMV over K expert slabs staged SoA: grid.y = the expert, the same per-row body as
// k_gemv verbatim (bit-identical accumulation), with the x/y and W planes advanced per expert.
// P4 (Q4_0) only today - the moe down experts; the other formats grow a case when a tier needs them.
template <int FMT>
__global__ void __launch_bounds__(256) k_gemv_b(PackedW W, const float* __restrict__ x, float* __restrict__ y,
                                                int64_t x_stride, int64_t y_stride, int64_t codes_stride,
                                                int64_t d_stride) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t row = (int64_t)blockIdx.x * 8 + warp;
    if (row >= W.rows) return;
    W.codes += (size_t)blockIdx.y * codes_stride;
    W.d = (uint16_t*)((uint8_t*)W.d + (size_t)blockIdx.y * d_stride);
    const float* __restrict__ xb = x + (int64_t)blockIdx.y * x_stride;
    float* __restrict__ yb = y + (int64_t)blockIdx.y * y_stride;
    const int64_t ng = W.cols / 32;
    float acc = 0.f;
    for (int64_t g = lane; g < ng; g += 32) {
        float w[32];
        deq32<FMT>(W, row, g, w);
        const float4* xv = (const float4*)(xb + g * 32);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float4 a = __ldg(xv + j);
            acc += w[4 * j] * a.x + w[4 * j + 1] * a.y + w[4 * j + 2] * a.z + w[4 * j + 3] * a.w;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) yb[row] = acc;
}

// W describes ONE expert slab (rows/cols); the codes/d strides are per-expert byte offsets in the
// SoA staging; x_stride/y_stride the per-expert activation/output element strides.
void launch_gemv_batched(const PackedW& W, const float* x, float* y, int64_t x_stride, int64_t y_stride, int batch,
                         cudaStream_t s) {
    const unsigned G = (unsigned)((W.rows + 7) / 8);
    const int64_t cs = W.rows * (W.cols / 2);          // P4 codes bytes per expert slab
    const int64_t ds = W.rows * (W.cols / 32) * 2;      // P4 fp16 d bytes per expert slab
    k_gemv_b<FMT_P4><<<dim3(G, batch), 256, 0, s>>>(W, x, y, x_stride, y_stride, cs, ds);
}

// ------------------------------------------------------------------------------------------- q8_1 activations
__global__ void k_quantize_q8_1(const float* __restrict__ x, int K, int8_t* xq, float* xd, float* xs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // K % 32 == 0, blockDim multiple of 32
    if (i >= K) return;
    const float xi = x[i];
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = amax / 127.0f;
    xq[i] = amax == 0.0f ? 0 : (int8_t)roundf(xi / d);
    if ((i & 31) == 0) {
        xd[i >> 5] = __half2float(__float2half_rn(d));
        xs[i >> 5] = __half2float(__float2half_rn(sum));
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
        const int64_t b = row * (W.cols / 32) + g;
        const uint8_t* c = W.codes + b * 16;
        const uint32_t qh = (uint32_t)W.hi[b * 4] | ((uint32_t)W.hi[b * 4 + 1] << 8) |
                            ((uint32_t)W.hi[b * 4 + 2] << 16) | ((uint32_t)W.hi[b * 4 + 3] << 24);
        const float dx = h2f(W.d[b]), mx = h2f(W.m[b]);
        int sumi0 = 0, sumi1 = 0;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const int xh0 = (int)(((qh >> j) << 4) & 0x10u);
            const int xh1 = (int)((qh >> (j + 12)) & 0x10u);
            sumi0 += ((c[j] & 0xF) | xh0) * x[j];
            sumi1 += ((c[j] >> 4) | xh1) * x[j + 16];
        }
        return (dx * d8) * (float)(sumi0 + sumi1) + mx * s8;
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
        const uint8_t* qs = W.codes + blk * 128 + 32 * ((g & 7) >> 1);
        const int hin = (int)(g & 1);
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 32; l++) {
            const int lo = hin ? (qs[l] >> 4) : (qs[l] & 15);
            sumi += lo * xv[l];
        }
        return (dx * yd) * (float)(sc * sumi) - (dminx * yd) * (float)(((int)bs0 + (int)bs1) * mi);
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
        const int sub = 2 * (int)(g & 7);  // K2: the subs s0, s0+1; K4: the bsums 2s, 2s+1
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
        default: break;
    }
}

// ------------------------------------------------------------------------- Q8_0 activations (the Q4_0 pairing)
// quantize_row_q8_0_ref verbatim: d = amax/127 rounded to fp16, id = d ? 1/d : 0, q = roundf(x*id).
__global__ void k_quantize_q8_0(const float* __restrict__ x, int K, int8_t* __restrict__ xq, float* __restrict__ xd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // K % 32 == 0
    if (i >= K) return;
    const float xi = x[i];
    float amax = fabsf(xi);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f;
    const float id = d ? 1.0f / d : 0.0f;
    xq[i] = (int8_t)roundf(xi * id);
    if ((i & 31) == 0) xd[i >> 5] = __half2float(__float2half_rn(d));
}

void launch_quantize_q8_0(const float* x, int K, int8_t* xq, float* xd, cudaStream_t s) {
    k_quantize_q8_0<<<(K + 255) / 256, 256, 0, s>>>(x, K, xq, xd);
}

// ggml_vec_dot_q4_0_q8_0_generic verbatim (the PLAIN form - no sum fold): v = nib-8, the two
// 16-elem int partials, then sumi * dx * dy (the left-assoc chain). The group IS the 32-block,
// so the integers are bit-identical to the oracle's own Q4_0xQ8_0 arithmetic.
__device__ __forceinline__ float dot_q8_0_p4(const PackedW& W, int64_t row, int64_t g, const int8_t* x, float dy) {
    const int64_t b = row * (W.cols / 32) + g;
    const uint4 c4 = *(const uint4*)(W.codes + b * 16);
    const uint8_t* c = (const uint8_t*)&c4;
    int s0 = 0, s1 = 0;
#pragma unroll
    for (int j = 0; j < 16; j++) {
        s0 += ((c[j] & 15) - 8) * x[j];
        s1 += ((c[j] >> 4) - 8) * x[j + 16];
    }
    return (float)(s0 + s1) * h2f(W.d[b]) * dy;
}

template <int FMT>
__global__ void __launch_bounds__(256) k_gemv_q80(PackedW W, const int8_t* __restrict__ xq,
                                                  const float* __restrict__ xd, float* __restrict__ y) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t row = (int64_t)blockIdx.x * 8 + warp;
    if (row >= W.rows) return;
    const int64_t ng = W.cols / 32;
    float acc = 0.f;
    for (int64_t g = lane; g < ng; g += 32) {
        int8_t xv[32];
        *(int4*)xv = __ldg((const int4*)(xq + g * 32));
        *(int4*)(xv + 16) = __ldg((const int4*)(xq + g * 32 + 16));
        acc += dot_q8_0_p4(W, row, g, xv, xd[g]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = acc;
}

// the batched variant for the 10-expert SoA-staged down slabs: grid.y = the expert, the
// xq/xd planes advance by the per-expert strides (the ffa [TOPK][EE] is flat, the expert
// boundaries are the 32-group boundaries, so one flat quantize feeds it).
__global__ void __launch_bounds__(256) k_gemv_q80_b(PackedW W, const int8_t* __restrict__ xq,
                                                    const float* __restrict__ xd, float* __restrict__ y,
                                                    int64_t x_stride, int64_t y_stride, int64_t xd_stride,
                                                    int64_t codes_stride, int64_t d_stride) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t row = (int64_t)blockIdx.x * 8 + warp;
    if (row >= W.rows) return;
    W.codes += (size_t)blockIdx.y * codes_stride;
    W.d = (uint16_t*)((uint8_t*)W.d + (size_t)blockIdx.y * d_stride);
    const int8_t* __restrict__ xb = xq + (int64_t)blockIdx.y * x_stride;
    const float* __restrict__ db = xd + (int64_t)blockIdx.y * xd_stride;
    float* __restrict__ yb = y + (int64_t)blockIdx.y * y_stride;
    const int64_t ng = W.cols / 32;
    float acc = 0.f;
    for (int64_t g = lane; g < ng; g += 32) {
        int8_t xv[32];
        *(int4*)xv = __ldg((const int4*)(xb + g * 32));
        *(int4*)(xv + 16) = __ldg((const int4*)(xb + g * 32 + 16));
        acc += dot_q8_0_p4(W, row, g, xv, db[g]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) yb[row] = acc;
}

void launch_gemv_q8_0(const PackedW& W, const int8_t* xq, const float* xd, float* y, cudaStream_t s) {
    const unsigned G = (unsigned)((W.rows + 7) / 8);
    k_gemv_q80<FMT_P4><<<G, 256, 0, s>>>(W, xq, xd, y);
}

void launch_gemv_q8_0_b(const PackedW& W, const int8_t* xq, const float* xd, float* y, int64_t x_stride,
                        int64_t y_stride, int64_t xd_stride, int batch, cudaStream_t s) {
    // W describes ONE expert slab; the codes/d strides are per-expert byte offsets (P4)
    const unsigned G = (unsigned)((W.rows + 7) / 8);
    const int64_t cs = W.rows * (W.cols / 2);
    const int64_t ds = W.rows * (W.cols / 32) * 2;
    k_gemv_q80_b<<<dim3(G, batch), 256, 0, s>>>(W, xq, xd, y, x_stride, y_stride, xd_stride, cs, ds);
}
