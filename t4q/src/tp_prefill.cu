// TP=2 batched prefill (milestone P): ubatches of tokens through all 64 layers with the W4A8 tensor-core GEMM
// (kernels/gemm.cuh) reading the decode weights in place, a token-sequential DeltaNet scan with the decode step's
// math (state in registers), causal attention writing the decode KV cache, and an fp32 all-reduce of the K-split
// partials through peer copies. Tokens [0, n-1) go through the batch path; the last prompt token runs as a normal
// decode step, which produces the logits / first generated token and leaves StepState exactly as decode expects.
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "kernels/gemm.cuh"
#include "kernels/gemm8.cuh"
#include "kernels/gemm16.cuh"
#include "kernels/gemm_r.cuh"
#include "kernels/rot.cuh"
#include "kernels/tp_kernels.h"
#include "model.h"
#include "tp.h"
#include "tp_api.h"

using hp::D;
using namespace t4q;

namespace {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float block_sum(float v, float* red) {  // same order as tp_kernels.cu block_sum
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < nw; i++) t += red[i];
    return t;
}
__device__ __forceinline__ float h2f_u16(uint16_t b) { return __half2float(__ushort_as_half(b)); }

// ------------------------------------------------------------------------------------------------ kernels
// h[t] = embed(ids[t]) (Q4_0 row dequant, same arithmetic as k_embed). grid (T, 20) x 256
__global__ void k_pf_embed(const uint8_t* __restrict__ embd, const int* __restrict__ ids, float* h) {
    const int t = blockIdx.x, i = blockIdx.y * 256 + threadIdx.x;
    int tok = ids[t];
    if (tok < 0 || tok >= 248320) tok = 0;
    const uint8_t* b = embd + (size_t)tok * 2880 + (i >> 5) * 18;
    const float d = h2f_u16((uint16_t)(b[0] | (b[1] << 8)));
    const int e = i & 31;
    const int q = e < 16 ? (b[2 + e] & 15) : (b[2 + e - 16] >> 4);
    h[(size_t)t * D + i] = __fmul_rn((float)(q - 8), d);
}

// [h += p0 + p1] (p0 = GPU0's partial on both GPUs, so both residuals stay bit-identical); xn = rmsnorm(h) * w.
// grid T x 256 threads (20 elements each)
__device__ __forceinline__ float ldp(const float* p, int i) { return p[i]; }
__device__ __forceinline__ float ldp(const __half* p, int i) { return __half2float(p[i]); }
template <class PT>
__global__ void __launch_bounds__(256) k_pf_add_norm(float* h, const PT* own, const PT* rx, int gpu, int add,
                                                     const float* __restrict__ w, float* xn) {
    __shared__ float red[8];
    const int t = blockIdx.x, tid = threadIdx.x;
    float* hp = h + (size_t)t * D;
    float x[20];
#pragma unroll
    for (int k = 0; k < 20; k++) x[k] = hp[tid + 256 * k];
    if (add) {
        const PT* p0 = (gpu == 0 ? own : rx) + (size_t)t * D;
        const PT* p1 = (gpu == 0 ? rx : own) + (size_t)t * D;
#pragma unroll
        for (int k = 0; k < 20; k++) {
            x[k] = x[k] + (ldp(p0, tid + 256 * k) + ldp(p1, tid + 256 * k));
            hp[tid + 256 * k] = x[k];
        }
    }
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 20; k++) ss += x[k] * x[k];
    ss = block_sum(ss, red);
    const float scale = rsqrtf(ss / 5120.f + 1e-6f);
#pragma unroll
    for (int k = 0; k < 20; k++) xn[(size_t)t * D + tid + 256 * k] = (x[k] * scale) * w[tid + 256 * k];
}

// ---- fused producers that quantize straight into the GEMM's activation layout (gemm.cuh, int8 path):
// xq [Tp][K] row tr, xs [K/32][Tp] {d/16, -(M + 128*sumq)*d/16}, xsum [K/32][Tp] d*sumq; one warp = one 32-group,
// lane = element (same math as gemm::quant_rows_kernel)
struct Q8Out {
    int8_t* xq;
    float2* xs;    // gemm.cuh meta (nullptr in gemm8 mode)
    float* xsum;
    float* dx;     // gemm8 block scales [K/ga][Tp] (nullptr in gemm.cuh mode)
    int K, Tp;
    int ga;        // gemm8 group: 32 (warp_q8) or 64 (warp pairs, q8_64)
};
// GA 64: v of element k with the 64-group amax (two consecutive warps) -> xq, dx (same math as gemm8::quant8_kernel<64>)
__device__ __forceinline__ void q8_64_write(float v, float amax, const Q8Out& q, int tr, int k) {
    const float d = amax / 127.f;
    const int qi = amax == 0.f ? 0 : __float2int_rn(v / d);
    q.xq[(size_t)tr * q.K + k] = (int8_t)qi;
    if ((threadIdx.x & 63) == 0) q.dx[(size_t)(k >> 6) * q.Tp + tr] = d;
}
__device__ __forceinline__ float warp_amax(float v) {
    float a = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    return a;
}
__device__ __forceinline__ void warp_q8(float v, const Q8Out& q, int tr, int k) {
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.f;
    const int qi = amax == 0.f ? 0 : __float2int_rn(v / d);
    int sq = qi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    q.xq[(size_t)tr * q.K + k] = (int8_t)qi;
    if ((threadIdx.x & 31) == 0) {
        const int b = k >> 5;
        if (q.dx) q.dx[(size_t)b * q.Tp + tr] = d;
        if (q.xs) {
            const float sc = d * 0.0625f;
            q.xs[(size_t)b * q.Tp + tr] = make_float2(sc, -(float)(12582912 + 128 * sq) * sc);
            q.xsum[(size_t)b * q.Tp + tr] = d * (float)sq;
        }
    }
}

// [h += p0 + p1]; x = rmsnorm(h) * w -> q8 (and fp32 xn if xn != nullptr). grid T x 256 (token tr of the sub-batch)
template <class PT>
__global__ void __launch_bounds__(256) k_pf_add_norm_q8(float* h, const PT* own, const PT* rx, int gpu, int add,
                                                        const float* __restrict__ w, float* xn, Q8Out q) {
    __shared__ float red[8];
    const int t = blockIdx.x, tid = threadIdx.x;
    float* hp = h + (size_t)t * D;
    float x[20];
#pragma unroll
    for (int k = 0; k < 20; k++) x[k] = hp[tid + 256 * k];
    if (add) {
        const PT* p0 = (gpu == 0 ? own : rx) + (size_t)t * D;
        const PT* p1 = (gpu == 0 ? rx : own) + (size_t)t * D;
#pragma unroll
        for (int k = 0; k < 20; k++) {
            x[k] = x[k] + (ldp(p0, tid + 256 * k) + ldp(p1, tid + 256 * k));
            hp[tid + 256 * k] = x[k];
        }
    }
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 20; k++) ss += x[k] * x[k];
    ss = block_sum(ss, red);
    const float scale = rsqrtf(ss / 5120.f + 1e-6f);
    if (q.ga == 64) {
        __shared__ float am[20][8];
#pragma unroll
        for (int k = 0; k < 20; k++) {
            const int e = tid + 256 * k;
            x[k] = (x[k] * scale) * w[e];
            if (xn) xn[(size_t)t * D + e] = x[k];
            const float a = warp_amax(x[k]);
            if ((tid & 31) == 0) am[k][tid >> 5] = a;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 20; k++)
            q8_64_write(x[k], fmaxf(am[k][tid >> 5], am[k][(tid >> 5) ^ 1]), q, t, tid + 256 * k);
        return;
    }
#pragma unroll
    for (int k = 0; k < 20; k++) {
        const int e = tid + 256 * k;
        const float y = (x[k] * scale) * w[e];
        if (xn) xn[(size_t)t * D + e] = y;
        warp_q8(y, q, t, e);
    }
}

// fp16 rows (K) -> GA64 q8: xq [Tp][K], dx [K/64][Tp]; grid (Tp, K/512) x 256 (warp = one 64-group); rows >= T zero
__global__ void __launch_bounds__(256) k_q8_64_half(const __half* __restrict__ x, int T, int Tp, int K, int8_t* xq,
                                                    float* dx) {
    const int t = blockIdx.x, lane = threadIdx.x & 31, grp = blockIdx.y * 8 + (threadIdx.x >> 5);
    const int k0 = grp * 64;
    float a = 0.f, b = 0.f;
    if (t < T) { a = __half2float(x[(size_t)t * K + k0 + lane]); b = __half2float(x[(size_t)t * K + k0 + 32 + lane]); }
    float m = fmaxf(fabsf(a), fabsf(b));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float d = m / 127.f;
    xq[(size_t)t * K + k0 + lane] = (int8_t)(m > 0.f ? __float2int_rn(a / d) : 0);
    xq[(size_t)t * K + k0 + 32 + lane] = (int8_t)(m > 0.f ? __float2int_rn(b / d) : 0);
    if (lane == 0) dx[(size_t)grp * Tp + t] = d;
}

// R512: gated RMSNorm of the 24 local heads (o * rsqrt(mean o^2) * w * silu(z)) into smem, then T(.) quantized per
// token (ssm_out input, K = 3072). grid T x 256 (warp w: heads 3w..3w+2, 4 elements per lane), dynamic smem 3072 floats
__global__ void __launch_bounds__(256) k_pf_gnorm_rot(const float* __restrict__ o, const float* __restrict__ y, int ldy,
                                                      const float* __restrict__ w, int8_t* xq, float* dx, float* dsr,
                                                      int Tp) {
    extern __shared__ __align__(16) float xs[];
    __shared__ float red[8];
    const int t = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int hh = 0; hh < 3; ++hh) {
        const int vl = warp * 3 + hh;
        const float4 x = *(const float4*)(o + (size_t)t * 3072 + vl * 128 + lane * 4);
        float ss = x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
        ss = warp_sum(ss);  // same value in every lane
        const float scale = rsqrtf(ss / 128.0f + 1e-6f);
        const float4 zz = *(const float4*)(y + (size_t)t * ldy + 5120 + vl * 128 + lane * 4);
        const float4 ww = *(const float4*)(w + lane * 4);
        float* d = xs + vl * 128 + lane * 4;
        d[0] = ((x.x * scale) * ww.x) * (zz.x / (1.0f + expf(-zz.x)));
        d[1] = ((x.y * scale) * ww.y) * (zz.y / (1.0f + expf(-zz.y)));
        d[2] = ((x.z * scale) * ww.z) * (zz.z / (1.0f + expf(-zz.z)));
        d[3] = ((x.w * scale) * ww.w) * (zz.w / (1.0f + expf(-zz.w)));
    }
    __syncthreads();
    rot::rot_quant_row(xs, xs, 3072, xq + (size_t)t * 3072, dx + t, red, dsr, t, Tp);
}

// R512: [h += partials]; x = rmsnorm(h) * w (xn written if non-null), then T(x) quantized per token into xq / dx.
// grid T x 256, dynamic smem 5120 floats
template <class PT>
__global__ void __launch_bounds__(256) k_pf_add_norm_rot(float* h, const PT* own, const PT* rx, int gpu, int add,
                                                         const float* __restrict__ w, float* xn, int8_t* xq, float* dx,
                                                         float* dsr, int Tp) {
    extern __shared__ __align__(16) float xs[];
    __shared__ float red[8];
    const int t = blockIdx.x, tid = threadIdx.x;
    float* hp = h + (size_t)t * D;
    float x[20];
#pragma unroll
    for (int k = 0; k < 20; k++) x[k] = hp[tid + 256 * k];
    if (add) {
        const PT* p0 = (gpu == 0 ? own : rx) + (size_t)t * D;
        const PT* p1 = (gpu == 0 ? rx : own) + (size_t)t * D;
#pragma unroll
        for (int k = 0; k < 20; k++) {
            x[k] = x[k] + (ldp(p0, tid + 256 * k) + ldp(p1, tid + 256 * k));
            hp[tid + 256 * k] = x[k];
        }
    }
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 20; k++) ss += x[k] * x[k];
    ss = block_sum(ss, red);
    const float scale = rsqrtf(ss / 5120.f + 1e-6f);
#pragma unroll
    for (int k = 0; k < 20; k++) {
        const int e = tid + 256 * k;
        const float y = (x[k] * scale) * w[e];
        xs[e] = y;
        if (xn) xn[(size_t)t * D + e] = y;
    }
    __syncthreads();
    rot::rot_quant_row(xs, xs, D, xq + (size_t)t * D, dx + t, red, dsr, t, Tp);
}

// int8 buffers a vs b: out[0] = max |a - b|, out[1] = count of |a - b| > 1, out[2] = count of a != b
__global__ void k_i8diff(const int8_t* a, const int8_t* b, size_t n, unsigned* out) {
    unsigned mx = 0, c1 = 0, c0 = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const int d = abs((int)a[i] - (int)b[i]);
        mx = max(mx, (unsigned)d); c1 += d > 1; c0 += d != 0;
    }
    atomicMax(out, mx); atomicAdd(out + 1, c1); atomicAdd(out + 2, c0);
}

// silu(g) * u (gate|up rows interleaved by 4) -> q8 (K = 8704). grid (T, 34) x 256
__global__ void k_pf_silu_q8(const float* __restrict__ y, int ldy, Q8Out q) {
    const int t = blockIdx.x, i = blockIdx.y * 256 + threadIdx.x;
    const int gr = (i >> 2) * 8 + (i & 3);
    const float g = y[(size_t)t * ldy + gr], u = y[(size_t)t * ldy + gr + 4];
    const float v = (g / (1.0f + expf(-g))) * u;
    if (q.ga == 64) {
        __shared__ float am[8];
        const float a = warp_amax(v);
        if ((threadIdx.x & 31) == 0) am[threadIdx.x >> 5] = a;
        __syncthreads();
        q8_64_write(v, fmaxf(am[threadIdx.x >> 5], am[(threadIdx.x >> 5) ^ 1]), q, t, i);
        return;
    }
    warp_q8(v, q, t, i);
}

// gated RMSNorm per head -> q8 (K = 3072). grid (T, 24) x 128
__global__ void __launch_bounds__(128) k_pf_gnorm_q8(const float* __restrict__ o, const float* __restrict__ y, int ldy,
                                                     const float* __restrict__ w, Q8Out q) {
    __shared__ float red[4];
    const int t = blockIdx.x, vl = blockIdx.y, i = threadIdx.x;
    const float x = o[(size_t)t * 3072 + vl * 128 + i];
    const float ss = block_sum(x * x, red);
    const float scale = rsqrtf(ss / 128.0f + 1e-6f);
    const float zz = y[(size_t)t * ldy + 5120 + vl * 128 + i];
    const float v = ((x * scale) * w[i]) * (zz / (1.0f + expf(-zz)));
    if (q.ga == 64) {
        __shared__ float am[4];
        const float a = warp_amax(v);
        if ((i & 31) == 0) am[i >> 5] = a;
        __syncthreads();
        q8_64_write(v, fmaxf(am[i >> 5], am[(i >> 5) ^ 1]), q, t, vl * 128 + i);
        return;
    }
    warp_q8(v, q, t, vl * 128 + i);
}

// yab[kz][t][i] = xn[t][kz-th K slice] . ab[i][same] (48 fp32 rows: 24 alpha then 24 beta); the consumers (k_pf_gdn,
// k_pf_gdnc) sum the AB_KS slices in a fixed order. grid (ceil(T/128), AB_KS), 128 threads: thread = 8 tokens
// (tg = tid & 15) x 6 rows (og = tid >> 4). K in tiles of 32 staged k-major in smem (x [k][128 tok], ab [k][48]) from
// registers loaded one tile ahead: two LDS.128 + three LDS.64 per 48 FMA. Per-thread summation order over k is the
// same as the round-3 kernel (sequential within the slice).
constexpr int AB_KS = 16;  // round 4: 4 -> 16 K slices (64 -> 256 blocks at T = 2048)
__global__ void __launch_bounds__(128) k_pf_ab(const float* __restrict__ xn, const float* __restrict__ ab, int T,
                                               float* yab, int sstride) {
    __shared__ __align__(16) float xs[32][132];
    __shared__ __align__(16) float as[32][52];
    const int tid = threadIdx.x, tg = tid & 15, og = tid >> 4, t0 = blockIdx.x * 128;
    float acc[8][6];
#pragma unroll
    for (int a = 0; a < 8; a++)
#pragma unroll
        for (int j = 0; j < 6; j++) acc[a][j] = 0.f;
    const int kbeg = blockIdx.y * (D / AB_KS), kend = kbeg + D / AB_KS;
    yab += (size_t)blockIdx.y * sstride;
    float4 rx[8], ra[3];
    const bool okx = t0 + tid < T;
    const float* xrow = xn + (size_t)(okx ? t0 + tid : 0) * D;
    auto load = [&](int k0) {
#pragma unroll
        for (int j = 0; j < 8; j++) rx[j] = okx ? *(const float4*)(xrow + k0 + 4 * j) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int j = 0; j < 3; j++) {
            const int i = tid + 128 * j, r = i >> 3, k = (i & 7) * 4;
            ra[j] = __ldg((const float4*)(ab + (size_t)r * D + k0 + k));
        }
    };
    load(kbeg);
    for (int k0 = kbeg; k0 < kend; k0 += 32) {
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 8; j++) {
            xs[4 * j][tid] = rx[j].x; xs[4 * j + 1][tid] = rx[j].y; xs[4 * j + 2][tid] = rx[j].z; xs[4 * j + 3][tid] = rx[j].w;
        }
#pragma unroll
        for (int j = 0; j < 3; j++) {
            const int i = tid + 128 * j, r = i >> 3, k = (i & 7) * 4;
            as[k][r] = ra[j].x; as[k + 1][r] = ra[j].y; as[k + 2][r] = ra[j].z; as[k + 3][r] = ra[j].w;
        }
        __syncthreads();
        if (k0 + 32 < kend) load(k0 + 32);
#pragma unroll 4
        for (int k = 0; k < 32; k++) {
            const float4 x0 = *(const float4*)&xs[k][tg * 8], x1 = *(const float4*)&xs[k][tg * 8 + 4];
            const float2 a01 = *(const float2*)&as[k][og * 6], a23 = *(const float2*)&as[k][og * 6 + 2],
                         a45 = *(const float2*)&as[k][og * 6 + 4];
            const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
            const float av[6] = {a01.x, a01.y, a23.x, a23.y, a45.x, a45.y};
#pragma unroll
            for (int t = 0; t < 8; t++)
#pragma unroll
                for (int j = 0; j < 6; j++) acc[t][j] += xv[t] * av[j];
        }
    }
#pragma unroll
    for (int a = 0; a < 8; a++) {
        const int t = t0 + tg * 8 + a;
        if (t < T) {
            float* yp = yab + (size_t)t * 48 + og * 6;
            *(float2*)yp = make_float2(acc[a][0], acc[a][1]);
            *(float2*)(yp + 2) = make_float2(acc[a][2], acc[a][3]);
            *(float2*)(yp + 4) = make_float2(acc[a][4], acc[a][5]);
        }
    }
}

// conv1d (4 taps over raw q|k|v inputs, history from the ring for positions < p0) + SiLU; q/k L2-normalized per head
// (decode's arithmetic). grid (T, 40 heads of 128 channels: 8 q, 8 k, 24 v), 128 threads
__global__ void __launch_bounds__(128) k_pf_conv(const float* __restrict__ y, int ldy, const float* __restrict__ ring,
                                                 const float* __restrict__ cw, int p0, float* out) {
    __shared__ float red[4];
    const int t = blockIdx.x, gi = blockIdx.y, tid = threadIdx.x;
    const int ch = gi < 8 ? gi * 128 + tid : gi < 16 ? 1024 + (gi - 8) * 128 + tid : 2048 + (gi - 16) * 128 + tid;
    float r[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int tt = t - 3 + j;
        r[j] = tt >= 0 ? __ldg(y + (size_t)tt * ldy + ch) : ring[((p0 + tt) & 3) * 5120 + ch];
    }
    const float4 wc = __ldg((const float4*)(cw + (size_t)ch * 4));
    float sum = 0.f;
    sum += r[0] * wc.x;
    sum += r[1] * wc.y;
    sum += r[2] * wc.z;
    sum += r[3] * wc.w;
    const float a = sum / (1.0f + expf(-sum));
    float val = a;
    if (gi < 16) {
        const float s = warp_sum(a * a);
        if ((tid & 31) == 0) red[tid >> 5] = s;
        __syncthreads();
        const float tot = (red[0] + red[1]) + (red[2] + red[3]);
        const float scale = rsqrtf(tot / 128.0f + 1e-6f / 128.0f);
        val = (a * scale) * (1.0f / sqrtf(128.0f));
    }
    out[(size_t)t * 5120 + ch] = val;
}

// ring slots of the last 3 positions of the ubatch get their raw inputs. grid 20 x 256
__global__ void k_pf_ring(const float* __restrict__ y, int ldy, int T, int p0, float* ring) {
    const int ch = blockIdx.x * 256 + threadIdx.x;
    for (int j = 1; j <= 3; j++) {
        const int tt = T - j;
        if (tt >= 0) ring[((p0 + tt) & 3) * 5120 + ch] = y[(size_t)tt * ldy + ch];
    }
}

// DeltaNet scan over T tokens with decode's per-token math; state in registers (decode layout S[vl][col][k]).
// grid 96 = (24 local v heads x 4 slices of 32 value columns), 256 threads: warp w owns columns sl*32 + 4w + j
// (j = lane >> 3); the 8 lanes of a column own k = 32m + 4*(lane & 7) + e (m, e < 4), so the two k-reductions per token
// are 3 shuffles each. Inputs are staged through shared memory in chunks of 8 tokens (double buffered); the decay and
// beta gates are computed once per token in the staging step.
constexpr int GCH = 8;  // 18.5 KB smem: 3 blocks per SM, all 96 resident
// V2 (pf_gdn2): one pass over the state per token computes the update, o_t = S_t q_t AND kv_{t+1} = S_t k_{t+1}
// (both reductions interleaved, 4 partial sums each), so a token costs one dependent pass + one shuffle tree instead
// of two; the first token of each staged chunk gets its own kv pass.
template <int V2>
__global__ void __launch_bounds__(256) k_pf_gdn(const float* __restrict__ qkv, const float* __restrict__ yab, int ab_ss, int T,
                                                const float* __restrict__ ssm_a, const float* __restrict__ ssm_dt,
                                                float* S, float* o) {
    __shared__ __align__(16) float sq[2][GCH][128], sk[2][GCH][128], svv[2][GCH][32], sab[2][GCH][2];
    const int bx = blockIdx.x, vl = bx >> 2, sl = bx & 3, kl = vl & 7, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5, j = lane >> 3, i8 = lane & 7;
    const int colw = warp * 4 + j;  // column within the slice
    const int col = sl * 32 + colw;
    float s[4][4];  // s[m][e] = S[col][32m + 4*i8 + e]
    {
        const float* Sp = S + ((size_t)vl * 128 + col) * 128 + 4 * i8;
#pragma unroll
        for (int m = 0; m < 4; m++) {
            const float4 v = *(const float4*)(Sp + 32 * m);
            s[m][0] = v.x; s[m][1] = v.y; s[m][2] = v.z; s[m][3] = v.w;
        }
    }
    const float dtv = ssm_dt[vl], av = ssm_a[vl];
    // staging (GCH = 8): q/k 8 tokens x 32 float4 (one each per thread), v 8 x 8 float4 (threads < 64), gates (8)
    float4 rq, rk, rv;
    float rbeta = 0.f, rg = 0.f;
    auto gload = [&](int c0) {
        {
            const int tt = tid >> 5, f = tid & 31, t = c0 + tt;
            if (t < T) {
                const float4* row = (const float4*)(qkv + (size_t)t * 5120);
                rq = __ldg(row + kl * 32 + f);
                rk = __ldg(row + 256 + kl * 32 + f);
            }
        }
        if (tid < 64) {
            const int tt = tid >> 3, f = tid & 7, t = c0 + tt;
            if (t < T) rv = __ldg((const float4*)(qkv + (size_t)t * 5120 + 2048 + vl * 128 + sl * 32) + f);
        }
        if (tid >= 64 && tid < 64 + GCH) {
            const int t = c0 + tid - 64;
            if (t < T) {
                float ya_ = 0.f, yb = 0.f;
#pragma unroll
                for (int kz = 0; kz < AB_KS; ++kz) {
                    ya_ += __ldg(yab + (size_t)kz * ab_ss + (size_t)t * 48 + vl);
                    yb += __ldg(yab + (size_t)kz * ab_ss + (size_t)t * 48 + 24 + vl);
                }
                rbeta = 1.0f / (1.0f + expf(-yb));
                const float xg = ya_ + dtv;
                const float sp = xg > 20.0f ? xg : logf(1.0f + expf(xg));
                rg = expf(sp * av);
            }
        }
    };
    auto sstore = [&](int b) {
        *(float4*)&sq[b][tid >> 5][(tid & 31) * 4] = rq;
        *(float4*)&sk[b][tid >> 5][(tid & 31) * 4] = rk;
        if (tid < 64) *(float4*)&svv[b][tid >> 3][(tid & 7) * 4] = rv;
        if (tid >= 64 && tid < 64 + GCH) { sab[b][tid - 64][0] = rg; sab[b][tid - 64][1] = rbeta; }
    };
    gload(0);
    sstore(0);
    __syncthreads();
    for (int c0 = 0, b = 0; c0 < T; c0 += GCH, b ^= 1) {
        if (c0 + GCH < T) gload(c0 + GCH);
        const int n = min(GCH, T - c0);
        if (V2) {
            float kr[4][4];
#pragma unroll
            for (int m = 0; m < 4; m++) {
                const float4 kv4 = *(const float4*)&sk[b][0][32 * m + 4 * i8];
                kr[m][0] = kv4.x; kr[m][1] = kv4.y; kr[m][2] = kv4.z; kr[m][3] = kv4.w;
            }
            float delta;
            {
                float p[4];
#pragma unroll
                for (int m = 0; m < 4; m++) p[m] = s[m][0] * kr[m][0] + s[m][1] * kr[m][1] + s[m][2] * kr[m][2] + s[m][3] * kr[m][3];
                float kv = (p[0] + p[1]) + (p[2] + p[3]);
                kv += __shfl_xor_sync(0xffffffffu, kv, 1);
                kv += __shfl_xor_sync(0xffffffffu, kv, 2);
                kv += __shfl_xor_sync(0xffffffffu, kv, 4);
                delta = (svv[b][0][colw] - sab[b][0][0] * kv) * sab[b][0][1];
            }
            for (int tt = 0; tt < n; tt++) {
                const float gv = sab[b][tt][0];
                const bool nx = tt + 1 < n;
                float qr[4][4], kn[4][4];
#pragma unroll
                for (int m = 0; m < 4; m++) {
                    const float4 qv4 = *(const float4*)&sq[b][tt][32 * m + 4 * i8];
                    qr[m][0] = qv4.x; qr[m][1] = qv4.y; qr[m][2] = qv4.z; qr[m][3] = qv4.w;
                    const float4 kn4 = nx ? *(const float4*)&sk[b][tt + 1][32 * m + 4 * i8] : make_float4(0.f, 0.f, 0.f, 0.f);
                    kn[m][0] = kn4.x; kn[m][1] = kn4.y; kn[m][2] = kn4.z; kn[m][3] = kn4.w;
                }
                float pa[4], pk[4];
#pragma unroll
                for (int m = 0; m < 4; m++) {
                    pa[m] = 0.f; pk[m] = 0.f;
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const float sn = gv * s[m][e] + kr[m][e] * delta;
                        pa[m] += sn * qr[m][e];
                        pk[m] += sn * kn[m][e];
                        s[m][e] = sn;
                    }
                }
                float a = (pa[0] + pa[1]) + (pa[2] + pa[3]), kv = (pk[0] + pk[1]) + (pk[2] + pk[3]);
                a += __shfl_xor_sync(0xffffffffu, a, 1);
                kv += __shfl_xor_sync(0xffffffffu, kv, 1);
                a += __shfl_xor_sync(0xffffffffu, a, 2);
                kv += __shfl_xor_sync(0xffffffffu, kv, 2);
                a += __shfl_xor_sync(0xffffffffu, a, 4);
                kv += __shfl_xor_sync(0xffffffffu, kv, 4);
                if (i8 == 0) o[(size_t)(c0 + tt) * 3072 + vl * 128 + col] = a * (1.0f / sqrtf(128.0f));
                if (nx) {
                    delta = (svv[b][tt + 1][colw] - sab[b][tt + 1][0] * kv) * sab[b][tt + 1][1];
#pragma unroll
                    for (int m = 0; m < 4; m++)
#pragma unroll
                        for (int e = 0; e < 4; e++) kr[m][e] = kn[m][e];
                }
            }
            if (c0 + GCH < T) sstore(b ^ 1);
            __syncthreads();
            continue;
        }
        for (int tt = 0; tt < n; tt++) {
            float kr[4][4], qr[4][4];
#pragma unroll
            for (int m = 0; m < 4; m++) {
                const float4 kv4 = *(const float4*)&sk[b][tt][32 * m + 4 * i8];
                const float4 qv4 = *(const float4*)&sq[b][tt][32 * m + 4 * i8];
                kr[m][0] = kv4.x; kr[m][1] = kv4.y; kr[m][2] = kv4.z; kr[m][3] = kv4.w;
                qr[m][0] = qv4.x; qr[m][1] = qv4.y; qr[m][2] = qv4.z; qr[m][3] = qv4.w;
            }
            const float gv = sab[b][tt][0], beta = sab[b][tt][1], sv = svv[b][tt][colw];
            float kv = 0.f;
#pragma unroll
            for (int m = 0; m < 4; m++)
#pragma unroll
                for (int e = 0; e < 4; e++) kv += s[m][e] * kr[m][e];
            kv += __shfl_xor_sync(0xffffffffu, kv, 1);
            kv += __shfl_xor_sync(0xffffffffu, kv, 2);
            kv += __shfl_xor_sync(0xffffffffu, kv, 4);
            const float delta = (sv - gv * kv) * beta;
            float a = 0.f;
#pragma unroll
            for (int m = 0; m < 4; m++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const float sn = gv * s[m][e] + kr[m][e] * delta;
                    a += sn * qr[m][e];
                    s[m][e] = sn;
                }
            a += __shfl_xor_sync(0xffffffffu, a, 1);
            a += __shfl_xor_sync(0xffffffffu, a, 2);
            a += __shfl_xor_sync(0xffffffffu, a, 4);
            if (i8 == 0) o[(size_t)(c0 + tt) * 3072 + vl * 128 + col] = a * (1.0f / sqrtf(128.0f));
        }
        if (c0 + GCH < T) sstore(b ^ 1);
        __syncthreads();
    }
    {
        float* Sp = S + ((size_t)vl * 128 + col) * 128 + 4 * i8;
#pragma unroll
        for (int m = 0; m < 4; m++) *(float4*)(Sp + 32 * m) = make_float4(s[m][0], s[m][1], s[m][2], s[m][3]);
    }
}

// ------------------------------------------------------------------------------------------------ chunked DeltaNet
// Chunked gated delta rule on fp16 tensor cores (mma.m16n8k8, fp32 accumulation), C = 64 tokens per chunk, one block
// per local value head (24 blocks x 256 threads). Same recurrence as k_pf_gdn (decode math):
//   S_t = d_t S_{t-1} + u_t k_t^T,  u_t = b_t (v_t - d_t S_{t-1} k_t),  o_t = S_t q_t / sqrt(128)
// With G = cumsum(log d) inside the chunk and S0 the state entering it:
//   (I - A) U = b * (V - e^G * (K S0^T)),   A[t][s] = -b_t e^{G_t - G_s} (k_t . k_s)  (s < t)
//   O = e^G * (Q S0^T) + (Q K^T * D) U,      D[t][s] = e^{G_t - G_s}  (s <= t)
//   S_C = e^{G_C} S0 + (e^{G_C - G} * U)^T K
// U comes from T = (I - A)^{-1} (forward substitution, fp32, one column per thread for 64 threads).
// Warp w owns state rows v in [16w, 16w + 16) for all 128 k, in m16n8 accumulator layout (64 fp32 registers), which is
// also the B-fragment layout that Q S0^T and K S0^T need, so S0 never goes through shared memory.
namespace gdnc {
constexpr int C = 64;
// smem layout (bytes): Q/A/T region, K, P, R/U, gates
constexpr int O_Q = 0;                 // Qh [64][128] fp16 (32 KB? no: 64*128*2 = 16 KB); later Am fp32 [64][64] (16 KB), then Th fp16
constexpr int O_K = 16384;             // Kh [64][128] fp16
constexpr int O_P = 32768;             // Ph [64][64] fp16 (8 KB)
constexpr int O_R = 40960;             // Rh / Uh [64][128] fp16 (16 KB)
constexpr int O_G = 57344;             // G[64], beta[64] fp32
constexpr int BYTES = O_G + 2 * 64 * 4;
constexpr int SCR = 8192 + 8192 + 512;  // k_pf_gdnp scratch bytes per (head, chunk)
// 16-B unit swizzles: 256-B rows (16 units) and 128-B rows (8 units)
__device__ __forceinline__ int off256(int row, int col) {  // fp16 element (row, col) in a [*][128] matrix
    return row * 256 + ((((col >> 3) ^ (row & 7))) << 4) + (col & 7) * 2;
}
__device__ __forceinline__ int off128(int row, int col) {  // fp16 element in a [*][64] matrix
    return row * 128 + ((((col >> 3) ^ (row & 7))) << 4) + (col & 7) * 2;
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t b) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b));
}
__device__ __forceinline__ void ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, unsigned sp) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(sp));
}
__device__ __forceinline__ void ldsm4t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, unsigned sp) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(sp));
}
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *(uint32_t*)&h;
}
}  // namespace gdnc

template <int PRE>
__global__ void __launch_bounds__(256, 1) k_pf_gdnc(const float* __restrict__ qkv, const float* __restrict__ yab, int ab_ss,
                                                    int T, const float* __restrict__ ssm_a,
                                                    const float* __restrict__ ssm_dt, float* S, float* o,
                                                    const unsigned char* __restrict__ scr) {
    using namespace gdnc;
    extern __shared__ __align__(16) unsigned char sm[];
    const int vl = blockIdx.x, kl = vl & 7, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t4 = lane & 3;
    const unsigned sb = (unsigned)__cvta_generic_to_shared(sm);
    float* Gs = (float*)(sm + O_G);
    float* Bs = Gs + 64;
    // state rows v = 16*warp + g (+8), k = 8j + 2t4 (+1): st[j][0..3] in accumulator order
    float st[16][4];
    {
        const float* Sp = S + ((size_t)vl * 128 + 16 * warp + g) * 128 + 2 * t4;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float2 a = *(const float2*)(Sp + 8 * j), b = *(const float2*)(Sp + 8 * 128 + 8 * j);
            st[j][0] = a.x; st[j][1] = a.y; st[j][2] = b.x; st[j][3] = b.y;
        }
    }
    const float dtv = ssm_dt[vl], av = ssm_a[vl];
    for (int c0 = 0; c0 < T; c0 += C) {
        const int n = min(C, T - c0);
        // ---- stage Q, K (fp16), gates
        for (int i = tid; i < 64 * 32; i += 256) {
            const int t = i >> 5, f = (i & 31) * 4;
            float4 q = make_float4(0.f, 0.f, 0.f, 0.f), kk = q;
            if (t < n) {
                const float* row = qkv + (size_t)(c0 + t) * 5120;
                q = *(const float4*)(row + kl * 128 + f);
                kk = *(const float4*)(row + 1024 + kl * 128 + f);
            }
            *(uint2*)(sm + O_Q + off256(t, f)) = make_uint2(pack_h2(q.x, q.y), pack_h2(q.z, q.w));
            *(uint2*)(sm + O_K + off256(t, f)) = make_uint2(pack_h2(kk.x, kk.y), pack_h2(kk.z, kk.w));
        }
        if (PRE) {  // G, beta, P from the k_pf_gdnp pass (scratch row of this (head, chunk))
            const unsigned char* cs = scr + ((size_t)vl * ((T + C - 1) / C) + c0 / C) * SCR;
            if (tid < 128) Gs[tid] = ((const float*)(cs + 16384))[tid];  // Gs[64] then Bs[64] (contiguous)
#pragma unroll
            for (int u = tid; u < 512; u += 256)
                *(int4*)(sm + O_P + off128(u >> 3, (u & 7) * 8)) = ((const int4*)(cs + 8192))[u];
        } else {
        if (warp == 0) {
            float lg[2], bt[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int t = lane * 2 + h;
                lg[h] = 0.f; bt[h] = 0.f;
                if (t < n) {
                    float ya_ = 0.f, yb = 0.f;
                    if (ab_ss < 0) {  // one slice, row stride -ab_ss (alpha/beta columns of the R512 qkvz GEMM)
                        ya_ = __ldg(yab + (size_t)(c0 + t) * (-ab_ss) + vl);
                        yb = __ldg(yab + (size_t)(c0 + t) * (-ab_ss) + 24 + vl);
                    } else {
#pragma unroll
                        for (int kz = 0; kz < AB_KS; ++kz) {
                            ya_ += __ldg(yab + (size_t)kz * ab_ss + (size_t)(c0 + t) * 48 + vl);
                            yb += __ldg(yab + (size_t)kz * ab_ss + (size_t)(c0 + t) * 48 + 24 + vl);
                        }
                    }
                    bt[h] = 1.0f / (1.0f + expf(-yb));
                    const float xg = ya_ + dtv;
                    const float sp = xg > 20.0f ? xg : logf(1.0f + expf(xg));
                    lg[h] = sp * av;
                }
            }
            // inclusive scan of lg over 64 tokens (2 per lane)
            float v = lg[0] + lg[1];
#pragma unroll
            for (int o2 = 1; o2 < 32; o2 <<= 1) {
                const float u = __shfl_up_sync(0xffffffffu, v, o2);
                if (lane >= o2) v += u;
            }
            const float ex = v - (lg[0] + lg[1]);  // exclusive prefix
            Gs[2 * lane] = ex + lg[0];
            Gs[2 * lane + 1] = ex + lg[0] + lg[1];
            Bs[2 * lane] = bt[0];
            Bs[2 * lane + 1] = bt[1];
        }
        }
        __syncthreads();
        const float GC = Gs[63];
        // ---- per warp: QS0 = Q S0^T and KS0 = K S0^T for v cols [16w, 16w+16): [64 t][16 v] = 4 mt x 2 nt tiles
        float qs[4][2][4], ks[4][2][4];
#pragma unroll
        for (int mt = 0; mt < 4; ++mt)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) { qs[mt][nt][e] = 0.f; ks[mt][nt][e] = 0.f; }
#pragma unroll
        for (int jj = 0; jj < 16; jj += 2) {  // two k8 steps per ldmatrix.x4
            uint32_t bq[2][2];  // [k step][nt]: B = S0^T (k x v): from st (rows v = g / g+8)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                bq[h][0] = pack_h2(st[jj + h][0], st[jj + h][1]);
                bq[h][1] = pack_h2(st[jj + h][2], st[jj + h][3]);
            }
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                const int row = mt * 16 + (lane & 15), col = jj * 8 + (lane >> 4) * 8;
                uint32_t a0, a1, a2, a3;  // matrices: (rows 0-7, k jj), (rows 8-15, k jj), (rows 0-7, k jj+1), (8-15, jj+1)
                // lanes 0-7 rows 0-7 / 8-15 rows 8-15 at col block jj; lanes 16-31 at col block jj+1
                ldsm4(a0, a1, a2, a3, sb + O_Q + off256(row, col));
                uint32_t k0, k1, k2, k3;
                ldsm4(k0, k1, k2, k3, sb + O_K + off256(row, col));
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    mma16816(qs[mt][nt], a0, a1, bq[0][nt]);
                    mma16816(qs[mt][nt], a2, a3, bq[1][nt]);
                    mma16816(ks[mt][nt], k0, k1, bq[0][nt]);
                    mma16816(ks[mt][nt], k2, k3, bq[1][nt]);
                }
            }
        }
        if (!PRE)
        // ---- KK and QK [64 t][64 s]: warp w computes rows mt = w & 3, cols s in [32 * (w >> 2), +32) (4 n8 tiles)
        {
            const int mt = warp & 3, s0 = 32 * (warp >> 2);
            float kk[4][4], qk[4][4];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) { kk[nt][e] = 0.f; qk[nt][e] = 0.f; }
#pragma unroll
            for (int jj = 0; jj < 16; jj += 2) {
                const int row = mt * 16 + (lane & 15), col = jj * 8 + (lane >> 4) * 8;
                uint32_t a0, a1, a2, a3, q0, q1, q2, q3;
                ldsm4(a0, a1, a2, a3, sb + O_K + off256(row, col));
                ldsm4(q0, q1, q2, q3, sb + O_Q + off256(row, col));
#pragma unroll
                for (int np = 0; np < 2; ++np) {  // pairs of n8 tiles: B = K rows s (non-trans), 2 k steps
                    // matrices: (s rows np*16 + 0-7, k jj), (s 8-15, k jj), (s 0-7, k jj+1), (s 8-15, k jj+1)
                    const int srow = s0 + np * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, scol = jj * 8 + (lane >> 4) * 8;
                    uint32_t b0, b1, b2, b3;
                    ldsm4(b0, b1, b2, b3, sb + O_K + off256(srow, scol));
                    mma16816(kk[np * 2], a0, a1, b0);
                    mma16816(kk[np * 2], a2, a3, b2);
                    mma16816(kk[np * 2 + 1], a0, a1, b1);
                    mma16816(kk[np * 2 + 1], a2, a3, b3);
                    mma16816(qk[np * 2], q0, q1, b0);
                    mma16816(qk[np * 2], q2, q3, b2);
                    mma16816(qk[np * 2 + 1], q0, q1, b1);
                    mma16816(qk[np * 2 + 1], q2, q3, b3);
                }
            }
            __syncthreads();  // all warps done reading Qh (QS0 above, QK here) before Am overwrites it
            float* Am = (float*)(sm + O_Q);
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int t = mt * 16 + g + (e >> 1) * 8, sI = s0 + nt * 8 + 2 * t4 + (e & 1);
                    const float dg = sI <= t ? __expf(Gs[t] - Gs[sI]) : 0.f;
                    Am[t * 64 + sI] = sI < t ? -Bs[t] * dg * kk[nt][e] : 0.f;
                    *(__half*)(sm + O_P + off128(t, sI)) = __float2half_rn(dg * qk[nt][e]);
                }
        }
        // ---- R = b * (V - e^G * KS0) -> Rh (fp16) [64 t][128 v] (warp: its 16 v cols)
#pragma unroll
        for (int mt = 0; mt < 4; ++mt)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int t = mt * 16 + g + hh * 8, v = 16 * warp + nt * 8 + 2 * t4;
                    float2 vv = make_float2(0.f, 0.f);
                    if (t < n) vv = *(const float2*)(qkv + (size_t)(c0 + t) * 5120 + 2048 + vl * 128 + v);
                    const float eg = __expf(Gs[t]), b = Bs[t];
                    const float r0 = b * (vv.x - eg * ks[mt][nt][hh * 2]), r1 = b * (vv.y - eg * ks[mt][nt][hh * 2 + 1]);
                    *(uint32_t*)(sm + O_R + off256(t, v)) = pack_h2(r0, r1);
                    ks[mt][nt][hh * 2] = 0.f; ks[mt][nt][hh * 2 + 1] = 0.f;  // reused as the P U accumulator
                }
        __syncthreads();
        if (PRE) {  // T = (I - A)^{-1} from scratch over Qh's region (QS0 is done: the barrier above)
            const unsigned char* cs = scr + ((size_t)vl * ((T + C - 1) / C) + c0 / C) * SCR;
#pragma unroll
            for (int u = tid; u < 512; u += 256)
                *(int4*)(sm + O_Q + off128(u >> 3, (u & 7) * 8)) = ((const int4*)cs)[u];
            __syncthreads();
        } else
        // ---- T = (I - A)^{-1}: 4 threads per column j (tid = 4j + r, r owns rows t = r mod 4), forward substitution
        // with the 4 partial sums combined by 2 shuffles per row; then Th (fp16, [64][64]) over Am's region
        {
            float tc[16];
            const int j = tid >> 2, r = tid & 3;
            const float* Am = (const float*)(sm + O_Q);
#pragma unroll
            for (int t = 0; t < 64; ++t) {
                float a0 = 0.f, a1 = 0.f;
#pragma unroll
                for (int i = 0; i < 16; i += 2) {
                    if (4 * i + r < t) a0 += Am[t * 64 + 4 * i + r] * tc[i];
                    if (4 * (i + 1) + r < t) a1 += Am[t * 64 + 4 * (i + 1) + r] * tc[i + 1];
                }
                float v = a0 + a1;
                v += __shfl_xor_sync(0xffffffffu, v, 1);
                v += __shfl_xor_sync(0xffffffffu, v, 2);
                if (r == (t & 3)) tc[t >> 2] = v + ((t == j) ? 1.f : 0.f);
            }
            __syncthreads();  // Am fully read
#pragma unroll
            for (int i = 0; i < 16; ++i) *(__half*)(sm + O_Q + off128(4 * i + r, j)) = __float2half_rn(tc[i]);
            __syncthreads();
        }
        // ---- U = T R: [64 t][16 v] per warp; A = T (rows t, k = s), B = R (k = s, n = v) via ldmatrix.trans
        float uu[4][2][4];
#pragma unroll
        for (int mt = 0; mt < 4; ++mt)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) uu[mt][nt][e] = 0.f;
#pragma unroll
        for (int jj = 0; jj < 8; jj += 2) {
            // B: matrices (s jj rows, v nt0), (s jj, v nt1), (s jj+1, v nt0), (s jj+1, v nt1)
            uint32_t b0, b1, b2, b3;
            {
                const int srow = jj * 8 + (lane & 7) + (lane >> 4) * 8, vcol = 16 * warp + ((lane >> 3) & 1) * 8;
                ldsm4t(b0, b1, b2, b3, sb + O_R + off256(srow, vcol));
            }
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                if (mt * 2 + 1 < jj) continue;  // T is lower triangular: rows < 16mt+16 need s < 16mt+16
                const int row = mt * 16 + (lane & 15), col = jj * 8 + (lane >> 4) * 8;
                uint32_t a0, a1, a2, a3;
                ldsm4(a0, a1, a2, a3, sb + O_Q + off128(row, col));
                mma16816(uu[mt][0], a0, a1, b0);
                mma16816(uu[mt][1], a0, a1, b1);
                mma16816(uu[mt][0], a2, a3, b2);
                mma16816(uu[mt][1], a2, a3, b3);
            }
        }
        __syncwarp();
        // U (fp16) over this warp's Rh columns (only this warp reads/writes them)
#pragma unroll
        for (int mt = 0; mt < 4; ++mt)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int t = mt * 16 + g + hh * 8, v = 16 * warp + nt * 8 + 2 * t4;
                    *(uint32_t*)(sm + O_R + off256(t, v)) = pack_h2(uu[mt][nt][hh * 2], uu[mt][nt][hh * 2 + 1]);
                }
        __syncwarp();
        // ---- O = e^G * QS0 + P U ; A = P (rows t, k = s), B = U (k = s, n = v) via .trans
#pragma unroll
        for (int jj = 0; jj < 8; jj += 2) {
            uint32_t b0, b1, b2, b3;
            {
                const int srow = jj * 8 + (lane & 7) + (lane >> 4) * 8, vcol = 16 * warp + ((lane >> 3) & 1) * 8;
                ldsm4t(b0, b1, b2, b3, sb + O_R + off256(srow, vcol));
            }
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                if (mt * 2 + 1 < jj) continue;  // P is lower triangular (incl. diagonal)
                const int row = mt * 16 + (lane & 15), col = jj * 8 + (lane >> 4) * 8;
                uint32_t a0, a1, a2, a3;
                ldsm4(a0, a1, a2, a3, sb + O_P + off128(row, col));
                // P U accumulates into qs after the e^G scaling below: use a separate accumulator (ks is free)
                mma16816(ks[mt][0], a0, a1, b0);
                mma16816(ks[mt][1], a0, a1, b1);
                mma16816(ks[mt][0], a2, a3, b2);
                mma16816(ks[mt][1], a2, a3, b3);
            }
        }
        // ---- outputs
#pragma unroll
        for (int mt = 0; mt < 4; ++mt)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int t = mt * 16 + g + hh * 8, v = 16 * warp + nt * 8 + 2 * t4;
                    if (t < n) {
                        const float eg = __expf(Gs[t]);
                        const float o0 = eg * qs[mt][nt][hh * 2] + ks[mt][nt][hh * 2];
                        const float o1 = eg * qs[mt][nt][hh * 2 + 1] + ks[mt][nt][hh * 2 + 1];
                        *(float2*)(o + (size_t)(c0 + t) * 3072 + vl * 128 + v) =
                            make_float2(o0 * (1.0f / sqrtf(128.0f)), o1 * (1.0f / sqrtf(128.0f)));
                    }
                }
        __syncwarp();
        // ---- state: S = e^{G_C} S0 + (e^{G_C - G} * U)^T K ; scale U rows in place (this warp's columns)
#pragma unroll
        for (int mt = 0; mt < 4; ++mt)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int t = mt * 16 + g + hh * 8, v = 16 * warp + nt * 8 + 2 * t4;
                    const float w = __expf(GC - Gs[t]);
                    *(uint32_t*)(sm + O_R + off256(t, v)) = pack_h2(w * uu[mt][nt][hh * 2], w * uu[mt][nt][hh * 2 + 1]);
                }
        __syncwarp();
        {
            const float eC = __expf(GC);
#pragma unroll
            for (int j = 0; j < 16; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) st[j][e] *= eC;
        }
#pragma unroll
        for (int jj = 0; jj < 8; jj += 2) {  // k-dim = t (64)
            // A = U'^T (rows v 16w.., cols t): from Rh[t][v] via .trans: matrices (v 0-7, t jj), (v 8-15, t jj),
            // (v 0-7, t jj+1), (v 8-15, t jj+1)
            uint32_t a0, a1, a2, a3;
            {
                const int trow = jj * 8 + (lane & 7) + (lane >> 4) * 8, vcol = 16 * warp + ((lane >> 3) & 1) * 8;
                ldsm4t(a0, a1, a2, a3, sb + O_R + off256(trow, vcol));
            }
#pragma unroll
            for (int kp = 0; kp < 16; kp += 2) {
                // B = K (k-dim t, n = key): from Kh[t][key] via .trans: matrices (t jj, key kp), (t jj, key kp+1),
                // (t jj+1, key kp), (t jj+1, key kp+1)
                uint32_t b0, b1, b2, b3;
                const int trow = jj * 8 + (lane & 7) + (lane >> 4) * 8, kcol = kp * 8 + ((lane >> 3) & 1) * 8;
                ldsm4t(b0, b1, b2, b3, sb + O_K + off256(trow, kcol));
                mma16816(st[kp], a0, a1, b0);
                mma16816(st[kp + 1], a0, a1, b1);
                mma16816(st[kp], a2, a3, b2);
                mma16816(st[kp + 1], a2, a3, b3);
            }
        }
        __syncthreads();  // smem reused by the next chunk
    }
    {
        float* Sp = S + ((size_t)vl * 128 + 16 * warp + g) * 128 + 2 * t4;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            *(float2*)(Sp + 8 * j) = make_float2(st[j][0], st[j][1]);
            *(float2*)(Sp + 8 * 128 + 8 * j) = make_float2(st[j][2], st[j][3]);
        }
    }
}

// k_pf_gdnc's state-independent part, for all chunks in parallel (pf_gdnc 2): per (head, chunk) the gates G / beta,
// P = (Q K^T * D) and T = (I - A)^{-1} (same arithmetic as k_pf_gdnc<0>) -> scr [head][chunk][T fp16 64x64 | P fp16
// 64x64 | G[64] B[64] fp32]. grid (24, nchunks), 256 threads.
__global__ void __launch_bounds__(256, 1) k_pf_gdnp(const float* __restrict__ qkv, const float* __restrict__ yab, int ab_ss,
                                                    int T, const float* __restrict__ ssm_a,
                                                    const float* __restrict__ ssm_dt, unsigned char* __restrict__ scr) {
    using namespace gdnc;
    extern __shared__ __align__(16) unsigned char sm[];
    const int vl = blockIdx.x, kl = vl & 7, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t4 = lane & 3;
    float* Gs = (float*)(sm + O_G);
    float* Bs = Gs + 64;
    const float dtv = ssm_dt[vl], av = ssm_a[vl];
    const int c0 = blockIdx.y * C;
    const int n = min(C, T - c0);
    const unsigned sb = (unsigned)__cvta_generic_to_shared(sm);
    unsigned char* cs = scr + ((size_t)vl * gridDim.y + blockIdx.y) * SCR;
    __half* Tg = (__half*)cs;
    __half* Pg = (__half*)(cs + 8192);
    {
        // ---- stage Q, K (fp16), gates
        for (int i = tid; i < 64 * 32; i += 256) {
            const int t = i >> 5, f = (i & 31) * 4;
            float4 q = make_float4(0.f, 0.f, 0.f, 0.f), kk = q;
            if (t < n) {
                const float* row = qkv + (size_t)(c0 + t) * 5120;
                q = *(const float4*)(row + kl * 128 + f);
                kk = *(const float4*)(row + 1024 + kl * 128 + f);
            }
            *(uint2*)(sm + O_Q + off256(t, f)) = make_uint2(pack_h2(q.x, q.y), pack_h2(q.z, q.w));
            *(uint2*)(sm + O_K + off256(t, f)) = make_uint2(pack_h2(kk.x, kk.y), pack_h2(kk.z, kk.w));
        }
        if (warp == 0) {
            float lg[2], bt[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int t = lane * 2 + h;
                lg[h] = 0.f; bt[h] = 0.f;
                if (t < n) {
                    float ya_ = 0.f, yb = 0.f;
                    if (ab_ss < 0) {  // one slice, row stride -ab_ss (alpha/beta columns of the R512 qkvz GEMM)
                        ya_ = __ldg(yab + (size_t)(c0 + t) * (-ab_ss) + vl);
                        yb = __ldg(yab + (size_t)(c0 + t) * (-ab_ss) + 24 + vl);
                    } else {
#pragma unroll
                        for (int kz = 0; kz < AB_KS; ++kz) {
                            ya_ += __ldg(yab + (size_t)kz * ab_ss + (size_t)(c0 + t) * 48 + vl);
                            yb += __ldg(yab + (size_t)kz * ab_ss + (size_t)(c0 + t) * 48 + 24 + vl);
                        }
                    }
                    bt[h] = 1.0f / (1.0f + expf(-yb));
                    const float xg = ya_ + dtv;
                    const float sp = xg > 20.0f ? xg : logf(1.0f + expf(xg));
                    lg[h] = sp * av;
                }
            }
            // inclusive scan of lg over 64 tokens (2 per lane)
            float v = lg[0] + lg[1];
#pragma unroll
            for (int o2 = 1; o2 < 32; o2 <<= 1) {
                const float u = __shfl_up_sync(0xffffffffu, v, o2);
                if (lane >= o2) v += u;
            }
            const float ex = v - (lg[0] + lg[1]);  // exclusive prefix
            Gs[2 * lane] = ex + lg[0];
            Gs[2 * lane + 1] = ex + lg[0] + lg[1];
            Bs[2 * lane] = bt[0];
            Bs[2 * lane + 1] = bt[1];
        }
        __syncthreads();
        if (tid < 128) ((float*)(cs + 16384))[tid] = Gs[tid];
        // ---- KK and QK [64 t][64 s]: warp w computes rows mt = w & 3, cols s in [32 * (w >> 2), +32) (4 n8 tiles)
        {
            const int mt = warp & 3, s0 = 32 * (warp >> 2);
            float kk[4][4], qk[4][4];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) { kk[nt][e] = 0.f; qk[nt][e] = 0.f; }
#pragma unroll
            for (int jj = 0; jj < 16; jj += 2) {
                const int row = mt * 16 + (lane & 15), col = jj * 8 + (lane >> 4) * 8;
                uint32_t a0, a1, a2, a3, q0, q1, q2, q3;
                ldsm4(a0, a1, a2, a3, sb + O_K + off256(row, col));
                ldsm4(q0, q1, q2, q3, sb + O_Q + off256(row, col));
#pragma unroll
                for (int np = 0; np < 2; ++np) {  // pairs of n8 tiles: B = K rows s (non-trans), 2 k steps
                    // matrices: (s rows np*16 + 0-7, k jj), (s 8-15, k jj), (s 0-7, k jj+1), (s 8-15, k jj+1)
                    const int srow = s0 + np * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, scol = jj * 8 + (lane >> 4) * 8;
                    uint32_t b0, b1, b2, b3;
                    ldsm4(b0, b1, b2, b3, sb + O_K + off256(srow, scol));
                    mma16816(kk[np * 2], a0, a1, b0);
                    mma16816(kk[np * 2], a2, a3, b2);
                    mma16816(kk[np * 2 + 1], a0, a1, b1);
                    mma16816(kk[np * 2 + 1], a2, a3, b3);
                    mma16816(qk[np * 2], q0, q1, b0);
                    mma16816(qk[np * 2], q2, q3, b2);
                    mma16816(qk[np * 2 + 1], q0, q1, b1);
                    mma16816(qk[np * 2 + 1], q2, q3, b3);
                }
            }
            __syncthreads();  // all warps done reading Qh (QS0 above, QK here) before Am overwrites it
            float* Am = (float*)(sm + O_Q);
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int t = mt * 16 + g + (e >> 1) * 8, sI = s0 + nt * 8 + 2 * t4 + (e & 1);
                    const float dg = sI <= t ? __expf(Gs[t] - Gs[sI]) : 0.f;
                    Am[t * 64 + sI] = sI < t ? -Bs[t] * dg * kk[nt][e] : 0.f;
                    Pg[t * 64 + sI] = __float2half_rn(dg * qk[nt][e]);
                }
        }
        __syncthreads();  // Am complete
        // ---- T = (I - A)^{-1}: 4 threads per column j (tid = 4j + r, r owns rows t = r mod 4), forward substitution
        // with the 4 partial sums combined by 2 shuffles per row; then Th (fp16, [64][64]) over Am's region
        {
            float tc[16];
            const int j = tid >> 2, r = tid & 3;
            const float* Am = (const float*)(sm + O_Q);
#pragma unroll
            for (int t = 0; t < 64; ++t) {
                float a0 = 0.f, a1 = 0.f;
#pragma unroll
                for (int i = 0; i < 16; i += 2) {
                    if (4 * i + r < t) a0 += Am[t * 64 + 4 * i + r] * tc[i];
                    if (4 * (i + 1) + r < t) a1 += Am[t * 64 + 4 * (i + 1) + r] * tc[i + 1];
                }
                float v = a0 + a1;
                v += __shfl_xor_sync(0xffffffffu, v, 1);
                v += __shfl_xor_sync(0xffffffffu, v, 2);
                if (r == (t & 3)) tc[t >> 2] = v + ((t == j) ? 1.f : 0.f);
            }
#pragma unroll
            for (int i = 0; i < 16; ++i) Tg[(4 * i + r) * 64 + j] = __float2half_rn(tc[i]);
        }
    }
}

// max |a - b| and max |b| over n floats (as positive float bits in two unsigned ints)
__global__ void k_maxdiff(const float* a, const float* b, size_t n, unsigned* out) {
    float e = 0.f, m = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        e = fmaxf(e, fabsf(a[i] - b[i]));
        m = fmaxf(m, fabsf(b[i]));
    }
    atomicMax(out, __float_as_uint(e));
    atomicMax(out + 1, __float_as_uint(m));
}

// gated RMSNorm per head: g = rmsnorm(o) * w * silu(z). grid (T, 24) x 128
__global__ void __launch_bounds__(128) k_pf_gnorm(const float* __restrict__ o, const float* __restrict__ y, int ldy,
                                                  const float* __restrict__ w, float* g) {
    __shared__ float red[4];
    const int t = blockIdx.x, vl = blockIdx.y, i = threadIdx.x;
    const float x = o[(size_t)t * 3072 + vl * 128 + i];
    const float ss = block_sum(x * x, red);
    const float scale = rsqrtf(ss / 128.0f + 1e-6f);
    const float zz = y[(size_t)t * ldy + 5120 + vl * 128 + i];
    g[(size_t)t * 3072 + vl * 128 + i] = ((x * scale) * w[i]) * (zz / (1.0f + expf(-zz)));
}

// q/k RMSNorm + partial NeoX RoPE, KV append at p0 + t (decode's arithmetic). grid (T, 14) x 256
__global__ void __launch_bounds__(256) k_pf_attn_prep(const float* __restrict__ ya, int ldy, const float* qw,
                                                      const float* kw, float* qa, __half* kc, __half* vc, int max_ctx,
                                                      int p0, float theta_scale) {
    __shared__ float red[8];
    __shared__ float yv[256];
    const int t = blockIdx.x, b = blockIdx.y, d = threadIdx.x;
    const int pos = p0 + t;
    const float* row = ya + (size_t)t * ldy;
    const bool isq = b < 12;
    const float* src = isq ? row + b * 512 : row + 6144 + (b - 12) * 256;
    const float* w = isq ? qw : kw;
    const float x = src[d];
    const float ss = block_sum(x * x, red);
    const float scale = rsqrtf(ss / 256.0f + 1e-6f);
    yv[d] = (x * scale) * w[d];
    __syncthreads();
    float out0 = 0.f, out1 = 0.f;
    if (d < 32) {
        const float theta = (float)pos * powf(theta_scale, (float)d);
        const float c = cosf(theta), s = sinf(theta);
        const float x0 = yv[d], x1 = yv[d + 32];
        out0 = x0 * c - x1 * s;
        out1 = x0 * s + x1 * c;
    }
    if (isq) {
        float* dst = qa + (size_t)t * 3072 + b * 256;
        if (d < 32) { dst[d] = out0; dst[d + 32] = out1; }
        else if (d >= 64) dst[d] = yv[d];
    } else {
        const int kv = b - 12;
        __half* kd = kc + ((size_t)kv * max_ctx + pos) * 256;
        if (d < 32) { kd[d] = __float2half_rn(out0); kd[d + 32] = __float2half_rn(out1); }
        else if (d >= 64) kd[d] = __float2half_rn(yv[d]);
        vc[((size_t)kv * max_ctx + pos) * 256 + d] = __float2half_rn(row[6656 + kv * 256 + d]);
    }
}

// causal attention for the ubatch queries over positions [0, p0 + t], online softmax (decode's per-key math), output
// gated by sigmoid(gate). grid (T, 2 kv heads) x 192 (warp = one of the 6 q heads of the kv head, lane = 8 dims)
__global__ void __launch_bounds__(192) k_pf_attn(const float* __restrict__ qa, const __half* __restrict__ kc,
                                                 const __half* __restrict__ vc, const float* __restrict__ ya, int ldy,
                                                 int max_ctx, int p0, float* out) {
    const int t = blockIdx.x, j = blockIdx.y, lane = threadIdx.x & 31, h6 = threadIdx.x >> 5;
    const int hl = j * 6 + h6;
    const int n_kv = p0 + t + 1;
    float q[8];
    {
        const float4* qp = (const float4*)(qa + (size_t)t * 3072 + hl * 256 + lane * 8);
        const float4 a = qp[0], b = qp[1];
        q[0] = a.x; q[1] = a.y; q[2] = a.z; q[3] = a.w; q[4] = b.x; q[5] = b.y; q[6] = b.z; q[7] = b.w;
    }
    float m = -FLT_MAX, l = 0.f, acc[8];
#pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.f;
    const __half* K = kc + (size_t)j * max_ctx * 256 + lane * 8;
    const __half* V = vc + (size_t)j * max_ctx * 256 + lane * 8;
    uint4 kraw = __ldg((const uint4*)K), vraw = __ldg((const uint4*)V);
    for (int s = 0; s < n_kv; s++) {
        const uint4 kcur = kraw, vcur = vraw;
        if (s + 1 < n_kv) {
            kraw = __ldg((const uint4*)(K + (size_t)(s + 1) * 256));
            vraw = __ldg((const uint4*)(V + (size_t)(s + 1) * 256));
        }
        float k[8], v[8];
        const __half2* kh = (const __half2*)&kcur;
        const __half2* vh = (const __half2*)&vcur;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float2 kf = __half22float2(kh[i]), vf = __half22float2(vh[i]);
            k[2 * i] = kf.x; k[2 * i + 1] = kf.y; v[2 * i] = vf.x; v[2 * i + 1] = vf.y;
        }
        float dot = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) dot += q[i] * k[i];
        dot = warp_sum(dot) * (1.0f / 16.0f);
        const float mn = fmaxf(m, dot);
        const float c = expf(m - mn), p = expf(dot - mn);
        l = l * c + p;
#pragma unroll
        for (int i = 0; i < 8; i++) acc[i] = acc[i] * c + p * v[i];
        m = mn;
    }
    const float* gate = ya + (size_t)t * ldy + hl * 512 + 256 + lane * 8;
    float* dst = out + (size_t)t * 3072 + hl * 256 + lane * 8;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const float g = gate[i];
        dst[i] = (acc[i] / l) * (1.0f / (1.0f + expf(-g)));
    }
}

// Tensor-core causal attention (FA2-style) for the ubatch queries: block = (16 query tokens, kv head j), 6 warps = the 6 q
// heads of j. Per warp: Q (16 x 256, pre-scaled by 1/16, fp16) in registers, O (16 x 256 fp32) in registers, keys in
// steps of 16 through shared memory (fp16 K/V straight from the decode cache), S = Q K^T and O += P V with
// mma.m16n8k8 f16 -> f32, online softmax in fp32, P rounded to fp16 for the PV product. Output gated by sigmoid(gate).
__device__ __forceinline__ void mma1688(float* c, uint32_t a0, uint32_t a1, uint32_t b) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b));
}
__device__ __forceinline__ void ldsm4(uint32_t* r, const void* p) {
    const unsigned sp = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sp));
}
__device__ __forceinline__ void ldsm4t(uint32_t* r, const void* p) {
    const unsigned sp = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sp));
}
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *(const uint32_t*)&h;
}
__global__ void __launch_bounds__(192, 1) k_pf_fa(const float* __restrict__ qa, const __half* __restrict__ kc,
                                                  const __half* __restrict__ vc, const float* __restrict__ ya, int ldy,
                                                  int T, int max_ctx, int p0, float* out) {
    __shared__ __align__(16) __half Ks[16 * 256], Vs[16 * 256];
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t4 = lane & 3;
    const int j = blockIdx.y, hl = j * 6 + w, q0 = blockIdx.x * 16;
    const int r0 = q0 + g, r1 = q0 + g + 8;
    const int qp0 = p0 + r0, qp1 = p0 + r1;
    // Q fragments (scaled by 1/16, exact)
    uint32_t qf[32][2];
#pragma unroll
    for (int kk = 0; kk < 32; kk++) {
        float2 a = make_float2(0.f, 0.f), b = make_float2(0.f, 0.f);
        if (r0 < T) a = *(const float2*)(qa + (size_t)r0 * 3072 + hl * 256 + kk * 8 + 2 * t4);
        if (r1 < T) b = *(const float2*)(qa + (size_t)r1 * 3072 + hl * 256 + kk * 8 + 2 * t4);
        qf[kk][0] = pack_h2(a.x * 0.0625f, a.y * 0.0625f);
        qf[kk][1] = pack_h2(b.x * 0.0625f, b.y * 0.0625f);
    }
    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
    const int kend = p0 + min(q0 + 15, T - 1);  // last key position any row of this block attends to
    const int nsteps = kend / 16 + 1;
    const __half* Kg = kc + (size_t)j * max_ctx * 256;
    const __half* Vg = vc + (size_t)j * max_ctx * 256;
    for (int ks = 0; ks < nsteps; ks++) {
        __syncthreads();
        for (int u = tid; u < 1024; u += 192) {  // 16 keys x 32 units of 16 B, K then V
            const int isv = u >> 9, key = (u >> 5) & 15, un = u & 31, kp = ks * 16 + key;
            int4 v = make_int4(0, 0, 0, 0);
            if (kp <= kend) v = __ldg((const int4*)((isv ? Vg : Kg) + (size_t)kp * 256) + un);
            *(int4*)((isv ? Vs : Ks) + key * 256 + ((un ^ (key & 7)) * 8)) = v;
        }
        __syncthreads();
        // S = Q K^T for keys ks*16 + 0..15 (two n-tiles of 8 keys)
        float sc[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
            const int key = nt * 8 + (lane & 7);
#pragma unroll
            for (int kk = 0; kk < 32; kk += 4) {
                uint32_t kb[4];
                ldsm4(kb, Ks + key * 256 + (((kk + (lane >> 3)) ^ (key & 7)) * 8));
#pragma unroll
                for (int i = 0; i < 4; i++) mma1688(sc[nt], qf[kk + i][0], qf[kk + i][1], kb[i]);
            }
        }
        // causal mask + online softmax (rows g and g + 8; a row's 16 scores sit in the 4 lanes of its quad)
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int kp = ks * 16 + nt * 8 + 2 * t4 + e;
                if (kp > qp0) sc[nt][e] = -1e30f;
                if (kp > qp1) sc[nt][2 + e] = -1e30f;
                mx0 = fmaxf(mx0, sc[nt][e]);
                mx1 = fmaxf(mx1, sc[nt][2 + e]);
            }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float c0 = expf(m0 - mx0), c1 = expf(m1 - mx1);
        float ps0 = 0.f, ps1 = 0.f;
        uint32_t pa[2][2];
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            const float a = expf(sc[nt][0] - mx0), b = expf(sc[nt][1] - mx0);
            const float cc = expf(sc[nt][2] - mx1), d = expf(sc[nt][3] - mx1);
            ps0 += a + b;
            ps1 += cc + d;
            pa[nt][0] = pack_h2(a, b);
            pa[nt][1] = pack_h2(cc, d);
        }
        ps0 += __shfl_xor_sync(0xffffffffu, ps0, 1);
        ps0 += __shfl_xor_sync(0xffffffffu, ps0, 2);
        ps1 += __shfl_xor_sync(0xffffffffu, ps1, 1);
        ps1 += __shfl_xor_sync(0xffffffffu, ps1, 2);
        l0 = l0 * c0 + ps0;
        l1 = l1 * c1 + ps1;
        m0 = mx0;
        m1 = mx1;
#pragma unroll
        for (int i = 0; i < 32; i++) { o[i][0] *= c0; o[i][1] *= c0; o[i][2] *= c1; o[i][3] *= c1; }
        // O += P V (two k-steps of 8 keys, 32 n-tiles of 8 dims)
#pragma unroll
        for (int kk2 = 0; kk2 < 2; kk2++) {
            const int key = kk2 * 8 + (lane & 7);
#pragma unroll
            for (int nd = 0; nd < 32; nd += 4) {
                uint32_t vb[4];
                ldsm4t(vb, Vs + key * 256 + (((nd + (lane >> 3)) ^ (key & 7)) * 8));
#pragma unroll
                for (int i = 0; i < 4; i++) mma1688(o[nd + i], pa[kk2][0], pa[kk2][1], vb[i]);
            }
        }
    }
    const float il0 = 1.0f / l0, il1 = 1.0f / l1;
#pragma unroll
    for (int nd = 0; nd < 32; nd++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int d = nd * 8 + 2 * t4 + e;
            if (r0 < T) {
                const float gt = ya[(size_t)r0 * ldy + hl * 512 + 256 + d];
                out[(size_t)r0 * 3072 + hl * 256 + d] = (o[nd][e] * il0) * (1.0f / (1.0f + expf(-gt)));
            }
            if (r1 < T) {
                const float gt = ya[(size_t)r1 * ldy + hl * 512 + 256 + d];
                out[(size_t)r1 * 3072 + hl * 256 + d] = (o[nd][2 + e] * il1) * (1.0f / (1.0f + expf(-gt)));
            }
        }
}

// silu(g) * u for the gate|up output interleaved by 4 rows per 8-row tile. grid (T, 34) x 256
__global__ void k_pf_silu(const float* __restrict__ y, int ldy, float* out) {
    const int t = blockIdx.x, i = blockIdx.y * 256 + threadIdx.x;  // 0..8703
    const int gr = (i >> 2) * 8 + (i & 3);
    const float g = y[(size_t)t * ldy + gr], u = y[(size_t)t * ldy + gr + 4];
    out[(size_t)t * 8704 + i] = (g / (1.0f + expf(-g))) * u;
}

// ------------------------------------------------------------------------------------------------ host side
constexpr int NSUB = 2;  // sub-batches per ubatch: the all-reduce copy of one overlaps the other's compute

struct PfGpu {
    int* ids = nullptr;
    float *h = nullptr, *xn = nullptr, *y = nullptr, *part = nullptr, *rx[2] = {nullptr, nullptr};
    float *yab = nullptr, *qkv = nullptr, *o = nullptr, *g32 = nullptr, *qa = nullptr;
    unsigned char* gscr = nullptr;  // k_pf_gdnp scratch [24][chunks][gdnc::SCR]
    cudaEvent_t ch_ev[2][8] = {};    // pf_arc: end of K-split GEMM chunk c of sub-batch s
    int8_t* xq = nullptr;
    float2* xs = nullptr;
    float* xsum = nullptr;
    float* dx = nullptr;  // gemm8 activation block scales
    int8_t* xq2 = nullptr;  // fused gate|up silu epilogue output (down GEMM input)
    float* dx2 = nullptr;
    int8_t* dsh = nullptr;  // gemm17 GSH shift deltas [K/64][Tp] (permuted token order)
    int8_t* w8r[4] = {nullptr, nullptr, nullptr, nullptr};  // R512 rotated int8 weights: qkvz|qkv_a, ssm_out|wo, gateup, down
    float* invq = nullptr;  // R512: invr of qkvz + the alpha/beta rows [8448]
    float* dsr = nullptr;   // R512 per-512-block scale ratios [17][Up] (pf_rgb)
    float* dxt = nullptr;   // gemm17 per-token final factor [Tp]
    cudaStream_t sc = nullptr;              // copy stream (AR payloads)
    cudaEvent_t part_ev[NSUB] = {};         // compute stream: partial rows of sub s written
    cudaEvent_t sent[NSUB][2] = {};         // copy stream: sub s rows of AR slot copied to the peer
};
// ------------------------------------------------------------------------------------------------ activation-format study
// (option pf_fq) Emulates an alternative GEMM activation format on the GA64 int8 input, in place: x^ = xq * dx is
// re-quantized with the studied format, the result re-quantized to GA64 (so the error is the studied format's plus a
// second GA64 rounding: a conservative emulation). Stats per GEMM type: [calls, tokens, residual entries, sum of the
// per-call union of channels that needed an exact path, max union]; top-n channel picks are counted in the union too.
constexpr int FQ_KMAX = 8704;
struct FqBufs {
    float* camax = nullptr;            // [FQ_KMAX] per-channel amax over the call's tokens (float bits, atomicMax)
    unsigned char* sel = nullptr;      // [FQ_KMAX] top-n exact channels
    unsigned char* uni = nullptr;      // [FQ_KMAX] channels with an exact residual entry in this call
    unsigned long long* st = nullptr;  // [6][5]
};
__global__ void k_fq_amax(const int8_t* __restrict__ xq, const float* __restrict__ dx, int K, int Tp, float* camax) {
    const int t = blockIdx.x;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const float v = fabsf((float)xq[(size_t)t * K + k] * dx[(size_t)(k >> 6) * Tp + t]);
        atomicMax((unsigned*)&camax[k], __float_as_uint(v));
    }
}
// n iterations of a block argmax over the not-yet-selected channels (1 block x 1024)
__global__ void k_fq_topn(const float* __restrict__ camax, int K, int n, unsigned char* sel) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    for (int k = threadIdx.x; k < K; k += blockDim.x) sel[k] = 0;
    __syncthreads();
    for (int it = 0; it < n; it++) {
        float v = -1.f;
        int idx = 0;
        for (int k = threadIdx.x; k < K; k += blockDim.x)
            if (!sel[k] && camax[k] > v) { v = camax[k]; idx = k; }
        for (int o = 16; o > 0; o >>= 1) {
            const float v2 = __shfl_xor_sync(0xffffffffu, v, o);
            const int i2 = __shfl_xor_sync(0xffffffffu, idx, o);
            if (v2 > v || (v2 == v && i2 < idx)) { v = v2; idx = i2; }
        }
        if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = v; bi[threadIdx.x >> 5] = idx; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float m = bv[0];
            int mi = bi[0];
            for (int w = 1; w < (int)(blockDim.x >> 5); w++)
                if (bv[w] > m || (bv[w] == m && bi[w] < mi)) { m = bv[w]; mi = bi[w]; }
            sel[mi] = 1;
        }
        __syncthreads();
    }
}
// grid Ts x 256, dynamic smem K floats
__global__ void __launch_bounds__(256) k_fq_apply(int8_t* xq, float* dx, int K, int Tp, int mode, float alpha, int G,
                                                  const unsigned char* __restrict__ sel, unsigned char* uni,
                                                  unsigned long long* st) {
    extern __shared__ float xs[];
    __shared__ float red[8];
    __shared__ float redm[8];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int k = tid; k < K; k += 256) xs[k] = (float)xq[(size_t)t * K + k] * dx[(size_t)(k >> 6) * Tp + t];
    __syncthreads();
    unsigned cnt = 0;
    // mode 6: random signs + block Walsh-Hadamard (size G) before the per-token quantization, inverse after;
    // mode 8: per-channel smoothing x / s_k, s_k = sqrt(ubatch channel amax) (camax passed through sel's slot)
    auto fwht = [&]() {
        const float nrm = rsqrtf((float)G);
        for (int k = tid; k < K; k += 256) xs[k] *= ((k * 2654435761u) & 0x80000000u) ? -nrm : nrm;
        __syncthreads();
        for (int h = 1; h < G; h <<= 1) {
            for (int idx = tid; idx < K / 2; idx += 256) {
                const int blk = idx / (G / 2), j = idx % (G / 2);
                const int i = blk * G + (j / h) * 2 * h + (j % h);
                const float a = xs[i], b = xs[i + h];
                xs[i] = a + b; xs[i + h] = a - b;
            }
            __syncthreads();
        }
    };
    auto ifwht = [&]() {
        for (int h = 1; h < G; h <<= 1) {
            for (int idx = tid; idx < K / 2; idx += 256) {
                const int blk = idx / (G / 2), j = idx % (G / 2);
                const int i = blk * G + (j / h) * 2 * h + (j % h);
                const float a = xs[i], b = xs[i + h];
                xs[i] = a + b; xs[i + h] = a - b;
            }
            __syncthreads();
        }
        const float nrm = rsqrtf((float)G);
        for (int k = tid; k < K; k += 256) xs[k] *= ((k * 2654435761u) & 0x80000000u) ? -nrm : nrm;
        __syncthreads();
    };
    const float* smooth = mode == 8 ? (const float*)sel : nullptr;
    if (mode == 8) sel = nullptr;
    if (mode == 6) fwht();
    if (mode == 8) {
        for (int k = tid; k < K; k += 256) xs[k] /= smooth[k];
        __syncthreads();
    }
    if (mode == 9) {  // per-token scale D times a power of two per G-group: step = D 2^-e / 127, e <= Emax (alpha)
        float am = 0.f;
        for (int k = tid; k < K; k += 256) am = fmaxf(am, fabsf(xs[k]));
        for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
        if (lane == 0) redm[warp] = am;
        __syncthreads();
        float D = 0.f;
        for (int w = 0; w < 8; w++) D = fmaxf(D, redm[w]);
        const int emax = (int)(alpha * 10.f + 0.5f);
        for (int g = warp; g < K / G; g += 8) {
            float m = 0.f;
            for (int k = lane; k < G; k += 32) m = fmaxf(m, fabsf(xs[g * G + k]));
            for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
            int e = emax;
            if (m > 0.f) e = min(emax, (int)floorf(log2f(D / m)));
            const float d = ldexpf(D, -e) / 127.f;
            if (D > 0.f)
                for (int k = lane; k < G; k += 32) xs[g * G + k] = d * rintf(xs[g * G + k] / d);
        }
    } else if (mode == 5) {
        for (int g = warp; g < K / G; g += 8) {
            float m = 0.f;
            for (int k = lane; k < G; k += 32) m = fmaxf(m, fabsf(xs[g * G + k]));
            for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
            const float d = m / 127.f;
            if (m > 0.f)
                for (int k = lane; k < G; k += 32) xs[g * G + k] = d * rintf(xs[g * G + k] / d);
        }
    } else {
        float ss = 0.f, am = 0.f, nc = 0.f;
        for (int k = tid; k < K; k += 256) {
            if (sel && sel[k]) continue;
            ss += xs[k] * xs[k];
            am = fmaxf(am, fabsf(xs[k]));
            nc += 1.f;
        }
        for (int o = 16; o > 0; o >>= 1) {
            ss += __shfl_xor_sync(0xffffffffu, ss, o);
            nc += __shfl_xor_sync(0xffffffffu, nc, o);
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
        }
        __shared__ float rn[8];
        if (lane == 0) { red[warp] = ss; redm[warp] = am; rn[warp] = nc; }
        __syncthreads();
        ss = 0.f; am = 0.f; nc = 0.f;
        for (int w = 0; w < 8; w++) { ss += red[w]; am = fmaxf(am, redm[w]); nc += rn[w]; }
        const float rms = sqrtf(ss / fmaxf(nc, 1.f));
        const float c = (mode == 2 || mode == 4 || (mode == 6 && alpha > 0.f)) ? fminf(am, alpha * rms) : am;
        const float d = c / 127.f;
        for (int k = tid; k < K; k += 256) {
            if (sel && sel[k]) continue;
            const float v = xs[k];
            if (fabsf(v) > c) { cnt++; uni[k] = 1; continue; }
            if (d > 0.f) xs[k] = d * rintf(v / d);
        }
    }
    __syncthreads();
    if (mode == 6) ifwht();
    if (mode == 8) {
        for (int k = tid; k < K; k += 256) xs[k] *= smooth[k];
        __syncthreads();
    }
    // back to GA64: one warp per 64-group
    for (int g = warp; g < K / 64; g += 8) {
        const float a0 = xs[g * 64 + lane], a1 = xs[g * 64 + 32 + lane];
        float m = fmaxf(fabsf(a0), fabsf(a1));
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        const float d = m / 127.f;
        xq[(size_t)t * K + g * 64 + lane] = (int8_t)(m > 0.f ? __float2int_rn(a0 / d) : 0);
        xq[(size_t)t * K + g * 64 + 32 + lane] = (int8_t)(m > 0.f ? __float2int_rn(a1 / d) : 0);
        if (lane == 0) dx[(size_t)g * Tp + t] = d;
    }
    for (int o = 16; o > 0; o >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
    if (lane == 0 && cnt) atomicAdd(&st[2], (unsigned long long)cnt);
}
// s_k = sqrt(camax_k) (floored) in place
__global__ void k_fq_smooth(float* camax, int K) {
    const int k = blockIdx.x * 256 + threadIdx.x;
    if (k < K) camax[k] = sqrtf(fmaxf(camax[k], 1e-6f));
}
__global__ void k_fq_uni(unsigned char* uni, const unsigned char* sel, int K, int T, unsigned long long* st) {
    __shared__ unsigned red[8];
    unsigned c = 0;
    for (int k = threadIdx.x; k < K; k += 256) {
        c += (uni[k] || (sel && sel[k])) ? 1u : 0u;
        uni[k] = 0;
    }
    for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = c;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned s = 0;
        for (int w = 0; w < 8; w++) s += red[w];
        st[0] += 1; st[1] += T; st[3] += s;
        if (s > st[4]) st[4] = s;
    }
}

struct Pf {
    int cap = 0;  // allocated ubatch capacity (tokens)
    PfGpu G[2];
    FqBufs fq[2];
};

template <class T>
T* dalloc(size_t n) {
    T* p;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

void pf_rot_prep(t4q_ctx* c);
Pf* pf_get(t4q_ctx* c) {
    tp::State& S = *c->tps;
    const int ub = S.pf_ub;
    Pf* P = (Pf*)S.pf;
    if (P && P->cap == ub) return P;  // reallocated whenever pf_ub changes (batched decode shrinks it to save VRAM)
    if (P) {
        for (int g = 0; g < 2; g++) {
            CK(cudaSetDevice(g));
            CK(cudaDeviceSynchronize());
            PfGpu& B = P->G[g];
            for (void* p : {(void*)B.ids, (void*)B.h, (void*)B.xn, (void*)B.y, (void*)B.part, (void*)B.rx[0],
                            (void*)B.rx[1], (void*)B.yab, (void*)B.gscr, (void*)B.qkv, (void*)B.o, (void*)B.g32, (void*)B.qa,
                            (void*)B.xq, (void*)B.xs, (void*)B.xsum, (void*)B.dx, (void*)B.xq2, (void*)B.dx2,
                            (void*)B.dsh, (void*)B.dxt, (void*)B.dsr})
                cudaFree(p);
            for (auto e : B.part_ev) cudaEventDestroy(e);
            for (auto& r : B.sent) for (auto e : r) cudaEventDestroy(e);
            for (auto& r : B.ch_ev) for (auto e : r) if (e) cudaEventDestroy(e);
            cudaStreamDestroy(B.sc);
        }
        delete P;
        S.pf = nullptr;  // a throw below must not leave S.pf dangling at the freed Pf
    }
    P = new Pf();
    P->cap = ub;
    const size_t U = ub, Up = (size_t)(ub + 255) / 256 * 256;
    for (int g = 0; g < 2; g++) {
        CK(cudaSetDevice(g));
        PfGpu& B = P->G[g];
        B.ids = dalloc<int>(U);
        B.h = dalloc<float>(U * D);
        B.xn = dalloc<float>(U * D);
        B.y = dalloc<float>(U * 17408);
        B.part = dalloc<float>(U * D);
        B.rx[0] = dalloc<float>(U * D);
        B.rx[1] = dalloc<float>(U * D);
        B.yab = dalloc<float>(U * 48 * AB_KS);
        B.gscr = dalloc<unsigned char>((size_t)24 * ((U + gdnc::C - 1) / gdnc::C) * gdnc::SCR);
        B.qkv = dalloc<float>(U * 5120);
        B.o = dalloc<float>(U * 3072);
        B.g32 = dalloc<float>(U * 8704);
        B.qa = dalloc<float>(U * 3072);
        B.xq = dalloc<int8_t>(Up * 8704);
        B.xs = dalloc<float2>(Up * (8704 / 32));
        B.xsum = dalloc<float>(Up * (8704 / 32));
        B.dx = dalloc<float>(Up * (8704 / 32));
        B.xq2 = dalloc<int8_t>(Up * 8704);
        B.dx2 = dalloc<float>(Up * (8704 / 32));
        B.dsh = dalloc<int8_t>(Up * (8704 / 64));
        B.dxt = dalloc<float>(Up);
        B.dsr = dalloc<float>(Up * 17);
        CK(cudaStreamCreateWithFlags(&B.sc, cudaStreamNonBlocking));
        for (auto& e : B.part_ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        for (auto& r : B.sent) for (auto& e : r) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        // the memsets above run on the legacy stream, which does not order against the engine's non-blocking streams
        CK(cudaDeviceSynchronize());
    }
    // gemm8 per-row scales of every prefill GEMM weight (once per context)
    for (int g = 0; g < 2; g++) {
        CK(cudaSetDevice(g));
        tp::Gpu& G = S.G[g];
        for (int il = 0; il < 64; il++) {
            tp::Layer& L = G.L[il];
            for (tp::FW* W : {&L.qkvz, &L.ssm_out, &L.qkv_a, &L.wo, &L.gateup, &L.down}) {
                if (!W->ok() || W->invs) continue;
                CK(cudaMalloc(&W->invs, (size_t)W->L.N * 4));
                gemm8::Args a = gemm8::make_args(W->L, W->base, W->invs, nullptr, nullptr, nullptr, 0, 0, 0);
                CK(gemm8::row_invs(W->L.fmt, W->L.rpl, a, W->invs, G.s));
            }
        }
        CK(cudaStreamSynchronize(G.s));
    }
    S.pf = P;
    return P;
}

// R512: rotated-row scales (once per context) and the per-GEMM-type int8 scratch
void pf_rot_prep(t4q_ctx* c) {
    tp::State& S = *c->tps;
    Pf* P = (Pf*)S.pf;
    if (!S.pf_rot || !P) return;
    for (int g = 0; g < 2; g++) {
        CK(cudaSetDevice(g));
        tp::Gpu& G = S.G[g];
        PfGpu& B = P->G[g];
        const size_t sz[4] = {(size_t)8448 * 5120, (size_t)5120 * 3072, (size_t)17408 * 5120, (size_t)5120 * 8704};
        if (!B.invq) CK(cudaMalloc(&B.invq, 8448 * 4));
        for (int i = 0; i < 4; i++)
            if (!B.w8r[i]) CK(cudaMalloc(&B.w8r[i], sz[i]));
        for (int il = 0; il < 64; il++) {
            tp::Layer& L = G.L[il];
            if (L.ab && !L.ab8r) {
                CK(cudaMalloc(&L.ab8r, (size_t)256 * D));
                CK(cudaMalloc(&L.ab8i, 256 * 4));
                rot::w8r_f32_kernel<<<32, 256, 0, G.s>>>(L.ab, 48, 256, D, L.ab8i, L.ab8r);
                CK(cudaGetLastError());
            }
            for (tp::FW* W : {&L.qkvz, &L.ssm_out, &L.qkv_a, &L.wo, &L.gateup, &L.down}) {
                if (!W->ok() || W->invr) continue;
                CK(cudaMalloc(&W->invr, (size_t)W->L.N * 4));
                gemm8::Args a = gemm8::make_args(W->L, W->base, nullptr, nullptr, nullptr, nullptr, 0, 0, 0);
                CK(rot::invr(W->L.fmt, W->L.rpl, a, W->invr, G.s));
            }
        }
        CK(cudaStreamSynchronize(G.s));
    }
    // persistent rotated weights within the VRAM budget (both GPUs cache the same layers)
    if (S.pf_wcache > 0) {
        size_t budget = (size_t)S.pf_wcache << 20;
        for (int g = 0; g < 2; g++) {
            CK(cudaSetDevice(g));
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            const size_t margin = (size_t)1200 << 20;
            budget = std::min(budget, fr > margin ? fr - margin : 0);
        }
        for (int il = 0; il < 64; il++) {
            for (int wi = 0; wi < 6; wi++) {
                tp::Layer& L0 = S.G[0].L[il];
                tp::FW* W0 = wi == 0 ? &L0.gateup : wi == 1 ? &L0.down : wi == 2 ? &L0.qkvz : wi == 3 ? &L0.qkv_a : wi == 4 ? &L0.ssm_out : &L0.wo;
                if (!W0->ok() || W0->w8c) continue;
                static const int wbits[6] = {4, 8, 1, 2, 16, 32};  // gateup, down, qkvz, qkv_a, ssm_out, wo
                if (!(S.pf_rot_mask & wbits[wi])) continue;       // only cache GEMM types that run on R512
                const bool abr = wi == 2 && L0.ab8r;
                const size_t rows = abr ? 8448 : W0->L.N, bytes = rows * W0->L.K;
                if (bytes > budget) continue;
                budget -= bytes;
                for (int g = 0; g < 2; g++) {
                    CK(cudaSetDevice(g));
                    tp::Gpu& G = S.G[g];
                    tp::Layer& L = G.L[il];
                    tp::FW* W = wi == 0 ? &L.gateup : wi == 1 ? &L.down : wi == 2 ? &L.qkvz : wi == 3 ? &L.qkv_a : wi == 4 ? &L.ssm_out : &L.wo;
                    CK(cudaMalloc(&W->w8c, bytes));
                    CK(cudaMalloc(&W->invc, rows * 4));
                    gemm8::Args q = gemm8::make_args(W->L, W->base, nullptr, nullptr, nullptr, nullptr, 0, 0, 0);
                    CK(rot::convert(W->L.fmt, W->L.rpl, q, W->invr, W->w8c, G.s, S.pf_rcf != 0));
                    CK(cudaMemcpyAsync(W->invc, W->invr, (size_t)W->L.N * 4, cudaMemcpyDeviceToDevice, G.s));
                    if (abr) {
                        CK(cudaMemcpyAsync(W->w8c + (size_t)8192 * W->L.K, L.ab8r, (size_t)256 * W->L.K, cudaMemcpyDeviceToDevice, G.s));
                        CK(cudaMemcpyAsync(W->invc + 8192, L.ab8i, 256 * 4, cudaMemcpyDeviceToDevice, G.s));
                    }
                    S.pf_wcache_used += (long long)bytes;
                }
            }
        }
        for (int g = 0; g < 2; g++) { CK(cudaSetDevice(g)); CK(cudaStreamSynchronize(S.G[g].s)); }
    }
}

void ck_launch(const char* w) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("prefill launch failed: ") + w + ": " + cudaGetErrorString(e));
}

struct PfRun {
    t4q_ctx* c;
    Pf* P;
    int T = 0, p0 = 0;
    int nsub = 1, st0[NSUB] = {0, 0}, sT[NSUB] = {0, 0};
    int ar = 0;  // all-reduce counter (slot = ar & 1); all sub-batches of one phase share the slot (disjoint rows)
    bool i4 = false;
    bool g8 = false;  // gemm8 path (pf_g8)
    int chk_done[2] = {0, 0};
    bool chunked[2][NSUB] = {};  // the last K-split GEMM of (gpu, sub-batch) ran in pf_arc chunks
    int rchk_done[2] = {0, 0};
    bool ar16 = false;  // pf_ar16: K-split GEMMs write fp16 partials, the all-reduce copies fp16
    int ga = 32;      // gemm8 activation scale group (pf_ga)
    bool g17 = false; // gemm17 shift-folded 64-groups (pf_g17): unfused producers + quant_gsh
    bool rot = false; // R512 (pf_rot): rotated per-token activations x rotated int8 weights (gemm17 per-token)
    bool rgb = false; // pf_rgb: R512 activations with per-512-block scales (gemm17 GSH 2)
    int rmask = 63;   // pf_rot_mask: GEMM types on R512 (1 qkvz, 2 qkv_a, 4 gateup, 8 down, 16 ssm_out, 32 attn_out)
    static int wbit(const tp::FW& W) {
        return W.L.N == 8192 ? 1 : W.L.N == 7168 ? 2 : W.L.N == 17408 ? 4 : W.L.K == 8704 ? 8 : W.L.fmt == gemv::FAST_K5 ? 16 : 32;
    }
    bool rotw(const tp::FW& W) const { return rot && (rmask & wbit(W)); }
    int emax = 7;
    int pfk = 0;            // gemm9 L2 weight prefetch distance in 32-blocks (batched decode option bd_pfk)
    bool gemmr = false;     // batched decode: P4 GEMMs at 32 / 64-token tiles through gemmr (register-direct A fragments)
    int lbm = 1;            // gemm9 line-batched weight loads at BN <= 64 (batched decode option bd_lbm)
    int ksplit = 1;         // batched decode: split-K slices for the K-split GEMMs (fp32 slices in kscr, summed by ksum)
    float* kscr[2] = {nullptr, nullptr};
    bool p2p_part = false;  // batched decode: fp16 K-split GEMM partials also stored straight into the peer's rx slot
    int tp_force = 0;  // batched decode: GEMM token padding (32, 64 or a multiple of 128), 0 = prefill rule
    // largest gemm9 token tile <= cap that divides Tp (Tp is a multiple of 32)
    static int bn_div(int Tp, int cap) {
        for (int bn = cap; bn > 32; bn >>= 1)
            if (Tp % bn == 0) return bn;
        return 32;
    }
    int tpad(int Ts) const { return tp_force ? tp_force : g8 ? (Ts + 255) / 256 * 256 : (Ts + 127) / 128 * 128; }
    // pf_arc: chunk rows (a multiple of 128) and count for sub-batch s; 1 chunk = off
    int chunk_rows(int s) const {
        const int n = c->tps->pf_arc;
        return ((sT[s] + n - 1) / n + 127) / 128 * 128;
    }
    int arc_chunks(int s) const {
        if (!g8 || c->tps->pf_arc <= 1) return 1;
        const int cs = chunk_rows(s);
        return (sT[s] + cs - 1) / cs;
    }
    // profiling (option pf_prof): events on GPU0's stream after each op group, named by the op that just ended
    std::vector<cudaEvent_t>* ev = nullptr;
    std::vector<const char*>* evn = nullptr;
    std::vector<cudaEvent_t>* ev1 = nullptr;  // batched decode: GPU1 events as well
    std::vector<const char*>* evn1 = nullptr;
    void mark(int g, const char* name) {
        if (g == 1 && ev1) {
            cudaEvent_t e;
            CK(cudaEventCreate(&e));
            CK(cudaEventRecord(e, c->tps->G[1].s));
            ev1->push_back(e);
            evn1->push_back(name);
            return;
        }
        if (!ev || g != 0) return;
        cudaEvent_t e;
        CK(cudaEventCreate(&e));
        CK(cudaEventRecord(e, c->tps->G[0].s));
        ev->push_back(e);
        evn->push_back(name);
    }

    // sub-batch s: x (fp32 rows t0.., row stride K) -> q8 activations for the GEMM (rows 0..Tp-1 of the shared buffer)
    void quant(int g, int s, const float* x, int K, bool rq = false) {
        tp::Gpu& G = c->tps->G[g];
        PfGpu& B = P->G[g];
        const int Ts = sT[s], Tps = tpad(Ts);
        const int n = Tps * (K >> 5);
        const float* xs0 = x + (size_t)st0[s] * K;
        if (rq) rot::quant_rot<float>(xs0, K, Ts, Tps, K, B.xq, B.dxt, G.s, rgb ? B.dsr : nullptr);
        else if (g17) g16::quant_gsh_kernel<float><<<Tps, 256, 0, G.s>>>(xs0, K, Ts, K, emax, B.xq, B.dsh, B.dxt, Tps);
        else if (g8) gemm8::quant8(xs0, K, Ts, Tps, K, B.xq, B.dx, G.s, ga);
        else if (i4) gemm::quant_rows_i4_kernel<<<(n + 127) / 128, 128, 0, G.s>>>(xs0, K, Ts, Tps, K, B.xq, B.xs, B.xsum);
        else gemm::quant_rows_kernel<<<(n + 127) / 128, 128, 0, G.s>>>(xs0, K, Ts, Tps, K, B.xq, B.xs, B.xsum);
        ck_launch("quant");
        mark(g, "quant");
    }
    // silu: gate|up GEMM whose epilogue writes q8(silu(gate) * up) into xq2/dx2; in2: read the activations from xq2/dx2
    void gemm(int g, int s, const tp::FW& W, float* y, int ldy, bool silu = false, bool in2 = false) {
        tp::Gpu& G = c->tps->G[g];
        PfGpu& B = P->G[g];
        const int Ts = sT[s], Tps = tpad(Ts);
        cudaError_t e;
        gemm::GemmArgs a = gemm::make_args(W.L, W.base, B.xq, B.xs, B.xsum, y + (size_t)st0[s] * ldy, ldy, Ts, Tps);
        if (y == B.part) chunked[g][s] = false;
        if (g8 && !g17 && !rotw(W) && c->tps->pf_fq) fq(g, s, W, in2);
        if (rotw(W)) {
            const int slot = (W.L.N == 8192 || W.L.N == 7168) ? 0 : W.L.N == 17408 ? 2 : W.L.K == 8704 ? 3 : 1;
            gemm8::Args q = gemm8::make_args(W.L, W.base, nullptr, nullptr, nullptr, nullptr, 0, 0, 0);
            const bool abrows = W.L.N == 8192 && ldy == 8448;  // + 256 alpha/beta rows (48 used) of this layer
            const tp::Layer* Lab = nullptr;
            if (abrows)
                for (int il2 = 0; il2 < 64 && !Lab; il2++)
                    if (&c->tps->G[g].L[il2].qkvz == &W) Lab = &c->tps->G[g].L[il2];
            const bool cached = W.w8c != nullptr && c->tps->pf_wcache > 0;
            if (s == 0 && abrows && !cached) {
                CK(cudaMemcpyAsync(B.w8r[0] + (size_t)8192 * W.L.K, Lab->ab8r, (size_t)256 * W.L.K, cudaMemcpyDeviceToDevice, G.s));
                CK(cudaMemcpyAsync(B.invq, W.invr, 8192 * 4, cudaMemcpyDeviceToDevice, G.s));
                CK(cudaMemcpyAsync(B.invq + 8192, Lab->ab8i, 256 * 4, cudaMemcpyDeviceToDevice, G.s));
            }
            if (s == 0 && !cached) {  // both sub-batches use the same converted weights
                CK(rot::convert(W.L.fmt, W.L.rpl, q, W.invr, B.w8r[slot], G.s, c->tps->pf_rcf != 0));
                if (c->tps->pf_rot_chk && !(rchk_done[g] & (1 << slot))) {  // fp16 converter vs the fp32 one, once per slot
                    rchk_done[g] |= 1 << slot;
                    const size_t n = (size_t)W.L.N * W.L.K;
                    int8_t* tmp;
                    unsigned* dd;
                    CK(cudaMallocAsync(&tmp, n, G.s));
                    CK(cudaMallocAsync(&dd, 12, G.s));
                    CK(cudaMemsetAsync(dd, 0, 12, G.s));
                    CK(rot::convert(W.L.fmt, W.L.rpl, q, W.invr, tmp, G.s, false));
                    k_i8diff<<<256, 256, 0, G.s>>>(B.w8r[slot], tmp, n, dd);
                    unsigned h[3];
                    CK(cudaMemcpyAsync(h, dd, 12, cudaMemcpyDeviceToHost, G.s));
                    CK(cudaStreamSynchronize(G.s));
                    char buf[200];
                    snprintf(buf, sizeof buf, "{\"gpu\": %d, \"slot\": %d, \"fmt\": %d, \"n\": %zu, \"maxdiff\": %u, \"gt1\": %u, \"ne\": %u}",
                             g, slot, W.L.fmt, n, h[0], h[1], h[2]);
                    tp::State& S2 = *c->tps;
                    if (S2.pf_gdnc_json.size() < 3000) S2.pf_gdnc_json += (S2.pf_gdnc_json.empty() ? "" : ", ") + std::string(buf);
                    CK(cudaFreeAsync(tmp, G.s));
                    CK(cudaFreeAsync(dd, G.s));
                }
                mark(g, "rot_convert");
            }
            g16::Args a7;
            a7.w8 = cached ? W.w8c : B.w8r[slot]; a7.invs = cached ? W.invc : abrows ? B.invq : W.invr; a7.xq = B.xq; a7.dx = B.dxt;
            a7.N = abrows ? 8448 : W.L.N; a7.K = W.L.K; a7.T = Ts; a7.Tp = Tps;
            if (rgb) a7.dsr = B.dsr;
            if (silu) {  // silu(gate) * up rows of 8704 into g32 (fp32; pf_h16: fp16)
                a7.ldy = 8704;
                if (c->tps->pf_h16) {
                    a7.yh = (__half*)B.g32 + (size_t)st0[s] * 8704;
                    e = rgb ? g16::launch17_t<0, gemv::FAST_P4, 4, 2, 2, 1>(a7, G.s) : g16::launch17_t<0, gemv::FAST_P4, 4, 2, 0, 1>(a7, G.s);
                } else {
                    a7.y = B.g32 + (size_t)st0[s] * 8704;
                    e = rgb ? g16::launch17_t<0, gemv::FAST_P4, 4, 3, 2, 1>(a7, G.s) : g16::launch17_t<0, gemv::FAST_P4, 4, 3, 0, 1>(a7, G.s);
                }
            } else {
                a7.y = y + (size_t)st0[s] * ldy; a7.ldy = ldy;
                if (ar16 && y == B.part) { a7.yh = (__half*)B.part + (size_t)st0[s] * ldy; a7.y = nullptr; }
                if (rgb) e = a7.yh ? g16::launch17_t<0, gemv::FAST_P4, 4, 1, 2, 1>(a7, G.s) : g16::launch17_t<0, gemv::FAST_P4, 4, 0, 2, 1>(a7, G.s);
                else e = a7.yh ? g16::launch17_t<0, gemv::FAST_P4, 4, 1, 0, 1>(a7, G.s) : g16::launch17_t<0, gemv::FAST_P4, 4, 0, 0, 1>(a7, G.s);
            }
        } else if (g17) {
            g16::Args a7;
            a7.q = gemm8::make_args(W.L, W.base, W.invs, nullptr, nullptr, nullptr, 0, 0, 0);
            a7.invs = W.invs; a7.xq = B.xq; a7.dx = B.dxt; a7.dsh = B.dsh;
            a7.y = y + (size_t)st0[s] * ldy; a7.ldy = ldy;
            if (ar16 && y == B.part) { a7.yh = (__half*)B.part + (size_t)st0[s] * ldy; a7.y = nullptr; }
            a7.N = W.L.N; a7.K = W.L.K; a7.T = Ts; a7.Tp = Tps;
            e = g16::launch17(W.L.fmt, W.L.rpl, 1, a7, G.s);
        } else if (g8 && silu) {
            gemm8::Args a8 = gemm8::make_args(W.L, W.base, W.invs, B.xq, B.dx, nullptr, 0, Ts, Tps);
            a8.oq = B.xq2; a8.odx = B.dx2;
            a8.pfk = pfk;
            a8.lbm = lbm;
            if (gemmr && ga == 64 && W.L.fmt == gemv::FAST_P4 && (Tps == 32 || Tps == 64))
                e = gemmr::launch(W.L.fmt, W.L.rpl, Tps, 2, true, a8, G.s);
            else
                e = gemm8::launch9_silu(W.L.fmt, W.L.rpl, a8, G.s, bn_div(Tps, 256));
        } else if (g8) {
            gemm8::Args a8 = gemm8::make_args(W.L, W.base, W.invs, in2 ? B.xq2 : B.xq, in2 ? B.dx2 : B.dx,
                                              y + (size_t)st0[s] * ldy, ldy, Ts, Tps);
            if (ar16 && y == B.part) {
                a8.yh = (__half*)B.part + (size_t)st0[s] * ldy;
                a8.y = nullptr;
                if (p2p_part) a8.yh2 = (__half*)P->G[1 - g].rx[ar & 1] + (size_t)st0[s] * ldy;  // send() copies nothing
            }
            a8.pfk = pfk;
            a8.lbm = lbm;
            if (gemmr && ga == 64 && W.L.fmt == gemv::FAST_P4 && (Tps == 32 || Tps == 64) &&
                !(y == B.part && ksplit > 1)) {
                e = gemmr::launch(W.L.fmt, W.L.rpl, Tps, W.L.N >= 8192 ? 2 : 1, false, a8, G.s);
                if (e != cudaSuccess) throw std::runtime_error(std::string("gemmr launch: ") + cudaGetErrorString(e));
                mark(g, W.L.N == 8192 ? "gemm_qkvz" : W.L.N == 7168 ? "gemm_attn_qkv" : W.L.N == 17408 ? "gemm_gateup"
                        : W.L.K == 8704 ? "gemm_down" : "gemm_attn_out");
                return;
            }
            const int pbn = c->tps->pf_bn;
            const int bn = Tps < 128 ? bn_div(Tps, 64) : pbn ? pbn : (Tps >= 512 ? 256 : 128);
            if (y == B.part && ksplit > 1 && kscr[g] && W.L.K % (256 * ksplit) == 0) {  // split-K, then a fixed-order sum
                gemm8::Args az = a8;
                az.kz = ksplit;
                az.y = kscr[g];
                az.zs = (long long)Tps * ldy;
                az.yh = nullptr; az.yh2 = nullptr;
                e = gemm8::launch9(W.L.fmt, W.L.rpl, (Tps % bn) ? bn_div(Tps, 128) : bn, ga, az, G.s);
                if (e == cudaSuccess) {
                    gemm8::ksum_kernel<<<dim3(Ts, ldy / 256), 256, 0, G.s>>>(kscr[g], az.zs, ksplit, ldy, a8.y, a8.yh, a8.yh2);
                    e = cudaGetLastError();
                }
                if (e != cudaSuccess) throw std::runtime_error(std::string("gemm split-K: ") + cudaGetErrorString(e));
                mark(g, W.L.K == 8704 ? "gemm_down" : W.L.fmt == gemv::FAST_K5 ? "gemm_ssm_out" : "gemm_attn_out");
                return;
            }
            const int nch = arc_chunks(s);
            if (y == B.part && nch > 1) {  // pf_arc: token chunks, each followed by an event for its AR copy (send)
                const int cs = chunk_rows(s);
                for (int ci = 0; ci < nch; ci++) {
                    const int r0 = ci * cs, rows = std::min(cs, Ts - r0);
                    gemm8::Args ac = a8;
                    ac.xq = a8.xq + (size_t)r0 * W.L.K;
                    ac.dx = a8.dx + r0;
                    ac.dxs = Tps;
                    if (ac.yh) ac.yh = a8.yh + (size_t)r0 * ldy;
                    if (ac.y) ac.y = a8.y + (size_t)r0 * ldy;
                    ac.T = rows;
                    ac.Tp = (rows + 127) / 128 * 128;
                    e = gemm8::launch9(W.L.fmt, W.L.rpl, (ac.Tp % 256) ? 128 : 256, ga, ac, G.s);
                    if (e != cudaSuccess) break;
                    if (!B.ch_ev[s][ci]) CK(cudaEventCreateWithFlags(&B.ch_ev[s][ci], cudaEventDisableTiming));
                    CK(cudaEventRecord(B.ch_ev[s][ci], G.s));
                }
                chunked[g][s] = true;
            } else {
                e = gemm8::launch9(W.L.fmt, W.L.rpl, (Tps % bn) ? bn_div(Tps, 128) : bn, ga, a8, G.s);
            }
        } else if (i4 && W.L.fmt != gemv::FAST_K5) {
            if (W.L.fmt == gemv::FAST_P4) e = W.L.rpl == 4 ? gemm::gemm_launch<gemv::FAST_P4, 4, 3>(a, G.s)
                                                             : gemm::gemm_launch<gemv::FAST_P4, 2, 3>(a, G.s);
            else e = W.L.rpl == 4 ? gemm::gemm_launch<gemv::FAST_P4M, 4, 3>(a, G.s)
                                  : gemm::gemm_launch<gemv::FAST_P4M, 2, 3>(a, G.s);
        } else {
            e = gemm::gemm_launch_fmt(W.L.fmt, W.L.rpl, a, G.s);
        }
        if (e != cudaSuccess) throw std::runtime_error(std::string("gemm launch: ") + cudaGetErrorString(e));
        mark(g, W.L.N == 8192 ? "gemm_qkvz" : W.L.N == 7168 ? "gemm_attn_qkv" : W.L.N == 17408 ? "gemm_gateup"
                : W.L.K == 8704 ? "gemm_down" : W.L.fmt == gemv::FAST_K5 ? "gemm_ssm_out" : "gemm_attn_out");
    }
    // activation-format study (pf_fq) on this GEMM's GA64 input
    void fq(int g, int s, const tp::FW& W, bool in2) {
        tp::State& S = *c->tps;
        const int type = W.L.N == 8192 ? 0 : W.L.N == 7168 ? 1 : W.L.N == 17408 ? 2 : W.L.K == 8704 ? 3
                         : W.L.fmt == gemv::FAST_K5 ? 4 : 5;
        if (ga != 64 || !((S.pf_fq_mask >> type) & 1)) return;
        tp::Gpu& G = S.G[g];
        PfGpu& B = P->G[g];
        FqBufs& F = P->fq[g];
        if (!F.camax) {
            CK(cudaMalloc(&F.camax, FQ_KMAX * 4));
            CK(cudaMalloc(&F.sel, FQ_KMAX));
            CK(cudaMalloc(&F.uni, FQ_KMAX));
            CK(cudaMalloc(&F.st, 6 * 5 * 8));
            CK(cudaMemsetAsync(F.sel, 0, FQ_KMAX, G.s));
            CK(cudaMemsetAsync(F.uni, 0, FQ_KMAX, G.s));
            CK(cudaMemsetAsync(F.st, 0, 6 * 5 * 8, G.s));
        }
        int8_t* xq = in2 ? B.xq2 : B.xq;
        float* dx = in2 ? B.dx2 : B.dx;
        const int K = W.L.K, Ts = sT[s], Tps = tpad(Ts), mode = S.pf_fq;
        const unsigned char* sel = nullptr;
        if (mode == 3 || mode == 4) {
            CK(cudaMemsetAsync(F.camax, 0, FQ_KMAX * 4, G.s));
            k_fq_amax<<<Ts, 256, 0, G.s>>>(xq, dx, K, Tps, F.camax);
            k_fq_topn<<<1, 1024, 0, G.s>>>(F.camax, K, S.pf_fq_n, F.sel);
            sel = F.sel;
        }
        if (mode == 8) {
            CK(cudaMemsetAsync(F.camax, 0, FQ_KMAX * 4, G.s));
            k_fq_amax<<<Ts, 256, 0, G.s>>>(xq, dx, K, Tps, F.camax);
            k_fq_smooth<<<(K + 255) / 256, 256, 0, G.s>>>(F.camax, K);
            sel = (const unsigned char*)F.camax;
        }
        if ((mode == 5 || mode == 6 || mode == 9) && (S.pf_fq_n < 32 || K % S.pf_fq_n)) throw std::runtime_error("pf_fq_n must divide K");
        k_fq_apply<<<Ts, 256, K * 4, G.s>>>(xq, dx, K, Tps, mode, S.pf_fq_a / 10.f, S.pf_fq_n, sel, F.uni, F.st + type * 5);
        k_fq_uni<<<1, 256, 0, G.s>>>(F.uni, mode == 8 ? nullptr : sel, K, Ts, F.st + type * 5);
        ck_launch("fq");
    }
    Q8Out q8out(int g, int s, int K) {
        PfGpu& B = P->G[g];
        Q8Out q;
        q.xq = B.xq; q.K = K; q.Tp = tpad(sT[s]);
        q.xs = g8 ? nullptr : B.xs; q.xsum = g8 ? nullptr : B.xsum; q.dx = g8 ? B.dx : nullptr;
        q.ga = g8 ? ga : 32;
        return q;
    }
    // fused producers write 32-element q8 groups (gemm.cuh layout or gemm8 GA 32); other GA use the quant kernels
    bool fused() const { return c->tps->pf_fuse && !i4 && !g17 && (!g8 || ga == 32 || ga == 64); }
    // K5 GEMMs need int8 activations even in i4 mode
    void qg(int g, int s, const tp::FW& W, const float* x, int K, float* y, int ldy) {
        const bool save = i4;
        if (W.L.fmt == gemv::FAST_K5) i4 = false;
        quant(g, s, x, K, rotw(W));
        gemm(g, s, W, y, ldy);
        i4 = save;
    }
    // after sub s's K-split GEMM (both GPUs): copy its partial rows to the peer's rx slot on the copy streams
    void send(int s) {
        tp::State& S = *c->tps;
        const int sl = ar & 1;
        const size_t off = (size_t)st0[s] * D, bytes = (size_t)sT[s] * D * 4;
        for (int g = 0; g < 2; g++) {
            CK(cudaSetDevice(g));
            PfGpu& B = P->G[g];
            if (p2p_part) {  // the GEMM epilogue already wrote the peer's rx slot: ordering only
                CK(cudaEventRecord(B.sent[s][sl], S.G[g].s));
                continue;
            }
            const int nch = (ar16 && chunked[g][s]) ? arc_chunks(s) : 1;
            if (nch > 1) {  // pf_arc: copy each chunk as soon as its GEMM launch is done
                const int cs = chunk_rows(s);
                for (int ci = 0; ci < nch; ci++) {
                    const int r0 = ci * cs, rows = std::min(cs, sT[s] - r0);
                    const size_t o2 = off + (size_t)r0 * D;
                    CK(cudaStreamWaitEvent(B.sc, B.ch_ev[s][ci], 0));
                    CK(cudaMemcpyPeerAsync((__half*)P->G[1 - g].rx[sl] + o2, 1 - g, (const __half*)B.part + o2, g,
                                           (size_t)rows * D * 2, B.sc));
                }
            } else {
            CK(cudaEventRecord(B.part_ev[s], S.G[g].s));
            CK(cudaStreamWaitEvent(B.sc, B.part_ev[s], 0));
            if (ar16)
                CK(cudaMemcpyPeerAsync((__half*)P->G[1 - g].rx[sl] + off, 1 - g, (const __half*)B.part + off, g, bytes / 2, B.sc));
            else
                CK(cudaMemcpyPeerAsync(P->G[1 - g].rx[sl] + off, 1 - g, B.part + off, g, bytes, B.sc));
            }
            CK(cudaEventRecord(B.sent[s][sl], B.sc));
        }
    }
    // before consuming AR (ar - 1) for sub s on GPU g: the peer's copy into our rx and our own copy out of part
    void recv(int g, int s) {
        tp::State& S = *c->tps;
        const int sl = (ar - 1) & 1;
        CK(cudaStreamWaitEvent(S.G[g].s, P->G[1 - g].sent[s][sl], 0));
        CK(cudaStreamWaitEvent(S.G[g].s, P->G[g].sent[s][sl], 0));
    }
    // q8: fused path, quantize into the GEMM layout (xn written only when want_xn)
    void add_norm(int g, int s, const float* w, bool add, bool q8 = false, bool want_xn = true, bool rq = false) {
        tp::Gpu& G = c->tps->G[g];
        PfGpu& B = P->G[g];
        if (add) recv(g, s);
        const int sl = (ar - 1) & 1;
        const size_t off = (size_t)st0[s] * D;
        if (q8 && rq) {
            if (ar16)
                k_pf_add_norm_rot<__half><<<sT[s], 256, D * 4, G.s>>>(B.h + off, (const __half*)B.part + off,
                                                                      (const __half*)B.rx[sl] + off, g, add ? 1 : 0, w,
                                                                      want_xn ? B.xn + off : nullptr, B.xq, B.dxt,
                                                                      rgb ? B.dsr : nullptr, tpad(sT[s]));
            else
                k_pf_add_norm_rot<float><<<sT[s], 256, D * 4, G.s>>>(B.h + off, B.part + off, B.rx[sl] + off, g, add ? 1 : 0,
                                                                     w, want_xn ? B.xn + off : nullptr, B.xq, B.dxt,
                                                                     rgb ? B.dsr : nullptr, tpad(sT[s]));
            // padding rows (tokens Ts..Tp-1) keep stale activations: gemm17 never writes their outputs
        } else if (q8)
            if (ar16)
                k_pf_add_norm_q8<__half><<<sT[s], 256, 0, G.s>>>(B.h + off, (const __half*)B.part + off, (const __half*)B.rx[sl] + off,
                                                                 g, add ? 1 : 0, w, want_xn ? B.xn + off : nullptr, q8out(g, s, D));
            else
                k_pf_add_norm_q8<float><<<sT[s], 256, 0, G.s>>>(B.h + off, B.part + off, B.rx[sl] + off, g, add ? 1 : 0, w,
                                                                want_xn ? B.xn + off : nullptr, q8out(g, s, D));
        else
            if (ar16)
                k_pf_add_norm<__half><<<sT[s], 256, 0, G.s>>>(B.h + off, (const __half*)B.part + off, (const __half*)B.rx[sl] + off,
                                                              g, add ? 1 : 0, w, B.xn + off);
            else
                k_pf_add_norm<float><<<sT[s], 256, 0, G.s>>>(B.h + off, B.part + off, B.rx[sl] + off, g, add ? 1 : 0, w,
                                                             B.xn + off);
        ck_launch("add_norm");
        mark(g, q8 ? "ar_wait+add_norm_q8" : "ar_wait+add_norm");
    }

    void mixer(int g, int il, int s, float theta_scale) {
        tp::State& S = *c->tps;
        tp::Gpu& G = S.G[g];
        tp::Layer& L = G.L[il];
        PfGpu& B = P->G[g];
        const int t0 = st0[s], Ts = sT[s], ps = p0 + t0;
        const bool rin = rotw(L.attn ? L.qkv_a : L.qkvz);  // input projection on R512
        const bool fu = fused(), fq8 = fu || rin;
        const bool abin = rin && !L.attn && S.pf_abq && L.ab8r;  // alpha/beta rows appended to the qkvz GEMM
        add_norm(g, s, L.attn_norm, il > 0, fq8, !L.attn && !abin, rin);
        const int qld = abin ? 8448 : 8192;
        if (!L.attn) {
            if (fq8) gemm(g, s, L.qkvz, B.y, qld);
            else qg(g, s, L.qkvz, B.xn, D, B.y, 8192);
            if (abin) {
            } else if (false) {  // (K-sliced rotated ab GEMM, superseded by the appended rows)
                for (int kz = 0; kz < AB_KS; kz++) {  // K-slices into the consumer's partial-sum slices
                    g16::Args aa;
                    aa.w8 = L.ab8r + kz * (D / AB_KS); aa.xq = B.xq + kz * (D / AB_KS); aa.kstride = D;
                    aa.invs = L.ab8i; aa.dx = B.dxt;
                    aa.y = B.yab + (size_t)kz * P->cap * 48 + (size_t)t0 * 48; aa.ldy = 48; aa.nvalid = 48;
                    aa.N = 256; aa.K = D / AB_KS; aa.T = Ts; aa.Tp = tpad(Ts);
                    CK((g16::launch17_t<0, gemv::FAST_P4, 4, 0, 0, 1>(aa, G.s)));
                }
            } else {
                k_pf_ab<<<dim3((Ts + 127) / 128, AB_KS), 128, 0, G.s>>>(B.xn + (size_t)t0 * D, L.ab, Ts, B.yab + (size_t)t0 * 48,
                                                                       P->cap * 48);
            }
            mark(g, "ab");
            k_pf_conv<<<dim3(Ts, 40), 128, 0, G.s>>>(B.y + (size_t)t0 * qld, qld, L.conv_ring, L.conv_w, ps,
                                                     B.qkv + (size_t)t0 * 5120);
            k_pf_ring<<<20, 256, 0, G.s>>>(B.y + (size_t)t0 * qld, qld, Ts, ps, L.conv_ring);
            mark(g, "conv");
            {
                static int carve[2] = {0, 0};
                if (!carve[g]) {
                    CK(cudaFuncSetAttribute(k_pf_gdn<0>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
                    CK(cudaFuncSetAttribute(k_pf_gdn<1>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
                    CK(cudaFuncSetAttribute(k_pf_gdnc<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, gdnc::BYTES));
                    CK(cudaFuncSetAttribute(k_pf_gdnc<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, gdnc::BYTES));
                    CK(cudaFuncSetAttribute(k_pf_gdnp, cudaFuncAttributeMaxDynamicSharedMemorySize, gdnc::BYTES));
                    carve[g] = 1;
                }
            }
            if (S.pf_gdnc && S.pf_gdnc_chk && !chk_done[g]) {  // one-time check of the chunked scan vs the sequential one
                chk_done[g] = 1;
                const size_t sn = (size_t)24 * 128 * 128, on = (size_t)Ts * 3072;
                float *s1, *o1;
                unsigned* dd;
                CK(cudaMallocAsync(&s1, sn * 4, G.s));
                CK(cudaMallocAsync(&o1, on * 4, G.s));
                CK(cudaMallocAsync(&dd, 16, G.s));
                CK(cudaMemsetAsync(dd, 0, 16, G.s));
                CK(cudaMemcpyAsync(s1, L.S, sn * 4, cudaMemcpyDeviceToDevice, G.s));
                k_pf_gdn<0><<<96, 256, 0, G.s>>>(B.qkv + (size_t)t0 * 5120, B.yab + (size_t)t0 * 48, P->cap * 48, Ts,
                                                 L.ssm_a, L.ssm_dt, s1, o1);
                k_pf_gdnc<0><<<24, 256, gdnc::BYTES, G.s>>>(B.qkv + (size_t)t0 * 5120, B.yab + (size_t)t0 * 48, P->cap * 48,
                                                            Ts, L.ssm_a, L.ssm_dt, L.S, B.o + (size_t)t0 * 3072, nullptr);
                k_maxdiff<<<64, 256, 0, G.s>>>(B.o + (size_t)t0 * 3072, o1, on, dd);
                k_maxdiff<<<64, 256, 0, G.s>>>(L.S, s1, sn, dd + 2);
                unsigned hd[4];
                CK(cudaMemcpyAsync(hd, dd, 16, cudaMemcpyDeviceToHost, G.s));
                CK(cudaStreamSynchronize(G.s));
                float f[4];
                memcpy(f, hd, 16);
                char buf[200];
                snprintf(buf, sizeof buf, "{\"gpu\": %d, \"T\": %d, \"o_maxerr\": %.3e, \"o_max\": %.3e, \"S_maxerr\": %.3e, \"S_max\": %.3e}",
                         g, Ts, f[0], f[1], f[2], f[3]);
                if (S.pf_gdnc_json.size() < 1500) S.pf_gdnc_json += (S.pf_gdnc_json.empty() ? "" : ", ") + std::string(buf);
                CK(cudaFreeAsync(s1, G.s));
                CK(cudaFreeAsync(o1, G.s));
                CK(cudaFreeAsync(dd, G.s));
                mark(g, "gdn_scan");
            } else if (S.pf_gdnc) {
                const float* yb = abin ? B.y + (size_t)t0 * qld + 8192 : B.yab + (size_t)t0 * 48;
                const int yss = abin ? -qld : P->cap * 48;
                if (S.pf_gdnc == 2) {  // state-independent part for all chunks in parallel, then the sequential scan
                    k_pf_gdnp<<<dim3(24, (Ts + gdnc::C - 1) / gdnc::C), 256, gdnc::BYTES, G.s>>>(
                        B.qkv + (size_t)t0 * 5120, yb, yss, Ts, L.ssm_a, L.ssm_dt, B.gscr);
                    k_pf_gdnc<1><<<24, 256, gdnc::BYTES, G.s>>>(B.qkv + (size_t)t0 * 5120, yb, yss, Ts, L.ssm_a, L.ssm_dt,
                                                                L.S, B.o + (size_t)t0 * 3072, B.gscr);
                } else {
                    k_pf_gdnc<0><<<24, 256, gdnc::BYTES, G.s>>>(B.qkv + (size_t)t0 * 5120, yb, yss, Ts, L.ssm_a, L.ssm_dt,
                                                                L.S, B.o + (size_t)t0 * 3072, nullptr);
                }
                mark(g, "gdn_scan");
            } else {
            auto gk = S.pf_gdn2 ? k_pf_gdn<1> : k_pf_gdn<0>;
            gk<<<96, 256, 0, G.s>>>(B.qkv + (size_t)t0 * 5120, B.yab + (size_t)t0 * 48, P->cap * 48, Ts, L.ssm_a, L.ssm_dt,
                                          L.S, B.o + (size_t)t0 * 3072);
            mark(g, "gdn_scan");
            }
            if (rotw(L.ssm_out)) {
                k_pf_gnorm_rot<<<Ts, 256, 3072 * 4, G.s>>>(B.o + (size_t)t0 * 3072, B.y + (size_t)t0 * qld, qld,
                                                           L.ssm_norm, B.xq, B.dxt, rgb ? B.dsr : nullptr, tpad(Ts));
                ck_launch("gnorm_rot");
                mark(g, "gnorm_q8");
                gemm(g, s, L.ssm_out, B.part, D);
            } else if (fu) {
                k_pf_gnorm_q8<<<dim3(Ts, 24), 128, 0, G.s>>>(B.o + (size_t)t0 * 3072, B.y + (size_t)t0 * qld, qld,
                                                              L.ssm_norm, q8out(g, s, 3072));
                ck_launch("deltanet");
                mark(g, "gnorm_q8");
                gemm(g, s, L.ssm_out, B.part, D);
            } else {
                k_pf_gnorm<<<dim3(Ts, 24), 128, 0, G.s>>>(B.o + (size_t)t0 * 3072, B.y + (size_t)t0 * qld, qld,
                                                           L.ssm_norm, B.g32 + (size_t)t0 * 3072);
                ck_launch("deltanet");
                mark(g, "gnorm");
                qg(g, s, L.ssm_out, B.g32, 3072, B.part, D);
            }
        } else {
            if (fq8) gemm(g, s, L.qkv_a, B.y, 7168);
            else qg(g, s, L.qkv_a, B.xn, D, B.y, 7168);
            k_pf_attn_prep<<<dim3(Ts, 14), 256, 0, G.s>>>(B.y + (size_t)t0 * 7168, 7168, L.q_norm, L.k_norm,
                                                          B.qa + (size_t)t0 * 3072, (__half*)L.kc, (__half*)L.vc,
                                                          S.max_ctx, ps, theta_scale);
            mark(g, "attn_prep");
            if (S.pf_fa)
                k_pf_fa<<<dim3((Ts + 15) / 16, 2), 192, 0, G.s>>>(B.qa + (size_t)t0 * 3072, (const __half*)L.kc,
                                                                  (const __half*)L.vc, B.y + (size_t)t0 * 7168, 7168,
                                                                  Ts, S.max_ctx, ps, B.g32 + (size_t)t0 * 3072);
            else
                k_pf_attn<<<dim3(Ts, 2), 192, 0, G.s>>>(B.qa + (size_t)t0 * 3072, (const __half*)L.kc,
                                                        (const __half*)L.vc, B.y + (size_t)t0 * 7168, 7168, S.max_ctx,
                                                        ps, B.g32 + (size_t)t0 * 3072);
            ck_launch("attention");
            mark(g, "attn");
            qg(g, s, L.wo, B.g32, 3072, B.part, D);
        }
    }
    void ffn(int g, int il, int s) {
        tp::State& S = *c->tps;
        tp::Gpu& G = S.G[g];
        tp::Layer& L = G.L[il];
        PfGpu& B = P->G[g];
        const int t0 = st0[s], Ts = sT[s];
        const bool fu = fused();
        const bool rg = rotw(L.gateup), rd = rotw(L.down);
        if (rg) {
            add_norm(g, s, L.post_norm, true, true, false, true);
            gemm(g, s, L.gateup, nullptr, 0, true);  // silu(gate) * up -> fp16 rows of g32
            const int Tps = tpad(Ts);
            if (S.pf_h16) {
                if (rd) rot::quant_rot<__half>((const __half*)B.g32 + (size_t)t0 * 8704, 8704, Ts, Tps, 8704, B.xq, B.dxt, G.s,
                                               rgb ? B.dsr : nullptr);
                else k_q8_64_half<<<dim3(Tps, 8704 / 64 / 8), 256, 0, G.s>>>((const __half*)B.g32 + (size_t)t0 * 8704, Ts, Tps, 8704, B.xq, B.dx);
            } else {
                if (rd) rot::quant_rot<float>(B.g32 + (size_t)t0 * 8704, 8704, Ts, Tps, 8704, B.xq, B.dxt, G.s, rgb ? B.dsr : nullptr);
                else gemm8::quant8(B.g32 + (size_t)t0 * 8704, 8704, Ts, Tps, 8704, B.xq, B.dx, G.s, 64);
            }
            ck_launch("quant_down");
            mark(g, "quant");
            gemm(g, s, L.down, B.part, D);
            return;
        }
        add_norm(g, s, L.post_norm, true, fu, false);
        if (fu && g8 && ga == 64 && S.pf_silu && L.gateup.L.fmt == gemv::FAST_P4 && !rd) {
            gemm(g, s, L.gateup, nullptr, 0, true);
            gemm(g, s, L.down, B.part, D, false, true);
            return;
        }
        if (fu && !rd) {
            gemm(g, s, L.gateup, B.y, 17408);
            k_pf_silu_q8<<<dim3(Ts, 34), 256, 0, G.s>>>(B.y + (size_t)t0 * 17408, 17408, q8out(g, s, 8704));
            ck_launch("silu_q8");
            mark(g, "silu_q8");
            gemm(g, s, L.down, B.part, D);
            return;
        }
        if (fu) gemm(g, s, L.gateup, B.y, 17408);  // GA64 activations from the fused add_norm
        else qg(g, s, L.gateup, B.xn, D, B.y, 17408);
        k_pf_silu<<<dim3(Ts, 34), 256, 0, G.s>>>(B.y + (size_t)t0 * 17408, 17408, B.g32 + (size_t)t0 * 8704);
        ck_launch("silu");
        mark(g, "silu");
        qg(g, s, L.down, B.g32, 8704, B.part, D);
    }

    void run(const int32_t* ids) {
        tp::State& S = *c->tps;
        nsub = (S.pf_nsub >= 2 && T >= std::max(256, S.pf_nsub_min)) ? 2 : 1;
        const int al = g8 ? 256 : 128;  // sub-batch 0 rows: a multiple of the GEMM token tile
        const int half = nsub == 2 ? ((T / 2 + al - 1) / al) * al : T;
        st0[0] = 0; sT[0] = std::min(T, half);
        st0[1] = sT[0]; sT[1] = T - sT[0];
        if (nsub == 2 && sT[1] <= 0) nsub = 1;
        for (int g = 0; g < 2; g++) {
            CK(cudaSetDevice(g));
            tp::Gpu& G = S.G[g];
            CK(cudaMemcpyAsync(P->G[g].ids, ids, (size_t)T * 4, cudaMemcpyHostToDevice, G.s));
            mark(g, "start");
            k_pf_embed<<<dim3(T, 20), 256, 0, G.s>>>(G.embd, P->G[g].ids, P->G[g].h);
            ck_launch("embed");
            mark(g, "embed");
        }
        const float theta_scale = powf(hp::ROPE_BASE, -2.0f / hp::NROT);
        for (int il = 0; il < 64; il++) {
            for (int s = 0; s < nsub; s++) {
                for (int g = 0; g < 2; g++) { CK(cudaSetDevice(g)); mixer(g, il, s, theta_scale); }
                send(s);
            }
            ar++;
            for (int s = 0; s < nsub; s++) {
                for (int g = 0; g < 2; g++) { CK(cudaSetDevice(g)); ffn(g, il, s); }
                send(s);
            }
            ar++;
        }
        // consume the last AR (keeps the part/rx reuse ordered for the next ubatch)
        for (int s = 0; s < nsub; s++)
            for (int g = 0; g < 2; g++) {
                CK(cudaSetDevice(g));
                add_norm(g, s, S.G[g].output_norm, true);
            }
    }
};

}  // namespace

// batched prefill of ids[0..n) at positions c->pos.. (all but the last token through the batch path)
int tp_prefill_batched(t4q_ctx* c, const int32_t* ids, int n, void (*run_last_step)(t4q_ctx*)) {
    tp::State& S = *c->tps;
    if (c->pos + n > S.max_ctx) throw std::runtime_error("context full");
    Pf* P = pf_get(c);
    pf_rot_prep(c);
    auto t0 = Clock::now();
    const bool head = S.pf_head != 0;
    const int nb = head ? n : n - 1;  // pf_head 0: the last token goes through the decode step
    int last_b0 = 0;
    std::vector<cudaEvent_t> ev;
    std::vector<const char*> evn;
    std::vector<std::pair<std::string, std::pair<double, int>>> prof;
    auto acc = [&](const std::string& k, double ms) {
        for (auto& p : prof)
            if (p.first == k) { p.second.first += ms; p.second.second++; return; }
        prof.push_back({k, {ms, 1}});
    };
    S.pf_fq_json.clear();
    for (int g = 0; g < 2; g++)
        if (P->fq[g].st) { CK(cudaSetDevice(g)); CK(cudaMemsetAsync(P->fq[g].st, 0, 6 * 5 * 8, S.G[g].s)); }
    for (int b0 = 0; b0 < nb; b0 += S.pf_ub) {
        PfRun R{c, P};
        R.T = std::min(S.pf_ub, nb - b0);
        last_b0 = b0;
        R.p0 = c->pos + b0;
        R.g8 = S.pf_g8 != 0;
        R.ar16 = R.g8 && S.pf_ar16;
        R.ga = S.pf_ga;
        R.g17 = R.g8 && S.pf_g17;
        R.rot = R.g8 && S.pf_rot && R.T >= S.pf_rot_min;
        R.rmask = S.pf_rot_mask;
        R.rgb = R.rot && S.pf_rgb;
        R.emax = S.pf_emax;
        R.i4 = S.pf_i4 != 0 && !R.g8;
        if (S.pf_prof) { R.ev = &ev; R.evn = &evn; }
        R.run(ids + b0);
        if (S.pf_keep_h) {
            std::vector<float>& hv = c->dumps["pf_h"];
            if (b0 == 0) hv.clear();
            hv.resize((size_t)(b0 + R.T) * D);
            CK(cudaSetDevice(0));
            CK(cudaStreamSynchronize(S.G[0].s));
            CK(cudaMemcpy(hv.data() + (size_t)b0 * D, P->G[0].h, (size_t)R.T * D * 4, cudaMemcpyDeviceToHost));
        }
        if (S.pf_prof) {
            CK(cudaSetDevice(0));
            CK(cudaStreamSynchronize(S.G[0].s));
            for (size_t i = 1; i < ev.size(); i++) {
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, ev[i - 1], ev[i]));
                acc(evn[i], ms);
            }
            for (auto e : ev) cudaEventDestroy(e);
            ev.clear();
            evn.clear();
        }
    }
    if (S.pf_prof) {
        std::string js = "{";
        char b[160];
        for (size_t i = 0; i < prof.size(); i++) {
            snprintf(b, sizeof b, "%s\"%s\": [%.2f, %d]", i ? ", " : "", prof[i].first.c_str(), prof[i].second.first,
                     prof[i].second.second);
            js += b;
        }
        S.pf_json = js + "}";
    }
    for (int g = 0; g < 2; g++) {
        CK(cudaSetDevice(g));
        CK(cudaStreamSynchronize(S.G[g].s));
    }
    const double t_batch = secs(t0);
    if (S.pf_fq && P->fq[0].st) {
        unsigned long long h[30];
        CK(cudaSetDevice(0));
        CK(cudaMemcpy(h, P->fq[0].st, sizeof h, cudaMemcpyDeviceToHost));
        static const char* nm[6] = {"qkvz", "attn_qkv", "gateup", "down", "ssm_out", "attn_out"};
        std::string js = "{";
        char b[200];
        for (int i = 0; i < 6; i++) {
            if (!h[i * 5]) continue;
            snprintf(b, sizeof b, "%s\"%s\": {\"calls\": %llu, \"tokens\": %llu, \"resid\": %llu, \"union_mean\": %.1f, \"union_max\": %llu}",
                     js.size() > 1 ? ", " : "", nm[i], h[i * 5], h[i * 5 + 1], h[i * 5 + 2],
                     (double)h[i * 5 + 3] / h[i * 5], h[i * 5 + 4]);
            js += b;
        }
        S.pf_fq_json = js + "}";
    }
    // decode step (or only its head) for the last token at position c->pos + n - 1
    const int pos_last = c->pos + n - 1;
    for (int g = 0; g < 2; g++) {
        CK(cudaSetDevice(g));
        CK(cudaMemcpy(&S.G[g].st->pos, &pos_last, 4, cudaMemcpyHostToDevice));
    }
    if (head) {
        // the batch already wrote KV / conv ring / DeltaNet state for every prompt token and left the final residual
        // of the last token in row (n - 1 - last_b0) of h: output norm + q8 (decode format), lm_head shard, argmax
        // exchange (advances StepState pos / step / token and the ring exactly like a decode step's head)
        const size_t lrow = (size_t)(n - 1 - last_b0);
        for (int g = 0; g < 2; g++) {
            CK(cudaSetDevice(g));
            tp::Gpu& G = S.G[g];
            tp::ar_norm(P->G[g].h + lrow * D, G.hb[1], nullptr, nullptr, nullptr, G.st, 127, G.output_norm, G.xn,
                        G.xq, G.xm, G.s);
            tp::gemv(G.lm, G.xq, G.xm, G.logits, G.s);
            tp::argmax_step(G.logits, 124160, 124160 * g, G.apart, S.p2p ? G.amb : G.hamb,
                            S.p2p ? G.flag + 2 : G.hflag + 2, G.peer_amb, G.peer_flag + 2, G.st,
                            g == 0 ? S.d_ring : nullptr, G.s);
            ck_launch("pf_head");
        }
        c->pos = pos_last + 1;
        c->steps++;
    } else {
        c->pos = pos_last;
        run_last_step(c);
    }
    for (int g = 0; g < 2; g++) {
        CK(cudaSetDevice(g));
        CK(cudaStreamSynchronize(S.G[g].s));
    }
    S.pf_last_batch_s = t_batch;
    S.pf_last_total_s = secs(t0);
    S.pf_last_n = n;
    S.pf_h_pos0 = pos_last - (n - 1) + last_b0;  // P->G[g].h rows: final residuals of positions pos0 .. pos0 + nrows - 1
    S.pf_h_n = nb - last_b0;
    return 0;
}

// final residual rows of the last batched-prefill ubatch on GPU g (speculative decoding's MTP prompt catch-up)
const float* tp_prefill_hrows(t4q_ctx* c, int g) {
    Pf* P = (Pf*)c->tps->pf;
    return P ? P->G[g].h : nullptr;
}

#include "tp_batch.cuh"
