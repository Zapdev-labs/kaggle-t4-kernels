// TP decode kernels: fast GEMV with fused AR publish / fp32 segment rows, and the small fused kernels between GEMVs.
#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "tp_kernels.h"

using namespace t4q::gemv;

namespace tp {
namespace {

constexpr unsigned long long WATCHDOG_NS = 4000000000ull;  // 4 s: a broken run must not hang the Kaggle session

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ unsigned ld_vol_u32(const unsigned* p) {
    unsigned v;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ float ld_vol_f32(const float* p) {
    float v;
    asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ void st_vol_u32(unsigned* p, unsigned v) {
    asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned epoch_of(const StepState* st, int idx) {
    return *(volatile const uint32_t*)&st->step * (unsigned)NAR + (unsigned)idx + 1u;
}
// spin until flag >= e (wrap-safe); returns false on watchdog
__device__ __forceinline__ bool wait_flag(const unsigned* flag, unsigned e) {
    const unsigned long long t0 = gtimer();
    unsigned spins = 0;
    while ((int)(ld_vol_u32(flag) - e) < 0) {
        if ((++spins & 1023u) == 0 && gtimer() - t0 > WATCHDOG_NS) return false;
    }
    return true;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// block sum, fixed order; red must hold blockDim/32 floats; result valid in all threads
__device__ __forceinline__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < nw; i++) t += red[i];
    return t;
}
// q8_1 quantization of one 32-group held one value per lane (same math as gemv.cuh quantize_q8_kernel)
__device__ __forceinline__ void quant_warp(float v, int8_t* xq_i, int2* xm_g) {
    const int lane = threadIdx.x & 31;
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.f;
    const int q = amax == 0.f ? 0 : (int)roundf(v / d);
    *xq_i = (int8_t)q;
    int s = q;
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const int s1 = __shfl_sync(0xffffffffu, s, 16);
    if (lane == 0) *xm_g = make_int2(__float_as_int(d), (int)((unsigned)(s & 0xffff) | ((unsigned)s1 << 16)));
}
__device__ __forceinline__ float h2f_u16(uint16_t b) { return __half2float(__ushort_as_half(b)); }

// extra block b of nb: prefetch the ranges into L2, one instruction per 32 B sector (Turing L2 fills sectors; one per
// 128 B line in M4 v4 showed no gain), then return
__device__ __forceinline__ void do_prefetch(const Pf& pf, int b, int nb) {
    const unsigned stride = (unsigned)nb * blockDim.x * 32u;
#pragma unroll
    for (int r = 0; r < 4; r++) {
        if (!pf.p[r]) continue;
        for (unsigned off = ((unsigned)b * blockDim.x + threadIdx.x) * 32u; off < pf.n[r]; off += stride)
            asm volatile("prefetch.global.L2 [%0];" ::"l"(pf.p[r] + off));
    }
}
// consumer-side publish: copy this GPU's partial (own slot, written by the previous kernel) to the peer's mailbox with
// coalesced float4 stores, then one system fence + flag (the M0 two-level pattern). Whole block participates.
__device__ __forceinline__ void publish_partial(const float* own_slot, float* peer_slot, unsigned* peer_flag_slot,
                                                unsigned epoch) {
    if (peer_slot) {  // nullptr: the producer GEMV already wrote the rows (arpub 2); only fence + flag here
        const float4* src = (const float4*)own_slot;
        float4* dst = (float4*)peer_slot;
        for (int i = threadIdx.x; i < 1280; i += blockDim.x) dst[i] = src[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(peer_flag_slot), "r"(epoch) : "memory");
    }
}

#define T4Q_PF_BLOCKS(NWORK)                                       \
    if ((int)blockIdx.x >= (NWORK)) {                              \
        do_prefetch(pf, blockIdx.x - (NWORK), gridDim.x - (NWORK)); \
        return;                                                    \
    }

// ------------------------------------------------------------------------------------------------ GEMV
__device__ __forceinline__ float4 ld_vol_f4(const float4* p) {
    float4 v;
    asm volatile("ld.volatile.global.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
}
// q8 quantization of one 32-group held by one thread (d = amax/127, q = round(v/d)) into the shared-memory x planes
__device__ __forceinline__ void quant_group(const float* v, int g, int8_t* s_lo, int8_t* s_hi, int2* s_mt) {
    float amax = 0.f;
#pragma unroll
    for (int e = 0; e < 32; e++) amax = fmaxf(amax, fabsf(v[e]));
    const float d = amax / 127.f;
    int qw[8];
    int s0 = 0, s1 = 0;
#pragma unroll
    for (int w = 0; w < 8; w++) {
        unsigned word = 0;
#pragma unroll
        for (int b = 0; b < 4; b++) {
            const int q = amax == 0.f ? 0 : (int)roundf(v[4 * w + b] / d);
            word |= (unsigned)(q & 0xff) << (8 * b);
            if (w < 4) s0 += q;
            else s1 += q;
        }
        qw[w] = (int)word;
    }
    ((int4*)s_lo)[g] = make_int4(qw[0], qw[1], qw[2], qw[3]);
    ((int4*)s_hi)[g] = make_int4(qw[4], qw[5], qw[6], qw[7]);
    s_mt[g] = make_int2(__float_as_int(d), (int)((unsigned)(s0 & 0xffff) | ((unsigned)s1 << 16)));
}

// q8 quantization of one 32-group (one value per lane, element i) into the shared-memory x planes
__device__ __forceinline__ void quant_smem(float v, int i, int8_t* s_lo, int8_t* s_hi, int2* s_mt) {
    const int lane = threadIdx.x & 31;
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.f;
    const int q = amax == 0.f ? 0 : (int)roundf(v / d);
    const int g = i >> 5;
    if (lane < 16) s_lo[g * 16 + lane] = (int8_t)q;
    else s_hi[g * 16 + lane - 16] = (int8_t)q;
    int s = q;
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const int s1 = __shfl_sync(0xffffffffu, s, 16);
    if (lane == 0) s_mt[g] = make_int2(__float_as_int(d), (int)((unsigned)(s & 0xffff) | ((unsigned)s1 << 16)));
}

int g_max_blocks = 80;

template <int FMT, int RPL, int NCH, bool AR, bool SEG, int PRO, bool SQ = false>
__global__ void __launch_bounds__(256, 2) k_gemv(const GemvArgs a, const ArArgs ar, const SegArgs sg, const ProArgs pa,
                                                 int tpw) {
    constexpr int D = 2, M = 1;
    constexpr int CVT = (FMT == FAST_P4 || FMT == FAST_Q8) ? 1 : 0;
    constexpr int NB = NCH * 16;
    constexpr int K = NCH * 512;
    const int tid = threadIdx.x, lane = tid & 31, h = lane >> 4, j = lane & 15, wib = tid >> 5;
    const int warp = blockIdx.x * 8 + wib;
    const int ntot = a.ntiles + (SEG ? sg.nrows : 0);
    const int tbeg = warp * tpw, tend = min(ntot, tbeg + tpw);
    // AR: a block owns the contiguous rows of its 8 * tpw tiles (tpw <= 2); they are staged here and sent to the peer
    // as coalesced float4 PCIe writes (4-byte scattered remote stores cost ~60 us per 20 KB in M2 v1).
    __shared__ float sy[AR ? 8 * 2 * 2 * RPL : 1];
    __shared__ __align__(16) int8_t s_lo[PRO ? NB * 16 : 16];
    __shared__ __align__(16) int8_t s_hi[PRO ? NB * 16 : 16];
    __shared__ int2 s_mt[PRO ? NB : 1];
    __shared__ __align__(16) float s_xf[(PRO == PRO_ARNORM || PRO == PRO_GNORM || (PRO == PRO_LEADER && SEG)) ? K : 4];
    __shared__ float red[128];
    __shared__ int s_flag;
    __shared__ float sa[SQ ? 8 * 8 * RPL : 1];  // SQ: silu(g) * u of the block's 8 * tpw * RPL outputs (tpw <= 8)

    WChunk<FMT, RPL> w[D];
    const bool pre = PRO != PRO_NONE && tbeg < tend && tbeg < a.ntiles;
    if (pre) {
#pragma unroll
        for (int c = 0; c < D; ++c)
            if (c < NCH) load_chunk<FMT, RPL, NCH>(w[c], a, tbeg, c, lane);
    }
    if (PRO == PRO_ARNORM) {
        static_assert(PRO != PRO_ARNORM || K == 5120, "ARNORM prologue needs K = 5120");
        const bool add = pa.add != 0;
        if (pa.flag) {
            if (tid == 0) {
                s_flag = wait_flag(pa.flag, epoch_of(pa.st, pa.idx));
                if (!s_flag) pa.st->err = 1000 + pa.idx;
            }
            __syncthreads();
        }
        float4* sx4 = (float4*)s_xf;
        float ss = 0.f;
        float4 xv[5];
#pragma unroll
        for (int k = 0; k < 5; k++) xv[k] = ((const float4*)pa.h_in)[tid + 256 * k];
        if (add) {
            float4 ov[5], rv[5];
#pragma unroll
            for (int k = 0; k < 5; k++) {
                ov[k] = ((const float4*)pa.own)[tid + 256 * k];
                rv[k] = ld_vol_f4((const float4*)pa.rx + tid + 256 * k);
            }
#pragma unroll
            for (int k = 0; k < 5; k++) {
                xv[k].x = xv[k].x + (ov[k].x + rv[k].x);
                xv[k].y = xv[k].y + (ov[k].y + rv[k].y);
                xv[k].z = xv[k].z + (ov[k].z + rv[k].z);
                xv[k].w = xv[k].w + (ov[k].w + rv[k].w);
                if (blockIdx.x == 0) ((float4*)pa.h_out)[tid + 256 * k] = xv[k];
            }
        }
#pragma unroll
        for (int k = 0; k < 5; k++) {
            sx4[tid + 256 * k] = xv[k];
            ss += xv[k].x * xv[k].x + xv[k].y * xv[k].y + xv[k].z * xv[k].z + xv[k].w * xv[k].w;
        }
        ss = block_sum(ss, red);  // its barriers also publish s_xf
        const float scale = rsqrtf(ss / 5120.f + 1e-6f);
        if (tid < NB) {
            float v[32];
            const float4* w4 = (const float4*)pa.nw + tid * 8;
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const int ee = e;
                const float4 x = sx4[tid * 8 + ee], wv = __ldg(w4 + ee);
                v[4 * ee] = (x.x * scale) * wv.x;
                v[4 * ee + 1] = (x.y * scale) * wv.y;
                v[4 * ee + 2] = (x.z * scale) * wv.z;
                v[4 * ee + 3] = (x.w * scale) * wv.w;
            }
            quant_group(v, tid, s_lo, s_hi, s_mt);
            if (SEG || blockIdx.x == 0) {
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    const int ee = e;
                    const float4 y = make_float4(v[4 * ee], v[4 * ee + 1], v[4 * ee + 2], v[4 * ee + 3]);
                    if (SEG) sx4[tid * 8 + ee] = y;
                    if (blockIdx.x == 0) ((float4*)pa.xn_out)[tid * 8 + ee] = y;
                }
            }
        }
        __syncthreads();
    } else if (PRO == PRO_LEADER) {
        static_assert(PRO != PRO_LEADER || K == 5120, "LEADER prologue needs K = 5120");
        const unsigned ep = epoch_of(pa.st, pa.idx);
        const unsigned xep = ep + 1u;  // x-ready value: never 0 (idx -1 at step 0 has epoch 0 = the initial flag)
        if (blockIdx.x == 0) {
            const bool add = pa.add != 0;
            if (pa.pub_peer_flag) publish_partial(pa.own, pa.pub_peer_rx, pa.pub_peer_flag, ep);
            if (pa.flag) {
                if (tid == 0) {
                    s_flag = wait_flag(pa.flag, ep);
                    if (!s_flag) pa.st->err = 1000 + pa.idx;
                }
                __syncthreads();
            }
            float* xf = SEG ? s_xf : pa.xn_out;  // fp32 x staging (SEG keeps it in smem for the fp32 rows)
            float4* sx4 = (float4*)xf;
            float ss = 0.f;
            float4 xv[5];
#pragma unroll
            for (int k = 0; k < 5; k++) xv[k] = ((const float4*)pa.h_in)[tid + 256 * k];
            if (add) {
                float4 ov[5], rv[5];
#pragma unroll
                for (int k = 0; k < 5; k++) {
                    ov[k] = ((const float4*)pa.own)[tid + 256 * k];
                    rv[k] = ld_vol_f4((const float4*)pa.rx + tid + 256 * k);
                }
#pragma unroll
                for (int k = 0; k < 5; k++) {
                    xv[k].x = xv[k].x + (ov[k].x + rv[k].x);
                    xv[k].y = xv[k].y + (ov[k].y + rv[k].y);
                    xv[k].z = xv[k].z + (ov[k].z + rv[k].z);
                    xv[k].w = xv[k].w + (ov[k].w + rv[k].w);
                    ((float4*)pa.h_out)[tid + 256 * k] = xv[k];
                }
            }
#pragma unroll
            for (int k = 0; k < 5; k++) {
                sx4[tid + 256 * k] = xv[k];
                ss += xv[k].x * xv[k].x + xv[k].y * xv[k].y + xv[k].z * xv[k].z + xv[k].w * xv[k].w;
            }
            ss = block_sum(ss, red);  // barriers publish xf within the block
            const float scale = rsqrtf(ss / 5120.f + 1e-6f);
            if (tid < NB) {
                float v[32];
                const float4* w4 = (const float4*)pa.nw + tid * 8;
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    const float4 x = sx4[tid * 8 + e], wv = __ldg(w4 + e);
                    v[4 * e] = (x.x * scale) * wv.x;
                    v[4 * e + 1] = (x.y * scale) * wv.y;
                    v[4 * e + 2] = (x.z * scale) * wv.z;
                    v[4 * e + 3] = (x.w * scale) * wv.w;
                }
                quant_group(v, tid, s_lo, s_hi, s_mt);
                ((int4*)pa.gxq)[2 * tid] = ((const int4*)s_lo)[tid];
                ((int4*)pa.gxq)[2 * tid + 1] = ((const int4*)s_hi)[tid];
                pa.gxm[tid] = s_mt[tid];
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    const float4 y = make_float4(v[4 * e], v[4 * e + 1], v[4 * e + 2], v[4 * e + 3]);
                    sx4[tid * 8 + e] = y;
                    if (SEG) ((float4*)pa.xn_out)[tid * 8 + e] = y;
                }
            }
            __threadfence();
            __syncthreads();
            if (tid == 0) asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(pa.xflag), "r"(xep) : "memory");
        } else {
            if (tid == 0) {
                s_flag = wait_flag(pa.xflag, xep);
                if (!s_flag) pa.st->err = 4000 + pa.idx;
            }
            __syncthreads();
            for (int g = tid; g < NB; g += blockDim.x) {  // x from L2 (written during this kernel: bypass L1)
                ((int4*)s_lo)[g] = __ldcg((const int4*)pa.gxq + 2 * g);
                ((int4*)s_hi)[g] = __ldcg((const int4*)pa.gxq + 2 * g + 1);
                s_mt[g] = __ldcg(pa.gxm + g);
            }
            if (SEG)
                for (int i = tid; i < K / 4; i += blockDim.x) ((float4*)s_xf)[i] = __ldcg((const float4*)pa.xn_out + i);
            __syncthreads();
        }
    } else if (PRO == PRO_SILU) {
        for (int g = tid; g < NB; g += blockDim.x) {
            float v[32];
            const float4* g4 = (const float4*)pa.gu + g * 8;
            const float4* u4 = (const float4*)(pa.gu + K) + g * 8;
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                float4 gv[4], uv[4];
#pragma unroll
                for (int e = 0; e < 4; e++) { gv[e] = g4[hh * 4 + e]; uv[e] = u4[hh * 4 + e]; }
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const float* gg = (const float*)&gv[e];
                    const float* uu = (const float*)&uv[e];
#pragma unroll
                    for (int q = 0; q < 4; q++) v[16 * hh + 4 * e + q] = (gg[q] / (1.0f + expf(-gg[q]))) * uu[q];
                }
            }
            quant_group(v, g, s_lo, s_hi, s_mt);
        }
        __syncthreads();
    } else if (PRO == PRO_GNORM) {
        static_assert(PRO != PRO_GNORM || K == 3072, "GNORM prologue needs K = 3072");
        float ov[32];
        if (tid < NB) {
            const float4* o4 = (const float4*)pa.o + tid * 8;
            float sq = 0.f;
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const float4 x = o4[e];
                ov[4 * e] = x.x; ov[4 * e + 1] = x.y; ov[4 * e + 2] = x.z; ov[4 * e + 3] = x.w;
                sq += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
            }
            red[tid] = sq;
        }
        __syncthreads();
        if (tid < NB) {
            const int hb = tid & ~3;
            const float tot = ((red[hb] + red[hb + 1]) + red[hb + 2]) + red[hb + 3];
            const float scale = rsqrtf(tot / 128.0f + 1e-6f);
            const float4* z4 = (const float4*)pa.z + tid * 8;
            const float4* w4 = (const float4*)pa.gw + (tid & 3) * 8;
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const float4 zz = z4[e], wv = __ldg(w4 + e);
                const float* zp = (const float*)&zz;
                const float* wp = (const float*)&wv;
#pragma unroll
                for (int q = 0; q < 4; q++)
                    ov[4 * e + q] = ((ov[4 * e + q] * scale) * wp[q]) * (zp[q] / (1.0f + expf(-zp[q])));
            }
            quant_group(ov, tid, s_lo, s_hi, s_mt);
        }
        __syncthreads();
    }

    for (int tile = tbeg; tile < tend; tile++) {
        if (SEG && tile >= a.ntiles) {
            const int row = tile - a.ntiles;
            const float4* w4 = (const float4*)(sg.w + (size_t)row * 5120);
            const float4* x4 = (PRO == PRO_ARNORM || PRO == PRO_LEADER) ? (const float4*)s_xf : (const float4*)sg.x;
            float acc = 0.f;
#pragma unroll 4
            for (int i = lane; i < 1280; i += 32) {
                const float4 wv = __ldg(w4 + i), x = x4[i];
                acc += wv.x * x.x + wv.y * x.y + wv.z * x.z + wv.w * x.w;
            }
            acc = warp_sum(acc);
            if (lane == 0) sg.y[row] = acc;
            continue;
        }
        float acc[RPL][M];
#pragma unroll
        for (int r = 0; r < RPL; ++r)
#pragma unroll
            for (int c = 0; c < M; ++c) acc[r][c] = 0.f;
        if (!(pre && tile == tbeg)) {
#pragma unroll
            for (int c = 0; c < D; ++c)
                if (c < NCH) load_chunk<FMT, RPL, NCH>(w[c], a, tile, c, lane);
        }
#pragma unroll
        for (int c = 0; c < NCH; ++c) {
            WChunk<FMT, RPL> cur = w[c % D];
            if (c + D < NCH) load_chunk<FMT, RPL, NCH>(w[c % D], a, tile, c + D, lane);
            const int kb = c * 16 + j;
#pragma unroll
            for (int col = 0; col < M; ++col) {
                int4 xl, xh;
                int2 mt;
                if (PRO != PRO_NONE) {
                    xl = ((const int4*)s_lo)[kb];
                    xh = ((const int4*)s_hi)[kb];
                    mt = s_mt[kb];
                } else {
                    const int4* xp = (const int4*)(a.xq + (size_t)col * K + kb * 32);
                    xl = __ldg(xp);
                    xh = __ldg(xp + 1);
                    mt = __ldg(a.xm + (size_t)col * NB + kb);
                }
                const float xd = __int_as_float(mt.x);
                const int s0 = (int)(short)(mt.y & 0xffff), s1 = mt.y >> 16;
                const int moff = FMT == FAST_P4 ? 0x4B400000 - 8 * (s0 + s1) : 0x4B400000;
#pragma unroll
                for (int r = 0; r < RPL; ++r) acc[r][col] += group_dot_r<FMT, CVT, RPL>(cur, r, xl, xh, xd, s0, s1, moff);
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
                if (SQ) {  // half 0 holds gate row r, half 1 the matching up row
                    const float u = __shfl_xor_sync(0xffffffffu, v, 16);
                    if (lane == 0) sa[(tile - blockIdx.x * 8 * tpw) * RPL + r] = (v / (1.0f + expf(-v))) * u;
                }
                if (j == 0 && row < a.N) {
                    a.y[(size_t)col * a.ldy + row] = v;
                    if (AR) sy[(tile - blockIdx.x * 8 * tpw) * 2 * RPL + h * RPL + r] = v;
                }
            }
    }
    if (SQ) {  // the block owns outputs [blockIdx.x * tpw * 32, +tpw * 32): tpw q8 groups
        __syncthreads();
        if (wib < tpw)
            quant_warp(sa[wib * 32 + lane], pa.sq_xq + (blockIdx.x * tpw + wib) * 32 + lane,
                       pa.sq_xm + blockIdx.x * tpw + wib);
    }
    if (AR) {
        __syncthreads();
        const int nrow = 8 * tpw * 2 * RPL;
        bool wrote = false;
        if (ar.fence != -2)
            for (int t = tid; t < nrow / 4; t += blockDim.x) {
                const int row0 = blockIdx.x * nrow + t * 4;
                if (row0 < a.N) {
                    *(float4*)(ar.y_peer + row0) = *(const float4*)(sy + t * 4);
                    wrote = true;
                }
            }
        if (ar.fence < 0) return;  // rows only: the consumer kernel publishes the flag
        if (wrote) {
            if (ar.fence == 2) __threadfence_system();
            else if (ar.fence == 1) __threadfence();
        }
        __syncthreads();
        if (tid == 0) {
            const unsigned old = atomicAdd(ar.cnt, 1u);
            if (old == gridDim.x - 1) {
                atomicExch(ar.cnt, 0u);
                __threadfence_system();
                st_vol_u32(ar.peer_flag, epoch_of(ar.st, ar.idx));
            }
        }
    }
}

template <int FMT, int RPL, int NCH, bool AR, bool SEG, int PRO, bool SQ = false>
void launch_gemv(const FW& W, const int8_t* xq, const int2* xm, float* y, cudaStream_t s, const ArArgs& ar,
                 const SegArgs& sg, const ProArgs& pa) {
    GemvArgs a = make_args(W.L, W.base, xq, xm, y, W.L.N);
    const int ntot = W.L.ntiles + (SEG ? sg.nrows : 0);
    // prologue kernels: co-resident grid (the prologue runs once per block); plain kernels: full grid (M0 policy)
    // leader prologue: grid must be co-resident (blocks wait on block 0) -> occupancy-checked cap
    int cap = g_max_blocks;
    if (PRO == PRO_LEADER) {
        static int occ[8] = {0};
        int dev = 0;
        cudaGetDevice(&dev);
        if (dev < 8 && !occ[dev]) {
            cudaFuncSetAttribute(k_gemv<FMT, RPL, NCH, AR, SEG, PRO, SQ>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                 100);
            int nb = 0, nsm = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_gemv<FMT, RPL, NCH, AR, SEG, PRO, SQ>, 256, 0);
            cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
            occ[dev] = nb * nsm;
            if (occ[dev] < 1) throw std::runtime_error("leader gemv: zero occupancy");
        }
        cap = std::min(cap, occ[dev < 8 ? dev : 0]);
    }
    const bool coresident = (PRO != PRO_NONE && !(SQ && PRO != PRO_LEADER));
    const int target = coresident ? std::min((ntot + 7) / 8, cap) : (ntot + 7) / 8;
    const int tpw = (ntot + 8 * target - 1) / (8 * target);
    const int blocks = (ntot + 8 * tpw - 1) / (8 * tpw);
    if (AR && (W.L.N % 4 || SEG || tpw > 2)) throw std::runtime_error("AR gemv needs N % 4 == 0, no segment, tpw <= 2");
    if (SQ && (tpw > 8 || RPL != 4 || SEG || AR || W.L.ntiles % (8 * tpw))) throw std::runtime_error("bad silu-quant gemv");
    static bool attr[8] = {false};  // per device: prefer max shared memory so two prologue blocks fit per SM
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 8 && !attr[dev]) {
        cudaFuncSetAttribute(k_gemv<FMT, RPL, NCH, AR, SEG, PRO, SQ>, cudaFuncAttributePreferredSharedMemoryCarveout,
                             100);
        attr[dev] = true;
    }
    k_gemv<FMT, RPL, NCH, AR, SEG, PRO, SQ><<<blocks, 256, 0, s>>>(a, ar, sg, pa, tpw);
}

}  // namespace

void set_max_blocks(int n) { g_max_blocks = n; }

void gemv(const FW& W, const int8_t* xq, const int2* xm, float* y, cudaStream_t s, const ArArgs* ar,
          const SegArgs* seg, const ProArgs* pro) {
    const ArArgs A = ar ? *ar : ArArgs{};
    const SegArgs S = seg ? *seg : SegArgs{};
    const ProArgs P = pro ? *pro : ProArgs{};
    const int f = W.L.fmt, nch = W.L.nch, rpl = W.L.rpl;
    const bool isar = ar != nullptr, isseg = seg != nullptr;
    int pk = PRO_NONE;
    if (pro) pk = pro->xflag ? PRO_LEADER : pro->h_in ? PRO_ARNORM : pro->gu ? PRO_SILU : pro->o ? PRO_GNORM : PRO_NONE;
    if (pro && pro->sq_xq) {  // gate|up with the silu-quant epilogue
        if (f == FAST_P4 && rpl == 4 && nch == 10 && !isar && !isseg && pk == PRO_NONE) {
            launch_gemv<FAST_P4, 4, 10, false, false, PRO_NONE, true>(W, xq, xm, y, s, A, S, P);
            return;
        }
        if (f == FAST_P4 && rpl == 4 && nch == 10 && !isar && !isseg && pk == PRO_ARNORM) {
            launch_gemv<FAST_P4, 4, 10, false, false, PRO_ARNORM, true>(W, xq, xm, y, s, A, S, P);
            return;
        }
        if (f == FAST_P4 && rpl == 4 && nch == 10 && !isar && !isseg && pk == PRO_LEADER) {
            launch_gemv<FAST_P4, 4, 10, false, false, PRO_LEADER, true>(W, xq, xm, y, s, A, S, P);
            return;
        }
        throw std::runtime_error("tp::gemv: no silu-quant instantiation");
    }
#define T4Q_G(FMT, RPL, NCH, AR_, SEG_, PRO_)                                                     \
    if (f == FMT && rpl == RPL && nch == NCH && isar == AR_ && isseg == SEG_ && pk == PRO_) {      \
        launch_gemv<FMT, RPL, NCH, AR_, SEG_, PRO_>(W, xq, xm, y, s, A, S, P);                    \
        return;                                                                                   \
    }
    T4Q_G(FAST_P4, 4, 10, false, true, PRO_ARNORM)   // DeltaNet qkvz + alpha/beta fp32 rows, AR + attn_norm
    T4Q_G(FAST_P4, 4, 10, false, false, PRO_ARNORM)  // attn q|k|v (attn_norm), ffn gate|up (post_norm)
    T4Q_G(FAST_K6, 2, 10, false, false, PRO_ARNORM)  // lm_head (output_norm)
    T4Q_G(FAST_P4, 4, 10, false, true, PRO_LEADER)   // leader-block prologue variants (fuse 3)
    T4Q_G(FAST_P4, 4, 10, false, false, PRO_LEADER)
    T4Q_G(FAST_K6, 2, 10, false, false, PRO_LEADER)
    T4Q_G(FAST_K5, 2, 6, true, false, PRO_GNORM)     // ssm_out (Q5_K) with gated norm prologue, AR publish
    T4Q_G(FAST_P4, 4, 6, true, false, PRO_NONE)      // attn_output, AR publish
    T4Q_G(FAST_P4, 4, 17, true, false, PRO_SILU)     // ffn_down Q4_0, silu prologue, AR publish
    T4Q_G(FAST_P4M, 4, 17, true, false, PRO_SILU)    // ffn_down Q4_1
    // unfused path (separate ar_norm / gnorm_q8 / silu_q8 kernels; option fuse=0)
    T4Q_G(FAST_P4, 4, 10, false, true, PRO_NONE)
    T4Q_G(FAST_K5, 2, 6, true, false, PRO_NONE)
    T4Q_G(FAST_P4, 4, 17, true, false, PRO_NONE)
    T4Q_G(FAST_P4M, 4, 17, true, false, PRO_NONE)
    // RPL=2 layouts of the P4/P4M weights (T4Q_RPL_P4=2 A/B)
    T4Q_G(FAST_P4, 2, 10, false, true, PRO_NONE)
    T4Q_G(FAST_P4, 2, 10, false, false, PRO_NONE)
    T4Q_G(FAST_P4, 2, 6, false, false, PRO_NONE)
    T4Q_G(FAST_P4, 2, 17, false, false, PRO_NONE)
    T4Q_G(FAST_P4M, 2, 17, false, false, PRO_NONE)
    T4Q_G(FAST_P4, 2, 6, true, false, PRO_NONE)
    T4Q_G(FAST_P4, 2, 17, true, false, PRO_NONE)
    T4Q_G(FAST_P4M, 2, 17, true, false, PRO_NONE)
    // self-test variants (x from global q8, no AR)
    T4Q_G(FAST_P4, 4, 10, false, false, PRO_NONE)
    T4Q_G(FAST_K6, 2, 10, false, false, PRO_NONE)
    T4Q_G(FAST_P4, 4, 6, false, false, PRO_NONE)
    T4Q_G(FAST_K5, 2, 6, false, false, PRO_NONE)
    T4Q_G(FAST_P4, 4, 17, false, false, PRO_NONE)
    T4Q_G(FAST_P4M, 4, 17, false, false, PRO_NONE)
#undef T4Q_G
    throw std::runtime_error("tp::gemv: no instantiation for fmt " + std::to_string(f) + " rpl " + std::to_string(rpl) +
                             " nch " + std::to_string(nch) + " ar " + std::to_string(isar) + " seg " +
                             std::to_string(isseg) + " pro " + std::to_string(pk));
}

// ------------------------------------------------------------------------------------------------ small kernels
namespace {

__global__ void k_embed(const uint8_t* __restrict__ embd, const StepState* st, const int* prompt, float* h, const Pf pf) {
    T4Q_PF_BLOCKS(20)
    const int pos = st->pos;
    int tok = pos < st->n_prompt ? prompt[pos] : st->token;
    if (tok < 0 || tok >= 248320) tok = 0;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint8_t* b = embd + (size_t)tok * 2880 + (i >> 5) * 18;
    const float d = h2f_u16((uint16_t)(b[0] | (b[1] << 8)));
    const int e = i & 31;
    const int q = e < 16 ? (b[2 + e] & 15) : (b[2 + e - 16] >> 4);
    h[i] = __fmul_rn((float)(q - 8), d);
}

template <bool WAIT>
__global__ void __launch_bounds__(1024) k_ar_norm(const float* h, float* h_out, const float* own, const float* rx,
                                                  const unsigned* flag, StepState* st, int idx,
                                                  const float* __restrict__ w, float* xn, int8_t* xq, int2* xm,
                                                  const Pf pf, float* pub_peer_rx, unsigned* pub_peer_flag) {
    T4Q_PF_BLOCKS(1)
    __shared__ float red[32];
    __shared__ int s_ok;
    const int tid = threadIdx.x;
    const int slot = idx & 1;
    if (pub_peer_flag)
        publish_partial(own + slot * 5120, pub_peer_rx ? pub_peer_rx + slot * 5120 : nullptr, pub_peer_flag + slot,
                        epoch_of(st, idx));
    if (WAIT) {
        if (tid == 0) {
            s_ok = wait_flag(flag + slot, epoch_of(st, idx));
            if (!s_ok) st->err = 1000 + idx;
        }
        __syncthreads();
    }
    float v[5];
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 5; k++) {
        const int i = tid + 1024 * k;
        float x = h[i];
        if (own) {
            x = x + (own[slot * 5120 + i] + ld_vol_f32(rx + slot * 5120 + i));
            h_out[i] = x;
        }
        v[k] = x;
        ss += x * x;
    }
    ss = block_sum(ss, red);
    const float scale = rsqrtf(ss / 5120.f + 1e-6f);
#pragma unroll
    for (int k = 0; k < 5; k++) {
        const int i = tid + 1024 * k;
        const float y = (v[k] * scale) * w[i];
        xn[i] = y;
        quant_warp(y, xq + i, xm + (i >> 5));
    }
}

// grid 96 = 24 local v heads x 4 slices of 32 value columns; 256 threads
__global__ void __launch_bounds__(256) k_gdn(const float* __restrict__ y, const float* __restrict__ yab, float* ring,
                                             const float* __restrict__ cw, const float* __restrict__ ssm_a,
                                             const float* __restrict__ ssm_dt, float* S, float* o,
                                             const StepState* st) {
    __shared__ float sq[128], sk[128], sv[32], red[8];
    const int vl = blockIdx.x >> 2, sl = blockIdx.x & 3, kl = vl & 7, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int pos = st->pos;
    const int p0 = (pos + 1) & 3, p1 = (pos + 2) & 3, p2 = (pos + 3) & 3, pw = pos & 3;  // pos-3, pos-2, pos-1, pos
    // state rows first: their DRAM latency overlaps the conv / L2-norm phase
    float s[4][4];
#pragma unroll
    for (int cc = 0; cc < 4; cc++) {
        const int col = sl * 32 + warp * 4 + cc;
        const float* Sp = S + ((size_t)vl * 128 + col) * 128;
#pragma unroll
        for (int r = 0; r < 4; r++) s[cc][r] = Sp[r * 32 + lane];
    }
    // q (tid < 128) or k (tid >= 128) channel of local k head kl
    {
        const int ch = tid < 128 ? kl * 128 + tid : 1024 + kl * 128 + (tid - 128);
        const float x = y[ch];
        const float* wc = cw + (size_t)ch * 4;
        float sum = 0.f;
        sum += ring[p0 * 5120 + ch] * wc[0];
        sum += ring[p1 * 5120 + ch] * wc[1];
        sum += ring[p2 * 5120 + ch] * wc[2];
        sum += x * wc[3];
        const float a = sum / (1.0f + expf(-sum));
        if (vl < 8 && sl == 0) ring[pw * 5120 + ch] = x;
        float ssq = warp_sum(a * a);
        if (lane == 0) red[warp] = ssq;
        __syncthreads();
        const float tot = tid < 128 ? (red[0] + red[1]) + (red[2] + red[3]) : (red[4] + red[5]) + (red[6] + red[7]);
        const float scale = rsqrtf(tot / 128.0f + 1e-6f / 128.0f);
        const float val = (a * scale) * (1.0f / sqrtf(128.0f));
        if (tid < 128) sq[tid] = val;
        else sk[tid - 128] = val;
    }
    if (tid < 32) {
        const int ch = 2048 + vl * 128 + sl * 32 + tid;
        const float x = y[ch];
        const float* wc = cw + (size_t)ch * 4;
        float sum = 0.f;
        sum += ring[p0 * 5120 + ch] * wc[0];
        sum += ring[p1 * 5120 + ch] * wc[1];
        sum += ring[p2 * 5120 + ch] * wc[2];
        sum += x * wc[3];
        sv[tid] = sum / (1.0f + expf(-sum));
        ring[pw * 5120 + ch] = x;
    }
    __syncthreads();
    const float beta = 1.0f / (1.0f + expf(-yab[24 + vl]));
    const float xg = yab[vl] + ssm_dt[vl];
    const float sp = xg > 20.0f ? xg : logf(1.0f + expf(xg));
    const float gv = expf(sp * ssm_a[vl]);
    float kr[4], qr[4];
#pragma unroll
    for (int r = 0; r < 4; r++) { kr[r] = sk[r * 32 + lane]; qr[r] = sq[r * 32 + lane]; }
#pragma unroll
    for (int cc = 0; cc < 4; cc++) {
        const int col = sl * 32 + warp * 4 + cc;
        float* Sp = S + ((size_t)vl * 128 + col) * 128;
        float kv = 0.f;
#pragma unroll
        for (int r = 0; r < 4; r++) kv += s[cc][r] * kr[r];
        kv = warp_sum(kv);
        const float delta = (sv[warp * 4 + cc] - gv * kv) * beta;
        float a = 0.f;
#pragma unroll
        for (int r = 0; r < 4; r++) {
            const float sn = gv * s[cc][r] + kr[r] * delta;
            a += sn * qr[r];
            Sp[r * 32 + lane] = sn;
        }
        a = warp_sum(a);
        if (lane == 0) o[vl * 128 + col] = a * (1.0f / sqrtf(128.0f));
    }
}

__global__ void __launch_bounds__(128) k_gnorm_q8(const float* o, const float* z, const float* __restrict__ w,
                                                  int8_t* xq, int2* xm, const Pf pf) {
    T4Q_PF_BLOCKS(24)
    __shared__ float red[4];
    const int vl = blockIdx.x, i = threadIdx.x;
    const float x = o[vl * 128 + i];
    const float ss = block_sum(x * x, red);
    const float scale = rsqrtf(ss / 128.0f + 1e-6f);
    const float zz = z[vl * 128 + i];
    const float val = ((x * scale) * w[i]) * (zz / (1.0f + expf(-zz)));
    quant_warp(val, xq + vl * 128 + i, xm + vl * 4 + (i >> 5));
}

__global__ void k_silu_q8(const float* gu, int n, int8_t* xq, int2* xm, const Pf pf) {
    T4Q_PF_BLOCKS((n + 255) / 256)
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // n % 32 == 0; whole warps in range
    if (i - (threadIdx.x & 31) >= n) return;
    const float g = gu[i], u = gu[n + i];
    const float val = (g / (1.0f + expf(-g))) * u;
    quant_warp(val, xq + i, xm + (i >> 5));
}

// blocks 0..11: local q heads, 12..13: local kv heads. blockDim 256
__global__ void k_attn_prep(const float* ya, const float* qw, const float* kw, float* qa, __half* kc, __half* vc,
                            int max_ctx, const StepState* st, float theta_scale) {
    __shared__ float red[8];
    __shared__ float yv[256];
    const int b = blockIdx.x, d = threadIdx.x;
    const int pos = st->pos;
    const bool isq = b < 12;
    const float* src = isq ? ya + b * 512 : ya + 6144 + (b - 12) * 256;
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
        float* dst = qa + b * 256;
        if (d < 32) { dst[d] = out0; dst[d + 32] = out1; }
        else if (d >= 64) dst[d] = yv[d];
    } else {
        const int kv = b - 12;
        __half* kd = kc + ((size_t)kv * max_ctx + pos) * 256;
        if (d < 32) { kd[d] = __float2half_rn(out0); kd[d + 32] = __float2half_rn(out1); }
        else if (d >= 64) kd[d] = __float2half_rn(yv[d]);
        vc[((size_t)kv * max_ctx + pos) * 256 + d] = __float2half_rn(ya[6656 + kv * 256 + d]);
    }
}

// grid (2 kv heads, NSPLIT), 256 threads, 2 blocks/SM (one wave). Warps 0-3 serve q heads 0-2 of the kv head, warps
// 4-7 heads 3-5, each group striding over the block's chunk of positions (two positions in flight per warp).
// ws layout: [(j * NSPLIT + s) * 6 + h6] x 258 floats {m, l, acc[256]}
__global__ void __launch_bounds__(256, 2) k_attn_split(const float* qa, const __half* kc, const __half* vc, float* ws,
                                                       int max_ctx, const StepState* st) {
    __shared__ float sm_m[8][3], sm_l[8][3];
    __shared__ float sacc[8][256];
    const int j = blockIdx.x, sidx = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, hg = warp >> 2, wq = warp & 3;
    const int n_kv = st->pos + 1;
    const int chunk = (n_kv + NSPLIT - 1) / NSPLIT;
    const int t0 = sidx * chunk, t1 = min(n_kv, t0 + chunk);
    float q[3][8];
#pragma unroll
    for (int hh = 0; hh < 3; hh++) {
        const float4* qp = (const float4*)(qa + (j * 6 + 3 * hg + hh) * 256 + lane * 8);
        const float4 a = qp[0], b = qp[1];
        q[hh][0] = a.x; q[hh][1] = a.y; q[hh][2] = a.z; q[hh][3] = a.w;
        q[hh][4] = b.x; q[hh][5] = b.y; q[hh][6] = b.z; q[hh][7] = b.w;
    }
    float m[3], l[3], acc[3][8];
#pragma unroll
    for (int hh = 0; hh < 3; hh++) {
        m[hh] = -FLT_MAX; l[hh] = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) acc[hh][i] = 0.f;
    }
    const __half* K = kc + (size_t)j * max_ctx * 256;
    const __half* Vv = vc + (size_t)j * max_ctx * 256;
    for (int t = t0 + wq; t < t1; t += 8) {
        const bool two = t + 4 < t1;
        uint4 kraw[2], vraw[2];
        kraw[0] = *(const uint4*)(K + (size_t)t * 256 + lane * 8);
        vraw[0] = *(const uint4*)(Vv + (size_t)t * 256 + lane * 8);
        if (two) {
            kraw[1] = *(const uint4*)(K + (size_t)(t + 4) * 256 + lane * 8);
            vraw[1] = *(const uint4*)(Vv + (size_t)(t + 4) * 256 + lane * 8);
        }
#pragma unroll
        for (int u = 0; u < 2; u++) {
            if (u == 1 && !two) break;
            float k[8], v[8];
            const __half2* kh = (const __half2*)&kraw[u];
            const __half2* vh = (const __half2*)&vraw[u];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float2 kf = __half22float2(kh[i]), vf = __half22float2(vh[i]);
                k[2 * i] = kf.x; k[2 * i + 1] = kf.y; v[2 * i] = vf.x; v[2 * i + 1] = vf.y;
            }
#pragma unroll
            for (int hh = 0; hh < 3; hh++) {
                float dot = 0.f;
#pragma unroll
                for (int i = 0; i < 8; i++) dot += q[hh][i] * k[i];
                dot = warp_sum(dot) * (1.0f / 16.0f);
                const float mn = fmaxf(m[hh], dot);
                const float c = expf(m[hh] - mn), p = expf(dot - mn);
                l[hh] = l[hh] * c + p;
#pragma unroll
                for (int i = 0; i < 8; i++) acc[hh][i] = acc[hh][i] * c + p * v[i];
                m[hh] = mn;
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int hh = 0; hh < 3; hh++) { sm_m[warp][hh] = m[hh]; sm_l[warp][hh] = l[hh]; }
    }
    __syncthreads();
#pragma unroll
    for (int hh = 0; hh < 3; hh++) {
        float M = -FLT_MAX;
#pragma unroll
        for (int w = 0; w < 4; w++) M = fmaxf(M, sm_m[4 * hg + w][hh]);
        const float sc = (l[hh] > 0.f) ? expf(m[hh] - M) : 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) sacc[warp][lane * 8 + i] = acc[hh][i] * sc;
        __syncthreads();
#pragma unroll
        for (int g2 = 0; g2 < 2; g2++) {
            float* out = ws + ((size_t)(j * NSPLIT + sidx) * 6 + 3 * g2 + hh) * 258;
            float a = 0.f;
#pragma unroll
            for (int w = 0; w < 4; w++) a += sacc[4 * g2 + w][tid];
            out[2 + tid] = a;
            if (tid == 0) {
                float M2 = -FLT_MAX;
                for (int w = 0; w < 4; w++) M2 = fmaxf(M2, sm_m[4 * g2 + w][hh]);
                float L = 0.f;
                for (int w = 0; w < 4; w++)
                    if (sm_l[4 * g2 + w][hh] > 0.f) L += sm_l[4 * g2 + w][hh] * expf(sm_m[4 * g2 + w][hh] - M2);
                out[0] = M2;
                out[1] = L;
            }
        }
        __syncthreads();
    }
}

// 12 blocks (local q heads) x 256: merge splits, sigmoid gate, q8 for attn_output
__global__ void __launch_bounds__(256) k_attn_combine_q8(const float* ws, const float* ya, const StepState* st,
                                                         int8_t* xq, int2* xm, const Pf pf) {
    T4Q_PF_BLOCKS(12)
    const int hl = blockIdx.x, d = threadIdx.x, j = hl / 6, h6 = hl % 6;
    const int n_kv = st->pos + 1;
    const int chunk = (n_kv + NSPLIT - 1) / NSPLIT;
    const int nsp = (n_kv + chunk - 1) / chunk;  // splits with at least one position
    float M = -FLT_MAX;
#pragma unroll 8
    for (int s = 0; s < nsp; s++) M = fmaxf(M, ws[((size_t)(j * NSPLIT + s) * 6 + h6) * 258]);
    float num = 0.f, den = 0.f;
#pragma unroll 8
    for (int s = 0; s < nsp; s++) {
        const float* p = ws + ((size_t)(j * NSPLIT + s) * 6 + h6) * 258;
        const float wgt = expf(p[0] - M);
        num += wgt * p[2 + d];
        den += wgt * p[1];
    }
    const float att = num / den;
    const float g = ya[hl * 512 + 256 + d];
    const float val = att * (1.0f / (1.0f + expf(-g)));
    quant_warp(val, xq + hl * 256 + d, xm + hl * 8 + (d >> 5));
}

constexpr int NBA = 160;
__global__ void __launch_bounds__(256) k_argmax_part(const float* x, int n, float* apart) {
    __shared__ float sv[256];
    __shared__ int si[256];
    const int per = (n + NBA - 1) / NBA;
    const int lo = blockIdx.x * per, hi = min(n, lo + per);
    float bv = -FLT_MAX;
    int bi = 0x7fffffff;
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const float v = x[i];
        if (v > bv) { bv = v; bi = i; }
    }
    sv[threadIdx.x] = bv;
    si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float ov = sv[threadIdx.x + s];
            const int oi = si[threadIdx.x + s];
            if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) {
                sv[threadIdx.x] = ov;
                si[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        apart[blockIdx.x] = sv[0];
        apart[NBA + blockIdx.x] = __int_as_float(si[0]);
    }
}

__global__ void k_argmax_final(const float* apart, int row0, float* amb, const unsigned* aflag, float* peer_amb,
                               unsigned* peer_aflag, StepState* st, int* ring) {
    if (threadIdx.x != 0) return;
    float bv = -FLT_MAX;
    int bi = 0x7fffffff;
    for (int b = 0; b < NBA; b++) {
        const float v = apart[b];
        const int i = __float_as_int(apart[NBA + b]);
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    bi += row0;
    const unsigned step = st->step;
    const int slot = step & 1;
    const unsigned e = step + 1u;
    peer_amb[slot * 2] = bv;
    peer_amb[slot * 2 + 1] = __int_as_float(bi);
    __threadfence_system();
    st_vol_u32(peer_aflag + slot, e);
    if (!wait_flag(aflag + slot, e)) st->err = 2000;
    const float pv = ld_vol_f32(amb + slot * 2);
    const int pi = __float_as_int(ld_vol_f32(amb + slot * 2 + 1));
    int best = bi;
    float bestv = bv;
    if (pv > bv || (pv == bv && pi < bi)) { best = pi; bestv = pv; }
    st->token = best;
    st->last_tok = best;
    st->last_val = bestv;
    if (ring) {
        ring[step % RING] = best;
        __threadfence_system();
    }
    st->pos = st->pos + 1;
    st->step = step + 1u;
}

__global__ void __launch_bounds__(640) k_pull(const unsigned* hflag, const float* hrx, float* rx, StepState* st,
                                              int idx, const float* own, float* pub_peer_rx, unsigned* pub_peer_flag) {
    __shared__ int ok;
    if (pub_peer_flag)
        publish_partial(own + (idx & 1) * 5120, pub_peer_rx ? pub_peer_rx + (idx & 1) * 5120 : nullptr,
                        pub_peer_flag + (idx & 1), epoch_of(st, idx));
    if (threadIdx.x == 0) {
        ok = wait_flag(hflag, epoch_of(st, idx));
        if (!ok) st->err = 3000 + idx;
    }
    __syncthreads();
    float4 v;
    const float* p = hrx + threadIdx.x * 8;
    asm volatile("ld.volatile.global.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    float4 u;
    asm volatile("ld.volatile.global.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(u.x), "=f"(u.y), "=f"(u.z), "=f"(u.w) : "l"(p + 4));
    ((float4*)rx)[threadIdx.x * 2] = v;
    ((float4*)rx)[threadIdx.x * 2 + 1] = u;
}

}  // namespace

void pull(const unsigned* hflag, const float* hrx, float* rx, StepState* st, int idx, cudaStream_t s,
          const float* own, float* pub_peer_rx, unsigned* pub_peer_flag) {
    k_pull<<<1, 640, 0, s>>>(hflag, hrx, rx, st, idx, own, pub_peer_rx, pub_peer_flag);
}

void embed(const uint8_t* embd, StepState* st, const int* prompt, float* h, cudaStream_t s, const Pf* pf) {
    const Pf P = pf ? *pf : Pf{};
    k_embed<<<20 + P.blocks, 256, 0, s>>>(embd, st, prompt, h, P);
}

void ar_norm(const float* h, float* h_out, const float* own, const float* rx, const unsigned* flag,
             const StepState* st, int idx, const float* w, float* xn, int8_t* xq, int2* xm, cudaStream_t s,
             const Pf* pf, float* pub_peer_rx, unsigned* pub_peer_flag) {
    const Pf P = pf ? *pf : Pf{};
    const int nb = 1 + (P.blocks + 3) / 4;  // 1024-thread blocks
    if (flag)
        k_ar_norm<true><<<nb, 1024, 0, s>>>(h, h_out, own, rx, flag, (StepState*)st, idx, w, xn, xq, xm, P,
                                            pub_peer_rx, pub_peer_flag);
    else
        k_ar_norm<false><<<nb, 1024, 0, s>>>(h, h_out, own, rx, flag, (StepState*)st, idx, w, xn, xq, xm, P,
                                             pub_peer_rx, pub_peer_flag);
}

void gdn(const float* y, const float* yab, float* ring, const float* conv_w, const float* ssm_a, const float* ssm_dt,
         float* S, float* o, const StepState* st, cudaStream_t s) {
    k_gdn<<<96, 256, 0, s>>>(y, yab, ring, conv_w, ssm_a, ssm_dt, S, o, st);
}

void gnorm_q8(const float* o, const float* z, const float* w, int8_t* xq, int2* xm, cudaStream_t s, const Pf* pf) {
    const Pf P = pf ? *pf : Pf{};
    k_gnorm_q8<<<24 + 2 * P.blocks, 128, 0, s>>>(o, z, w, xq, xm, P);
}

void silu_q8(const float* gu, int n, int8_t* xq, int2* xm, cudaStream_t s, const Pf* pf) {
    const Pf P = pf ? *pf : Pf{};
    k_silu_q8<<<(n + 255) / 256 + P.blocks, 256, 0, s>>>(gu, n, xq, xm, P);
}

void attn_prep(const float* ya, const float* qw, const float* kw, float* qa, uint16_t* kc, uint16_t* vc, int max_ctx,
               const StepState* st, float theta_scale, cudaStream_t s) {
    k_attn_prep<<<14, 256, 0, s>>>(ya, qw, kw, qa, (__half*)kc, (__half*)vc, max_ctx, st, theta_scale);
}

void attn_split(const float* qa, const uint16_t* kc, const uint16_t* vc, float* ws, int max_ctx, const StepState* st,
                cudaStream_t s) {
    k_attn_split<<<dim3(2, NSPLIT), 256, 0, s>>>(qa, (const __half*)kc, (const __half*)vc, ws, max_ctx, st);
}

void attn_combine_q8(const float* ws, const float* ya, const StepState* st, int8_t* xq, int2* xm, cudaStream_t s,
                     const Pf* pf) {
    const Pf P = pf ? *pf : Pf{};
    k_attn_combine_q8<<<12 + P.blocks, 256, 0, s>>>(ws, ya, st, xq, xm, P);
}

void argmax_step(const float* logits, int n, int row0, float* apart, float* amb, const unsigned* aflag,
                 float* peer_amb, unsigned* peer_aflag, StepState* st, int* ring, cudaStream_t s) {
    k_argmax_part<<<NBA, 256, 0, s>>>(logits, n, apart);
    k_argmax_final<<<1, 32, 0, s>>>(apart, row0, amb, aflag, peer_amb, peer_aflag, st, ring);
}

}  // namespace tp
