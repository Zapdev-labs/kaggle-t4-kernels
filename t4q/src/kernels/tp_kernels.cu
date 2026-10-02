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
// phase timing (option dbgts): block (0,0) thread 0 adds globaltimer offsets from its own start into d_dbg[base + k]
__device__ unsigned long long* d_dbg = nullptr;
#define DBG_T0 \
    unsigned long long _dbg_t0 = 0; \
    unsigned long long* _dbg = (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) ? d_dbg : nullptr; \
    if (_dbg) _dbg_t0 = gtimer();
#define DBG_PH(base, k) \
    if (_dbg) atomicAdd(_dbg + (base) + (k), gtimer() - _dbg_t0);
#define DBG_N(base) \
    if (_dbg) atomicAdd(_dbg + (base) + 15, 1ull);
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
__device__ int d_spin_ns = 0;  // > 0: __nanosleep between polls of a spin wait (option spin_ns; power under the cap)
// spin until flag >= e (wrap-safe); returns false on watchdog
__device__ __forceinline__ bool wait_flag(const unsigned* flag, unsigned e) {
    const unsigned long long t0 = gtimer();
    unsigned spins = 0;
    const int ns = d_spin_ns;
    while ((int)(ld_vol_u32(flag) - e) < 0) {
        if (ns) __nanosleep(ns);
        if ((++spins & 1023u) == 0 && gtimer() - t0 > WATCHDOG_NS) return false;
    }
    return true;
}
__device__ __forceinline__ float2 ld_vol_f2(const float2* p) {
    float2 v;
    asm volatile("ld.volatile.global.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(p));
    return v;
}
__device__ __forceinline__ void st_vol_f2(float2* p, float x, float y) {
    asm volatile("st.volatile.global.v2.f32 [%0], {%1,%2};" ::"l"(p), "f"(x), "f"(y) : "memory");
}
// LL mailbox (option ll): the peer writes {value, epoch tag} pairs with single 8-byte stores, so a value is valid once
// its tag reaches the epoch; no fence, counter or flag on the producer side. Returns the value (NaN on watchdog).
__device__ __forceinline__ float wait_ll(const float2* p, unsigned e, bool& ok) {
    float2 v = ld_vol_f2(p);
    if ((int)(__float_as_uint(v.y) - e) >= 0) return v.x;
    const unsigned long long t0 = gtimer();
    unsigned spins = 0;
    const int ns = d_spin_ns;
    for (;;) {
        if (ns) __nanosleep(ns);
        v = ld_vol_f2(p);
        if ((int)(__float_as_uint(v.y) - e) >= 0) return v.x;
        if ((++spins & 1023u) == 0 && gtimer() - t0 > WATCHDOG_NS) { ok = false; return 0.f; }
    }
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

// AR tail block t of nt (blockDim elements each): see ArArgs fence -4
__device__ __noinline__ void ar_tail(const ArArgs& ar, unsigned nwork, int t, float* red) {
    const int tid = threadIdx.x, nt = ar.tb;
    const int e = t * blockDim.x + tid;
    const int slot = ar.idx & 1;
    const unsigned ep = epoch_of(ar.st, ar.idx);
    const float wv = ar.nw[e];
    const float hv = ar.h_in[e];
    if (tid == 0) {  // all work blocks of this GPU done (their y rows are written and fenced)
        const unsigned long long t0 = gtimer();
        unsigned spins = 0;
        bool ok = true;
        while (ld_vol_u32(ar.cnt) < nwork) {
            if ((++spins & 1023u) == 0 && gtimer() - t0 > WATCHDOG_NS) { ok = false; break; }
        }
        if (!ok) ((StepState*)ar.st)->err = 6000 + ar.idx;
    }
    __syncthreads();
    __threadfence();
    const float ov = __ldcg(ar.own + e);
    ar.y_peer[e] = ov;  // my slice of the partial to the peer's mailbox (coalesced)
    __syncthreads();
    if (tid == 0) {
        __threadfence_system();
        st_vol_u32(ar.peer_tflag + slot * TFLAGS + t, ep);
        const unsigned o = atomicAdd(ar.cnt2, 1u);
        if (o == (unsigned)nt - 1u) {  // every tail block is past the counter wait: reset for the next K-split GEMV
            *ar.cnt = 0u;
            *ar.cnt2 = 0u;
        }
    }
    if (tid < nt) {
        if (!wait_flag(ar.tflag + slot * TFLAGS + tid, ep)) ((StepState*)ar.st)->err = 7000 + ar.idx;
    }
    __syncthreads();
    float ss = 0.f;
    const int n4 = 1280 / blockDim.x;
    for (int k = 0; k < n4; k++) {
        const int i4 = tid + blockDim.x * k;
        const float4 h4 = __ldcg((const float4*)ar.h_in + i4);
        const float4 o4 = __ldcg((const float4*)ar.own + i4);
        const float4 r4 = ld_vol_f4((const float4*)ar.rx + i4);
        const float x0 = h4.x + (o4.x + r4.x), x1 = h4.y + (o4.y + r4.y);
        const float x2 = h4.z + (o4.z + r4.z), x3 = h4.w + (o4.w + r4.w);
        ss += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
    }
    ss = block_sum(ss, red);
    const float scale = rsqrtf(ss / 5120.f + 1e-6f);
    const float x = hv + (ov + ld_vol_f32(ar.rx + e));
    ar.h_out[e] = x;
    const float y = (x * scale) * wv;
    ar.xn[e] = y;
    quant_warp(y, ar.xq + e, ar.xm + (e >> 5));
}

int g_max_blocks = 80;
int g_sq_threads = 256;  // gate|up silu-quant GEMV block size (option sqt: 128 -> 2 tiles per warp, 256 -> 1)
int g_threads = 128;  // plain-x GEMV block size (M4 v11 selftest: 128 >= 256 on every shape)

template <int FMT, int RPL, int NCH, bool AR, bool SEG, int PRO, bool SQ = false, int CVX = 1>
__global__ void __launch_bounds__(256, 2) k_gemv(const GemvArgs a, const ArArgs ar, const SegArgs sg, const ProArgs pa,
                                                 int tpw) {
    constexpr int D = 2, M = 1;
    constexpr int CVT = FMT == FAST_P4 ? CVX : FMT == FAST_Q8 ? 1 : 0;  // CVX 2: P4 unsigned high-nibble dp4a
    constexpr int NB = NCH * 16;
    constexpr int K = NCH * 512;
    const int tid = threadIdx.x, lane = tid & 31, h = lane >> 4, j = lane & 15, wib = tid >> 5;
    const int wpb = blockDim.x >> 5;  // warps per block (8, or 4 for 128-thread plain kernels)
    const int warp = blockIdx.x * wpb + wib;
    if (AR && ar.fence == -4 && (int)blockIdx.x >= (int)gridDim.x - ar.tb) {
        __shared__ float tred[8];
        ar_tail(ar, gridDim.x - ar.tb, blockIdx.x - (gridDim.x - ar.tb), tred);
        return;
    }
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
                const int moff = FMT == FAST_P4 ? 0x4B400000 - (CVT == 2 ? 128 : 8) * (s0 + s1) : 0x4B400000;
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
                    if (lane == 0) sa[(tile - blockIdx.x * wpb * tpw) * RPL + r] = (v / (1.0f + expf(-v))) * u;
                }
                if (j == 0 && row < a.N) {
                    a.y[(size_t)col * a.ldy + row] = v;
                    if (AR) sy[(tile - blockIdx.x * wpb * tpw) * 2 * RPL + h * RPL + r] = v;
                }
            }
    }
    if (SQ) {  // the block owns outputs [blockIdx.x * ng * 32, +ng * 32), ng = wpb * tpw * RPL / 32 q8 groups
        __syncthreads();
        const int ng = wpb * tpw * RPL / 32;
        if (wib < ng)
            quant_warp(sa[wib * 32 + lane], pa.sq_xq + (blockIdx.x * ng + wib) * 32 + lane,
                       pa.sq_xm + blockIdx.x * ng + wib);
    }
    if (AR && ar.fence == -4) {  // AR tail mode: rows stay local; fence the y writes, then count this block
        __threadfence();
        __syncthreads();
        if (tid == 0) atomicAdd(ar.cnt, 1u);
        return;
    }
    if (AR) {
        __syncthreads();
        const int nrow = wpb * tpw * 2 * RPL;  // AR tail mode: rows stay local; counted below
        if (ar.fence == -3) {  // LL rows: {value, epoch} 8-byte stores into the peer's float2 mailbox
            const unsigned tag = epoch_of(ar.st, ar.idx);
            for (int t = tid; t < nrow; t += blockDim.x) {
                const int row = blockIdx.x * nrow + t;
                if (row < a.N) st_vol_f2((float2*)ar.y_peer + row, sy[t], __uint_as_float(tag));
            }
            return;
        }
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

template <int FMT, int RPL, int NCH, bool AR, bool SEG, int PRO, bool SQ = false, int CVX = 1>
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
            cudaFuncSetAttribute(k_gemv<FMT, RPL, NCH, AR, SEG, PRO, SQ, CVX>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                 100);
            int nb = 0, nsm = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_gemv<FMT, RPL, NCH, AR, SEG, PRO, SQ, CVX>, 256, 0);
            cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
            occ[dev] = nb * nsm;
            if (occ[dev] < 1) throw std::runtime_error("leader gemv: zero occupancy");
        }
        cap = std::min(cap, occ[dev < 8 ? dev : 0]);
    }
    const bool coresident = (PRO != PRO_NONE && !(SQ && PRO != PRO_LEADER));
    // block size: 128 threads (T4Q_THREADS) only for plain-x kernels; prologue kernels assume 256
    const int threads = PRO == PRO_NONE ? (SQ ? g_sq_threads : g_threads) : 256;
    const int wpb = threads / 32;
    const int target = coresident ? std::min((ntot + wpb - 1) / wpb, cap) : (ntot + wpb - 1) / wpb;
    int tpw = (ntot + wpb * target - 1) / (wpb * target);
    if (SQ) tpw = std::max(tpw, 32 / (wpb * RPL));  // a block must own whole q8 groups
    const int blocks = (ntot + wpb * tpw - 1) / (wpb * tpw);
    if (AR && (W.L.N % 4 || SEG || tpw > 2)) throw std::runtime_error("AR gemv needs N % 4 == 0, no segment, tpw <= 2");
    if (SQ && (tpw > 8 || RPL != 4 || SEG || AR || W.L.ntiles % (wpb * tpw) || (wpb * tpw * RPL) % 32))
        throw std::runtime_error("bad silu-quant gemv");
    if (AR && wpb * tpw * 2 * RPL > 8 * 2 * 2 * RPL) throw std::runtime_error("AR gemv: staging too small");
    static bool attr[8] = {false};  // per device: prefer max shared memory so two prologue blocks fit per SM
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 8 && !attr[dev]) {
        cudaFuncSetAttribute(k_gemv<FMT, RPL, NCH, AR, SEG, PRO, SQ, CVX>, cudaFuncAttributePreferredSharedMemoryCarveout,
                             100);
        attr[dev] = true;
    }
    const int tail = (AR && ar.fence == -4) ? ar.tb : 0;
    if (tail && tail * threads != 5120) throw std::runtime_error("AR tail: tb * threads != 5120");
    k_gemv<FMT, RPL, NCH, AR, SEG, PRO, SQ, CVX><<<blocks + tail, threads, 0, s>>>(a, ar, sg, pa, tpw);
}

}  // namespace

int g_p4u = 1;  // 1: P4 GEMVs use the unsigned high-nibble dp4a path (bit-identical, fewer instructions)
void set_p4u(int v) { g_p4u = v; }
void set_sq_threads(int v) { g_sq_threads = v == 256 ? 256 : 128; }
void set_max_blocks(int n) { g_max_blocks = n; }
int gemv_threads() { return g_threads; }
void set_threads(int n) { g_threads = (n == 128) ? 128 : 256; }

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
            if (g_p4u) launch_gemv<FAST_P4, 4, 10, false, false, PRO_NONE, true, 2>(W, xq, xm, y, s, A, S, P);
            else launch_gemv<FAST_P4, 4, 10, false, false, PRO_NONE, true>(W, xq, xm, y, s, A, S, P);
            return;
        }
        if (f == FAST_P4 && rpl == 4 && nch == 10 && !isar && !isseg && pk == PRO_ARNORM) {
            if (g_p4u) launch_gemv<FAST_P4, 4, 10, false, false, PRO_ARNORM, true, 2>(W, xq, xm, y, s, A, S, P);
            else launch_gemv<FAST_P4, 4, 10, false, false, PRO_ARNORM, true>(W, xq, xm, y, s, A, S, P);
            return;
        }
        if (f == FAST_P4 && rpl == 4 && nch == 10 && !isar && !isseg && pk == PRO_LEADER) {
            if (g_p4u) launch_gemv<FAST_P4, 4, 10, false, false, PRO_LEADER, true, 2>(W, xq, xm, y, s, A, S, P);
            else launch_gemv<FAST_P4, 4, 10, false, false, PRO_LEADER, true>(W, xq, xm, y, s, A, S, P);
            return;
        }
        throw std::runtime_error("tp::gemv: no silu-quant instantiation");
    }
#define T4Q_G(FMT, RPL, NCH, AR_, SEG_, PRO_)                                                     \
    if (f == FMT && rpl == RPL && nch == NCH && isar == AR_ && isseg == SEG_ && pk == PRO_) {      \
        if (FMT == FAST_P4 && g_p4u)                                                              \
            launch_gemv<FMT, RPL, NCH, AR_, SEG_, PRO_, false, (FMT == FAST_P4 ? 2 : 1)>(W, xq, xm, y, s, A, S, P); \
        else                                                                                      \
            launch_gemv<FMT, RPL, NCH, AR_, SEG_, PRO_>(W, xq, xm, y, s, A, S, P);                \
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
    // K5 (ssm_out) RPL 1 / 4 layouts (T4Q_RPL_K5 A/B)
    T4Q_G(FAST_K5, 4, 6, true, false, PRO_NONE)
    T4Q_G(FAST_K5, 4, 6, false, false, PRO_NONE)
    T4Q_G(FAST_K5, 1, 6, true, false, PRO_NONE)
    T4Q_G(FAST_K5, 1, 6, false, false, PRO_NONE)
#undef T4Q_G
    throw std::runtime_error("tp::gemv: no instantiation for fmt " + std::to_string(f) + " rpl " + std::to_string(rpl) +
                             " nch " + std::to_string(nch) + " ar " + std::to_string(isar) + " seg " +
                             std::to_string(isseg) + " pro " + std::to_string(pk));
}

// ------------------------------------------------------------------------------------------------ small kernels
namespace {

__global__ void k_embed(const uint8_t* __restrict__ embd, const StepState* st, const int* prompt, float* h, const Pf pf) {
    T4Q_PF_BLOCKS(20)
    DBG_T0
    const int pos = st->pos;
    int tok = pos < st->n_prompt ? prompt[pos] : st->token;
    if (tok < 0 || tok >= 248320) tok = 0;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint8_t* b = embd + (size_t)tok * 2880 + (i >> 5) * 18;
    const float d = h2f_u16((uint16_t)(b[0] | (b[1] << 8)));
    const int e = i & 31;
    const int q = e < 16 ? (b[2 + e] & 15) : (b[2 + e - 16] >> 4);
    h[i] = __fmul_rn((float)(q - 8), d);
    DBG_PH(96, 0)
    DBG_N(96)
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

// multi-block AR + RMSNorm + q8 (default since M4 round 2): 20 blocks x 256 threads, every block reduces the full sum
// of squares redundantly (60 KB of L2 reads each) and then normalizes / quantizes its own 256 elements (8 q8 groups),
// one element per thread. The single-block kernel above spent 13-15 us per call (graph trace, M4 v14) in its serial
// per-thread loop of 5 dependent load + quant_warp rounds.
constexpr int ARN_BLOCKS = 20;
// MF (arpub 3): every block copies its own 256-element slice of the partial to the peer and sets its own flag
// (pub_peer_flag / flag are [2][TFLAGS] arrays); threads 0..19 then wait for the peer's 20 slice flags
template <bool WAIT, bool MF = false>
__global__ void __launch_bounds__(256) k_ar_norm_mb(const float* h, float* h_out, const float* own, const float* rx,
                                                    const unsigned* flag, StepState* st, int idx,
                                                    const float* __restrict__ w, float* xn, int8_t* xq, int2* xm,
                                                    const Pf pf, float* pub_peer_rx, unsigned* pub_peer_flag) {
    T4Q_PF_BLOCKS(ARN_BLOCKS)
    if (MF) {
        const int sl = idx & 1, e0 = blockIdx.x * 256 + threadIdx.x;
        pub_peer_rx[sl * 5120 + e0] = own[sl * 5120 + e0];
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            st_vol_u32(pub_peer_flag + sl * TFLAGS + blockIdx.x, epoch_of(st, idx));
        }
        if (threadIdx.x < ARN_BLOCKS) {
            if (!wait_flag(flag + sl * TFLAGS + threadIdx.x, epoch_of(st, idx))) st->err = 1000 + idx;
        }
        __syncthreads();
    }
    __shared__ __align__(16) float sx[5120];
    __shared__ float red[8];
    __shared__ int s_ok;
    const int tid = threadIdx.x;
    const int slot = idx & 1;
    const int e = blockIdx.x * 256 + tid;
    DBG_T0
    const float wv = w[e];  // independent of the wait: issue early
    if (!MF && pub_peer_flag && blockIdx.x == 0)
        publish_partial(own + slot * 5120, pub_peer_rx ? pub_peer_rx + slot * 5120 : nullptr, pub_peer_flag + slot,
                        epoch_of(st, idx));
    float4 xv[5], ov[5];
#pragma unroll
    for (int k = 0; k < 5; k++) xv[k] = ((const float4*)h)[tid + 256 * k];
    if (own) {
#pragma unroll
        for (int k = 0; k < 5; k++) ov[k] = ((const float4*)(own + slot * 5120))[tid + 256 * k];
    }
    if (WAIT) {
        if (tid == 0) {
            s_ok = wait_flag(flag + slot, epoch_of(st, idx));
            if (!s_ok) st->err = 1000 + idx;
        }
        __syncthreads();
    }
    if (own) {
        float4 rv[5];
#pragma unroll
        for (int k = 0; k < 5; k++) rv[k] = ld_vol_f4((const float4*)(rx + slot * 5120) + tid + 256 * k);
#pragma unroll
        for (int k = 0; k < 5; k++) {
            xv[k].x = xv[k].x + (ov[k].x + rv[k].x);
            xv[k].y = xv[k].y + (ov[k].y + rv[k].y);
            xv[k].z = xv[k].z + (ov[k].z + rv[k].z);
            xv[k].w = xv[k].w + (ov[k].w + rv[k].w);
        }
    }
    DBG_PH(64, 0)
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 5; k++) {
        ss += xv[k].x * xv[k].x + xv[k].y * xv[k].y + xv[k].z * xv[k].z + xv[k].w * xv[k].w;
        ((float4*)sx)[tid + 256 * k] = xv[k];
    }
    ss = block_sum(ss, red);  // its barriers also publish sx
    DBG_PH(64, 1)
    const float scale = rsqrtf(ss / 5120.f + 1e-6f);
    const float x = sx[e];
    if (own) h_out[e] = x;
    const float y = (x * scale) * wv;
    xn[e] = y;
    quant_warp(y, xq + e, xm + (e >> 5));
    DBG_PH(64, 2)
    DBG_N(64)
}

// No-P2P AR + norm in one kernel (option pn, arpub 1 semantics; replaces pull + ar_norm): block b of 20
//   copies its 256-element slice of this GPU's partial to the peer's host-mapped mailbox and sets its slice flag,
//   waits for the peer's 20 slice flags (host-mapped, local), reads only its own slice of the peer partial from host
//   memory, then the blocks exchange their sums of squares through tagged {value, epoch} pairs in device memory
//   (fixed summation order, identical on both GPUs) and each normalizes / quantizes its slice.
__global__ void __launch_bounds__(256) k_pull_norm(const float* h, float* h_out, const float* own, const float* hrx,
                                                   const unsigned* htflag, StepState* st, int idx,
                                                   const float* __restrict__ w, float* xn, int8_t* xq, int2* xm,
                                                   float* peer_hrx, unsigned* peer_htflag, float2* ssb) {
    __shared__ float red[8];
    __shared__ float s_tot;
    const int tid = threadIdx.x, b = blockIdx.x, sl = idx & 1;
    const int e = b * 256 + tid;
    const unsigned ep = epoch_of(st, idx);
    const float wv = w[e], hv = h[e];
    const float ov = own[sl * 5120 + e];
    peer_hrx[sl * 5120 + e] = ov;
    __syncthreads();
    if (tid == 0) {
        __threadfence_system();
        st_vol_u32(peer_htflag + sl * TFLAGS + b, ep);
    }
    if (tid < ARN_BLOCKS) {
        if (!wait_flag(htflag + sl * TFLAGS + tid, ep)) st->err = 3000 + idx;
    }
    __syncthreads();
    const float r = ld_vol_f32(hrx + sl * 5120 + e);
    const float x = hv + (ov + r);
    h_out[e] = x;
    const float ss = block_sum(x * x, red);
    if (tid == 0) st_vol_f2(ssb + sl * 32 + b, ss, __uint_as_float(ep));
    if (tid < 32) {  // gather the 20 block sums (tagged) and add them in block order
        float v = 0.f;
        if (tid < ARN_BLOCKS) {
            bool ok = true;
            v = wait_ll(ssb + sl * 32 + tid, ep, ok);
            if (!ok) st->err = 3500 + idx;
        }
        // fixed-order sum: lane 0 adds lanes 0..19 sequentially via shuffles
        float tot = 0.f;
        for (int i = 0; i < ARN_BLOCKS; i++) tot += __shfl_sync(0xffffffffu, v, i);
        if (tid == 0) s_tot = tot;
    }
    __syncthreads();
    const float scale = rsqrtf(s_tot / 5120.f + 1e-6f);
    const float y = (x * scale) * wv;
    xn[e] = y;
    quant_warp(y, xq + e, xm + (e >> 5));
}

// LL variant of k_ar_norm: h += own + rxl.value once every tag reached epoch(idx); same arithmetic order
__global__ void __launch_bounds__(1024) k_ar_norm_ll(const float* h, float* h_out, const float* own, const float2* rxl,
                                                     StepState* st, int idx, const float* __restrict__ w, float* xn,
                                                     int8_t* xq, int2* xm) {
    __shared__ float red[32];
    const int tid = threadIdx.x;
    const int slot = idx & 1;
    const unsigned e = epoch_of(st, idx);
    float v[5], o[5], hv[5];
#pragma unroll
    for (int k = 0; k < 5; k++) {
        const int i = tid + 1024 * k;
        hv[k] = h[i];
        o[k] = own[slot * 5120 + i];
    }
    bool ok = true;
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 5; k++) {
        const int i = tid + 1024 * k;
        const float r = wait_ll(rxl + slot * 5120 + i, e, ok);
        const float x = hv[k] + (o[k] + r);
        h_out[i] = x;
        v[k] = x;
        ss += x * x;
    }
    if (!ok) st->err = 5000 + idx;
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
__device__ __forceinline__ void d_gdn(const float* __restrict__ y, const float* __restrict__ yab, float* ring,
                                             const float* __restrict__ cw, const float* __restrict__ ssm_a,
                                             const float* __restrict__ ssm_dt, float* S, float* o,
                                             const StepState* st, int bx) {
    __shared__ float sq[128], sk[128], sv[32], red[8];
    const int vl = bx >> 2, sl = bx & 3, kl = vl & 7, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    DBG_T0
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
    // gate inputs (independent of the conv phase: issue early)
    const float yb = __ldcg(yab + 24 + vl), ya_ = __ldcg(yab + vl), dtv = ssm_dt[vl], av = ssm_a[vl];
    // v-channel conv inputs (warp 0 computes them after the q/k phase)
    const int chv = 2048 + vl * 128 + sl * 32 + (tid & 31);
    float vx = 0.f, vr0 = 0.f, vr1 = 0.f, vr2 = 0.f;
    float4 vw = make_float4(0.f, 0.f, 0.f, 0.f);
    if (tid < 32) {
        vx = __ldcg(y + chv);
        vr0 = ring[p0 * 5120 + chv];
        vr1 = ring[p1 * 5120 + chv];
        vr2 = ring[p2 * 5120 + chv];
        vw = __ldg((const float4*)(cw + (size_t)chv * 4));
    }
    // q (tid < 128) or k (tid >= 128) channel of local k head kl
    {
        const int ch = tid < 128 ? kl * 128 + tid : 1024 + kl * 128 + (tid - 128);
        const float x = __ldcg(y + ch);
        const float4 wc = __ldg((const float4*)(cw + (size_t)ch * 4));
        float sum = 0.f;
        sum += ring[p0 * 5120 + ch] * wc.x;
        sum += ring[p1 * 5120 + ch] * wc.y;
        sum += ring[p2 * 5120 + ch] * wc.z;
        sum += x * wc.w;
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
    DBG_PH(48, 0)
    if (tid < 32) {
        float sum = 0.f;
        sum += vr0 * vw.x;
        sum += vr1 * vw.y;
        sum += vr2 * vw.z;
        sum += vx * vw.w;
        sv[tid] = sum / (1.0f + expf(-sum));
        ring[pw * 5120 + chv] = vx;
    }
    __syncthreads();
    DBG_PH(48, 1)
    const float beta = 1.0f / (1.0f + expf(-yb));
    const float xg = ya_ + dtv;
    const float sp = xg > 20.0f ? xg : logf(1.0f + expf(xg));
    const float gv = expf(sp * av);
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
    DBG_PH(48, 2)
    DBG_N(48)
}

__global__ void __launch_bounds__(256) k_gdn(const float* __restrict__ y, const float* __restrict__ yab, float* ring,
                                             const float* __restrict__ cw, const float* __restrict__ ssm_a,
                                             const float* __restrict__ ssm_dt, float* S, float* o,
                                             const StepState* st) { d_gdn(y, yab, ring, cw, ssm_a, ssm_dt, S, o, st, blockIdx.x); }

// gdn, then the last of the 4 blocks of each head applies the gated RMSNorm + q8 for that head (same arithmetic as
// k_gnorm_q8: its 128-thread block_sum plus four zero warp partials)
__global__ void __launch_bounds__(256) k_gdn_gn(const float* __restrict__ y, const float* __restrict__ yab, float* ring,
                                                const float* __restrict__ cw, const float* __restrict__ ssm_a,
                                                const float* __restrict__ ssm_dt, float* S, float* o,
                                                const StepState* st, unsigned* cnt, const float* z,
                                                const float* __restrict__ gw, int8_t* xq, int2* xm) {
    const int vl = blockIdx.x >> 2, tid = threadIdx.x;
    const float zz = tid < 128 ? __ldcg(z + vl * 128 + tid) : 0.f;  // gated-norm inputs, loaded before the gdn work
    const float gwv = tid < 128 ? gw[tid] : 0.f;
    d_gdn(y, yab, ring, cw, ssm_a, ssm_dt, S, o, st, blockIdx.x);
    __shared__ int s_last;
    __shared__ float red2[8];
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned old = atomicAdd(cnt + vl, 1u);
        s_last = old == 3u;
        if (s_last) cnt[vl] = 0u;  // the next use is a later kernel
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    const float x = tid < 128 ? __ldcg(o + vl * 128 + tid) : 0.f;
    const float ss = block_sum(x * x, red2);
    if (tid >= 128) return;
    const float scale = rsqrtf(ss / 128.0f + 1e-6f);
    const float val = ((x * scale) * gwv) * (zz / (1.0f + expf(-zz)));
    quant_warp(val, xq + vl * 128 + tid, xm + vl * 4 + (tid >> 5));
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
__device__ __forceinline__ void d_attn_prep(const float* ya, const float* qw, const float* kw, float* qa, __half* kc, __half* vc,
                            int max_ctx, const StepState* st, float theta_scale, int bx) {
    __shared__ float red[8];
    __shared__ float yv[256];
    const int b = bx, d = threadIdx.x;
    DBG_T0
    const int pos = st->pos;
    const bool isq = b < 12;
    const float* src = isq ? ya + b * 512 : ya + 6144 + (b - 12) * 256;
    const float* w = isq ? qw : kw;
    const float x = __ldcg(src + d);
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
        vc[((size_t)kv * max_ctx + pos) * 256 + d] = __float2half_rn(__ldcg(ya + 6656 + kv * 256 + d));
    }
    DBG_PH(32, 0)
    DBG_N(32)
}

__global__ void k_attn_prep(const float* ya, const float* qw, const float* kw, float* qa, __half* kc, __half* vc,
                            int max_ctx, const StepState* st, float theta_scale) { d_attn_prep(ya, qw, kw, qa, kc, vc, max_ctx, st, theta_scale, blockIdx.x); }

// grid (2 kv heads, NSPLIT), 256 threads, 2 blocks/SM (one wave). Warps 0-3 serve q heads 0-2 of the kv head, warps
// 4-7 heads 3-5, each group striding over the block's chunk of positions (two positions in flight per warp).
// ws layout: [(j * NSPLIT + s) * 6 + h6] x 258 floats {m, l, acc[256]}
__device__ __forceinline__ void d_attn_split(const float* qa, const __half* kc, const __half* vc, float* ws,
                                                       int max_ctx, const StepState* st, int bx, int by) {
    __shared__ float sm_m[8][3], sm_l[8][3];
    __shared__ float sacc[8][256];
    const int j = bx, sidx = by;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, hg = warp >> 2, wq = warp & 3;
    DBG_T0
    const int n_kv = st->pos + 1;
    const int chunk = (n_kv + NSPLIT - 1) / NSPLIT;
    const int t0 = sidx * chunk, t1 = min(n_kv, t0 + chunk);
    DBG_PH(0, 0)
    float q[3][8];
#pragma unroll
    for (int hh = 0; hh < 3; hh++) {
        const float4* qp = (const float4*)(qa + (j * 6 + 3 * hg + hh) * 256 + lane * 8);
        const float4 a = __ldcg(qp), b = __ldcg(qp + 1);
        q[hh][0] = a.x; q[hh][1] = a.y; q[hh][2] = a.z; q[hh][3] = a.w;
        q[hh][4] = b.x; q[hh][5] = b.y; q[hh][6] = b.z; q[hh][7] = b.w;
    }
    DBG_PH(0, 1)
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
        kraw[0] = __ldcg((const uint4*)(K + (size_t)t * 256 + lane * 8));
        vraw[0] = __ldcg((const uint4*)(Vv + (size_t)t * 256 + lane * 8));
        if (two) {
            kraw[1] = __ldcg((const uint4*)(K + (size_t)(t + 4) * 256 + lane * 8));
            vraw[1] = __ldcg((const uint4*)(Vv + (size_t)(t + 4) * 256 + lane * 8));
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
    DBG_PH(0, 2)
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
    DBG_PH(0, 3)
    DBG_N(0)
}

__global__ void __launch_bounds__(256, 2) k_attn_split(const float* qa, const __half* kc, const __half* vc, float* ws,
                                                       int max_ctx, const StepState* st) { d_attn_split(qa, kc, vc, ws, max_ctx, st, blockIdx.x, blockIdx.y); }

// Split-K decode attention v2 (option attn2, default since M4 round 2): per block (kv head j, chunk of positions)
//   scores: warp per group of 4 positions, lanes hold 8 dims; the 6 heads x 4 positions partial dots (32 slots with
//           2 pad heads) are reduce-scattered across the warp in 31 shuffles (instead of a 5-shuffle sum per value)
//   softmax: warp per head over the chunk (max, exp, sum) -> ws m, l
//   P.V: warp per position subset, lanes hold 8 dims, all 6 heads share each V row; cross-warp sum via smem
// ws layout and the combine kernel are unchanged. Chunk <= ATT2_MAXCH positions (max_ctx 4096 / NSPLIT 40 -> 103).
constexpr int ATT2_MAXCH = 128;
// one reduce-scatter step: lanes with bit W set keep slots [W, 2W), the others [0, W); v[0..W) holds the result
template <int W>
__device__ __forceinline__ void rs_step(float* v, int lane) {
    const bool up = (lane & W) != 0;
#pragma unroll
    for (int i = 0; i < W; i++) {
        const float send = up ? v[i] : v[i + W];
        const float keep = up ? v[i + W] : v[i];
        v[i] = keep + __shfl_xor_sync(0xffffffffu, send, W);
    }
}
__global__ void __launch_bounds__(256, 2) k_attn_split2(const float* qa, const __half* kc, const __half* vc, float* ws,
                                                        int max_ctx, const StepState* st) {
    __shared__ float sS[8][ATT2_MAXCH];  // scores, then probabilities (heads 6, 7 unused)
    __shared__ float sacc[8][256];
    __shared__ float s_m[6], s_l[6];
    const int j = blockIdx.x, sidx = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    DBG_T0
    const int n_kv = st->pos + 1;
    const int chunk = (n_kv + NSPLIT - 1) / NSPLIT;
    const int t0 = sidx * chunk, t1 = min(n_kv, t0 + chunk);
    const int n = max(0, t1 - t0);
    if (chunk > ATT2_MAXCH) {  // max_ctx > 5120: unsupported chunk (the engine checks max_ctx at load)
        if (tid == 0) ((StepState*)st)->err = 8000;
        return;
    }
    const __half* K = kc + ((size_t)j * max_ctx + t0) * 256;
    const __half* Vv = vc + ((size_t)j * max_ctx + t0) * 256;
    // q for the 6 heads: 8 dims per lane (registers)
    float q[6][8];
#pragma unroll
    for (int h = 0; h < 6; h++) {
        const float4* qp = (const float4*)(qa + (j * 6 + h) * 256 + lane * 8);
        const float4 a = __ldcg(qp), b = __ldcg(qp + 1);
        q[h][0] = a.x; q[h][1] = a.y; q[h][2] = a.z; q[h][3] = a.w;
        q[h][4] = b.x; q[h][5] = b.y; q[h][6] = b.z; q[h][7] = b.w;
    }
    DBG_PH(0, 0)
    // ---- scores
    for (int g4 = warp * 4; g4 < n; g4 += 32) {
        uint4 kr[4];
#pragma unroll
        for (int u = 0; u < 4; u++)
            kr[u] = (g4 + u < n) ? __ldcg((const uint4*)(K + (size_t)(g4 + u) * 256 + lane * 8)) : make_uint4(0, 0, 0, 0);
        float v[32];
#pragma unroll
        for (int u = 0; u < 4; u++) {
            float k[8];
            const __half2* kh = (const __half2*)&kr[u];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float2 f = __half22float2(kh[i]);
                k[2 * i] = f.x; k[2 * i + 1] = f.y;
            }
#pragma unroll
            for (int h = 0; h < 8; h++) {
                float d = 0.f;
                if (h < 6) {
#pragma unroll
                    for (int i = 0; i < 8; i++) d += q[h][i] * k[i];
                }
                v[h * 4 + u] = d;
            }
        }
        // reduce-scatter: lane L ends with the full sum of slot L (head L >> 2, position g4 + (L & 3))
        rs_step<16>(v, lane);
        rs_step<8>(v, lane);
        rs_step<4>(v, lane);
        rs_step<2>(v, lane);
        rs_step<1>(v, lane);
        const int h = lane >> 2, u = lane & 3;
        if (h < 6 && g4 + u < n) sS[h][g4 + u] = v[0] * (1.0f / 16.0f);
    }
    __syncthreads();
    DBG_PH(0, 1)
    // ---- softmax per head (warps 0..5)
    if (warp < 6) {
        float m = -FLT_MAX;
        for (int t = lane; t < n; t += 32) m = fmaxf(m, sS[warp][t]);
        m = warp_max(m);
        float l = 0.f;
        for (int t = lane; t < n; t += 32) {
            const float p = expf(sS[warp][t] - m);
            sS[warp][t] = p;
            l += p;
        }
        l = warp_sum(l);
        if (lane == 0) { s_m[warp] = m; s_l[warp] = l; }
    }
    __syncthreads();
    // ---- P.V
    float acc[6][8];
#pragma unroll
    for (int h = 0; h < 6; h++)
#pragma unroll
        for (int i = 0; i < 8; i++) acc[h][i] = 0.f;
    for (int t = warp; t < n; t += 32) {
        uint4 vr[4];
#pragma unroll
        for (int u = 0; u < 4; u++)
            if (t + 8 * u < n) vr[u] = __ldcg((const uint4*)(Vv + (size_t)(t + 8 * u) * 256 + lane * 8));
#pragma unroll
        for (int u = 0; u < 4; u++) {
            if (t + 8 * u >= n) break;
            float v[8];
            const __half2* vh = (const __half2*)&vr[u];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float2 f = __half22float2(vh[i]);
                v[2 * i] = f.x; v[2 * i + 1] = f.y;
            }
#pragma unroll
            for (int h = 0; h < 6; h++) {
                const float p = sS[h][t + 8 * u];
#pragma unroll
                for (int i = 0; i < 8; i++) acc[h][i] += p * v[i];
            }
        }
    }
    DBG_PH(0, 2)
    // ---- cross-warp sum, write the split record per head
#pragma unroll
    for (int h = 0; h < 6; h++) {
#pragma unroll
        for (int i = 0; i < 8; i++) sacc[warp][lane * 8 + i] = acc[h][i];
        __syncthreads();
        float a = 0.f;
#pragma unroll
        for (int w = 0; w < 8; w++) a += sacc[w][tid];
        float* out = ws + ((size_t)(j * NSPLIT + sidx) * 6 + h) * 258;
        out[2 + tid] = a;
        if (tid == 0) {
            out[0] = n > 0 ? s_m[h] : -FLT_MAX;
            out[1] = n > 0 ? s_l[h] : 0.f;
        }
        __syncthreads();
    }
    DBG_PH(0, 3)
    DBG_N(0)
}

// Fused attention (option attnf, default since M4 round 2): q/k RMSNorm + RoPE + KV append, split-K flash decode
// and the split merge + sigmoid gate + q8 in one kernel. grid (2 kv heads, NSPLIT), 256 threads, 2 blocks/SM.
//   phase 1: warps 0-5 build the 6 q heads of kv head j in smem (warp per head); in the block whose chunk holds pos,
//            warp 6 appends k (norm + rope) and warp 7 appends v
//   phase 2: as d_attn_split, four positions in flight per warp
//   phase 3: the last block of kv head j (atomic counter) merges the splits for its 6 heads, warp per head
constexpr int ATT_MIN_CHUNK = 32;
__global__ void __launch_bounds__(256, 2) k_attn_fused(const float* ya, const float* __restrict__ qw,
                                                       const float* __restrict__ kw, __half* kc, __half* vc,
                                                       float* ws, int max_ctx, const StepState* st, float theta_scale,
                                                       unsigned* cnt, int8_t* xq, int2* xm) {
    __shared__ float sq[6][256];
    __shared__ float sm_m[8][3], sm_l[8][3];
    __shared__ float sacc[8][256];
    __shared__ int s_last;
    const int j = blockIdx.x, sidx = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, hg = warp >> 2, wq = warp & 3;
    const int pos = st->pos;
    const int n_kv = pos + 1;
    const int chunk = max(ATT_MIN_CHUNK, (n_kv + NSPLIT - 1) / NSPLIT);
    const int t0 = sidx * chunk, t1 = min(n_kv, t0 + chunk);
    const bool has_pos = t0 <= pos && pos < t1;
    __shared__ float s_cos[32], s_sin[32];
    if (tid < 32) {  // RoPE table for dims 0..31 (same expressions as k_attn_prep)
        const float theta = (float)pos * powf(theta_scale, (float)tid);
        s_cos[tid] = cosf(theta);
        s_sin[tid] = sinf(theta);
    }
    __syncthreads();
    // ---- phase 1
    if (warp < 6 || has_pos) {
        const bool isq = warp < 6;
        const float* src = isq ? ya + (6 * j + warp) * 512 : (warp == 6 ? ya + 6144 + j * 256 : ya + 6656 + j * 256);
        float x[8];
        {
            const float4 a = __ldcg((const float4*)(src + lane * 8)), b = __ldcg((const float4*)(src + lane * 8 + 4));
            x[0] = a.x; x[1] = a.y; x[2] = a.z; x[3] = a.w; x[4] = b.x; x[5] = b.y; x[6] = b.z; x[7] = b.w;
        }
        if (warp == 7) {
            __half* vd = vc + ((size_t)j * max_ctx + pos) * 256 + lane * 8;
#pragma unroll
            for (int i = 0; i < 8; i++) vd[i] = __float2half_rn(x[i]);
        } else {
            float ss = 0.f;
#pragma unroll
            for (int i = 0; i < 8; i++) ss += x[i] * x[i];
            ss = warp_sum(ss);
            const float scale = rsqrtf(ss / 256.0f + 1e-6f);
            const float* w = isq ? qw : kw;
            float y[8], pr[8];
#pragma unroll
            for (int i = 0; i < 8; i++) y[i] = (x[i] * scale) * w[lane * 8 + i];
#pragma unroll
            for (int i = 0; i < 8; i++) pr[i] = __shfl_xor_sync(0xffffffffu, y[i], 4);
            if (lane < 8) {  // rotary dims 0..63: d < 32 pairs with d + 32 (lane ^ 4)
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const int d = (lane & 3) * 8 + i;
                    const float c = s_cos[d], sn = s_sin[d];
                    y[i] = lane < 4 ? y[i] * c - pr[i] * sn : pr[i] * sn + y[i] * c;
                }
            }
            if (isq) {
#pragma unroll
                for (int i = 0; i < 8; i++) sq[warp][lane * 8 + i] = y[i];
            } else {
                __half* kd = kc + ((size_t)j * max_ctx + pos) * 256 + lane * 8;
#pragma unroll
                for (int i = 0; i < 8; i++) kd[i] = __float2half_rn(y[i]);
            }
        }
    }
    __syncthreads();
    // ---- phase 2
    float q[3][8];
#pragma unroll
    for (int hh = 0; hh < 3; hh++)
#pragma unroll
        for (int i = 0; i < 8; i++) q[hh][i] = sq[3 * hg + hh][lane * 8 + i];
    float m[3], l[3], acc[3][8];
#pragma unroll
    for (int hh = 0; hh < 3; hh++) {
        m[hh] = -FLT_MAX; l[hh] = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) acc[hh][i] = 0.f;
    }
    const __half* K = kc + (size_t)j * max_ctx * 256;
    const __half* Vv = vc + (size_t)j * max_ctx * 256;
    for (int t = t0 + wq; t < t1; t += 16) {
        uint4 kraw[4], vraw[4];
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const int tt = t + 4 * u;
            if (tt < t1) {
                kraw[u] = __ldcg((const uint4*)(K + (size_t)tt * 256 + lane * 8));
                vraw[u] = __ldcg((const uint4*)(Vv + (size_t)tt * 256 + lane * 8));
            }
        }
#pragma unroll
        for (int u = 0; u < 4; u++) {
            if (t + 4 * u >= t1) break;
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
    // ---- phase 3: the last block of kv head j merges
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned old = atomicAdd(cnt + j, 1u);
        s_last = old == (unsigned)(NSPLIT - 1);
        if (s_last) cnt[j] = 0u;
    }
    __syncthreads();
    if (!s_last || warp >= 6) return;
    __threadfence();
    const int nsp = (n_kv + chunk - 1) / chunk;
    const int h6 = warp, hl = 6 * j + h6;
    float M = -FLT_MAX;  // max over the splits: lanes load split lane and lane + 32 in parallel
    for (int s2 = lane; s2 < nsp; s2 += 32) M = fmaxf(M, __ldcg(ws + ((size_t)(j * NSPLIT + s2) * 6 + h6) * 258));
    M = warp_max(M);
    float num[8], den = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) num[i] = 0.f;
#pragma unroll 8
    for (int s2 = 0; s2 < nsp; s2++) {
        const float* p = ws + ((size_t)(j * NSPLIT + s2) * 6 + h6) * 258;
        const float wgt = expf(__ldcg(p) - M);
        den += wgt * __ldcg(p + 1);
        const float2* a2 = (const float2*)(p + 2 + lane * 8);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float2 a = __ldcg(a2 + i);
            num[2 * i] += wgt * a.x;
            num[2 * i + 1] += wgt * a.y;
        }
    }
    float val[8], amax = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const float att = num[i] / den;
        const float g = __ldcg(ya + hl * 512 + 256 + lane * 8 + i);
        val[i] = att * (1.0f / (1.0f + expf(-g)));
        amax = fmaxf(amax, fabsf(val[i]));
    }
    // q8 of the 32-group held by lanes 4g..4g+3 (same rule as quant_warp)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const float d = amax / 127.f;
    int s8 = 0;
    unsigned w0 = 0, w1 = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int qv = amax == 0.f ? 0 : (int)roundf(val[i] / d);
        s8 += qv;
        if (i < 4) w0 |= (unsigned)(qv & 0xff) << (8 * i);
        else w1 |= (unsigned)(qv & 0xff) << (8 * (i - 4));
    }
    *(uint2*)(xq + hl * 256 + lane * 8) = make_uint2(w0, w1);
    const int s16 = s8 + __shfl_xor_sync(0xffffffffu, s8, 1);  // lanes 4g, 4g+1: elements 0..15; 4g+2, 4g+3: 16..31
    const int s16b = __shfl_xor_sync(0xffffffffu, s16, 2);
    if ((lane & 3) == 0)
        xm[hl * 8 + (lane >> 2)] = make_int2(__float_as_int(d), (int)((unsigned)(s16 & 0xffff) | ((unsigned)s16b << 16)));
}

// 12 blocks (local q heads) x 256: merge splits, sigmoid gate, q8 for attn_output
__device__ __forceinline__ void d_attn_combine_q8(const float* ws, const float* ya, const StepState* st,
                                                         int8_t* xq, int2* xm, const Pf pf, int bx) {
    const int hl = bx, d = threadIdx.x, j = hl / 6, h6 = hl % 6;
    DBG_T0
    const int n_kv = st->pos + 1;
    const int chunk = (n_kv + NSPLIT - 1) / NSPLIT;
    const int nsp = (n_kv + chunk - 1) / chunk;  // splits with at least one position
    // warp 0: split weights exp(m_s - M) and the denominator, in parallel over the splits (lane s, s + 32)
    __shared__ float s_w[NSPLIT];
    __shared__ float s_den;
    if (d < 32) {
        float mv[2] = {-FLT_MAX, -FLT_MAX}, lv[2] = {0.f, 0.f};
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int s = d + 32 * u;
            if (s < nsp) {
                const float* p = ws + ((size_t)(j * NSPLIT + s) * 6 + h6) * 258;
                mv[u] = __ldcg(p);
                lv[u] = __ldcg(p + 1);
            }
        }
        const float M = warp_max(fmaxf(mv[0], mv[1]));
        float den = 0.f;
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int s = d + 32 * u;
            if (s < nsp) {
                const float w = expf(mv[u] - M);
                s_w[s] = w;
                den += w * lv[u];
            }
        }
        den = warp_sum(den);
        if (d == 0) s_den = den;
    }
    __syncthreads();
    DBG_PH(16, 0)
    float num = 0.f;
#pragma unroll 8
    for (int s = 0; s < nsp; s++) num += s_w[s] * __ldcg(ws + ((size_t)(j * NSPLIT + s) * 6 + h6) * 258 + 2 + d);
    const float den = s_den;
    DBG_PH(16, 1)
    const float att = num / den;
    const float g = __ldcg(ya + hl * 512 + 256 + d);
    const float val = att * (1.0f / (1.0f + expf(-g)));
    quant_warp(val, xq + hl * 256 + d, xm + hl * 8 + (d >> 5));
    DBG_PH(16, 2)
    DBG_N(16)
}

__global__ void __launch_bounds__(256) k_attn_combine_q8(const float* ws, const float* ya, const StepState* st,
                                                         int8_t* xq, int2* xm, const Pf pf) {
    T4Q_PF_BLOCKS(12)
    d_attn_combine_q8(ws, ya, st, xq, xm, pf, blockIdx.x);
}

constexpr int NBA = 160;
__device__ __forceinline__ void d_argmax_part(const float* x, int n, float* apart, int bx, int nb) {
    __shared__ float sv[256];
    __shared__ int si[256];
    const int per = (n + nb - 1) / nb;
    const int lo = bx * per, hi = min(n, lo + per);
    float bv = -FLT_MAX;
    int bi = 0x7fffffff;
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const float v = __ldcg(x + i);
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
        apart[bx] = sv[0];
        apart[nb + bx] = __int_as_float(si[0]);
    }
}
__global__ void __launch_bounds__(256) k_argmax_part(const float* x, int n, float* apart) {
    d_argmax_part(x, n, apart, blockIdx.x, NBA);
}

__device__ __forceinline__ void d_argmax_final(const float* apart, int nb, int row0, float* amb, const unsigned* aflag,
                                               float* peer_amb, unsigned* peer_aflag, StepState* st, int* ring) {
    if (threadIdx.x >= 32) return;
    float bv = -FLT_MAX;  // warp scan of the partials (max value, lowest index on ties: same result as a serial scan)
    int bi = 0x7fffffff;
    for (int b = threadIdx.x; b < nb; b += 32) {
        const float v = __ldcg(apart + b);
        const int i = __float_as_int(__ldcg(apart + nb + b));
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (threadIdx.x != 0) return;
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
__global__ void k_argmax_final(const float* apart, int row0, float* amb, const unsigned* aflag, float* peer_amb,
                               unsigned* peer_aflag, StepState* st, int* ring) {
    d_argmax_final(apart, NBA, row0, amb, aflag, peer_amb, peer_aflag, st, ring);
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

// L2 warm-up experiment kernels: touch n bytes with real loads (cg) or prefetch instructions
__global__ void k_touch(const uint8_t* p, size_t n, int mode, float* sink) {
    unsigned acc = 0;
    for (size_t off = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 32; off < n;
         off += (size_t)gridDim.x * blockDim.x * 32) {
        if (mode == 0) acc ^= __ldcg((const unsigned*)(p + off));
        else asm volatile("prefetch.global.L2 [%0];" ::"l"(p + off));
    }
    if (acc == 0x12345678u) sink[0] = 1.f;
}
void touch(const uint8_t* p, size_t n, int mode, float* sink, cudaStream_t s) {
    k_touch<<<80, 256, 0, s>>>(p, n, mode, sink);
}

void pull(const unsigned* hflag, const float* hrx, float* rx, StepState* st, int idx, cudaStream_t s,
          const float* own, float* pub_peer_rx, unsigned* pub_peer_flag) {
    k_pull<<<1, 640, 0, s>>>(hflag, hrx, rx, st, idx, own, pub_peer_rx, pub_peer_flag);
}

void embed(const uint8_t* embd, StepState* st, const int* prompt, float* h, cudaStream_t s, const Pf* pf) {
    const Pf P = pf ? *pf : Pf{};
    k_embed<<<20 + P.blocks, 256, 0, s>>>(embd, st, prompt, h, P);
}

int g_arn = 0;  // 0: multi-block ar_norm (default), 1: single 1024-thread block (M4 round 1)
void set_arn(int v) { g_arn = v; }

void ar_norm(const float* h, float* h_out, const float* own, const float* rx, const unsigned* flag,
             const StepState* st, int idx, const float* w, float* xn, int8_t* xq, int2* xm, cudaStream_t s,
             const Pf* pf, float* pub_peer_rx, unsigned* pub_peer_flag) {
    const Pf P = pf ? *pf : Pf{};
    if (g_arn == 0) {
        const int nb = ARN_BLOCKS + P.blocks;
        if (flag)
            k_ar_norm_mb<true><<<nb, 256, 0, s>>>(h, h_out, own, rx, flag, (StepState*)st, idx, w, xn, xq, xm, P,
                                                  pub_peer_rx, pub_peer_flag);
        else
            k_ar_norm_mb<false><<<nb, 256, 0, s>>>(h, h_out, own, rx, flag, (StepState*)st, idx, w, xn, xq, xm, P,
                                                   pub_peer_rx, pub_peer_flag);
        return;
    }
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

void gdn_gn(const float* y, const float* yab, float* ring, const float* conv_w, const float* ssm_a, const float* ssm_dt,
            float* S, float* o, const StepState* st, unsigned* cnt, const float* z, const float* gw, int8_t* xq,
            int2* xm, cudaStream_t s) {
    k_gdn_gn<<<96, 256, 0, s>>>(y, yab, ring, conv_w, ssm_a, ssm_dt, S, o, st, cnt, z, gw, xq, xm);
}

void ar_norm_mf(const float* h, float* h_out, const float* own, const float* rx, const unsigned* tflag,
                const StepState* st, int idx, const float* w, float* xn, int8_t* xq, int2* xm, cudaStream_t s,
                float* peer_rx, unsigned* peer_tflag) {
    k_ar_norm_mb<false, true><<<ARN_BLOCKS, 256, 0, s>>>(h, h_out, own, rx, tflag, (StepState*)st, idx, w, xn, xq, xm,
                                                         Pf{}, peer_rx, peer_tflag);
}

void pull_norm(const float* h, float* h_out, const float* own, const float* hrx, const unsigned* htflag,
               const StepState* st, int idx, const float* w, float* xn, int8_t* xq, int2* xm, cudaStream_t s,
               float* peer_hrx, unsigned* peer_htflag, float2* ssb) {
    k_pull_norm<<<ARN_BLOCKS, 256, 0, s>>>(h, h_out, own, hrx, htflag, (StepState*)st, idx, w, xn, xq, xm, peer_hrx,
                                           peer_htflag, ssb);
}

void ar_norm_ll(const float* h, float* h_out, const float* own, const float2* rxl, const StepState* st, int idx,
                const float* w, float* xn, int8_t* xq, int2* xm, cudaStream_t s) {
    k_ar_norm_ll<<<1, 1024, 0, s>>>(h, h_out, own, rxl, (StepState*)st, idx, w, xn, xq, xm);
}

void set_spin_ns(int ns) { cudaMemcpyToSymbol(d_spin_ns, &ns, sizeof ns); }
void set_dbg(unsigned long long* p) { cudaMemcpyToSymbol(d_dbg, &p, sizeof p); }

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

int g_attn2 = 0;
void set_attn2(int v) { g_attn2 = v; }
void attn_split(const float* qa, const uint16_t* kc, const uint16_t* vc, float* ws, int max_ctx, const StepState* st,
                cudaStream_t s) {
    if (g_attn2) {
        k_attn_split2<<<dim3(2, NSPLIT), 256, 0, s>>>(qa, (const __half*)kc, (const __half*)vc, ws, max_ctx, st);
        return;
    }
    k_attn_split<<<dim3(2, NSPLIT), 256, 0, s>>>(qa, (const __half*)kc, (const __half*)vc, ws, max_ctx, st);
}

void attn_combine_q8(const float* ws, const float* ya, const StepState* st, int8_t* xq, int2* xm, cudaStream_t s,
                     const Pf* pf) {
    const Pf P = pf ? *pf : Pf{};
    k_attn_combine_q8<<<12 + P.blocks, 256, 0, s>>>(ws, ya, st, xq, xm, P);
}

void attn_fused(const float* ya, const float* qw, const float* kw, uint16_t* kc, uint16_t* vc, float* ws, int max_ctx,
                const StepState* st, float theta_scale, unsigned* cnt, int8_t* xq, int2* xm, cudaStream_t s) {
    k_attn_fused<<<dim3(2, NSPLIT), 256, 0, s>>>(ya, qw, kw, (__half*)kc, (__half*)vc, ws, max_ctx, st, theta_scale, cnt,
                                                  xq, xm);
}

void argmax_step(const float* logits, int n, int row0, float* apart, float* amb, const unsigned* aflag,
                 float* peer_amb, unsigned* peer_aflag, StepState* st, int* ring, cudaStream_t s) {
    k_argmax_part<<<NBA, 256, 0, s>>>(logits, n, apart);
    k_argmax_final<<<1, 32, 0, s>>>(apart, row0, amb, aflag, peer_amb, peer_aflag, st, ring);
}


// ================================================================================================ persistent layer kernels
namespace {

// sense-reversal grid barrier; requires all blocks co-resident. bar[0] = arrivals, bar[1] = generation
__device__ __forceinline__ void grid_sync(unsigned* bar, StepState* st, int code) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned gen = ld_vol_u32(bar + 1);
        __threadfence();
        const unsigned old = atomicAdd(bar, 1u);
        if (old == gridDim.x - 1) {
            atomicExch(bar, 0u);
            __threadfence();
            atomicAdd(bar + 1, 1u);
        } else {
            const unsigned long long t0 = gtimer();
            unsigned spins = 0;
            while (ld_vol_u32(bar + 1) == gen) {
                if ((++spins & 1023u) == 0 && gtimer() - t0 > WATCHDOG_NS) {
                    st->err = code;
                    break;
                }
            }
        }
        __threadfence();
    }
    __syncthreads();
}

__device__ __forceinline__ void x_ready_set(unsigned* f, unsigned v) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) st_vol_u32(f, v);
}
__device__ __forceinline__ void x_ready_wait(const unsigned* f, unsigned v, StepState* st, int code, int* s_flag) {
    if (threadIdx.x == 0) {
        *s_flag = wait_flag(f, v);
        if (!*s_flag) st->err = code;
        __threadfence();
    }
    __syncthreads();
}
// copy q8 x planes (ng groups) and optionally fp32 x from global (written during this kernel: L2 only)
__device__ __forceinline__ void x_load(const int8_t* gxq, const int2* gxm, int ng, int8_t* s_lo, int8_t* s_hi,
                                       int2* s_mt, const float* gxf, float* s_xf, int nf) {
    for (int g = threadIdx.x; g < ng; g += blockDim.x) {
        ((int4*)s_lo)[g] = __ldcg((const int4*)gxq + 2 * g);
        ((int4*)s_hi)[g] = __ldcg((const int4*)gxq + 2 * g + 1);
        s_mt[g] = __ldcg(gxm + g);
    }
    if (s_xf)
        for (int i = threadIdx.x; i < nf / 4; i += blockDim.x) ((float4*)s_xf)[i] = __ldcg((const float4*)gxf + i);
    __syncthreads();
}

// block 0: [publish own partial,] wait for the peer, x = h + own + rx (or h), RMSNorm * nw, q8 -> smem planes and
// global gxq/gxm, fp32 normalized x -> xn (and s_xf if given). 256 threads, K = 5120.
__device__ void leader_arnorm(const ProArgs& pa, float* s_xf, int8_t* s_lo, int8_t* s_hi, int2* s_mt, float* red,
                              int* s_flag, int8_t* gxq, int2* gxm, float* xn) {
    const int tid = threadIdx.x;
    const unsigned ep = epoch_of(pa.st, pa.idx);
    if (pa.pub_peer_flag) publish_partial(pa.own, pa.pub_peer_rx, pa.pub_peer_flag, ep);
    if (pa.flag) {
        if (tid == 0) {
            *s_flag = wait_flag(pa.flag, ep);
            if (!*s_flag) pa.st->err = 1000 + pa.idx;
        }
        __syncthreads();
    }
    float* xf = s_xf ? s_xf : xn;
    float4* sx4 = (float4*)xf;
    float ss = 0.f;
    float4 xv[5];
#pragma unroll
    for (int k = 0; k < 5; k++) xv[k] = ((const float4*)pa.h_in)[tid + 256 * k];
    if (pa.add) {
        float4 ov[5], rv[5];
#pragma unroll
        for (int k = 0; k < 5; k++) {
            ov[k] = __ldcg((const float4*)pa.own + tid + 256 * k);
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
    ss = block_sum(ss, red);
    const float scale = rsqrtf(ss / 5120.f + 1e-6f);
    if (tid < 160) {
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
        ((int4*)gxq)[2 * tid] = ((const int4*)s_lo)[tid];
        ((int4*)gxq)[2 * tid + 1] = ((const int4*)s_hi)[tid];
        gxm[tid] = s_mt[tid];
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const float4 y = make_float4(v[4 * e], v[4 * e + 1], v[4 * e + 2], v[4 * e + 3]);
            sx4[tid * 8 + e] = y;
            if (s_xf) ((float4*)xn)[tid * 8 + e] = y;
        }
    }
}

// block 0: gated RMSNorm of o (24 heads x 128) with z -> q8 x (96 groups) to smem planes and global
__device__ void leader_gnorm(const float* o, const float* z, const float* gw, int8_t* s_lo, int8_t* s_hi, int2* s_mt,
                             float* red, int8_t* gxq, int2* gxm) {
    const int tid = threadIdx.x;
    float ov[32];
    if (tid < 96) {
        const float4* o4 = (const float4*)o + tid * 8;
        float sq = 0.f;
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const float4 x = __ldcg(o4 + e);
            ov[4 * e] = x.x; ov[4 * e + 1] = x.y; ov[4 * e + 2] = x.z; ov[4 * e + 3] = x.w;
            sq += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
        }
        red[tid] = sq;
    }
    __syncthreads();
    if (tid < 96) {
        const int hb = tid & ~3;
        const float tot = ((red[hb] + red[hb + 1]) + red[hb + 2]) + red[hb + 3];
        const float scale = rsqrtf(tot / 128.0f + 1e-6f);
        const float4* z4 = (const float4*)z + tid * 8;
        const float4* w4 = (const float4*)gw + (tid & 3) * 8;
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const float4 zz = __ldcg(z4 + e), wv = __ldg(w4 + e);
            const float* zp = (const float*)&zz;
            const float* wp = (const float*)&wv;
#pragma unroll
            for (int q = 0; q < 4; q++)
                ov[4 * e + q] = ((ov[4 * e + q] * scale) * wp[q]) * (zp[q] / (1.0f + expf(-zp[q])));
        }
        quant_group(ov, tid, s_lo, s_hi, s_mt);
        ((int4*)gxq)[2 * tid] = ((const int4*)s_lo)[tid];
        ((int4*)gxq)[2 * tid + 1] = ((const int4*)s_hi)[tid];
        gxm[tid] = s_mt[tid];
    }
}

// warp tiles of a GEMV on the persistent grid (blocked: warp w owns tiles [w * tpw, +tpw))
template <int FMT, int RPL, int NCH>
__device__ __forceinline__ bool gemv_preload(WChunk<FMT, RPL>* w, const GemvArgs& a, int nseg) {
    const int nw = gridDim.x * (blockDim.x >> 5), ntot = a.ntiles + nseg;
    const int tpw = (ntot + nw - 1) / nw;
    const int tbeg = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * tpw;
    if (tbeg >= min(ntot, tbeg + tpw) || tbeg >= a.ntiles) return false;
#pragma unroll
    for (int c = 0; c < 2; ++c)
        if (c < NCH) load_chunk<FMT, RPL, NCH>(w[c], a, tbeg, c, threadIdx.x & 31);
    return true;
}

template <int FMT, int RPL, int NCH, bool SEG, bool SQ, bool ROWS>
__device__ __forceinline__ void gemv_tiles(const GemvArgs& a, const SegArgs& sg, WChunk<FMT, RPL>* w, bool pre,
                                           const int8_t* s_lo, const int8_t* s_hi, const int2* s_mt,
                                           const float* s_xf, float* sy, float* sa, int8_t* sq_xq, int2* sq_xm,
                                           float* y_peer) {
    constexpr int D = 2;
    constexpr int CVT = (FMT == FAST_P4 || FMT == FAST_Q8) ? 1 : 0;
    const int tid = threadIdx.x, lane = tid & 31, h = lane >> 4, j = lane & 15, wib = tid >> 5;
    const int wpb = blockDim.x >> 5, nw = gridDim.x * wpb;
    const int ntot = a.ntiles + (SEG ? sg.nrows : 0);
    const int tpw = (ntot + nw - 1) / nw;
    const int tbeg = (blockIdx.x * wpb + wib) * tpw, tend = min(ntot, tbeg + tpw);
    for (int tile = tbeg; tile < tend; tile++) {
        if (SEG && tile >= a.ntiles) {
            const int row = tile - a.ntiles;
            const float4* w4 = (const float4*)(sg.w + (size_t)row * 5120);
            const float4* x4 = (const float4*)s_xf;
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
        float acc[RPL];
#pragma unroll
        for (int r = 0; r < RPL; ++r) acc[r] = 0.f;
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
            const int4 xl = ((const int4*)s_lo)[kb], xh = ((const int4*)s_hi)[kb];
            const int2 mt = s_mt[kb];
            const float xd = __int_as_float(mt.x);
            const int s0 = (int)(short)(mt.y & 0xffff), s1 = mt.y >> 16;
            const int moff = FMT == FAST_P4 ? 0x4B400000 - 8 * (s0 + s1) : 0x4B400000;
#pragma unroll
            for (int r = 0; r < RPL; ++r) acc[r] += group_dot_r<FMT, CVT, RPL>(cur, r, xl, xh, xd, s0, s1, moff);
        }
#pragma unroll
        for (int r = 0; r < RPL; ++r) {
            float v = acc[r];
            v += __shfl_xor_sync(0xffffffffu, v, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 2);
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            const int row = tile * 2 * RPL + h * RPL + r;
            if (SQ) {
                const float u = __shfl_xor_sync(0xffffffffu, v, 16);
                if (lane == 0) sa[(tile - blockIdx.x * wpb * tpw) * RPL + r] = (v / (1.0f + expf(-v))) * u;
            }
            if (j == 0 && row < a.N) {
                a.y[row] = v;
                if (ROWS) sy[(tile - blockIdx.x * wpb * tpw) * 2 * RPL + h * RPL + r] = v;
            }
        }
    }
    if (SQ) {
        __syncthreads();
        const int ng = wpb * tpw * RPL / 32;
        if (blockIdx.x * wpb * tpw < a.ntiles && wib < ng)
            quant_warp(sa[wib * 32 + lane], sq_xq + (blockIdx.x * ng + wib) * 32 + lane, sq_xm + blockIdx.x * ng + wib);
    }
    if (ROWS) {
        __syncthreads();
        const int nrow = wpb * tpw * 2 * RPL;
        for (int t = tid; t < nrow / 4; t += blockDim.x) {
            const int row0 = blockIdx.x * nrow + t * 4;
            if (row0 < a.N) *(float4*)(y_peer + row0) = *(const float4*)(sy + t * 4);
        }
    }
}

__device__ __forceinline__ unsigned xready_val(const StepState* st, int xid) {
    return *(volatile const uint32_t*)&st->step * 1024u + (unsigned)xid + 1u;
}

__global__ void __launch_bounds__(256, 2) k_mega_dn(const MegaDN m) {
    __shared__ __align__(16) int8_t s_lo[160 * 16];
    __shared__ __align__(16) int8_t s_hi[160 * 16];
    __shared__ int2 s_mt[160];
    __shared__ __align__(16) float s_xf[5120];
    __shared__ __align__(16) float sy[256];
    __shared__ float red[128];
    __shared__ int s_flag;
    StepState* st = m.c.st;
    const unsigned xv = xready_val(st, m.c.xid);
    // A: AR-in + attn_norm (block 0), qkvz + alpha/beta rows
    WChunk<FAST_P4, 2> wq[2];
    const bool pq = gemv_preload<FAST_P4, 2, 10>(wq, m.qkvz, m.ab.nrows);
    if (blockIdx.x == 0) {
        leader_arnorm(m.c.in, s_xf, s_lo, s_hi, s_mt, red, &s_flag, m.c.gxq, m.c.gxm, m.c.xn);
        x_ready_set(m.c.xrdy, xv);
    } else {
        x_ready_wait(m.c.xrdy, xv, st, 5100, &s_flag);
        x_load(m.c.gxq, m.c.gxm, 160, s_lo, s_hi, s_mt, m.c.xn, s_xf, 5120);
    }
    gemv_tiles<FAST_P4, 2, 10, true, false, false>(m.qkvz, m.ab, wq, pq, s_lo, s_hi, s_mt, s_xf, nullptr, nullptr,
                                                   nullptr, nullptr, nullptr);
    grid_sync(m.c.bar, st, 5101);
    // B: DeltaNet recurrence (96 items)
    for (int it = blockIdx.x; it < 96; it += gridDim.x) {
        d_gdn(m.y, m.yab, m.ring, m.ring_w, m.ssm_a, m.ssm_dt, m.S, m.o, st, it);
        __syncthreads();
    }
    WChunk<FAST_K5, 2> wk[2];
    const bool pk = gemv_preload<FAST_K5, 2, 6>(wk, m.ssm_out, 0);
    grid_sync(m.c.bar, st, 5102);
    // C: gated norm (block 0) -> ssm_out, rows to the peer
    if (blockIdx.x == 0) {
        leader_gnorm(m.o, m.y + 5120, m.ssm_norm, s_lo, s_hi, s_mt, red, m.c.gxq, m.c.gxm);
        x_ready_set(m.c.xrdy, xv + 1);
    } else {
        x_ready_wait(m.c.xrdy, xv + 1, st, 5103, &s_flag);
        x_load(m.c.gxq, m.c.gxm, 96, s_lo, s_hi, s_mt, nullptr, nullptr, 0);
    }
    gemv_tiles<FAST_K5, 2, 6, false, false, true>(m.ssm_out, SegArgs{}, wk, pk, s_lo, s_hi, s_mt, nullptr, sy, nullptr,
                                                  nullptr, nullptr, m.y_peer);
}

template <int DF>
__global__ void __launch_bounds__(256, 2) k_mega_ffn(const MegaFFN m) {
    __shared__ __align__(16) int8_t s_lo[272 * 16];
    __shared__ __align__(16) int8_t s_hi[272 * 16];
    __shared__ int2 s_mt[272];
    __shared__ __align__(16) float sy[256];
    __shared__ float sa[256];
    __shared__ float red[128];
    __shared__ int s_flag;
    StepState* st = m.c.st;
    const unsigned xv = xready_val(st, m.c.xid);
    WChunk<FAST_P4, 4> wg[2];
    const bool pg = gemv_preload<FAST_P4, 4, 10>(wg, m.gateup, 0);
    if (blockIdx.x == 0) {
        leader_arnorm(m.c.in, nullptr, s_lo, s_hi, s_mt, red, &s_flag, m.c.gxq, m.c.gxm, m.c.xn);
        x_ready_set(m.c.xrdy, xv);
    } else {
        x_ready_wait(m.c.xrdy, xv, st, 5200, &s_flag);
        x_load(m.c.gxq, m.c.gxm, 160, s_lo, s_hi, s_mt, nullptr, nullptr, 0);
    }
    gemv_tiles<FAST_P4, 4, 10, false, true, false>(m.gateup, SegArgs{}, wg, pg, s_lo, s_hi, s_mt, nullptr, nullptr, sa,
                                                   m.xq2, m.xm2, nullptr);
    WChunk<DF, 4> wd[2];
    const bool pd = gemv_preload<DF, 4, 17>(wd, m.down, 0);
    grid_sync(m.c.bar, st, 5201);
    x_load(m.xq2, m.xm2, 272, s_lo, s_hi, s_mt, nullptr, nullptr, 0);
    gemv_tiles<DF, 4, 17, false, false, true>(m.down, SegArgs{}, wd, pd, s_lo, s_hi, s_mt, nullptr, sy, nullptr, nullptr,
                                              nullptr, m.y_peer);
}

__global__ void __launch_bounds__(256, 2) k_mega_attn(const MegaAttn m) {
    __shared__ __align__(16) int8_t s_lo[160 * 16];
    __shared__ __align__(16) int8_t s_hi[160 * 16];
    __shared__ int2 s_mt[160];
    __shared__ __align__(16) float sy[256];
    __shared__ float red[128];
    __shared__ int s_flag;
    StepState* st = m.c.st;
    const unsigned xv = xready_val(st, m.c.xid);
    WChunk<FAST_P4, 2> wq[2];
    const bool pq = gemv_preload<FAST_P4, 2, 10>(wq, m.qkv, 0);
    if (blockIdx.x == 0) {
        leader_arnorm(m.c.in, nullptr, s_lo, s_hi, s_mt, red, &s_flag, m.c.gxq, m.c.gxm, m.c.xn);
        x_ready_set(m.c.xrdy, xv);
    } else {
        x_ready_wait(m.c.xrdy, xv, st, 5300, &s_flag);
        x_load(m.c.gxq, m.c.gxm, 160, s_lo, s_hi, s_mt, nullptr, nullptr, 0);
    }
    gemv_tiles<FAST_P4, 2, 10, false, false, false>(m.qkv, SegArgs{}, wq, pq, s_lo, s_hi, s_mt, nullptr, nullptr,
                                                    nullptr, nullptr, nullptr, nullptr);
    grid_sync(m.c.bar, st, 5301);
    for (int it = blockIdx.x; it < 14; it += gridDim.x) {
        d_attn_prep(m.ya, m.qw, m.kw, m.qa, (__half*)m.kc, (__half*)m.vc, m.max_ctx, st, m.theta_scale, it);
        __syncthreads();
    }
    grid_sync(m.c.bar, st, 5302);
    for (int it = blockIdx.x; it < 2 * NSPLIT; it += gridDim.x) {
        d_attn_split(m.qa, (const __half*)m.kc, (const __half*)m.vc, m.ws, m.max_ctx, st, it / NSPLIT, it % NSPLIT);
        __syncthreads();
    }
    WChunk<FAST_P4, 4> wo[2];
    const bool po = gemv_preload<FAST_P4, 4, 6>(wo, m.wo, 0);
    grid_sync(m.c.bar, st, 5303);
    for (int it = blockIdx.x; it < 12; it += gridDim.x) {
        d_attn_combine_q8(m.ws, m.ya, st, m.c.gxq, m.c.gxm, Pf{}, it);
        __syncthreads();
    }
    grid_sync(m.c.bar, st, 5304);
    x_load(m.c.gxq, m.c.gxm, 96, s_lo, s_hi, s_mt, nullptr, nullptr, 0);
    gemv_tiles<FAST_P4, 4, 6, false, false, true>(m.wo, SegArgs{}, wo, po, s_lo, s_hi, s_mt, nullptr, sy, nullptr,
                                                  nullptr, nullptr, m.y_peer);
}

__global__ void __launch_bounds__(256, 2) k_mega_head(const MegaHead m) {
    __shared__ __align__(16) int8_t s_lo[160 * 16];
    __shared__ __align__(16) int8_t s_hi[160 * 16];
    __shared__ int2 s_mt[160];
    __shared__ float red[128];
    __shared__ int s_flag;
    StepState* st = m.c.st;
    const unsigned xv = xready_val(st, m.c.xid);
    WChunk<FAST_K6, 2> wl[2];
    const bool pl = gemv_preload<FAST_K6, 2, 10>(wl, m.lm, 0);
    if (blockIdx.x == 0) {
        leader_arnorm(m.c.in, nullptr, s_lo, s_hi, s_mt, red, &s_flag, m.c.gxq, m.c.gxm, m.c.xn);
        x_ready_set(m.c.xrdy, xv);
    } else {
        x_ready_wait(m.c.xrdy, xv, st, 5400, &s_flag);
        x_load(m.c.gxq, m.c.gxm, 160, s_lo, s_hi, s_mt, nullptr, nullptr, 0);
    }
    gemv_tiles<FAST_K6, 2, 10, false, false, false>(m.lm, SegArgs{}, wl, pl, s_lo, s_hi, s_mt, nullptr, nullptr, nullptr,
                                                    nullptr, nullptr, nullptr);
    grid_sync(m.c.bar, st, 5401);
    d_argmax_part(m.logits, 124160, m.apart, blockIdx.x, gridDim.x);
    grid_sync(m.c.bar, st, 5402);
    if (blockIdx.x == 0)
        d_argmax_final(m.apart, gridDim.x, m.row0, m.amb, m.aflag, m.peer_amb, m.peer_aflag, st, m.ring);
}

template <class KF>
int mega_grid(KF kf) {
    static int cap[8] = {0};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 8 && !cap[dev]) {
        cudaFuncSetAttribute(kf, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        int nb = 0, nsm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kf, 256, 0);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        cap[dev] = nb * nsm;
    }
    return cap[dev < 8 ? dev : 0];
}

}  // namespace

int mega_capacity() {
    int c = 1 << 30;
    c = std::min(c, mega_grid(k_mega_dn));
    c = std::min(c, mega_grid(k_mega_ffn<FAST_P4>));
    c = std::min(c, mega_grid(k_mega_ffn<FAST_P4M>));
    c = std::min(c, mega_grid(k_mega_attn));
    c = std::min(c, mega_grid(k_mega_head));
    return c;
}
void mega_dn(const MegaDN& m, int grid, cudaStream_t s) { k_mega_dn<<<grid, 256, 0, s>>>(m); }
void mega_ffn(const MegaFFN& m, int down_fmt, int grid, cudaStream_t s) {
    if (down_fmt == FAST_P4M) k_mega_ffn<FAST_P4M><<<grid, 256, 0, s>>>(m);
    else k_mega_ffn<FAST_P4><<<grid, 256, 0, s>>>(m);
}
void mega_attn(const MegaAttn& m, int grid, cudaStream_t s) { k_mega_attn<<<grid, 256, 0, s>>>(m); }
void mega_head(const MegaHead& m, int grid, cudaStream_t s) { k_mega_head<<<grid, 256, 0, s>>>(m); }

}  // namespace tp
