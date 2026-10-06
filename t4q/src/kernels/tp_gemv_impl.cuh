// Shared TP decode device helpers + the fast GEMV kernel template (k_gemv). Included by tp_kernels.cu and
// tp_spec.cu; everything is in an anonymous namespace (one copy per translation unit).
#pragma once
#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "tp_kernels.h"

namespace tp {
namespace {
using namespace t4q::gemv;


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
// the AR-wait watchdog as a device global (the r13 deferred 8000-race fix): the default is the
// old 4 s; the spec driver extends it to 30 s for a fresh process's FIRST spec step only (the
// v42 flake: the previous process's driver-global CUDA teardown races this process's first spec
// graph launches, and the first AR wait tripped the 4-s watchdog, st.err = 8000+idx). Load once
// per wait_flag call; it is only ever set/restored between steps, never mid-wait.
__device__ unsigned long long d_watchdog_ns = WATCHDOG_NS;
// spin until flag >= e (wrap-safe); returns false on watchdog
__device__ __forceinline__ bool wait_flag(const unsigned* flag, unsigned e) {
    const unsigned long long t0 = gtimer();
    const unsigned long long wd = d_watchdog_ns;
    unsigned spins = 0;
    const int ns = d_spin_ns;
    while ((int)(ld_vol_u32(flag) - e) < 0) {
        if (ns) __nanosleep(ns);
        if ((++spins & 1023u) == 0 && gtimer() - t0 > wd) return false;
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

// M > 1 (spec verify / MTP catch-up, tp_spec.cu): M activation columns (xq + col * K, xm + col * K / 32,
// y + col * ldy; SEG: x + col * 5120, y + col * 64; SQ: sq_xq + col * N / 2, sq_xm + col * N / 64). Every column's
// arithmetic is the M = 1 code path, so each column is bit-identical to a single-token GEMV. M > 1 supports
// PRO_NONE without AR only.
template <int FMT, int RPL, int NCH, bool AR, bool SEG, int PRO, bool SQ = false, int CVX = 1, int M = 1>
__global__ void __launch_bounds__(256, 2) k_gemv(const GemvArgs a, const ArArgs ar, const SegArgs sg, const ProArgs pa,
                                                 int tpw) {
    constexpr int D = 2;
    static_assert(M == 1 || (!AR && PRO == PRO_NONE), "multi-column GEMV: PRO_NONE without AR only");
    constexpr int CVT = FMT == FAST_P4 ? CVX : FMT == FAST_Q8 ? 1 : 0;  // CVX 2: P4 unsigned high-nibble dp4a
    constexpr int NB = NCH * 16;
    constexpr int K = NCH * 512;
    const int tid = threadIdx.x, lane = tid & 31, h = lane >> 4, j = lane & 15, wib = tid >> 5;
    const int wpb = blockDim.x >> 5;  // warps per block (8, or 4 for 128-thread plain kernels)
    const int warp = blockIdx.x * wpb + wib;
    if (pa.pf_next.blocks && (int)blockIdx.x >= (int)gridDim.x - pa.pf_next.blocks) {
        do_prefetch(pa.pf_next, blockIdx.x - (gridDim.x - pa.pf_next.blocks), pa.pf_next.blocks);
        return;
    }
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
    __shared__ float sa[SQ ? 8 * 8 * RPL * M : 1];  // SQ: silu(g) * u of the block's 8 * tpw * RPL outputs (tpw <= 8)

    WChunk<FMT, RPL> w[D];
    // SEG: the fp32 rows are work items 0..nrows-1 (first, so their latency is not the kernel tail), quantized
    // tiles follow (work item t -> tile t - nrows)
    const int nseg = SEG ? sg.nrows : 0;
    const int qbeg = tbeg - nseg;
    const bool pre = PRO != PRO_NONE && tbeg < tend && qbeg >= 0 && qbeg < a.ntiles;
    if (pre) {
#pragma unroll
        for (int c = 0; c < D; ++c)
            if (c < NCH) load_chunk<FMT, RPL, NCH>(w[c], a, qbeg, c, lane);
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

    for (int item = tbeg; item < tend; item++) {
        const int tile = item - nseg;
        if (SEG && item < nseg) {
            const int row = item;
            const float4* w4 = (const float4*)(sg.w + (size_t)row * 5120);
            if (M == 1) {
                const float4* x4 = (PRO == PRO_ARNORM || PRO == PRO_LEADER) ? (const float4*)s_xf : (const float4*)sg.x;
                float acc = 0.f;
#pragma unroll 4
                for (int i = lane; i < 1280; i += 32) {
                    const float4 wv = __ldg(w4 + i), x = x4[i];
                    acc += wv.x * x.x + wv.y * x.y + wv.z * x.z + wv.w * x.w;
                }
                acc = warp_sum(acc);
                if (lane == 0) sg.y[row] = acc;
            } else {
                float acc[M];
#pragma unroll
                for (int col = 0; col < M; col++) acc[col] = 0.f;
#pragma unroll 4
                for (int i = lane; i < 1280; i += 32) {
                    const float4 wv = __ldg(w4 + i);
#pragma unroll
                    for (int col = 0; col < M; col++) {
                        const float4 x = ((const float4*)(sg.x + (size_t)col * 5120))[i];
                        acc[col] += wv.x * x.x + wv.y * x.y + wv.z * x.z + wv.w * x.w;
                    }
                }
#pragma unroll
                for (int col = 0; col < M; col++) {
                    const float v = warp_sum(acc[col]);
                    if (lane == 0) sg.y[(size_t)col * 64 + row] = v;
                }
            }
            continue;
        }
        float acc[RPL][M];
#pragma unroll
        for (int r = 0; r < RPL; ++r)
#pragma unroll
            for (int c = 0; c < M; ++c) acc[r][c] = 0.f;
        if (!(pre && item == tbeg)) {
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
                    if (lane == 0) sa[col * 8 * 8 * RPL + (tile - blockIdx.x * wpb * tpw) * RPL + r] = (v / (1.0f + expf(-v))) * u;
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
#pragma unroll
        for (int col = 0; col < M; col++)
            if (wib < ng)
                quant_warp(sa[col * 8 * 8 * RPL + wib * 32 + lane],
                           pa.sq_xq + (size_t)col * (a.N / 2) + (blockIdx.x * ng + wib) * 32 + lane,
                           pa.sq_xm + (size_t)col * (a.N / 64) + blockIdx.x * ng + wib);
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

template <int FMT, int RPL, int NCH, bool AR, bool SEG, int PRO, bool SQ = false, int CVX = 1, int M = 1>
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
            cudaFuncSetAttribute(k_gemv<FMT, RPL, NCH, AR, SEG, PRO, SQ, CVX, M>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                 100);
            int nb = 0, nsm = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_gemv<FMT, RPL, NCH, AR, SEG, PRO, SQ, CVX, M>, 256, 0);
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
        cudaFuncSetAttribute(k_gemv<FMT, RPL, NCH, AR, SEG, PRO, SQ, CVX, M>, cudaFuncAttributePreferredSharedMemoryCarveout,
                             100);
        attr[dev] = true;
    }
    const int tail = (AR && ar.fence == -4) ? ar.tb : 0;
    if (tail && tail * threads != 5120) throw std::runtime_error("AR tail: tb * threads != 5120");
    if (tail && pa.pf_next.blocks) throw std::runtime_error("gemv: AR tail and prefetch blocks are exclusive");
    k_gemv<FMT, RPL, NCH, AR, SEG, PRO, SQ, CVX, M><<<blocks + tail + pa.pf_next.blocks, threads, 0, s>>>(a, ar, sg, pa,
                                                                                                    tpw);
}


// ------------------------------------------------------------------------------------------------ attention / argmax
// (shared with tp_spec.cu; the position is a parameter so verify columns can pass pos + t)
// blocks 0..11: local q heads, 12..13: local kv heads. blockDim 256
__device__ __forceinline__ void d_attn_prep(const float* ya, const float* qw, const float* kw, float* qa, __half* kc, __half* vc,
                            int max_ctx, int pos, float theta_scale, int bx) {
    __shared__ float red[8];
    __shared__ float yv[256];
    const int b = bx, d = threadIdx.x;
    DBG_T0
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

// grid (2 kv heads, NSPLIT), 256 threads, 2 blocks/SM (one wave). Warps 0-3 serve q heads 0-2 of the kv head, warps
// 4-7 heads 3-5, each group striding over the block's chunk of positions (two positions in flight per warp).
// ws layout: [(j * NSPLIT + s) * 6 + h6] x 258 floats {m, l, acc[256]}
__device__ __forceinline__ void d_attn_split(const float* qa, const __half* kc, const __half* vc, float* ws,
                                                       int max_ctx, int pos, int bx, int by) {
    __shared__ float sm_m[8][3], sm_l[8][3];
    __shared__ float sacc[8][256];
    const int j = bx, sidx = by;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, hg = warp >> 2, wq = warp & 3;
    DBG_T0
    const int n_kv = pos + 1;
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

// 12 blocks (local q heads) x 256: merge splits, sigmoid gate, q8 for attn_output
__device__ __forceinline__ void d_attn_combine_q8(const float* ws, const float* ya, int pos,
                                                         int8_t* xq, int2* xm, const Pf pf, int bx) {
    const int hl = bx, d = threadIdx.x, j = hl / 6, h6 = hl % 6;
    DBG_T0
    const int n_kv = pos + 1;
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

}  // namespace
}  // namespace tp
