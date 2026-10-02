// TP decode kernels: fast GEMV with fused AR publish / fp32 segment rows, and the small fused kernels between GEMVs.
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

// ------------------------------------------------------------------------------------------------ GEMV
template <int FMT, int RPL, int M, int NCH, bool AR, bool SEG>
__global__ void __launch_bounds__(256, 2) k_gemv(const GemvArgs a, const ArArgs ar, const SegArgs sg) {
    constexpr int D = 2;
    constexpr int CVT = (FMT == FAST_P4 || FMT == FAST_Q8) ? 1 : 0;
    const int lane = threadIdx.x & 31, h = lane >> 4, j = lane & 15;
    const int warp = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int nwarps = gridDim.x * (blockDim.x >> 5);
    constexpr int NB = NCH * 16;
    constexpr int K = NCH * 512;
    const int ntot = a.ntiles + (SEG ? sg.nrows : 0);
    // AR: the launcher guarantees one tile per warp, so a block owns rows [blockIdx.x * 16 * RPL, +16 * RPL); they
    // are staged here and sent to the peer as a few coalesced float4 PCIe writes (4-byte scattered remote stores
    // cost ~60 us per 20 KB in M2 v1).
    __shared__ float sy[AR ? 8 * 2 * RPL : 1];

    for (int tile = warp; tile < ntot; tile += nwarps) {
        if (SEG && tile >= a.ntiles) {
            const int row = tile - a.ntiles;
            const float4* w4 = (const float4*)(sg.w + (size_t)row * 5120);
            const float4* x4 = (const float4*)sg.x;
            float acc = 0.f;
#pragma unroll 4
            for (int i = lane; i < 1280; i += 32) {
                const float4 w = __ldg(w4 + i), x = x4[i];
                acc += w.x * x.x + w.y * x.y + w.z * x.z + w.w * x.w;
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
                const int4* xp = (const int4*)(a.xq + (size_t)col * K + kb * 32);
                const int4 xl = __ldg(xp), xh = __ldg(xp + 1);
                const int2 mt = __ldg(a.xm + (size_t)col * NB + kb);
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
                if (j == 0 && row < a.N) {
                    a.y[(size_t)col * a.ldy + row] = v;
                    if (AR) sy[(threadIdx.x >> 5) * 2 * RPL + h * RPL + r] = v;
                }
            }
    }
    if (AR) {
        __syncthreads();
        const int row0 = blockIdx.x * 16 * RPL + threadIdx.x * 4;
        if (threadIdx.x < 4 * RPL && row0 < a.N) {
            *(float4*)(ar.y_peer + row0) = *(const float4*)(sy + threadIdx.x * 4);
            __threadfence_system();
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const unsigned old = atomicAdd(ar.cnt, 1u);
            if (old == gridDim.x - 1) {
                atomicExch(ar.cnt, 0u);
                __threadfence_system();
                st_vol_u32(ar.peer_flag, epoch_of(ar.st, ar.idx));
            }
        }
    }
}

template <int FMT, int RPL, int NCH, bool AR, bool SEG>
void launch_gemv(const FW& W, const int8_t* xq, const int2* xm, float* y, cudaStream_t s, const ArArgs& ar,
                 const SegArgs& sg) {
    GemvArgs a = make_args(W.L, W.base, xq, xm, y, W.L.N);
    const int ntot = W.L.ntiles + (SEG ? sg.nrows : 0);
    const int blocks = (ntot + 7) / 8;  // full grid: one tile per warp (required by the AR epilogue)
    if (AR && (W.L.N % 4 || SEG)) throw std::runtime_error("AR gemv needs N % 4 == 0 and no segment");
    k_gemv<FMT, RPL, 1, NCH, AR, SEG><<<blocks, 256, 0, s>>>(a, ar, sg);
}

}  // namespace

void gemv(const FW& W, const int8_t* xq, const int2* xm, float* y, cudaStream_t s, const ArArgs* ar,
          const SegArgs* seg) {
    const ArArgs A = ar ? *ar : ArArgs{};
    const SegArgs S = seg ? *seg : SegArgs{};
    const int f = W.L.fmt, nch = W.L.nch, rpl = W.L.rpl;
    const bool isar = ar != nullptr, isseg = seg != nullptr;
#define T4Q_G(FMT, RPL, NCH, AR_, SEG_)                                                         \
    if (f == FMT && rpl == RPL && nch == NCH && isar == AR_ && isseg == SEG_) {                  \
        launch_gemv<FMT, RPL, NCH, AR_, SEG_>(W, xq, xm, y, s, A, S);                           \
        return;                                                                                 \
    }
    T4Q_G(FAST_P4, 4, 10, false, true)   // DeltaNet qkvz + alpha/beta fp32 rows
    T4Q_G(FAST_P4, 4, 10, false, false)  // attn q|k|v, ffn gate|up
    T4Q_G(FAST_K5, 2, 6, true, false)    // ssm_out (Q5_K), K-split, AR
    T4Q_G(FAST_P4, 4, 6, true, false)    // attn_output, K-split, AR
    T4Q_G(FAST_P4, 4, 17, true, false)   // ffn_down Q4_0, K-split, AR
    T4Q_G(FAST_P4M, 4, 17, true, false)  // ffn_down Q4_1, K-split, AR
    T4Q_G(FAST_K6, 2, 10, false, false)  // lm_head Q6_K
    T4Q_G(FAST_P4, 4, 6, false, false)   // self-test variants without AR
    T4Q_G(FAST_K5, 2, 6, false, false)
    T4Q_G(FAST_P4, 4, 17, false, false)
    T4Q_G(FAST_P4M, 4, 17, false, false)
#undef T4Q_G
    throw std::runtime_error("tp::gemv: no instantiation for fmt " + std::to_string(f) + " rpl " + std::to_string(rpl) +
                             " nch " + std::to_string(nch) + " ar " + std::to_string(isar) + " seg " +
                             std::to_string(isseg));
}

// ------------------------------------------------------------------------------------------------ small kernels
namespace {

__global__ void k_embed(const uint8_t* __restrict__ embd, const StepState* st, const int* prompt, float* h) {
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
__global__ void __launch_bounds__(1024) k_ar_norm(float* h, const float* own, const float* rx, const unsigned* flag,
                                                  StepState* st, int idx, const float* __restrict__ w, float* xn,
                                                  int8_t* xq, int2* xm) {
    __shared__ float red[32];
    __shared__ int s_ok;
    const int tid = threadIdx.x;
    const int slot = idx & 1;
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
        if (WAIT) {
            x = x + (own[slot * 5120 + i] + ld_vol_f32(rx + slot * 5120 + i));
            h[i] = x;
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
    float s[4][4];
#pragma unroll
    for (int cc = 0; cc < 4; cc++) {
        const int col = sl * 32 + warp * 4 + cc;
        const float* Sp = S + ((size_t)vl * 128 + col) * 128;
#pragma unroll
        for (int r = 0; r < 4; r++) s[cc][r] = Sp[r * 32 + lane];
    }
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
                                                  int8_t* xq, int2* xm) {
    __shared__ float red[4];
    const int vl = blockIdx.x, i = threadIdx.x;
    const float x = o[vl * 128 + i];
    const float ss = block_sum(x * x, red);
    const float scale = rsqrtf(ss / 128.0f + 1e-6f);
    const float zz = z[vl * 128 + i];
    const float val = ((x * scale) * w[i]) * (zz / (1.0f + expf(-zz)));
    quant_warp(val, xq + vl * 128 + i, xm + vl * 4 + (i >> 5));
}

__global__ void k_silu_q8(const float* gu, int n, int8_t* xq, int2* xm) {
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

// grid (2 kv heads, NSPLIT), 256 threads. Each block: 6 q heads of its kv head over a chunk of positions.
// ws layout: [(j * NSPLIT + s) * 6 + h6] x 258 floats {m, l, acc[256]}
__global__ void __launch_bounds__(256) k_attn_split(const float* qa, const __half* kc, const __half* vc, float* ws,
                                                    int max_ctx, const StepState* st) {
    __shared__ float sm_m[8][6], sm_l[8][6];
    __shared__ float sacc[8][256];
    const int j = blockIdx.x, sidx = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n_kv = st->pos + 1;
    const int chunk = (n_kv + NSPLIT - 1) / NSPLIT;
    const int t0 = sidx * chunk, t1 = min(n_kv, t0 + chunk);
    float q[6][8];
#pragma unroll
    for (int h6 = 0; h6 < 6; h6++) {
        const float4* qp = (const float4*)(qa + (j * 6 + h6) * 256 + lane * 8);
        const float4 a = qp[0], b = qp[1];
        q[h6][0] = a.x; q[h6][1] = a.y; q[h6][2] = a.z; q[h6][3] = a.w;
        q[h6][4] = b.x; q[h6][5] = b.y; q[h6][6] = b.z; q[h6][7] = b.w;
    }
    float m[6], l[6], acc[6][8];
#pragma unroll
    for (int h6 = 0; h6 < 6; h6++) {
        m[h6] = -FLT_MAX; l[h6] = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) acc[h6][i] = 0.f;
    }
    const __half* K = kc + (size_t)j * max_ctx * 256;
    const __half* Vv = vc + (size_t)j * max_ctx * 256;
    // two positions per warp iteration (both K/V rows in flight before the math)
    for (int t = t0 + warp; t < t1; t += 16) {
        const bool two = t + 8 < t1;
        uint4 kraw[2], vraw[2];
        kraw[0] = *(const uint4*)(K + (size_t)t * 256 + lane * 8);
        vraw[0] = *(const uint4*)(Vv + (size_t)t * 256 + lane * 8);
        if (two) {
            kraw[1] = *(const uint4*)(K + (size_t)(t + 8) * 256 + lane * 8);
            vraw[1] = *(const uint4*)(Vv + (size_t)(t + 8) * 256 + lane * 8);
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
            for (int h6 = 0; h6 < 6; h6++) {
                float dot = 0.f;
#pragma unroll
                for (int i = 0; i < 8; i++) dot += q[h6][i] * k[i];
                dot = warp_sum(dot) * (1.0f / 16.0f);
                const float mn = fmaxf(m[h6], dot);
                const float c = expf(m[h6] - mn), p = expf(dot - mn);
                l[h6] = l[h6] * c + p;
#pragma unroll
                for (int i = 0; i < 8; i++) acc[h6][i] = acc[h6][i] * c + p * v[i];
                m[h6] = mn;
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int h6 = 0; h6 < 6; h6++) { sm_m[warp][h6] = m[h6]; sm_l[warp][h6] = l[h6]; }
    }
    __syncthreads();
#pragma unroll
    for (int h6 = 0; h6 < 6; h6++) {
        float M = -FLT_MAX;
#pragma unroll
        for (int w = 0; w < 8; w++) M = fmaxf(M, sm_m[w][h6]);
        const float sc = (l[h6] > 0.f) ? expf(m[h6] - M) : 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) sacc[warp][lane * 8 + i] = acc[h6][i] * sc;
        __syncthreads();
        float* out = ws + ((size_t)(j * NSPLIT + sidx) * 6 + h6) * 258;
        float a = 0.f;
#pragma unroll
        for (int w = 0; w < 8; w++) a += sacc[w][tid];
        out[2 + tid] = a;
        if (tid == 0) {
            float L = 0.f;
            for (int w = 0; w < 8; w++)
                if (sm_l[w][h6] > 0.f) L += sm_l[w][h6] * expf(sm_m[w][h6] - M);
            out[0] = M;
            out[1] = L;
        }
        __syncthreads();
    }
}

// 12 blocks (local q heads) x 256: merge splits, sigmoid gate, q8 for attn_output
__global__ void __launch_bounds__(256) k_attn_combine_q8(const float* ws, const float* ya, const StepState* st,
                                                         int8_t* xq, int2* xm) {
    const int hl = blockIdx.x, d = threadIdx.x, j = hl / 6, h6 = hl % 6;
    const int n_kv = st->pos + 1;
    const int chunk = (n_kv + NSPLIT - 1) / NSPLIT;
    const int nsp = (n_kv + chunk - 1) / chunk;  // splits with at least one position
    float M = -FLT_MAX;
    for (int s = 0; s < nsp; s++) M = fmaxf(M, ws[((size_t)(j * NSPLIT + s) * 6 + h6) * 258]);
    float num = 0.f, den = 0.f;
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

}  // namespace

void embed(const uint8_t* embd, StepState* st, const int* prompt, float* h, cudaStream_t s) {
    k_embed<<<20, 256, 0, s>>>(embd, st, prompt, h);
}

void ar_norm(float* h, const float* own, const float* rx, const unsigned* flag, const StepState* st, int idx,
             const float* w, float* xn, int8_t* xq, int2* xm, cudaStream_t s) {
    if (flag) k_ar_norm<true><<<1, 1024, 0, s>>>(h, own, rx, flag, (StepState*)st, idx, w, xn, xq, xm);
    else k_ar_norm<false><<<1, 1024, 0, s>>>(h, own, rx, flag, (StepState*)st, idx, w, xn, xq, xm);
}

void gdn(const float* y, const float* yab, float* ring, const float* conv_w, const float* ssm_a, const float* ssm_dt,
         float* S, float* o, const StepState* st, cudaStream_t s) {
    k_gdn<<<96, 256, 0, s>>>(y, yab, ring, conv_w, ssm_a, ssm_dt, S, o, st);
}

void gnorm_q8(const float* o, const float* z, const float* w, int8_t* xq, int2* xm, cudaStream_t s) {
    k_gnorm_q8<<<24, 128, 0, s>>>(o, z, w, xq, xm);
}

void silu_q8(const float* gu, int n, int8_t* xq, int2* xm, cudaStream_t s) {
    k_silu_q8<<<(n + 255) / 256, 256, 0, s>>>(gu, n, xq, xm);
}

void attn_prep(const float* ya, const float* qw, const float* kw, float* qa, uint16_t* kc, uint16_t* vc, int max_ctx,
               const StepState* st, float theta_scale, cudaStream_t s) {
    k_attn_prep<<<14, 256, 0, s>>>(ya, qw, kw, qa, (__half*)kc, (__half*)vc, max_ctx, st, theta_scale);
}

void attn_split(const float* qa, const uint16_t* kc, const uint16_t* vc, float* ws, int max_ctx, const StepState* st,
                cudaStream_t s) {
    k_attn_split<<<dim3(2, NSPLIT), 256, 0, s>>>(qa, (const __half*)kc, (const __half*)vc, ws, max_ctx, st);
}

void attn_combine_q8(const float* ws, const float* ya, const StepState* st, int8_t* xq, int2* xm, cudaStream_t s) {
    k_attn_combine_q8<<<12, 256, 0, s>>>(ws, ya, st, xq, xm);
}

void argmax_step(const float* logits, int n, int row0, float* apart, float* amb, const unsigned* aflag,
                 float* peer_amb, unsigned* peer_aflag, StepState* st, int* ring, cudaStream_t s) {
    k_argmax_part<<<NBA, 256, 0, s>>>(logits, n, apart);
    k_argmax_final<<<1, 32, 0, s>>>(apart, row0, amb, aflag, peer_amb, peer_aflag, st, ring);
}

}  // namespace tp
