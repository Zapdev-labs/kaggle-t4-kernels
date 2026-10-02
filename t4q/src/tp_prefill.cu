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
__global__ void __launch_bounds__(256) k_pf_add_norm(float* h, const float* own, const float* rx, int gpu, int add,
                                                     const float* __restrict__ w, float* xn) {
    __shared__ float red[8];
    const int t = blockIdx.x, tid = threadIdx.x;
    float* hp = h + (size_t)t * D;
    float x[20];
#pragma unroll
    for (int k = 0; k < 20; k++) x[k] = hp[tid + 256 * k];
    if (add) {
        const float* p0 = (gpu == 0 ? own : rx) + (size_t)t * D;
        const float* p1 = (gpu == 0 ? rx : own) + (size_t)t * D;
#pragma unroll
        for (int k = 0; k < 20; k++) {
            x[k] = x[k] + (p0[tid + 256 * k] + p1[tid + 256 * k]);
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

// yab[t][i] = xn[t] . ab[i] (48 fp32 rows: 24 alpha then 24 beta). grid ceil(T/8) x 128; thread = (token tid & 7,
// rows (tid >> 3) + 16j, j < 3)
__global__ void __launch_bounds__(128) k_pf_ab(const float* __restrict__ xn, const float* __restrict__ ab, int T,
                                               float* yab) {
    __shared__ float xs[8][132];
    __shared__ __align__(16) float as[48][132];
    const int tid = threadIdx.x, tt = tid & 7, r0 = tid >> 3, t0 = blockIdx.x * 8;
    float acc[3] = {0.f, 0.f, 0.f};
    for (int k0 = 0; k0 < D; k0 += 128) {
        __syncthreads();
        for (int i = tid; i < 8 * 32; i += 128) {
            const int r = i >> 5, k = (i & 31) * 4;
            const float4 v = (t0 + r < T) ? *(const float4*)(xn + (size_t)(t0 + r) * D + k0 + k) : make_float4(0.f, 0.f, 0.f, 0.f);
            xs[r][k] = v.x; xs[r][k + 1] = v.y; xs[r][k + 2] = v.z; xs[r][k + 3] = v.w;
        }
        for (int i = tid; i < 48 * 32; i += 128) {
            const int r = i >> 5, k = (i & 31) * 4;
            *(float4*)&as[r][k] = __ldg((const float4*)(ab + (size_t)r * D + k0 + k));
        }
        __syncthreads();
#pragma unroll 8
        for (int k = 0; k < 128; k++) {
            const float xv = xs[tt][k];
#pragma unroll
            for (int j = 0; j < 3; j++) acc[j] += xv * as[r0 + 16 * j][k];
        }
    }
    if (t0 + tt < T)
#pragma unroll
        for (int j = 0; j < 3; j++) yab[(size_t)(t0 + tt) * 48 + r0 + 16 * j] = acc[j];
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
// grid 96 = (24 local v heads x 4 slices of 32 value columns), 256 threads (warp = 4 columns, lane = 4 key rows).
// Inputs are staged through shared memory in chunks of 16 tokens (double buffered; the next chunk's global loads are in
// flight while the current one is scanned), which keeps the per-token step off the global-memory latency.
constexpr int GCH = 8;  // 18.5 KB smem: 3 blocks per SM, all 96 resident
__global__ void __launch_bounds__(256) k_pf_gdn(const float* __restrict__ qkv, const float* __restrict__ yab, int T,
                                                const float* __restrict__ ssm_a, const float* __restrict__ ssm_dt,
                                                float* S, float* o) {
    __shared__ __align__(16) float sq[2][GCH][128], sk[2][GCH][128], svv[2][GCH][32], sab[2][GCH][2];
    const int bx = blockIdx.x, vl = bx >> 2, sl = bx & 3, kl = vl & 7, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    float s[4][4];
#pragma unroll
    for (int cc = 0; cc < 4; cc++) {
        const int col = sl * 32 + warp * 4 + cc;
        const float* Sp = S + ((size_t)vl * 128 + col) * 128;
#pragma unroll
        for (int r = 0; r < 4; r++) s[cc][r] = Sp[r * 32 + lane];
    }
    const float dtv = ssm_dt[vl], av = ssm_a[vl];
    // staging (GCH = 8): q/k 8 tokens x 32 float4 (one each per thread), v 8 x 8 float4 (threads < 64), ab 16 floats
    float4 rq, rk, rv;
    float rab = 0.f;
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
        if (tid < 16) {
            const int tt = tid >> 1, t = c0 + tt;
            if (t < T) rab = __ldg(yab + (size_t)t * 48 + (tid & 1) * 24 + vl);
        }
    };
    auto sstore = [&](int b) {
        *(float4*)&sq[b][tid >> 5][(tid & 31) * 4] = rq;
        *(float4*)&sk[b][tid >> 5][(tid & 31) * 4] = rk;
        if (tid < 64) *(float4*)&svv[b][tid >> 3][(tid & 7) * 4] = rv;
        if (tid < 16) sab[b][tid >> 1][tid & 1] = rab;
    };
    gload(0);
    sstore(0);
    __syncthreads();
    for (int c0 = 0, b = 0; c0 < T; c0 += GCH, b ^= 1) {
        if (c0 + GCH < T) gload(c0 + GCH);
        const int n = min(GCH, T - c0);
        for (int tt = 0; tt < n; tt++) {
            float kr[4], qr[4];
#pragma unroll
            for (int r = 0; r < 4; r++) { kr[r] = sk[b][tt][r * 32 + lane]; qr[r] = sq[b][tt][r * 32 + lane]; }
            const float4 v4 = *(const float4*)&svv[b][tt][warp * 4];
            const float sv[4] = {v4.x, v4.y, v4.z, v4.w};
            const float ya_ = sab[b][tt][0], yb = sab[b][tt][1];
            const float beta = 1.0f / (1.0f + expf(-yb));
            const float xg = ya_ + dtv;
            const float sp = xg > 20.0f ? xg : logf(1.0f + expf(xg));
            const float gv = expf(sp * av);
            float outv[4];
#pragma unroll
            for (int cc = 0; cc < 4; cc++) {
                float kv = 0.f;
#pragma unroll
                for (int r = 0; r < 4; r++) kv += s[cc][r] * kr[r];
                kv = warp_sum(kv);
                const float delta = (sv[cc] - gv * kv) * beta;
                float a = 0.f;
#pragma unroll
                for (int r = 0; r < 4; r++) {
                    const float sn = gv * s[cc][r] + kr[r] * delta;
                    a += sn * qr[r];
                    s[cc][r] = sn;
                }
                outv[cc] = warp_sum(a);
            }
            if (lane == 0)
                *(float4*)(o + (size_t)(c0 + tt) * 3072 + vl * 128 + sl * 32 + warp * 4) =
                    make_float4(outv[0] * (1.0f / sqrtf(128.0f)), outv[1] * (1.0f / sqrtf(128.0f)),
                                outv[2] * (1.0f / sqrtf(128.0f)), outv[3] * (1.0f / sqrtf(128.0f)));
        }
        if (c0 + GCH < T) sstore(b ^ 1);
        __syncthreads();
    }
#pragma unroll
    for (int cc = 0; cc < 4; cc++) {
        const int col = sl * 32 + warp * 4 + cc;
        float* Sp = S + ((size_t)vl * 128 + col) * 128;
#pragma unroll
        for (int r = 0; r < 4; r++) Sp[r * 32 + lane] = s[cc][r];
    }
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
    int8_t* xq = nullptr;
    float2* xs = nullptr;
    float* xsum = nullptr;
    cudaStream_t sc = nullptr;              // copy stream (AR payloads)
    cudaEvent_t part_ev[NSUB] = {};         // compute stream: partial rows of sub s written
    cudaEvent_t sent[NSUB][2] = {};         // copy stream: sub s rows of AR slot copied to the peer
};
struct Pf {
    int cap = 0;  // allocated ubatch capacity (tokens)
    PfGpu G[2];
};

template <class T>
T* dalloc(size_t n) {
    T* p;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

Pf* pf_get(t4q_ctx* c) {
    tp::State& S = *c->tps;
    const int ub = S.pf_ub;
    Pf* P = (Pf*)S.pf;
    if (P && P->cap >= ub) return P;
    if (P) {
        for (int g = 0; g < 2; g++) {
            CK(cudaSetDevice(g));
            CK(cudaDeviceSynchronize());
            PfGpu& B = P->G[g];
            for (void* p : {(void*)B.ids, (void*)B.h, (void*)B.xn, (void*)B.y, (void*)B.part, (void*)B.rx[0],
                            (void*)B.rx[1], (void*)B.yab, (void*)B.qkv, (void*)B.o, (void*)B.g32, (void*)B.qa,
                            (void*)B.xq, (void*)B.xs, (void*)B.xsum})
                cudaFree(p);
            for (auto e : B.part_ev) cudaEventDestroy(e);
            for (auto& r : B.sent) for (auto e : r) cudaEventDestroy(e);
            cudaStreamDestroy(B.sc);
        }
        delete P;
    }
    P = new Pf();
    P->cap = ub;
    const size_t U = ub, Up = (size_t)(ub + 127) / 128 * 128;
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
        B.yab = dalloc<float>(U * 48);
        B.qkv = dalloc<float>(U * 5120);
        B.o = dalloc<float>(U * 3072);
        B.g32 = dalloc<float>(U * 8704);
        B.qa = dalloc<float>(U * 3072);
        B.xq = dalloc<int8_t>(Up * 8704);
        B.xs = dalloc<float2>(Up * (8704 / 32));
        B.xsum = dalloc<float>(Up * (8704 / 32));
        CK(cudaStreamCreateWithFlags(&B.sc, cudaStreamNonBlocking));
        for (auto& e : B.part_ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        for (auto& r : B.sent) for (auto& e : r) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        // the memsets above run on the legacy stream, which does not order against the engine's non-blocking streams
        CK(cudaDeviceSynchronize());
    }
    S.pf = P;
    return P;
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
    // profiling (option pf_prof): events on GPU0's stream after each op group, named by the op that just ended
    std::vector<cudaEvent_t>* ev = nullptr;
    std::vector<const char*>* evn = nullptr;
    void mark(int g, const char* name) {
        if (!ev || g != 0) return;
        cudaEvent_t e;
        CK(cudaEventCreate(&e));
        CK(cudaEventRecord(e, c->tps->G[0].s));
        ev->push_back(e);
        evn->push_back(name);
    }

    // sub-batch s: x (fp32 rows t0.., row stride K) -> q8 activations for the GEMM (rows 0..Tp-1 of the shared buffer)
    void quant(int g, int s, const float* x, int K) {
        tp::Gpu& G = c->tps->G[g];
        PfGpu& B = P->G[g];
        const int Ts = sT[s], Tps = (Ts + 127) / 128 * 128;
        const int n = Tps * (K >> 5);
        const float* xs0 = x + (size_t)st0[s] * K;
        if (i4) gemm::quant_rows_i4_kernel<<<(n + 127) / 128, 128, 0, G.s>>>(xs0, K, Ts, Tps, K, B.xq, B.xs, B.xsum);
        else gemm::quant_rows_kernel<<<(n + 127) / 128, 128, 0, G.s>>>(xs0, K, Ts, Tps, K, B.xq, B.xs, B.xsum);
        ck_launch("quant");
        mark(g, "quant");
    }
    void gemm(int g, int s, const tp::FW& W, float* y, int ldy) {
        tp::Gpu& G = c->tps->G[g];
        PfGpu& B = P->G[g];
        const int Ts = sT[s], Tps = (Ts + 127) / 128 * 128;
        gemm::GemmArgs a = gemm::make_args(W.L, W.base, B.xq, B.xs, B.xsum, y + (size_t)st0[s] * ldy, ldy, Ts, Tps);
        cudaError_t e;
        if (i4 && W.L.fmt != gemv::FAST_K5) {
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
    // K5 GEMMs need int8 activations even in i4 mode
    void qg(int g, int s, const tp::FW& W, const float* x, int K, float* y, int ldy) {
        const bool save = i4;
        if (W.L.fmt == gemv::FAST_K5) i4 = false;
        quant(g, s, x, K);
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
            CK(cudaEventRecord(B.part_ev[s], S.G[g].s));
            CK(cudaStreamWaitEvent(B.sc, B.part_ev[s], 0));
            CK(cudaMemcpyPeerAsync(P->G[1 - g].rx[sl] + off, 1 - g, B.part + off, g, bytes, B.sc));
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
    void add_norm(int g, int s, const float* w, bool add) {
        tp::Gpu& G = c->tps->G[g];
        PfGpu& B = P->G[g];
        if (add) recv(g, s);
        const int sl = (ar - 1) & 1;
        const size_t off = (size_t)st0[s] * D;
        k_pf_add_norm<<<sT[s], 256, 0, G.s>>>(B.h + off, B.part + off, B.rx[sl] + off, g, add ? 1 : 0, w, B.xn + off);
        ck_launch("add_norm");
        mark(g, "ar_wait+add_norm");
    }

    void mixer(int g, int il, int s, float theta_scale) {
        tp::State& S = *c->tps;
        tp::Gpu& G = S.G[g];
        tp::Layer& L = G.L[il];
        PfGpu& B = P->G[g];
        const int t0 = st0[s], Ts = sT[s], ps = p0 + t0;
        add_norm(g, s, L.attn_norm, il > 0);
        if (!L.attn) {
            qg(g, s, L.qkvz, B.xn, D, B.y, 8192);
            k_pf_ab<<<(Ts + 7) / 8, 128, 0, G.s>>>(B.xn + (size_t)t0 * D, L.ab, Ts, B.yab + (size_t)t0 * 48);
            mark(g, "ab");
            k_pf_conv<<<dim3(Ts, 40), 128, 0, G.s>>>(B.y + (size_t)t0 * 8192, 8192, L.conv_ring, L.conv_w, ps,
                                                     B.qkv + (size_t)t0 * 5120);
            k_pf_ring<<<20, 256, 0, G.s>>>(B.y + (size_t)t0 * 8192, 8192, Ts, ps, L.conv_ring);
            mark(g, "conv");
            {
                static int carve[2] = {0, 0};
                if (!carve[g]) {
                    CK(cudaFuncSetAttribute(k_pf_gdn, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
                    carve[g] = 1;
                }
            }
            k_pf_gdn<<<96, 256, 0, G.s>>>(B.qkv + (size_t)t0 * 5120, B.yab + (size_t)t0 * 48, Ts, L.ssm_a, L.ssm_dt,
                                          L.S, B.o + (size_t)t0 * 3072);
            mark(g, "gdn_scan");
            k_pf_gnorm<<<dim3(Ts, 24), 128, 0, G.s>>>(B.o + (size_t)t0 * 3072, B.y + (size_t)t0 * 8192, 8192,
                                                       L.ssm_norm, B.g32 + (size_t)t0 * 3072);
            ck_launch("deltanet");
            mark(g, "gnorm");
            qg(g, s, L.ssm_out, B.g32, 3072, B.part, D);
        } else {
            qg(g, s, L.qkv_a, B.xn, D, B.y, 7168);
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
        add_norm(g, s, L.post_norm, true);
        qg(g, s, L.gateup, B.xn, D, B.y, 17408);
        k_pf_silu<<<dim3(Ts, 34), 256, 0, G.s>>>(B.y + (size_t)t0 * 17408, 17408, B.g32 + (size_t)t0 * 8704);
        ck_launch("silu");
        mark(g, "silu");
        qg(g, s, L.down, B.g32, 8704, B.part, D);
    }

    void run(const int32_t* ids) {
        tp::State& S = *c->tps;
        nsub = (S.pf_nsub >= 2 && T >= 256) ? 2 : 1;
        const int half = nsub == 2 ? ((T / 2 + 127) / 128) * 128 : T;  // sub-batch 0 rows: a multiple of 128
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
    auto t0 = Clock::now();
    const int nb = n - 1;  // the last token goes through the decode step
    std::vector<cudaEvent_t> ev;
    std::vector<const char*> evn;
    std::vector<std::pair<std::string, std::pair<double, int>>> prof;
    auto acc = [&](const std::string& k, double ms) {
        for (auto& p : prof)
            if (p.first == k) { p.second.first += ms; p.second.second++; return; }
        prof.push_back({k, {ms, 1}});
    };
    for (int b0 = 0; b0 < nb; b0 += S.pf_ub) {
        PfRun R{c, P};
        R.T = std::min(S.pf_ub, nb - b0);
        R.p0 = c->pos + b0;
        R.i4 = S.pf_i4 != 0;
        if (S.pf_prof) { R.ev = &ev; R.evn = &evn; }
        R.run(ids + b0);
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
    // decode step for the last token at position c->pos + n - 1
    const int pos_last = c->pos + nb;
    for (int g = 0; g < 2; g++) {
        CK(cudaSetDevice(g));
        CK(cudaMemcpy(&S.G[g].st->pos, &pos_last, 4, cudaMemcpyHostToDevice));
    }
    c->pos = pos_last;
    run_last_step(c);
    for (int g = 0; g < 2; g++) {
        CK(cudaSetDevice(g));
        CK(cudaStreamSynchronize(S.G[g].s));
    }
    S.pf_last_batch_s = t_batch;
    S.pf_last_total_s = secs(t0);
    S.pf_last_n = n;
    return 0;
}
