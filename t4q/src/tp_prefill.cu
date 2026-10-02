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

// yab[t][i] = xn[t] . ab[i] (48 fp32 rows: 24 alpha then 24 beta). grid ceil(T/32) x 256; thread = (token tt, rows w+8j)
__global__ void __launch_bounds__(256) k_pf_ab(const float* __restrict__ xn, const float* __restrict__ ab, int T,
                                               float* yab) {
    __shared__ float xs[32][129];
    __shared__ float as[48][128];
    const int tid = threadIdx.x, tt = tid & 31, w = tid >> 5, t0 = blockIdx.x * 32;
    float acc[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int k0 = 0; k0 < D; k0 += 128) {
        __syncthreads();
        for (int i = tid; i < 32 * 128; i += 256) {
            const int r = i >> 7, k = i & 127;
            xs[r][k] = (t0 + r < T) ? xn[(size_t)(t0 + r) * D + k0 + k] : 0.f;
        }
        for (int i = tid; i < 48 * 128; i += 256) {
            const int r = i >> 7, k = i & 127;
            as[r][k] = ab[(size_t)r * D + k0 + k];
        }
        __syncthreads();
#pragma unroll 4
        for (int k = 0; k < 128; k++) {
            const float xv = xs[tt][k];
#pragma unroll
            for (int j = 0; j < 6; j++) acc[j] += xv * as[w + 8 * j][k];
        }
    }
    if (t0 + tt < T)
#pragma unroll
        for (int j = 0; j < 6; j++) yab[(size_t)(t0 + tt) * 48 + w + 8 * j] = acc[j];
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
// grid 96 = (24 local v heads x 4 slices of 32 value columns), 256 threads (warp = 4 columns, lane = 4 key rows)
__global__ void __launch_bounds__(256) k_pf_gdn(const float* __restrict__ qkv, const float* __restrict__ yab, int T,
                                                const float* __restrict__ ssm_a, const float* __restrict__ ssm_dt,
                                                float* S, float* o) {
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
    const int qoff = kl * 128 + lane, koff = 1024 + kl * 128 + lane, voff = 2048 + vl * 128 + sl * 32 + warp * 4;
    // prefetch registers for token t
    float kn[4], qn[4], vn[4], ya_n, yb_n;
    auto load = [&](int t) {
        const float* row = qkv + (size_t)t * 5120;
#pragma unroll
        for (int r = 0; r < 4; r++) { kn[r] = __ldg(row + koff + r * 32); qn[r] = __ldg(row + qoff + r * 32); }
        const float4 v4 = __ldg((const float4*)(row + voff));
        vn[0] = v4.x; vn[1] = v4.y; vn[2] = v4.z; vn[3] = v4.w;
        ya_n = __ldg(yab + (size_t)t * 48 + vl);
        yb_n = __ldg(yab + (size_t)t * 48 + 24 + vl);
    };
    if (T > 0) load(0);
    for (int t = 0; t < T; t++) {
        float kr[4], qr[4], sv[4];
#pragma unroll
        for (int r = 0; r < 4; r++) { kr[r] = kn[r]; qr[r] = qn[r]; sv[r] = vn[r]; }
        const float ya_ = ya_n, yb = yb_n;
        if (t + 1 < T) load(t + 1);
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
            *(float4*)(o + (size_t)t * 3072 + vl * 128 + sl * 32 + warp * 4) =
                make_float4(outv[0] * (1.0f / sqrtf(128.0f)), outv[1] * (1.0f / sqrtf(128.0f)),
                            outv[2] * (1.0f / sqrtf(128.0f)), outv[3] * (1.0f / sqrtf(128.0f)));
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

// silu(g) * u for the gate|up output interleaved by 4 rows per 8-row tile. grid (T, 34) x 256
__global__ void k_pf_silu(const float* __restrict__ y, int ldy, float* out) {
    const int t = blockIdx.x, i = blockIdx.y * 256 + threadIdx.x;  // 0..8703
    const int gr = (i >> 2) * 8 + (i & 3);
    const float g = y[(size_t)t * ldy + gr], u = y[(size_t)t * ldy + gr + 4];
    out[(size_t)t * 8704 + i] = (g / (1.0f + expf(-g))) * u;
}

// ------------------------------------------------------------------------------------------------ host side
struct PfGpu {
    int* ids = nullptr;
    float *h = nullptr, *xn = nullptr, *y = nullptr, *part = nullptr, *rx[2] = {nullptr, nullptr};
    float *yab = nullptr, *qkv = nullptr, *o = nullptr, *g32 = nullptr, *qa = nullptr;
    int8_t* xq = nullptr;
    float2* xs = nullptr;
    float* xsum = nullptr;
    cudaEvent_t sent[2] = {nullptr, nullptr};
};
struct Pf {
    int ub = 0, ubp = 0;
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
    if (P && P->ub == ub) return P;
    if (P) {
        for (int g = 0; g < 2; g++) {
            CK(cudaSetDevice(g));
            PfGpu& B = P->G[g];
            for (void* p : {(void*)B.ids, (void*)B.h, (void*)B.xn, (void*)B.y, (void*)B.part, (void*)B.rx[0],
                            (void*)B.rx[1], (void*)B.yab, (void*)B.qkv, (void*)B.o, (void*)B.g32, (void*)B.qa,
                            (void*)B.xq, (void*)B.xs, (void*)B.xsum})
                cudaFree(p);
            for (auto e : B.sent) cudaEventDestroy(e);
        }
        delete P;
    }
    P = new Pf();
    P->ub = ub;
    P->ubp = (ub + 127) / 128 * 128;
    const size_t U = ub, Up = P->ubp;
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
        for (auto& e : B.sent) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
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
    int T, Tp, p0;
    int ar = 0;  // all-reduce counter (slot = ar & 1)
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

    // x (fp32 [T][K], row stride K) -> q8 activations for the GEMM
    void quant(int g, const float* x, int K) {
        tp::Gpu& G = c->tps->G[g];
        PfGpu& B = P->G[g];
        const int n = Tp * (K >> 5);
        if (i4) gemm::quant_rows_i4_kernel<<<(n + 127) / 128, 128, 0, G.s>>>(x, K, T, Tp, K, B.xq, B.xs, B.xsum);
        else gemm::quant_rows_kernel<<<(n + 127) / 128, 128, 0, G.s>>>(x, K, T, Tp, K, B.xq, B.xs, B.xsum);
        ck_launch("quant");
        mark(g, "quant");
    }
    void gemm(int g, const tp::FW& W, float* y, int ldy) {
        tp::Gpu& G = c->tps->G[g];
        PfGpu& B = P->G[g];
        gemm::GemmArgs a = gemm::make_args(W.L, W.base, B.xq, B.xs, B.xsum, y, ldy, T, Tp);
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
    void quant_for(int g, const tp::FW& W, const float* x, int K) {
        const bool save = i4;
        if (W.L.fmt == gemv::FAST_K5) i4 = false;
        quant(g, x, K);
        i4 = save;
    }
    void gemm_for(int g, const tp::FW& W, float* y, int ldy) {
        const bool save = i4;
        if (W.L.fmt == gemv::FAST_K5) i4 = false;
        gemm(g, W, y, ldy);
        i4 = save;
    }
    // send this GPU's partial to the peer (rx slot of the current AR); the peer's stream waits for it
    void exchange() {
        tp::State& S = *c->tps;
        const int sl = ar & 1;
        for (int g = 0; g < 2; g++) {
            CK(cudaSetDevice(g));
            CK(cudaMemcpyPeerAsync(P->G[1 - g].rx[sl], 1 - g, P->G[g].part, g, (size_t)T * D * 4, S.G[g].s));
            CK(cudaEventRecord(P->G[g].sent[sl], S.G[g].s));
            mark(g, "ar_copy");
        }
        for (int g = 0; g < 2; g++) {
            CK(cudaSetDevice(g));
            CK(cudaStreamWaitEvent(S.G[g].s, P->G[1 - g].sent[sl], 0));
        }
    }
    // consume the pending AR (if any) and normalize with w
    void add_norm(int g, const float* w, bool add) {
        tp::Gpu& G = c->tps->G[g];
        PfGpu& B = P->G[g];
        const int sl = (ar - 1) & 1;
        k_pf_add_norm<<<T, 256, 0, G.s>>>(B.h, B.part, B.rx[sl], g, add ? 1 : 0, w, B.xn);
        ck_launch("add_norm");
        mark(g, "ar_wait+add_norm");
    }

    void run(const int32_t* ids) {
        tp::State& S = *c->tps;
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
            // ---- mixer
            for (int g = 0; g < 2; g++) {
                CK(cudaSetDevice(g));
                tp::Gpu& G = S.G[g];
                tp::Layer& L = G.L[il];
                PfGpu& B = P->G[g];
                add_norm(g, L.attn_norm, il > 0);
                if (!L.attn) {
                    quant_for(g, L.qkvz, B.xn, D);
                    gemm_for(g, L.qkvz, B.y, 8192);
                    k_pf_ab<<<(T + 31) / 32, 256, 0, G.s>>>(B.xn, L.ab, T, B.yab);
                    mark(g, "ab");
                    k_pf_conv<<<dim3(T, 40), 128, 0, G.s>>>(B.y, 8192, L.conv_ring, L.conv_w, p0, B.qkv);
                    k_pf_ring<<<20, 256, 0, G.s>>>(B.y, 8192, T, p0, L.conv_ring);
                    mark(g, "conv");
                    k_pf_gdn<<<96, 256, 0, G.s>>>(B.qkv, B.yab, T, L.ssm_a, L.ssm_dt, L.S, B.o);
                    mark(g, "gdn_scan");
                    k_pf_gnorm<<<dim3(T, 24), 128, 0, G.s>>>(B.o, B.y, 8192, L.ssm_norm, B.g32);
                    ck_launch("deltanet");
                    mark(g, "gnorm");
                    quant_for(g, L.ssm_out, B.g32, 3072);
                    gemm_for(g, L.ssm_out, B.part, D);
                } else {
                    quant_for(g, L.qkv_a, B.xn, D);
                    gemm_for(g, L.qkv_a, B.y, 7168);
                    k_pf_attn_prep<<<dim3(T, 14), 256, 0, G.s>>>(B.y, 7168, L.q_norm, L.k_norm, B.qa, (__half*)L.kc,
                                                                 (__half*)L.vc, S.max_ctx, p0, theta_scale);
                    mark(g, "attn_prep");
                    k_pf_attn<<<dim3(T, 2), 192, 0, G.s>>>(B.qa, (const __half*)L.kc, (const __half*)L.vc, B.y, 7168,
                                                           S.max_ctx, p0, B.g32);
                    ck_launch("attention");
                    mark(g, "attn");
                    quant_for(g, L.wo, B.g32, 3072);
                    gemm_for(g, L.wo, B.part, D);
                }
            }
            exchange();
            ar++;
            // ---- FFN
            for (int g = 0; g < 2; g++) {
                CK(cudaSetDevice(g));
                tp::Gpu& G = S.G[g];
                tp::Layer& L = G.L[il];
                PfGpu& B = P->G[g];
                add_norm(g, L.post_norm, true);
                quant_for(g, L.gateup, B.xn, D);
                gemm_for(g, L.gateup, B.y, 17408);
                k_pf_silu<<<dim3(T, 34), 256, 0, G.s>>>(B.y, 17408, B.g32);
                ck_launch("silu");
                mark(g, "silu");
                quant_for(g, L.down, B.g32, 8704);
                gemm_for(g, L.down, B.part, D);
            }
            exchange();
            ar++;
        }
        // the final residual of these tokens is not needed (the next ubatch / decode step starts from embeddings);
        // consume the last AR so the rx slots alternate consistently
        for (int g = 0; g < 2; g++) {
            CK(cudaSetDevice(g));
            add_norm(g, S.G[g].output_norm, true);
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
    for (int b0 = 0; b0 < nb; b0 += P->ub) {
        PfRun R{c, P};
        R.T = std::min(P->ub, nb - b0);
        R.Tp = (R.T + 127) / 128 * 128;
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
