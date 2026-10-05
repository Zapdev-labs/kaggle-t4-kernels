// Speculative decoding (milestone M5, DESIGN.md section 7): MTP drafts (blk.64) + a batched verify of k + 1 tokens
// whose every column is bit-identical to a single-token decode step, so the greedy output equals plain greedy decode.
//
// Per iteration, per GPU, two CUDA graphs (all position state on the device, no host round trip):
//   draft  : MTP catch-up over the k + 1 verify rows (token y_j paired with the target h_final of the same column, at
//            position vpos + 1 + j; only rows j <= nacc are valid, later rows are overwritten before they are read),
//            row nacc -> draft head -> d1; then k - 1 chained single-row MTP passes (token d_{i-1}, h' of the previous
//            pass) -> d2..dk. vt = [y_nacc, d1..dk].
//   verify : the 64 trunk layers + head on the k + 1 tokens vt at positions pos .. pos + k (M-column GEMVs, per-column
//            attention with the decode split boundaries, DeltaNet over the k + 1 tokens with a state snapshot after
//            every token), argmax per column, exchange, device-side greedy acceptance (n = longest prefix with
//            vt[i + 1] == y_i), emit y_0..y_n, pos += n + 1, DeltaNet snapshot index += n + 1.
// The conv ring has CR = 16 slots (positions pos - 3 .. pos + k never alias), and DeltaNet keeps NS = k + 2 snapshot
// buffers, so rejected tokens need no replay. The verify and draft graphs use separate all-reduce mailboxes (their AR
// counts differ, so slot parity would not alternate across graph boundaries on a shared mailbox).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#include <map>

#include "cupti_trace.h"
#include "kernels/tp_gemv_impl.cuh"
#include "kernels/tp_gemv_tc.cuh"
#include "model.h"
#include "tp.h"
#include "tp_api.h"

namespace tp {
extern int g_p4u;
namespace spec {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

// 1: M > 1 P4 GEMVs go through the int4 tensor-core kernel (bit-identical to the dp4a path, see the header). Read at
// graph capture time (the dispatch is baked into the graph).
int g_spec_tc = 0;

constexpr int DM = 5120;
constexpr size_t SZS = (size_t)24 * 128 * 128;          // DeltaNet state floats per layer per GPU
constexpr int WSZ = 2 * tp::NSPLIT * 6 * 258;            // attention split workspace floats per column
constexpr int AMB = 2 * tp::MMAX;                         // argmax mailbox floats per slot
constexpr size_t RBS = (size_t)tp::MMAX * 24 * 128;      // replay stash floats per layer
constexpr int TPD_SPEC = tp::MMAX + 1;                   // spec_tc token tile (8): xq2 / xm2 pad row

// ------------------------------------------------------------------------------------------------ kernels

__global__ void k_embed_m(const uint8_t* __restrict__ embd, const int* tok, float* h) {
    const int t = blockIdx.y;
    int tk = tok[t];
    if (tk < 0 || tk >= 248320) tk = 0;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint8_t* b = embd + (size_t)tk * 2880 + (i >> 5) * 18;
    const float d = h2f_u16((uint16_t)(b[0] | (b[1] << 8)));
    const int e = i & 31;
    const int q = e < 16 ? (b[2 + e] & 15) : (b[2 + e - 16] >> 4);
    h[(size_t)t * DM + i] = __fmul_rn((float)(q - 8), d);
}

// AR + RMSNorm + q8 over M rows (grid (20, M) x 256): row r = blockIdx.y; each block reduces the full row's sum of
// squares exactly like k_ar_norm_mb and normalizes its own 256 elements, so every row is bit-identical to the
// single-token kernel. own / rx: slot bases (rows of 5120). pub (P2P): every block copies its slice of its row's
// partial to the peer's mailbox; the last block to finish (local counter) sets the peer flag. WAIT: spin on the local
// flag (peer-written). own == nullptr: no all-reduce (h is used as is).
template <bool WAIT>
__global__ void __launch_bounds__(256) k_ar_norm_m(const float* h, float* h_out, const float* own, const float* rx,
                                                   const unsigned* flag, StepState* st, int idx,
                                                   const float* __restrict__ w, float* xn, int8_t* xq, int2* xm,
                                                   float* pub_peer_rx, unsigned* pub_peer_flag, unsigned* cnt) {
    __shared__ float red[8];
    __shared__ int s_ok;
    const int tid = threadIdx.x, row = blockIdx.y;
    const int e = blockIdx.x * 256 + tid;
    const size_t ro = (size_t)row * DM;
    const unsigned ep = epoch_of(st, idx);
    const float wv = w[e];
    if (pub_peer_flag && blockIdx.x == 0) {  // one publisher block per row: coalesced 20 KB copy, one system fence
        const float4* src = (const float4*)(own + ro);
        float4* dst = (float4*)(pub_peer_rx + ro);
#pragma unroll
        for (int k = 0; k < 5; k++) dst[tid + 256 * k] = src[tid + 256 * k];
        __syncthreads();
        if (tid == 0) {
            __threadfence_system();
            const unsigned old = atomicAdd(cnt, 1u);
            if (old == gridDim.y - 1) {
                atomicExch(cnt, 0u);
                __threadfence_system();
                st_vol_u32(pub_peer_flag, ep);
            }
        }
    }
    float4 xv[5], ov[5];
    const float4* h4 = (const float4*)(h + ro);
#pragma unroll
    for (int k = 0; k < 5; k++) xv[k] = h4[tid + 256 * k];
    if (own) {
#pragma unroll
        for (int k = 0; k < 5; k++) ov[k] = ((const float4*)(own + ro))[tid + 256 * k];
    }
    if (WAIT) {
        if (tid == 0) {
            s_ok = wait_flag(flag, ep);
            if (!s_ok) st->err = 8000 + idx;
        }
        __syncthreads();
    }
    if (own) {
        float4 rv[5];
#pragma unroll
        for (int k = 0; k < 5; k++) rv[k] = ld_vol_f4((const float4*)(rx + ro) + tid + 256 * k);
#pragma unroll
        for (int k = 0; k < 5; k++) {
            xv[k].x = xv[k].x + (ov[k].x + rv[k].x);
            xv[k].y = xv[k].y + (ov[k].y + rv[k].y);
            xv[k].z = xv[k].z + (ov[k].z + rv[k].z);
            xv[k].w = xv[k].w + (ov[k].w + rv[k].w);
        }
    }
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 5; k++) ss += xv[k].x * xv[k].x + xv[k].y * xv[k].y + xv[k].z * xv[k].z + xv[k].w * xv[k].w;
    ss = block_sum(ss, red);
    const float scale = rsqrtf(ss / 5120.f + 1e-6f);
    float x = h[ro + e];
    if (own) {
        x = x + (own[ro + e] + ld_vol_f32(rx + ro + e));
        h_out[ro + e] = x;
    }
    const float y = (x * scale) * wv;
    xn[ro + e] = y;
    quant_warp(y, xq + ro + e, xm + (size_t)row * 160 + (e >> 5));
}

// no-P2P fallback for M rows (grid M x 640): block r publishes own row r to the peer's host-mapped mailbox, the last
// block sets the peer's host flag; all wait for the local host flag and copy the peer's row r to the device rx
__global__ void __launch_bounds__(640) k_pull_m(const unsigned* hflag, const float* hrx, float* rx, StepState* st,
                                                int idx, const float* own, float* pub_peer_hrx, unsigned* pub_peer_hflag,
                                                unsigned* cnt) {
    __shared__ int ok;
    const int row = blockIdx.x, tid = threadIdx.x;
    const size_t ro = (size_t)row * DM;
    const unsigned ep = epoch_of(st, idx);
    {
        const float4* src = (const float4*)(own + ro);
        float4* dst = (float4*)(pub_peer_hrx + ro);
        for (int i = tid; i < 1280; i += blockDim.x) dst[i] = src[i];
    }
    __syncthreads();
    if (tid == 0) {
        __threadfence_system();
        const unsigned old = atomicAdd(cnt, 1u);
        if (old == gridDim.x - 1) {
            atomicExch(cnt, 0u);
            __threadfence_system();
            st_vol_u32(pub_peer_hflag, ep);
        }
        ok = wait_flag(hflag, ep);
        if (!ok) st->err = 8300 + idx;
    }
    __syncthreads();
    const float* p = hrx + ro + tid * 8;
    float4 v = ld_vol_f4((const float4*)p), u = ld_vol_f4((const float4*)(p + 4));
    ((float4*)(rx + ro))[tid * 2] = v;
    ((float4*)(rx + ro))[tid * 2 + 1] = u;
}

// DeltaNet over M consecutive tokens (grid 96 = 24 local v heads x 4 slices of 32 value columns; 256 threads), the
// per-token arithmetic of d_gdn with the state kept in registers between tokens. Conv history of token t at position
// pos - j comes from column t - j of y (same values the single-token kernel stores in its ring) or, before the first
// column, from the CR-slot ring. The state after token t goes to snapshot buffer (sidx + 1 + t) % ns. The last block of
// each head then applies the gated RMSNorm + q8 for every column (k_gdn_gn arithmetic).
// RB (rollback by replay, option spec_rb): only the state after the last token goes to buffer sidx ^ 1, and per token
// the k row, the v conv output and decay / beta are stashed (rb_k / rb_d [M][24][128], rb_g [M][24]) for k_gdn_replay
template <bool RB>
__global__ void __launch_bounds__(256) k_gdn_m(const float* __restrict__ y, int ldy, const float* __restrict__ yab,
                                               float* ring, const float* __restrict__ cw,
                                               const float* __restrict__ ssm_a, const float* __restrict__ ssm_dt,
                                               float* Sl, size_t bstride, int ns, float* o, const StepState* st,
                                               int M, unsigned* cnt, const float* __restrict__ gw, int8_t* xq,
                                               int2* xm, float* rb_k, float* rb_d, float* rb_g) {
    __shared__ float sq[128], sk[128], sv[32], red[8];
    const int vl = blockIdx.x >> 2, sl = blockIdx.x & 3, kl = vl & 7, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int pos0 = st->pos, sidx = st->sidx;
    const float* Sin = Sl + (size_t)sidx * bstride;
    float s[4][4];
#pragma unroll
    for (int cc = 0; cc < 4; cc++) {
        const int col = sl * 32 + warp * 4 + cc;
        const float* Sp = Sin + ((size_t)vl * 128 + col) * 128;
#pragma unroll
        for (int r = 0; r < 4; r++) s[cc][r] = Sp[r * 32 + lane];
    }
    const float dtv = ssm_dt[vl], av = ssm_a[vl];
    const int chv = 2048 + vl * 128 + sl * 32 + (tid & 31);
    float4 vw = make_float4(0.f, 0.f, 0.f, 0.f);
    if (tid < 32) vw = __ldg((const float4*)(cw + (size_t)chv * 4));
    const int ch = tid < 128 ? kl * 128 + tid : 1024 + kl * 128 + (tid - 128);
    const float4 wc = __ldg((const float4*)(cw + (size_t)ch * 4));
    for (int t = 0; t < M; t++) {
        const int pos = pos0 + t;
        const float* yt = y + (size_t)t * ldy;
        // value of conv channel c at position pos - j (j = 1..3)
        auto hist = [&](int c, int j) -> float {
            return t - j >= 0 ? __ldcg(y + (size_t)(t - j) * ldy + c) : ring[((pos - j) & (CR - 1)) * 5120 + c];
        };
        const float yb = __ldcg(yab + t * 64 + 24 + vl), ya_ = __ldcg(yab + t * 64 + vl);
        float vx = 0.f, vr0 = 0.f, vr1 = 0.f, vr2 = 0.f;
        if (tid < 32) {
            vx = __ldcg(yt + chv);
            vr0 = hist(chv, 3);
            vr1 = hist(chv, 2);
            vr2 = hist(chv, 1);
        }
        {
            const float x = __ldcg(yt + ch);
            const float r0 = hist(ch, 3), r1 = hist(ch, 2), r2 = hist(ch, 1);
            float sum = 0.f;
            sum += r0 * wc.x;
            sum += r1 * wc.y;
            sum += r2 * wc.z;
            sum += x * wc.w;
            const float a = sum / (1.0f + expf(-sum));
            if (vl < 8 && sl == 0) ring[(pos & (CR - 1)) * 5120 + ch] = x;
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
            float sum = 0.f;
            sum += vr0 * vw.x;
            sum += vr1 * vw.y;
            sum += vr2 * vw.z;
            sum += vx * vw.w;
            sv[tid] = sum / (1.0f + expf(-sum));
            ring[(pos & (CR - 1)) * 5120 + chv] = vx;
        }
        __syncthreads();
        const float beta = 1.0f / (1.0f + expf(-yb));
        const float xg = ya_ + dtv;
        const float sp = xg > 20.0f ? xg : logf(1.0f + expf(xg));
        const float gv = expf(sp * av);
        float kr[4], qr[4];
#pragma unroll
        for (int r = 0; r < 4; r++) { kr[r] = sk[r * 32 + lane]; qr[r] = sq[r * 32 + lane]; }
        float* Sout = Sl + (size_t)(RB ? (sidx ^ 1) : (sidx + 1 + t) % ns) * bstride;
        if (RB) {
            if (sl == 0 && tid < 128) rb_k[((size_t)t * 24 + vl) * 128 + tid] = sk[tid];
            if (sl == 0 && tid == 0) { rb_g[(t * 24 + vl) * 2] = gv; rb_g[(t * 24 + vl) * 2 + 1] = beta; }
        }
#pragma unroll
        for (int cc = 0; cc < 4; cc++) {
            const int col = sl * 32 + warp * 4 + cc;
            float* Sp = Sout + ((size_t)vl * 128 + col) * 128;
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
                s[cc][r] = sn;
                if (!RB || t == M - 1) Sp[r * 32 + lane] = sn;
            }
            if (RB && lane == 0) rb_d[((size_t)t * 24 + vl) * 128 + col] = sv[warp * 4 + cc];
            a = warp_sum(a);
            if (lane == 0) o[(size_t)t * 3072 + vl * 128 + col] = a * (1.0f / sqrtf(128.0f));
        }
        __syncthreads();  // shared q/k/v rows are rewritten by the next token
    }
    __shared__ int s_last;
    __shared__ float red2[8];
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned old = atomicAdd(cnt + vl, 1u);
        s_last = old == 3u;
        if (s_last) cnt[vl] = 0u;
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    const float gwv = tid < 128 ? gw[tid] : 0.f;
    for (int t = 0; t < M; t++) {
        const float zz = tid < 128 ? __ldcg(y + (size_t)t * ldy + 5120 + vl * 128 + tid) : 0.f;
        const float x = tid < 128 ? __ldcg(o + (size_t)t * 3072 + vl * 128 + tid) : 0.f;
        const float ss = block_sum(x * x, red2);
        if (tid < 128) {
            const float scale = rsqrtf(ss / 128.0f + 1e-6f);
            const float val = ((x * scale) * gwv) * (zz / (1.0f + expf(-zz)));
            quant_warp(val, xq + (size_t)t * 3072 + vl * 128 + tid, xm + (size_t)t * 96 + vl * 4 + (tid >> 5));
        }
    }
}

// rollback by replay: after a partial acceptance (nacc < k) rebuild the state after token nacc from the verify's input
// state (buffer sidx ^ 1, sidx already flipped by the accept) and the stash, with the k_gdn_m update arithmetic.
// (kv and delta are recomputed per lane exactly as in k_gdn_m; the stash holds k, the v conv output, decay and beta)
// grid (96, 48 layers) x 256
__global__ void __launch_bounds__(256) k_gdn_replay(float* Sb, size_t bstride, const StepState* st, int k,
                                                    const float* rbk, const float* rbd, const float* rbg,
                                                    size_t rbstride) {
    const int n = st->nacc;
    if (!st->rbp || n >= k) return;
    const int sidx = st->sidx, li = blockIdx.y;
    const int vl = blockIdx.x >> 2, sl = blockIdx.x & 3, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float* Sin = Sb + (size_t)(sidx ^ 1) * bstride + (size_t)li * SZS;
    float* Sout = Sb + (size_t)sidx * bstride + (size_t)li * SZS;
    const float* rb_k = rbk + (size_t)li * rbstride;
    const float* rb_d = rbd + (size_t)li * rbstride;
    const float* rb_g = rbg + (size_t)li * MMAX * 48;
    float s[4][4];
#pragma unroll
    for (int cc = 0; cc < 4; cc++) {
        const int col = sl * 32 + warp * 4 + cc;
        const float* Sp = Sin + ((size_t)vl * 128 + col) * 128;
#pragma unroll
        for (int r = 0; r < 4; r++) s[cc][r] = Sp[r * 32 + lane];
    }
    for (int t = 0; t <= n; t++) {
        const float gv = rb_g[(t * 24 + vl) * 2], beta = rb_g[(t * 24 + vl) * 2 + 1];
        float kr[4];
#pragma unroll
        for (int r = 0; r < 4; r++) kr[r] = rb_k[((size_t)t * 24 + vl) * 128 + r * 32 + lane];
#pragma unroll
        for (int cc = 0; cc < 4; cc++) {
            const int col = sl * 32 + warp * 4 + cc;
            float kv = 0.f;
#pragma unroll
            for (int r = 0; r < 4; r++) kv += s[cc][r] * kr[r];
            kv = warp_sum(kv);
            const float delta = (rb_d[((size_t)t * 24 + vl) * 128 + col] - gv * kv) * beta;
#pragma unroll
            for (int r = 0; r < 4; r++) {
                const float sn = gv * s[cc][r] + kr[r] * delta;
                s[cc][r] = sn;
            }
        }
    }
#pragma unroll
    for (int cc = 0; cc < 4; cc++) {
        const int col = sl * 32 + warp * 4 + cc;
        float* Sp = Sout + ((size_t)vl * 128 + col) * 128;
#pragma unroll
        for (int r = 0; r < 4; r++) Sp[r * 32 + lane] = s[cc][r];
    }
}

// attention over M columns: column t at position *pbase + poff + t (decode arithmetic and split boundaries)
__global__ void k_attn_prep_m(const float* ya, int ldy, const float* qw, const float* kw, float* qa, __half* kc,
                              __half* vc, int max_ctx, const int* pbase, int poff, float theta_scale) {
    const int t = blockIdx.y;
    d_attn_prep(ya + (size_t)t * ldy, qw, kw, qa + (size_t)t * 12 * 256, kc, vc, max_ctx, *pbase + poff + t,
                theta_scale, blockIdx.x);
}
__global__ void __launch_bounds__(256, 2) k_attn_split_m(const float* qa, const __half* kc, const __half* vc, float* ws,
                                                         int max_ctx, const int* pbase, int poff) {
    const int t = blockIdx.z;
    d_attn_split(qa + (size_t)t * 12 * 256, kc, vc, ws + (size_t)t * WSZ, max_ctx, *pbase + poff + t, blockIdx.x,
                 blockIdx.y);
}
__global__ void __launch_bounds__(256) k_attn_combine_m(const float* ws, const float* ya, int ldy, const int* pbase,
                                                        int poff, int8_t* xq, int2* xm) {
    const int t = blockIdx.y;
    d_attn_combine_q8(ws + (size_t)t * WSZ, ya + (size_t)t * ldy, *pbase + poff + t, xq + (size_t)t * 3072,
                      xm + (size_t)t * 96, Pf{}, blockIdx.x);
}

__global__ void __launch_bounds__(256) k_argmax_part_m(const float* x, int n, int ldx, float* apart) {
    d_argmax_part(x + (size_t)blockIdx.y * ldx, n, apart + (size_t)blockIdx.y * 2 * NBA, blockIdx.x, NBA);
}

// per-column argmax over this GPU's partials, exchange with the peer (max value, lowest index on ties), then
// MODE 0 (verify): greedy acceptance + state update + token emission; MODE 1 (draft di): vt[di] = argmax
struct XArgs {
    const float* apart;
    int M, row0;
    float* amb;             // local mailbox [2][AMB] (peer-written)
    const unsigned* aflag;  // local flags [2]
    float* peer_amb;
    unsigned* peer_aflag;
    StepState* st;
    int idx;                // epoch index (slot = idx & 1)
    int k, ns, di, rb;
    int* vt;                // verify input tokens [MMAX]
    int* yv;                // verify argmax per column [MMAX]
    int* ring;              // host-mapped token ring (GPU0) or nullptr
    int* hcnt;              // host-mapped emitted-token counter (GPU0) or nullptr
    int* hist;              // [MMAX + 1] accepted-length histogram (device); [MMAX + 1 ..] the same for n-gram steps
    int* htok;              // token history by position (G.prompt): emitted tokens are appended
    int max_ctx;
};
template <int MODE>
__global__ void k_argmax_x(const XArgs a) {
    __shared__ int s_y[MMAX];
    __shared__ int s_ok;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int slot = a.idx & 1;
    const unsigned e = epoch_of(a.st, a.idx);
    float bv = -FLT_MAX;
    int bi = 0x7fffffff;
    if (w < a.M) {
        const float* ap = a.apart + (size_t)w * 2 * NBA;
        for (int b = lane; b < NBA; b += 32) {
            const float v = __ldcg(ap + b);
            const int i = __float_as_int(__ldcg(ap + NBA + b));
            if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        bi += a.row0;
        if (lane == 0) {
            a.peer_amb[slot * AMB + 2 * w] = bv;
            a.peer_amb[slot * AMB + 2 * w + 1] = __int_as_float(bi);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        st_vol_u32(a.peer_aflag + slot, e);
        s_ok = wait_flag(a.aflag + slot, e);
        if (!s_ok) a.st->err = 8500 + a.idx;
    }
    __syncthreads();
    if (w < a.M && lane == 0) {
        const float pv = ld_vol_f32(a.amb + slot * AMB + 2 * w);
        const int pi = __float_as_int(ld_vol_f32(a.amb + slot * AMB + 2 * w + 1));
        int best = bi;
        if (pv > bv || (pv == bv && pi < bi)) best = pi;
        s_y[w] = best;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    StepState* st = a.st;
    if (MODE == 1) {
        a.vt[a.di] = s_y[0];
        return;
    }
    int n = 0;
    while (n < a.k && a.vt[n + 1] == s_y[n]) n++;
    for (int t = 0; t < a.M; t++) a.yv[t] = s_y[t];
    const int p = st->pos;
    st->vpos = p;
    st->nacc = n;
    st->pos = p + n + 1;
    st->sidx = a.rb ? (st->sidx ^ 1) : (st->sidx + n + 1) % a.ns;
    st->rbp = a.rb && n < a.k;
    st->token = s_y[n];
    st->last_tok = s_y[n];
    for (int i = 0; i <= n; i++)
        if (p + 1 + i < a.max_ctx) a.htok[p + 1 + i] = s_y[i];
    int ne = st->nemit;
    if (a.ring) {
        for (int i = 0; i <= n; i++) a.ring[(ne + i) % RING] = s_y[i];
    }
    ne += n + 1;
    st->nemit = ne;
    if (a.hcnt) {
        __threadfence_system();
        *(volatile int*)a.hcnt = ne;
    }
    if (a.hist) a.hist[(st->ngu ? MMAX + 1 : 0) + n]++;
    st->step = st->step + 1u;
}

// MTP catch-up: row nacc of the shared-head-norm output feeds the draft head; vt[0] = y_nacc (the next verify's first
// token). grid 20 x 256
__global__ void k_select(const float* hp, const int8_t* xq, const int2* xm, const StepState* st, float* hs,
                         int8_t* xqd, int2* xmd, const int* yv, int* vt) {
    const int n = st->nacc;
    const int e = blockIdx.x * 256 + threadIdx.x;
    hs[e] = hp[(size_t)n * DM + e];
    xqd[e] = xq[(size_t)n * DM + e];
    if ((e & 31) == 0) xmd[e >> 5] = xm[(size_t)n * 160 + (e >> 5)];
    if (e == 0) vt[0] = yv[n];
}

// prompt lookup (n-gram) drafts: the longest (<= ngmax) match of the history suffix ending at P = st->pos (htok[P] =
// vt[0], the pending token) that ends at an earlier position j (most recent on ties); with a match of >= ngmin tokens
// the drafts become htok[j + 1 .. j + k] (periodically extended past P), else the MTP drafts stay. 1 block x 1024
__global__ void __launch_bounds__(1024) k_ngram(const int* htok, StepState* st, int* vt, int k, int ngmin, int ngmax) {
    __shared__ int s_best;
    const int P = st->pos, lo = st->ng_lo;
    if (threadIdx.x == 0) s_best = -1;
    __syncthreads();
    const int pt = htok[P];
    for (int j = lo + threadIdx.x; j < P; j += blockDim.x) {
        if (htok[j] != pt) continue;
        int l = 1;
        while (l < ngmax && j - l >= lo && htok[j - l] == htok[P - l]) l++;
        if (l >= ngmin) atomicMax(&s_best, (l << 20) | j);
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    const int b = s_best;
    st->ngu = b >= 0;
    if (b < 0) return;
    const int j = b & 0xFFFFF, per = P - j;
    for (int i = 1; i <= k; i++) {
        int q = j + i;
        while (q > P) q -= per;
        vt[i] = htok[q];
    }
}

// timeline marker between graphs (spec_trace)
__global__ void k_mark_draft_end() {}
__global__ void k_mark_verify_end() {}

// debug (spec_force): drafts are the expected continuation stored in the prompt buffer
__global__ void k_force(int* vt, const int* prompt, const StepState* st, int k, int max_ctx) {
    const int p = st->pos;
    for (int i = 1; i <= k; i++) vt[i] = p + i < max_ctx ? prompt[p + i] : 0;
}

// conv ring conversion between the decode engine's 4-slot ring and the spec engine's CR-slot ring (positions
// pos - 3 .. pos - 1); grid 20 x 256
__global__ void k_ring_conv(const float* src, int src_slots, float* dst, int dst_slots, const int* pp) {
    const int pos = *pp;
    const int e = blockIdx.x * 256 + threadIdx.x;
    for (int j = 1; j <= 3; j++) {
        const int q = pos - j;
        if (q < 0) continue;
        dst[(size_t)(q % dst_slots) * 5120 + e] = src[(size_t)(q % src_slots) * 5120 + e];
    }
}

// ------------------------------------------------------------------------------------------------ GEMV dispatch (M columns)
template <int M>
void gemv_mt(const FW& W, const int8_t* xq, const int2* xm, float* y, cudaStream_t s, const SegArgs* seg,
             const ProArgs* sq) {
    const ArArgs A{};
    const SegArgs S = seg ? *seg : SegArgs{};
    const ProArgs P = sq ? *sq : ProArgs{};
    const int f = W.L.fmt, nch = W.L.nch, rpl = W.L.rpl;
    const bool isseg = seg != nullptr, issq = sq != nullptr;
    // spec_tc: the M > 1 P4 GEMVs on the int4 tensor cores (bit-identical to the dp4a path below, header for why).
    // qkvz's 48 fp32 alpha/beta rows run in k_seg_m (the k_gemv SEG arithmetic verbatim).
    if (g_spec_tc && M > 1 && f == FAST_P4 && (W.L.K % 64) == 0) {
        t4q::gtc::TcArgs A;
        A.q = t4q::gemm8::make_args(W.L, W.base, nullptr, nullptr, nullptr, nullptr, 0, 0, 0);
        A.xq = xq;
        A.xms = xm;
        A.y = y;
        A.ldy = W.L.N;
        A.N = W.L.N;
        A.K = W.L.K;
        A.T = M;
        if (issq) {
            A.sq_xq = P.sq_xq;
            A.sq_xm = P.sq_xm;
        }
        CK(t4q::gtc::gemv_tc_launch(rpl, issq, A, s));
        if (isseg) t4q::gtc::k_seg_m<<<(S.nrows + 7) / 8, 256, 0, s>>>(S.w, S.x, S.y, S.nrows, M);
        return;
    }
#define T4Q_GM(FMT, RPL, NCH, SEG_, SQ_)                                                                        \
    if (f == FMT && rpl == RPL && nch == NCH && isseg == SEG_ && issq == SQ_) {                                 \
        launch_gemv<FMT, RPL, NCH, false, SEG_, PRO_NONE, SQ_, (FMT == FAST_P4 ? 2 : 1), M>(W, xq, xm, y, s, A, S, \
                                                                                             P);                \
        return;                                                                                                 \
    }
    T4Q_GM(FAST_P4, 2, 10, true, false)    // DeltaNet qkvz + alpha/beta rows (RPL 2 for the N-split weights)
    T4Q_GM(FAST_P4, 2, 10, false, false)   // attention q|k|v (rpl 2 default: v41 qkvz-class N sweep)
    T4Q_GM(FAST_P4, 4, 10, false, false)   // attention q|k|v packed rpl 4 (T4Q_RPL_QKV_A A/B, N 7168)
    T4Q_GM(FAST_K5, 2, 6, false, false)    // ssm_out
    T4Q_GM(FAST_P4, 4, 6, false, false)    // attn_output
    T4Q_GM(FAST_P4, 4, 10, false, true)    // gate|up + silu q8 epilogue
    T4Q_GM(FAST_P4, 4, 17, false, false)   // ffn_down Q4_0
    T4Q_GM(FAST_P4M, 4, 17, false, false)  // ffn_down Q4_1
    T4Q_GM(FAST_K6, 2, 10, false, false)   // lm_head / draft head
    T4Q_GM(FAST_Q8, 2, 10, false, false)   // MTP eh_proj
#undef T4Q_GM
    throw std::runtime_error("spec gemv: no instantiation for fmt " + std::to_string(f) + " rpl " +
                             std::to_string(rpl) + " nch " + std::to_string(nch) + " seg " + std::to_string(isseg) +
                             " sq " + std::to_string(issq));
}
void gemv_m(int M, const FW& W, const int8_t* xq, const int2* xm, float* y, cudaStream_t s,
            const SegArgs* seg = nullptr, const ProArgs* sq = nullptr) {
    if (tp::g_p4u == 0) throw std::runtime_error("spec decoding needs p4u = 1");
    switch (M) {
        case 1: return gemv_mt<1>(W, xq, xm, y, s, seg, sq);
        case 2: return gemv_mt<2>(W, xq, xm, y, s, seg, sq);
        case 3: return gemv_mt<3>(W, xq, xm, y, s, seg, sq);
        case 4: return gemv_mt<4>(W, xq, xm, y, s, seg, sq);
        case 5: return gemv_mt<5>(W, xq, xm, y, s, seg, sq);
        case 6: return gemv_mt<6>(W, xq, xm, y, s, seg, sq);
        case 7: return gemv_mt<7>(W, xq, xm, y, s, seg, sq);
        default: throw std::runtime_error("spec gemv: M out of range");
    }
}

// ------------------------------------------------------------------------------------------------ buffers
struct SG {  // per GPU
    float *hb[2] = {nullptr, nullptr};  // residual rows [MMAX][5120]
    float *xn = nullptr, *hf = nullptr, *hp = nullptr, *hs = nullptr, *zero = nullptr;
    float *y = nullptr, *yab = nullptr, *o = nullptr, *qa = nullptr, *ws = nullptr, *logits = nullptr, *logd = nullptr;
    int8_t *xq = nullptr, *xq2 = nullptr, *xqd = nullptr;
    int2 *xm = nullptr, *xm2 = nullptr, *xmd = nullptr;
    float* part = nullptr;  // [2 slots][MMAX][5120]
    // all-reduce mailboxes: [0] verify graph, [1] draft graph. P2P: rx/flag in VRAM; else host-mapped hrx/hflag
    float* rx[2] = {nullptr, nullptr};
    unsigned* flag[2] = {nullptr, nullptr};
    float* hrx[2] = {nullptr, nullptr};
    unsigned* hflag[2] = {nullptr, nullptr};
    float* peer_rx[2] = {nullptr, nullptr};
    unsigned* peer_flag[2] = {nullptr, nullptr};
    // argmax exchange mailboxes ([0] verify, [1] draft)
    float* amb[2] = {nullptr, nullptr};
    unsigned* aflag[2] = {nullptr, nullptr};
    float* peer_amb[2] = {nullptr, nullptr};
    unsigned* peer_aflag[2] = {nullptr, nullptr};
    unsigned* cnt = nullptr;   // [8] publish counters
    unsigned* gcnt = nullptr;  // [32] gdn head counters
    float* apart = nullptr;    // [MMAX][2 * NBA]
    int *vt = nullptr, *yv = nullptr, *hist = nullptr, *pscr = nullptr;
    float* Sb = nullptr;       // [NSNAP][48][SZS] DeltaNet state snapshots
    float *rbk = nullptr, *rbd = nullptr, *rbg = nullptr;  // replay stash [48][MMAX][24][128] (rbg [48][MMAX][24])
    float* ring = nullptr;     // [48][CR][5120]
    float* hpr = nullptr;      // prompt catch-up h rows
    size_t hpr_n = 0;
    cudaGraphExec_t ev = nullptr, ed = nullptr;  // verify / draft graphs
    cudaEvent_t e0 = nullptr, e1 = nullptr, e2 = nullptr;
};
struct Spec {
    SG G[2];
    int k = 0, dv = -1, force = 0, ng = 0, ngmax = 0, rb = -1, sqt = 0, tc = 0;  // configuration of the captured graphs
    bool rb_cfg = false;
    int* h_cnt = nullptr;           // host-mapped emitted-token counter (GPU0 writes)
    int* d_cnt = nullptr;
    bool active = false;            // spec state (snapshots / ring / MTP KV) is in sync with the engine
    // stats
    long steps = 0, emitted = 0, drafted = 0, iters_timed = 0;
    double gen_s = 0, ms_draft = 0, ms_verify = 0;
    int last_hist[MMAX + 1] = {0};
    int last_hist_ng[MMAX + 1] = {0};  // steps whose drafts came from prompt lookup
    long dv_hits = 0, dv_total = 0;  // target tokens inside the draft vocab (M5b in-subset rate)
    double prep_s = 0;
    // last t4q_generate call
    int last_k = 0, last_dv = 0;
    long last_steps = 0, last_acc = 0;
    double last_s = 0;
    std::string trace_json;
};

template <class T>
T* dz(size_t n) {
    T* p;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}
template <class T>
T* hz(size_t n) {
    T* p;
    CK(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocMapped | cudaHostAllocPortable));
    memset(p, 0, n * sizeof(T));
    return p;
}

Spec* spec_get(t4q_ctx* c) {
    tp::State& S = *c->tps;
    if (S.spec) return (Spec*)S.spec;
    if (!S.G[0].mtp.eh.ok()) throw std::runtime_error("spec: the GGUF has no MTP block (blk.64)");
    Spec* P = new Spec();
    for (int g = 0; g < 2; g++) {
        SG& B = P->G[g];
        CK(cudaSetDevice(g));
        for (int i = 0; i < 2; i++) B.hb[i] = dz<float>((size_t)MMAX * DM);
        B.xn = dz<float>((size_t)MMAX * DM);
        B.hf = dz<float>((size_t)MMAX * DM);
        B.hp = dz<float>((size_t)MMAX * DM);
        B.hs = dz<float>(DM);
        B.zero = dz<float>((size_t)MMAX * DM);
        B.y = dz<float>((size_t)MMAX * 17408);
        B.yab = dz<float>((size_t)MMAX * 64);
        B.o = dz<float>((size_t)MMAX * 3072);
        B.qa = dz<float>((size_t)MMAX * 12 * 256);
        B.ws = dz<float>((size_t)MMAX * WSZ);
        B.logits = dz<float>((size_t)MMAX * 124160);
        B.logd = dz<float>(124160);
        B.xq = dz<int8_t>((size_t)MMAX * 17408);
        B.xm = dz<int2>((size_t)MMAX * 17408 / 32);
        // 8 rows (not MMAX): the spec_tc kernel's 8-token tile reads the pad row 7 (stale bytes feed discarded lanes)
        B.xq2 = dz<int8_t>((size_t)TPD_SPEC * 8704);
        B.xm2 = dz<int2>((size_t)TPD_SPEC * 8704 / 32);
        B.xqd = dz<int8_t>(DM);
        B.xmd = dz<int2>(DM / 32);
        B.part = dz<float>((size_t)2 * MMAX * DM);
        for (int i = 0; i < 2; i++) {
            if (S.p2p) {
                B.rx[i] = dz<float>((size_t)2 * MMAX * DM);
                B.flag[i] = dz<unsigned>(8);
                B.amb[i] = dz<float>(2 * AMB);
                B.aflag[i] = dz<unsigned>(8);
            } else {
                B.rx[i] = dz<float>((size_t)2 * MMAX * DM);
                B.hrx[i] = hz<float>((size_t)2 * MMAX * DM);
                B.hflag[i] = hz<unsigned>(8);
                B.amb[i] = hz<float>(2 * AMB);
                B.aflag[i] = hz<unsigned>(8);
            }
        }
        B.cnt = dz<unsigned>(8);
        B.gcnt = dz<unsigned>(32);
        B.apart = dz<float>((size_t)MMAX * 2 * NBA);
        B.vt = dz<int>(MMAX);
        B.yv = dz<int>(MMAX);
        B.hist = dz<int>(2 * (MMAX + 1));
        B.pscr = dz<int>(4);
        B.Sb = dz<float>((size_t)NSNAP * 48 * SZS);
        B.rbk = dz<float>((size_t)48 * RBS);
        B.rbd = dz<float>((size_t)48 * RBS);
        B.rbg = dz<float>((size_t)48 * MMAX * 48);
        B.ring = dz<float>((size_t)48 * CR * 5120);
        CK(cudaEventCreate(&B.e0));
        CK(cudaEventCreate(&B.e1));
        CK(cudaEventCreate(&B.e2));
    }
    for (int g = 0; g < 2; g++) {
        SG& B = P->G[g];
        SG& Q = P->G[1 - g];
        for (int i = 0; i < 2; i++) {
            B.peer_rx[i] = S.p2p ? Q.rx[i] : Q.hrx[i];
            B.peer_flag[i] = S.p2p ? Q.flag[i] : Q.hflag[i];
            B.peer_amb[i] = Q.amb[i];
            B.peer_aflag[i] = Q.aflag[i];
        }
    }
    CK(cudaSetDevice(0));
    CK(cudaHostAlloc(&P->h_cnt, 64, cudaHostAllocMapped | cudaHostAllocPortable));
    memset(P->h_cnt, 0, 64);
    CK(cudaHostGetDevicePointer((void**)&P->d_cnt, P->h_cnt, 0));
    S.spec = P;
    return P;
}

// ------------------------------------------------------------------------------------------------ enqueue
struct SEnq {
    t4q_ctx* c;
    Spec* P;
    int g;
    tp::State& S;
    tp::Gpu& G;
    SG& B;
    cudaStream_t s;
    int mb = 0;  // mailbox: 0 verify, 1 draft
    SEnq(t4q_ctx* c_, Spec* P_, int g_)
        : c(c_), P(P_), g(g_), S(*c_->tps), G(c_->tps->G[g_]), B(P_->G[g_]), s(c_->tps->G[g_].s) {}
    void chk(const char* w) {
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) throw std::runtime_error(std::string("spec launch ") + w + ": " + cudaGetErrorString(e));
    }
    // AR idx (add) or plain norm (h used as is): rows of h_in -> [h_out], norm(w) -> xn / xq / xm rows
    void ar(int idx, const float* h_in, float* h_out, bool add, const float* w, float* xn, int8_t* xq, int2* xm,
            int M) {
        const dim3 grid(20, M);
        if (!add) {
            k_ar_norm_m<false><<<grid, 256, 0, s>>>(h_in, nullptr, nullptr, nullptr, nullptr, G.st, idx, w, xn, xq, xm,
                                                    nullptr, nullptr, nullptr);
            chk("ar_norm_m");
            return;
        }
        const int sl = idx & 1;
        const size_t so = (size_t)sl * MMAX * DM;
        const float* own = B.part + so;
        if (S.p2p) {
            k_ar_norm_m<true><<<grid, 256, 0, s>>>(h_in, h_out, own, B.rx[mb] + so, B.flag[mb] + sl, G.st, idx, w,
                                                   xn, xq, xm, B.peer_rx[mb] + so, B.peer_flag[mb] + sl, B.cnt + mb);
        } else {
            k_pull_m<<<M, 640, 0, s>>>(B.hflag[mb] + sl, B.hrx[mb] + so, B.rx[mb] + so, G.st, idx, own,
                                       B.peer_rx[mb] + so, B.peer_flag[mb] + sl, B.cnt + 2 + mb);
            k_ar_norm_m<false><<<grid, 256, 0, s>>>(h_in, h_out, own, B.rx[mb] + so, nullptr, G.st, idx, w, xn, xq,
                                                    xm, nullptr, nullptr, nullptr);
        }
        chk("ar_m");
    }
    float* part(int idx) { return B.part + (size_t)(idx & 1) * MMAX * DM; }
    void attention(const tp::Layer& L, int M, const int* pbase, int poff, int idx_out) {
        gemv_m(M, L.qkv_a, B.xq, B.xm, B.y, s);
        k_attn_prep_m<<<dim3(14, M), 256, 0, s>>>(B.y, L.qkv_a.L.N, L.q_norm, L.k_norm, B.qa, (__half*)L.kc,
                                                  (__half*)L.vc, S.max_ctx, pbase, poff,
                                                  powf(hp::ROPE_BASE, -2.0f / hp::NROT));
        k_attn_split_m<<<dim3(2, NSPLIT, M), 256, 0, s>>>(B.qa, (const __half*)L.kc, (const __half*)L.vc, B.ws,
                                                          S.max_ctx, pbase, poff);
        k_attn_combine_m<<<dim3(12, M), 256, 0, s>>>(B.ws, B.y, L.qkv_a.L.N, pbase, poff, B.xq, B.xm);
        chk("attention");
        gemv_m(M, L.wo, B.xq, B.xm, part(idx_out), s);
    }
    void ffn(const tp::Layer& L, int M, int idx_out) {
        ProArgs sq;
        sq.sq_xq = B.xq2;
        sq.sq_xm = B.xm2;
        gemv_m(M, L.gateup, B.xq, B.xm, B.y, s, nullptr, &sq);
        gemv_m(M, L.down, B.xq2, B.xm2, part(idx_out), s);
    }
    // ---- verify graph: M = k + 1 tokens vt at positions st->pos ..
    void verify(int k, bool rb) {
        mb = 0;
        const int M = k + 1;
        k_embed_m<<<dim3(20, M), 256, 0, s>>>(G.embd, B.vt, B.hb[0]);
        chk("embed");
        const size_t bstride = (size_t)48 * SZS;
        const int ns = k + 2;
        for (int il = 0; il < 64; il++) {
            const tp::Layer& L = G.L[il];
            int idx = 2 * il - 1;
            if (idx < 0) ar(-1, B.hb[0], nullptr, false, L.attn_norm, B.xn, B.xq, B.xm, M);
            else ar(idx, B.hb[idx & 1], B.hb[(idx + 1) & 1], true, L.attn_norm, B.xn, B.xq, B.xm, M);
            idx = 2 * il;
            if (!L.attn) {
                SegArgs sg;
                sg.w = L.ab; sg.x = B.xn; sg.y = B.yab; sg.nrows = 48;
                gemv_m(M, L.qkvz, B.xq, B.xm, B.y, s, &sg);
                const int li = il - il / 4;
                const size_t ro = (size_t)li * RBS;
                if (rb)
                    k_gdn_m<true><<<96, 256, 0, s>>>(B.y, L.qkvz.L.N, B.yab, B.ring + (size_t)li * CR * 5120, L.conv_w,
                                                     L.ssm_a, L.ssm_dt, B.Sb + (size_t)li * SZS, bstride, ns, B.o, G.st,
                                                     M, B.gcnt, L.ssm_norm, B.xq, B.xm, B.rbk + ro, B.rbd + ro,
                                                     B.rbg + (size_t)li * MMAX * 48);
                else
                    k_gdn_m<false><<<96, 256, 0, s>>>(B.y, L.qkvz.L.N, B.yab, B.ring + (size_t)li * CR * 5120, L.conv_w,
                                                      L.ssm_a, L.ssm_dt, B.Sb + (size_t)li * SZS, bstride, ns, B.o, G.st,
                                                      M, B.gcnt, L.ssm_norm, B.xq, B.xm, nullptr, nullptr, nullptr);
                chk("gdn_m");
                gemv_m(M, L.ssm_out, B.xq, B.xm, part(idx), s);
            } else {
                attention(L, M, &G.st->pos, 0, idx);
            }
            ar(idx, B.hb[idx & 1], B.hb[(idx + 1) & 1], true, L.post_norm, B.xn, B.xq, B.xm, M);
            ffn(L, M, 2 * il + 1);
        }
        // head: h_final rows -> hf (the MTP catch-up's h input), lm_head, argmax, accept
        ar(127, B.hb[1], B.hb[0], true, G.output_norm, B.hf, B.xq, B.xm, M);
        gemv_m(M, G.lm, B.xq, B.xm, B.logits, s);
        k_argmax_part_m<<<dim3(NBA, M), 256, 0, s>>>(B.logits, 124160, 124160, B.apart);
        XArgs a;
        a.apart = B.apart; a.M = M; a.row0 = 124160 * g;
        a.amb = B.amb[0]; a.aflag = B.aflag[0]; a.peer_amb = B.peer_amb[0]; a.peer_aflag = B.peer_aflag[0];
        a.st = G.st; a.idx = 250; a.k = k; a.ns = ns; a.di = 0; a.rb = rb;
        a.vt = B.vt; a.yv = B.yv;
        a.ring = g == 0 ? S.d_ring : nullptr;
        a.hcnt = g == 0 ? P->d_cnt : nullptr;
        a.hist = B.hist;
        a.htok = G.prompt;
        a.max_ctx = S.max_ctx;
        k_argmax_x<0><<<1, 32 * M, 0, s>>>(a);
        chk("accept");
    }
    // ---- one MTP pass over M rows: tokens tok[t], h rows hsrc (GPU1's half), positions *pbase + poff + t; AR idx0..+2
    // on the draft mailbox; output: shared-head-norm rows -> xn_out / xq_out / xm_out
    void mtp_pass(int M, const int* tok, const float* hsrc, const int* pbase, int poff, int idx0, float* xn_out,
                  int8_t* xq_out, int2* xm_out) {
        mb = 1;
        const tp::Mtp& T = G.mtp;
        if (g == 0) {
            k_embed_m<<<dim3(20, M), 256, 0, s>>>(G.embd, tok, B.hb[0]);
            chk("mtp embed");
            ar(-1, B.hb[0], nullptr, false, T.enorm, B.xn, B.xq, B.xm, M);
        } else {
            ar(-1, hsrc, nullptr, false, T.hnorm, B.xn, B.xq, B.xm, M);
        }
        gemv_m(M, T.eh, B.xq, B.xm, part(idx0), s);
        ar(idx0, B.zero, B.hb[1], true, T.L.attn_norm, B.xn, B.xq, B.xm, M);
        attention(T.L, M, pbase, poff, idx0 + 1);
        ar(idx0 + 1, B.hb[1], B.hb[0], true, T.L.post_norm, B.xn, B.xq, B.xm, M);
        ffn(T.L, M, idx0 + 2);
        ar(idx0 + 2, B.hb[0], B.hb[1], true, T.shnorm, xn_out, xq_out, xm_out, M);
    }
    void draft_head(int di, int k, bool dvh) {
        const FW& H = dvh ? G.mtp.lmd : G.lm;
        const int rows = H.L.N;
        gemv_m(1, H, B.xqd, B.xmd, B.logd, s);
        k_argmax_part_m<<<dim3(NBA, 1), 256, 0, s>>>(B.logd, rows, rows, B.apart);
        XArgs a;
        a.apart = B.apart; a.M = 1; a.row0 = rows * g;
        a.amb = B.amb[1]; a.aflag = B.aflag[1]; a.peer_amb = B.peer_amb[1]; a.peer_aflag = B.peer_aflag[1];
        a.st = G.st; a.idx = 200 + di; a.k = k; a.ns = k + 2; a.di = di; a.rb = 0;
        a.vt = B.vt; a.yv = B.yv; a.ring = nullptr; a.hcnt = nullptr; a.hist = nullptr; a.htok = nullptr;
        a.max_ctx = S.max_ctx;
        k_argmax_x<1><<<1, 32, 0, s>>>(a);
        chk("draft argmax");
    }
    // ---- draft graph: catch-up (k + 1 rows) + first draft, then k - 1 chained drafts
    void draft(int k, bool dvh, bool force, int ng, int ngmax) {
        if (P->rb_cfg) {
            k_gdn_replay<<<dim3(96, 48), 256, 0, s>>>(B.Sb, (size_t)48 * SZS, G.st, k, B.rbk, B.rbd, B.rbg, RBS);
            chk("gdn_replay");
        }
        mtp_pass(k + 1, B.yv, B.hf, &G.st->vpos, 1, 0, B.hp, B.xq, B.xm);
        k_select<<<20, 256, 0, s>>>(B.hp, B.xq, B.xm, G.st, B.hs, B.xqd, B.xmd, B.yv, B.vt);
        chk("select");
        draft_head(1, k, dvh);
        for (int i = 2; i <= k; i++) {
            mtp_pass(1, B.vt + (i - 1), B.hs, &G.st->pos, i - 1, 3 * (i - 1), B.hs, B.xqd, B.xmd);
            draft_head(i, k, dvh);
        }
        if (ng > 0) {
            k_ngram<<<1, 1024, 0, s>>>(G.prompt, G.st, B.vt, k, ng, ngmax);
            chk("ngram");
        }
        if (force) {
            k_force<<<1, 1, 0, s>>>(B.vt, G.prompt, G.st, k, S.max_ctx);
            chk("force");
        }
    }
};

void sync2(t4q_ctx* c) {
    for (int g = 0; g < 2; g++) {
        CK(cudaSetDevice(g));
        CK(cudaStreamSynchronize(c->tps->G[g].s));
    }
}

void check_err2(t4q_ctx* c) {
    for (int g = 0; g < 2; g++) {
        tp::StepState st;
        CK(cudaSetDevice(g));
        CK(cudaMemcpy(&st, c->tps->G[g].st, sizeof st, cudaMemcpyDeviceToHost));
        if (st.err) throw std::runtime_error("spec: device error code " + std::to_string(st.err) + " on gpu " +
                                             std::to_string(g));
    }
}

void destroy_graphs(Spec* P) {
    for (int g = 0; g < 2; g++) {
        SG& B = P->G[g];
        CK(cudaSetDevice(g));
        if (B.ev) { cudaGraphExecDestroy(B.ev); B.ev = nullptr; }
        if (B.ed) { cudaGraphExecDestroy(B.ed); B.ed = nullptr; }
    }
}

void capture(t4q_ctx* c, Spec* P, int k, bool dvh, bool force) {
    destroy_graphs(P);
    for (int g = 0; g < 2; g++) {
        tp::Gpu& G = c->tps->G[g];
        SG& B = P->G[g];
        CK(cudaSetDevice(g));
        for (int which = 0; which < 2; which++) {
            cudaGraph_t gr;
            CK(cudaStreamBeginCapture(G.s, cudaStreamCaptureModeThreadLocal));
            SEnq q(c, P, g);
            try {
                if (which == 0) q.verify(k, P->rb_cfg);
                else q.draft(k, dvh, force, c->tps->spec_ng, c->tps->spec_ngmax);
            } catch (...) {
                cudaStreamEndCapture(G.s, &gr);
                throw;
            }
            CK(cudaStreamEndCapture(G.s, &gr));
            CK(cudaGraphInstantiate(which == 0 ? &B.ev : &B.ed, gr, 0));
            cudaGraphDestroy(gr);
        }
    }
    P->k = k;
    P->dv = dvh;
    P->force = force;
    P->ng = c->tps->spec_ng;
    P->ngmax = c->tps->spec_ngmax;
    P->rb = c->tps->spec_rb;
    P->sqt = c->tps->spec_sqt;
}

// enter spec mode: DeltaNet state -> snapshot 0, conv ring -> CR slots, MTP KV catch-up over the prompt rows,
// first draft inputs (y_0 = the pending token, h = the last prompt position's h_final)
void spec_enter(t4q_ctx* c, Spec* P) {
    tp::State& S = *c->tps;
    auto t0 = Clock::now();
    sync2(c);
    tp::StepState st;
    CK(cudaSetDevice(0));
    CK(cudaMemcpy(&st, S.G[0].st, sizeof st, cudaMemcpyDeviceToHost));
    const int pos = st.pos;  // position of the pending token (st.token)
    if (pos != c->pos) throw std::runtime_error("spec: device pos != host pos");
    for (int g = 0; g < 2; g++) {
        tp::Gpu& G = S.G[g];
        SG& B = P->G[g];
        CK(cudaSetDevice(g));
        for (int il = 0; il < 64; il++) {
            if (G.L[il].attn) continue;
            const int li = il - il / 4;
            CK(cudaMemcpyAsync(B.Sb + (size_t)li * SZS, G.L[il].S, SZS * 4, cudaMemcpyDeviceToDevice, G.s));
            k_ring_conv<<<20, 256, 0, G.s>>>(G.L[il].conv_ring, 4, B.ring + (size_t)li * CR * 5120, CR, &G.st->pos);
        }
        // spec fields of the step state
        tp::StepState s2 = st;
        CK(cudaMemcpy(&s2, G.st, sizeof s2, cudaMemcpyDeviceToHost));
        s2.vpos = pos - 1;
        s2.nacc = 0;
        s2.sidx = 0;
        s2.nemit = 0;
        s2.ngu = 0;
        s2.rbp = 0;
        s2.ng_lo = st.n_prompt == pos ? 0 : pos;  // tokens decoded outside spec mode are not in the history
        CK(cudaMemcpyAsync(G.prompt + pos, &st.token, 4, cudaMemcpyHostToDevice, G.s));
        CK(cudaMemcpyAsync(G.st, &s2, sizeof s2, cudaMemcpyHostToDevice, G.s));
        CK(cudaMemcpyAsync(B.yv, &st.token, 4, cudaMemcpyHostToDevice, G.s));
        CK(cudaMemcpyAsync(B.hf, G.xn, DM * 4, cudaMemcpyDeviceToDevice, G.s));  // h_final of position pos - 1
        CK(cudaMemsetAsync(B.hist, 0, 2 * (MMAX + 1) * 4, G.s));
        CK(cudaStreamSynchronize(G.s));
    }
    *(volatile int*)P->h_cnt = 0;
    // MTP prompt catch-up: rows (x_q, h_{q-1}) at positions q of the last prefill ubatch (h_{-1} = 0)
    const int r0 = S.pf_h_pos0, nr = S.pf_h_n;
    if (nr > 0 && r0 + nr <= pos) {
        for (int g = 0; g < 2; g++) {
            SG& B = P->G[g];
            tp::Gpu& G = S.G[g];
            CK(cudaSetDevice(g));
            const float* hrows = tp_prefill_hrows(c, g);
            if (!hrows) throw std::runtime_error("spec: no prefill rows");
            if (B.hpr_n < (size_t)nr + 1) {
                if (B.hpr) CK(cudaFree(B.hpr));
                B.hpr = dz<float>((size_t)(nr + 1) * DM);
                B.hpr_n = nr + 1;
            }
            // hpr row 0: h of position r0 - 1 (zero when r0 == 0, else unknown -> zero); rows 1..nr = output_norm(res)
            CK(cudaMemsetAsync(B.hpr, 0, DM * 4, G.s));
            SEnq q(c, P, g);
            for (int r = 0; r < nr; r += MMAX) {
                const int m = std::min(MMAX, nr - r);
                q.ar(-1, hrows + (size_t)r * DM, nullptr, false, G.output_norm, B.hpr + (size_t)(r + 1) * DM, B.xq,
                     B.xm, m);
            }
        }
        // the MTP passes need both GPUs (all-reduces): interleave the launches per chunk; the prompt ids are in
        // G.prompt (set_prompt), the h rows in hpr
        for (int r = 0; r < nr; r += MMAX) {
            const int m = std::min(MMAX, nr - r);
            const int q0 = r0 + r;
            for (int g = 0; g < 2; g++) {
                SG& B = P->G[g];
                tp::Gpu& G = S.G[g];
                CK(cudaSetDevice(g));
                CK(cudaMemcpyAsync(B.pscr + (r / MMAX & 1), &q0, 4, cudaMemcpyHostToDevice, G.s));
                CK(cudaStreamSynchronize(G.s));  // q0 is a stack value
                SEnq q(c, P, g);
                q.mtp_pass(m, G.prompt + q0, B.hpr + (size_t)r * DM, B.pscr + (r / MMAX & 1), 0, 0, B.xn, B.xq, B.xm);
            }
            // the passes of one chunk use AR idx 0..2 at the same step epoch: finish before the next chunk reuses them
            sync2(c);
            check_err2(c);
            // advance the epoch base so the next chunk's flags are new
            for (int g = 0; g < 2; g++) {
                tp::Gpu& G = S.G[g];
                CK(cudaSetDevice(g));
                tp::StepState s3;
                CK(cudaMemcpy(&s3, G.st, sizeof s3, cudaMemcpyDeviceToHost));
                s3.step += 1;
                CK(cudaMemcpy(G.st, &s3, sizeof s3, cudaMemcpyHostToDevice));
            }
        }
    }
    sync2(c);
    check_err2(c);
    P->active = true;
    P->prep_s += secs(t0);
}

// leave spec mode: state of the accepted tokens back into the decode engine's buffers
void spec_leave(t4q_ctx* c, Spec* P) {
    tp::State& S = *c->tps;
    sync2(c);
    if (P->rb_cfg)  // the last verify's partial acceptance has not been replayed yet (the next draft graph would)
        for (int g = 0; g < 2; g++) {
            SG& B = P->G[g];
            CK(cudaSetDevice(g));
            k_gdn_replay<<<dim3(96, 48), 256, 0, S.G[g].s>>>(B.Sb, (size_t)48 * SZS, S.G[g].st, P->k, B.rbk, B.rbd,
                                                              B.rbg, RBS);
            CK(cudaStreamSynchronize(S.G[g].s));
        }
    tp::StepState st;
    CK(cudaSetDevice(0));
    CK(cudaMemcpy(&st, S.G[0].st, sizeof st, cudaMemcpyDeviceToHost));
    for (int g = 0; g < 2; g++) {
        tp::Gpu& G = S.G[g];
        SG& B = P->G[g];
        CK(cudaSetDevice(g));
        for (int il = 0; il < 64; il++) {
            if (G.L[il].attn) continue;
            const int li = il - il / 4;
            CK(cudaMemcpyAsync(G.L[il].S, B.Sb + ((size_t)st.sidx * 48 + li) * SZS, SZS * 4,
                               cudaMemcpyDeviceToDevice, G.s));
            k_ring_conv<<<20, 256, 0, G.s>>>(B.ring + (size_t)li * CR * 5120, CR, G.L[il].conv_ring, 4, &G.st->pos);
        }
        // h_final of the last accepted position, for a later spec_enter (decode leaves it in G.xn as well)
        CK(cudaMemcpyAsync(G.xn, B.hf + (size_t)st.nacc * DM, DM * 4, cudaMemcpyDeviceToDevice, G.s));
        CK(cudaStreamSynchronize(G.s));
    }
    c->pos = st.pos;
    P->active = false;
}

}  // namespace spec
}  // namespace tp

using namespace tp::spec;
using tp::MMAX;
using tp::RING;

// ================================================================================================ API
void tp_spec_reset(t4q_ctx* c) {
    if (!c->tps || !c->tps->spec) return;
    Spec* P = (Spec*)c->tps->spec;
    P->active = false;
}

int tp_spec_force(t4q_ctx* c, const int32_t* ids, int n) {
    tp::State& S = *c->tps;
    if (c->pos + n > S.max_ctx) n = S.max_ctx - c->pos;
    for (int g = 0; g < 2; g++) {
        CK(cudaSetDevice(g));
        CK(cudaMemcpy(S.G[g].prompt + c->pos, ids, (size_t)n * 4, cudaMemcpyHostToDevice));
    }
    return 0;
}

int tp_spec_generate(t4q_ctx* c, int32_t* out, int max_new, const int32_t* stop, int n_stop) {
    tp::State& S = *c->tps;
    Spec* P = spec_get(c);
    const int k = S.spec_k;
    if (k < 1 || k > MMAX - 1) throw std::runtime_error("spec_k out of range");
    const bool dvh = S.spec_dv && S.G[0].mtp.dv > 0;
    const bool force = S.spec_force != 0;
    if (!P->G[0].ev || P->k != k || P->dv != (int)dvh || P->force != (int)force || P->ng != S.spec_ng ||
        P->ngmax != S.spec_ngmax || P->rb != S.spec_rb || P->sqt != S.spec_sqt || P->tc != S.spec_tc) {
        P->rb_cfg = S.spec_rb != 0;
        P->tc = S.spec_tc;
        g_spec_tc = S.spec_tc;
        tp::g_sq_threads = S.spec_sqt == 128 ? 128 : 256;
        sync2(c);
        capture(c, P, k, dvh, force);
    }
    auto t0 = Clock::now();
    spec_enter(c, P);
    tp::StepState st;
    CK(cudaSetDevice(0));
    CK(cudaMemcpy(&st, S.G[0].st, sizeof st, cudaMemcpyDeviceToHost));
    const int pos0 = st.pos;
    auto is_stop = [&](int t) {
        for (int j = 0; j < n_stop; j++)
            if (stop[j] == t) return true;
        return false;
    };
    int n = 0;
    out[n++] = st.token;
    bool done = is_stop(st.token) || max_new <= 1;
    const int AHEAD = S.spec_ahead > 0 ? S.spec_ahead : 4;
    long iters = 0, launched = 0;
    int got = 0;  // tokens read from the ring
    const bool timed = S.spec_prof != 0;
    const bool dbg = S.spec_dbg != 0;
    std::vector<float>* dl = nullptr;
    std::vector<float>* dn = nullptr;
    if (dbg) {
        dl = &c->dumps["spec_logits"];
        dn = &c->dumps["spec_n"];
        dl->clear();
        dn->clear();
    }
    if (S.spec_trace > 0 && !dbg) {
        // CUPTI timeline of spec_trace iterations, split into draft / verify segments by marker kernels
        std::string err;
        const int nt = std::min<long>(S.spec_trace, (S.max_ctx - pos0 - 2 * (k + 1)) / (k + 1));
        if (nt > 1 && trace::begin(err)) {
            for (int it = 0; it < nt; it++) {
                for (int g = 0; g < 2; g++) {
                    SG& B = P->G[g];
                    CK(cudaSetDevice(g));
                    CK(cudaGraphLaunch(B.ed, S.G[g].s));
                    k_mark_draft_end<<<1, 32, 0, S.G[g].s>>>();
                    CK(cudaGraphLaunch(B.ev, S.G[g].s));
                    k_mark_verify_end<<<1, 32, 0, S.G[g].s>>>();
                }
                launched++;
            }
            sync2(c);
            std::vector<trace::Rec> R = trace::end();
            iters = launched;
            std::string js = "{\"iters\": " + std::to_string(nt);
            for (int g = 0; g < 2; g++) {
                std::map<std::string, std::pair<double, int>> agg[2];
                double busy[2] = {0, 0}, span[2] = {0, 0};
                int seg = 0, it = 0;
                uint64_t s0 = 0, last_end = 0;
                for (auto& r : R) {
                    if (r.dev != g) continue;
                    const bool m1 = r.name.find("k_mark_draft_end") != std::string::npos;
                    const bool m2 = r.name.find("k_mark_verify_end") != std::string::npos;
                    if (m1 || m2) {
                        if (it > 0 && s0) span[seg] += (last_end - s0) * 1e-3;  // skip the first iteration
                        seg = m1 ? 1 : 0;
                        if (m2) it++;
                        s0 = 0;
                        continue;
                    }
                    if (!s0) s0 = r.start;
                    last_end = r.end;
                    if (it == 0) continue;
                    const double d = (r.end - r.start) * 1e-3;
                    agg[seg][r.name].first += d;
                    agg[seg][r.name].second++;
                    busy[seg] += d;
                }
                const int m = std::max(1, it - 1);
                for (int sg = 0; sg < 2; sg++) {
                    char b[200];
                    snprintf(b, sizeof b, ", \"gpu%d_%s\": {\"span_us\": %.1f, \"busy_us\": %.1f, \"by_name\": {", g,
                             sg ? "verify" : "draft", span[sg] / m, busy[sg] / m);
                    js += b;
                    bool first = true;
                    for (auto& kv : agg[sg]) {
                        snprintf(b, sizeof b, "%s\"%s\": [%.1f, %.2f]", first ? "" : ", ", kv.first.c_str(),
                                 kv.second.first / m, (double)kv.second.second / m);
                        js += b;
                        first = false;
                    }
                    js += "}}";
                }
            }
            P->trace_json = js + "}";
        } else if (!err.empty()) {
            P->trace_json = "{\"error\": \"" + err + "\"}";
        }
    }
    while (!done) {
        // worst case each iteration advances k + 1 positions; the verify touches pos .. pos + k
        const int ahead = (timed || dbg) ? 1 : AHEAD;
        int enq = 0;
        while (enq < ahead) {
            const long worst = pos0 + (long)(launched + 1) * (k + 1) + k + 1;
            if (worst >= S.max_ctx) break;
            if (n + (launched - iters) >= max_new) break;  // every iteration emits at least one token
            for (int g = 0; g < 2; g++) {
                SG& B = P->G[g];
                CK(cudaSetDevice(g));
                if (timed && g == 0) CK(cudaEventRecord(B.e0, S.G[g].s));
                CK(cudaGraphLaunch(B.ed, S.G[g].s));
                if (timed && g == 0) CK(cudaEventRecord(B.e1, S.G[g].s));
                CK(cudaGraphLaunch(B.ev, S.G[g].s));
                if (timed && g == 0) CK(cudaEventRecord(B.e2, S.G[g].s));
            }
            launched++;
            enq++;
        }
        if (enq == 0) {
            if (launched == iters) break;  // context full
        }
        CK(cudaSetDevice(0));
        CK(cudaStreamSynchronize(S.G[0].s));
        if (timed) {
            float a = 0, b = 0;
            CK(cudaEventElapsedTime(&a, P->G[0].e0, P->G[0].e1));
            CK(cudaEventElapsedTime(&b, P->G[0].e1, P->G[0].e2));
            P->ms_draft += a;
            P->ms_verify += b;
            P->iters_timed++;
        }
        iters = launched;
        const int cnt = *(volatile int*)P->h_cnt;
        if (dbg) {
            // logits of the accepted columns of the last verify (GPU0 rows 0..124159 and GPU1 rows of each column)
            sync2(c);
            tp::StepState s0;
            CK(cudaMemcpy(&s0, S.G[0].st, sizeof s0, cudaMemcpyDeviceToHost));
            const int na = s0.nacc + 1;
            const size_t base = dl->size();
            dl->resize(base + (size_t)na * hp::V);
            for (int g = 0; g < 2; g++) {
                CK(cudaSetDevice(g));
                for (int t = 0; t < na; t++)
                    CK(cudaMemcpy(dl->data() + base + (size_t)t * hp::V + (size_t)124160 * g,
                                  P->G[g].logits + (size_t)t * 124160, 124160 * 4, cudaMemcpyDeviceToHost));
            }
            dn->push_back((float)na);
        }
        for (; got < cnt && !done;) {
            const int t = S.h_ring[got % RING];
            got++;
            if (S.G[0].mtp.dv > 0) {
                P->dv_total++;
                if (t < S.G[0].mtp.dv) P->dv_hits++;
            }
            out[n++] = t;
            if (is_stop(t) || n >= max_new) done = true;
        }
        if (enq == 0 && got >= cnt) break;
    }
    sync2(c);
    check_err2(c);
    {
        // stats: accepted-length histogram (GPU0), steps
        std::vector<int> h(2 * (MMAX + 1));
        CK(cudaSetDevice(0));
        CK(cudaMemcpy(h.data(), P->G[0].hist, 2 * (MMAX + 1) * 4, cudaMemcpyDeviceToHost));
        long steps = 0, acc = 0;
        for (int i = 0; i <= MMAX; i++) {
            P->last_hist[i] = h[i] + h[MMAX + 1 + i];
            P->last_hist_ng[i] = h[MMAX + 1 + i];
            steps += P->last_hist[i];
            acc += (long)i * P->last_hist[i];
        }
        P->steps += steps;
        P->drafted += steps * k;
        P->emitted += acc + steps;
        P->last_k = k;
        P->last_dv = dvh;
        P->last_steps = steps;
        P->last_acc = acc;
    }
    spec_leave(c, P);
    P->last_s = secs(t0);
    P->gen_s += secs(t0);
    c->gen_s += secs(t0);
    c->gen_tokens += n;
    return n;
}

std::string tp_spec_stats(t4q_ctx* c) {
    if (!c->tps || !c->tps->spec) return "";
    Spec* P = (Spec*)c->tps->spec;
    char b[1536];
    std::string hist, hng;
    for (int i = 0; i <= MMAX; i++) hist += (i ? "," : "") + std::to_string(P->last_hist[i]);
    for (int i = 0; i <= MMAX; i++) hng += (i ? "," : "") + std::to_string(P->last_hist_ng[i]);
    snprintf(b, sizeof b,
             ", \"spec\": {\"k\": %d, \"dv\": %d, \"steps\": %ld, \"drafted\": %ld, \"emitted\": %ld, "
             "\"accept_rate\": %.4f, \"tokens_per_step\": %.4f, \"gen_s\": %.3f, \"prep_s\": %.3f, "
             "\"ms_draft\": %.3f, \"ms_verify\": %.3f, \"iters_timed\": %ld, \"last_hist\": [%s], "
             "\"dv_in_subset\": %.4f, \"dv_total\": %ld, \"last\": {\"k\": %d, \"dv\": %d, \"steps\": %ld, "
             "\"accepted\": %ld, \"accept_rate\": %.4f, \"tokens_per_step\": %.4f, \"secs\": %.3f, \"ng\": %d, "
             "\"hist_ng\": [%s]}}",
             P->k, P->dv, P->steps, P->drafted, P->emitted,
             P->drafted ? (double)(P->emitted - P->steps) / P->drafted : 0.0,
             P->steps ? (double)P->emitted / P->steps : 0.0, P->gen_s, P->prep_s,
             P->iters_timed ? P->ms_draft / P->iters_timed : 0.0, P->iters_timed ? P->ms_verify / P->iters_timed : 0.0,
             P->iters_timed, hist.c_str(), P->dv_total ? (double)P->dv_hits / P->dv_total : 0.0, P->dv_total, P->last_k,
             P->last_dv, P->last_steps, P->last_acc,
             P->last_steps ? (double)P->last_acc / ((double)P->last_steps * std::max(1, P->last_k)) : 0.0,
             P->last_steps ? (double)(P->last_acc + P->last_steps) / P->last_steps : 0.0, P->last_s, P->ng, hng.c_str());
    std::string r = b;
    if (!P->trace_json.empty()) r += ", \"spec_trace\": " + P->trace_json;
    return r;
}
