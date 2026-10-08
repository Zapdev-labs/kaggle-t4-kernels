// cf-m1: the CYBER-FROST kernels. Exact decode math transcribed from the qwen4exp graph:
// hyper-connection mixers, sigmoid-gated GDN norm, 2-kv-head dense attention, the PLE
// n-gram conv, and the MoE combine. The rest reuses the M1 kernel set unchanged.
#include <cfloat>
#include <cmath>

#include <cuda_fp16.h>

#include "cf_model.h"

using namespace cf;

namespace {
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ float bsum256(float v, float* sh) {
    v = wsum(v);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) t += sh[i];
    return t;
}
__device__ float bmax256(float v, float* sh) {
    v = wmax(v);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = -FLT_MAX;
#pragma unroll
    for (int i = 0; i < 8; i++) t = fmaxf(t, sh[i]);
    return t;
}
}  // namespace

// grouped RMSNorm over the [D, HC] wide residual: one block per stream, rms over D.
__global__ void k_cf_hc_norm(const float* res, const float* w, float* xn, float eps) {
    __shared__ float red[8];
    const int s = blockIdx.x;
    const float* x = res + (size_t)s * D;
    float* y = xn + (size_t)s * D;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += 256) ss += x[i] * x[i];
    // reduce across the block
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) t += red[i];
        red[0] = rsqrtf(t / D + eps);
    }
    __syncthreads();
    const float scale = red[0];
    const float* ws = w + (size_t)s * D;
    for (int i = threadIdx.x; i < D; i += 256) y[i] = (x[i] * scale) * ws[i];
}

void launch_cf_hc_norm(const float* res, const float* w, float* xn, cudaStream_t s) {
    k_cf_hc_norm<<<HC, 256, 0, s>>>(res, w, xn, EPS);
}

// lo = silu(y * 1/HC)
__global__ void k_cf_hc_lo(const float* y, float* lo) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < LORA) {
        const float x = y[i] * (1.0f / HC);
        lo[i] = x / (1.0f + expf(-x));
    }
}

void launch_cf_hc_lo(const float* y, float* lo, cudaStream_t s) {
    k_cf_hc_lo<<<(LORA + 255) / 256, 256, 0, s>>>(y, lo);
}

// mixed[c] = (1/HC) * sum_s xn[s*D+c] * sigmoid(y_up[s*D+c])
__global__ void k_cf_hc_mixed(const float* xn, const float* y_up, float* mixed) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float acc = 0.f;
#pragma unroll
    for (int s = 0; s < HC; s++) {
        const float gg = y_up[(size_t)s * D + c];
        acc += xn[(size_t)s * D + c] * (1.0f / (1.0f + expf(-gg)));
    }
    mixed[c] = acc * (1.0f / HC);
}

void launch_cf_hc_mixed(const float* xn, const float* y_up, float* mixed, cudaStream_t s) {
    k_cf_hc_mixed<<<(D + 255) / 256, 256, 0, s>>>(xn, y_up, mixed);
}

// res[s*D+c] += 2*sigmoid(inj[s] * 1/HC) * block[c]
__global__ void k_cf_hc_combine(float* res, const float* block, const float* inj) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HCD) return;
    const int s = i / D;
    const float w = inj[s] * (1.0f / HC);
    res[i] += (2.0f / (1.0f + expf(-w))) * block[i % D];
}

void launch_cf_hc_combine(float* res, const float* block, const float* inj, cudaStream_t s) {
    k_cf_hc_combine<<<(HCD + 255) / 256, 256, 0, s>>>(res, block, inj);
}

__global__ void k_cf_res_init(float* res, const float* emb) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HCD) res[i] = emb[i % D];
}

void launch_cf_res_init(float* res, const float* emb, cudaStream_t s) {
    k_cf_res_init<<<(HCD + 255) / 256, 256, 0, s>>>(res, emb);
}

// the qwen4exp GDN output gate: sigmoid(z), not silu
__global__ void k_cf_gdn_gnorm(const float* o, const float* z, const float* w, float* out, float eps) {
    const int h = blockIdx.x, i = threadIdx.x;
    const float x = o[h * 128 + i];
    __shared__ float sh[4];
    float ss = x * x;
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) sh[0] = rsqrtf((sh[0] + sh[1] + sh[2] + sh[3]) / 128.0f + eps);
    __syncthreads();
    const float zz = z[h * 128 + i];
    out[h * 128 + i] = ((x * sh[0]) * w[i]) * (1.0f / (1.0f + expf(-zz)));
}

void launch_cf_gdn_gnorm(const float* o, const float* z, const float* w, float* out, float eps, cudaStream_t s) {
    k_cf_gdn_gnorm<<<HV, 128, 0, s>>>(o, z, w, out, eps);
}

// blocks 0..23: q heads ([24][512]: q 256 | gate 256), 24..25: the 2 kv heads
__global__ void k_cf_qk_norm_rope(const float* qfull, const float* k, const float* qw, const float* kw, float* qn,
                                  float* kn, const int* pos_dev, float eps, float theta_scale, int n_rot) {
    __shared__ float red[8];
    __shared__ float y[256];
    const int b = blockIdx.x, d = threadIdx.x;
    const int pos = *pos_dev;  // r19v: the step params (capture-constant; the same int)
    const bool isq = b < HQ;
    const float* src = isq ? qfull + b * 512 : k + (b - HQ) * 256;
    const float* w = isq ? qw : kw;
    float* dst = isq ? qn + b * 256 : kn + (b - HQ) * 256;
    const float x = src[d];
    const float ss = bsum256(x * x, red);
    const float scale = rsqrtf(ss / 256.0f + eps);
    y[d] = (x * scale) * w[d];
    __syncthreads();
    const int half_rot = n_rot / 2;
    if (d < half_rot) {
        const float theta = (float)pos * powf(theta_scale, (float)d);
        const float c = cosf(theta), si = sinf(theta);
        const float x0 = y[d], x1 = y[d + half_rot];
        dst[d] = x0 * c - x1 * si;
        dst[d + half_rot] = x0 * si + x1 * c;
    } else if (d >= n_rot) {
        dst[d] = y[d];
    }
}

void launch_cf_qk_norm_rope(const float* qfull, const float* k, const float* qw, const float* kw, float* qn, float* kn,
                            const int* pos_dev, float eps, float freq_base, int n_rot, cudaStream_t s) {
    const float theta_scale = powf(freq_base, -2.0f / n_rot);
    k_cf_qk_norm_rope<<<HQ + HKV, 256, 0, s>>>(qfull, k, qw, kw, qn, kn, pos_dev, eps, theta_scale, n_rot);
}

__global__ void k_cf_kv_store(const float* k, const float* v, uint16_t* kc, uint16_t* vc, const int* pos_dev,
                              int max_ctx) {
    const int j = blockIdx.x, d = threadIdx.x;
    const int pos = *pos_dev;  // r19v: the step params (capture-constant; the same int)
    __half* kh = (__half*)kc;
    __half* vh = (__half*)vc;
    kh[((int64_t)j * max_ctx + pos) * 256 + d] = __float2half_rn(k[j * 256 + d]);
    vh[((int64_t)j * max_ctx + pos) * 256 + d] = __float2half_rn(v[j * 256 + d]);
}

void launch_cf_kv_store(const float* k, const float* v, uint16_t* kc, uint16_t* vc, const int* pos_dev, int max_ctx,
                        cudaStream_t s) {
    k_cf_kv_store<<<HKV, 256, 0, s>>>(k, v, kc, vc, pos_dev, max_ctx);
}

__global__ void k_cf_attn_decode(const float* q, const uint16_t* kc, const uint16_t* vc, float* out, float* scores,
                                 const int* pos_dev, int max_ctx, float scale) {
    __shared__ float qs[256];
    __shared__ float red[8];
    const __half* kh = (const __half*)kc;
    const __half* vh = (const __half*)vc;
    const int h = blockIdx.x, j = h / 12;  // GQA 12
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n_kv = *pos_dev + 1;  // r19v: the step params (the caller passed pos+1)
    qs[tid] = q[h * 256 + tid];
    __syncthreads();
    const __half* K = kh + (int64_t)j * max_ctx * 256;
    const __half* V = vh + (int64_t)j * max_ctx * 256;
    float* sc = scores + (int64_t)h * max_ctx;
    for (int t = warp; t < n_kv; t += 8) {
        const __half* kr = K + (int64_t)t * 256;
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) a += qs[lane + 32 * i] * __half2float(kr[lane + 32 * i]);
        a = wsum(a);
        if (lane == 0) sc[t] = a * scale;
    }
    __syncthreads();
    float m = -FLT_MAX;
    for (int t = tid; t < n_kv; t += 256) m = fmaxf(m, sc[t]);
    m = bmax256(m, red);
    float sum = 0.f;
    for (int t = tid; t < n_kv; t += 256) {
        const float e = expf(sc[t] - m);
        sc[t] = e;
        sum += e;
    }
    sum = bsum256(sum, red);
    __syncthreads();
    float acc = 0.f;
    for (int t = 0; t < n_kv; t++) acc += sc[t] * __half2float(V[(int64_t)t * 256 + tid]);
    out[h * 256 + tid] = acc / sum;
}

void launch_cf_attn_decode(const float* q, const uint16_t* kc, const uint16_t* vc, float* out, float* scores,
                           const int* pos_dev, int max_ctx, float scale, cudaStream_t s) {
    k_cf_attn_decode<<<HQ, 256, 0, s>>>(q, kc, vc, out, scores, pos_dev, max_ctx, scale);
}

// PLE: per-stream s = sum_c key*query / sqrt(D), then gate = sigmoid(sgn(s)*sqrt(clamp(|s|,1e-6)))
__global__ void k_cf_ple_sg(const float* key, const float* query, float* s_out, float* gate) {
    const int s = threadIdx.x;
    if (s >= HC) return;
    const float* kk = key + (size_t)s * D;
    const float* qq = query + (size_t)s * D;
    float acc = 0.f;
    for (int c = 0; c < D; c++) acc += kk[c] * qq[c];
    const float sv = acc / sqrtf((float)D);
    s_out[s] = sv;
    const float a = fabsf(sv) > 1e-6f ? fabsf(sv) : 1e-6f;
    gate[s] = 1.0f / (1.0f + expf(-(sv < 0.f ? -1.f : 1.f) * sqrtf(a)));
}

void launch_cf_ple_sg(const float* key, const float* query, float* s_out, float* gate, cudaStream_t s) {
    k_cf_ple_sg<<<1, 32, 0, s>>>(key, query, s_out, gate);
}

// gated[i] = value[i % D] * gate[i / D]
__global__ void k_cf_ple_gated(const float* value, const float* gate, float* gated) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HCD) gated[i] = value[i % D] * gate[i / D];
}

void launch_cf_ple_gated(const float* value, const float* gate, float* gated, cudaStream_t s) {
    k_cf_ple_gated<<<(HCD + 255) / 256, 256, 0, s>>>(value, gate, gated);
}

// depthwise causal conv, dilated by the n-gram size: out[i] = silu(sum_k w[i*4+k] * x[i, t-(3-k)*3]),
// x = [hist 9 columns | the current gnorm column]. Then roll the ring: hist[j] = hist[j+1], hist[8] = gnorm.
__global__ void k_cf_ple_conv(const float* gnorm, float* hist, const float* w, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HCD) return;
    const float* wc = w + (size_t)i * 4;
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < PLE_CONV; k++) {
        const int col = PLE_HIST - (PLE_CONV - 1 - k) * PLE_NGRAM;  // k=0 -> oldest
        const float xv = col < PLE_HIST ? hist[(size_t)col * HCD + i] : gnorm[i];
        sum += wc[k] * xv;
    }
    out[i] = sum / (1.0f + expf(-sum));
    // roll the ring, this thread owns element i in every column
#pragma unroll
    for (int j = 0; j + 1 < PLE_HIST; j++) hist[(size_t)j * HCD + i] = hist[(size_t)(j + 1) * HCD + i];
    hist[(size_t)(PLE_HIST - 1) * HCD + i] = gnorm[i];
}

void launch_cf_ple_conv(const float* gnorm, float* hist, const float* w, float* out, cudaStream_t s) {
    k_cf_ple_conv<<<(HCD + 255) / 256, 256, 0, s>>>(gnorm, hist, w, out);
}

// out[c] = sum_e we[e]*ye[e*D+c] + sigmoid(*sh_gate_raw) * ysh[c]
__global__ void k_cf_moe_out(const float* ye, const float* we, const float* ysh, const float* sh_gate_raw,
                             float* out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float acc = 0.f;
#pragma unroll
    for (int e = 0; e < TOPK; e++) acc += we[e] * ye[(size_t)e * D + c];
    const float gg = *sh_gate_raw;
    out[c] = acc + (1.0f / (1.0f + expf(-gg))) * ysh[c];
}

void launch_cf_moe_out(const float* ye, const float* we, const float* ysh, const float* sh_gate_raw, float* out,
                       cudaStream_t s) {
    k_cf_moe_out<<<(D + 255) / 256, 256, 0, s>>>(ye, we, ysh, sh_gate_raw, out);
}

// cf-m6 r6b (the section 6 freeze, the per-side partials): the OWNER-side partial sum over
// the side's COMPACT slots (out[c] = sum_{k<n} we[k]*ye[k*D+c], n = the side's pick count -
// the unowned picks are simply absent from the side's table). The same loop expression as
// k_cf_moe_out's, only the bound is the side's n instead of TOPK, so the per-element
// arithmetic is the combine's own (the split changes only the ADD ORDER across sides - the
// ~1e-6 class, honestly noted in the r6b record; the requant's own error dwarfs it).
__global__ void k_cf_moe_partial(const float* ye, const float* we, int n, float* out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float acc = 0.f;
    for (int k = 0; k < n; k++) acc += we[k] * ye[(size_t)k * D + c];
    out[c] = acc;
}

void launch_cf_moe_partial(const float* ye, const float* we, int n, float* out, cudaStream_t s) {
    k_cf_moe_partial<<<(D + 255) / 256, 256, 0, s>>>(ye, we, n, out);
}

// cf-m6 r6c part 2 (the verify's split-moe): the per-row GATHER partial over the row's
// OWNED pick slots. The verify's per-row ye plane is the PICK-SLOT layout (the amortized
// dn writes the row's picks at their k slots, the unowned slots are stale garbage), so
// the side's partial sums the row's OWN k-list (we[ks[j]]*ye[ks[j]*D+c] - the row's we
// slice + the window-built k-list) instead of the compact [0,n) slots the greedy's
// partial rides. Together with the final it is the per-row moe_out's own arithmetic with
// the add order split across the sides (the same reassociation class as the greedy's).
__global__ void k_cf_moe_partial_k(const float* ye, const float* we, const int* ks, int nk, float* out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float acc = 0.f;
    for (int j = 0; j < nk; j++) {
        const int k = ks[j];
        acc += we[k] * ye[(size_t)k * D + c];
    }
    out[c] = acc;
}

void launch_cf_moe_partial_k(const float* ye, const float* we, const int* ks, int nk, float* out, cudaStream_t s) {
    k_cf_moe_partial_k<<<(D + 255) / 256, 256, 0, s>>>(ye, we, ks, nk, out);
}

// the r6b final combine: block = p0 + p1 + sigmoid(gate)*ysh (the shared expert ran on
// GPU0 with the core; only side 0's partial could have summed it, but the shared's own
// contribution lands here once - the moe_out's own tail form, p0/p1 in place of the Σ)
__global__ void k_cf_moe_final(const float* p0, const float* p1, const float* ysh, const float* sh_gate_raw,
                               float* out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    const float gg = *sh_gate_raw;
    out[c] = p0[c] + p1[c] + (1.0f / (1.0f + expf(-gg))) * ysh[c];
}

void launch_cf_moe_final(const float* p0, const float* p1, const float* ysh, const float* sh_gate_raw, float* out,
                         cudaStream_t s) {
    k_cf_moe_final<<<(D + 255) / 256, 256, 0, s>>>(p0, p1, ysh, sh_gate_raw, out);
}

// the 10-expert batched silu(g)*u over the stacked [gate n | up n] staging (stride 2n per expert).
// Same expression as k_silu_mul, so bit-identical per element; one launch instead of 10 x 3-block
// underfilled ones (the r18 verdict: the batched launch is mandatory at the measured 112->144 GB/s).
__global__ void k_cf_silu_mul_b(const float* gu, float* out, int n_per, int stride) {
    const int b = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_per) return;
    const float x = gu[(size_t)b * stride + i];
    out[(size_t)b * n_per + i] = (x / (1.0f + expf(-x))) * gu[(size_t)b * stride + n_per + i];
}

void launch_cf_silu_mul_b(const float* gu, float* out, int n_per, int batch, cudaStream_t s) {
    k_cf_silu_mul_b<<<dim3((n_per + 255) / 256, batch), 256, 0, s>>>(gu, out, n_per, 2 * n_per);
}

// cf-m4 (r19w): the MTP draft's eh_proj input gather - the per-stream [e_norm ; h_norm_s]
// concat: out[s][0:D] = e_norm (the shared half), out[s][D:2D] = h_norm[s*D:(s+1)*D]
// (the per-stream half). out flat [4][2D]; one launch, HC*2D threads.
__global__ void k_cf_eh_gather(const float* e_norm, const float* h_norm, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HC * 2 * D) return;
    const int s = i / (2 * D), j = i % (2 * D);
    out[i] = (j < D) ? e_norm[j] : h_norm[s * D + (j - D)];
}

void launch_cf_eh_gather(const float* e_norm, const float* h_norm, float* out, cudaStream_t s) {
    k_cf_eh_gather<<<(HC * 2 * D + 255) / 256, 256, 0, s>>>(e_norm, h_norm, out);
}
