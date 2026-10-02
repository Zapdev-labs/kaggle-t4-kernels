// Gated DeltaNet decode kernels (M1, unfused). GGUF tiled v-head order: v head h uses k head h % 16.
#include "kernels.h"

namespace {
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ float bsum128(float v) {  // blockDim.x == 128
    __shared__ float sh[4];
    v = wsum(v);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    return sh[0] + sh[1] + sh[2] + sh[3];
}
}  // namespace

__global__ void k_gdn_conv(const float* qkv, float* cs, const float* w, float* y, int nch) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nch) return;
    const float s0 = cs[c * 3 + 0], s1 = cs[c * 3 + 1], s2 = cs[c * 3 + 2], x = qkv[c];
    const float* wc = w + (int64_t)c * 4;
    float sum = 0.f;
    sum += s0 * wc[0];
    sum += s1 * wc[1];
    sum += s2 * wc[2];
    sum += x * wc[3];
    y[c] = sum / (1.0f + expf(-sum));
    cs[c * 3 + 0] = s1;
    cs[c * 3 + 1] = s2;
    cs[c * 3 + 2] = x;
}

void launch_gdn_conv(const float* qkv, float* conv_state, const float* conv_w, float* y, int nch, cudaStream_t s) {
    k_gdn_conv<<<(nch + 255) / 256, 256, 0, s>>>(qkv, conv_state, conv_w, y, nch);
}

// blocks 0..15: q heads (y[0:2048]); 16..31: k heads (y[2048:4096])
__global__ void k_gdn_l2(const float* y, float* qn, float* kn, float eps) {
    const int b = blockIdx.x;
    const float* src = y + b * 128;
    float* dst = b < 16 ? qn + b * 128 : kn + (b - 16) * 128;
    const float x = src[threadIdx.x];
    const float ss = bsum128(x * x);
    const float scale = rsqrtf(ss / 128.0f + eps / 128.0f);
    dst[threadIdx.x] = (x * scale) * (1.0f / sqrtf(128.0f));
}

void launch_gdn_l2(const float* y, float* qn, float* kn, float eps, cudaStream_t s) {
    k_gdn_l2<<<32, 128, 0, s>>>(y, qn, kn, eps);
}

__global__ void k_gdn_gates(const float* b_raw, const float* a_raw, const float* ssm_a, const float* dt, float* beta,
                            float* g, int nv) {
    const int h = threadIdx.x;
    if (h >= nv) return;
    beta[h] = 1.0f / (1.0f + expf(-b_raw[h]));
    const float x = a_raw[h] + dt[h];
    const float sp = x > 20.0f ? x : logf(1.0f + expf(x));
    g[h] = sp * ssm_a[h];
}

void launch_gdn_gates(const float* b_raw, const float* a_raw, const float* ssm_a, const float* dt, float* beta,
                      float* g, int nv, cudaStream_t s) {
    k_gdn_gates<<<1, 64, 0, s>>>(b_raw, a_raw, ssm_a, dt, beta, g, nv);
}

// grid (48 heads, 32), block (32, 4): each warp owns one value column `col`; lanes cover k rows i = r*32 + lane.
__global__ void k_gdn_recur(float* S, const float* qn, const float* kn, const float* v, const float* beta,
                            const float* g, float* o, float scale) {
    const int h = blockIdx.x;
    const int lane = threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int kh = h % 16;
    float* st = S + ((int64_t)h * 128 + col) * 128;
    const float* q = qn + kh * 128;
    const float* k = kn + kh * 128;
    float s[4], kr[4], qr[4];
#pragma unroll
    for (int r = 0; r < 4; r++) {
        s[r] = st[r * 32 + lane];
        kr[r] = k[r * 32 + lane];
        qr[r] = q[r * 32 + lane];
    }
    const float gv = expf(g[h]);
    const float bv = beta[h];
    float kv = 0.f;
#pragma unroll
    for (int r = 0; r < 4; r++) kv += s[r] * kr[r];
    kv = wsum(kv);
    const float delta = (v[h * 128 + col] - gv * kv) * bv;
    float a = 0.f;
#pragma unroll
    for (int r = 0; r < 4; r++) {
        s[r] = gv * s[r] + kr[r] * delta;
        a += s[r] * qr[r];
        st[r * 32 + lane] = s[r];
    }
    a = wsum(a);
    if (lane == 0) o[h * 128 + col] = a * scale;
}

void launch_gdn_recur(float* S, const float* qn, const float* kn, const float* v, const float* beta, const float* g,
                      float* o, float scale, cudaStream_t s) {
    k_gdn_recur<<<dim3(48, 32), dim3(32, 4), 0, s>>>(S, qn, kn, v, beta, g, o, scale);
}

__global__ void k_gdn_gnorm(const float* o, const float* z, const float* w, float* out, float eps) {
    const int h = blockIdx.x, i = threadIdx.x;
    const float x = o[h * 128 + i];
    const float ss = bsum128(x * x);
    const float scale = rsqrtf(ss / 128.0f + eps);
    const float zz = z[h * 128 + i];
    out[h * 128 + i] = ((x * scale) * w[i]) * (zz / (1.0f + expf(-zz)));
}

void launch_gdn_gnorm(const float* o, const float* z, const float* w, float* out, int nv, float eps, cudaStream_t s) {
    k_gdn_gnorm<<<nv, 128, 0, s>>>(o, z, w, out, eps);
}
