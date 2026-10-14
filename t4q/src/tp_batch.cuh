// Batched decode (milestone B), included at the end of tp_prefill.cu (reuses its PfRun: fused add+norm+q8 producers,
// gemm9 GEMMs, gate|up silu epilogue, fp16 peer-copy all-reduce).
//
// B concurrent sequences live in "slots". Each slot owns, per GPU: a DeltaNet recurrent state per DeltaNet layer
// (24 local heads x 128 x 128, fp32 or fp16 storage with fp32 math), a conv ring (4 x 5120 raw inputs) and an fp16 KV
// cache of slot_ctx positions per attention layer. One step feeds one token per listed slot:
//   embed -> 64 x [add+norm+q8 -> GEMM (m = B, 64-token tile) -> batched gdn step / batched attention -> GEMM -> AR]
//   -> output norm -> lm_head (gemm9 on Q6_K, or the dp4a GEMV 8 columns per pass) -> per-row argmax on each GPU ->
//   host combines the two shards (max value, lowest index on ties: the decode engine's rule).
// Every kernel is per-row: a sequence's result does not depend on B or on the other rows (attention splits use a
// fixed number of positions per block, gemm9 accumulates each output in a fixed order). The numerics differ from the
// single-stream decode engine (dp4a GEMVs with exact Q4 x q8 per 32) the same way batched prefill does: per-row int8
// weights and GA64 activations in the GEMMs, fp16 all-reduce partials.
// Prompts are prefilled by the single-stream engine (batched prefill), then its state is copied into the slot.

namespace {

constexpr int BD_MAXB = 128;
constexpr int BD_NSMAX = 64;  // attention split blocks per row at most (slot_ctx <= 64 * bd_ch)

struct BdGpu {
    void* S = nullptr;        // [48 DeltaNet layers][n_slots][24 * 128 * 128] float or __half
    float* ring = nullptr;    // [48][n_slots][4][5120]
    __half* kv = nullptr;     // [16 attention layers][2 (k, v)][n_slots][2 kv heads][slot_ctx][256]
    int* meta = nullptr;      // [3][BD_MAXB]: slot, pos, input token of each row
    float* ws = nullptr;      // attention split partials [n_slots][2][nsmax][6][258]
    float* o = nullptr;       // [BD_MAXB][3072] DeltaNet outputs
    float* qa = nullptr;      // [BD_MAXB][3072] roped queries
    float* logits = nullptr;  // [BD_MAXB][124160] this GPU's logits shard
    float2* am = nullptr;     // [BD_MAXB] per-row argmax (value, index bits)
    float* kscr = nullptr;    // split-K fp32 slices [4][BD_MAXB][5120]
    int8_t* hq = nullptr;     // [BD_MAXB][5120] decode-format q8 of the normed rows (dp4a head)
    int2* hm = nullptr;
    float2* h_am = nullptr;   // pinned host copy of am
    int* h_meta = nullptr;    // pinned host meta
};

struct Bd {
    int n_slots = 0, slot_ctx = 0, sf16 = 0, nsmax = 0;
    size_t s_elems = (size_t)24 * 128 * 128;
    BdGpu G[2];
    std::vector<int> pos, tok;  // per slot: next position, next input token (-1: empty slot)
    int last_B = 0;
    std::vector<int> last_slots;
    double step_s = 0, prefill_s = 0, enqueue_s = 0;
    long steps = 0, rows = 0, prefills = 0;
    std::vector<std::pair<std::string, std::pair<double, int>>> prof, prof1;  // GPU0 / GPU1 per-op ms
    size_t bytes = 0;
};

inline int dn_index(int il) { return il - (il + 1) / 4; }  // DeltaNet layer ordinal (layers with (il + 1) % 4 != 0)
inline int at_index(int il) { return il / 4; }             // attention layer ordinal

__device__ __forceinline__ float bd_lds(const float* p) { return *p; }
__device__ __forceinline__ float bd_lds(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ void bd_sts(float* p, float v) { *p = v; }
__device__ __forceinline__ void bd_sts(__half* p, float v) { *p = __float2half_rn(v); }

// DeltaNet step for row b (slot meta[b] at position meta[MAXB + b]): the decode kernel's arithmetic (tp_kernels.cu
// d_gdn) with per-row state / ring / output pointers. y row: q | k | v | z (ldy). alpha/beta from the AB_KS K-slices of
// k_pf_ab, summed in k_pf_gdn's order. grid (96, B) x 256
template <class SF>
__global__ void __launch_bounds__(256) k_bd_gdn(const float* __restrict__ yall, int ldy, const float* __restrict__ yab,
                                                int ab_ss, const int* __restrict__ meta, float* ringb,
                                                const float* __restrict__ cw, const float* __restrict__ ssm_a,
                                                const float* __restrict__ ssm_dt, SF* Sb, float* oall) {
    __shared__ float sq[128], sk[128], sv[32], red[8];
    const int b = blockIdx.y;
    const int slot = meta[b], pos = meta[BD_MAXB + b];
    const float* y = yall + (size_t)b * ldy;
    float* ring = ringb + (size_t)slot * 4 * 5120;
    SF* S = Sb + (size_t)slot * 24 * 128 * 128;
    float* o = oall + (size_t)b * 3072;
    const int bx = blockIdx.x;
    const int vl = bx >> 2, sl = bx & 3, kl = vl & 7, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int p0 = (pos + 1) & 3, p1 = (pos + 2) & 3, p2 = (pos + 3) & 3, pw = pos & 3;  // pos-3, pos-2, pos-1, pos
    float s[4][4];
#pragma unroll
    for (int cc = 0; cc < 4; cc++) {
        const int col = sl * 32 + warp * 4 + cc;
        const SF* Sp = S + ((size_t)vl * 128 + col) * 128;
#pragma unroll
        for (int r = 0; r < 4; r++) s[cc][r] = bd_lds(Sp + r * 32 + lane);
    }
    float ya_ = 0.f, yb = 0.f;
#pragma unroll
    for (int kz = 0; kz < AB_KS; ++kz) {
        ya_ += __ldg(yab + (size_t)kz * ab_ss + (size_t)b * 48 + vl);
        yb += __ldg(yab + (size_t)kz * ab_ss + (size_t)b * 48 + 24 + vl);
    }
    const float dtv = ssm_dt[vl], av = ssm_a[vl];
    const int chv = 2048 + vl * 128 + sl * 32 + (tid & 31);
    float vx = 0.f, vr0 = 0.f, vr1 = 0.f, vr2 = 0.f;
    float4 vw = make_float4(0.f, 0.f, 0.f, 0.f);
    if (tid < 32) {
        vx = __ldg(y + chv);
        vr0 = ring[p0 * 5120 + chv];
        vr1 = ring[p1 * 5120 + chv];
        vr2 = ring[p2 * 5120 + chv];
        vw = __ldg((const float4*)(cw + (size_t)chv * 4));
    }
    {
        const int ch = tid < 128 ? kl * 128 + tid : 1024 + kl * 128 + (tid - 128);
        const float x = __ldg(y + ch);
        const float4 wc = __ldg((const float4*)(cw + (size_t)ch * 4));
        float sum = 0.f;
        sum += ring[p0 * 5120 + ch] * wc.x;
        sum += ring[p1 * 5120 + ch] * wc.y;
        sum += ring[p2 * 5120 + ch] * wc.z;
        sum += x * wc.w;
        const float a = sum / (1.0f + expf(-sum));
        float ssq = warp_sum(a * a);
        if (lane == 0) red[warp] = ssq;
        __syncthreads();
        if (vl < 8 && sl == 0) ring[pw * 5120 + ch] = x;  // slot pw is never read in this step
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
        ring[pw * 5120 + chv] = vx;
    }
    __syncthreads();
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
        SF* Sp = S + ((size_t)vl * 128 + col) * 128;
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
            bd_sts(Sp + r * 32 + lane, sn);
        }
        a = warp_sum(a);
        if (lane == 0) o[vl * 128 + col] = a * (1.0f / sqrtf(128.0f));
    }
}

// q/k RMSNorm + partial NeoX RoPE, KV append (fp16) for row b at its slot / position (decode d_attn_prep arithmetic).
// blocks x: 0..11 local q heads, 12..13 local kv heads; grid (14, B) x 256. kc / vc: this layer's [n_slots][2][ctx][256]
__global__ void __launch_bounds__(256) k_bd_attn_prep(const float* __restrict__ yall, int ldy, const float* __restrict__ qw,
                                                      const float* __restrict__ kw, float* qall, __half* kc, __half* vc,
                                                      int slot_ctx, const int* __restrict__ meta, float theta_scale) {
    __shared__ float red[8];
    __shared__ float yv[256];
    const int bb = blockIdx.y, b = blockIdx.x, d = threadIdx.x;
    const int slot = meta[bb], pos = meta[BD_MAXB + bb];
    const float* ya = yall + (size_t)bb * ldy;
    const bool isq = b < 12;
    const float* src = isq ? ya + b * 512 : ya + 6144 + (b - 12) * 256;
    const float* w = isq ? qw : kw;
    const float x = __ldg(src + d);
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
        float* dst = qall + (size_t)bb * 3072 + b * 256;
        if (d < 32) { dst[d] = out0; dst[d + 32] = out1; }
        else if (d >= 64) dst[d] = yv[d];
    } else {
        const int kvh = b - 12;
        const size_t row = (((size_t)slot * 2 + kvh) * slot_ctx + pos) * 256;
        __half* kd = kc + row;
        if (d < 32) { kd[d] = __float2half_rn(out0); kd[d + 32] = __float2half_rn(out1); }
        else if (d >= 64) kd[d] = __float2half_rn(yv[d]);
        vc[row + d] = __float2half_rn(__ldg(ya + 6656 + kvh * 256 + d));
    }
}

// split-K decode attention (decode d_attn_split per row): block (kv head j, split s, row b) covers positions
// [s*ch, min(n_kv, (s+1)*ch)); warps 0-3 serve q heads 0-2 of the kv head, 4-7 heads 3-5.
// ws: [b][j][nsmax][6][258] {m, l, acc[256]}. grid (2, nsplits, B) x 256
__global__ void __launch_bounds__(256, 2) k_bd_attn_split(const float* __restrict__ qall, const __half* __restrict__ kc,
                                                          const __half* __restrict__ vc, float* ws, int slot_ctx,
                                                          const int* __restrict__ meta, int ch, int nsmax) {
    __shared__ float sm_m[8][3], sm_l[8][3];
    __shared__ float sacc[8][256];
    const int j = blockIdx.x, sidx = blockIdx.y, bb = blockIdx.z;
    const int slot = meta[bb], pos = meta[BD_MAXB + bb];
    const int n_kv = pos + 1;
    const int t0 = sidx * ch, t1 = min(n_kv, t0 + ch);
    if (t0 >= n_kv) return;  // the combine reads only splits with positions
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, hg = warp >> 2, wq = warp & 3;
    const float* qa = qall + (size_t)bb * 3072;
    float q[3][8];
#pragma unroll
    for (int hh = 0; hh < 3; hh++) {
        const float4* qp = (const float4*)(qa + (j * 6 + 3 * hg + hh) * 256 + lane * 8);
        const float4 a = __ldg(qp), b = __ldg(qp + 1);
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
    const __half* K = kc + ((size_t)slot * 2 + j) * slot_ctx * 256;
    const __half* Vv = vc + ((size_t)slot * 2 + j) * slot_ctx * 256;
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
            float* out = ws + ((((size_t)bb * 2 + j) * nsmax + sidx) * 6 + 3 * g2 + hh) * 258;
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

__device__ __forceinline__ float bd_warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// merge the splits of row b, sigmoid output gate -> g32 [b][3072] fp32 (quantized for attn_output by quant8).
// grid (12 local q heads, B) x 256
__global__ void __launch_bounds__(256) k_bd_attn_combine(const float* __restrict__ ws, const float* __restrict__ yall,
                                                         int ldy, const int* __restrict__ meta, int ch, int nsmax,
                                                         float* g32) {
    const int hl = blockIdx.x, bb = blockIdx.y, d = threadIdx.x, j = hl / 6, h6 = hl % 6;
    const int n_kv = meta[BD_MAXB + bb] + 1;
    const int nsp = (n_kv + ch - 1) / ch;
    const float* wb = ws + (((size_t)bb * 2 + j) * nsmax) * 6 * 258;
    __shared__ float s_w[BD_NSMAX];
    __shared__ float s_den;
    if (d < 32) {
        float mv[2] = {-FLT_MAX, -FLT_MAX}, lv[2] = {0.f, 0.f};
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int s = d + 32 * u;
            if (s < nsp) {
                const float* p = wb + ((size_t)s * 6 + h6) * 258;
                mv[u] = p[0];
                lv[u] = p[1];
            }
        }
        const float M = bd_warp_max(fmaxf(mv[0], mv[1]));
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
    float num = 0.f;
#pragma unroll 8
    for (int s = 0; s < nsp; s++) num += s_w[s] * wb[((size_t)s * 6 + h6) * 258 + 2 + d];
    const float att = num / s_den;
    const float g = __ldg(yall + (size_t)bb * ldy + hl * 512 + 256 + d);
    g32[(size_t)bb * 3072 + hl * 256 + d] = att * (1.0f / (1.0f + expf(-g)));
}

// per-row argmax of this GPU's logits shard (max value, lowest index on ties). grid B x 1024
__global__ void __launch_bounds__(1024) k_bd_argmax(const float* __restrict__ x, int ld, int n, int row0, float2* out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    const float* r = x + (size_t)blockIdx.x * ld;
    float bv = -FLT_MAX;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += 1024) {
        const float v = r[i];
        if (v > bv) { bv = v; bi = i; }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if ((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = bv; si[threadIdx.x >> 5] = bi; }
    __syncthreads();
    if (threadIdx.x >= 32) return;
    bv = sv[threadIdx.x]; bi = si[threadIdx.x];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (threadIdx.x == 0) out[blockIdx.x] = make_float2(bv, __int_as_float(bi + row0));
}

__global__ void k_bd_f2h(const float* __restrict__ a, __half* b, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        b[i] = __float2half_rn(a[i]);
}

// ------------------------------------------------------------------------------------------------ host
Bd* bd_of(t4q_ctx* c) {
    Bd* b = (Bd*)c->tps->bd;
    if (!b) throw std::runtime_error("batch not initialized (t4q_batch_init)");
    return b;
}

size_t bd_s_bytes(const Bd& b) { return b.s_elems * (b.sf16 ? 2 : 4); }
char* bd_S(const Bd& b, int g, int il, int slot) {
    return (char*)b.G[g].S + ((size_t)dn_index(il) * b.n_slots + slot) * bd_s_bytes(b);
}
float* bd_ring(const Bd& b, int g, int il, int slot) {
    return b.G[g].ring + ((size_t)dn_index(il) * b.n_slots + slot) * 4 * 5120;
}
// kv layer base: kc (kv = 0) or vc (kv = 1) of attention layer il: [n_slots][2][slot_ctx][256]
__half* bd_kv(const Bd& b, int g, int il, int kv) {
    return b.G[g].kv + ((size_t)at_index(il) * 2 + kv) * b.n_slots * 2 * b.slot_ctx * 256;
}

void bd_free(Bd* b) {
    if (!b) return;
    for (int g = 0; g < 2; g++) {
        CK(cudaSetDevice(g));
        CK(cudaDeviceSynchronize());
        BdGpu& G = b->G[g];
        for (void* p : {G.S, (void*)G.ring, (void*)G.kv, (void*)G.meta, (void*)G.ws, (void*)G.o, (void*)G.qa,
                        (void*)G.logits, (void*)G.am, (void*)G.hq, (void*)G.hm, (void*)G.kscr})
            if (p) cudaFree(p);
        if (G.h_am) cudaFreeHost(G.h_am);
        if (G.h_meta) cudaFreeHost(G.h_meta);
    }
    delete b;
}

struct BdRun : PfRun {
    Bd* bd = nullptr;
    int nsplit = 1;  // attention split blocks launched per row (max over the batch)
    float theta_scale = 0.f;

    void mixer_bd(int g, int il) {
        tp::State& S = *c->tps;
        tp::Gpu& G = S.G[g];
        tp::Layer& L = G.L[il];
        PfGpu& B = P->G[g];
        BdGpu& Q = bd->G[g];
        const int T = sT[0];
        if (!L.attn) {
            add_norm(g, 0, L.attn_norm, il > 0, true, true);
            gemm(g, 0, L.qkvz, B.y, 8192);
            k_pf_ab<<<dim3((T + 127) / 128, AB_KS), 128, 0, G.s>>>(B.xn, L.ab, T, B.yab, P->cap * 48);
            mark(g, "ab");
            float* ring = bd_ring(*bd, g, il, 0);
            if (bd->sf16)
                k_bd_gdn<__half><<<dim3(96, T), 256, 0, G.s>>>(B.y, 8192, B.yab, P->cap * 48, Q.meta, ring, L.conv_w,
                                                               L.ssm_a, L.ssm_dt, (__half*)bd_S(*bd, g, il, 0), Q.o);
            else
                k_bd_gdn<float><<<dim3(96, T), 256, 0, G.s>>>(B.y, 8192, B.yab, P->cap * 48, Q.meta, ring, L.conv_w,
                                                              L.ssm_a, L.ssm_dt, (float*)bd_S(*bd, g, il, 0), Q.o);
            ck_launch("bd_gdn");
            mark(g, "gdn");
            k_pf_gnorm_q8<<<dim3(T, 24), 128, 0, G.s>>>(Q.o, B.y, 8192, L.ssm_norm, q8out(g, 0, 3072));
            ck_launch("bd_gnorm");
            mark(g, "gnorm_q8");
            gemm(g, 0, L.ssm_out, B.part, D);
        } else {
            add_norm(g, 0, L.attn_norm, il > 0, true, false);
            gemm(g, 0, L.qkv_a, B.y, 7168);
            __half* kc = bd_kv(*bd, g, il, 0);
            __half* vc = bd_kv(*bd, g, il, 1);
            k_bd_attn_prep<<<dim3(14, T), 256, 0, G.s>>>(B.y, 7168, L.q_norm, L.k_norm, Q.qa, kc, vc, bd->slot_ctx,
                                                         Q.meta, theta_scale);
            k_bd_attn_split<<<dim3(2, nsplit, T), 256, 0, G.s>>>(Q.qa, kc, vc, Q.ws, bd->slot_ctx, Q.meta, S.bd_ch,
                                                                bd->nsmax);
            k_bd_attn_combine<<<dim3(12, T), 256, 0, G.s>>>(Q.ws, B.y, 7168, Q.meta, S.bd_ch, bd->nsmax, B.g32);
            ck_launch("bd_attention");
            mark(g, "attn");
            qg(g, 0, L.wo, B.g32, 3072, B.part, D);
        }
    }

    void head(int g) {
        tp::State& S = *c->tps;
        tp::Gpu& G = S.G[g];
        PfGpu& B = P->G[g];
        BdGpu& Q = bd->G[g];
        const int T = sT[0], Tp = tpad(T);
        tp::FW& W = G.lm;
        if (W.L.fmt != gemv::FAST_K6) throw std::runtime_error("batched head expects a Q6_K lm_head");
        if (S.bd_head == 1) {
            if (!W.invs) {
                CK(cudaMalloc(&W.invs, (size_t)W.L.N * 4));
                gemm8::Args a = gemm8::make_args(W.L, W.base, W.invs, nullptr, nullptr, nullptr, 0, 0, 0);
                CK(gemm8::row_invs(W.L.fmt, W.L.rpl, a, W.invs, G.s));
            }
            gemm8::quant8(B.xn, D, T, Tp, D, B.xq, B.dx, G.s, 64);
            gemm8::Args a8 = gemm8::make_args(W.L, W.base, W.invs, B.xq, B.dx, Q.logits, 124160, T, Tp);
            a8.pfk = pfk;
            a8.lbm = lbm;
            const cudaError_t e = gemm8::launch9(W.L.fmt, W.L.rpl, bn_div(Tp, 128), 64, a8, G.s);
            if (e != cudaSuccess) throw std::runtime_error(std::string("bd head gemm: ") + cudaGetErrorString(e));
        } else {
            const int nq = T * (D / 32) * 32;
            gemv::quantize_q8_kernel<<<(nq + 255) / 256, 256, 0, G.s>>>(B.xn, D, T, Q.hq, Q.hm);
            for (int c0 = 0; c0 < T; c0 += 8) {
                const int M = std::min(8, T - c0);
                gemv::GemvArgs a = gemv::make_args(W.L, W.base, Q.hq + (size_t)c0 * D, Q.hm + (size_t)c0 * (D / 32),
                                                   Q.logits + (size_t)c0 * 124160, 124160);
                cudaError_t e = cudaErrorInvalidValue;
                switch (M) {
#define T4Q_BDH(m)                                                                                                  \
    case m: {                                                                                                       \
        static int occ[2] = {0, 0};                                                                                 \
        if (!occ[g]) occ[g] = std::max(1, gemv::gemv_fast_occupancy<gemv::FAST_K6, 2, m, 10, 1, false>(256));      \
        e = gemv::gemv_fast_launch<gemv::FAST_K6, 2, m, 10, 1, false>(a, occ[g] * 40, 256, G.s);                   \
        break;                                                                                                      \
    }
                    T4Q_BDH(1) T4Q_BDH(2) T4Q_BDH(3) T4Q_BDH(4) T4Q_BDH(5) T4Q_BDH(6) T4Q_BDH(7) T4Q_BDH(8)
#undef T4Q_BDH
                }
                if (e != cudaSuccess) throw std::runtime_error(std::string("bd head gemv: ") + cudaGetErrorString(e));
            }
        }
        mark(g, "lm_head");
        k_bd_argmax<<<T, 1024, 0, G.s>>>(Q.logits, 124160, 124160, 124160 * g, Q.am);
        ck_launch("bd_argmax");
        CK(cudaMemcpyAsync(Q.h_am, Q.am, (size_t)T * sizeof(float2), cudaMemcpyDeviceToHost, G.s));
        mark(g, "argmax");
    }

    void run_bd() {
        tp::State& S = *c->tps;
        nsub = 1;
        st0[0] = 0; sT[0] = T;
        theta_scale = powf(hp::ROPE_BASE, -2.0f / hp::NROT);
        for (int g = 0; g < 2; g++) {
            CK(cudaSetDevice(g));
            tp::Gpu& G = S.G[g];
            BdGpu& Q = bd->G[g];
            CK(cudaMemcpyAsync(Q.meta, Q.h_meta, 3 * BD_MAXB * sizeof(int), cudaMemcpyHostToDevice, G.s));
            mark(g, "start");
            k_pf_embed<<<dim3(T, 20), 256, 0, G.s>>>(G.embd, Q.meta + 2 * BD_MAXB, P->G[g].h);
            ck_launch("bd_embed");
            mark(g, "embed");
        }
        for (int il = 0; il < 64; il++) {
            for (int g = 0; g < 2; g++) { CK(cudaSetDevice(g)); mixer_bd(g, il); }
            send(0);
            ar++;
            for (int g = 0; g < 2; g++) { CK(cudaSetDevice(g)); ffn(g, il, 0); }
            send(0);
            ar++;
        }
        for (int g = 0; g < 2; g++) {
            CK(cudaSetDevice(g));
            add_norm(g, 0, S.G[g].output_norm, true);
            head(g);
        }
    }
};

void bd_copy_state(t4q_ctx* c, Bd* b, int g, int il, const void* srcS, const float* srcRing, int dst_slot,
                   bool src_is_slot) {
    tp::Gpu& G = c->tps->G[g];
    char* dS = bd_S(*b, g, il, dst_slot);
    if (src_is_slot || !b->sf16) {
        CK(cudaMemcpyAsync(dS, srcS, src_is_slot ? bd_s_bytes(*b) : b->s_elems * 4, cudaMemcpyDeviceToDevice, G.s));
    } else {
        k_bd_f2h<<<240, 256, 0, G.s>>>((const float*)srcS, (__half*)dS, b->s_elems);
        ck_launch("bd_f2h");
    }
    CK(cudaMemcpyAsync(bd_ring(*b, g, il, dst_slot), srcRing, 4 * 5120 * 4, cudaMemcpyDeviceToDevice, G.s));
}

}  // namespace

// ================================================================================================ batch API
int tp_batch_init(t4q_ctx* c, int n_slots, int slot_ctx, int sf16) {
    tp::State& S = *c->tps;
    if (n_slots < 1 || n_slots > BD_MAXB) throw std::runtime_error("n_slots must be 1..128");
    if (slot_ctx < 16 || slot_ctx > 64 * 256) throw std::runtime_error("slot_ctx out of range");
    bd_free((Bd*)S.bd);
    S.bd = nullptr;
    pf_get(c);  // prefill / GEMM activation buffers first (they are shared with the batch step)
    Bd* b = new Bd();
    b->n_slots = n_slots;
    b->slot_ctx = slot_ctx;
    b->sf16 = sf16 ? 1 : 0;
    b->nsmax = std::min(BD_NSMAX, (slot_ctx + 63) / 64);
    b->pos.assign(n_slots, 0);
    b->tok.assign(n_slots, -1);
    const size_t nS = (size_t)48 * n_slots * bd_s_bytes(*b), nR = (size_t)48 * n_slots * 4 * 5120 * 4,
                 nKV = (size_t)16 * 2 * n_slots * 2 * slot_ctx * 256 * 2,
                 nWS = (size_t)n_slots * 2 * b->nsmax * 6 * 258 * 4;
    const size_t need = nS + nR + nKV + nWS + (size_t)BD_MAXB * (3072 * 8 + 124160 * 4 + 5120 + 5120 / 32 * 8 + 64);
    for (int g = 0; g < 2; g++) {
        CK(cudaSetDevice(g));
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const size_t margin = (size_t)300 << 20;
        if (need + margin > fr) {
            delete b;
            char m[200];
            snprintf(m, sizeof m, "batch buffers need %.0f MiB per GPU, %.0f MiB free (gpu %d)", need / 1048576.0,
                     fr / 1048576.0, g);
            throw std::runtime_error(m);
        }
    }
    b->bytes = need;
    try {
    for (int g = 0; g < 2; g++) {
        CK(cudaSetDevice(g));
        BdGpu& G = b->G[g];
        CK(cudaMalloc(&G.S, nS));
        CK(cudaMemset(G.S, 0, nS));
        G.ring = dalloc<float>(nR / 4);
        CK(cudaMalloc(&G.kv, nKV));
        G.meta = dalloc<int>(3 * BD_MAXB);
        G.ws = dalloc<float>(nWS / 4);
        G.o = dalloc<float>((size_t)BD_MAXB * 3072);
        G.qa = dalloc<float>((size_t)BD_MAXB * 3072);
        G.logits = dalloc<float>((size_t)BD_MAXB * 124160);
        G.am = dalloc<float2>(BD_MAXB);
        G.kscr = dalloc<float>((size_t)4 * BD_MAXB * 5120);
        G.hq = dalloc<int8_t>((size_t)BD_MAXB * 5120);
        G.hm = dalloc<int2>((size_t)BD_MAXB * 160);
        CK(cudaMallocHost(&G.h_am, BD_MAXB * sizeof(float2)));
        CK(cudaMallocHost(&G.h_meta, 3 * BD_MAXB * sizeof(int)));
        CK(cudaDeviceSynchronize());
    }
    } catch (...) {
        bd_free(b);  // nullptr-init fields make partial-init teardown safe
        throw;
    }
    S.bd = b;
    return 0;
}

void tp_batch_free(t4q_ctx* c) {
    if (!c->tps) return;
    bd_free((Bd*)c->tps->bd);
    c->tps->bd = nullptr;
}

int tp_batch_prefill(t4q_ctx* c, int slot, const int32_t* ids, int n) {
    tp::State& S = *c->tps;
    Bd* b = bd_of(c);
    if (slot < 0 || slot >= b->n_slots) throw std::runtime_error("bad slot");
    if (n < 1 || n + 1 > b->slot_ctx) throw std::runtime_error("prompt does not fit the slot context");
    auto t0 = Clock::now();
    tp_reset(c);
    tp_prefill(c, ids, n);  // single-stream batched prefill: KV, conv ring, DeltaNet state, first token
    int first = -1;
    for (int g = 0; g < 2; g++) {
        CK(cudaSetDevice(g));
        tp::Gpu& G = S.G[g];
        CK(cudaStreamSynchronize(G.s));
        if (g == 0) {
            tp::StepState st;
            CK(cudaMemcpy(&st, G.st, sizeof st, cudaMemcpyDeviceToHost));
            if (st.err) throw std::runtime_error("prefill error " + std::to_string(st.err));
            first = st.last_tok;
        }
        for (int il = 0; il < 64; il++) {
            tp::Layer& L = G.L[il];
            if (!L.attn) {
                bd_copy_state(c, b, g, il, L.S, L.conv_ring, slot, false);
            } else {
                for (int kv = 0; kv < 2; kv++)
                    for (int j = 0; j < 2; j++)
                        CK(cudaMemcpyAsync(bd_kv(*b, g, il, kv) + ((size_t)slot * 2 + j) * b->slot_ctx * 256,
                                           (kv ? L.vc : L.kc) + (size_t)j * S.max_ctx * 256, (size_t)n * 512,
                                           cudaMemcpyDeviceToDevice, G.s));
            }
        }
    }
    for (int g = 0; g < 2; g++) { CK(cudaSetDevice(g)); CK(cudaStreamSynchronize(S.G[g].s)); }
    b->pos[slot] = n;
    b->tok[slot] = first;
    b->prefill_s += secs(t0);
    b->prefills++;
    return first;
}

int tp_batch_clone(t4q_ctx* c, int src, int dst) {
    tp::State& S = *c->tps;
    Bd* b = bd_of(c);
    if (src < 0 || src >= b->n_slots || dst < 0 || dst >= b->n_slots || b->tok[src] < 0) throw std::runtime_error("bad clone slots");
    if (src == dst) return 0;
    for (int g = 0; g < 2; g++) {
        CK(cudaSetDevice(g));
        tp::Gpu& G = S.G[g];
        for (int il = 0; il < 64; il++) {
            if (!G.L[il].attn) {
                bd_copy_state(c, b, g, il, bd_S(*b, g, il, src), bd_ring(*b, g, il, src), dst, true);
            } else {
                for (int kv = 0; kv < 2; kv++)
                    CK(cudaMemcpyAsync(bd_kv(*b, g, il, kv) + (size_t)dst * 2 * b->slot_ctx * 256,
                                       bd_kv(*b, g, il, kv) + (size_t)src * 2 * b->slot_ctx * 256,
                                       (size_t)2 * b->slot_ctx * 512, cudaMemcpyDeviceToDevice, G.s));
            }
        }
    }
    for (int g = 0; g < 2; g++) { CK(cudaSetDevice(g)); CK(cudaStreamSynchronize(S.G[g].s)); }
    b->pos[dst] = b->pos[src];
    b->tok[dst] = b->tok[src];
    return 0;
}

int tp_batch_set_token(t4q_ctx* c, int slot, int token) {
    Bd* b = bd_of(c);
    if (slot < 0 || slot >= b->n_slots) throw std::runtime_error("bad slot");
    if (token < -1 || token >= hp::V) throw std::runtime_error("bad token");
    b->tok[slot] = token;
    return 0;
}

int tp_batch_pos(t4q_ctx* c, int slot) {
    Bd* b = bd_of(c);
    if (slot < 0 || slot >= b->n_slots) throw std::runtime_error("bad slot");
    return b->pos[slot];
}

int tp_batch_step(t4q_ctx* c, int n, const int32_t* slots, int32_t* out) {
    tp::State& S = *c->tps;
    Bd* b = bd_of(c);
    if (n < 1 || n > b->n_slots) throw std::runtime_error("bad batch size");
    if (S.bd_ch < 16 || (b->slot_ctx + S.bd_ch - 1) / S.bd_ch > b->nsmax) throw std::runtime_error("bd_ch too small for slot_ctx");
    std::vector<char> seen(b->n_slots, 0);
    int maxpos = 0;
    for (int i = 0; i < n; i++) {
        const int s = slots[i];
        if (s < 0 || s >= b->n_slots || seen[s]) throw std::runtime_error("bad or duplicate slot in batch");
        seen[s] = 1;
        if (b->tok[s] < 0) throw std::runtime_error("slot " + std::to_string(s) + " has no pending token");
        if (b->pos[s] >= b->slot_ctx) throw std::runtime_error("slot " + std::to_string(s) + " context full");
        maxpos = std::max(maxpos, b->pos[s]);
    }
    auto t0 = Clock::now();
    Pf* P = pf_get(c);
    for (int g = 0; g < 2; g++) {
        int* hm = b->G[g].h_meta;
        for (int i = 0; i < n; i++) {
            hm[i] = slots[i];
            hm[BD_MAXB + i] = b->pos[slots[i]];
            hm[2 * BD_MAXB + i] = b->tok[slots[i]];
        }
    }
    std::vector<cudaEvent_t> ev, ev1;
    std::vector<const char*> evn, evn1;
    BdRun R;
    R.c = c;
    R.P = P;
    R.bd = b;
    R.T = n;
    R.tp_force = n <= 32 ? 32 : n <= 64 ? 64 : (n + 127) / 128 * 128;
    R.g8 = true;
    R.ga = 64;
    R.ar16 = S.pf_ar16 != 0;
    R.p2p_part = S.bd_p2p && S.p2p && R.ar16;
    R.pfk = S.bd_pfk;
    R.lbm = S.bd_lbm;
    R.gemmr = S.bd_gemmr != 0;
    R.ksplit = S.bd_ksplit;
    R.kscr[0] = b->G[0].kscr;
    R.kscr[1] = b->G[1].kscr;
    R.nsplit = (maxpos + 1 + S.bd_ch - 1) / S.bd_ch;
    if (S.bd_prof) { R.ev = &ev; R.evn = &evn; R.ev1 = &ev1; R.evn1 = &evn1; }
    R.run_bd();
    b->enqueue_s += secs(t0);
    for (int g = 0; g < 2; g++) { CK(cudaSetDevice(g)); CK(cudaStreamSynchronize(S.G[g].s)); }
    for (int i = 0; i < n; i++) {
        const float2 a = b->G[0].h_am[i], p = b->G[1].h_am[i];
        int best, pi;
        memcpy(&best, &a.y, 4);
        memcpy(&pi, &p.y, 4);
        float bv = a.x;
        if (p.x > bv || (p.x == bv && pi < best)) { best = pi; bv = p.x; }
        out[i] = best;
        b->tok[slots[i]] = best;
        b->pos[slots[i]]++;
    }
    if (S.bd_prof) {
        for (int g = 0; g < 2; g++) {
            auto& E = g ? ev1 : ev;
            auto& N = g ? evn1 : evn;
            auto& PR = g ? b->prof1 : b->prof;
            CK(cudaSetDevice(g));
            for (size_t i = 1; i < E.size(); i++) {
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, E[i - 1], E[i]));
                bool found = false;
                for (auto& p : PR)
                    if (p.first == N[i]) { p.second.first += ms; p.second.second++; found = true; break; }
                if (!found) PR.push_back({N[i], {ms, 1}});
            }
            for (auto e : E) cudaEventDestroy(e);
        }
    }
    b->last_B = n;
    b->last_slots.assign(slots, slots + n);
    b->step_s += secs(t0);
    b->steps++;
    b->rows += n;
    return n;
}

int tp_batch_logits(t4q_ctx* c, int row, float* out) {
    tp::State& S = *c->tps;
    Bd* b = bd_of(c);
    if (row < 0 || row >= b->last_B) throw std::runtime_error("bad row");
    for (int g = 0; g < 2; g++) {
        CK(cudaSetDevice(g));
        CK(cudaStreamSynchronize(S.G[g].s));
        CK(cudaMemcpy(out + (size_t)124160 * g, b->G[g].logits + (size_t)row * 124160, 124160 * 4, cudaMemcpyDeviceToHost));
    }
    return 0;
}

std::string tp_batch_stats(t4q_ctx* c) {
    tp::State& S = *c->tps;
    Bd* b = (Bd*)S.bd;
    if (!b) return "";
    char buf[512];
    snprintf(buf, sizeof buf,
             ", \"batch\": {\"n_slots\": %d, \"slot_ctx\": %d, \"state_f16\": %d, \"mib_per_gpu\": %.0f, \"steps\": %ld, "
             "\"rows\": %ld, \"step_s\": %.4f, \"enqueue_s\": %.4f, \"prefills\": %ld, \"prefill_s\": %.4f, \"head\": %d, "
             "\"ch\": %d, \"p2p\": %d, \"last_B\": %d",
             b->n_slots, b->slot_ctx, b->sf16, b->bytes / 1048576.0, b->steps, b->rows, b->step_s, b->enqueue_s,
             b->prefills, b->prefill_s, S.bd_head, S.bd_ch, S.bd_p2p, b->last_B);
    std::string s = buf;
    for (int g = 0; g < 2; g++) {
        const auto& PR = g ? b->prof1 : b->prof;
        if (PR.empty()) continue;
        s += g ? ", \"profile1\": {" : ", \"profile\": {";
        for (size_t i = 0; i < PR.size(); i++) {
            snprintf(buf, sizeof buf, "%s\"%s\": [%.3f, %d]", i ? ", " : "", PR[i].first.c_str(), PR[i].second.first,
                     PR[i].second.second);
            s += buf;
        }
        s += "}";
    }
    return s + "}";
}

void tp_batch_reset_stats(t4q_ctx* c) {
    Bd* b = (Bd*)c->tps->bd;
    if (!b) return;
    b->step_s = 0; b->steps = 0; b->rows = 0; b->prefill_s = 0; b->prefills = 0; b->enqueue_s = 0;
    b->prof.clear();
    b->prof1.clear();
}
