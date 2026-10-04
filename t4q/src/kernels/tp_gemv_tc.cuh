// t4q/src/kernels/tp_gemv_tc.cuh -- spec verify GEMV on the int4 tensor cores (option spec_tc, milestone M5 round 2).
//
// The verify pass (tp_spec.cu) runs every decode GEMV over M = k + 1 activation columns. k_gemv runs the M = 1 dp4a
// body per column: the weight loads amortize, but the per-column dp4a / x-read / epilogue work grows with M and the
// one-tile-per-warp grid no longer fits its blocks per SM, so gate|up falls to ~185 GB/s at M = 4 and k >= 4 loses to
// k = 3. This kernel keeps the weight stream (the same decode-layout P4 codes, read through gemm8::WPtr) and moves the
// dot products onto mma.m8n8k32 s4/u4: the int32 accumulation is exact and the 8-column C tile absorbs every M <= 8
// column at once (TPD = 8 token slots; MMAX = 7 real + padding, whose lanes are discarded by the tok < T guards).
//
// Numerical contract vs k_gemv's P4 CVT = 2 (p4u = 1) arithmetic:
//   dp4a: per 32-group, s16 = 16 * sum(x * c) (sl * 16 + sh, group_dot CVT 2) and the folded float is
//         h2f(d_w) * ((x_d * 0.0625f) * (__int_as_float(s16 + moff) - 12582912.f))
//         with moff = 0x4B400000 - 128 * (s0 + s1), so the integer inside is exactly 16 * P with
//         P = sum(x * (c - 8)), and (x_d * 0.0625) * float(16 P) == x_d * float(P) exactly (power of two).
//   mma:  weight code as s4 (c ^ 8), activation split into nibble planes x = 16 * hi + lo (d4_from_q8's exact
//         shuffle), so 16 * H + L - MAGIC = P: the same integer (int32 accumulation is exact, order-free).
//   The epilogue replicates the dp4a float expression verbatim per group, so every group product is bit-identical.
//   The ACCUMULATION ORDER differs: k_gemv sums 16 per-lane chunk partials through a shfl_xor tree at the end,
//   this kernel folds group products in flat ascending block order - results agree to ULPs (same products, other
//   rounding sequence), like the t4q-vs-llama.cpp selftest bound. The spec_check "tc" gate is therefore argmax
//   equality of the forced-draft verify logits (tc 0 vs 1) + a reported max-abs-diff, plus V5 byte-identity of
//   the greedy token stream; NOT bit equality.
//
// Activations are the engine's existing per-32 q8_1 buffers (xq / xm), read directly: the digit planes are built
// in-flight at smem staging (no conversion pass, no extra buffer). Padding rows M..7 are never written by producers;
// their stale content only feeds discarded lanes.
//
// Shape: one token tile of TPD = 8 (one mma m-tile), BR weight rows per block (64 or 128, chosen at launch so the
// grid fills the SMs in a single wave), k-stages of 64 (two Q4 blocks) with a register double buffer, as gemm20.
#pragma once
#include "gemm8.cuh"

#ifdef __CUDACC__
namespace t4q {
namespace gtc {

using gemv::FAST_P4;

struct TcArgs {
    gemm8::Args q;               // decode-layout P4 weights (codes / d planes, N, K, ntiles, cm)
    const int8_t* xq = nullptr;  // [T][K] int8 (row stride K); rows >= T stale, discarded
    const int2* xms = nullptr;   // [T][K/32] q8_1 meta (d bits in .x); rows >= T stale, discarded
    float* y = nullptr;          // SQ == 0: y[col * ldy + row]
    int ldy = 0, N = 0, K = 0, T = 0;
    int8_t* sq_xq = nullptr;     // SQ == 1: [T][N/2] q8 of silu(gate) * up per column (k_gemv SQ layout)
    int2* sq_xm = nullptr;       // SQ == 1: [T][N/64]
};

constexpr int TPD = 8;  // token slots (mma m-dim); T = M <= MMAX = 7

__device__ __forceinline__ void ldsm_x1(uint32_t& r0, const void* p) {
    const unsigned sp = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(r0) : "r"(sp));
}
__device__ __forceinline__ void ldsm_x2(uint32_t& r0, uint32_t& r1, const void* p) {
    const unsigned sp = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(sp));
}
__device__ __forceinline__ void mma_ss(int& d0, int& d1, uint32_t a, uint32_t b, int c0, int c1) {
    asm("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1}, {%2}, {%3}, {%4,%5};"
        : "=r"(d0), "=r"(d1) : "r"(a), "r"(b), "r"(c0), "r"(c1));
}
__device__ __forceinline__ void mma_us(int& d0, int& d1, uint32_t a, uint32_t b, int c0, int c1) {
    asm("mma.sync.aligned.m8n8k32.row.col.s32.u4.s4.s32 {%0,%1}, {%2}, {%3}, {%4,%5};"
        : "=r"(d0), "=r"(d1) : "r"(a), "r"(b), "r"(c0), "r"(c1));
}
// 32-B weight rows: 16-B unit swizzle so 8 consecutive rows hit distinct banks (gemm20's)
__device__ __forceinline__ int wswz(int row, int u) { return u ^ ((row >> 2) & 1); }

// q8_1 quantization of one 32-group held one value per lane (copy of tp::quant_warp: same math, bit-identical)
__device__ __forceinline__ void tc_quant_warp(float v, int8_t* xq_i, int2* xm_g) {
    const int lane = threadIdx.x & 31;
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.f;
    const int q = amax == 0.f ? 0 : (int)roundf(v / d);
    *xq_i = (int8_t)q;
    int s = q;
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const int s1 = __shfl_sync(0xffffffffu, s, 16);
    if (lane == 0) *xm_g = make_int2(__float_as_int(d), (int)((unsigned)(s & 0xffff) | ((unsigned)s1 << 16)));
}

template <int BR, int SQ>
struct Cfg {
    static constexpr int O_X = 0;                  // [TPD][64 B] digit planes per k64 stage
    static constexpr int O_W = O_X + TPD * 64;     // [BR][32 B] codes per stage
    static constexpr int O_WS = O_W + BR * 32;     // [BR] float4 {d0, d1, -, -}
    static constexpr int O_XS = O_WS + BR * 16;    // [2][TPD] activation d per 32-group
    static constexpr int STAGE = O_XS + 2 * TPD * 4;
    static constexpr int O_SQ = 2 * STAGE;         // SQ: [BR][TPD] fp32 outputs staged by row
    static constexpr int SMEM = 2 * STAGE + (SQ ? BR * TPD * 4 : 0);
    static constexpr int WN = BR / 4;              // rows per warp (4 row groups wn)
    static constexpr int NG = WN / 8;              // 8-row n tiles per warp
};

template <int RPL, int BR, int SQ>
__global__ void __launch_bounds__(256, BR == 64 ? 3 : 2) gemv_tc_kernel(const TcArgs a) {
    using C = Cfg<BR, SQ>;
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wn = warp >> 1;  // 4 row groups; the token tile is one mma m-tile (tokens 0..7, lane >> 2)
    const int row0 = blockIdx.y * BR;
    const int K = a.K, nst = K >> 6, K32 = K >> 5;
    const int t4 = lane & 3;

    // staging roles: X: threads 0..15 (token tid >> 1, k32 half tid & 1); W: BR rows x 2 halves (BR <= 128 -> all
    // 256 threads for BR 128, threads 0..127 for BR 64); XS: threads 0..TPD-1. Weight rows are clamped: the last
    // tile may hang over N (its lanes compute garbage that the epilogue discards).
    const int xt = tid >> 1;
    const uint8_t* xraw = (const uint8_t*)a.xq + (size_t)xt * K + (tid & 1) * 32;
    const int wrow = tid >> 1;                     // local row in the tile
    const int wrow_g = min(row0 + wrow, a.N - 1);  // clamped global row for the loads
    const int wh = tid & 1;
    const gemm8::WPtr<FAST_P4, RPL> P(a.q, wrow_g, wh);
    const int xtk = tid < TPD ? tid : 0;

    float F[C::NG][2];
#pragma unroll
    for (int g = 0; g < C::NG; ++g) F[g][0] = F[g][1] = 0.f;

    int4 sx[2], sw[2];
    uint32_t sd0 = 0;
    float sxs0 = 0.f, sxs1 = 0.f;
    auto load = [&](int st) {
        if (tid < TPD * 2) {
            // raw q8 bytes of one k32 -> [lo 16 B | hi 16 B] digit planes (d4_from_q8's exact shuffle)
            const int4 A = __ldg((const int4*)(xraw + (size_t)st * 64));
            const int4 Bv = __ldg((const int4*)(xraw + (size_t)st * 64 + 16));
            const uint32_t Aw[4] = {(uint32_t)A.x, (uint32_t)A.y, (uint32_t)A.z, (uint32_t)A.w};
            const uint32_t Bw[4] = {(uint32_t)Bv.x, (uint32_t)Bv.y, (uint32_t)Bv.z, (uint32_t)Bv.w};
            uint32_t lo[4], hi[4];
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                lo[w] = (Aw[w] & 0x0F0F0F0Fu) | ((Bw[w] & 0x0F0F0F0Fu) << 4);
                hi[w] = ((Aw[w] >> 4) & 0x0F0F0F0Fu) | (Bw[w] & 0xF0F0F0F0u);
            }
            sx[0] = make_int4((int)lo[0], (int)lo[1], (int)lo[2], (int)lo[3]);
            sx[1] = make_int4((int)hi[0], (int)hi[1], (int)hi[2], (int)hi[3]);
        }
        const int c = (st * 2) >> 4, jj = (st * 2) & 15;  // jj is even: wh + jj stays inside the h half
        // L2 prefetch ~12 stages ahead (~770 k): the 2-deep register buffer cannot cover the DRAM latency on
        // its own (v13: 63-107 GB/s, latency-bound); the dp4a gemv uses the same trick (gemm8::prefetch9)
        if (st + 12 < nst) {
            const int c2 = ((st + 12) * 2) >> 4, jj2 = ((st + 12) * 2) & 15;
            gemm8::prefetch_l2(P.code + c2 * P.cs_code + jj2 * 16);
            gemm8::prefetch_l2(P.sa + c2 * P.cs_s + jj2 * RPL * 2);
        }
        const uint8_t* cp = P.code + c * P.cs_code + jj * 16;
        const uint8_t* dp = P.sa + c * P.cs_s + jj * RPL * 2;
        sw[0] = gemv::ldg_nc_v4(cp);
        sd0 = (uint32_t)__ldg((const unsigned short*)dp);
        if (tid < TPD) {
            const int2 m = a.xms[(size_t)xtk * K32 + st * 2];
            sxs0 = __int_as_float(m.x);
            sxs1 = __int_as_float(a.xms[(size_t)xtk * K32 + st * 2 + 1].x);
        }
    };
    auto store = [&](int b) {
        unsigned char* B = smem + b * C::STAGE;
        if (tid < TPD * 2) {
            const int xr = tid >> 1, xu = (tid & 1) * 2;
            *(int4*)(B + C::O_X + xr * 64 + gemm8::swz(xr, xu) * 16) = sx[0];
            *(int4*)(B + C::O_X + xr * 64 + gemm8::swz(xr, xu + 1) * 16) = sx[1];
        }
        auto flip = [](int4 v) {
            return make_int4(v.x ^ 0x88888888, v.y ^ 0x88888888, v.z ^ 0x88888888, v.w ^ 0x88888888);
        };
        if (wrow < BR) {  // BR 64: threads >= 128 own no row
            *(int4*)(B + C::O_W + wrow * 32 + wswz(wrow, wh) * 16) = flip(sw[0]);
            // the pair partner (tid ^ 1) holds the stage's other k32 d for this row
            const float dm = gemm8::h2f(sd0), dp2 = __shfl_xor_sync(0xffffffffu, dm, 1);
            if (wh == 0) *(float4*)(B + C::O_WS + wrow * 16) = make_float4(dm, dp2, 0.f, 0.f);
        }
        if (tid < TPD) {
            *(float*)(B + C::O_XS + tid * 4) = sxs0;
            *(float*)(B + C::O_XS + TPD * 4 + tid * 4) = sxs1;
        }
    };

    load(0);
    store(0);
    __syncthreads();
    int buf = 0;
    for (int s = 0; s < nst; ++s) {
        if (s + 1 < nst) load(s + 1);
        const unsigned char* Bp = smem + buf * C::STAGE;
        uint32_t fa[2][2], fb[2][C::NG];
#pragma unroll
        for (int bb = 0; bb < 2; ++bb)
#pragma unroll
            for (int dg = 0; dg < 2; ++dg)
                ldsm_x1(fa[dg][bb], Bp + C::O_X + (lane & 7) * 64 + gemm8::swz(lane & 7, bb * 2 + dg) * 16);
#pragma unroll
        for (int bb = 0; bb < 2; ++bb) {
            const int r = wn * C::WN + (lane >> 3) * 8 + (lane & 7);
            if (C::NG == 4)
                gemm8::ldsm_x4(fb[bb][0], fb[bb][1], fb[bb][2], fb[bb][3],
                               Bp + C::O_W + r * 32 + wswz(r, bb) * 16);
            else if (C::NG == 2)
                ldsm_x2(fb[bb][0], fb[bb][1], Bp + C::O_W + r * 32 + wswz(r, bb) * 16);
        }
        const float xd0 = *(const float*)(Bp + C::O_XS + (lane >> 2) * 4);
        const float xd1 = *(const float*)(Bp + C::O_XS + TPD * 4 + (lane >> 2) * 4);
#pragma unroll
        for (int g = 0; g < C::NG; ++g) {
            float4 ws[2];  // per row: {d of the stage's even k32, d of the odd k32} (gemm20's layout)
#pragma unroll
            for (int e = 0; e < 2; ++e)
                ws[e] = *(const float4*)(Bp + C::O_WS + (wn * C::WN + 8 * g + 2 * t4 + e) * 16);
            int h0[2], l0[2], h1[2], l1[2];
            mma_ss(h0[0], h0[1], fa[1][0], fb[0][g], 0, 0);
            mma_us(l0[0], l0[1], fa[0][0], fb[0][g], gemm8::MAGIC_I, gemm8::MAGIC_I);
            mma_ss(h1[0], h1[1], fa[1][1], fb[1][g], 0, 0);
            mma_us(l1[0], l1[1], fa[0][1], fb[1][g], gemm8::MAGIC_I, gemm8::MAGIC_I);
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                // P = 16 * H + L - MAGIC: the exact per-32 integer of the dp4a path (sum(x * (c - 8)))
                const int P0 = ((h0[e] << 4) + l0[e]) - gemm8::MAGIC_I;
                const int P1 = ((h1[e] << 4) + l1[e]) - gemm8::MAGIC_I;
                // magic I2F of 16 * P (|16 P| <= 32 * 127 * 8 * 16 < 2^22), then the dp4a epilogue verbatim
                const float f0 = __int_as_float((P0 << 4) + gemm8::MAGIC_I) - gemm8::MAGIC_F;
                const float f1 = __int_as_float((P1 << 4) + gemm8::MAGIC_I) - gemm8::MAGIC_F;
                F[g][e] += ws[e].x * ((xd0 * 0.0625f) * f0);
                F[g][e] += ws[e].y * ((xd1 * 0.0625f) * f1);
            }
        }
        if (s + 1 < nst) store(buf ^ 1);
        __syncthreads();
        buf ^= 1;
    }
    if (SQ == 0) {
#pragma unroll
        for (int g = 0; g < C::NG; ++g) {
            const int row = row0 + wn * C::WN + 8 * g + 2 * t4;
            if (row >= a.N) continue;
            const int tok = lane >> 2;
            if (tok >= a.T) continue;
            float2* p = (float2*)(a.y + (size_t)tok * a.ldy + row);
            p->x = F[g][0];
            p->y = F[g][1];
        }
    } else {
        // fp32 y too (dead downstream in the spec engine, but keeps tc and dp4a memory-faithful for A/B checks)
#pragma unroll
        for (int g = 0; g < C::NG; ++g) {
            const int row = row0 + wn * C::WN + 8 * g + 2 * t4;
            if (row >= a.N) continue;
            const int tok = lane >> 2;
            if (tok >= a.T) continue;
            *(float2*)(a.y + (size_t)tok * a.ldy + row) = make_float2(F[g][0], F[g][1]);
        }
        // silu(gate) * up -> q8 per column (the gate|up tensor's 8-row tiles: 4 gate rows + 4 up rows; the down row
        // for staged gate row grow is ob = row0/2 + (grow >> 1) - ... in group form below)
        __syncthreads();
        float* s_q = (float*)(smem + C::O_SQ);
#pragma unroll
        for (int g = 0; g < C::NG; ++g)
#pragma unroll
            for (int e = 0; e < 2; ++e)
                s_q[(wn * C::WN + 8 * g + 2 * t4 + e) * TPD + (lane >> 2)] = F[g][e];
        __syncthreads();
        const int nog = BR / 64;  // down rows per block = BR/2 (4 gate + 4 up rows per 8-row tile) = nog * 32
        for (int job = warp; job < a.T * nog; job += 8) {
            const int col = job / nog, ogi = job % nog;
            if (row0 / 2 + ogi * 32 + 32 > a.N / 2) continue;  // whole group out of range (padded tail tile)
            const int o = ogi * 32 + lane;
            const int grow = 8 * (o >> 2) + (o & 3);
            const float gval = s_q[grow * TPD + col], uval = s_q[(grow + 4) * TPD + col];
            const float val = (gval / (1.0f + expf(-gval))) * uval;
            const int ob = row0 / 2 + o;
            tc_quant_warp(val, a.sq_xq + (size_t)col * (a.N / 2) + ob, a.sq_xm + (size_t)col * (a.N / 64) + (ob >> 5));
        }
    }
}

template <int RPL, int BR, int SQ>
static cudaError_t launch_t(const TcArgs& a, cudaStream_t s) {
    if (a.K % 64 || a.T < 1 || a.T > TPD) return cudaErrorInvalidValue;
    using C = Cfg<BR, SQ>;
    static int attr_mask = 0;  // per device: prefer max smem carveout
    int dev = 0;
    cudaGetDevice(&dev);
    if (!(attr_mask & (1 << dev))) {
        cudaError_t e = cudaFuncSetAttribute(gemv_tc_kernel<RPL, BR, SQ>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                             100);
        if (e != cudaSuccess) return e;
        attr_mask |= 1 << dev;
    }
    dim3 grid(1, (a.N + BR - 1) / BR);
    gemv_tc_kernel<RPL, BR, SQ><<<grid, 256, C::SMEM, s>>>(a);
    return cudaGetLastError();
}

// pick BR so the grid fills the machine in one wave (T4: 80 SMs): big N -> 128-row tiles, small N -> 64
static inline cudaError_t gemv_tc_launch(int rpl, bool sq, const TcArgs& a, cudaStream_t s) {
    const bool big = a.N >= 56 * 128;
    if (rpl == 2) return sq ? cudaErrorInvalidValue : (big ? launch_t<2, 128, 0>(a, s) : launch_t<2, 64, 0>(a, s));
    if (rpl == 4) return sq ? (big ? launch_t<4, 128, 1>(a, s) : launch_t<4, 64, 1>(a, s))
                            : (big ? launch_t<4, 128, 0>(a, s) : launch_t<4, 64, 0>(a, s));
    return cudaErrorInvalidValue;
}

// qkvz's 48 fp32 alpha/beta rows (k_gemv SEG M>1 path verbatim): grid ceil(nrows / 8) x 256, warp per row
__global__ void k_seg_m(const float* __restrict__ w, const float* __restrict__ x, float* y, int nrows, int M) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= nrows) return;
    const int lane = threadIdx.x & 31;
    const float4* w4 = (const float4*)(w + (size_t)row * 5120);
    for (int col = 0; col < M; col++) {
        const float4* x4 = (const float4*)(x + (size_t)col * 5120);
        float acc = 0.f;
#pragma unroll 4
        for (int i = lane; i < 1280; i += 32) {
            const float4 wv = __ldg(w4 + i), xv = __ldg(x4 + i);
            acc += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) y[(size_t)col * 64 + row] = acc;
    }
}

}  // namespace gtc
}  // namespace t4q
#endif
