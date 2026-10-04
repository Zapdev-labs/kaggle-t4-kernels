// t4q/src/kernels/tp_gemv_tc.cuh -- spec verify GEMV on the int4 tensor cores (option spec_tc, milestone M5 r2-r5).
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
//   equality of the forced-draft verify logits (tc 0 vs 1) + a reported max-abs-diff, plus accept-rate parity of
//   the greedy stream; NOT bit equality.
//
// r4 structure (v25+): same warp / row / fragment ownership and the same numeric path as r3, but the weight stream is
// restructured after t4q-b's diagnosis: a warp that keeps only one 64-el stage of loads in flight rings ~105-115 GB/s
// no matter the instruction mix (r3 measured exactly that band), while the dp4a's D=2 512-el register ring sustains
// 250+. So r4 loads in GROUPS of 4 stages (256 k-elements = the 8 consecutive Q4 blocks of one plane-half unit run):
//   * per lane (row = lane>>1, half wh = lane&1): 4 code LDG.128 + 4 d LDG.u16 per group, the 4 code units
//     (jj0 + wh*4 + k, k = 0..3) CONTIGUOUS = 64 B in one L2 line pair - held in a 2-deep register ring, so every
//     load batch has two full groups (~8 stages of mma work) to land before its STS, like the dp4a's chunk ring;
//   * the smem W slice is a 2-deep ring of GROUP buffers ([4 stages x 16 rows x 32 B codes][16 rows x 4 x 8 B d]);
//     one __syncwarp per group (4x fewer), one ldsm_x4 per stage (was 2 x ldsm_x2), and the stage d pair is packed
//     as one float2 per row (r3 staged {dm, dp2, 0, 0} float4s via a shfl + half-wasted stores; r4's lane already
//     holds both d words of its two stages straight from the 4-word d batch, so the shfl and the waste are gone);
//   * A fragments stay r3's per-stage in-register build from the L2-hot global q8 (the gemm9 line proves a
//     1-stage activation ring is fine at 4x this kernel's weight throughput).
// r5 structure (v26+): v25 measured the flaw - r4's smem (44 KB at BR 128 / 22 KB at BR 64) capped every shape at
// 8 warps/SM while r3 ran 16, so the doubled per-warp bytes in flight were spent buying back the lost occupancy
// (only the longest-K shape, down 110 -> 142 GB/s, came out ahead). The ring is now parameterized, RREG x RSMEM:
//   * <2,1> (the engine default): one smem group buffer (20/10 KB -> 16 warps/SM again) with r4's 2-group register
//     ring. Step t: STS(G_t) -> syncwarp -> LDG(G_{t+RREG}) into the just-consumed slot -> compute(G_t) ->
//     syncwarp; the STS rides RREG group-times behind its LDG, and the closing syncwarp orders the step's ldsm
//     reads before the next STS rewrites the single buffer.
//   * <4,1>/<4,2>: a 4-group register ring (~164 regs, 8-12 warps/SM) - twice the DRAM-latency cover per warp.
//   * BR 64 forced for big N (tc_bench variant 4): BR 128 at N 8192 is 64 blocks on 40 SMs, a 2:1 SM imbalance.
// tc_bench --variants A/Bs the matrix per shape; the numeric path is byte-identical across variants (same
// mma / fold / epilogue order - only the ring scheduling and the smem buffer count differ).
// r6 (v27+): v26's matrix killed the r5 occupancy theory - <2,1> (16 warps/SM) matched <2,2> (8) within noise on
// every shape and <4,*> lost ~10%, while the whole matrix plateaued at 105-125 GB/s against dp4a's 250-260 on the
// same weight stream (down 142 -> 127 across runs says cross-run numbers drift; only the within-run A/B counts).
// dp4a's recipe is a FEW HUGE load batches per warp: 3 loads (2x LDG.128 + 1 u32) feed 1024 elements of dp4a
// work, so its batches always land inside the consume time. r6 chases the two remaining structural deltas, both
// as tc_bench variants in the same run:
//   * HOIST (v 5/6/7/8/9): compute's per-stage A-side global reads (4x LDG.32 q8 + 1x LDG.128 xms per lane per
//     stage, non-volatile so ptxas left them inline per stage, ~270+ cy of L2 latency each on the mma path) move
//     to the TOP of the group: all 16 q8 words + 4 xms words issue back-to-back as pinned volatile LDGs before
//     the first ldsm, ~4 stages of W work of cover.
//   * BAT (v 6/7/8): the per-step 4x LDG.128 W batch becomes a PAIR/QUAD burst - the block's 2/4 STSes go first
//     (RSMEM 2), then ONE burst of 8/16 consecutive LDG.128 + d loads into the just-freed slots, then the
//     block's computes; the burst rides RREG groups of cover before its STS (dp4a's shape: ~4.6/9.2 KB of weight
//     bytes issued at once vs the step's 2.3 KB). Bursts drop gload's L2 prefetch (hardware MLP covers DRAM
//     latency once 8-16 LDGs are outstanding; the 8 prefetch ops per group only doubled the LSU pressure).
// r3 structure (v18+): BR/16 warps of 32 lanes, each warp owning exactly 16 weight rows = 2 mma n-tiles (v16's
// 256-thread block mapped two warps onto every tile and computed each one twice). Each warp owns its rows end to end:
//   * W staging is warp-private, so the k-loop needs only __syncwarp, never __syncthreads (the SQ epilogue's one
//     cross-warp handoff excepted);
//   * A fragments are built in-register from the L2-hot global q8: no X smem stage and no ldmatrix on the
//     activation side at all.
// Shape: one token tile of TPD = 8 (one mma m-tile), BR weight rows per block (64 or 128, chosen at launch so the
// grid fills the SMs in a single wave), k-groups of 256 elements with register + smem double rings.
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
__device__ __forceinline__ int4 flip4(int4 v) {
    return make_int4(v.x ^ 0x88888888, v.y ^ 0x88888888, v.z ^ 0x88888888, v.w ^ 0x88888888);
}

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

template <int BR, int SQ, int RSMEM>
struct Cfg {
    static constexpr int WN = 16;        // rows per warp = 2 mma n-tiles
    static constexpr int NW = BR / WN;   // warps per block: 4 (BR 64, 128 threads) or 8 (BR 128, 256 threads)
    static constexpr int NG = 2;         // mma n-tiles per warp (16 rows)
    // per-warp GROUP buffer (4 stages of 64 el): [4 stages x WN x 32 B codes][WN rows x 4 stages x 8 B {d_e, d_o}]
    static constexpr int WGRP = 4 * WN * 32 + WN * 32;
    static constexpr int STAGE = NW * WGRP;         // ONE smem ring buffer (a group); the ring is RSMEM deep
    static constexpr int O_SQ = RSMEM * STAGE;      // SQ: [BR][TPD] fp32 outputs staged by row
    static constexpr int SMEM = O_SQ + (SQ ? BR * TPD * 4 : 0);
    // <*,1>: BR 128 20 KB (+SQ 4) = 24 KB -> 2 blocks/SM, BR 64 10 KB -> 4 = 16 warps/SM; <*,2> is r4's 8
};

template <int RPL, int BR, int SQ, int RREG, int RSMEM, int HOIST, int BAT>
__global__ void __launch_bounds__(BR / 16 * 32,
                                  HOIST ? (RREG == 2 ? (BR == 64 ? 3 : 1) : (BR == 64 ? 2 : 1))
                                        : (RREG == 2 ? (BR == 64 ? 4 : 2) : (BR == 64 ? 2 : 1)))
gemv_tc_kernel(const TcArgs a) {
    using C = Cfg<BR, SQ, RSMEM>;
    static_assert(RSMEM >= 2 || BAT == 0, "burst schedules store both pair buffers before compute");
    static_assert(BAT != 2 || RREG == 4, "burst-4 is the 4-group register ring's maximum batch");
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int row0 = blockIdx.y * BR;
    const int K = a.K, nst = K >> 6, ng = nst >> 2, K32 = K >> 5;
    const int t4 = lane & 3, tok = lane >> 2;

    // this warp owns WN weight rows; the row's 8 group code units (128 B, one plane-half unit run) live in the
    // lane pair (lane >> 1 = row, lane & 1 = wh): wh 0 holds the group's stages 0-1 (blocks 8gi..8gi+3), wh 1
    // stages 2-3, each lane's 4 units one CONTIGUOUS 64-B run of the row's plane-half.
    const int wrow = warp * C::WN + (lane >> 1);   // local row in the tile
    const int wrow_g = min(row0 + wrow, a.N - 1);  // clamped global row for the loads (padded tile tail)
    const int wh = lane & 1;
    const gemm8::WPtr<FAST_P4, RPL> P(a.q, wrow_g, 0);  // blk 0: P.code / P.sa sit at the row's plane-half base
    const int8_t* xtok = (const int8_t*)a.xq + (size_t)tok * K;  // own token's q8 row (tok >= T: stale, discarded)

    float F[C::NG][2];
#pragma unroll
    for (int g = 0; g < C::NG; ++g) F[g][0] = F[g][1] = 0.f;

    // W register ring (RREG groups deep, the dp4a's chunk ring): group gi = blocks 8gi..8gi+7 = units
    // jj0..jj0+7 of the row's plane-half, chunk c = gi >> 1, jj0 = (gi & 1) << 3. The lane's 4 code units are
    // (jj0 + wh*4 + k), k = 0..3: 64 B contiguous, and its 4 d words are the FULL d pairs of its two stages
    // (blocks 8gi+4wh+2u and +1), so the d staging needs no shfl and wastes no smem. Group gi rides slot
    // gi % RREG / smem buffer gi % RSMEM: both indices are call-site constants (the unrolled step loop), so
    // the arrays stay in registers (a runtime ring index lands the whole set in local memory - the first r4
    // draft's 160-B stack).
    int4 rq[RREG][4];
    uint32_t rd[RREG][4];
    auto gload = [&](int gi, int4 (&rq)[4], uint32_t (&rd)[4]) {
        const int c = gi >> 1, j = ((gi & 1) << 3) + wh * 4;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            rq[k] = gemv::ldg_nc_v4(P.code + c * P.cs_code + (j + k) * 16);
            rd[k] = (uint32_t)__ldg((const unsigned short*)(P.sa + c * P.cs_s + (j + k) * RPL * 2));
        }
        if (BAT == 0 && gi + RREG + 2 < ng) {  // L2 prefetch beyond the ring horizon (BAT 0 only: with 8-16 LDGs
            // outstanding the hardware MLP covers DRAM latency and the prefetches only add LSU pressure)
            const int gp = gi + RREG + 2, c2 = gp >> 1, j2 = ((gp & 1) << 3) + wh * 4;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                gemm8::prefetch_l2(P.code + c2 * P.cs_code + (j2 + k) * 16);
                gemm8::prefetch_l2(P.sa + c2 * P.cs_s + (j2 + k) * RPL * 2);
            }
        }
    };
    auto gstore = [&](const int4 (&rq)[4], const uint32_t (&rd)[4], unsigned char* B) {
        const int row = lane >> 1;
        // unit (j + k) = block 8gi + wh*4 + k: stage sloc = 2*wh + (k >> 1) (k = 2*sloc + bb), code 16 B per row
        // per stage at wswz(row, bb), d pair {d(2st), d(2st+1)} one float2 per row per stage
#pragma unroll
        for (int k = 0; k < 4; ++k)
            *(int4*)(B + (2 * wh + (k >> 1)) * (C::WN * 32) + row * 32 + wswz(row, k & 1) * 16) = flip4(rq[k]);
#pragma unroll
        for (int u = 0; u < 2; ++u)
            *(float2*)(B + 4 * C::WN * 32 + row * 32 + (2 * wh + u) * 8) =
                make_float2(gemm8::h2f(rd[2 * u]), gemm8::h2f(rd[2 * u + 1]));
    };
    auto compute = [&](int gi, const unsigned char* Bp) {
        // A-side (HOIST, r6): the whole group's activation reads up front, pinned volatile so ptxas cannot
        // sink them back into the stage loop - 16 q8 words (the lane's lo/hi nibble pair of each stage's 2
        // blocks) + 4 xms words issue back-to-back and hide their L2 latency under the group's W work, instead
        // of each stage's mma waiting on its own fresh ~270+ cy loads
        uint32_t aq[4][4];  // [stage][2*bb + half]: q8 word (half 0 = lo4 base, 1 = +16)
        int4 axd[4];        // [stage]: xms words (d(2st) pair .x/.y, d(2st+1) pair .z/.w)
        if (HOIST) {
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const int st = gi * 4 + s;
                axd[s] = gemv::ldg_nc_v4((const int4*)(a.xms + (size_t)tok * K32 + st * 2));
#pragma unroll
                for (int bb = 0; bb < 2; ++bb) {
                    aq[s][2 * bb] = gemv::ldg_nc_u32(xtok + st * 64 + bb * 32 + t4 * 4);
                    aq[s][2 * bb + 1] = gemv::ldg_nc_u32(xtok + st * 64 + bb * 32 + t4 * 4 + 16);
                }
            }
        }
#pragma unroll
        for (int s = 0; s < 4; ++s) {  // stage = 64 el (2 blocks); the r3 numeric path, order and epilogue verbatim
            const int st = gi * 4 + s;
            // own-token A fragments straight from global q8: the fragment nibble at (byte m, half h) is the digit
            // of k = bb*32 + t4*4 + m + 16h, the same GGUF interleave the staged W codes carry, so mma nibble j
            // of A and B always multiply the same k (the int32 sum is order-free anyway)
            uint32_t fa[2][2];  // [dg][bb]: dg 0 = lo digits (u4), dg 1 = hi digits (s4)
#pragma unroll
            for (int bb = 0; bb < 2; ++bb) {
                const uint32_t lo4 = HOIST ? aq[s][2 * bb] : *(const uint32_t*)(xtok + st * 64 + bb * 32 + t4 * 4);
                const uint32_t hi4 = HOIST ? aq[s][2 * bb + 1]
                                           : *(const uint32_t*)(xtok + st * 64 + bb * 32 + t4 * 4 + 16);
                fa[0][bb] = (lo4 & 0x0F0F0F0Fu) | ((hi4 & 0x0F0F0F0Fu) << 4);
                fa[1][bb] = ((lo4 >> 4) & 0x0F0F0F0Fu) | (hi4 & 0xF0F0F0F0u);
            }
            // both bb x g B fragments in ONE ldsm_x4 (r3: two ldsm_x2): lanes 8m..8m+7 address matrix m,
            // m = 2*bb + g over (rows 8g..8g+7, unit wswz(row, bb)); r3's lanes 16..31 idle addresses are gone
            uint32_t fb[2][2];
            {
                const int m = lane >> 3, r = 8 * (m & 1) + (lane & 7), bb = m >> 1;
                gemm8::ldsm_x4(fb[0][0], fb[0][1], fb[1][0], fb[1][1],
                               Bp + s * (C::WN * 32) + r * 32 + wswz(r, bb) * 16);
            }
            // activation d of this token at the stage's two k32 blocks (one 16-B load: the index is 16-B aligned)
            const int4 m2 = HOIST ? axd[s] : __ldg((const int4*)(a.xms + (size_t)tok * K32 + st * 2));
            const float xd0 = __int_as_float(m2.x), xd1 = __int_as_float(m2.z);
#pragma unroll
            for (int g = 0; g < C::NG; ++g) {
                float2 ws[2];  // per row: {d of the stage's even k32, d of the odd k32} (gemm20's layout, packed)
#pragma unroll
                for (int e = 0; e < 2; ++e)
                    ws[e] = *(const float2*)(Bp + 4 * C::WN * 32 + (8 * g + 2 * t4 + e) * 32 + s * 8);
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
        }
    };

    // One STEP per group (r5, BAT 0): STS(G_t) (its LDG was RREG groups back) -> syncwarp -> refill the
    // just-consumed slot with G_{t+RREG} -> compute(G_t) -> syncwarp. The opening syncwarp makes the STS visible
    // to the whole warp's ldsm; the closing one orders the step's ldsm reads before any later STS rewrites the
    // buffer (the single-buffer RSMEM=1 case needs it, the 2-buffer case gets it free).
    auto step = [&](int t, int4 (&rqs)[4], uint32_t (&rds)[4], unsigned char* B) {
        gstore(rqs, rds, B);
        __syncwarp();
        if (t + RREG < ng) gload(t + RREG, rqs, rds);
        compute(t, B);
        __syncwarp();
    };
    // group t's smem buffer: the ring is RSMEM (1 or 2) deep, warp-private slice
    auto buf = [&](int t) { return smem + (RSMEM == 2 ? (t & 1) : 0) * C::STAGE + warp * C::WGRP; };
#pragma unroll
    for (int j = 0; j < RREG; ++j)
        if (j < ng) gload(j, rq[j], rd[j]);
    if (BAT == 0) {
        // the outer loop unrolls by RREG so every slot / buffer index is a call-site constant and the
        // register ring never lands in local memory (a runtime ring index does - r4's 160-B stack lesson)
        for (int base = 0; base < ng; base += RREG)
#pragma unroll
            for (int j = 0; j < RREG; ++j) {
                const int t = base + j;
                if (t >= ng) break;
                step(t, rq[j], rd[j], buf(t));
            }
    } else if (BAT == 1) {
        // r6 BURST-2 (RSMEM >= 2): per PAIR, both STSes first (both buffers live - RSMEM 2 is what makes the
        // pair storeable up front), then ONE 8-wide load burst into the pair's just-freed slots (dp4a's few
        // huge batches: ~4.6 KB of weight bytes per warp in flight at once vs the step's 2.3 KB), then both
        // computes; the burst rides RREG groups of cover before its STS. The unrolled pair slots stay
        // call-site constants (a runtime slot index lands the ring in local memory).
        for (int base = 0; base < ng; base += RREG)
#pragma unroll
            for (int p = 0; p < RREG / 2; ++p) {
                const int t0 = base + 2 * p, t1 = t0 + 1;
                unsigned char *B0 = buf(t0), *B1 = buf(t1);
                if (t0 < ng) gstore(rq[2 * p], rd[2 * p], B0);
                if (t1 < ng) gstore(rq[2 * p + 1], rd[2 * p + 1], B1);
                __syncwarp();
                if (t0 + RREG < ng) gload(t0 + RREG, rq[2 * p], rd[2 * p]);
                if (t1 + RREG < ng) gload(t1 + RREG, rq[2 * p + 1], rd[2 * p + 1]);
                if (t0 < ng) compute(t0, B0);
                if (t1 < ng) compute(t1, B1);
                __syncwarp();
            }
    } else {
        // r6 BURST-4 (RREG 4, RSMEM 2): the block's four STSes, then ONE 16-wide LDG.128 burst (~9.2 KB per
        // warp in flight at once), then the block's four computes - the maximum batch the 4-group register
        // ring holds, a full RREG groups of DRAM-latency cover behind it
        for (int base = 0; base < ng; base += 4) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if (base + j < ng) gstore(rq[j], rd[j], buf(base + j));
            __syncwarp();
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if (base + 4 + j < ng) gload(base + 4 + j, rq[j], rd[j]);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if (base + j < ng) compute(base + j, buf(base + j));
            __syncwarp();
        }
    }
    if (SQ == 0) {
#pragma unroll
        for (int g = 0; g < C::NG; ++g) {
            const int row = row0 + warp * C::WN + 8 * g + 2 * t4;
            if (row >= a.N) continue;
            if (tok >= a.T) continue;
            float2* p = (float2*)(a.y + (size_t)tok * a.ldy + row);
            p->x = F[g][0];
            p->y = F[g][1];
        }
    } else {
        // fp32 y too (dead downstream in the spec engine, but keeps tc and dp4a memory-faithful for A/B checks)
#pragma unroll
        for (int g = 0; g < C::NG; ++g) {
            const int row = row0 + warp * C::WN + 8 * g + 2 * t4;
            if (row >= a.N) continue;
            if (tok >= a.T) continue;
            *(float2*)(a.y + (size_t)tok * a.ldy + row) = make_float2(F[g][0], F[g][1]);
        }
        // silu(gate) * up -> q8 per column (the gate|up tensor's 8-row tiles: 4 gate rows + 4 up rows; the down row
        // for staged gate row grow is ob = row0/2 + (grow >> 1) - ... in group form below). One cross-warp handoff:
        // every warp's F lands in shared, then any warp quantizes any output group.
        __syncthreads();
        float* s_q = (float*)(smem + C::O_SQ);
#pragma unroll
        for (int g = 0; g < C::NG; ++g)
#pragma unroll
            for (int e = 0; e < 2; ++e)
                s_q[(warp * C::WN + 8 * g + 2 * t4 + e) * TPD + tok] = F[g][e];
        __syncthreads();
        const int nog = BR / 64;  // down rows per block = BR/2 (4 gate + 4 up rows per 8-row tile) = nog * 32
        for (int job = warp; job < a.T * nog; job += C::NW) {
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

template <int RPL, int BR, int SQ, int RREG, int RSMEM, int HOIST, int BAT>
static cudaError_t launch_t(const TcArgs& a, cudaStream_t s) {
    if (a.K % 256 || a.T < 1 || a.T > TPD) return cudaErrorInvalidValue;  // K%256: groups of 4 x 64-el stages
    using C = Cfg<BR, SQ, RSMEM>;
    static int attr_mask = 0;  // per device: prefer max smem carveout
    int dev = 0;
    cudaGetDevice(&dev);
    if (!(attr_mask & (1 << dev))) {
        cudaError_t e = cudaFuncSetAttribute(gemv_tc_kernel<RPL, BR, SQ, RREG, RSMEM, HOIST, BAT>,
                                             cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        if (e != cudaSuccess) return e;
        attr_mask |= 1 << dev;
    }
    dim3 grid(1, (a.N + BR - 1) / BR);
    gemv_tc_kernel<RPL, BR, SQ, RREG, RSMEM, HOIST, BAT><<<grid, C::NW * 32, C::SMEM, s>>>(a);
    return cudaGetLastError();
}

// pick BR so the grid fills the machine in one wave (T4: 40 SMs): big N -> 128-row tiles, small N -> 64. The
// engine default ring is <RREG 2, RSMEM 1, hoist 0, bat 0> (v26's best; v27's tc_bench matrix A/Bs the r6
// levers and promotes the winner here once measured).
static inline cudaError_t gemv_tc_launch(int rpl, bool sq, const TcArgs& a, cudaStream_t s) {
    const bool big = a.N >= 56 * 128;
    if (rpl == 2) return sq ? cudaErrorInvalidValue
                           : (big ? launch_t<2, 128, 0, 2, 1, 0, 0>(a, s) : launch_t<2, 64, 0, 2, 1, 0, 0>(a, s));
    if (rpl == 4) return sq ? (big ? launch_t<4, 128, 1, 2, 1, 0, 0>(a, s) : launch_t<4, 64, 1, 2, 1, 0, 0>(a, s))
                            : (big ? launch_t<4, 128, 0, 2, 1, 0, 0>(a, s) : launch_t<4, 64, 0, 2, 1, 0, 0>(a, s));
    return cudaErrorInvalidValue;
}

// bench variant dispatch (the r6 matrix): vreg 2/4, vsmem 1/2, hoist 0/1, bat 0/1/2, br64 forces BR 64 for
// big N (the ragged-grid A/B: BR 128 at N 8192 is 64 blocks over 40 SMs, a 2:1 imbalance). BAT > 0 needs
// RSMEM 2 (the burst stages both/all buffers before compute). Valid combos: <2,1,0,0> (V1/V4/V9's no-hoist
// step), <2,1,1,0> (V5), <2,2,1,1> (V7), <4,2,1,1> (V6), <4,2,1,2> (V8).
static inline cudaError_t gemv_tc_launch_v(int rpl, bool sq, int vreg, int vsmem, int hoist, int bat, bool br64,
                                           const TcArgs& a, cudaStream_t s) {
    const bool big = br64 ? false : (a.N >= 56 * 128);
    const bool v2 = vreg == 2, v4 = vreg == 4, s1 = vsmem == 1, s2 = vsmem == 2, h0 = hoist == 0, h1 = hoist == 1,
              b0 = bat == 0, b1 = bat == 1, b2 = bat == 2;
    if (rpl == 2 && !sq) {
        if (v2 && s1 && h0 && b0)   // V1 (big) / V4 (br64): v26's best
            return big ? launch_t<2, 128, 0, 2, 1, 0, 0>(a, s) : launch_t<2, 64, 0, 2, 1, 0, 0>(a, s);
        if (v2 && s1 && h1 && b0)   // V5 / V9: A-hoist at the 2-group ring
            return big ? launch_t<2, 128, 0, 2, 1, 1, 0>(a, s) : launch_t<2, 64, 0, 2, 1, 1, 0>(a, s);
        if (v2 && s2 && h1 && b1)   // V7: A-hoist + burst-2 at the 2-group ring (12 warps/SM at BR 64)
            return big ? launch_t<2, 128, 0, 2, 2, 1, 1>(a, s) : launch_t<2, 64, 0, 2, 2, 1, 1>(a, s);
        return cudaErrorInvalidValue;
    }
    if (rpl == 4) {
        if (v2 && s1 && h0 && b0)   // V1 / V4 at rpl 4 (sq 0/1)
            return sq ? (big ? launch_t<4, 128, 1, 2, 1, 0, 0>(a, s) : launch_t<4, 64, 1, 2, 1, 0, 0>(a, s))
                      : (big ? launch_t<4, 128, 0, 2, 1, 0, 0>(a, s) : launch_t<4, 64, 0, 2, 1, 0, 0>(a, s));
        if (v2 && s1 && h1 && b0)   // V5 / V9 at rpl 4
            return sq ? (big ? launch_t<4, 128, 1, 2, 1, 1, 0>(a, s) : launch_t<4, 64, 1, 2, 1, 1, 0>(a, s))
                      : (big ? launch_t<4, 128, 0, 2, 1, 1, 0>(a, s) : launch_t<4, 64, 0, 2, 1, 1, 0>(a, s));
        if (v2 && s2 && h1 && b1)   // V7 at rpl 4
            return sq ? (big ? launch_t<4, 128, 1, 2, 2, 1, 1>(a, s) : launch_t<4, 64, 1, 2, 2, 1, 1>(a, s))
                      : (big ? launch_t<4, 128, 0, 2, 2, 1, 1>(a, s) : launch_t<4, 64, 0, 2, 2, 1, 1>(a, s));
        if (v4 && s2 && h1 && b1)   // V6: A-hoist + burst-2 at the 4-group ring (8 warps/SM)
            return sq ? (big ? launch_t<4, 128, 1, 4, 2, 1, 1>(a, s) : launch_t<4, 64, 1, 4, 2, 1, 1>(a, s))
                      : (big ? launch_t<4, 128, 0, 4, 2, 1, 1>(a, s) : launch_t<4, 64, 0, 4, 2, 1, 1>(a, s));
        if (v4 && s2 && h1 && b2)   // V8: A-hoist + burst-4 (16 LDG.128 back-to-back, ~9.2 KB/warp in flight)
            return sq ? (big ? launch_t<4, 128, 1, 4, 2, 1, 2>(a, s) : launch_t<4, 64, 1, 4, 2, 1, 2>(a, s))
                      : (big ? launch_t<4, 128, 0, 4, 2, 1, 2>(a, s) : launch_t<4, 64, 0, 4, 2, 1, 2>(a, s));
        return cudaErrorInvalidValue;
    }
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
