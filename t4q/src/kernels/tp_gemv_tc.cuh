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
//   * BAT (v 6/7/8): the per-step 4x LDG.128 W batch becomes a PAIR burst - the block's 2 STSes go first
//     (RSMEM 2), then ONE burst of 8 consecutive LDG.128 + d loads into the just-freed slots, then the
//     block's computes; the burst rides RREG groups of cover before its STS (dp4a's shape: ~4.6 KB of weight
//     bytes issued at once vs the step's 2.3 KB). Bursts drop gload's L2 prefetch (hardware MLP covers DRAM
//     latency once 8-16 LDGs are outstanding; the 8 prefetch ops per group only doubled the LSU pressure).
//     The 4 d u16s stay separate loads: the FAST_P4 d words interleave the RPL planes per 16-row group
//     (sa = d + (g*RPL + r)*2 B), so odd-plane rows are 2 mod 4 aligned and no wider d window exists (v28
//     fused them into 16-B windows and the whole matrix FATALed with cudaErrorMisalignedAddress).
// v27 measured: burst-2 (V7: HOIST + RREG 2/RSMEM 2) took down_tp 127 -> 168.8 GB/s (M-flat), BEATING dp4a at
// M 7/8 (168.4 vs 158.6, 164.0 vs 138.3) - the first shape fully beaten; the quad burst (BAT 2) is WRONG
// (it stages four smem buffers at once but RSMEM 2 gives two, so groups base and base+2 collide) and is dropped;
// ncu is unusable on Kaggle (ERR_NVGPUCTRPERM, counters admin-only). r7 (v28+): the remaining M 4 gap (168.5 vs
// dp4a's 231 on down) is the A-side latency that survives the r6 hoist - the hoist issues the group's 16 q8 + 4
// xms words at the TOP of compute but stage 0's fa build consumes them IMMEDIATELY, so every group still eats one
// ~270+ cy A-load stall on its first mma. AR (v 4/5): the A-side moves one GROUP early - compute(t) issues group
// t+1's A words into the ping-pong A buffer (t+1)&1 at its top (aq/axd ring 2 deep, compile-time buffer indices:
// groups are even/odd by construction in every schedule), a full group of W work of cover.
// r8 (v31+): the SASS of the star showed a healthy burst/ldsm/mma interleave, but the RSMEM-2 ring needed
// 40 KB of smem at BR 128 = 8 warps/SM (and 20 KB = 12 at BR 64) - dp4a holds NO smem at all and its 250+ GB/s
// rides 16-20 warps/SM. The d words were the smem parasite: they now ride a 4-deep REGISTER ring (gload packs
// each lane's two stages' d pairs into one u32 per stage; compute shfls the row's pair from the owning lane - one
// shfl per pair, warp-synchronous, no barrier), STAGE loses its d area (20 -> 16 KB at BR 128, 10 -> 8 at BR 64),
// which lifts the star to 16 warps/SM at both BRs: dp4a's occupancy with dp4a's batch sizes. The d-ring is why
// RREG is pinned to 2 (the live d set {t, t+1, t+RREG, t+RREG+1} needs 4 slots; RREG 4's refill collides) and
// why the schedules unroll their outer loops by 4 (the slot indices must be call-site literals or the ring
// lands in local memory - r4's lesson).
// v31 measured the r8 kernel clean (min-window timing, download joined first): the star/AR64 reached
// 16-warp occupancy parity and WON gateup at M7/8 (154 vs 149/119), down at M8 (134 vs 127), within 1.45x
// of dp4a on qkvz at M8 - but every shape still lost at M 2-4, and the diagnosis is structural: TC's per-launch
// time is M-INDEPENDENT (the mma m-tile covers all 8 token slots and the A loads fetch them all), while dp4a
// scales with M. r9 gates the A loads on tok < a.T (quad-uniform, the stale quads skip their ~24 of ~30 LSU
// ops per group and zero their registers - the epilogue discards those slots anyway, and the ungated loads
// read past a.T-token xq/xms buffers, a latent OOB); the mma itself stays m8 (the B-fragment pattern needs all
// 32 lanes) but it is <2% of the time, so the tensor-pipe waste is free. The engine default is promoted to
// v31's per-shape winners (gemv_tc_launch below).
// r10 (v33+): v32's engine trace (per-kernel CUPTI) explained the rest of the gap: the verify's TC launches
// run at the matrix's per-launch times (qkvz 167.9 vs bench 165.7 us) under a uniform ~1.28x sustained-clock
// inflation, and GB/s across shapes tracks the grid's TOTAL warps = N/16 (attn N 4096: 256 warps = 6.4/SM ->
// 102 GB/s; down N 5120: 320 = 8/SM -> 216; gateup 1088 capped) - the small-N shapes are warp-starved, not
// latency- or tensor-bound. WN 8: each warp owns ONE mma n-tile (8 rows, the lane QUAD loader - one stage, 2
// code int4s + 1 packed d u32 per lane per group, ldsm_x2 for the two k32 halves), so the total warps double
// (N/8) with STAGE unchanged and every row still warp-private (no cross-warp reduction). The star8 lands at
// 80-95 regs (3 blocks = 24 warps/SM at BR 64); the AR8's ring keeps 127-128 regs (2 blocks = 16) - its BR 64
// launch-bounds target is 2 blocks (3 spills the A-ring, 48 B of local).
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

template <int BR, int SQ, int RSMEM, int ROWSW>
struct Cfg {
    static_assert(ROWSW == 8 || ROWSW == 16, "r10: 8 (one mma n-tile) or 16 (two) rows per warp");
    static constexpr int WN = ROWSW;      // rows per warp: 8 (one mma n-tile, r10) or 16 (two)
    static constexpr int NW = BR / WN;   // warps per block: WN 16: 4/8 (128/256 threads at BR 64/128);
                                         // WN 8: 8/16 (256/512) - SAME smem, double the warps per block
    static constexpr int NG = WN / 8;    // mma n-tiles per warp: 2 (16 rows) or 1 (8 rows)
    // per-warp GROUP buffer (4 stages of 64 el): [4 stages x WN x 32 B codes]. The d words ride NO smem (r8):
    // gload packs each lane's two stages' d pairs into ONE u32 per stage of a 4-deep REGISTER ring, and
    // compute shfls the row's pair from the owning lane pair (warp-synchronous, no barrier) - STAGE loses
    // the d area, 20 -> 16 KB at BR 128 / 10 -> 8 KB at BR 64, lifting the RSMEM-2 ring from 8 to 16
    // warps/SM (BR 128) and the BR 64 star from 12 to 16: dp4a's occupancy with dp4a's batch sizes.
    // r10 WN 8: the warp owns ONE n-tile (8 rows) - the group buffer halves per warp (1 KB) while the block
    // doubles its warp count, so STAGE is UNCHANGED and the grid's TOTAL warps double (N/16 -> N/8): the
    // v32 engine trace showed GB/s tracking total warps (attn N 4096: 256 warps = 6.4/SM -> 102 GB/s; down
    // N 5120: 320 = 8/SM -> 216), and the small-N shapes are the engine's laggards (attn/out 0.5x dp4a).
    static constexpr int WGRP = 4 * WN * 32;
    static constexpr int STAGE = NW * WGRP;         // ONE smem ring buffer (a group); the ring is RSMEM deep
    static constexpr int O_SQ = RSMEM * STAGE;      // SQ: [BR][TPD] fp32 outputs staged by row
    static constexpr int SMEM = O_SQ + (SQ ? BR * TPD * 4 : 0);
    // r8 <*,2>: BR 128 non-sq 32 KB -> 2 blocks = 16 warps/SM (was 8), BR 64 non-sq 16 KB -> 4 blocks
    // (122-reg star) = 16 warps (was 12), BR 64 sq 20 KB -> 3 blocks = 12; <*,1>: 16/8 KB -> 4/8 blocks.
    // r10 WN 8: 256 threads at BR 64 - the per-thread reg budget halves (fewer F/fb/rq regs, the A-ring
    // unchanged), so the star8 should hold 2 blocks = 16 warps/SM while serving 2x the total warps.
    static constexpr int RQW = WN == 8 ? 2 : 4;  // code int4s per lane per group (WN 8: 1 stage = 2 units)
    static constexpr int RDW = WN == 8 ? 1 : 2;  // d pairs packed per lane per group (1 u32 each)
};

template <int RPL, int BR, int SQ, int RREG, int RSMEM, int HOIST, int BAT, int AR, int WN = 16>
__global__ void __launch_bounds__(BR / WN * 32,
                                  HOIST ? (RREG == 2 ? (BR == 64 ? (WN == 8 ? 2 : 3) : 1) : (BR == 64 ? 2 : 1))
                                        : (RREG == 2 ? (BR == 64 ? 4 : 2) : (BR == 64 ? 2 : 1)))
gemv_tc_kernel(const TcArgs a) {
    using C = Cfg<BR, SQ, RSMEM, WN>;
    static_assert(RSMEM >= 2 || BAT == 0, "burst schedules store both pair buffers before compute");
    static_assert(BAT == 0 || BAT == 1,
                  "burst-4 was dropped in r7: RSMEM 2 holds two live buffers, groups base and base+2 collide");
    static_assert(AR == 0 || HOIST == 1, "the A-ring rides the hoist's pinned A-side loads");
    static_assert(RREG == 2,
                  "r8's register d-ring is 4 deep (the live set {t, t+1, t+RREG, t+RREG+1}): RREG 4's refill of "
                  "d-slot g%4 collides with compute(g)'s pending read - V8's buffer collision, twice over");
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int row0 = blockIdx.y * BR;
    const int K = a.K, nst = K >> 6, ng = nst >> 2, K32 = K >> 5;
    const int t4 = lane & 3, tok = lane >> 2;
    // r9: only the first a.T token quads do A-side work - the mma m-tile is fixed at 8 slots (the B-fragment
    // pattern needs all 32 lanes), but the STALE slots' A loads (16 q8 words + 4 xms words per lane per group,
    // ~24 of the warp's ~30 LSU ops!) are pure waste at the engine's real M (2-6): their outputs are discarded
    // by the epilogue anyway, and ungated they even read past a.T-token xq/xms buffers. The gate is
    // quad-uniform (tok = lane >> 2), so the warp skips whole LDGs, and the stale registers zero out so the
    // discarded F slots stay NaN-free (the int mma path is bounded either way).
    const bool tlive = tok < a.T;

    // this warp owns WN weight rows (r10: 8 or 16 = one or two mma n-tiles); the row's 8 group code units
    // (128 B, one plane-half unit run) live in the lane team: WN 16: the lane PAIR (lane >> 1 = row,
    // lane & 1 = wh), wh 0 holds the group's stages 0-1 (blocks 8gi..8gi+3), wh 1 stages 2-3, each lane's
    // 4 units one CONTIGUOUS 64-B run of the row's plane-half; WN 8: the lane QUAD (lane >> 2 = row,
    // lane & 3 = qh), qh owning exactly ONE stage (blocks 8gi+2qh..8gi+2qh+1, 32 B contiguous) - the
    // group's W bytes spread over ALL 32 lanes either way.
    const int wrow = warp * C::WN + (WN == 8 ? (lane >> 2) : (lane >> 1));  // local row in the tile
    const int wrow_g = min(row0 + wrow, a.N - 1);  // clamped global row for the loads (padded tile tail)
    const int wh = WN == 8 ? (lane & 3) : (lane & 1);  // WN 8: the lane's stage; WN 16: its stage pair
    const gemm8::WPtr<FAST_P4, RPL> P(a.q, wrow_g, 0);  // blk 0: P.code / P.sa sit at the row's plane-half base
    const int8_t* xtok = (const int8_t*)a.xq + (size_t)tok * K;  // own token's q8 row (tok >= T: stale, discarded)

    float F[C::NG][2];
#pragma unroll
    for (int g = 0; g < C::NG; ++g) F[g][0] = F[g][1] = 0.f;

    // W register rings (the dp4a chunk-ring idea, r4/r8): group gi = blocks 8gi..8gi+7 = units
    // jj0..jj0+7 of the row's plane-half, chunk c = gi >> 1, jj0 = (gi & 1) << 3. The lane's code units are
    // (jj0 + wh*4 + k), k = 0..RQW-1: WN 16: 64 B contiguous (2 stages); WN 8: 32 B (1 stage, wh = the
    // stage). Group gi rides code slot gi % RREG (the smem STS frees it a refill early) and d slot gi % 4 of
    // the REGISTER d-ring (r8: d never touches smem - the codes' STS->ldsm handoff frees rq early, but d
    // lives ONLY in registers, so its slot must outlive the refill horizon: at the pair's burst the live d set
    // is {t, t+1, t+RREG, t+RREG+1} - four groups, hence depth 4 and RREG 2 only (RREG 4's refill of slot
    // t%4 collides with compute(t)'s pending read, V8's lesson twice over). Both ring indices are call-site
    // constants (the schedules unroll by 4), so the arrays stay in registers (a runtime ring index lands the
    // whole set in local memory - the first r4 draft's 160-B stack).
    int4 rq[RREG][C::RQW];
    uint32_t rd_p[4][C::RDW];  // [d slot = group & 3][pair]: (d(2st) | d(2st+1) << 16) per stage of the row
    auto gload = [&](int gi, int4 (&rqs)[C::RQW], uint32_t (&rds)[C::RDW]) {
        const int c = gi >> 1, j = ((gi & 1) << 3) + (WN == 8 ? 2 * wh : 4 * wh);
#pragma unroll
        for (int k = 0; k < C::RQW; ++k) {
            rqs[k] = gemv::ldg_nc_v4(P.code + c * P.cs_code + (j + k) * 16);
        }
        // the lane's d u16s pair up per stage (the pair's two blocks of the same row), each pair packed
        // into ONE u32 register word - the shfl consumer (r8) fetches the pair in a single shfl_sync
#pragma unroll
        for (int u = 0; u < C::RDW; ++u)
            rds[u] = (uint32_t)__ldg((const unsigned short*)(P.sa + c * P.cs_s + (j + 2 * u) * RPL * 2)) |
                     ((uint32_t)__ldg((const unsigned short*)(P.sa + c * P.cs_s + (j + 2 * u + 1) * RPL * 2)) << 16);
        if (BAT == 0 && gi + RREG + 2 < ng) {  // L2 prefetch beyond the ring horizon (BAT 0 only: with 8 LDGs
            // outstanding the hardware MLP covers DRAM latency and the prefetches only add LSU pressure)
            const int gp = gi + RREG + 2, c2 = gp >> 1, j2 = ((gp & 1) << 3) + (WN == 8 ? 2 * wh : 4 * wh);
#pragma unroll
            for (int k = 0; k < C::RQW; ++k) {
                gemm8::prefetch_l2(P.code + c2 * P.cs_code + (j2 + k) * 16);
                gemm8::prefetch_l2(P.sa + c2 * P.cs_s + (j2 + k) * RPL * 2);
            }
        }
    };
    auto gstore = [&](const int4 (&rqs)[C::RQW], unsigned char* B) {
        const int row = WN == 8 ? (lane >> 2) : (lane >> 1);
        // unit (j + k) = block 8gi + (WN 16: wh*4 / WN 8: 2*wh) + k: stage sloc = (WN 16: 2*wh + (k >> 1) /
        // WN 8: wh), code 16 B per row per stage at wswz(row, k & 1) - CODES ONLY (r8): the d pairs stay in
        // the register d-ring until the compute-side shfl, so the smem group buffer loses its d area
#pragma unroll
        for (int k = 0; k < C::RQW; ++k)
            *(int4*)(B + (WN == 8 ? wh : 2 * wh + (k >> 1)) * (C::WN * 32) + row * 32 + wswz(row, k & 1) * 16) =
                flip4(rqs[k]);
    };
    // A-side (HOIST, r6/r7): the group's 16 q8 words (the lane's lo/hi nibble pair of each stage's 2 blocks) +
    // 4 xms words as one back-to-back pinned-volatile batch (ptxas cannot sink them into the stage loop). The
    // words live in a 2-deep ping-pong A ring indexed by group parity - every schedule visits groups in
    // ascending pairs (base, base+1), so both buffer indices are call-site constants: AR 0 (r6) loads THIS
    // group's words at the top of compute(gi) (stage 0's fa build still eats their L2 latency once), AR 1
    // (r7) loads group gi+1's words instead, one full group of W work of cover before compute(gi+1) reads
    // them - the surviving ~270+ cy first-mma stall of the r6 hoist
    uint32_t aq[2][4][4];  // [ring buffer = group & 1][stage][2*bb + half]: q8 word (half 0 = lo4 base, 1 = +16)
    int4 axd[2][4];        // [ring buffer][stage]: xms words (d(2st) pair .x/.y, d(2st+1) pair .z/.w)
    auto aload = [&](int gi, uint32_t (&aqb)[4][4], int4 (&axb)[4]) {
        if (!tlive) {  // r9: stale token quad - zero the words, skip the loads (the discarded slots' F stays 0)
#pragma unroll
            for (int s = 0; s < 4; ++s) {
#pragma unroll
                for (int w = 0; w < 4; ++w) aqb[s][w] = 0;
                axb[s] = make_int4(0, 0, 0, 0);
            }
            return;
        }
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int st = gi * 4 + s;
            axb[s] = gemv::ldg_nc_v4((const int4*)(a.xms + (size_t)tok * K32 + st * 2));
#pragma unroll
            for (int bb = 0; bb < 2; ++bb) {
                aqb[s][2 * bb] = gemv::ldg_nc_u32(xtok + st * 64 + bb * 32 + t4 * 4);
                aqb[s][2 * bb + 1] = gemv::ldg_nc_u32(xtok + st * 64 + bb * 32 + t4 * 4 + 16);
            }
        }
    };
    auto compute = [&](int gi, const unsigned char* Bp, uint32_t (&aqg)[4][4], int4 (&axg)[4],
                       uint32_t (&aqn)[4][4], int4 (&axn)[4], uint32_t (&rdc)[C::RDW]) {
        if (HOIST) {
            if (AR) {
                if (gi + 1 < ng)
                    aload(gi + 1, aqn, axn);  // r7: NEXT group's A words NOW - a full group of cover
            } else {
                aload(gi, aqg, axg);  // r6: this group's words (stage 0 eats their latency once)
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
                const uint32_t lo4 =
                    HOIST ? aqg[s][2 * bb] : (tlive ? *(const uint32_t*)(xtok + st * 64 + bb * 32 + t4 * 4) : 0u);
                const uint32_t hi4 = HOIST ? aqg[s][2 * bb + 1]
                                            : (tlive ? *(const uint32_t*)(xtok + st * 64 + bb * 32 + t4 * 4 + 16)
                                                     : 0u);
                fa[0][bb] = (lo4 & 0x0F0F0F0Fu) | ((hi4 & 0x0F0F0F0Fu) << 4);
                fa[1][bb] = ((lo4 >> 4) & 0x0F0F0F0Fu) | (hi4 & 0xF0F0F0F0u);
            }
            // the warp's B fragments: WN 16: both bb x g matrices in ONE ldsm_x4 (r3: two ldsm_x2), lanes
            // 8m..8m+7 addressing matrix m = 2*bb + g over (rows 8g..8g+7, unit wswz(row, bb)); WN 8: the
            // single n-tile's two bb halves in an ldsm_x2 - lanes 0-7 address (row = lane, bb 0), lanes
            // 8-15 (row = lane-8, bb 1); the fragment regs are fb[bb][g] either way (mma: fb[0][g], fb[1][g])
            uint32_t fb[2][C::NG];
            if constexpr (WN == 16) {
                const int m = lane >> 3, r = 8 * (m & 1) + (lane & 7), bb = m >> 1;
                gemm8::ldsm_x4(fb[0][0], fb[0][1], fb[1][0], fb[1][1],
                               Bp + s * (C::WN * 32) + r * 32 + wswz(r, bb) * 16);
            } else {
                // lanes 16-31's addresses are ignored by the x2 but still clamped to the row's own 32 B so
                // no architecture ever wanders off the warp's group buffer
                const int r = lane & 7, bb = lane < 16 ? (lane >> 3) : (lane & 1);
                gemm8::ldsm_x2(fb[0][0], fb[1][0], Bp + s * (C::WN * 32) + r * 32 + wswz(r, bb) * 16);
            }
            // activation d of this token at the stage's two k32 blocks (one 16-B load: the index is 16-B
            // aligned); r9: the stale token quads read 0 (gated, no OOB past a.T-token xms; the hoisted path
            // zeroes in aload already)
            const int4 m2z = make_int4(0, 0, 0, 0);
            const int4 m2 = HOIST ? axg[s] : (tlive ? __ldg((const int4*)(a.xms + (size_t)tok * K32 + st * 2)) : m2z);
            const float xd0 = __int_as_float(m2.x), xd1 = __int_as_float(m2.z);
#pragma unroll
            for (int g = 0; g < C::NG; ++g) {
                float2 ws[2];  // per row: {d of the stage's even k32, d of the odd k32} (gemm20's layout, packed)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    // r8: the row's d pair rides the LOADER lane's register d-ring word - ONE shfl per pair,
                    // warp-synchronous, no smem round-trip, no barrier. WN 16: the loader lane PAIR owns the
                    // row (lane 2*row + (s >> 1) packed its stage-(s & 1) pair); r10 WN 8: the loader lane
                    // QUAD owns it (lane 4*row + s packed exactly stage s's pair)
                    uint32_t dp;
                    if constexpr (WN == 8)
                        dp = __shfl_sync(0xffffffffu, rdc[0], 4 * (8 * g + 2 * t4 + e) + s);
                    else
                        dp = __shfl_sync(0xffffffffu, rdc[s & 1], 2 * (8 * g + 2 * t4 + e) + (s >> 1));
                    ws[e] = make_float2(gemm8::h2f(dp & 0xffffu), gemm8::h2f(dp >> 16));
                }
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
    // just-consumed slots with G_{t+RREG} -> compute(G_t) -> syncwarp. The opening syncwarp makes the STS visible
    // to the whole warp's ldsm; the closing one orders the step's ldsm reads before any later STS rewrites the
    // buffer (the single-buffer RSMEM=1 case needs it, the 2-buffer case gets it free). r8: the step loop
    // unrolls by FOUR (not RREG) so the d-ring slot t%4 is the literal j - group t's d must outlive the
    // gload(t+RREG) refill two steps away, and the codes' smem handoff does not cover the register-only d.
    auto step = [&](int t, int4 (&rqs)[C::RQW], uint32_t (&rdc)[C::RDW], uint32_t (&rds)[C::RDW], unsigned char* B,
                    uint32_t (&aqc)[4][4], int4 (&axc)[4], uint32_t (&aqn2)[4][4], int4 (&axn2)[4]) {
        gstore(rqs, B);
        __syncwarp();
        if (t + RREG < ng) gload(t + RREG, rqs, rds);
        compute(t, B, aqc, axc, aqn2, axn2, rdc);
        __syncwarp();
    };
    // group t's smem buffer: the ring is RSMEM (1 or 2) deep, warp-private slice
    auto buf = [&](int t) { return smem + (RSMEM == 2 ? (t & 1) : 0) * C::STAGE + warp * C::WGRP; };
    if (HOIST && AR && ng > 0) aload(0, aq[0], axd[0]);  // r7 A-ring prime: group 0's words (compute(0) issues
    // group 1's into buffer 1 - the AR branch's gi+1 load needs its buffer pre-primed with THIS group)
#pragma unroll
    for (int j = 0; j < RREG; ++j)
        if (j < ng) gload(j, rq[j], rd_p[j]);
    if (BAT == 0) {
        // the outer loop unrolls by 4 so every slot / buffer index is a call-site constant and the register
        // rings never land in local memory (a runtime ring index does - r4's 160-B stack lesson)
        for (int base = 0; base < ng; base += 4)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int t = base + j;
                if (t >= ng) break;
                step(t, rq[j & 1], rd_p[j], rd_p[j ^ 2], buf(t), aq[j & 1], axd[j & 1], aq[(j + 1) & 1],
                     axd[(j + 1) & 1]);
            }
    } else {
        // r6 BURST-2 (RSMEM >= 2): per PAIR, both STSes first (both buffers live - RSMEM 2 is what makes the
        // pair storeable up front), then ONE 8-wide load burst into the pair's just-freed slots (dp4a's few
        // huge batches: ~4.6 KB of weight bytes per warp in flight at once vs the step's 2.3 KB), then both
        // computes; the burst rides RREG groups of cover before its STS. r8: the outer loop unrolls by TWO
        // PAIRS (base += 4, p in 0..1) so the d-ring slots (2p, 2p+1 for the computes, 2p^2 / (2p+1)^2 for
        // the refills) are literals; the 4-wide quad (BAT 2) is structurally wrong at RSMEM 2 and dropped
        // (v27), and RREG 4 is dropped with it (its refill would collide with the live d of compute(t)).
        for (int base = 0; base < ng; base += 4)
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int t0 = base + 2 * p, t1 = t0 + 1;  // rq slots: (base+2p) % RREG 2 = 0 / (base+2p+1) % 2 = 1
                unsigned char *B0 = buf(t0), *B1 = buf(t1);
                if (t0 < ng) gstore(rq[0], B0);
                if (t1 < ng) gstore(rq[1], B1);
                __syncwarp();
                if (t0 + RREG < ng) gload(t0 + RREG, rq[0], rd_p[2 * p ^ 2]);
                if (t1 + RREG < ng) gload(t1 + RREG, rq[1], rd_p[(2 * p + 1) ^ 2]);
                if (t0 < ng) compute(t0, B0, aq[0], axd[0], aq[1], axd[1], rd_p[2 * p]);
                if (t1 < ng) compute(t1, B1, aq[1], axd[1], aq[0], axd[0], rd_p[2 * p + 1]);
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

template <int RPL, int BR, int SQ, int RREG, int RSMEM, int HOIST, int BAT, int AR, int WN = 16>
static cudaError_t launch_t(const TcArgs& a, cudaStream_t s) {
    if (a.K % 256 || a.T < 1 || a.T > TPD) return cudaErrorInvalidValue;  // K%256: groups of 4 x 64-el stages
    using C = Cfg<BR, SQ, RSMEM, WN>;
    static int attr_mask = 0;  // per device: prefer max smem carveout
    int dev = 0;
    cudaGetDevice(&dev);
    if (!(attr_mask & (1 << dev))) {
        cudaError_t e = cudaFuncSetAttribute(gemv_tc_kernel<RPL, BR, SQ, RREG, RSMEM, HOIST, BAT, AR, WN>,
                                             cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        if (e != cudaSuccess) return e;
        attr_mask |= 1 << dev;
    }
    dim3 grid(1, (a.N + BR - 1) / BR);
    gemv_tc_kernel<RPL, BR, SQ, RREG, RSMEM, HOIST, BAT, AR, WN><<<grid, C::NW * 32, C::SMEM, s>>>(a);
    return cudaGetLastError();
}

// engine default = the measured per-shape winners, all at BR 64 (r8 cut the star's smem to 16 KB -> 4 blocks
// = 16 warps/SM; the finer grid tails better than BR 128's 2-block 16): rpl 4 non-sq (down/out) the star
// <4,64,0,2,2,1,1,0> (down 217 GB/s at M4, +53% over ctrl); rpl 4 sq (gateup) the AR ring <4,64,1,2,2,1,1,1>
// (156-186, beats dp4a at M7/8); rpl 2 non-sq: the AR ring <2,64,0,2,2,1,1,1> for the BIG N (qkvz/clamp,
// N >= 8192: v36 has AR64 > star64 at EVERY M, +23..+55% at M2 where the r9 T-gate makes the A-side the
// largest fraction - the AR ring covers exactly that), the star <2,64,0,2,2,1,1,0> for the small N (attn
// N 4096: the star's M2 131 vs the AR's 120, ties at M4-8). r10's WN 8 (one n-tile per warp, 2x the total
// warps) is REFUTED by the same matrix: it lost ~25% on every shape (the per-warp in-flight bytes halved
// while the fixed per-group costs - the syncwarp pair, the AR loads, the ldsm setup - doubled per byte; the
// warp count was never the constraint: GB/s tracks the in-flight bytes per SM, not the warps).
static inline cudaError_t gemv_tc_launch(int rpl, bool sq, const TcArgs& a, cudaStream_t s) {
    if (rpl == 2)
        return sq ? cudaErrorInvalidValue
                  : (a.N >= 8192 ? launch_t<2, 64, 0, 2, 2, 1, 1, 1>(a, s) : launch_t<2, 64, 0, 2, 2, 1, 1, 0>(a, s));
    if (rpl == 4)
        return sq ? launch_t<4, 64, 1, 2, 2, 1, 1, 1>(a, s) : launch_t<4, 64, 0, 2, 2, 1, 1, 0>(a, s);
    return cudaErrorInvalidValue;
}

// bench variant dispatch (the r8 matrix + r10): vreg 2 (the register d-ring is RREG 2 only), vsmem 1/2,
// hoist 0/1, bat 0/1, ar 0/1, wn 8/16 (r10: one n-tile per warp, double the total warps), br64 forces BR 64
// for big N (the ragged-grid A/B: BR 128 at N 8192 is 64 blocks over 40 SMs, a 2:1 imbalance). BAT 1 needs
// RSMEM 2 (the burst stages both buffers before compute). Valid combos: <2,1,0,0,0> (V0/V1: control step),
// <2,2,1,1,0> (V2/V3: the burst-2 star), <2,2,1,1,1> (V4/V5: + the r7 A-ring), and the r10 WN 8 twins V6
// (the star8) / V7 (the AR8). r8 dropped the 4-group ring (star4): its refill collides with the live d.
static inline cudaError_t gemv_tc_launch_v(int rpl, bool sq, int vreg, int vsmem, int hoist, int bat, int ar,
                                           bool br64, int wn, const TcArgs& a, cudaStream_t s) {
    const bool big = br64 ? false : (a.N >= 56 * 128);
    const bool v2 = vreg == 2, s1 = vsmem == 1, s2 = vsmem == 2, h0 = hoist == 0, h1 = hoist == 1,
              b0 = bat == 0, b1 = bat == 1, a0 = ar == 0, a1 = ar == 1, w8 = wn == 8;
    if (w8) {
        // r10: the WN 8 variants run at BR 64 ONLY (the engine's promoted BR): the big-N ternary would
        // instantiate both BRs of every combo and v33's tc_bench compile (30 gemv_tc instantiations) OOM'd
        // the 13 GB Kaggle worker mid-matrix - the dedicated branch halves the added count
        if (v2 && s2 && h1 && b1 && a0) {
            if (rpl == 2)
                return sq ? cudaErrorInvalidValue : launch_t<2, 64, 0, 2, 2, 1, 1, 0, 8>(a, s);
            if (rpl == 4)
                return sq ? launch_t<4, 64, 1, 2, 2, 1, 1, 0, 8>(a, s)
                          : launch_t<4, 64, 0, 2, 2, 1, 1, 0, 8>(a, s);
        }
        if (v2 && s2 && h1 && b1 && a1) {
            if (rpl == 2)
                return sq ? cudaErrorInvalidValue : launch_t<2, 64, 0, 2, 2, 1, 1, 1, 8>(a, s);
            if (rpl == 4)
                return sq ? launch_t<4, 64, 1, 2, 2, 1, 1, 1, 8>(a, s)
                          : launch_t<4, 64, 0, 2, 2, 1, 1, 1, 8>(a, s);
        }
        return cudaErrorInvalidValue;
    }
    if (rpl == 2 && !sq) {
        if (v2 && s1 && h0 && b0 && a0 && !w8)  // V0/V1: control (per-step W issue, per-stage A reads)
            return big ? launch_t<2, 128, 0, 2, 1, 0, 0, 0>(a, s) : launch_t<2, 64, 0, 2, 1, 0, 0, 0>(a, s);
        if (v2 && s2 && h1 && b1 && a0 && !w8)  // V2/V3: the burst-2 star, r8 smem cut (16 warps/SM at BR 64)
            return big ? launch_t<2, 128, 0, 2, 2, 1, 1, 0>(a, s) : launch_t<2, 64, 0, 2, 2, 1, 1, 0>(a, s);
        if (v2 && s2 && h1 && b1 && a1 && !w8)  // V4/V5: the star + r7 A-ring (12 warps/SM at BR 64, 166 regs)
            return big ? launch_t<2, 128, 0, 2, 2, 1, 1, 1>(a, s) : launch_t<2, 64, 0, 2, 2, 1, 1, 1>(a, s);
        return cudaErrorInvalidValue;
    }
    if (rpl == 4) {
        if (v2 && s1 && h0 && b0 && a0 && !w8)  // V0/V1 at rpl 4 (sq 0/1)
            return sq ? (big ? launch_t<4, 128, 1, 2, 1, 0, 0, 0>(a, s) : launch_t<4, 64, 1, 2, 1, 0, 0, 0>(a, s))
                      : (big ? launch_t<4, 128, 0, 2, 1, 0, 0, 0>(a, s) : launch_t<4, 64, 0, 2, 1, 0, 0, 0>(a, s));
        if (v2 && s2 && h1 && b1 && a0 && !w8)  // V2/V3 at rpl 4
            return sq ? (big ? launch_t<4, 128, 1, 2, 2, 1, 1, 0>(a, s) : launch_t<4, 64, 1, 2, 2, 1, 1, 0>(a, s))
                      : (big ? launch_t<4, 128, 0, 2, 2, 1, 1, 0>(a, s) : launch_t<4, 64, 0, 2, 2, 1, 1, 0>(a, s));
        if (v2 && s2 && h1 && b1 && a1 && !w8)  // V4/V5 at rpl 4
            return sq ? (big ? launch_t<4, 128, 1, 2, 2, 1, 1, 1>(a, s) : launch_t<4, 64, 1, 2, 2, 1, 1, 1>(a, s))
                      : (big ? launch_t<4, 128, 0, 2, 2, 1, 1, 1>(a, s) : launch_t<4, 64, 0, 2, 2, 1, 1, 1>(a, s));
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
