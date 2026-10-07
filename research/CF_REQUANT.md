# CF_REQUANT: the expert requant to the iq1_s class (the full-residency stretch, census-unlocked)

The frozen spec for the cf-m6 requant stretch (PLAN_CF section 5). The unlock is the r19n
census verdict: the router is FLAT (168,938 of ~168,960 (layer,expert) slots touched in
353 steps, the top-64 carries 10.7%) - the tiering does not pay, fewer expert bytes is the
only route to full residency, and full residency kills the engine's dominant cost class
(the union expert reads, ~112 ms full-UVA / ~155 ms RAM-capped split per verify round,
~114 ms per greedy step). Every structural fact below is from the primary sources fetched
in r19ac/r19ad (ggml-common.h, ggml-quants.c, ggml-cuda/vecdotq.cuh @ ggml-org/llama.cpp
master); every rate number is the honest roof math, measured at the L4, never assumed.

## 1. The frame and the pick

The expert pool (from the in-repo shapes): gu 80.5 G elems (512 experts/layer x 48 layers
x [1280, 2560]) + dn 40.2 G (512 x 48 x [2560, 640]) = 120.7 G elems. The 2x T4 room after
the ~2.6 GB core, the KV, the draft, the scratch is ~24-25 GB. The stock candidates
(r19ac): iq1_s 1.5625 bpw -> 23.6 GB (the ONLY stock fit); iq1_m 1.75 -> 26.4 (over);
iq2_xxs 2.0625 -> 31.1 (well over); no gu/dn hybrid fits (28.6 / 26.1). THE PICK: BOTH
expert tensors at the iq1_s structure - the stock, battle-tested 2048-entry ternary
lattice, not a custom form. The named fallback if the L4 quality gate fails: the 2-grid
custom (~1.69 bpw, 25.5 GB, +1 lattice-select bit per 8-elem group) - it needs the MEASURED
VRAM inventory (25.5 GB only fits the aggressive end of the room), so it is a fallback,
not the pick. Per expert: gu 0.64 MB + dn 0.32 MB = 0.96 MB; per greedy step 48 x 10 = 480
expert tensors = 461 MB of reads, 2.36 G elem-dots.

## 2. The format (the exact arithmetic, ggml-common.h verbatim facts)

block_iq1_s, 50 B / 256 elems = 1.5625 bpw: `ggml_half d; uint8_t qs[32]; uint16_t
qh[8]`. The planes for PackedW (packed.h): `codes` = qs (32 B/block), `hi` = qh (16
B/block), `d` = the f16 (2 B/block) - a clean FMT_IQ1S, the same planar SoA the K2/K4
formats already use. Constants: QI1_S = 8 (int32s of qs per block), QR1_S = 8.

The value of element j (in 32-elem group ib, grid byte k = ib*4 + j/8, nibble n = j%8):

```
L     = the nibble at iq1s_grid_gpu[ qs[k] | ((qh[ib] >> 3*k) & 7) << 8 ]   // in {0,1,2}
delta = (qh[ib] & 0x8000) ? -0.125 : +0.125          // IQ1S_DELTA = 0.125
scale = h2f(d) * (2*((qh[ib] >> 12) & 7) + 1)         // the per-32 multiplier
w     = scale * ( (L-1) + delta )                    // the lattice value in {-1,0,+1}
```

The GPU grid is the REPACKED nibble form (NOT the CPU byte form): `iq1s_grid_gpu` =
uint32[2048], each entry 8 nibbles = 8 L-indices of one 8-elem group; the CPU reference
grid (uint64[2048], bytes {0xff, 0x00, 0x01} = L-1) is packer-side only. The stock table
values are data (the lattice), the same interop class as the Q2_K/Q4_K block formats the
repo already loads from the GGUF; the kernels and the quantizer are this repo's own code.

## 3. The decode (the device form and the exact instruction count)

Per 32-elem group, one thread (the ggml MMVQ form, VDR_IQ1_S = 1 - the reference for the
repo's own kernel):
4 iterations (l0 = 0,2,4,6), each: the index build (~4 int ops: `qh >> 3*(l0/2)`, `& 7`,
`<< 8`, `| qs[l0/2]`), ONE shared-memory grid load (the 8 KB uint32[2048] table
in shared), the nibble unpack (3 ops: two `& 0x0F0F0F0F` + one `>> 4`), 2 q8 loads, 2
dp4a. Then the tail: the scale extract (~4), the delta (~4), the final FMA (~3).
TOTAL ~48 INT-pipe ops (dp4a inclusive) + ~13 loads per 32 elems = ~1.5 ops/elem.

THE CORRECTION (the exact math, why the nibble form works): dp4a dots the L-INDEX against
the q8 activation, but the true value is (L-1) + delta, so the group dot =
d8*(sumi + (delta-1)*sum_q8_group) - the sum of the group's q8 quants, which the repo's
q8_K activation path ALREADY produces (the bsums: per-16 int16 sums; a 32-group =
bs[2g] + bs[2g+1], the same bsums the K2/K4 kernels read for the min corrections). No new
activation format; the r18 launch_quantize_q8_K preprocessing is reused verbatim.

## 4. The kernels at the CF shapes (what lands, the honest rate roof)

- FMT_IQ1S in gemv() (cf_engine.cu): the M=1 greedy path - quantize_q8_K + the new
  launch_gemv_iq1s (the ~48 ops/32 elems form above). Replaces the union-staging +
  repack + gemv_q8k class for every RESIDENT expert pick.
- The verify's M=8: NOT the drop-in launch_gemv_q8k_b form (it re-decodes W per (row,
  batch) pair - at 1.5 ops/elem-dot x 18.9 G elem-dots/round that is ~28 ms). The
  AMORTIZED form is frozen: per (expert, 32-group) decode ONCE (~40 ops), then 8 dp4a
  pairs (one per draft row) + the 8 per-row bsums corrections -> ~56 ops per 256
  elem-dots = 0.22 ops/elem-dot.
- THE ROOF (T4: ~16 INT32/dp4a lanes/SM/cycle x 40 SMs x ~1.59 GHz ~ 1.0e12 INT ops/s
  per GPU; the HBM ~300 GB/s): the greedy step's expert class 2.36 G elem-dots x 1.5 =
  3.5e9 ops ~ 3.4 ms single-GPU, ~1.7-2 ms at the TP half-split (the HBM side 461 MB ->
  1.5 ms single-GPU, under the INT roof); the verify round (M=8 amortized) ~4.1e9 ops ~
  4 ms single-GPU, ~2 ms split. FROM ~114 ms / ~112-155 ms. Even at 2-3x off-roof (the
  shared-grid LDS bank conflicts, the load counts, the un-tuned tile) the class lands in
  the ~5-15 ms band - a ~10-25x collapse of the engine's dominant cost. MEASURED AT THE
  L4; the roof is the argument, not the claim.

## 5. The packer (the one-time Kaggle kernel)

Per 32-elem group (the ggml quantize_row_iq1_s_impl form, replicated in this repo's own
code): the weighted sort (32 pairs), the prefix sums, the EXHAUSTIVE 2-boundary split
search (i1 x i2 over 33 boundaries, both shift directions x_p = {-1+d, d, 1+d} /
x_m = {-1-d, -d, 1-d} - ~1.1k O(1) evaluations), the ternary L assignment, then per
8-elem group the grid mapping: the kmap (u = the 16-bit packed L vector -> the grid
index; 6561 valid u of 65536) with the weighted-distance NEIGHBOR fallback
(iq1_find_best_neighbour2: the list [count, idx...] from the kmap's negative encoding,
the full-grid search as the last resort). The scale quantize: l = round(0.5*(id*scale-1))
clamped [0,7], `l |= 8` when shift == -1, qh[ib] = (l << 12) | (the 3-bit index halves at
3k); the block d = (max_scale/15) * 1.125 (the ggml fudge factors - the 1.125 on d and
the 1.05 in the neighbor scale - are part of the tuned format, replicated EXACTLY).
THE WEIGHTS: the first form is the SELF-SCALED weights (w = sqrt(sigma2 + x*x),
sigma2 = 2*sum(x^2)/256 per block, qw = 1) - NO imatrix pass. The honest risk: the ggml
IQ1_S quality is imatrix-tuned; if the L4 quality gate fails, the fallback is the
llama.cpp imatrix collected over the Q2_K_S GGUF on the Kaggle CPU (hours, one extra
one-time kernel) and a re-pack. The gate decides, nothing is assumed.
THE PIPELINE: stream `Blackfrost-AI/CYBER-FROST-3.8-BF16` (355 GB) per (layer, tensor)
over HTTP range reads (the largest single tensor is one layer's gu = 3.36 GB BF16 - fits
the 28 GB host RAM; never hold two), quantize, write the packed planes. The output file
is THIS REPO'S OWN layout (not a GGUF): per (layer, tensor) a slab header + the PackedW
planes, expert-major rows so each GPU's half is one contiguous range. COST: ~150-300
ops/elem x 120.7 G elems ~ 2-4e13 ops, OpenMP over rows on the Kaggle 4 vCPU ~ 1.5-3 h +
the streaming (~15-30 min at the ~200-400 MB/s class) + the 23.6 GB write - one one-time
Kaggle kernel, the same one-time-cost pattern as the baseline's llama.cpp binaries.
Attach as a Kaggle Dataset (~24 GiB).

## 6. The residency and the TP split (the staged landing)

STAGE 1 (small engine change, no TP): extend the existing TIERED resident-slab path -
the load-time resident set becomes the iq1_s slabs for as many layers as fit (the
conservative inventory: the layers [0,20) ~ 9.8 GB resident next to the 2.6 GB core =
12.9 GB, safe; the aggressive: ~24 layers = 11.8 GB, 15.1 GB total - the L4 VRAM
inventory decides the count). The HIT picks read the resident slab with ZERO staging
(the tiered branch that already exists); the MISS picks keep the current UVA/chunked
staging + the Q2_K_S repack path UNCHANGED. The round cost: the resident layers ~1-2 ms +
the miss layers' UVA ~60-80 ms -> the interim ~60-80 ms/round class. The quality gate
runs on THE MIX (the resident experts carry the requant values, the miss experts the
trunk's own) - which is exactly the configuration stage 1 serves.
STAGE 2 (the full win, the minimal TP): the experts split BY ID across the 2 T4s (GPU0
owns [0,256) per layer per tensor, GPU1 [256,512)), the CORE REPLICATED on both (2.6 GB
x 2 - the core forward is deterministic, no AR for it), the router on both (the same
input -> the same top-10, no AR), the expert gemv on the owner, ONE per-layer MoE
partial-sum combine (2560 floats x 48 layers: the P2P all-reduce or the pinned-staged
combine - the r14 AR floor class, ~0.2-1 ms/step, the honest TP tax, measured at the L4).
Per GPU: 2.6 core + 11.8 half-pool + KV + scratch ~ 14.9-15.1 GB - the L4 inventory
confirms. THE END STATE: the expert class ~2-4 ms/round, the greedy step and the verify
round both core-bound - the ~4-8x end-to-end projection, MEASURED AT THE L4.
The load: the dataset file read per GPU's contiguous expert range + cudaMemcpy into the
per-(layer,tensor) slabs (~23.6 GB total, ~1-2 min one-time load).

## 7. The gates

1. ROUND-TRIP (local, no GPU): the packer's output decoded by the FMT_IQ1S deq32 host sim
   (the test_deq_sim.cpp pattern, the T4Q_HOST_SIM form) must be bit-identical to the
   packer's own reference decode of every block - the loader round-trip check pattern.
2. SMOKE (L4): the add function, the resident-mix greedy run.
3. QUALITY (L4): the perplexity-class A/B - the Q2_K_S trunk vs the requant-experts
   trunk (the stage-1 mix, then the full) - plus the greedy-agreement rate at temp 0 on
   a 256-token coding prompt. THE BAR: the Q2_K trunk's own quality class, NOT
   bit-identity. If the A/B fails: the imatrix re-pack, then the 2-grid fallback form.
4. RATE (L4): the resident-expert gemv GB/s vs the r18 K-quant family (Q2_K ~179-200
   GB/s) - the roof math says the INT pipe caps ~150 GB/s/GPU of expert bytes at M=1; the
   measured number decides the kernel-tuning rounds.
5. VRAM (L4): the per-GPU inventory at the stage-1 and stage-2 forms.

## 8. The honest risk register

- QUALITY is the primary risk: the ternary lattice + the +-0.125 shift at 1.5625 bpw vs
  the trunk's 2.625 gu / 4.5 dn is a substantial per-expert noise step, and the ggml
  quality is imatrix-tuned while the first pack is self-scaled. The L4 A/B measures it;
  the imatrix and the 2-grid are the named fallbacks. NO assumption either way.
- RATE: the nibble grid decode is NOT the dp4a nibble path the r18 family rides; the
  shared-grid gather (1 LDS per 8 elems, bank conflicts) and the ~1.5 ops/elem sit below
  the K-quant rates per byte. The roof math (section 4) says the class still collapses
  ~10-25x; the kernel-tuning rounds close the gap the L4 measures.
- THE TP TAX: the per-layer combine (~0.2-1 ms/step) and the replicated core (~2.6 GB)
  are the price of the split; the stage-1 by-layer form pays neither.
- THE ONE-TIME COSTS: the packer kernel (~1.5-3 h) + the 24 GiB dataset + the imatrix
  fallback (hours) if the gate demands it - all one-time, all inside the Kaggle budget.

## 9. The order (the implementation rounds)

r1 [LANDED, all local]: FMT_IQ1S in packed.h + the deq32 template + the host sim + the
packer core (the SSD search, the kmap init, the scale/d/shift packing) + the round-trip
test - ALL LOCAL (no GPU needed), the r1 gate is the round-trip. LANDED AS:
t4q/src/iq1s_table.h (the extracted 2048-entry u16 lattice, host/device form switch),
FMT_IQ1S = 9 (packed.h, 50 B/256 = 1.5625 bpw), the deq32 FMT_IQ1S branch (deq.cuh,
the exact reference fp-op order), t4q/src/requant.h (the packer core: the standard
fp16 converters with RNE and payload-preserving inf/NaN, the exact iq2xs_init_impl
3-pass table build, the exact quantize_row_iq1_s_impl port with the self-scaled
weights, find_best_neighbour2 with NO fudge, the d=(max_scale/15)*1.125 fudge
replicated exactly, and the independent {1,3,5}-byte-grid reference decode),
t4q/tests/test_iq1s.cpp + the Makefile test_iq1s target. THE GATE PASSED: the fp16
all-65536 round-trip + 11 RNE spot checks (the ties, the subnormals, the 65520
overflow boundary); the kmap self-check + the byte-grid/u16-table agreement over all
2048 entries; the packer->deq32(host sim) vs the reference decode BIT-IDENTICAL over
4096 rows x 2048 cols across 4 input classes (normal / all-zero-eps / all-negative /
constant); the RMSE diagnostic rel ~0.40 on the synthetic mix (the honest
ternary-lattice + 15-step-scale error on uniform/constant synthetic data - NOT a
quality verdict; the quality gate is the L4 A/B, section 7). The nvcc podman build
gate: the library + cf_run compile clean, ptxas warnings exactly HEAD's 77 (41
tp_prefill + 36 tp_spec, pre-existing). THE GATE'S TWO CATCHES (both fixed): the
reference decode was missing the 32*ib group offset (every group overwrote slots
0-31 - found by the probe at the first differing element, the deq32 kernel path was
the correct one), and the test's own LCG normalization bug (a 24-bit return treated
as 8-bit -> ~1e5 inputs -> a false 2494 RMSE explosion). Verified against the fetched
primary source: the eps path really does skip the packing (a zero group inside a live
block decodes to the -0.875*d bias quirk - the ggml behavior, replicated), and the
ggml itself uses the float-pairs + int-aliasing + value-only comparator form (the
port's form is canonical, not an invention).
r2 [LANDED, build-gated locally]: the tables (the 2048-entry nibble grid) + the
launch_gemv_iq1s M=1 kernel + the FMT_IQ1S gemv dispatch. LANDED AS: the second table in
iq1s_table.h - t4q_iq1s_grid_gpu[2048] uint32, the ggml iq1s_grid_gpu values VERBATIM
(fetched ggml-common.h), the halves-interleave packing (the LOW nibble of byte b = L_b =
the sub-vector elems 0-3, the HIGH nibble = L_{b+4} = elems 4-7) - the packing pinned
EMPIRICALLY at the extraction (the halves form matches the fetched table 2048/2048; the
naive nibble-j=L_j form matches only 42/2048 and would mispair L_2 with q_1 in the dp4a);
the same host/device form switch as the u16 table. The kernel: the dot_q8k FMT_IQ1S
branch in gemv_ref.cu - the ggml vec_dot_iq1_s_q8_1 translated to the PackedW planes +
the q8_K pairing (4 iterations per 32-group: the index build, ONE u32 grid load, the
3-op nibble unpack to 2 dp4a byte-quads, 2 dp4a; then the d1q = h2f(d)*((qh>>11)&0xE)+1
tail with the ggml's one-shift trick, delta = -1 + 0.125 - (qh&0x8000)*(2*0.125/0x8000)
= -0.875/-1.125, and the group dot = d1q*yd*(sumi + delta*(bs0+bs1)) - the SAME two
bsums the K2/K4 branch reads, the ggml q8_1 s-term's equivalent at the q8_K pairing, NO
new activation format); the launch_gemv_q8k FMT_IQ1S case (the M=1 greedy path, the
k_gemv_q8k warp-per-row form unchanged) + the gemv() dispatch branch in cf_engine.cu
(FMT_IQ1S joins the K2/K4 q8_K pairing). NOT the batched _b form - the spec's r5
amortized M=8 kernel replaces it, the drop-in re-decodes W per (row, batch) pair. THE
GATES (test_iq1s.cpp, host): the u32-vs-u16 packing identity 2048/2048; the int-exact
sumi gate - the nibble-unpacked dp4a sum vs the INDEPENDENT u16 per-elem walk,
INTEGER-IDENTICAL over 4096 rows x 64 groups (0 mismatches - the pairing bug class is
deterministically excluded); the fp-dot gate - the full kernel-arithmetic twin (incl.
the host q8_K quantizer twin: the first-occurrence argmax, iscale = -127/maxv,
MIN(127, v), the bsums, d = 1/iscale) vs the deq32-decode dot over the same x: rel
6.9e-05 L1-normalized (the honest q8_K activation error class). THE BUILD GATE: nvcc
clean, k_gemv_q8k<9> = 41 registers / 0 spills / 0 smem (the K2 twin is 63), the ptxas
warnings exactly HEAD's 77. THE DEVICE-FORM NOTE: the grid reads go straight to the
__device__ table (the ggml MMVQ's own form, L1-resident at 8 KB) - the spec's
shared-memory staging stays a RATE-tuning option for the L4 measure, not a correctness
difference.
r3 [LANDED, all local except the Kaggle streaming driver]: the packer tool + THE TILING
FINDING + the half-block amendment. THE GATE'S CATCH (the round's reason): the first
pack_gate run FAILED on the dn plane - the dn rows are 640 wide and 640 % 256 = 128,
the 256-elem iq1_s block CANNOT tile the dn (this is why the GGUF's dn experts are
Q4_0, a 32-block format); the naive 768-pad would put the pool at 25.17 GB, OVER the
~24-25 GB room. THE AMENDMENT: FMT_IQ1SH = 10, the 128-elem half block - 16 B codes +
8 B hi + 2 B d = 26 B/128 = 1.625 bpw, 640 = 5x128 EXACTLY; the SAME per-32-group
search/scale/lattice/dp4a arithmetic, only the block scope halves (the max_scale and d
span 4 groups, not 8 - a FINER scale grid on the dn, a marginal quality gain if
anything); the corrected pool gu 15.73 GB (80.5 G elems at 1.5625 bpw) + dn 8.18 GB
(40.2 G elems at 1.625 bpw) = 23.90 GB, INSIDE the room. THE PAIRING (settled here):
the SH dot pairs with q8_0, NOT q8_K - the q8_K 256-super-blocks also cannot tile 640,
while the q8_0 32-blocks tile anything, and its per-32 SIGNED int sum xs[g] is exactly
the correction term the dot needs (the weight value = L + delta - 1, so the group dot
= d1q*dy*(sumi + delta*s32) - the ggml (delta-1)*sum_q8 correction at the repo's own
q8_0 activation). LANDED AS: requant.h TEMPLATED (BlockT<NG> with the 50/26-B
static_asserts, quant_row_t<NG>, dequant_row_ref_t<NG>, the stock-256 wrappers
quant_row/dequant_row_ref unchanged - the r1 gate passes unmodified through the
refactor); FMT_IQ1SH = 10 in packed.h; the deq32 FMT_IQ1SH branch in deq.cuh (nb =
cols/128, the 16/8/2-B plane strides, the group decode IDENTICAL); the
dot_q8_0_iq1sh branch + k_gemv_iq1sh in gemv_ref.cu (the k_gemv_q80 warp-per-row
twin: the nibble-grid dp4a identical per group, the index build one u32 grid load,
the s32 correction, the d1q/delta tail) + the launch_gemv_q8_0 FMT_IQ1SH case + the
gemv() dispatch branch in cf_engine.cu (FMT_IQ1SH joins the P4/Q8 q8_0 pairing);
tools/cf_requant_pack.cpp v2 (the 96-B SlabHdr with fmt + dn_fmt + the plane offsets,
the dn planes at the half-block strides, quantize_tensor<NG> OpenMP, the synthetic
source generator, the verify reading BOTH plane sets and decoding the dn via
deq32<FMT_IQ1SH> + dequant_row_ref_t<4>) + the Makefile pack_gate target. THE GATES
(all local): test_iq1s EXTENDED - the r1/r2 gates UNCHANGED through the BlockT
refactor (the 256-block round-trip still BIT-IDENTICAL, rmse rel 0.4022 the same
value, the M=1 dot twin 6.9e-05 the same - the refactor is byte-identical); the NEW
SH gates: the 640-wide round-trip (2048 rows x 5 blocks, the same input classes incl.
the eps/constant paths) BIT-IDENTICAL deq32-vs-ref, rmse rel 0.4021 (the same honest
lattice class); the SH M=1 dot twin (the host q8_0 quantizer twin: amax/127, the
fp16-rounded d, roundf(x*id), the per-32 signed sum; the full kernel-arithmetic dot)
vs the deq32-decode dot: sumi INTEGER-EXACT (0 mismatches vs the independent u16
per-elem walk) and rel 1.23e-04 L1-normalized (the honest q8_0 activation class,
bound 2e-2). pack_gate GREEN: ne=4 synthetic, the slab 3,891,296 B EXACTLY (96-B
header + 2,560,000 gu + 1,331,200 dn - the plane arithmetic checks to the byte), all
15,360 rows verified deq32-vs-ref BIT-IDENTICAL, src rmse rel 0.4024 (the diagnostic,
NOT a quality verdict). THE BUILD GATE: nvcc clean, k_gemv_iq1sh = 64 registers /
0 spills / 0 smem (the k_gemv_q80 P4 twin's own 64 - the family budget), ZERO ptxas
warnings in the rebuilt TUs (HEAD's 77 are all in tp_prefill 41 + tp_spec 36, objects
unchanged - the counts re-confirmed from the tracked ptxas logs). STILL OPEN in r3's
frame: the Kaggle streaming driver notebook (the 131-shard HTTP-range read of the
expert tensors, the largest 3.36 GB, feeding this packer) + the dataset push - rides
the Saturday L4 window.
r4: the stage-1 tiered residency (the loader reads the slabs, the HIT branch dispatches
FMT_IQ1S) + the L4 smoke + the A/B + the rate measure.
r5: the M=8 amortized verify kernel + its L4 measure.
r6: the stage-2 TP split (the by-ID residency, the replicated core, the per-layer
combine) + the full L4 battery.
The L4-blocked r19w-r19aa battery (the MTP graphs/prefetch A/Bs) rides the same sessions.
