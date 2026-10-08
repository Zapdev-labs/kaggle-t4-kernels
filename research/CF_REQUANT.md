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
THE PIPELINE (LANDED, the r3 driver): stream `Blackfrost-AI/CYBER-FROST-3.8-BF16` (355
GB) per (layer, tensor) over HTTP range reads (the largest single tensor is one layer's
gu = 3.36 GB BF16 - fits the 28 GB host RAM; never hold two), quantize, write the packed
planes. The output file is THIS REPO'S OWN layout (not a GGUF): per layer ONE slab (the
96-B v2 header + the gu planes + the dn planes, expert-major rows so each GPU's half is
one contiguous range) - `kaggle/cfreqa|cfreqb` (t4q/tools/stage_cfreq.py, generated by
mkkernel.py --template cfreq --define CFREQ_LO/HI). THE LANDED FORM'S DEPARTURES from
the sketch above, all forced by real limits: (1) TWO CPU-ONLY kernels, not one GPU
kernel - the ~24.4 GB pool (49 layers x 498.07 MB, the MTP layer's ne is also 512,
verified from the shard headers) is over the 20 GB kernel-output cap, so cfreqa packs
layers [0,24) ~11.95 GB and cfreqb [24,49) ~12.45 GB, each under the 19.5 GB working
dir; the runtime battery attaches BOTH as kernel_sources (the p/pg pattern), NO dataset
push and NO secrets needed; and a CPU kernel burns NO GPU quota - the whole L4 window
stays for the runtime battery. (2) The streaming is SEGMENTED (256 MB range segments,
per-segment retry + resume from the segment's own start, 10 attempts) with the shard
headers cached (one 8-B read + one header read per shard) and the producer thread one
layer ahead of the packer (2 rotating raw slots on /tmp, ~10 GB peak). (3) The slab
lands ATOMICALLY: pack to .tmp, --verify a 32-row sample against the raw sources (the
deq32-vs-ref bit-identity + the src-RMSE diag), the byte-count check to the byte (96 +
ne*1280*500 + ne*2560*130), then the rename - a killed session never leaves a partial
slab in the output. (4) The manifest.json (the per-layer ne/bytes/verify line) is the
r4 loader's input. COST: ~150-300 ops/elem x 120.7 G elems ~ 2-4e13 ops, OpenMP over
rows on the Kaggle 4 vCPU ~ 1.5-3 h + the streaming overlapped + the ~24.4 GB write -
each kernel ~1.5-2.5 h of the 12 h CPU cap (DEADLINE 11 h + the r19b/r19f watchdog
pair). THE LIVE SMOKE (r3, against the real endpoint): the index fetch, the shard
header parse, the 206 range reads, layer 0's gu = BF16 [512,1280,2560] with
data_offsets summing to EXACTLY 3,355,443,200 B (the pinned number), real weight values
decoding (|w| ~ 1.2 class), and the MTP dn shape [512,2560,640] on its own shard.

## 6. The residency and the TP split (the staged landing)

STAGE 1 (small engine change, no TP): extend the existing TIERED resident-slab path -
the load-time resident set becomes the iq1_s slabs for as many layers as fit (the
conservative inventory at the LANDED per-layer 498.07 MB: the layers [0,20) ~ 9.96 GB
resident next to the 2.6 GB core = 12.6 GB, safe; the aggressive: ~24 layers = 11.95
GB, 14.6 GB total - the L4 VRAM inventory decides the count; the whole 49-layer pool is
24.41 GB, so stage 1's by-layer form caps at ~24 layers/GPU and stage 2's by-ID split
is the full-residency form). The HIT picks read the resident slab with ZERO staging
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
unchanged - the counts re-confirmed from the tracked ptxas logs). THE DRIVER (the r3
pipeline remainder, LANDED in the follow-up commit): t4q/tools/stage_cfreq.py + the
mkkernel.py --template/--define/--cpu extensions, generating kaggle/cfreqa (layers
[0,24)) + kaggle/cfreqb ([24,49) incl. the MTP) - the two-CPU-kernel form (the 20 GB
output cap and the GPU-quota economy, section 5's landed-form notes) with the
segmented retrying stream, the producer one layer ahead, the atomic .tmp-then-rename
slabs, the per-layer verify + the byte-count check, and the manifest for the r4 loader.
LIVE-SMOKED against the real endpoint: the index, the shard headers, the 206 range
reads, layer 0's gu 3,355,443,200 B EXACT, the MTP ne=512. The kernel PUSHES ride the
Saturday window with the L4 battery.
r4 [LANDED, build- and host-gated locally; the value gates are L4]: the stage-1 tiered
residency. LANDED AS: (1) The kernels - the launch_gemv_q8k_b FMT_IQ1S case (the batched
resident gu: dot_q8k<9> in the _b walk, the SAME activation reads the walk already does -
bsb[sb*16+sub]/[..+1] the group's two 16-sums, db[sb] the super-block d) and
k_gemv_iq1sh_b + launch_gemv_iq1sh_b (the batched SH dn on the k_gemv_q80_b walk form,
the same q8_0 planes/strides). (2) The loader - SlabHdr moved to packed.h (ONE shared
definition, the writer and the reader pinned by the same 96-B static_assert); the
T4Q_CF_IQSLAB=<dir> + T4Q_CF_IQN=<1..48> block: the count EXPLICIT (no auto-fit
guesswork - the L4 VRAM inventory decides, the spec's gate 5), the free-VRAM check, the
per-slab header + plane-offset validation to the byte, the pageable one-time uploads,
the identity hot map (hn = NE, every pick a HIT), the MTP slab deferred to stage 2; the
hot-set skip (the covered layers' ids consumed + discarded - the file is per-layer
sequential); the uncovered-layer all-miss safety pass (the tiered host_router derefs
hot_idx - a layer covered by NEITHER tier would crash); the UVA prefix exclusion
(uva_lo: the span + alias loops over [uva_lo, uva_n), the fully-covered case logged +
off). (3) The engine - host_router's IQ1S hit-view branch; emit_moe_rest's per-layer
iqs branch (the SAME quantizes - the pairings COINCIDE: the gu rides q8_K for both K2
and IQ1S, the dn rides q8_0 for both P4 and IQ1SH - with the r4 batched dots;
whole-layer residency means NO within-layer format mix, one branch per layer not per
pick); vfy_window's resident early path + vfy_moe_em's branch. THE MTP CONSISTENCY
FINDING: the verify MUST read the SAME weights the greedy path reads or the acceptance
compares two different models (the originals-staged verify would diverge from the
resident-reading greedy at the covered layers - the byte-identity gate's class), so the
covered layers skip the union staging entirely (nu = 0, the dedup skipped) and the
per-row views point at the resident planes. THE GATE'S TWO CATCHES: (1) alloc_packed
had NO FMT_IQ1S/FMT_IQ1SH cases - the silent switch fall-through left 0-B/NULL planes
and the tier's uploads would land nowhere (a runtime crash NO local gate could see; the
fix is the two cases PLUS the loader's alloc-vs-slab cross-check that catches the class
AT LOAD, not at first use); (2) the d-offset UNITS bug in the view math - d is
uint16_t*, the offset is in ELEMENTS with no byte factor (the existing K2 form's own
convention, (EE/32) with no *2) - the first draft had *2 on both d lines, caught in
review against the K2 form, and the new view gate pins the correct class. THE GATES:
test_iq1s + the NEW resident-view gate (a 2-expert expert-major plane, the expert-1
view at the loader's EXACT offsets, the SH dot twin vs the deq32 decode of expert 1's
own rows: sumi INTEGER-EXACT, the dot inside the 2e-2 bound) - the r1/r2/r3 gates
unchanged; pack_gate green through the SlabHdr move; the nvcc build clean with ZERO
ptxas warnings in the rebuilt TUs (HEAD's 77 all in the untouched tp_*); the registers:
k_gemv_q8k_b<9> = 47 (LIGHTER than the K2 twin's 63 - the nibble-grid dot is
register-cheap), k_gemv_iq1sh_b = 64 (the q80_b family budget), both 0 spills / 0 stack
/ 0 smem. THE GRAPH NOTE, honestly: the gmode&&tiered check turns the graphs OFF under
the IQSLAB (the conservative form) - a FULL-coverage run is capture-constant in
principle (the W-table H2D per layer is fixed-size, nmiss = 0 always) but the verify's
union staging for uncovered layers and the draft block still vary; the re-enable is an
L4-measured later round, not assumed. THE VALUE GATES (L4, Saturday): the smoke, the
perplexity A/B on the mix, the greedy agreement, the resident-expert rate vs the r18
K-quant family, the VRAM inventory.
r5 [LANDED, host- and build-gated locally; the rate is L4]: the AMORTIZED M=nr verify
kernel, the spec's frozen form. LANDED AS: k_gemv_iq1s_vfy + k_gemv_iq1sh_vfy (gemv_ref.cu)
- the walk is the _b family's own (warp = the W row, lane strides the 32-groups, the xor
tree, lane 0 writes) with the row loop UNROLLED over the fixed 8 slots and GUARDED by the
pick mask (block-uniform, so no intra-warp divergence): per (union pick, 32-group) the
decode happens ONCE - the 8 grid quads held in registers, d1q/delta computed once - then
every row that picked the expert dots against the SHARED quads (2 dp4a per 8 elems per
row + the per-row tail: the bsums pair at the q8_K pairing, the s32 at the q8_0 pairing),
the outputs at the SAME per-row planes the _b form wrote (s.logits + k*2n / ye + r*TOPK*D
+ k*D). The interface: the picks' UNION views (uv_gu/uv_dn, vfy_window's dedup + the
resident-slab offsets verbatim) + the row map (rowmap[u*8+r] = row r's pick index of slot
u, -1 = not picked) + the FIXED per-row plane table (VfyMoeTab in packed.h - the scratch
pointers never move, ONE upload at the verify alloc). vfy_moe_em's resident branch
restructured to the phased form: the nr quantizes, ONE gu launch over the union, the
per-row silu + q8_0 quantizes, ONE dn launch, then the per-row tail verbatim. THE
BIT-IDENTITY CLAIM, pinned: the per-(row, pick) accumulation over g is EXACTLY the _b
form's (the same g-sequence per lane, the same tree, the same tail expression) - the new
gate's amortized sim (the kernel's own decode-then-rowloop structure) vs the per-(row,
pick) dot_iq1s_sim/dot_iq1sh_sim over the same views + activations is BIT-IDENTICAL, plus
the sumi INT-exactness vs the independent u16 walk, the rel bound vs the deq32 decode dot
(2.18e-04), and the union/rowmap COVERAGE check (every pick at exactly one (slot, k), the
uidx sweep leaves no residue). THE GRAPH NOTE: the amortized launch's grid rides nu (the
union count, varying per layer AND per step) - not capture-constant, exactly the class
the r19z comment names; the verify's V1 graphs are already dead under the tier (the
loader's gmode&&tiered kill at load), so no capture ever sees the varying grid. THE
REGISTERS: k_gemv_iq1s_vfy = 64 / k_gemv_iq1sh_vfy = 60, both 0 stack / 0 local / 0 smem
(no spills - the masked unroll keeps the 8 accs + the 8 quads live without blowing the
family budget). THE HONEST NOTES: (1) the win is the pick overlap across the nr rows (the
drafts are near-duplicates - their top-10s overlap heavily); at ZERO overlap (nu =
nr*TOPK) the cost is the _b form's own + the rowmap overhead, so the form is win-neutral
at worst - the L4 measures the real overlap; (2) the UNCOVERED layers (past the iq1_s
prefix at stage 1) keep the _b form - their verify re-decode stays; the amortization
extends there only if the L4 shows the uncovered verify cost matters (at stage 2's full
pool every layer is resident and the amortized path covers everything). THE VALUE GATE
(L4): the verify-round wall before/after at the covered layers (the spec's ~28 ms ->
~4 ms class at full overlap, measured not assumed).
r6 [THE DESIGN FREEZE - the implementation arc's spec; cf-m6 stage 2, the full-pool
by-ID TP split]: the exact forms, frozen for the implementation rounds (r6a/b/c).
r6a [LANDED, host- and build-gated; the emission is r6b]: the SPLIT TIER. LANDED AS:
T4Q_CF_IQTP=1 alongside the IQSLAB pair - the loader's block forks: the per-GPU
free-VRAM checks (the half need per side, the ~2.6 GB core replication charged at r6b,
noted), and per covered layer each side reads its OWN half's planes as ONE CONTIGUOUS
byte range per plane (the planes are expert-major: the experts [g*NE/2, (g+1)*NE/2) are
one range - one fseek+fread per plane per GPU, the offsets = the header's plane bases +
o*rows_per_half*plane-units with o = g*NE/2) into the per-side pair (res_gu/res_dn =
GPU0's own - every landed r4/r5 read site untouched - plus the NEW res_gu1/res_dn1 =
GPU1's, the allocs at [NE/2*2*EE, D] / [NE/2*D, EE] on each device with the
alloc-vs-slab half cross-check); the identity owner map stands (hn = NE, every pick a
hit across the two GPUs, the engine derives owner(e) = e >> 8 inline at r6b). THE
ENGINE GATES: cf_step/cf_verify/cf_draft_step THROW under iqtp until r6b lands (the
load succeeds + the tier's prints + the step-time throw = the r6a form's own L4 probe).
THE GATE: the split-plane view twin (a 4-expert SH plane, the owner-1 half copied as the
loader's one-range read verbatim, expert 3's view INTO the half at le = 1, the dot twin
vs the deq32 decode of expert 3's OWN rows from the FULL plane - a half-offset slip
reads the symmetric neighbor and the dot lands far outside the bound): sumi EXACT, dot
OK. THE BUILD: clean, zero ptxas warnings in the rebuilt TUs. THE REST IS r6b (the
per-side emission + the dispatch + the combine kernels + the sync pairs + the scratch/
rolling-state replication + the core replication) + r6c (the draft/verify TP forms + the
battery extension), per the freeze below. THE
r6b [LANDED, host- and build-gated; the draft/verify TP forms are r6c]: THE FREEZE
AMENDED FIRST (the design re-check at the emission head): the REPLICATED CORE is
replaced by the ACTIVATION SCATTER - the core (the attention/deltanet family, the hc
mixers, the shared expert, the router, the rolling states, the KV, the PLE ring) runs
on GPU0 ONLY, the per-layer mixed [D] ships 0->1, and GPU1 is a PURE MoE ACCELERATOR
(its owned picks' quantize + dots + partial). THE VERDICT: the replication bought
NOTHING on the critical path (the core wall is the same - the same core work ran in
lockstep on both, and the moe split dominates either way) while costing the ~2.6 GB
core twin + the double launch wall + the lockstep-determinism hazard (the rolling
states divergent across the pair - a byte-level correctness risk the scatter removes:
the states stay GPU0's own, the single source of truth). THE AMENDED FORM: the loader's
GPU1 scratch is the CfIqtp plane set (<1 MB: the mixed/logits/ffa/ye/partial/partial0/
xq* planes + the per-side W tables + the we_c pair) + the stream + the cross-device
event pair + the best-effort PCIe P2P both directions (the D2Ds fall back to the
runtime's host-staged form either way - correct, the L4 measures which); NO CfScratch
twin, NO rolling-state twins, NO block peer-copy (GPU1's only per-layer input is the
mixed). THE COMPOSE (host_router): the covered layer's picks are ALL hits on the
per-side pair - the owner dispatch (owner(e) = e >> 8, le = e & 255) lands each pick
in its side's COMPACT slot with its compact we, the view math the tiered branch's own
(le*rows*plane-units into the SIDE's plane: GPU0 res_gu/res_dn, GPU1 res_gu1/res_dn1),
n[0]+n[1] = TOPK, tier_nmiss = 0. THE EMISSION (emit_moe_rest): GPU1's side FIRST (so
the peer work overlaps GPU0's own): ev0 (st's tail - the mixed + everything before it)
waited on st1, the W-table + we_c[1] H2Ds + the mixed [D] D2D 0->1, the q8_K quantize
+ the FMT_IQ1S gu dot + the silu + the q8_0 + the FMT_IQ1SH dn dot + the partial (the
count = n1, the compact slots), ev1; then GPU0's side: the W-table + we_c[0] H2Ds, the
SAME quantize/dot/silu/quantize/dot chain over n0 (the trunk's own scratch), the
partial into partial0, THE SHARED EXPERT verbatim (GPU0's own weights + scratch), then
the combine: st waits ev1, the partial [D] D2D 1->0 into partial0+D, the final
(p0 + p1 + sigmoid(gate)*ysh - the moe_out's own arithmetic, the add order split
across the sides, the reassociation class measured 2.4e-07 in the gate) into the
block. ONE event pair + 2 [D] D2Ds per layer (vs the freeze's ~2 pairs + the block
copy) - the TP tax class the L4 measures. THE SYNC DISCIPLINE (the reuse hazards
closed BOTH ways by the same pair): the mixed's overwrite on st is behind the
combine's ev1 wait; the host tables' overwrite at the next compose is behind the
router's st sync, which is behind the prior combine, which waited ev1 = st1's full
tail (the H2D reads included); the partial's overwrite on st1 is behind the next ev0
chain. THE KERNELS: k_cf_moe_partial (43 regs, 0 spills) + k_cf_moe_final (15 regs, 0
spills). cf_step's THROW LIFTED (the greedy path is landed); cf_verify/cf_draft_step
still THROW (r6c). THE GATE: the split-moe combine twin (the REAL id space e in
[0,512), owner = e>>8, BOTH owners forced - a pure-lcg draw once landed 10/0 and left
side 1 unexercised; the forced 5/5): the dispatch + the compact slots + the per-side
partials + the final vs the moe_out's own single-loop form over the same values - max
|d| 2.384e-07 (the fp32 add-order class, bounded 1e-4), the structure checks (no pick
lost/doubled, n0+n1 = 10) OK; the r1-r6a lines UNCHANGED-green. THE BUILD: clean, ZERO
ptxas warnings; the 77 pre-existing nvcc front-end warnings (#128-D, unreachable
loops) unchanged in the untouched tp_* TUs (41 tp_prefill + 36 tp_spec), the one
host-gcc -Wformat-truncation at cf_loader.cu:405 verified IDENTICAL at HEAD (the r4
tier's own layer_%03d snprintf - provably safe, il < n <= 48; not introduced by r6b).
THE REST IS r6c (the draft/verify TP forms + the battery's IQTP phases).
r6c part 1 [LANDED, host- and build-gated; the verify TP form is part 2]: THE DRAFT
SPLIT - the draft's 512 Q8_0 experts split BY ID too (the freeze's ~1.33 GiB/side
arithmetic): the loader's draft block under iqtp takes the per-side free checks (the
r6a form), the per-side allocs (res_gu/res_dn shrink to GPU0's [0,256) half at
[256*2*EE, D]/[256*D, EE], res_gu1/res_dn1 join on GPU1), and the chunk loop NEVER
STRADDLING the 256 boundary (each chunk stages + repacks on its OWN side's stream; GPU1
gets its OWN raw staging plane - a kernel cannot read a remote pointer without the P2P
the loader only tries best-effort; the SHARED host staging's refill syncs BOTH streams
per chunk, the load-time conservative form; the local row offsets (e0 - NE/2)*rows);
the draft's VRAM print carries the split. THE EMISSION (cf_draft_step): the greedy r6b
discipline REUSED VERBATIM - the CfIqtp planes (every draft plane fits: the gu y
[TOPK*2*EE], the ffa [TOPK*EE], the ye [TOPK*D], the q8_0 planes [HCD >= TOPK*EE]), the
event pair, the owner dispatch (the compact slots + the compact we), the per-side dots
(ONLY the W-table views differ: the draft's Q8_0 slabs at the local row offsets
le*2*EE*D / le*2*EE*(D/32) and le*D*EE / le*D*(EE/32) into the SIDE's half), the
partial/combine pair, and the serial-use discipline (the draft runs between greedy
steps; the streams' own order serializes every plane reuse, the same ev1 chain closes
the host arrays). cf_draft_step's THROW LIFTED (the greedy + draft paths are landed);
cf_verify still throws (part 2). cf_free: GPU1's planes now torn down (the device-1
reset under iqtp - the r6a halves + the CfIqtp + the draft's halves all fall to it).
THE GATE: the NEW draft q8_0 split-view twin (a 4-expert Q8_0 pair, the owner-1 half at
the local row offsets, expert 3's view INTO the half at le = 1 with the EMISSION'S OWN
compose offsets, byte-identity vs the FULL plane's expert-3 rows - gu and dn at their
own strides, the fp16-scale d plane as raw 2-B words): BYTE-IDENTICAL (256 rows); the
r1-r6b lines UNCHANGED-green; the build clean (MAKE_EXIT 0, zero new warnings - the 77
#128-D + the one HEAD-identical host-gcc truncation both unchanged). THE REST IS r6c
part 2 (the verify TP: the per-side sub-unions + the per-side VfyMoeTab + the amortized
dots per side + the per-row gather-partial + the per-row ships/finals) + the battery's
IQTP phases.
r6c part 2 [LANDED, host- and build-gated; the TP arc is COMPLETE - the battery's IQTP
phases remain]: THE VERIFY TP - the freeze's hardest piece. THE STRUCTURE: the loader's
verify-alloc extension under iqtp (GPU1's OWN per-row planes - the shipped mixed, the
q8_K/q8_0 activations, the gu y + the silu out + the dn y at the PICK-SLOT layout; the
side's FIXED tab (one upload, the r5 form); the sub-union W tables [MAXR*TOPK]; the
rowmap pair; the per-row owned k-lists; the we copy; the [2*MAXR*D] partial pair - p0
at [0,MAXR*D), the shipped p1 landing at [MAXR*D, ...) - the greedy's own 2-slot form);
all trivial scratch (<1 MB). vfy_window's split branch: the union dedup (the r5 form)
then the SPLIT BY OWNER - each union pick lands in its side's sub-union (the compact
slot ug) with the W views into the SIDE's resident planes (the r5 view math verbatim
with le + the side's pair), the rowmaps split per side (rowmap_g[ug][r] = k), and the
per-row OWNED k-lists built for the gather partials (the per-row ye plane is the
PICK-SLOT layout - the amortized dn writes the row's picks at their k slots, the
unowned slots are stale - so a side's partial needs the row's own k-list, the compact
form would sum stale slots); side 0 rides the EXISTING uv/rowmap buffers, side 1 the
CfIqtp pair on st1 (+ the we_dev1 H2D from the pinned we_h). vfy_moe_em's split branch:
GPU1 (st1, behind ev0 = st's tail) runs the per-row mixed ships + the q8_K quantizes +
its sub-union's AMORTIZED gu dot + the per-row silu/q8_0 (the full-plane batch, the
stale slots never read - the rowmap-guarded dot, the r5 form's own class) + its
amortized dn dot + the per-row GATHER partials; GPU0 runs the r5 phases verbatim over
its sub-union + its gather partials; the per-row shared experts launch BEFORE the ev1
wait (GPU0's own work overlapping GPU1's dots), then the ev1 wait + the ONE [nr*D]
partial ship + the per-row finals (p0_r + p1_r + sigmoid(gate_r)*ysh_r - the per-row
moe_out's own arithmetic, the add order split across the sides, the reassociation
class measured 2.384e-07 in the gate). THE KERNEL: k_cf_moe_partial_k (56 registers, 0
spills) - the moe_out's own per-element expression with the k-gather. cf_verify's
THROW LIFTED (the greedy + draft + verify paths all landed). THE GATE: the verify
split-moe twin (3 rows x 10 picks, BOTH owners forced, the REAL id space): the r5
union dedup sim + the split compose sim (the sub-unions + the per-side rowmaps + the
k-lists) with the COVERAGE checks (each (row, pick k) in EXACTLY ONE side's rowmap slot
+ one ks entry; nk0+nk1 = 10 per row; the ks entries are the row's own pick indices;
nus0+nus1 = nu - the sub-unions cover the union) + the gather-partial/final arithmetic
vs the moe_out's own single-loop form over the same values: OK (sub-unions 15+15 of
30, coverage 0 bad, max |d| 2.384e-07); the r1-r6b-part-1 lines UNCHANGED-green; the
build clean (the 77 #128-D + the one HEAD-identical host-gcc truncation both
unchanged, zero ptxas warnings). THE REMAINING r6c: the battery's IQTP phases (the
cfbat extension - the smoke + the sweep + the A/B + the spec wall at T4Q_CF_IQTP=1).
r6c part 3 [LANDED, gated; the r6 TP arc is COMPLETE]: THE BATTERY'S IQTP PHASES -
the cfbat template's phase 6 (the driver regenerated, 708 kB): the cfrun --iqtp flag
(T4Q_CF_IQTP=1 in the env), the split SMOKE + the greedy rate at IQN=IQN_AB (the TP tax
vs the same-IQN single-GPU run in the sweep), the full-48 SPLIT-CEILING probe (the
per-side throw carries the free-GiB number; the SPLIT's own ceiling arithmetic -
floor(free/(PER_GIB/2)), twice the single-GPU form - the honest note: a SUCCESS = the
whole 24.41 GB pool fits the split, the informative inventory for the Saturday
decision), and the IQTP verify round (the r6c wall: verify_ms + mean_union at the
split vs the spec_ab run) - all in the priority order AFTER the r5 phases (a deadline
cut loses the newest first). THE GATES: the regenerated driver ast-parses with the
defines verified; the split-ceiling arithmetic functionally gated on the loader's
REAL per-side throw form (floor(11.62/0.232) = 50 vs the single-GPU 25, the
no-match None); the phase order gated (the IQTP phases after the r5 phases). THE
SATURDAY KERNEL CURRENCY: the cfreqa + cfreqb payloads REGENERATED against the
current tree (the r3-era payloads predated the r4+ packed.h/requant changes - each
payload is self-consistent, but the regenerated ones carry the current sources so
the staleness question closes; the outputs are format-identical either way, the
SlabHdr v2 layout never moved) - all three Saturday kernels (cfreqa, cfreqb, cfbat)
now carry the r6c-era tree, gated (the ast-parse, the baked layer ranges, the
CPU-only no-source metadata, the distinct payloads). THE
SATURDAY SEQUENCE unchanged: cfreqa + cfreqb first, their outputs, then cfbat (the
sources resolve), the battery, the r19w-r19aa kernels. THE
OWNER MAP: owner(e) = e >> 8 (NE = 512, the halves 256: GPU0 [0,256), GPU1 [256,512)),
the local index le = e & 255. THE SPLIT PLANES: the slab planes are EXPERT-MAJOR, so the
owner's half is ONE CONTIGUOUS BYTE RANGE per plane (the codes/hi/d of the experts
[g*256, (g+1)*256)) - ONE fseek+fread per plane per GPU (no per-expert seeks), the
alloc_packed(gpu=g, FMT_IQ1S, 256*2*EE, D) + (gpu=g, FMT_IQ1SH, 256*D, EE) on EACH
device, the free-VRAM check per GPU (the per-GPU need: the core ~2.6 GB replicated +
the half-pool 0.232 GiB/layer + the per-GPU scratch + the KV; the spec's 14.9-15.1 GB,
the L4 inventory confirms). THE RESIDENT FIELDS: the per-GPU pair (res_gu1/res_dn1 JOIN the existing res_gu/res_dn -
GPU0's own - so every landed r4/r5 read site stays UNTOUCHED and r6b's per-side loops
read the pair; the "arrays" sketch was the idea, the pair is the landed compat form),
the hot maps the OWNER form (hn = NE -
every pick a hit ACROSS the two GPUs; hot_ids[e] = e, hot_idx[e] = the packed
(owner<<8 | le)... or the engine derives owner(e) inline - the simpler form, the
identity map keeps hn=NE). THE ENGINE (r6b, the greedy emission): (1) [AMENDED at
r6b - the r6b record above: the REPLICATED CORE is replaced by the ACTIVATION SCATTER
(the core on GPU0 only, the mixed ships 0->1, GPU1 a pure MoE accelerator) - the
replication bought nothing on the critical path while costing the ~2.6 GB twin + the
double launch wall + the lockstep-determinism hazard] THE REPLICATED
CORE - every core op (the attention family, the hc mixers, the shared expert, the
router gemv, the rolling states - the KV cells, the GDN S/conv, the PLE ring) runs on
BOTH GPUs on the SAME inputs - deterministic identical states, NO all-reduce for the
core; the per-GPU scratch (the CfScratch per device) + the per-GPU rolling states at
load. (2) THE ROUTER - the gemv on both, the D2H from GPU0 only (the same values). (3)
THE MoE DISPATCH - the picks split by owner: per side g the owned picks' W views (the
local row base le*2*EE / le*D on the side's res_gu/res_dn), the q8_K quantize of the
SIDE'S mixed (already there - the replicated forward produced it), the batched gu dot
(launch_gemv_q8k_b FMT_IQ1S on the side's stream) over the OWNED picks' views, the
silu_mul + the q8_0 quantize per side, the dn dot writing the COMPACT per-side slots
(the W-table's y_stride does it - no gather kernel), the per-side PARTIAL
(Σ_{owned} we_k*ye_k, no shared - a small new launch_cf_moe_partial), then (4) THE
BOTH-WAYS COMBINE per layer: GPU1's partial [D] peer-copied 1->0 (the event pair:
GPU1's stream event waited on GPU0's stream before the copy node), the final
combine on GPU0 (p0 + p1 + ysh*gate - the shared ran on BOTH but only GPU0's partial
sums it), then the block [D] peer-copied 0->1 (the second event pair - GPU1's forward
needs the block for the next layer's mixed); ~2 event pairs + 2 D2Ds per layer = the
spec's ~0.2-1 ms TP tax, measured at the L4. [AMENDED at r6b - the r6b record above:
the scatter form needs NO block copy - GPU1's only per-layer input is the mixed; ONE
event pair + 2 [D] D2Ds per layer.] (5) THE GRAPH NOTE: the tier already
kills the graphs (the gmode&&tiered check) - the TP form rides the direct launches,
the launch-wall question is the L4's (the r19z segment-graph form would need the
per-side captures + the inter-side waits, a later round if the wall shows). THE DRAFT
BLOCK (r6c): the draft's 512 Q8_0 experts split by ID too (1.25 GB/GPU - the VRAM has
no room for the 2.5 GB replication; the same owner math at the Q8_0 format, the same
dispatch + compact slots + partial/combine in cf_draft_step). THE VERIFY BLOCK (r6c,
the hardest piece): the per-row scratch per side (the verify's forward runs per side -
the replicated rolling states), the union views split by owner (the per-side
sub-unions + the per-side rowmaps over the SAME plane table - the VfyMoeTab per side),
the amortized dots per side (launch_gemv_iq1s_vfy / iq1sh_vfy on the side's stream
over the side's sub-union), the per-row per-side partial ye combines + the both-ways
block combines per layer (the same event-pair discipline). THE GATES (each round):
the host twins (the owner math, the split-plane offset math - the r4 view-gate class,
the compact-slot math), the build + the registers (the family budgets: 64/0 spills),
the L4 battery extension (the cfbat phases at T4Q_CF_IQTP: the smoke, the sweep, the
A/B, the spec wall - the full-pool rate vs the stage-1 prefix, the TP tax measured).
THE COMMIT PLAN: r6a the loader (the split tier + the per-GPU scratch/rolling states +
the core replication + the draft split + the twins; the emission gates T4Q_CF_IQTP
with a THROW until r6b), r6b the greedy emission (the dispatch + the partial/combine
kernels + the sync pairs + the twins), r6c the draft/verify TP forms + the battery
extension. THE HONEST RISK NOTE: the cross-device discipline (the event pairs, the
peer-copy ordering) is L4-only verifiable - the local gates pin the MATH (the owner
arithmetic, the offsets, the compact slots), the L4 battery pins the SYNC (the
byte-match gates at both sides, the intermittent-hang class the 27B's tp_engine
patterns already solved - the ar-mailbox + the event pairs are that engine's own
forms, adopted not invented).

THE L4 BATTERY DRIVER (designed + gated locally, ready for the Saturday window): the
kaggle/cfbat GPU kernel (t4q/tools/stage_cfbat.py -> mkkernel, the sources = the baseline
kernel's libllama + BOTH cfreq pack outputs). THE FORM: the cf1 helpers verbatim (the v7
no-stat downloader, the r19b/r19f watchdog pair, the pre-download static writes) + the
SLAB MERGE (both kernels' layer_*.bin symlinked into ONE dir - the loader reads
layer_%03d.bin contiguous; a 24 GB copy is too slow, symlinks are instant - plus the
manifest validation + the REAL per-layer RMSE stats off the pack's own verify lines) +
the tokenizer (the oracle chatw on the baked prompt, the raw-id fallback if no libllama)
+ THE FIVE PHASES in priority order (a deadline cut loses the least): (1) the resident
smoke + the VRAM ceiling (the IQN=28 probe - the loader's throw carries the free-GiB
number, ceiling = floor(free/0.4641); a SUCCESS extends the sweep); (2) the trunk
baseline time (the Q2_K_S reference tok/s); (3) the IQN sweep time {8,16,24} (+
28 if it fits - the requant-mix rate curve); (4) the QUALITY A/B: gen 64 tokens at
IQN=0 vs IQN=16, the CF_GEN streams compared by the driver (the greedy agreement - the
honest requant-quality gate, no oracle needed); (5) the MTP battery (k=3): the draft
alpha1 smoke (IQN=0), the verify BYTE-MATCH at IQN=16 (THE r4 MTP-consistency bar), the
spec round timers at IQN=0 vs 16 - THE r5 WALL MEASURE: verify_ms before/after the
amortized dots + mean_union (the real pick overlap). THE cf_run EXTENSIONS the battery's
eyes needed (both build-gated): the gen mode ALWAYS prints the CF_GEN stream without an
oracle (the A/B's input), and verify/spec print mean_union (CfVerify's nu_sum/nu_cnt,
accumulated in vfy_window's both paths). THE mkkernel --sources FLAG (the docstring
promised it, the parser never had it - the battery needs the 3-source list). THE LOCAL
GATES: the generated driver ast-parses with the defines verified + the driver's own
parsing logic functionally gated (ceiling_from on the loader's REAL throw form ->
floor(11.62/0.4641) = 25; parse_gen_stream on the r5 CF_GEN form; the CF-json parse; the
manifest RMSE regex on the packer's real verify line); the nvcc build clean (the cf_run/
model changes); test_iq1s unchanged-green. THE SATURDAY SEQUENCE (the window is the
bottleneck): push cfreqa + cfreqb first (the pack runs, ~2.5 h each CPU) -> their
outputs complete -> push cfbat (the sources resolve) -> the battery (~2.5-3 h GPU,
~11 loads) -> the r19w-r19aa kernels ride the rest of the window.
