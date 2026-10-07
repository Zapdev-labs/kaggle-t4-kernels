# PLAN_CF: how fast can CYBER-FROST-3.8 go on Kaggle 2x T4, with fully custom kernels

Written at the cf recon round (2026-10-05), right after `research/cf-arch.md` was verified
against the real GGUF header bytes. Standing directive: run CYBER-FROST-3.8 as fast as this
hardware allows, with this repo's own kernel stack - no llama.cpp engine, no borrowed
codebase. All t4q measured numbers (bandwidths, AR, clocks) carry over from PROGRESS.md.

## 1. What changed vs the Qwen3.8-27B problem

The 27B was a dense hybrid whose whole packed weight set (14.2 GB) fits VRAM with room to
spare, so decode was a pure weight-stream problem and the only lever was bytes/bandwidth.
CYBER-FROST-3.8 (`qwen4exp`) is a 177B-param / ~6B-active MoE: the pool is **77.15 GiB**
(experts 50.6 GiB incl. the Q8_0 MTP graft, the PLE n-gram table 26.85 GiB, everything else
~2.6 GiB) against ~30.7 GiB VRAM + ~29 GB host RAM + /tmp disk. Decode touches only
**~3.02 GB per token** (cf-arch.md section 8), so the arithmetic ceiling is high - the whole
game is WHICH bytes are resident where, and what streams over PCIe/disk at miss time.

| quantity | 27B (qwen35) | CYBER-FROST (qwen4exp) |
|---|---|---|
| hidden | 5120 | 2560 |
| layers | 64 trunk + MTP | 48 trunk + MTP (36 GDN + 12 attn) |
| FFN | dense 17408 | MoE 512 experts x 640, top-10 + shared |
| norms | RMSNorm | hyper-connections (4 streams, LoRA mixers) |
| extra | - | PLE n-gram table (26.85 GiB, 1440 B/token) |
| packed pool | 14.2 GB (all resident) | 77.15 GiB (cannot all be resident) |
| per-token reads | 14.2 GB | ~3.02 GB |
| dense ceiling | ~42 | **~168 tok/s** (2x254 GB/s) |
| MTP k=3 ceiling | ~73 measured / ~250 modeled | **~125-135 tok/s** (draft 0.44 GiB/token) |

## 2. The honest ceiling ladder

Every number below uses measured T4 facts. **cf-m0 is CLOSED (r18, v47): the first-cut
dequant rates and the platform paths are measured, no more assumptions**:

- DRAM per GPU: 278-280 GB/s (re-confirmed on both cards, 80-240 blocks).
- Pinned warm tier (zero-copy, cudaHostAllocMapped): 11.5 GB/s per GPU, **11.53 GB/s
  AGGREGATE when both GPUs read concurrently** (5.76 each); cudaMemcpyAsync 11.37 - the
  shared PCIe/host controller is the wall, not the copy choice. (Corrects the r17
  "~8.4 GB/s per direction per GPU" assumption.)
- Disk (/tmp = overlayfs on the host's 8 TB volume, 87% full): write 0.22 GB/s, seq read
  0.25 GB/s (possibly co-tenant-contended; the 82.85 GB download is ~6+ min write-bound),
  random 2 MiB 1.51 GB/s (may be host-cache-warm; re-measured against the real file at
  cf-m3), random 4 KiB ~91 us (11.0k IOPS), mmap first-fault ~98 us/row (the 16-row PLE
  gather = 1.56 ms/token serial -> the 1-step-ahead async prefetch pipeline is mandatory).
- First-cut dequant GEMV at the real shapes (packed weight bytes, host-model-checked):
  Q2_K 178.8 GB/s at a full grid (102.7 at the underfilled per-expert N=640 launch),
  Q4_0 expert-down 144.2 batched (112.2 as 10 separate launches), Q4_K lm_head class
  88.6 (the repack target is the 150-200 class), Q5_1 127.7.

- **All-active-resident dense, first-cut kernels**: ~21 ms/token = **~48 tok/s**
  (Q2_K 1.30 GB @179 + Q4_0 0.45 @144 + Q4_K 0.34 @88.6 + Q5_1 0.465 @128 + ~3 misc).
- **All-active-resident, cf-m1 kernels** (batched gathers + the Q4_K/Q5_1 repack + launch
  batching): ~13-15 ms/token = **~65-75 tok/s**.
- **All-active-resident + MTP k=2-3** (the draft's own Q8 MoE + the shared lm_head at M):
  the stretch is **~55-80 tok/s**.
- **Two-tier (VRAM hot + pinned warm)**: the warm tier's 11.5 GB/s SHARED ceiling can carry
  only ~10-15% of the expert traffic at 60 tok/s even fully pipelined - the census decides
  whether the top-~250 experts/layer capture enough of the routed mass for the VRAM tier
  to close the rest. Disk-tier misses at the measured 0.25-1.5 GB/s cap the flat-router
  fallback at ~10-25 tok/s (the requant path, section 5, is the escape).
- **The mmap floor** (the pack author's laptop and llama.cpp mmap on the same Kaggle box:
  6-9 tok/s): the floor to beat 5-10x.
- **The cf-m1 staging wall (r19i's static decomposition, computed from the engine + the
  measured 11.53 GB/s + the ~22 us/launch class)**: the MoE stages ~20 MB/layer (10 experts'
  gate|up 1280x840 B + down 2560x360 B) x 48 layers = ~960 MB/token. The host MMAP->pinned
  memcpys ~62 ms/token run GPU-AND-PCIe IDLE (after each layer's router sync, sequential
  with everything); the H2D ~83 ms at 11.53 GB/s overlaps only the ~1 ms/layer of dense GPU
  work; the GPU MoE itself is ~7 ms. The serialized critical path is ~150-165 ms/token =
  the ~6-8 tok/s mmap floor DECODED - the floor is the staging path itself, not the disk.
  The launch count is ~54/layer = ~2600/token (the q8 fast paths doubled the gemv calls
  to quantize+dot); at the measured ~22 us/launch fixed that is ~57 ms/token, HIDDEN under
  the staging today but the NEXT wall after the tiering. Lever order confirmed with
  numbers: (1) the tiering (cf-m3: the resident hits skip the staging entirely; the misses
  pay it), (2) the CUDA-graph capture (the launch overhead pays only after the staging
  drops below it), (3) MTP. The r19e fp32 batched gemv (k_gemv_b) is orphaned dead code
  since r19g's q8 rewire - removed r19i; the tiering lands the eidx-table variants instead
  (the resident-hit batched gemvs read a per-pick resident-index table; the identity table
  reproduces the current uniform-stride behavior bit-exactly, so the default path can adopt
  the mechanism as a stepping stone with zero behavior change).

## 3. The two gating unknowns (cf-m0 and cf-m2)

1. **Kaggle /tmp disk bandwidth** (sequential, random 4 KiB, and 2 MiB expert-sized random
   reads). The download observed 228 MB/s network; the disk tier's real numbers set the
   floor for cold misses and the PLE-table row reads.
2. **Router concentration** P(top-H resident experts cover the routed mass) vs H, per layer,
   on real text (coding prompts, the add-function smoke, generation). With 512 experts and
   top-10 per token this curve decides everything:
   - hot-set 250-300/layer covers >= 95% -> the VRAM tier alone nearly closes the gap
     -> 100-150 tok/s class.
   - covers ~85-90% -> the host-RAM tier carries the misses -> 60-90 tok/s class.
   - flat router (no concentration) -> the disk tier caps at ~10-25 tok/s and the only way
     up is the requant path (section 5).
   Measured by our own engine's router instrumentation at cf-m2 (dump top-10 expert ids per
   layer per token; no assumptions from other architectures).

## 4. The milestone ladder

- **cf-m0 - platform probes, no model** (CLOSED r18, v47, kaggle/cf0): tc_bench grew `--cfprobe`
  (disk write/seq/rand-2MiB/rand-4KiB/mmap-fault, pinned zero-copy per GPU + both + memcpy, the
  dramprobe re-run on both, and the Q2_K/Q4_K/Q5_1/Q4_0 dequant GEMV at the real shapes with
  host-model checks). Verdict: the platform numbers in section 2 + the two probe report-label
  bugs (the raw ms fields were right) fixed in-tree; Q2_K at a full grid 178.8 GB/s (the ~200
  gate marginally missed at the first cut - the mask-dp4a path is correct, the headroom is in
  the derive/load engineering at cf-m1), the launch geometry verdict (per-expert N=640 launches
  are grid-underfilled: the batched gather is mandatory, 102.7 -> 178.8 same kernel).
- **cf-m1 - the exact-forward port, correctness-first** (BUILT r19, the Kagulate gate round
  pushed as kaggle/cf1): the format layer landed (FMT_K2/K4/Q51 in packed.h/deq.cuh/
  repack.cu/gemv_ref.cu/quant_cpu.cpp, all bit-exact transcriptions; the exact-u64 KV parse
  for the PLE hash constants) and the model layer landed (cf_model.h / cf_kernels.cu /
  cf_loader.cu / cf_engine.cu / tools/cf_run.cu, the full podman nvcc build 0 errors,
  8-53 regs zero spill). The layout decisions verified against the ggml semantics: the
  wide residual is stream-major flat [4][2560]; the rope decodes to the SAME partial NeoX
  as the 27B (rope_multi sections [11,11,10,0] with indep_sects=false carries no restart);
  the GDN output gate is sigmoid; the dense attention is 24q/2kv GQA 12. The 512 experts +
  the PLE table stay in the host mmap (cf-m1a is the correctness gate: per token the 10
  experts' raw slabs stage into pinned RAM, one upload, two repack launches into the
  [12800,2560]/[25600,640] staging PackedWs, then the gemvs); the PLE hash runs host-side
  exact-u64 (EOS 248044 cuts predecessors at-or-before it, heads 0-7 bigram 8-15 trigram).
  The tokenizer is owned by the ORACLE (the new chatw job: the GGUF chat template +
  llama_tokenize write the ids), so no t4q tokenizer port exists at all. The Kaggle round
  (kaggle/cf1): the 82.85 GB download, the build, the oracle jobs (chatw x2, seq tail-8
  x2, gen 8 x2), then cf_run seq (rel < 1e-3 + top1 agree vs the oracle's own tbt), gen
  (byte-compare the greedy), time (the steady tok/s), census (the router top-10 dump +
  the local analysis). Gate: byte-identical greedy vs llama.cpp b10975 on both prompts.
  Round state (r19b-l): five oracle-side platform bugs fixed v1-v5; v6 WEDGED 12 h past
  every timeout and burned the weekly 30 h GPU quota - DECODED (r19f): the downloader's own
  p.stat() on the fresh 82.85 GB file held the GIL through a D-state overlayfs stall and
  froze the whole process (thread watchdog included; the GPUs never ran); hardened with a
  separate-process watchdog + a child-stat size probe. TWO payload-killing engine bugs
  were then caught statically pre-quota (the r19l cluster): the Q4_K source-block order
  (c980111: d/dmin were read from the block tail, poisoning the Q4_K token_embd +
  lm_head) and THE BLOCK INPUT (ec1faac: every block projection consumed the raw
  grouped-norm xn instead of the hc_mix OUTPUT s.mixed - the deltanet's qkv/z/beta/alpha,
  the attention's wq/wk/wv, the moe's router + gate|up + shared experts; cf-arch.md
  section 1 is unambiguous that x = hc_mix(res_hc) feeds every block, so every trunk
  output would have diverged). The regenerated payload (ec1faac, byte-verified, sha
  80f407cbc57876d5) now gates base + census + batching + q8 + the dp4a dots + the Q4_K
  fix + the block-input fix + the zero-change identity W-table + the dual-path tiering +
  the hot-set loader + the census emission + the bisect tooling (T4Q_CF_DUMP /
  T4Q_CF_NOFAST), with the failure ladder ordered by behavior-change size: the
  block-input fix (the biggest real change) -> the Q4_K fix (c980111) -> the dp4a
  (942e86d, sim-verified 120/120) -> the zero-change mechanisms (78d480e the identity
  table, e27b610/ec1faac the tiering's OFF path - both bit-exact by construction,
  near-zero suspicion) -> c3fb34c -> 35d8c2d -> 2f33431. The r18 worklist is CLOSED:
  every K-quant/P4 gemv pairs the activation with the oracle's own activation
  quantization (Q2_K/Q4_K <-> Q8_K, Q5_1 <-> Q8_1, Q4_0 <-> Q8_0, the exact ggml
  arithmetic, sim-verified 120/120), and the integer partials run on the T4's IDP.4A
  (r19i, 942e86d: the 27B's own measured class, the P4's factored -8 via the xs sums
  plane, the K4's nibble plane, the Q51's nibble-spread qh fold; the K2's 2-bit extract
  stays scalar by a thin margin). The r19k W-table mechanism + the r19l tiering are
  BOTH LANDED (78d480e + e27b610 + ec1faac): both moe batched gemvs read the pick's
  slab view from a device table wt[blockIdx.y], the identity table reproduces the old
  stride pointers bit for bit, and with a hot-set file loaded the moe is DUAL-PATH
  (the hits read the load-time resident slabs with zero staging; only the misses
  stage, compacted, scaling the memcpys + H2D + repacks with the miss count;
  absent file = OFF = the verbatim full-staging path) - the kernels untouched in
  both steps, and the tiering is complete except the miss-pipeline tuning (the
  sticky-speculation prefetch), which needs the census's churn rate. Next speed
  levers, in order, all AFTER the base gates - the order set by the staging-wall
  decomposition above: (1) the census verdict -> the H choice -> the tiered A/B
  (cf-m3: the staging IS the ~150-165 ms/token wall; the resident hits skip it
  entirely), (2) the CUDA-graph capture of the static-shape sections (the ~57 ms
  launch overhead is hidden under the staging today and becomes the wall only after
  the tiering), (3) MTP (cf-m4).
- **cf-m2 - the census**: the router concentration curve (section 3.2) + the per-bucket step
  trace (the 27B trace method). Verdict: the tier split for cf-m3. Gate: the curve + the
  chosen H per layer recorded, the projected tok/s with a measured miss model.
- **cf-m3 - the tiered engine**: the resident tier (the hottest H experts/layer packed into
  VRAM at load from the census's hot-set file; absent file = OFF = the exact current staging
  path, so the landing is attribution-clean) + the dual-path moe (the hit picks run the
  W-table batched gemvs over the resident slabs - NO staging, NO PCIe; the miss picks pay
  the current MMAP->pinned->H2D staging for only their own rows) + the miss-pipeline
  tuning the census measures (the sticky-speculation prefetch: the previous token's layer-L
  picks prefetched into a double-buffered staging during the dense GPU window, so the miss
  memcpys hide under the H2D; the churn rate decides its value) + the PLE 16-row async
  prefetch after each sampling. The resident layout is the same packed staging shape
  ([H*2*EE] K2 gate|up + [H*D] P4 down per layer), and the W-table mechanism is ALREADY
  LANDED in the default path (r19k, 78d480e: k_gemv_q8k_b + the k_gemv_q80_b table form,
  the identity table = the staged slab views reproduces the old uniform-stride pointers bit
  for bit, zero behavior change) - the tiering only uploads the per-hit resident views into
  the same table, with zero kernel change. Gate: >= 25 tok/s single-stream (3-4x the mmap
  floor), correctness gates intact.
- **cf-m4 - MTP spec**: the draft/verify/rollback wiring on the tiered engine, the n-gram
  table prefetch driven by the sampled token, k tuned on the measured acceptance.
  Gate: >= 40-60 tok/s, byte-identical greedy at every k.
- **cf-m5 - the closure rounds**: the r4-r16 method (every lever A/B'd, every bucket measured
  or roof-closed, PROGRESS.md sections per round). Stretch goals: the TC verify columns at
  the M=4 batch (the int4 mma path exists), the draft's lm_head truncation (0.34 GiB of the
  0.44 GiB draft step) if quality holds, CUDA-graph the whole step.

## 5. The requant stretch (only if the census says the hot set does not fit)

If cf-m2 shows a flat router, the only route to full residency is fewer expert bytes: a
custom ~1.6-2.0 bit/elem expert format. Source = `Blackfrost-AI/CYBER-FROST-3.8-BF16`
(355 GB), streamed tensor-by-tensor over HTTP range reads (never holding more than one
tensor), packed once into a Kaggle Dataset (~25 GiB) and attached to every later kernel -
the same one-time-cost pattern as the baseline kernel's llama.cpp binaries. Gate: the smoke
test (the add function) + greedy agreement vs the Q2_K_S trunk at a temperature of 0 on a
256-token coding prompt; the quality bar is the Q2_K trunk's own, not a bit-identity gate.

## 6. Risk register

- The Q2_K/Q5_1 GEMV rates at the small expert shapes may sit below the P4-class 254 GB/s
  (super-block dequant is more ALU per byte). cf-m0 measures before any design is frozen.
- The host-RAM tier's ~8.4 GB/s is a measured P2P figure; host-pinned reads may differ
  (cf-m0 measures the real number per GPU).
- The worker OOM killer (the r14-r16 lesson): the stage script downloads the 82.8 GB file to
  /tmp (352 s observed), so RAM hygiene and the new-questions-first ordering carry over; the
  PLE table must be opened read-only mmap, never cached in RAM.
- KV is f16 (52 KiB/token incl. the draft layer): at 262k ctx that is 6.7 GiB split across
  the GPUs; long-context runs trade KV against the expert tier.
- The account runs 2 GPU sessions; a session slot must be free for every cf round (retry
  every 60 s on the session-cap error, as before).
- The engine's TP split: attention heads 12+12, DeltaNet value-heads 24+24, experts split
  BY EXPERT ID (each GPU owns half the pool per layer, router on both, top-10 intersected);
  the hc mixers and the MoE partial sums need per-layer all-reduces (2560 floats - half the
  27B's AR width, the r14 AR floor carries over).

## 7. What "done" looks like

A Kaggle kernel that serves CYBER-FROST-3.8-Q2_K_S with this repo's fully custom kernels at
the measured ceiling for its router concentration class - with the ceiling, the tiering
decision, and every lever's A/B recorded in PROGRESS.md, and the correctness gates
(byte-identical greedy vs the llama.cpp oracle, logits KL) passing at every round.
