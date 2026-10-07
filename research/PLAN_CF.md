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
  Round state (r19b-m): five oracle-side platform bugs fixed v1-v5; v6 WEDGED 12 h past
  every timeout and burned the weekly 30 h GPU quota - DECODED (r19f): the downloader's own
  p.stat() on the fresh 82.85 GB file held the GIL through a D-state overlayfs stall and
  froze the whole process (thread watchdog included; the GPUs never ran); hardened with a
  separate-process watchdog + a child-stat size probe. THE v7 THEN WEDGED THE SAME 12 H
  (r19m, Oct 6: 12 h burned, the GPUs never ran - 17,207 clock samples, zero util rows):
  the r19f child-stat was NO defense (the stat syscall D-locks on the fresh inode regardless
  of the process; subprocess.run(timeout)'s kill cannot reclaim a D-state child, and its
  post-kill communicate() waits forever - dl_stat.txt never written), and the process
  watchdog never fired (its mark() writes ride the same stalled disk - a frozen mark blocks
  the kill). THE DRIVER HARDENING (r19m): NO size probe on the fresh model file in ANY
  form (the hf/curl rc certifies the transfer), the static work-file writes moved before
  the download thread starts (the /tmp-health window), the process watchdog KILL-FIRST (no
  disk writes before os.kill), and stream()'s final p.wait() bounded + the abandoned-child
  report (the last unbounded wait on the main path). FOUR payload-killing engine bugs and
  one tooling race were caught statically pre-quota (the r19l cluster + r19m): the Q4_K
  source-block order (c980111: d/dmin read from the block tail, poisoning the Q4_K
  token_embd + lm_head), THE BLOCK INPUT (ec1faac: every block projection consumed the
  raw grouped-norm xn instead of the hc_mix OUTPUT s.mixed - cf-arch.md section 1 is
  unambiguous that x = hc_mix(res_hc) feeds every block), the q8_0 PLANE OVERFLOW
  (7657d1c: the xq0/xd0/xs0 planes sized TOPK*EE = 6400 while every layer's hc_ffn_down
  is Q4_0 with K = 10240 - the generic gemv helper's FMT_P4 branch overran the planes
  into the neighboring scratch, NaN'ing the moe outputs; fixed to the largest gemv K,
  HCD), plus the dump-stream race (the NULL-stream copies read pre-write garbage; the
  z~0 phantom) and the q8_1 ACTIVATION-SUM (0c99eec: k_quantize_q8_1 stored the raw float
  sum while ggml's quantize_row_q8_1_ref stores fp16(d * sum(q)) over the rounded ints -
  the m-terms of the q4_1/q5_1 dots consume it, so every layer's hc_attn_down/up would have
  diverged; the q4_0 dot's m-term is 0). The regenerated payload (the FIFTH, at the 0c99eec
  tree + the hardened driver, byte-verified, sha f88057d6de2c8ef3)
  now gates base + census + batching + q8 + the dp4a dots + all four fixes + the
  zero-change identity W-table + the dual-path tiering + the hot-set loader + the
  census emission + the bisect tooling (T4Q_CF_DUMP / T4Q_CF_NOFAST) + the driver
  hardening, with the failure
  ladder ordered by behavior-change size: the block-input fix (the biggest real change)
  -> the Q4_K fix (c980111) -> the q8_1 sum fix (0c99eec) -> the q8_0 plane fix
  (7657d1c) -> the dp4a (942e86d,
  sim-verified 120/120) -> the zero-change mechanisms (78d480e the identity table,
  e27b610/ec1faac the tiering's OFF path - both bit-exact by construction,
  near-zero suspicion) -> c3fb34c -> 35d8c2d -> 2f33431. ~18 h of the weekly quota
  remains after the v7's 12-h burn: if the v8 wedges the same way, STOP - no more
  pushes until a human-driven diagnosis. The r18 worklist is CLOSED:
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
  sticky-speculation prefetch), which needs the census's churn rate. THE CENSUS VERDICT
  LANDED (r19n, the local L4 battery: the same tree as the passed gates): the routing is
  NEAR-UNIFORM - 168,938 unique (layer,expert) slots of ~168,960 touched in 353 steps, the
  top-64 slots carrying only 10.7% of the draws - so the residency tiering DOES NOT PAY at
  this routing entropy; the mechanism stays landed + attribution-clean (absent hot-set =
  OFF = the verbatim staging path), and the staging wall's fix is the UVA ZERO-COPY of the
  mmap'd experts instead, matching the moe-l2 conclusion. Next speed
  levers, in order - the order set by the staging-wall decomposition above and the census
  verdict: (1) the UVA zero-copy (cf-m3 rebased: cudaHostRegister the mmap'd GGUF expert
  region as mapped host memory and have the moe gemvs read the RAW GGUF bytes directly
  over PCIe - no pinned staging, no H2D, no repack, the single touch; ~960 MB/token over
  the T4's gen3 x16 (~12.8 GB/s) ~ 75 ms vs the ~150-165 ms staged path, a ~2x staging
  cut; the L4's gen4 makes it ~38 ms), (2) the CUDA-graph capture - the r19p LAUNCH CENSUS
  (static, from the engine code; the full table is the PROGRESS r19p record): ~2,578
  launches/token (54/layer on the 36 GDN layers, 51 on the 12 attn layers, the PLE 12, the
  head/tail 10) x the measured ~22 us/launch = the ~57 ms wall CONFIRMED independently of
  the r19i decomposition, and the step splits into ~49 sync-bounded segments (the moe
  router's D2H + host softmax/top-10 x48 + the logits sync x1) with the ~1,440 host
  staging memcpys (~62 ms) riding the windows; every launch's grid is CONSTANT ACROSS
  STEPS (the fixed families' grids are compile-time, the gemv grids are the load-time
  W.rows) - only VALUES vary (pos in the 3 attention kernels, fixable by a device
  step-params buffer the graph's own memcpy node refreshes; the W-table/we contents are
  host tables the graph's memcpy nodes re-upload) - so BOTH graph forms are shape-viable:
  **G1** (pre-UVA, the ~50 per-segment graphs: the pos-params fix + the pinned tables +
  the capture wrapper; the launch wall ~57 ms -> ~1 ms while the staging wall stays -
  the staged ~150-165 -> ~95-110 ms/token class; the no-UVA HEDGE if the probe's ratio
  disappoints) and **G2** (post-UVA, the full-step single graph: the device-side router -
  the softmax/top-10/the W-table view build/the we renorm as kernels - kills the 48
  mid-step syncs so the segments merge, and the staging memcpys are already dead) = ~1
  replay/token (the T4: ~75 ms UVA reads + ~7 ms GPU + ~1 replay ~ ~83 ms/token class).
  G2 ORDER-EXACTNESS SPEC (r19t): the device-side router MUST reproduce the host loops'
  exact fp accumulation ORDER - the softmax sum and the we renorm accumulated
  sequentially over i=0..NE-1 (a single warp's serial loop, ~2-5 us - the free
  correctness), and the top-10 scan keeping the host's strict-> first-max tie rule -
  because a warp-parallel/tree reduction reorders the sum and perturbs the renormalized
  we by ~1 ulp, which feeds moe_out -> a ~1e-6-class logit perturbation -> the near-tie
  argmax flips at a ~1-per-1e3-1e4-token rate: the SHORT gen gates (8 tokens) PASS while
  the long generations silently diverge from the byte-identical-greedy bar (the r19n
  battery's own flips sat at gaps 2.68/0.49 under the +-2..5 q8 noise; the G2
  perturbation is ~6 orders smaller but the bar is byte-identity, not statistics). The
  W-table view build is integer-offset arithmetic - exact by construction, no risk. The
  G1 form keeps the host router as-is (no new arithmetic, no risk).
  G1 LANDED (r19v, build-clean 0 errors, OFF by default - absent T4Q_CF_GRAPH = the
  verbatim direct-emission path): the emission/driver split (one op source, two drivers -
  the moved verbatim bodies: emit_head/emit_ple_kernels/emit_pre/emit_router/emit_moe_rest/
  emit_post/emit_tail + ple_host/host_router holding ALL the per-step host state), the 49
  sync-bounded segment graphs captured at the FIRST step right before each segment's first
  replay (the ops recorded, not executed; every arg a fixed steady-state buffer - the census
  found no varying arg after the pos device word, and every varying memcpy content rides a
  pinned host source the captured H2D nodes re-carry at each replay), the host windows
  between replays exactly the direct path's host work (the sync + the router softmax/
  top-10/we + the census + the OFF staging memcpys + the PLE gather). The gmode exclusions:
  tiered (the miss-varying H2D sizes are not capture-constant; the census verdict says the
  tiering pays ~nothing at this entropy anyway) and T4Q_CF_DUMP (the mid-step D2H probes are
  not capture-legal). The graphs survive cf_reset; cf_free destroys them. Two self-review
  catches landed pre-battery: the h_params[0] = pos write must live in the DRIVER (the
  emission runs only at capture time in the graph path - the stale-pos trap), and it must
  precede the params H2D's ENQUEUE (a pinned async copy reads its source at execution time -
  the write-after-enqueue race). Runtime verification: the gates with T4Q_CF_GRAPH=1 (the
  greedy byte-compare vs the OFF path) + the A/B tok/s on the L4, then the T4 at the next
  quota window; the graph_probe's measured differentials decide the adoption class.
  The UVA stays first (the bigger cut, and it unlocks G2); G1 is the hedge that does not
  need it, (3) MTP (cf-m4).
  PROBE (r19s, landed: t4q/tools/graph_probe.cu + the build/graph_probe target, compiles
  clean in the 12.8 podman): the census-shaped chain itself (every 4th launch a gemv-ish
  kernel streaming a ~13 MB slab - the packed-slab read class of one moe gemv, the rest
  the tiny norm class, one stream so the chain is strictly serial) measured three ways,
  the kernel work identical in all three: (A) launch-by-launch (the current engine's
  form), (B) ONE captured full-step graph (the G2 form), (C) ~50 per-segment graphs with
  a cudaStreamSynchronize + the ~30 us router host loop between replays (the G1 form) -
  the differentials (A)-(B)/(A)-(C) are the pure graph reclamation on this hardware (the
  ~22 us/launch class was inherited from the 27B-era measurements), the instantiate
  times reported as the one-time load cost. Run: `./build/graph_probe [2578] [50]` on
  each host (the L4 now, the T4 at the next quota window).
- **cf-m2 - the census**: VERDICT LANDED (r19n): near-uniform routing (168,938/168,960
  slots touched in 353 steps; the top-64 = 10.7% of the draws) - no tier split pays; the
  UVA zero-copy is the lever. The census tooling stays (any future model/file re-checks
  the same way).
- **cf-m3 - the UVA zero-copy moe** (REBASED from the tiered engine on the r19n census
  verdict; the tiering mechanism stays landed + OFF + attribution-clean but pays ~nothing
  at this entropy): cudaHostRegister(cudaHostRegisterMapped) the mmap'd GGUF expert
  region once at load, and the moe gemvs read the RAW GGUF blocks directly from the mapped
  host pages over PCIe - the dequant fused into the gemv (the block decode + the dp4a dot
  per block, the r19i dot arithmetic unchanged) - eliminating the per-step MMAP->pinned
  memcpys (~62 ms), the H2D (~83 ms), AND the repack launches, the single PCIe touch
  (~960 MB/token ~ 75 ms on the T4's gen3 x16 vs the ~150-165 ms staged total). The
  dual-path/W-table mechanism already supports per-pick views; the raw-reading gemv
  becomes the third path (OFF by default until the A/B passes). The PLE 16-row async
  prefetch after each sampling stays on the list. PROBE FIRST (r19o, landed:
  t4q/tools/uva_probe.cu + the build/uva_probe target, compiles clean in the 12.8 podman):
  the mapped-read bandwidth vs the staged H2D+read on the same warm page-cache bytes - the
  A/B ratio decides the form (>= ~0.6 supports the minimal pointer-swap UVA: cudaHostRegister
  (Mapped) the expert region once at load + the repack's input pointer swapped to the
  mapped pages - the SAME repack + the SAME gemvs, byte-identical by construction, no
  pinned staging, no H2D; a halved ratio -> the fused raw gemv (the dequant fused into
  the dot, the repack eliminated) or the staged path stays). Run on the L4 host:
  `./build/uva_probe <model.gguf> 0 2 b` (the registration time, the RSS growth, the
  ratio all print). Gate: the correctness gates intact +
  the staged-vs-UVA A/B tok/s on the same hardware (the L4 locally, the T4 on the next
  quota window).
  THE RAM CAP (r19r, from the tensor map): the trunk's expert region is 47.9 GiB (the
  per-layer 1,022 MB: the gate 275.25 + the up 275.25 + the down 471.86, x 48) - MORE
  than the Kaggle host's ~29 GB RAM, so the FULL registration does not fit on the
  TARGET platform; the full-region form is the big-RAM host's (the probe's RSS growth
  + registration time decide that host's feasibility). THE KAGGLE FORM: the PARTIAL
  registration - the first N layers' expert spans (the layer-granular contiguous
  ~1 GiB/layer spans), N driven by the measured usable RAM (~28-29 layers at ~28.8 GiB,
  ~60% of the pool), carried by the LANDED dual-path/W-table mechanism: the registered
  layers' picks read via the alias (the repack's input = the per-pick alias offsets),
  the unregistered layers run the verbatim staged path. The honest Kaggle ladder: the
  mixed staging ~29 x 1.56 + ~19 x 3.02 ~ 103 ms/token + ~7 GPU + the segmented graphs
  (the staged layers keep their router syncs, so the full-step G2 graph is the
  full-UVA host's; the Kaggle stays G1-class) ~1 -> ~110 ms/token ~ 9 t/s, vs the
  full-UVA host's ~83 ms (~12 t/s, the G2). The probe's RSS output feeds N per host.
  LANDED AS THE THIRD PATH (r19u, build-clean 0 errors, OFF by default - absent
  T4Q_CF_UVA_LAYERS = the verbatim staging path): the loader registers the first n
  layers' expert tensors as ONE coalesced page-aligned cudaHostRegisterMapped span (the
  per-tensor page-spans of the file-adjacent tensors would double-register on the shared
  boundary pages), and the moe's UVA branch reads the picks' raw slabs through the device
  aliases by the per-pick-id scatter repack (k_repack_eid_q2k/q4 in repack.cu: the
  address math is exactly the OFF path's memcpy sources, the same repack block decode,
  the same identity W table + the same gemvs - byte-identical by construction; only the
  read path changes: no host memcpys, no H2D, no raw staging, +2 launches/layer for the
  eid upload + the two scatter repacks). Runtime verification: the gates + the A/B on
  the L4 (T4Q_CF_UVA_LAYERS=n, n=48 on a big-RAM host / n~28-29 on the Kaggle), then the
  T4 at the next quota window; the v9 payload regen carries it after the L4 passes.
- **cf-m4 - MTP spec**: the DESIGN IS FROZEN (r19q, `research/CF_MTP.md` - the
  pre-implementation spec: the draft block's exact forward (cf-arch section 6), the Q8_0
  gemv path ALREADY IN-REPO and gate-proven (the repack's GT_Q8_0 case + the FAST_Q8
  int8-dp4a gemv family, the 27B's byte-identical spec gates), the activation pairing
  (the existing launch_quantize_q8_0, the symmetric form), the eh_proj gather (the only
  new op), the resident 2.49 GiB draft experts (staging-free ~2-3 ms draft steps), the
  catch-up (pending_h ring + the draft's KV over the prompt), the batched verify's
  kernel surface (the 27B's k_pf templates) with THE GATE (the batched rows MUST
  reproduce the sequential decode bit-exactly - the near-tie argmaxes flip otherwise),
  the rollback state list, and the speed math (the UVA'd T4: k=3 ~ 55 ms/token ~ 18
  t/s, a ~1.5x over the G2 engine)). Implementation: the draft/verify/rollback wiring
  on the UVA engine, the n-gram table prefetch driven by the sampled token, k tuned on
  the measured acceptance. Gate: >= 40-60 tok/s, byte-identical greedy at every k.
  THE DRAFT BLOCK LANDED (r19w, build-clean 0 errors/0 warnings, OFF by default - absent
  T4Q_CF_MTP=1 at load the draft is not loaded and the 2.49 GiB stays free): the CfDraft
  struct (a CfLayer at il=48/attn=true + the nextn extras + the OWN scratch + the
  all-512 resident slabs + the per-step W tables + the own KV pos), the loader wiring
  (the weights/nextn extras via the normal upload path, the 512 experts through the
  SAME chunked raw->pinned->H2D->repack pass as the hot-set tier - the trunk-sized
  raw_stage takes 3 Q8_0 draft experts per pass, the layout math self-reviewed against
  the physical region boundaries: 10.75 MB gu / 3.48 MB per expert = 3, 9.22 MB dn /
  1.74 MB = 5, ch = 3), the FMT_Q8 case in gemv() (the Q8_0 activation pairing with the
  gate-proven FAST_Q8 dot), the ONE new op k_cf_eh_gather (the per-stream
  [e_norm ; h_norm_s] concat, 19 regs 0 spills), the host_top10 extraction (ONE
  order-exact softmax/top-10/we source shared by the trunk's host window and the
  draft's forward - the draft's routing MUST be the same order-exact form), and
  cf_draft_step: the exact CF_MTP section 1 forward - the pair (x_q, h_{q-1}) with
  h_{-1} = 0 (the position-0 pair), the e host-dequant, e_norm/h_norm, the gather, the
  4 eh_proj gemvs composing res', the attention twin (the trunk's exact op list on the
  draft's own KV/scratch, the pos word riding d_params[1] - the spare slot, written in
  the driver BEFORE the upload per the r19v discipline), the MoE via host_top10 + the
  ALL-RESIDENT W-table compose (the tiering's mechanism, all hits, staging-free) + the
  gated shared expert, the draft's OWN final mixer, the SHARED lm_head, the logits D2H
  into the draft's pinned row. cf_reset rewinds the draft's pos; cf_free drops the
  pinned rows (the device buffers fall to the cudaDeviceReset, the trunk's own style).
  The shared host router buffers (h_router/eid/we_h) are safe because the draft step
  NEVER interleaves a trunk step mid-flight (the smoke is strictly sequential; the
  future speculative driver must keep that rule). THE ACCEPTANCE SMOKE: cf_run's new
  `draft` mode - the prompt + n greedy with the per-pair draft forward interleaved
  (the pre-loop (ids[0], 0) pair, then after each trunk step at i the lagged 1-step
  compare (the draft's prediction from its pair at i vs the run's ACTUAL token at
  i+1) + the next pair (x_{i+1}, h_i = the trunk's pre-final-mixer residual)), printing
  alpha1 (the 1-step acceptance the whole MTP speed math rides on) + the draft/trunk
  mean ms. Runtime verification (the L4): T4Q_CF_MTP=1 + the draft mode -> the measured
  alpha1 decides the adoption class (the 27B's class ~2.2 accepted/verify); the draft
  forward's byte-exactness rides the trunk's own argument (every op is the trunk's op
  with the draft's own buffers - the same kernels, the same args, the same order - and
  the Q8_0 pairing is the 27B's gate-proven arithmetic). STILL AHEAD: the catch-up
  (pending_h ring), the batched verify (THE GATE: bit-exact vs sequential), the
  rollback, the speculative driver, k tuned on the measured acceptance.
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
