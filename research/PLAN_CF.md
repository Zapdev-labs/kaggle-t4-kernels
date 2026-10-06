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

Every number below uses measured T4 facts (254 GB/s per GPU at the real shapes, ~8.4 GB/s
sustained PCIe per direction, 15.36 GiB usable VRAM each). The two unknowns that gate the
ladder are measured at cf-m0/cf-m2, never assumed:

- **All-active-resident dense**: 3.02 GB/token -> 5.95 ms -> **~168 tok/s**.
- **All-active-resident + MTP k=3**: ~4.3 GB per ~3.4 accepted tokens -> **~125-135 tok/s**.
- **Two-tier (VRAM hot experts + host-RAM warm tier)**: miss bytes cross PCIe at ~8.4 GB/s.
  Equilibrium: ~78 tok/s at 90% VRAM hit, ~150 at 95%, disk-floor below that.
- **The mmap floor** (what naive streaming gives, the pack author's laptop and llama.cpp
  mmap on the same Kaggle box: 6-9 tok/s): the floor to beat 5-15x.

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

- **cf-m0 - platform probes, no model**: tc_bench grows `--cfprobe`: /tmp disk (seq/rand4k/
  2 MiB), pinned-host-RAM read bandwidth from each GPU, and the new dequant GEMV shapes:
  Q2_K and Q5_1 kernels (the engine has Q4_0/Q4_K-class paths already) at the real expert
  shapes ([2560,640] gate/up, [640,2560] down, the 10-expert gather-GEMV pattern, the
  [2560,512] router, the hc LoRA [10240,320]/[320,10240] pair). Gate: Q2_K GEMV >= ~200
  GB/s at the gate/up shapes; disk + host-RAM numbers recorded.
- **cf-m1 - the exact-forward port, correctness-first**: GGUF loader for the Q2_K_S trunk
  (Q2_K/Q4_0/Q5_1/Q4_K/F16), the repack into the engine layout (experts stored per-expert
  contiguous for the gather + the tier manager; the PLE table split out; the inert indexer
  tensors dropped), and the kernel set: hc mixer/combine, PLE (gather + signed-sqrt gate +
  dilated conv), DeltaNet (the 27B port at D=2560 + sigmoid gate), dense GQA flash-decode
  (24q/2kv, head 256), the MoE router + 10-expert gather-GEMV + gated shared expert, the
  final hc mixer + lm_head, the MTP draft + verify + rollback (DeltaNet S, conv, PLE
  history). Slow-but-correct tiering (synchronous misses). Oracle: llama.cpp b10975+master
  built on the Kaggle box (the baseline kernel pattern), greedy 256-token byte-compare +
  logits KL on a short prompt, plus the arch.md section-6-style cross-checks (tiled V
  permutation, ssm row order) against the BF16 safetensors via HTTP range reads.
  Gate: byte-identical greedy vs llama.cpp; a working tok/s number (expected single digits,
  the mmap class).
- **cf-m2 - the census**: the router concentration curve (section 3.2) + the per-bucket step
  trace (the 27B trace method). Verdict: the tier split for cf-m3. Gate: the curve + the
  chosen H per layer recorded, the projected tok/s with a measured miss model.
- **cf-m3 - the tiered engine**: the VRAM LRU (hottest H experts/layer), the pinned host-RAM
  warm tier, the async miss pipeline (the router for layer L+1 runs during layer L's MoE so
  the 10 expert rows prefetch one layer ahead), the PLE 16-row async prefetch after each
  sampling. Gate: >= 25 tok/s single-stream (3-4x the mmap floor), correctness gates intact.
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
