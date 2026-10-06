# t4q — from-scratch dual-T4 inference engines for the Qwen3.8 family (dense 27B done; CYBER-FROST-3.8 MoE underway)

`t4q` is a custom CUDA inference stack for the Qwen3.8 hybrid (DeltaNet + full attention)
models on quantized GGUF weights, targeting Kaggle's dual Tesla T4 (2 x 16 GB, 70 W power
cap each). No llama.cpp engine, no vendored kernels: custom dp4a GEMV and int8 tensor-core
GEMM kernels, tensor-parallel (TP=2) execution, a P2P mailbox all-reduce that beats NCCL ~2x
on the AR critical path, CUDA-graph decode, batched prefill, continuous batching, and MTP
speculative decoding — validated token-for-token and logit-for-logit against a llama.cpp
oracle built in the same session on the same box.

Two programs live here:

1. **Qwen3.8-27B (dense, `qwen35`)** — complete through M5: the engine, the spec-decode
   closure, and a fully measured performance roofline (see below).
2. **CYBER-FROST-3.8 (MoE, `qwen4exp`, 177B/6B-active)** — the active program: the same
   method (exact decode math from real GGUF bytes, custom kernels, measured everything)
   applied to the fine-tune of Qwen3.8-Flash-Next. Recon is done, the plan is written.

## Qwen3.8-27B: headline speedups (vs llama.cpp `-sm tensor`, same GGUF, same session)

| workload | t4q | llama.cpp | speedup |
|---|---|---|---|
| Single-stream greedy decode (tok/s) | **30.4** | 21.2 | **1.4x** |
| Single-stream + MTP speculative decode, k=3 (tok/s) | **73.7** | 37.9 | **2.4x** (2.3-2.6x) |
| Prompt processing pp512 (tok/s) | **904.7** | 532.6 | **1.7x** |
| Prompt processing pp2048 (tok/s) | **1033.2** | 512.3 | **2.0x** |
| Batched decode, B=16 (tok/s aggregate) | **181.2** | 105.3 | **1.7x** |
| Batched decode, B=32 (tok/s aggregate) | **314.7** | 129.2 | **2.4x** |
| Batched decode, B=64 (tok/s aggregate) | **443.0** | 146.0 | **3.0x** |

Numbers from an independent verification audit (`research/VERIFY.md`) on a P2P box with
llama.cpp a4cb4c61 measured in the same session, same model file, same token ids. On
Kaggle's 2x T4 nodes (the development target, ~9% slower GPUs than the audit box), the
M5 engine holds **66-71 tok/s MTP spec** (k3_dv1, best min-over-prompts 66.35).

### The measured roofline (r14-r16): why the dense engine closes at ~66-71 tok/s on Kaggle

Every claim below is a direct measurement, not a model (PROGRESS.md, rounds 14-16):

- **The node's sustained DRAM read ceiling is ~277 GB/s** (275.0/277.0/277.6 at 80/160/240
  blocks, a pure-read probe in the same run as the anchors). The dp4a GEMV anchors run at
  208-259 GB/s per shape = 75-93% of that ceiling.
- **The instruction-count lever is refuted by direct experiment**: the SASS census found the
  tile loop at 1808 instructions/tile with ~25% tile-invariant x-side overhead; staging the
  `(moff, xd)` table in smem cut the loop to 1691 (-6.5%) bit-identically — and the
  controlled A/B measured **8/8 matrix anchors slower** (-1.3 to -6.3%): the removed ops
  were stall filler inside the dp4a accumulator dependency chains, and the loop is
  latency/dependency-bound, not issue-count-bound. The 277 GB/s ceiling is unreachable by
  instruction cuts; the cut was reverted.
- Every >1 ms step bucket is measured- or roof-closed: AR all-reduce at the PCIe gen3 x16
  floor (~11 us fixed protocol + ~8.4 GB/s marginal), the lm head ratio-closed on the K6
  OR-trick (~161 GB/s at 63% of the P4 class), the attention trio at the graph-launch floor,
  the seg rows fused into the dp4a GEMV at the goal config; tree/ngram/low-bit speculative
  levers refuted on measured roofs.

Milestones (all gated on correctness before speed):

- **Decode engine**: ~30 tok/s single-stream, byte-identical greedy output to the reference.
- **MTP speculative decoding**: 66-73 tok/s — 2.2-2.4x plain, output **byte-identical** to
  plain greedy (zero quality loss), acceptance 0.73-0.82 per draft, 3.2-3.5 tokens/verify.
- **Batched prefill**: W4A8 int8 tensor-core GEMM (`mma.m8n8k16`) reading the decode P4
  layouts in place; ~1000 tok/s at pp2048.
- **Continuous batching**: OpenAI-compatible server; 32 concurrent coding requests at
  **436 total tok/s** (226 generated, 72.5 s wall).

## CYBER-FROST-3.8 (qwen4exp): the active program

`freakyskittle/CYBER-FROST-3.8-GGUF` (a Blackfrost-AI fine-tune of `Qwen/Qwen3.8-Flash-Next`):
177B params, ~6B active, 48 layers (36 DeltaNet + 12 full attention), 512 experts x 640 with
top-10 routing + a shared expert, hyper-connection residuals (4 streams, LoRA mixers, no
layer norms), a 26.85 GiB hashed n-gram PLE embedding table, and an in-file Q8_0 MTP draft
block. The only runnable trunk on 2x T4 class hardware is the 77.15 GiB Q2_K_S.

The recon is complete and written up:

- **`research/cf-arch.md`** — the exact decode math, verified against the real GGUF header
  (parsed over HTTP Range reads: every tensor name/dims/type/offset), the HF config,
  transformers main, and llama.cpp b10975 + master's MTP graph. Includes the complete tensor
  map, the per-token byte budget (~3.02 GB/token), and the gotchas checklist (sigmoid gates,
  the PLE hash multipliers, the dilated PLE conv, the per-stream MTP `eh_proj`).
- **`research/PLAN_CF.md`** — the honest ceiling ladder and the milestone plan. The
  arithmetic ceiling is **~168 tok/s dense / ~125-135 tok/s MTP k=3** if every touched byte
  is VRAM-resident — but the 77.15 GiB pool cannot be, so the game is the tiering: hot
  experts in VRAM, warm tier in pinned host RAM over PCIe (a SHARED 11.53 GB/s wall,
  5.76 each when both GPUs read — corrected by the cf-m0 measurement), the PLE
  table on disk with a 16-row async prefetch (1440 B/token). The two gating unknowns — the
  Kaggle disk bandwidth and the router concentration curve — are measured first (cf-m0,
  cf-m2), never assumed. Milestones: cf-m0 probes -> cf-m1 the exact-forward port with the
  llama.cpp oracle -> cf-m2 the router census -> cf-m3 the tiered engine (gate: >= 25 tok/s,
  vs the 6-9 tok/s mmap floor) -> cf-m4 MTP (gate: >= 40-60) -> cf-m5 closure rounds.

**cf-m0 is closed (measured, `kaggle/cf0`)** and **cf-m1 is built**: the format layer
(FMT_K2/FMT_K4/FMT_Q51 through the whole packed/deq/repack/gemv/cpu-dequant chain, the
exact-u64 KV parse for the PLE hash constants) and the complete model layer
(`t4q/src/cf_model.h`, `cf_kernels.cu`, `cf_loader.cu`, `cf_engine.cu`, `tools/cf_run.cu`)
— the full qwen4exp decode path in this repo's own kernels: the stream-major [4][2560]
hyper-connection residual with the LoRA mixers, the sigmoid-gated DeltaNet, the 24q/2kv
dense attention (its rope decodes to the same partial NeoX as the 27B's), the host-exact
PLE n-gram hash + dilated conv, and the MoE router with the 10-expert pinned staging ->
one upload -> two repack launches -> the gemv combine. Build-validated (0 errors,
zero spill); the Kaggle correctness round (`kaggle/cf1`) is gated by the platform: five
oracle-side issues were found and fixed across v1-v6 (the 26.3 GiB PLE-table prefetch
OOM, the CPU repack OOM, a chatw path bug), and the v6 run wedged past every designed
timeout and consumed the weekly 30h GPU quota — the v7 hardening (a deadline watchdog so
every future round yields its logs, live progress prints, a trimmed oracle scope) plus
the cf-m2 census ride-along are committed and ready to push when the quota resets. The
tokenizer stays owned by the oracle (a `chatw` job writes the ids), so no t4q tokenizer
port exists. No pass or speed claim is made for the engine until the gate runs.

## Correctness (the 27B engine; the same gates apply to CYBER-FROST)

- Greedy output matches the llama.cpp layer-split oracle 256/256 tokens on the coding
  prompt; the one divergence is a near-tie where llama's own two paths disagree with each
  other more than t4q does.
- Logits: mean KL 1.55e-3 vs llama's batch path — 0.29x llama's own internal
  batch-vs-token disagreement floor.
- Repack: 497 tensors, 30272 rows checked, bit-exact on every quant format.
- Batched rows bit-identical regardless of batch composition.
- Speculative decoding output byte-identical to plain greedy at every k; the M4/M8 verify
  columns bit-identical between the dp4a and tensor-core paths.

## Repository layout

- `t4q/` — the engine: CUDA kernels (`src/kernels/`), GGUF loader, TP engine, prefill,
  batched and speculative decode, C ABI, Python driver and tests.
- `research/` — the method, in reading order: `arch.md` (qwen35 exact decode math),
  `cf-arch.md` (qwen4exp / CYBER-FROST exact decode math), `DESIGN.md` (the 27B design),
  `PLAN_500.md` (the 27B spec-decode plan), `PLAN_CF.md` (the CYBER-FROST plan),
  `gguf.md`, `kernels.md`, `baseline.md` (the llama.cpp baseline + binaries),
  `VERIFY.md` (the independent speed audit), and the per-round result tables.
- `kaggle/` — Kaggle kernel staging per milestone (metadata + packed driver scripts).
- `PROGRESS.md` — the dated per-session handoff log: every measured number, every A/B,
  every refutation, rounds 1-16 so far.
- `t4q/tools/mkkernel.py` — packs the tree into a Kaggle kernel per stage.

## Hardware notes (measured on Kaggle 2x T4)

The 70 W software power cap is the dominant constraint: under sustained tensor-core load
the SM clock drops to 300-1050 MHz, so prefill is energy-bound. Decode is bound by the
measured roofs: **~277 GB/s sustained DRAM reads per GPU** (the GEMV anchors at 75-93% of
it), **~8.4 GB/s per direction sustained over PCIe gen3 x16** (the all-reduce and any
host-tier traffic), and dp4a group loops that are latency/dependency-bound (the r16
instruction-cut experiment). P2P remote stores work between the T4s; the custom AR mailbox
sits on them. The account allows 2 concurrent GPU sessions; stage scripts retry every 60 s
on the session cap.
