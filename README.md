# t4q — a from-scratch dual-T4 inference engine for Qwen3.8-27B

`t4q` is a custom CUDA inference stack for the Qwen3.8-27B (hybrid DeltaNet +
attention) model on quantized GGUF weights, targeting Kaggle's dual Tesla T4
(2 x 16 GB, 70 W power cap each). It includes custom dp4a GEMV and int8
tensor-core GEMM kernels, tensor-parallel (TP=2) execution, CUDA-graph decode,
batched prefill, continuous batching, and MTP speculative decoding — validated
token-for-token and logit-for-logit against a llama.cpp oracle.

## Headline speedups (vs llama.cpp `-sm tensor`, same GGUF, same session)

| workload | t4q | llama.cpp | speedup |
|---|---|---|---|
| Single-stream greedy decode (tok/s) | **30.4** | 21.2 | **1.4x** |
| Single-stream + MTP speculative decode, k=3 (tok/s) | **73.7** | 37.9 | **2.4x** (2.3-2.6x) |
| Prompt processing pp512 (tok/s) | **904.7** | 532.6 | **1.7x** |
| Prompt processing pp2048 (tok/s) | **1033.2** | 512.3 | **2.0x** |
| Batched decode, B=16 (tok/s aggregate) | **181.2** | 105.3 | **1.7x** |
| Batched decode, B=32 (tok/s aggregate) | **314.7** | 129.2 | **2.4x** |
| Batched decode, B=64 (tok/s aggregate) | **443.0** | 146.0 | **3.0x** |

Numbers from an independent verification audit (`research/VERIFY.md`) run on a
P2P box with llama.cpp a4cb4c61 measured in the same session on the same model
file and token ids.

Milestone highlights:

- **Decode engine**: ~30 tok/s single-stream, byte-identical greedy output to
  the validated reference path.
- **Speculative decoding (MTP)**: **73.7 tok/s** — 2.4x plain t4q decode on the
  same box, with output **byte-identical** to plain greedy (zero quality loss).
  Acceptance 0.73-0.82 per draft, 3.2-3.5 tokens per verify step at k=3.
- **Batched prefill**: W4A8 int8 tensor-core GEMM (`mma.m8n8k16`) reading the
  decode P4/P4M/K5 weight layouts in place; ~1000 tok/s at pp2048.
- **Continuous batching**: an OpenAI-compatible server with a
  continuous-batching scheduler; 32 concurrent coding requests completed at
  **436 total tok/s** (226 generated tok/s, 72.5 s wall).
- **Custom all-reduce**: a P2P mailbox (remote stores + epoch flags) beating
  NCCL by ~2x on the AR critical path, with an automatic host-mapped fallback
  for boxes without P2P.

## Correctness (validated, not just benchmarked)

- Greedy output matches the llama.cpp layer-split oracle 256/256 tokens on the
  coding prompt; the one divergence is a near-tie where llama's own two paths
  (batch vs token-by-token) disagree with each other more than t4q does.
- Logits: mean KL 1.55e-3 vs llama's batch path — 0.29x llama's own internal
  batch-vs-token disagreement floor.
- Repack: 497 tensors, 30272 rows checked, bit-exact on every quant format.
- Batched rows are bit-identical regardless of batch composition.
- Speculative decoding output is byte-identical to plain greedy at every k.

## Repository layout

- `t4q/` — the engine: CUDA kernels (`src/kernels/`), GGUF loader, TP engine,
  prefill, batched and speculative decode paths, C ABI, Python driver and tests.
- `research/` — design docs, per-milestone result tables, the verification audit.
- `kaggle/` — Kaggle kernel staging (metadata + packed driver scripts).
- `PROGRESS.md` — dated per-session handoff log with all measured numbers.
- `t4q/tools/mkkernel.py` — packs the tree into a Kaggle kernel per stage.

## Hardware notes

The 70 W software power cap on Kaggle T4s is the dominant constraint: under
sustained tensor-core load the SM clock drops to 300-1050 MHz, so prefill speed
is energy-bound, not occupancy-bound. The decode GEMV path (dp4a) stays
memory-bound and holds 250-260 GB/s of packed weight bytes even when throttled.
