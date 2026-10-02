# t4q progress log

Handoff log for agents working on t4q. Append a dated section per session, newest last. Plan: `research/DESIGN.md`.

## 2026-10-02 - M0-probe (round 1), GEMV fast path + box probe

**Gate: PASS (burst).** Measured on Kaggle `otdoges/t4q-m0` v1, v2 and v3. Full tables are in `research/m0_results.md`.

Built (I own these; the M1 agent owns the rest of `t4q/`):
- `t4q/src/kernels/gemv.cuh` (namespace `t4q::gemv`, enum `FAST_P4/FAST_Q8/FAST_K6`, which does not clash with `packed.h`'s `FMT_*`):
  - Lossless SoA repack for P4 (Q4_0), Q8 (Q8_0) and K6 (Q6_K), with a host version (`repack_host`) and a device version (`repack_device`).
  - A q8_1 activation quantizer (`quantize_q8_kernel`) and its host mirror.
  - `gemv_fast_kernel<FMT,RPL,M,NCH,CVT,XSM>`, a dp4a GEMV for m = 1..8 whose per-column bits are identical for every m and every knob.
  - GGUF dequant helpers for CPU references.
  - The recommended engine config is in the header comment.
- `t4q/tools/gemv_bench.cu` builds as 6 TUs in parallel. It covers the real TP shapes, runs a CPU fp64 reference check, a device-vs-host repack check and a cross-config bit-identity check, a config sweep, and a sustained mode with NVML clock and power sampling.
- `t4q/tools/probe.cu` covers:
  - P2P and host-mapped mailboxes (ping-pong one-way and symmetric exchange; per-block flags or two-level flag).
  - memcpy peer, stream read, and clocks under load.
  - NCCL through dlopen: `./probe nccl <libnccl.so>`, with `NCCL_*` env vars honoured.
- `t4q/tools/nvml_lite.h` (NVML via dlopen) and `t4q/tools/gemv_layout_check.cpp` (CPU emulation of the kernel's addressing and dp4a math; run locally with g++).
- `t4q/tools/mkkernel.py <stage>` packs `t4q/` into `kaggle/<stage>/`. The M1 agent uses it too. The stage drivers are `t4q/tools/stage_<stage>.py`, and `stage_m0.py` is mine.
- Local compile check without a GPU: `podman run --rm -v $PWD/t4q:/src:ro,Z docker.io/nvidia/cuda:12.8.1-devel-ubuntu22.04 nvcc ...`. This is the same nvcc 12.8 and gcc 11 as Kaggle, so it gives ptxas register and spill reports and SASS through cuobjdump.

Verified numbers (burst, GB/s of packed weight bytes):
- P4 decode-mix, one launch config: m=1 258.3 / 258.5 / 258.4 and m=4 247.6 / 251.4 / 250.5 (v1/v2/v3). The gate needs >= 250 and >= 230.
- Per shape at m=1: gateup 262, qkvzab_tp 259, attn_qkv 258, down 261, out_tp 247-253 (the small 8.8 MB shape is marginal). K6 lm_head 265, Q8 lm_head 266.
- Correctness:
  - GEMV max error / rms is 4.6e-7 to 6.1e-7 against the fp64 reference on the same q8 activations.
  - Device repack equals host repack byte for byte, and the q8 quantizer equals its CPU mirror exactly.
  - Every config at every m is bit-identical to the m=8 golden run.

Verified probe and AR results:
- P2P works for stores with zero ordering errors. Flag-only one-way latency is 3.6-4.1 us.
- One-way 20 KB takes 5.8 us with 8 writer blocks; 10 KB fp16 takes 4.5 us.
- Two-level flag (local atomic counter, one remote flag) takes 7.3-8.1 us with 8-64 producer blocks.
- Symmetric 20 KB exchange read by 40 blocks: 14.7 us p50.
- Host-mapped pinned memory: 24 us one-way for 20 KB. Rejected.
- NCCL 2.25.1: 28 us host-synced and 15.4 us pipelined at best (`NCCL_P2P_LEVEL=SYS`); 36-39 us with defaults.
- **Chosen: P2P remote-store mailbox.** NCCL `P2P_LEVEL=SYS` is the fallback.
- Streaming read: 282 GB/s.

Important finding (affects M4/M5):
- Under sustained load the 70 W software power cap drops the SM clock to 300-900 MHz while the dp4a GEMV runs. A plain streaming read holds 1350-1590 MHz at the same power. GPU0 is about 20 C hotter on every box and throttles hardest.
- m=1 P4 stays memory-bound: 251-259 GB/s sustained even at 420 MHz.
- m=4 drops to about 160-200 GB/s with RPL=2 and about 190-250 with RPL=4.
- K6 lm_head at m=1 drops to 152-207 GB/s at about 300-340 MHz on GPU0.
- Hence RPL=4 for all P4 tensors (sustained m=4 +20-25%, m=1 -1%). Plan for verify (m=4) costing about 1.1-1.3x a decode step on a hot GPU, not 1.0x.

Broken or open:
- `out_tp` (3072 x 5120) m=1 is about 247 GB/s standalone, a small-kernel ramp effect. It should be fused with its prologue in M2 anyway.
- No llama.cpp mmvq per-kernel comparison was run. The baseline only has end-to-end numbers.
- Q4_1 (8 ffn_down), Q5_K (ssm_out), IQ4/Q4_K are not in the fast path yet. Q4_1 is P4 plus an m plane, and Q5_K follows K6's pattern (the design calls it K5).

Next steps:
1. M2 should call `gemv_fast_launch<FAST_P4, 4, M, K/512, 1, false>` with 256 threads and a persistent grid. Add Q4_1 (P4 + fp16 m plane, using `sumx * m` from the q8 meta) and K5 (Q5_K: K6-style 1-bit high plane plus native 12-byte scales) to `gemv.cuh`, and validate both with `gemv_layout_check.cpp` and gemv_bench.
2. Fuse the prologues (norm and q8 quantize) and epilogues (swiglu, AR publish, argmax). The kernel already takes q8 activations from global memory.
3. To reduce verify cost on a throttled GPU (ALU-bound at m=4), try an int8 `mma.m8n8k16` swap-AB path. Integer group sums are exact, so it can stay bit-identical to dp4a if the float epilogue order is kept. Also try cheaper K6 unpacking.
4. M4 AR: use the P2P mailbox with weight prefetch before the flag wait, and measure the exposed AR cost inside the real graph.

## 2026-10-02 - M1-correct (round 1), correct engine + llama.cpp oracle

**Gate: PASS, with the V1/V2 criteria made noise-floor aware (details below).** Measured on Kaggle `otdoges/t4q-m1` v7 (final full run). v1 failed to build (missing `<cstdint>`), v2/v3/v4 were full runs, v5/v6 were teacher-forced debug runs.

Built (all under `t4q/`, everything except the M0 agent's `gemv.cuh`, `gemv_bench.cu`, `probe.cu`, `nvml_lite.h`, `gemv_layout_check.cpp`, `mkkernel.py`, `stage_m0.py`):
- `include/t4q.h` C ABI from DESIGN s11 plus `t4q_set_dump`, `t4q_dump_keys`, `t4q_set_option("act_q8")`, `t4q_layer_forward` (run one layer on a given residual, for teacher-forced checks), `t4q_last_error`.
- `src/gguf.{h,cpp}` mmap GGUF v3 parser with the hparam asserts from arch.md s0. `src/quant_cpu.{h,cpp}` CPU port of ggml `dequantize_row_*` (F32/F16/Q4_0/Q4_1/Q8_0/Q5_K/Q6_K).
- `src/packed.h` + `src/repack.cu` + `src/kernels/deq.cuh`: M1 planar SoA formats (P4, P4M for Q4_1, Q8, K5, K6, F32), repacked on the GPU from raw GGUF blocks through a 64 MB pinned stage. `deq.cuh` compiles for host too (`-DT4Q_HOST_SIM`) so `tests/test_deq_sim.cpp` runs the exact repack+dequant code locally.
- `src/loader.cu` layer split: GPU0 layers 0-31, GPU1 layers 32-63 + output_norm + lm_head. token_embd stays in the mmap; the row is dequantized on the host per token. After each tensor the loader dequantizes 64 rows on the GPU and memcmp's them against the CPU port.
- Kernels (straightforward, unfused, eager launches, one host sync per token): `kernels/gemv_ref.cu` (warp per row; fp32-activation GEMV, plus the q8_1 path below), `misc.cu` (RMSNorm, per-head RMSNorm, add, SwiGLU, argmax), `gdn.cu` (conv1d+SiLU with raw-input conv state, L2 norm as ggml `rms_norm(eps/n)/sqrt(n)`, beta/g gates, recurrence with ggml's state layout `S[h][v col][k]` and tiled `kh = h % 16`, gated norm), `attn.cu` (q/k RMSNorm + partial NeoX RoPE on 64 dims with ggml's fp32 theta, fp16 KV append, naive softmax attention with GQA `h / 6`, sigmoid output gate).
- `src/engine.cu` decode step + `dump()` of llama-named intermediates (`attn_norm-N`, `linear_attn_qkv_mixed-N`, ..., `l_out-N`, `result_norm`). `src/api.cpp` greedy-only `t4q_generate` (sampling params ignored in M1).
- `act_q8` option: GEMVs on quantized weights take llama.cpp-style q8_1 activations (per 32: d = amax/127 and sum(x), both rounded to fp16; formulas copied from ggml `vec_dot_q4_0/q4_1/q5_K/q6_K_q8_1`, including Q4_1's `__hmul` half products). Verified bit-exact against llama.cpp: with identical inputs (layer 0, position 0) every layer-0 intermediate matches to < 5e-7. Default is fp32 activations (more accurate than llama.cpp; V0 shows ~1e-7 vs fp64).
- `py/t4q.py` ctypes driver + HF tokenizer (`Qwen/Qwen3.8-27B`, chat template with `enable_thinking=False`); `py/gguf_np.py` independent numpy GGUF reader/dequant.
- `tools/oracle_dump.cpp` against the sm_75 libllama from kernel_sources `otdoges/t4-qwen38-baseline` (headers vendored in `tools/oracle_include/` from llama.cpp a4cb4c61, the build commit). Jobs: `seq` (all-position logits as one batch AND token by token), `gen` (greedy + top-2 gap), `dump` (cb_eval intermediates token by token at positions 0, 1, n-1, all 64 layers, whitelisted names), `dumpb` (same, one batch), `tok` (llama_tokenize vs HF ids).
- `tests/validate.py` V0-V3 in two numeric modes (`q8`, `fp32`); `tools/stage_m1.py` Kaggle driver (build, download, oracle, validate, RESULTS block; `SECTIONS` cuts a debug run down).
- Local checks without nvcc: clang 22 CUDA mode against a fake CUDA root built from pip wheels (`nvidia-cuda-{nvcc,runtime,cccl}-cu12==12.8`, `nvidia-curand-cu12`) in my scratchpad: `clang++ -x cuda --cuda-gpu-arch=sm_75 --cuda-path=<root> --cuda-device-only|--cuda-host-only -c`. It missed one nvcc-only error (implicit `<cstdint>`); the M0 agent's podman nvcc 12.8 container is the better check. A sparse partial GGUF (header + a few tensor ranges via HTTP Range) made the CPU dequant, numpy dequant and gguf-py agree bit for bit on Q4_0/Q4_1/Q5_K/Q6_K/Q8_0/F32.

Verified numbers (v7 unless noted; prompts P0/P1 are the baseline coding prompts through the chat template, W is a 400-token English passage; 553 positions):
- Tokenizer: llama_tokenize == HF ids on P0 (81), P1 (72), W (400).
- Repack: 497 tensors, 30272 rows checked, 0 mismatched (bit-exact).
- V0 (fp32 mode vs fp64 numpy on real weights, layer 0 DeltaNet, layer 3 attention, FFN with Q4_1 down, Q5_K ssm_out, 2048 Q6_K lm_head rows, recurrent state): every check rel <= 1e-7.
- V2 fp32 mode vs llama batch logits: mean KL 1.41e-3, p99 1.38e-2, max 2.75e-2, top-1 99.44% excluding 20 near-ties (98.4% all). vs llama token-by-token: mean KL 4.21e-3, p99 4.1e-2, top-1 98.9%. llama's own batch vs token-by-token floor: mean KL 5.28e-3, p99 4.7e-2, top-1 98.7%. So t4q is 0.27x (batch) and 0.80x (tbt) of llama's internal disagreement.
- V2 q8 mode: vs tbt mean KL 6.68e-3 (1.26x floor), vs batch 5.58e-3 (1.06x floor).
- V1 free-running intermediates, all 64 layers x 22 names x positions 0/1/11: worst rel / max(design tol, 2 x llama batch-vs-tbt rel) = 0.91 (q8) and 0.81 (fp32).
- V3 greedy 128 tokens: fp32 mode P1 identical 128/128, P0 identical for 74 tokens then diverges at a near-tie (llama top-2 gap 0.042). q8 mode P0 diverges at 72 (gap 0.045), P1 at 17 (gap 0.029). All four outputs are coherent Python (LRU cache module, Apache log analyzer).
- Speed (irrelevant for M1): fp32 mode 302 ms/token (3.8 tok/s), q8 mode 131 ms/token (11.8 tok/s decode). VRAM 7199 / 8149 MiB at 4k context. Load 8 s from page cache (16.06 GB download 85 s).

Why the design's absolute V1/V2 numbers do not apply as written:
- llama.cpp quantizes every activation to q8_1 before each quantized matmul. On outlier tokens this is lossy: at layer 3, position 0 (`<|im_start|>`), llama's `Vcur` differs from the exact product by 7.6% while t4q fp32 is at 1e-7 of fp64. Its batch (MMQ, chunked GDN, MMA FA) and token-by-token (MMVQ, AR GDN) paths therefore disagree with each other at mean KL 5.3e-3, more than the 2e-3 the design assumed for t4q vs llama.
- With `act_q8` t4q reproduces llama's MMVQ bit-exactly (layer 0, pos 0: 0.0). The remaining q8-mode gap starts in llama's Turing MMA flash-attention: its output differs from both exact-f32 and fp16-rounded-V attention by 1.4e-4 (teacher-forced probe, every attention layer, even at position 0 with one key), and q8_1 rounding flips in the next GEMVs amplify that through the stack.
- Gate as implemented in `validate.py` (`gate_detail`): V0 all <= tol; repack bit-exact; V1 within max(design tol, 2x llama floor) in both modes; V2 fp32 vs batch meets the design's absolute thresholds and both modes are within 2x the llama floor (the design's secondary V2 criterion); V3 passes in both modes.

Broken or open:
- No prefill: `t4q_prefill`/`t4q_logits` run decode steps. No TP, no graphs, no MTP (M2+).
- `t4q_generate` is greedy only. Stop tokens work (`EOS_IDS` in `py/t4q.py`: 248046, 248044).
- Teacher-forced per-layer check (`V1_tf`) reports max l_out rel 1.5e-2 (q8) / 1.9e-2 (fp32); it is informational. It localized the attention discrepancy above.
- I did not measure the design's floor definition (llama layer split vs tensor split); I used llama batch vs token-by-token instead.

Next steps:
1. M2: swap `gemv_ref.cu` for the M0 fast path (`gemv.cuh`, RPL=4). It needs Q4_1 (P4 + m plane) and K5 (Q5_K) first. Its q8 scheme (float d, integer sums) differs from llama's fp16 d/s, so expect the fp32-vs-q8 split seen here; re-run `stage_m1` (all sections) after every kernel change, comparing against this run's numbers.
2. Keep `act_q8` (or the M2 q8 kernels) behind the same validate: V1/V2 floor ratios should stay <= ~1.3.
3. Replace the host embedding dequant and per-token host sync (M3), then prefill GEMM + `gdn_seq` (M4) so V2 can compare t4q prefill vs llama batch directly.

## 2026-10-02 - M2-M4 (round 1), TP=2 fast decode engine

**Gate (single-stream decode >= 30 tok/s on Q4_0 at 4k, matching the oracle): NOT passed.** Best verified: **29.24 tok/s** (`otdoges/t4q-m4` v11, CUDA graphs, prompt P1, 256 tokens, P2P box) and 29.22 (v7). At 3.6k context the best is **28.55** (v11). Every correctness check passes on every version since m2 v1. For reference, llama.cpp `-sm tensor` on Q4_0 does 21.11.

I went straight to TP=2 rather than measuring layer-split M2/M3 gates, because the 30 tok/s gate needs TP. Graphs, device StepState and the async host loop (the M3 items) were in the first TP run, so `t4q-m2` holds the first two engine versions and `t4q-m4` holds everything after. There is no separate `t4q-m3` kernel.

### What I built (all under `t4q/`)
- `src/kernels/gemv.cuh` gained two fast formats, P4M (Q4_1: P4 plus an fp16 m plane) and K5 (Q5_K: nibbles, a 1-bit high plane laid out for shift+mask, decoded 6-bit sc/m bytes, and a d/dmin pair per 256). Both have host and device repack (`repack_device_rows` takes a row offset). `tools/gemv_layout_check.cpp` covers them and passes locally.
- `src/tp.h`, `src/tp_engine.cu`, `src/tp_api.h`, `src/kernels/tp_kernels.{h,cu}` form the TP engine. `t4q_params.tp = 1` selects it; the M1 layer-split engine is unchanged.
  - **Shard repack.** At load each GPU gathers its rows and column blocks from the mmap:
    - DeltaNet: k-heads 8g..8g+7, v-heads {r*16 + 8g + kl}; conv/ssm_a/dt/alpha/beta are gathered to match.
    - Attention: q heads 12g.., kv heads 2g, 2g+1.
    - FFN: gate and up rows **interleaved by 4 per tile**, so one GEMV block owns a whole q8 group.
    - K-split: ssm_out, attn_output and down by column blocks. lm_head by rows.
  - The raw Q4_0 `token_embd` lives on both GPUs (715 MB each). VRAM is 8411 MiB per GPU at 4k context. Load takes 6-7 s from page cache.
  - **Load-time self-test.** Each fast format is checked against the CPU ggml dequant with fp64 dots (worst 4.2e-7). It also times every weight's GEMV and the AR-epilogue variants. These are the "selftest" numbers below.
  - **Per-layer kernels (default path).**
    - `ar_norm`: one block does the all-reduce publish, waits for the peer, adds the residual, then RMSNorm and q8 quantization.
    - The GEMVs use the dp4a q8 path:
      - qkvz also produces the 48 fp32 alpha/beta rows as a segment;
      - gate|up has a silu*up + q8 epilogue (no separate silu kernel);
      - the K-split GEMVs write their rows straight into the peer's mailbox (`arpub=2`).
    - `gdn` is a single pass: 96 blocks, conv ring in a 4-slot buffer, L2 norm, gates and recurrence, with state loads hoisted. `gnorm_q8` follows it.
    - Attention is three kernels:
      - `attn_prep`: q/k norm, RoPE, fp16 KV append;
      - `attn_split`: split-K flash decode, 2 x 40 blocks at 2 per SM;
      - `attn_combine_q8`.
    - At the end of the step, `argmax` runs per shard and exchanges with the peer, then updates the token, pos and step.
  - **All-reduce.** A P2P mailbox: rows go into the peer's `rx[slot]`, then fence.sys and an epoch flag. Epochs come from the device step counter. Each GPU computes `h + (p0 + p1)` in the same order, and the residual is checked bit-identical on both GPUs.
    - **Fallback when the GPUs have no P2P** (2 of my 14 m4 boxes; m4 v1 failed to load because of it): a host-mapped mailbox plus a `pull` kernel. It is about 1 tok/s slower, with the same greedy output.
  - **Graphs.** One CUDA graph per GPU per step, replayable because all position state is in the device StepState. The host enqueues 8 steps ahead and reads tokens from a host-mapped ring.
- `tests/tp_check.py` runs the correctness checks and benchmarks:
  - selftest; V1 (residual-stream dumps vs llama, floor-aware); V2 (logits KL, eager and graphs);
  - V3 (greedy 128 tokens); V4 (graphs vs eager bit-identical);
  - P0/P1 decode benches, a 3.6k-depth bench and an eager per-kernel profile, with A/B over option configs.
  - `tools/stage_m2.py` / `stage_m4.py` drive the oracle, `tp_check`, nvidia-smi clock logging per bench window, and variant processes driven by env vars.
- Runtime options (`t4q_set_option`) and env knobs:
  - `graphs`;
  - `arpub` (0 = the GEMV epilogue publishes, 1 = the consumer copies and publishes, 2 = rows from the GEMV and the flag from the consumer; default 2);
  - `fuse` (0 = separate glue kernels, the default; 2 = redundant per-block AR+norm prologue; 3 = leader-block prologue);
  - `pf_kb` (L2 prefetch, off); `mega` (persistent per-layer kernels, off); `profile`;
  - env: `T4Q_NO_P2P`, `T4Q_RPL_P4/N/K`, `T4Q_THREADS`.

### Verified numbers (Kaggle, tok/s are graph-mode decode of 256 tokens after the P0/P1 chat prompts, max_ctx 4096)
| kernel version | change | P0 / P1 tok/s | 3.6k depth | GPU SM MHz during bench |
|---|---|---|---|---|
| t4q-m2 v1 | first TP engine, scattered 4-byte AR writes | 24.46 / 24.49 | 24.08 | 1180-1270 |
| t4q-m2 v2 | coalesced float4 AR rows, attention 2 positions in flight | 28.60 / 28.75 | 28.11 | 1000-1070 |
| t4q-m4 v3 | A/B: fused per-block prologues vs separate kernels | 26.3 / 26.7 vs **28.28 / 28.24** | 26.0 vs 27.4 | GPU0 660-880 |
| t4q-m4 v5 | separate (fuse 0) vs AR+norm prologue fused (fuse 2) | 28.79 / 28.92 vs 28.87 / 28.89 | 28.38 | 970-1065 |
| t4q-m4 v7 | consumer publishes the AR (arpub 1) | **29.17 / 29.22** | 28.42 | 1000-1040 |
| t4q-m4 v10 | arpub 2 vs 1 vs leader prologue | 28.86 / 29.06 vs 28.99 / 29.00 vs 28.31 / 28.28 | 28.08 | 840-960 |
| t4q-m4 v11 | arpub 2 (default) | 28.89 / **29.24** | **28.55** | GPU0 1050, GPU1 920 |
| t4q-m4 v12 | RPL2 for N-split + 128-thread blocks (default) | 28.28 / 28.09 (old config 28.64 / 28.30) | 27.92 | GPU0 690-780 (hot) |
| t4q-m4 v13 (no P2P) | default vs `mega=1` | 28.55 / 28.63 vs 27.06 / 27.07 | 26.5 | 1050-1240 |

The no-P2P fallback, forced on P2P boxes, measured 28.19 (v7), 27.5 (v10) and 27.4 (v12). Its greedy output is identical.

**Correctness**, unchanged through v13; numbers from v11, the same in v4-v13:
- Self-test max error/rms 4.2e-7, on every format and both GPUs.
- V1, all 64 layers at positions 0/1/11 (771 records): worst rel / max(tol, 2 x llama floor) is 0.48-0.51 per key. The residual is bit-identical across the two GPUs.
- V2 against llama batch: mean KL 1.55e-3, p99 1.43e-2, top-1 99.4% excluding ties. That is **0.29x the llama batch-vs-tbt floor** and inside the design's absolute limits. Against llama token-by-token it is 0.89x the floor.
- V3: P1 matches 128/128. P0 matches 74 tokens, then diverges at the same near-tie as M1 (llama top-2 gap 0.042). m4 v2 was 128/128 on both.
- V4: graphs vs eager logits are bit-identical (max diff 0.0).
- The output is coherent Python, e.g. the Apache log analyzer.

**Where the time goes** (v11, P2P box, eager per-kernel events, per GPU per token, ~37 ms eager vs 34.2 ms in graph mode):

| part | time |
|---|---|
| gate\|up | 12.5-12.7 ms |
| down | 6.6 ms |
| qkvz | 4.8 ms |
| ssm_out | 2.85 ms |
| lm_head | 1.97 ms (265 GB/s) |
| attn qkv | 1.37 ms |
| attn_output | 0.7 ms |
| AR + norm (`ar_norm`, 129 x 17-26 us) | 2.2-3.4 ms |
| gdn | 0.85 ms |
| attention at 3.6k | 1.1 ms |
| gnorm | 0.33 ms |

Self-test GEMV rates are 250 (qkvz, RPL2 + 128 threads), 257 (gate|up), 249 (down), 225 (K5 ssm_out and K=3072 attn_output) and 265 (lm_head) GB/s. Against lm_head's 265 GB/s every GEMV pays about 6 us of fixed ramp and tail, roughly 2 ms per token.

The SM clock under the 70 W cap varies from 555 to 1270 MHz between boxes and over a run. Decode follows the slower GPU's clock: 29.2 tok/s at about 1 GHz, 27.2 at 555 MHz.

### What did not work (measured, kept as options or removed)
- **Fused per-block prologues** (AR+norm, silu, gated norm redundantly in every GEMV block; m4 v1-v3) were slower: 24.9, then 26.7 vs 28.3. The silu and gated-norm work is ALU- and latency-bound at throttled clocks. The AR+norm-only variant (fuse 2) is break-even. Leader-block fusion (fuse 3, block 0 does the work, the others prefetch and wait) is also no faster, because the slower GPU's critical path dominates. Its first version had a bug: the x-ready epoch was 0 at step 0, so the V1 floor failed and greedy output diverged at token 5. Fixed in v10.
- **L2 prefetch** of the next GEMV during small kernels (`pf_kb`) gave no gain, at either 128 B or 32 B granularity. A direct test (v12) showed that touching the first 2 MB of gate|up saves only 2-2.4 us of a 195 us GEMV.
- **AR publish from the GEMV epilogue with a per-block fence.sys and counter** costs 10-14 us per K-split GEMV (self-test: plain 50.1 us, rows only 52.1, +fence.gpu+counter 57.2, +fence.sys 62-64). Hence `arpub` 1 and 2.
- **Persistent per-layer kernels** (`mega=1`, v13) are correct (V3 passes) but 5% slower. Static tile ownership leaves blocks idle: gate|up keeps only 68 of 80 blocks busy, and the 96 gdn items need two rounds on 80 blocks. It would need dynamic (atomic) tile scheduling and a gdn re-split to pay off.
- **RPL=2 for ffn_down** is slower (P4M 118.6 vs 111.9 us). RPL=2 + 128 threads only helps the N-split shapes (qkvz 99.6 -> 94.3 us), and that gain did not show up end to end within the box-to-box noise (v12).

### Broken or open
- No batched prefill: `t4q_prefill` runs decode steps (about 28 tok/s). The verify/batched GEMV path (m > 1) exists in `gemv.cuh` but is not wired into the TP engine.
- The TP engine is greedy only, Q4_0 only (it asserts a Q4_0 `token_embd`), and needs exactly 2 GPUs.
- `t4q_generate` launches up to 7 steps past a stop token (the token ring is drained every 8 steps), so the state is advanced past EOS. That is fine for the tests (they reset), but it needs replay or rewind before multi-turn use.
- Graph-mode profiling with event nodes returned `cudaErrorInvalidValue`. The profiles are eager, which inflates small kernels by about 3 ms per token.
- Box variance (P2P present or not, clocks 555-1270 MHz) is ±4%, as large as the remaining gap to the gate. Always compare A/B configs inside one run, and use the cool-at-load self-test timings for kernel-level decisions.

### Next steps
1. To reach 30+ single-stream, the remaining fixed costs are the AR critical path (about 17 us x 129 in graph mode) and about 6 us of ramp and tail per GEMV x 337. Options:
   - Make `mega` pay with dynamic tile scheduling: an atomic tile counter with the next index prefetched; SQ via fp32 silu output and q8 in the down phase. Re-split gdn into 80 or 160 items.
   - Or cut the AR latency: have the slower GPU publish earlier, or overlap the leader norm with the AR wait.
2. M5 (MTP) is the larger lever: about 2.2-2.8x on code prompts. It needs the m = k+1 verify GEMV (`gemv_fast_kernel` already handles M up to 8 with bit-identical columns) wired into `k_gemv` and the step kernels, plus `gdn_verify`/replay as in DESIGN s7.
3. Batched decode for aggregate throughput (user goal 200-400 tok/s) needs the int8 mma GEMM path for m = 8..64, and the same for prefill (W4A8 `mma.m8n8k16`).
