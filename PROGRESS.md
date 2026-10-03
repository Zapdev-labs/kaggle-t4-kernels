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

## 2026-10-02 - M2-M4 (round 2), graph-mode profiling, AR and small-kernel work

**Gate (single-stream decode >= 30 tok/s on Q4_0, outputs matching the oracle): PASS on P2P boxes, marginal.**
- `otdoges/t4q-m4` v24: default config **30.04** tok/s (P1), `pf_kb=1536` config **30.07 / 30.05**, `tp_check` `gate_30: true`, every correctness check passing (selftest, V1 floor, TP residual identical, V2 and V3 in eager and graphs, V4 bit-identical).
- v26, same code path (the option that differed only acts without P2P): **30.15 / 30.17**, V1-V4 passing. A later config in that run crashed the bench loop (arpub 4, see below), so v26 has no gate summary.
- At 3.6k context the best is **29.33** (v24); the depth gate in `tp_check` (`gate_30_depth`) is not met.
- Box lottery matters more than anything I changed this round. Over v14-v27 I saw three kinds of box: fast P2P (rows to the peer cost ~3 us per K-split GEMV), slow P2P (the same rows cost 16-31 us) and no P2P (host-mapped mailbox). With the final defaults: slow-P2P boxes give 29.9-30.2, no-P2P boxes 28.3-29.4 (v23, v25, v27). I did not get a fast-P2P box after v16.

All tok/s are graph-mode decode of 256 tokens after the P0/P1 chat prompts, max_ctx 4096, as in round 1.

### What I built (all under `t4q/`)
- **CUPTI kernel timeline** (`src/cupti_trace.{h,cpp}`, option `trace`, `tp_check --trace N --trace_dir`). It dlopens libcupti (Kaggle has `/usr/local/cuda/lib64/libcupti.so`), records every kernel inside the CUDA graphs and writes per-position durations, launch gaps and start offsets for both GPUs. This replaced the eager event profile, which overstated small kernels. `t4q-m4` v14 onwards has a trace per config in `kaggle/m4/out*/traces/`.
- **globaltimer phase probes** (option `dbgts`): block 0 of the small kernels adds per-phase offsets into a device buffer. It showed that the single-block `ar_norm` spent 13-15 us on its own arithmetic.
- **Multi-block AR + RMSNorm + q8** (`k_ar_norm_mb`, default; `arn=1` restores the old kernel): 20 blocks x 256, each reduces the full sum of squares redundantly and normalizes its own 256 elements. Per call: 18.6 -> 10.6 us. v15: 29.77 vs 29.05 for the old kernel in the same run.
- **AR transport auto-select** (`arpub -1`, default): at load I time the ssm_out GEMV with and without rows to the peer. If the rows cost more than 10 us (slow P2P), or there is no P2P, it uses arpub 1, where `ar_norm` block 0 copies the 20 KB partial in one coalesced pass and then publishes the flag. Otherwise it uses arpub 2 (rows from the GEMV blocks).
  - v20 (slow P2P): 29.66 vs 27.67 for arpub 2.
  - v19 (no P2P): 29.14 vs 27.21.
  - Stats report `arpub_auto` and `ar_rows_cost_us`.
- **gate|up with 256-thread blocks** (`sqt=256`, default): one tile per warp instead of two. gate|up went from 199 to 193 us (v21-v22).
- **qkvz fp32 alpha/beta rows first** in the GEMV work list, so they are no longer the kernel tail: 97.4 -> 95.8 us.
- **Fused gdn + gated norm** (`gdnf=1`, default): the last block of each head does the norm and q8. Gate loads and v-conv loads are hoisted. Saves the gnorm launch, about 1.6 us per DeltaNet layer.
- **P4 unsigned high-nibble dp4a** (`p4u=1`, default): `dp4a.u32.s32` on `q & 0xF0F0F0F0` removes the shifts. It is 5.6% fewer instructions in the gate|up kernel and bit-identical; the selftest compares every P4 weight against the reference path and fails on any bit difference. No measurable speed change, since the kernel is memory-bound.
- **Chunk-major weight layout** (`Layout::cm`, default 1; `T4Q_CM=0` restores the old layout): `[chunk][tile]` instead of `[tile][chunk]`, with the same bytes. Selftest is 0.5-1% faster. It also makes `pf_kb` prefetch hit the first-wave tiles: the GEMVs after a prefetching kernel run 2-4 us faster, but the prefetching kernel gets longer by the same amount, so the net is neutral (`pf_kb=0` stays the default).
- **argmax_final** uses a warp instead of one thread.
- **Combine** computes the split weights in parallel.
- `gemv_layout_check.cpp` covers `cm=0/1`.
- `tp_check` additions:
  - `--rounds` interleaves configs against clock drift;
  - options absent from a config reset to config 0's value;
  - benches of an alternative config only count if its greedy V3 passes;
  - it reports `gate_30` (short context) and `gate_30_depth` separately.
- `stage_m4.py` logs `nvidia-smi topo` and picks up the CUPTI path automatically.

### Where the time goes now (v24 trace, slow-P2P box, GPU0, per token, 33.58 ms step)
| part | time |
|---|---|
| gate\|up | 64 x 193 us = 12.4 ms |
| down | 56 x 97 us + 8 x 115 us (Q4_1) = 6.4 ms |
| qkvz | 48 x 96 us = 4.6 ms |
| ssm_out (K5) | 48 x 54 us = 2.6 ms |
| lm_head | 1.94 ms |
| qkv_a | 16 x 81 us = 1.3 ms |
| attn_output | 16 x 40 us = 0.64 ms |
| AR (`ar_norm_mb`, arpub 1) | 128 x 13-16 us = 1.7-2.0 ms |
| gdn | 48 x 17.5 us = 0.84 ms |
| attention (prep + split + combine) at 330 positions | 16 x ~34 us = 0.55 ms |
| launch gaps | 0.39 ms |

- GEMVs total about 29.8 ms. At lm_head's 268 GB/s the same bytes would take about 27.8 ms, so the remaining per-kernel ramp, tail and epilogue cost is about 2 ms. The worst offenders are ssm_out (13 us over), gate|up (6 us) and attn_output (7 us).
- No-P2P boxes add `pull` (15-20 us) before every `ar_norm` (7-8 us).
- At 3.6k context, `attn_split` grows to 50-58 us per layer.

### Measured dead ends (kept as options, default off)
- **LL all-reduce** (`ll=1`, tagged 8-byte rows, no flags): `ar_norm` 24.7 -> 28.9 us (v14).
- **AR tail blocks inside the K-split GEMV** (`tail=1`, no `ar_norm` kernel): +6-8 us per AR, because the copy, `fence.sys` and flag sit on the critical path (v17).
- **Fused attention** (`attnf=1`): 36-40 us vs 34-41 for the three kernels.
- **Split attention v2** (`attn2=1`; reduce-scatter scores, separate softmax, shared-V P.V): 27.6 vs 20.9 us at short context, 48 vs 48 at depth (v20).
- **arpub 3** (each of the 20 blocks publishes its own slice with its own flag): `ar_norm` 19.9 vs 15.5 us on a slow-P2P box (v26).
- **arpub 4** (copy-engine `cudaMemcpyPeerAsync` node): fails inside stream capture. It is now rejected by `set_option`.
- **No-P2P one-kernel variants**: `pn=1` (`pull_norm`, slice publish with tagged partial sums) took 34 us vs 25 us for pull + ar_norm; `pn=2` (`pull_arn`, a transport block plus norm blocks) took 30.6 vs 27.3 us.
- **Smaller knobs**:
  - `spin_ns` nanosleep backoff: no effect.
  - `pf_gemv` (gate|up tail blocks prefetching down): gate|up slower, down unchanged.
  - `T4Q_THREADS=256` for the plain GEMVs: neutral.
  - K5 RPL 1 or 4: neutral.

### Broken or open
- The gate holds only on P2P boxes, and only just. No-P2P boxes (about 40% of my runs) top out at 29.4.
- The depth number (29.33 at 3.6k) is below 30. `attn_split` at depth runs at about half the KV bandwidth.
- Everything from round 1 still applies: no batched prefill, greedy only, Q4_0 only, up to 7 steps past EOS.
- When an option throws during graph capture, the engine stays in a failed state for the rest of the process. v26's bench loop died that way after arpub 4.

### Next steps
1. To make 30 robust (no-P2P boxes, depth), the remaining fixed costs are about 2 ms of GEMV ramp and tail and about 2 ms of AR. The candidate that attacks both is a pipelined persistent GEMV: GEMV kernels launched in alternating graph branches, each with at most one block per SM. Each would prefetch its first chunks during the previous kernel's tail and wait on a device flag for x, with dynamic (atomic) tile scheduling. Round 1's `mega` lost 5% mainly to static tile ownership.
2. Attention at depth: score tiles with `mma.m16n8k16` f16 (Q in fp16, as llama's Turing FA does), or a better split of positions per warp. The v2 kernel was latency-bound in phases I could not explain with the phase probes.
3. M5 (MTP) is still the big single-stream lever, and batched decode or prefill needs the int8 mma W4A8 path.

## 2026-10-02 - P-prefill (round 1), W4A8 tensor-core GEMM + batched TP prefill

**Gate (pp2048 >= 1400 and pp512 >= 1200 tok/s, correctness preserved): NOT passed.** Best verified: **pp512 583.5, pp2048 633.8 tok/s** (`otdoges/t4q-p` v5, P2P box), with correctness passing on every configuration. llama.cpp `-sm tensor` on Q4_0 does pp512 516. Full tables are in `research/p_results.md`.

### What I built (all under `t4q/`)
- `src/kernels/gemm.cuh`: W4A8 GEMM on int8 tensor cores (`mma.m8n8k16.u8.s8`).
  - It reads the decode weights in place (P4/P4M/K5 layouts, rpl 2 or 4, cm 0/1).
  - The int32 result is converted to float with the magic constant on the accumulator. The exact per-32 epilogue is two FFMAs.
  - Tiling: 128 x 128 block tiles, ldmatrix fragments, register-staged double buffer.
  - The prefill q8 quantizer (`quant_rows_kernel`) has a host mirror.
  - Kept as A/B code: timing-only epilogue variants (`EPI` 0/1), an fp16 HMMA variant (`gemm_f16_kernel`) and an int4 m8n8k32 split-activation variant (`EPI` 3, `quant_rows_i4_kernel`; it has a 4.6e-4 rounding error from its nq constant and is not used).
- `tools/gemm_bench.cu`: real TP shapes at T 512/2048, an fp64 check on sampled rows, and sustained both-GPU runs per variant with NVML clocks and power.
- `src/tp_prefill.cu`: batched TP prefill, used by `t4q_prefill` for n >= 2 (option `pf`).
  - Ubatches (`pf_ub`, default 2048) are split into 2 interleaved sub-batches (`pf_nsub`). The fp32 AR partials go through `cudaMemcpyPeerAsync` on a copy stream, so one sub-batch's copy overlaps the other's compute.
  - Fused producers (`pf_fuse`): add+RMSNorm, silu*up and the gated norm quantize straight into the GEMM layout.
  - DeltaNet: conv with decode arithmetic (history from the conv ring, ring updated), then a sequential scan with decode's per-token math. The scan keeps the state in registers in the decode layout, stages inputs through smem, and keeps all 96 blocks resident.
  - Attention: tensor-core flash attention (`k_pf_fa`, option `pf_fa`) that writes the decode KV cache.
  - The last prompt token runs as a normal decode step (graph), so StepState, logits and the first generated token come out exactly as the decode engine expects.
  - Per-op profile: option `pf_prof`, reported in stats as `pf_profile`.
- API: `t4q_last_logits`. Options: `pf`, `pf_ub`, `pf_nsub`, `pf_fa`, `pf_fuse`, `pf_i4`, `pf_prof`. Stats: `pf_last_batch_s`, `pf_last_total_s`.
- `tools/oracle_dump.cpp`: new `last` job (last-position logits with the prompt in 512-token batches); `gen` now prefills in 512-token batches, so prompts > 512 work.
- `tests/prefill_check.py`, compared against the decode path (pf=0) and the llama.cpp oracle:
  - last-token KL and top-1;
  - 16 teacher-forced continuation positions;
  - greedy 32 tokens;
  - pp512/pp2048 benches per config with a profile run.
- `tools/stage_p.py`: `SECTIONS` "gemm" (bench only, no download) and "engine" (download, oracle, prefill_check).

### Verified
- GEMM: rel L2 1.3e-5 to 3.1e-5 against fp64 on the same q8 activations, for every shape and format (v1-v3).
- Sustained gateup T=2048, both GPUs (v2/v3):

  | variant | TOPS per GPU | MHz |
  |---|---|---|
  | exact W4A8 | 23.8-26.0 | 790-980 |
  | no epilogue at all (timing only) | 27.4-35.3 | 665-875 |
  | fp16 HMMA | 18.4-19.4 | 960-1016 |
  | int4 split | 21.8-25.2 | |
  | W4A4 cost | 24.9-30.4 | |

  All of them sit at the 70 W cap.
- Engine correctness (v6 default ub 2048; every v5 and v6 config passes), KL(decode path ‖ batched prefill) at the last token:

  | prompt | KL |
  |---|---|
  | P0 | 3.4e-5 |
  | P1 | 6.6e-5 |
  | W (400) | 3.9e-4 |
  | L (2048) | 2.3e-4 |

  - Top-1 is equal everywhere.
  - Greedy 32 tokens match the decode path on P0/P1/L. On W the batched path matches the oracle for all 32, while the decode path diverges at token 25 at a near-tie (gap 0.08).
  - KL against llama.cpp's batch logits is 1.2e-5 to 8.4e-4 (llama's own batch-vs-tbt floor: 3e-5 on P0/P1, 2.6e-3 on W).
- Speed by version:

  | version | box | pp512 | pp2048 |
  |---|---|---|---|
  | v4 | no P2P | 574 | 551 |
  | v5 | P2P | 583.5 | 633.8 |
  | v6 | P2P, GPU0 throttled to 680-750 MHz | 579 | 597 |
- Profile (v5, pp2048): GEMMs 2.2 s of a 3.2 s batch (gateup 1.0 s), GDN scan 0.33 s, AR wait 0.31 s, flash attention 0.055 s (the SIMT kernel took 0.385 s).

### Why the gate is not reachable with this design (measured)
- The T4s idle at about 30 W (`clocks.csv`), which leaves about 40 W for compute under the 70 W cap.
- At about 600 tok/s each GPU spends about 110 mJ per token.
- The linear layers are 24.3 GOP per token per GPU. Even the timing-only GEMM with no epilogue sustained only 27-35 TOPS, which caps linear alone at 1150-1440 tok/s. The exact kernel's 24-26 TOPS caps it at 990-1070.
- Overlapping the AR shows the same limit: it removed about 1 s of waiting, but the clocks dropped from about 1080 to about 800 MHz (604 vs 577 tok/s).
- So prefill speed is energy per token, and 1400 tok/s needs roughly half the current energy per MAC.

### Broken or open
- The int4-split GEMM path (`pf_i4`) has a 4.6e-4 rounding error (its nq constant -(M + 8 sq) d is not exact). It was no faster, so it is off and untested end to end.
- The GDN scan is limited by shared-memory wavefronts and shuffles: 285 ms at pp2048 against a ~50 ms FMA floor.
- `k_pf_ab` (fp32 SIMT, 48 rows) takes 72-111 ms at pp2048.
- The AR payload is fp32.
- Prefill speed varies about ±5% with box and GPU temperature; GPU0 often runs 20 C hotter and throttles harder.

### Next steps
1. **GEMM energy.** Hoist the kb-loop address math (about 90 IMAD/LEA/SEL per 128 IMMA) and try other block shapes and KBU values. Then measure, with prefill_check KL, the bounded-loss options:
   - per-128 activation groups with a one-FFMA-per-block epilogue (about +10% in the EPI1 test);
   - per-row-256 int8 requantization of each layer's weights into a scratch buffer, which would approach the no-epilogue 28-35 TOPS.
2. **GDN.** Either the chunked WY form on tensor cores, or a reduce-scatter layout (8 columns per warp, about 3 shuffles and 1 smem wavefront per column per token).
3. **Smaller items:**
   - ab rows on tensor cores, or folded into another kernel;
   - silu*up + q8 in the gate|up GEMM epilogue (removes the 17408-wide fp32 round trip);
   - FA output + q8 fused;
   - fp16 AR payload.
4. **Physics.** On these T4s at 70 W, 1400 tok/s pp2048 probably needs the lossy GEMM options above plus every other part near zero. A realistic exact target is about 800-900.

## 2026-10-02 - P-prefill (round 2), in-kernel W4->int8 GEMM, chunked DeltaNet, fused silu, fp16 AR

**Gate (pp2048 >= 1400 and pp512 >= 1200 tok/s, correctness preserved): NOT passed.** Best verified: **pp2048 988.6, pp512 855.4 tok/s** (`otdoges/t4q-p` v17, P2P box, both GPUs at 960-1050 MHz), correctness passing on every prompt. Round 1 best was 633.8 / 583.5, so this is +56% / +47%. Full tables: `research/p_results.md` (round 2 section).

### What I built
- **`t4q/src/kernels/gemm8.cuh`**, the new prefill GEMM family.
  - **`gemm9_kernel`** is the production kernel. It reads the decode weights in place (P4/P4M/K5, rpl 2/4) and converts each 32-block to int8 in fp16x2 while staging. Each row gets one scale, `invs[n] = 127 / max|w[n]|`, computed once per weight by `row_invs_kernel`, so the weight scale leaves the inner loop.
  - Activations are q8 with one scale per 64 elements (GA 64). That costs one FFMA per output per 64 instead of two per 32. The magic constant handles the int-to-float conversion, with the bias subtracted every 256 k.
  - Layout: 128 x 256 tile, 8 warps of 64 x 64, hoisted per-thread pointers.
  - Kernel variants kept for A/B: GA 32 (exact q8 blocks) and GA 0 (per token).
  - Requantizing to per-row int8 adds about 1% of Q4_0's own noise variance. I measured this on real Qwen3.8 tensors downloaded with HTTP range requests.
  - Experiments kept in the file, none faster under the cap (see below):
    - `gemm8_kernel` (first version);
    - `gemm10` (CUTLASS-style pipeline, 128 x 128);
    - `gemm11` (256 x 128 tile);
    - `gemm12` (2 blocks/SM);
    - `gemm13` (per-token probe);
    - `gemm14` (GA 64 on the CUTLASS pipeline; it spills).
- **Fused gate|up epilogue** (`launch9_silu`, option `pf_silu`): the gate|up GEMM writes q8(silu(gate) * up) straight into the down GEMM's input. This removes the 17408-wide fp32 round trip and the silu kernel. Results are bit-identical to the unfused path (same KL).
- **Chunked DeltaNet** (`k_pf_gdnc`, option `pf_gdnc`): C = 64 on fp16 mma.m16n8k8 with fp32 accumulation.
  - Formulas: `(I - A) U = b (V - e^G K S0^T)`, `O = e^G Q S0^T + (QK^T * D) U`, `S = e^{G_C} S0 + (e^{G_C - G} U)^T K`.
  - T = (I - A)^-1 comes from fp32 forward substitution.
  - The state stays in registers in the accumulator layout, which is also the B-fragment layout that Q S0^T needs.
  - One block per head.
  - Option `pf_gdnc_chk` compares it against the sequential scan. Max error relative to max |value|: o about 3e-4 to 1e-3, S about 3e-4 to 6e-4.
- **fp16 all-reduce partials** (`pf_ar16`): the K-split GEMMs write fp16, the peer copy moves half the bytes, and add_norm sums in fp32.
- **GA 64 fused producers**: add_norm, silu and gated norm compute the 64-group amax across warp pairs.
- **`k_pf_ab`**: 128-token tiles, 4-way K split, summed by the scan.
- New options: `pf_g8`, `pf_ga` (32/64/0), `pf_bn`, `pf_silu`, `pf_gdnc`, `pf_gdnc_chk`, `pf_ar16`, `pf_gdn2`. Defaults: g8 1, ga 64, silu 1, gdnc 1, ar16 1, gdn2 0.
- Tools:
  - `t4q/tools/ref_bench.cu`: cuBLAS, cuBLASLt and CUTLASS v3.5.1 (cloned at run time) int8/int4/fp16 references.
  - `gemm_bench.cu`: gemm8-14 variants, a host mirror check for the requant GEMMs, ablation and data-toggle probes.
  - `stage_pg.py`: GEMM-only runs on a second kernel id (`otdoges/t4q-pg`), so they can run beside engine runs.

### Verified (all on Kaggle)
- **Engine speed by version** (correctness passing on all of them):

  | version | change | pp2048 | pp512 |
  |---|---|---|---|
  | v12 | gemm9 GA64, unfused | 925 | 800 |
  | v15 | + fused silu | 888 | 780 (GPU1 throttled to 900 MHz) |
  | v16 | + chunked GDN | 921 | 819 (GPU0 throttled to 875 MHz) |
  | v17 | + fp16 AR | **988.6** | **855.4** (both GPUs about 960-1050 MHz) |

  On the v17 box, the round-1 path measured 747/699 in v12.
- **v17 correctness**, KL(decode path ‖ batched prefill) at the last token:

  | prompt | KL |
  |---|---|
  | P0 | 6.7e-4 |
  | P1 | 1.1e-3 |
  | W | 3.2e-4 |
  | L (2048) | 5.4e-4 |

  - Top-1 is equal everywhere.
  - Greedy 32 tokens are identical to the decode path on P0/P1/L. On W the output matches the oracle and the decode path diverges at its known near-tie (token 25).
  - KL against llama.cpp's batch logits is 6.6e-4 to 1.3e-3.
- **Rejected for accuracy**: GA 0 (one activation scale per token) gave L KL 0.31 and greedy diverged at token 1 (v12). Massive activations break per-token int8.
- **v17 pp2048 profile, GPU0** (2.04 s):
  - GEMMs 1703 ms: gateup 806, down 354, qkvz 298, ssm_out 122, attn_qkv 86, attn_out 37;
  - AR wait 105;
  - GDN 78 (the sequential scan took 224);
  - ab 53, attention 49, conv 23, gated norm 21.

### Why the gate is still out of reach (measured)
- **The GEMM is 83% of the time, and the 70 W cap sets its speed.** Sustained TOPS with both GPUs loaded, gateup at T = 2048:

  | kernel | TOPS | clock |
  |---|---|---|
  | round 1 W4A8 | 23-26 | |
  | **gemm9 GA64** | **30.9-33.5** | about 1040 MHz, 775 ops/clk/SM = 38% of peak |
  | cuBLAS int8 | 45-55 | |
  | CUTLASS int8 128x256 | 45-55 | 1400 ops/clk/SM |
  | CUTLASS int4 | 122-126 | at 55-59 W, not capped |

- **Ablation of gemm9** (v11):
  - full kernel: 32.4 TOPS;
  - no weight conversion: 31.7;
  - no FFMA epilogue: 33.9;
  - L1-hot loads: 39.1;
  - no global loads: 49-50;
  - no global loads, no smem stores or barriers: 67-70;
  - pure ldmatrix + mma loop: 86-90.

  Data movement and the barrier phase cost the most, not the math. All-zero activations gave +17-20% (datapath toggling).
- **What did not help:**
  - a CUTLASS-style pipeline on 128 x 128 (27 TOPS, lower clock);
  - a 256 x 128 tile (no change);
  - 2 blocks/SM (20-24 TOPS);
  - interleaved stores (-7%);
  - GA 64 on the CUTLASS pipeline (spills, 16 TOPS).
  The per-token CUTLASS-style probe reached 38-40 TOPS but is not accurate.
- **No clock or power control on Kaggle** (pg v7). `nvidia-smi -lgc`, `-pl` and `-rgc` return "insufficient permissions". `-lmc` is "not supported" on this GPU. `-ac` offers memory 5001 MHz only.
- **Arithmetic.** 1400 tok/s pp2048 is 1.46 s per ubatch. At 32 TOPS the GEMMs alone take 1.55 s, so the gate needs a GEMM at about 41+ TOPS sustained on the slower GPU plus at most 0.25 s for everything else.

### Broken or open
- **Box variance is ±7%.** One GPU often runs 15-20% slower (hotter). The fixed TP split then turns into AR wait on the faster GPU (370-400 ms in v13-v15).
- `k_pf_ab` still takes about 50 ms, 3-4x more than its FLOPs need, and I don't know why yet.
- `k_pf_gdnc` uses only 24 blocks (one per head) and a sequential 64-step T solve. There is room left (78 ms).
- gemm14 (accurate GA 64 on the CUTLASS pipeline) is correct but spills 640 B. gemm15 makes the same pipeline fit in registers: quarter tiles, dx read per pass, bias subtracted before the barrier, loads issued mid-stage, about 12 B of spill. It is correct but sustains only 27-28 TOPS against 31-32 for gemm9 (pg v8), because the extra A-fragment reloads cost more than the pipeline saves.
- No decode numbers were measured this round. The decode engine is untouched apart from shared headers.

### Next steps
1. **GEMM**, the only lever big enough for 1400.
   - The CUTLASS-style pipeline with per-64 activation scales is now measured both ways: gemm14 spills and gemm15 is slower. A more promising route is a different activation format that removes the per-group FFMA, so the per-token pipeline speed (1010 ops/clk, 38-40 TOPS) can be kept.
   - Or try LLM.int8-style outlier extraction: a per-token int8 main GEMM through the CUTLASS-like gemm13 path (38-40 TOPS), plus a small dense fp16 side GEMM over the union of outlier channels per ubatch. Check it with the KL suite; GA 0 alone fails.
2. **Non-GEMM**, now about 340 ms:
   - more blocks for `k_pf_gdnc` (split value columns, blocked T inverse);
   - find the `k_pf_ab` slowdown;
   - fuse the FA output into the q8 producer;
   - fuse conv + q/k norm into the chunk kernel's staging.
3. Keep `pf_ga=64` and check new GEMM ideas with `tests/prefill_check.py`. KL against the decode path has stayed at or below 1.1e-3.
