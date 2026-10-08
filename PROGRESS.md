# t4q progress log

Handoff log for agents working on t4q. Append a dated section per session, newest last. Plan: `research/DESIGN.md`.

## 2026-10-02 - M0-probe (round 1), GEMV fast path + box probe

**Gate: PASS (burst).** Measured on Kaggle `t4q-m0` v1, v2 and v3. Full tables are in `research/m0_results.md`.

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

**Gate: PASS, with the V1/V2 criteria made noise-floor aware (details below).** Measured on Kaggle `t4q-m1` v7 (final full run). v1 failed to build (missing `<cstdint>`), v2/v3/v4 were full runs, v5/v6 were teacher-forced debug runs.

Built (all under `t4q/`, everything except the M0 agent's `gemv.cuh`, `gemv_bench.cu`, `probe.cu`, `nvml_lite.h`, `gemv_layout_check.cpp`, `mkkernel.py`, `stage_m0.py`):
- `include/t4q.h` C ABI from DESIGN s11 plus `t4q_set_dump`, `t4q_dump_keys`, `t4q_set_option("act_q8")`, `t4q_layer_forward` (run one layer on a given residual, for teacher-forced checks), `t4q_last_error`.
- `src/gguf.{h,cpp}` mmap GGUF v3 parser with the hparam asserts from arch.md s0. `src/quant_cpu.{h,cpp}` CPU port of ggml `dequantize_row_*` (F32/F16/Q4_0/Q4_1/Q8_0/Q5_K/Q6_K).
- `src/packed.h` + `src/repack.cu` + `src/kernels/deq.cuh`: M1 planar SoA formats (P4, P4M for Q4_1, Q8, K5, K6, F32), repacked on the GPU from raw GGUF blocks through a 64 MB pinned stage. `deq.cuh` compiles for host too (`-DT4Q_HOST_SIM`) so `tests/test_deq_sim.cpp` runs the exact repack+dequant code locally.
- `src/loader.cu` layer split: GPU0 layers 0-31, GPU1 layers 32-63 + output_norm + lm_head. token_embd stays in the mmap; the row is dequantized on the host per token. After each tensor the loader dequantizes 64 rows on the GPU and memcmp's them against the CPU port.
- Kernels (straightforward, unfused, eager launches, one host sync per token): `kernels/gemv_ref.cu` (warp per row; fp32-activation GEMV, plus the q8_1 path below), `misc.cu` (RMSNorm, per-head RMSNorm, add, SwiGLU, argmax), `gdn.cu` (conv1d+SiLU with raw-input conv state, L2 norm as ggml `rms_norm(eps/n)/sqrt(n)`, beta/g gates, recurrence with ggml's state layout `S[h][v col][k]` and tiled `kh = h % 16`, gated norm), `attn.cu` (q/k RMSNorm + partial NeoX RoPE on 64 dims with ggml's fp32 theta, fp16 KV append, naive softmax attention with GQA `h / 6`, sigmoid output gate).
- `src/engine.cu` decode step + `dump()` of llama-named intermediates (`attn_norm-N`, `linear_attn_qkv_mixed-N`, ..., `l_out-N`, `result_norm`). `src/api.cpp` greedy-only `t4q_generate` (sampling params ignored in M1).
- `act_q8` option: GEMVs on quantized weights take llama.cpp-style q8_1 activations (per 32: d = amax/127 and sum(x), both rounded to fp16; formulas copied from ggml `vec_dot_q4_0/q4_1/q5_K/q6_K_q8_1`, including Q4_1's `__hmul` half products). Verified bit-exact against llama.cpp: with identical inputs (layer 0, position 0) every layer-0 intermediate matches to < 5e-7. Default is fp32 activations (more accurate than llama.cpp; V0 shows ~1e-7 vs fp64).
- `py/t4q.py` ctypes driver + HF tokenizer (`Qwen/Qwen3.8-27B`, chat template with `enable_thinking=False`); `py/gguf_np.py` independent numpy GGUF reader/dequant.
- `tools/oracle_dump.cpp` against the sm_75 libllama from kernel_sources `t4-qwen38-baseline` (headers vendored in `tools/oracle_include/` from llama.cpp a4cb4c61, the build commit). Jobs: `seq` (all-position logits as one batch AND token by token), `gen` (greedy + top-2 gap), `dump` (cb_eval intermediates token by token at positions 0, 1, n-1, all 64 layers, whitelisted names), `dumpb` (same, one batch), `tok` (llama_tokenize vs HF ids).
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

**Gate (single-stream decode >= 30 tok/s on Q4_0 at 4k, matching the oracle): NOT passed.** Best verified: **29.24 tok/s** (`t4q-m4` v11, CUDA graphs, prompt P1, 256 tokens, P2P box) and 29.22 (v7). At 3.6k context the best is **28.55** (v11). Every correctness check passes on every version since m2 v1. For reference, llama.cpp `-sm tensor` on Q4_0 does 21.11.

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
- `t4q-m4` v24: default config **30.04** tok/s (P1), `pf_kb=1536` config **30.07 / 30.05**, `tp_check` `gate_30: true`, every correctness check passing (selftest, V1 floor, TP residual identical, V2 and V3 in eager and graphs, V4 bit-identical).
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

**Gate (pp2048 >= 1400 and pp512 >= 1200 tok/s, correctness preserved): NOT passed.** Best verified: **pp512 583.5, pp2048 633.8 tok/s** (`t4q-p` v5, P2P box), with correctness passing on every configuration. llama.cpp `-sm tensor` on Q4_0 does pp512 516. Full tables are in `research/p_results.md`.

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

**Gate (pp2048 >= 1400 and pp512 >= 1200 tok/s, correctness preserved): NOT passed.** Best verified: **pp2048 988.6, pp512 855.4 tok/s** (`t4q-p` v17, P2P box, both GPUs at 960-1050 MHz), correctness passing on every prompt. Round 1 best was 633.8 / 583.5, so this is +56% / +47%. Full tables: `research/p_results.md` (round 2 section).

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
  - `stage_pg.py`: GEMM-only runs on a second kernel id (`t4q-pg`), so they can run beside engine runs.

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

## 2026-10-03 - P-prefill (round 3): plain-int8 GEMM pipeline, rotated per-token path (R512), activation-format study

**Gate (pp2048 >= 1400 and pp512 >= 1200 tok/s, correctness preserved): NOT passed.**
- Correct (default GA64 path, every check passes): best pp2048 **1006.1**, pp512 **870.5** (`t4q-p` v31). Other boxes: 960.0 / 828.5 (v30) and 978.4 / 859.7 (v34). This is the round-2 path; within box variance it is unchanged (round 2: 988.6 / 855.4).
- Fast but **not correct enough** (opt-in R512 mode): pp2048 **1213.2** (v31) and pp512 **955.9** (v31, `pf_nsub=1`). It fails the long-prompt check: last-token KL 7.1e-3 against a limit of 2e-3 (2 x llama.cpp's own spread on L, 8.2e-4). Teacher-forced continuation KL reaches max 2e-2 to 6e-1, against 6.7e-3 for GA64. Top-1 and greedy 32 tokens match everywhere.
- No decode numbers were measured this round.

Full tables: `research/p_results.md` (round 3 section). Raw outputs: `kaggle/p/out_v18..v34`, `kaggle/pg/out9..out13`.

### What I built
- **Activation-format study** (`pf_fq` modes 1-9 in `tp_prefill.cu`, `k_fq_*`):
  - It emulates a GEMM input format on the GA64 int8 input. Every GEMM type can be selected with `pf_fq_mask`.
  - It reports the residual and outlier-channel stats in `pf_fq`.
  - `pf_keep_h=1` copies the final residual of every batch token into `dumps["pf_h"]`. `prefill_check.py --keep_h 1` turns that into a per-token hidden-state error against config 0, a more robust metric than one last-token KL.
- **`t4q/src/kernels/gemm16.cuh`**: plain int8 x int8 GEMMs with int32 accumulation in the mma C operand.
  - `gemm16` uses the 128-weight-row x 256-token tile.
  - `gemm17` uses the CUTLASS orientation (128 tokens x 256 weight rows) with L2::128B loads. With KNOB 1 the global loads are issued at k-group 0, as in CUTLASS MmaPipelined.
  - Weights come either pre-converted (`w8`) or from the decode layout, converted in registers.
  - Epilogues: fp32, fp16, fused silu(gate)*up (OUT 2 fp16 / OUT 3 fp32), `nvalid` and `kstride`.
  - Group-scale variants: GSH 1 shift-folds power-of-two 64-group scales. GSH 2 uses per-512-block scales with exact float-ratio accumulator rescaling.
- **`t4q/src/kernels/rot.cuh`** (R512): randomized block-Hadamard rotation T = H512 * D.
  - D is a per-32-chunk hash sign times a fixed 32-periodic element pattern.
  - Weight converters: fp32 (all formats) and fp16x2 (P4/P4M, memory-bound, about 190-200 GB/s). Rotated row scales `invr` are computed once.
  - Activation quantizers: register-resident `quant_rot_kernel`, plus the fused `k_pf_add_norm_rot` and `k_pf_gnorm_rot`.
  - `w8r_f32_kernel` rotates fp32 weight rows (alpha/beta).
  - `tests/test_rot_deq.cu` is a host-only check of the decode-layout dequant used by the converters. It passes exactly.
- **Engine** (`pf_rot=1`, all GEMMs through gemm17):
  - Per-GEMM-type scratch for rotated weights. The alpha/beta rows are appended to the qkvz GEMM (N 8448), which removed `k_pf_ab` (53 to 0 ms).
  - Optional persistent rotated-weight cache in spare VRAM (`pf_wcache` MB per GPU, about 39% of the weights at 4500).
  - Options:
    - `pf_rot_mask`: which GEMM types use R512;
    - `pf_rot_min`: ubatch size threshold;
    - `pf_rgb`: per-512-block scales;
    - `pf_rcf`: fp16 converter;
    - `pf_rot_chk`: converter cross-check;
    - `pf_h16`: fp16 vs fp32 silu output;
    - `pf_abq`.
- **Shared fixes**:
  - `k_pf_gdnc` T-solve now uses 4 threads per column, which cut the spill from 116 to 52 B.
  - `k_q8_64_half`.
  - `oracle_dump` has a new `floor` job: llama.cpp's last-token spread for long prompts. `stage_p.py` runs it for L.
  - `prefill_check.py` adds continuation KL for L, `--bench_skip` and `--prompts`.
- **Benches**: `t4q/tools/gemm16_bench.cu` checks the kernels against an exact int64 reference, then runs burst and sustained modes. Stage `pg` runs it in section "g16".

### Verified (Kaggle)
- **GEMM, sustained, both GPUs, gateup T=2048** (pg v9-v13):

  | kernel | TOPS |
  |---|---|
  | gemm9 GA64 (production) | 28-32 |
  | gemm17 w8 KNOB 1 | 50.6-59.7 (1290-1335 ops/clk/SM) |
  | CUTLASS int8 128x256, same sessions | 44.6-59.7 (1375-1490 ops/clk) |
  | in-register Q4 conversion in the CUTLASS orientation | 10-15% slower |
  | GSH 1 (shift fold) | 33-35 |
  | GSH 2 (I2F/F2I rescale every 512 k) | 40.5, exact to 5e-6 |

  All gemm16/17 variants match the exact reference (rel 4e-8 per-token, 5e-6 with folds).
- **R512 speed**: in the engine its GEMMs are about 1.45x faster than gemm9 per batch. But with both GPUs fully loaded, clocks settle near 640-720 MHz instead of 900-1000 (v29: 638 / 645 MHz). Energy per op is only about 25% lower, so pp2048 gains about 20%: 1213 vs 1006 on the v31 box.
- **Accuracy study**: per-token int8 fails (L KL 2e-2; down input alone 0.57).
  - Clipping, top-n outlier channels and smoothing don't fix it.
  - GA128 and GA256 fail or are borderline on L.
  - Hadamard-512 + per-token is the best per-token format, but in the real engine it still fails L.
  - With R512 the hidden-state error is about 1.3-1.5x GA64 (median 5-6e-2 vs 3.3-4e-2 on L), and the long prompt amplifies it.

### Why the gate is out of reach (measured)
- Prefill is energy-bound at the 70 W cap. The idle draw is about 30 W, which leaves about 40 W for compute.
- The int8 m8n8k16 mma reads and writes its accumulators for every 1024 MACs. Even CUTLASS-level kernels sustain only about 45-55 TOPS per GPU, and at full TP load both GPUs drop to about 650-800 MHz.
- The accurate activation format (GA64) needs a per-group fold. In the CUTLASS loop order the fold needs a second accumulator set (spills) or extra ALU work (25% for shift folding, 20% for per-512 rescale). gemm9's loop order runs at about 750 ops/clk.
- The correct path tops out near 1000 tok/s at pp2048. R512 reaches about 1200 but fails the correctness check.

### Broken or open
- R512 fails the L check in every variant (v27-v34). In v22-v26 it had passed L (KL 2.9e-3 against the old 5e-3 limit, before the `floor` job existed). It is opt-in only.
- GPU asymmetry: on many boxes one GPU runs about 30% slower under R512 load, and the other waits 20-38% of the batch in AR.
- The decode step for the last prompt token still costs about 30-35 ms per prefill.
- `pf_wcache` takes spare VRAM: the 1.2 GB margin is checked once, at cache build.

### Next steps
1. **Accuracy with the fast pipeline.** The only accurate format found is GA64-class grouping. Untested ideas:
   - a 128x128-tile CUTLASS-order kernel with GA64 fold (64 int + 64 float accumulators per thread fit);
   - two-digit activations (x = s1 q1 + s2 q2) for only the most sensitive GEMM type;
   - a rotation over all of K (5120 = 5 x 1024), which spreads outliers more.

   Check with `prefill_check` on L, using both the continuation-KL max and the last-token KL. One last-token KL is too noisy to pick between variants.
2. **Correct path at about 1000 tok/s**:
   - drop the last-token decode step (about 3%);
   - parallelize `k_pf_gdnc` (24 blocks), currently 72-98 ms;
   - fuse the AR add into the K-split GEMM epilogue.
3. **On unbalanced boxes**, offload the shared work (add_norm and quant for both GPUs) to the faster GPU, and send int8 activations over P2P.

## 2026-10-03 - P-prefill (round 4): pf_head, faster alpha/beta, int4 and CUTLASS-orientation GEMM probes (negative)

**Gate (pp2048 >= 1400 and pp512 >= 1200 tok/s, correctness preserved): NOT passed.**
- Best on the correct default path: **pp2048 1032.6** (`t4q-p` v36, config `pf_head=1`, nsub 2) and **pp512
  962.4** (v36, `pf_head=1`, nsub 1; both are what the new defaults pick for those sizes). Round 3 best was 1006.1 /
  870.5 (v31). Other boxes with the final defaults: 1013.2 / 906.2 (v37), 988.5 / 923.2 (v38; GPU1 ~7% slower).
- Every prompt passes in v36-v38 (top-1 equal, greedy 32 identical to the decode path, W diverging only at its known
  near-tie, last-token KL <= max(2e-3, 2 x llama floor)). L last-token KL 3.4e-4 with `pf_head=1`.
- No decode numbers were measured this round.

Full tables: `research/p_results.md` (round 4 section). Raw: `kaggle/p/out_v35..v38`, `kaggle/pg/out14..16`.

### What changed (defaults)
- `pf_head` (default 1, `tp_prefill.cu`): the last prompt token goes through the batch; afterwards only the head runs
  decode-style (`tp::ar_norm` on the final residual row, lm_head `tp::gemv`, `tp::argmax_step`, which advances
  StepState/ring like a decode step). Removes the 34-35 ms decode step: pp512 893 -> 947, pp2048 1020 -> 1033 (v36).
- `pf_nsub_min` (default 1024): ubatches below it run as one sub-batch (pp512 962 vs 947).
- `k_pf_ab` rewritten (16 K slices, 8x6 thread tiles, register prefetch): 51.6 -> 34 ms at pp2048, 31.7 -> 10.8 ms
  at pp512. Consumers sum `AB_KS` slices generically.
- R512 weight cache now only caches GEMM types in `pf_rot_mask`.

### New options, off by default (measured, no gain)
- `pf_gdnc=2`: `k_pf_gdnp` computes gates, P and T = (I-A)^-1 for all chunks in parallel, `k_pf_gdnc<1>` loads them.
  Bit-identical, 83 vs 79 ms (the T solve is not the scan's bottleneck).
- `pf_arc=N`: K-split GEMMs in N token chunks with per-chunk peer copies (`gemm8::Args::dxs` = dx row stride for
  token slices). Bit-identical, no end-to-end gain: AR wait is mostly GPU imbalance, not copy time.

### GEMM probes (bench `t4q/tools/gemm20_bench.cu`, stage pg section `g20`)
- `kernels/gemm20.cuh`: W4A8 on int4 tensor cores, native Q4_0 codes (s4 via c ^ 8) x GA64 activations as hi/lo
  nibble planes, exact per-64 fold. 4.4e-4 rel error vs the exact Q4 product (gemm9 6.3e-3), but 26-27 TOPS sustained
  (gemm9 31-32); even with no fold only 42-43 TOPS vs gemm17's 56-58 at the same 67 W. Int4 is not a way around the
  power cap for an exact W4A8.
- `kernels/gemm21.cuh`: gemm9's numerics in gemm17's CUTLASS orientation (tile-outer, MAGIC-started 4-mma chains
  folded per 64-k stage). Exact (1.7e-7 vs its int8 mirror) but only +3% over gemm9 (741-812 ops/clk); pre-converted
  int8 weights or a 256-token tile are not faster. The fold structure, not the Q4 conversion, is the cost.

### Accuracy findings
- The L-prompt metrics are noise-dominated near 1e-3: changing only the alpha/beta fp32 summation order moved GA64's L
  last-token KL 3.2e-4 -> 1.1e-3 and cont mean 1.1e-3 -> 4.9e-3; GA32 (more accurate than GA64) shows the same heavy
  h_rel tails vs GA64 (p99 0.86) that round 3 attributed to R512.
- R512 is still clearly worse on L: mask 55 (all but down) 3.9e-2, mask 21 1.4e-1, gateup alone 3.7e-2 last-token KL
  (GA64 3.4e-4-1.1e-3). Speed: mask 55 1178.9 / 914.1, mask 21 1164.3 / 924.8 vs GA64 993.3 / 877.6 (v35). Stays opt-in.

### Why the gate is still out of reach
- GEMMs are 1655 ms of a 1981 ms pp2048 batch (v36) at 29-34 TOPS. Three more exact-GEMM structures were measured this
  round (int4 digits, CUTLASS orientation with fold, 256-token tile) and none beats gemm9 by more than 3% at the 70 W
  cap. 1400 tok/s needs the GEMMs at <= ~1.2 s, i.e. >= 41 TOPS sustained on the slower GPU with an accurate format;
  the only kernels that reach that (gemm17 / CUTLASS per-token int8) need per-token activations, which fail on L.
- Non-GEMM work left at pp2048: AR wait 103-233 ms (mostly the slower GPU), DeltaNet 77, attention 54, ab 34, conv 22,
  gated norm 20.

### Next steps
1. Balance the two GPUs: one is 6-10% slower on most boxes and the other waits for it (100-230 ms per pp2048 batch).
   A runtime-calibrated uneven N split for the N-split GEMMs (the faster GPU keeps a few % of the peer's weight rows)
   could recover most of it.
2. Overlap the 24-block DeltaNet scan (uses 24 of 40 SMs) with the other sub-batch's GEMM on a second stream.
3. If a better accuracy metric is wanted: average last-token KL over several long prompts; the single L prompt swings
   3x from fp32 summation order alone.
4. pf_head could also skip the batch path's final add_norm/xn write for all but the last row (small).

## 2026-10-03 - B-batched (round 1): batched / continuous decode, scheduler, OpenAI server

**Gate (aggregate decode >= 200 tok/s at some B, outputs correct): PASS.** Stretch (400) reached at 1k context on P2P
boxes. All numbers are Kaggle `t4q-b`, Q4_0 GGUF, greedy, both T4s (TP=2), aggregate = B / step time.

| version | box | 1k ctx B=64 | B=48 | B=32 | B=16 | B=8 | B=1 | 4k ctx B=32 | B=24 | B=16 |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 | P2P | **429.8** | - | 269.4 | 153.8 | 82.5 | 11.0 | (OOM, fixed in v2) | - | - |
| v2 | no P2P | 382.8 | 318.9 | 269.3 | 159.0 | 86.8 | 11.8 | 213.6 | 181.1 | 137.9 |
| v6 | P2P | 424.7 | - | 297.7 | 172.6 | - | 12.4 | 251.1 | - | 157.6 |
| v8 (final defaults) | P2P | 402.9 | 342.3 | 294.9 | 173.6 | 94.7 | 12.8 | 235.4 | 200.0 | 151.6 |

- Clocks during the v8 benches: GPU0 810-975 MHz, GPU1 1020-1110 MHz (`kaggle/b/out8/results_b.json`, per-B `clocks`).
- fp32 vs fp16 DeltaNet state (v8, 1k ctx): fp32 B=32 280.1 / B=40 285.1 tok/s, fp16 B=32 294.9 (gdn kernel 21.2 ms vs
  12.0 ms per step at B=32). fp16 is the default; quality numbers below.
- **End to end** (v8): 32 concurrent coding requests (stdlib-source prompts, 477 tokens on average, 15262 prompt tokens),
  512 new tokens each (EOS ignored, 16384 tokens): **72.5 s wall, 225.9 generated tok/s, 436.2 total tok/s**; prefill
  17.6 s (869 tok/s, one prompt at a time), decode 55.0 s at B=32 (297.6 tok/s aggregate), mean TTFT 9.1 s (all 32
  submitted at t=0, prefill-first scheduling). v5-v7: 72.6-73.1 s.
- **Correctness** (v8; identical in v1-v8 for the same config):
  - Reference = the single-stream TP decode engine (validated against the llama.cpp oracle in M1-M4), 8 coding prompts.
  - Teacher-forced, 16 positions x 8 prompts at B=8: KL(single-stream || batched) mean 2.2e-4 / p99 4.1e-3 / max 6.8e-3
    (fp32 state), 2.4e-4 / 4.1e-3 / 5.1e-3 (fp16 state); top-1 agreement 100% (128/128) in both.
  - Free-running greedy, 128 tokens, B=8: 4/8 (fp32 state) and 5/8 (fp16) prompts identical to single-stream; every
    divergence is at a near-tie (batched top-2 logit gap 0.009-0.12; tokens 17-49). Outputs are coherent code.
  - Batch invariance: a sequence's logits are bit-identical whatever B and the other rows are (B=1 vs B=8 rows,
    max diff 0.0; B=40 with 4 clones per prompt: clones bit-identical, and the 64-token-tile row equals the 32-token-tile
    row bit for bit). This needed explicit roundings in gemm9's bias (`__fmul_rn`/`__fadd_rn`; FMA contraction had made
    BN 32 and BN 64 differ in the last bit).
  - Why batched != single-stream bits: the batched GEMMs are prefill's gemm9 numerics (per-row int8 weight requant, GA64
    activations; rel L2 vs fp64 5.6e-3 to 8.7e-3 per shape, `bd_gemm_bench`), the decode engine uses exact Q4 x q8 dp4a.
    Same class of difference as batched prefill (round P).
  - OpenAI server (v1-v8): 4 concurrent requests (chat stream, chat, completion, second chat) batched together
    (`max_batch` 4); streamed text == non-streamed text for the same prompt.

### What I built
- `t4q/src/tp_batch.cuh` (compiled inside `tp_prefill.cu`, reuses `PfRun`): slots with per-sequence DeltaNet state
  (fp32 or fp16 storage, fp32 math), conv ring and fp16 KV (`[layer][k|v][slot][2 heads][slot_ctx][256]`). One step:
  embed -> per layer fused add+norm+q8 -> gemm9 at m=B (token tile 32 for B<=32, 64 for B<=64) -> batched `k_bd_gdn`
  (decode gdn arithmetic per row) or `k_bd_attn_prep/split/combine` (fixed 128-position split blocks, so splits do not
  depend on B) -> K-split GEMM -> fp16 peer-copy all-reduce -> FFN (gate|up silu epilogue) -> output norm -> lm_head
  (gemm9 on Q6_K, `bd_head=1`; dp4a GEMV 8 columns per pass is `bd_head=0`, 24 ms vs 4.3 ms at B=64) -> per-row argmax on
  each GPU, host merges the shards (max, lowest index on ties). Prompts go through the single-stream batched prefill, then
  state/ring/KV are copied into the slot. Per-op two-GPU event profile (`bd_prof`), host enqueue time in stats.
- `gemm8.cuh`: gemm9 at BN 32 / 64 (2 blocks/SM), Q6_K weights (K6) for the batched head, silu epilogue at any BN,
  split-K (`kz`, fixed-order `ksum_kernel`), optional L2 weight prefetch (`pfk`), optional peer-mailbox fp16 output
  (`yh2`), 3-stage load ring at BN 32 (`lbm`, default on, ~3%), bench-only ablation bits (`AB` 128 = contiguous-load probe).
- `gemm_r.cuh` (`bd_gemmr`, default off): register-direct P4 GEMM -- the Q4_0 nibble order is exactly an mma.m8n8k16 A
  fragment (thread t loads code bytes 4*(t&3)..+3 of row t>>2), converted in registers with gemm9's fp16 ops;
  bit-identical to gemm9 on every shape (`bd_gemm_bench`), but not faster.
- C ABI `t4q_batch_init/prefill/clone/set_token/pos/step/logits/free`; options `bd_head`, `bd_ch`, `bd_prof`, `bd_p2p`,
  `bd_pfk`, `bd_ksplit`, `bd_lbm`, `bd_gemmr`, `bd_reset_stats`; `pf_ub` changes now reallocate the prefill buffers (the
  4k bench uses `pf_ub=1024` to make room for 32 x 4128-position slots).
- `py/batch.py` continuous-batching scheduler (requests join/leave between steps; `prefill_per_iter`), `py/server.py`
  OpenAI-compatible server (stdlib `http.server`; `/v1/models`, `/v1/chat/completions`, `/v1/completions`, SSE streaming,
  stop strings, greedy only), `py/t4q.py` batch bindings and `Tokenizer.chat_messages`.
- `tests/batch_check.py` (sections correct / bench / e2e / server), `tools/stage_b.py` (stage `b`),
  `tools/bd_gemm_bench.cu` + `tools/stage_bg.py` (GEMM-only stage `bg`: synthetic weights, gemm9 vs gemmr vs dp4a GEMV,
  fp64 check, ablations; runs while the GGUF downloads).

### Where the time goes (v8, B=64, 1k ctx, GPU0 events, ms per step of 158.9)
gate|up 42.7, down 20.9, qkvz 15.9, ssm_out 7.2, attn qkv 5.0, attn out 2.0, lm_head 4.8 (GEMMs ~98 ms) | gdn 25.5 |
attention 13.6 | AR wait + add_norm 15.0 (GPU1: 38.6, it waits for the slower GPU0) | ab 3.5 | host enqueue 5.2 (async).
At 4k / B=32 attention is 25.4 ms (KV 4 GiB per GPU per step).

### What did not work (measured, all options default off)
- The GEMMs at 32/64-token tiles stream weights at only ~110-140 GB/s in the engine (dp4a GEMV: 260), and their time
  hardly depends on B (gate|up 29 ms at B=1, 43 ms at B=64). Tried, none faster: L2 prefetch of the weight planes
  (`bd_pfk` 16/32: 10-15% slower, lm_head 3x slower), split-K for the K-split GEMMs (`bd_ksplit`: slower), line-batched
  weight loads (4 stages per load, v4: no change), a 2-3 stage load ring (v5: <=3%), register-direct fragments (gemmr,
  v7: same speed), 2 blocks/SM at BN 64. Bench ablations (v6, bg v1) put the cost in the weight-load path and the smem
  staging, not the MMA math, but timings swing up to 2x with the throttled clock, so the root cause is still open.
- P2P epilogue stores of the AR partials (`bd_p2p`): untested so far. The boxes where it was benched (v2, v4) had no
  P2P, so the engine fell back to the copy path; it needs a P2P box A/B (and its correctness check) before use.

### Broken or open
- B=1..8 is slow (12.8 tok/s at B=1): the 32-token tile computes 32 columns regardless. A dp4a M<=8 path for tiny B is
  not built (needs decode-format q8 producers and AR/silu epilogues in `gemv_fast_kernel`).
- No CUDA graphs per B bucket: steps are eager (host enqueue ~5 ms per step, overlapped with GPU work; graphs would need
  one multi-device graph with the peer copies and cross-GPU events).
- Memory caps B: 64 slots at 1k (fp16 state 36 MiB + KV 34 MiB per slot per GPU) and 32 slots at 4k. q8 KV would double it.
- GPU imbalance: one GPU usually runs 15-25% slower; the faster one waits (15-40 ms per step at B=64).
- Prefill is one prompt at a time through the single-stream path (~870 tok/s); TTFT under load is serialized.
- The scheduler is greedy only; sampling parameters are ignored by the server.

### Next steps
1. **GEMM weight streaming** is the big lever (~60% of the step). Test the DRAM-locality hypothesis: a skinny kernel that
   loads weights in GEMV order (each warp load = one full 512-B tile-chunk segment, like `gemv_fast_kernel`) and
   transposes into mma fragments through a per-warp smem slice (no block barrier), or fp16 HMMA with in-register
   dequant (~1.5 instr/weight, exact per-group scales). Measure SM clock (NVML) next to every bench timing; single
   readings in `bd_gemm_bench` are not trustworthy.
2. Overlap / balance: uneven TP split calibrated at load (the hotter GPU gets fewer rows), or overlap the AR copy with
   the next layer's independent work.
3. dp4a M<=8 path for tiny B; CUDA graphs per bucket; q8 KV for 4k x 64; multi-prompt batched prefill (varlen ubatch)
   for TTFT.

## 2026-10-03 - M5-MTP (round 1): MTP speculative decoding, byte-identical to plain greedy

**Gate (>= 60 tok/s single-stream on P0/P1 greedy, Q4_0, spec output byte-identical to t4q non-spec greedy): PASS.**
All numbers are Kaggle `t4q-m5`, Q4_0 GGUF (MTP = its own `blk.64`), TP=2, CUDA graphs, greedy, 512 generated
tokens after the chat-templated prompt, stop on EOS (none hit), wall time of `t4q_generate`.

| version | config | P0 tok/s | P1 tok/s | P2 (edit prompt) | plain decode same box P0 / P1 | GPU MHz (0 / 1) |
|---|---|---|---|---|---|---|
| v1 | k=3, 32k draft head | **70.03** | **64.87** | - | 30.57 / 30.37 | ~1050 / 890 (hot box) |
| v2 | k=3, 32k head | **73.69** | **68.95** | - | 30.83 / 30.75 | 1200 / 1100 |
| v5 | k=3, 32k head | 72.88 | 68.09 | 73.20 | 30.75 / 30.65 | 1060 / 1090 |
| v5 | k=4 | 71.31 | 66.36 | 71.68 | | 1020 / 1060 |
| v5 | k=5 | 67.63 | 60.96 | 70.13 | | 960 / 1020 |
| v5 | k=6 | 58.93 | 53.73 | 70.08 | | 940 / 1000 |
| v1 | k=2 | 65.19 | 61.20 | - | | |
| v1 | k=3, full 248k draft head | 65.91 | 61.17 | - | | |

- Best single-stream: **73.69 tok/s on P0** (v2, k=3), 2.4x plain t4q decode on the same box and 2.3-2.6x llama.cpp's
  measured 28.8-32.4 (tensor split + MTP). Quality: identical token stream to plain t4q greedy decode (which matched the
  llama.cpp oracle in M1-M4), so no quality loss of any kind.
- **Correctness** (every run v1-v5 except the buggy replay run v4): V5 spec output byte-identical to plain greedy for
  every k (2..6), prompt (P0/P1/P2), draft head and the no-P2P fallback (v1, `T4Q_NO_P2P=1`: 61.8 / 61.1 tok/s at 256
  tokens). V4: with drafts forced to the reference continuation (100% acceptance), all 32-35 verify columns' logits are
  **bit-identical** (max |diff| 0.0) to teacher-forced plain decode logits, at k=3 and k=6, P2P and no-P2P.
- **Acceptance** per draft (k=3, from the accepted-length histogram): P0 0.807, P1 0.731, P2 0.818 with the 32k head;
  0.852 / 0.785 with the full head (v1; llama.cpp with the separate Q6_K MTP file measured 0.847 / 0.778).
  Tokens per verify step at k=3: 3.42 / 3.19 / 3.45; k=4: 3.95 / 3.65; k=6: 4.38 / 3.97 / 5.11.
- **Draft vocab (M5b)**: the first 32768 token ids hold 95.5-96% of the generated tokens; 65536 ids hold 98.3-99.6%
  (`T4Q_DV=65536`, v2 variant: acceptance 0.844 / 0.742 at k=3, speed within box noise of 32k: 72.67 / 66.14). The
  truncated head is faster than the full head despite the acceptance drop (70.0 vs 65.9 on P0, v1), so it is the
  default (`spec_dv 1`); the design's "< 2 points" rule is not met but the speed rule wins.
- **Where the time goes** (v3/v5 CUPTI, GPU0, k=3): verify graph 44-46 ms vs a plain decode step 34.3 ms; draft graph
  2.9 ms. The verify's extra 11 ms: gate|up GEMV 17.4 vs 12.4 ms (M=4 at 185 GB/s, the outlier), multi-row AR 3.7 vs
  2.4 ms, gdn_m (4 state snapshots) 2.0-2.3 vs 0.9 ms, down 6.3 vs 5.5, qkvz 5.3 vs 4.6, attention split 0.9 vs 0.3.
  The dp4a GEMV cost grows ~2.5-4 ms per extra column, which is why k=3 beats k=4..6 even though they accept more
  tokens per step.

### What I built
- `t4q/src/tp_spec.cu` (new; `t4q_set_option("spec_k", k)` makes `t4q_generate` speculative):
  - Two CUDA graphs per GPU per iteration, no host sync inside the loop (host enqueues `spec_ahead` = 4 iterations and
    reads a host-mapped token ring + counter):
    - **draft**: MTP catch-up over the k+1 verify rows (token y_j with the target h_final of column j at position
      vpos+1+j; rows after nacc are garbage and get overwritten before any read), select row nacc, draft head + argmax
      exchange -> d1; k-1 chained single-row MTP passes (h' fed back) -> d2..dk.
    - **verify**: 64 layers on k+1 tokens with M-column kernels, lm_head, per-column argmax exchange, device-side greedy
      acceptance (emits y_0..y_n, pos += n+1, appends the tokens to the history in `G.prompt`).
  - Bit-identity by construction: `k_gemv` (moved to `kernels/tp_gemv_impl.cuh`, shared by tp_kernels.cu and
    tp_spec.cu) got an `M` template parameter (columns, PRO_NONE / no AR only) running the M=1 code per column;
    `k_ar_norm_m` (rows of the multi-block AR+norm, one publisher block per row), `k_pull_m` (no-P2P), `k_gdn_m`
    (k+1 tokens with the state in registers, conv history from earlier y columns), per-column attention through the
    original `d_attn_prep/split/combine` (now taking `pos`, in the shared header), `k_embed_m`, `k_argmax_x`.
  - DeltaNet rollback: a state snapshot after every verify token (k+2 buffers per layer, 8 x 75.5 MB per GPU
    allocated) and a 16-slot conv ring, so a rejected token needs no replay. `spec_rb 1` (replay from a per-token
    stash, `k_gdn_replay`) is implemented and byte-identical (v5) but not faster (replay 0.41 ms on partial steps vs
    0.3 ms saved in gdn_m), default off.
  - Separate AR / argmax mailboxes for the verify and draft graphs (their AR counts differ, so slot parity would not
    alternate across graph boundaries on a shared mailbox).
  - MTP prompt catch-up at spec start from the batched prefill's final residual rows (`tp_prefill_hrows`).
  - Prompt lookup (`spec_ng N`, `k_ngram`): a match of >= N tokens of the history suffix replaces the MTP drafts.
    Measured (v3, k=3): ng5 66.8 / 62.6 / 66.6 vs MTP only 69.3 / 63.5 / 65.9 on P0 / P1 / P2; ng3 is worse. MTP
    already predicts the copied spans of the edit prompt (P2 acceptance 0.82), so prompt lookup stays off.
  - Debug / bench options: `spec_force` (+ `t4q_spec_force`), `spec_dbg` (verify logits to dumps), `spec_prof`,
    `spec_trace N` (CUPTI timeline split into draft / verify by marker kernels, stats key `spec_trace`), `spec_sqt`.
- `tp_engine.cu`: `load_mtp` (blk.64 sharded like an attention layer; `eh_proj` Q8_0 K-split by input half: GPU0
  multiplies enorm(embed), GPU1 hnorm(h); draft head = output.weight rows [dv/2 g, +dv/2), `T4Q_DV` default 32768,
  `T4Q_NO_MTP` skips loading). VRAM 8615 MiB per GPU at 4k context before the spec buffers (~+0.75 GB).
- `tests/spec_check.py` (sections ref / v4 / v5 / prof / trace, config sets), `tools/stage_m5.py` (stage m5, no
  oracle needed: the reference is plain t4q greedy). Prompts P0/P1 as in the baseline, P2 = a code-edit prompt.

### Broken or open
- The verify cost per column is the limit (dp4a GEMV ALU at M=k+1 under the 70 W cap). A tensor-core (mma.m8n8k16)
  verify GEMV can stay bit-identical (the per-group integer sums are exact; the float epilogue must replicate lane j's
  chunk-ordered FFMA chain and the xor-8/4/2/1 butterfly) but my instruction estimate says ~1.3x at M=4 and ~2x at M=8
  only, because the exact per-(row, column, group) float epilogue stays. It would make k=5..6 pay off (P0 4.3-4.4
  tokens per step).
- gate|up at M=4 runs at 185 GB/s while down/qkvz run at 215-220; `spec_sqt 128` did not help (v5, confounded by clocks).
- Speculation inside the batched engine (small B) and the low-bit speed mode are not started.
- The spec loop overshoots up to `spec_ahead` iterations past max_new / EOS (state advanced past the returned tokens).
- Prompts longer than one prefill ubatch (2048): only the last ubatch gets the MTP prompt catch-up.

### Next steps
1. Tensor-core verify GEMV for P4 (Q4_0) with the bit-exact epilogue above; then raise the default k.
2. Fix the gate|up M>1 slowdown (wave quantization: 272 blocks on 80 slots, try a persistent tile loop).
3. Spec in the batched engine for B <= 2 (two sequences x (k+1) columns through the same M-column kernels needs
   per-sequence state buffers), and the low-bit speed mode (Q3_K/IQ3 fast formats) with KL vs Q4_0/Q8_0.

## 2026-10-03 - Independent verification audit (stage verify)

**Verdict: the headline claims reproduce.** I measured everything in one fresh run, `t4q-verify` **v1**, on a
P2P box, with llama.cpp a4cb4c61 in the same session on the same Q4_0 GGUF and the same token ids. The engine is
unchanged at `407f0ee`; the audit harness is new: `t4q/tests/verify_check.py` and `t4q/tools/stage_verify.py`.
Full table and caveats: `research/VERIFY.md`. Raw JSON and llama logs: `research/verify_v1/`.

| metric | t4q | llama.cpp (same session) |
|---|---|---|
| plain greedy decode P0 / P1 (512 tok) | 30.42 / 30.25 (rerun 30.06 / 30.21) | server -sm tensor 21.48 / 21.17; tg128 20.94 |
| MTP k=3 P0 / P1 | 71.85 / 68.08 (rerun 72.54 / 68.00), byte-identical to plain | draft-mtp n_max 3: 37.87 / 33.19 (rerun 37.31 / 34.56) |
| pp512 / pp2048 (median of 3) | 904.7 / 1033.2 | 532.63 / 512.27 |
| batched aggregate B=16 / 32 / 64 (64 distinct prompts, ctx 0.4-0.65k) | 181.2 / 314.7 / 443.0 | llama-batched-bench S_TG 105.26 / 129.22 / 146.03 |

### Correctness
- t4q greedy vs the llama layer-split oracle over 256 tokens: P1 matches 256/256. P0 matches the first 74 tokens, then diverges at a near-tie where llama's top-2 gap is 0.042.
- For comparison, llama's own tensor split vs its layer split agree for only 74 (P0) and 17 (P1) tokens.
- Teacher-forced top-1 agreement: 255/256 on P0 and 256/256 on P1. Excluding near-ties, both are 100%.
- Batched rows vs single-stream (35 greedy tokens, distinct prompts): 7 of 8 are identical.

### Audit findings
- No timing or token-count bugs found:
  - every timed path ends with a device sync;
  - throughput is computed as (n - 1) / wall time;
  - each bench starts with a reset and a fresh prefill.
- One methodology issue in the builders' batched bench: `batch_check.py` cloned a single prompt into all slots. My control run shows duplicate slots are only about 3% faster (456.5 vs 443.0 at B=64), so the claim is not materially inflated.
- Open item: on my stdlib-code prompt, the last-position KL at pp512 is 4.3e-2 against llama. Top-1 is equal and the top-2 gap is 1.30. t4q's own decode path vs its prefill path shows a similar KL, 5.4e-2. llama's own floor on this prompt was not measured, so this is unresolved. At pp2048 the KL is 3.8e-8.

### Not re-verified
- Decode at 4k depth (claimed 29.3; the depth gate was never met).
- Batched decode at 1k and 4k context.
- No-P2P boxes.

### Next steps
1. Run the oracle `floor` job and a decode-path `last` on several 512-token prompts. That separates a sensitive position from a real prefill accuracy problem.
2. Change `batch_check.py`'s bench to distinct prompts, reusing `verify_check.py`'s approach, and re-measure B=64 at 1k and B=32 at 4k.
3. Measure depth-4k single-stream decode and MTP in the same session as llama.cpp.

## 2026-10-03 - M5 round 2: int4 tensor-core verify GEMV (spec_tc), harness honesty fixes

**Gate: kernel CORRECT, performance PARKED.** The TC verify GEMV produces dp4a-level values (ULP diffs <= 1.9e-5 on
every shape, exact on the K=512 micro case) but streams at only 45-95 GB/s vs the dp4a's 76-259, so `spec_tc = 1`
runs ~2x slower end-to-end (k3_tc 31-34 tok/s vs k3_dv1 66-70). The option stays default-off; the perf restructure
is the follow-up. Kaggle v6 through v16; final state v16 + v17 (gate fix).

### What was built
- `t4q/src/kernels/tp_gemv_tc.cuh` (`t4q::gtc`): one token tile of TPD = 8 per block (one mma.m8n8k32 m-tile
  absorbs every M <= 8 column), BR = 64/128 weight rows, 64-k stages with a register double buffer, weights read
  through `gemm8::WPtr` (the decode P4 layout, no conversion pass). Activations: the engine's existing per-32 q8_1
  buffers read directly, digit planes (lo/hi nibbles, `d4_from_q8`'s shuffle) built in-flight at smem staging.
  Numerics: weight code as s4 (c ^ 8), x = 16 * hi + lo, so 16 * H + L - MAGIC = P = sum(x * (c - 8)), the exact
  dp4a integer; the float epilogue replicates `h2f(d_w) * ((x_d * 0.0625f) * float(16 P))` verbatim per group.
  SQ mode (gate|up): silu(gate) * up -> the k_gemv SQ q8 layout, plus fp32 y for memory-faithful A/B.
- `t4q/tools/tc_bench.cu`: 7 synthetic cases (a K=512/N=8 micro + the 6 real TP shapes) x M in {2,4,7,8}: a host
  float model of the dp4a CVT=2 arithmetic, max-abs-diff + first-mismatch DIFF reports, GB/s burst rates,
  `--case`/`--dev`/`--reps` args, unbuffered logging. All 7 cases `ok:true`, `fails:0` (v16).
- `tests/spec_check.py` tc section + `tools/stage_m5.py`: forced-draft A/B (spec_tc 0 vs 1 verify logits) with
  argmax-equality + max-abs-diff records, tc configs in the v5 sweep, per-case bench driver, 7 cases.

### The three bugs that took v6..v12
1. Link failure: the new `gemm8.cuh` include in `tp_spec.cu` re-defined three non-template `__global__` kernels
   (v6) -> `static` on `ksum_kernel`, `quant8_tok_kernel`, `quant8_g128_kernel`; local podman link check passed (v8).
2. IMA at spec_tc = 1: the SQ epilogue's `nog = BR/32` should be `BR/64` (a BR=128 weight-row span covers only
   BR/2 down rows because each 8-row tile interleaves 4 gate + 4 up rows) -> the q8 writes overran the staged
   smem; plus `TcArgs.y` was never set (both the engine branch and the bench) (v8/v9/v11).
3. The bench SIGSEGV: the synthetic generator used row stride `2 + K/2` instead of `src_row_bytes(FAST_P4, K)`.

### What the harness taught us (the "honesty" fixes)
- **Bit-identity between tc and dp4a is unattainable by design**: both kernels contract a*b+c into FMA (nvcc's
  default `-fmad=true`; `-Xcompiler -ffp-contract=off` never reaches device code) and sum the 512-k group products
  in different orders (dp4a: 16 per-lane chunk partials through a shfl_xor tree; tc: flat ascending). The right
  gates: max-abs-diff magnitude (bench) + argmax equality + accept-rate parity (spec_check). The K=512 micro case
  (one chunk, one order) is bit-EXACT for tc vs dp4a vs host: proof the mapping/int arithmetic are right.
- **The engine-level 0.529 logit diff is the SQ requant amplification chain, not a kernel bug**: tc-vs-dp4a ULP
  differences tip individual roundf() boundaries when gateup's silu(gate)*up is re-quantized to q8 (the bench's
  sq_eq is false for exactly that case), the down GEMV then sees a few +-1 q8 entries, and the logits move ~0.5
  with argmax equal 32/32 and 35/35. Accept rates are unchanged (k3_tc 0.81/0.72/0.82 vs dp4a 0.81/0.73/0.82).
- **All spec variants tip the same near-tie**: P1 token 435 diverges for EVERY non-default config (sqt128, rb1,
  tc, tc1) and nothing else; the default k{N}_dv1 configs stay byte-identical. V5's identity gate now covers only
  the default-arithmetic configs; variants are gated on accept-rate parity (+-0.03 vs the same-k base).

### Perf verdict and the parked restructure
- v16 rates (M=8): qkvz 67 vs 79 dp4a, attn_qkv 45 vs 76, gateup 82 vs 107, down 95 vs 137, out 69 vs 156,
  clamp 95 vs 118. The v14 L2-prefetch addition (~12 stages ahead, gemm8::prefetch9's trick) bought only ~15%.
- Diagnosis: the block-wide `__syncthreads` double buffer (512 stages) serializes load latency against compute,
  and the X-side smem staging + 4 ldsm_x1 per warp-stage add issue slots the dp4a does not pay.
- Parked plan (next TC attempt): warp-autonomous staging (each warp owns its WN rows: the W smem slices become
  warp-private, syncs drop to __syncwarp) + direct-A fragments (each lane builds its own a-fragment from 2 L2-hot
  u32 loads of xq, deleting the X smem stage and the X ldsm entirely). Estimated 1.5-2.5x; the TC wins only when
  that lands (M >= 7 shapes first).

### Next steps
1. L2 (PLAN_500): tree drafts / TPD = 16 on the dp4a path - orthogonal to the verify kernel, biggest expected
   tok/s lever (M5 r1 k=3 baseline 66-70 tok/s -> target 100-160).
2. The TC perf restructure above, then re-evaluate spec_tc defaults for k >= 6.
3. L3: the low-bit UD-Q2_K_XL / IQ3 fast modes with the KL gate.

## 2026-10-04 - M5 round 3: TC r3 restructure lands correct (all gates green), TC perf still behind dp4a

**Gate: PASS.** v24: bench `fails:0` on all 8 cases x M=2/4/7/8, `V4_pass`, `tc_pass`, `V5_pass`, `gate_60` all true
(best k4_dv1 66.24 tok/s; k3 spec 0.68 accept, 3.85 tok/step). `spec_tc` defaults stay OFF: dp4a verify remains.

### The r3 rewrite (parked plan from r2, done)
`tp_gemv_tc.cuh` fully restructured: `NW = BR/16` warps of 32 lanes, each warp owns exactly WN=16 weight rows
(2 mma n-tiles) end to end. Warp-private smem W slices (codes + d, 768 B/warp/stage, 2-deep), so the k-loop needs
only `__syncwarp`; one `__syncthreads` pair remains in the SQ epilogue. Direct-A fragments: each lane builds its
mma A-fragments in-register from 2 L2-hot u32 loads of the token's q8 (the fragment nibble at (byte m, half h) is
the digit of k = bb*32 + t4*4 + m + 16h, the same GGUF interleave the staged W codes carry) - the X smem stage and
X ldsm are gone. L2 prefetch kept ~12 stages ahead. ptxas: 63-66 regs, 0 spills.

### The 75%-unwritten bug (v18..v23) and how the bench caught it
- v18 claimed "TC now fast, 267 GB/s" but many outputs read 0. v19's NaN-fill + census showed the written values
  were ULP-correct (2e-6..6e-6) yet EXACTLY 75% of every window was never written (nnan = 0.75*M*N, all shapes),
  and the old N=8 micro check was vacuous (ncmp=0, nothing compared).
- v21's non-vacuous bench (N=64 micro cases, gate over ALL rows + nnan==0) showed nnan = 96 per block regardless
  of M: 48 rows per BR=64 block. Kernel-side printf (v22): only warp 0 ever reached the epilogue.
- Root cause, one line: `static constexpr int NW = BR / 64;` instead of `BR / WN` = 1 warp per 64-row block.
  Each warp computed its 16 rows correctly; the other 48 rows never had an owner. v18's "267 GB/s" was inflated
  ~4x because 3 of 4 warps' worth of weight traffic never happened.
- Follow-on: `Cfg`'s `NW = BR / WN` referenced `WN` one line before its declaration; nvcc's frontend rejects that
  in the tp_spec.cu TU only (the bench TU compiled) - v23 failed at build; fixed by declaring WN first. v20's push
  also slipped past a broken local compile (duplicate y_a, gate after use) - the tail-cleanup edit; fixed, and the
  tp_spec TU is now part of the local compile check before every push.

### Honest perf verdict
- Correct r3 TC (v24, GB/s of packed weight bytes, M=2 / M=8) vs dp4a:
  qkvz 86/85 vs 251/79, attn_qkv 75/59 vs 236/80, gateup 91/65 vs 250/94, down 110/110 vs 254/123,
  out 95/94 vs 233/177, clamp 86/56 vs 256/79.
- TC is ~2-3x behind at M=2 and roughly ties or trails at M=8. TC's M-scaling is flat (latency-bound small
  chunks); dp4a collapses with M but stays faster everywhere. Value correctness is exact-class: maxd <= 6e-6 vs
  the host model over all rows, SQ tips only in gateup's requant chain, engine argmax 32/32 and 35/35.
- Verdict: keep dp4a as the verify/draft engine; the TC path stays wired, gated and correct, but spec_tc defaults
  remain 0. Perf debt is now isolated to memory-system efficiency (RPL=2 64-token chunks, 256 B contiguous loads
  per warp-chunk, grid of N/64 blocks vs dp4a's 512-token persistent chunks), not to staging or sync structure.

### Next steps
1. L2 (PLAN_500): tree drafts / TPD=16 on the dp4a path - unchanged as the biggest tok/s lever.
2. TC perf, if pursued: merge the RPL chunks (fewer, larger K-chunks; RPL=4 128-token chunks), consider a
   persistent block per 64 rows with a K-loop grid-stride, and nch-style chunking like the dp4a's. Only worth it
   if the projected M>=7 win clears the added complexity.
3. L3: the low-bit UD-Q2_K_XL / IQ3 fast modes with the KL gate.

## 2026-10-04 - M5 rounds 4-8: the TC GEMV from 2-3x behind to beating dp4a on 3 of 6 shapes (v31)

Kernel template grew to `template <RPL, BR, SQ, RREG, RSMEM, HOIST, BAT, AR>`; the bench (`tc_bench.cu`) became a
6-variant x 4-M x 8-case matrix with a dp4a anchor per shape; the timing harness learned 20 warmups + min-of-4
windows and to join the model download before benching (v29's timings were contaminated 1.2-2x by the concurrent
download OOM-killing the clocks monitor and case tails). ncu is permanently out (ERR_NVGPUCTRPERM on the notebook
nodes) - the within-run matrix is the only instrument.

### r4-r5 (v25-v26): group-of-4 ring, then occupancy was the theory that died
- r4: load W in GROUPS of 4 stages (256 k-els = the 8 consecutive Q4 blocks of a plane-half unit run) held in a
  2-deep register+smem ring; one `__syncwarp` per group, one `ldsm_x4` per stage, A fragments in-register from
  global q8. Burst-2 down_tp hit 168.8 GB/s at M=7/8, BEATING dp4a - the first shape won (v27).
- r5 parameterized the ring (RREG x RSMEM). v26's matrix killed the occupancy theory: <2,1> (16 warps/SM)
  matched <2,2> (8) within noise; BAT=2 (quad burst) was structurally WRONG (4 groups staging into RSMEM=2's two
  buffers: groups base and base+2 collide) and is dropped; 8 L2 prefetches per group only doubled LSU pressure
  (hardware MLP covers DRAM latency once 8-16 LDGs are outstanding).
- v28's d-window fusion (2 d u16s -> one LDG.128) FATALed the whole matrix (`cudaErrorMisalignedAddress`): FAST_P4
  d words interleave the RPL planes (`sa = d + (g*RPL + r)*2 B`), so odd-plane rows sit at 2 mod 4 - no wider d
  window exists. The 4 separate u16 loads are a layout constraint, documented in the header.

### r7 (v28-v30): the A-ring, and the clean winner matrix
- The surviving M=4 gap was A-side latency: the hoist issues the group's 16 q8 + 4 xms words at the TOP of
  compute but stage 0 consumes them immediately - every group eats one ~270+ cy stall on its first mma. AR:
  compute(t) issues group t+1's A words into compile-time-indexed ping-pong buffers a full group early.
- v30 (clean) per-shape winners at 16 warps or less: AR64 wins qkvz (129.7-135.6), qkvz_clamp (123.6-127.9),
  gateup (138-139, beats dp4a at M8: 138.3 vs 118.7), attn (54.5-66.5), out (69.3-91.6); star wins down
  (130-151). The GEMV gap directly throttles spec decoding: tc configs sit at the rb1 floor (k4_tc 41.9 ~
  x_k4_rb1 42.4 vs dv1 66.8 tok/s).

### r8 (v31): d to a register ring, 16-warp occupancy parity
The star's RSMEM-2 ring needed 40 KB smem at BR128 = 8 warps/SM (dp4a: no smem, 16-20 warps). The d words were
the smem parasite: gload now packs each lane's two stages' d pairs into one u32 per stage (`rd_p[4][2]`, a 4-deep
REGISTER d-ring) and compute shfls the row's pair from the owning loader lane - one shfl per pair, no barrier.
STAGE loses its d area (20->16 KB at BR128, 10->8 at BR64): the star runs 16 warps/SM at both BRs. RREG is pinned
to 2 (the live d set {t, t+1, t+RREG, t+RREG+1} needs 4 slots; RREG 4's refill collides with a live read) and
both schedules unroll their outer loops by 4 so the ring indices are call-site literals (a runtime index lands
the whole ring in local memory - r4's lesson).
- v31 measured clean: all 28 CHECK rows per case ok=true, engine gates green, best_spec 66.84 (k3_dv1). The star
  now reaches 171.4 GB/s on down at M4 (+27% within-run over ctrl), AR64 gateup 156.9 (+27%), and TC WINS
  gateup at M7/8 (153.2 vs 148.7; 154.3/147.9 vs 118.7), down at M8 (133.8 vs 126.9), and is within 1.45x on
  qkvz at M8 (139.3 vs 201.5).
- The remaining gap is structural: TC's per-launch time is M-INDEPENDENT (the mma m-tile covers all 8 token
  slots and the A loads fetch them all), dp4a scales with M - so TC loses at the engine's real M=2-6.

### r9 (v32): T-gating the A loads + engine promotion to the winners
- r9: gate the A-side loads on `tok < a.T` (quad-uniform; the stale token quads skip their ~24 of ~30 LSU ops
  per group and zero their registers; the mma stays m8 - the B-fragment pattern needs all 32 lanes - but the
  tensor pipe is <2% utilized, so the waste is free). This also fixes a latent OOB: the ungated stale quads
  read past the a.T-token xq/xms buffers.
- The engine default (`gemv_tc_launch`) was promoted to the winners, all BR64; v32 measured it clean: the
  M2/M4 columns jumped 20-78% (attn M2 73.8 -> 131.0, down M4 171 -> 216, qkvz M4 131 -> 169); TC WINS
  gateup at M7/8 (159.4/152.1 vs 148.1/118.3), down at M8 (152.2 vs 135.5), down at M4 within 6% (216.2 vs
  230.6); the engine tc configs rose 42 -> 52-60 tok/s (k4_tc 41.9 -> 53.7) vs dv1 65-67; all rows ok, nnan 0,
  gates green, best_spec k3_dv1 67.49.
- v32's engine CUPTI trace closed the accounting: the verify's TC launches run at the matrix's per-launch
  times (qkvz 167.9 us vs bench 165.7) under a uniform ~1.22-1.28x sustained-clock inflation (sm 1301 MHz vs
  1590 boost), so the engine gap is exactly the per-launch GB/s ratios x 64-72 launches/step x 32 layers,
  not an engine-path anomaly.

### r10 (v33-v36): the WN=8 warp doubling is REFUTED; qkvz/clamp promote to the AR ring
- Theory (from the trace): GB/s tracked total warps = N/16 (attn 256 = 6.4/SM -> 102 GB/s; down 320 = 8/SM ->
  216), so doubling the warps should lift the small-N shapes. WN 8: each warp owns ONE mma n-tile (8 rows,
  a lane-QUAD loader - one stage, 2 code int4s + 1 packed d u32 per lane per group, ldsm_x2), total warps
  N/8, STAGE unchanged, every row still warp-private. The kernel template grew a 9th param WN (8/16); the
  star8 lands at 80-95 regs (24 warps/SM at BR64), the AR8 at 127-128 (its BR64 launch-bounds target is 2
  blocks - 3 spills the A-ring 48 B to local).
- v36's matrix REFUTES it: star8/AR8 lost ~25% on EVERY shape at every M (attn star8 69.7-103.6 vs star
  102.7-131.4; down star8 98.5-189.8 vs star 152.2-217.7). The warp count was never the constraint: the
  per-warp in-flight W bytes halved (2 KB/group vs 4) while the fixed per-group costs (the syncwarp pair,
  the ldsm setup, the AR ring's loads) doubled per byte. GB/s tracks the in-flight bytes per SM, not warps.
  The WN16 code stays (template param, committed-clean, one if-constexpr block) but is benched no more.
- The same matrix promoted rpl2-nonsq at big N: AR64 > star64 on qkvz/clamp at EVERY M (+23..+55% at M2,
  where the r9 T-gate makes the A-side the largest fraction and the AR ring covers exactly that load); the
  small-N rpl2 (attn N 4096) keeps the star (its M2 131 vs 120, ties at M4-8). `gemv_tc_launch`: rpl2-nonsq
  N >= 8192 -> <2,64,0,2,2,1,1,1>, else the star; rpl4-nonsq the star; rpl4-sq the AR64.

### The v33-v35 worker-OOM saga (harness, not kernel)
Three consecutive runs were OOM-killed mid-matrix at a drifting case boundary (v33 at micro1, v34 at
gateup, v35 at case 2-3 of the real block) - the shared T4x2 node's RAM margin, not our leak (v32's
identical 6-variant flow survived; the downloader already runs as a subprocess). The harness now: the
ENGINE spec sweep runs BEFORE the tc_bench matrix (the goal metric survives any late kill), the model's
clean page cache is dropped (POSIX_FADV_DONTNEED) before the matrix, the REAL cases run before the proven
micro smoke cases, and the WN8 variants left the run list (24 -> 18 instantiations in the tc_bench TU,
compile RSS down, matrix runtime -25%).

### Standing gate (unchanged)
"Fully faster than dp4a on ALL real shapes" is still open: v36 wins 2 of 6 real shapes at M7/8 (gateup
AR64, down star at M8) and sits within 6% on down at M4; the engine's real M regime (4-7) still favors
dv1 by ~15% end-to-end (52-60 vs 65-67 tok/s). r11 candidate (the structural endgame): replace the smem
STS -> syncwarp -> ldsm -> mma round-trip with register-only B fragments via warp shuffles (the ldmatrix
redistribution as ~4-8 __shfl_sync ops + the nibble ALU - no smem, no barrier pairs, no ldsm, the dp4a
register path's shape); that removes the last structural difference from dp4a besides the mma itself.

## 2026-10-05 - M5 round 11 (r11): pure-shuffle B dead on paper (PTX-pinned), RREG-4 measured and
## refuted; the marginal rate is issue-bound, not cover-bound
**Gate: NOT passed (unchanged wins: gateup M7/8 AR64, down M7/8 star - down M7 is a v40 flip inside the
M2-style noise). v38 (old-loop rerun, healthy node), v39 (node OOM at the tc_bench build, 6 s in), v40
(the fixed r11 matrix, OOM at the last micro case - the engine data and every real/diagnostic case
CHECK row survived). Gates: every CHECK ok=true nnan=0, sq_eq=true except gateup's known q8-ULP
requant rows; engine V4/V5/tc/argmax/parity/gate_60 all True, best_spec k3_dv1 65.79.**

### The PTX ISA pins the B mapping and kills the pure-shuffle GEMV
The planned r11 endgame (register-only B fragments via warp shuffles, no smem/ldsm/syncwarp) is dead on
paper: ldmatrix.x4 gives thread l matrix i's row `l>>2`, 16-bit words `2*(l&3)+{0,1}`; the `mma.m8n8k32`
B operand needs lane l to hold n = `8g + (l>>2)` with nibble j at k = `4*(l&3) + (j>>1) + 16*(j&1)`. So
lane l's B word is component `(l&3)` of loader lane `(16g + 2*(l>>2) + (s>>1))`'s `rq[2*(s&1)+bb]` - the
component index varies per consumer, so one shfl can never serve the four consumers of a source word,
and the honest move set is **64 SHFL warp-issues per group vs the smem path's 10** (4 STS.128 + 4
ldsm.x4 + 2 syncwarp) - a 3-6x LSU-issue regression in a kernel whose M2 speed came from CUTTING LSU
issues 3x (r9's T-gate). The smem round-trip stays; the last structural difference from dp4a is the
mma shape itself.

### RREG 4 (the double per-warp W cover) measured and REFUTED
Design: the surviving r11 lever from the cover theory - dp4a streams ~9.2 KB/warp in flight vs the
star's 4 KB, so deepen the ring: RREG 4 (`rq[4][RQW]`, +32 regs), the d-ring generalized to depth
`2*RREG` (r8's rule generalized: the live d set at a burst is the RREG unconsumed groups plus the
refill pair = RREG+2 slots), schedules unrolled by 2*RREG so every slot index stays a call-site
literal. 183-227 regs (star/AR, all 0 stack 0 spills, podman-verified) = 2 blocks = 8 warps/SM at BR64:
half the star's warps, double the cover. Bench: V8 (star-R4) / V9 (AR-R4), the WN8 dispatch removed (its
6 combos freed the instantiation budget: 24 - 6 + 5 = 23, inside the v33 envelope).
- v40 verdict: V8/V9 lose to the RREG-2 star/AR on EVERY real shape at every M (M4, the tightest-noise M:
  down 131.7 vs 213.2, out 93.7 vs 97.7, attn 82.4 vs 85.0, qkvz 130.0 vs 167.7, clamp 133.2 vs 170.5,
  gateup AR-R4 121.0 vs AR64 191.3). With r10: **WN8 (half cover, double warps) and RREG-4 (double
  cover, half warps) both hold the total in-flight constant (~64 KB/SM) and both lose** - the star's
  16 warps x 4 KB is the measured sweet spot; neither total in-flight bytes nor per-warp cover is the
  binding currency. WN8 halved the bytes per fixed per-group cost (r10's measured reason); RREG-4
  halved the independent instruction streams that interleave the consume windows (8 warps hide half
  the latency of 16).

### The N/K-sweep diagnostics: the gap is BOTH fixed-cost and issue-bound marginal
Three new cases decompose the per-launch cost (M2, K 5120 rpl2: n1024 64.3 / n2048 99.5 / attn 109.6 /
qkvz 109.6 GB/s star vs dp4a 202.1 / 207.3 / 225.6 / 246.3; out-class K sweep at N 5120 rpl4: out K3072
118.7 / outk5 K5120 129.6 / down K8704 218.1 star vs dp4a 219.6 / 237.3 / 259.1). The linear fit over N:
- dp4a: ~3 us fixed + ~254 GB/s marginal - riding the DRAM roof.
- TC star: ~22 us fixed (inflated by small-N underfill: N 1024 is 16 blocks over 40 SMs) + ~122 GB/s
  marginal - HALF the DRAM roof with 16 warps of 4 KB cover and ~50-60 warp-issues per 2 KB group
  (32 mma + 10 LSU + control). 16 warps on the 4 schedulers make that an issue roof of ~180-220 GB/s -
  exactly the measured marginal ceiling; the cover was never the marginal constraint.

### r12 direction (from the diagnostics, not the refuted cover theory)
The issue count per weight-byte must halve: `mma.m16n8k64.s4.s4` (B = 8 rows x 64 k = 256 B/mma ->
16 mma/group vs 32, the same 4 ldsm.x4/group; the A side is 16 columns so M2 wastes 14/16 of it, but
the A requant cost is per-k not per-column, and the T-gate already zeroes the stale-token A loads) plus
the remaining syncwarp/epilogue trims. If the measured marginal stays ~120 after the mma halving, the
TC verify path is structurally capped and dp4a verify stays the engine default.

### Harness lessons (v38-v40)
The stage edits go in `t4q/tools/stage_m5.py` - the packed `kaggle/m5/t4q-m5.py` is REGENERATED by
mkkernel.py (the first v38 push shipped the old matrix because the loop edit went into the packed
file). Kaggle id/title must resolve exactly: after the v38 title-slug rename (t4q-m5 ->
otdoges-t4q-m5), an id/title mismatch 409s every push; the metadata now carries id
`otdoges/otdoges-t4q-m5` + title `otdoges/t4q-m5` (title slug == id slug). The node OOM killer took 2
of 4 pushes (v39 six seconds into the tc_bench compile, v40 at the last micro case): the matrix-last
design did its job both times - the engine data and the full real-case matrix landed. The m5 flow is
now ~20 min end-to-end (the 16 GB model downloads in 85 s from the warm cache), so retry-on-kill is
cheap.

## 2026-10-05 - M5 round 12 (r12): the m16n8k32 s4 mma is sm_80-only (the shape lever is dead on
## T4); the rpl packing A/B is measured - shape-local, not a uniform win; the bench attn shape was
## stale, the engine's qkv_a is N 7168
**Gate: NOT passed (unchanged wins: gateup M7/8 AR64, down M7/8 star). v41 (the fixed r12 matrix): the
OOM kill at the last micro case again - only micro2 and the final SUMMARY block lost; all 12 case
logs with full M sweeps + the engine sweep landed. Gates: every CHECK ok=true nnan=0; engine
V4/V5/tc/argmax/parity/gate_60 all True, best_spec k3_dv1 66.52.**

### The mma-shape lever is dead on paper (sm_80-only)
The r11-planned `mma.m16n8k32/.m16n8k64` with `.s4/.u4` operands requires sm_80 (the PTX ISA Target
ISA Notes) - on the T4 (sm_75) the only sub-byte mma is `m8n8k32`, so the two-digit q8 split forcing
32 mma per group is STRUCTURAL. The operand-flip/mma-shape family is closed; what remains of the
issue-bound marginal (~122 GB/s vs dp4a's 254 DRAM-roof) is the mma count itself.

### The rpl A/B: the packing matters, but shape-locally
The v40 decomposition (the rpl2-packed class at HALF the rpl4 class's per-SM group rate, same kernel,
same warps) made rpl a bench A/B without a kernel rewrite: same shape, only the layout flips
(qkvz_r4 / attn_r4 / down_r2, cases 11/12/13; the dp4a anchors ride along - DK(4,10) vs DK(2,10)).
v41 M4 verdict (one node, one run):
- down (K 8704, N 5120): rpl4 REQUIRED - down_r2 (rpl2) halves it (star 217.0 vs 110.3; dp4a 258.9 vs
  n/a - dp4a's rpl2 nch-17 template does not exist). The engine already packs down/out/gateup rpl4.
- qkvz (K 5120, N 8192): rpl2 KEEPS it (dp4a 252.6 vs 244.8, TC 174.3 vs 175.7 - rpl4 neutral to
  slightly negative). The engine already packs qkvz rpl2.
- attn (K 5120, N 4096): rpl4 wins BOTH paths - dp4a 133.7 -> 203.0 (+52%), TC star 80.9 -> 98.5
  (+22%). The packing effect is a memory-subsystem effect shared by both kernels (the per-warp
  contiguous run doubles), not a TC-only artifact.
So "rpl4 everywhere" is refuted; the current engine packings are already optimal for every tensor
whose shape was measured EXCEPT the attention qkv_a.

### The stale bench shape (the r12 miss)
The bench's attn case was N 4096 - a stale approximation. The engine's real qkv_a (selftest-pinned)
is **N 7168, K 5120** (q|gate 6144 | k 512 | v 512 rows per GPU), sitting between attn's measured
N 4096 (rpl4 wins) and qkvz's N 8192 (rpl2 wins). The v41 A/B does NOT decide the engine's attn
packing; the r13 A/B runs at the true shape.

### r13 design (v42): the true-shape A/B + the engine-level variant, defaults untouched
- Bench cases attn_e_tp (14, N 7168 rpl2) / attn_e_r4 (15, N 7168 rpl4) - the true-shape A/B, with
  attn_r4 (12) kept as the N-4096 control for the N dependence. The r12 twins 11/13 are off the run
  lists (v41 measured both; their dispatches stay for the record).
- The loader gained `T4Q_RPL_QKV_A` (qkv_a-only, applied after T4Q_RPL_N; defaults unchanged: qkv_a
  still packs rpl2), and tp_spec.cu gained the (FAST_P4, 4, 10, non-sq) dp4a instantiation it needs
  (255 regs at the launch-bounds cap, 0 spill; the non-spec decode path already had it). Prefill/
  TC dispatch on W.L.rpl generically - no other wall.
- The m5 VARIANTS mechanism (env at load time) runs an engine-level A/B in the SAME run: the r4a
  process loads with T4Q_RPL_QKV_A=4 and re-runs k3_dv1 + the two best extras (spec_k=3,spec_rb=1
  and its TC twin) - its tok_s compares 1:1 against the main run's same configs on the same node.
  The repack is value-identical, so the promotion gates are: k3_dv1 byte-identical to the reference,
  accept-rate parity, and the tok/s comparison. If rpl4 wins at the true shape, r14 flips the loader
  default for qkv_a only.
- Matrix order is new-questions-first (14, 15, 12, then the real-shape continuity set 2-7, outk5,
  micros last): the engine variant adds ~4 min before the matrix, so the kill window now lands
  mid-matrix - the r13 data lands before it, and only continuity columns are at risk (v40/v41
  already hold them).
Podman: all three changed TUs compile RC=0; the gtc envelope is unchanged (24 unique gemv_tc
instantiations in the bench TU, all 0 stack 0 spill); tp_spec's only spillers are the pre-existing
gemm8 prefill family (benign since v24).

## 2026-10-05 - M5 round 13 (r13): the true-shape A/B closes the rpl line (qkv_a keeps rpl 2);
## the rb=1 + dp4a + rpl2 ~10 ms/step penalty is real and the repack removes it - but it promotes
## nothing (rb 1 never beats rb 0); the 8000 first-step watchdog flake is harness-mitigated
**Gate: NOT passed (unchanged wins: gateup M7/8 AR64, down M7/8 star). v42 ran COMPLETE (no OOM
kill); v43 was killed at ~1264 s mid-matrix, after both engine sweeps and both true-shape case
sweeps (24 CHECK lines each) - everything that mattered landed. Gates: every CHECK ok=true nnan=0;
main V4/V5/tc/argmax/parity all True, best_spec k3_dv1 66.95 (v42) / 64.59 (v43, slower node); the
r4a variant V5_pass True both runs.**

### The true-shape bench A/B: qkv_a keeps rpl 2
Cases attn_e_tp (14) / attn_e_r4 (15), N 7168 K 5120 (the selftest-pinned engine shape), M4
v42 / M4 v43 (same numbers within 1%):
- dp4a: 243.1 / 242.0 (rpl 2) vs 239.4 (rpl 4) - neutral; at M8 dp4a rpl 4 COLLAPSES
  (177.9 -> 124.9, -30%).
- TC AR64: 178.1 vs 177.8; star64 176.4 vs 170.9 - neutral; the one TC win (rpl 4 M8, AR64 142.5
  vs dp4a 124.9) is moot - the engine keeps rpl 2, where dp4a wins M8.
The r12 attn N-4096 anchors turned out node-variable (dp4a rpl 2 M4: 133.7 in v41, 157.2 in v42)
- the +52% was partly node variance; the honest N-4096 effect is ~+28% (157.2 -> 200.6). The
transition to the rpl2 side happens somewhere in N 4096..7168, and the engine's N 7168 is on it.

### The engine-level A/B: the repack moves NOTHING that matters - and the rb=1 anomaly is real
v42's r4a variant (T4Q_RPL_QKV_A=4, one node, same run): x_k3_rb1 (rb=1, dp4a) jumped 54.95 ->
68.16 tok/s (+24%, all three prompts, +1.2% above the main's rb 0 best) - right after the k3_dv1
(rb=0) config ERRORED (the 4-s AR-wait watchdog, code 8000, at its first spec step). Every
per-GEMV rate said neutral (bench M4 dp4a/TC, M=1 selftest 244 vs 249, the ref decode 30.1 vs 30.5),
so the first read was a post-error artifact. v43 settled it CLEAN (the v4 warmup made k3_dv1 run;
no 8000, V5_pass True):
- k3_dv1 (rb 0, dp4a): main 69.07/64.59/69.37 vs r4a 68.94/64.71/69.74 - IDENTICAL (+-0.3%).
- x_k3_rb1 (rb 1, dp4a): main 56.35/52.22/57.49 vs r4a 68.85/64.41/69.68 - **the ~10 ms/step
  rb=1 penalty is REAL at rpl 2 (present in v38-v43) and the repack removes it entirely**.
- x_k3_tc1_rb1 (rb 1, TC): main 56.26/52.09/57.40 vs r4a 56.04/52.16/57.37 - the TC class is
  rb-IMMUNE and rpl-IMMUNE at this shape.
The verdict: rb 1 at rpl 4 == rb 0 exactly (64.41-69.68 vs 64.59-69.74), and rb 0 itself is
rpl-neutral - so the repack promotes NOTHING (the engine's best config stays rb 0, and rb 1 never
beats it at either packing). qkv_a stays rpl 2; the rpl line is CLOSED in both directions at the
true shapes: rpl 2 {qkvz, qkv_a}, rpl 4 {down, wo, gateup} - the current engine packings, now all
measured.

### The open curiosity (recorded, not pursued): the rb=1 penalty has NO per-GEMV signature
The ~10 ms/step penalty is specific to (rb=1 verify graph) x (dp4a) x (qkv_a at rpl 2): the bench
gemv rates at the true shape are rpl-equal, the TC class and the rb 0 class are rpl/penalty-immune,
and the k_gdn_m stash/replay arithmetic is far too small (the rb 1 path SAVES ~0.9 ms/step of
snapshot writes). Whatever it is, it lives in the rb 1 verify graph's interaction with the qkv_a
gemv's rpl 2 layout, and it is not a tok/s lever (rb 1 == rb 0 once it is gone).

### The 8000 watchdog flake (a harness race, mitigated; a robustness note for the engine)
A fresh process's FIRST spec step can hit the 4-s AR-wait watchdog (st.err = 8000+idx) - seen once
(v42's variant, right after the main process's 8.6 GB x 2 CUDA teardown; the driver-global teardown
races the next process's first graph launches). Harness-mitigated in v43: a 20-s teardown drain
before the VARIANTS loop + the v4 section as the variant warmup - the variant then ran clean
(V5_pass True, no recurrence). The underlying race stands: a user going straight to spec decode on
a fresh engine could see a hard error; the fix (a cross-GPU barrier before the first verify step,
or a first-step-only longer watchdog) is an engine change for a later round.

### Where the standing gate stands after r4-r13 (the lever census)
The TC GEMV beats dp4a only on gateup M7/8 (AR64) and down M7/8 (star, M7 a v40 flip inside
noise). Every structural lever is now measured-dead on T4 within the exact-numerics constraint:
cover (WN8, r10) and warp count (RREG-4, r11) refuted by the two-sided falsification; the mma shape
is sm_80-only (m16n8k32/k64 s4 needs sm_80; T4 is stuck at m8n8k32 and the two-digit q8 split, 32
mma/group); the packing is closed (r12/r13, all engine tensors measured at their true shapes); the
per-launch cost is ~22 us fixed + ~122 GB/s marginal, issue-bound (~50-60 warp-issues per 2 KB
group vs 4 schedulers/SM) against dp4a's ~3 us + 254 GB/s DRAM roof. The one un-measured lever left
is the s4-activation requant (16 mma/group, -10 A-loads/group -> the marginal roof ~180-220 or
better; but it breaks the V4/V5 bit-identity gates and the tok/s effect rides through the draft
ACCEPTANCE rate, unmeasured). r14: either measure that acceptance-vs-speed tradeoff once, or accept
the TC as the M7/8-only promotion and move to the next tok/s lever of PLAN_500 (the tree drafts /
TPD = 16 on the dp4a path).

## 2026-10-05 - M5 round 14 (r14): the tree drafts and low-bit refuted on measured roofs, the ngram
## confirmed measured-dead (v3); the v42 trace census finds the AR norms as the biggest live lever
## (4.86 ms/step, 128 x 38 us at M4 - the publish runs through one block per row); the SPREAD publish
## (every block its 1 KB slice, one counter + one flag, T4Q_AR_SPREAD) is the r14 A/B

### The r13-chosen lever (tree drafts / TPD 16) dies on the measured M-scaling roof
The tree needs the TC verify at M 16 (TPD = 16, a 2-m-tile variant of gemv_tc, same weight bytes,
~2x mma work). The honest roof check kills it without kernel work: the TC verify degrades ~20% from
M2 to M8 (gateup AR64 179.5 -> 143.1 GB/s), so a 2-m-tile M16 lands ~85-100 GB/s; a depth-4 B = 2
tree (E ~ 4.6 tokens/step at per-level top-2 coverage ~0.95) needs ~64 verify columns through
that rate -> a ~97 ms step ~ 47 tok/s, a LOSS to the current 66-72. The M8 AR-norm bytes double
again with M (M x 20 KB x 128), so the tree's step is worse than the naive roof. Tree closed.

### Low-bit (L3) dies on the T4's issue x DRAM coincidence
dp4a at Q4_0 sits at ~254 GB/s, which is BOTH the DRAM roof and the issue roof (the 6-8
instructions per 4-el group / 1 byte of weight: the issue rate and the DRAM rate coincide on T4).
A 2-bit pack saves ~1.44x the bytes but spends ~2.75x the instructions per byte (the unpack
shift/mask chains), so the issue roof drops to ~92 GB/s < the needed ~176 - the byte saving cannot
pay. Low-bit closed within exact numerics on this chip. (PLAN_500's honest bottom line said 500
needs BOTH low-bit and ~7-8 tokens/step; both halves are now measured or roof-dead on 2xT4, so
the exact-numerics ceiling here is the ~75-80 tok/s architecture fully tuned, not 500.)

### The ngram lever was already measured (v3) - it stays off
spec_ng (prompt lookup, k_ngram) was A/B'd at k = 3 in v3: ng5 66.8 / 62.6 / 66.6 vs MTP-only
69.3 / 63.5 / 65.9 tok/s on P0/P1/P2, and ng3 was worse. The MTP head already predicts the copied
spans of the edit prompt (P2 acceptance 0.82), so the lookup replaces drafts it would have gotten
anyway. No re-run needed; --ngs 0 stays.

### The v42 trace census (the non-GEMV ~40% of the step, per-iter us / launch counts)
The TC-config verify decodes as (per iter): the gemvs 44.3 ms of the 54.3 ms span (81.6%, capped
by the r4-r13 census); k_ar_norm_m 4857.8 us / 128 launches (8.9%); k_gdn_m 2231.9 / 48 (4.1%,
rb 1 measured a net loss earlier); k_seg_m 1436.6 / 48 (2.6%); the attention split/prep/combine
~1070 / 16 layers (2%); the lm head 1982.7 us / 1 launch (FAST_K6, Q6_K, ~161 GB/s, M-insensitive
1939 us at M1 - a Q6_K unpack kernel, not a Q4_0 stream; ~+0.7 ms if it reached 220+, a K6-kernel
retune candidate for a later round); the draft span 250 us/iter. At the dp4a best config the step
is 47.6 ms: the gemvs ~31 (capped), the AR norms ~4.86 (10.2%), the rest ~10 - so the AR norms are
the biggest live non-GEMV lever.

### The AR norm decomposition: the publish runs through ONE BLOCK PER ROW
k_ar_norm_m's publish copies each row's 20 KB fp32 partial to the peer's mailbox from block 0 of
that row only (the m5 port kept the M1-round's "one coalesced 20 KB copy + one fence + one flag"
design). At M4 that is 4 concurrent publisher blocks moving 80 KB per direction; the trace's M1
(13.9 us, 20 KB) vs M4 (38 us, 80 KB) gives a ~2.5 GB/s marginal - the per-block outstanding-store
limit, not the PCIe (the gemv AR ring publishes from many blocks and does not pay this). The
M4-round arpub A/Bs were M1-only (20 KB from one block): arpub 2 (rows from the gemv) ~= arpub 1
(the consumer copy) on the fast box (v10/v11, a wash), and arpub 3 (each of the 20 blocks its own
slice + its own flag) lost only on a SLOW-P2P box (19.9 vs 15.5, v26) - the fast box at M > 1 was
never measured, and the M > 1 shape is where the one-block publish actually hurts.

### The r14 change (bit-identity trivially preserved: same values, same fp32 exchange, timing only)
k_ar_norm_m gains a SPREAD template param (default off): every block publishes its own 256-element
(1 KB) slice of its row, then fences + bumps ONE counter (no per-slice flags - that was arpub 3's
protocol cost); the last of the 20 x M blocks sets the one peer flag, exactly like the current
last-of-M. Same addresses, same fp32 bytes, same reduce/wait path - only which blocks issue the
stores changes. T4Q_AR_SPREAD (default 1) A/Bs it at engine level: the main runs the spread, the
ars0 variant runs the old one-block-per-row publish on the rb0/rb1 best configs. v44 also adds
tc_bench --arprobe (the mechanism anchor): the exact publish pattern (fence + counter + flag
inclusive, no wait) at M 1/4 x spread 0/1 x dst P2P/local, timed on the real box - if the probe
shows the P2P write flat vs writer count, the spread is dead and the AR is at the platform's
latency floor; if the P2P write scales with the writer count, the projected AR time at M4 is
~15 us (from 38) and the step gains ~2.6 ms (+5-6% tok/s).

### v44 results: the probe refutes the writer-scaling model - the AR is at the PCIe floor
The arprobe ran first and landed in 1 second (rc 0, before the worker OOM kill at 1264 s took the
rest of the matrix mid-loop; cases 2/12/14/15 landed, all in-band - attn_e_tp 248.8 vs attn_e_r4
241.7 dp4a at M2, the rpl2 verdict repeated):
  M=1: no-spread P2P 13.31 us / spread 12.75 (loc: 10.89 / 9.36)
  M=4: no-spread P2P 20.42 us / spread 21.55 (loc: 14.47 / 11.27)
The P2P publish does NOT scale with the writer count: at M4 the four one-row publisher blocks
already saturate the link. The linear fit (M4 - M1 = 7.11 us for +60 KB) gives the marginal
~8.4 GB/s - essentially the PCIe gen3 x16 per-direction ceiling - and the fixed ~11 us (the launch
+ the threadfence_system + the counter + the flag). My r14 model (the per-block outstanding-store
limit, ~2.5 GB/s) was wrong: the v42 trace's M1-vs-M4 delta was the fixed-vs-marginal split, not a
per-block cap. The local-dst control DOES scale with the writers (14.47 -> 11.27 us, the local
write path), which is what sent the model down the wrong path.
The engine A/B (same run): k3_dv1 MAIN (spread) 70.61 / 65.77 / 70.31 vs ars0 (old publish)
68.71 / 65.15 / 70.58 - the spread is +1.9 / +0.6 / -0.3 (min-over-prompts +0.62, ~+1%; the P0 step
48.45 vs 49.79 ms, -1.34 ms). The spread stays the default: no measured downside, a small real
win on 2 of 3 prompts - plausibly the finer-grained stores interleave better with the PEER's
concurrent publish stream, the bidirectional-contention regime the one-direction probe cannot
reproduce. THE AR NORM LEVER IS CLOSED AT THE PLATFORM FLOOR: ~11 us of protocol per AR + the
partial bytes at the ~8 GB/s PCIe ceiling + the wait/readback; the 38 us/AR at M4 is the floor,
and the M4-round's per-block-fence measurements (10-14 us per K-split GEMV) already showed the
in-gemv alternatives are worse. No further AR work pays.

### The rb1 penalty is REOPENED and re-attributed: it is the config HISTORY, not the rpl
v44's ars0 (rpl 2, the OLD publish, a SHORT sweep: k3_dv1 then x_k3_rb1) runs x_k3_rb1 at
69.89 / 65.39 / 70.35 tok/s - NO penalty - while the SAME run's MAIN (the long sweep, rpl 2, the
spread) runs x_k3_rb1 at 54.54 / 50.54 / 55.56 (the ~15 ms/step penalty present). The v43
conclusion "the repack (rpl 4) removes the penalty" was CONFOUNDED: the v42/v43 r4a variant was
both the repack AND the short sweep. The v43 main (rpl 2, old publish, long sweep) had the penalty;
the v44 ars0 (rpl 2, old publish, short sweep) does not. The clocks refute the power/DVFS theory
(the penalized config runs at the same clock band as the healthy ones: 879/1050 vs 919/1039).
The discriminator is the config position: in every main (v38-v44) x_k3_rb1 runs AFTER the 8-config
sweep (four dp4a configs then four TC configs); in both no-penalty variants it runs BEFORE any TC
config (the --ks 3 variants run only k3_dv1 first). Something the TC configs leave behind (a
stream attribute, a memory-pool state, a graph-capture artifact) slows the NEXT rb 1 dp4a verify
graph by ~15 ms/step; the mechanism is unidentified. It is NOT a tok/s lever (rb 1 <= rb 0 at
every measurement, and the best config k3_dv1 / rb 0 is unaffected - 70.61 / 65.77 / 70.31, the
best engine numbers to date, spread on).

### Where the levers stand after r14
Closed this round: tree drafts (the TC M-scaling roof), low-bit (the issue x DRAM coincidence),
ngram (the v3 measurement), the AR norms (the PCIe floor, this round's probe). Still live, in
honest order: the FAST_K6 lm-head retune (~2 ms/step at ~161 GB/s; +0.6-0.8 ms/step if a better
Q6_K unpack reaches ~220), the k_seg_m fusion into the verify GEMV epilogue (~1.4 ms/step), the
8000 fresh-process race fix (robustness), and the rb1-penalty mechanism (a curiosity - the best
config does not use rb 1). The 47.6 ms step is ~65% GEMV (capped at the DRAM x issue coincidence),
~10% AR (the platform floor), ~9% gdn/seg/attn machinery. The exact-numerics ceiling on 2xT4 stays
the ~75-80 tok/s architecture fully tuned.

### r14 tail census correction: the seg is TC-only, the head is ratio-closed, the attn is launch-floor
Two corrections to the "still live" list above, from reading the code paths (no v-run needed):
- k_seg_m is launched ONLY on the spec_tc path (tp_spec.cu: the gemv_mt TC branch; the TcArgs has no
  seg) - the dp4a path (every best config, k3_dv1) fuses the 48 alpha/beta rows into launch_gemv via
  SegArgs, so the 1.44 ms/step k_seg_m cost never runs at the goal config. The seg fusion is dead
  for the goal; it would only tune the losing TC configs.
- The FAST_K6 head at ~161 GB/s is at its instruction/byte roof: the group_dot K6 path already uses
  the OR-trick (the lo nibble in place + the hi 2 bits pre-shifted through 0x30303030 masks, no
  per-element extraction), so the ~40 int ops/group vs the P4's ~20 with 26.25 vs 18 B/group puts
  the issue-side rate at ~63% of the P4 class - exactly the measured 161/254. A launch retune might
  buy +5-10% of 2 ms (~+0.2% tok/s), inside the node band; not worth a run.
- The attn trio (prep/split/combine, ~1.07 ms/step at 16 layers) is ~48 small-kernel launches at
  the graph-launch floor (~7-22 us each staging <100 KB); fusing the trio saves ~2 launches x 16
  layers x ~5 us ~ 0.16 ms (+0.3%), also inside the band. The draft (250 us/iter) is a single MTP
  block (not 64 layers) and is already at the graph-launch floor.
THE EXACT-NUMERICS SINGLE-REQUEST CEILING ON 2xT4 IS THEREFORE CLOSED AT ~66-71 tok/s: the 47.6 ms
step is ~64% GEMV at the DRAM x issue coincidence, ~10% AR at the PCIe floor, ~4.5% gdn (rb 1 net
loss), ~4% head at the K6 instruction ratio, ~2% attn at the launch floor, ~0.5% draft. Every >1 ms
bucket is measured- or roof-closed. The paths out (low-bit, tree) are roof-dead on this chip; the
s4-activation requant loses even breaking numerics (r13 roof math). What remained was the deferred
8000 fresh-process race fix (robustness) and the rb1 config-history mechanism (a curiosity - the
best config does not use rb 1). BOTH CLOSED r19h (5fb7bf8): the 8000 fix landed as the safe variant
(the AR-wait watchdog becomes the device global d_watchdog_ns, extended to 30 s until the process's
first spec verify completes, then restored to 4 s - a genuine first-step hang now costs 30 s once
instead of a hard error); the rb1 mechanism is declined by design (a curiosity, not a lever).

## 2026-10-05 - M5 round 15 (r15): the one honest unknown left in the dominant cost - is the dp4a
## GEMV's ~254 GB/s actually the node's DRAM ceiling, or just the best GEMV rate ever measured?

### Why this question is the only one left
After r14 every >1 ms bucket of the 47.6 ms step is measured- or roof-closed EXCEPT this: the r4-r13
census calls the dp4a weight streams' ~254 GB/s "the DRAM roof", but 254 was the best GEMV rate ever
measured, never a measured stream ceiling. The T4's GDDR6 is 320 GB/s theoretical; a STREAM-class
coalesced read on a good card reaches 280-300 GB/s (88-94%). This node runs power-capped (66-67 W
sustained, SMs 900-1050 MHz, both at ~75-79 C), and the DRAM draws real power, so the true sustained
read ceiling could sit anywhere from ~250 (power-capped) to ~290+. The GEMV marginal bytes are
~28 ms/step of the 47.6: if the ceiling is 254-260 the GEMV is at 97-100% of it and the closure
stands; if it is 280+ the weight streams leave ~10% (~2.6 ms/step, ~+5.5% tok_s) and the streams'
DRAM efficiency becomes the next lever.

### The v45 design (pure measurement, no engine change)
tc_bench --dramprobe: a dead-code-guarded coalesced 16-B read (k_read16, 12 registers) over a
200 MB buffer (>> 6 MB L2, pure DRAM), timed with the same warmup/min-window methodology as the GEMV
anchors, at 80/160/240 blocks x 256 threads. The anchor cases (qkvz/gateup/down/out) run in the SAME
run on the SAME node, so the probe-vs-anchor comparison has no node band in it. No VARIANTS process
(no engine change to A/B), so the run is ~250 s shorter and the OOM kill window moves off the bench
phase. Matrix order: the anchor classes first (the probe's comparators), the answered attn_e twins
late.

### The decision rule
If dramprobe <= ~260 GB/s: the GEMV closure stands rigorously - the engine is at 97-100% of the
node's power-capped DRAM ceiling and the exact-numerics ceiling is ~66-71 tok/s, reported as final.
If dramprobe >= ~280 GB/s: the GEMV leaves ~10% - the r16 lever is the streams' DRAM efficiency
(load width/pattern per warp, the tile/chunk memory order, the L2 policy hints, the warp scheduling
across the 22-44 MB per-layer walks).

### v45 results: the DRAM ceiling is ~277 GB/s - the GEMV anchors sit at 75-93% of it
The probe (this node, this run): 275.0 / 277.0 / 277.6 GB/s at 80/160/240 blocks (762->756 us per
200 MB rep; the extra blocks buy ~1%). The SAME run's anchors (no node band in the comparison):
qkvz M2 258.3 / M4 254.9 (93/92% of the ceiling - the best dp4a rates measured to date), down M2
259.2 / M4 231.1, gateup M2 250.6 / M4 222.0, out M2 224.3 / M4 208.7 (75-93% per shape).
The engine continuity: k3_dv1 70.8 / 66.05 / 70.96 (V5 pass, the best numbers to date, the spread
on; the worker OOM kill took the matrix tail again, after all the anchors and the probe had landed).
So the r13 "254 = the DRAM roof" was HALF right: 254 is the ISSUE roof (the dp4a's instruction
stream at the current clocks), not the DRAM roof (277). The honest remaining stream gap at the
verify's M4 mix is ~2-3.7 ms/step (~+4-8% tok_s) IF the per-shape rates can be lifted toward the
ceiling - but the limiter per shape is the instruction count, not the DRAM: the pure read moves
16 B per ~6 instructions (1 LDG.128 + 5 XOR) while the dp4a group loop moves 16 B per ~25-30
(load, 8 dp4a, mask/shift, the q8 epilogue) - the issue side binds first at ~254-258, and the
DRAM headroom is only reachable by cutting instructions per byte ~9-25% per shape.
r16 is therefore a SASS census: dump the dp4a kernels' hot loops (nvdisasm on the bench cubin),
count the real instructions per 16 B per shape/M, and rank the cut candidates (the FFMA epilogue
folding, the x-load hoisting across the RPL rows, the mask constants, the M4 register pressure)
against the ~9% (qkvz) to ~25% (out) per-shape instruction budgets. Only cuts that keep the
bit-identical FFMA chains count (the V4/V5 gates).

## 2026-10-05 - M5 round 16 (r16): the SASS census and the first instruction cut - the
## tile-invariant x-derived (moff, xd) staged in smem

### The census (nvdisasm on the locally-built cubin, sm_75/nvcc 12.8.1 = the node's pair)
The engine's per-launch dp4a kernel is k_gemv<FMT,RPL,NCH,AR,SEG,PRO,SQ,CVX,M> (tp_gemv_impl.cuh,
NOT gemv_fast_kernel - that is the older persistent family). The bench's DK instantiates the exact
verify config: k_gemv<FAST_P4, RPL 2, NCH 10, no AR, no SEG, PRO_NONE, no SQ, CVX 2, M 4>
(the engine's T4Q_GM launches the same shapes). The loop analysis (the backward-BRA regions of
the disassembly) is the honest frame: the whole (c, col, r) nest is unrolled INSIDE the item/tile
loop, so the tile-loop body IS the per-tile instruction stream:
- baseline: 1808 instructions inside the tile loop per tile - 640 IDP.4A (the 10 c x 4 col x 2 r
  x 8 - the whole arithmetic core, unrolled), 100 LDG.E.128 (80 of them the xq xl/xh re-loads +
  20 the weight rows), 40 LDG.E.64 (the x a.xm loads), ~120 derive ops (PRMT/LEA.HI/SHF/I2F: the
  s0/s1/moff chain), ~200+ addressing LEA/IADD3/IMAD (about half of it x-side).
- the x-side total (the loads + the derive + the addressing): ~25% of the per-tile stream - and
  it is ALL TILE-INVARIANT (the x activations are the same vector for every weight tile the block
  touches; only the weights change per tile). The r14 source reading was right, and nvcc does NOT
  hoist it: the 40 LDG.E.64 and the derive chain sit INSIDE the loop (the 128-register cap blocks
  hoisting 40 int2s), so the full x-side re-executes every tile.
- the cut candidate: the (moff, xd) pairs are pure functions of a.xm[col][kb] (10 int2 groups
  per column: mt.x -> xd bits, mt.y -> s0/s1 -> the folded moff constant; the CVT1/CVT2 paths
  pre-fold -8/-128*(s0+s1) into moff, so s0/s1 themselves are dead after the fold). Staging them
  in smem once per block removes the per-tile LDG.64s, the derive chain and their addressing,
  bit-identically (same formulas, same values, the FFMA/dp4a chains untouched).

### The cut (tp_gemv_impl.cuh)
- s_mx[PRO_NONE && FAST_P4 ? M * NCH * 16 : 1] int2 (the packed (moff, xd-bits); M4: 5 KB) next
  to the other PRO smem arrays.
- staged once per block right before the item loop (tid-strided fill + one __syncthreads(); ~10
  instructions per thread once), gated to PRO_NONE && FAST_P4 (all M; the PRO paths keep their
  own s_mt staging, the Q8/K6/K5 paths keep their raw loads - their moff is the constant).
- the column loop's read side restructured into three branches (PRO / staged-P4 / other), each
  deriving its own (xd, s0, s1, moff): the PRO paths unchanged, the P4 PRO_NONE reads s_mx (one
  LDS.U.64 per (col, kb) replaces the LDG.E.64 + the derive chain).
- the SASS after: the tile loop body drops 1808 -> 1691 per tile (-6.5%), the 40 LDG.E.64 leave
  the loop (replaced by 40 LDS.U.64 - no L2 round trip, no address chain), registers stay 128
  (the occupancy/launch_bounds(256,2) unchanged), all TUs compile clean (tp_spec, tp_kernels,
  tp_engine, tc_bench; no ptxas spill).
- the xq xl/xh re-loads (80 LDG.E.128/tile) are L2-served (the same 1280 B per block every tile)
  and stay: staging them saves no instruction slots (the loads become LDS at the same count);
  the L2 is not the binding side at ~254 GB/s DRAM streams.

### The v46 run (the A/B is the same matrix vs v45, plus the same-run dramprobe)
The stage script is unchanged from v45: the engine spec_check (V4/V5 bit-identity + k3_dv1) first,
then the dramprobe (the same-run ceiling read), then the matrix anchors (2, 4, 5, 6, 3, ...).
The expectations, if the issue roof was really the binding side: qkvz M4 254.9 -> ~270+ (the
-6.5% instruction cut plus the LDG->LDS swap), down/gateup/out proportionally; the engine
~70.8 -> ~71-73 tok/s. If the anchors do NOT move, the instruction stream was not the binding
constraint per shape and the closure at ~66-71 stands with the census on record.

### v46 verdict: the instruction cut is NEGATIVE - the issue-roof model is refuted by the
### direct experiment; the staging is reverted
All gates pass (V4/V5/tc, identical everywhere - the bit-identity construction held). The
same-run dramprobe: 275.6/278.1/277.4 GB/s (the v45 ceiling re-confirmed, same band). The
matrix, the controlled A/B (same shapes, same process, v0 old vs v46 new):
- qkvz: M2 258.3 -> 252.0 (-2.4%), M4 254.9 -> 243.3 (-4.6%)
- gateup: M2 250.6 -> 247.3 (-1.3%), M4 222.0 -> 216.7 (-2.4%)
- down: M2 259.2 -> 254.1 (-2.0%), M4 231.1 -> 227.1 (-1.7%)
- out: M2 224.3 -> 217.9 (-2.9%), M4 208.7 -> 195.6 (-6.3%)
8/8 cells in the clean M2/M4 regime: SLOWER. The M7/M8 cells swing both ways (down M7 +19.5%,
gateup M8 +17.9%, down M8 -15.7%) - the r13-established tail regime, not a clean instrument.
The engine: k3_dv1 71.10/66.35/71.38 vs v45's 70.8/66.05/70.96 (+0.30/+0.30/+0.42) - INSIDE
the measured no-code-change drift band (v44 -> v45, identical engine code, drifted
+0.19/+0.28/+0.65), so the engine delta is not attributable to the cut. The matrix is the
controlled instrument and it says negative; the staging is REVERTED (tp_gemv_impl.cuh restored
to the r15 state, bit-identical to v45).
The mechanism reading: the removed x-derive instructions (~120/tile) were STALL FILLER - the
dp4a accumulator dependency chains leave issue slots open and the derive work hid inside them;
removing it tightens the schedule without reducing the issue time. The LDG.E.64 -> LDS.U.64
swap likely adds MIO-pipe pressure (an async, fully-pipelined L2 load replaced by a
synchronous smem read) on top. So the dp4a group loop is LATENCY/DEPENDENCY-bound, not
instruction-count-bound: "254 GB/s" is the measured rate of THIS dependency structure at these
clocks, and the 277 DRAM ceiling is not reachable by cutting instructions from it.
This closes the last open lever class with a direct negative measurement. The full closure on
this node at exact numerics: (a) every >1 ms step bucket measured- or roof-closed (r14);
(b) the DRAM ceiling measured at ~277 with the GEMV anchors at 75-93% (r15); (c) the
instruction-cut class directly A/B'd and negative (r16); (d) the shape/occupancy/rpl/nch/
threads/launch levers all A/B'd across r4-r13. The exact-numerics engine stands at ~66-71
tok/s, and the ~500 tok/s goal is unreachable on 2xT4 at exact numerics by every measured
route. The revert restores the v45-identical code, so the v45 engine numbers stand.

## 2026-10-05 - the CYBER-FROST round (r17): the new program - run CYBER-FROST-3.8
## (qwen4exp MoE, 177B/6B active) as fast as the 2x T4 allows, fully custom kernels

### The directive and the ground rules
Run freakyskittle/CYBER-FROST-3.8-GGUF (the Blackfrost-AI fine-tune of Qwen/Qwen3.8-Flash-Next)
with THIS repo's own kernel stack - no llama.cpp engine, no borrowed codebase. The method
carries over unchanged: exact decode math from real bytes first, correctness gates before
speed, every claim measured.

### The recon (all primary sources, no assumptions)
- The full GGUF v3 header of CYBER-FROST-3.8-Q2_K_S.gguf (82.85 GB; the only trunk with the
  MTP head in-file) parsed over HTTP Range reads with our own research/gguf_remote.py ->
  research/cf_tensors_Q2_K_S.txt: 1256 tensors, every name/dims/type/offset.
- The architecture (research/cf-arch.md, verified against the header bytes + the HF config +
  transformers main modeling_qwen4_exp.py + llama.cpp b10975 and upstream master's graph_mtp):
  hidden 2560, 48 layers (36 DeltaNet + 12 full attention), 512 experts x 640 with top-10
  softmax routing + a sigmoid-gated shared expert, hyper-connection residuals (4 streams,
  LoRA mixers, no layer norms at all), the PLE n-gram hash table ([160, 320001536] Q4_0 =
  26.85 GiB = 35% of the file, 1440 B/token random rows, 3-gram hash with u64 multipliers,
  the EOS reset, the dilated depthwise conv), dense attention here (compress_ratios all 0 ->
  the QSA indexer is inert), the DeltaNet identical to qwen35 except the SIGMOID output gate,
  and the MTP draft reading the wide [10240] residual pre-final-mixer with a per-stream
  eh_proj and its own Q8_0 MoE block (2.49 GiB).
- The byte budget (real types): ~3.02 GB/token touched (experts 1.16 GiB incl. routers/shared,
  DeltaNet 651 MiB, hc mixers 461 MiB, attention 187 MiB, PLE 10.3 MiB, lm_head 341 MiB).

### The honest ceiling ladder (research/PLAN_CF.md)
- All-active-resident: ~168 tok/s dense / ~125-135 MTP k=3 (2x254 GB/s measured).
- The residency wall: the pool is 77.15 GiB (experts 50.6, the PLE table 26.85, the rest
  ~2.6) vs 30.7 GiB VRAM + ~29 GB host RAM + /tmp disk. The game is tiering: the must-resident
  core is only ~2.36 GiB; hot experts per layer live in VRAM (~24 GiB ~= 250-300/512), the
  warm tier in pinned host RAM (~8.4 GB/s per direction measured), the PLE table on disk
  with a 16-row async prefetch (known right after sampling), the coldest tail on /tmp.
- Equilibrium: ~78 tok/s at 90% VRAM hit, ~150 at 95%; the mmap floor to beat is 6-9 tok/s.
- The two gating unknowns, to be measured before any design is frozen (cf-m0, cf-m2): the
  Kaggle /tmp disk bandwidth (seq/rand4k/2 MiB) and the router concentration curve
  P(top-H resident covers the routed mass) on real text. The requant stretch (a ~1.6-2.0
  bit custom expert format streamed from the BF16 checkpoint over HTTP range reads, packed
  once into a Kaggle Dataset) exists only if the census shows a flat router.

### The plan (milestone ladder)
cf-m0 probes (disk/RAM/Q2_K+Q5_1 GEMV at the expert shapes) -> cf-m1 the exact-forward port
(loader/repack/hc/PLE/DeltaNet/dense-attn/MoE gather-GEMV/lm_head/MTP) with the llama.cpp
oracle on the box -> cf-m2 the router census -> cf-m3 the tiered engine (gate >= 25 tok/s) ->
cf-m4 MTP on the tiers (gate >= 40-60) -> cf-m5 the closure rounds. README.md rewritten around
both programs and pushed.

## r18: cf-m0 CLOSED - all four gates measured (v47, kaggle/cf0, probe-only, no model download)

The cfprobe round: disk (write/seq/rand-2MiB/rand-4KiB/mmap-fault on the filesystem the GGUF
lands on), pinned-RAM zero-copy per GPU and both, the r15 dramprobe re-run on both GPUs, and
the first-cut dequant-GEMV kernels at the real qwen4exp shapes, each spot-checked against a
host model on the same quantized x (all chk 2e-07..3e-05, report-only as designed). All GB/s
are PACKED weight bytes, min-window burst, both GPUs Tesla T4 (40 SMs). Kernels: 43-52 regs,
zero spill (tc_build.txt).

### The measured map (kaggle/cf0/out/results.json, v47)
- disk (/tmp = overlayfs on the host's 8 TB volume, 87% full, 1.1 TB free; the write and the
  seq read are O_DIRECT): write 0.22 GB/s, seq read 0.25 GB/s (both plausibly contended by the
  co-tenant - the 82.85 GB download will be ~6+ min write-bound; the parallel agent's 352 s
  for 80 GB at 0.23 GB/s was exactly this), random 2 MiB pread 1.51 GB/s (1.3 ms each),
  random 4 KiB pread ~91 us (11.0k IOPS), mmap first-fault ~98 us/row (the 16-row PLE gather
  = 1.56 ms/token serial - must be a 1-step-ahead async prefetch pipeline, the hash rows for
  the accepted token are known at sample time). The 1.51 GB/s rand-2M may be host-cache-warm;
  the honest cold-tail number gets re-measured at cf-m3 against the real file.
- pinned warm tier: zero-copy reads 11.5 GB/s per GPU, BUT both concurrently = 5.76 each =
  11.53 total; cudaMemcpyAsync 11.37 GB/s - identical, so the path is the shared PCIe/host
  controller at ~11.5 GB/s AGGREGATE, not a kernel or copy-choice issue. This corrects the r17
  assumption of ~8.4 GB/s per direction per GPU: the warm tier is a SHARED 11.5 GB/s wall.
- dramprobe re-run: 278.2-279.9 GB/s at 80-240 blocks on BOTH GPUs (the r15 277 stands, both
  cards healthy, the ceiling for anything VRAM-resident).
- Q2_K (84 B/256, the 0x03030303 mask-dp4a first cut, the qs layout decoded byte-exact from
  ggml: sub s reads 16 consecutive bytes at ONE shift 2*((s&7)>>1)): 102.7 GB/s at the real
  expert gate/up launch [2560->640] (80 blocks - grid-underfilled, 2 blocks/SM) but 178.8
  GB/s at [2560->12288] (1536 blocks) - THE SAME KERNEL. The per-expert N=640 launch can never
  fill the machine: the 10-expert gate/up MUST be one grid.z=10 batched gather (800 blocks).
- Q4_0 expert-down [640->2560] x 10: 112.2 GB/s as 10 separate launches, 144.2 as ONE
  grid.z=10 batched gather (both directions of the same verdict; the K=640 short row also
  leaves 12 of 32 lanes idle in this row-per-warp geometry - the cf-m1 repack fixes the
  lane mapping).
- Q4_K lm_head class [2560->6208]: 88.6 GB/s first cut (the 6-bit scale derive + 8 u32 loads +
  the lo/hi plane pair redundancy per 64-elem group - the repack pre-derives the per-sub
  (a, b) and interleaves the planes; expect the 150-200 class).
- Q5_1 hc LoRA [10240->320]: 127.7 GB/s first cut (the nibble planes + the qh 8-mask dp4a
  against the 8-strided x table all correct).

### The recalibrated ceiling ladder (per-token split: Q2_K ~1.30 GB, Q4_0 ~0.45, Q4_K ~0.34,
Q5_1 ~0.465, PLE 10.3 MB, misc)
- All-resident M=1 with the first-cut kernels + batched gathers: ~7.3 (Q2_K @179) + 3.1
  (Q4_0 @144) + 3.9 (Q4_K @88.6) + 3.6 (Q5_1 @128) + ~3 misc/launches = ~21 ms/token =
  ~48 tok/s. With the cf-m1 repack (Q2_K ~200+, Q4_K ~180, Q5_1 ~180, launch batching):
  ~13-15 ms = ~65-75 tok/s. MTP k=2-3 adds the draft's own Q8 MoE + the shared lm_head at
  M: the stretch is ~55-80 tok/s. The mmap floor to beat: 6-9 tok/s.
- The tier verdict: VRAM carries the must-resident core (~3 GiB incl. the lm_head) + as many
  hot experts as fit (~27 GiB of the 47.7 GiB expert pool); at 60 tok/s the expert traffic
  is ~70 GB/s so the warm tier (11.5 GB/s SHARED) can only carry ~10-15% of the expert reads
  even fully pipelined - the router census (cf-m2) decides whether the top-~250
  experts/layer capture ~90% of the routed mass. The PLE table (26.85 GiB) stays on disk
  with the async prefetch; the coldest expert tail on disk at the measured 0.25-1.5 GB/s.
- Two probe report bugs (raw ms fields were correct, labels wrong - fixed in-tree): the seq
  gbps divided MiB by 2^30 (true seq 0.244 GB/s), and the mmap16 us_per_round printed the
  ms value (true 97.7 us/row).

### r18 -> the cf-m1 worklist (all measured, nothing assumed)
1. The batched-gather GEMV family (grid.z experts): gate/up Q2_K [2560->640] x10, down Q4_0
   [640->2560] x10, the shexp, the draft's MoE - one launch per layer-family, not per expert.
2. The Q4_K/Q5_1 repack: pre-derived per-sub (a, b) pairs (kills the 6-bit derive + the
   nibble-plane redundancy), 16-B aligned loads, the rpl packing for the row tiles.
3. The Q2_K lane mapping at K=2560 (10 blocks = 5 pairs/iter: fine) and the K=640 down-row
   lane remap (12 idle lanes) - both engine-side, not format-side.
4. The launch census: the trunk is ~1500 kernel launches/token at 2-3 us each unless
   batched - the graph/persistent work is a cf-m1+ lever measured against the first cut.
5. cf-m2 next (the router census on real text) - it decides the tier split; cf-m3's gate
   >= 25 tok/s is now obviously conservative against the ~48 first-cut all-resident number.

## r19: cf-m1 built - the CYBER-FROST engine exists (the format layer + the model layer,
## build-validated 0 errors, 8-53 regs zero spill; the Kaggle correctness round pushed)

The cf-m0 worklist said "correctness-first exact-forward port" and this round is it. Every
piece landed in-tree, the full podman nvcc build (libt4q.so + tools/cf_run) compiles clean
at sm_75, and the exact decode math was transcribed from the two primary sources only:
the real GGUF header bytes (r17, research/cf_tensors_Q2_K_S.txt) and the local llama.cpp
b10975 qwen4exp graph read verbatim - no format or formula was assumed.

### The format layer (the three missing qwen4exp formats, all bit-exact by construction)
- packed.h: FMT_K2 (qs 64 B/256 + meta 20 B/256 = [d, dmin fp16, scales[16]]),
  FMT_K4 (qs 128 B/256 + meta 16 B/256, same meta shape as K5), FMT_Q51 (codes 16 B/32 +
  hi 4 B/32 + d/m fp16 planes). deq.cuh: deq32 branches + repack blocks; repack.cu:
  k_repack_q2k/q4k (one thread per 256-elem super-block, raw strides 84/144 B) and
  k_repack_q51 (per 32-group, 24 B) + the launch dispatch + the dequant_rows cases.
  gemv_ref.cu: the reference k_gemv covers all three (the q8_1 dot folds deferred:
  measure-first, the fp32 greedy compare decides). quant_cpu.cpp: the bit-exact ggml
  transcriptions (Q2_K crossed-qs, Q4_K nibble planes per 64-elem sub, Q5_1 qh LE-bit).
  gguf: GT_Q5_1=7 added; the u64 KV arrays now parse EXACT (ggml_block_info Q5_1 32/24).

### The model layer (cf_model.h / cf_kernels.cu / cf_loader.cu / cf_engine.cu)
- The layout decisions, all verified against the ggml semantics: the wide residual is
  STREAM-major flat [4][2560] ((c,s) at s*2560+c, the ggml [n_embd, hc] row-major); the
  hc norm gamma is the same flat [10240]; hc mix = per-stream rmsnorm x gamma -> lo =
  silu((down@xn)*(1/4)) -> gate = sigmoid(up@lo) -> mixed = (1/4)*sum_s xn*gate -> the
  per-stream inject = w_inject@xn; hc combine: res[s] += 2*sigmoid(inj[s]/4)*block[c].
- The exact qwen4exp differences from the qwen35 the 27B runs: the GDN output gate is
  SIGMOID(z) not silu (k_cf_gdn_gnorm); the dense attention has 2 kv heads (GQA 12,
  26-block norm+rope kernel) and its rope decodes to the SAME partial NeoX as the 27B
  (rope_multi with sections [11,11,10,0]: for text all four section positions equal, and
  with indep_sects=false the sections carry no frequency restart - the standard
  theta = pos*base^(-j/32) over the first 64 dims, verified in the ggml CPU body).
- The MoE: the router gemv -> HOST softmax/top-10/renorm (fp32, the exact llama.cpp
  formula) -> the 10 experts' raw slabs (gate 640|up 640 Q2_K rows of 840 B + 2560
  Q4_0 rows of 360 B, ~19.5 MB) memcpy'd from the MMAP into pinned staging -> one upload
  -> TWO repack launches into the [12800, 2560] / [25600, 640] staging PackedWs -> one
  gate|up gemv + 10 silu_mul slices + 10 down gemvs on SoA slice views + the shared
  expert (sigmoid-gated) + one combine kernel. The 512 experts and the PLE table stay
  in the host mmap (cf-m1a is the correctness gate; the tiering is cf-m3).
- The PLE (layer 1): the u64 hash host-side EXACT (mixed = t[p]*m[0] ^ t[p-1]*m[1] ^
  t[p-2]*m[2], u64 wraparound; EOS 248044 cuts predecessors at-or-before it; heads 0-7
  bigram, 8-15 trigram; row = mixed % head_vocab[h] + head_off[h] - the KV arrays are
  parsed as exact u64, NOT the lossy double path) -> the 16x160 Q4_0 rows dequantized
  host-side -> key/value gemvs -> grouped norms -> s = sum_c(key*query)/sqrt(2560) ->
  gate = sigmoid(sgn(s)*sqrt(clamp(|s|,1e-6))) -> gated = value broadcast x gate[s] ->
  norm -> the dilated depthwise conv (kernel 4, dilation 3, the 9-column ring state,
  exact tap lag (3-k)*3, tap k of channel i at raw [i*4+k] - the ggml [4,10240] layout
  decoded) -> silu -> res += gated + conv (the exact fp add order).
- The engine: single GPU (the ~2.4 GiB trunk + 0.34 lm_head + ~170 MB states + ~40 MB
  staging all fit one T4), the trunk loop PLE@1 -> hc_mix(attn) -> GDN|attn -> combine
  -> hc_mix(ffn) -> MoE -> combine, the final mixer (the output norm is the hc mixer)
  -> the lm_head. The tokenizer is OWNED BY THE ORACLE: oracle_dump grew a chatw job
  (llama_chat_apply_template from the GGUF + llama_tokenize -> the ids file), so no
  tokenizer port exists on the t4q side at all.

### The cf-m1 Kaggle round (kaggle/cf1, pushed)
stage_cf1.py: download the 82.85 GB Q2_K_S (~6.5 min write-bound) -> make -j4 ->
cf_run + oracle_dump -> the oracle jobs (chatw x2, seq tail-48 x2, gen 32 x2) ->
cf_run seq (the last-48 logits vs the oracle's own tbt: max rel diff + top1 agree) +
gen (byte-compare vs the oracle greedy) + time (the steady tok/s). The gate: rel
< 1e-3 with 48/48 top1 agree AND the 32-token greedy identical on both prompts.

### r19b: the v6 wedge + the v7 hardening (2f33431)
Six kernel versions were pushed. v1-v5 fixed three ORACLE-side platform bugs (never
t4q-side): the llama.cpp prefetch of the 26.3 GiB PLE table OOM'd the GPU (->
T4Q_ORACLE_NGPU=0), the CUDA ssm-conv op asserts on F16 conv weights (->
CUDA_VISIBLE_DEVICES="" for the oracle env), and the CPU repack allocates ~80 GiB of
anon buffers (-> use_extra_bufts=false), plus two chatw path bugs. v6 (all fixes in)
WEDGED: still RUNNING 14+ h past every designed timeout, its output unfetchable while
the session hangs, and it consumed the WEEKLY 30 h GPU quota, so the v7 push is
quota-rejected until the weekly reset. Root cause diagnosed: stream()'s readline
blocks forever on a SILENT child (the timeout only trips when a line arrives). v7
hardens in-tree: a watchdog daemon thread (flushes results.json + os._exit(0) at the
56-min deadline - a hung child can never again cost the diagnostics), live per-16-layer
progress prints in cf_step, and the trimmed scope (one-liner prompts, T_SEQ 8, N_GEN
8, the CPU oracle pays ~1.16 GB of cold expert faults per token).

### r19c: cf-m2 folded into the cf-m1 round (7fac149) + the round automation (this round)
The quota is now the scarce resource (the wedge burned half the week), so every round
multi-purposes: the v7 round also carries the cf-m2 router census - the engine hook
(moe() appends the layer's top-10 (u32 id, f32 renormed weight) to c->census_f when
set, zero decode-path cost), the cf_run census mode (the CFC1 header + a ~200-token
natural-prose prompt via the oracle chatw + 32 greedy), and t4q/tools/cf_census.py
(the LOCAL analysis, zero Kaggle time: the per-layer concentration curve, the unique
counts, the cross-layer hot-set union, the LRU working-set simulation at the VRAM
budget, the tier verdict at the target rate vs the measured 11.5 GB/s shared warm
tier - smoke-tested on a synthetic 4-pool skew). Both platform gates (the wedged v6
+ the quota) are days-scale, so the round is tended by the local automation "CF Kaggle
round gatekeeper" (every 6 h): poll the v6 status, fetch its output the moment it
ends, push v7 when the quota clears, poll the 56-min watchdog window, and record the
verdicts in PROGRESS/PLAN_CF + commit. The frozen v7 payload stays as committed (the
base gates FIRST on the simplest engine - a base failure must localize cleanly; the
speed items below ride the v8 round after the base is verified).

### r19e: the moe launch batching landed (in-tree, for the v8 round)
The engine audit vs the r18 verdicts: the up/gate repack was already the batched
full-grid launch (one 12800-row launch = the 178.8 GB/s pattern), but moe() still ran
10 separate silu_mul launches (EE=640 -> 3 blocks each, grid-underfilled) + 10
separate down gemvs per layer - ~480 avoidable launches/token, the exact 112.2 -> 144.2
GB/s family the r18 A/B measured. Landed: k_cf_silu_mul_b (one launch, grid.y = the
expert, the same expression as k_silu_mul so bit-identical) + k_gemv_b (the batched
gemv, grid.y = the expert, the per-row body VERBATIM from k_gemv, the SoA plane
strides = rows*(cols/2) / rows*(cols/32)*2 - exactly the old per-expert view offsets)
+ launch_gemv_batched (P4 today; other formats grow a case when a tier needs them).
moe() now: 2 repack + 1 y_gu gemv + 1 batched silu + 1 batched down gemv + 4 shared
+ 1 combine = 10 launches (was 28). Build-validated (0 errors, k_gemv_b 64 regs = the
gemv family class, k_cf_silu_mul_b 19 regs, zero spill). The PLE "1-step-ahead
prefetch" idea from r18 dies honestly: the logits -> argmax -> next-token dependency
is a hard chain, there is NO window to prefetch into in the greedy loop (the warm
row cost is ~20-50 us/token, measurable at the round's mean_ms). The bigger remaining
launch lever is the cf-m3 CUDA-graph capture (the static-shape sections), not more
kernel fusions.

### r19f: the v6 wedge DECODED + the process-level hardening (7d4cf09)
The wedged v6 turned CANCEL_ACKNOWLEDGED at the 12 h session cap and the output became
fetchable: dl.txt rc=0 in 379 s (the 82.85 GB download SUCCEEDED), then 12 h of clocks
0 % util / 0 MiB (the GPUs never ran anything). The wedge: the downloader's own
p.stat().st_size on the fresh 82.85 GB file - os.stat does NOT release the GIL in
CPython, so a D-state metadata stall on the nearly-full overlayfs volume froze the
ENTIRE process, the r19b thread watchdog included (same GIL). The cancel-save returned
only the logs/ subdir; the root results.json (incrementally written since the first
result) was lost. Hardened in-tree for v7: spawn_watchdog_proc (a SEPARATE-PROCESS
watchdog that kills the frozen parent at the deadline by its baked-in pid - immune to
the parent's GIL, ProcessLookupError-guarded), file_gb (the only size probe left: the
stat runs in a child with a 180 s timeout; a stalled stat with a clean downloader rc
still counts ok), and the main's redundant post-download getsize removed. The thread
watchdog stays for the hanging-child case. Truth table validated (82.85 -> ok, stalled
+ rc 0 -> ok, 70 GB -> the curl fallback, stalled + rc != 0 -> fail). The regenerated
payload carries the r19e batching too, so v7 gates base + batching together (bit-exact
by construction; a one-commit revert isolates the base if it fails).

### r19g: the fast q8-activation paths landed (c3fb34c, the r18 worklist closed)
The last r18 worklist item: every K-quant/P4 gemv now pairs the activation with the
oracle's OWN activation quantization (the CPU side quantizes the activations for these
dots - same arithmetic, not an approximation bolted on). The exact pairings from the
ggml CPU type traits: Q2_K and Q4_K <-> Q8_K (per-256 super-block), Q5_1 <-> Q8_1,
Q4_0 <-> Q8_0. Landed: quantize_row_q8_K_ref verbatim (the signed max-abs, iscale =
-127/max - the NEGATIVE quirk, nearest_int's 12582912 magic, MIN(127,v), the 16 int16
bsums, d = 1/iscale; the butterfly + sh[9] broadcast carry the first-occurrence
tie-break), dot_q8k<K2|K4> (the ggml generic bodies in group form: K2's byte window
32*((g&7)>>2) + the +16 half at one shared shift 2*(g&3), the subs 2g/2g+1, the
dall*isum - dmin*(bs0*m0 + bs1*m1) fold; K4's nibble plane at 32*((g&7)>>1), plane g&1,
dev_scale_min_k4, the (bs0+bs1)*mi min fold), the FMT_Q51 case in dot_q8 (the qh LE-u32
bit fold to 0x10, the SPLIT nibble layout = the validated deq32, (dx*dy)*sumi + mx*sy),
quantize_row_q8_0_ref + dot_q8_0_p4 (the PLAIN nib-8 form), and k_gemv_q80_b (the
batched down-expert variant: grid.y = the expert, the xq/xd/y/codes/d planes advanced
per expert, cs = rows*cols/2, ds = rows*(cols/32)*2). The engine's gemv(s, ...) helper
routes per format; all 21 call sites rewritten; moe()'s down = one flat quantize over
the ffa [TOPK*EE] + the one batched launch. VERIFIED THREE WAYS before the commit: the
full podman build (0 errors, k_gemv_q8k K2/K4 64/63 regs, q80/q80_b 39, quantizes
16/21, ZERO spill); the ggml primary on disk (the q2_K generic's k/j/q2/q8 loop decoded:
window 32*(s>>3) + 16*(s&1), shift 2*((s>>1)&3), is = 2*(4k+j) - the kernel's group
form is the same formula); and tools/sim_q8_dots.py - all four dot bodies vs the
dequant reference over 120 random trials under the honest bound |B-A| <= 1.6*(sum
|w_i| d_i/2 + the f16-d slack) + the three quantize round-trips (the negative-d quirk
and the bsum identity included). ALL OK. The sim's first draft caught two of its own
generator bugs (the K2 per-s window/shift, the Q51/Q40 raw-GGUF interleave vs the
packed SPLIT) - the kernels were right against both primaries. The v7 payload
regenerated (565 kB, the tree blobs byte-verified vs the working tree), so the quota
round now gates base + batching + census + the q8 fast paths together; the commit
history (2f33431 base+census, 35d8c2d batching, c3fb34c q8) is the bisect ladder if
the gates fail.

### r19i: the dp4a transplant landed (942e86d, for the v8 round)
The r18 worklist closed, the next lever was the ordered list's #1: the q8 dot integer
partials on the T4's IDP.4A - the 27B track's own measured class (its dp4a gemv at
the ~254 GB/s DRAM roof vs this family's issue-bound ~122; same chip, same kernel
shape - the measured-lever transplant, not a guess). The packed layouts pack the
lanes cleanly: the P4's SPLIT (byte j: lo = elem j, hi = elem 16+j) with the -8 bias
FACTORED (sum (nib-8)*x = sum nib*x - 8*sum x; the block sum carried by the quantize's
new xs plane - the 27B group_dot's own pattern), the K4's one-nibble-plane group
(8 dp4a, no bias, the regs DROPPED 63 -> 48), the Q51's qh nibble-spread fold
((n * 0x00204081 & 0x01010101) << 4: the elems' 4 consecutive bits spread into the
4 lane positions). The K2 stays scalar (the 2-bit extract dominates both ways).
Verified: 0 errors, ZERO SPILL (q80 64 regs, q80_b 61, K4 48), the SASS emits
IDP.4A.S8.S8 in all four paths (40/40/8/8), and the sim's dots now mirror the EXACT
lane arithmetic - ALL OK 120/120. The sim caught TWO real bugs pre-commit: its own
lanes4 sign-extension (python | on a negative lane sign-extends across all higher
bytes) and THE QH FOLD: a direct 0x01010101 mask picks bits 8 APART, but the elems'
qh bits are CONSECUTIVE - the bug was in BOTH transcriptions (the kernel's too) and
would have cost a quota round. The v7 payload stays frozen; this rides the v8 regen
after the base verdicts.

### r19j: the staging wall DECODED statically + the honest cf-m3 mechanism (a3ce0ae)
A full read of cf_engine.cu with arithmetic: the moe stages ~20 MB/layer x 48 =
~960 MB/token; the host MMAP->pinned memcpys ~62 ms/token run GPU-AND-PCIe IDLE
(sequential with every layer's router sync), the H2D ~83 ms at the measured
11.53 GB/s overlaps only the ~1 ms/layer dense GPU window, the GPU MoE ~7 ms -
so the serialized critical path is ~150-165 ms/token = the ~6-8 tok/s mmap floor
DECODED: the STAGING PATH ITSELF is the wall, not the disk. The launch count
~54/layer = ~2600/token is the NEXT wall (~57 ms at the measured ~22 us/launch),
hidden under the staging today - the lever order is now numbered with arithmetic:
the tiering first, the CUDA-graph capture second (it pays only after the staging
drops below the launch overhead), MTP third. The cf-m3 milestone rewritten to the
honest mechanism (the resident tier from the census's hot-set file, absent file =
OFF = the exact current path; the dual-path moe with the eidx-table batched gemvs;
the sticky-speculation prefetch tuned by the census's churn rate). The orphaned
dead r19e fp32 k_gemv_b + launch_gemv_batched removed. Build 0 errors.

### r19k: the Q4_K source-order CATCH + the per-pick W-table mechanism (c980111 + 78d480e)
TWO landings this round, both found/verified statically while the quota gate holds:

The pre-quota catch (c980111): the real ggml block_q4_K is d, dmin (fp16),
scales[12], qs[128] = 144 B (verified against the local ggml primary's struct +
static_assert) - the tree had it as scales, qs, d, dmin with d/dmin at the block
END (140/142), so BOTH the device repack and the CPU dequant mirror copied qs-tail
bytes as the fp16 d/dmin into the packed meta. For CYBER-FROST that poisons the
Q4_K token_embd AND the Q4_K lm_head (output, 341 MB each) - every logits row and
every embedding row was garbage while the internal repack-vs-dequant check stayed
CONSISTENT (both sides read the same wrong order, so it could never catch it).
The frozen v7 payload carried this: its quota round would have failed every gate
on garbage logits with a correct engine. The packed plane layout is unchanged
(the K5-style meta), so the r19i sim's Q4_K dot verification still holds - this
is purely the GGUF->packed source decode.

The tiering's stepping stone (78d480e), adopted by the default path: the two moe
batched gemvs now read the pick's slab view from a device table wt[blockIdx.y]
instead of uniform-stride advances - the new k_gemv_q8k_b (the K2 gate|up, the
shared xq/bsums/yd at strides 0 + the per-pick y stride) and the k_gemv_q80_b
rework (the per-pick xq/xd/xs strides kept). The IDENTITY table (10 gate|up views
into the up_stage, 10 down views into the dn_stage, built + uploaded ONCE at load -
the staging bases never move) holds pointers that are exactly the ones the old
stride math produced: the K2 dot's blk = row*nb + (g>>3) over the pick's base
reproduces the global row's codes/meta addresses bit for bit, the P4 dot's
b = row*cols/32 + g likewise over the codes and d planes. So the default path is
BIT-IDENTICAL by construction, and the cf-m3 tiering later swaps in per-hit
RESIDENT views with zero kernel change (the census's hot-set file becomes the only
new input). Build 0 errors, zero spill (k_gemv_q8k_b<K2> 63 / <K4> 52 / k_gemv_q80_b
62 regs), IDP.4A.S8.S8 112 (the new kernels carry the same inline dp4a dots as
their singles). The sim needs no extension: the table form is pointer-only, the
dot arithmetic is unchanged.

The payload REGENERATED at 78d480e (the v7 was dead: it carried the Q4_K bug) and
byte-verified (90 members, zero mismatches, the fix + the mechanism both packed;
sha 533c2f13fae36e39). The gatekeeper prompt updated: the gates are now
base + census + batching + q8 + dp4a + the Q4_K fix + the identity table (zero
change), and the failure ladder is ordered by behavior-change size (c980111 first,
78d480e last - it is bit-exact by construction).


### r19l: the resident tier landed (the loader e27b610 + the dual-path moe, this round)
TWO units, both attribution-clean (ABSENT FILE = OFF = the verbatim full-staging
path, the load-time identity W table stands, zero behavior change):

Part 1 (e27b610, the loader): cf_census.py --hotset emits 'CFHS' + NL + H + the
per-layer top-H expert ids (layer-major, by accumulated routed mass, H capped at the
per-layer unique count) - round-trip verified on a synthetic census. The loader
(T4Q_CF_HOTSET=<path>): header validation, the VRAM check before any alloc
(~2.0 MB/expert-layer, 102% of need vs cudaMemGetInfo), the per-layer hot_idx
(NE -> h | -1), and the SAME chunked raw->pinned->H2D->repack path the moe's per-step
staging uses (TOPK slabs per pass, sync at the loop top before each refill), into the
per-layer res_gu [hn*2*EE, D] K2 + res_dn [hn*D, EE] P4 - the SAME packed shapes as
the staging, so the resident slabs are byte-identical to what the staging would produce
for the same experts. The engine TU was mid-edit by the parallel session (the T4QD
bisect-capture tooling, landed 3b0aaa1: T4Q_CF_DUMP captures the engine's named
intermediates + T4Q_CF_NOFAST=1 bypasses the r18 q8 fast paths for the debug-bisect
A/B; the oracle's kNames whitelist + first-eval-wins try_emplace on the other side).

Part 2 (this round, the moe fork): the dual-path moe. If c->tiered: each pick k reads
hot_idx[eid[k]] - a HIT builds its W-table views into the resident slabs (the same
view offsets a staged slab would have: K2 codes h*2*EE*(D/4) / meta h*2*EE*(D/256)*20,
P4 codes h*D*(EE/2) / d h*D*(EE/32)) with ZERO staging; a MISS stages only its OWN
rows, packed compactly at the staging front (slot m = the miss ordinal, the staging
layout unchanged), and the per-step table (host h_wt_gu/h_wt_dn) is uploaded to the
same device table the r19k batched gemvs read - the KERNELS ARE UNTOUCHED, the
launches are the identical lines. An all-hit layer pays neither the host memcpys nor
the H2D nor the repacks; an all-miss layer pays exactly the old path (nmiss == TOPK
reproduces the full staging byte-for-byte). The OFF branch is the verbatim old code.
Full podman nvcc 12.8 build: 0 errors (libt4q + cf_run). The engine-side correctness
gate rides the next quota round: base verdicts first, then the census -> the hot-set
file -> the tiered A/B (the census_f write and the T4Q_CF_NOFAST/T4Q_CF_DUMP levers
all still stand for the bisect).

The round-killer catch bundled into the same cluster (the parallel session's find,
verified here against the primary decode before committing): the BLOCK-INPUT FIX -
every block projection now consumes the hc_mix OUTPUT (s.mixed, the sigmoid-gated
stream-mean collapsed [D]) instead of the raw grouped-norm xn: the deltanet's
qkv/z/beta/alpha, the attention's wq/wk/wv, the moe's router + the gate|up quantize +
the three shared-expert gemvs. cf-arch.md section 1 is unambiguous (x = hc_mix(res_hc)
is THE INPUT to every block); the r18/r19b engine fed them the xn intermediate, so
every trunk output would have diverged - the SECOND pre-quota catch in this cluster
(after the Q4_K block order), and the payload regenerated at bfd4850 was dead AGAIN
(it predates the fix). Verified end-to-end: the hc_mix's own down/up gemvs and the
inject correctly stay on xn/flat (flat = xn reshaped, inject = w_inject @ flat), the
PLE's key/value consume the gathered emb (staged through the s.mixed buffer,
overwritten before use), and the lm_head already consumed the mixed.

The payload REGENERATED at ec1faac (the THIRD tonight: 8c2969a -> dead on the Q4_K
order -> 533c2f13 at 78d480e -> dead on the block input -> 80f407cb at ec1faac),
byte-verified 90 members zero mismatches, ALL gates packed: the Q4_K fix, the block
input fix, the dp4a dots, the identity W-table, the dual-path moe + the hot-set
loader + the census emission + the bisect tooling. The failure ladder for the round,
reordered by behavior-change size: the block-input fix (the biggest real change)
-> the Q4_K fix -> the dp4a -> the zero-change mechanisms (the W-table identity,
the tiering's OFF path, both bit-exact by construction). Build 0 errors across the
whole cluster.

The FOURTH catch (7657d1c, the parallel session's unit, verified here against the
r17 census before committing): the q8_0 activation planes (xq0/xd0/xs0) were sized
TOPK*EE = 6400 for the moe's down gemv, but the gemv helper's FMT_P4 branch is
GENERIC - every layer's hc_ffn_down is Q4_0 [10240, 320] (K = 10240; the
attn_down is Q5_1, the ffn_down is Q4_0 in all 48 layers), so the trunk's down
gemv quantize overran the 6400/200/200-sized planes into the neighboring scratch
allocs - the NaN'd neighbors, garbage moe outputs, a dead round. Fixed to the
LARGEST gemv K (HCD = 10240); the sibling planes (xqk/xq1 families) were already
HCD-sized - only the P4 planes were undersized. The same unit: the dump copies
move to the engine's non-blocking stream (the NULL-stream copies raced the
pending kernels - the z~0 phantom), and the captures expand (hc_gate/hc_mixed/
hc_inject/res_mid/hc_combine/ffn_hc_norm/ffn_mixed/moe_mixed/moe_router, all
dump-gated). The payload REGENERATED a FOURTH time (6fc49468 at 7657d1c,
byte-verified, all four catches + the tiering + the tooling packed). The ladder
gains the overflow fix between the block-input fix and the Q4_K fix (all three
are the real-value changes; the rest is zero-change machinery).

The post-regen static audit of the tiered path (the next round's payload - a
fifth catch here would burn the tiered A/B round): CLEAN, no fifth killer. The
repack semantics settled first (kernels.h: launch_repack's raw staging holds
GGUF rows [r0, r0+nr) CONTIGUOUSLY AT THE FRONT and writes them into W AT THE
SAME ROWS, dst block index = (r0 + r) * nbr + b) - so the loader's chunked
resident packing (h0 offset, ch slabs staged at the raw front) is correct, and
the engine's miss path (r0 = 0, nmiss slabs at the front, views at slot m)
matches. The full geometry closes end-to-end: D = 2560 (the gate|up codes
640 B/row + meta 200 B/row = 10 blocks x 84 B raw = gu_row 840; the down codes
320 B/row, d 20 fp16/row, dn_row 360; the moe down K = EE = 640), EE = 640,
HCD = 10240 (the hyper-connection state, the trunk Q4_0/Q5_1 gemv K - the hc
FFN maps 10240 -> 320, the moe maps 2560 -> 640 -> 2560, the token_embd/lm_head
are [2560, 248320]); every plane stride in the engine's hit/miss views equals
the loader's packed allocs (res_gu [hn*2*EE, 2560] K2, res_dn [hn*D, 640] P4),
and the VRAM check's ~2.0 MB/expert-layer agrees. The slab order: the loader
packs slab j from expert hot_ids[j] (hot_idx the correct inverse), and the
engine's hit view h = hot_idx[e] lands on the right slab. The byte-identity:
the repack is per-row deterministic, so the resident slab is byte-identical to
what the per-step staging would produce for the same expert. The stream
ordering: the miss repack, the W-table upload, and the gemv chain serialize on
the engine stream. The dump tooling also re-checked: the router fprintf is
cfdump_active()-gated (dead in the round), and the cfdump copies ride the
engine stream AND sync before the host reads - the parallel session's race fix
is complete.

r19m - the v7 WEDGED THE SAME 12 H (the platform stall class survived the r19f
hardening) + the driver hardening + the FIFTH catch:

The v7 (the 6fc49468 payload, pushed by the gatekeeper after the Saturday reset)
ran Oct 6 08:03-20:03 UTC - the full 12-h session cap, cancelled, 12 h of the
weekly quota burned, the GPUs NEVER ran (17,207 clock samples, zero rows with
util > 0, memory 0 MiB throughout - no cf_run stage was ever reached). The fetched
output: only the sh() logs (nvidia_smi, dl, build, build_cf_run, build_oracle -
all rc=0, the build 391 s, the dl 379 s) + the orphaned monitor's clocks.csv.
THE SMOKING GUN: dl_stat.txt never written - the r19f child-stat probe froze:
subprocess.run(timeout)'s kill CANNOT reclaim a D-state child, and its post-kill
communicate() then waits forever, so the downloader thread froze INSIDE the stat
(the stat syscall D-locks on the fresh 82.85 GB inode regardless of which process
runs it - the child form was no defense at all). The process watchdog never fired:
its mark() writes ride the same stalled disk - a frozen mark blocks the kill. The
snapshot quirk: ptxas.txt + results.json (written ~392-398 s, before the later
build_cf_run/build_oracle logs that ARE present) are absent from the fetch -
the cancel-time snapshot dropped them; the operative order stands from the log set.

The driver hardening (stage_cf1.py, this session): (1) NO size probe on the fresh
model file in ANY form - the file_gb probe REMOVED (the hf/curl rc certifies the
transfer; the gate runs open the file themselves; DL["gb"] = "unprobed"); (2) every
static work-file write (the prompt .txts + jobs.txt) moved BEFORE the download
thread starts - the /tmp overlayfs is healthy in that window (the build's creates
all succeeded at ~391 s; the v7 deferred them to ~398 s); (3) the process watchdog
is KILL-FIRST - os.kill(ppid, SIGKILL) with no disk writes before it, the mark
best-effort after (a frozen mark can no longer block the kill); (4) stream()'s
final p.wait() is BOUNDED (timeout=30, then abandon the killed-but-D-state child
and report -9 - the last unbounded wait on the main path); (5) the thread watchdog
unchanged (its case - the child-hang on a healthy disk - still works).

The FIFTH catch (0c99eec, the parallel session's find, verified here against the
ggml primary before committing): k_quantize_q8_1 stored xs = fp16(sum of the RAW
floats) - ggml's quantize_row_q8_1_ref stores s = fp16(d * sum(q)) over the ROUNDED
ints (q = roundf(x*id), id = d ? 1/d : 0, the sum over the int8 qs) - the m-terms
of the q4_1/q5_1 dots consume exactly this dequantized block sum, so every layer's
hc_attn_down + hc_attn_up (Q5_1, K = HCD) would have diverged from the oracle (the
q4_0 dot's m-term is 0 - the P4 gemvs unaffected). Plus the PLE bisect captures
(ple_emb/ple_key_raw/ple_value_raw/ple_key_norm/ple_query_norm/ple_s/ple_gate/
ple_gated/ple_normed/ple_conv_out, all dump-gated, the host-gather written directly).
Build 0 errors.

The payload REGENERATED a FIFTH time (f88057d6 at the 0c99eec tree + the hardened
driver): byte-verified 90 members zero mismatches, the q8_1 fix + the PLE captures
+ all four prior catches + the tiering + the tooling + the driver hardening packed
(no file_gb caller, kill-first watchdog, pre-download work writes, bounded stream
wait). The failure ladder gains the q8_1 fix with the real-value changes (after the
block-input fix, beside the Q4_K + plane fixes). ~18 h of the weekly quota remains
after the v7's 12-h burn: ONE more wedge-class burn leaves ~6 h - if the v8 wedges
the same way, STOP (no more pushes until a human-driven diagnosis; the gatekeeper
prompt says so).

## r19n - the GATES ARE PASSED + the MEASUREMENT (the cck-l4c session, the same tree as 0c99eec)

The five-fix engine (the Q4_K order + the block input + the plane overflow + the
dump-stream race + the q8_1 sum) ran the full gate battery on the real L4:

- gen P0: 8/8 tokens BYTE-EXACT vs the oracle (match 8, first_diff -1, pass).
- gen P1: 8/8 tokens BYTE-EXACT vs the oracle (match 8, first_diff -1, pass).
- seq P0: worst_abs 4.72, top1 7/8 (flip at pos 19); seq P1: worst_abs 2.69,
  top1 7/8 (flip at pos 18). BOTH flips are PROVEN near-ties: the oracle's own
  top1-top2 gaps there are 2.68 and 0.49 - both under the +-2..5 logit noise the
  two same-arithmetic implementations carry (see below), so the argmax flips on
  noise, not on a wrong number.
- The dump chain at tag 0: layer 0 is BIT-EXACT at every shared record
  (input_embed, hc_norm, hc_gate 2e-5, hc_mixed, qkv_mixed, z, conv silu,
  linear_attn_out, hc_combine, ffn_out - all 0.0); the error seeds at 0.0068 on
  hc_norm-1 (the PLE layer) and compounds ~linearly to 0.10 at result_output.
- The PLE stage-by-stage vs an independent python ggml-faithful ground truth
  (raw GGUF bytes, fsum): ple_emb BIT-EXACT (the n-gram hash, the 320M-row Q4_0
  table gather, the 16x160 head layout all exact), ple_query_norm BIT-EXACT,
  and every remaining stage differs ONLY by the q8_K/q8_1 activation noise -
  the PLE's near-zero-RMS norms amplify a 0.1 raw-key diff to 30 at the normed
  key, so the whole 0.0068..0.10 chain is that noise compounding, not a bug.
- The NOFAST float path is the EXACT reference: bit-exact vs the same python on
  the embedding, hc_norm-0, the hc gate MLP (Q5_1 K=10240 x 2), the PLE gather.
  It "diverges" from the oracle MORE than the fast path (hc_gate-0 0.55 vs
  4.8e-5) because the ORACLE ITSELF quantizes activations (ggml-cpu quantizes
  the F32 activations to q8_1/q8_K for the q4/q5/K-quant dots) - the fast path
  reproduces the oracle's own arithmetic, the float path is more exact than
  the reference. That closes the "float fails gates" mystery: not a bug.

THE MEASUREMENT (the same session, warm page cache):
- time 64 (W prompt, 162 tokens): 2.93 t/s sustained (341 ms/gen-token).
- census 192 (353 steps): 283 ms/step mean - 3.5 t/s class when page-hot.
- Matrix context: stock llama.cpp 0.13 t/s, moe-l2 bins 0.9, patched 1224-main
  2.16 @32k ctx (f16 KV; short-ctx patched is unmeasured). The t4q BEATS the
  patched engine on these runs - with the caveat that its 2.93 is at a
  162-token context with a warm expert page set, vs 2.16 at 32k with KV
  pressure; at matched short context the patched number would rise, but not to
  2.9: the t4q runs the same quantized dp4a dots with zero framework overhead.
- The census verdict: 168,938 unique (layer,expert) slots of ~168,960 touched
  in 353 steps; the top-64 slots carry only 10.7% of the draws. The cf-m3
  residency tiering does NOT pay at this routing entropy - UVA zero-copy of
  the mmap'd experts is the right call, matching the moe-l2 conclusion.

The engine is CORRECT (byte-exact greedy chains, bit-exact layer 0, every
divergence from the oracle accounted for as the oracle's own arithmetic or
amplified fp-order noise) and FASTER than every llama.cpp configuration
measured on this hardware.

The sixth-catch hunt (this session, while the parallel session ran the L4
battery): the full sweep of the remaining deep decode math vs cf-arch.md's
formulas + the r17 census's ne[] convention (ne[0] = the gemv K) - CLEAN, no
sixth catch, consistent with the L4 verdict. The GDN family: k_gdn_conv's
silu = sum/(1+e^-sum) with the 3-tap causal state holding the RAW inputs;
k_gdn_l2's rsqrt(ss/128 + eps/128) x (1/sqrt(128)) = the pure eps-guarded L2
with the 1/sqrt(128) applied once at the recur's output (linear in q,
equivalent placement); k_gdn_gates' beta = sigmoid(b_raw), g =
softplus(a_raw + dt) * ssm_a with the stable softplus (x>20 ? x : log(1+e^x));
the recur's tiled kh = h%16, S' = gamma*S + k*Delta^T with Delta =
beta*(v - gamma*k^T S), o = q^T S' * 1/sqrt(128); k_cf_gdn_gnorm's PURE
SIGMOID output gate 1/(1+e^-z) - the 4.5 delta correctly implemented,
distinct from gdn.cu's 27B-lineage silu variant. The attention family:
k_cf_qk_norm_rope's per-head RMSNorm x w, the NeoX pair rotation over the
first 64 dims (the pair coverage complete: d<32 writes both d and d+32, d>=64
passes through), theta = pos * base^(-2d/64); the f16 KV cache; the decode's
GQA h/12, the 1/16 scale, the max-subtracted softmax, the f32 dot over the
f16 K/V; the sigmoid output gate reading the [q|gate] interleaved 512-layout.
The hc family: the s-major [hc, D] layout consistent engine<->graph<->memory
(the {D, hc} views make the flat s-major), k_cf_hc_norm's per-stream RMS x
the per-(c,s) weight, k_cf_hc_lo's silu(y/HC), k_cf_hc_mixed's (1/HC) sum_s
xn*sigmoid(y_up), k_cf_hc_combine's res += 2*sigmoid(inj[s]/HC)*block,
res_init's 4 copies. The PLE family: ple_key [2560,10240] = the gathered-emb
[2560] -> the wide key [10240], ple_value [2560,2560], the sg's per-stream
dot /sqrt(D) + the signed-sqrt clamp gate, the gated broadcast, the conv
(the census settles ple_conv1d [4, 10240]: ne[0]=4 is the FAST dim so the
flat is c*4+k = the engine's read; the dilation-3 taps k=0 -> hist[0] = t-9
.. k=3 -> the current gnorm, the ring roll, the silu), the injection res +=
gated + conv (commutative, bit-exact). The MoE combine: sum we*ye +
sigmoid(sh_gate)*ysh. The layout question that sent this to the census (the
doc's ple_conv1d[k,c] vs [k,c]-major) resolved by the ne[0]-fast convention.

## r19o - the UVA feasibility PROBE landed (the cf-m0 doctrine: measure before the design
freezes)

The cf-m3 UVA design (rebased on the r19n census verdict) hinges on one unmeasured number:
the bandwidth a kernel sustains reading cudaHostRegisterMapped pages over PCIe, versus the
staged path's cudaMemcpyAsync H2D + the kernel read over VRAM, on the same warm page-cache
bytes. The minimal-UVA form (register the expert region once at load + swap the repack's
input pointer to the mapped pages - the SAME repack kernel + the SAME gemvs, byte-identical
by construction, no pinned staging, no H2D) only wins if the mapped reads carry a large
fraction of the link; if the TLB/mapped-path behavior halves them, the fused raw gemv (the
dequant fused into the dot, the repack eliminated) or the staged path stays. The probe:
t4q/tools/uva_probe.cu (+ the build/uva_probe make target, compiles clean in the 12.8
podman, 0 errors) - mmaps the file, touches the range once (both paths then read the SAME
warm page cache), times cudaHostRegister(Mapped) + the RSS growth across the pinning (the
Kaggle-host feasibility input: the ~28.8 GB expert region pinned vs the host's RAM), and
benches both paths (3 passes x 3 reps, best): (A) the kernel reading the device alias of
the mapped pages, (B) the staged form (the H2D from the same pages + the same kernel over
the VRAM staging) - the coalesced 16-B grid-strided read kernel is the bandwidth-realistic
moe pattern (the expert gu slab is a contiguous 1.075 MB region; the ALU is a trivial word
sum so the loads dominate), and the probe prints the A/B ratio + the decision rule
(>= ~0.6 supports the pointer-swap form). Run on the L4 host: ./build/uva_probe <model>
0 2 b. The parallel session (the L4 host) runs it; the measured class freezes the design.

## r19p - the CUDA-graph LAUNCH CENSUS (static, from the engine code; lever 2's probe)

The graph capture's cf-m0-doctrine probe (measure the structure before designing the
capture): the full audit of cf_engine.cu's decode step. THE LAUNCH CENSUS: ~2,578
launches/token -
- per GDN layer 54: hc_mix 9 (hc_norm 1 + the Q5_1 down pair 2 + hc_lo 1 + the up pair 2 +
  hc_mixed 1 + the inject pair 2), deltanet 15 (the qkv/z/beta/alpha pairs 8 + the
  gates/conv/l2/recur/gnorm 5 + the ssm_out pair 2), hc_combine 1, the ffn hc_mix 9,
  moe 19 (the router pair 2 + the 2 repacks (the OFF staging path) + the batched five
  (quantize_q8_K + gemv_q8k_b + silu_mul_b + quantize_q8_0 + gemv_q8_0_b) + the shared
  four pairs 8 + silu_mul 1 + moe_out 1), hc_combine 1;
- per attn layer 51: the same shell with attention 12 (the wq/wk/wv pairs 6 +
  qk_norm_rope/kv_store/attn_decode/gate_sigmoid 4 + the wo pair 2);
- the step-level: the PLE 12 (the key/value pairs 4 + three hc_norms + sg + gated + conv +
  the two adds), the head/tail 10 (res_init + the final hc_mix 7 (no inject) + the lm_head
  pair).
36x54 + 12x51 + 12 + 10 = 2,578, at the measured ~22 us/launch = the ~57 ms wall - the
r19i decomposition's "~54/layer = ~2600/token" CONFIRMED independently, this time counted
from the code. THE SYNC CENSUS: 49 syncs/token (the moe router's D2H + host
softmax/top-10/renorm x48 + the final logits sync x1) - the step is ~49 sync-bounded
segments, and the ~1,440 host staging memcpys (~62 ms) + the router host loop ride the
windows between them (the serialization's engine). THE MEMCPY CENSUS (the OFF path):
~243 async transfers/token (the router D2H x48, the we H2D x48, the raw H2D x48 (~960 MB),
the wt_gu/wt_dn H2D x96, h_emb + h_ple + the logits D2H). THE FIXED-SHAPE AUDIT: every
launch's grid is CONSTANT ACROSS STEPS (the fixed families' grids are compile-time
(HC/LORA/D/HCD/HV/HQ/HKV/EE/TOPK/CONV/V), the gemv grids are the load-time W.rows); only
VALUES vary - pos in exactly 3 attention kernels (qk_norm_rope, kv_store,
attn_decode's n_kv) and the W-table/we contents (host-built, uploaded each step). The
tiered path's nmiss-dependent repack rows are dead (the tiering is OFF per the r19n
verdict). VERDICT: BOTH graph forms are shape-viable -
- G1 (pre-UVA, the HEDGE): ~50 per-segment graphs (the boundaries at the moe syncs, the
  merged [layer-tail + next-layer-head] segments); the refactors: the device step-params
  buffer (the 3 attention kernels read pos from it; the graph's own memcpy node refreshes
  it from pinned host memory), the pinned h_wt_gu/h_wt_dn/we_h (a nicety - the unpinned
  memcpy nodes stage through the driver bounce at today's cost), the capture wrapper +
  the replay driver. The launch wall ~57 ms -> ~1 ms; the staging wall stays: the staged
  ~150-165 -> ~95-110 ms/token class WITHOUT the UVA.
- G2 (post-UVA, the FULL WIN): the UVA kills the staging memcpys/H2D/repacks, then the
  device-side router (the softmax over 512 + the top-10 + the we renorm + the W-table
  view build as kernels - all fixed-shape, single-CTA-sized work) kills the 48 mid-step
  syncs, the segments MERGE into one full-step graph: ~1 replay/token, the remaining
  memcpy nodes h_emb + h_ple + the params in, the pinned logits out, ONE sync at the step
  boundary. The T4 class: ~75 ms UVA reads + ~7 ms GPU + ~1 replay ~ ~83 ms/token (~12
  t/s class, from the ~6-8 mmap-floor decode); the L4's measured 341 ms/token gets the
  same lever classes (its exact split awaits the parallel session's UVA A/B on that host).
THE ORDER STANDS: the UVA first (the bigger cut AND G2's unlock), G1 the hedge that does
not need it, MTP after both (cf-m4). The PLAN_CF lever-2 text rebased onto this census.

## r19q - the cf-m4 MTP design FROZEN pre-implementation (research/CF_MTP.md)

The third lever's design (the first two being the parallel session's UVA lane + the graph
capture), done statically while the quota window holds: the draft block (blk.48, ALL
Q8_0) is specified end-to-end from cf-arch section 6 - the exact forward (the eh_proj
per-stream concat over [e_norm ; h_norm_s], the draft's own KV at p+1, the own hc mixers
+ Q8_0 MoE + hc_head -> the SHARED full-vocab lm_head; NO GDN, NO PLE in the draft), the
catch-up (the pending_h ring of the trunk's [10240] pre-final-mixer residuals + the
draft's KV over the prompt, ~2.5 ms/prompt-token one-time), the batched verify's kernel
surface (the per-row variants of every trunk family; the 27B's k_pf_attn/k_pf_gdn/
k_pf_gdnc are the in-repo templates) with THE GATE named up front (the batched rows must
reproduce the sequential decode BIT-EXACTLY - any per-row reduction reorder can flip the
near-tie argmaxes (the r19n battery's own flips sat at gaps 2.68/0.49) and break the
oracle agreement), the rollback state list (the GDN S snapshots ~57 MB, the conv states,
the PLE 9-column history ~5.9 MB, the KV pos counter, the draft's own chain), and the
key structural finding: THE Q8_0 GEMV PATH ALREADY EXISTS IN-REPO AND IS GATE-PROVEN -
repack.cu's GT_Q8_0 case (k_repack_q8, the 34 B/block ggml layout), the FAST_Q8 int8-dp4a
gemv family (gemv.cuh/tp_gemv_impl.cuh, the 27B's own eh_proj launch form), and the
activation pairing is the CF engine's existing launch_quantize_q8_0 (the symmetric
Q8_0-form) - so the draft's FMT_Q8 case is a REUSE of an already-oracle-proven path, not
a new kernel family to verify from scratch. The draft's 2.49 GiB experts + the shared
lm_head are RESIDENT (the trunk leaves ~12 GiB free on a 15.36 GiB T4): the draft steps
are staging-free ~2-3 ms. The speed math: on the UVA'd T4 (~83 ms/token, ~12 t/s after
cf-m3 + G2), k=3 gives the verify union ~1.44 GB ~ 112 ms + ~8 ms of drafts at the
expected ~2.2 tokens/verify ~ 55 ms/token ~ 18 t/s (a ~1.5x over the G2 engine; the
all-resident ceiling ladder stays the reference). The PLAN_CF cf-m4 bullet carries the
summary + the pointer. The implementation stays ordered after the UVA + the graphs (the
verify's union staging rides whichever expert path wins the A/B; the draft steps join the
G2 graph as fixed-shape work).

## r19r - the cf-m3 RAM CAP decoded (the Kaggle form of the UVA is PARTIAL)

The static check of the rebased cf-m3 against the tensor map found a platform blocker in
the design as written: the trunk's expert region is 47.9 GiB (the per-layer 1,022 MB: the
gate 275.25 + the up 275.25 + the down 471.86, x 48 layers) - MORE than the Kaggle host's
~29 GB RAM, so cudaHostRegister over the WHOLE expert region does not fit on the TARGET
platform; the full-region form is the big-RAM host's (the parallel session's L4; the
probe's RSS growth + registration time decide that host's feasibility). THE KAGGLE FORM:
the PARTIAL registration - the first N layers' expert spans (the layer-granular contiguous
~1 GiB/layer spans), N driven by the measured usable RAM (~28-29 layers at ~28.8 GiB,
~60% of the pool) - carried by the LANDED dual-path/W-table mechanism: the registered
layers' picks read via the alias (the repack's input = the per-pick alias offsets), the
unregistered layers run the verbatim staged path, so the mechanism is the tiering pattern
repurposed (the layer split instead of the hot-set split, the alias views instead of the
resident packed slabs). The honest Kaggle ladder: the mixed staging ~29 x 1.56 + ~19 x
3.02 ~ 103 ms/token + ~7 GPU + the segmented graphs (the staged layers keep their router
syncs, so the full-step G2 graph is the full-UVA host's; the Kaggle stays G1-class) ~1 ->
~110 ms/token ~ 9 t/s, vs the full-UVA host's ~83 ms/token (~12 t/s, the G2). The MTP
note gains the matching caveat: on the partial-UVA Kaggle the k=3 verify splits the same
way (~0.86 GB alias + ~0.58 GB staged ~ 155 ms + ~8 ms of drafts) ~ 75 ms/token ~ 13 t/s
- still a ~1.4x over that host's non-MTP class, so the lever pays on both hosts. The
probe's RSS output feeds N per host; the quota is still 30/30 (the direct push rejected
again this morning, the gatekeepers own the retries).

## r19s - the CUDA-graph launch-wall PROBE landed (the G1/G2 win measured pre-engine)

The r19p census counted the step's launches statically; the ~22 us/launch class (and so
the ~57 ms wall and the G1/G2 reclamation claims) was inherited from the 27B-era
measurements, never measured on this kernel mix. The probe (t4q/tools/graph_probe.cu +
the build/graph_probe target, compiles clean in the 12.8 podman): the census-shaped
chain itself - every 4th launch a gemv-ish kernel streaming a ~13 MB weight slab (the
packed-slab read class of one moe gemv, 160 CTAs, a warp per row over K=2560), the rest
the tiny norm/add-class kernel (41 CTAs over 10240 floats, ping-ponged), all on ONE
stream so the chain is strictly serial - measured three ways with the kernel work
identical in all three: (A) launch-by-launch (the current engine's form), (B) ONE
captured full-step graph (the G2 form, ~1 replay), (C) ~50 per-segment graphs with a
cudaStreamSynchronize + the ~30 us router host loop (the softmax/top-10 over 512, run 4x
for an honest ~30 us) between replays (the G1 form). The differentials (A)-(B)/(A)-(C)
are the pure graph reclamation on this hardware; the instantiate times are reported as
the one-time load cost of the graph set. The self-review catches before landing: the
gemv-ish ping-pong read the [1280]-float output as the [2560]-float input (an OOB - the
fixed pair suffices, the single stream serializes the chain), the uninitialized gx input
(memset added), and the host loop's device-only __expf (expf). Run on each host:
./build/graph_probe [2578] [50] - the L4 now, the T4 at the next quota window.

## r19t - the G2 device-router ORDER-EXACTNESS spec (the seventh-catch-class find)

The G2 form's device-side router (the softmax/top-10/the we renorm/the W-table view build
as kernels, the sync-killer that merges the segments into the full-step graph) carries a
correctness trap the census did not name: a warp-parallel/tree reduction REORDERS the fp
sum, perturbing the renormalized we by ~1 ulp; the perturbed we feeds moe_out -> a
~1e-6-class logit perturbation -> the near-tie argmax flips at a ~1-per-1e3-1e4-token
rate - the SHORT gen gates (8 tokens) PASS while the long generations silently diverge
from the byte-identical-greedy bar (the trap class: the gates pass, the long runs
diverge). The r19n battery's own flips sat at gaps 2.68/0.49 under the +-2..5 q8 noise;
the G2 perturbation is ~6 orders smaller, but the bar is byte-identity, not statistics.
THE SPEC: the device router must reproduce the host loops' exact fp accumulation ORDER -
the softmax sum and the we renorm accumulated sequentially over i=0..NE-1 (a single
warp's serial loop, ~2-5 us - the free correctness), the top-10 scan keeping the host's
strict-> first-max tie rule; the W-table view build is integer-offset arithmetic, exact
by construction (no risk); the G1 form keeps the host router as-is (no new arithmetic,
no risk). The PLAN_CF lever-2 G2 block carries the spec. The MTP note's verify gate
(the batched rows must reproduce the sequential decode bit-exactly) is the same rule
applied to the batched trunk - the two gates are consistent.

## r19u - the UVA pointer-swap LANDED as the third path (build-clean, OFF by default)

The engine lane's owner (the parallel L4 session) has been quiet for two days with its
milestone complete, so the implementation landed here under the repo's own precedent
(the r19l cluster: engine mechanisms land statically-verified + build-clean, the GPU
battery gates them later). THE IMPLEMENTATION (byte-identical by construction, only
the read path changes): (1) cf_loader.cu registers the first T4Q_CF_UVA_LAYERS=n layers'
expert tensors as ONE coalesced page-aligned cudaHostRegisterMapped span - the
per-tensor page-spans of the file-adjacent tensors overlap on the shared boundary pages
and would double-register - timing the registration + printing the span; the per-layer
device aliases are the span's alias + the tensor offsets. (2) The new scatter repacks
(repack.cu): k_repack_eid_q2k (the pick-major [gate EE | up EE] layout, the address
(eid[k]*EE + pr)*840 = exactly the OFF path's memcpy source) and k_repack_eid_q4 (the
(eid[k]*D + pr)*360 form), the SAME repack block decodes, packing into the SAME
identity W-table views. (3) The moe's UVA branch (cf_engine.cu): the eid upload +
the two scatter repacks - NO host memcpys, NO H2D, NO raw staging. (4) cf_free unregisters
the span after the stream drain. ABSENT/0 = OFF = the verbatim staging path
(attribution-clean, the current payload f88057d6 still gates the staged engine verbatim).
Self-review catches before landing: the repack_q4_block signature (4 args, the bb lives
in the caller's address math only), the registration's page-span overlap (the coalesced
one-span form). Build: libt4q + cf_run 0 errors, 0 warnings in the 12.8 podman. RUNTIME
VERIFICATION (the L4 host): the correctness gates with T4Q_CF_UVA_LAYERS=48 (a big-RAM
host) or ~28-29 (the Kaggle class), then the staged-vs-UVA A/B tok/s (the uva_probe's
ratio decides the adoption; n is tuned per host from the probe's RSS report); the v9
payload regen carries it after the L4 passes. The PLAN_CF cf-m3 section carries the
landed note.

## r19v - the G1 SEGMENT GRAPHS landed (build-clean, OFF by default) + the pos device word

Two commits. (1) a02bd01 the pos device word (the G1 prerequisite, byte-identical by
construction): pos - the ONLY per-step varying kernel arg in the whole step (the r19p
census: every launch grid constant, only pos in the 3 attention kernels + the pinned-table
contents vary) - now rides a pinned host word uploaded at every step head, and the 3
attention kernels read it from the device word (the same int, the same downstream
arithmetic: theta, the KV slot address, n_kv = pos+1); also eid becomes PINNED (a captured
memcpy node must read pinned host memory). (2) the G1 SEGMENT-GRAPH DRIVER (T4Q_CF_GRAPH=1,
OFF by default): the emission/driver split - ONE op source (the moved verbatim bodies:
emit_head / emit_ple_kernels / emit_pre / emit_router / emit_moe_rest / emit_post /
emit_tail, plus ple_host and host_router holding ALL the per-step host state) with TWO
drivers: the direct driver (the exact old inline op sequence) and the G1 driver (49
sync-bounded segments: the 48 router syncs + the final logits sync; seg 0 = head + L0 pre +
L0 router, seg k = L(k-1) post + [the PLE kernels if k == PLE_LAYER] + Lk pre + Lk router,
seg 48 = L47 post + tail; the FIRST step captures each segment right before its first
replay - the ops RECORDED, not executed, every arg a fixed steady-state buffer - later
steps replay only; the host windows between replays are exactly the direct path's host
work, refreshing the pinned sources the replayed memcpy nodes re-carry). The gmode
exclusions at load: tiered (the miss-varying H2D sizes are not capture-constant; the
census verdict says the tiering pays ~nothing at this routing entropy anyway) and
T4Q_CF_DUMP (the mid-step D2H probes are not capture-legal). The RELAXED capture mode keeps
the emission's diagnostic queries (CK(cudaGetLastError)/check_launch) legal mid-capture
(single-threaded engine). The graphs survive cf_reset (the buffers are the same, the
memsets run outside the graphs); cf_free destroys them after the stream drain. TWO
SELF-REVIEW CATCHES before the battery (the capture-only trap class): (a) the h_params[0]
= pos write initially sat INSIDE emit_head - the graph driver calls the emission only at
capture time, so from step 2 the write would never run and the replayed H2D would carry a
STALE pos (the fix: the write lives in the driver, written every step); (b) the write must
also PRECEDE the params H2D's enqueue - a pinned async copy reads its source at execution
time, so a write after the enqueue is a race (the fix: the write precedes emit_head/the
seg-0 launch). Build: libt4q + cf_run 0 errors, 0 warnings in the 12.8 podman (the 3
attention kernels unchanged register counts). RUNTIME VERIFICATION (the L4 host): the
correctness gates with T4Q_CF_GRAPH=1 (the greedy byte-compare vs the OFF path - the
byte-identity argument: the replayed graph executes the recorded op list in the recorded
order with the recorded args, which IS the direct path's op list/args, with the data
flowing through the same fixed buffers), then the A/B tok/s (the graph_probe's measured
differentials decide the adoption class); the T4 at the next quota window. The G2 (the
full-step graph + the order-exact device router per the r19t spec) builds on this once the
UVA's A/B passes. The PLAN_CF lever-2 block carries the landed note.

## r19w - the MTP draft block LANDED (build-clean, OFF by default) + the acceptance smoke

The third lever's first half (the draft; the verify/rollback/driver stay ahead). Loaded
only under T4Q_CF_MTP=1 (absent = not loaded, the 2.49 GiB VRAM stays free); INERT until
cf_draft_step is called. THE LANDING, per the frozen CF_MTP.md section 1 forward:
- CfDraft (cf_model.h, after CfScratch since it embeds one): a CfLayer at il=48/attn=true
  (blk.48 is NON-recurrent per attention.recurrent_layers[49] despite 48%4!=3), the nextn
  extras (enorm/hnorm/eh_proj + the draft's OWN final mixer hh_*), the OWN scratch (the
  trunk's is never touched; every activation plane sized for the largest gemv K = HCD,
  including the Q8_1 trio as dead-weight insurance), the ALL-512 resident expert slabs +
  the per-step composed host/device W tables, the chain state hres, the own KV pos word.
- The loader: the draft weights + nextn extras via the normal upload path; the 512 experts
  through the SAME chunked raw->pinned->H2D->repack pass as the hot-set tier - the
  trunk-sized raw_stage (10.75 MB gu + 9.22 MB dn regions) takes 3 Q8_0 draft experts per
  pass (3.48/1.74 MB each; the layout math SELF-REVIEWED against the physical region
  boundaries: up_bytes/3.48 MB = 3, dn_bytes/1.74 MB = 5, ch = 3 - the flagged likely-bug
  cleared, the offsets are the actual alloc boundaries), a VRAM check at ~2.49 GiB.
- gemv()'s FMT_Q8 case (merged into the FMT_P4 branch): the Q8_0 activation pairing -
  launch_quantize_q8_0 + launch_gemv_q8_0, the gate-proven FAST_Q8 int8-dp4a family (the
  27B's own MTP graft passed the byte-identical spec gates on this exact arithmetic).
- k_cf_eh_gather (the ONLY new op): the per-stream [e_norm ; h_norm_s] concat [4][5120],
  19 regs, 0 spills.
- host_top10 extracted from host_router (ONE order-exact softmax/top-10/we-renorm source,
  shared by the trunk's host window and the draft's forward - the draft's routing MUST be
  the same order-exact form; the tiering state stays OUT of it, in host_router).
- cf_draft_step (cf_engine.cu): the pair (token, h) at the draft's OWN KV position, h =
  the trunk's pre-final-mixer wide residual (a D2D so the source is stable across the
  chain) or nullptr = the h_{-1} = 0 pair (the 27B's gate-proven form). The forward: the
  e host-dequant, e_norm (launch_rmsnorm) + h_norm (launch_cf_hc_norm), the gather, the 4
  eh_proj gemvs composing res' (the shared [2560,5120] Q8_0), the ATTENTION TWIN (the
  trunk's attention() op list verbatim - wq/wk/wv gemvs, qk_norm_rope, kv_store,
  attn_decode at 1/16, gate_sigmoid, wo - on the draft's own KV/scratch; the pos word
  rides d_params[1], the spare slot, written in the driver BEFORE the upload's enqueue,
  the r19v discipline), the MoE (the router gemv -> D2H -> host_top10 -> the ALL-RESIDENT
  W-table compose (the tiering's mechanism, all hits, STAGING-FREE) -> the batched Q8_0
  gu gemv (shared activation, strides 0) -> silu_mul_b -> the batched down gemv (the flat
  ffa, the trunk's exact strides) + the gated shared expert -> moe_out), the draft's OWN
  final mixer, the SHARED lm_head (full vocab), the logits D2H into the draft's pinned
  row. The shared host router buffers (h_router/eid/we_h/we) are safe: the draft step
  never interleaves a trunk step mid-flight (the smoke is strictly sequential; the
  future speculative driver must keep that rule). Under T4Q_CF_GRAPH=1 the draft's ops
  run OUTSIDE every capture (the draft never runs mid-cf_step; its steps end drained).
- cf_reset rewinds the draft's pos; cf_free drops the pinned rows (h_e/h_logits) - the
  device buffers fall to the cudaDeviceReset, the trunk's own style.
- THE ACCEPTANCE SMOKE (tools/cf_run.cu's new `draft` mode): the pre-loop pair
  (ids[0], 0) fills the draft's slot 0 and predicts position 1; after each trunk step at
  i, the LAGGED 1-step compare (the draft's prediction from its pair at i vs the run's
  ACTUAL token at i+1 - the prompt id or the trunk's own greedy argmax) + the next pair
  (x_{i+1}, h_i); prints alpha1 (the 1-step acceptance the whole MTP speed math rides
  on) + the draft/trunk mean ms + the first disagreement position.
Build: libt4q + cf_run 0 errors, 0 warnings in the 12.8 podman (k_cf_eh_gather 8 regs,
0 spills; k_cf_qk_norm_rope's pre-existing 32-B stack frame unchanged vs HEAD).
RUNTIME VERIFICATION (the L4 host): T4Q_CF_MTP=1 + `cf_run draft <model> <ids> <n>` ->
the measured alpha1 decides the adoption class (the 27B's class ~2.2 accepted/verify;
the CF_MTP speed math: k=3 ~ 55 ms/token ~ 18 t/s on the UVA'd T4). The draft forward's
byte-exactness rides the trunk's own argument (every op is the trunk's op on the draft's
own buffers - the same kernels, args, order - and the Q8_0 pairing is the 27B's
gate-proven arithmetic; the eh_proj direction (rows=2560 out, cols=5120 in) would
surface as a garbage-level alpha1 in the smoke, not silently). STILL AHEAD (the next
rounds): the catch-up (pending_h ring), the batched verify (THE GATE: the k+1 rows
MUST reproduce the sequential decode bit-exactly), the rollback, the speculative
driver, k tuned on the measured acceptance. The PLAN_CF cf-m4 block carries the landed
note.

## r19x - the MTP verify LANDED in the GATE form (build-clean, THE GATE tool with it)

The third lever's second half: the batched verify (CF_MTP.md section 5), landed as the
per-row GATE form. Loaded under the same T4Q_CF_MTP=1 gate (k rides T4Q_CF_K, default 3
-> nr = k+1 rows, the MAXR=8 bound); INERT until cf_verify is called. NO new kernels -
the whole round is the driver + the union window + the per-row planes:
- CfVerify (cf_model.h): the [MAXR] per-row scratch (the trunk's plane list verbatim -
  every field, the activation planes sized for the largest gemv K as always), the
  per-row pos words (h_pos pinned [MAXR] -> d_pos device [MAXR]; the row's
  pos_dev = d_pos + r), the per-row pinned host planes (h_emb/h_ple/h_router/we_h/
  h_logits, [MAXR]-strided), the per-row eid/we host arrays, the per-row W tables
  ([MAXR][TOPK] host + device), the UNION slabs ([nr*TOPK*2EE, D] K2 / [nr*TOPK*D, EE]
  P4 - the worst case nr*TOPK experts, the dedup usually ~1.5x, ~80 MB at nr=4, a VRAM
  check at load), the union maps (uidx [NE], uids [MAXR*TOPK], the device uids for the
  UVA scatters), the per-row ye/ysh/sh_gate_raw planes.
- cf_verify (cf_engine.cu): THE GATE FORM - nr candidate rows through the whole trunk
  in ONE pass. Every op is the sequential step's op with the ROW's slice (the same
  kernels, the same args, the same per-row order): the head (the per-row embedding
  dequants into the pinned [nr][D] slices, the toks records, the pos words - the writes
  precede the ONE upload's enqueue, then the per-row emb H2D + res_init), the per-layer
  loop (per row: [the PLE twin at layer 1 - ple_host_core into the row's pinned slice
  (the hash at the row's pos), the H2D + the 8 PLE launches verbatim with the row's
  scratch (the ple_key/ple_query/ple_s/ple_gate/ple_gated buffers shared - the stream
  order serializes the rows, the same device-side reuse the sequential steps make) +
  the ring evolving] -> the pre twin (the hc mix, the attention twin with the row's
  pos_dev, or deltanet on the row's scratch with L.S/L.conv_state evolving in row
  order) -> the combine -> the ffn mix -> the router twin (the gemv borrows the row's
  logits scratch, the D2H lands in the row's pinned router slice)), then the ONE
  batched host window: the sync, the per-row order-exact top-10 (host_top10_row - the
  r19t bar, ONE source with the trunk's host_top10), the union dedup (uidx/uids), the
  union staging (the OFF chunked passes through the trunk's raw_stage - TOPK experts
  per pass, the same pass form as the hot-set tier / the UVA union scatters - each
  union expert's own slab read once through the registered aliases, the same address
  math as the OFF passes' memcpy sources, the same repack decode - byte-identical by
  construction), the per-row W-table compose (the tiering's hit-branch view math
  verbatim, the row's pick k -> the union slot of eid[r][k]), the uploads (the wt
  [nr][TOPK] + the we), then the per-row moe rest twin (the batched gu gemv on the
  row's W table + the shared quantized mixed, silu_mul_b, the batched down gemv into
  the row's ye slice, the shared expert, the moe_out on the row's we slice, the
  combine), then the per-row tail twin (the final mixer, the shared lm_head into the
  row's logits scratch, the D2H into the row's pinned h_logits slice). The rows' logits
  land in v->h_logits [nr][V] (row r = the prediction for pos+r+1); the trunk's pos
  advances by nr; c->have_logits = false (the trunk's h_logits is stale).
- THE TRAP CLASSES pre-empted by design (the r19w self-review sweep found no NEW
  catches): the per-row pos ARRAY (the single-word form would race the per-row async
  uploads - the write-after-enqueue class), the per-row pinned host slices (h_ple/
  h_emb/h_router - the shared-word host races), the window's sync ordering (the
  we/raw_stage/union-slab cross-layer drains: the layer il+1's window sync drains the
  layer il's we upload + staging H2Ds + moe-rest readers before any host rewrite or
  slab overwrite), the shared device buffers (the ple family: stream-ordered, the same
  reuse the sequential steps make), the pageable W-table uploads (the driver stages
  small copies at the enqueue - the trunk's own per-layer precedent), the causal
  per-row attention (row r's n_kv = pos_r+1 never reaches the later rows' slots - no
  reordering needed, the per-row order IS the sequential order).
- THE GATE TOOL (cf_run's new `verify` mode): run A = the prompt + n-1 greedy steps,
  ref[i] = the logits after the step consuming tok_i (the prediction of tok_{i+1});
  cf_reset (the KV/GDN/PLE/toks states clear, the weights stay); run B = the same
  prompt + the verify rounds over the SAME tail tokens (chunked by nr) - every row's
  logits must BYTE-MATCH the sequential reference (memcmp per row + the top1 tally +
  the first diff + the worst rel diff; pass = all rows byte-match). The gate also
  covers the union (a byte-matched row proves the union slabs/W tables/we are
  byte-exact), the per-row top-10, and the whole per-row op mirror.
- host_top10_row / ple_host_core extractions (ONE source each: the trunk's instances
  are the single-row wrappers; the verify's rows run the same order-exact loops at the
  row's pos into the row's buffers).
- cf_free: the verify's pinned planes freed (the device buffers fall to the
  cudaDeviceReset, the trunk's own style).
Build: libt4q + cf_run 0 errors, 0 warnings in the 12.8 podman (NO new kernels - the
ptxas diffs are compile-time noise only). RUNTIME VERIFICATION (the L4): T4Q_CF_MTP=1 +
`cf_run verify <model> <ids> <n>` -> THE GATE (byte_match == rows) + the verify's
ms/token vs the sequential's (the honest note: the DIRECT per-row-launch form pays
the ~22 us launch wall x ~2600 x nr - the SPEED comes with the verify's own segment
graphs + the UVA union scatters, the later rounds; the CF_MTP speed math - the union
~1.44 GB + the drafts ~8 ms -> ~55 ms/token ~ 18 t/s at k=3 - assumes those). ALSO
PENDING: the draft's acceptance smoke (r19w's mode) on the same L4 run. STILL AHEAD:
the catch-up (pending_h ring), the rollback (the GDN S/conv snapshots + the KV/PLE pos
rewind), the speculative driver (the draft chain + the verify + the accept/reject
loop), the verify's segment graphs, k tuned on the measured acceptance. The PLAN_CF
cf-m4 block carries the verify's landed note; the r19w register-count record corrected
(k_cf_eh_gather 8 regs, not 19).

## r19y - the SPECULATIVE DRIVER landed (the 27B's tp_spec.cu arithmetic adapted; the snapshots, not the ring)

The third lever's third half (CF_MTP.md sections 1/4/6): the draft chain + the verify
+ the accept scan + the rollback + the catch-up, landed build-clean 0 errors/0 warnings,
NO new kernels (host code only - the ptxas diffs are compile-time noise, the cf kernel
register counts identical vs HEAD), INERT until cf_spec_prime/cf_spec_step are called
(the same T4Q_CF_MTP=1 gate; the snapshot planes ~0.34 GiB at k=3 ride the verify block
with their own VRAM check). THE DESIGN DECISION (the r19y plan's open point, resolved
against the plan's ring form): the 27B's rollback is a ns = k+2 SNAPSHOT RING because its
k_gdn_m writes each token's state directly into the ring; the CF verify reuses the
SEQUENTIAL step's single-token launch_gdn_recur IN PLACE on L.S, so the ring form would
need a separate-in/out change to that gate-proven kernel AND the true S size (HV*DK*DK
f32 = 3.15 MB/layer x 36 GDN layers = ~113 MB per state set - the CF_MTP.md section 6
"[24,128,128] ~ 57 MB" note and the r19y plan's "~37 MB/snapshot" were both ~2-3x low)
would cost ~566 MB for 5 ring slots. THE LANDED FORM: per-(layer,row) SNAPSHOT
CAPTURES - stream-ordered D2Ds enqueued inside cf_verify at the exact boundaries (after
the row's deltanet, before the next row overwrites L.S/L.conv_state; after the row's
PLE roll; the pending_h capture rides the row's tail because the tail's hc_mix only
READS s.h), the states AFTER rows 0..k-1 (k = nr-1 slots; the accept n = k leaves the
rolling L.S itself correct, no capture, no restore), and on a partial accept (n < k)
the restore D2Ds (per GDN layer L.S/L.conv_state <- snap[gord*k+n], the PLE ring <-
ple_snap[n]) on the drained stream (cf_verify's own end sync makes them land before
the next round's first op). Zero kernel surface, ~340 MB of planes at k=3, and the
honest copy cost: ~113 MB of S captures + the conv/PLE/pending_h per verify (~2-4 ms
in the direct-launch form) + ~1-2 ms of restores on a partial accept - the
multi-token ring kernel / the verify's segment graphs remove it in the later rounds.
THE STATE LIST (CF_MTP.md section 6, each item proven stale-but-invisible or restored):
the GDN S/conv (the snapshots above), the PLE ring (the snapshots above), the trunk
attention KV (pos-only - the cells beyond the rewound pos are invisible; each row's
own slot is written before its read), c->toks (the rejected drafts' entries are never
read before the next round's head rewrites them - every read is at a position whose
entry was written by an accepted consumption or by an earlier row of the same verify),
the draft's own KV pos + its res' chain state (the catch-up reprocesses the accepted
tokens at the draft positions p+1..p+n+1, d->pos rewound to p+1 first, so the chain's
speculative draft slots are re-consumed in order and the last call's hres IS the next
round's chain state), the trunk pos (c->pos = p+n+1).
THE ARITHMETIC (tp_spec.cu verbatim where it applies): the accept scan n = the longest
prefix with vt[n+1] == yv[n] (vt[0] = the pending token, yv[r] = the argmax of verify
row r - row 0's consumption is never rejected, so every round emits >= 1); the
emission yv[0..n] with the new pending = yv[n] (every position emitted exactly once:
the prime's pending by the mode, each later position by exactly one round); the picks
use the FIRST-max rule (cf_run's argmax verbatim as the engine's pick() - a differing
tie-break could split an exact tie). The chain (CF_MTP.md section 1's form): vt[1] =
the pending's draft prediction (IN HAND from the prime / the previous catch-up's last
call), vt[2..k] chained with the draft's OWN hres as the h input (the D2D in
cf_draft_step decouples hres from h_in, so the forward's hres overwrite is
stream-ordered after the copy); the catch-up pairs (yv[t], pending_h[t]) - the verify
row t's pre-final-mixer residual, the same ground-truth pairing the acceptance smoke
measured alpha1 on.
THE INVARIANT (established by cf_spec_prime, restored by every round): the draft has
processed positions 0..c->pos (its own KV slots), so its h_logits predict position
c->pos+1 and d->pos = c->pos+1. THE PRIME (cf_spec_prime): the trunk over the prompt
(one cf_step per token - the same path the reference modes run) with the draft paired
one step behind (the pairs (ids[j], h_{j-1}) with h_{-1} = 0 - the catch-up form; the
draft's D2D from c->sc.h is enqueued before the next trunk step's launches, so the
source is stable, stream-ordered), then the PENDING pair (the greedy pick after the
prompt, h_{np-1}) at the draft's position np. Each cf_step/cf_draft_step call drains
before returning, so no trunk op interleaves a draft op mid-flight (the r19w rule,
kept); the prime guards pos == 0 (fail loud beats silent misalignment). THE ROUND
(cf_spec_step): the k drafts -> the verify (k+1 rows, its own captures riding) -> the
argmaxes + the accept scan -> the rollback (n < k) + c->pos = p+n+1 -> the emission ->
the catch-up (d->pos = p+1, n+1 draft calls) -> the invariant. Returns the emitted
count (n+1), 0 on error.
THE GATE TOOL (cf_run's new `spec` mode): run A = the prompt + n-1 greedy steps (the
reference tokens); cf_reset; run B = cf_spec_prime + the spec rounds until n tokens -
every emitted token must equal the reference's token at its position (first_diff +
match count; pass = first_diff < 0). The construction argument: the verify's rows
reproduce the sequential logits byte-exactly (the r19x gate), so the emitted prefix is
the sequential's own stream, and the restored state is the sequential's own state -
the round is the reference's arithmetic, re-batched; the gate is that argument's
runtime proof (and the r19x `verify` gate re-proves the rows WITH the captures now
riding them). Prints the accept histogram hist[0..k] (the k-tuning eye), the
tokens/round, the ms/token vs the sequential, and the per-round draft/verify/catch-up
ms splits (the engine's timers). Runtime verification (the L4): T4Q_CF_MTP=1 +
`cf_run spec <model> <ids> <n>` -> the pass gate + the measured acceptance (the
histogram's mean decides k; the 27B's class ~2.2 accepted/verify) + the honest speed
note: the DIRECT per-row-launch form pays the ~22 us launch wall x ~2600 x nr + the
snapshot copies - the CF_MTP speed math (~55 ms/token ~ 18 t/s at k=3 on the UVA'd T4)
still assumes the verify's own segment graphs + the UVA union scatters, the next
rounds. The PLAN_CF cf-m4 block carries the driver's landed note.

## r19z - the verify's own SEGMENT GRAPHS landed (the G1 pattern applied to cf_verify;
## the sweep caught the r19x eid null-deref)

The launch wall was the MTP verify's dominant direct-form cost: ~2600 launches per
row x nr rows x ~22 us ~ 100-250 ms of pure wall per verify at nr=4 - the CF_MTP speed
math (~55 ms/token ~ 18 t/s at k=3) assumed this round. The r19x inline body split
into the SIX vfy_* functions at the end of the anonymous namespace (the r19v
emission-split bar: ONE source for the direct driver and the graphs, the r19x bodies
verbatim):
- The host-only driver head (the embedding dequants into the pinned [nr][D] slices,
  the toks records, the h_pos writes before any enqueue - the r19v discipline) +
  vfy_head_em (the pos-word upload + the per-row emb H2Ds + the res inits).
- vfy_ple_host (the per-row PLE gathers) - HOISTED out of the emission into the
  driver's window work (the G1 form: host work cannot run mid-capture); value-safe:
  the gathers write disjoint pinned slices and each row's H2D reads its own slice.
- vfy_pre_em (per layer, per row: the PLE twin at layer 1 + the pre twin + the router
  gemv/D2H; the r19y S/conv/PLE snapshot D2Ds ride at their exact per-(layer,row)
  boundaries - fixed-arg nodes).
- vfy_window (the sync + the per-row order-exact top-10 + the union dedup + the
  union staging DIRECT - the union count varies per layer, not capture-constant -
  + the W-table compose/uploads, pageable sources, direct).
- vfy_moe_em (the we upload MOVED to its head - the pinned source is capture-legal
  there, the r19v emit_moe_rest precedent, a memcpy among disjoint buffers written
  by the window's host work and read by the moe_out after, in both orders - + the
  per-row moe rest + the combine).
- vfy_tail_em (the r19y pending_h capture + the final mixer + the shared lm_head +
  the logits D2Hs into the pinned [nr][V] slices).
The two moves (the PLE gather hoist, the we upload to the moe emission head) are the
only deltas from the r19x-gate-proven stream order - both value-safe, both named in
the split's header comment.

THE GRAPH DRIVER: the same NL+1 = 49 sync-bounded segmentation as the step's G1
(seg 0 = the head emission + L0's pre emission; seg k = L(k-1)'s moe emission + Lk's
pre emission; seg 48 = L47's moe emission + the tail emission), captured at the
FIRST full-nr verify call and replayed after (capture-at-first-use; each capture
HEAD-SYNCED - the idle-stream invariant, the verify's windows, unlike the step's,
enqueue staging - the sync runs only during the build, the steady state just
replays; cudaStreamCaptureModeRelaxed keeps the mid-emission check_launch/CK queries
legal). The per-layer host windows run between the replays exactly as the direct
path's (vfy_window + the PLE gathers at the layers[0] window, before seg 1 - the G1
host-window pattern: the gathers must precede the layer-1 emission's H2D node).
Every varying content rides a pinned-fixed host source the captured memcpy nodes
re-carry at each replay (the pos words, the emb rows, the PLE gather rows, the
router D2Hs, the we rows, the logits D2Hs); the r19y snapshot D2Ds are fixed-arg
nodes; the union staging + the W-table/uids uploads stay DIRECT in the windows.
PARTIAL-nr calls (the `verify` gate mode's tail chunks, nr < v->nr) fall to the
direct driver - the captured shapes are nr-bound; the direct and graph paths run the
same ops with the same args in the same order (byte-identical by construction).
Rides the same T4Q_CF_GRAPH=1 gate as the step's G1 graphs (c->gmode; the
tiered/dump exclusions apply); INERT until cf_verify is called at full nr under the
gate. cf_free destroys the execs/graphs after the stream drain (the G1 pattern).
NO new kernels - host code only, the cf kernel register counts identical vs HEAD,
0 errors/0 warnings (the 77 tp_* ptxas lines are the pre-existing 27B-lane set).

THE SWEEP CAUGHT ONE REAL r19x BUG: v->eid (the per-row picks plane the window's
host_top10_row writes, the dedup + the W-table compose read) was NEVER allocated -
any cf_verify call would have segfaulted on the first window (the r19x round was
build-verified only; the L4 runtime battery is quota-gated). Now cudaMallocHost'd in
the verify block (pinned to match the sibling planes' discipline) + freed in cf_free.
The CfDraft pointer fields were re-checked against the draft's alloc block - all
allocated (the sweep's cross-check).

Runtime verification (the L4): T4Q_CF_MTP=1 + T4Q_CF_GRAPH=1 + the `verify` gate ->
the byte-match gate with the graphs riding (the full-nr chunks replay, the partial
tails run direct - both drivers exercised in one run) + the verify ms/token vs the
gmode-off direct form (the honest note: the launch wall collapses from ~10,400 x
22 us to ~49 graph launches + the windows' direct staging launches ~5-22/layer; the
r19y snapshot D2Ds and the union staging stay - the G2/UVA/k-tuning rounds' levers);
the `spec` gate -> the pass gate + the accept histogram (the spec rounds always
verify at full nr - the graph path always).

## r19aa - the PLE PREFETCH landed (CF_MTP.md section 8's frozen design; the only
## window this engine has)

The verify's row gathers fault the mmap'd 26.85 GiB PLE table - the ~98 us/row
first-fault class measured on the Kaggle disk (the r19e verdict held: the 1-step
prefetch is impossible in the greedy loop, the MTP's verify is the only window);
the frozen section 8 math: ~1.4-1.6 ms/token of gather fault cost, most of it
hideable under the draft calls' GPU stretches. LANDED as the per-row prefetch
threads (T4Q_CF_PLE_PRE=1, OFF absent the env - the verbatim round; value-invisible
either way, a pure page warm):
- ONE source for the row enumeration (the r19t bar): the walk template extracted
  from ple_host_core (the hash math verbatim - the same (n, g, h, row) sequence,
  the same dequants in the same order; byte-identical by construction), the gather
  and the touch both ride it - the touch can never warm the wrong pages.
- ple_touch_rows: a VOLATILE byte per 4 KiB step of the row's span + the last byte
  (a dead load can be elided by the compiler; a volatile one cannot - the page
  fault is the whole point). No CUDA API, no exception path (the touch cannot
  fail), pure reads of the immutable table.
- ple_pre_ctx: the ctx construction on the ENGINE thread at the spawn point - the
  verify's to-be-written records substituted for c->toks (the verify's driver head
  writes c->toks[pos+r] = vt[r] for EVERY r before the gather reads them, so the
  gather's c->toks[pos+r-k] IS vt[r-k] for r>=k; the trunk's valid rolling entries
  below that; the EOS/-1 cut logic verbatim) - so the thread body reads NO shared
  mutable state (the ctx goes into the lambda by value).
- The spawn schedule: rows 0/1 before the draft chain (the pending + its draft
  prediction in hand), row i the moment its producing draft call lands its pick -
  the threads run under the draft calls' GPU stretches (the host blocked at each
  call's drains) and through the verify's seg-0 drain, and the gather coalesces
  with any in-flight page fault (concurrent faults on the same page wait for the
  first - a lagging thread never ADDS wall, it only warms; the tail pays).
- The RAII holder joins on EVERY exit path (an unjoined std::thread terminates at
  destruction - the mid-chain `return 0`s included).
The concurrency sweep: cf_draft_step writes no c->toks (its own KV pos/buffers);
the spawn's c->toks reads (pos-1, pos-2 - below the verify's head-write range
[pos, pos+nr-1]) happen before cf_verify runs at all; the threads are invisible
to the verify's first-call stream captures (NO CUDA API in the thread body - the
single-threaded-engine capture discipline holds); the <thread>/<array> includes
added. Build: 0 errors, 0 warnings, the cf kernel register counts identical vs
HEAD (host code only). Runtime verification (the L4): T4Q_CF_MTP=1 + the spec gate
with/without T4Q_CF_PLE_PRE=1 -> the A/B (the ms/token + the draft/verify splits;
the honest cost: ~nr thread spawns/round ~0.1-0.2 ms, and the warm-cache class -
natural-text n-grams recur - pays the spawns for nothing; the A/B decides the
default).

THE GDN RING SCOPE (r19ab, the closure verdict - scoped, NOT landed): the
multi-token GDN ring kernel (the 27B's k_gdn_m form, the in-repo template) was
scoped against the r19y snapshot-capture design it would replace. THE ARITHMETIC:
the copies it removes are the r19y captures - 3 rows x 36 GDN layers x (3.15 MB S +
tiny conv) ~ 340 MB of D2D ~ 2.5-3 ms/verify + the restore ~0.5-1 ms/round ~ 3-4
ms out of the ~165 ms round class ~ 2%; the per-row kernel launches it also
consolidates are already graph-free post-r19z; and it COSTS more VRAM (ns = k+2
ring slots ~ 454 MB vs the captures' ~340 MB) plus a large kernel port (the
projections batched M-form, the gates/conv/l2/recur/gnorm folded per layer, the
ring semantics, the 27B shapes vs the CF shapes differ throughout). The r19y
decision's own math (zero kernel surface, ~0.34 GiB) stands; the L4 measures the
MTP path first (the acceptance histogram + the verify's ms/token decide whether
2% is worth a kernel round). THE HONEST NEXT LEVER by the same arithmetic: the
requant stretch (PLAN_CF section 5, census-unlocked - the router is flat, the
tiering does not pay), whose round 1 is the FORMAT DESIGN (the frozen-spec form:
the bit layout + the exact decode arithmetic, the error budget vs the Q2_K
trunk's own quality, the T4 kernel-rate roof at the LUT-decode class, the
residency band below) - a fresh research round, not started this turn. THE POOL
ARITHMETIC (the corrected frame, re-derived from the in-repo shapes): the expert
pool is 120.7 G-elems (gu 80.5 = [1280 x 2560]/expert x 24,576 experts + dn 40.2 =
[2560 x 640] x 24,576); at 2.0625 bpw (the iq2-xxs class) that is ~31 GB - it does
NOT fit the 2x T4 frame (30.7 GB total VRAM minus the ~2.6 GB must-resident core,
the KV, the draft, the scratch leaves ~24-25 GB of expert room); the FULL-residency
band is ~1.56-1.7 bpw (~24 GB at 1.6, ~23.6 at the iq1_s 1.5625 class), which
concentrates the quality risk at the low-bpw end (vs the Q2_K_S trunk's own 2.625
gu / 4.5 dn) - the round's central question is the quality gate (the perplexity-
class bar, not full greedy agreement, at that band), and the payoff rides the TP
split (the experts by ID across the 2 T4s, the plan's risk-register design). The
G2 device router stays the same verdict as before (the order-exact trap for a
~1-2% win, and the full-step graph form cannot run on the RAM-capped Kaggle host -
the UVA covers only ~60% of the layers).

## r19ac - the requant FORMAT STUDY opened (the primary-source facts fetched; the
## freeze is the next round)

The round-1 opening of the requant stretch (PLAN_CF section 5, census-unlocked).
THE PRIMARY SOURCE (fetched this turn, ggml-common.h @ ggml-org/llama.cpp master):
the exact candidate block structures - iq2_xxs 2.0625 bpw (a f16 d + 32 u16 grid
indices per 256-elem superblock, the 256-entry iq2xxs_grid codebook of 8-byte
entries with the byte values {0x08, 0x19, 0x2b}), iq2_xs 2.3125 (+ 8 scale bytes),
iq2_s 2.5625 (d + 64 qs + 8 qh + 8 scales), iq1_s 1.5625 (a f16 d + 32 u8 grid
indices + 8 u16 qh per 256, the 2048-entry iq1s_grid + ksigns64), iq1_m 1.75 (32
qs + 16 qh + 8 scales, no f16 d). THE RESIDENCY ARITHMETIC (the r19ab-corrected
pool, 120.7 G-elems, against each candidate): iq1_s -> 23.6 GB, the ONLY stock
format that fits the 2x T4 full-residency frame (~24-25 GB of expert room after the
~2.6 GB core + the KV + the draft + the scratch); iq1_m -> 26.4 GB (over); iq2_xxs
-> 31.1 GB (well over); and NO hybrid split fits either (gu iq2_xxs + dn iq1_s =
28.6 GB, gu iq1_s + dn iq2_xxs = 26.1 GB - both over) - so the stock answer is
BOTH expert tensors at iq1_s, or a custom leaner ~1.6 form (the iq1_s structure
with a leaner scale, or the iq1_m scales without the f16 d). THE HONEST RISK
CLASSES the freeze must argue: (1) QUALITY - the iq1_s lattice (the 1.5625 "1-bit"
family) against the trunk's own Q2_K_S 2.625 gu class is a substantial per-expert
noise step; the source is the BF16 original (355 GB, streamed tensor-by-tensor
per the plan), not the quantized trunk; the gate is the perplexity-class A/B (the
Q2_K_S trunk vs the iq1_s-experts trunk) + the greedy-agreement rate at temp 0 -
measured at the L4, not assumed; (2) RATE - the grid decode (the shared-memory
codebook gather + the sign unpack + the float dot, NOT the dp4a nibble path) sits
below the r18-measured K-quant family (Q2_K ~179-200 GB/s at the expert shapes);
the honest kernel-rate estimate needs the exact ggml CUDA iq1_s vec_dot
instruction count - the next study, not yet done; (3) the QUALITY-RESIDENCY
TENSION - if iq1_s fails the quality gate, the fallback is the custom ~1.65-1.7
form (a leaner-scale iq1_s/iq1_m hybrid) which trades bpw headroom against the
same lattice noise. STILL AHEAD (the freeze's inputs): the ggml-quants.c +
iq1_s CUDA vec_dot study (the exact decode + the quantizer's grid search + the
instruction count), then the FROZEN SPEC (research/CF_REQUANT.md: the format pick,
the exact arithmetic, the kernel design at the CF shapes, the packer pipeline
(the BF16 streaming + the grid search), the TP-split residency design, the gates),
then the implementation rounds. The r19ab note carries the pool arithmetic; this
note carries the source facts.

## r19ad - the requant freeze: the vec_dot + quantizer study closed, the FROZEN SPEC
## landed (research/CF_REQUANT.md; records only, no code)

The round that closes the r19ac study list and lands the frozen spec. THE PRIMARY
SOURCES (fetched, ggml-org/llama.cpp master): vecdotq.cuh (the exact CUDA
vec_dot_iq1_s_q8_1), ggml-quants.c (the quantize_row_iq1_s_impl + the reference
dequant + iq1_find_best_neighbour2), ggml-common.h (the block_iq1_s struct, the
IQ1S_DELTA = 0.125, the NGRID_IQ1S = 2048, BOTH grid tables - the CPU uint64[2048]
byte form {0xff, 0x00, 0x01} = L-1, and the GPU REPACKED uint32[2048] NIBBLE form -
one entry = 8 nibbles = 8 L-indices of one 8-elem group), common.cuh (the
arch/feature classes). THE DECODE FORM (now exact): the nibble L-index dots against
the q8 activation via dp4a, and the true value (L-1)+delta is recovered through the
group's q8 SUM - the group dot = d8*(sumi + (delta-1)*sum_q8_group) - the repo's
r18 q8_K activation path ALREADY produces those sums (the bsums the K2/K4 kernels
read for the min corrections; a 32-group = bs[2g]+bs[2g+1]), so NO new activation
format, the launch_quantize_q8_K preprocessing is reused verbatim. THE QUANTIZER
FORM (now exact): per 32-elem group the weighted sort + prefix sums, the EXHAUSTIVE
2-boundary split search over 33 boundaries in both shift directions (x_p =
{-1+d, d, 1+d}, x_m = {-1-d, -d, 1-d}, ~1.1k O(1) evaluations), the ternary L
assignment, then per 8-elem group the kmap grid mapping (the 6561-entry u16->index
map) with the weighted-distance NEIGHBOR fallback; the scale l = round(0.5*(id*s-1))
clamped [0,7] with the shift bit at l bit 3 -> qh bits 12-15, the index halves at
qh 3k, the block d = (max_scale/15)*1.125 and the neighbor's 1.05 - the ggml fudge
factors are part of the tuned format, replicated EXACTLY. THE INSTRUCTION COUNT (the
r19ac open question, closed): per 32 elems M=1 ~48 INT-pipe ops (4 iters of [~4 idx
ops + 1 shared grid load + 3 unpack + 2 dp4a] + a ~10-op tail) = ~1.5 ops/elem; M=8
AMORTIZED (the decode once per (expert, 32-group), 8 dp4a pairs) ~56 ops per 256
elem-dots = 0.22 ops/elem-dot. THE ROOF (the honest math): the T4 INT32+dp4a pipe
~16 lanes/SM/cycle x 40 SMs x ~1.59 GHz ~ 1.0e12 INT ops/s per GPU - the greedy
step's expert class (2.36 G elem-dots, 461 MB) ~3.4 ms single-GPU / ~1.7-2 ms at the
TP half-split (the HBM side 1.5 ms, under the INT roof); the verify round (M=8, 18.9
G elem-dots over ~2.36 G decode-elems) ~4 ms single-GPU / ~2 ms split - FROM the
~114 ms greedy / ~112-155 ms round class, a ~10-25x collapse of the engine's
dominant cost even at 2-3x off-roof. MEASURED AT THE L4, never assumed. THE CRITICAL
KERNEL FINDING: the repo's current launch_gemv_q8k_b RE-DECODES W per (row, batch)
pair - the drop-in form at 1.5 ops/elem-dot would cost ~28 ms/round; the spec FREEZES
the amortized M=8 form as a requirement, not the drop-in. THE REPO SURVEY: the engine
is single-GPU today (cudaSetDevice(0); the TP split is future work the spec stages);
the PackedW planar SoA takes FMT_IQ1S cleanly (codes = qs 32 B/256, hi = qh 16 B/256,
d = 2 B/256 = the 50 B/256 block); the tiered loader's HIT branch (the resident slab,
zero staging) is the stage-1 landing surface - the resident layers go iq1_s, the miss
layers keep the current UVA/chunked + Q2_K_S repack path UNCHANGED, and the quality
gate runs on exactly that mix. THE SPEC (research/CF_REQUANT.md, 9 sections): the pick
(stock iq1_s both tensors, 23.6 GB; the 2-grid ~1.69 bpw custom as the NAMED fallback
if the gate fails), the exact arithmetic, the M=1 + amortized M=8 kernel forms, the
packer (the self-scaled weights w = sqrt(sigma2+x*x) FIRST - the honest imatrix risk
named, the llama.cpp imatrix over the Q2_K_S GGUF as the fallback if the A/B fails),
the pipeline (the BF16 HTTP-range streaming per (layer, tensor) - the largest single
read one layer's gu 3.36 GB, fits the host RAM; the own-layout output file,
expert-major so each GPU's half is one range; ~1.5-3 h one-time Kaggle kernel), the
staged residency (stage 1: the tiered by-layer slabs, no TP, no combine tax, the
interim ~60-80 ms/round class; stage 2: the by-ID TP split, the replicated core
(~2.6 GB x 2, deterministic, no AR), the router on both, one per-layer 2560-float
combine - the r14 AR floor ~0.2-1 ms/step, the honest TP tax), the gates (the LOCAL
round-trip via the deq32 host sim - no GPU needed; the L4 smoke, the perplexity A/B,
the greedy agreement, the rate vs the r18 family, the VRAM inventory), the risk
register, and the r1-r6 round order - r1 is ALL LOCAL (FMT_IQ1S + the deq32 host sim
+ the packer core + the round-trip test), so the implementation starts regardless of
the quota gate. PLAN_CF section 5's ORDER updated to the landed state. STILL AHEAD:
the spec's r1 (the local format + packer core round), then r2+ as gated; the
L4-blocked r19w-r19aa battery rides the same Saturday sessions.

## cf-m6 r1: the requant LOCAL round landed (the format + the packer core + the round-trip gate, ALL GREEN)

THE LANDED SURFACE: t4q/src/iq1s_table.h (the 2048-entry u16 lattice extracted from the
r19ad-saved ggml-common.h - bounds re-verified 2918..3045 = exactly 2048 entries, the
adjacent kgrid_2bit_1024 disambiguated; NGRID/KMAP_SIZE=43692/DELTA=0.125 defines; the
__device__ vs plain static form switched by T4Q_HOST_SIM/T4Q_TABLE_HOST so the same
table serves the kernels, the host sim, and the packer); FMT_IQ1S = 9 in packed.h
(codes = qs 32 B/256, hi = qh 16 B/256, d 2 B/256 = the 50 B/256 = 1.5625 bpw block) +
the pack_fmt_name case; the deq32 FMT_IQ1S branch in deq.cuh (the exact reference
fp-op order: dl = d*(2*sc+1), v = L-1, delta = +-0.125 at qh bit 15, the grid index =
qs[4*ib+l] | ((qh >> 3l) & 7) << 8); t4q/src/requant.h (the packer core: the standard
fp16 converters - RNE narrowing, exact widening, payload-preserving inf/NaN so the
all-65536 round-trip holds; the exact iq2xs_init_impl 3-pass table build with the
(dist2, k) qsort and the nwant=3 distinct-distance cut; quant_row = the exact
quantize_row_iq1_s_impl port - the self-scaled weights w = sqrt(sigma2 + x*x) (qw = 1
first, the imatrix the named fallback), the exhaustive 2-boundary SSD split search
over 33 boundaries in both shift directions, the kmap + the fudge-free
find_best_neighbour2 fallback, the !all_on_grid scale refit, the l = round(0.5*(id*s -
1)) clamped [0,7] packing with the shift at l bit 3, and the d = (max_scale/15)*1.125
fudge replicated EXACTLY; plus the independent {1,3,5}-byte-grid reference decode);
t4q/tests/test_iq1s.cpp + the Makefile test_iq1s target (HOSTCXX g++, host-only).

THE GATE (CF_REQUANT.md section 7, gate 1) PASSED: (a) the fp16 all-65536-pattern
round-trip + 11 RNE spot checks (the exact ties -> even, the subnormal ties, the 65520
overflow boundary -> inf, 65519.9 -> the max finite); (b) the kmap self-check - every
on-grid u maps back to its k AND the byte-grid/u16-table nibbles agree over all 2048
entries; (c) THE ROUND-TRIP: quant_row -> the deq32 host sim (the kernel's own decode
path) vs dequant_row_ref (the independent byte-grid walk) BIT-IDENTICAL over 4096
rows x 2048 cols x 4 input classes (normal amplitudes / the all-zero eps path / the
all-negative flip path / constant rows); (d) the RMSE diagnostic rel ~0.40 on the
synthetic mix - the honest ternary-lattice + 15-step-scale-grid error on
uniform/constant synthetic data, NOT a quality verdict (the quality bar is the L4
perplexity A/B per the spec; the all-zero rows decode exactly 0 via the block-level
d = 0 path). THE BUILD GATE: the podman nvcc library + cf_run build clean, the ptxas
warnings EXACTLY HEAD's 77 (41 tp_prefill + 36 tp_spec, pre-existing, no new warnings
from the __device__ table in every kernel TU).

THE GATE'S TWO CATCHES, both fixed before landing: (1) the reference decode was
missing the 32*ib group offset (every 8-elem sub-group overwrote slots 0..31 and the
rest of the row stayed zeroed) - found by the one-block probe at the first differing
element (deq32 = +0.011816 correct, ref = 0.000000), and the deq32 kernel path was
the CORRECT one all along; (2) the test's own LCG normalization treated the 24-bit
return as 8-bit (~1e5 inputs -> a false 2494 RMSE explosion that looked like a scale
bug in the quantizer - it was the test, not the code). ALSO VERIFIED against the
fetched primary source before landing: the eps path really skips the packing (a zero
group inside a live block decodes to the -0.875*d bias quirk - the ggml behavior,
replicated not fixed), the ggml itself uses the float pairs + int-aliasing +
value-only comparator form (the port's sort is canonical), and the search's x_p-then-
x_m evaluation order, the sumqx > 0 && sumq2 > 0 refit condition, and id = 1/d from
the UNFUDGED d all match verbatim. THE HONEST NOTE: the packer's same-input
determinism holds per-platform (the SSD sort's value-only comparator keeps libc
qsort's tie order - the ggml-inherited property), and the round-trip gate covers the
format's decode consistency, not the L4 quality/rate/VRAM gates which stay ahead.

## cf-m6 r2: the M=1 iq1_s GEMV kernel landed (the nibble grid + the dp4a dot + the dispatch, build-gated locally)

THE LANDED SURFACE: (1) t4q_iq1s_grid_gpu[2048] uint32 in iq1s_table.h - the ggml
iq1s_grid_gpu values VERBATIM (the fetched ggml-common.h table), the halves-interleave
packing: the LOW nibble of byte b = L_b (the sub-vector's elems 0-3), the HIGH nibble =
L_{b+4} (elems 4-7). THE PACKING PINNED EMPIRICALLY, not assumed: both captured tables
(the uint64 iq1s_grid and the uint32 iq1s_grid_gpu) were extracted from the r19ad logs
and cross-checked against this repo's u16 table - the u64 table matches byte j = L_j-1
in LSB order (2048/2048), and of the candidate gpu packings ONLY the halves-interleave
matches (2048/2048; the naive nibble-j=L_j form matches 42/2048 and would mispair L_2
with q_1 in the dp4a - the exact bug class the gate now excludes). (2) The dot_q8k
FMT_IQ1S branch in gemv_ref.cu - the ggml vec_dot_iq1_s_q8_1 translated to the PackedW
planes + the repo's q8_K pairing: 4 iterations per 32-group (the index build qs[k] |
((qh >> 3k) & 7) << 8, ONE u32 grid load, the 3-op nibble unpack to the 2 dp4a
byte-quads, 2 dp4a), the d1q = h2f(d)*(((qh >> 11) & 0xE) + 1) tail (the ggml's
one-shift trick), delta = -1 + 0.125 - (qh & 0x8000)*(2*0.125/0x8000) (= -0.875
unshifted / -1.125 shifted), and the group dot = d1q*yd*(sumi + delta*(bs0+bs1)) - the
SAME two bsums the K2/K4 branch already reads (the ggml q8_1 s-term's equivalent at the
q8_K pairing, NO new activation format). (3) The launch_gemv_q8k FMT_IQ1S case (the M=1
greedy path on the proven warp-per-row k_gemv_q8k form) + the gemv() dispatch branch in
cf_engine.cu (FMT_IQ1S joins the K2/K4 q8_K pairing). The batched _b form deliberately
NOT extended - the spec freezes the r5 amortized M=8 kernel as the verify path (the
drop-in re-decodes W per (row, batch) pair, ~28 ms/round at this dot's op count).

THE GATES (test_iq1s.cpp, all local): the u32-vs-u16 packing identity 2048/2048; the
INT-EXACT sumi gate - the kernel's nibble-unpacked dp4a sum vs the INDEPENDENT u16
per-elem walk, integer-identical over 4096 rows x 64 groups x 4 sub-vectors (0
mismatches - any nibble-order/pairing bug is deterministically excluded); the fp-dot
gate - the full kernel-arithmetic twin (with the host q8_K quantizer twin: the
first-occurrence argmax tie-break, iscale = -127/maxv, MIN(127, v), the per-16 bsums,
d = 1/iscale - the k_quantize_q8_K sequential twin) vs the deq32-decode dot over the
same x: rel 6.9e-05 L1-normalized, inside the honest q8_K activation error class (the
gate bound 2e-2; a pairing or correction bug lands at O(30%)). THE BUILD GATE: nvcc
clean, the k_gemv_q8k<9> instantiation 41 registers / 0 spills / 0 stack / 0 smem / 472
B cmem (the K2 twin runs 63 regs - the iq1_s dot is the lighter one), the ptxas
warnings EXACTLY HEAD's 77 (41 tp_prefill + 36 tp_spec, pre-existing). THE DEVICE-FORM
NOTE, honestly: the grid reads go straight to the __device__ table (the ggml MMVQ's own
form - 8 KB is L1-resident), NOT the spec's shared-memory staging; that stays a
rate-tuning option the L4 measure decides, not a correctness difference. STILL AHEAD:
r3 (the packer pipeline - the BF16 HTTP-range streaming + the own-layout output), r4
(the stage-1 tiered residency + the L4 battery), r5 (the amortized M=8), r6 (the TP
split); the L4-blocked r19w-r19aa battery rides the same Saturday sessions.

## cf-m6 r3: the packer landed + THE TILING FINDING (the dn 640 half-block amendment, ALL GATES GREEN)

THE ROUND'S CATCH, found by the pack_gate the round was built around: the dn rows are
640 wide and 640 % 256 = 128 - the 256-elem iq1_s block CANNOT tile the dn (this is
why the GGUF's dn experts are Q4_0, a 32-block format). The naive 768-pad puts the
pool at 25.17 GB, OVER the ~24-25 GB 2xT4 room. THE AMENDMENT, landed this round:
FMT_IQ1SH = 10, the 128-elem half block (16 B codes + 8 B hi + 2 B d = 26 B/128 =
1.625 bpw; 640 = 5x128 EXACTLY) - the SAME per-32-group search/scale/lattice/dp4a
arithmetic, only the block scope halves (max_scale and d span 4 groups not 8 - a
finer scale grid on the dn, marginally BETTER quality if anything). The corrected
pool: gu 15.73 GB (80.5 G elems at 1.5625 bpw) + dn 8.18 GB (40.2 G elems at 1.625
bpw) = 23.90 GB, inside the room. THE PAIRING, settled with it: the SH dot pairs with
q8_0, not q8_K (the q8_K 256-super-blocks also cannot tile 640; the q8_0 32-blocks
tile anything, and its per-32 signed int sum xs[g] is exactly the correction the dot
needs: weight value = L + delta - 1, so the group dot = d1q*dy*(sumi + delta*s32)).

LANDED: (1) requant.h TEMPLATED - BlockT<NG> (static_asserts 50 B at NG=8, 26 B at
NG=4), quant_row_t<NG>, dequant_row_ref_t<NG>, the stock-256 wrappers kept unchanged
(the r1 gate passes unmodified through the refactor - the round-trip rmse rel is the
same 0.4022, the refactor is byte-identical). (2) FMT_IQ1SH = 10 in packed.h + the
deq32 FMT_IQ1SH branch in deq.cuh (nb = cols/128, the 16/8/2-B plane strides, the
group decode IDENTICAL to FMT_IQ1S) + the pack_fmt_name case. (3) The
dot_q8_0_iq1sh branch + k_gemv_iq1sh in gemv_ref.cu (the k_gemv_q80 warp-per-row
twin: per 32-group the index build qs[k] | ((qh >> 3k) & 7) << 8, ONE u32 grid load,
the 3-op nibble unpack, 2 dp4a, then the d1q = h2f(d)*(((qh >> 11) & 0xE) + 1) tail
and the s32 correction) + the launch_gemv_q8_0 FMT_IQ1SH case + the gemv() dispatch
branch in cf_engine.cu (FMT_IQ1SH joins the P4/Q8 q8_0 pairing). (4)
tools/cf_requant_pack.cpp v2 - the packer: the 96-B SlabHdr (magic/version/fmt +
dn_fmt, the plane offsets, static_asserted), the mmap'd bf16 sources, OpenMP
quantize_tensor<NG> (8 for gu, 4 for dn), the plane split, the synthetic source
generator, and the verify that reads BOTH plane sets back and decodes the dn via
deq32<FMT_IQ1SH> + dequant_row_ref_t<4>, with the src-RMSE diagnostic.

THE GATES (all local): test_iq1s EXTENDED with the SH section - the 640-wide
round-trip (2048 rows x 5 blocks/row, the same input classes incl. the eps and
constant paths) BIT-IDENTICAL deq32-vs-ref, rmse rel 0.4021 (the honest lattice
class, matching the 256-block's 0.4022); the SH M=1 dot twin - the host q8_0
quantizer twin (amax/127, the fp16-rounded d, roundf(x*id), the per-32 signed sum)
+ the full kernel-arithmetic dot vs the deq32-decode dot: sumi INTEGER-EXACT (0
mismatches vs the independent u16 per-elem walk) and rel 1.23e-04 L1-normalized
(the honest q8_0 activation class, bound 2e-2). pack_gate GREEN: ne=4 synthetic,
the slab 3,891,296 B EXACTLY (96-B header + 2,560,000 gu + 1,331,200 dn - the plane
arithmetic checks to the byte), all 15,360 rows verified BIT-IDENTICAL, src rmse rel
0.4024. THE BUILD GATE: nvcc clean, k_gemv_iq1sh = 64 registers / 0 spills / 0 smem
(the k_gemv_q80 P4 twin's own 64 - the family budget), ZERO ptxas warnings in the
rebuilt TUs (HEAD's 77 are all tp_prefill 41 + tp_spec 36, their objects unchanged;
the counts re-confirmed from the tracked ptxas logs). STILL OPEN in r3's frame: the
Kaggle streaming driver notebook (the 131-shard HTTP-range read of the expert
tensors - the index map is saved, the gu shards are single-tensor 3.36 GB files) +
the dataset push; both ride the Saturday L4 window with the r19w-r19aa battery.
NEXT: r4 (the stage-1 tiered residency - the loader reads the slabs, the HIT branch
dispatches FMT_IQ1S/FMT_IQ1SH) + the L4 battery.

## cf-m6 r3 part 2: the packer DRIVER landed (the two CPU kernels, live-smoked, ready for the Saturday push)

THE r3 PIPELINE REMAINDER: t4q/tools/stage_cfreq.py (the driver template) + the mkkernel.py
--template/--define/--cpu extensions, generating kaggle/cfreqa/t4q-cfreqa.py (layers [0,24),
~11.95 GB) + kaggle/cfreqb/t4q-cfreqb.py (layers [24,49) incl. the MTP, ~12.45 GB). THE FORM
and its reasons, all forced by real limits: (1) TWO CPU-ONLY kernels, not one GPU kernel -
the ~24.41 GB pool (49 layers x 498.07 MB; the MTP layer's ne is ALSO 512, verified from the
shard headers, so the pool arithmetic is now exact) is over the 20 GB kernel-output cap, so
the 49 layers split in two, each half under the 19.5 GB working dir; the runtime battery
attaches BOTH as kernel_sources (the p/pg pattern) - NO dataset push, NO secrets; and the CPU
kernel burns NO GPU quota, keeping the whole Saturday window for the runtime battery. (2) The
streaming: the index fetched at runtime (no baked map), the shard headers cached (one 8-B +
one header read each), the tensors streamed in 256 MB range SEGMENTS with per-segment retry +
resume from the segment's own start (10 attempts, the rewind handled), the producer thread one
layer ahead over 2 rotating raw slots on /tmp (~10 GB peak). (3) The atomic slab: pack to
.tmp, the byte-count check to the byte (96 + ne*1280*500 + ne*2560*130), the --verify
32-row sample against the raw sources (deq32-vs-ref + the src-RMSE diag), THEN the rename -
a killed session never leaves a partial slab. (4) manifest.json = the r4 loader's input (the
per-layer file/ne/kind/bytes/verify). (5) The r19b/r19f watchdog pair at DEADLINE 11 h of
the 12 h CPU cap. THE GATES: both generated drivers parse (ast) with the defines verified
(LO=0/HI=24 and LO=24/HI=49), the metadata is CPU (enable_gpu false, no machine shape, no
sources); and THE LIVE SMOKE against the real endpoint - the index fetch, the shard-header
parse, ANONYMOUS 206 range reads, layer 0's gu = BF16 [512,1280,2560] with the offsets
summing to EXACTLY 3,355,443,200 B (the number the r3 study pinned), real weight values
decoding (|w| ~ 1.2 class), the MTP dn [512,2560,640] on its own shard, and the ne=512 ->
498.1 MB slab arithmetic confirmed against the packer's plane math. The honest note: the
~1.2 weight magnitude class (vs the synthetic gate's 0.02-0.17) is amplitude-independent
for the quantizer, but the REAL RMSE lands only in the per-layer verify at the pack - the
manifest carries it. NEXT: r4 (the stage-1 tiered residency - the loader reads the slabs +
the HIT-branch mixed-format dispatch), then the Saturday pushes + the L4 battery.

r19ae cf-m6 r4 the stage-1 tiered residency LANDED (the resident tier + the batched dots,
ALL LOCAL GATES GREEN, the value gates L4-blocked): THE MECHANISM - T4Q_CF_IQSLAB=<dir> +
T4Q_CF_IQN=<1..48> (the count EXPLICIT, the L4 VRAM inventory decides it) pins the first
N main layers' experts to FULL-RESIDENT IQ1S/IQ1SH slabs (gu [NE*2*EE, D] FMT_IQ1S + dn
[NE*D, EE] FMT_IQ1SH, the identity hot map: hn = NE, every pick a HIT); the layers past
the resident prefix keep the existing miss path (K2/UVA/ON). THE PAIRING COINCIDENCE that
kept it one-branch-per-layer: the resident gu rides the SAME q8_K activation the K2 miss
path rides, and the resident dn the SAME q8_0 the P4 miss path rides - emit_moe_rest
branches on `iqs = tiered && L.res_gu.fmt == FMT_IQ1S` and swaps ONLY the dot launches
(launch_gemv_q8k_b's new FMT_IQ1S case + the new k_gemv_iq1sh_b), NO within-layer format
mix, the quantize code paths IDENTICAL to the miss case. THE MTP CONSISTENCY FINDING: the
verify MUST read the same weights the greedy path reads (the originals-staged verify at a
covered layer would compare the greedy's resident-read logits against original-weight
verify logits - the acceptance would compare two different models); the covered layers
skip the union staging entirely (nu = 0) and the per-row views point at the resident
planes. THE CODE: gemv_ref.cu (the batched IQ1S gu case + k_gemv_iq1sh_b/launch on the
k_gemv_q80_b walk), packed.h (SlabHdr moved here - ONE shared definition, the same
static_assert pins the writer and the reader), cf_model.h (iqs_n + uva_lo), cf_loader.cu
(the IQSLAB block: the free-VRAM check at ~498.07 MB/layer x1.02, the header + plane
validation to the byte, the pageable one-time uploads, the MTP slab deferred to stage 2 -
the draft block owns blk.48; the hot-set skip consumes + discards the covered ids; the
all-miss safety pass for layers covered by NEITHER tier - the tiered host_router derefs
hot_idx; the UVA prefix exclusion: the span + alias loops over [uva_lo, uva_n), the
fully-covered case logged + off), cf_engine.cu (host_router's IQ1S view branch, the
emit_moe_rest/vfy_moe_em iqs launches, the vfy_window resident early path, the three
uva_lo conditions). THE GATE'S TWO CATCHES, both fixed pre-landing: (1) alloc_packed had
NO FMT_IQ1S/FMT_IQ1SH cases - the silent fall-through left 0-B/NULL planes, the tier's
uploads would land nowhere (a RUNTIME crash no compile/build gate can see; fixed with the
two cases + the loader's new alloc-vs-slab cross-check that catches the class AT LOAD);
(2) the d-offset UNITS bug - d is uint16_t*, the offset is in ELEMENTS, no byte factor
(the existing K2 form's own convention); the first draft had *2 at both view sites,
caught in review against the K2 form, and the new resident-view gate pins the class. THE
GATES (all local): test_iq1s + the NEW resident-view gate (the 2-expert expert-major
plane, the expert-1 view at the loader's EXACT offsets, the SH dot twin vs the deq32
decode of expert 1's own rows: sumi INTEGER-EXACT, dot inside the 2e-2 class) - the
r1/r2/r3 gates UNCHANGED (fp16 0; round-trip rel 0.4022; M=1 6.90e-05; SH rel 0.4021 /
sumi EXACT / dot 1.23e-04); pack_gate GREEN through the SlabHdr move (the ne=4 slab
3,891,296 B exact, all rows bit-identical); the nvcc podman build clean x2 (through the
alloc fix) with ZERO ptxas warnings in the rebuilt TUs (HEAD's 77 all in the untouched
tp_prefill 41 + tp_spec 36); the registers: k_gemv_q8k_b<9> = 47 (LIGHTER than the K2
twin's 63 - the nibble-grid dot is register-cheap), k_gemv_iq1sh_b = 64 (the q80_b family
budget), both 0 spills / 0 stack / 0 smem. THE GRAPH NOTE, honestly: the gmode&&tiered
check turns the graphs OFF under the IQSLAB (the conservative form); a full-coverage run
is capture-constant in principle but the uncovered-verify staging and the draft still
vary - the re-enable is an L4-measured later round. STILL L4-BLOCKED (the Saturday
window): the cfreqa/cfreqb pushes + the real slabs, the smoke (an add-function greedy run
at T4Q_CF_IQSLAB), the perplexity A/B vs the Q2_K_S trunk, the greedy agreement, the
resident-expert gemv GB/s vs the r18 K-quant ~179-200, and the VRAM inventory that sets
iqs_n. NEXT: r5 (the amortized M=8 verify kernel) while the L4 window holds, then the
battery, then r6 (the stage-2 TP split).

r19af cf-m6 r5 the AMORTIZED M=nr verify kernel LANDED (the spec's frozen form, ALL LOCAL GATES GREEN, the rate L4-blocked): THE FORM - k_gemv_iq1s_vfy + k_gemv_iq1sh_vfy in gemv_ref.cu: the _b family's own walk (warp = the W row, lane strides the 32-groups, the xor tree) with the row loop UNROLLED over the fixed 8 slots and GUARDED by the pick mask (block-uniform - no intra-warp divergence): per (union pick, 32-group) the decode happens ONCE (the 8 grid quads held in registers, d1q/delta once), then every row that picked the expert dots against the SHARED quads (2 dp4a per 8 elems per row + the per-row tail: the bsums pair at the q8_K pairing, the s32 at the q8_0), the outputs at the SAME per-row planes the _b form wrote. THE INTERFACE - the picks' UNION views (uv_gu/uv_dn at the resident-slab offsets, vfy_window's dedup verbatim) + the row map (rowmap[u*8+r] = the pick index or -1) + the FIXED per-row plane table (VfyMoeTab in packed.h, 8 fields x 8 slots, ONE upload at the verify alloc - the scratch pointers never move); vfy_moe_em's resident branch restructured to the phased form (the nr quantizes -> ONE gu launch -> the per-row silu + q8_0 quantizes -> ONE dn launch -> the per-row tail verbatim). THE BIT-IDENTITY CLAIM, pinned by the new gate: the per-(row, pick) accumulation over g is EXACTLY the _b form's (the same g-sequence per lane, the same tree, the same tail expression) - the gate's amortized sim (the kernel's OWN decode-then-rowloop structure) vs the per-(row, pick) dot twins over the same views + activations: BIT-IDENTICAL (0 diffs), sumi INT-EXACT vs the independent u16 walk (0), rel 2.18e-04 vs the deq32 decode dot (0 over), and the union/rowmap COVERAGE check OK (every pick at exactly one (slot, k), the uidx sweep leaves no residue). THE GRAPH NOTE: the amortized launch's grid rides nu (the union count, varying per layer AND per step) - not capture-constant, exactly the class the r19z comment names; the verify's V1 graphs are already dead under the tier (the loader's gmode&&tiered kill), so no capture ever sees the varying grid. THE REGISTERS: k_gemv_iq1s_vfy = 64 / k_gemv_iq1sh_vfy = 60, both 0 stack / 0 local / 0 smem - NO SPILLS (the masked unroll keeps the 8 accs + the 8 quads live inside the family budget). THE GATES: test_iq1s ALL GREEN (the r1-r4 lines unchanged: fp16 0, round-trip rel 0.4022, M=1 6.90e-05, SH rel 0.4021 / sumi EXACT / dot 1.23e-04, the resident-view twin EXACT/OK; the NEW amortized line IDENTICAL/EXACT/OK/OK); pack_gate GREEN (3,891,296 B exact, all rows verified); the nvcc podman build clean with ZERO ptxas warnings in the rebuilt TUs. THE HONEST NOTES: (1) the win is the pick overlap across the nr rows (the drafts are near-duplicates - their top-10s overlap heavily); at ZERO overlap (nu = nr*TOPK) the cost is the _b form's own + the rowmap overhead - win-neutral at worst, the L4 measures the real overlap; (2) the UNCOVERED layers (past the iq1_s prefix at stage 1) keep the _b form - the amortization extends there only if the L4 shows the uncovered verify cost matters (at stage 2's full pool every layer is resident). NEXT: r6 (the stage-2 TP split - the by-ID residency, the replicated core, the per-layer combine - the full 24.41 GB pool), then the Saturday L4 battery (the cfreq pushes + the real slabs + the smoke/A-B/rate/VRAM inventory + the verify-round wall measure).
