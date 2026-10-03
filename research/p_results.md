# Milestone P (prefill) results, round 1

Every number below was measured on Kaggle (`otdoges/t4q-p`, versions 1-6, 2026-10-02). Raw outputs are in `kaggle/p/out` (v2; v1 was overwritten by v2), `kaggle/p/out_v3` ... `kaggle/p/out_v6`, each with `results.json`, `results_pf.json` (engine runs) and `logs/` (`gemm_u*_dev0.txt`, `gemm_sustain_dev*.txt`, `prefill_check.log`, `clocks.csv`).

## Gate: NOT passed

| | pp512 tok/s | pp2048 tok/s | box |
|---|---|---|---|
| llama.cpp `-sm tensor`, Q4_0 (baseline.md) | 516 | - | |
| t4q-p v4 (ub 512, no AR overlap) | 574 | 551 | no P2P |
| **t4q-p v5** (best config per size) | **583.5** | **633.8** | P2P, GPU0 ~850 MHz |
| t4q-p v6 (fused q8 producers, GDN re-layout) | 579 | 597 | P2P, GPU0 throttled to 680-750 MHz |
| gate | >= 1200 | >= 1400 | |

Correctness passes on every configuration of v5 and v6 (below).

## W4A8 GEMM (`t4q/src/kernels/gemm.cuh`, bench `t4q/tools/gemm_bench.cu`)

- Reads the decode GEMV's packed weights in place (P4 = Q4_0, P4M = Q4_1, K5 = Q5_K; layout rpl 2 or 4, chunk-major), so prefill needs no second weight copy.
- `mma.sync.m8n8k16.u8.s8` (codes in the high bits of each byte), int32 accumulation on top of the magic constant 0x4B400000 so the int result is a float without I2F, fp32 epilogue per 32-block: `t = f*s + nq; acc = fma(d_w, t, acc)` (+ one FFMA for Q4_1/Q5_K mins).
- Block 128 rows x 128 tokens, 8 warps (64 x 32 warp tile), ldmatrix fragments, register-staged double buffer, 1 block/SM.
- Correctness: rel L2 vs an fp64 reference on the same q8 activations 1.3e-5 to 3.1e-5 for every shape and format; the GPU quantizer equals its host mirror bit for bit.

Burst TOPS (single GPU, T = 512 / 2048; clocks drift with temperature, so treat these as ranges):

| shape (per GPU) | fmt | T=512 | T=2048 |
|---|---|---|---|
| qkvz 5120 -> 8192 | P4 rpl 2 | 15-38 | 19-36 |
| gateup 5120 -> 17408 | P4 | 27-29 | 23-24 |
| down 8704 -> 5120 | P4 | 40-42 | 35-41 |
| down (Q4_1 layers) | P4M | 31-35 | 27-30 |
| ssm_out 3072 -> 5120 | K5 | 25-30 | 27-31 |
| attn qkv 5120 -> 7168 | P4 rpl 2 | 33-40 | 33-42 |
| attn out 3072 -> 5120 | P4 | 35-39 | 35-40 |

**Sustained, both GPUs loaded (gateup T=2048, 10 s per variant, 70 W cap):**

| variant | v2 GPU0 / GPU1 TOPS | MHz | v3 GPU0 / GPU1 TOPS |
|---|---|---|---|
| i8 W4A8 (correct) | 24.2 / 25.7 | 897 / 955 | 23.8 / 26.0 |
| i8, one FFMA epilogue (timing only) | 26.1 / 28.9 | 788 / 879 | - |
| i8, no epilogue at all (timing only) | 28.3 / 32.0 | 688 / 785 | 27.4 / 35.3 |
| fp16 HMMA m16n8k8, weights dequantized in registers (correct, rel 2.5e-4 vs fp32 x) | 18.4 / 19.4 | 962 / 1016 | - |
| int4 m8n8k32, activations split into exact hi/lo nibbles (correct up to a rounding issue, 4.6e-4) | - | - | 21.8 / 25.2 |
| int4 W4A4 cost (one m8n8k32 per block, timing only) | - | - | 24.9 / 30.4 |

Findings:
- Every variant runs into the 70 W cap, at 680-1000 MHz. The GPUs idle at about 30 W (`clocks.csv`: 29.6 W at 705 MHz, 0% utilization), so only about 40 W is available for compute. Throughput is therefore energy per op, not peak rate.
- The exact per-32 epilogue (2 FFMA per output per 32-block, against 2 IMMA) costs about 20-30%. Halving the IMMA count (W4A4) gains less than removing the epilogue. The int4 split and fp16 paths are not more efficient.
- The linear layers are 24.3 GOP per token per GPU. Linear alone is capped at 990-1070 tok/s with the exact kernel's sustained rate, and at 1150-1440 tok/s even with no epilogue at all. **pp2048 >= 1400 is out of reach for this design at 70 W.**

## Batched TP prefill engine (`t4q/src/tp_prefill.cu`)

Tokens [0, n-1) go through the batch path in ubatches (default 2048) of 2 interleaved sub-batches. The last prompt token runs as a normal decode step (graph), which yields the logits and leaves StepState as decode expects. Per layer:
- **Norm + q8**: AR add + RMSNorm, quantized straight into the GEMM activation layout (`k_pf_add_norm_q8`).
- **DeltaNet**: W4A8 GEMMs, then the ab rows (fp32 SIMT), the conv kernel (decode's arithmetic, history from the conv ring), a sequential scan with decode's per-token math (state in registers, smem-staged inputs; the state stays in the decode layout), and gated norm + q8.
- **Attention**: tensor-core flash attention (`k_pf_fa`: Q/O in registers, mma m16n8k8, fp16 P), writing the decode KV cache.
- **FFN**: gate|up GEMM, silu + q8, down GEMM.
- **All-reduce**: fp32 partials by `cudaMemcpyPeerAsync` on a copy stream; the copy of one sub-batch overlaps the other's compute.

Correctness (v6 default config; pf0 = the decode path, pf1 = batched prefill):

| prompt (tokens) | KL(pf0 ‖ pf1) last token | KL(oracle batch ‖ pf1) | llama floor (batch vs tbt) | top-1 | greedy 32 |
|---|---|---|---|---|---|
| P0 (81) | 3.4e-5 | 1.2e-5 | 3.0e-5 | equal | identical to pf0 and oracle |
| P1 (72) | 6.6e-5 | 1.6e-4 | 3.1e-5 | equal | identical |
| W (400) | 3.9e-4 | 8.4e-4 | 2.6e-3 | equal | pf1 = oracle for all 32; pf0 diverges from both at 25 (oracle gap 0.08) |
| L (2048, 1 ubatch) | 2.3e-4 | 6.9e-4 | - | equal | identical to pf0; both diverge from the oracle at 8 (gap 0.39) |

On top of that, for the short prompts (P0/P1/W), 16 teacher-forced positions after prefill have KL(pf0 ‖ pf1) of 4e-5 to 7e-4, which checks the resulting DeltaNet/conv/KV state.

Profile (v5, pp2048, ub 2048, GPU0 stream ms, 3.2 s batch time):

| part | ms |
|---|---|
| GEMMs | 2204 |
| ... gateup | 1015 |
| ... down | 499 |
| ... qkvz | 356 |
| ... ssm_out | 178 |
| ... attn_qkv | 110 |
| ... attn_out | 46 |
| GDN scan | 329 (285 after the v6 re-layout) |
| AR wait + norm | 307 |
| quant | 95 (mostly removed by the fused producers in v6) |
| ab | 72 |
| silu | 56 |
| flash attention | 55 (SIMT kernel: 385) |
| conv | 25 |
| gated norm | 20 |

A/B results:
- **AR overlap.** With 2 sub-batches, pp2048 is 604 against 577 for 1 sub-batch at ub 512 (v5). The overlap removes about 1 s of waiting, but the GPUs then run at about 800 instead of about 1080 MHz, the same energy argument as above.
- **Flash attention** against the SIMT kernel: 634/604 against 555 at pp2048.
- **Fused q8 producers**: 597 against 594 (v6, within noise; GPU0 was throttled).

# Round 2 (2026-10-02, `otdoges/t4q-p` v7-v17, `otdoges/t4q-pg` v1-v7)

Raw outputs: `kaggle/p/out_v7` ... `kaggle/p/out_v17` and `kaggle/pg/out*`. They are git-ignored but kept on disk.

## Gate: NOT passed

| version | config | pp2048 | pp512 | clocks GPU0 / GPU1 |
|---|---|---|---|---|
| v12 | round-1 path (`pf_g8=0`) | 746.8 | 699.0 | ~1035 / 1072 |
| v12 | gemm9 GA32, fused producers | 859.4 | 762.6 | |
| v12 | gemm9 GA64, unfused | 925.2 | 799.6 | |
| v12 | GA0 (per token) | 998.8 | 849.2 | **fails correctness** (L KL 0.31) |
| v15 | GA64 + fused silu | 887.9 | 779.6 | 1065 / 908 |
| v16 | + chunked GDN | 920.9 | 819.2 | 875 / 1010 |
| **v17** | **+ fp16 AR (current defaults)** | **988.6** | **855.4** | 960 / 975 |
| v17 | same, fp32 AR | 967.7 | 841.1 | |

v17 correctness for the default config:

| prompt | KL(pf0 ‖ pf1) | KL(oracle ‖ pf1) | top-1 | greedy 32 vs pf0 |
|---|---|---|---|---|
| P0 | 6.7e-4 | 6.6e-4 | equal | identical |
| P1 | 1.1e-3 | 7.0e-4 | equal | identical |
| W | 3.2e-4 | 1.2e-3 | equal | diverges at 25; pf1 matches the oracle for all 32 |
| L (2048) | 5.4e-4 | 1.3e-3 | equal | identical |

## GEMM: sustained TOPS per GPU, both GPUs loaded, gateup T=2048

| kernel | TOPS | MHz | ops/clk/SM | note |
|---|---|---|---|---|
| cuBLAS int8 GemmEx (v7) | 54.9 / 44.6 | 991 / 781 | ~1400 | random int8 data |
| cuBLASLt int8 COL32 (v7) | 56.4 | 1530 | ~920 | memset data, low toggling, not comparable |
| cuBLAS fp16 (v7) | 30.4 / 27.2 | 884 / 783 | | |
| CUTLASS int8 128x256 (v7) | 54.8 / 45.0 | 977 / 759 | 1400 | |
| CUTLASS int8 128x128 (v7) | 44.3 / 34.8 | 777 / 591 | 1424 | |
| CUTLASS int4 128x256 (v7) | 125.9 / 121.6 | 1321 / 1254 | 2380 | 59 / 55 W, not at the cap |
| round-1 W4A8 (`gemm.cuh`) | 22-26 | 815-960 | 670 | |
| gemm8 (GA32, in-kernel requant) | 26.4 / 27.0 | ~1030 | 650 | v8 |
| **gemm9 GA64 128x256** | **30.9-33.5** | 990-1080 | 770-780 | production |
| gemm9 GA32 | 24.2-28.4 | | 690-710 | |
| gemm10 CUTLASS-style 128x128 GA64/128 | 27.0-28.3 | 811-878 | 805-832 | |
| gemm11 256x128 | 31.7-33.8 | | 796-814 | |
| gemm12 2 blocks/SM (128 thr) | 20.5-24.4 | 651-813 | 750-785 | |
| gemm13 CUTLASS-style, per token (probe) | 37.9-40.0 | 927-994 | 1006-1021 | inaccurate (GA0) |
| gemm14 GA64 on the CUTLASS pipeline | 15.7-15.9 | | 413-426 | spills 640 B |
| gemm15 GA64, quarter tiles (pg v8) | 27.2 / 28.4 | 999 / 1055 | 673-681 | correct, about 12 B spill; slower than gemm9 (30.7 / 32.3 in the same run) |

gemm9 ablations (v11, timing only, GA64 unless noted):

| variant | TOPS | ops/clk/SM |
|---|---|---|
| full | 32.4 | 779 |
| per-token scale (no FFMA) | 35.2 | 835 |
| no weight conversion | 31.7 | 757 |
| no FFMA | 33.9 | 833 |
| L1-hot loads | 39.1 | 879 |
| no global loads | 49-50 | 1086 |
| no global loads, no smem stores or barriers | 67-70 | 1459 |
| per token, no loads, no stores | 86 | 1605 |
| pure ldmatrix + mma loop (v9) | 90 | 1599 |

Data toggling (pg v4, gemm9 GA64, TOPS):

| inputs | TOPS |
|---|---|
| real-ish data | 31.3 |
| zero activations | 37.5 |
| zero weights | 35.4 |
| constant activations | 35.0 |
| 7/8 of activation bytes zero | 33.6 |

Per-row requantization error on real Q4_0 tensors (blk.20 ffn_down/gate/qkv, blk.60 ffn_down, blk.0 ffn_down Q4_1). Added variance as a fraction of Q4's own noise variance:

| group size | added variance | rel_y |
|---|---|---|
| per row | 0.9-1.7% | 0.84-1.05% |
| per 256 | 0.6% | |

Group sizes are in elements. rel_y is the relative error in y.

In the kernels, gemm9's rel L2 against the host mirror is 1.0e-4 to 2.3e-4 (fp32 magic-bias accumulation), and against the exact Q4 product 6e-3 (synthetic random weights).

## Chunked DeltaNet (`k_pf_gdnc`, v16)
- Against the sequential scan on the same inputs (first DeltaNet layer):
  - o: max error 1e-3 to 4e-3 against max |o| of 3-6;
  - S: max error 6e-3 to 1.7e-2 against max |S| of 27-41.
- pp2048 GPU0 time: 224 ms (sequential, 96 blocks) down to 84 ms (24 blocks).

## Kaggle box controls (pg v7)
- The container runs as root, but:
  - `nvidia-smi -lgc`, `-rgc` and `-pl` give "insufficient permissions";
  - `-lmc` is "not supported for GPU";
  - `-ac` with a 405 MHz memory clock is "not supported" (the only memory application clock is 5001 MHz).
- So the 70 W cap and the memory clock are fixed.

# Round 3 (2026-10-02/03, `otdoges/t4q-p` v18-v34, `otdoges/t4q-pg` v9-v13)

Raw outputs: `kaggle/p/out_v18` ... `kaggle/p/out_v34`, `kaggle/pg/out9` ... `kaggle/pg/out13` (git-ignored, kept on disk).

## Gate: NOT passed

| version | config | pp2048 | pp512 | clocks GPU0 / GPU1 (pp2048) | correctness |
|---|---|---|---|---|---|
| v30 | GA64 (round-2 default) | 960.0 | 828.5 | 908 / 938 | pass |
| v31 | GA64 (round-2 default) | 1006.1 | 870.5 | 975 / 990 | pass |
| v26 | R512 + weight cache 4.5 GB, nsub 2 | 1173.8 | 856.1 | 998 / 728 | pass (L KL 2.9e-3; build before the T-solve change) |
| v30 | R512 + cache, nsub 2 | 1197.5 | 879.7 | 825 / 720 | fails L (KL 7.1e-3) |
| v34 | GA64 (round-2 default) | 978.4 | 859.7 | 945 / 982 | pass |
| **v31** | **R512 + cache, nsub 2** | **1213.2** | 895.7 | 825 / 705 | **fails L (KL 7.1e-3 vs limit 2e-3)** |
| v31 | R512 + cache, nsub 1 | 1125.7 | **955.9** | 878 / 908 | fails L |
| gate | | >= 1400 | >= 1200 | | |

R512 = Hadamard-rotated per-token int8 activations x rotated per-row int8 weights (below). "fails L": last-token
KL(decode path || batched prefill) on the 2048-token prompt is 7.1e-3; llama.cpp's own batch-vs-step spread on L
(new `floor` oracle job, v31) is 8.2e-4, so the limit is 2e-3. Top-1 and greedy 32 tokens match on every prompt.
**R512 is not correctness-preserving** (v32-v34): every R512 variant fails L (last-token KL 2.5e-3 to 1.6e-1,
teacher-forced continuation KL max up to 0.64), while GA64 stays at KL 3.2e-4 and continuation max 6.7e-3. It stays
an opt-in fast mode (`pf_rot=1`); the default remains GA64.

R512 accuracy variants on L (v32-v34; KL = last token vs the decode path, cont = 16 teacher-forced positions):

| variant | L KL | L cont mean / max | W KL |
|---|---|---|---|
| GA64 (default) | 3.2e-4 | 1.1e-3 / 6.7e-3 | 4.3e-4 |
| R512, all GEMMs | 7.1e-3 | 2.0e-3 / 2.3e-2 | 1.6e-3 |
| R512, fp32 silu output (`pf_h16=0`) | 3.1e-3 | 4.1e-2 / 6.4e-1 | 1.0e-3 |
| R512, fp32 alpha/beta (`pf_abq=0`) | 5.9e-3 | 3.0e-3 / 4.6e-2 | 2.1e-3 |
| R512 + per-512-block scales (`pf_rgb=1`) | 1.6e-2 | 1.2e-3 / 1.0e-2 | 2.8e-3 |
| R512 except ssm_out / attn_out (mask 15) | 3.0e-3 | 4.5e-3 / 6.3e-2 | 9.7e-4 |
| R512 only gateup + down (mask 12) | 4.3e-2 | 3.4e-2 / 3.6e-1 | 2.2e-3 |
| R512 only ssm_out + attn_out (mask 48) | 1.7e-3 | 1.6e-3 / 2.2e-2 | 1.5e-3 |

No subset of GEMMs on R512 gets the continuation error down to GA64's level, so per-token int8 (even after rotation)
is too coarse for this model on long prompts; GA128 and GA256 already failed L in the emulation study.

## Activation-format study (v18-v20, `pf_fq`: emulated on the GA64 input of every GEMM)

Hidden-state metric: relative L2 error of the final residual of every prompt token vs the GA32 run (median shown).

| format | L last-token KL | L median h error | note |
|---|---|---|---|
| GA64 (production) | 5.4e-4 | 3.3e-2 | reference point |
| per token | 2.1e-2 | 2.0e-1 | fails; down input alone: KL 0.57 |
| per token, clip 8 rms + exact residual | 1.0e-2 | | union of residual channels 2307 of 8704 (down) |
| top-32 channels exact + per token | 1.2e-1 | | outliers are not channel-structured |
| GA256 | 5.0e-3 | 6.4e-2 | borderline |
| GA512 | 9.1e-2 | | fails |
| block Hadamard 512 + per token | 1.5e-3 | 4.6e-2 | best per-token format (emulation adds a GA64 rounding) |
| per-channel smoothing + per token | 8.5e-1 | 8.1e-2 | fails |
| per token x 2^-e per 64 (shift-foldable, e <= 7) | 5.9e-2 | 5.3e-2 | |

## GEMM (pg v9-v13, gateup T=2048, sustained, both GPUs)

| kernel | TOPS | ops/clk/SM | note |
|---|---|---|---|
| gemm9 GA64 (production) | 28-32 | 730-760 | |
| gemm16 (128 rows x 256 tokens, plain int8, per token) | 35-39 | 1050-1090 | in-kernel Q4 conversion costs nothing in this orientation |
| gemm16 swap (128 tokens x 256 rows, CUTLASS orientation) | 39-42 | 1320-1350 | |
| gemm16 swap + L2::128B loads | 44.5-52.6 | 1200-1260 | |
| gemm17 w8, loads at k-group 0 (KNOB 1, used by R512) | 50.6-59.7 | 1290-1335 | |
| gemm17 in-place Q4 (swap) | 38.6-42.8 | 1140-1170 | conversion costs 10-15% in this orientation |
| gemm17 shift-folded GA64 (GSH 1) | 33-35 | 850-890 | 2 shifts per accumulator per stage |
| gemm17 per-512-block scales, I2F/F2I rescale (GSH 2) | 40.3-40.7 | 1040-1050 | exact (rel 5e-6), 20% slower |
| CUTLASS int8 128x256 (same sessions) | 44.6-59.7 | 1375-1490 | |

`.satfinite` and non-serpentine mma order: no gain. Rotated-weight converter (`rot::convert`): fp32 version 135 GB/s,
fp16x2 version 190-200 GB/s (Q4 read + int8 write), memory-bound.

## R512 engine profile (v31, pp2048, GPU0 = the faster GPU, ms per 2048-token batch)

| part | GA64 (v31) | R512 + cache (v31) |
|---|---|---|
| GEMMs | 1665 | 1165 |
| weight rotation (uncached 61%) | - | 64 |
| alpha/beta | 52 | 0 (rows appended to the qkvz GEMM) |
| quant (down input, attn out) | 3 | 26 |
| DeltaNet scan | 72 | 82 |
| attention + prep | 55 | 66 |
| conv / gnorm | 43 | 40 |
| AR wait + add_norm | 104 | 212 |
| batch total | 2003 | 1663 |

Under R512 load both GPUs settle near 640-720 MHz when equally loaded (v29: 638 / 645 MHz), against 900-1000 MHz under
GA64. The rotated int8 GEMM does about 1.75x the ops per clock of gemm9, so per-op energy is only about 25% lower and
the clock drops. On boxes where one GPU is less efficient it runs at about 700 MHz while the other idles 20-35% in
the all-reduce wait.

# Round 4 (2026-10-02/03, `otdoges/t4q-p` v35-v38, `otdoges/t4q-pg` v14-v16)

Raw outputs: `kaggle/p/out_v35` ... `kaggle/p/out_v38`, `kaggle/pg/out14` ... `kaggle/pg/out16` (git-ignored).

## Gate: NOT passed

All rows pass the correctness check (every prompt, top-1 equal, greedy 32 identical or diverging only at W's known
near-tie, last-token KL within max(2e-3, 2 x llama floor)). Same-run comparisons only; boxes differ by 5-10%.

| version | config | pp512 | pp2048 | note |
|---|---|---|---|---|
| v36 | round-3 default (last token through a decode step) | 893.4 | 1019.9 | same box as the next three rows |
| v36 | **+ pf_head 1** (last token in the batch, head only) | 947.1 | **1032.6** | |
| v36 | + pf_head 1, one sub-batch | **962.4** | 922.4 | pp512 prefers nsub 1, pp2048 nsub 2 |
| v36 | GA32 + pf_head 1 (accuracy reference) | 828.3 | 904.9 | W greedy diverges at 25 vs pf0 (near-tie) |
| v37 | new default (pf_head 1, nsub 2 only from 1024 tokens) | 906.2 | 1013.2 | GPU imbalance box (pp512 AR wait 142 ms) |
| v38 | new default | 923.2 | 988.5 | GPU1 ~7% slower than GPU0 |
| gate | | >= 1200 | >= 1400 | |

Round-3 best on the correct path was 870.5 / 1006.1 (v31).

## pf_head (v36)

The last prompt token used to run a full 64-layer decode step after the batch (34-35 ms, `total_s - batch_s`). It now
goes through the batch with the other tokens, and only the head runs decode-style: `tp::ar_norm` (output norm + q8 of
the final residual row), `tp::gemv` lm_head shard, `tp::argmax_step` (advances StepState pos/step/token and the ring
exactly like a decode step's head). KV, conv ring and DeltaNet state for the last token come from the batch path.

| prompt | KL pf0 vs pf1, head 0 | head 1 | KL vs llama.cpp batch logits, head 1 |
|---|---|---|---|
| P0 | 9.3e-4 | 1.0e-3 | 1.1e-3 |
| P1 | 1.4e-4 | 2.7e-4 | 5.3e-5 |
| W | 4.6e-4 | 2.4e-4 | 1.2e-3 |
| L | 1.1e-3 | 3.4e-4 | 2.1e-4 |

Greedy 32 tokens identical to the decode path on every prompt (W: both diverge from llama.cpp at its near-tie 25).

## Alpha/beta kernel (v35-v36)

`k_pf_ab` rewritten: 16 K slices (was 4), 128 threads with 8 tokens x 6 rows per thread, register prefetch of the
next k tile. pp2048: 51.6 -> 34 ms; pp512: 31.7 -> 10.8 ms. Changing the slice count changes the fp32 summation
order only, yet it moved the GA64 L metrics from (last-token KL 3.2e-4, cont mean 1.1e-3, cont max 6.7e-3) to
(1.1e-3, 4.9e-3, 3.9e-2): the L-prompt metrics are dominated by chaotic sensitivity, not by format accuracy (below).

## Accuracy: the L-prompt metrics are noise-dominated (v35, v36)

- A pure summation-order change (alpha/beta K slices 4 -> 16) changed GA64's L last-token KL 3.4x and cont mean 4.7x.
- GA32 (exact q8 blocks, strictly more accurate than GA64) vs GA64: hidden-state h_rel on L median 0.042, p99 0.86;
  cont max 4.0e-2 against GA64's 6.7e-2. The heavy h_rel tails that round 3 read as R512 inaccuracy appear between two
  accurate formats too.
- R512 is still clearly worse at the last token on L: subsets measured in v35 (all with GA64 elsewhere):

| R512 GEMM mask | L last-token KL | L cont mean / max | pp2048 | pp512 |
|---|---|---|---|---|
| none (GA64) | 1.1e-3 | 4.9e-3 / 3.9e-2 | 993.3 | 877.6 |
| 55 (all but down) | 3.9e-2 | 3.9e-2 / 0.62 | 1178.9 | 914.1 |
| 21 (qkvz, gateup, ssm_out) | 1.4e-1 | 2.7e-3 / 3.5e-2 | 1164.3 | 924.8 |
| 4 (gateup only) | 3.7e-2 | 3.3e-2 / 0.51 | 1062.0 | 872.2 |

  Even gateup alone on R512 fails L by 20x, so per-token rotated activations stay opt-in.

## Int4 tensor-core W4A8 GEMM (`kernels/gemm20.cuh`, pg v14)

Native Q4_0 codes as the s4 B operand (c ^ 8 per nibble, no requantization), GA64 activations split into
hi (s4) / lo (u4) nibble planes in the weights' nibble order, two m8n8k32 mmas per 32-block, exact fold per 64
(`v = fma(d0, P0', -(d0 + d1) MAGIC); v = fma(d1, P1', v); F = fma(a, v, F)`). Error vs the exact Q4_0 x q8 product
4.4e-4 rel L2 (gemm9: 6.3e-3). Sustained, both GPUs, gateup T = 2048:

| kernel | TOPS | ops/clk/SM | MHz | W |
|---|---|---|---|---|
| gemm9 GA64 (production) | 31.0-31.9 | 742-744 | 1041-1073 | 67-68 |
| gemm17 w8 per token (R512 GEMM) | 56.3-57.9 | 1296-1300 | 1082-1117 | 64-66 |
| gemm20 exact | 26.0-26.8 | 609-613 | 1060-1100 | 67-68 |
| gemm20 without the fold (timing only, wrong numbers) | 42.3-43.2 | 968-976 | 1082-1115 | 67 |

Even with no fold the int4 two-digit form does less work per joule than plain int8 (43 vs 57 TOPS at the same
power); the fold (2.5 ALU instructions per int4 mma) costs another 38%. CUTLASS's int4 GEMM (126 TOPS at 59 W, round 2)
is far more efficient than this kernel, but any exact W4A8 form needs the per-32 weight fold. Dead end.

## gemm9 numerics in the CUTLASS orientation (`kernels/gemm21.cuh`, pg v15-v16)

Tile-outer loop within each 64-k stage, MAGIC-started 4-mma chains folded right away (FB 0 exact: 1.7e-7 rel vs the
int8 mirror; FB 1 bias trick: 1.2e-4, same as gemm9). Sustained gateup T = 2048, both GPUs:

| kernel | v15 TOPS | v16 TOPS | ops/clk/SM |
|---|---|---|---|
| gemm9 GA64 | 30.7 / 31.0 | 29.2 / 31.4 | 743-749 |
| gemm21 128 tok x 256 rows, in-place Q4 | 31.7-32.1 | 30.9 / 33.5 | 741-812 |
| gemm21 256 tok x 128 rows | - | 25.9 / 27.7 | 606-608 |
| gemm21 pre-converted int8 weights | - | 28.9 / 30.9 | 691-695 |
| gemm17 w8 per token | 44.7 / 46.8 | 40.1 / 42.2 | 1384-1428 |

+3% over gemm9 at best, and removing the weight conversion does not help: the fold structure (short MAGIC chains,
255 registers, 4-80 B spills) is the cost, not the conversion. Not integrated.

## pf_gdnc 2 and pf_arc (v37, v38)

- `pf_gdnc 2`: the chunked DeltaNet's state-independent part (gates, P, T = (I - A)^-1) runs first for all chunks in
  parallel (`k_pf_gdnp`, grid 24 x chunks), the scan loads it. Bit-identical results, but 83.4 vs 78.8 ms: the T
  solve is not what limits the 24-block scan. Off by default.
- `pf_arc N`: K-split GEMMs (ssm_out, attn_out, down) in N token chunks, each chunk's peer copy issued as soon as it
  is done. Bit-identical. pp512 nsub 1: AR wait 138 -> 113-117 ms but the chunked GEMMs are slower; pp2048 nsub 1 +
  arc 4/8: 954-957 vs 988-999 with nsub 2. The AR wait is mostly GPU imbalance (the profiled GPU0 waits for a 6-10%
  slower GPU1), not copy time. Off by default.

## v36 pp2048 profile (GPU0, pf_head 1, ms per 2047-token batch)

GEMMs 1655 (gateup 779, down 348, qkvz 286, ssm_out 122, attn_qkv 83, attn_out 38); AR wait + add_norm 103; DeltaNet
scan 77; attention 47 + prep 7; ab 34; conv 22; gated norm 20; quant 3. Batch 1.981 s, total 1.983 s.
