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
