# M0 results: GEMV bench and box probe (Kaggle 2x T4)

Kernel `t4q-m0` (private), three versions run on 2026-10-02:
- v1 at 11:34 UTC: first sweep, probe and NCCL.
- v2 at 11:46 UTC: added RPL=4, the CVT variant, and rotated sustained A/B runs.
- v3 at 11:55 UTC: per-shape RPL A/B under sustained load and the two-level-flag AR probe.

Every number below was measured on Kaggle. Raw outputs are in `kaggle/m0/out` (v1), `kaggle/m0/out_v2` and `kaggle/m0/out_v3`, each with `results.json`, `logs/gemv_bench.txt` (one JSON row per measurement), `logs/sustain_dev*.txt`, `logs/probe.txt`, `logs/nccl_*.txt` and `logs/smi_monitor.csv`.

Code:
- `t4q/src/kernels/gemv.cuh`: fast GEMV, host and device repack, q8_1 quantizer.
- `t4q/tools/gemv_bench.cu`, `t4q/tools/probe.cu`, `t4q/tools/nvml_lite.h`.
- `t4q/tools/gemv_layout_check.cpp`: a CPU emulation of the kernel's addressing and integer math.
- `t4q/tools/stage_m0.py` and `t4q/tools/mkkernel.py`.

## Gate verdict: PASS (burst), with a sustained-load caveat

| gate item | requirement | measured (v1 / v2 / v3) | |
|---|---|---|---|
| P4 m=1 | >= 250 GB/s | **258.3 / 258.5 / 258.4** decode-mix, one launch config for all shapes | pass |
| P4 m=4 | >= 230 GB/s | **247.6 / 251.4 / 250.5** decode-mix, same config | pass |
| AR path chosen with measured latency | | P2P remote-store mailbox: 5.8 us one-way for 20 KB with 8 writer blocks; 14.6-14.8 us p50 for a full symmetric 20 KB exchange read by 40 blocks; 0 errors | pass |

Notes on the gate:
- "Decode-mix" is total bytes divided by total time over the per-GPU TP decode GEMVs, weighted by how often each runs per token: gateup x64, qkvzab_tp x48, attn_qkv_tp x16, down_tp x64, out_tp x64.
- Per shape, every P4 shape clears 250 at m=1 except `out_tp` (3072 x 5120, 8.8 MB, a 35 us kernel), which measured 252.7 / 247.3 / 247.1. Every P4 shape clears 230 at m=4 (minimum 237.9 in v3).
- Correctness held in every version:
  - The device repack equals the host repack byte for byte.
  - The GPU q8_1 quantization equals the CPU mirror exactly.
  - GEMV versus the fp64 CPU reference on the same q8 activations gives max error / rms(ref) of 4.6e-7 to 6.1e-7. All rows were checked except on the lm_head shapes (5404 sampled rows there).
  - Every config (RPL, CVT, XSM, threads, grid) at every m returns results bit-identical to the m=8 golden run, on every row.
  - Error against unquantized fp32 x is rel L2 0.0083-0.0087, all from q8_1 activation quantization.

**Caveat (important for M4/M5).** Under sustained back-to-back load, the 70 W software power cap pulls the SM clock down to 300-900 MHz on these T4s. A pure streaming read keeps 1350-1590 MHz at the same power. GPU0 runs about 20 C hotter (77-81 C) than GPU1 on all three boxes and throttles hardest.
- m=1 P4 stays memory-bound even at 420 MHz: 251-259 GB/s sustained.
- m=4 becomes ALU-bound when throttled: 161-202 GB/s on GPU0 with RPL=2, and 191-206 with RPL=4.
- K6 (lm_head) at m=1 fell to 152-207 GB/s at 300-340 MHz on GPU0.

## Box facts (all three boxes)

- 2x Tesla T4, 40 SMs, driver 580, nvcc 12.8. P2P is reported both ways, with no native P2P atomics and perf rank 0.
- Streaming read (float4 `__ldg`, 512 MB): **282.1-282.4 GB/s**, with both GPUs at 281.8-282.2 sustained. D2D memcpy runs 234.7-236.4 GB/s (read+write).
- Clocks under 12 s of sustained streaming on both GPUs:
  - GPU0: 1350-1440 MHz at 68 W, 78-81 C, swpower.
  - GPU1: 1455-1590 MHz at 68-70 W, 60-75 C.

## Kernel design (as built)

- **Layout.** A warp owns a tile of 2*RPL rows. Half-warp h owns RPL rows, and lane j of the half owns 32-weight group `c*16 + j` of 512-weight chunk c.
  - Code planes are `[tile][chunk][r][(part)][lane][16 B]`, so every load instruction reads 512 contiguous bytes. Scale planes are `[tile][chunk][lane][r]`.
  - The layout is lossless and costs the same bytes as GGUF: P4 18 B / 32 weights, Q8 34 B / 32, K6 210 B / 256.
- **K6** repacks Q6_K's 6-bit codes into 16 B of low nibbles (Q4_0 order) plus 8 B of high bits per 32 weights. Each word is laid out so that one shift and mask puts the high bits at bit 4 of each byte. Q6_K's int8 sub-scales and fp16 d stay native.
- **Activations** use q8_1 per 32 (int8 plus fp32 d plus two 16-element int sums). The P4 offset is applied as `sumi - 8*S`, and K6 as `scA*(sa - 32*s0) + scB*(sb - 32*s1)`.
- **Ordering.** Each lane accumulates in fp32 in chunk order, then an xor tree runs across the 16 lanes. The order depends neither on m nor on any launch knob, so the per-column bits are identical for m = 1..8 (needed for V4 and V5).
- **Template knobs** in `gemv_fast_kernel<FMT, RPL, M, NCH, CVT, XSM>`:
  - RPL 2 or 4.
  - CVT 0 uses I2F. CVT 1 uses an exact magic-number int to float that folds in the -8S offset. They are bit-identical.
  - XSM stages x in shared memory once per block.
  - `__launch_bounds__(256, 2)` (128-register cap).
  - Runtime knobs are 128 or 256 threads, and a full or persistent grid.
- **Measured dead ends:**
  - RPL=1 loses at m>=3 (v1).
  - A 64-register cap (`__launch_bounds__(256,4)`) is worse on every shape (v1).
  - The register ring depth D compiled to identical SASS (nvcc schedules the loads itself).
  - CVT=1 made no measurable difference (I2F is not the bottleneck).
  - XSM only helps at m>=6.

## GEMV burst results (v3, best config per cell, GB/s of packed weight bytes)

| shape (per GPU, TP) | fmt | K x N | MB | m=1 | m=2 | m=3 | m=4 | m=5 | m=6 | m=7 | m=8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gateup_tp | P4 | 5120 x 17408 | 50.1 | 262.4 | 262.7 | 261.3 | 257.3 | 253.2 | 247.8 | 226.9 | 213.9 |
| qkvzab_full | P4 | 5120 x 16480 | 47.5 | 263.0 | 262.5 | 260.2 | 256.7 | 253.3 | 247.7 | 227.8 | 211.8 |
| qkvzab_tp | P4 | 5120 x 8240 | 23.7 | 259.1 | 257.1 | 255.8 | 251.5 | 239.1 | 233.5 | 230.6 | 189.9 |
| attn_qkv_tp | P4 | 5120 x 7168 | 20.6 | 257.5 | 256.3 | 254.8 | 250.5 | 232.8 | 231.5 | 208.0 | 199.1 |
| down_tp | P4 | 8704 x 5120 | 25.1 | 261.0 | 258.6 | 256.5 | 247.9 | 240.8 | 227.0 | 203.9 | 176.7 |
| out_tp | P4 | 3072 x 5120 | 8.8 | 247.1 | 243.0 | 241.3 | 237.9 | 224.3 | 217.5 | 205.5 | 195.5 |
| lmhead_tp_k6 | K6 (Q6_K) | 5120 x 124160 | 521.5 | 265.2 | 265.1 | 262.5 | 261.0 | 257.3 | 228.5 | 220.1 | 204.3 |
| lmhead_tp_q8 | Q8 (Q8_0) | 5120 x 124160 | 675.4 | 266.4 | 265.1 | 263.4 | 264.2 | 262.4 | 262.0 | 260.6 | 259.8 |

The P4 decode-mix with one launch config (`rpl 2, I2F, no XSM, 128 threads, full grid`): m1 258.4, m2 258.6, m3 255.9, m4 250.5, m5 240.9, m6 226.0, m7 207.8, m8 189.0. With the best config per shape: m1 259.9, m4 252.1, m8 197.7. v1 and v2 agree within about 1%.

Times at m=1: gateup 190 us, down 96 us, qkvzab_tp 91 us, attn_qkv 80 us, out 35 us, lm_head K6 1.96 ms.

## Sustained load (both GPUs at once, 6 s per config, rotated forward then reverse, mean of 2 reps)

Columns are GB/s / SM MHz / W / bytes per SM clock.

| v3 | GPU0 RPL=2 | GPU0 RPL=4 | GPU1 RPL=2 | GPU1 RPL=4 |
|---|---|---|---|---|
| down_tp m=1 | 254.0 / 638 / 66.7 | 251.8 / 608 / 66.6 | 258.8 / 1118 / 69.1 | 257.1 / 1103 / 69.2 |
| out_tp m=1 | 233.2 / 818 / 65.9 | 229.0 / 825 / 66.0 | 243.0 / 1208 / 69.0 | 237.1 / 1178 / 68.8 |
| down_tp m=4 | 165.8 / 623 | **205.7** / 705 | 221.2 / 878 | 220.2 / 1140 |
| gateup_tp m=4 | 167.2 / 653 | **203.7** / 488 | 213.6 / 833 | **252.7** / 885 |
| qkvzab_tp m=4 | 161.0 / 630 | **201.8** / 555 | 213.1 / 840 | **240.4** / 983 |
| out_tp m=4 | 162.7 / 705 | **191.2** / 548 | 211.1 / 960 | **222.9** / 1028 |
| lmhead_k6 m=4 | 170.4 / 465 | - | 217.0 / 585 | - |

| v2 (gateup_tp unless noted) | GPU0 | GPU1 |
|---|---|---|
| m=1 RPL2 / RPL4 | 257.8 / 258.0 GB/s at 600-675 MHz | 259.3 / 259.9 at 840-885 MHz |
| m=4 RPL2 (I2F / magic) | 191.6 / 189.3 at 745 MHz (6.4 B/clk/SM) | 202.0 / 200.5 |
| m=4 RPL4 (I2F / magic / magic+XSM) | 221.4 / 220.5 / 226.4 at 510-525 MHz (10.5-11.1 B/clk/SM) | 231.8 / 231.7 / 234.1 |
| m=8 RPL2+XSM / RPL4+XSM | 138.2 / 118.8 | 147.9 / 120.6 |
| lmhead_k6 m=1 RPL2 / RPL4 | 203.3 / 207.6 at 338 MHz | 260.7 / 259.8 |

v1 (GPU0 hot): gateup m=1 251.6 GB/s at 420 MHz, gateup m=4 157.2 at 615 MHz, and lmhead K6 m=1 152.4 at 300 MHz.

**Engine policy from this data:**
- **Repack every P4 tensor with RPL=4.** It costs at most 1-2% at m=1 and gains 20-25% sustained at m=4 on the throttled GPU.
- Launch with 256 threads and a persistent grid (`min(ntiles/8, occupancy*40)` blocks), CVT=1, and XSM=0 (XSM=1 only for m>=6).
- Keep K6 at RPL=2 (RPL=4 spills at m>=6).

Static SASS mix per 32 weights per lane-row (RPL=2):
- P4 m=1: 37 instructions (8 IDP, 10 LOP3, 5 SHF, 3 LDG, 1 I2F).
- P4 m=4: 86 (32 IDP, 7.5 LDG, 4 I2F).
- K6 m=1: 66 (20 LOP3, 13 IMAD, 11 SHF, 8 IDP).
- Q8 m=1: 25.

A float4 streaming read issues about 8. That gap explains the power-cap clock sag.

## All-reduce probe (5120 fp32 = 20 KB unless noted; p50 / p99 us)

| path | test | v1 | v2 | v3 |
|---|---|---|---|---|
| P2P remote stores, flag only | one-way (ping-pong / 2) | 4.06 / 4.24 | 3.81 / 4.11 | 3.60 / 4.10 |
| P2P, 1 writer block, 10 KB / 20 KB | one-way | 7.23 / 9.23 | 6.82 / 9.25 | 6.45 / 9.26 |
| P2P, 8 writer blocks, 10 KB / 20 KB | one-way | - | **4.48 / 5.78** | 4.54 / 5.82 |
| P2P, 40 writer blocks, per-block flags, 10 / 20 KB | one-way | - | 6.59 / 7.57 | 6.94 / 7.78 |
| P2P, two-level flag (local atomic counter, 1 remote flag), 8 / 40 / 64 blocks, 20 KB | one-way | - | - | 7.30 / 7.87 / 8.08 |
| P2P symmetric exchange, 40 blocks, each block reads all 20 KB | AR round | 14.78 / 16.45 | 14.72 / 16.45 | 14.85 / 16.42 |
| same, each block reads only its slice | AR round | 10.24 / 14.62 | 10.24 / 14.34 | 10.24 / 14.37 |
| P2P exchange, 1 block | AR round | 10.27 / 12.29 | 10.27 / 12.29 | 10.27 / 12.29 |
| host-mapped pinned, 1 block, 10 / 20 KB | one-way | 13.26 / 23.55 | 14.32 / 24.82 | - |
| host-mapped exchange 40 blocks | AR round | 321.9 | 319.5 | - |
| cudaMemcpyPeerAsync 20 KB | host-synced / pipelined | 9.38 / 3.89 | 8.63 / 3.71 | - |
| NCCL 2.25.1 f32 default | host-synced p50 / p99 / pipelined | 36.0 / 57.9 / 22.7 | 36.6 / 62.0 / 22.3 | 38.6 / 65.6 / 23.4 |
| NCCL `NCCL_P2P_LEVEL=SYS` | same | 27.2 / 52.1 / 15.7 | 28.0 / 58.7 / 15.8 | 28.2 / 59.2 / 15.4 |
| NCCL `P2P_LEVEL=SYS PROTO=LL` | same | 27.4 / 54.6 / 15.6 | 27.9 / 56.9 / 16.3 | 28.0 / 41.9 / 15.4 |
| NCCL f16 (10 KB), `P2P_LEVEL=SYS` | same | 24.3 / 51.2 / 12.3 | - | - |

Zero data or ordering errors in any mailbox test. Every payload word was checked against its epoch. No watchdog fired.

**Decision: primary AR = P2P remote-store mailbox** (DESIGN 4.1).
- The producer writes its partial into the peer's VRAM, then `__threadfence_system()`, then a flag store.
- Host-mapped memory is rejected: polling or reading a 20 KB payload over PCIe costs 24 us per hop and collapses with many reader blocks.
- NCCL stays the fallback, at 28 us synced and 15.4 us pipelined at best, with `NCCL_P2P_LEVEL=SYS`.

What this means for M4:
- The latency floor is about 3.6-4 us per hop. A 20 KB payload costs about 5.8 us one-way when written by about 8 blocks. 10 KB fp16 costs 4.5 us.
- 128 ARs per token at about 6 us is 0.77 ms if fully exposed. So the weight prefetch overlap in DESIGN 4.1 is what decides the final number (2-4 us exposed per AR expected).
- Waiting on many per-block flags costs about 2 us more (40 flags vs 8). The two-level flag (local counter, last block fences and writes one flag) is equally correct and stays at 7.3-8.1 us with 8-64 producers. Use it when the producer is a GEMV epilogue with many blocks, or have a few blocks own the publish.
- Reading the full 20 KB vector in each of 40 consumer blocks adds about 4.5 us (14.8 vs 10.2). The prefetch overlap matters there as well.
- The 4 us floor is per hop. A mailbox exchange costs about 10 us of round trip when nothing overlaps.
