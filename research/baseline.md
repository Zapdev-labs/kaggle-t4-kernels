# Kaggle 2x T4 baseline: upstream llama.cpp + Qwen3.8-27B

Kernel `otdoges/t4-qwen38-baseline` (private). v1 (2026-10-02 09:27 UTC) measured the box and microbenchmarks but the
llama.cpp CUDA configure failed (fix below). v2 (09:59 UTC, 2806 s wall) built llama.cpp master from source for sm_75
and ran every benchmark. v2 push first bounced for ~15 min with "Maximum batch GPU session count of 2 reached" because
`otdoges/cyber-frost-t4-fast` and `otdoges/cyber-frost-t4-ub4k` were RUNNING; the account allows 2 concurrent GPU sessions.

Files:
- Kernel dir: `/home/dih/kaggle-custom-kernals/kaggle/baseline/` (`t4-qwen38-baseline.py` is generated from `template.py` + `bw.cu`; `kernel-metadata.json`)
- v2 output: `/home/dih/kaggle-custom-kernals/kaggle/baseline/out/` (`results.json`, `logs/` with every llama-bench/server log and output samples, `bw.cu`)
- Reusable binaries: `out/llama-bin-sm75/` and `out/llama-bin-sm75.tgz` (137 MB, llama-bench/server/cli/speculative-simple/batched-bench + shared libs, sm_75, CUDA 12.8, NCCL on).
  On Kaggle attach with `"kernel_sources": ["otdoges/t4-qwen38-baseline"]` and they appear under `/kaggle/input/t4-qwen38-baseline/`; set `LD_LIBRARY_PATH` to that dir.
- v1 output: `/home/dih/kaggle-custom-kernals/kaggle/baseline/out_v1/`

## Headline

| config (UD-Q4_K_XL unless noted, -fa on, f16 KV, -ngl 99) | decode tok/s | prefill tok/s |
|---|---|---|
| prior KoboldCpp baseline (Q4_K_S finetune, 262k ctx, q8 KV) | ~9.2 | ~360 @25k |
| llama-bench `-sm layer` tg128 / pp512 | 13.13 | 310.8 |
| llama-bench `-sm tensor` tg128 / pp512 | **18.70** | **504.7** |
| llama-bench Q4_0 `-sm layer` | 14.66 | 310.1 |
| llama-bench Q4_0 `-sm tensor` | **21.11** | **516.0** |
| server, coding prompt, `-sm layer`, no spec | 12.54 / 12.66 | - |
| server, `-sm layer` + MTP draft (n_max 4) | 17.7-18.9 | - |
| server, `-sm tensor`, no spec | 16.86 / 16.93 | - |
| **server, `-sm tensor` + MTP draft (n_max 3)** | **28.8-32.4** | - |
| server, Q4_0 `-sm layer` + MTP (n_max 3) | 21.4-24.1 | - |

Best measured: **~30 tok/s decode** (UD-Q4_K_XL, `-sm tensor`, MTP draft, greedy coding prompt), 3.3x the KoboldCpp
baseline. Q4_0 + tensor + MTP was not run (my script only paired Q4_0 with layer split); from the ratios above it
should land around 34-36 tok/s.

## Box (measured, run 1, 2026-10-02)

| item | value |
|---|---|
| GPUs | 2x Tesla T4, sm_75, 40 SMs, 1590 MHz max SM clock, 15360 MiB, 70 W cap, L2 4 MiB, 64 KiB smem/block opt-in |
| Driver | 580.178.04 (CUDA driver API 13.0) |
| Toolkit | nvcc 12.8.93 at `/usr/local/cuda` (-> cuda-12.8); cmake 3.31.10, ninja 1.13, gcc 11.4 |
| OS | Ubuntu 22.04.5, kernel 6.18 |
| CPU | 4 vCPU Xeon @ 2.0 GHz, AVX-512 (f/bw/cd/dq/vl), FMA, F16C |
| RAM / disk | 31 GB RAM, no swap; `/` overlay 1.1 TB free; `/kaggle/working` 20 GB |
| Python stack | torch 2.10.0+cu128, NCCL 2.27.5 (torch) / system libnccl 2.25.1, triton 3.6.0, cupy 14.0.1, numba 0.60, pycuda, transformers 5.0.0; no vllm / flash_attn / xformers / bitsandbytes |
| Topology | `nvidia-smi topo -m`: GPU0<->GPU1 = **PHB** (through the CPU host bridge), PCIe gen3 x16 max |
| P2P | `nvidia-smi topo -p2p r` = OK; `cudaDeviceCanAccessPeer(0,1)=1`, `(1,0)=1`; torch agrees; perf rank 0, no native atomics |
| HF download | UD-Q4_K_XL 17.56 GB in 85 s (208 MB/s), Q4_0 16.06 GB in 70 s (230 MB/s), MTP 1.37 GB in 4 s |

## Microbenchmarks (bw.cu, nvcc -O3 -arch=sm_75)

| test | result |
|---|---|
| theoretical DRAM BW (5001 MHz x 256 bit x 2) | 320.1 GB/s |
| float4 `__ldg` read kernel, 1 GiB, best config (1280 blocks x 512) | **279.2 GB/s** (87% of peak; flat 275-279 across grid 80..1280, 256..1024 threads) |
| read kernel, 8 MB working set | 268.1 GB/s (31.3 us) |
| read kernel, 45 MB (one q4 FFN matrix) | 284.0 GB/s (166 us) |
| read kernel, 128 MB | 282.2 GB/s |
| cudaMemcpy D2D 1 GiB (read+write counted) | 241.3 GB/s |
| copy kernel (read+write) | 226.5 GB/s |
| empty kernel, back-to-back stream launches | 2.69 us/launch |
| empty kernel inside a CUDA graph | 0.81 us/kernel |
| launch + cudaDeviceSynchronize round trip | 7.03 us |
| 10 KB GPU0->GPU1 `cudaMemcpyPeerAsync`+sync, P2P **off** | 18.53 us (17.16 us pipelined) |
| 10 KB host-staged (D2H + H2D, pinned) | 18.26 us |
| 10 KB GPU0->GPU1 P2P **on** | **11.68 us** sync, **7.57 us** pipelined |
| 256 MB peer copy, P2P off / on | 9.81 / 9.90 GB/s |
| GPU1 kernel reading GPU0 memory directly (P2P) | 7.25 GB/s |
| pinned D2H / H2D 256 MB | 13.14 / 12.37 GB/s |

Takeaways for the custom engine:
- Weight-streaming roofline per GPU is ~279 GB/s, not 320. Bytes touched per decoded token (all `blk.*` + `output`,
  excluding `token_embd` which is a 1-row gather), summed from the GGUF tensor tables in this dir:
  UD-Q4_K_XL = 16.83 GB, Q4_0 = 15.33 GB. Layer split is sequential, so the bound is
  16.83 / 279 = 60.3 ms = **16.6 tok/s** (Q4_K_XL) and 15.33 / 279 = 55.0 ms = **18.2 tok/s** (Q4_0).
  Tensor parallel halves bytes per GPU: 30.2 ms (33 tok/s) and 27.5 ms (36 tok/s) before sync cost.
- Tensor parallel sync is cheap enough in principle: P2P works, a 10 KB activation hop costs ~7.6-11.7 us.
  hidden=5120 fp16 = 10 KB, so with 2 all-reduce style exchanges per layer x 64 layers = 128 hops x ~10 us
  = ~1.3 ms/token. That keeps TP at ~30 ms/token = ~33 tok/s bound, about 1.8x layer split.
- Kernel launch overhead is material: 2.69 us per eager launch vs 0.81 us in a graph. Decode with ~15-20 kernels
  per layer x 64 layers is ~1000+ launches = ~2.7 ms eager vs ~0.8 ms graphed. Use CUDA graphs (or a persistent kernel).

## Build fix for upstream llama.cpp on this image

Run 1 configure failed with `Target "ggml-cuda" links to: CUDA::cuda_driver but the target was not found`.
The image has no `libcuda.so` where FindCUDAToolkit looks. Fix used in v2: locate a stub or the driver library
(`/usr/local/cuda*/lib64/stubs/libcuda.so`, `ldconfig -p`, `find /usr /lib /opt`) and pass
`-DCUDA_cuda_driver_LIBRARY=<path> -DCUDAToolkit_ROOT=/usr/local/cuda`. The prebuilt fallback also failed because
GitHub's `releases/latest` for ggml-org/llama.cpp is now `v0.5.0` (no binaries); v2 picks the newest `b<NNNN>` tag
(b11344 at the time: `llama-b11344-bin-ubuntu-cuda-12.8-x64.tar.gz`).

llama.cpp master: v1 cloned `8d81559fa7b8`; v2 (the benchmarked build) cloned `a4cb4c61fd9d` (2026-10-02 11:56 +0200, #29818). libcuda used: `/usr/local/nvidia/lib64/libcuda.so` (first candidate; also present: `/usr/local/cuda-12.8/compat/libcuda.so*`). Build of 5 targets took 1622 s.
Configure took 124 s; NCCL was auto-detected (`Found NCCL: /usr/lib/x86_64-linux-gnu/libnccl.so`), which matters
for `-sm tensor`.

## How llama.cpp master runs Qwen3.8 MTP

- `--spec-type draft-mtp` enables MTP. With `-md <file>` it loads the head from a separate GGUF (unsloth ships
  `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`, 1.37 GB); without `-md` it builds the MTP context on the target model itself
  (works only if the main GGUF carries `blk.N.nextn.*` tensors). Code: `common/speculative.cpp`
  (`common_speculative_impl_draft_mtp`, context type `LLAMA_CONTEXT_TYPE_MTP`).
- Draft length: `--spec-draft-n-max` (default 3), per request `"speculative.n_max"`. Server timings report
  `draft_n` and `draft_n_accepted`.
- `-sm` accepts `none|layer|row|tensor` (tensor = EXPERIMENTAL; splits weights and KV).

## llama-bench results (llama.cpp master a4cb4c61, built on Kaggle in 1622 s with ninja -j4)

`llama-bench -m M -ngl 99 -fa 1 -t 4 -p 512 -n 128 -r 3 -sm <mode>`; KV f16, ubatch 512.

| model | -sm | pp512 tok/s | tg128 tok/s | tg ms/token | % of BW roofline |
|---|---|---|---|---|---|
| UD-Q4_K_XL (17.56 GB) | layer | 310.80 +- 1.27 | 13.13 +- 0.01 | 76.2 | 79% of 60.3 ms |
| UD-Q4_K_XL | row | FAIL: `device CUDA0 does not support split buffers` | | | |
| UD-Q4_K_XL | tensor | 504.66 +- 1.96 | 18.70 +- 0.28 | 53.5 | 56% of 30.2 ms |
| UD-Q4_K_XL | layer, depth 16384 | 221.28 +- 12.48 | 12.29 +- 0.05 | 81.4 | |
| Q4_0 (16.06 GB) | layer | 310.08 +- 0.19 | 14.66 +- 0.07 | 68.2 | 81% of 55.0 ms |
| Q4_0 | row | FAIL (same split-buffer error) | | | |
| Q4_0 | tensor | 515.99 +- 0.59 | 21.11 +- 0.04 | 47.4 | 58% of 27.5 ms |

Prefill vs ubatch (UD-Q4_K_XL, layer, pp2048, -b 2048): ub 512 = **424.97**, ub 1024 = 338.14, ub 2048 = 259.96 tok/s.
Bigger ubatches are slower on T4 here; keep -ub 512.

Notes:
- `-sm row` is broken on master for this model/build (model load error, not OOM).
- `-sm tensor` (EXPERIMENTAL in master) works for qwen35, uses NCCL (found at configure), and gives +42% decode and
  +62% prefill over layer split. It only reaches 56-58% of its roofline, so per-layer sync + small-kernel overhead
  dominates the remaining gap; that is the gap a custom engine can attack.
- Decode at 16k depth loses only 6% (12.29 vs 13.13): 48 of 64 layers are DeltaNet with O(1) state, so KV reads
  barely matter at these lengths.

## llama-server results (coding prompts, greedy)

`llama-server -ngl 99 -sm <mode> -fa on -c 16384 -np 1 -fit off -t 4 -cram 0`, MTP adds
`--spec-type draft-mtp -md MTP/mtp-Qwen3.8-27B-Q4_0.gguf --spec-draft-n-max N --spec-draft-ngl 99`.
Requests: `/v1/chat/completions`, temperature 0, top_k 1, `enable_thinking: false`, max_tokens 512, two prompts
(P0 = thread-safe LRU cache module with unittest, P1 = Apache log analyzer). All runs produced 512 tokens of real Python.

| model / -sm | spec | P0 decode tok/s | P1 decode tok/s | draft accept (P0 / P1) | mean accepted len | GPU mem MiB (gpu0,gpu1) |
|---|---|---|---|---|---|---|
| Q4_K_XL / layer | none | 12.54 | 12.66 | - | - | 8161, 9441 |
| Q4_K_XL / layer | MTP, n_max 4 | 18.39-18.85 | 17.73-18.28 | 0.757-0.765 / 0.745 | 3.96-4.06 | 8465, 11169 |
| Q4_K_XL / tensor | none | 16.86 | 16.93 | - | - | 8803, 8803 |
| Q4_K_XL / tensor | MTP, n_max 3 | **32.43 / 31.10** | **28.79 / 30.34** | 0.847 / 0.778 | 3.33-3.54 | 9757, 9757 |
| Q4_0 / layer | MTP, n_max 3 | 24.10 / 23.71 | 21.40 / 22.10 | 0.829 / 0.734 | 3.19-3.50 | 8089, 9961 |

- MTP speedup: 1.46x on layer split, **1.8x on tensor split**.
- Per-request `"speculative.n_max"` had no effect; the startup `--spec-draft-n-max` wins (layer logs show mean len 4.0
  with n_max 4, tensor logs 3.3-3.5 with n_max 3, identical across "n_max 1..4" requests). So the n_max sweep is not a
  real sweep; a proper sweep needs a server restart per value.
- Log warning with MTP: `spec common_specu: backend offload failed for seq_id=0; using CPU sampler` (draft sampling
  runs on the CPU; a small cost on a 2.0 GHz vCPU).
- Greedy output with tensor+MTP is byte-identical to tensor no-spec and layer no-spec (same md5 of first 1500 chars);
  layer+MTP diverges slightly (batched verify numerics).
- Server prefill numbers (110-190 tok/s) are for 72-81 token prompts and are not meaningful; use llama-bench pp.

## What this means for a custom engine

- Target to beat: **18.7 tok/s plain / ~30 tok/s with MTP** (Q4_K_XL, tensor split); **21.1 tok/s plain** with Q4_0.
- Rooflines at 279 GB/s per GPU: Q4_K_XL TP 33 tok/s, Q4_0 TP 36 tok/s plain; with MTP at ~3.4 accepted tokens per
  verify step and a verify step costing roughly one weight pass, the ceiling is ~100 tok/s, realistically 60-80.
- llama.cpp tensor mode reaches only 56-58% of roofline; layer mode reaches 79-81%. A fused, graph-captured TP
  engine with P2P (7.6-11.7 us per 10 KB hop) instead of NCCL should close most of that.
