# Qwen3.8-27B decode on Kaggle 2x T4: kernel research and plan

Target: batch-1 (and batch 2-8 for speculative verify) decode of Qwen3.8-27B (qwen35 arch) on 2x Tesla T4 (sm_75), as fast as possible.
Baseline to beat: KoboldCpp/llama.cpp CUDA, layer split, Q4_K_S, q8 KV: **9.2 tok/s** at 25k ctx.

## 0. TL;DR

- Decode is pure weight streaming. Real bytes per token (parsed from the unsloth GGUF headers, main model only, excluding token_embd and the MTP block 64):
  UD-IQ4_XS **13.34 GB**, UD-Q4_K_S 14.45 GB, Q4_0 15.07 GB, UD-Q4_K_M 15.39 GB, UD-Q4_K_XL 16.48 GB, UD-Q3_K_XL 12.24 GB.
- The 9.2 tok/s baseline works out to about **144 GB/s effective, about 45% of one T4's 320 GB/s peak**, because layer split runs the GPUs one after the other, so only one T4's bandwidth is ever in use.
- A realistic GEMV on T4 reaches **240-280 GB/s (75-87%)**. Reported T4 streaming read is about 277 GB/s. 90% of peak is not realistic, so plan around 260.
- **Tensor parallel (TP=2) roughly doubles the ceiling** because each GPU streams half the weights. The cost is 128 small all-reduces per token (2 per layer, 20 KB fp32 or 10 KB fp16 each).
  - Ceiling at 260 GB/s and 4k ctx, UD-IQ4_XS: layer split **18.6 tok/s**, TP with an 8 us all-reduce **35.5**, TP with a 30 us NCCL all-reduce **32.3**.
  - With MTP speculative decoding (k=2-3, about 75-82% acceptance): roughly **x1.8-2.2 on top**, so **55-70 tok/s** is the stretch ceiling and **40+ tok/s** is a realistic target.
- Kaggle T4 pairs show `PHB` (through the CPU host bridge, no NVLink). P2P may be reported but cannot be assumed. Probe it first (script in section 3). For the all-reduce, use a **custom spin-flag all-reduce in host-mapped pinned memory** (zero-copy). It works whether or not P2P is available, runs inside CUDA graphs, and should take about 5-10 us. NCCL is the fallback.
- Kernel strategy on Turing: **int8 activations (q8_1) + dp4a** for batch-1 GEMV (the llama.cpp mmvq approach; it uses the least ALU per weight on sm_75). For verify batches of 2-8, use **int4 to fp16 lop3 dequant + `mma.m16n8k8` with swapped A/B** (weights as M=16, tokens as N=8). Marlin is sm_80+ only. ExLlamaV3 has community sm_75 paths.
- DeltaNet recurrent state is 302 MB/token of read+write at fp32 (151 MB per GPU under TP). That is about 2% of traffic, so keep it fp32, do one read and one write per element, and handle all k+1 verify tokens in a single pass.

## 1. Model facts that matter for kernels (verified against the GGUF header)

`general.architecture=qwen35`, `block_count=65`: 64 main blocks plus **blk.64, which is the MTP block** (full attention + FFN, 0.425 G params), with `nextn.eh_proj/enorm/hnorm/shared_head_norm`. hidden 5120, FFN 17408, vocab 248320, `rope.dimension_count=64`, `freq_base=1e7`, `rms_eps=1e-6`, `ssm.conv_kernel=4`, `ssm.state_size=128`, `ssm.group_count=16`, `ssm.time_step_rank=48`, `ssm.inner_size=6144`, `full_attention_interval=4`.

Matrices per layer (K -> N):

| layer kind (count) | tensor | shape | params |
|---|---|---|---|
| DeltaNet (48) | attn_qkv | 5120 -> 10240 (q 2048, k 2048, v 6144) | 52.4M |
| | attn_gate (z) | 5120 -> 6144 | 31.5M |
| | ssm_alpha, ssm_beta | 5120 -> 48 each | 0.5M |
| | ssm_out | 6144 -> 5120 | 31.5M |
| Full attn (16) | attn_q (q + output gate interleaved per head) | 5120 -> 12288 | 62.9M |
| | attn_k, attn_v | 5120 -> 1024 each | 10.5M |
| | attn_output | 6144 -> 5120 | 31.5M |
| all (64) | ffn_gate, ffn_up | 5120 -> 17408 each | 178.3M |
| | ffn_down | 17408 -> 5120 | 89.1M |
| head | output (lm_head) | 5120 -> 248320 | 1271M |

Totals: layers 24.353 G, lm_head 1.271 G, token_embd 1.271 G (a row gather only; it can live in host memory), MTP block 0.425 G.

Bytes per token by quant (parsed with an HTTP range read of each GGUF header):

| GGUF | layers GB (bpw) | lm_head GB (type) | **decode GB/token** | file GB |
|---|---|---|---|---|
| UD-Q3_K_XL | 11.36 (3.73) | 0.874 (Q5_K) | **12.24** | 13.15 |
| UD-IQ4_XS | 12.47 (4.10) | 0.874 (Q5_K) | **13.34** | 14.25 |
| UD-Q4_K_S | 13.41 (4.40) | 1.043 (Q6_K) | **14.45** | 15.36 |
| Q4_0 | 14.02 (4.61) | 1.043 (Q6_K) | **15.07** | 16.06 |
| UD-Q4_K_M | 14.34 (4.71) | 1.043 (Q6_K) | **15.39** | 16.46 |
| UD-Q4_K_XL | 15.44 (5.07) | 1.043 (Q6_K) | **16.48** | 17.56 |
| custom int4 g64 sym fp16 scale + 4.5 bpw head | 12.94 (4.25) | 0.715 | **13.65** | - |

Kernel-coverage warning: the UD-* files mix many block types even inside one tensor class. UD-IQ4_XS uses IQ4_XS, IQ4_NL, IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS, Q2_K..Q6_K, and Q8_0; UD-Q4_K_M similarly. **Q4_0 is nearly homogeneous**: Q4_0, plus Q4_1 on some ffn_down, Q5_K on ssm_out, Q6_K on lm_head, F32 ssm_alpha/beta, and Q8_0 eh_proj. That makes it the fastest path to a correct engine.

Semantics to replicate (from `~/oxidize/oxidize-c/src/backends/cuda_qwen35.cu`, `src/model/qwen35_delta.c`):
- DeltaNet: causal conv1d (4 taps, state of 3 past inputs) over the 10240 qkv channels, then SiLU. L2-normalize q and k per key head (eps 1e-6). q *= 1/sqrt(128).
  - `beta = sigmoid(b)`, `decay = exp(ssm_a * softplus(alpha + dt_bias))`, where GGUF `ssm_a = -exp(A_log)`.
  - Per value head h, using key head `h % 16` (**tiled GQA, ggml_repeat order**, as the oxidize comment notes):
    `S = decay*S; d = (v - S k) * beta; S += d k^T; o = S q`.
  - Then a per-head RMSNorm(128, `ssm_norm`) * SiLU(z), then ssm_out.
- Full attention: attn_q gives q and the output gate per head. q_norm and k_norm are RMSNorm over 256. RoPE applies to the first 64 dims only (partial 0.25). For text-only positions the interleaved MRoPE sections [11,11,10] collapse to plain RoPE with `inv_freq_i = 1e7^(-2i/64)`, i < 32 (check the neox vs interleave pair layout against oxidize `k_qk_norm_rope`). Scale 1/16. `o *= sigmoid(gate)`, then attn_output.

## 2. Quantized GEMV on sm_75

### 2.1 What exists

| kernel family | sm_75? | notes |
|---|---|---|
| llama.cpp `mmvq` (`ggml/src/ggml-cuda/mmvq.cu`, `vecdotq.cuh`) | yes | Activations quantized to q8_1 (int8 per 32, plus fp16 d and sum). `vec_dot_*_q8_1` uses `dp4a` (`__dp4a`). It supports ncols 1..8 (`MMVQ_MAX_BATCH_SIZE`), so the same kernel serves spec verify. IQ4_XS/IQ4_NL use a 16-entry int8 codebook lookup via `__byte_perm` (prmt), which costs about 2 prmt per 4 weights. Weak point: it reads 4-byte ints out of AoS block structs (Q4_K is 144 B/256 w), not 16-byte vector loads. |
| exllamav2 | yes (CC 6.x+) | GPTQ/EXL2 GEMV with fp16 dequant (lop3), reorders rows/groups for coalescing. A good source for the layout ideas. |
| exllamav3 (EXL3/QTIP trellis) | **community sm_75 paths**: upstream PR #325, issues #411/#417, fork rafatxf/exllamav3 | Turing has no `cp.async`, its `mma` is m16n8k8 (2 instructions per m16n8k16 tile), and shared memory is capped at 64 KB. On a 2080 Ti (616 GB/s) the fork's m<=8 GEMV streams **300-450 GB/s (49-73%)**. Qwen 27B 4.0 bpw decode went 17.9 -> 28.2 tok/s at 4k and 11.9 -> 25.9 at 64k. It includes 4-bit-cache flash decoding and GDN kernels. Its trellis decode is ALU-heavy, which hurts on a 70 W T4. |
| AWQ GEMV / GEMM (llm-awq, AutoAWQ) | yes | int4 g128 asym, fp16 dequant + HFMA2. Vector loads are fine, but the ALU per weight is higher than dp4a. |
| GPTQ Marlin / Machete | **no** (sm_80+: cp.async, m16n8k16, bf16) | Not usable. |
| FlashAttention-2 | no (sm_80+) | Write our own decode attention. |

### 2.2 Bandwidth reality on T4

- Spec 320 GB/s (256-bit GDDR6 at 10 Gbps effective). Reported streaming read on T4 is about **277 GB/s (87%)**. Expect a well-tuned GEMV at **250-270 GB/s**. 300+ is not realistic.
- The T4 is 70 W. Memory clock is fixed (5001 MHz), but SM clocks sag under load (Colab T4s have been seen at 600-800 MHz against a 1590 MHz boost). That hurts ALU-heavy dequant formats (IQ*, trellis) much more than Q4_0/dp4a. Measure `nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,clocks_throttle_reasons.active --format=csv -lms 100` during a run.
- Little's law: about 280 GB/s x about 1 us loaded latency = about 280 KB in flight device-wide, so **at least 8 KB per SM, aim for 16-32 KB**. For example, 256 threads x 16 B x 4-deep unroll = 16 KB per block, with 2 blocks per SM.
- ALU budget per GPU per token under TP: about 12.8 G weights at 1.4 GHz x 40 SM x 64 lanes = 3.6 T lane-ops/s.
  - dp4a path: about 0.5-0.6 instr/weight (lop3/shift unpack about 0.25, dp4a 0.25, scale FMA amortized), about 2 ms. Hidden under about 25 ms of memory time.
  - fp16 path (lop3 magic + HFMA2): about 1.5 instr/weight, about 5-6 ms. Still hidden at m=1, but it scales with m.
  - For m=4-8 verify, dequant once and then use either dp4a xm or mma.

### 2.3 Ideal weight layout ("repack at load")

GGUF blocks are AoS: Q4_0 is 18 B = fp16 d + 16 B nibbles; Q4_K is 144 B per 256. Repack at load into SoA, warp-tiled:

```
for each row-tile of R rows (R = 2 or 4 per warp), for each 1024-weight K-chunk:
  qs:     [R][32 lanes][16 B]   // lane l holds 32 weights at k = chunk*1024 + l*32 .. +31, i.e. one Q4_0 block per lane
  scales: [R][32] fp16          // one 64-B line per row per chunk, read as one u32 per lane pair or prefetched into smem
```

- Each lane does one `ld.global.nc.v4.u32` (`__ldg(int4*)`) per row per chunk, so a warp reads 512 contiguous bytes per instruction. That is fully coalesced, sector-perfect, and needs no shared-memory staging.
- Activations x are quantized once per GEMV in the prologue (q8_1: int8 x 32 + fp16 d, s) into shared memory (5120 B for K=5120, 17408 B for ffn_down).
- For Q4_0: `sumi = dp4a(lo_nibbles, xq[0:4]) + ...`, `acc += d_w * (d_x*sumi - 8*s_x)` (llama.cpp formula: `d4 * (sumi * ds8.x - (8*vdr/QI4_0) * ds8.y)`).
- For Q4_K/Q5_K/Q6_K: same tiling at 256-weight superblocks, 8 lanes per superblock, with the 6-bit sub-scales unpacked once per superblock (amortized over 256 weights).
- Grid: one warp per R rows. N >= 5120 everywhere after fusion, giving 1280-2560 warps against 40 SM x 32 warps resident, so 1-2 waves. Use **row-interleaved assignment** so the last wave is not a tail. Avoid split-K except for small N.
- Format choice:
  - **int4 + fp16 group scale (g32, sym) = 4.5 bpw.** Simplest and fastest to decode. Quality is about Q4_0.
  - **Q4_K (asym g32, 6-bit scales/mins, fp16 super d/dmin) = 4.5 bpw.** Better quality at the same bytes; unpacking costs a few extra instructions per 32 weights. Asymmetric is fine with dp4a because the min term uses the precomputed q8_1 block sum.
  - **IQ4_XS = 4.25 bpw.** 6% fewer bytes than Q4_K, about Q4_K_S quality, a prmt codebook lookup, about 2x the ALU of Q4_0. Fine at m=1 if SM clocks hold.
  - **Custom int4 g64 sym fp16 scale = 4.25 bpw.** Same bytes as IQ4_XS, trivial decode, but needs our own quantizer from BF16 (GPTQ/imatrix-quality is needed to match IQ4_XS).

### 2.4 Fast int4 -> fp16 dequant (for the mma path)

- LOP3 magic (FasterTransformer, "Who Says Elephants Can't Run" / AWQ): `h = lop3(q >> s, 0x000f000f, 0x64006400, (a & b) | c)` gives the fp16 pair `1024 + q`. Then `__hsub2(h, 1032)` for symmetric, or `__hfma2(h, scale, -(1024+z)*scale)` to fold scale and zero into one HFMA2.
  - Pre-permute the nibbles at repack time (order 0,2,4,6,1,3,5,7) so the `>>4` variant yields consecutive pairs.
- int8 -> fp16: `prmt` with 0x6480 magic (`__byte_perm(q, 0x64646464, 0x4140/0x4342)`) then subtract 1152.
- **Swap-AB mma for verify:** `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` with A = 16 weight rows (dequantized in registers from the 16-B loads), B = up to 8 token columns (activations in shared memory), fp32 accumulation. The exllamav3 Turing notes say the fp32-accumulate HMMA path gave +10% decode. Tensor-core math is effectively free here, so verify at m=2..8 costs about the same as m=1 as long as the dequant ALU stays hidden.

## 3. Multi-GPU on Kaggle T4 x2

### 3.1 Topology facts

- `nvidia-smi topo -m` on Kaggle T4 x2 reports **PHB** (both GPUs behind the CPU host bridge, no NVLink). The image ships NCCL (2.27.5 in a recent validated profile, with CUDA 12.8 and driver 580.x).
- NCCL's default `NCCL_P2P_LEVEL` allows P2P only up to PXB, so **on PHB NCCL uses SHM (host-staged) transport by default** unless forced with `NCCL_P2P_LEVEL=PHB`. Inside a VM (Kaggle runs on GCP), P2P is frequently disabled or broken. `cudaDeviceCanAccessPeer` is a lookup table, not a link test, so confirm with an actual copy.
- Observed: vLLM TP=2 on Kaggle T4 x2 with Qwen2.5-7B got **58 tok/s single-stream vs 32 tok/s on one T4 (1.8x)**, using NCCL with custom all-reduce disabled. So PHB TP=2 does pay off for batch-1 decode at 7B+, and it pays off more at 27B where per-layer weight time dominates comms.

### 3.2 Probe to run first on Kaggle (put it in `probe.cu`)

1. `cudaDeviceCanAccessPeer(0,1)` and `(1,0)`. Then `cudaDeviceEnablePeerAccess` and a 20 KB plus 64 MB `cudaMemcpyPeerAsync`, checksum verified, recording latency and GB/s.
2. Host-mapped ping-pong: `cudaHostAlloc(..., cudaHostAllocMapped|cudaHostAllocPortable)`. A kernel on GPU0 writes 20 KB plus an epoch flag (`st.global.release.sys` / `__threadfence_system()`); a kernel on GPU1 spins on `ld.volatile`/`ld.acquire.sys`, adds, and writes back. Measure the round trip over 10k iterations.
3. NCCL `ncclAllReduce` of 5120 fp32 (and fp16), both default and with `NCCL_P2P_LEVEL=PHB`, plus `NCCL_PROTO=LL` and `NCCL_ALGO=Ring`. Report p50/p99.
4. Per-GPU: a `cudaMemcpy` D2D 1 GB bandwidth test, plus a pure streaming-read kernel (`ld.global.nc.v4`, 512 MB), plus clocks/power while running.

Expected results (to be measured): PCIe Gen3 x16 one-way latency about 1-2 us and about 11-12 GB/s. The host-mapped all-reduce should land around **5-10 us**; NCCL over SHM for 20 KB is typically **20-40 us**.

### 3.3 TP vs layer-split math for this model

- **Layer split:** GPU0 runs blocks 0-31, GPU1 runs 32-63 plus lm_head. Comms are one 10-20 KB hidden-state handoff per token (negligible). At batch 1, only one GPU works at a time, so tok/s is about BW_eff / total bytes. This is what KoboldCpp did.
- **TP=2 (Megatron):** split per GPU as follows:
  - DeltaNet: key heads 0-7 plus their 24 value heads. **Permute the value heads so that v head h with `h%16 < 8` goes to GPU0**, because of the tiled mapping. Rows of attn_qkv, attn_gate, ssm_alpha, ssm_beta, and the conv channels are sliced accordingly, and the ssm_out columns are sliced.
  - Full attention: 12 q heads and 2 kv heads per GPU.
  - FFN: 8704 of the gate/up rows and the down columns.
  - lm_head: 124160 rows each, with a local argmax/top-k followed by exchanging (val, idx).
  - Comms: 2 all-reduces per layer, 128 per token, of 5120 fp32 = 20 KB (10 KB in fp16).
  - Per GPU per token: about 6.7 GB of weights (IQ4_XS), 151 MB of DeltaNet state, and half the KV.
- **Fuse the all-reduce into the next kernel:** the producer GEMV epilogue writes its partial sum to its own slot in host-mapped (or peer) memory and bumps an epoch flag. The consumer kernel's prologue (RMSNorm + q8_1 quantize of the next GEMV) spins on the peer flag, sums `own + peer + residual`, and writes the new residual. This adds no extra kernel and no host involvement, and it works inside CUDA graphs.
- llama.cpp status: `-sm row` (old) splits rows and is weak over PCIe. `-sm tensor` (PR #19378, experimental) splits along any dimension with AllReduce, **requires flash attention, and did not support a quantized KV cache at merge**. ik_llama.cpp has its own TP split mode. The torad-labs llama.cpp fork shows `-sm tensor` with fused f16/q8_0 recurrent-state writes for GDN/KDA, 5080 + 5070 Ti at about 104 tok/s.

### 3.4 Roofline ceilings (tok/s, batch 1, no speculation)

Model: t = (W + 0.302 GB DeltaNet state + KV_q8(ctx)) / BW plus 0.6 ms graph/launch gaps. TP: bytes/2 + 128 x L_allreduce.

| BW_eff | ctx | quant (GB/token) | layer split | TP, 8 us AR | TP, 30 us AR |
|---|---|---|---|---|---|
| 260 | 4k | UD-IQ4_XS (13.34) | 18.6 | **35.5** | 32.3 |
| 260 | 4k | UD-Q4_K_S (14.45) | 17.3 | 33.0 | 30.2 |
| 260 | 4k | Q4_0 (15.07) | 16.6 | 31.8 | 29.2 |
| 260 | 4k | UD-Q4_K_M (15.39) | 16.3 | 31.2 | 28.7 |
| 260 | 4k | UD-Q3_K_XL (12.24) | 20.2 | 38.4 | 34.7 |
| 260 | 32k | UD-IQ4_XS | 17.4 | 33.3 | 30.4 |
| 260 | 32k | UD-Q4_K_M | 15.3 | 29.4 | 27.2 |
| 240 | 4k | UD-IQ4_XS | 17.2 | 32.9 | 30.2 |
| 280 | 4k | UD-IQ4_XS | 20.1 | 38.1 | 34.4 |

Full-attention KV per token: 16 layers x 4 kv heads x 256 x (K+V) gives **64 KB fp16, 34 KB q8_0, 18 KB q4_0**.
- At 32k ctx q8: 1.14 GB per token read (+8% traffic).
- At 128k: 4.6 GB (+33%).
- At 262k: 9.1 GB q8, which exceeds the weights. Long-context decode is KV-bound; use 4-bit KV (Hadamard-rotated) there.

## 4. Gated DeltaNet and attention decode kernels

### 4.1 DeltaNet state traffic

- Per layer: 48 heads x 128 x 128 x 4 B = 3.15 MB, so read+write is 6.3 MB. Over 48 layers that is **302 MB/token**, about 1.16 ms at 260 GB/s (about 2% of a layer-split token, about 3.5% per GPU under TP).
- fp16 state would save about 0.29 ms per GPU (about 1%). Not worth the precision risk at first. The torad-labs f16/q8_0 state PR measured bit-identical-range PPL and +2.3-2.6%, mostly by deleting an extra copy kernel, which we never write in the first place.
- On-chip residency across tokens is impossible: 151 MB of total state against 4 MB of L2 and 40 x 256 KB registers. `cudaAccessPolicyWindow` L2 persistence measured -0.1% in that PR, so skip it.
- The oxidize reference kernel touches each state row twice (two loops). A fused kernel must do exactly **one read and one write** per element.

### 4.2 Fused mixer kernel design (one launch per DeltaNet layer)

- Grid: (head, value-row slice), 4 slices of 32 rows, giving 192 blocks per layer (96 per GPU under TP). With only 48 whole-head blocks on 40 SMs, 8 SMs would run 2 heads and double the time.
- Block = 128 threads: thread j owns column j of a 32x128 slice, so 32 fp32 registers of state. Coalesced 512 B row loads.
- Prologue (redundant per block, cheap):
  - conv1d over the 4 taps for this head's q/k (128 each) and v-slice channels, then SiLU.
  - Update the conv state (3 x 10240 fp32 per layer; only the slice owner of v writes it; q/k channels are written by slice 0).
  - L2-norm q and k (warp reductions); q *= 1/sqrt(128).
  - beta and decay from the ssm_alpha/ssm_beta GEMV outputs.
- Body: `S *= decay; sk = S.k (block reduction over 128 columns per row); d = (v - sk)*beta; S += d k^T; o = S.q`. Write o (32 values per block).
- **Speculative verify:** loop t = 0..k inside the kernel with S in registers. Write o_t for every t. Write S only once at the end into the alternate buffer (double-buffered states). Stash the per-token conv inputs.
  - On partial acceptance a < k+1, run one "replay" kernel over all 48 layers from the old buffer for the a accepted tokens, using stashed (q, k, v, beta, decay) of about 41 KB per layer per token. That costs one more 302 MB pass (about 1.2 ms), paid only on rejection.
  - Alternative: write the per-t state snapshot only for slices where it is cheap. Replay is simpler.
- Gated RMSNorm (per-head 128) * SiLU(z) moves into the **ssm_out GEMV prologue**, which reads all 6144 (3072 per GPU) values anyway. This removes the cross-slice reduction problem.

### 4.3 Full-attention decode (16 layers)

- One kernel per layer: q/k RMSNorm, partial RoPE (64 dims), KV append (fp16 or q8_0), then **split-K flash decoding**.
  - Grid = kv_head x ctx-chunk (chunk about 256-512 positions so there are at least 160 blocks).
  - Each block processes **all 6 q heads of its GQA group** so K/V are read once. Pad to n=8 and use mma m16n8k8: K-tile as A, q as B.
  - Online softmax in fp32, then a tiny combine kernel that also applies `sigmoid(gate)` and quantizes for attn_output.
- KV q8_0 with on-the-fly dequant to fp16 in registers (prmt trick). For more than 64k ctx, use q4_0 with a Hadamard rotation (as exllamav3's fdq4: x8-11 at 128k vs a naive kernel).
- Verify (k+1 queries) folds into the same kernel as extra query columns: n = 6 x (k+1), still m16n8k8-friendly.

## 5. Launch overhead and fusion

Kaggle has 4 slow vCPUs, so **the whole decode step must be one CUDA graph per GPU**, with argmax/sampling on the GPU. vLLM on T4 halved throughput in eager mode. Programmatic dependent launch is sm_90+, so it does not exist here.

Fused kernel list per token:
- **DeltaNet layer (5 kernels):**
  1. `[AR-wait + residual + RMSNorm + q8_1]` -> GEMV attn_qkv|attn_gate|alpha|beta fused into one 16480-row matrix (8240 per GPU).
  2. Mixer.
  3. `[gated norm + q8_1]` -> GEMV ssm_out -> `[partial publish]`.
  4. `[AR-wait + residual + RMSNorm + q8_1]` -> GEMV gate|up (rows interleaved so one warp produces gate_i and up_i) -> epilogue `silu(g)*u` -> q8_1.
  5. GEMV down -> `[partial publish]`.
- **Full-attn layer (6 kernels):** fused q|k|v GEMV, then norm/rope/append/attention, combine + gate, attn_output, gate|up, down.
- **Head:** lm_head GEMV with a fused block-argmax (or top-k) epilogue, without writing 248k logits, then a tiny final reduce.
- **Embedding:** gather one row from host-mapped memory (frees 0.55-0.72 GB VRAM).
- About 340 kernels per token. At about 1.5-3 us of graph node gap each, that is about 0.5-1 ms (2-3%).
- A persistent megakernel (Hazy Research / Mirage-MPK style) could reclaim that. Do it last, if ever.

## 6. Speculative decoding with the built-in MTP head

MTP step: `h' = eh_proj([enorm(embed(tok)) ; hnorm(h_last)])`, then the blk.64 full-attention block (with its own 1-layer KV cache), then `shared_head_norm`, then the shared lm_head.

Draft cost per token: blk.64 (0.27-0.35 GB) + eh_proj (0.04) + lm_head (0.87-1.04) = **about 1.3 GB, about 9% of a main-model token**. Most of it is the lm_head. **Draft with a frequency-truncated vocab head** (FR-Spec style: the top 32k tokens, 0.13 GB) to cut the draft to about 3%. Verify still uses the full head.

Acceptance reported for Qwen3.5/3.6 MTP in llama.cpp:
- About 82% with `--spec-draft-n-max 2`, about 72% with 3. The usual 70-85% range.
- One llama.cpp issue (#23322) showed acceptance dropping to 35% from a hybrid-cache invalidation bug, so implement state rollback correctly (section 4.2).
- Chaining the single MTP layer past 1 step degrades acceptance.

Model: E[tokens per step] = (1 - a^(k+1)) / (1 - a). Cost = T_verify(k+1) + k * T_draft. With swap-AB mma or dp4a x m, T_verify(k+1) is about (1 + 0.04k) T1, still memory-bound.

| a | k | draft cost | E | step cost | speedup |
|---|---|---|---|---|---|
| 0.80 | 1 | 9% | 1.80 | 1.13 | 1.59x |
| 0.82 | 2 | 9% | 2.49 | 1.26 | 1.98x |
| 0.82 | 2 | 3% (truncated vocab) | 2.49 | 1.14 | **2.19x** |
| 0.72 | 3 | 3% | 2.61 | 1.21 | 2.16x |
| 0.85 | 3 | 3% | 3.19 | 1.21 | 2.63x |

(These assume i.i.d. acceptance, so they are optimistic. Plan for about 1.8x.)

Combined target: TP at about 33 tok/s x about 1.8 gives **about 55-60 tok/s**. Without TP: about 18 x 1.8 = 32 tok/s.

## 7. Concrete build plan (ordered)

1. **probe.cu** (section 3.2): P2P, host-mapped ping-pong, NCCL latency, streaming read GB/s, clocks. This decides the all-reduce design and the BW_eff for all estimates.
2. **gemv_bench.cu:** Q4_0 repacked SoA + q8_1 + dp4a at m=1..8, shapes 5120x16480, 5120x34816, 17408x5120, 6144x5120, 5120x124160. Compare against llama.cpp mmvq on identical data, plus cuBLAS fp16 GEMV as a sanity check. Goal: at least 250 GB/s at m=1 and at least 230 at m=4. Then add the Q4_K / Q5_K / Q6_K / IQ4_XS variants.
3. **Single-GPU correctness engine (Q4_0 GGUF):** loader plus repack and the fused kernels from section 5. Compare logits against llama.cpp (`llama-eval-callback` / perplexity on wikitext 2k) and against oxidize's semantics.
4. **TP=2** with the fused host-mapped all-reduce, graph-captured per GPU, with a single host thread launching both graphs.
5. **MTP speculative decode** with double-buffered DeltaNet state and the replay kernel, a truncated-vocab draft head, and on-GPU accept logic.
6. Swap in the UD-IQ4_XS / Q4_K_M kernels, or a custom 4.25 bpw format, for quality per byte. Then q4 KV for long context.

## Sources

- llama.cpp multi-GPU docs, split modes: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- llama.cpp `-sm tensor` discussion: https://github.com/ikawrakow/ik_llama.cpp/discussions/1247, https://github.com/ikawrakow/ik_llama.cpp/discussions/979
- ExLlamaV3 Turing support: https://github.com/turboderp-org/exllamav3/pull/325, https://github.com/turboderp-org/exllamav3/issues/417, https://github.com/turboderp-org/exllamav3/pull/411, https://github.com/rafatxf/exllamav3
- Kaggle T4x2 vLLM profile (NCCL 2.27.5, CUDA 12.8, custom AR disabled): https://github.com/kaggle-vllm/kaggle-vllm
- Kaggle T4x2 PHB topology and TP=2 at 58 vs 32 tok/s: https://github.com/Chebaleomkar/ZeroHost-vLLM-v2
- NCCL P2P troubleshooting: https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2312/user-guide/docs/troubleshooting/gpu_troubleshooting.html
- P2P in VMs: https://github.com/NVIDIA/nccl/issues/1329
- T4 graphs and throttling notes: https://www.dhruvvakharwala.dev/blogs/bear-the-tokens-qwen-t4-throughput
- T4 product brief (320 GB/s): https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/T4%20Product%20Brief.pdf
- Batch-1 decode efficiency: https://arxiv.org/pdf/2605.30571
- GEMV grid starvation: https://github.com/microsoft/onnxruntime/issues/32382
- f16/q8_0 recurrent state and L2 persistence: https://github.com/torad-labs/llama.cpp/pull/82
- Chunked GDN prefill: https://github.com/torad-labs/llama.cpp/pull/10
- MTP acceptance: https://johnpaulwile.substack.com/p/multi-token-prediction-mtp-in-llamacpp, https://github.com/ggml-org/llama.cpp/issues/23322
- Qwen3.8-27B GGUF speed thread: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/32
- GGUF tensor types and sizes: parsed from https://huggingface.co/unsloth/Qwen3.8-27B-GGUF headers (HTTP range read; parser at scratchpad `gg/parse.py`).
