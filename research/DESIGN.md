# t4q: a Qwen3.8-27B decode engine for Kaggle 2x T4

Status: design, 2026-10-02. Nothing in t4q has been compiled yet. This design builds on four research files in this directory: `arch.md` (exact math), `gguf.md` (container, block formats, per-file types), `kernels.md` (kernel strategy and rooflines) and `baseline.md` (measured box and llama.cpp numbers).

t4q is a standalone C++17/CUDA engine for sm_75 with no runtime dependencies beyond the CUDA runtime. It loads an unsloth GGUF, repacks the weights on the GPU, and runs batch-1 decode, MTP speculative decode and batched prefill across both T4s with tensor parallelism (TP). Python drives it through a plain C ABI loaded with ctypes.

## 0. Targets and numbers to beat

| config (all measured on this Kaggle box) | decode tok/s | prefill pp512 tok/s |
|---|---|---|
| prior KoboldCpp (layer split, Q4_K_S, q8 KV) | 9.2 | ~360 @25k |
| llama.cpp master, Q4_0, `-sm layer` | 14.66 | 310 |
| llama.cpp master, Q4_0, `-sm tensor` | **21.11** | **516** |
| llama.cpp master, UD-Q4_K_XL, `-sm tensor` | 18.70 | 505 |
| llama.cpp master, UD-Q4_K_XL, tensor + MTP (n_max 3), greedy coding prompts | **28.8-32.4** | - |

t4q milestone gates (same prompts, greedy, 4k context):
- M4 plain decode on Q4_0: **at least 30 tok/s**, with ~33 expected and a ~35 roofline.
- M5 with MTP: **at least 60 tok/s** on the baseline coding prompts P0/P1, with 70-85 expected and ~95 as the ceiling.
- Prefill: **at least 700 tok/s** at pp512.

## 1. Decisions at a glance

| topic | decision |
|---|---|
| source GGUF | **`Qwen3.8-27B-Q4_0.gguf`** (16.06 GB) for M1-M5. **UD-Q4_K_M** is the production default from M6, with UD-IQ4_XS as the speed option. The MTP head comes from `blk.64` inside the main file. No requantization. |
| on-device format | Lossless SoA repack done on the GPU at load: separate code, high-bit and scale planes, native two-level scales, 16 B per lane per load (section 3) |
| GPU split | **TP=2** for every layer plus the lm_head. M1 uses layer split only for bring-up, and both modes share the same shard abstraction. |
| all-reduce | A custom in-kernel mailbox with direct **P2P stores** plus per-block epoch flags. The consumer prefetches its weights before waiting. Fallbacks are host-mapped pinned memory, then NCCL. |
| GEMV | q8_1 int8 activations with `dp4a`, for m = 1..8 columns. The per-column result is bit-identical for every m. |
| prefill GEMM | In-register dequant to fp16 and `mma.sync.m16n8k8.f32.f16.f16.f32`, with register-staged double buffering (sm_75 has no cp.async) |
| DeltaNet state | **fp32**, stored in t4q's own register-tile order, read once and written once per step, double-buffered for speculation |
| residual stream / norms / gates | fp32 |
| KV cache | fp16 by default (64 KiB/token, 32 KiB per GPU). q8_0 is an option (M6) for contexts above about 160k. |
| graphs | One CUDA graph per GPU for each of decode-1, verify-(k+1), mtp-draft and replay. All step state lives in device memory, so the host never syncs inside a step. |
| sampling | On the GPU. Argmax or top-k is fused into the lm_head epilogue. |
| MTP | k = 3 drafts by default (sweep 2-5). The catch-up and the first draft are fused into one batched MTP pass. Rollback uses replay, not per-token snapshots. M5b adds a truncated 32k-vocab draft head. |
| tokenizer | HF `tokenizers` in the Python driver. Token ids are the interface, and both t4q and the llama.cpp oracle consume the same id file. |
| validation | `oracle_dump`, a 150-line C++ file linked against the sm_75 `libllama.so` already built in `t4-qwen38-baseline`, dumps full logits and named intermediates |

## 2. Resolved disagreements between the research files

1. **Bytes per token.** `arch.md`/`kernels.md`/`gguf.md` give Q4_0 = 15.07 GB and Q4_K_XL = 16.48 GB. `baseline.md` gives 15.33 and 16.83. The gap is the MTP block `blk.64` (0.265 / 0.351 GB), which `baseline.md` counted. **Use the figures without MTP for plain decode** and add `blk.64` only to the draft cost.
2. **Achievable bandwidth.** The research files planned around 260 GB/s, while `baseline.md` measured a 279.2 GB/s float4 read (87%) and 284 GB/s on a 45 MB matrix-sized buffer. **279 is the hard roofline and 265 (95% of the measured read) is the GEMV target.** All tables below use both.
3. **P2P.** `kernels.md` assumed PHB would mean no usable P2P and NCCL over SHM. The box actually reports `cudaDeviceCanAccessPeer = 1` both ways, and a 10 KB P2P hop costs 7.57 us pipelined (11.68 us synced) against 17-18 us host-staged. **Primary: direct P2P stores into the peer's mailbox.** Fallback 1 is host-mapped pinned memory. Fallback 2 is NCCL. The probe (M0) measures remote-store latency specifically, because `baseline.md` measured only memcpy and peer *reads* (7.25 GB/s). `topo -p2p` showed no native atomics, so **the protocol uses no remote atomics**, only stores and flags.
4. **Which GGUF.** `gguf.md` recommends UD-Q4_K_M for production, `kernels.md` suggests IQ4_XS or a custom format, and `baseline.md` benchmarked Q4_K_XL. **Use Q4_0 for M1-M5.** It has 6 types, it is llama.cpp's fastest measured config (21.1 tok/s), and it gives a clean oracle. **Then UD-Q4_K_M** (+2% bytes, imatrix quality). UD-IQ4_XS is the speed option once native IQ3/IQ2 kernels exist; upcasting its 2.5 GB of low-bit tensors would add 0.8 GB and erase most of its edge. A custom requant is rejected: it would need a 51 GB BF16 download and would throw away the imatrix work.
5. **Scale layout.** `kernels.md` sketched fp16 per-group scales. `gguf.md` showed that expanding K-quant scales costs +11% bytes. **Keep the native scale bytes and move them to separate planes.** Q4_K's `d, dmin, scales[12]` is exactly 16 B, already one aligned vector.
6. **Verify GEMV.** `kernels.md` proposes swap-AB `mma` for m = 2..8. **t4q uses dp4a for m ≤ 8 first**, which is llama.cpp's mmvq precedent. At m = 4 that is about 1.4 instr/weight, roughly 5 ms of ALU per GPU against ~28 ms of memory time, so it stays hidden. It also gives bit-identical columns, so speculative output is exactly the non-spec output. Swap-AB mma is the fallback if `nvidia-smi` shows SM clocks throttled below ~1.1 GHz under verify.
7. **MTP rollback.** llama.cpp (`arch.md`) keeps K per-token state snapshots, while `kernels.md` proposes a double buffer plus replay. Snapshots cost k × 75.5 MB of extra writes per GPU on every step. Replay costs one 151 MB read+write per GPU, and only on partial acceptance. **Use replay.**
8. **Draft head.** llama.cpp drafts with the full 248k lm_head, about 1 GB, which is most of the draft cost. **M5a uses the full head for parity; M5b uses a 32k-row subset** (section 7.4) behind a flag, kept only if acceptance falls by less than 2 points.
9. **MTP weights.** `baseline.md` ran MTP from the separate `MTP/mtp-...-Q4_0.gguf`, whose `blk.64` is Q6_K/Q4_K. `gguf.md` notes that the main files already carry `blk.64`. **t4q uses the main file's `blk.64`** (Q4_0 with a Q8_0 `eh_proj` in the Q4_0 file). Acceptance may differ slightly from the 0.73-0.85 measured with the separate file. M5 measures it, and t4q can load the separate file's `blk.64` if acceptance turns out worse.
10. **MTP speedup.** `kernels.md` plans for 1.8x and `baseline.md` measured 1.8x on llama tensor split. t4q's draft is cheaper (no CPU sampler, fused catch-up, optional truncated head) and its verify costs about 1.1x a token. So **2.2-2.8x is expected on code** (section 9). The gate stays at 60 tok/s.
11. **EOS.** The GGUF has 248046 (`<|im_end|>`) and the HF config has 248044. **Stop on both.**
12. **RoPE pairing.** `kernels.md` left neox versus interleaved open. `arch.md` settled it: NeoX pairs (i, i+32) over the first 64 of 256 dims, θ_i = p·1e7^(−2i/64), with p·inv_freq formed in fp32.
13. **L2-norm epsilon.** oxidize uses `1/sqrt(max(Σx², eps))`. **t4q uses the HF/ggml form `x·rsqrt(Σx² + 1e-6)`.**
14. **V-head order.** GGUF stores V heads tiled (`kh = h % 16`); HF and the oxidize HF path use grouped (`kh = h / 3`). **t4q reads only GGUF, so it uses `h % 16` everywhere.** The TP split must give GPU g the v heads `{r·16 + kh : r ∈ 0..2, kh ∈ [8g, 8g+8)}`.
15. **Embedding placement.** `token_embd` (0.715 GB in Q4_0) is a one-row gather. **It stays in host-mapped pinned memory**, freeing VRAM for KV. The gather is 2880 B per token over PCIe, under 2 us, and is graph-capturable.
16. **KV format.** fp16 by default, since llama.cpp `-sm tensor` didn't support quantized KV at merge, and t4q is not bound by that. Per-GPU capacity (section 8) holds about 200k tokens in fp16. **q8_0 KV is M6, for 262k.**

## 3. Loader and on-device repack

### 3.1 GGUF parsing
`src/gguf.cpp` uses `mmap` and the v3 layout in `gguf.md` §1: tensor data starts at the end of the tensor infos rounded up to 32 (10,996,704 for Q4_0). It reads the hyperparameters from the `qwen35.*` keys and asserts them against `arch.md` §0. It ignores the tokenizer arrays, because Python handles tokenization. The GGUF lives on `/tmp` (the 1.1 TB overlay), not `/kaggle/working`, which is limited to 20 GB and is the output.

### 3.2 Pipeline
For each tensor and each GPU shard:
1. Copy the shard's source rows (or column block ranges for K-split tensors) into a pinned 2 x 256 MB double-buffered staging area with `memcpy` from the mmap (4 threads).
2. `cudaMemcpyAsync` H2D at about 12 GB/s.
3. Run the repack kernel into the final allocation.

The load is bound by disk and page cache (16 GB at about 1-2 GB/s from a fresh download), so plan for **15-25 s after a ~70 s download**. Optionally write the repacked blob as a Kaggle dataset later (skips download and repack; not needed for M1-M5).

### 3.3 Packed formats (lossless; row = output channel, rows tiled by 2 per warp)

| t4q fmt | GGUF sources | planes per row (K weights) | per-lane 16 B load covers |
|---|---|---|---|
| P4 | Q4_0, Q4_1, IQ4_NL | `codes` K/2 B (ggml nibble order kept: lo = j, hi = j+16); `d` fp16 per 32 (+`m` fp16 for Q4_1); IQ4_NL sets the LUT flag (int8 `kvalues_iq4nl` via `__byte_perm`) | one 32-weight group |
| K4 | Q4_K, IQ4_XS | `codes` 128 B per 256; `meta` 16 B per 256 (Q4_K: d, dmin, scales[12] verbatim; IQ4_XS: d, scales_h, scales_l[4] + pad = 8 B, or 16 B aligned) | 32 weights; 8 lanes = 1 superblock |
| K4L | Q3_K, IQ3_S (upcast, M6) | 4-bit code + per-tensor int8 LUT16 + int8 sub-scale per 16/32 + fp16 d per 256 | as K4 |
| K5 | Q5_K | K4 planes + `qh` 1-bit plane 32 B per 256 | as K4 + one 4 B load |
| K6 | Q6_K | `ql` 128 B + `qh` 64 B + int8 `sc[16]` + fp16 `d` per 256 | as K4 + 8 B |
| Q8 | Q8_0 | int8 32 B per 32 + fp16 `d` | 16 weights |
| H | F32 `ssm_alpha`/`ssm_beta` | fp16 if the round-trip is exact (these are bf16-origin values, and fp16 has 2 more mantissa bits), else keep F32 | - |

- Each plane is stored `[row_pair][k_chunk][lane]`, so **a warp instruction reads 512 contiguous bytes** with `ld.global.nc.v4`. Scale planes are read once per 32 or 256 weights.
- **TP slicing happens at repack time.** Row slices (q/k/v/qkv/z/alpha/beta/gate/up/lm_head) are free. K-splits (ffn_down at 8704, ssm_out and attn_output at 3072, eh_proj at 5120) land on 256-weight boundaries (34x256, 12x256, 20x256). The DeltaNet V rows are a non-contiguous gather (item 14 in section 2).
- **Interleave rows for fused epilogues:** gate and up are interleaved per 32 rows, so one warp-pair yields `silu(g)*u` for 32 consecutive outputs and can q8_1-quantize them in-warp.
- After repack the loader checks `dequant(packed) == ggml_dequant(gguf)` bit-exactly on 64 random rows per tensor, using a CPU port of `dequantize_row_*`.

### 3.4 Per-GPU memory (TP, Q4_0, 4k context)
Weights 7.53 GB + `blk.64` 0.13 GB + 32k draft head 0.07 GB + DeltaNet state 2 x 75.5 MB + conv state 2 x 2.9 MB + verify stash 2 MB + KV 34 KiB/token (trunk 32 KiB + MTP 2 KiB) + prefill workspace about 0.35 GB + CUDA context about 0.35 GB comes to **about 8.6 GB plus KV**. That leaves about 6.8 GB of KV per GPU: **about 200k tokens in fp16, or 262k+ in q8_0.**

## 4. GPU split: TP=2

Why TP: on this box llama.cpp's tensor split beats its layer split by 1.42x decode and 1.62x prefill, even through NCCL. Layer split is capped near 17.7 tok/s at 279 GB/s because only one GPU streams at a time. TP is capped near 35.

| block | GPU g owns | collective after |
|---|---|---|
| DeltaNet in-proj (qkv, z, alpha, beta) | k heads 8g..8g+7: q rows 128kh, k rows 2048+128kh, v rows for heads {r·16+kh} (3072 rows), z rows for the same 24 heads, 24 alpha + 24 beta rows. Total 8240 rows | none |
| DeltaNet conv + recurrence + gated norm | its 5120 conv channels, 24 v heads of state (75.5 MB) | none |
| ssm_out | 3072 input columns (its 24 heads) to 5120 partial | **AR #1** |
| full attn q/k/v | q heads 12g..12g+11 (attn_q rows 6144g..+6143, contiguous because of the per-head [q\|gate] interleave), kv heads 2g, 2g+1 (rows 512g..+511 in k and v) | none |
| attention core | 2 kv heads x 6 q heads each, local KV cache | none |
| attn_output | 3072 input columns | **AR #1** |
| FFN gate/up | rows 8704g..+8703 (interleaved) | none |
| ffn_down | 8704 input columns | **AR #2** |
| lm_head | rows 124160g..+124159 | (val, idx) top-k exchange |
| MTP eh_proj | input columns 5120g..: GPU0 multiplies `enorm(embed)` and GPU1 multiplies `hnorm(h)` | AR |
| MTP block 64 | like a full-attn layer | 2 AR |

Each token needs **128 all-reduces** of 5120 fp32 values (20 KB). Both GPUs then hold an identical residual: each computes `res + (p0 + p1)` with p0 always added first, so the fp32 bits match on both cards without further sync.

### 4.1 Mailbox all-reduce (`src/kernels/allreduce.cuh`)
- Each GPU allocates in its own VRAM `rx[2 slots][5120] fp32` plus `flag[2 slots][MAXBLK] u32`, then enables peer access. The peer gets those pointers.
- **Producer** (the K-split GEMV epilogue, i.e. ssm_out/attn_output/ffn_down): block b writes its rows' partials to its local `own[slot]` and to the peer's `rx[slot]` (posted PCIe writes), then calls `__threadfence_system()`, then writes `peer.flag[slot][b] = epoch`. The epoch is a device-side u32 that increases on every AR, so the flags never need resetting.
- **Consumer** (the next GEMV, `norm_q8_gemv`):
  1. Prefetch its own weight tile first. Weights don't depend on x, so about 40 KB of loads per block go into flight.
  2. Spin with `ld.volatile` on the **local** flags (one thread per producer block). The spin has a 2^26-iteration watchdog that sets an error word and exits, so a broken run can't hang a Kaggle session.
  3. Compute `res = res + own + rx` (block 0 writes res back), then RMSNorm and q8_1 into shared memory, then the dp4a body.

  The prefetch hides PCIe latency and the skew between GPUs behind useful work. **Expected exposed cost is 2-4 us per AR, about 0.4 ms per token.**
- Slots alternate, so a fast GPU can never overwrite a mailbox its peer hasn't consumed. Stream order serializes each GPU's own kernels, so two slots are sufficient.
- Fallbacks are selected at init from the probe:
  - (a) `rx` and flags in `cudaHostAllocMapped|Portable` memory, with the consumer spinning over PCIe;
  - (b) the AR becomes its own `ncclAllReduce` graph node between kernels. NCCL 2.27.5 ships in torch; `-lnccl` comes from the system libnccl 2.25.1.

## 5. Decode kernels, fusion plan and per-op budget

Kernels for one token per GPU in the TP Q4_0 decode-1 graph are below. The DeltaNet bytes are per layer (x48); the attention bytes are per layer (x16).

| # | kernel | fused work | bytes/GPU | time @265 GB/s |
|---|---|---|---|---|
| D1 | `norm_q8_gemv<seg>` | AR-wait, residual add, RMSNorm(attn_norm), q8_1(x); segmented GEMV: qkv (5120 rows P4) + z (3072 P4) + alpha/beta (48 rows H) | 24.0 MB | 91 us |
| D2 | `gdn_step` | conv4 + state shift (raw inputs), SiLU, L2-norm q/k (q·1/√128), beta = σ(b), g = ssm_a·softplus(a+dt), recurrence over 24 heads x 128 rows x 128 (fp32 state, 1 R + 1 W), output o | 3.1 MB | 12 us |
| D3 | `gnorm_q8_gemv` | prologue: per-head RMSNorm(o)·ssm_norm·silu(z) to q8_1; GEMV ssm_out (Q5_K, 3072 to 5120); epilogue publishes the AR partial | 10.8 MB | 41 us |
| D4 | `norm_q8_gemv<swiglu>` | AR-wait, residual, RMSNorm(post_attention_norm), q8_1; GEMV gate\|up interleaved (17408 local rows); epilogue silu(g)·u to q8_1 (8704) | 50.1 MB | 189 us |
| D5 | `q8_gemv<partial>` | GEMV ffn_down (8704 to 5120, Q4_0/Q4_1), publishes the AR partial | 25.1 MB | 95 us |
| A1 | `norm_q8_gemv<seg>` | AR-wait, residual, RMSNorm, q8_1; attn_q 6144 + attn_k 512 + attn_v 512 rows | 20.6 MB | 78 us |
| A2 | `attn_decode` | q/k per-head RMSNorm, partial NeoX RoPE on 64 dims, KV append (fp16), split-K flash-decode: grid = 2 kv heads x 64 splits, each block serves the 6 q heads of its group (K/V read once), online softmax in fp32 | 32 KiB x ctx | 0.5 ms @4k (all 16 layers) |
| A3 | `attn_combine_q8` | merge splits, o·σ(gate) to q8_1 (3072) | ~0.2 MB | 3 us |
| A4 | `q8_gemv<partial>` | attn_output (3072 to 5120), AR publish | 8.8 MB | 33 us |
| A5, A6 | as D4, D5 | FFN | 75.2 MB | 284 us |
| E0 | `embed_gather` | Q4_0 row from host-mapped memory, dequant to fp32 residual; token id read from the device ring | 2.9 KB PCIe | ~2 us |
| H1 | `norm_q8_gemv<argmax>` | final AR-wait, residual, output_norm (also stores `h_final` for MTP), q8_1; lm_head Q6_K 124160 rows; epilogue per-block top-k (k = 1 greedy, 20 sampled) | 0.52 GB | 1.97 ms |
| H2 | `select_token` | exchange per-GPU candidates through the mailbox, temperature/top-k/top-p with Philox, write the token to the device ring and to a host-mapped ring, bump pos/epoch | - | 3 us |

Totals per GPU per token:
- 48 x 5 + 16 x 6 + 3 = **339 graph nodes**. At about 0.8-1 us per node, including tails, that is 0.35 ms.
- Bytes: weights 7.53 GB + DeltaNet state 0.151 GB + KV.

Notes:
- **GEMV body (P4, m columns).** The block is 256 threads with 2 rows per warp. The K=5120 loop runs 5 iterations x 2 rows x 16 B per lane, all issued before use (≥ 40 KB in flight per block). The math is `sumi = dp4a(lo, xq[0:4]) + dp4a(hi, xq[16:20]) ...` and `acc += d_w·(d_x·sumi − 8·s_x)`, accumulated in fp32 in a fixed order independent of m. The grid is row-interleaved so the last wave isn't a tail. K-format bodies unpack the sub-scales once per 256 weights.
- **Segmented GEMV.** A single launch covers up to 4 segments with different formats or row counts that share the same x, dispatched by blockIdx range. This is how qkv|z|alpha|beta and q|k|v fuse even in UD files, where each tensor has its own type.
- **`gdn_step` layout.** The grid is (24 heads x 4 row-slices) = 96 blocks of 128 threads. Thread (row i, quarter c) owns `S[i][32c..32c+31]` (32 fp32 registers). `S·k` and `S·q` are 32 FMAs plus a 2-step shuffle reduce across 4 lanes. The state is stored in register-tile order `S_mem[h][slice][r][tid]` (float4), so every warp load is 512 contiguous bytes. q/k come from the full 128-dim key head that each block recomputes from qkv (cheap). The conv state is written by the slice that owns the channel.
- **What is not fused, and why.** gated-norm-before-ssm_out is fused, but D2 → D3 isn't: the gated norm needs all 128 rows of a head while a D2 block owns 32. A3 isn't fused into A4 because each A4 block would redundantly merge 786 KB of split results.

## 6. CUDA graphs and host loop

- **Graphs per GPU, each captured once:**
  - `G_dec1`: E0 → 64 layers → H1/H2
  - `G_ver[k]`: the same with m = k+1 columns, captured for k = 1..5
  - `G_draft`
  - `G_replay`
  - `G_prefill[ub]`: ubatch = 512, with a tail ubatch padded and masked
- **Graph invariance:** every position-dependent quantity (`pos`, `kv_len`, the AR epoch, the state-buffer index, the token ring head, `n_accepted`) lives in a device `StepState` struct. Kernels read it, and H2 or the accept kernel advances it. Attention uses a fixed grid (64 splits; each split covers `ceil(len/64)` positions and returns early when empty), so no launch parameters change between steps.
- **Host:** one thread launches `G_x` on stream 0 (GPU0) and stream 1 (GPU1). The two graphs synchronize only through the mailboxes. The host enqueues **8 steps ahead** and syncs once per 8 tokens (or per 8 spec steps) to read the host-mapped token ring and check for EOS or the length limit. Tokens produced after EOS are discarded and the state is rewound by the same replay mechanism; for a plain step that means resetting `pos`, and for DeltaNet keeping the last 8 steps' stash. In M3 that is simplified to sync every step if needed: about 7 us per token, well under 1%.

## 7. MTP speculative decoding (M5)

### 7.1 Draft model (exact, from `arch.md` §7)
`u = W_eh · [enorm(embed(x)) ; hnorm(h)]`, then one full-attention block (`blk.64`, its own KV cache, same RoPE), then `shared_head_norm` to give h', then logits through the lm_head. Here h is the **post-`output_norm`** target hidden from the previous position, or the previous draft's h' when chaining.

### 7.2 Step (greedy). State: the last emitted token t at position p, and `H` = target `h_final` rows from the last verify
1. **Fused catch-up + first draft** (`G_draft`, batched): the MTP layer runs over the n+1 rows `(x_{q}, h_{q−1})` for the n tokens accepted in the last verify plus the bonus row `(t, h_{p−1})`. The early rows only overwrite the MTP KV entries the earlier drafts wrote using h' instead of the true h. The last row's logits give d1, and lm_head runs for that row only.
2. **Chained drafts** d2..dk: single-row MTP passes, feeding back h'.
3. **Verify** (`G_ver[k]`): the target decodes `[t, d1..dk]` at positions p..p+k. DeltaNet runs `gdn_verify`, which loops over the k+1 tokens with S in registers, reads buffer A and writes the final S to buffer B, writes o_t for every t, and stashes per-token (k̂_t, v_t, β_t, decay_t, raw conv inputs), about 17 KB per layer per token per GPU. Attention appends k+1 KV rows with an in-batch causal mask. H1 emits argmax for all k+1 rows and stores all k+1 `h_final` rows.
4. **Accept** (`accept` kernel, device side): n is the length of the longest prefix with `d_i == y_{i−1}`. Emit d1..dn and y_n. Set kv_len = p+n+1 (rejected KV rows are simply overwritten later).
5. **Rollback** (`G_replay`, always launched, device-branching):
   - If n = k, flip the state index to B.
   - Otherwise replay buffer A in place over the n+1 accepted tokens from the stash (no outputs) and rebuild the conv state from the stash. This costs 151 MB R+W per GPU, about 0.6 ms, only on partial acceptance.
   - The MTP KV is fixed by step 1 of the next iteration.
- **No host round trip inside the loop.** The host enqueues draft, verify and replay graphs back to back for many steps and drains the token ring periodically.
- **Exactness:** dp4a columns are bit-identical across m, and the attention split boundaries depend only on position. So **spec greedy output must be byte-identical to plain greedy**, which is a hard test (V5 below). llama.cpp's own spec output matched only under tensor split.

### 7.3 Sampling (M6)
For non-greedy sampling, use standard speculative rejection sampling on the GPU (accept d with min(1, p/q), resample from norm(max(p−q, 0))). This needs full draft and target probabilities over the top-k=20 candidates. Restrict both to top-k 20 / top-p 0.95, as in the GGUF defaults.

### 7.4 Truncated draft head (M5b)
Keep the 32768 most frequent token ids, taken from a token-frequency count over the target's own greedy outputs plus a code and text sample tokenized once (stored as `draft_vocab.bin`, 128 KB). The `output.weight` rows for those ids are gathered into a separate repacked 32k-row head, 0.069 GB per GPU. The draft cost falls from about 2.6 ms to about 0.85 ms. The flag is kept only if the in-subset rate for target tokens is ≥ 98% and acceptance drops by less than 2 points.

## 8. Prefill (batched, ubatch 512)

- **Linear layers:** `gemm_f16` dequantizes the weight tile in registers (lop3 `0x64006400` magic for 4-bit codes, prmt for LUT/int8), stores fp16 to shared memory, and runs `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`.
  - Tile: M = 128 weight rows x N = 64 or 128 tokens x K = 64, 256 threads, register-staged double buffering (no cp.async on sm_75), ≤ 48 KB static shared memory.
  - Activations: RMSNorm output in fp16. Normalized values are bounded, but **check for overflow in M1** and fall back to per-row-scaled fp16 if any layer exceeds 60000.
  - The residual stays fp32, and AR runs per ubatch with 512 x 20 KB = 10 MB per AR at about 10 GB/s, which is 1 ms. That is ~128 ms per ubatch and would dominate, so **during prefill the AR payload is fp16** (5 MB, 0.5 ms), and the next GEMM overlaps with the transfer by splitting the ubatch into 4 sub-chunks pipelined through the mailbox. That brings the AR cost down to about 20-30 ms per 512 tokens.
- **DeltaNet prefill:** `gdn_seq` runs the same register-resident recurrence as `gdn_verify`, looping over all ubatch tokens. It needs ~0.3 us per token step per layer, so about 7 ms per 512 tokens per layer group, which is small. The chunked WY algorithm (chunk 64, mma) waits for M6 and only gets built if profiling shows `gdn_seq` above 15% of prefill time.
- **Full-attention prefill:** a custom sm_75 flash-attention kernel (FA2 does not run on sm_75) with head_dim 256, mma m16n8k8, a 64-query x 64-key tile, and online softmax in fp32. M1 uses a naive kernel.
- **Roofline:** 2 x 27.6 GFLOP per token in the linear layers. T4 fp16 HMMA is 65 TFLOPS peak, about 25-30 sustained at 70 W, so the ceiling for two GPUs is about 1000 tok/s. **Target ≥ 700, against llama.cpp's 516.**
- **int8 option (M6):** `mma.m8n8k16.s8` with q8_1 activations (llama.cpp MMQ style) has 2x the peak, at the cost of per-sub-block int32 rescaling.

## 9. Roofline and target tables

**Plain decode.**
- Per-GPU time = (W/2 + 0.151 GB state + KV_ctx/2)/BW + 0.35 ms graph gaps + 0.4 ms exposed AR.
- Layer split = (W + 0.302 GB + KV)/BW + 0.35 ms.
- KV fp16 is 64 KiB per token in total.

| GGUF (decode GB/token) | ctx | layer split @279 | TP @279 (roofline) | **TP @265 (t4q target)** | TP @265, q8 KV | llama.cpp best measured |
|---|---|---|---|---|---|---|
| Q4_0 (15.07) | 4k | 17.7 | 34.8 | **33.1** | 33.3 | 21.1 (tensor) |
| Q4_0 | 32k | 15.8 | 31.1 | 29.6 | 31.4 | - |
| Q4_0 | 128k | 11.6 | 22.9 | 21.8 | 26.1 | - |
| UD-Q4_K_M (15.39) | 4k | 17.4 | 34.1 | **32.4** | 32.7 | - |
| UD-Q4_K_M | 32k | 15.6 | 30.6 | 29.1 | 30.8 | - |
| UD-IQ4_XS (13.35) | 4k | 19.9 | 38.9 | **37.0** | 37.4 | - |
| UD-IQ4_XS | 32k | 17.6 | 34.4 | 32.7 | 34.9 | - |
| UD-Q4_K_XL (16.48) | 4k | 16.3 | 31.9 | 30.4 | 30.6 | 18.7 (tensor) |

**MTP (Q4_0, 4k, k = 3, TP @265).**
- Cost model: step = 1.10 x T1 (verify) + catch-up/draft-1 (~1.0 ms) + (k−1) x T_draft + P(reject) x 0.6 ms.
- T_draft = 0.85 ms with the 32k head or 2.6 ms with the full head.
- Tokens per step = 1 + k·r, where r is the per-draft acceptance (llama.cpp's reported "draft accept"). llama.cpp measured 3.3-3.5 on P0/P1.

| case | tokens/step | step ms (32k head) | tok/s | step ms (full head) | tok/s |
|---|---|---|---|---|---|
| coding, r = 0.80 (measured 0.78-0.85) | 3.4 | 36.3 | **94** | 39.8 | 85 |
| general chat, r ≈ 0.6 | 2.8 | 36.3 | 77 | 39.8 | 70 |
| pessimistic, r = 0.5 | 2.5 | 36.5 | 68 | 40.0 | 62 |

Gate **≥ 60**. Expect 70-85 after a 10-15% derate for jitter and clock sag. This is 2.3-2.8x llama.cpp's best measured 30 tok/s.

**Per-op time budget (TP Q4_0, 4k, per GPU per token, target):**

| part | ms |
|---|---|
| FFN | 18.2 |
| DeltaNet in/out proj | 6.3 |
| attention projections | 1.8 |
| lm_head | 1.97 |
| DeltaNet state | 0.57 |
| KV | 0.5 |
| AR | 0.4 |
| graph gaps | 0.35 |
| **total** | **≈ 30.1 (33 tok/s)** |

If a profile shows any GEMV below 240 GB/s, fix that before doing anything else.

## 10. Numerical validation plan

The oracle is llama.cpp a4cb4c61, the sm_75 build in kernel output `t4-qwen38-baseline` (`llama-bin-sm75/libllama.so`, CUDA 12.8). `tools/oracle_dump.cpp` compiles against `llama.h`/`ggml.h` from that commit (a shallow clone takes seconds; there's no rebuild of ggml). It does three things:
- loads the same GGUF with `-ngl 99 -sm layer`, f16 KV, FA on;
- evaluates a fixed **token-id file** two ways: as a 512-token batch (MMQ path), and then token by token for the last 64 positions (MMVQ path, the same regime as t4q decode);
- writes the full fp32 logits for every position (`.npy`, 248320 floats per row) and, through `cb_eval`, full dumps of the named intermediates `attn_norm-N`, `linear_attn_qkv_mixed-N`, `z-N`, `beta_sigmoid-N`, `gate-N`, `conv_output_silu-N`, `q_conv_predelta-N`, `k_conv_predelta-N`, `v_conv_predelta-N`, `attn_output-N`, `final_output-N`, `linear_attn_out-N`, `Qcur-N`, `attn_gated-N`, `ffn_out-N`, `l_out-N`, `result_norm` and `result_output` for N ∈ {0, 3, 31, 63}, on the first and last positions.

t4q exposes the same names through `t4q_dump()`.

| id | check | pass criterion |
|---|---|---|
| V0 | each kernel vs `tests/ref.c` (fp64 scalar reference of `arch.md` §2-5 on real weights from the GGUF, 1 layer at a time); repack round-trip bit-exact | GEMV rel err ≤ 2e-3 (q8_1 activations); gdn state rel ≤ 1e-5; attention ≤ 1e-3 |
| V1 | intermediates vs oracle, layers 0/3/31/63 | rel L2 ≤ 1e-2 at layer 0, ≤ 3e-2 at layer 63 |
| V2 | full logits, 576 positions (P0 + P1 + 512-token wikitext slice), t4q prefill vs oracle batch, t4q decode vs oracle token-by-token | top-1 agreement ≥ 99% (excluding positions where the oracle's top-2 gap is < 0.1 logit); **mean KL(p_llama‖p_t4q) ≤ 2e-3 nats, p99 ≤ 2e-2**; and ≤ 2x the noise floor = KL(llama layer split ‖ llama tensor split) measured the same way |
| V3 | greedy 256 tokens on P0/P1 | identical to llama.cpp, or the first divergence happens at a near-tie (gap < 0.05 logit, logged) |
| V4 | internal consistency: layer split vs TP; decode-1 vs verify-m logits; prefill vs decode | TP vs layer KL ≤ 1e-4; verify-m vs decode-1 **bit-identical**; prefill vs decode KL ≤ 2e-3 |
| V5 | spec vs non-spec greedy, 512 tokens on P0/P1 | **byte-identical** token stream |
| V6 | long-context sanity: 32k-token needle prompt | the needle is retrieved; KL vs oracle at the last 16 positions within V2 limits |

Each run writes `results.json`, which records a `validation` block with the numbers above.

## 11. Tokenizer and API

- **Tokenizer:** the Python driver downloads only `tokenizer.json`, `tokenizer_config.json` and `chat_template` from `Qwen/Qwen3.8-27B` (public; no token needed) and uses `tokenizers` / transformers 5.0 (already on the image) with `apply_chat_template(..., enable_thinking=False)`.
  - Every prompt is tokenized once, saved as `int32` ids, and those ids go to both t4q and `oracle_dump`, so tokenizer differences can't contaminate validation.
  - A one-off check compares ids against `llama_tokenize` from libllama on the test prompts.
  - Detokenization uses `tokenizers.decode` on the host-mapped token ring.
- **C ABI** (`include/t4q.h`, `extern "C"`, no pybind):
```c
typedef struct t4q_ctx t4q_ctx;
typedef struct { int n_gpu; int tp; int max_ctx; int kv_q8; int spec_k; int draft_vocab; int verbose; } t4q_params;
typedef struct { float temp; int top_k; float top_p; uint64_t seed; } t4q_sampling;
t4q_ctx* t4q_load(const char* gguf, const t4q_params*);
int  t4q_prefill(t4q_ctx*, const int32_t* ids, int n);                 /* returns 0 or error */
int  t4q_generate(t4q_ctx*, int32_t* out, int max_new, const t4q_sampling*, const int32_t* stop, int n_stop);
int  t4q_logits(t4q_ctx*, const int32_t* ids, int n, float* out);     /* debug: logits for every position */
int  t4q_dump(t4q_ctx*, const char* name, int layer, float* out, size_t cap);
void t4q_stats(t4q_ctx*, char* json, int cap);   /* tok/s, accept rate, per-graph ms, clocks */
void t4q_reset(t4q_ctx*);  void t4q_free(t4q_ctx*);
```
- **Binaries:** `libt4q.so`, a `t4q-bench` CLI (pp/tg like llama-bench), `probe`, `gemv_bench` and `oracle_dump`.

## 12. Source tree and Kaggle harness

```
/home/dih/kaggle-custom-kernals/t4q/
  include/t4q.h
  src/gguf.cpp  src/loader.cu  src/repack.cu  src/engine.cu (graphs, StepState, host loop)  src/api.cpp
  src/kernels/{gemv.cuh (P4/K4/K5/K6/Q8 bodies, segments, epilogues), gdn.cu, attn.cu, allreduce.cuh,
               lmhead.cu, sample.cu, prefill_gemm.cu, fa_prefill.cu, misc.cu}
  tools/{probe.cu, gemv_bench.cu, oracle_dump.cpp, mkkernel.py}
  tests/{ref.c, test_kernels.cu, validate.py}
  py/t4q.py (ctypes wrapper + tokenizer + driver)
  Makefile    # nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo -Xptxas -v; ~1-2 min on 4 vCPU
```

- **Build:** a plain Makefile with no CMake. On the Kaggle image, nvcc 12.8 is at `/usr/local/cuda`, and the libcuda stub is not needed because t4q uses only the runtime API. If NCCL is enabled, `-lnccl` links against the system 2.25.1.
- **Kernel packaging:** `tools/mkkernel.py <stage>` tars `t4q/` and inlines it as base64 into `kaggle/<stage>/t4q-<stage>.py`, the same pattern as `kaggle/baseline/template.py`. That writes `kernel-metadata.json` with:
  - `id` = `t4q-<stage>`, `is_private` true, `kernel_type` script, `enable_gpu` and `enable_internet` true;
  - `machine_shape` `NvidiaTeslaT4`;
  - the same pinned `docker_image` as the baseline (`gcr.io/kaggle-private-byod/python@sha256:37c64f7d...`);
  - `kernel_sources: ["t4-qwen38-baseline"]`, which supplies libllama for the oracle.

  It never copies anything from the `prior/` kernels: they contain leaked tokens, and t4q needs no secrets because every download is public.
- **Script flow:**
  1. Run `nvidia-smi` and log the clocks monitor in the background.
  2. Start the GGUF download to `/tmp/t4q/models` in a thread (Q4_0 takes 70 s) while `make` runs.
  3. Run the stage's tests and benches.
  4. Write `results.json`, `logs/`, `*.npy` (KL inputs only, small) to `/kaggle/working`.

  The stage scripts have a `DEADLINE` guard like the baseline's.
- **Loop:**
  - `kaggle kernels push -p kaggle/<stage>`
  - poll `kaggle kernels status t4q-<stage>`
  - `kaggle kernels output t4q-<stage> -p kaggle/<stage>/out`

  The account allows **2 concurrent GPU sessions**, and another agent's `cyber-frost-*` kernels were seen holding both. On "Maximum batch GPU session count", retry every 60 s.

## 13. Milestones

| M | deliverable | exit gate |
|---|---|---|
| M0 | `probe.cu`: P2P remote-store latency for 20 KB plus a flag (target ≤ 5 us); host-mapped ping-pong; NCCL 20 KB AR p50/p99; streaming read; clocks under load. `gemv_bench.cu`: P4 dp4a at m = 1..8 on the real shapes (5120 x 16480 / 7168 / 17408, 8704 x 5120, 3072 x 5120, 5120 x 124160 Q6_K) vs llama.cpp mmvq timings | P4 m = 1 ≥ 250 GB/s; m = 4 ≥ 230 GB/s; AR path chosen |
| M1 | correct engine: GGUF load + repack (P4, K5, K6, Q8, H), layer split (GPU0 layers 0-31, GPU1 layers 32-63 + head), straightforward unfused kernels, greedy decode, `t4q_logits`, `oracle_dump` | V0-V3 pass on Q4_0. Speed is irrelevant (expect ~8-12 tok/s) |
| M2 | fast GEMV: SoA layouts, segmented GEMVs, fused norm/q8 prologues and swiglu/argmax epilogues, `gdn_step` single pass | layer-split decode ≥ 16 tok/s at 4k (≥ 90% of the 17.7 roofline); V2 still passes |
| M3 | CUDA graphs + device StepState + 8-step async host loop + E0/H2 on GPU | layer split ≥ 17 tok/s; host CPU < 10%; graph nodes ≤ 345 per token |
| M4 | TP=2: shard repack, mailbox AR with weight prefetch, top-k exchange, fp16-payload prefill AR; prefill GEMM (mma m16n8k8) + `gdn_seq` + naive attention prefill | **decode ≥ 30 tok/s @4k Q4_0** (beats llama.cpp tensor's 21.1 by 1.4x); prefill ≥ 600 tok/s; V4 passes |
| M5 | MTP: `G_draft` (fused catch-up), `gdn_verify` + stash + `G_replay`, device accept; M5b 32k draft head; k sweep 2..5 | **≥ 60 tok/s on P0/P1 greedy**; V5 byte-identical; acceptance reported per prompt |
| M6 | production quality and long context: K4/K4L/IQ4_XS kernels and UD-Q4_K_M default; q8_0 KV (262k); sm_75 flash-attention prefill; spec rejection sampling; optional int8 MMQ prefill and chunked GDN | UD-Q4_K_M ≥ 30 tok/s plain / ≥ 55 MTP; 262k context loads; prefill ≥ 700 tok/s |

## 14. Risks

- **SM clock sag at 70 W** (Colab T4s have been seen at 600-800 MHz). dp4a at m = 4 and the K-quant unpacks get ALU-bound. Mitigation: log `clocks.sm` and `clocks_throttle_reasons` during every bench; fall back to swap-AB mma for verify; prefer P4/K4 over IQ formats.
- **P2P stores inside the GCP VM could be slow or incoherent** even though memcpy works. M0 measures this directly; if needed, use host-mapped memory (+~5 us per AR, about −2 tok/s).
- **fp16 overflow in the prefill activations** (outlier channels; `post_attention_norm` min 0.0039). Detect in M1; fall back to per-row-scaled fp16 or the int8 path.
- **Lower MTP acceptance with the main file's Q4_0 `blk.64`** than with unsloth's separate Q6_K MTP file. M5 measures both; t4q can load `blk.64` from the separate file.
- **Kaggle GPU session contention** (2 slots shared with another agent). Keep stage runs under 30 min and cache the built `libt4q.so` in kernel outputs.
