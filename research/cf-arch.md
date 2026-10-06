# CYBER-FROST-3.8 (`qwen4exp` / GGUF arch `qwen4exp`) exact decode math

Status: verified 2026-10-05 against
- **Real bytes**: the full GGUF v3 header of `freakyskittle/CYBER-FROST-3.8-GGUF`
  `CYBER-FROST-3.8-Q2_K_S.gguf` (82,849,634,592 B) parsed over HTTP Range reads
  (`research/gguf_remote.py` -> `research/cf_tensors_Q2_K_S.txt`): 1256 tensors, 67 KV pairs,
  every name/dims/type/offset captured. This is the only trunk file with the MTP head in-file
  (grafted as `blk.48`, Q8_0, from `mtp-CYBER-FROST-3.8-Q8_0.gguf`).
- HF config: `Qwen/Qwen3.8-Flash-Next` `config.json` (`Qwen4ExpForConditionalGeneration`,
  text-only stack; the fine-tune checkpoint is `Blackfrost-AI/CYBER-FROST-3.8-BF16`).
- transformers main: `src/transformers/models/qwen4_exp/modeling_qwen4_exp.py` (trunk only; it
  ignores `mtp.*`).
- llama.cpp `src/models/qwen4exp.cpp` (local checkout b10975 for the trunk + upstream master
  for `graph_mtp`, which the local checkout predates). Attention here is **dense**: this file's
  `qwen4exp.attention.compress_ratios` is all zeros, so the QSA/indexer path never builds.

---

## 0. Hyperparameters (text model)

| name | value | GGUF key |
|---|---|---|
| hidden `D` | **2560** | `qwen4exp.embedding_length` |
| layers | 48 trunk + 1 MTP | `qwen4exp.block_count` = **49**, `qwen4exp.nextn_predict_layers` = 1 |
| layer type | `il % 4 == 3` full attention (3,7,...,47) = 12; else DeltaNet = 36 | `qwen4exp.attention.recurrent_layers[49]` (blk.48 also non-recurrent) |
| MoE | 512 experts, 10 per token, expert FFN 640, shared expert FFN 640 | `expert_count`, `expert_used_count`, `expert_feed_forward_length`, `expert_shared_feed_forward_length` |
| attention | Hq=24, Hkv=2 (GQA 12), head_dim 256 | `attention.head_count{,_kv}`, `attention.key_length` = 256 |
| RoPE | 64 of 256 dims (partial 0.25), theta 1e7, NeoX pairs, sections [11,11,10,0] | `rope.dimension_count`=64, `rope.freq_base`=1e7, `rope.dimension_sections` |
| DeltaNet | 16 key heads x 128, 48 value heads x 128, conv 4 | `ssm.group_count`=16, `ssm.time_step_rank`=48, `ssm.state_size`=128, `ssm.inner_size`=6144, `ssm.conv_kernel`=4 |
| hyper-connections | hc=4 streams, low-rank 320 | `qwen4exp.hyper_connection.count`=4, `.low_rank`=320 |
| PLE | at layer 1 only; 3-gram, 8 heads/n-gram (16 heads), 160-dim rows | `qwen4exp.ple.layers`=[1], `.ngram_size`=3, `.heads_per_ngram`=8, `embedding_length_per_layer_input`=160 |
| indexer | 4 heads x 128, top-2048, **compress_ratios all 0 -> inert** | `attention.indexer.*`, `attention.compress_ratios` |
| eps | 1e-6 everywhere | `attention.layer_norm_rms_epsilon` |
| vocab | 248320, untied | `output.weight` separate |
| tokens | bos 248044, **eos 248046** (`<|im_end|>`), pad 248044; config eos is 248044. Stop on both. | `tokenizer.ggml.*` |
| sampling defaults | temp 1.0, top_k 20, top_p 0.95 | `general.sampling.*` |
| context | 262144 | `qwen4exp.context_length` |

Derived: key_dim = 16*128 = 2048; value_dim = 48*128 = 6144; conv_dim = 2*2048+6144 = **10240**;
hc_dim = 4*2560 = **10240** (same number, different thing); q_proj out = 24*256*2 = 12288;
o_proj in = 6144. The tokenizer is byte-identical to the Qwen3.8-27B family (`pre = 'qwen35'`,
248320 vocab, same merges) and the chat template ships inside the GGUF.

DeltaNet sizes are IDENTICAL to Qwen3.8-27B (arch.md sections 4): qkv [2560,10240], gate
[2560,6144], ssm_out [6144,2560]; only `D` halves (5120 -> 2560). The GDN math is the same
except one gate (section 4.5 below).

---

## 1. The residual stream: 4 hyper-connection channels (no layer norms!)

There are no `attn_norm`/`post_attention_norm`/`output_norm` tensors. Instead the residual
state is `res_hc[D, hc=4, T]` (fp32), started as 4 copies of the embedding, and every mixer
input/output goes through an HC LoRA mixer. Per layer:

```
if il == 1: res_hc = PLE(res_hc, ple_emb)              # section 5
x      = hc_mix(res_hc, hc_attn_{norm,down,up}, il)     # the [D] input to the token mixer
a      = il%4==3 ? FullAttn(il, x, p) : DeltaNet(il, x)  # sections 4, 6
res_hc = hc_combine(res_hc, a, inject_attn)            # scatter back into the 4 streams
x      = hc_mix(res_hc, hc_ffn_{norm,down,up}, il)
m      = MoE(il, x) + sigmoid(gate_sh) * SharedFFN(il, x)   # section 7
res_hc = hc_combine(res_hc, m, inject_ffn)
```

### 1.1 `hc_mix(x[D,hc,T], w_norm[D,hc], w_down[10240,320], w_up[320,10240]) -> [D,T]`

```
xn[c,s,t] = rsqrt(mean_c(x[c,s,t]^2) + 1e-6) * x[c,s,t] * w_norm[c,s]     # grouped RMSNorm per stream
flat     = xn reshaped [10240, T]
lo       = silu( (w_down @ flat) * (1/hc) )          # [320, T]
gate     = sigmoid( w_up @ lo )                       # [10240, T]
gated    = flat * gate                                # [10240, T]
mixed    = mean over the 4 streams of gated           # [D, T]: sum_s gated[:,s,:] then * (1/hc)
```

### 1.2 `hc_combine` (with the inject from the *same* mixer that produced `x`)

```
w[s,t] = 2 * sigmoid( inject[:,s,t] / hc )           # inject = w_inject[10240,4] @ flat, per stream
res_hc[c,s,t] += w[s,t] * block_out[c,t]             # scatter: each stream gets its own weight
```

A zero inject gives weight 1 per stream (a plain residual add). The final mixer
`output_hc_{norm,down,up}` (top-level tensors, = `hc_head_*`) is an `hc_mix` and IS the output
norm; `result_norm` is its `mixed` output.

---

## 2. Top-level decode step (one token, position `p`)

```
e = token_embd[tok]                                  # row of 2560 (Q4_K)
res_hc = [e, e, e, e]                                # 4 streams
for il in 0..47:  (section 1 loop)
h_wide = res_hc                                      # [10240] - the MTP "h" input (section 8)
logits = output.weight @ hc_mix(h_wide, output_hc_*)  # [248320], Q4_K lm_head every token
```

Keep `res_hc`, the DeltaNet state, the norms and the gates in fp32 (T4 has no bf16).

---

## 3. PLE: the n-gram hashed embedding (layer 1 only)

Per token, 16 rows are gathered from `per_layer_token_embd` [160, 320001536] (Q4_0, 26.85 GiB,
35% of the file). The row index is computed host-side (llama.cpp does it on the host because
ggml has no int64/xor):

```
ctx[0] = tok; ctx[s] = the token s positions back (from the KV cells' token ids)
          an EOS (248044) in the window, or a missing predecessor, resets ctx[s..] to EOS
mixed_n = (ctx[0]*mult[0]) ^ (ctx[1]*mult[1]) ^ ... ^ (ctx[n-1]*mult[n-1])   # u64 wrap
row[h]  = mixed_n % vocab_size[h] + head_offset[h]
```

- `ple_layer_multipliers` (u64) = [23703573157769, 20109073645365, 8052911324071]
- bigram heads (n=2): h = 0..7, offsets 0..140000297, vocab ~20.000M each
- trigram heads (n=3): h = 8..15, offsets 160000374..300001275, vocab ~20.000M each
- total rows 320001536 (= last offset + last vocab). Per token the gather reads
  16 rows x 160 dims x 18/32 B = **1440 B**.

Then (all per token, `T` = 1 at decode):

```
emb   = concat_h gather = [160*16 = 2560]            # == D by construction
key   = ple_key @ emb          # [10240], then grouped RMSNorm (ple_norm_key [D,hc]) -> [10240]
query = grouped RMSNorm(res_hc, ple_norm_query)       # [10240]
s     = sum_rows(key * query) / sqrt(2560)           # one value per (stream, token)
gate  = sigmoid( sgn(s) * sqrt(clamp(|s|, 1e-6, inf)) )
value = ple_value @ emb         # [2560], broadcast to the 4 streams
gated = value * gate           # [D, hc, T]
gnorm = grouped RMSNorm(gated, ple_norm_conv)         # [10240]
conv  = depthwise causal conv over time, kernel 4, DILATION = ngram_size (3), over gnorm
        (per channel c: sum_k ple_conv1d[k,c] * gnorm[c, t-(3-k)*3]); then silu
res_hc += gated + conv          # wide injection into all 4 streams
```

The conv state (`(4-1)*3 = 9` columns x 10240 x fp32) is a per-sequence recurrent row and must
roll back on MTP rejections. Predecessors come from the attention KV cells' stored token ids.

---

## 4. Full-attention block (layers 3,7,...,47 and the MTP layer) - DENSE here

Same structure as arch.md section 3 (Qwen3.8-27B) with these changes: `D` 2560, **Hkv = 2**
(GQA group 12), `attn_q` [2560,12288] holds `[q256|gate256] x 24` interleaved per head, k/v
[2560,512], o [6144,2560]. Per-head RMSNorm on q (attn_q_norm) and each kv head (attn_k_norm),
partial NeoX RoPE (64/256, theta 1e7, IMROPE text-reduces to plain partial RoPE), scale
`1/sqrt(256) = 1/16`, **sigmoid** output gate, causal. The indexer tensors
(`indexer.{q,k}_proj` Q5_1, `indexer.{q,k}_norm` F32) exist in the file but
`compress_ratios` = 0, so no QSA: attention reads the whole KV. Upstream master builds
QSA top-k (top-2048 blocks) when ratios > 0; a future file may enable it, ours does not.

KV cache: f16 K+V, 2 heads x 256 = 1024 B per K, 1024 B per V, per token per layer = 4 KiB;
13 layers with KV (12 trunk + the MTP layer's own cache) = **52 KiB/token** (6.7 GiB at 262k,
872 MiB at 16k). Quantized KV is unverified for this arch (the pack author says it crashes;
treat f16 as the reference).

---

## 4.5. DeltaNet block (36 layers) - deltas vs arch.md section 4

Identical math and tensor set to Qwen3.8-27B (tiled V-head mapping `kh = h % 16`, ssm_a =
`-exp(A_log)`, decay `exp(ssm_a * softplus(alpha + dt_bias))`, conv state holds raw inputs,
L2 norm then `1/sqrt(128)` on q, gated RMSNorm with `ssm_norm`), with:
- `attn_qkv` [2560,10240], `attn_gate` [2560,6144], `ssm_out` [6144,2560] (D halves)
- **the output gate is sigmoid, not silu**: `o = norm(o) * ssm_norm * sigmoid(z_h)` (llama.cpp
  `build_norm_gated`: "the one numerical difference from Qwen3.5's GDN")
- persistent state: S [36][48][128][128] fp32 = 113 MiB + conv [36][10240][3] = 4.4 MiB
  (+ 1 MiB x n_rs_seq rollback copies for MTP verify, as in the 27B engine)

VERIFY at cf-m1 (same method as arch.md section 6): the tiled V-head permutation and the
`ssm_a`/`dt` row order against the BF16 safetensors (HTTP range reads of the
`Blackfrost-AI/CYBER-FROST-3.8-BF16` index). Assume qwen35's converter rules hold
(`conversion/qwen.py` `_LinearAttentionVReorderBase` is shared) until checked.

---

## 5. MoE FFN (every layer, incl. blk.48) + shared expert

```
logits_r = ffn_gate_inp @ x                # [512] (Q2_K [2560,512])
probs    = softmax(logits_r) in fp32       # llama.cpp SOFTMAX gating
(top10, w10) = top-k by prob, renormalized # w10 sums to 1 (norm=true)
y_e      = sum over the 10 experts e of w10_e * ( silu(gate_e @ x) * (up_e @ x) )
            gate_e, up_e: [2560 -> 640]; down_e: [640 -> 2560]; expert rows are 2-contiguous
            in the fused tensors ffn_{gate,up}_exps [2560,640,512] / ffn_down_exps [640,2560,512]
sh       = sigmoid(ffn_gate_inp_shexp @ x) * ( silu(gate_sh @ x) * (up_sh @ x) ) @ down_sh
out      = y_e + sh
```

`expert_weights_scale` defaults to 1.0 (verify at cf-m1). The shared expert has its own
sigmoid scalar gate per token. Per expert per token the reads are 537600 (gate Q2_K) +
537600 (up Q2_K) + 921600 (down Q4_0) = **1.997 MiB**.

---

## 6. The MTP draft block (blk.48, in-file, Q8_0)

Single draft layer, full-attention type, with its own hc mixers and MoE (512 experts, Q8_0:
3 x 891289600 B per layer = 2.49 GiB) and its own final mixer (`blk.48.nextn.hc_head_*`).
The trunk hands it the **wide residual before the final mixer** (`t_h_nextn` = res_hc flat
[10240, T]). Draft forward at position p+1, chained (h from the previous draft step):

```
e      = token_embd[tok_{p+1}]
h      = [D,hc] wide residual from the trunk (or the previous draft step)
e_norm = RMSNorm(e, enorm) repeated to the 4 streams        # enorm [2560]
h_norm = grouped RMSNorm(h, hnorm) per stream              # hnorm [2560, 4]
u_s    = eh_proj @ concat(e_norm, h_norm[:, s])            # eh_proj [5120,2560], PER STREAM
res'   = [u_0..u_3]                                        # the draft's 4-stream state
a      = FullAttn(blk.48, hc_attn_mix(res'), pos p+1)      # own KV cache (2 kv heads)
res'   = hc_combine(res', a, inject)
m      = MoE(blk.48, hc_ffn_mix(res')) + gated shared      # the Q8_0 experts
res'   = hc_combine(res', m, inject)
h_next = res'                                              # chains as the next h
logits = output.weight @ hc_mix(res', nextn.hc_head_*)     # shared lm_head, FULL vocab
```

Draft cost per token: attn 15.6 MiB + hc mixers ~19 MiB + eh_proj 13.3 MiB + experts 10 x
5.25 MiB (Q8_0) + shared 5 MiB + hc_head 4.7 MiB + **lm_head 341 MiB** = **~0.44 GiB**, of
which 77% is the lm_head (same observation as the 27B: a truncated draft vocab is the
engine-side lever, gated on quality). Catch-up/verify/rollback semantics are qwen35's
(arch.md section 7): after every target batch run the draft over the same tokens (pairing
token k with h_{k-1}), keep `pending_h`, greedy-sample drafts (top-k 10), verify as one
batch, roll back DeltaNet S + conv + **the PLE conv history** for rejections.

---

## 7. Complete tensor map (from the parsed header, Q2_K_S file)

Per-layer quant types (uniform across trunk layers; blk.48 is all Q8_0):

| family | shape | type | per-layer bytes |
|---|---|---|---|
| ffn_gate_exps / ffn_up_exps | [2560,640,512] | Q2_K | 275.25 MB each |
| ffn_down_exps | [640,2560,512] | Q4_0 | 471.86 MB |
| ffn_gate_inp (router) | [2560,512] | Q2_K | 420 KB |
| ffn_{gate,up}_shexp | [2560,640] | Q2_K | 525 KB each |
| ffn_down_shexp | [640,2560] | Q4_0 | 900 KB |
| ffn_gate_inp_shexp | [2560,1] | Q2_K | 840 B |
| attn_qkv (recurrent) | [2560,10240] | Q2_K | 8.20 MB |
| attn_gate (recurrent) | [2560,6144] | Q2_K | 4.92 MB |
| ssm_out | [6144,2560] | Q2_K | 4.92 MB |
| ssm_{alpha,beta} | [2560,48] | Q2_K | 39 KB each |
| ssm_conv1d | [4,10240] | F16 | 80 KB |
| ssm_{a,dt,norm} | 48/48/128 | F32 | tiny |
| attn_q (full) | [2560,12288] | Q2_K | 9.85 MB |
| attn_{k,v} (full) | [2560,512] | Q2_K | 420 KB each |
| attn_output (full) | [6144,2560] | Q2_K | 4.92 MB |
| attn_{q,k}_norm | 256 | F32 | 1 KB |
| indexer.{q,k}_proj | [2560,512]/[2560,128] | Q5_1 | 960/240 KB (**inert**: ratios 0) |
| hc_{attn,ffn}_{norm} | [2560,4] | F32 | 40 KB |
| hc_{attn,ffn}_down / _up | [10240,320]/[320,10240] | Q5_1 (mostly) | 2.34 MB each |
| hc_{attn,ffn}_inject | [10240,4] | Q5_1 | 30 KB |
| ple_key / ple_value (layer 1) | [2560,10240]/[2560,2560] | Q2_K | 8.20 MB / 2.05 MB |
| ple_conv1d | [4,10240] | F16 | 80 KB |
| ple_norm_{key,query,conv} | [2560,4] | F32 | 40 KB |

Top-level: `token_embd` [2560,248320] Q4_K 341 MB; `output` [2560,248320] Q4_K 341 MB;
`per_layer_token_embd` [160,320001536] **Q4_0 26.85 GiB**; `output_hc_{norm,down,up}`
(F32/Q5_1/Q5_1) 4.7 MB; `blk.48.nextn.{enorm,hnorm,eh_proj,hc_head_*}` (enorm F32 [2560],
hnorm F32 [2560,4], eh_proj Q8_0 [5120,2560] 13.3 MB).

Totals by type: Q4_0 48.04 GiB (experts-down 21.92 + the PLE table 26.85), Q2_K 25.51 GiB
(gate/up exps 13.14 + 13.14 minus the draft's Q8), Q8_0 2.58 GiB (the blk.48 graft), Q5_1
0.35 GiB (hc LoRA + inject), Q4_K 0.67 GiB (embed + lm_head), F32/F16 tiny.
**77.15 GiB of tensor data.** New dequant formats for the engine vs the 27B: **Q2_K**
(84 B/256: 2 super-blocks, fp16 d/dmin + 16 u8 4-bit scales + 64 u8 quants), **Q5_1**
(24 B/32), Q4_K (have), Q4_0 (have), F16 conv rows.

---

## 8. Per-token byte budget and the honest ceilings on Kaggle 2x T4

Per decoded token (all layers touched exactly once, lm_head once):

| part | MiB/token |
|---|---|
| MoE experts (10 x 1.997) + routers + shared | 1045 + 21 + 96 = **1162** |
| DeltaNet x36 (qkv+gate+out+small) | **651** |
| full-attn x12 (q,k,v,o; dense) | **187** |
| hc mixers x48 x2 + final | **461** |
| PLE (layer 1) + table rows (1440 B) | **10.3** |
| lm_head (Q4_K, full vocab) + token row | **341** |
| **total** | **~2.81 GiB = 3.02 GB** |

Ceilings (both GPUs streaming at the measured ~254 GB/s, TP=2, AR overhead amortized as in
the 27B engine - the per-layer all-reduces are 2560-wide, half the 27B's):

- dense single-stream, all-active-resident: 3.02 GB / 508 GB/s = 5.95 ms -> **~168 tok/s**
- MTP k=3 (draft ~0.44 GiB/token, verify M=4 with expert dedup ~1.5x): ~4.3 GB per ~3.4
  accepted tokens -> **~125-135 tok/s** spec ceiling

The residency wall is the whole game: the pool is 77.15 GiB (experts 50.6 GiB incl. the Q8
draft block, PLE table 26.85 GiB, everything else ~2.6 GiB) against 2 x 15.36 GiB VRAM
(~30.7 GiB) + ~29 GB host RAM + /tmp disk. Design consequences (see PLAN_CF.md):

- Must-resident in VRAM (touched every token): DeltaNet + attn + hc + PLE + routers + shared
  experts + lm_head + token_embd ~= **2.36 GiB** (+ KV, + 113 MiB DeltaNet S, + workspaces).
- The PLE table (26.85 GiB, 1440 B/token, random rows, index known right after sampling the
  previous token) goes to disk/page-cache with an async 16-row prefetch - it never needs to
  be resident.
- The routed experts (50.6 GiB pool, ~1.14 GiB/token touched) cannot all be resident. The
  two-tier plan: hottest experts per layer in VRAM (~24 GiB ~= 250-300 of 512 per layer),
  next tier pinned in host RAM (~25 GiB) read over PCIe (~8.4 GB/s measured sustained),
  coldest tail on /tmp disk. Equilibrium estimate at 90% VRAM hit: ~78 tok/s; at 95%:
  ~150 tok/s; the disk tail sets the floor if the router is flat.
- The two numbers that decide everything (both measured at cf-m0/cf-m2, never assumed): the
  Kaggle /tmp disk bandwidth, and the router concentration curve P(hit) vs resident-experts
  per layer on real text.

---

## 9. Gotchas checklist

- No layer norms at all: every norm is an hc grouped RMSNorm `[D,hc]` (the flat [10240]
  tensors), and the final `output_hc_*` mixer replaces `output_norm`.
- hc_mix returns the MEAN of the 4 gated streams; the residual keeps all 4 streams; the
  LoRA silu is scaled 1/hc BEFORE the up-projection.
- hc_combine weight: `2*sigmoid(inject/hc)` - a zero inject is identity.
- The DeltaNet output gate is SIGMOID (not silu, unlike qwen35); everything else in section 4
  of arch.md carries over, including the tiled V mapping (`kh = h%16`) - verify at cf-m1.
- The attention output gate is sigmoid; q_proj is `[q256|gate256] x 24` per head.
- GQA is 12:1 (24 q heads, 2 kv heads); head_dim 256 with partial RoPE on 64.
- The MoE renormalizes the top-10 probs; the shared expert is gated by its own sigmoid
  scalar, not folded into the router.
- The PLE gathers rows for BOTH the bigram (heads 0-7) and trigram (heads 8-15) hashes of the
  last 3 tokens; EOS in the window resets to EOS; the conv is DILATED by 3 (kernel 4,
  dilation 3) and its 9-column history rolls back with MTP.
- The MTP h input is the WIDE [10240] residual pre-final-mixer; eh_proj is applied PER STREAM
  to `[enorm(e) ; hnorm_s(h)]`; the draft's own hc_head mixer produces its logits via the
  shared lm_head.
- eos: stop on 248046 (`<|im_end|>`) AND 248044 (`<|endoftext|>`); the GGUF embeds the whole
  tokenizer + chat template (pre 'qwen35', same as the 27B family).
- The indexer tensors are dead weight in this file (compress_ratios 0): do not read them.
- KV stays f16 until quantized KV is validated for this arch.
