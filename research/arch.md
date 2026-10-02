# Qwen3.8-27B (`qwen3_5` / GGUF arch `qwen35`) exact decode math

Status: verified 2026-10-02 against
- HF `transformers` main: `src/transformers/models/qwen3_5/modeling_qwen3_5.py` (+ `configuration_qwen3_5.py`, `qwen3_next/modeling_qwen3_next.py`)
- `Qwen/Qwen3.8-27B/config.json` and `model.safetensors.index.json` (1199 tensors, 333 visual)
- llama.cpp master: `src/models/qwen35.cpp`, `src/models/delta-net-base.cpp`, `src/models/models.h`, `src/llama-graph.cpp`, `src/llama-model.cpp`, `ggml/src/ggml-cuda/{gated_delta_net.cu,ssm-conv.cu,rope.cu,unary.cu}`, `conversion/qwen.py` (the converter is now split into `conversion/*.py`), `common/speculative.cpp`
- vLLM main: `vllm/model_executor/models/{qwen3_next.py,qwen3_next_mtp.py,qwen3_5.py}`
- oxidize reference: `~/oxidize/oxidize-c/src/model/qwen35_delta.c`, `src/backends/cuda_qwen35.cu`
- **Real bytes**: I parsed the GGUF headers of `unsloth/Qwen3.8-27B-GGUF` `Qwen3.8-27B-Q4_0.gguf`, `UD-Q4_K_S.gguf`, `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` (HTTP range reads), pulled small F32 tensors from the Q4_0 file, and compared them with BF16 values from the HF safetensors. Every converter transform listed below is confirmed numerically, not just read from code.

Scratch copies of all sources: `/tmp/claude-1000/-home-dih-kaggle-custom-kernals/46241a61-a4b1-4d72-ad81-648a1054bb5d/scratchpad/src/` (header dumps `q40.txt`, `hdr_*.gguf.txt`, parser `ggufhdr.py`, size calc `sizes.py`).

---

## 0. Hyperparameters (text model)

| name | value | GGUF key |
|---|---|---|
| hidden `D` | 5120 | `qwen35.embedding_length` |
| layers | 64 trunk + 1 MTP | `qwen35.block_count` = **65**, `qwen35.nextn_predict_layers` = 1 |
| layer type | `il % 4 == 3` is full attention (3,7,...,63) = 16 layers; others DeltaNet = 48 | `qwen35.full_attention_interval` = 4 (no explicit recurrent array in unsloth files; loader computes `is_recr = (i < 64) && ((i+1) % 4 != 0)`) |
| FFN `F` | 17408, SiLU-gated | `feed_forward_length` |
| attn heads | Hq=24, Hkv=4, head_dim=256 (GQA group 6) | `attention.head_count{,_kv}`, `key_length`/`value_length` = 256 |
| RoPE | rotary dims 64 of 256 (partial 0.25), theta 1e7, NeoX pairing, IMROPE sections [11,11,10,0] | `rope.dimension_count`=64, `rope.freq_base`=1e7, `rope.dimension_sections` |
| DeltaNet | Hk=16 key heads x dk=128, Hv=48 value heads x dv=128, conv kernel 4 | `ssm.group_count`=16, `ssm.time_step_rank`=48, `ssm.state_size`=128, `ssm.inner_size`=6144, `ssm.conv_kernel`=4 |
| eps | 1e-6 for every RMSNorm, gated norm and L2 norm | `attention.layer_norm_rms_epsilon` |
| vocab | 248320, untied (`output.weight` separate from `token_embd.weight`) | |
| tokens | GGUF: bos 248044, **eos 248046** (`<|im_end|>`), pad 248055. HF config says eos 248044 (`<|endoftext|>`). Stop on both. | |
| sampling defaults (GGUF) | temp 1.0, top_k 20, top_p 0.95 | `general.sampling.*` |

Derived sizes: key_dim = 16*128 = 2048; value_dim = 48*128 = 6144; conv_dim = 2*2048 + 6144 = **10240**; q_proj out = 24*256*2 = **12288**; o_proj in = 6144; kv proj out = 4*256 = 1024.

---

## 1. Conventions

- GGUF/ggml matrix `W` with dims `[ne0, ne1]` = `[in, out]`, `ne0` contiguous. This is byte-identical to the PyTorch `[out, in]` row-major weight: output row `r` is `ne0` contiguous elements. `y[r] = sum_c W[r][c] * x[c]`. No transposes are inserted by the converter for any 2D weight here (only the V-head row/column permutation in section 6).
- `RMSNorm(x, w) = x * rsqrt(mean(x^2) + 1e-6) * w`, computed in fp32. **In GGUF the weight already contains the `+1`** (see 6). In HF, `Qwen3_5RMSNorm` is zero-centered: weight initialised to zeros and applied as `x_normed * (1.0 + w)` in fp32 before the cast back.
- Exception: the DeltaNet gated norm (`linear_attn.norm`, GGUF `ssm_norm`) is a plain-weight RMSNorm (init ones, no +1) both in HF and GGUF.
- `silu(x) = x * sigmoid(x)`, `softplus(x) = x > 20 ? x : log1p(exp(x))` (torch threshold 20; ggml CUDA uses `logf(1+expf(x))` with the same threshold).

---

## 2. Top level decode step (one token, position `p`, 0-based)

```
h = token_embd[tok]                          # row of 5120 (Q4_0 in the Q4_0 file; dequant to fp32)
for il in 0..63:
    r = h
    x = RMSNorm(h, attn_norm[il])            # GGUF blk.N.attn_norm.weight (HF input_layernorm, +1 baked)
    if il % 4 == 3:  a = FullAttn(il, x, p)
    else:            a = DeltaNet(il, x)
    h = r + a
    r = h
    x = RMSNorm(h, post_attention_norm[il])  # GGUF blk.N.post_attention_norm.weight
    h = r + W_down @ (silu(W_gate @ x) * (W_up @ x))
h_final = RMSNorm(h, output_norm)            # this exact tensor is also the MTP "h" input (llama.cpp t_h_nextn)
logits  = output.weight @ h_final            # 248320 x 5120 (Q6_K in both unsloth Q4 files)
```

No embedding scaling, no logit softcap, no biases anywhere (`attention_bias=false`, conv has no bias).
Residual stream: keep it **fp32**. `post_attention_norm` weights go down to 0.0039 in layer 0 (min over channels), which points at massive-activation channels; fp16 residuals are a real overflow risk on T4.

---

## 3. Full-attention block (layers 3,7,...,63 and the MTP layer)

GGUF tensors (per layer): `attn_q.weight [5120,12288]`, `attn_k.weight [5120,1024]`, `attn_v.weight [5120,1024]`, `attn_q_norm.weight [256]`, `attn_k_norm.weight [256]`, `attn_output.weight [6144,5120]`. (llama.cpp also accepts a fused `attn_qkv [5120, 12288+1024+1024]`; unsloth files use separate q/k/v for attention layers.)

```
qg = W_q @ x            # 12288
k  = W_k @ x            # 1024 = 4 heads x 256
v  = W_v @ x            # 1024
for head h in 0..23:                     # q and gate are INTERLEAVED PER HEAD:
    q_h    = qg[h*512 + 0   : h*512 + 256]
    gate_h = qg[h*512 + 256 : h*512 + 512]
q_h = RMSNorm(q_h, attn_q_norm)          # per head over 256, eps 1e-6, +1 baked
k_j = RMSNorm(k_j, attn_k_norm)          # per kv head j in 0..3
# v has no norm
RoPE(q_h), RoPE(k_j) at position p       # section 3.1
o_h = softmax_t( (q_h . K_{j,t}) / 16 ) @ V_{j,t},   j = h / 6  (grouped GQA, repeat_kv; t = 0..p)
o   = concat_h(o_h)                      # 6144
o   = o * sigmoid(gate)                  # elementwise over all 6144, head-aligned
a   = W_o @ o                            # 5120
```

- Scale: `1/sqrt(256) = 0.0625` (`head_dim**-0.5`; llama.cpp `kq_scale = 1/sqrtf(n_embd_head)`).
- Output gate is **sigmoid**, not swish. `config.json` has `"output_gate_type": "swish"`, but `Qwen3_5TextConfig` does not read that key (grep finds no use in transformers), and HF (`attn_output * torch.sigmoid(gate)`), vLLM (`attn_output * torch.sigmoid(gate)`) and llama.cpp (`ggml_sigmoid(gate)`) all use sigmoid. Treat the config key as dead metadata.
- Causal mask only, with no sliding window, sinks or softcap.
- KV cache per token: 16 layers x 2 x 4 heads x 256 = 32768 values = 64 KiB fp16 (32 KiB q8). At 262144 ctx that is 16 GiB fp16 / ~8.5 GiB q8_0. The MTP layer needs its own extra 4 KiB/token fp16.

### 3.1 RoPE (exact)

Rotary dims `n_rot = 64` (first 64 of 256), NeoX "rotate_half" pairing inside those 64:

```
for i in 0..31:
    theta = p * 1e7^(-2i/64)            # = p * inv_freq[i], inv_freq computed in fp32
    x0 = v[i]; x1 = v[i+32]
    v[i]    = x0*cos(theta) - x1*sin(theta)
    v[i+32] = x0*sin(theta) + x1*cos(theta)
# dims 64..255 unchanged
```

Why interleaved MRoPE reduces to that: HF builds 3 position streams (T,H,W) and `recomposition_frequencies` takes frequency index `i` from H if `i%3==1 && i<33`, from W if `i%3==2 && i<30`, else from T. llama.cpp IMROPE (`rope.cu`, `sector = i % 32`) does the same, with a 4th stream for the leftovers. For text tokens llama.cpp `llm_graph_input_pos::set_input` writes `pos,pos,pos,0`, and with sections [11,11,10] every one of the 32 frequency slots maps to T/H/W (none reaches the 4th stream). So for text-only decode every slot uses `p`, which is plain partial NeoX RoPE. HF text path also expands `position_ids` identically to all streams. Precision note: HF and ggml both form `p*inv_freq` in fp32. Match that, not fp64, if you want bit-close agreement at long context.

---

## 4. Gated DeltaNet block (48 layers)

GGUF tensors (per layer), Q4_0 file types in brackets:

| GGUF name | shape `[ne0,ne1]` | HF source | role |
|---|---|---|---|
| `blk.N.attn_qkv.weight` | [5120, 10240] Q4_0 | `linear_attn.in_proj_qkv` | rows: q 0..2047, k 2048..4095, v 4096..10239 (**V rows head-permuted**) |
| `blk.N.attn_gate.weight` | [5120, 6144] Q4_0 | `linear_attn.in_proj_z` | z (output gate), rows head-permuted |
| `blk.N.ssm_beta.weight` | [5120, 48] F32 | `in_proj_b` | beta logits, rows head-permuted |
| `blk.N.ssm_alpha.weight` | [5120, 48] F32 | `in_proj_a` | decay logits `a`, rows head-permuted |
| `blk.N.ssm_dt.bias` | [48] F32 | `dt_bias` (renamed `dt_proj.bias`) | permuted |
| `blk.N.ssm_a` | [48] F32 | `A_log` -> **stored as `-exp(A_log)`** | permuted |
| `blk.N.ssm_conv1d.weight` | [4, 10240] F32 | `conv1d.weight [10240,1,4]` squeezed | 4 taps per channel contiguous, V channels permuted |
| `blk.N.ssm_norm.weight` | [128] F32 | `linear_attn.norm.weight` (no +1) | gated RMSNorm weight, shared by all heads |
| `blk.N.ssm_out.weight` | [6144, 5120] Q5_K | `out_proj` | input columns head-permuted |

Persistent state per layer: conv state `[10240][3]` fp32 (last 3 pre-conv inputs), recurrent `S[48][128 v][128 k]` fp32.
Totals for 48 layers: S = 48*48*128*128*4 B = **144 MiB**, conv = 48*10240*3*4 B = 5.6 MiB. That's per sequence and independent of context length.

Decode step (indices are GGUF/llama.cpp order):

```
qkv = W_qkv  @ x        # 10240
z   = W_gate @ x        # 6144  -> 48 heads x 128
b   = W_beta @ x        # 48
a   = W_alpha@ x        # 48

# causal depthwise conv, kernel 4, with state; then SiLU
for c in 0..10239:
    win = [cs[c][0], cs[c][1], cs[c][2], qkv[c]]        # oldest .. newest
    y[c] = silu( sum_{j=0..3} w_conv[c][j] * win[j] )   # w_conv[c][3] multiplies the CURRENT token
    cs[c] = [cs[c][1], cs[c][2], qkv[c]]                 # state stores RAW (pre-conv, pre-silu) inputs
q = y[0:2048]    -> 16 heads x 128
k = y[2048:4096] -> 16 heads x 128
v = y[4096:]     -> 48 heads x 128

for kh in 0..15:
    q[kh] = q[kh] / sqrt(sum(q[kh]^2) + 1e-6) * (1/sqrt(128))   # L2 norm then the dk^-0.5 query scale
    k[kh] = k[kh] / sqrt(sum(k[kh]^2) + 1e-6)

for h in 0..47:                            # value head, GGUF order
    kh    = h % 16                         # TILED mapping in GGUF order (see section 6)
    beta  = sigmoid(b[h])
    g     = ssm_a[h] * softplus(a[h] + ssm_dt[h])   # ssm_a = -exp(A_log) < 0, so g < 0
    decay = exp(g)
    # S[h][i][j]: i = value dim (0..127), j = key dim (0..127); row i contiguous over j
    S *= decay
    for i: kv[i]    = sum_j S[i][j] * k[kh][j]
    for i: delta[i] = beta * (v[h][i] - kv[i])
    for i,j: S[i][j] += delta[i] * k[kh][j]
    for i: o[h][i]  = sum_j S[i][j] * q[kh][j]

    # gated RMSNorm (norm then gate), per head over 128
    o[h] = o[h] * rsqrt(mean(o[h]^2) + 1e-6) * ssm_norm * silu(z[h])

a_out = W_ssm_out @ concat_h(o[h])        # 6144 -> 5120
```

Equivalent fused form, as in ggml-cuda `gated_delta_net_cuda` (one warp per value column i, 128 lanes cover j):
`kv = decay * (S_old k)`, `delta = beta (v - kv)`, `S = decay*S_old + delta k^T`, `o = S q`. ggml applies the `1/sqrt(dk)` on the output (`attn*scale`) instead of on q. Same thing.

Notes:
- HF state is `[B, Hv, dk, dv]` (key-major), so `kv_mem = (S * k[...,None]).sum(-2)`. ggml and oxidize store the transpose `[Hv][dv][dk]`, which gives contiguous dot products over k. The math is the same.
- HF L2 norm `x * rsqrt(sum x^2 + 1e-6)`. ggml: `rms_norm(x, eps/n)/sqrt(n)`, the same value. oxidize uses `1/sqrt(max(sum, eps))`, which differs only for near-zero vectors.
- HF reference dtypes: projections and conv run in model dtype (bf16), and `g` is computed in fp32 (`A_log.float()`, `a.float()`). The recurrence casts q,k,v,beta,g to fp32 (`mamba_ssm_dtype: float32`). The gated norm normalizes in fp32, casts to model dtype, multiplies by the weight, then multiplies by `silu(gate.float())`. On T4 (no bf16), keep conv output, q/k/v, the state and the norm in fp32. Only the GEMV inputs/weights need be fp16/int.
- Value ranges seen in layer 0 (Q4_0 GGUF): `ssm_a` in [-0.338, -0.0038]. `ssm_dt.bias` in [-5.7, 19.25], so softplus can be about 19 and decay = exp(-0.04*19) ~ 0.47 for some heads, while others are ~1 (long memory). Never compute `exp(A_log)` in fp16.
- Prefill (seq > 1) uses the chunked algorithm (HF `torch_chunk_gated_delta_rule`, chunk 64; llama.cpp `build_delta_net_chunking` or the fused op looping tokens). It is mathematically identical to iterating the step above token by token, which is the validation oracle.

---

## 5. MLP

GGUF: `ffn_gate.weight [5120,17408]`, `ffn_up.weight [5120,17408]`, `ffn_down.weight [17408,5120]`.
`y = W_down @ (silu(W_gate @ x) * (W_up @ x))`. All 64 layers plus MTP are dense (no MoE).
In the unsloth Q4_0 file, `ffn_down` of layers 0-7 is **Q4_1**. Everything else in the FFN is Q4_0.

---

## 6. Converter transforms (HF -> GGUF), all verified numerically

From `conversion/qwen.py` (`Qwen3NextModel.modify_tensors`, `_LinearAttentionVReorderBase`, `_QwenMtpMixin`, `Qwen3_5TextModel` registered for `Qwen3_5ForConditionalGeneration`):

1. **`+1` on every `*norm.weight` except `linear_attn.norm.weight`**. Applies to `input_layernorm`, `post_attention_layernorm`, `model.norm`, `q_norm`, `k_norm`, `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`, `mtp.norm`, `mtp.layers.0.*norm`.
   Checks: HF `model.norm[0:3]` = 0.9609, 0.75, 0.9375, and GGUF `output_norm` = 1.9609, 1.75, 1.9375. HF `q_norm` (L3) = 0.2021, and GGUF 1.2021. HF `linear_attn.norm` = 0.8828, and GGUF `ssm_norm` = 0.8828 (no +1).
2. **`A_log -> -exp(A_log)`** stored as `ssm_a` (F32). Check: HF L0 head 0 gives -exp(A_log) = -0.0406, and GGUF `ssm_a[0]` = -0.0406.
3. **`dt_bias` renamed** to `ssm_dt.bias`.
4. **`conv1d.weight` squeezed** `[10240,1,4] -> [10240,4]` (GGUF dims `[4,10240]`). Tap order is unchanged. Check: first 4 values 0.0009, 0.0012, -0.0016, -0.0771 are identical in both.
5. **V-head reorder, grouped -> tiled** (Hk=16, Hv=48, r=3 v heads per k head). HF stores v heads grouped by k head: HF v head `hf = kh*3 + m` uses k head `kh` (HF does `q.repeat_interleave(3, dim=heads)`). GGUF stores **GGUF head `j` = HF head `(j % 16)*3 + j // 16`**, so GGUF head `j` pairs with k head `j % 16`. This lets ggml use a tiled broadcast (`iq1 = h_idx % neqk1` in the CUDA op). Applied to `in_proj_qkv` V rows only (q/k rows untouched), `in_proj_z` rows (blocks of 128), `in_proj_a`/`in_proj_b` rows (blocks of 1), `A_log` and `dt_bias` (blocks of 1), conv1d V channels (blocks of 128, q/k channels untouched), and `out_proj` **columns** (blocks of 128).
   Check: GGUF `ssm_a[0..5]` = -0.0406, -0.0095, -0.0591, -0.0198, -0.0074, -0.0115, which matches HF `-exp(A_log)` at indices 0, 3, 6, 9, 12, 15. GGUF `ssm_dt` = -3.4688, -1.0703, 18.375, ..., which equals HF `dt_bias[0,3,6,...]`.
   **If you load HF safetensors directly instead of GGUF, use `kh = h / 3` (grouped). If you load GGUF, use `kh = h % 16` (tiled).** Mixing these up is exactly the bug oxidize documents: it degenerates slowly into repeated tokens rather than producing obvious garbage.
6. Full-attention `q_proj` is **not** permuted (Qwen uses NeoX RoPE, so no llama-style q/k permute). The per-head `[q(256), gate(256)]` interleave is preserved.
7. MTP renames: `mtp.fc -> blk.64.nextn.eh_proj`, `mtp.pre_fc_norm_embedding -> blk.64.nextn.enorm`, `mtp.pre_fc_norm_hidden -> blk.64.nextn.hnorm`, `mtp.norm -> blk.64.nextn.shared_head_norm`, `mtp.layers.0.X -> blk.64.X` (standard names). `block_count` becomes 65 and `nextn_predict_layers` = 1. Visual tensors are dropped (they go to `mmproj-*.gguf`).
8. Legacy Qwen3-Next only: a fused `in_proj_qkvz` gets split per k-group into `attn_qkv` + `attn_gate`. Qwen3.5/3.8 already ships separate `in_proj_qkv` / `in_proj_z`, so this path is unused.

### 6.1 Complete GGUF tensor map (Q4_0 file: 866 tensors)

| GGUF | dims | Q4_0 file | UD-Q4_K_S file (varies per layer) | HF |
|---|---|---|---|---|
| `token_embd.weight` | [5120, 248320] | Q4_0 | Q3_K | `model.language_model.embed_tokens` |
| `output.weight` | [5120, 248320] | Q6_K | Q6_K | `lm_head` |
| `output_norm.weight` | [5120] | F32 | F32 | `model.norm` (+1) |
| `blk.N.attn_norm.weight` | [5120] | F32 | F32 | `input_layernorm` (+1) |
| `blk.N.post_attention_norm.weight` | [5120] | F32 | F32 | `post_attention_layernorm` (+1) |
| `blk.N.ffn_{gate,up}.weight` | [5120, 17408] | Q4_0 | Q4_K / IQ4_XS / IQ3_S / IQ2_* mix | `mlp.{gate,up}_proj` |
| `blk.N.ffn_down.weight` | [17408, 5120] | Q4_0 (Q4_1 for N=0..7) | IQ4_XS / IQ4_NL / Q4_K / Q5_K mix | `mlp.down_proj` |
| DeltaNet (N%4!=3) `attn_qkv` | [5120, 10240] | Q4_0 | IQ4_XS etc. | `linear_attn.in_proj_qkv` (V permuted) |
| `attn_gate` | [5120, 6144] | Q4_0 | Q5_K etc. | `linear_attn.in_proj_z` (permuted) |
| `ssm_alpha.weight` / `ssm_beta.weight` | [5120, 48] | F32 | Q8_0 | `in_proj_a` / `in_proj_b` (permuted) |
| `ssm_a` | [48] | F32 | F32 | `-exp(A_log)` (permuted) |
| `ssm_dt.bias` | [48] | F32 | F32 | `dt_bias` (permuted) |
| `ssm_conv1d.weight` | [4, 10240] | F32 | F32 | `conv1d.weight` squeezed (V ch permuted) |
| `ssm_norm.weight` | [128] | F32 | F32 | `linear_attn.norm` (no +1) |
| `ssm_out.weight` | [6144, 5120] | Q5_K | Q5_K etc. | `out_proj` (cols permuted) |
| Attn (N%4==3) `attn_q.weight` | [5120, 12288] | Q4_0 | IQ4_NL etc. | `self_attn.q_proj` |
| `attn_k.weight` / `attn_v.weight` | [5120, 1024] | Q4_0 | Q4_K / Q5_K | `k_proj` / `v_proj` |
| `attn_q_norm.weight` / `attn_k_norm.weight` | [256] | F32 | F32 | `q_norm` / `k_norm` (+1) |
| `attn_output.weight` | [6144, 5120] | Q4_0 | Q5_K | `o_proj` |
| MTP `blk.64.attn_{q,k,v,output}`, `attn_{q,k}_norm`, `attn_norm`, `post_attention_norm`, `ffn_*` | as attention layer | Q4_0 | Q6_K attn / Q4_K ffn | `mtp.layers.0.*` |
| `blk.64.nextn.eh_proj.weight` | [10240, 5120] | Q8_0 | Q4_K | `mtp.fc` |
| `blk.64.nextn.enorm.weight` / `hnorm.weight` | [5120] | F32 | F32 | `mtp.pre_fc_norm_embedding` / `_hidden` (+1) |
| `blk.64.nextn.shared_head_norm.weight` | [5120] | F32 | F32 | `mtp.norm` (+1) |
| (optional, absent) `nextn.embed_tokens`, `nextn.shared_head_head` | | | | fall back to `token_embd` / `output` |

Both unsloth main files already contain the MTP block (`blk.64.*`). `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` (1.37 GB, 18 tensors) is an "mtp_only" file: `token_embd`, `output`, `output_norm` plus `blk.64.*`, for use as a separate draft model.

Quant block formats you will need (from `ggml-common.h`): Q4_0 = {fp16 d; u8 qs[16]} per 32, `w = d*((nib)-8)`, low nibbles are elements 0..15, high nibbles 16..31. Q4_1 = {fp16 d, m; qs[16]}, `w = d*nib + m`. Q8_0 = {fp16 d; i8 qs[32]}. Q5_K = 176 B/256, Q6_K = 210 B/256, Q4_K = 144 B/256, IQ4_XS = 136 B/256 (super-block formats: port the ggml dequant).

### 6.2 Bytes read per decoded token (from the parsed headers)

| file | trunk (64 layers) | lm_head | token_embd (1 row read) | MTP block | **per-token weight bytes** |
|---|---|---|---|---|---|
| Q4_0 | 13.06 GiB | 0.97 GiB | 0.67 GiB total | 0.25 GiB | **14.03 GiB = 15.07 GB** |
| UD-Q4_K_S | 12.49 GiB | 0.97 GiB | 0.51 GiB | 0.33 GiB | 13.46 GiB = 14.45 GB |

Q4_0 trunk breakdown: ffn_down 3.03 GiB, ffn_gate 2.99, ffn_up 2.99, attn_qkv 1.32, ssm_out 0.97, attn_gate 0.79, attn_q 0.53, attn_output 0.26, ssm_alpha+beta 0.09 (F32), attn_k+v 0.09.
Per DeltaNet layer about 221 MB; per attention layer about 209 MB.
Bandwidth ceilings on T4 (320 GB/s peak, about 250-280 achievable): layer-split pipeline (one GPU active at a time) gives ~17-18 tok/s max. Tensor-parallel (both GPUs streaming concurrently, ~128 all-reduces (2 per layer) of 5120 floats per token over PCIe) gives ~30-35 tok/s max before MTP. The model (~14.7 GiB incl. embeddings) cannot fit on one 15 GiB T4 with KV/state, so a 2-GPU split is mandatory. Embeddings can stay on the host (a 1-row gather) to save 0.67 GiB VRAM.

---

## 7. MTP layer and llama.cpp MTP speculative decoding

HF `transformers` ignores `mtp.*` (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`). The reference semantics come from vLLM `qwen3_next_mtp.py` and llama.cpp `qwen35.cpp::graph_mtp`, and they agree.

MTP forward for draft position `p+1` (predicts the token at `p+2`):

```
inputs: tok = x_{p+1} (token just sampled / accepted), hprev = h_final at position p
        (target: post-output_norm hidden, the vector that produced logits for x_{p+1})
e   = token_embd[tok]                                  # shared embedding (mtp_use_dedicated_embeddings=false)
u   = W_eh @ concat( RMSNorm(e, enorm), RMSNorm(hprev, hnorm) )   # [10240] -> 5120, EMBEDDING FIRST
r = u; x = RMSNorm(u, blk64.attn_norm)
u = r + FullAttn(blk64, x, pos = p+1)                  # same block as section 3, own KV cache, same RoPE
r = u; x = RMSNorm(u, blk64.post_attention_norm)
u = r + MLP(blk64, x)
h' = RMSNorm(u, shared_head_norm)                      # also the next chained "hprev"
logits_draft = output.weight @ h'                      # shared lm_head (full 248320 vocab)
```

How llama.cpp drives it (`common/speculative.cpp`, `common_speculative_impl_draft_mtp`, qwen35 = single head, not chained):
- The draft context has its own KV cache for layer 64 only (`mtp_on_hybrid_qwen` uses a plain attention KV cache, no recurrent state).
- **Catch-up (`process`)**: after every target batch (prefill or verify), the MTP layer is run over the same tokens, pairing token `x_k` with target `h_{k-1}` (shifted right by one; the first token pairs with `pending_h` from the previous call). This fills the MTP KV cache for every accepted position. The last target h row is stashed as `pending_h`.
- **Draft**: start from `(id_last, pending_h)` at `pos0`. Sample greedily (top-k 10 sampler, take candidate 0). Stop if `p < p_min` (default 0) or after `n_max` tokens (default `n_max = 3`). Each next step feeds `(drafted_token, h' from the previous MTP step)` at `pos0+i+1`.
- **Verify**: the target decodes `[id_last, d1..dn]` in one batch. Accepted prefix + 1 bonus token. The DeltaNet state must be rolled back for rejected tokens. llama.cpp does this by keeping `n_rs_seq = n_max` per-token state snapshots (`ggml_gated_delta_net(..., K = n_rs_seq+1)` writes the last K states, plus K conv-state slots in `build_conv_state`). Then `seq_rm` picks the snapshot for the accepted length. A custom engine needs the same: store S (144 MiB) and conv state per verified position, or recompute from the accepted prefix.
- Cost per draft token: MTP block (~0.25 GiB) + **full lm_head (0.97 GiB Q6_K)**, about 1.2 GiB or ~8.5% of a target forward. A truncated-vocab draft head (top-N frequent tokens) would cut that a lot. That's an engine optimization, not part of the model.

---

## 8. Validation plan (numerical oracle)

1. **Op-level**: implement the scalar C/numpy step from sections 3-4 and compare custom CUDA kernels against it on random inputs (fp32 tolerance about 1e-5 relative for the state update).
2. **Layer-level vs llama.cpp**: build llama.cpp (CUDA, sm_75) on Kaggle and run `llama-eval-callback -m Qwen3.8-27B-Q4_0.gguf -p "..." -n 1` (examples/eval-callback). It dumps every named graph node. Useful names from `qwen35.cpp`: `attn_norm-N`, `linear_attn_qkv_mixed-N`, `z-N`, `beta_sigmoid-N`, `a_softplus-N`, `gate-N` (=g), `conv_output_silu-N`, `q_conv_predelta-N`, `k_conv_predelta-N`, `v_conv_predelta-N`, `attn_output-N` (DeltaNet core), `final_output-N` (after gated norm), `linear_attn_out-N`, `Qcur_normed-N`, `Kcur_normed-N`, `Qcur-N` (post-RoPE), `gate_reshaped-N`, `attn_pregate-N`, `attn_gated-N`, `attn_output-N`, `ffn_out-N`, `l_out-N`, `result_norm`, `result_output`; MTP: `mtp_enorm`, `mtp_hnorm`, `mtp_eh_proj`, ..., `h_nextn`. The same GGUF gives the same weights, so differences are kernel error only.
3. **Model-level**: top-1 agreement and KL vs llama.cpp logits (`llama-perplexity --kl-divergence-base` on a short corpus), plus a greedy 256-token generation match.
4. **HF cross-check without loading the model**: the BF16 safetensors can be range-read per tensor (index at `https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/model.safetensors.index.json`) to check permutation and +1 handling for any tensor, as done above.

---

## 9. Gotchas checklist

- GGUF V-head order is tiled (`kh = h % 16`). HF order is grouped (`kh = h / 3`). The same applies to z, a, b, A, dt_bias, conv V channels and out_proj columns.
- Norm weights in GGUF already have +1; do not add it again. `ssm_norm` never has +1.
- `ssm_a` is already `-exp(A_log)`. Decay = `exp(ssm_a * softplus(a + dt))`.
- q_proj output is per-head interleaved `[q256|gate256] x 24`, not `[all q | all gate]`.
- Attention output gate = sigmoid (ignore `output_gate_type: swish`). DeltaNet output gate = SiLU(z) after the RMSNorm.
- Query scale: attention uses 1/16 (head_dim 256). DeltaNet uses 1/sqrt(128) after the L2 norm.
- RoPE: only the first 64 of 256 dims, NeoX pairs (i, i+32), theta 1e7.
- Conv state holds raw pre-activation inputs. Tap 3 is the current token. SiLU comes after the conv.
- The MTP `h` input is the **post-final-norm** hidden. The concat order is `[enorm(embed), hnorm(h)]`.
- Keep the residual stream, the DeltaNet state, the L2/RMS norms and softplus/exp in fp32 on T4 (no bf16; fp16 range risk).
