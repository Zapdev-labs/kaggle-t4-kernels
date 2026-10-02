# GGUF loading for the Qwen3.8-27B T4 engine

Source repo: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF (tree API: `https://huggingface.co/api/models/unsloth/Qwen3.8-27B-GGUF/tree/main?recursive=1`).
All tensor tables below were read remotely with HTTP Range requests by `research/gguf_remote.py` (dependency-free hand parser, fetches 8 MiB chunks until the tensor-info table is parsed). For every file, `data_start + sum(tensor bytes) == file size` exactly, so the parser and the block-size table are verified.
Dequant formulas were checked against `ggml/src/ggml-common.h` and `ggml/src/ggml-quants.c` in the local llama.cpp checkout (`~/Projects/llama.cpp/llama.cpp`, commit bb463c525). Qwen3.5-specific tensor transforms were checked against `conversion/qwen.py` and `src/models/qwen35.cpp` in the same tree.

Files in this directory:

| file | content |
|---|---|
| `gguf_remote.py` | `python3 gguf_remote.py <resolve-url> <size> <out.txt>` remote header dumper |
| `roles.py` | per-role type histogram (`python3 roles.py Q4_0 UD-Q4_K_M ...`, run inside `research/`) |
| `gguf_tensors_<variant>.txt` | full KV list + every tensor: `name \| type \| [ne0, ne1] \| offset_rel \| bytes` + per-type summary |
| `gguf_roles_by_variant.txt` | per-role (DeltaNet layer / full-attn layer / MTP) type counts and MB for each variant |

Variants dumped: `Q4_0`, `UD-Q4_K_M`, `UD-IQ4_XS`, `MTP-Q4_0` (requested), plus `UD-Q4_K_XL`, `UD-Q4_K_S`, `UD-Q5_K_M`, `Q8_0`.

---

## 1. GGUF v3 container

Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md . Everything is little-endian.

```
offset 0   uint32 magic      = 0x46554747  ("GGUF")
       4   uint32 version    = 3
       8   uint64 n_tensors  (866 for the main files, 18 for the MTP file)
      16   uint64 n_kv       (50 for UD files, 51 for Q4_0/Q8_0)
      24   kv[n_kv]
           tensor_info[n_tensors]
           <pad to general.alignment>          (default 32 if the key is absent; absent here -> 32)
data_start: tensor data, each tensor at data_start + offset (offset is a multiple of alignment)
```

`gguf_string` = `uint64 len` + `len` bytes UTF-8, no NUL.

KV entry: `gguf_string key; uint32 value_type; value`.
value_type: `0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool(1 byte), 8 string, 9 array, 10 u64, 11 i64, 12 f64`.
array = `uint32 elem_type; uint64 n; n elems` (elements of a string array are each `gguf_string`; arrays of arrays are legal but not used here).

tensor_info: `gguf_string name; uint32 n_dims; uint64 ne[n_dims]; uint32 ggml_type; uint64 offset`.
`ne[0]` is the fastest-varying (contiguous) dim. For a weight `[ne0, ne1] = [in, out]`, row `r` (one output channel) is the `ne0` input weights stored contiguously at `offset + r * row_bytes`, `row_bytes = ne0 / block_elems * block_bytes`. Quant blocks never straddle rows, so splitting by output rows (ne1) is always lossless; splitting along ne0 must land on a block boundary (256 for K/IQ-K types).

Header size here is dominated by the tokenizer arrays (248320 tokens + 247587 merges): header ends at 10,996,700 bytes (Q4_0/Q8_0, data_start 10,996,704) or 10,996,621 (UD files, data_start 10,996,640); MTP file data_start 10,945,408. So a loader needs ~11 MB of header before the first tensor.

ggml_type ids, block elements and block bytes (bpw = 8*bytes/elems):

| id | type | elems | bytes | bpw |
|---|---|---|---|---|
| 0 | F32 | 1 | 4 | 32 |
| 1 | F16 | 1 | 2 | 16 |
| 2 | Q4_0 | 32 | 18 | 4.5 |
| 3 | Q4_1 | 32 | 20 | 5.0 |
| 8 | Q8_0 | 32 | 34 | 8.5 |
| 10 | Q2_K | 256 | 84 | 2.625 |
| 11 | Q3_K | 256 | 110 | 3.4375 |
| 12 | Q4_K | 256 | 144 | 4.5 |
| 13 | Q5_K | 256 | 176 | 5.5 |
| 14 | Q6_K | 256 | 210 | 6.5625 |
| 17 | IQ2_XS | 256 | 74 | 2.3125 |
| 18 | IQ3_XXS | 256 | 98 | 3.0625 |
| 20 | IQ4_NL | 32 | 18 | 4.5 |
| 21 | IQ3_S | 256 | 110 | 3.4375 |
| 22 | IQ2_S | 256 | 82 | 2.5625 |
| 23 | IQ4_XS | 256 | 136 | 4.25 |
| 30 | BF16 | 1 | 2 | 16 |

### Model KVs present (arch `qwen35`)

```
qwen35.block_count = 65          (64 main + 1 MTP; blk.64 is the MTP layer)
qwen35.nextn_predict_layers = 1
qwen35.context_length = 262144
qwen35.embedding_length = 5120
qwen35.feed_forward_length = 17408
qwen35.attention.head_count = 24, head_count_kv = 4, key_length = value_length = 256
qwen35.attention.layer_norm_rms_epsilon = 1e-6
qwen35.rope.freq_base = 1e7, rope.dimension_count = 64, rope.dimension_sections = [11,11,10,0]
qwen35.full_attention_interval = 4         (layer il is full attention iff (il+1) % 4 == 0 -> 3,7,...,63)
qwen35.ssm.conv_kernel = 4, ssm.state_size = 128, ssm.group_count = 16 (k heads),
qwen35.ssm.time_step_rank = 48 (v heads), ssm.inner_size = 6144 (48*128)
tokenizer.ggml.model = gpt2, pre = qwen35, 248320 tokens, bos 248044, eos 248046, pad 248055
general.file_type: Q4_0 -> 2, MTP file -> 14 (Q4_K_S label)
general.sampling: top_k 20, top_p 0.95, temp 1.0
imatrix: UD files built with imatrix (496 entries)
```

### Tensor list (identical names and shapes in every main file; only types differ)

Global: `token_embd.weight [5120, 248320]`, `output.weight [5120, 248320]` (untied), `output_norm.weight [5120] F32`.

DeltaNet layer (il % 4 != 3, 48 layers), 14 tensors each:

| tensor | shape [ne0, ne1] | meaning |
|---|---|---|
| attn_norm.weight | [5120] F32 | input RMSNorm (already `1 + w`) |
| attn_qkv.weight | [5120, 10240] | rows: q 0..2047 (16x128), k 2048..4095, v 4096..10239 (48x128, **tiled order**, see below) |
| attn_gate.weight | [5120, 6144] | z (output gate), 48x128, tiled order |
| ssm_alpha.weight | [5120, 48] | a projection (F32 in Q4_0, Q8_0 in UD) |
| ssm_beta.weight | [5120, 48] | b projection |
| ssm_a | [48] F32 | already `-exp(A_log)` |
| ssm_dt.bias | [48] F32 | dt bias |
| ssm_conv1d.weight | [4, 10240] F32 | depthwise causal conv, channels in qkv order (V part tiled) |
| ssm_norm.weight | [128] F32 | gated RMSNorm weight (NOT +1 shifted) |
| ssm_out.weight | [6144, 5120] | out_proj |
| post_attention_norm.weight | [5120] F32 | pre-FFN RMSNorm (`1 + w`) |
| ffn_gate/ffn_up.weight | [5120, 17408] | SwiGLU |
| ffn_down.weight | [17408, 5120] | |

Full-attention layer (il = 3, 7, ..., 63; 16 layers), 11 tensors each:

| tensor | shape | meaning |
|---|---|---|
| attn_norm.weight | [5120] F32 | |
| attn_q.weight | [5120, 12288] | per head h: rows 512h..512h+255 = q, 512h+256..512h+511 = output gate (interleaved per head, NOT q-block then gate-block) |
| attn_k.weight / attn_v.weight | [5120, 1024] | 4 kv heads x 256 |
| attn_q_norm / attn_k_norm.weight | [256] F32 | per-head RMSNorm (`1 + w`) |
| attn_output.weight | [6144, 5120] | |
| post_attention_norm, ffn_gate/up/down | as above | |

Gate application: `out = attn(q,k,v) * sigmoid(gate)` before `attn_output`. DeltaNet: `g = ssm_a * softplus(alpha_proj + ssm_dt.bias)` (log-decay), `beta = sigmoid(beta_proj)`, output `rmsnorm(o; ssm_norm) * silu(z)`.

MTP layer `blk.64` (17 tensors in main files): full-attention block (attn_q/k/v/output, q/k norms, attn_norm, post_attention_norm, ffn_*) plus `nextn.eh_proj.weight [10240, 5120]`, `nextn.enorm`, `nextn.hnorm`, `nextn.shared_head_norm` (all [5120] F32). eh_proj input is `concat(enorm(embed(tok)), hnorm(h))` (embedding half first, per HF `pre_fc_norm_embedding`/`pre_fc_norm_hidden` order); logits go through `shared_head_norm` then the shared `output.weight`.

**Important conversion facts (llama.cpp `conversion/qwen.py`)** that a custom engine must honour when reading these GGUFs, or undo when comparing against HF:
1. All `*norm.weight` except `linear_attn.norm` (= `ssm_norm`) had `+1` added. So use `y = x * rsqrt(mean(x^2)+eps) * w` directly.
2. `ssm_a = -exp(A_log)`.
3. V heads are reordered from grouped (HF: v head `g*3 + r` belongs to k head g) to tiled order: GGUF v head index `r*16 + g` (r in 0..2, g in 0..15). This applies to the V rows of attn_qkv, attn_gate rows, ssm_alpha/ssm_beta rows, ssm_a, ssm_dt.bias, the V channels of ssm_conv1d, and the input columns of ssm_out. Consequence: v head `j` uses k head `j % 16` (not `j / 3`). The oxidize reference (`~/oxidize/oxidize-c/src/model/qwen35_delta.c`) must be checked for which convention it assumes before porting.
4. `conv1d` is squeezed to [4, 10240].

---

## 2. Block layouts and dequant formulas

Notation: `d`, `dmin`, `m` are fp16 unless stated; `q` integers; element index `e` within the block. All formulas are bit-exact transcriptions of `dequantize_row_*` in ggml-quants.c.

### Q4_0 (18 B / 32)
```
struct { half d; uint8 qs[16]; }
e = j      (j 0..15): q = qs[j] & 0xF
e = j + 16 (j 0..15): q = qs[j] >> 4
w = d * (q - 8)
```

### Q4_1 (20 B / 32)
```
struct { half d; half m; uint8 qs[16]; }      same nibble map as Q4_0
w = d * q + m
```

### Q8_0 (34 B / 32)
```
struct { half d; int8 qs[32]; }   w = d * qs[e]
```

### Q4_K (144 B / 256)
```
struct { half d; half dmin; uint8 scales[12]; uint8 qs[128]; }   offsets 0,2,4,16
8 sub-blocks of 32, 6-bit scale sc[s] and 6-bit min m[s]:
  s < 4 : sc = scales[s] & 63;                          m = scales[s+4] & 63
  s >= 4: sc = (scales[s+4] & 0xF) | ((scales[s-4] >> 6) << 4)
          m  = (scales[s+4] >> 4)  | ((scales[s]   >> 6) << 4)
chunk c = 0..3 (64 elems), l = 0..31:
  e = 64c + l      : q = qs[32c + l] & 0xF,  s = 2c
  e = 64c + 32 + l : q = qs[32c + l] >> 4,   s = 2c + 1
w = d*sc[s]*q - dmin*m[s]
```
Note the nibble pairing: one byte holds elements 32 apart (two different sub-blocks), unlike Q4_0/IQ4_XS where it holds elements 16 apart in the same block.

### Q5_K (176 B / 256)
```
struct { half d; half dmin; uint8 scales[12]; uint8 qh[32]; uint8 qs[128]; }   offsets 0,2,4,16,48
scales/mins and low-nibble map exactly as Q4_K.
  e = 64c + l      : q = (qs[32c+l] & 0xF) + 16*((qh[l] >> (2c))   & 1)
  e = 64c + 32 + l : q = (qs[32c+l] >> 4)  + 16*((qh[l] >> (2c+1)) & 1)
w = d*sc[s]*q - dmin*m[s]          (q in 0..31)
```

### Q6_K (210 B / 256)
```
struct { uint8 ql[128]; uint8 qh[64]; int8 scales[16]; half d; }   offsets 0,128,192,208 (d is LAST)
half n = 0,1 (128 elems): ql' = ql + 64n, qh' = qh + 32n, sc' = scales + 8n; l = 0..31, is = l/16
  e = 128n +      l : q = ((ql'[l]    & 0xF) | (((qh'[l] >> 0) & 3) << 4)) - 32, scale sc'[is+0]
  e = 128n + 32 + l : q = ((ql'[l+32] & 0xF) | (((qh'[l] >> 2) & 3) << 4)) - 32, scale sc'[is+2]
  e = 128n + 64 + l : q = ((ql'[l]    >> 4)  | (((qh'[l] >> 4) & 3) << 4)) - 32, scale sc'[is+4]
  e = 128n + 96 + l : q = ((ql'[l+32] >> 4)  | (((qh'[l] >> 6) & 3) << 4)) - 32, scale sc'[is+6]
w = d * sc * q        (equivalently the scale index for element e is e/16)
```

### IQ4_NL (18 B / 32) and the IQ4 table
```
kvalues_iq4nl[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 }
struct { half d; uint8 qs[16]; }       nibble map as Q4_0
w = d * kvalues_iq4nl[q]
```

### IQ4_XS (136 B / 256)
```
struct { half d; uint16 scales_h; uint8 scales_l[4]; uint8 qs[128]; }   offsets 0,2,4,8
sub-block ib = 0..7:
  ls = ((scales_l[ib/2] >> (4*(ib%2))) & 0xF) | (((scales_h >> (2*ib)) & 3) << 4)    (6-bit)
  dl = d * (ls - 32)
  j = 0..15: e = 32ib + j      -> q = qs[16ib + j] & 0xF
             e = 32ib + 16 + j -> q = qs[16ib + j] >> 4
w = dl * kvalues_iq4nl[q]
```

### Lower-bit types that the UD files also contain

The unsloth "UD" files are per-tensor mixes, so a loader for them must also handle these (grid tables `iq3s_grid[512]` (u32), `iq3xxs_grid[256]` (u32), `iq2s_grid[1024]` (u64), `iq2xs_grid[512]` (u64), `ksigns_iq2xs[128]`, `kmask_iq2xs[8] = {1,2,4,...,128}` are copied verbatim from ggml-common.h):

- **Q3_K** (110 B): `{ uint8 hmask[32]; uint8 qs[64]; uint8 scales[12]; half d; }`. 16 6-bit scales via the kmask unpack in `dequantize_row_q3_K`. For half n (128 elems), j = 0..3 (shift 2j), l = 0..15: `e = 128n + 32j + l` uses `qs[32n+l]`, `hmask[l]` bit `4n+j`, scale `8n+2j`; `e = 128n+32j+16+l` uses `qs[32n+16+l]`, `hmask[16+l]`, scale `8n+2j+1`. `w = d*(sc-32)*(((qs>>2j)&3) - (hbit ? 0 : 4))`.
- **Q2_K** (84 B): `{ uint8 scales[16]; uint8 qs[64]; half d; half dmin; }`. Same element map as Q3_K. `w = d*(sc&0xF)*((qs>>2j)&3) - dmin*(sc>>4)`.
- **IQ3_S** (110 B): `{ half d; uint8 qs[64]; uint8 qh[8]; uint8 signs[32]; uint8 scales[4]; }`, scale per 32 `d*(1+2*nibble)`, 9-bit index into iq3s_grid (4 values per entry), sign bit per element. Grid magnitudes are exactly {1,3,5,...,15}.
- **IQ3_XXS** (98 B): `{ half d; uint8 qs[96]; }` (64 grid bytes + 8x u32 scale/sign words), `db = d*(0.5 + (aux>>28))*0.5`, 7-bit sign index into ksigns. Grid magnitudes {4,12,20,28,36,44,52,62}.
- **IQ2_S** (82 B) / **IQ2_XS** (74 B): scale per 16 `d*(0.5+nibble)*0.25`, 10-/9-bit index into 8-wide grids; magnitudes {8,25,43}.

Key fact for repacking: every one of these types (and Q4_0/Q4_1/Q4_K/IQ4_*) dequantizes as `w = scale_g * LUT[code] (+ min_g)` with a 16-entry (or smaller) signed LUT, i.e. they all fit **losslessly** into a 4-bit-code format with a per-tensor LUT: IQ3_S has 16 signed values (±1..±15 odd), IQ3_XXS 16 (±8 magnitudes), IQ2 6 values, Q3_K 8 values, Q2_K 4. Only Q5_K, Q6_K and Q8_0 need more than 4 bits.

---

## 3. Which types each file actually uses (Qwen3.8-27B)

File sizes (HF tree API, bytes) for all relevant files:

| file | bytes | GiB |
|---|---|---|
| Q4_0 | 16,056,478,688 | 14.95 |
| Q4_1 | 17,540,705,248 | 16.34 |
| Q8_0 | 29,047,086,048 | 27.05 |
| UD-IQ3_XXS | 10,934,860,704 | 10.18 |
| UD-Q3_K_XL | 13,146,393,504 | 12.24 |
| UD-IQ4_XS | 14,252,845,984 | 13.27 |
| UD-Q4_K_S | 15,358,213,024 | 14.30 |
| UD-Q4_K_M | 16,464,440,224 | 15.33 |
| UD-Q4_K_XL | 17,559,178,144 | 16.35 |
| UD-Q5_K_S / M / XL | 18,665,753,504 / 19,771,509,664 / 20,876,938,144 | 17.38 / 18.41 / 19.44 |
| UD-Q6_K | 21,983,677,344 | 20.47 |
| BF16 (2 shards) | 49,986,159,616 + 4,671,576,000 | 50.9 |
| MTP/mtp-Qwen3.8-27B-Q4_0 | 1,369,590,656 | 1.28 |
| imatrix_unsloth.gguf | 13,642,656 | |

Bytes that decode must stream per token (64 main layers + output head; excludes the token_embd row gather and the MTP layer), and type sets:

| variant | total tensor bytes | decode bytes/token | token_embd | blk.64 MTP | ggml types present |
|---|---|---|---|---|---|
| Q4_0 | 16.045 GB | **15.065 GB** | Q4_0 0.715 GB | 0.265 GB | Q4_0 (352), Q4_1 (8, ffn_down of 8 layers), Q5_K (48 = all ssm_out), Q6_K (output), Q8_0 (MTP eh_proj), F32 (456: norms, conv, ssm_a/dt, **ssm_alpha/beta as F32**) |
| UD-IQ4_XS | 14.242 GB | **13.345 GB** | Q3_K 0.546 GB | 0.351 GB | IQ4_XS 211, IQ3_S 46, Q4_K 55, IQ3_XXS 17, Q3_K 9, Q6_K 8, IQ2_S 7, IQ2_XS 2, IQ4_NL 1, Q2_K 1, Q8_0 98, F32 360; output Q5_K |
| UD-Q4_K_S | 15.347 GB | 14.450 GB | Q3_K | 0.351 GB | IQ4_XS 172, Q4_K 95, Q5_K 80, IQ3_S 15, Q3_K 13, IQ4_NL 7, IQ3_XXS 5, IQ2_S/XS 1+1, Q8_0 99; output Q6_K |
| UD-Q4_K_M | 16.453 GB | **15.387 GB** | Q4_K 0.715 GB | 0.351 GB | Q5_K 131, IQ4_XS 117, Q4_K 104, Q6_K 30, Q3_K 7, IQ4_NL 7, IQ3_S 4, Q8_0 106, F32 360; output Q6_K |
| UD-Q4_K_XL | 17.548 GB | **16.482 GB** | Q4_K 0.715 GB | 0.351 GB | Q5_K 191, IQ4_XS 70, Q4_K 69, Q6_K 56, IQ4_NL 6, Q3_K 3, IQ3_S 1, Q8_0 110; output Q6_K |
| UD-Q5_K_M | 19.761 GB | 18.694 GB | Q4_K | 0.351 GB | Q5_K 189, Q6_K 160, IQ4_XS 19, Q4_K 12, IQ4_NL 2, Q8_0 124 |
| Q8_0 | 29.036 GB | 27.234 GB | Q8_0 1.351 GB | 0.451 GB | Q8_0 506, F32 360 |
| MTP-Q4_0 (separate file) | 1.359 GB | n/a | Q3_K 0.546 GB | 0.266 GB | blk.64: attn Q6_K, ffn + eh_proj Q4_K; plus its own Q3_K `token_embd` and `output` (0.546 GB each) |

Per-role patterns (full table in `gguf_roles_by_variant.txt`):
- UD files: `ssm_alpha`, `ssm_beta` are always Q8_0; norms, conv1d, ssm_a, ssm_dt.bias always F32; MTP blk.64 is Q6_K (ffn/attn_q/attn_output/eh_proj) + Q8_0 (attn_k/v) in every UD file.
- UD types vary **layer by layer** for every big matrix (e.g. UD-Q4_K_M dn:ffn_down = 18 IQ4_XS, 16 Q5_K, 9 Q4_K, 2 IQ3_S, 2 IQ4_NL, 1 Q3_K). So a kernel cannot be specialized per role; it must dispatch on the per-tensor type (or the loader must normalize the format, section 4).
- Q4_0 is near-uniform: everything big is Q4_0 except all 48 ssm_out (Q5_K), ffn_down of 8 layers (Q4_1), output (Q6_K).

**The main GGUFs already contain the MTP layer (blk.64).** The separate `MTP/mtp-...-Q4_0.gguf` is only needed for main files converted without MTP; for our engine skip it and reuse the main file's `token_embd`/`output` (its own embed/head are a lower-quality Q3_K duplicate, 1.09 GB wasted).

---

## 4. Recommendation: source file and T4 repack

### Memory budget (2x T4, 15360 MiB each, ~14.6 GiB usable after context)
- KV cache (16 full-attn layers, 4 kv heads x 256, K+V): 64 KiB/token in fp16 (+1/16 for the MTP layer). 32k ctx = 2 GiB, 131k = 8 GiB, 262k = 16 GiB fp16 or ~8.5 GiB at q8.
- DeltaNet state: 48 x 48 x 128 x 128 fp32 = 151 MB; conv state 48 x 3 x 10240 fp32 = 5.9 MB. Negligible.
- Token embedding can stay in host RAM (one row gather per token) and save 0.5 to 0.7 GB of VRAM.
- Workspace: largest prefill dequant tile 17408 x 5120 fp16 = 178 MB, logits 248320 fp32 = 1 MB.

With tensor parallel (each GPU holds half of every matrix), UD-Q4_K_M needs (15.39 + 0.35) / 2 = 7.9 GB = 7.3 GiB per GPU, leaving about 7 GiB per GPU for KV: 131k fp16 or 262k q8 fits. UD-Q4_K_XL needs 7.8 GiB per GPU, still 131k fp16 / ~200k+ q8.

### Speed (decode is weight-bandwidth bound)
Decode time per token ~= decode_bytes / (n_gpu_in_parallel * achieved_BW) + sync overhead. Assume ~260 GB/s achieved per T4 (81% of 320).

| source | bytes/token | 2-GPU layer split (sequential) | 2-GPU tensor parallel (+~3 ms for 128 allreduces) |
|---|---|---|---|
| UD-IQ4_XS | 13.35 GB | ~19.5 tok/s | ~35 tok/s |
| UD-Q4_K_S | 14.45 GB | ~18 tok/s | ~32 tok/s |
| Q4_0 | 15.07 GB | ~17 tok/s | ~31 tok/s |
| UD-Q4_K_M | 15.39 GB | ~17 tok/s | ~30 tok/s |
| UD-Q4_K_XL | 16.48 GB | ~16 tok/s | ~28 tok/s |
| UD-Q5_K_M | 18.69 GB | ~14 tok/s | ~25 tok/s |

These are ceilings before MTP speculation (which multiplies by accepted tokens per step, typically 1.5 to 2x with one MTP layer). The prior KoboldCpp run at 9.2 tok/s is about 55% of the layer-split ceiling for its Q4_K_S file.

### Pick
1. **Bring-up / kernel correctness: `Qwen3.8-27B-Q4_0.gguf`** (16.06 GB). Only 6 types (F32, Q4_0, Q4_1, Q5_K, Q6_K, Q8_0), Q4_0 is trivially vectorizable, and it is the same speed class as UD-Q4_K_M. Good first target for end-to-end greedy-decode parity against llama.cpp.
2. **Production default: `Qwen3.8-27B-UD-Q4_K_M.gguf`** (16.46 GB file, 15.39 GB/token). imatrix-guided K/IQ4 mix with Q5_K/Q6_K on sensitive tensors; better quality than Q4_0 at +2% bytes, fits 131k fp16 KV with TP. Download time at ~280 MB/s ~= 60 s.
3. **Quality option: UD-Q4_K_XL** (+7% bytes, -7% speed). **Speed option: UD-IQ4_XS** (-13% bytes, +15% speed, but contains IQ3/IQ2 tensors in 40 layers).

All three UD files share one loader if it implements the type set {F32, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL, Q3_K, IQ3_S} (+ Q2_K, IQ3_XXS, IQ2_S, IQ2_XS for IQ4_XS/Q4_K_S). I would not requantize from BF16 (51 GB download, would also lose the imatrix work) and not requantize between GGUF types (compounding error). Repack losslessly instead.

### Repack into a T4 layout at load time, on GPU: yes
GGUF block structs are bad for coalesced 128-bit loads (Q4_0 18 B, IQ4_XS 136 B, Q6_K 210 B are not 16-byte multiples, and scales are interleaved with codes). Repack per tensor into separate planes, which is a pure bit shuffle and costs nothing in quality:

- **Format A, "u4+LUT"** (all 4-bit-or-less types: Q4_0, Q4_1, Q4_K, IQ4_NL, IQ4_XS, Q3_K, Q2_K, IQ3_S, IQ3_XXS, IQ2_*): code plane `uint4 codes[rows][K/32]` (32 weights per 16 B, k-order interleaved for the lop3/prmt fp16 magic-number dequant trick), per-tensor 16-entry fp16 LUT (identity-minus-8 for Q4_0, `kvalues_iq4nl` for IQ4, odd values for IQ3_S, etc.), scale plane: keep the native two-level scales (fp16 `d`/`dmin` per 256 + 6/8-bit int `sc`/`m` per 32 or 16) rather than expanding to fp16 per group, because expanding adds ~0.5 bpw (+11% bytes) and fp16 rounding of `d*sc`. Upcasting the 2/3-bit types to 4 bits costs bytes only for those tensors (UD-Q4_K_M: ~0.4 GB of Q3_K/IQ3_S -> +0.13 GB; UD-IQ4_XS: ~2.5 GB -> +0.8 GB, which erodes most of its speed edge, so for IQ4_XS keep native kernels for IQ3/IQ2 or accept it).
- **Format B, "u4 + 1-bit plane"** for Q5_K, **Format C, "u4 + 2-bit plane"** for Q6_K (per-16 int8 scales + fp16 d), **Format D int8** for Q8_0.
- Kernel dispatch is per tensor on {A, B, C, D} and group size {16, 32}; 4 GEMV variants cover every file.
- Row-major by output channel, so TP splits along ne1 (q/k/v/gate/up/qkv/z/alpha/beta/output) are free; splits along ne0 for ffn_down (17408/2 = 8704 = 34 x 256), ssm_out and attn_output (6144/2 = 3072 = 12 x 256) land on superblock boundaries. DeltaNet TP split: GPU0 takes k heads 0..7 and, because of the tiled V order, v heads {r*16 + g : g < 8, r < 3} (non-contiguous rows, gathered during repack).

Load pipeline: `mmap` the GGUF (from `/kaggle/working`, 20 GB limit; 16.5 GB fits but check `df -h` for a larger scratch disk), for each tensor `cudaMemcpyAsync` raw blocks (rows for that GPU) into a pinned double-buffered 256 MB staging area, launch the per-type repack kernel into the final allocation, then release staging. GPU repack is far faster than PCIe (~12 GB/s H2D on Kaggle at best), so total load is disk/download bound (~60 s download, ~10 to 20 s read). Optional: write the repacked blob once and attach it as a private Kaggle dataset to skip both download and repack on later runs.

Kernel notes for sm_75: no bf16, no cp.async, mma.sync m16n8k8 fp16 and m8n8k16 int8 are available. For batch-1 decode use fp16 dequant + half2 FMA or dp4a with q8_1-quantized activations; for MTP verify (2 to 4 tokens) and prefill prefer int8 mma (130 TOPS) or fp16 mma (65 TFLOPS) with in-register dequant.

---

## Links
- GGUF spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- Block structs and IQ tables: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
- Reference dequant: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c (`dequantize_row_*`)
- CUDA reference dequant/MMVQ: https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda (`convert.cu`, `vecdotq.cuh`, `mmvq.cu`)
- Qwen3.5 converter transforms: llama.cpp `conversion/qwen.py` (`Qwen3NextModel.modify_tensors`, `_LinearAttentionVReorderBase`)
- Qwen3.5 graph: llama.cpp `src/models/qwen35.cpp`
- Files: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main
