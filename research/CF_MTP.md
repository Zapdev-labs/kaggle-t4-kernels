# CF_MTP: the cf-m4 speculative decoding design (draft / verify / rollback on the CF engine)

The pre-implementation spec for cf-m4 (the third lever, after the UVA + the graphs). The
math is cf-arch.md section 6 (already exact); this note fixes the IMPLEMENTATION structure:
the port sources, the kernel surface, the state, the gates, and the speed math, so the
implementation starts from a verified design instead of discovering it.

## 1. The draft block (blk.48, in-file, ALL Q8_0)

Single draft layer, full-attention type, own hc mixers + own MoE (512 experts, Q8_0,
2.49 GiB) + own final mixer (`blk.48.nextn.hc_head_*`). The trunk hands it the WIDE
residual pre-final-mixer (`t_h_nextn` = res_hc flat [10240]); the draft forward at
position p+1, chained (h from the previous draft step):

```
e      = token_embd[tok_{p+1}]
e_norm = RMSNorm(e, enorm) repeated to the 4 streams      # enorm [2560] (in-file)
h_norm = grouped RMSNorm(h, hnorm) per stream            # hnorm [2560, 4]
u_s    = eh_proj @ concat(e_norm, h_norm[:, s])          # eh_proj [5120,2560] PER STREAM
res'   = [u_0..u_3]                                       # the draft's 4-stream state
a      = FullAttn(blk.48, hc_attn_mix(res'), pos p+1)    # own KV cache, 2 kv heads
res'   = hc_combine(res', a, inject)
m      = MoE(blk.48, hc_ffn_mix(res')) + gated shared    # the Q8_0 experts
res'   = hc_combine(res', m, inject)
h_next = res'   # chains as the next draft step's h
logits = output.weight @ hc_mix(res', nextn.hc_head_*)   # the SHARED lm_head, full vocab
```

NO GDN and NO PLE in the draft (full-attention type; the PLE is layer 1 of the trunk) -
the draft's only rolling state is its own KV cache. Draft cost ~0.44 GiB/token, 77% of it
the shared lm_head (341 MiB, resident with the trunk's non-expert set).

## 2. The Q8_0 gemv path (the critical port item - it already exists, gate-proven)

The whole graft is Q8_0, a format the CF engine's FMT set (K2/K4/Q51/P4) does not cover.
The repo ALREADY owns the complete path, verified by the 27B's spec gates (byte-identical
greedy at every k with the 27B's own Q8 MTP graft):
- the REPACK: `repack.cu`'s `GT_Q8_0` case -> `k_repack_q8` / `repack_q8_block` (the
  34 B/block ggml layout: the f16 d + 32 int8) -> the FAST_Q8 packed planes (the codes
  plane + the d plane).
- the GEMV: `kernels/gemv.cuh` + `tp_gemv_impl.cuh`'s FAST_Q8 template (the int8 x int8
  dp4a, CVT = 1, the d_w x d_a scale fold), the launch form the 27B uses at
  `T4Q_GM(FAST_Q8, 2, 10, ...)` for its own eh_proj.
- the ACTIVATION pairing: the symmetric Q8_0-form activation (amax/127, no s-term) - the
  CF engine's `launch_quantize_q8_0` already produces exactly this form for the FMT_P4
  pairing, so the same quantizer feeds the Q8-weight dot.

The port is IN-REPO reuse (legal under this repo's own-kernels rule; the 27B stack is this
repo), and it is the SAFEST form because its arithmetic is already gate-proven against
the oracle's own Q8_0 dot. The CF-side work: an FMT_Q8 case in the CF `gemv()` dispatch
calling the shared FAST_Q8 gemv launch, plus the loader wiring (the draft tensors ->
the packed PackedWs at load, the ~2.49 GiB resident).

## 3. The draft's kernel surface (existing families, own buffers)

Everything except eh_proj and the FMT_Q8 case is the existing CF kernel set with the
draft's own buffers: the hc mixers (`k_cf_hc_norm/lo/mixed/combine`), the attention
(`k_cf_qk_norm_rope` + `k_cf_kv_store` + `k_cf_attn_decode` + `launch_gate_sigmoid` on the
draft's OWN kc/vc at the draft positions p+1), the MoE shell (the batched gu/up/down
gemvs over the draft's RESIDENT expert slabs - no staging, the resident W-table views),
`k_cf_moe_out`. The NEW op is eh_proj's input: a tiny gather that concatenates
[e_norm ; h_norm_s] into a [5120] buffer per stream (the e half is shared across the 4
streams, the h half is per-stream; one small kernel or 4 strided copies).

## 4. The catch-up (the draft's KV over the prompt)

Before the first speculative step the draft's KV must cover the prompt: the draft forward
over the (token_k, h_{k-1}) pairs, one pass per prompt token (~2-3 ms each, the same
resident-read cost class as a draft step), h from the trunk's per-step wide residual. The
engine keeps the last k+1 `pending_h` rows (the [10240] pre-final-mixer residual at each
step, a ring buffer) - the 27B's `pending_h` pattern. One-time prompt cost ~2.5 ms/token.

## 5. The verify (the batched trunk over k+1 rows) and THE GATE

The verify is the row-batched trunk forward over the k+1 candidate tokens in ONE pass:
the per-row variants of every trunk family - the hc mixers (per-row norms), the attention
(per-row q over the shared KV; the 27B's `k_pf_attn` is the in-repo template), the GDN
(per-row S + conv recurrence; `k_pf_gdn`/`k_pf_gdnc` are the templates), the PLE (per-row
hashes + per-row 16-row gathers), the MoE (per-row routing + the top-10 UNION per layer,
staged/UVA-read once, the dedup factor ~1.5x per cf-arch).
THE GATE: byte-identical greedy at every k - the batched rows MUST reproduce the
sequential decode bit-exactly (the per-row reductions unchanged, the row-parallel grid
only; the 27B's prefill proved the pattern holds). This is the binding correctness
constraint: any batched kernel that reorders a per-row accumulation can flip the
near-tie argmaxes (the r19n battery's own flips sat at gaps 2.68/0.49 under +-2..5 noise)
and break the oracle agreement. The acceptance comparison itself (draft token vs the
verify argmax) then inherits a clean bit-exact base.

## 6. The rollback (the rejection path)

On a rejected draft: the GDN S snapshots (36 layers x [24,128,128] f32 ~ 57 MB, a D2D
copy ~0.2 ms taken once per verify) + the GDN conv states (tiny) + the PLE conv history
(the per-head 9-column ring ~5.9 MB) + the KV (the pos counter rollback - the cells
beyond the pos are invisible, no copy) + the draft's own KV pos and its res' chain
(restart from the last accepted h). All cheap; the 27B's snapshot/ring structure is the
template.

## 7. The VRAM budget and the speed math

The draft's 2.49 GiB experts + the shared lm_head are RESIDENT: the trunk's non-expert
set (~2.6 GiB) + the staging (~12 MB) + the KV/scratch leaves ~12 GiB free on a 15.36 GiB
T4. The draft step is staging-free: ~0.44 GiB of resident reads at the DRAM rate ~
1.7 ms + the lm_head read (resident) - a draft step is ~2-3 ms.
On the UVA'd T4 (~83 ms/token after cf-m3 + the G2 graph, ~12 t/s): k=3 gives the verify
union ~1.44 GB over the UVA path ~112 ms + 3 drafts ~8 ms, at the measured acceptance
(~2.2 tokens/verify expected from the 27B's class) ~ 55 ms/token ~ 18 t/s (a ~1.5x over
the G2 engine; the ceiling ladder in PLAN_CF section 2 stays the reference - the
all-resident stretch ~125-135 t/s).
On the RAM-CAPPED Kaggle host (the partial-UVA form, ~60% of the layers alias-read -
PLAN_CF cf-m3's RAM cap): the verify's union splits the same way (~0.86 GB alias-read
~67 ms + ~0.58 GB staged ~87 ms incl. its memcpys) ~ 155 ms + the drafts ~8 -> at ~2.2
accepted ~ 75 ms/token ~ 13 t/s - still a ~1.4x over that host's non-MTP ~110 ms/token
class, so the lever pays on both hosts. k is tuned on the measured acceptance at the
round.
The lm_head truncation (341 MiB of the 0.44 GiB draft step) stays the quality-gated
stretch (the 27B's `dv` pattern; the CF graft has NO own draft head, so it is an
engine-side vocab subset, gated on the smoke test + greedy agreement).

## 8. The PLE prefetch: the honest window class (the r19e chain verdict holds)

The 1-step-ahead row prefetch is IMPOSSIBLE in the greedy loop (r19e): the chain
gather_i -> the step's GPU work -> the logits -> the argmax -> gather_{i+1} is hard - the
next gather's hash needs THIS step's argmax, which exists only at the step's end, and the
gather is the very next op. The MTP changes the class but only PARTIALLY: the verify's
(k+1) x 16 rows become known at the DRAFTS (the draft tokens), so the prefetch window is
the drafts themselves (~8 ms for k=3) against the ~184 faulted pages (~17 ms at the
measured ~11k IOPS / ~98 us first-fault class) - MOST of the verify's PLE faults can
hide under the draft steps, the tail pays. Realistic win: most of the ~1.4-1.6 ms/token
gather fault cost, not all of it; the MTP's verify is the only window this engine has.

## 9. The order

The design is now frozen ahead of the implementation; the IMPLEMENTATION order stays
(1) the UVA (cf-m3), (2) the graphs, (3) this spec (cf-m4) - the verify's union staging
rides whichever expert path wins the UVA A/B, and the draft steps join the G2 graph as
fixed-shape work (the draft's launches are the same fixed-grid families).
