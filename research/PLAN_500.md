# PLAN_500: the road to ~500 tok/s single-request decode (and decent prefill)

Written at M5 round 2 (spec_tc / int4 tensor-core verify GEMV) push. All measured numbers are from
PROGRESS.md (2x T4, Kaggle, Qwen3.8-27B, box variance +/-4-7% between runs).

## 1. Where we are

| mode | tok/s (single request) | note |
|---|---|---|
| plain greedy TP=2 (M2-M4) | ~30 | memory-bound: every step streams all packed weights |
| MTP spec k=3 (M5 r1) | ~73.7 / 69.0 (P0/P1) | verify = M=k+1 dp4a GEMV columns, bit-identical to plain |
| prefill pp2048 (P r4) | ~1030 | gemm8 W4A16 / W4A8 paths |
| batched engine (B) | ~443 aggregate | different regime (multi-request) |

## 2. The roofline is unforgiving: steps are weight streams

Single-request decode is bandwidth-bound. Per verify STEP the engine streams every packed weight byte
exactly once, regardless of how many draft columns ride along (that is the whole point of spec decode):

```
step_floor  = packed_bytes / BW_eff
max_tok_s   = step_rate * tokens_per_step      tokens_per_step <= k + 1 <= MMAX + 1 = 8
```

Q4_0 packs to ~14.2 GB. Effective combined bandwidth on 2x T4 measures ~600 GB/s (the M=1 GEMVs run
at ~310 GB/s per GPU), so the step floor is ~24 ms and plain decode can never exceed ~42 tok/s no
matter what we do to the kernels. Every multiplier must come from tokens_per_step or from fewer bytes:

| weights | packed | step floor | k+1=8 ceiling | k+1=6 ceiling | today (k=3) |
|---|---|---|---|---|---|
| Q4_0      | 14.2 GB | ~24 ms | ~330 tok/s | ~250 | ~73 |
| UD-IQ3_S  | 12.0 GB | ~20 ms | ~400 | ~300 | — |
| UD-Q2_K_XL| 9.83 GB | ~16.5 ms| ~485 | ~365 | — |

(ceilings assume 100% acceptance; realistic is E = sum a^i with per-token acceptance a.)

## 3. Levers, in order of measured leverage

### L1. Make extra verify columns (nearly) free: int4 tensor-core verify GEMV — in flight (M5 r2)
The M5 r1 verify pays dp4a ALU cost that grows ~2.5-4 ms per extra column (gate|up drops to ~185 GB/s
at M=4), so k>=4 LOSES to k=3. The TC verify (kernels/tp_gemv_tc.cuh, `spec_tc=1`) puts the k+1
columns on the mma.m8n8k32 int4 pipes, bit-identical to the dp4a path (-ffp-contract=off; exact
per-32 integer fold + verbatim float epilogue). If the marginal column cost drops near zero, k=5-7
become pure wins and tokens_per_step goes from ~2.4 (k=3) toward ~3.2-3.8 (k=6 at a~0.72):
~110-160 tok/s at Q4_0. This is the round-2 gate.

### L2. More accepted tokens per step: draft quality and shape
- **acceptance a is the exponential lever**: E = (1 - a^(k+1)) / (1 - a). a=0.72, k=6 gives E~3.2;
  a=0.85 gives E~5.1. Options, roughly in cost order:
  - prompt-lookup hybrids (spec_ng): free, huge on copy-heavy text (already wired, `--ngs`).
  - draft-head domain tuning later; the truncated head (spec_dv=1) is already the fast variant.
  - **tree drafts**: MMAX=7 caps the chain at 8; a beam/tree of candidates verified in one M-column
    pass needs M>8 (a 2-m-tile TPD=16 variant of gemv_tc, same weight bytes, ~2x mma work only).
    Expected tokens/step ~4.5-5.5 at a~0.8.
- **two draft rounds per verify** (draft is 1 layer, ~4% of a step): chain depth beyond k+1 without
  raising the verify width — needs the tree form above.

### L3. Fewer bytes: low-bit speed mode (Q3/Q2)
The unsloth repo ships UD-IQ3_S (12.0 GB) and UD-Q2_K_XL (9.83 GB). A Q2_K_XL mode lifts EVERY
bandwidth-bound number by ~1.44x: the k=6 TC spec ceiling goes from ~250 to ~365 tok/s at realistic
acceptance, and the k+1=8 tree ceiling from ~330 to ~485. Requires new FAST_ formats (K2/K3 with the
group-scale layout) + KL-checked quality gate (VERIFY.md protocol) before trusting it. This is the
round-3 candidate after spec_tc lands.

### L4. Cheap steps (already mostly done, keep)
- AR-fused GEMV prologues (M4), one-pass tail AR blocks, graph capture of draft+verify, P2P mailbox,
  replay-only-on-partial rollback (spec_rb). The remaining per-step overheads are the draft forward
  (~1 layer + ring conv + norm) and the q8 requant between GEMVs — both already in graphs.

### L5. Prefill ("decent")
Prefill is compute-bound: the int8/W4A8 GEMMs (gemm8/gemm20 family) are the right tools and pp2048
~1030 tok/s is ~2.2x the llama.cpp box. Paths to ~1.5-2k:
- run prefill GEMMs on BOTH GPUs' tensor cores with better wave quantization (persistent tile loop,
  the gate|up M>1 fix generalizes);
- fp16 accumulate for the W4A8 Q4_1 path where the audit allows (check VERIFY quality bounds);
- overlap the second GPU's GEMM tails with the AR (already partially done in the B engine).

## 4. Honest bottom line

500 tok/s single-request requires BOTH the low-bit mode (~1.44x) and ~7-8 tokens per verify step at
high acceptance (tree drafts, M=16 verify, tuned heads): ~485 * (E/8) hits 500 only at E~8.2 — i.e.
near-perfect acceptance at k=7 — OR the Q2 floor with a tree E~5 and a slightly higher step rate than
modeled. The realistic 2-3 round arc is:

1. r2 (now): spec_tc bit-identical TC verify -> k=5-6 viable -> expect 110-160 tok/s at Q4_0.
2. r3: tree drafts (TPD=16 verify, M=16) + prompt-lookup tuning -> ~200-300 tok/s.
3. r4: UD-Q2_K_XL fast mode with KL gate -> ~300-450 tok/s, re-audit vs llama.cpp.
4. prefill: TC tiling pass -> 1.5-2k tok/s pp2048, folded into any round with spare kernel time.

Each step is gated on bit-identity / byte-identity vs the plain path (V4/V5) and the independent
llama.cpp audit, per VERIFY.md.
