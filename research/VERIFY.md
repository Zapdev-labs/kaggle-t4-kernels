# t4q independent verification (2026-10-03)

One fresh Kaggle run, `otdoges/t4q-verify` **v1** (2x T4, P2P box, both GPUs PHB). t4q was built from the tree at commit
`407f0ee` (the last engine commit) plus my audit scripts (`a300824`: `t4q/tests/verify_check.py`, `t4q/tools/stage_verify.py`).
llama.cpp is master `a4cb4c61` (the sm_75 binaries from kernel output `otdoges/t4-qwen38-baseline`), run in the **same
session** after t4q. Both use the same Q4_0 GGUF (`unsloth/Qwen3.8-27B-GGUF`, 16.06 GB) and the same token ids
(HF chat template, `enable_thinking=False`; llama-server got raw token arrays through `/completion`). Raw output is in
`kaggle/verify/out/` (`results.json`, `verify_single.json`, `verify_batch.json`, `logs/`, including a 1 Hz clock log).

## Final table

| metric | t4q (measured) | llama.cpp same session | t4q / llama | claimed by builders | verdict |
|---|---|---|---|---|---|
| single-stream greedy decode, P0 / P1, 512 tok, short ctx | **30.42 / 30.25**, rerun 30.06 / 30.21 | `-sm tensor` server 21.48 / 21.17 (rerun 21.41 / 21.41); llama-bench tg128 20.94 | 1.41-1.44x | 29.24 (r1), 30.17 (r2) | confirmed (just above 30) |
| MTP k=3 greedy, P0 / P1, 512 tok | **71.85 / 68.08**, rerun 72.54 / 68.00 | `-sm tensor` + draft-mtp n_max 3 (same Q4_0 blk.64): 37.87 / 33.19, rerun 37.31 / 34.56 | 1.9-2.05x | 73.69 / 68.95 | confirmed (within ~2%) |
| MTP acceptance per draft, P0 / P1 | 0.807 / 0.731 (3.42 / 3.19 tokens per verify) | 0.843-0.852 / 0.733 | | 0.807 / 0.731 | same |
| prefill pp512 (median of 3) | **904.7** (best 908.5) | llama-bench 532.63 | 1.70x | 855.4 (r2 exact path; 955.9 only in the lossy R512 mode) | confirmed |
| prefill pp2048 (median of 3) | **1033.2** (best 1033.9) | llama-bench 512.27 | 2.02x | 1032.6 | confirmed |
| batched aggregate decode, B=16, 64 distinct prompts | **181.2** (88.3 ms/step) | llama-batched-bench S_TG 105.26 | 1.72x | 172.6-173.6 (clones, 1k ctx) | confirmed |
| batched aggregate decode, B=32 | **314.7** (101.7 ms/step) | 129.22 | 2.44x | 294.9-297.7 | confirmed |
| batched aggregate decode, B=64 | **443.0** (144.5 ms/step) | 146.03 | 3.03x | 429.8 (best), 402.9 (final defaults) | confirmed at ~0.4-0.65k ctx, see caveats |
| B=64, one prompt cloned into all slots (control) | 456.5 (140.2 ms/step) | | | | duplicates are 3% faster, not a large effect |

t4q context in the batched runs: prompts 379-577 tokens, positions 413-644 during timing. llama-batched-bench:
npp 512, ntg 64, one process per B, `-sm tensor -fa on -ub 512`. SM clocks during the t4q timings were 930-1100 MHz;
during llama's runs 950-1050 MHz (mean), so no run got a clock advantage that explains the gaps.

## Correctness evidence

- **MTP vs plain:** all 4 MTP runs are byte-identical to t4q plain greedy (512 tokens each, first divergence none).
- **t4q plain greedy vs llama.cpp oracle** (`oracle_dump gen`, layer split, 256 tokens): P1 identical 256/256. P0
  identical for 74 tokens, then diverges at a near-tie (llama's top-2 gap 0.042 logit; t4q's gap there 0.069). This is
  the same divergence M1 reported.
- **llama.cpp's own spread for scale:** llama `-sm tensor` vs llama `-sm layer` greedy agree for only 74 (P0) and 17
  (P1) tokens. t4q vs llama tensor: 142 (P0) and 17 (P1). t4q agrees with llama's layer split better than llama's two
  split modes agree with each other.
- **Teacher-forced top-1 agreement** (t4q logits after prompt + llama's 256 greedy tokens, argmax vs llama's token):
  P0 255/256 (99.61%; the one miss is the 0.042 near-tie), P1 256/256. Excluding llama near-ties (gap < 0.1): 100% on both.
- **Prefill path:** last-position top-1 equals the oracle's at pp512 and pp2048. KL(llama || t4q) at the last position:
  pp2048 3.8e-8 (peaked distribution, gap 13.6), **pp512 4.3e-2** (gap 1.30). The 32 greedy tokens after the
  batched prefill equal those after a decode-path prefill (32/32), but the decode-path vs prefill-path KL at that
  same position is also 5.4e-2. See caveat 3.
- **Batched rows vs single-stream:** slots 0..7 (distinct prompts), 35 greedy tokens: 7/8 identical, 1 diverges at
  token 33. All 64 token streams in the B=64 run are distinct (the control clone run shows 1 distinct stream, as expected).
- Load-time self-test of the fast weight formats against a CPU fp64 reference: worst rel 4.2e-7.

## Code audit (what I looked for)

- **Timing scope:** `tp_generate` and `tp_spec_generate` time the whole generate call and end with a device sync of both
  GPUs. The spec loop's overshoot past `max_new` (up to `spec_ahead` iterations) is inside the timed region, which
  makes the MTP number slightly pessimistic, not optimistic. Batched `t4q_batch_step` syncs both GPUs before it returns
  tokens, so per-step wall time covers the work. Prefill ends with a sync of both streams.
- **Token counts:** the builders' scripts and mine use `(n - 1) / wall`, because the first token comes from the
  prefill. Prompt tokens are never counted as generated.
- **Skipped layers or cached outputs:** none found. Every bench resets state and re-prefills. The 99.6-100%
  teacher-forced top-1 agreement over 512 positions would not survive skipped layers.
- **Oracle:** `oracle_dump` links the a4cb4c61 libllama and evaluates the same GGUF on the same token ids. I also added
  an independent llama reference (llama-server `-sm tensor`) for the greedy texts.
- **Batched duplicates (real methodology issue, small effect):** the builders' `batch_check.py` bench prefilled one
  prompt and **cloned it into all B slots**. The engine does not share work across slots (separate KV and state, the
  GEMM computes every column), and in my control clones were only 3% faster than distinct prompts (456.5 vs 443.0).
  The claimed numbers are not inflated by more than that, but they were measured on duplicate sequences.
- **Best-of-N:** the prefill bench reports the best of 3 reps. The spread was under 1% here, so it does not matter.

## Caveats (honest list)

1. **Single box, single run.** Earlier rounds saw ±4% between boxes (clocks 555-1270 MHz, P2P present or not). This
   box had P2P. Without P2P the builders measured about 1 tok/s less, so the plain-decode ">= 30" gate holds only
   marginally on P2P boxes.
2. **Short context only.** Single-stream numbers are at < 600 positions and batched numbers at 413-644. I did not
   re-measure decode at 4k depth, where the builders report 29.3 tok/s (the 4k depth gate was never met). Batched at
   1k/4k contexts was not re-measured (claims: 402.9-429.8 at 1k, 235.4 at 4k B=32).
3. **pp512 last-position KL 4.3e-2 vs llama** is far above the design's 2e-3 target, on my stdlib-code prompt cut at
   512 tokens. t4q's own decode path disagrees with its prefill path by a similar 5.4e-2 at that position, and top-1
   and the next 32 greedy tokens still match, so it looks like a sensitive (flat) position, not a broken kernel. But I
   did not measure llama's own batch-vs-token floor on this prompt, so this stays **unresolved**. The batched prefill
   and batched decode both use lossy int8 per-row weight requantization (gemm9). They are not bit-equivalent to the
   decode path. That is a documented design choice, not hidden.
4. **Prompt mix favours MTP.** P0/P1 are code-generation prompts with 73-81% draft acceptance. MTP speed on chatty or
   high-entropy text will be lower. The numbers compare t4q and llama on the same prompts with the same acceptance
   (llama's acceptance is similar or higher), so the ratio is fair, but the absolute 68-72 tok/s is prompt-dependent.
5. **llama.cpp settings.** I compared against llama.cpp's best measured mode for this box (`-sm tensor`, FA on, f16 KV,
   ub 512, Q4_0). I did not tune llama.cpp further (for example other `-ub`, `-sm layer` with MTP, or graph options).
   llama-batched-bench processes each sequence's prompt separately, with identical prompt tokens across sequences.
   Its decode work is per sequence, so that does not favour llama.
6. **Batched t4q is slow at tiny B** (12.8 tok/s at B=1 per the builders; not re-measured). The single-stream engine is
   the right path for B=1.
7. **Greedy only.** Sampling is not implemented in the spec or batched paths, so every number here is greedy.
8. Build took 274 s on this box (vs 100-120 s before), which matters only for session budgeting.

## Verdict

The four headline claims hold on a fresh build with an independent harness, in the same session as llama.cpp:
- single-stream about **30.2-30.4 tok/s** (1.42x llama.cpp tensor split);
- MTP **68-72.5 tok/s**, byte-identical to plain greedy (about 2x llama.cpp + MTP);
- prefill **905 / 1033 tok/s** at pp512 / pp2048 (1.7x / 2.0x);
- batched aggregate **181 / 315 / 443 tok/s** at B = 16 / 32 / 64 with distinct prompts (1.7x / 2.4x / 3.0x
  llama-batched-bench).

Greedy output tracks the llama.cpp oracle as closely as llama.cpp's own split modes track each other. The open items
are the 4k-depth numbers I did not re-measure and the unresolved pp512 last-position KL.
