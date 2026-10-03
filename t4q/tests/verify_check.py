"""Independent audit (stage verify): re-measure t4q's claims with a fresh harness that does not reuse the earlier
check scripts' timing code.

usage: python verify_check.py --section single|batch --model M.gguf --lib libt4q.so --work WORK --oracle ORC --out R.json

single:
  - plain greedy (CUDA graphs) on P0/P1, 512 new tokens, stop on EOS, 2 rounds; tok/s = (n - 1) / wall time of
    t4q_generate (the first token comes out of the prefill, so it is not counted as decode work)
  - MTP k=3 on the same prompts, 2 rounds; output must be byte-identical to plain greedy
  - vs llama.cpp oracle (oracle_dump gen, layer split, 256 greedy tokens): first divergence of t4q plain greedy (with
    llama's top-2 gap there), and teacher-forced top-1 agreement: t4q logits after prompt + llama's tokens, argmax vs
    llama's greedy token at every one of the 256 positions (all, and excluding llama near-ties gap < 0.1)
  - prefill pp512 / pp2048 on a stdlib-source prompt: 1 warm-up + 3 timed reps (median and best); last-position
    logits vs the oracle's (top-1, KL); 32 greedy tokens after the batched prefill vs after a decode-path prefill (pf=0)
batch:
  - 64 DISTINCT prompts (different stdlib files, 400-600 tokens) prefilled into 64 slots
  - single-stream greedy references for slots 0..7 first (35 tokens)
  - B=16 on slots 0..15, B=32 on slots 16..47 (fresh), B=64 on all 64 slots: 2 warm-up + N timed steps each,
    aggregate tok/s = B * N / wall time of the N synchronous t4q_batch_step calls
  - batched greedy tokens of slots 0..7 vs the single-stream references (prefix match)
  - row-distinctness check (generated token rows are not all identical)
  - control: slot 63 cloned into all 64 slots (duplicate sequences), same B=64 timing
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))
from t4q import T4Q, EOS_IDS  # noqa: E402

R = {}
OUT = None


def log(*a):
    print("[verify]", *a, flush=True)


def save():
    with open(OUT, "w") as f:
        json.dump(R, f, indent=1, default=float)


def rd(p):
    return np.fromfile(p, dtype=np.int32)


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def kl(p_logits, q_logits):
    lp, lq = log_softmax(p_logits), log_softmax(q_logits)
    return float(np.sum(np.exp(lp) * (lp - lq)))


def gen_timed(eng, ids, n, stop):
    eng.reset()
    eng.prefill(ids)
    t0 = time.time()
    out = eng.generate(n, stop=stop)
    t1 = time.time()
    return out, t0, t1


def sec_single(eng, a):
    P = {p: rd(os.path.join(a.work, p + ".i32")) for p in ("P0", "P1")}
    eng.set_option("graphs", 1)
    R["single"] = S = {"plain": {}, "mtp": {}}
    plain_ref = {}
    # ---- plain greedy
    for rnd in range(2):
        for p, ids in P.items():
            out, t0, t1 = gen_timed(eng, ids, 512, EOS_IDS)
            if rnd == 0:
                plain_ref[p] = out
                out.tofile(os.path.join(a.work, f"t4q_plain_{p}.i32"))
            same = bool(np.array_equal(out, plain_ref[p]))
            S["plain"][f"{p}_r{rnd}"] = {"n": int(len(out)), "tok_s": round((len(out) - 1) / (t1 - t0), 2),
                                         "secs": round(t1 - t0, 3), "same_as_r0": same, "t0": t0, "t1": t1}
            log("plain", p, rnd, S["plain"][f"{p}_r{rnd}"])
    save()
    # ---- MTP k = 3 (default draft head)
    try:
        for rnd in range(2):
            for p, ids in P.items():
                eng.set_option("spec_k", 3)
                out, t0, t1 = gen_timed(eng, ids, 512, EOS_IDS)
                st = (eng.stats().get("spec") or {}).get("last", {})
                eng.set_option("spec_k", 0)
                ref = plain_ref[p]
                ident = bool(len(out) == len(ref) and np.array_equal(out, ref))
                nmin = min(len(out), len(ref))
                neq = np.nonzero(out[:nmin] != ref[:nmin])[0]
                S["mtp"][f"{p}_r{rnd}"] = {"n": int(len(out)), "tok_s": round((len(out) - 1) / (t1 - t0), 2),
                                           "secs": round(t1 - t0, 3), "identical_to_plain": ident,
                                           "first_div": int(neq[0]) if len(neq) else (-1 if ident else nmin),
                                           "accept_rate": st.get("accept_rate"),
                                           "tokens_per_step": st.get("tokens_per_step"), "t0": t0, "t1": t1}
                log("mtp", p, rnd, S["mtp"][f"{p}_r{rnd}"])
    except Exception:  # noqa: BLE001
        S["mtp_error"] = traceback.format_exc()[-2500:]
        log(S["mtp_error"])
        try:
            eng.set_option("spec_k", 0)
        except Exception:  # noqa: BLE001
            pass
    save()
    # ---- vs llama.cpp oracle
    S["oracle"] = {}
    for p, ids in P.items():
        gp = os.path.join(a.oracle, f"gen_{p}.gen.i32")
        if not os.path.exists(gp):
            S["oracle"][p] = {"error": "no oracle gen"}
            continue
        og = rd(gp)
        gaps = np.array([float(l.split()[3]) for l in open(os.path.join(a.oracle, f"gen_{p}.gen.txt"))])
        mine = plain_ref[p]
        m = min(len(og), len(mine))
        neq = np.nonzero(og[:m] != mine[:m])[0]
        div = int(neq[0]) if len(neq) else m
        # teacher forced: logits after prompt + og[:j] for j = 0..len(og)-1
        tf_ids = np.concatenate([ids, og[:-1]]).astype(np.int32)
        eng.reset()
        lg = eng.logits(tf_ids)[len(ids) - 1:]
        am = lg.argmax(axis=1)
        part = np.partition(lg, -2, axis=1)
        my_gap = part[:, -1] - part[:, -2]
        agree = am == og
        nt = gaps >= 0.1
        S["oracle"][p] = {
            "n_oracle": int(len(og)), "greedy_prefix_match": div, "llama_gap_at_div": float(gaps[div]) if div < m else None,
            "t4q_plain_identical_first_256": bool(div >= m),
            "tf_top1_agree": int(agree.sum()), "tf_positions": int(len(og)),
            "tf_top1_agree_pct": round(100.0 * agree.mean(), 2),
            "tf_top1_agree_pct_excl_neartie": round(100.0 * agree[nt].mean(), 2), "n_neartie": int((~nt).sum()),
            "tf_disagree_positions": [{"j": int(j), "llama_gap": round(float(gaps[j]), 4),
                                       "t4q_gap": round(float(my_gap[j]), 4)} for j in np.nonzero(~agree)[0][:20]],
        }
        del lg
        log("oracle", p, S["oracle"][p])
        save()
    # ---- prefill
    L = rd(os.path.join(a.work, "L.i32"))
    S["prefill"] = {}
    for n in (512, 2048):
        ids = L[:n]
        try:
            eng.set_option("pf", 1)
            eng.reset()
            eng.prefill(ids)  # warm-up
            ts = []
            for _ in range(3):
                eng.reset()
                w0 = time.time()
                eng.prefill(ids)
                ts.append(time.time() - w0)
            last = eng.last_logits()
            r = {"n": n, "tok_s_median": round(n / float(np.median(ts)), 1), "tok_s_best": round(n / min(ts), 1),
                 "tok_s_all": [round(n / x, 1) for x in ts], "t0": w0 - sum(ts[:-1]), "t1": w0 + ts[-1]}
            op = os.path.join(a.oracle, f"L{n}.last.f32")
            if os.path.exists(op):
                ol = np.fromfile(op, dtype=np.float32)
                r["oracle_top1_equal"] = bool(int(ol.argmax()) == int(last.argmax()))
                r["kl_llama_t4q"] = kl(ol, last)
                srt = np.sort(ol)
                r["oracle_top2_gap"] = float(srt[-1] - srt[-2])
            if n == 512:
                # batched prefill path vs decode-path prefill (pf=0): next 32 greedy tokens
                eng.reset()
                eng.prefill(ids)
                g1 = eng.generate(32)
                eng.set_option("pf", 0)
                eng.reset()
                eng.prefill(ids)
                last0 = eng.last_logits()
                g0 = eng.generate(32)
                eng.set_option("pf", 1)
                mm = min(len(g0), len(g1))
                ne = np.nonzero(g0[:mm] != g1[:mm])[0]
                r["greedy32_pf_vs_decode_prefix"] = int(ne[0]) if len(ne) else mm
                r["kl_decodepath_vs_prefill_last"] = kl(last0, last)
            S["prefill"][f"pp{n}"] = r
            log("prefill", n, r)
        except Exception:  # noqa: BLE001
            S["prefill"][f"pp{n}_error"] = traceback.format_exc()[-2500:]
            log(S["prefill"][f"pp{n}_error"])
            eng.set_option("pf", 1)
        save()


def sec_batch(eng, a):
    prompts = [np.asarray(x, np.int32) for x in json.load(open(os.path.join(a.work, "batch_prompts.json")))]
    R["batch"] = Bt = {"n_prompts": len(prompts), "lens": [int(len(p)) for p in prompts],
                       "n_distinct_prompts": len({p.tobytes() for p in prompts})}
    eng.set_option("graphs", 1)
    # single-stream references (35 tokens) for slots 0..7
    refs = {}
    for i in range(8):
        eng.reset()
        eng.prefill(prompts[i])
        refs[i] = eng.generate(35)
    save()
    eng.batch_init(64, 1088, 1)
    t = time.time()
    first = [eng.batch_prefill(i, prompts[i]) for i in range(64)]
    Bt["prefill_64_s"] = round(time.time() - t, 1)
    toks = {i: [first[i]] for i in range(64)}
    steps = a.steps
    runs = {}

    def run(name, slots):
        for _ in range(2):  # warm-up
            o = eng.batch_step(slots)
            for s, x in zip(slots, o):
                toks[s].append(int(x))
        rows = []
        t0 = time.time()
        for _ in range(steps):
            o = eng.batch_step(slots)
            rows.append(o.copy())
        t1 = time.time()
        for o in rows:
            for s, x in zip(slots, o):
                toks[s].append(int(x))
        rows = np.stack(rows)  # steps x B
        distinct_cols = len({rows[:, j].tobytes() for j in range(rows.shape[1])})
        B = len(slots)
        runs[name] = {"B": B, "steps": steps, "ms_per_step": round(1e3 * (t1 - t0) / steps, 2),
                      "agg_tok_s": round(B * steps / (t1 - t0), 1), "distinct_token_streams": distinct_cols,
                      "pos_range": [int(min(eng.batch_pos(s) for s in slots)), int(max(eng.batch_pos(s) for s in slots))],
                      "t0": t0, "t1": t1}
        log("batch", name, runs[name])
        Bt["runs"] = runs
        save()

    run("B16", list(range(16)))
    run("B32", list(range(16, 48)))
    run("B64", list(range(64)))
    # correctness: slots 0..7 batched greedy (first + 2 warm-up + steps of the B16 run) vs single-stream references
    cmp = {}
    for i in range(8):
        b = np.asarray(toks[i][:35], np.int32)
        r = refs[i]
        m = min(len(b), len(r))
        ne = np.nonzero(b[:m] != r[:m])[0]
        cmp[i] = int(ne[0]) if len(ne) else m
    Bt["batched_vs_single_prefix_match_of_35"] = cmp
    # control: duplicate sequences (slot 63 cloned into all)
    for s in range(63):
        eng.batch_clone(63, s)
    run("B64_clones", list(range(64)))
    eng.batch_free()
    save()


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=32)
    a = ap.parse_args()
    OUT = a.out
    t = time.time()
    eng = T4Q(a.model, lib=a.lib, max_ctx=4096, verbose=1, tp=1)
    st = eng.stats()
    R["load"] = {"secs": round(time.time() - t, 1), "p2p": st.get("p2p"), "vram": st.get("vram_used_mib"),
                 "selftest_worst": (st.get("selftest") or {}).get("worst"), "mtp": st.get("mtp")}
    log("load", R["load"])
    save()
    try:
        {"single": sec_single, "batch": sec_batch}[a.section](eng, a)
    except Exception:  # noqa: BLE001
        R["error"] = traceback.format_exc()[-3000:]
        log(R["error"])
    save()
    eng.close()


if __name__ == "__main__":
    main()
