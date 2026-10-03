"""Milestone P check + bench: batched TP prefill (option pf=1) vs the decode path (pf=0) and the llama.cpp oracle,
then pp512 / pp2048 timing.

usage: python prefill_check.py --model M.gguf --work WORK --oracle ORC --out results_pf.json --lib libt4q.so
         [--configs 'pf_i4=0;pf_i4=1'] [--bench_n 512,2048] [--reps 2] [--prompts P0,P1,W,L]

Correctness per prompt (ids from WORK/<name>.i32):
  last-token logits after prefill: pf=1 vs pf=0 (KL, top-1), both vs the oracle's batch path (<name>.last.f32), and the
  llama floor (oracle batch vs token-by-token at the last position, from the seq job, where available)
  continuation: 16 teacher-forced positions (the oracle's greedy tokens) after each prefill, KL pf=1 vs pf=0
  greedy: 32 tokens after each prefill, pf=1 vs pf=0 and vs the oracle (first divergence + oracle top-2 gap there)
Bench: wall time of t4q_prefill on the first n ids of prompt L (warm-up run first), per config.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "py"))
from t4q import T4Q  # noqa: E402

V = 248320
R = {}


def log(*a):
    print("[prefill_check]", *a, flush=True)


def log_softmax(x):
    x = np.asarray(x, np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def kl(p, q):
    lp, lq = log_softmax(p), log_softmax(q)
    return float((np.exp(lp) * (lp - lq)).sum(-1).mean())


def top2gap(x):
    s = np.partition(np.asarray(x, np.float64), -2)[-2:]
    return float(abs(s[1] - s[0]))


def read_ids(p):
    return np.fromfile(p, dtype=np.int32)


def apply(eng, opts):
    for kv in [x for x in opts.split(",") if x]:
        k, v = kv.split("=")
        eng.set_option(k.strip(), int(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--configs", default="pf_i4=0;pf_i4=1")
    ap.add_argument("--bench_n", default="512,2048")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--prompts", default="P0,P1,W,L")
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--sections", default="correct,bench")
    ap.add_argument("--keep_h", type=int, default=0)
    ap.add_argument("--bench_skip", default="", help="comma list of config indices not benched")
    a = ap.parse_args()
    secs = set(a.sections.split(","))
    cfgs = [c for c in a.configs.split(";")]

    def save():
        with open(a.out, "w") as f:
            json.dump(R, f, indent=1, default=float)

    t = time.time()
    eng = T4Q(a.model, lib=a.lib, max_ctx=4096, verbose=1, tp=1)
    R["load_s"] = round(time.time() - t, 1)
    st = eng.stats()
    R["p2p"] = st.get("p2p")
    R["selftest_worst"] = (st.get("selftest") or {}).get("worst")
    log("load", R["load_s"], "s, p2p", R["p2p"])
    eng.set_option("graphs", 1)
    if a.keep_h:
        eng.set_option("pf_keep_h", 1)
    save()

    def run(ids, pf, opts, cont):
        apply(eng, opts)
        eng.set_option("pf", pf)
        eng.reset()
        t0 = time.time()
        eng.prefill(ids)
        dt = time.time() - t0
        fqs = eng.stats().get("pf_fq") if pf else None
        hid = eng.dump("pf_h") if (pf and a.keep_h) else None
        last = eng.last_logits().copy()
        gen = eng.generate(a.gen)
        cl = np.zeros((0, V), np.float32)
        if len(cont):
            eng.reset()
            eng.prefill(ids)
            cl = eng.logits(cont)
        return {"last": last, "gen": gen, "cont": cl, "secs": dt, "fq": fqs, "h": hid}

    if "correct" in secs:
        C = R.setdefault("correct", {})
        ok_all = True
        for name in a.prompts.split(","):
            try:
                p = os.path.join(a.work, name + ".i32")
                if not os.path.exists(p):
                    continue
                ids = read_ids(p)
                og = os.path.join(a.oracle, f"gen_{name}.gen.i32")
                ogen = read_ids(og) if os.path.exists(og) else np.zeros(0, np.int32)
                ogap = np.loadtxt(os.path.join(a.oracle, f"gen_{name}.gen.txt"))[:, 3] if os.path.exists(og) else None
                cont = ogen[:16] if len(ogen) >= 16 and len(ids) <= 4096 else np.zeros(0, np.int32)
                A = run(ids, 0, cfgs[0], cont)
                res = {"n": int(len(ids)), "decode_path_prefill_s": round(A["secs"], 2)}
                olast_p = os.path.join(a.oracle, name + ".last.f32")
                olast = np.fromfile(olast_p, dtype=np.float32) if os.path.exists(olast_p) else None
                tbt_p = os.path.join(a.oracle, name + ".tbt.f32")
                if olast is not None and os.path.exists(tbt_p):
                    tb = np.fromfile(tbt_p, dtype=np.float32).reshape(-1, V)[-1]
                    res["llama_floor_kl_batch_vs_tbt"] = kl(olast, tb)
                if olast is not None:
                    res["kl_oracle_vs_pf0"] = kl(olast, A["last"])
                    res["oracle_top2_gap"] = top2gap(olast)
                href = None
                for ci, cfg in enumerate(cfgs):
                    B = run(ids, 1, cfg, cont)
                    r = {"kl_pf0_vs_pf1": kl(A["last"], B["last"]),
                         "top1_equal": bool(int(np.argmax(A["last"])) == int(np.argmax(B["last"]))),
                         "pf0_top2_gap": top2gap(A["last"]), "prefill_s": round(B["secs"], 3)}
                    if B.get("fq"):
                        r["fq"] = B["fq"]
                    if B.get("h") is not None:
                        hb = B["h"].reshape(-1, 5120).astype(np.float64)
                        if ci == 0:
                            href = hb
                        elif href is not None and href.shape == hb.shape:
                            e = np.linalg.norm(hb - href, axis=1) / np.maximum(np.linalg.norm(href, axis=1), 1e-30)
                            r["h_rel"] = {"mean": float(e.mean()), "med": float(np.median(e)),
                                          "p99": float(np.quantile(e, 0.99)), "max": float(e.max()),
                                          "argmax": int(e.argmax())}
                    if olast is not None:
                        r["kl_oracle_vs_pf1"] = kl(olast, B["last"])
                        r["top1_equal_oracle"] = bool(int(np.argmax(olast)) == int(np.argmax(B["last"])))
                    if len(cont):
                        kls = [kl(A["cont"][i], B["cont"][i]) for i in range(len(cont))]
                        r["cont_kl_mean"] = float(np.mean(kls))
                        r["cont_kl_max"] = float(np.max(kls))
                        r["cont_top1_agree"] = float(np.mean([np.argmax(A["cont"][i]) == np.argmax(B["cont"][i])
                                                              for i in range(len(cont))]))
                    m = min(len(A["gen"]), len(B["gen"]))
                    d = np.nonzero(A["gen"][:m] != B["gen"][:m])[0]
                    r["gen_first_div_vs_pf0"] = int(d[0]) if len(d) else -1
                    if len(ogen):
                        m2 = min(len(ogen), len(B["gen"]))
                        d2 = np.nonzero(ogen[:m2] != B["gen"][:m2])[0]
                        r["gen_first_div_vs_oracle"] = int(d2[0]) if len(d2) else -1
                        if len(d2) and ogap is not None:
                            r["oracle_gap_at_div"] = float(ogap[d2[0]])
                        d3 = np.nonzero(ogen[:m2] != A["gen"][:m2])[0]
                        r["pf0_gen_first_div_vs_oracle"] = int(d3[0]) if len(d3) else -1
                    # pass: same greedy token after the prompt (or a near-tie), KL to pf0 within 2x the llama floor
                    # (or 2e-3 where no floor), greedy text identical or diverging only at a near-tie
                    floor = res.get("llama_floor_kl_batch_vs_tbt")
                    lim = max(2e-3, 2 * floor) if floor is not None else 5e-3
                    gen_ok = r["gen_first_div_vs_pf0"] < 0 or (len(ogen) and (r.get("gen_first_div_vs_oracle", -1) < 0 or r.get("oracle_gap_at_div", 1) < 0.05))
                    r["pass"] = bool((r["top1_equal"] or r["pf0_top2_gap"] < 0.05) and r["kl_pf0_vs_pf1"] <= lim
                                     and gen_ok)
                    if ci == 0:
                        ok_all &= r["pass"]
                    res[cfg or "default"] = r
                    log(name, cfg, json.dumps(r))
                C[name] = res
            except Exception:  # noqa: BLE001
                C[name + "_error"] = traceback.format_exc()[-3000:]
                log(C[name + "_error"])
                ok_all = False
            save()
        R["correct_pass"] = bool(ok_all)

    if "bench" in secs:
        Bn = R.setdefault("bench", {})
        L = read_ids(os.path.join(a.work, "L.i32"))
        for n in [int(x) for x in a.bench_n.split(",")]:
            ids = L[:n]
            skip = {int(x) for x in a.bench_skip.split(",") if x}
            for ci, cfg in enumerate(cfgs):
                if ci in skip:
                    continue
                try:
                    apply(eng, cfg)
                    eng.set_option("pf", 1)
                    eng.reset()
                    eng.prefill(ids)  # warm-up (buffer allocation, first launches)
                    ts = []
                    for _ in range(a.reps):
                        eng.reset()
                        w0 = time.time()
                        eng.prefill(ids)
                        ts.append(time.time() - w0)
                    stt = eng.stats()
                    best = min(ts)
                    eng.set_option("pf_prof", 1)
                    eng.reset()
                    eng.prefill(ids)
                    prof = eng.stats().get("pf_profile")
                    eng.set_option("pf_prof", 0)
                    Bn[f"pp{n}_{cfg}"] = {"tok_s": round(n / best, 1), "tok_s_all": [round(n / x, 1) for x in ts],
                                          "batch_s": stt.get("pf_last_batch_s"), "total_s": stt.get("pf_last_total_s"),
                                          "t0": w0, "t1": w0 + ts[-1], "profile_gpu0_ms": prof}
                    log(f"pp{n}", cfg, json.dumps(Bn[f"pp{n}_{cfg}"]))
                except Exception:  # noqa: BLE001
                    Bn[f"pp{n}_{cfg}_error"] = traceback.format_exc()[-2000:]
                    log(Bn[f"pp{n}_{cfg}_error"])
                save()
        best = {}
        for k, v in Bn.items():
            if isinstance(v, dict) and "tok_s" in v:
                nn = k.split("_")[0]
                best[nn] = max(best.get(nn, 0), v["tok_s"])
        R["best_pp"] = best
        R["pf_gdnc_check"] = eng.stats().get("pf_gdnc_check")
        R["gate"] = bool(best.get("pp2048", 0) >= 1400 and best.get("pp512", 0) >= 1200 and R.get("correct_pass", False))
        log("best", best, "gate", R["gate"])
    save()
    eng.close()


if __name__ == "__main__":
    main()
