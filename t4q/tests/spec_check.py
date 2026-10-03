"""Speculative decoding check + bench (stage m5).

usage: python spec_check.py --model M.gguf --work WORK --out results_spec.json --lib libt4q.so
         [--gen 512] [--ks 2,3,4,5,6] [--sections ref,v4,v5,prof]

  ref : plain greedy decode (CUDA graphs) of --gen tokens after each coding prompt: the reference token stream and the
        plain tok/s of this box (validated against llama.cpp in M1-M4)
  v4  : verify columns vs single-token decode: drafts forced to the reference continuation (100% acceptance), the
        accepted columns' logits must be bit-identical to teacher-forced plain decode logits
  v5  : MTP speculative greedy decode for every k (and draft head variant): output must be byte-identical to ref;
        tok/s, acceptance per prompt, accepted-length histogram, draft-vocab in-subset rate
  prof: per-graph GPU times (draft / verify) with one iteration in flight
Every timing is wall time of t4q_generate (prefill excluded), tokens = generated tokens.
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


def log(*a):
    print("[spec_check]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--ks", default="2,3,4,5,6")
    ap.add_argument("--dvs", default="1", help="draft head variants for the k sweep (1 = truncated, 0 = full)")
    ap.add_argument("--dv0_ks", default="3", help="k values also run with the full draft head")
    ap.add_argument("--prompts", default="P0,P1")
    ap.add_argument("--sections", default="ref,v4,v5,prof")
    ap.add_argument("--stop", type=int, default=1, help="stop on EOS")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--trace_ks", default="3")
    ap.add_argument("--ngs", default="0", help="prompt-lookup thresholds to run for each k (0 = MTP only)")
    ap.add_argument("--gate_prompts", default="P0,P1")
    ap.add_argument("--extra", default="", help="';'-separated extra option sets, e.g. spec_k=3,spec_rb=0")
    a = ap.parse_args()
    secs = set(a.sections.split(","))

    def save():
        with open(a.out, "w") as f:
            json.dump(R, f, indent=1, default=float)

    ids = {p: np.fromfile(os.path.join(a.work, p + ".i32"), dtype=np.int32) for p in a.prompts.split(",")}
    t = time.time()
    eng = T4Q(a.model, lib=a.lib, max_ctx=4096, verbose=1, tp=1)
    R["load_s"] = round(time.time() - t, 1)
    st = eng.stats()
    R["selftest_worst"] = (st.get("selftest") or {}).get("worst")
    R["selftest_mtp"] = [x for x in (st.get("selftest") or {}).get("tests", []) if "blk.64" in x.get("w", "") or
                         "draft" in x.get("w", "")]
    R["vram_used_mib"] = st.get("vram_used_mib")
    R["p2p"] = st.get("p2p")
    R["mtp"] = st.get("mtp")
    log("load", R["load_s"], "s; selftest worst", R["selftest_worst"], "vram", R["vram_used_mib"], "p2p", R["p2p"],
        "mtp", R["mtp"])
    save()
    eng.set_option("graphs", 1)
    stop = EOS_IDS if a.stop else ()

    def run(p, n, opts):
        for k, v in opts.items():
            eng.set_option(k, v)
        eng.reset()
        eng.prefill(ids[p])
        t0 = time.time()
        out = eng.generate(n, stop=stop)
        t1 = time.time()
        return out, t0, t1

    # ---------------------------------------------------------------- reference: plain greedy decode
    ref = {}
    if "ref" in secs or "v5" in secs or "v4" in secs:
        R["ref"] = {}
        for p in ids:
            out, t0, t1 = run(p, a.gen, {"spec_k": 0})
            ref[p] = out
            R["ref"][p] = {"n": int(len(out)), "tok_s": round((len(out) - 1) / (t1 - t0), 2), "t0": t0, "t1": t1,
                           "head": out[:16].tolist()}
            log("ref", p, R["ref"][p]["n"], "tokens", R["ref"][p]["tok_s"], "tok/s")
        save()

    # ---------------------------------------------------------------- V4: forced drafts, verify logits bit-identity
    if "v4" in secs:
        try:
            res = {}
            for k in (3, 6):
                p = list(ids)[0]
                n = 33
                cont = ref[p]
                eng.reset()
                eng.prefill(ids[p])
                eng.set_option("spec_k", k)
                eng.set_option("spec_force", 1)
                eng.set_option("spec_dbg", 1)
                eng.spec_force(np.concatenate([cont, np.zeros(64, np.int32)]))
                out = eng.generate(n, stop=())
                sl = eng.dump("spec_logits")
                sn = eng.dump("spec_n")
                eng.set_option("spec_force", 0)
                eng.set_option("spec_dbg", 0)
                eng.set_option("spec_k", 0)
                ncol = int(sum(sn)) if sn is not None else 0
                sl = sl.reshape(-1, eng.n_vocab) if sl is not None else np.zeros((0, eng.n_vocab), np.float32)
                # plain teacher-forced logits: row j = logits after feeding cont[j] (position n_prompt + j)
                eng.reset()
                eng.prefill(ids[p])
                pl = eng.logits(cont[:ncol])
                same = [bool(np.array_equal(sl[j].view(np.uint32), pl[j].view(np.uint32))) for j in range(ncol)]
                maxd = [float(np.max(np.abs(sl[j] - pl[j]))) for j in range(ncol)]
                res[f"k{k}"] = {"out_identical": bool(np.array_equal(out, cont[:len(out)])), "n_out": int(len(out)),
                                "accepted_per_step": [int(x) for x in sn] if sn is not None else [],
                                "columns": ncol, "bitident_cols": int(sum(same)), "first_bad": (same.index(False)
                                                                                                if False in same else -1),
                                "max_abs_diff": max(maxd) if maxd else None,
                                "argmax_equal": int(sum(int(np.argmax(sl[j]) == np.argmax(pl[j])) for j in range(ncol)))}
                log("V4", k, res[f"k{k}"])
            R["V4"] = res
            R["V4_pass"] = all(v["columns"] > 0 and v["bitident_cols"] == v["columns"] and v["out_identical"]
                               for v in res.values())
        except Exception:  # noqa: BLE001
            R["V4_error"] = traceback.format_exc()[-3000:]
            log(R["V4_error"])
            for o in ("spec_force", "spec_dbg", "spec_k"):
                try:
                    eng.set_option(o, 0)
                except Exception:  # noqa: BLE001
                    pass
        save()

    # ---------------------------------------------------------------- V5 + speed: MTP spec decode per k
    if "v5" in secs:
        R["spec"] = {}
        ks = [int(x) for x in a.ks.split(",") if x]
        cfgs = []
        ngs = [int(x) for x in a.ngs.split(",") if x]
        base = {"spec_rb": 1, "spec_sqt": 256}
        for dvh in [int(x) for x in a.dvs.split(",")]:
            for k in ks:
                for ng in ngs:
                    cfgs.append((f"k{k}_dv{dvh}" + (f"_ng{ng}" if ng else ""),
                                 dict(base, spec_k=k, spec_dv=dvh, spec_ng=ng)))
        for k in [int(x) for x in a.dv0_ks.split(",") if x]:
            cfgs.append((f"k{k}_dv0", dict(base, spec_k=k, spec_dv=0, spec_ng=0)))
        for ex in [x for x in a.extra.split(";") if x]:  # e.g. "spec_k=3,spec_rb=0"
            o = dict(base, spec_dv=1, spec_ng=0)
            for kv in ex.split(","):
                kk, vv = kv.split("=")
                o[kk] = int(vv)
            cfgs.append(("x_" + ex.replace("spec_", "").replace(",", "_").replace("=", ""), o))
        for rnd in range(a.rounds):
            for cname, opts in cfgs:
                name = cname + (f"_r{rnd}" if rnd else "")
                res = {}
                try:
                    for p in ids:
                        out, t0, t1 = run(p, a.gen, opts)
                        s = eng.stats().get("spec", {}).get("last", {})
                        rp = ref[p]
                        nmin = min(len(out), len(rp))
                        ident = bool(len(out) == len(rp) and np.array_equal(out, rp))
                        div = -1
                        if not ident:
                            neq = np.nonzero(out[:nmin] != rp[:nmin])[0]
                            div = int(neq[0]) if len(neq) else nmin
                        res[p] = {"n": int(len(out)), "tok_s": round((len(out) - 1) / (t1 - t0), 2), "identical": ident,
                                  "first_divergence": div, "accept_rate": s.get("accept_rate"),
                                  "tokens_per_step": s.get("tokens_per_step"), "steps": s.get("steps"),
                                  "hist_ng": s.get("hist_ng"),
                                  "hist": eng.stats().get("spec", {}).get("last_hist"),
                                  "dv_in_subset": eng.stats().get("spec", {}).get("dv_in_subset"), "t0": t0, "t1": t1}
                        log("spec", name, p, res[p])
                    eng.set_option("spec_k", 0)
                    eng.set_option("spec_ng", 0)
                except Exception:  # noqa: BLE001
                    res["error"] = traceback.format_exc()[-3000:]
                    log(res["error"])
                    try:
                        eng.set_option("spec_k", 0)
                    except Exception:  # noqa: BLE001
                        pass
                R["spec"][name] = res
                save()
        ok = [v for v in R["spec"].values() if "error" not in v]
        R["V5_pass"] = bool(ok) and all(v[p]["identical"] for v in ok for p in ids)
        best = None
        gp = [p for p in a.gate_prompts.split(",") if p in ids]
        for name, v in R["spec"].items():
            if "error" in v or not all(v[p]["identical"] for p in ids):
                continue
            m = min(v[p]["tok_s"] for p in gp)
            if best is None or m > best[1]:
                best = (name, m)
        R["best_spec"] = {"config": best[0], "min_tok_s_over_prompts": best[1]} if best else None
        R["gate_60"] = bool(best and best[1] >= 60.0)
        log("best", R["best_spec"], "gate_60", R["gate_60"], "V5_pass", R["V5_pass"])
        save()

    # ---------------------------------------------------------------- per-graph profile
    if "prof" in secs:
        try:
            R["prof"] = {}
            for k in (3,):
                eng.set_option("spec_prof", 1)
                p = list(ids)[0]
                # stats accumulate: read before / after
                s0 = eng.stats().get("spec", {})
                out, t0, t1 = run(p, 128, {"spec_k": k, "spec_dv": 1})
                s1 = eng.stats().get("spec", {})
                eng.set_option("spec_prof", 0)
                eng.set_option("spec_k", 0)
                R["prof"][f"k{k}"] = {"ms_draft": s1.get("ms_draft"), "ms_verify": s1.get("ms_verify"),
                                      "iters": s1.get("iters_timed"), "tok_s_sync_each_iter":
                                          round((len(out) - 1) / (t1 - t0), 2), "before": s0.get("iters_timed")}
                log("prof", R["prof"])
        except Exception:  # noqa: BLE001
            R["prof_error"] = traceback.format_exc()[-2000:]
        save()
    # ---------------------------------------------------------------- CUPTI timelines: plain decode vs spec graphs
    if "trace" in secs:
        try:
            p = list(ids)[0]
            eng.set_option("spec_k", 0)
            eng.reset()
            eng.prefill(ids[p])
            eng.set_option("trace", 24)
            R["trace_plain"] = eng.stats().get("trace")
            for k in [int(x) for x in a.trace_ks.split(",") if x]:
                eng.set_option("spec_trace", 12)
                out, t0, t1 = run(p, 64, {"spec_k": k, "spec_dv": 1})
                eng.set_option("spec_trace", 0)
                eng.set_option("spec_k", 0)
                R[f"trace_spec_k{k}"] = eng.stats().get("spec_trace")
                log("trace k", k, str(R[f"trace_spec_k{k}"])[:400])
        except Exception:  # noqa: BLE001
            R["trace_error"] = traceback.format_exc()[-2000:]
            log(R["trace_error"])
        save()
    R["final_stats"] = {k: v for k, v in eng.stats().items() if k in ("spec", "mtp", "p2p", "arpub")}
    save()


if __name__ == "__main__":
    main()
