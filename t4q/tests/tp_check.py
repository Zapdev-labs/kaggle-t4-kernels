"""TP engine check + bench (stages m2-m4): correctness vs the llama.cpp oracle, then decode speed.

usage: python tp_check.py --model M.gguf --work WORK --oracle ORC --out results_tp.json --lib libt4q.so
         [--modes eager,graphs] [--bench 1] [--gen 256] [--depth 3584]

Correctness (DESIGN.md s10, adapted to TP; criteria as in PROGRESS.md for M1):
  selftest: every fast GEMV format vs fp64 CPU dequant with the same q8 activations (load time)
  V1: residual-stream intermediates (attn_norm, attn_residual, attn_post_norm, l_out, result_norm) vs llama.cpp
      token-by-token dumps, within max(design tol, 2 x llama batch-vs-tbt floor); TP residual bit-identical on both GPUs
  V2: full logits on P0/P1/W vs oracle batch and token-by-token; mean KL within 2x the llama floor
  V3: greedy 128 tokens on P0/P1 identical to llama.cpp or first divergence at a near-tie (gap < 0.05)
  V4: graphs vs eager greedy tokens and logits bit-identical
Bench: decode tok/s after P0/P1 (generate N), and at depth (prefill DEPTH tokens, generate 128); per-kernel profile.
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
import validate as VA  # noqa: E402
from t4q import T4Q, Tokenizer, EOS_IDS  # noqa: E402

R = {}


def log(*a):
    print("[tp_check]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--modes", default="eager,graphs")
    ap.add_argument("--bench", type=int, default=1)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3584)
    ap.add_argument("--sections", default="v1,v2,v3,v4")
    ap.add_argument("--configs", default="fuse=0,pf_kb=2048;fuse=0,pf_kb=0",
                    help="';'-separated option sets to bench; the first is the default (validated by V1-V4)")
    a = ap.parse_args()
    secs = set(a.sections.split(","))
    modes = [m for m in a.modes.split(",") if m]

    def save():
        with open(a.out, "w") as f:
            json.dump(R, f, indent=1, default=float)

    man = json.load(open(os.path.join(a.work, "manifest.json")))
    t = time.time()
    eng = T4Q(a.model, lib=a.lib, max_ctx=4096, verbose=1, tp=1)
    R["load_s"] = round(time.time() - t, 1)
    st = eng.stats()
    R["selftest"] = st.get("selftest")
    R["vram_used_mib"] = st.get("vram_used_mib")
    R["p2p"] = st.get("p2p")
    log("load", R["load_s"], "s; selftest worst", (st.get("selftest") or {}).get("worst"), "vram", st.get("vram_used_mib"))
    save()
    tok = None
    try:
        tok = Tokenizer()
    except Exception:  # noqa: BLE001
        pass

    # ---------------------------------------------------------------- V1 (eager dumps)
    if "v1" in secs:
        try:
            d = man["dump"]
            dids = VA.read_ids(os.path.join(a.work, d["ids"]))
            op = os.path.join(a.oracle, d["name"] + ".dump.bin")
            opb = os.path.join(a.oracle, d["name"] + ".dumpb.bin")
            orc = VA.read_oracle_dump(op)
            orcb = VA.batch_slice(VA.read_oracle_dump(opb), len(dids)) if os.path.exists(opb) else None
            eng.set_option("graphs", 0)
            tdumps, _ = VA.run_dump(eng, dids)
            mism = {k: float(v.get("tp_h_mismatch", [0])[0]) for k, v in tdumps.items()}
            r1, worst, wfloor = VA.v1(orc, tdumps, len(dids), orcb)
            R["V1"] = {"worst_ratio_vs_floor_by_key": wfloor, "worst_ratio_by_key": worst,
                       "n_compared": len(r1), "tp_h_mismatch_by_pos": mism,
                       "l_out_rel": {k: v for k, v in r1.items() if k.startswith("l_out-63") or k.startswith("result")}}
            R["V1_floor_pass"] = bool(wfloor) and max(wfloor.values()) <= 1.0
            R["TP_h_identical"] = all(v == 0 for v in mism.values())
            log("V1 floor ratios", json.dumps(wfloor), "h mismatch", mism)
        except Exception:  # noqa: BLE001
            R["V1_error"] = traceback.format_exc()[-3000:]
            log(R["V1_error"])
        save()

    # ---------------------------------------------------------------- V2/V3 per mode
    first_mode_v3 = None
    for mode in modes:
        M = R.setdefault(mode, {})
        eng.set_option("graphs", 1 if mode == "graphs" else 0)
        if "v2" in secs:
            try:
                v2 = VA.run_v2(eng, man, a)
                Af, Tt, F = v2["ALL_vs_batch"], v2["ALL_vs_tbt"], v2["ALL_floor"]
                M["V2"] = {k: v2[k] for k in ("ALL_vs_batch", "ALL_vs_tbt", "ALL_floor", "t4q_ms_per_step")}
                M["V2_kl_over_floor"] = {"batch": Af["mean_kl"] / max(F["mean_kl"], 1e-12),
                                         "tbt": Tt["mean_kl"] / max(F["mean_kl"], 1e-12)}
                M["V2_pass"] = bool(M["V2_kl_over_floor"]["batch"] <= 2.0 and M["V2_kl_over_floor"]["tbt"] <= 2.0)
                log(mode, "V2", json.dumps(M["V2"]), json.dumps(M["V2_kl_over_floor"]))
            except Exception:  # noqa: BLE001
                M["V2_error"] = traceback.format_exc()[-3000:]
                log(M["V2_error"])
            save()
        if "v3" in secs:
            try:
                v3, ok = VA.run_v3(eng, man, a, tok)
                M["V3"] = v3
                M["V3_pass"] = ok
                if first_mode_v3 is None:
                    first_mode_v3 = (mode, v3)
                elif "v4" in secs:
                    m0, ref = first_mode_v3
                    same = all(v3[k].get("t4q_text") == ref[k].get("t4q_text") for k in v3)
                    R["V4_graphs_vs_eager_greedy_identical"] = bool(same)
            except Exception:  # noqa: BLE001
                M["V3_error"] = traceback.format_exc()[-3000:]
                log(M["V3_error"])
            save()

    # V4: graphs vs eager logits bit-identical on a short sequence
    if "v4" in secs and "graphs" in modes and "eager" in modes:
        try:
            ids = VA.read_ids(os.path.join(a.work, "P1.i32"))[:24]
            out = {}
            for mode in ("eager", "graphs"):
                eng.set_option("graphs", 1 if mode == "graphs" else 0)
                eng.reset()
                out[mode] = eng.logits(ids)
            R["V4_graphs_vs_eager_logits_bitident"] = bool(np.array_equal(out["eager"], out["graphs"]))
            R["V4_max_abs_diff"] = float(np.abs(out["eager"] - out["graphs"]).max())
            log("V4", R["V4_graphs_vs_eager_logits_bitident"], R["V4_max_abs_diff"])
        except Exception:  # noqa: BLE001
            R["V4_error"] = traceback.format_exc()[-3000:]
            log(R["V4_error"])
        save()

    # ---------------------------------------------------------------- bench
    cfgs = []
    for c in a.configs.split(";"):
        kv = dict(x.split("=") for x in c.split(",") if x)
        cfgs.append(("_".join(f"{k}{v}" for k, v in kv.items()), {k: int(v) for k, v in kv.items()}))

    def apply(opts):
        for k, v in opts.items():
            eng.set_option(k, v)

    apply(cfgs[0][1])
    if a.bench:
        B = R.setdefault("bench", {})

        def bench_prompt(key, name, n):
            ids = VA.read_ids(os.path.join(a.work, name + ".i32"))
            eng.reset()
            t0 = time.time()
            eng.prefill(ids)
            tp_ = time.time() - t0
            w0 = time.time()
            g = eng.generate(n)
            tg = time.time() - w0
            B[key] = {"n_gen": int(len(g)), "decode_tok_s": round((len(g) - 1) / tg, 2),
                      "prefill_tok_s": round(len(ids) / tp_, 2), "t0": w0, "t1": w0 + tg,
                      "text_head": tok.decode(g[:60]) if tok else ""}
            log("bench", key, json.dumps(B[key])[:400])

        for fi, (fz, opts) in enumerate(cfgs):
            apply(opts)
            for mode in (modes if fi == 0 else [modes[-1]]):
                eng.set_option("graphs", 1 if mode == "graphs" else 0)
                for name in ("P0", "P1"):
                    try:
                        bench_prompt(f"{mode}_{fz}_{name}", name, a.gen)
                    except Exception:  # noqa: BLE001
                        B[f"{mode}_{fz}_{name}_error"] = traceback.format_exc()[-2000:]
                        log(B[f"{mode}_{fz}_{name}_error"])
                    save()
            if fi > 0 and "v3" in secs:  # greedy check of the alternative kernel structure
                try:
                    v3, ok = VA.run_v3(eng, man, a, None)
                    R.setdefault("V3_alt", {})[fz] = {k: (v["first_divergence"], v["oracle_gap_at_div"])
                                                      for k, v in v3.items()} | {"pass": ok}
                except Exception:  # noqa: BLE001
                    R.setdefault("V3_alt", {})[fz] = {"error": traceback.format_exc()[-2000:]}
                save()
        # depth bench: long prompt through decode steps, then generate 128 per fuse setting at ~depth tokens of context
        if a.depth > 0:
            mode = modes[-1]
            try:
                apply(cfgs[0][1])
                eng.set_option("graphs", 1 if mode == "graphs" else 0)
                w = VA.read_ids(os.path.join(a.work, "W.i32"))
                p0 = VA.read_ids(os.path.join(a.work, "P0.i32"))
                body = np.concatenate([np.tile(w, a.depth // len(w) + 1)[: a.depth - len(p0)], p0]).astype(np.int32)
                eng.reset()
                t0 = time.time()
                eng.prefill(body)
                tp_ = time.time() - t0
                B["depth_prefill_via_decode_tok_s"] = round(len(body) / tp_, 2)
                for fz, opts in cfgs:
                    apply(opts)
                    ctx0 = eng.pos
                    w0 = time.time()
                    g = eng.generate(128)
                    tg = time.time() - w0
                    B[f"depth{a.depth}_{mode}_{fz}"] = {"ctx_start": int(ctx0), "n_gen": int(len(g)),
                                                          "decode_tok_s": round((len(g) - 1) / tg, 2), "t0": w0,
                                                          "t1": w0 + tg, "text_head": tok.decode(g[:40]) if tok else ""}
                    log("bench depth", json.dumps(B[f"depth{a.depth}_{mode}_{fz}"])[:400])
                    eng.set_option("profile", 1)
                    R.setdefault("profile_at_depth", {})[fz] = eng.stats().get("profile")
                    log("profile", fz, json.dumps(R["profile_at_depth"][fz]))
                apply(cfgs[0][1])
            except Exception:  # noqa: BLE001
                B["depth_error"] = traceback.format_exc()[-2000:]
                log(B["depth_error"])
            save()
    R["final_stats"] = {k: v for k, v in eng.stats().items() if k not in ("selftest", "profile")}
    # gate
    best = 0.0
    for k, v in R.get("bench", {}).items():
        if isinstance(v, dict) and "decode_tok_s" in v:
            best = max(best, v["decode_tok_s"])
    dep = [v["decode_tok_s"] for k, v in R.get("bench", {}).items() if k.startswith("depth") and isinstance(v, dict)]
    corr = {"selftest": bool((R.get("selftest") or {}).get("pass")), "V1_floor": R.get("V1_floor_pass"),
            "TP_h_identical": R.get("TP_h_identical")}
    for mode in modes:
        corr[f"V2_{mode}"] = R.get(mode, {}).get("V2_pass")
        corr[f"V3_{mode}"] = R.get(mode, {}).get("V3_pass")
    if "V4_graphs_vs_eager_logits_bitident" in R:
        corr["V4_logits"] = R["V4_graphs_vs_eager_logits_bitident"]
    R["gate_detail"] = corr
    R["correct"] = all(bool(v) for v in corr.values())
    R["best_decode_tok_s"] = best
    R["depth_decode_tok_s"] = max(dep) if dep else None
    R["gate_30"] = bool(R["correct"] and dep and max(dep) >= 30.0)
    save()
    log("GATE correct", R["correct"], "best decode", best, "depth decode", R["depth_decode_tok_s"], "gate", R["gate_30"])


if __name__ == "__main__":
    main()
