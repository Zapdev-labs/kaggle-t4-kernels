"""Milestone B check + bench: batched decode (t4q_batch_*) vs the single-stream TP decode engine, aggregate decode
throughput vs B at 1k / 4k context, end-to-end continuous batching (32 coding requests), OpenAI server smoke test.

usage: python batch_check.py --model M.gguf --lib libt4q.so --out results_b.json [--sections correct,bench,e2e,server]

correct: 8 coding prompts. Reference = single-stream greedy (graphs) and teacher-forced single-stream logits.
  Per config (head, state dtype): every prompt prefilled into its own slot; TF steps with the reference tokens at
  B = 8 (KL / top-1 vs the single-stream logits), then free-running greedy at B = 8 (first divergence vs the reference,
  logit gap there), then slots 0/1 re-run alone at B = 1 (must be bit-identical to their B = 8 rows).
bench: one long prompt prefilled into slot 0 and cloned into B slots; ms per step and aggregate tok/s per B, with
  nvidia-smi clocks (the stage driver joins them by t0 / t1).
e2e: 32 requests (~500-token code prompts, 512 new tokens each, EOS ignored) through py/batch.py.
server: py/server.py on localhost: concurrent chat stream / chat / completion requests.
"""
import argparse
import json
import os
import sys
import threading
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))
from t4q import EOS_IDS, T4Q, Tokenizer  # noqa: E402
from batch import Request, Scheduler  # noqa: E402

R = {}
OUT = None


def log(*a):
    print("[batch_check]", *a, flush=True)


def save():
    with open(OUT, "w") as f:
        json.dump(R, f, indent=1, default=str)


def log_softmax(x):
    x = np.asarray(x, np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def kl(p, q):
    lp, lq = log_softmax(p), log_softmax(q)
    return float((np.exp(lp) * (lp - lq)).sum(-1))


PROMPTS = {
    "P0": "Write a complete Python module `lru_cache.py` that implements a thread-safe LRU cache class with get, put, "
          "delete, resize and a `__len__`, using an OrderedDict and a threading.Lock. Include type hints, docstrings, "
          "and a full unittest test suite at the bottom covering eviction order, resizing, and concurrent access.",
    "P1": "Write a Python script that parses an Apache access log file, aggregates requests per IP, per status code and "
          "per hour, detects IPs with more than 100 requests per minute, and prints a report. Use argparse, "
          "dataclasses, collections.Counter and re. Include docstrings and example usage.",
    "C2": "Implement a C function `int parse_csv_line(const char *line, char **fields, int max_fields)` that handles "
          "quoted fields with embedded commas and escaped quotes, and write a small main() that tests it.",
    "C3": "Write a Rust program that reads a text file given on the command line, counts word frequencies "
          "case-insensitively, and prints the 20 most common words with their counts, using a HashMap.",
    "C4": "Write a TypeScript React component `TodoList` with add, toggle and delete, persisting the list to "
          "localStorage, with proper typing for the todo items and a short explanation of the design.",
    "C5": "Write a SQL schema for a library system (books, authors, members, loans) with foreign keys and indexes, "
          "then write queries for: overdue loans, the most borrowed authors this year, and members with no loans.",
    "C6": "Write a Python function that solves Sudoku puzzles with backtracking and constraint propagation, plus a "
          "pretty printer and an example puzzle. Explain the time complexity briefly.",
    "C7": "Write a Go HTTP server with two endpoints: POST /items stores a JSON item in memory with a mutex, GET "
          "/items/{id} returns it, with proper status codes and a graceful shutdown on SIGINT.",
}


def apply(eng, opts):
    for kv in [x for x in opts.split(",") if x]:
        k, v = kv.split("=")
        eng.set_option(k.strip(), int(v))


def sec_correct(eng, tok, a):
    ids = {k: np.asarray(tok.encode(tok.chat_text(v)), np.int32) for k, v in PROMPTS.items()}
    names = list(ids)
    G, TF = a.gen, a.tf
    eng.set_option("graphs", 1)
    refs, refL = {}, {}
    t = time.time()
    for nm in names:
        eng.reset()
        eng.prefill(ids[nm])
        refs[nm] = [int(x) for x in eng.generate(G, stop=())]
        eng.reset()
        eng.prefill(ids[nm])
        refL[nm] = eng.logits(np.asarray(refs[nm][:TF], np.int32))
        log(f"ref {nm}: {len(ids[nm])} prompt tokens, first {refs[nm][:8]}")
    R["correct_ref"] = {"secs": round(time.time() - t, 1), "prompt_tokens": {k: int(len(v)) for k, v in ids.items()},
                        "ref_text_P1": tok.decode(refs["P1"])[:600]}
    save()
    res = {}
    nslot = len(names)
    for cfg in a.correct_configs.split(";"):
        opts, _, sf = cfg.partition("|")
        sf16 = int(sf or 0)
        key = f"{opts}|sf16={sf16}"
        log("config", key)
        apply(eng, opts)
        eng.batch_init(nslot, a.slot_ctx, sf16)
        r = {}
        # teacher-forced at B = nslot
        firsts = [eng.batch_prefill(i, ids[nm]) for i, nm in enumerate(names)]
        r["first_token_match"] = sum(int(f == refs[nm][0]) for f, nm in zip(firsts, names))
        kls, top1 = [], 0
        logits0 = None
        for k in range(TF):
            for i, nm in enumerate(names):
                eng.batch_set_token(i, refs[nm][k])
            eng.batch_step(list(range(nslot)))
            for i, nm in enumerate(names):
                lg = eng.batch_logits(i)
                if k == 0 and i == 0:
                    logits0 = lg.copy()
                kls.append(kl(refL[nm][k], lg))
                top1 += int(np.argmax(lg) == np.argmax(refL[nm][k]))
        kls = np.asarray(kls)
        r["tf"] = {"n": int(len(kls)), "kl_mean": float(kls.mean()), "kl_p99": float(np.quantile(kls, 0.99)),
                   "kl_max": float(kls.max()), "top1": top1 / len(kls)}
        # free-running greedy at B = nslot
        for i, nm in enumerate(names):
            eng.batch_prefill(i, ids[nm])
        outs = [[refs[nm][0]] for nm in names]  # first token checked above
        div = {nm: None for nm in names}
        t0 = time.time()
        for k in range(G - 1):
            o = eng.batch_step(list(range(nslot)))
            for i, nm in enumerate(names):
                outs[i].append(int(o[i]))
                if div[nm] is None and int(o[i]) != refs[nm][k + 1]:
                    lg = eng.batch_logits(i)
                    div[nm] = {"at": k + 1, "gap": float(lg[int(o[i])] - lg[refs[nm][k + 1]])}
        dt = time.time() - t0
        r["greedy"] = {nm: (div[nm] if div[nm] else {"at": None, "match": G}) for nm in names}
        r["greedy_full_match"] = sum(1 for nm in names if div[nm] is None)
        r["greedy_ms_per_step_B8"] = round(1e3 * dt / (G - 1), 2)
        # B-invariance: slot 0 alone, then slot 1 alone (bit-identical tokens and first-step logits)
        inv = {}
        for i in (0, 1):
            nm = names[i]
            eng.batch_prefill(i, ids[nm])
            eng.batch_set_token(i, refs[nm][0])
            seq = [refs[nm][0]]
            for k in range(G - 1):
                o = eng.batch_step([i])
                if k == 0 and i == 0:
                    lg = eng.batch_logits(0)
                    inv["logits0_maxdiff_vs_B8"] = float(np.abs(lg - logits0).max())
                seq.append(int(o[0]))
            inv[nm] = seq == outs[i]
        r["b_invariance"] = inv
        r["sample_text_P0"] = tok.decode(outs[0])[:400]
        r["outs"] = outs
        res[key] = r
        R["correct"] = res
        save()
        log(key, json.dumps({k: v for k, v in r.items() if k not in ("greedy", "sample_text_P0", "outs")}))
    keys = list(res)
    R["correct_same_tokens"] = {f"{x} == {y}": res[x]["outs"] == res[y]["outs"] for i, x in enumerate(keys)
                                for y in keys[i + 1:]}
    for r in res.values():
        r.pop("outs", None)
    save()
    eng.batch_free()


def long_ids(tok, n):
    import glob
    src = sorted(glob.glob(os.path.join(os.path.dirname(os.__file__), "*.py")))
    acc = []
    for p in src:
        try:
            acc += tok.encode(open(p, encoding="utf-8", errors="replace").read()[:20000])
        except Exception:  # noqa: BLE001
            continue
        if len(acc) >= n:
            break
    return np.asarray(acc[:n], np.int32)


def sec_bench(eng, tok, a):
    """spec: depth:n_slots:slot_ctx:sf16:B1,B2,..[:pf_ub[:opts1/opts2/..]] -- every option set runs every B on the
    same slots (options are runtime switches), interleaved per B so clock drift hits all sets alike"""
    res = R.setdefault("bench", {})
    for spec in a.bench.split(";"):
        parts = spec.split(":")
        depth, ns, sc, sf, bl = parts[:5]
        eng.set_option("pf_ub", int(parts[5]) if len(parts) > 5 and parts[5] else 2048)
        osets = (parts[6] if len(parts) > 6 else "").split("/")
        depth, ns, sc, sf = int(depth), int(ns), int(sc), int(sf)
        Bs = [int(x) for x in bl.split(",")]
        key = f"d{depth}_sf{sf}_n{ns}"
        log("bench", key, "slots", ns, "ctx", sc, "B", Bs, "option sets", osets)
        try:
            eng.batch_init(ns, sc, sf)
        except Exception as e:  # noqa: BLE001
            res[key] = {"error": str(e)}
            save()
            log("bench init failed", e)
            continue
        ids = long_ids(tok, depth)
        t = time.time()
        src = ns - 1  # pristine source slot; every run starts from clones of it at the same depth
        eng.batch_prefill(src, ids)
        rk = {"prefill_s": round(time.time() - t, 2), "depth": depth, "n_slots": ns, "slot_ctx": sc, "runs": {}}
        for B in Bs:
            if B > ns:
                continue
            for opts in osets:
                apply(eng, opts)
                for s in range(B):
                    eng.batch_clone(src, s)
                slots = list(range(B))
                eng.batch_step(slots)  # warm-up (each step advances every slot by one position)
                eng.batch_step(slots)
                n = a.steps
                eng.set_option("bd_reset_stats", 1)
                t0 = time.time()
                for _ in range(n):
                    eng.batch_step(slots)
                t1 = time.time()
                enq = (eng.stats().get("batch") or {}).get("enqueue_s", 0.0)
                ms = 1e3 * (t1 - t0) / n
                eng.set_option("bd_prof", 1)
                for _ in range(2):
                    eng.batch_step(slots)
                bst = eng.stats().get("batch") or {}
                eng.set_option("bd_prof", 0)
                prof = {k: round(v[0] / 2, 2) for k, v in (bst.get("profile") or {}).items()}  # ms per step, GPU0
                prof1 = {k: round(v[0] / 2, 2) for k, v in (bst.get("profile1") or {}).items()}  # GPU1
                rk["runs"][f"B{B}|{opts}"] = {"B": B, "opts": opts, "ms_per_step": round(ms, 2),
                                              "agg_tok_s": round(B * 1e3 / ms, 1), "enqueue_ms": round(1e3 * enq / n, 2),
                                              "t0": t0, "t1": t1, "profile_ms": prof, "profile1_ms": prof1}
                log(f"  B={B} [{opts}]: {ms:.1f} ms/step, {B * 1e3 / ms:.1f} tok/s aggregate, enqueue "
                    f"{1e3 * enq / n:.1f} ms; profile {prof}")
                res[key] = rk
                save()
        if src >= max(Bs):
            pass
    eng.batch_free()


def code_prompts(tok, n, target):
    import glob
    src = sorted(glob.glob(os.path.join(os.path.dirname(os.__file__), "*.py")))
    tasks = ["Explain what this code does and point out any bugs.", "Write unit tests for the main functions here.",
             "Refactor this code for readability and add type hints.", "Summarize the public API of this module.",
             "Suggest performance improvements for this code.", "Rewrite the core logic of this code in Rust."]
    out = []
    for i, p in enumerate(src):
        if len(out) >= n:
            break
        try:
            txt = open(p, encoding="utf-8", errors="replace").read()
        except Exception:  # noqa: BLE001
            continue
        body = tok.encode(txt[:12000])
        if len(body) < target:
            continue
        chunk = tok.decode(body[:target - 60])
        task = tasks[len(out) % len(tasks)]
        ids = tok.encode(tok.chat_text(f"Here is part of a Python module:\n```python\n{chunk}\n```\n{task}"))
        out.append(np.asarray(ids, np.int32))
    return out


def sec_e2e(eng, tok, a):
    prompts = code_prompts(tok, a.e2e_n, a.e2e_prompt)
    log("e2e prompts", len(prompts), "mean tokens", float(np.mean([len(p) for p in prompts])))
    ctx = max(len(p) for p in prompts) + a.e2e_gen + 8
    apply(eng, a.e2e_opts)
    sched = Scheduler(eng, a.e2e_n, ctx, a.e2e_sf16, prefill_per_iter=a.e2e_prefill_per_iter)
    reqs = [Request(p, max_new=a.e2e_gen, ignore_eos=True) for p in prompts]
    t0 = time.time()
    for r in reqs:
        sched.submit(r)
    sched.run_until_idle()
    t1 = time.time()
    gen = sum(len(r.out) for r in reqs)
    ptok = sum(len(r.ids) for r in reqs)
    st = sched.stats
    R["e2e"] = {"requests": len(reqs), "prompt_tokens": ptok, "gen_tokens": gen, "wall_s": round(t1 - t0, 2),
                "gen_tok_s": round(gen / (t1 - t0), 1), "total_tok_s": round((gen + ptok) / (t1 - t0), 1),
                "prefill_s": round(st["prefill_s"], 2), "decode_s": round(st["step_s"], 2),
                "decode_steps": st["steps"], "decode_agg_tok_s": round(st["decode_tokens"] / max(st["step_s"], 1e-9), 1),
                "prefill_tok_s": round(st["prefill_tokens"] / max(st["prefill_s"], 1e-9), 1),
                "max_batch": st["max_batch"], "ttft_mean_s": round(float(np.mean([r.t_first - r.t_submit for r in reqs])), 2),
                "slot_ctx": ctx, "state_f16": a.e2e_sf16, "opts": a.e2e_opts, "t0": t0, "t1": t1,
                "sample": tok.decode(reqs[0].out[:120])[:500]}
    save()
    log("e2e", json.dumps({k: v for k, v in R["e2e"].items() if k != "sample"}))
    eng.batch_free()


def sec_server(eng, tok, a):
    import urllib.request
    import server as srv
    sched = Scheduler(eng, 8, 1024, 1)
    httpd, _ = srv.start(sched, tok, "127.0.0.1", a.port)
    base = f"http://127.0.0.1:{a.port}/v1"

    def post(path, body, stream=False):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            if not stream:
                return json.loads(r.read())
            text, chunks, usage = "", 0, None
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                obj = json.loads(line[6:])
                chunks += 1
                d = obj["choices"][0].get("delta", {})
                text += d.get("content") or ""
                usage = obj.get("usage") or usage
            return {"text": text, "chunks": chunks, "usage": usage}

    msgs = [{"role": "user", "content": PROMPTS["P1"]}]
    out = {}
    jobs = {
        "chat_stream": lambda: post("/chat/completions", {"model": "x", "messages": msgs, "max_tokens": 48, "stream": True}, True),
        "chat": lambda: post("/chat/completions", {"model": "x", "messages": msgs, "max_tokens": 48}),
        "completion": lambda: post("/completions", {"model": "x", "prompt": "def fibonacci(n):", "max_tokens": 32}),
        "chat2": lambda: post("/chat/completions", {"model": "x", "messages": [{"role": "user", "content": PROMPTS["C3"]}], "max_tokens": 40}),
    }
    ths = []
    t0 = time.time()
    for k, f in jobs.items():
        def run(k=k, f=f):
            try:
                out[k] = f()
            except Exception as e:  # noqa: BLE001
                out[k] = {"error": repr(e)}
        th = threading.Thread(target=run)
        th.start()
        ths.append(th)
    for th in ths:
        th.join(timeout=900)
    models = json.loads(urllib.request.urlopen(base + "/models", timeout=30).read())
    httpd.shutdown()
    sched.shutdown()
    chat_txt = (out.get("chat") or {}).get("choices", [{}])[0].get("message", {}).get("content")
    R["server"] = {"secs": round(time.time() - t0, 1), "models": models, "max_batch": sched.stats["max_batch"],
                   "stream_equals_nonstream": chat_txt is not None and chat_txt == (out.get("chat_stream") or {}).get("text"),
                   "results": out}
    save()
    log("server", json.dumps(R["server"])[:1500])
    eng.batch_free()


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sections", default="correct,bench,e2e,server")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--tf", type=int, default=16)
    ap.add_argument("--slot_ctx", type=int, default=1024)
    ap.add_argument("--correct_configs", default="bd_head=1,bd_lbm=1|0;bd_head=1,bd_lbm=1|1;bd_head=1,bd_lbm=0|1;"
                                                 "bd_head=0,bd_lbm=1,bd_p2p=1|1")
    ap.add_argument("--bench", default="1000:64:1088:1:1,16,32,64::bd_lbm=0/bd_lbm=1/bd_lbm=1,bd_p2p=1;"
                                       "4000:32:4128:1:16,32:1024:bd_lbm=0,bd_p2p=0/bd_lbm=1,bd_p2p=0/bd_lbm=1,bd_p2p=1")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--e2e_n", type=int, default=32)
    ap.add_argument("--e2e_prompt", type=int, default=500)
    ap.add_argument("--e2e_gen", type=int, default=512)
    ap.add_argument("--e2e_sf16", type=int, default=1)
    ap.add_argument("--e2e_opts", default="bd_lbm=1,bd_p2p=0,pf_ub=2048")
    ap.add_argument("--e2e_prefill_per_iter", type=int, default=64)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max_ctx", type=int, default=4224)
    a = ap.parse_args()
    OUT = a.out
    t = time.time()
    eng = T4Q(a.model, lib=a.lib, max_ctx=a.max_ctx, tp=1)
    st = eng.stats()
    R["load"] = {"secs": round(time.time() - t, 1), "p2p": st.get("p2p"), "vram": st.get("vram_used_mib") or st.get("vram")}
    tok = Tokenizer()
    save()
    for sec in a.sections.split(","):
        t = time.time()
        try:
            {"correct": sec_correct, "bench": sec_bench, "e2e": sec_e2e, "server": sec_server}[sec](eng, tok, a)
        except Exception:  # noqa: BLE001
            R[f"{sec}_error"] = traceback.format_exc()[-3000:]
            log(sec, "failed:", R[f"{sec}_error"])
            try:
                eng.batch_free()
            except Exception:  # noqa: BLE001
                pass
        R.setdefault("section_secs", {})[sec] = round(time.time() - t, 1)
        save()
    R["stats"] = {k: v for k, v in eng.stats().items() if k in ("batch", "pf_last_batch_s", "p2p", "arpub")}
    save()


if __name__ == "__main__":
    main()
