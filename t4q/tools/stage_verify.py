"""t4q stage verify: independent audit of t4q's claimed numbers, apples-to-apples with llama.cpp in the same session.

Generated into kaggle/verify/t4q-verify.py by t4q/tools/mkkernel.py. No secrets; every download is public.
Flow: unpack -> (download Q4_0 GGUF + MTP GGUF in a thread) build libt4q + oracle_dump -> tokenize ->
  oracle (llama.cpp layer split: greedy 256 on P0/P1 with top-2 gaps, last-position logits of L512/L2048) ->
  t4q single (tests/verify_check.py: plain, MTP, oracle agreement, prefill) -> t4q batch (64 distinct prompts) ->
  llama-bench -sm tensor (pp512, pp2048, tg128) -> llama-server -sm tensor no-spec and MTP on the same token ids ->
  llama-batched-bench -sm tensor at B = 16/32/64 -> RESULTS block.
"""
import base64
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

T0 = time.time()
DEADLINE = 44 * 60
OUT = Path("/kaggle/working")
LOGS = OUT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
W = Path("/tmp/t4q")
W.mkdir(parents=True, exist_ok=True)
MD = W / "models"
MD.mkdir(exist_ok=True)
WORK = W / "work"
WORK.mkdir(exist_ok=True)
ORC = W / "oracle"
ORC.mkdir(exist_ok=True)
STAGE = "verify"
RESULTS = {"stage": STAGE}
TGZ = "__T4Q_TGZ_B64__"
REPO = "unsloth/Qwen3.8-27B-GGUF"
GGUF = "Qwen3.8-27B-Q4_0.gguf"
MTP_FILE = "MTP/mtp-Qwen3.8-27B-Q4_0.gguf"


def el():
    return round(time.time() - T0)


def log(*a):
    print(f"[{el():5d}s]", *a, flush=True)


def result(key, val):
    RESULTS[key] = val
    s = json.dumps({key: val}, default=str)
    print("RESULT " + (s if len(s) < 6000 else s[:6000] + "..."), flush=True)
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=1, default=str))


def sh(cmd, timeout=None, env=None, logname=None, cwd=None):
    t = time.time()
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout, env=env,
                           cwd=cwd)
        out, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        def dec(x):
            return x.decode(errors="replace") if isinstance(x, bytes) else (x or "")
        out = dec(e.stdout) + dec(e.stderr) + f"\n<<TIMEOUT after {timeout}s>>"
        rc = -9
    if logname:
        (LOGS / logname).write_text(f"$ {cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))}\n"
                                    f"rc={rc} secs={time.time() - t:.1f}\n{out}")
    return rc, out


def stream(cmd, logname, timeout, env=None, cwd=None):
    """run with output streamed to the log and to stdout (so the Kaggle log shows progress)"""
    t = time.time()
    with open(LOGS / logname, "w") as lf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=cwd)
        lines = []
        try:
            for line in p.stdout:
                lf.write(line)
                lf.flush()
                lines.append(line)
                if len(line) < 1500:
                    print("   |", line.rstrip(), flush=True)
                if time.time() - t > timeout:
                    p.kill()
                    lines.append(f"<<TIMEOUT after {timeout}s>>\n")
                    break
        finally:
            rc = p.wait()
    return rc, "".join(lines)


def unpack():
    src = W / "src"
    shutil.rmtree(src, ignore_errors=True)
    src.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(TGZ)), mode="r:gz") as tf:
        tf.extractall(src)
    return src / "t4q"


DL = {}


def downloader():
    t = time.time()
    env = dict(os.environ, HF_XET_HIGH_PERFORMANCE="1", HF_HUB_ENABLE_HF_TRANSFER="1", HF_HUB_DISABLE_PROGRESS_BARS="1",
               HF_HOME=str(W / "hf"))
    env.pop("HF_TOKEN", None)
    if not shutil.which("hf"):
        sh("pip install -q -U 'huggingface_hub[hf_xet]' hf_transfer", timeout=600, logname="pip_hf.txt")
    p = MD / GGUF
    rc, o = sh(["hf", "download", REPO, GGUF, "--local-dir", str(MD)], env=env, timeout=1800, logname="dl.txt")
    if rc or not p.exists():
        sh(f"curl -fL --retry 5 -o {p} https://huggingface.co/{REPO}/resolve/main/{GGUF}", timeout=1800,
           logname="dl_curl.txt")
    ok = p.exists() and p.stat().st_size > 1.6e10
    DL["path"] = str(p) if ok else None
    result("download", {"ok": ok, "secs": round(time.time() - t), "gb": round(p.stat().st_size / 1e9, 2) if ok else 0})


def clocks_monitor():
    f = open(LOGS / "clocks.csv", "w")
    return subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,"
                             "utilization.gpu,memory.used,clocks_throttle_reasons.active", "--format=csv,noheader",
                             "-lms", "1000"], stdout=f, stderr=subprocess.STDOUT)


def clocks_between(t0, t1):
    """mean SM clock / power / temp per GPU from logs/clocks.csv between unix times t0 and t1"""
    import datetime
    acc = {}
    try:
        for ln in (LOGS / "clocks.csv").read_text().splitlines():
            p = [x.strip() for x in ln.split(",")]
            if len(p) < 7:
                continue
            try:
                ts = datetime.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            except ValueError:
                continue
            if not (t0 <= ts <= t1):
                continue
            g = acc.setdefault(p[1], {"n": 0, "sm": 0.0, "w": 0.0, "c": 0.0, "min_sm": 1e9})
            sm = float(p[2].split()[0]); w = float(p[4].split()[0]); c = float(p[5])
            g["n"] += 1; g["sm"] += sm; g["w"] += w; g["c"] += c; g["min_sm"] = min(g["min_sm"], sm)
    except Exception as e:  # noqa: BLE001
        return {"err": str(e)}
    return {k: {"samples": v["n"], "sm_mhz": round(v["sm"] / v["n"]), "min_sm_mhz": v["min_sm"],
                "power_w": round(v["w"] / v["n"], 1), "temp_c": round(v["c"] / v["n"], 1)} for k, v in acc.items()
            if v["n"]}


def setup_llama():
    cands = sorted(glob.glob("/kaggle/input/**/libllama.so*", recursive=True))
    dst = Path("/tmp/llb")
    dst.mkdir(exist_ok=True)
    if cands:
        srcdir = Path(cands[0]).parent
        for f in srcdir.iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)
    else:
        tg = sorted(glob.glob("/kaggle/input/**/llama-bin-sm75.tgz", recursive=True))
        if tg:
            sh(f"tar -xzf {tg[0]} -C /tmp && cp -a /tmp/llama-bin-sm75/* {dst}/", timeout=600)
    # make sure soname links exist (kernel outputs may flatten symlinks)
    for f in list(dst.glob("*.so.*")):
        base = f.name.split(".so")[0] + ".so"
        parts = f.name.split(".so.")[1].split(".")
        for k in range(len(parts) + 1):
            name = base + ("." + ".".join(parts[:k]) if k else "")
            if not (dst / name).exists():
                os.symlink(f.name, dst / name)
    return dst, sorted(x.name for x in dst.iterdir())


PROMPTS = {
    "P0": "Write a complete Python module `lru_cache.py` that implements a thread-safe LRU cache class with get, put, "
          "delete, resize and a `__len__`, using an OrderedDict and a threading.Lock. Include type hints, docstrings, "
          "and a full unittest test suite at the bottom covering eviction order, resizing, and concurrent access.",
    "P1": "Write a Python script that parses an Apache access log file, aggregates requests per IP, per status code and "
          "per hour, detects IPs with more than 100 requests per minute, and prints a report. Use argparse, "
          "dataclasses, collections.Counter and re. Include docstrings and example usage.",
}


def dl_mtp():
    env = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1", HF_HOME=str(W / "hf"))
    env.pop("HF_TOKEN", None)
    p = MD / MTP_FILE
    for _ in range(60):
        if shutil.which("hf"):
            break
        time.sleep(5)
    sh(["hf", "download", REPO, MTP_FILE, "--local-dir", str(MD)], env=env, timeout=900, logname="dl_mtp.txt")
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        sh(f"curl -fL --retry 5 -o {p} https://huggingface.co/{REPO}/resolve/main/{MTP_FILE}", timeout=900)
    DL["mtp"] = str(p) if p.exists() and p.stat().st_size > 1e9 else None


def prepare(t4q):
    sys.path.insert(0, str(t4q / "py"))
    from t4q import Tokenizer  # noqa: E402
    import numpy as np
    tok = Tokenizer()
    info = {"tokenizer": tok.kind}
    for name, p in PROMPTS.items():
        x = np.asarray(tok.encode(tok.chat_text(p)), dtype=np.int32)
        x.tofile(WORK / f"{name}.i32")
        info[name] = int(len(x))
    src = sorted(glob.glob(os.path.join(os.path.dirname(os.__file__), "*.py")))
    # L: long stdlib-source prompt for pp512 / pp2048 (raw code tokens)
    acc = []
    for p in src[::-1]:
        try:
            acc += tok.encode(open(p, encoding="utf-8", errors="replace").read()[:20000])
        except Exception:  # noqa: BLE001
            continue
        if len(acc) >= 2048:
            break
    L = np.asarray(acc[:2048], np.int32)
    L.tofile(WORK / "L.i32")
    L[:512].tofile(WORK / "L512.i32")
    L.tofile(WORK / "L2048.i32")
    info["L"] = int(len(L))
    # 64 distinct chat prompts (different stdlib files, different tasks, 400-600 tokens)
    tasks = ["Explain what this code does and point out any bugs.", "Write unit tests for the main functions here.",
             "Refactor this code for readability and add type hints.", "Summarize the public API of this module.",
             "Suggest performance improvements for this code.", "Rewrite the core logic of this code in Rust."]
    bp = []
    for p in src:
        if len(bp) >= 64:
            break
        try:
            txt = open(p, encoding="utf-8", errors="replace").read()
        except Exception:  # noqa: BLE001
            continue
        body = tok.encode(txt[:12000])
        target = 400 + (len(bp) * 37) % 200
        if len(body) < target:
            continue
        chunk = tok.decode(body[:target - 60])
        ids = tok.encode(tok.chat_text(f"Here is part of a Python module:\n```python\n{chunk}\n```\n{tasks[len(bp) % 6]}"))
        bp.append([int(t) for t in ids])
    (WORK / "batch_prompts.json").write_text(json.dumps(bp))
    info["batch_prompts"] = len(bp)
    info["batch_distinct"] = len({tuple(x) for x in bp})
    info["batch_len_min_max"] = [min(map(len, bp)), max(map(len, bp))]
    jobs = ["gen gen_P0 %s 256" % (WORK / "P0.i32"), "gen gen_P1 %s 256" % (WORK / "P1.i32"),
            "last L512 %s 0" % (WORK / "L512.i32"), "last L2048 %s 0" % (WORK / "L2048.i32")]
    (WORK / "jobs.txt").write_text("\n".join(jobs) + "\n")
    result("inputs", info)


def left():
    return DEADLINE - el()


def bench(llb, env, model, tag, extra, timeout=900):
    cmd = [str(llb / "llama-bench"), "-m", model, "-ngl", "99", "-fa", "1", "-o", "json", "-t", "4"] + extra
    t = time.time()
    rc, o = sh(cmd, env=env, timeout=timeout, logname=f"llama_bench_{tag}.txt")
    rows = []
    m = re.search(r"\[\s*\{.*\}\s*\]", o, re.S)
    if m:
        try:
            for r in json.loads(m.group(0)):
                rows.append({"test": f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}",
                             "split_mode": r.get("split_mode"), "avg_ts": round(r["avg_ts"], 2),
                             "stddev_ts": round(r["stddev_ts"], 2), "n_batch": r.get("n_batch"),
                             "n_ubatch": r.get("n_ubatch")})
        except Exception as e:  # noqa: BLE001
            rows = [f"parse error {e}"]
    result(f"llama_bench_{tag}", {"rc": rc, "secs": round(time.time() - t), "rows": rows, "t0": t, "t1": time.time(),
                                  "clocks": clocks_between(t, time.time()), "err": "" if rows else o[-2500:]})


def serve(llb, env, model, tag, spec_args, timeout_load=420):
    port = 18080
    args = [str(llb / "llama-server"), "-m", model, "--host", "127.0.0.1", "--port", str(port), "-ngl", "99", "-sm",
            "tensor", "-fa", "on", "-c", "8192", "-np", "1", "-fit", "off", "-t", "4", "--no-webui", "-cram", "0"]
    args += spec_args
    logp = LOGS / f"server_{tag}.log"
    srv = subprocess.Popen(args, env=env, stdout=open(logp, "w"), stderr=subprocess.STDOUT)
    t = time.time()
    ok = False
    while srv.poll() is None and time.time() - t < timeout_load:
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).status == 200:
                ok = True
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    res = {"ok": ok, "load_s": round(time.time() - t), "args": " ".join(args[3:]), "runs": []}
    if not ok:
        srv.kill()
        res["log_tail"] = logp.read_text()[-2500:]
        result(f"server_{tag}", res)
        return res

    def req(ids, n=512):
        body = {"prompt": ids, "n_predict": n, "temperature": 0.0, "top_k": 1, "seed": 1, "cache_prompt": False,
                "return_tokens": True}
        rq = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                    headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(rq, timeout=900).read())

    try:
        import numpy as np
        req([int(x) for x in np.fromfile(WORK / "P0.i32", dtype=np.int32)][:16], 8)  # warm-up
        for rnd in range(2):
            for p in ("P0", "P1"):
                ids = [int(x) for x in np.fromfile(WORK / f"{p}.i32", dtype=np.int32)]
                t0 = time.time()
                r = req(ids)
                t1 = time.time()
                tm = r.get("timings", {})
                toks = r.get("tokens") or []
                if rnd == 0:
                    np.asarray(toks, np.int32).tofile(WORK / f"llama_{tag}_{p}.i32")
                    (LOGS / f"sample_{tag}_{p}.txt").write_text(r.get("content", ""))
                row = {"prompt": p, "round": rnd, "prompt_n": tm.get("prompt_n"), "decode_n": tm.get("predicted_n"),
                       "decode_tok_s": round(tm.get("predicted_per_second", 0), 2), "n_tokens": len(toks),
                       "draft_n": tm.get("draft_n"), "draft_accepted": tm.get("draft_n_accepted"),
                       "clocks": clocks_between(t0, t1)}
                if row["draft_n"]:
                    row["accept_rate"] = round(row["draft_accepted"] / row["draft_n"], 3)
                res["runs"].append(row)
                log(f"server {tag} {row}")
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)
    srv.terminate()
    try:
        srv.wait(60)
    except Exception:  # noqa: BLE001
        srv.kill()
    lt = logp.read_text()
    res["spec_log"] = re.findall(r".*(?:draft|mtp|MTP|nextn).*", lt)[:10]
    result(f"server_{tag}", res)
    return res


def batched_bench(llb, env, model, B, npp=512, ntg=64):
    c = B * (npp + ntg) + 512
    cmd = [str(llb / "llama-batched-bench"), "-m", model, "-ngl", "99", "-fa", "on", "-sm", "tensor", "-c", str(c),
           "-b", "2048", "-ub", "512", "-npp", str(npp), "-ntg", str(ntg), "-npl", str(B), "-t", "4"]
    t = time.time()
    rc, o = sh(cmd, env=env, timeout=min(600, max(60, left() - 60)), logname=f"batched_bench_B{B}.txt")
    rows = []
    for ln in o.splitlines():
        cells = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(cells) >= 10 and cells[0].isdigit():
            try:
                rows.append({"PP": int(cells[0]), "TG": int(cells[1]), "B": int(cells[2]), "N_KV": int(cells[3]),
                             "S_PP": float(cells[5]), "T_TG": float(cells[6]), "S_TG": float(cells[7])})
            except ValueError:
                pass
    result(f"llama_batched_B{B}", {"rc": rc, "secs": round(time.time() - t), "rows": rows,
                                   "clocks": clocks_between(t, time.time()), "err": "" if rows else o[-2000:]})


def compare_tokens():
    import numpy as np
    out = {}
    for p in ("P0", "P1"):
        f = {}
        for name, path in [("t4q_plain", WORK / f"t4q_plain_{p}.i32"), ("oracle_layer", ORC / f"gen_{p}.gen.i32"),
                           ("llama_tensor_nospec", WORK / f"llama_nospec_{p}.i32"),
                           ("llama_tensor_mtp", WORK / f"llama_mtp_{p}.i32")]:
            if path.exists():
                f[name] = np.fromfile(path, dtype=np.int32)
        names = sorted(f)
        r = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = f[names[i]], f[names[j]]
                m = min(len(a), len(b))
                ne = np.nonzero(a[:m] != b[:m])[0]
                r[f"{names[i]}~{names[j]}"] = {"common": int(m), "prefix_match": int(ne[0]) if len(ne) else int(m)}
        out[p] = r
    result("token_agreement", out)


def main():
    mon = None
    try:
        sh("nvidia-smi; nvidia-smi topo -m; nvidia-smi topo -p2p w", logname="nvidia_smi.txt")
        mon = clocks_monitor()
        t4q = unpack()
        th = threading.Thread(target=downloader, daemon=True)
        th.start()
        th2 = threading.Thread(target=dl_mtp, daemon=True)
        th2.start()
        rc, o = sh(f"make -C {t4q} -j4", timeout=1200, logname="build.txt")
        result("build", {"ok": rc == 0, "secs": el(), "tail": o[-3000:] if rc else ""})
        if rc:
            return
        llb, files = setup_llama()
        result("llama_libs", {"n": len(files), "commit": (llb / "COMMIT.txt").read_text()[:200]
                              if (llb / "COMMIT.txt").exists() else None})
        rc, o = sh(f"make -C {t4q} build/oracle_dump LLAMA_LIB={llb}", timeout=600, logname="build_oracle.txt")
        result("build_oracle", {"ok": rc == 0, "tail": o[-3000:] if rc else ""})
        prepare(t4q)
        th.join(timeout=1800)
        model = DL.get("path")
        if not model:
            result("fatal", "download failed")
            return
        env = dict(os.environ, LD_LIBRARY_PATH=f"{llb}:/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", ""))
        for b in llb.glob("llama-*"):
            b.chmod(0o755)
        # ---- 1. oracle (llama.cpp layer split, the validated reference)
        t = time.time()
        rc, o = stream([str(t4q / "build" / "oracle_dump"), model, str(WORK / "jobs.txt"), str(ORC), "layer"],
                       "oracle.log", timeout=900, env=env)
        result("oracle", {"rc": rc, "secs": round(time.time() - t), "files": sorted(x.name for x in ORC.iterdir()),
                          "tail": o[-2000:] if rc else ""})
        lib = str(t4q / "build" / "libt4q.so")
        # ---- 2. t4q single stream
        for sec, extra_env in (("single", {}), ("batch", {"T4Q_NO_MTP": "1"})):
            if left() < 600:
                result(f"t4q_{sec}", "skipped (deadline)")
                continue
            vout = OUT / f"verify_{sec}.json"
            t = time.time()
            rc, o = stream([sys.executable, "-u", str(t4q / "tests" / "verify_check.py"), "--section", sec, "--model",
                            model, "--lib", lib, "--work", str(WORK), "--oracle", str(ORC), "--out", str(vout)],
                           f"verify_{sec}.log", timeout=min(900, left() - 300), env=dict(os.environ, **extra_env))
            val = json.loads(vout.read_text()) if vout.exists() else {}
            for grp in (val.get("single") or {}).values():
                if isinstance(grp, dict):
                    for v in grp.values():
                        if isinstance(v, dict) and "t0" in v:
                            v["clocks"] = clocks_between(v["t0"], v["t1"])
            for v in ((val.get("batch") or {}).get("runs") or {}).values():
                v["clocks"] = clocks_between(v["t0"], v["t1"])
            vout.write_text(json.dumps(val, indent=1, default=str))
            result(f"t4q_{sec}", {"rc": rc, "secs": round(time.time() - t), "tail": o[-2500:] if rc else ""} | val)
        # ---- 3. llama.cpp, same session
        if left() > 300:
            bench(llb, env, model, "tensor", ["-sm", "tensor", "-p", "512,2048", "-n", "128", "-b", "2048", "-ub", "512",
                                              "-r", "3"], timeout=min(900, left() - 120))
        if left() > 300:
            serve(llb, env, model, "nospec", [])
        if left() > 240:
            r = serve(llb, env, model, "mtp", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3",
                                               "--spec-draft-ngl", "99"])
            if not r.get("ok") or not any(x.get("draft_n") for x in r.get("runs", [])):
                RESULTS["server_mtp_own_blk64_attempt"] = RESULTS.get("server_mtp")
                th2.join(timeout=300)
                if DL.get("mtp") and left() > 240:
                    serve(llb, env, model, "mtp", ["--spec-type", "draft-mtp", "-md", DL["mtp"], "--spec-draft-n-max",
                                                   "3", "--spec-draft-ngl", "99"])
        compare_tokens()
        for B in (16, 32, 64):
            if left() > 200:
                batched_bench(llb, env, model, B)
    except Exception:  # noqa: BLE001
        import traceback
        result("fatal", traceback.format_exc()[-3000:])
    finally:
        if mon:
            mon.kill()
        print("RESULTS " + json.dumps(RESULTS, default=str)[:30000], flush=True)


if __name__ == "__main__":
    main()
