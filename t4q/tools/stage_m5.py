"""t4q stage m5: MTP speculative decoding (tests/spec_check.py): plain greedy reference, verify-column bit-identity
(forced drafts), spec greedy byte-identical to plain for k = 2..6, tok/s and acceptance per prompt.

Generated into kaggle/<stage>/t4q-<stage>.py by t4q/tools/mkkernel.py. No secrets; every download is public.
Flow: unpack -> (download Q4_0 GGUF in a thread) build libt4q -> tokenize P0/P1 -> spec_check -> RESULTS block.
"""
import base64
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
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
STAGE = "m5"
RESULTS = {"stage": STAGE}
TGZ = "__T4Q_TGZ_B64__"
REPO = "unsloth/Qwen3.8-27B-GGUF"
GGUF = "Qwen3.8-27B-Q4_0.gguf"
SPEC_ARGS = ["--gen", "512", "--ks", "3,4,5,6", "--dvs", "1", "--dv0_ks", "", "--sections", "ref,v4,tc,v5,trace",
             "--trace_ks", "3", "--prompts", "P0,P1,P2", "--ngs", "0",
             "--extra", "spec_k=3,spec_rb=1;spec_k=4,spec_rb=1;spec_k=3,spec_sqt=128;spec_k=4,spec_sqt=128;spec_k=6;"
                        "spec_k=3,spec_tc=1,spec_rb=1;spec_k=4,spec_tc=1,spec_rb=1;spec_k=6,spec_tc=1"]
NOP2P_ARGS = ["--gen", "256", "--ks", "3", "--dvs", "1", "--dv0_ks", "", "--sections", "ref,v4,v5"]
RUN_NOP2P = False
# extra processes (env at load time): (name, env, args).
# r13/r14: the qkv_a (attention q|k|v, N 7168 K 5120) repack A/B at ENGINE level. The v42 bench A/B at the
# TRUE shape refuted the repack per-GEMV (M4 dp4a 243.1 vs 239.4, TC AR64 178.1 vs 177.8, rpl4 -30% at M8
# dp4a) - but the v42 r4a variant showed TWO anomalies its bench anchors contradict: x_k3_rb1 (rb=1,
# dp4a) jumped to rb=0 speed (54.95 -> 68.16 tok/s, +24%, all 3 prompts, same node, same run) right after
# the k3_dv1 (rb=0) config ERRORED (the 4-s AR-wait watchdog 8000 on its first spec step), and the class
# landed +1.2% above the main's rb0 best on every prompt. Either the errored config left the engine
# running rb0 semantics (an option-state artifact) or the repack removes a real ~10 ms/step rb=1 penalty.
# v43 settles it: the v4 section warms the spec machinery (spec-enter + first-step) so k3_dv1 (rb0) runs
# clean at rpl4, and the within-variant rb0/rb1 pair plus the main's rpl2 pair (47.6/57.7 ms per step)
# decide. If rb0@rpl4 > rb0@rpl2 the default flips; if the pair matches rpl2 the line closes.
VARIANTS = [("r4a", {"T4Q_RPL_QKV_A": "4"},
             ["--gen", "512", "--ks", "3", "--dvs", "1", "--dv0_ks", "", "--sections", "v4,v5",
              "--prompts", "P0,P1,P2", "--ngs", "0",
              "--extra", "spec_k=3,spec_rb=1;spec_k=3,spec_tc=1,spec_rb=1"])]


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


# P2: a code-edit prompt (the answer repeats most of the input), for prompt-lookup drafting
P2_CODE = """import os
import json


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    out = {}
    for k, v in cfg.items():
        if isinstance(v, str) and v.startswith("$"):
            out[k] = os.environ.get(v[1:], "")
        else:
            out[k] = v
    return out


def merge(a, b):
    res = dict(a)
    for k, v in b.items():
        if k in res and isinstance(res[k], dict) and isinstance(v, dict):
            res[k] = merge(res[k], v)
        else:
            res[k] = v
    return res


def flatten(d, prefix=""):
    items = []
    for k, v in d.items():
        key = prefix + "." + k if prefix else k
        if isinstance(v, dict):
            items.extend(flatten(v, key).items())
        else:
            items.append((key, v))
    return dict(items)


class Registry:
    def __init__(self):
        self.items = {}

    def register(self, name, fn):
        if name in self.items:
            raise KeyError(name)
        self.items[name] = fn

    def get(self, name):
        return self.items[name]

    def names(self):
        return sorted(self.items)
"""
PROMPTS = {
    "P0": "Write a complete Python module `lru_cache.py` that implements a thread-safe LRU cache class with get, put, "
          "delete, resize and a `__len__`, using an OrderedDict and a threading.Lock. Include type hints, docstrings, "
          "and a full unittest test suite at the bottom covering eviction order, resizing, and concurrent access.",
    "P1": "Write a Python script that parses an Apache access log file, aggregates requests per IP, per status code and "
          "per hour, detects IPs with more than 100 requests per minute, and prints a report. Use argparse, "
          "dataclasses, collections.Counter and re. Include docstrings and example usage.",
    "P2": "Add type hints and a short docstring to every function and method in this Python module. Do not change any "
          "logic. Reply with the complete updated module in one code block.\n\n```python\n" + P2_CODE + "```",
}
WIKI = (
    "The history of computing hardware covers the developments from early simple devices to aid calculation to modern "
    "day computers. The first aids to computation were purely mechanical devices which required the operator to set "
    "up the initial values of an elementary arithmetic operation, then manipulate the device to obtain the result. "
    "Later, computers represented numbers in a continuous form, for instance distance along a scale, rotation of a "
    "shaft, or a voltage. Numbers could also be represented in the form of digits, automatically manipulated by a "
    "mechanism. Although this approach generally required more complex mechanisms, it greatly increased the precision "
    "of results. The development of transistor technology and then the integrated circuit chip led to a series of "
    "breakthroughs, starting with transistor computers and then integrated circuit computers, causing digital "
    "computers to largely replace analog computers. Metal-oxide-semiconductor large-scale integration then enabled "
    "semiconductor memory and the microprocessor, leading to another key breakthrough, the miniaturized personal "
    "computer, in the 1970s. The cost of computers gradually became so low that personal computers by the 1990s, and "
    "then mobile computers, smartphones and tablets in the 2000s, became ubiquitous.\n\n"
    "Devices have been used to aid computation for thousands of years, mostly using one-to-one correspondence with "
    "fingers. The earliest counting device was probably a form of tally stick. Later record keeping aids throughout "
    "the Fertile Crescent included calculi, clay spheres, cones, and so on, which represented counts of items, probably "
    "livestock or grains, sealed in hollow unbaked clay containers. The use of counting rods is one example. The abacus "
    "was early used for arithmetic tasks. What we now call the Roman abacus was used in Babylonia as early as 2400 BC. "
    "Since then, many other forms of reckoning boards or tables have been invented. In a medieval European counting "
    "house, a checkered cloth would be placed on a table, and markers moved around on it according to certain rules, "
    "as an aid to calculating sums of money.\n\n"
    "Several analog computers were constructed in ancient and medieval times to perform astronomical calculations. "
    "These included the astrolabe and Antikythera mechanism from the Hellenistic world. In Roman Egypt, Hero of "
    "Alexandria made mechanical devices including automata and a programmable cart. Other early mechanical devices "
    "used to perform one or another type of calculations include the planisphere and other mechanical computing "
    "devices invented by Abu Rayhan al-Biruni; the equatorium and universal latitude-independent astrolabe by Abu "
    "Ishaq Ibrahim al-Zarqali; the astronomical analog computers of other medieval Muslim astronomers and engineers; "
    "and the astronomical clock tower of Su Song during the Song dynasty. The castle clock, a hydropowered mechanical "
    "astronomical clock invented by Ismail al-Jazari in 1206, was the first programmable analog computer."
)


def prepare_inputs(t4q):
    sys.path.insert(0, str(t4q / "py"))
    from t4q import Tokenizer  # noqa: E402
    tok = Tokenizer()
    log("tokenizer:", tok.kind)
    import numpy as np
    man = {"seqs": [], "gens": [], "dump": None}
    jobs = []
    texts = {}
    for name, p in PROMPTS.items():
        texts[name] = tok.chat_text(p)
    texts["W"] = WIKI
    ids = {}
    for name, txt in texts.items():
        x = tok.encode(txt)
        if name == "W":
            x = x[:400]
            txt = tok.decode(x)
        ids[name] = np.asarray(x, dtype=np.int32)
        ids[name].tofile(WORK / f"{name}.i32")
        (WORK / f"{name}.txt").write_text(txt)
        jobs.append(f"tok {name} {WORK / (name + '.txt')} {WORK / (name + '.i32')}")
    for name in ("P0", "P1", "W"):
        T = 100000  # token-by-token oracle for every position after the first
        man["seqs"].append({"name": name, "ids": f"{name}.i32", "tbt": T})
        jobs.append(f"seq {name} {WORK / (name + '.i32')} {T}")
    for name in ("P0", "P1"):
        man["gens"].append({"name": "gen_" + name, "ids": f"{name}.i32", "n": 128})
        jobs.append(f"gen gen_{name} {WORK / (name + '.i32')} 128")
    ids["D"] = ids["P0"][:12]
    ids["D"].tofile(WORK / "D.i32")
    layers = ",".join(str(i) for i in range(64))
    man["dump"] = {"name": "D", "ids": "D.i32", "layers": layers}
    jobs.append(f"dump D {WORK / 'D.i32'} {layers}")
    jobs.append(f"dumpb D {WORK / 'D.i32'} {layers}")
    (WORK / "manifest.json").write_text(json.dumps(man, indent=1))
    (WORK / "jobs.txt").write_text("\n".join(jobs) + "\n")
    result("inputs", {k: int(len(v)) for k, v in ids.items()} | {"tokenizer": tok.kind,
                                                                  "P0_head": texts["P0"][:120]})


def prepare_prompts(t4q):
    sys.path.insert(0, str(t4q / "py"))
    from t4q import Tokenizer  # noqa: E402
    import numpy as np
    tok = Tokenizer()
    out = {}
    for name, p in PROMPTS.items():
        x = np.asarray(tok.encode(tok.chat_text(p)), dtype=np.int32)
        x.tofile(WORK / f"{name}.i32")
        out[name] = int(len(x))
    result("inputs", out | {"tokenizer": tok.kind})


def summarize(val):
    for p, v in (val.get("ref") or {}).items():
        if isinstance(v, dict) and "t0" in v:
            v["clocks"] = clocks_between(v["t0"], v["t1"])
    for name, cfg in (val.get("spec") or {}).items():
        for p, v in cfg.items():
            if isinstance(v, dict) and "t0" in v:
                v["clocks"] = clocks_between(v["t0"], v["t1"])
    keep = ("load_s", "selftest_worst", "vram_used_mib", "p2p", "mtp", "ref", "V4", "V4_pass",
            "V4_error", "tc", "tc_pass", "tc_error", "spec", "V5_pass", "best_spec", "gate_60", "prof", "prof_error",
            "final_stats", "trace_error")
    return {k: val.get(k) for k in keep if k in val}


def main():
    mon = None
    try:
        sh("nvidia-smi; nvidia-smi topo -m; nvidia-smi topo -p2p w; nvidia-smi -q -d CLOCK,POWER | head -80",
           logname="nvidia_smi.txt")
        mon = clocks_monitor()
        t4q = unpack()
        th = threading.Thread(target=downloader, daemon=True)
        th.start()
        rc, o = sh(f"make -C {t4q} -j4", timeout=1500, logname="build.txt")
        ptx = "".join(Path(p).read_text() for p in glob.glob(f"{t4q}/build/**/*.ptxas.txt", recursive=True))
        (LOGS / "ptxas.txt").write_text(ptx)
        result("build", {"ok": rc == 0, "secs": el(), "tail": o[-3000:] if rc else ""})
        if rc:
            return
        # the model download must be DONE before the bench phase: v29's concurrent download (RAM + xet
        # buffers + page-cache churn) had the OOM killer reaping the clock monitor and every case process's
        # tail mid-bench, and the surviving TC timings came out 1.2-2x slow while dp4a and the engine sweep
        # (which run later, after the download finished) stayed at speed
        th.join(timeout=2400)
        # the ENGINE spec sweep runs BEFORE the tc_bench matrix (v33: the worker OOM-killed the notebook
        # mid-matrix at case 6/8 and the engine data - the actual goal metric - was lost; the matrix is
        # synthetic P4 and needs no model, so it goes last and any late kill only costs the matrix)
        prepare_prompts(t4q)
        th.join(timeout=1800)
        model = DL.get("path")
        if not model:
            result("fatal", "download failed")
            return
        cup = sorted(glob.glob("/usr/local/cuda/**/libcupti.so*", recursive=True)) + sorted(
            glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/cuda_cupti/lib/libcupti.so*")) + sorted(
            glob.glob("/usr/local/cuda*/extras/CUPTI/lib64/libcupti.so*"))
        result("cupti", cup[:3])
        if cup:
            os.environ["T4Q_CUPTI"] = cup[0]
        t = time.time()
        vout = OUT / "results_spec.json"
        remaining = DEADLINE - el() - 60
        rc, o = stream([sys.executable, "-u", str(t4q / "tests" / "spec_check.py"), "--model", model, "--work",
                        str(WORK), "--out", str(vout), "--lib", str(t4q / "build" / "libt4q.so")] + SPEC_ARGS,
                       "spec_check.log", timeout=max(300, remaining))
        val = json.loads(vout.read_text()) if vout.exists() else {}
        result("spec_check", {"rc": rc, "secs": round(time.time() - t), "tail": o[-3000:] if rc else ""})
        result("summary", summarize(val))
        # v42: the r4a variant's FIRST spec step hit the 4-s AR-wait watchdog (8000): the main process's
        # CUDA teardown (8.6 GB x 2 GPUs) races the next process's first graph launches on the driver-global
        # locks, so let the teardown drain before the variant processes start
        time.sleep(20)
        for vname, venv, vargs in VARIANTS:
            if DEADLINE - el() < 420:
                break
            vo = OUT / f"results_spec_{vname}.json"
            rc, o = stream([sys.executable, "-u", str(t4q / "tests" / "spec_check.py"), "--model", model, "--work",
                            str(WORK), "--out", str(vo), "--lib", str(t4q / "build" / "libt4q.so")] + vargs,
                           f"spec_check_{vname}.log", timeout=DEADLINE - el() - 60, env=dict(os.environ, **venv))
            vv = json.loads(vo.read_text()) if vo.exists() else {}
            result(vname, {"rc": rc, "tail": o[-2000:] if rc else ""} | summarize(vv))
        if RUN_NOP2P and val.get("p2p") and DEADLINE - el() > 420:
            vout2 = OUT / "results_spec_nop2p.json"
            rc, o = stream([sys.executable, "-u", str(t4q / "tests" / "spec_check.py"), "--model", model, "--work",
                            str(WORK), "--out", str(vout2), "--lib", str(t4q / "build" / "libt4q.so")] + NOP2P_ARGS,
                           "spec_check_nop2p.log", timeout=DEADLINE - el() - 60,
                           env=dict(os.environ, T4Q_NO_P2P="1"))
            v2 = json.loads(vout2.read_text()) if vout2.exists() else {}
            result("nop2p", {"rc": rc, "tail": o[-2000:] if rc else ""} | summarize(v2))
        # v33/v34 both died mid-matrix to the worker's OOM killer with the 16 GB model's page cache still
        # resident: drop its CLEAN pages (POSIX_FADV_DONTNEED) before the synthetic-P4 matrix so the 8 case
        # processes (a ~1.2 GB dual-GPU CUDA context each, 12 cases in the r13 matrix) get the full RAM headroom
        try:
            fd = os.open(model, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except OSError:
            pass
        # int4 tensor-core verify GEMV vs dp4a: bit-identity + per-shape GB/s (synthetic P4, no model needed).
        # One process per case: a crash in one shape leaves the others' results intact. LAST (see the note at
        # prepare_prompts): the engine data lands first, and any late worker OOM only costs matrix columns.
        # ncu is unusable on Kaggle (ERR_NVGPUCTRPERM, admin-gated counters), so the within-run variant matrix
        # stays the only profiling instrument.
        tbr, tbo = sh(f"nvcc -O3 -std=c++17 -arch=sm_75 -Xcompiler -ffp-contract=off -I{t4q}/include "
                      f"{t4q}/tools/tc_bench.cu -o {W / 'tc_bench'}", timeout=900, logname="tc_bench_build.txt")
        if tbr == 0:
            parts = []
            # r13 question FIRST: the true qkv_a shape A/B (14/15: attn_e_tp/attn_e_r4, N 7168 - the engine's
            # real attention q|k|v, selftest-pinned; the stale N 4096 case stays as the control 12 attn_r4 that
            # pairs with them for the N dependence), then the real-shape continuity set (2-7, the v37/v40
            # comparison set), then outk5 (10, the rpl4-K5120 slow-regime point), micros last. The r12 A/B
            # twins 11 qkvz_r4 / 13 down_r2 are off the run lists (v41 measured both; verdicts recorded) - their
            # dispatches stay for the record, like the r11 RREG-4 twins 8/9. The engine now runs a VARIANTS
            # process before the matrix, so the kill window lands mid-matrix: new-questions-first protects the
            # r13 data, each case's process leaves its CHECK lines in its own log, and a late worker-OOM kill
            # only costs the continuity columns
            for ci in (14, 15, 12, 2, 3, 4, 5, 6, 7, 10, 0, 1):
                crc, cout = sh(f"{W / 'tc_bench'} --case {ci} --reps 100 --variants 0,1,2,3,4,5", timeout=900,
                               logname=f"tc_bench_{ci}.txt", cwd=str(W))
                tail = (OUT / "logs" / f"tc_bench_{ci}.txt")
                txt = tail.read_text()[-1400:] if tail.exists() else cout[-800:]
                parts.append(f"case{ci} rc={crc}\n{txt}")
            result("tc_bench", {"rc": 0, "out": ("\n").join(parts)[-5000:]})
        else:
            result("tc_bench", {"rc": tbr, "out": tbo[-3000:]})
    except Exception:  # noqa: BLE001
        import traceback
        result("fatal", traceback.format_exc()[-3000:])
    finally:
        if mon:
            mon.kill()
        print("RESULTS " + json.dumps(RESULTS, default=str)[:20000], flush=True)


if __name__ == "__main__":
    main()
