"""t4q stage p (prefill milestone): W4A8 int8-mma GEMM bench (tools/gemm_bench.cu) + TP prefill engine checks.

SECTIONS selects what runs: "gemm" (no model download), "engine" (download, oracle, tests/prefill_check.py).

Generated into kaggle/<stage>/t4q-<stage>.py by t4q/tools/mkkernel.py. No secrets; every download is public.
Flow: unpack -> (download Q4_0 GGUF in a thread) build libt4q + oracle_dump -> tokenize prompts -> oracle (llama.cpp
a4cb4c61 sm_75 libllama from kernel_sources otdoges/t4-qwen38-baseline) -> tests/tp_check.py -> RESULTS block.
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
STAGE = "p"
SECTIONS = ["engine"]
PF_CONFIGS = "pf_g8=1,pf_ga=64,pf_fuse=1,pf_silu=1,pf_gdnc=1,pf_gdnc_chk=0;pf_g8=1,pf_ga=64,pf_fuse=1,pf_silu=1,pf_gdnc=0,pf_gdnc_chk=0;pf_gdnc=1,pf_gdnc_chk=1"
RESULTS = {"stage": STAGE}
TGZ = "__T4Q_TGZ_B64__"
REPO = "unsloth/Qwen3.8-27B-GGUF"
GGUF = "Qwen3.8-27B-Q4_0.gguf"
CONFIGS = "arpub=-1;arpub=-1,pf_kb=1536"
TP_ARGS = ["--modes", "eager,graphs", "--gen", "256", "--depth", "3584", "--configs", CONFIGS, "--rounds", "3",
           "--trace", "24", "--trace_dir", str(OUT / "traces")]
VARIANT_CONFIGS = "arpub=-1"


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
    # L: >= 2048 tokens of mixed text (wiki paragraphs and the two coding prompts, numbered sections)
    parts, k = [], 0
    while sum(len(x) for x in parts) < 2100:
        k += 1
        parts.append(tok.encode(f"\n\nSection {k}. " + (WIKI if k % 3 else PROMPTS["P0"] + " " + PROMPTS["P1"])))
    ids["L"] = np.concatenate(parts)[:2048].astype(np.int32)
    for name, x in ids.items():
        x.tofile(WORK / f"{name}.i32")
    for name in ("P0", "P1", "W"):
        jobs.append(f"seq {name} {WORK / (name + '.i32')} 1")
    for name in ("P0", "P1", "W", "L"):
        jobs.append(f"last {name} {WORK / (name + '.i32')} 0")
    for name, n in (("P0", 64), ("P1", 64), ("W", 32), ("L", 32)):
        jobs.append(f"gen gen_{name} {WORK / (name + '.i32')} {n}")
    (WORK / "jobs.txt").write_text("\n".join(jobs) + "\n")
    result("inputs", {k: int(len(v)) for k, v in ids.items()} | {"tokenizer": tok.kind})

def gemm_section(t4q):
    bdir = W / "gb"
    bdir.mkdir(exist_ok=True)
    flags = [NVCC, "-O3", "-std=c++17", "-arch=sm_75", "-lineinfo", "-Xptxas", "-v"]
    procs = []
    for u in GEMM_KBU:
        cmd = flags + [f"-DT4Q_GEMM_KBU={u}", str(t4q / "tools" / "gemm_bench.cu"), "-o", str(bdir / f"gemm_bench_u{u}"),
                       "-ldl", "-lpthread"]
        procs.append((u, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))
    built = []
    for u, p in procs:
        o, _ = p.communicate(timeout=900)
        (LOGS / f"build_gemm_u{u}.txt").write_text(o)
        if p.returncode == 0:
            built.append(u)
        else:
            log(f"gemm build u{u} failed:\n{o[-3000:]}")
    result("gemm_build", {"built": built, "secs": el()})
    for fn in GEMM_SASS:
        sh(f"cuobjdump -sass -fun '{fn}' {bdir / ('gemm_bench_u' + str(built[0]))} > {LOGS / ('sass_' + fn[-40:].replace('/', '_') + '.txt')}",
           timeout=300) if built else None
    out = {}
    for u in (built if GEMM_TABLE else []):
        rc, o = stream([str(bdir / f"gemm_bench_u{u}"), "--dev", "0"], f"gemm_u{u}_dev0.txt", timeout=900)
        rows = [json.loads(l[2:]) for l in o.splitlines() if l.startswith("R ")]
        checks = [json.loads(l[6:]) for l in o.splitlines() if l.startswith("CHECK ")]
        rv = [json.loads(l[3:]) for l in o.splitlines() if l.startswith("RV ")]
        summ = [json.loads(l[8:]) for l in o.splitlines() if l.startswith("SUMMARY ")]
        out[f"u{u}"] = {"rc": rc, "rows": [{k: r[k] for k in ("shape", "T", "us", "TOPS", "sm_mhz", "power_w",
                                                               "pct_of_peak_at_clock", "quant_us")} for r in rows],
                        "checks": [{k: c.get(k) for k in ("shape", "variant", "T", "rel_l2_vs_q8ref", "rel_l2_vs_fp32x",
                                                           "quant_mismatch", "ok")} for c in checks],
                        "variants": rv,
                        "summary": summ[0] if summ else None, "fatal": [l for l in o.splitlines() if "FATAL" in l]}
        result(f"gemm_u{u}", out[f"u{u}"])
    # sustained: both GPUs at once, best build
    if built:
        best = built[0]
        if len(built) > 1:
            def tot(u):
                s = (out.get(f"u{u}") or {}).get("summary") or {}
                return -(s.get("T2048") or {}).get("TOPS", 0)
            best = sorted(built, key=tot)[0]
        procs = [subprocess.Popen([str(bdir / f"gemm_bench_u{best}"), "--dev", str(d), "--sustain", str(GEMM_SUSTAIN),
                                   "--shape", "none", "--T", "2048", "--variants", GEMM_SVARS], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True) for d in (0, 1)]
        S = []
        for d, p in enumerate(procs):
            o, _ = p.communicate(timeout=900)
            (LOGS / f"gemm_sustain_dev{d}.txt").write_text(o)
            S += [json.loads(l[2:]) for l in o.splitlines() if l.startswith("S ")]
        agg = {}
        for s in S:
            if s["t"] < 2:
                continue
            a = agg.setdefault(f'{s.get("variant")}_dev{s["dev"]}', [])
            a.append((s["TOPS"], s["sm_mhz"], s["power_w"], s["temp"]))
        summ = {d: {"windows": len(v), "TOPS_mean": round(sum(x[0] for x in v) / len(v), 2),
                    "TOPS_min": round(min(x[0] for x in v), 2), "sm_mhz_mean": round(sum(x[1] for x in v) / len(v)),
                    "sm_mhz_min": min(x[1] for x in v), "power_w_mean": round(sum(x[2] for x in v) / len(v), 1),
                    "temp_max": max(x[3] for x in v)} for d, v in agg.items() if v}
        result("gemm_sustain", {"build": f"u{best}", "per_dev": summ})


def ref_section(t4q):
    """cuBLAS / CUTLASS reference GEMMs (tools/ref_bench.cu): burst on dev0, then sustained on both GPUs at once."""
    bdir = W / "rb"
    bdir.mkdir(exist_ok=True)
    rc, o = sh("git clone -q --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git " + str(W / "cutlass"),
               timeout=300, logname="cutlass_clone.txt")
    have_cut = (W / "cutlass" / "include" / "cutlass" / "gemm" / "device" / "gemm.h").exists()
    src = str(t4q / "tools" / "ref_bench.cu")
    flags = [NVCC, "-O3", "-std=c++17", "-arch=sm_75", src, "-lcublas", "-lcublasLt", "-ldl"]
    builds = [("blas", flags + ["-o", str(bdir / "ref_blas")])]
    if have_cut:
        builds.append(("cut", flags + ["-DT4Q_CUTLASS", "-I" + str(W / "cutlass" / "include"), "-o", str(bdir / "ref_cut")]))
    procs = [(n, subprocess.Popen(c, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)) for n, c in builds]
    built = []
    for n, p in procs:
        o, _ = p.communicate(timeout=1200)
        (LOGS / f"build_ref_{n}.txt").write_text(o)
        if p.returncode == 0:
            built.append(n)
        else:
            log(f"ref build {n} failed:\n{o[-3000:]}")
    result("ref_build", {"built": built, "cutlass": have_cut, "secs": el()})
    if not built:
        return
    exe = str(bdir / ("ref_cut" if "cut" in built else "ref_blas"))
    rc, o = stream([exe, "--dev", "0"], "ref_burst_dev0.txt", timeout=600)
    result("ref_burst", [json.loads(l[2:]) for l in o.splitlines() if l.startswith("B ")])
    procs = [subprocess.Popen([exe, "--dev", str(d), "--sustain", str(REF_SUSTAIN)], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) for d in (0, 1)]
    S = []
    for d, p in enumerate(procs):
        o, _ = p.communicate(timeout=1200)
        (LOGS / f"ref_sustain_dev{d}.txt").write_text(o)
        S += [json.loads(l[2:]) for l in o.splitlines() if l.startswith("S ")]
    result("ref_sustain", summarize_sustain(S))


def summarize_sustain(S):
    agg = {}
    for s in S:
        if s["t"] < 2:
            continue
        agg.setdefault(f'{s.get("variant")}_dev{s["dev"]}', []).append((s["TOPS"], s["sm_mhz"], s["power_w"], s["temp"]))
    return {d: {"windows": len(v), "TOPS_mean": round(sum(x[0] for x in v) / len(v), 2),
                "TOPS_min": round(min(x[0] for x in v), 2), "sm_mhz_mean": round(sum(x[1] for x in v) / len(v)),
                "sm_mhz_min": min(x[1] for x in v), "power_w_mean": round(sum(x[2] for x in v) / len(v), 1),
                "temp_max": max(x[3] for x in v)} for d, v in agg.items() if v}


REF_SUSTAIN = 8
GEMM_KBU = [2]
GEMM_TABLE = True
GEMM_SASS = ["_ZN3t4q5gemm812gemm9_kernelILi0ELi4ELi256ELi64ELi0EEEvNS0_4ArgsE"]
GEMM_SUSTAIN = 4
GEMM_SVARS = "17,27,28,29,17"
NVCC = "/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else (shutil.which("nvcc") or "nvcc")


def main():
    mon = None
    try:
        sh("nvidia-smi; nvidia-smi topo -m; nvidia-smi topo -p2p w; nvidia-smi -q -d CLOCK,POWER | head -80",
           logname="nvidia_smi.txt")
        mon = clocks_monitor()
        t4q = unpack()
        if "ref" in SECTIONS:
            ref_section(t4q)
        if "gemm" in SECTIONS:
            gemm_section(t4q)
        if "engine" not in SECTIONS:
            return
        th = threading.Thread(target=downloader, daemon=True)
        th.start()
        rc, o = sh(f"make -C {t4q} -j4", timeout=900, logname="build.txt")
        ptx = "".join(Path(p).read_text() for p in glob.glob(f"{t4q}/build/**/*.ptxas.txt", recursive=True))
        (LOGS / "ptxas.txt").write_text(ptx)
        result("build", {"ok": rc == 0, "secs": el(), "tail": o[-3000:] if rc else ""})
        if rc:
            return
        llb, files = setup_llama()
        result("llama_libs", files)
        rc, o = sh(f"make -C {t4q} build/oracle_dump LLAMA_LIB={llb}", timeout=600, logname="build_oracle.txt")
        result("build_oracle", {"ok": rc == 0, "tail": o[-3000:] if rc else ""})
        prepare_inputs(t4q)
        th.join(timeout=1800)
        model = DL.get("path")
        if not model:
            result("fatal", "download failed")
            return
        env = dict(os.environ, LD_LIBRARY_PATH=f"{llb}:/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", ""))
        (OUT / "traces").mkdir(exist_ok=True)
        cup = sorted(glob.glob("/usr/local/cuda/**/libcupti.so*", recursive=True)) + sorted(
            glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/cuda_cupti/lib/libcupti.so*")) + sorted(
            glob.glob("/usr/local/cuda*/extras/CUPTI/lib64/libcupti.so*"))
        result("cupti", cup[:6])
        if cup:
            os.environ["T4Q_CUPTI"] = cup[0]
        if (t4q / "build" / "oracle_dump").exists():
            t = time.time()
            rc, o = stream([str(t4q / "build" / "oracle_dump"), model, str(WORK / "jobs.txt"), str(ORC), "layer"],
                           "oracle.log", timeout=1200, env=env)
            result("oracle", {"rc": rc, "secs": round(time.time() - t),
                              "lines": [ln for ln in o.splitlines() if ln.startswith("ORACLE")][-40:],
                              "tail": o[-2500:] if rc else ""})
        t = time.time()
        vout = OUT / "results_pf.json"
        remaining = DEADLINE - el() - 60
        rc, o = stream([sys.executable, "-u", str(t4q / "tests" / "prefill_check.py"), "--model", model, "--work",
                        str(WORK), "--oracle", str(ORC), "--out", str(vout), "--lib", str(t4q / "build" / "libt4q.so"),
                        "--configs", PF_CONFIGS, "--bench_n", "512,2048", "--reps", "2"],
                       "prefill_check.log", timeout=max(300, remaining))
        val = json.loads(vout.read_text()) if vout.exists() else {}
        for k, v in (val.get("bench") or {}).items():
            if isinstance(v, dict) and "t0" in v:
                v["clocks"] = clocks_between(v["t0"], v["t1"])
        result("prefill_check", {"rc": rc, "secs": round(time.time() - t), "tail": o[-3000:] if rc else ""})
        result("summary", {k: val.get(k) for k in ("load_s", "p2p", "selftest_worst", "correct_pass", "best_pp", "gate",
                                                   "correct", "bench")})
    except Exception:  # noqa: BLE001
        import traceback
        result("fatal", traceback.format_exc()[-3000:])
    finally:
        if mon:
            mon.kill()
        print("RESULTS " + json.dumps(RESULTS, default=str)[:20000], flush=True)


if __name__ == "__main__":
    main()
