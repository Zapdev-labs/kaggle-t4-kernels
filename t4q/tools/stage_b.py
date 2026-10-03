"""t4q stage b (batched decode milestone): tests/batch_check.py on Kaggle 2x T4.

Generated into kaggle/b/t4q-b.py by t4q/tools/mkkernel.py (--no-sources). No secrets; every download is public.
Flow: unpack -> (download Q4_0 GGUF in a thread) build libt4q -> batch_check (correctness vs the single-stream
decode engine, throughput vs B at 1k / 4k context, 32-request end-to-end, OpenAI server smoke test) -> RESULTS block
with nvidia-smi clocks joined into every bench window.
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
STAGE = "b"
BC_ARGS = []  # extra tests/batch_check.py arguments
RESULTS = {"stage": STAGE}
TGZ = "__T4Q_TGZ_B64__"
REPO = "unsloth/Qwen3.8-27B-GGUF"
GGUF = "Qwen3.8-27B-Q4_0.gguf"


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


NVCC = "/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else (shutil.which("nvcc") or "nvcc")


def main():
    mon = None
    try:
        sh("nvidia-smi; nvidia-smi topo -m; nvidia-smi topo -p2p w; nvidia-smi -q -d CLOCK,POWER | head -80",
           logname="nvidia_smi.txt")
        mon = clocks_monitor()
        t4q = unpack()
        th = threading.Thread(target=downloader, daemon=True)
        th.start()
        rc, o = sh(f"make -C {t4q} -j4", timeout=1200, logname="build.txt")
        ptx = "".join(Path(p).read_text() for p in glob.glob(f"{t4q}/build/**/*.ptxas.txt", recursive=True))
        (LOGS / "ptxas.txt").write_text(ptx)
        result("build", {"ok": rc == 0, "secs": el(), "tail": o[-3000:] if rc else ""})
        if rc:
            return
        th.join(timeout=1800)
        model = DL.get("path")
        if not model:
            result("fatal", "download failed")
            return
        t = time.time()
        vout = OUT / "results_b.json"
        remaining = DEADLINE - el() - 60
        rc, o = stream([sys.executable, "-u", str(t4q / "tests" / "batch_check.py"), "--model", model, "--lib",
                        str(t4q / "build" / "libt4q.so"), "--out", str(vout)] + BC_ARGS,
                       "batch_check.log", timeout=max(300, remaining))
        val = json.loads(vout.read_text()) if vout.exists() else {}
        for k, v in (val.get("bench") or {}).items():
            for b, w in ((v or {}).get("per_B") or {}).items():
                if isinstance(w, dict) and "t0" in w:
                    w["clocks"] = clocks_between(w["t0"], w["t1"])
        if isinstance(val.get("e2e"), dict) and "t0" in val["e2e"]:
            val["e2e"]["clocks"] = clocks_between(val["e2e"]["t0"], val["e2e"]["t1"])
        (OUT / "results_b.json").write_text(json.dumps(val, indent=1, default=str))
        result("batch_check", {"rc": rc, "secs": round(time.time() - t), "tail": o[-3000:] if rc else ""})
        for k in ("load", "correct_ref", "correct", "bench", "e2e", "server", "correct_error", "bench_error",
                  "e2e_error", "server_error", "section_secs", "stats"):
            if k in val:
                result(k, val[k])
    except Exception:  # noqa: BLE001
        import traceback
        result("fatal", traceback.format_exc()[-3000:])
    finally:
        if mon:
            mon.kill()
        print("RESULTS " + json.dumps(RESULTS, default=str)[:30000], flush=True)


if __name__ == "__main__":
    main()
