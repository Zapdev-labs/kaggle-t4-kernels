"""t4q stage cf0: the CYBER-FROST platform + format probes (r17 cf-m0). NO model download - the probe is
synthetic-in-VRAM: the disk write/seq/random-2MiB/random-4KiB/mmap-fault rates of the filesystem the 82.85 GB
GGUF will land on, the pinned-RAM zero-copy read ceiling per GPU and both concurrently (the warm tier), and
the Q2_K/Q4_K/Q5_1/Q4_0 dequant-GEMV rates at the real qwen4exp shapes (packed weight-byte GB/s, each kernel
spot-checked against a host model) - the four unmeasured rates the tier design and the honest ceiling ladder
hang on. Flow: unpack -> build tc_bench -> --cfprobe (cfdisk, cfpinned, dramprobe x2, cfq2k/cfq4k/cfq51/cfp4d)
-> parsed R lines into RESULTS.
"""
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
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
STAGE = "cf0"
RESULTS = {"stage": STAGE}
TGZ = "__T4Q_TGZ_B64__"


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


def clocks_monitor():
    f = open(LOGS / "clocks.csv", "w")
    return subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,"
                             "utilization.gpu,memory.used,clocks_throttle_reasons.active", "--format=csv,noheader",
                             "-lms", "1000"], stdout=f, stderr=subprocess.STDOUT)


def main():
    mon = None
    try:
        sh("nvidia-smi; nvidia-smi topo -m; nvidia-smi topo -p2p w; nvidia-smi -q -d CLOCK,POWER | head -80",
           logname="nvidia_smi.txt")
        # the tier design hangs on what /tmp actually is (overlay-on-disk vs tmpfs) - log it explicitly
        sh("df -h /tmp /kaggle/working /dev/shm; stat -f /tmp /kaggle/working; mount | grep -E ' /tmp | /kaggle '",
           logname="df.txt")
        mon = clocks_monitor()
        t4q = unpack()
        rc, o = sh(f"nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo -Xptxas -v {t4q}/tools/tc_bench.cu -o {WORK}/tc_bench",
                   timeout=1500, logname="tc_build.txt")
        result("build", {"ok": rc == 0, "secs": el(), "tail": o[-3000:] if rc else ""})
        if rc:
            return
        rc, out = stream([str(WORK / "tc_bench"), "--cfprobe", "--cfdir", str(MD), "--reps", "60"],
                         "cfprobe.txt", timeout=DEADLINE - 240)
        rows = []
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("R "):
                try:
                    rows.append(json.loads(line[2:]))
                except Exception:  # noqa: BLE001
                    pass
        result("probe_rows", len(rows))
        result("probe", rows)
        # the decision lines, surfaced from the parsed rows for the log
        keep = {}
        for r in rows:
            for k in ("cfdisk", "cfpinned", "cfq2k", "cfq4k", "cfq51", "cfp4d", "dramprobe"):
                if k in r:
                    keep.setdefault(k, []).append(r[k])
        result("probe_key", keep)
        result("probe_rc", rc)
    finally:
        if mon:
            mon.kill()
    log("cf0 done at", el(), "s")


if __name__ == "__main__":
    main()
