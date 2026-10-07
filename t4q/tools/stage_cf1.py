"""t4q stage cf1: the CYBER-FROST correctness gate + the first-cut rate (cf-m1). Fully custom kernels, this repo.

Generated into kaggle/cf1/t4q-cf1.py by t4q/tools/mkkernel.py. No secrets; the download is public.
Flow: unpack -> (download the 82.85 GB Q2_K_S GGUF in a thread, ~6+ min write-bound) -> build libt4q +
oracle_dump + cf_run -> oracle jobs with the CF GGUF (chatw: the GGUF chat template + tokenize own the
tokenizer; seq: the last-48 per-position token-by-token logits; gen: 32 greedy) -> cf_run seq (rel diff +
top1 agree) + gen (byte-compare vs the oracle greedy) + time (the steady tok/s) -> RESULTS.
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
DEADLINE = 56 * 60
OUT = Path("/kaggle/working")
LOGS = OUT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
W = Path("/tmp/t4q")
W.mkdir(parents=True, exist_ok=True)
MD = W / "models"
MD.mkdir(exist_ok=True)
WORK = W / "work"
WORK.mkdir(parents=True, exist_ok=True)
ORC = W / "oracle"
ORC.mkdir(parents=True, exist_ok=True)
RESULTS = {"stage": "cf1"}
TGZ = "__T4Q_TGZ_B64__"
REPO = "freakyskittle/CYBER-FROST-3.8-GGUF"
GGUF = "CYBER-FROST-3.8-Q2_K_S.gguf"
T_SEQ = 8   # per-position logits compared (the oracle's own tbt tail)
N_GEN = 8   # greedy tokens byte-compared


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
            try:
                rc = p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # r19m (the v7 lesson): a killed-but-D-state child never exits - never wait
                # forever; abandon it (the session teardown reaps it) and report -9
                rc = -9
    return rc, "".join(lines)


def unpack():
    src = W / "src"
    shutil.rmtree(src, ignore_errors=True)
    src.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(TGZ)), mode="r:gz") as tf:
        tf.extractall(src)
    return src / "t4q"


DL = {}


def file_gb(p, logname):
    """REMOVED (r19m, the v7 lesson): the child-stat was no defense at all. The stat
    syscall D-locks on the fresh 82.85 GB inode regardless of which process runs it;
    subprocess.run(timeout)'s kill cannot reclaim a D-state child, and its post-kill
    communicate() then waits forever - the v7 downloader froze exactly there (dl_stat.txt
    never written). No caller remains; nothing may probe the fresh model file."""


def downloader():
    t = time.time()
    env = dict(os.environ, HF_XET_HIGH_PERFORMANCE="1", HF_HUB_ENABLE_HF_TRANSFER="1",
               HF_HUB_DISABLE_PROGRESS_BARS="1", HF_HOME=str(W / "hf"))
    env.pop("HF_TOKEN", None)
    if not shutil.which("hf"):
        sh("pip install -q -U 'huggingface_hub[hf_xet]' hf_transfer", timeout=600, logname="pip_hf.txt")
    p = MD / GGUF
    rc, o = sh(["hf", "download", REPO, GGUF, "--local-dir", str(MD)], env=env, timeout=2400, logname="dl.txt")
    if rc != 0:  # the hf path failed: the curl fallback
        rc, o = sh(f"curl -fL --retry 5 -o {p} https://huggingface.co/{REPO}/resolve/main/{GGUF}", timeout=2400,
                   logname="dl_curl.txt")
    # r19m (the v7 lesson): NO size probe on the fresh 82.85 GB file, in ANY form. The v6
    # froze the whole process on p.stat(); the v7 moved it to a child with a timeout and the
    # downloader STILL froze exactly there (dl_stat.txt never written): the stat syscall
    # D-locks on the fresh inode regardless of which process runs it, subprocess.run's
    # timeout kill cannot reclaim a D-state child, and its post-kill communicate() then
    # waits forever. The hf/curl rc already certifies the transfer; the gate runs open the
    # file themselves.
    ok = rc == 0
    DL["path"] = str(p) if ok else None
    DL["gb"] = "unprobed" if ok else "download failed"
    result("download", {"ok": ok, "secs": round(time.time() - t), "rc": rc})


def clocks_monitor():
    f = open(LOGS / "clocks.csv", "w")
    return subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,"
                             "utilization.gpu,memory.used", "--format=csv", "-l", "5"], stdout=f, stderr=subprocess.STDOUT)


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
    for f in list(dst.glob("*.so.*")):
        base = f.name.split(".so")[0] + ".so"
        parts = f.name.split(".so.")[1].split(".")
        for k in range(len(parts) + 1):
            name = base + ("." + ".".join(parts[:k]) if k else "")
            if not (dst / name).exists():
                os.symlink(f.name, dst / name)
    return dst, sorted(x.name for x in dst.iterdir())


# short prompts (the oracle decodes the whole 82.85 GB model CPU-side, and its cold expert
# reads dominate: every prompt token costs the oracle ~1.16 GB of faults, so the prompts stay
# minimal while still exercising every path: one coding ask, one prose ask)
PROMPTS = {
    "P0": "Write a one-line Python lambda that reverses a string.",
    "P1": "Name three benefits of quantized model weights.",
}

# the census prompt: ~200 tokens of natural prose (the same text class the 27B rounds used)
WIKI = (
    "The history of computing hardware covers the developments from early simple devices to "
    "aid calculation to modern day computers. The first aids to computation were purely "
    "mechanical devices which required the operator to set up the initial values of an "
    "elementary arithmetic operation, then manipulate the device to obtain the result. "
    "Numbers could also be represented in the form of digits, automatically manipulated by a "
    "mechanism. Although this approach generally required more complex mechanisms, it "
    "greatly increased the precision of results. The development of transistor technology "
    "and then the integrated circuit chip led to a series of breakthroughs, causing digital "
    "computers to largely replace analog computers. Semiconductor memory and the "
    "microprocessor led to the miniaturized personal computer in the 1970s, and personal "
    "computers became ubiquitous by the 1990s."
)


def parse_cf(out):
    vals = {}
    for ln in out.splitlines():
        if ln.startswith("CF {") and ln.rstrip().endswith("}"):
            try:
                vals.update(json.loads(ln[3:]))
            except Exception:  # noqa: BLE001
                pass
    return vals


def watchdog():
    """r19b lesson: a child that hangs silently blocks stream()'s readline. This thread
    owns that case (the GIL is free when a CHILD hangs): at DEADLINE-120 it flags, at
    DEADLINE it flushes and exits. The GIL-FROZEN case (the v6 wedge: the main blocked
    inside a non-GIL-releasing syscall) is owned by the process watchdog below."""
    time.sleep(max(60, DEADLINE - 120))
    result("watchdog", "deadline approaching in 120s")
    time.sleep(120)
    result("watchdog", "deadline hit - flushing and exiting; see logs/ for where each run stopped")
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=1, default=str))
    os._exit(0)


def spawn_watchdog_proc():
    """r19f lesson (the v6 post-mortem): the wedge froze the MAIN inside os.stat on the
    fresh 82.85 GB file - os.stat does NOT release the GIL, so the thread watchdog above
    froze with it (it shares the process's single GIL). This one is a separate PROCESS:
    it cannot be frozen by the parent. At the deadline it kills the (possibly frozen)
    parent by its baked-in pid so the session ends cleanly and the incremental
    results.json + the logs become the fetchable output."""
    ppid = os.getpid()
    src = (
        "import os, signal, time\n"
        f"ppid = {ppid}\n"
        f"deadline = {DEADLINE}\n"
        "def mark(msg):\n"
        "    try:\n"
        "        with open('/kaggle/working/logs/watchdog_proc.txt', 'a') as f:\n"
        "            f.write(msg + '\\n')\n"
        "    except OSError:\n"
        "        pass\n"
        "time.sleep(max(120, deadline - 120))\n"
        "time.sleep(120)\n"
        "# r19m (the v7 lesson): KILL FIRST, no disk writes before it - under the overlayfs\n"
        "# stall the mark() write froze this watchdog and the kill below never fired\n"
        "try:\n"
        "    os.kill(ppid, signal.SIGKILL)\n"
        "except (ProcessLookupError, PermissionError):\n"
        "    pass  # the parent already ended cleanly\n"
        "mark('deadline hit - killed the parent (kill-first; the mark is best-effort)')\n"
    )
    subprocess.Popen([sys.executable, "-c", src], start_new_session=True)


def main():
    mon = None
    try:
        sh("nvidia-smi", logname="nvidia_smi.txt")
        mon = clocks_monitor()
        spawn_watchdog_proc()
        threading.Thread(target=watchdog, daemon=True).start()
        t4q = unpack()
        # r19m (the v7 lesson): every static work-file write happens HERE, before the download
        # thread starts - the /tmp overlayfs is healthy in this window (the build's creates
        # all succeeded at ~391 s), and the v7 froze with the writes deferred to ~398 s
        for name, p in PROMPTS.items():
            (WORK / f"{name}.txt").write_text(p)
        jobs = []
        for name in PROMPTS:
            jobs.append(f"chatw {name} {WORK / (name + '.txt')} {WORK / (name + '.i32')}")
            jobs.append(f"seq {name} {WORK / (name + '.i32')} {T_SEQ}")
            jobs.append(f"gen gen_{name} {WORK / (name + '.i32')} {N_GEN}")
        (WORK / "jobs.txt").write_text("\n".join(jobs) + "\n")
        th = threading.Thread(target=downloader, daemon=True)
        th.start()
        rc, o = sh(f"make -C {t4q} -j4", timeout=1200, logname="build.txt")
        ptx = "".join(Path(p).read_text() for p in glob.glob(f"{t4q}/build/**/*.ptxas.txt", recursive=True))
        (LOGS / "ptxas.txt").write_text(ptx)
        spills = [ln for ln in ptx.splitlines() if "spill" in ln and "0 bytes spill" not in ln]
        result("build", {"ok": rc == 0, "secs": el(), "spill_lines": spills[:8],
                         "tail": o[-2000:] if rc else ""})
        if rc:
            return
        rc, o = sh(f"make -C {t4q} build/cf_run", timeout=600, logname="build_cf_run.txt")
        result("build_cf_run", {"ok": rc == 0, "tail": o[-1500:] if rc else ""})
        if rc:
            return
        llb, files = setup_llama()
        result("llama_libs", files)
        rc, o = sh(f"make -C {t4q} build/oracle_dump LLAMA_LIB={llb}", timeout=600, logname="build_oracle.txt")
        result("build_oracle", {"ok": rc == 0, "tail": o[-1500:] if rc else ""})
        if rc:
            return
        for name, p in PROMPTS.items():
            (WORK / f"{name}.txt").write_text(p)
        th.join(timeout=2400)
        model = DL.get("path")
        if not model:
            result("fatal", "download failed")
            return
        result("model", {"path": model, "gb": DL.get("gb")})  # no fresh getsize: the v6 lesson
        # oracle jobs: the chat template + tokenizer + the tbt tail + the greedy gens
        # (the work files were written pre-download: the r19m /tmp-health window)
        # the oracle runs CPU-only twice over: the 26.3 GiB PLE table prefetch OOMs a T4, and the
        # b10975 CUDA ssm-conv op asserts on the qwen4exp F16 conv weights (the CPU op handles them).
        # Hiding the GPU from the oracle leaves the cf_run gates on the GPU untouched.
        env = dict(os.environ, LD_LIBRARY_PATH=f"{llb}:/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", ""),
                   T4Q_ORACLE_NGPU="0", CUDA_VISIBLE_DEVICES="")
        t = time.time()
        rc, o = stream([str(t4q / "build" / "oracle_dump"), model, str(WORK / "jobs.txt"), str(ORC)],
                       "oracle.log", timeout=1500, env=env)
        result("oracle", {"rc": rc, "secs": round(time.time() - t),
                          "lines": [ln for ln in o.splitlines() if ln.startswith("ORACLE")][-40:],
                          "tail": o[-2000:] if rc else ""})
        if rc:
            return
        # the correctness gates, then the timed run
        summary = {}
        for name in PROMPTS:
            rc, o = stream([str(t4q / "build" / "cf_run"), "seq", model, str(WORK / f"{name}.i32"),
                            str(ORC / f"{name}.tbt.f32"), str(T_SEQ)], f"cf_seq_{name}.log", timeout=1800)
            summary[f"seq_{name}"] = parse_cf(o) or {"rc": rc, "tail": o[-1200:]}
            rc, o = stream([str(t4q / "build" / "cf_run"), "gen", model, str(WORK / f"{name}.i32"), str(N_GEN),
                            str(ORC / f"gen_{name}.gen.i32")], f"cf_gen_{name}.log", timeout=1800)
            summary[f"gen_{name}"] = parse_cf(o) or {"rc": rc, "tail": o[-1200:]}
            result(f"gate_{name}", summary[f"seq_{name}"] | summary[f"gen_{name}"])
        seq_pass = all(summary[f"seq_{n}"].get("pass") for n in PROMPTS)
        gen_pass = all(summary[f"gen_{n}"].get("pass") for n in PROMPTS)
        # the timed run: prompt + N_GEN greedy tokens, the steady tok/s
        rc, o = stream([str(t4q / "build" / "cf_run"), "time", model, str(WORK / "P0.i32"), str(N_GEN)],
                       "cf_time.log", timeout=1800)
        result("time", parse_cf(o) or {"rc": rc, "tail": o[-1200:]})
        # cf-m2 folded into the same quota-scarce round: the router census on natural text.
        # The oracle chatw tokenizes the wiki slice (the same text class the 27B rounds used),
        # then the census run dumps every layer's top-10 (id, renormed weight) per token.
        (WORK / "W.txt").write_text(WIKI)
        (WORK / "jobs2.txt").write_text(f"chatw W {WORK / 'W.txt'} {WORK / 'W.i32'}\n")
        rc, o = stream([str(t4q / "build" / "oracle_dump"), model, str(WORK / "jobs2.txt"), str(ORC)],
                       "oracle_census.log", timeout=600, env=env)
        if (WORK / "W.i32").exists():
            rc, o = stream([str(t4q / "build" / "cf_run"), "census", model, str(WORK / "W.i32"), "32",
                            str(WORK / "census.bin")], "cf_census.log", timeout=1800)
            cb = {}
            if (WORK / "census.bin").exists():
                shutil.copy2(WORK / "census.bin", OUT / "census.bin")
                cb["bin_bytes"] = (WORK / "census.bin").stat().st_size
            result("census", (parse_cf(o) or {"rc": rc}) | cb)
        else:
            result("census", {"error": "no W.i32", "tail": o[-800:]})
        result("summary", {"seq_pass": seq_pass, "gen_pass": gen_pass,
                           "seq": {k: v for k, v in summary.items() if k.startswith("seq")},
                           "gen": {k: v for k, v in summary.items() if k.startswith("gen")}})
    except Exception:  # noqa: BLE001
        import traceback
        result("fatal", traceback.format_exc()[-3000:])
    finally:
        if mon:
            mon.kill()
        print("RESULTS " + json.dumps(RESULTS, default=str)[:20000], flush=True)


if __name__ == "__main__":
    main()
