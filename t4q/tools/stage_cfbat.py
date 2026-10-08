"""t4q stage cfbat: the cf-m6 requant L4 BATTERY (r4's value gates + r5's wall measure, over the cfreq slabs).

Generated into kaggle/cfbat/t4q-cfbat.py by t4q/tools/mkkernel.py. No secrets; the download is public.
Sources: the baseline kernel (libllama for the tokenizer) + the two cfreq PACK kernels (their
/kaggle/working/slabs outputs - layer_000..047.bin + layer_mtp.bin + manifest.json, the r4 loader's
T4Q_CF_IQSLAB input).
Flow: unpack -> (download the 82.85 GB Q2_K_S GGUF in a thread, ~6+ min write-bound) -> build
libt4q + cf_run + oracle_dump -> the SLAB MERGE (symlink both kernels' layer_*.bin into one dir +
the manifest validation + the real-RMSE stats) -> the tokenizer (oracle chatw on the baked prompt)
-> THE PHASES, priority-ordered so a deadline cut loses the least:
  (1) the resident SMOKE + the VRAM CEILING: the IQN=__IQN_HI__ probe - the loader's throw
      carries the free-GiB number (the ceiling = floor(free/0.4641)), a SUCCESS extends the sweep;
  (2) the trunk BASELINE time (the Q2_K_S reference tok/s);
  (3) the IQN SWEEP time (the requant-mix rate curve);
  (4) the QUALITY A/B: gen __N_GEN__ tokens at IQN=0 vs the best IQN, the CF_GEN streams
      compared by the driver (the greedy agreement - the honest requant-quality gate);
  (5) the MTP battery (T4Q_CF_MTP=1, k=__K__): the draft alpha1 smoke (IQN=0), the verify
      BYTE-MATCH at the covered layers (THE r4 MTP-consistency bar), and the spec round
      timers at IQN=0 vs IQN=__IQN_AB__ - THE r5 MEASURE: verify_ms before/after the
      amortized dots, plus mean_union (the real pick overlap the amortization rides).
-> RESULTS.
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
from pathlib import Path

T0 = time.time()
DEADLINE = __DEADLINE_H__ * 3600
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
RESULTS = {"stage": "cfbat"}
TGZ = "__T4Q_TGZ_B64__"
REPO = "freakyskittle/CYBER-FROST-3.8-GGUF"
GGUF = "CYBER-FROST-3.8-Q2_K_S.gguf"
SLABS = W / "slabs"          # the merged dir the loader reads (T4Q_CF_IQSLAB)
IQN_SWEEP = __IQNS__          # the rate curve (the ceiling probe may extend it)
IQN_HI = __IQN_HI__           # the probe (the ceiling finder)
IQN_AB = __IQN_AB__           # the A/B + the MTP wall's covered-layer count
N_GEN = __N_GEN__             # the greedy agreement length
N_TIME = __N_TIME__           # the timed runs' greedy length
K_DRAFT = __K__               # T4Q_CF_K (nr = k+1 verify rows)
PROMPT = "Write a one-line Python lambda that reverses a string."
PER_GIB = 0.4641              # the per-layer resident need (498,073,600 B, the loader's own arithmetic)


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
    # r19m (the v7 lesson): NO size probe on the fresh 82.85 GB file, in ANY form (the stat
    # D-locks; the hf/curl rc already certifies the transfer; the gates open the file themselves)
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
        return dst, [f.name for f in dst.iterdir()]
    # no libllama mounted: the tokenizer falls back to a raw id prompt (the A/B still compares
    # like-for-like - the SAME ids both sides; the note lands in the results)
    return None, []


def parse_cf(out):
    for ln in out.splitlines():
        if ln.startswith("CF {"):
            try:
                return json.loads(ln[3:])
            except Exception:
                pass
    return None


def parse_gen_stream(logname):
    """the CF_GEN t4q: line from a gen run's log (cf-m6 r5: always printed without an oracle)"""
    try:
        for ln in (LOGS / logname).read_text().splitlines():
            if ln.startswith("CF_GEN t4q:"):
                return [int(x) for x in ln.split(":")[1].split()]
    except OSError:
        pass
    return None


def merge_slabs():
    """the two cfreq kernels' outputs -> ONE dir of layer_*.bin symlinks (the loader reads
    layer_%03d.bin contiguous; a 24 GB copy is too slow, symlinks are instant) + the
    manifest validation + the REAL per-layer RMSE stats (the pack's own verify lines)."""
    found = sorted(glob.glob("/kaggle/input/**/slabs/layer_*.bin", recursive=True))
    mans = sorted(glob.glob("/kaggle/input/**/slabs/manifest.json", recursive=True))
    shutil.rmtree(SLABS, ignore_errors=True)
    SLABS.mkdir(parents=True)
    for f in found:
        os.symlink(os.path.realpath(f), SLABS / Path(f).name)
    names = sorted(p.name for p in SLABS.iterdir() if p.name.endswith(".bin"))
    main = [n for n in names if n != "layer_mtp.bin"]
    ok = len(main) >= 24 and main == [f"layer_{i:03d}.bin" for i in range(len(main))]
    stats = {"ok": ok, "files": len(names), "main": len(main), "mtp": "layer_mtp.bin" in names,
             "head": names[:2], "tail": names[-2:]}
    rmse = []
    for m in mans:
        try:
            man = json.loads(Path(m).read_text())
            stats.setdefault("manifests", []).append({"lo": man.get("lo"), "hi": man.get("hi"),
                                                      "layers": len(man.get("layers", []))})
            for rec in man.get("layers", []):
                mo = re.search(r"rel=([0-9.]+)", rec.get("verify", ""))
                if mo:
                    rmse.append(float(mo.group(1)))
        except Exception as e:
            stats.setdefault("manifests", []).append({"error": repr(e)})
    if rmse:
        stats["rmse_rel"] = {"n": len(rmse), "mean": round(sum(rmse) / len(rmse), 5),
                             "max": round(max(rmse), 5)}
    result("slabs", stats)
    return ok


def cfrun(mode, args, logname, iqn=None, mtp=False, timeout=5400):
    env = dict(os.environ)
    if iqn is not None:
        env["T4Q_CF_IQSLAB"] = str(SLABS)
        env["T4Q_CF_IQN"] = str(iqn)
    if mtp:
        env["T4Q_CF_MTP"] = "1"
        env["T4Q_CF_K"] = str(K_DRAFT)
    cf = W / "src" / "t4q" / "build" / "cf_run"
    model = DL.get("path")
    rc, o = stream([str(cf), mode, model] + [str(a) for a in args], logname, timeout, env=env)
    return rc, o, parse_cf(o)


def ceiling_from(o):
    """the loader's throw carries the free number: 'the iq1_s tier needs X GiB resident,
    only Y GiB free' -> the max IQN (floor(Y/PER_GIB)); a SUCCESS means the probe fits"""
    mo = re.search(r"only ([0-9.]+) GiB free", o)
    if mo:
        return int(float(mo.group(1)) / PER_GIB)
    return None


def watchdog():
    """r19b lesson: a child that hangs silently blocks stream()'s readline. This thread
    owns that case (the GIL is free when a CHILD hangs): at DEADLINE-120 it flags, at
    DEADLINE it flushes and exits. The GIL-FROZEN case is owned by the process watchdog."""
    time.sleep(max(60, DEADLINE - 120))
    result("watchdog", "deadline approaching in 120s")
    time.sleep(120)
    result("watchdog", "deadline hit - flushing and exiting; see logs/ for where each run stopped")
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=1, default=str))
    os._exit(0)


def spawn_watchdog_proc():
    """r19f lesson (the v6 post-mortem): the wedge froze the MAIN inside os.stat on the
    fresh 82.85 GB file - os.stat does NOT release the GIL, so the thread watchdog above
    froze with it. This one is a separate PROCESS: it cannot be frozen by the parent. At
    the deadline it kills the (possibly frozen) parent by its baked-in pid (KILL FIRST,
    no disk writes before it - the v7 lesson) so the session ends cleanly and the
    incremental results.json + the logs become the fetchable output."""
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
        "# r19m (the v7 lesson): KILL FIRST, no disk writes before it\n"
        "try:\n"
        "    os.kill(ppid, signal.SIGKILL)\n"
        "except (ProcessLookupError, PermissionError):\n"
        "    pass\n"
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
        # thread starts (the /tmp overlayfs is healthy in this window)
        (WORK / "P.txt").write_text(PROMPT)
        (WORK / "jobs.txt").write_text(f"chatw P {WORK / 'P.txt'} {WORK / 'P.i32'}\n")
        th = threading.Thread(target=downloader, daemon=True)
        th.start()
        rc, o = sh(f"make -C {t4q} -j4", timeout=1800, logname="build.txt")
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
        if not merge_slabs():
            result("fatal", "the cfreq slabs are not mounted/complete (both pack kernels must have run first)")
            return
        llb, files = setup_llama()
        result("llama_libs", files)
        if llb:  # the tokenizer's oracle (the cf1 form: the baseline source's libllama)
            rc, o = sh(f"make -C {t4q} build/oracle_dump LLAMA_LIB={llb}", timeout=600, logname="build_oracle.txt")
            result("build_oracle", {"ok": rc == 0, "tail": o[-1500:] if rc else ""})
            if rc:
                llb = None  # the raw-id fallback below
        th.join(timeout=2400)
        model = DL.get("path")
        if not model:
            result("fatal", "download failed")
            return
        result("model", {"path": model, "gb": DL.get("gb")})
        # the tokenizer: the oracle chatw (the GGUF chat template + tokenize - the same class
        # every prior round used); the no-llama fallback bakes a raw id prompt
        ids = WORK / "P.i32"
        if llb:
            env = dict(os.environ, LD_LIBRARY_PATH=f"{llb}:/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", ""),
                       T4Q_ORACLE_NGPU="0", CUDA_VISIBLE_DEVICES="")
            rc, o = stream([str(t4q / "build" / "oracle_dump"), model, str(WORK / "jobs.txt"), str(ORC)],
                           "oracle_tok.log", timeout=900, env=env)
            result("tokenizer", {"ok": ids.exists(), "rc": rc})
        if not ids.exists():
            ids.write_bytes(b"".join((1000 + i * 7).to_bytes(4, "little") for i in range(24)))
            result("tokenizer", {"ok": True, "fallback": "raw ids (no libllama mounted)"})
        # ---- phase 1: the resident SMOKE + the VRAM CEILING (the IQN_HI probe) ----
        rc, o, cf = cfrun("time", [ids, N_TIME], f"smoke_iqn{IQN_HI}.log", iqn=IQN_HI)
        ceil = ceiling_from(o)
        result("smoke", {"iqn": IQN_HI, "rc": rc, "cf": cf,
                         "ceiling": ceil if ceil is not None else ("fits" if rc == 0 else None)})
        sweep = list(IQN_SWEEP)
        if rc == 0:  # the probe FIT: extend the sweep past it
            sweep = sorted(set(sweep) | {IQN_HI})
        # ---- phase 2: the trunk BASELINE (the Q2_K_S reference) ----
        rc, o, cf = cfrun("time", [ids, N_TIME], "time_base.log", iqn=None)
        result("time_base", {"rc": rc, "cf": cf})
        # ---- phase 3: the IQN SWEEP (the requant-mix rate curve) ----
        curve = {}
        smoke_cf = RESULTS.get("smoke", {}).get("cf")
        for n in sweep:
            if n == IQN_HI and smoke_cf and smoke_cf.get("mode") == "time":
                curve[n] = smoke_cf  # the probe already measured it (and it fit)
                continue
            rc, o, cf = cfrun("time", [ids, N_TIME], f"time_iqn{n}.log", iqn=n)
            curve[n] = cf or {"rc": rc}
        result("sweep", curve)
        # ---- phase 4: the QUALITY A/B (the greedy agreement, the same prompt both sides) ----
        rc, o, cf = cfrun("gen", [ids, N_GEN], "gen_base.log", iqn=None)
        result("gen_base", {"rc": rc, "cf": cf})
        rc, o, cf = cfrun("gen", [ids, N_GEN], f"gen_iqn{IQN_AB}.log", iqn=IQN_AB)
        result("gen_ab", {"rc": rc, "cf": cf})
        a = parse_gen_stream("gen_base.log")
        b = parse_gen_stream(f"gen_iqn{IQN_AB}.log")
        if a and b:
            m = min(len(a), len(b))
            agree = sum(x == y for x, y in zip(a[:m], b[:m]))
            first = next((i for i in range(m) if a[i] != b[i]), -1)
            result("agreement", {"n": m, "agree": agree, "first_diff": first,
                                 "tok_per": round(agree / m, 4), "same": first < 0})
        else:
            result("agreement", {"error": "no CF_GEN stream", "a": bool(a), "b": bool(b)})
        # ---- phase 5: the MTP battery (the alpha1 smoke, the verify byte-match, the r5 wall) ----
        rc, o, cf = cfrun("draft", [ids, 8], "draft_base.log", iqn=None, mtp=True)
        result("draft_base", {"rc": rc, "cf": cf})
        rc, o, cf = cfrun("verify", [ids, 8], f"verify_iqn{IQN_AB}.log", iqn=IQN_AB, mtp=True)
        result("verify_ab", {"rc": rc, "cf": cf})  # THE r4 MTP-consistency bar (byte_match at the covered layers)
        rc, o, cf = cfrun("spec", [ids, 24], "spec_base.log", iqn=None, mtp=True)
        result("spec_base", {"rc": rc, "cf": cf})
        rc, o, cf = cfrun("spec", [ids, 24], f"spec_iqn{IQN_AB}.log", iqn=IQN_AB, mtp=True)
        result("spec_ab", {"rc": rc, "cf": cf})  # THE r5 wall: verify_ms + mean_union vs spec_base
        wall = {}
        for tag in ("base", "ab"):
            d = RESULTS.get(f"spec_{tag}", {}).get("cf") or {}
            if d:
                wall[tag] = {k: d.get(k) for k in ("verify_ms", "draft_ms", "catch_ms", "spec_ms_per_tok",
                                                   "seq_ms", "mean_union", "tok_per_round")}
        result("r5_wall", wall)
        result("summary", {"sweep": {k: (v or {}).get("gen_ms_per_tok") for k, v in curve.items()},
                           "agreement": RESULTS.get("agreement"), "r5_wall": wall})
    except Exception:  # noqa: BLE001
        import traceback
        result("fatal", traceback.format_exc()[-3000:])
    finally:
        if mon:
            mon.kill()
        print("RESULTS " + json.dumps(RESULTS, default=str)[:20000], flush=True)


if __name__ == "__main__":
    main()
