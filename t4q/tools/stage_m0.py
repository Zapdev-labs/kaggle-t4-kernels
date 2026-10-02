"""t4q stage m0: GEMV bench (P4/Q8/K6 dp4a, m=1..8, real TP shapes) + box probe (mailbox AR paths, NCCL, clocks).

Generated into kaggle/m0/t4q-m0.py by t4q/tools/mkkernel.py. No secrets, no model download.
Prints RESULT lines and a final RESULTS block; writes /kaggle/working/results.json and logs/.
"""
import base64
import glob
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

T0 = time.time()
DEADLINE = 40 * 60
OUT = Path("/kaggle/working")
LOGS = OUT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
W = Path("/tmp/t4q")
W.mkdir(parents=True, exist_ok=True)
RESULTS = {"stage": "m0"}
TGZ = "__T4Q_TGZ_B64__"
NVCC = "/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else (shutil.which("nvcc") or "nvcc")


def el():
    return round(time.time() - T0)


def log(*a):
    print(f"[{el():5d}s]", *a, flush=True)


def result(key, val):
    RESULTS[key] = val
    s = json.dumps({key: val})
    print("RESULT " + (s if len(s) < 4000 else s[:4000] + "..."), flush=True)
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=1))


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


def unpack():
    src = W / "src"
    shutil.rmtree(src, ignore_errors=True)
    src.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(TGZ)), mode="r:gz") as tf:
        tf.extractall(src)
    return src / "t4q"


def build(t4q):
    tools = t4q / "tools"
    bdir = W / "build"
    bdir.mkdir(exist_ok=True)
    flags = [NVCC, "-O3", "-std=c++17", "-arch=sm_75", "-lineinfo", "-Xptxas", "-v"]
    procs = []
    for p in range(6):
        cmd = flags + [f"-DPART={p}", "-c", str(tools / "gemv_bench.cu"), "-o", str(bdir / f"gb{p}.o")]
        procs.append((f"gb{p}", subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))
    procs.append(("probe", subprocess.Popen(flags + [str(tools / "probe.cu"), "-o", str(bdir / "probe"), "-ldl", "-lpthread"],
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))
    ok = True
    spills = {}
    for name, pr in procs:
        out, _ = pr.communicate(timeout=1500)
        (LOGS / f"build_{name}.txt").write_text(out)
        if pr.returncode:
            ok = False
            log(f"BUILD FAIL {name}\n{out[-3000:]}")
        # ptxas: "Function properties for <mangled>" ... "N bytes spill stores"
        sp = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", out)
        spills[name] = sum(1 for a, b in sp if int(a) or int(b))
    rc, o = sh(flags[:5] + [str(bdir / f"gb{p}.o") for p in range(6)] + ["-o", str(bdir / "gemv_bench"), "-ldl", "-lpthread"],
               timeout=600, logname="build_link.txt")
    ok = ok and rc == 0
    rc2, o2 = sh(f"g++ -O2 -std=c++17 {tools / 'gemv_layout_check.cpp'} -o {bdir / 'glc'} && {bdir / 'glc'}", timeout=600,
                 logname="layout_check.txt")
    result("build", {"ok": ok, "secs": el(), "kernels_with_spills": spills, "layout_check": o2.strip().splitlines()[-1:] })
    return bdir if ok else None


def parse_json_lines(out, prefix):
    res = []
    for line in out.splitlines():
        if line.startswith(prefix + " "):
            try:
                res.append(json.loads(line[len(prefix) + 1:]))
            except Exception:  # noqa: BLE001
                pass
    return res


def gemv(bdir):
    rc, o = sh([str(bdir / "gemv_bench"), "--dev", "0"], timeout=20 * 60, logname="gemv_bench.txt")
    checks = parse_json_lines(o, "CHECK")
    summ = parse_json_lines(o, "SUMMARY")
    rows = parse_json_lines(o, "R")
    fatal = [l for l in o.splitlines() if l.startswith("FATAL")]
    result("gemv_checks", checks)
    result("gemv_rc", {"rc": rc, "fatal": fatal, "n_rows": len(rows)})
    if summ:
        result("gemv_summary", summ[0])
    for l in o.splitlines():
        if l.startswith("BEST"):
            log(l)
    return summ[0] if summ else None, rows


def sustain(bdir, cfg):
    out = {}
    for shape, m in [("gateup_tp", 1), ("gateup_tp", 4), ("lmhead_tp_k6", 1)]:
        c = cfg if shape.startswith("gateup") else "2,4,0,256,1"
        procs = [subprocess.Popen([str(bdir / "gemv_bench"), "--dev", str(d), "--sustain", "10", "--shape", shape, "--cfg",
                                   f"{c},{m}"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for d in (0, 1)]
        for d, p in enumerate(procs):
            o, _ = p.communicate(timeout=120)
            (LOGS / f"sustain_{shape}_m{m}_dev{d}.txt").write_text(o)
            j = parse_json_lines(o, "SUSTAIN")
            out[f"{shape}_m{m}_dev{d}"] = j[0] if j else o[-500:]
    result("gemv_sustain_both_gpus", out)


def nccl_libs():
    libs = []
    for c in ["/usr/lib/x86_64-linux-gnu/libnccl.so.2", "/usr/lib/x86_64-linux-gnu/libnccl.so"]:
        if os.path.exists(c):
            libs.append(c)
            break
    rc, o = sh(["python", "-c", "import nvidia.nccl, os; print(os.path.dirname(nvidia.nccl.__file__) or list(nvidia.nccl.__path__)[0])"],
               timeout=120)
    if rc == 0:
        for cand in glob.glob(o.strip().splitlines()[-1] + "/lib/libnccl.so*"):
            libs.append(cand)
            break
    return libs


def probe(bdir):
    rc, o = sh([str(bdir / "probe")], timeout=10 * 60, logname="probe.txt")
    P = parse_json_lines(o, "P")
    result("probe", P)
    if "PROBE_DONE" not in o:
        result("probe_error", o[-2000:])
    nccl = []
    for lib in nccl_libs():
        for envs in [{}, {"NCCL_P2P_LEVEL": "SYS"}, {"NCCL_PROTO": "LL"}, {"NCCL_P2P_LEVEL": "SYS", "NCCL_PROTO": "LL"},
                     {"NCCL_P2P_DISABLE": "1", "NCCL_PROTO": "LL"}]:
            if time.time() - T0 > DEADLINE:
                break
            env = dict(os.environ, **envs)
            tag = os.path.basename(os.path.dirname(os.path.dirname(lib))) + "_" + "_".join(f"{k}={v}" for k, v in envs.items())
            rc, o = sh([str(bdir / "probe"), "nccl", lib], timeout=180, env=env, logname=f"nccl_{len(nccl)}.txt")
            for j in parse_json_lines(o, "P"):
                j["lib"] = lib
                nccl.append(j)
    result("nccl", nccl)
    return P, nccl


def choose_ar(P, nccl):
    cands = []
    for p in P:
        if p.get("test") == "mailbox" and p.get("mode") == "exchange" and p.get("bytes") == 20480 and p.get("read_all") == 1 \
                and p.get("blocks") == 40 and sum(p.get("errors", [1])) == 0:
            cands.append({"path": f"mailbox_{p['kind']}", "us_p50": p["us_p50"], "us_p99": p["us_p99"], "src": "exchange 20KB 40 blocks"})
    for n in nccl:
        if n.get("dtype") == "f32" and n.get("correct") == 1:
            cands.append({"path": "nccl", "env": n.get("env"), "lib": n.get("lib"), "us_p50": n["sync_us_p50"],
                          "us_p99": n["sync_us_p99"], "pipelined_us": n["pipelined_us"], "src": "host-synced allreduce"})
    cands.sort(key=lambda c: c["us_p50"])
    return {"chosen": cands[0] if cands else None, "candidates": cands}


def main():
    sh("nvidia-smi", timeout=60, logname="nvidia_smi.txt")
    mon = subprocess.Popen("nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,"
                           "clocks_throttle_reasons.active --format=csv -lms 500", shell=True,
                           stdout=open(LOGS / "smi_monitor.csv", "w"), stderr=subprocess.STDOUT)
    try:
        t4q = unpack()
        bdir = build(t4q)
        if not bdir:
            result("fatal", "build failed")
            return
        summ, rows = gemv(bdir)
        cfg = "2,4,0,256,1"
        if summ and summ.get("mix_single_cfg"):
            cfg = summ["mix_single_cfg"][0]["cfg"]
        sustain(bdir, cfg)
        P, nccl = probe(bdir)
        ar = choose_ar(P, nccl)
        result("allreduce_choice", ar)
        gate = {}
        if summ:
            p4 = [s for s in summ["shapes"] if s["fmt"] == "P4"]
            per = {s["shape"]: {f"m{b['m']}": b["GBps"] for b in s["best"]} for s in p4}
            gate["p4_best_per_shape"] = per
            gate["mix_best_per_shape"] = summ.get("mix_best_per_shape")
            gate["mix_single_cfg"] = summ.get("mix_single_cfg", [])[:1]
            mix = summ.get("mix_single_cfg", [{}])[0] if summ.get("mix_single_cfg") else {}
            gate["m1_single_cfg_mix"] = mix.get("m1")
            gate["m4_single_cfg_mix"] = mix.get("m4")
            gate["min_shape_m1"] = min((v.get("m1", 0) for v in per.values()), default=0)
            gate["min_shape_m4"] = min((v.get("m4", 0) for v in per.values()), default=0)
            gate["pass_mix"] = bool(mix.get("m1", 0) >= 250 and mix.get("m4", 0) >= 230)
            gate["all_checks_ok"] = bool(summ.get("all_ok"))
        gate["ar_chosen"] = bool(ar["chosen"])
        result("gate", gate)
    finally:
        mon.terminate()
        print("RESULTS_BEGIN")
        print(json.dumps({k: RESULTS.get(k) for k in ["build", "gemv_checks", "gemv_summary", "gemv_sustain_both_gpus",
                                                       "allreduce_choice", "gate"]}, indent=1)[:60000])
        print("RESULTS_END", flush=True)


main()
