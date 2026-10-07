"""t4q stage cfreq: the CYBER-FROST expert requant PACK (cf-m6 r3, CF_REQUANT.md sections 5+9).
Generated into kaggle/cfreqa|cfreqb/t4q-cfreq*.py by t4q/tools/mkkernel.py --template cfreq --define
CFREQ_LO/HI. No secrets; the source repo is public. CPU-ONLY (no GPU quota burned).
Flow: unpack the t4q tree -> build cf_requant_pack (g++ -fopenmp, host-only) -> fetch
model.safetensors.index.json -> per layer in [LO, HI): stream the layer's gate_up + down expert
tensors from the BF16 safetensors shards (HTTP range reads, 256 MB segments, retry + resume, the
producer one layer ahead) -> cf_requant_pack --gu --dn --out (the gu planes at the iq1_s 256-block
1.5625 bpw, the dn planes at the iq1_s-half 128-block 1.625 bpw - the 640-tiling the r3 gate caught)
-> --verify a 32-row sample against the raw sources (the deq32-vs-ref bit-identity + the src-RMSE
diag) -> delete the raws. The slab lands ATOMICALLY (.tmp + rename after the verify passes, so a
killed session never leaves a partial slab). Output: /kaggle/working/slabs/layer_{000..047,mtp}.bin
(~498 MB/layer) + manifest.json - the r4 loader's input. TWO pack kernels cover the 49 layers (the
~24.4 GB pool over the 20 GB kernel-output cap): cfreqa = [0,24), cfreqb = [24,49) incl. the MTP.
"""
import base64
import io
import json
import os
import queue
import shutil
import struct
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import requests

T0 = time.time()
DEADLINE = 11 * 3600  # the 12 h CPU kernel cap minus the flush margin
OUT = Path("/kaggle/working")
LOGS = OUT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
W = Path("/tmp/t4q")
W.mkdir(parents=True, exist_ok=True)
RAW = W / "raw"
RAW.mkdir(parents=True, exist_ok=True)
SLABS = OUT / "slabs"
SLABS.mkdir(parents=True, exist_ok=True)
TGZ = "__T4Q_TGZ_B64__"

LAYER_LO = __CFREQ_LO__
LAYER_HI = __CFREQ_HI__
RESULTS = {"stage": "cfreq", "lo": LAYER_LO, "hi": LAYER_HI}
BASE = "https://huggingface.co/Blackfrost-AI/CYBER-FROST-3.8-BF16/resolve/main/"
SEG = 1 << 28  # 256 MB range segments


def el():
    return round(time.time() - T0)


def log(*a):
    print(f"[{el():5d}s]", *a, flush=True)


def result(key, val):
    RESULTS[key] = val
    s = json.dumps({key: val}, default=str)
    print("RESULT " + (s if len(s) < 6000 else s[:6000] + "..."), flush=True)
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=1, default=str))


def sh(cmd, timeout=None, logname=None, cwd=None):
    t = time.time()
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout, cwd=cwd)
        out, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        def dec(x):
            return x.decode(errors="replace") if isinstance(x, bytes) else (x or "")
        out = dec(e.stdout) + dec(e.stderr) + f"\n<<TIMEOUT after {timeout}s>>"
        rc = -9
    if logname:
        (LOGS / f"{logname}.txt").write_text(out)
    dt = time.time() - t
    tail = [l for l in out.splitlines() if l.strip()][-4:]
    log(f"{' '.join(cmd[:3])}... rc={rc} {dt:.0f}s | " + " / ".join(tail))
    return rc, out


def unpack():
    src = W / "src"
    shutil.rmtree(src, ignore_errors=True)
    src.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(TGZ)), mode="r:gz") as tf:
        tf.extractall(src)
    return src / "t4q"


def layers():
    out = []
    for i in range(48):
        out.append({"idx": i, "kind": "main", "name": f"layer_{i:03d}",
                    "gu": f"model.language_model.layers.{i}.mlp.experts.gate_up_proj",
                    "dn": f"model.language_model.layers.{i}.mlp.experts.down_proj"})
    out.append({"idx": 48, "kind": "mtp", "name": "layer_mtp",
                "gu": "mtp.layers.0.mlp.experts.gate_up_proj",
                "dn": "mtp.layers.0.mlp.experts.down_proj"})
    return out[LAYER_LO:LAYER_HI]


# ---- the safetensors shard headers (one 8-B read + one header read per shard, cached) ----
_hdr = {}


def shard_entry(shard, name):
    if shard not in _hdr:
        for att in range(8):
            try:
                r = requests.get(BASE + shard, headers={"Range": "bytes=0-7"}, timeout=(30, 60))
                r.raise_for_status()
                n = struct.unpack("<Q", r.content)[0]
                assert 0 < n < (1 << 22), f"bad header len {n}"
                r = requests.get(BASE + shard, headers={"Range": f"bytes=8-{8 + n - 1}"}, timeout=(30, 120))
                r.raise_for_status()
                _hdr[shard] = (n, json.loads(r.content))
                break
            except Exception as e:
                log(f"header {shard} attempt {att + 1}: {e}")
                time.sleep(5 * (att + 1))
        else:
            raise RuntimeError(f"shard header failed: {shard}")
    n, hdr = _hdr[shard]
    return 8 + n, hdr[name]


def stream_tensor(wm, name, path):
    """Stream one tensor's bytes to path in SEG-byte range segments (retry + resume; the
    file position is tracked so a short/failed segment re-reads from its own start)."""
    shard = wm[name]
    base_off, ent = shard_entry(shard, name)
    lo, hi = ent["data_offsets"]
    total = hi - lo
    got = 0
    t0 = time.time()
    with open(path, "wb") as f:
        while got < total:
            seg = min(SEG, total - got)
            a = base_off + lo + got
            att = 0
            while True:
                att += 1
                try:
                    with requests.get(BASE + shard, headers={"Range": f"bytes={a}-{a + seg - 1}"},
                                      stream=True, timeout=(30, 300)) as r:
                        if r.status_code != 206:
                            raise RuntimeError(f"status {r.status_code} (Range ignored?)")
                        n = 0
                        for chunk in r.iter_content(1 << 22):
                            f.write(chunk)
                            n += len(chunk)
                    if n != seg:
                        raise RuntimeError(f"short segment {n}/{seg}")
                    break
                except Exception as e:
                    if att >= 10:
                        raise RuntimeError(f"{name}: segment at {got} failed after {att} attempts: {e}")
                    f.seek(got)
                    time.sleep(min(60, 5 * att))
            got += seg
    if got != total or os.path.getsize(path) != total:
        raise RuntimeError(f"{name}: size {os.path.getsize(path)} != {total}")
    mbps = total / 1e6 / max(0.1, time.time() - t0)
    log(f"  streamed {name} {total / 1e9:.2f} GB at {mbps:.0f} MB/s")
    return ent


def producer(wm, q, slots):
    try:
        for k, L in enumerate(LAYERS):
            s = k % 2
            slots[s].acquire()
            gu_p = RAW / f"slot{s}_gu.bf16"
            dn_p = RAW / f"slot{s}_dn.bf16"
            stream_tensor(wm, L["gu"], gu_p)
            stream_tensor(wm, L["dn"], dn_p)
            q.put({"ok": True, "L": L, "s": s, "gu": gu_p, "dn": dn_p})
        q.put({"ok": True, "end": True})
    except Exception as e:
        q.put({"ok": False, "err": repr(e)})


def pack_layer(PACK, L, gu_p, dn_p, wm):
    """One layer: pack to .tmp, verify a 32-row sample vs the raw sources, then the atomic
    rename. The byte-count check to the byte (the plane arithmetic the r3 gate pinned)."""
    _, ent_g = shard_entry(wm[L["gu"]], L["gu"])
    ne = ent_g["shape"][0]
    assert ent_g["dtype"] == "BF16" and list(ent_g["shape"][1:]) == [1280, 2560], ent_g
    _, ent_d = shard_entry(wm[L["dn"]], L["dn"])
    assert ent_d["dtype"] == "BF16" and list(ent_d["shape"][1:]) == [2560, 640], ent_d
    out = SLABS / f"{L['name']}.bin"
    tmp = SLABS / f"{L['name']}.bin.tmp"
    rc, _ = sh([str(PACK), "--gu", str(gu_p), "--dn", str(dn_p), "--out", str(tmp),
                "--ne", str(ne), "--threads", "4"], timeout=2 * 3600, logname=f"pack_{L['name']}")
    if rc != 0:
        raise RuntimeError(f"pack {L['name']} rc={rc} (logs/pack_{L['name']}.txt)")
    want = 96 + ne * 1280 * 500 + ne * 2560 * 130  # hdr + gu 50B/256 + dn 26B/128
    if tmp.stat().st_size != want:
        raise RuntimeError(f"{L['name']}: slab {tmp.stat().st_size} != {want}")
    rc, vout = sh([str(PACK), "--verify", str(tmp), "--rows", "32",
                   "--verify-gu", str(gu_p), "--verify-dn", str(dn_p)],
                  timeout=1800, logname=f"verify_{L['name']}")
    vline = [l for l in vout.splitlines() if l.startswith("verify")][-1]
    if rc != 0 or " OK" not in vline:
        raise RuntimeError(f"verify {L['name']} rc={rc}: {vline}")
    os.replace(tmp, out)
    return {"file": out.name, "idx": L["idx"], "kind": L["kind"], "ne": ne, "bytes": want, "verify": vline}


def main():
    global LAYERS
    src = unpack()
    PACK = src / "build" / "cf_requant_pack"
    rc, _ = sh(["make", "build/cf_requant_pack"], cwd=str(src), timeout=600, logname="build")
    if rc != 0 or not PACK.exists():
        raise RuntimeError("cf_requant_pack build failed (logs/build.txt)")
    log(f"packer built, layers [{LAYER_LO}, {LAYER_HI})")

    idx = requests.get(BASE + "model.safetensors.index.json", timeout=(30, 300)).json()
    wm = idx["weight_map"]
    LAYERS = layers()
    log(f"{len(LAYERS)} layers to pack, {sum(1 for _ in LAYERS) * 498} MB class output")

    q = queue.Queue(maxsize=4)
    slots = [threading.Semaphore(1) for _ in range(2)]
    threading.Thread(target=producer, args=(wm, q, slots), daemon=True).start()

    done = []
    while True:
        item = q.get()
        if not item["ok"]:
            raise RuntimeError(f"producer failed: {item['err']}")
        if item.get("end"):
            break
        L, s = item["L"], item["s"]
        t0 = time.time()
        rec = pack_layer(PACK, L, item["gu"], item["dn"], wm)
        os.remove(item["gu"])
        os.remove(item["dn"])
        slots[s].release()
        rec["sec"] = round(time.time() - t0)
        done.append(rec)
        result("slab_" + L["name"], f"{rec['bytes']} B in {rec['sec']}s, {rec['verify']}")
        if item["gu"].exists() or item["dn"].exists():
            raise RuntimeError("raw cleanup failed")

    manifest = {
        "version": 2, "lo": LAYER_LO, "hi": LAYER_HI, "total_layers": 49,
        "gu_fmt": "FMT_IQ1S (iq1_s 256-elem block, 50 B/block = 1.5625 bpw)",
        "dn_fmt": "FMT_IQ1SH (the r3 half-block, 128-elem, 26 B/block = 1.625 bpw; the dn 640 tiling)",
        "slab_bytes_formula": "96 + ne*1280*500 + ne*2560*130",
        "layers": done,
    }
    (SLABS / "manifest.json").write_text(json.dumps(manifest, indent=1))
    gib = sum(r["bytes"] for r in done) / (1 << 30)
    result("manifest", f"{len(done)} slabs, {gib:.2f} GiB, all verified")
    result("done", "PACK COMPLETE")


def watchdog():
    """r19b lesson: a child that hangs silently blocks stream()'s readline. At
    DEADLINE-120 it flags, at DEADLINE it flushes and exits."""
    time.sleep(max(60, DEADLINE - 120))
    result("watchdog", "deadline approaching in 120s")
    time.sleep(120)
    result("watchdog", "deadline hit - flushing and exiting; completed slabs are whole (the .tmp rename)")
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=1, default=str))
    os._exit(0)


def spawn_watchdog_proc():
    """r19f lesson: the GIL-frozen main (a non-GIL-releasing syscall) takes the thread
    watchdog with it - this separate PROCESS cannot be frozen. At the deadline it kills
    the parent so the incremental results.json + logs + the completed slabs survive."""
    ppid = os.getpid()
    src = (
        "import os, signal, time\n"
        f"ppid = {ppid}\n"
        f"deadline = {DEADLINE}\n"
        "while True:\n"
        "    time.sleep(30)\n"
        "    try:\n"
        "        os.kill(ppid, 0)\n"
        "    except OSError:\n"
        "        os._exit(0)\n"
        "    if time.time() - START > deadline:\n"
        "        break\n"
        "os.kill(ppid, signal.SIGKILL)\n"
    )
    env = dict(os.environ, START=str(time.time()))
    subprocess.Popen([sys.executable, "-c", src], env=env, start_new_session=True)


if __name__ == "__main__":
    spawn_watchdog_proc()
    threading.Thread(target=watchdog, daemon=True).start()
    try:
        main()
        result("status", "OK")
    except Exception as e:
        import traceback
        traceback.print_exc()
        result("status", f"FAILED: {e!r}")
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=1, default=str))
