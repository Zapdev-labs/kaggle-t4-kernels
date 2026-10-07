#!/usr/bin/env python3
"""Pack t4q/ into a Kaggle script kernel: kaggle/<stage>/t4q-<stage>.py + kernel-metadata.json.

Usage: python3 t4q/tools/mkkernel.py <stage> [--sources t4-qwen38-baseline ...] [--no-sources]
       [--template <name>] [--cpu] [--define KEY=VAL ...]
The stage driver is t4q/tools/stage_<stage>.py (--template overrides the file name, for one
template generating several stage variants); it must contain the placeholder __T4Q_TGZ_B64__ (a
base64 tar.gz of t4q/, minus build outputs), which it unpacks to /tmp/t4q/src at runtime. Each
--define KEY=VAL substitutes __KEY__ in the template (the cfreq range split). --cpu generates a
CPU-only kernel (no GPU quota burned; enable_gpu false, no machine shape). No secrets are ever
packed.
"""
import base64
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
T4Q = ROOT / "t4q"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
SKIP_DIRS = {"build", "__pycache__", "out", ".git"}
SKIP_SUFFIX = {".o", ".so", ".gguf", ".npy", ".tgz"}


def pack() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in sorted(T4Q.rglob("*")):
            rel = p.relative_to(T4Q)
            if any(part in SKIP_DIRS for part in rel.parts) or p.suffix in SKIP_SUFFIX or not p.is_file():
                continue
            if p.stat().st_size > 4 << 20:
                continue
            tf.add(p, arcname=str(Path("t4q") / rel))
    return base64.b64encode(buf.getvalue()).decode()


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    stage = sys.argv[1]
    sources = ["t4-qwen38-baseline"]
    template = stage
    cpu = False
    defines = {}
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--no-sources":
            sources = []
        elif a == "--cpu":
            cpu = True
        elif a == "--template":
            i += 1
            template = args[i]
        elif a == "--define":
            i += 1
            k, _, v = args[i].partition("=")
            assert k and v, f"bad --define {args[i]!r}"
            defines[k] = v
        i += 1
    drv = T4Q / "tools" / f"stage_{template}.py"
    text = drv.read_text()
    assert "__T4Q_TGZ_B64__" in text, "driver lacks __T4Q_TGZ_B64__ placeholder"
    for k, v in defines.items():
        ph = f"__{k}__"
        assert ph in text, f"driver lacks the {ph} placeholder"
        text = text.replace(ph, v)
    import re
    left = re.findall(r"__[A-Z][A-Z0-9_]*__", text.replace("__T4Q_TGZ_B64__", ""))
    assert not left, f"unsubstituted placeholders remain: {left}"
    b64 = pack()
    out = ROOT / "kaggle" / stage
    out.mkdir(parents=True, exist_ok=True)
    code = f"t4q-{stage}.py"
    (out / code).write_text(text.replace("__T4Q_TGZ_B64__", b64))
    meta = {
        "id": f"otdoges/otdoges-t4q-{stage}", "title": f"otdoges/t4q-{stage}", "code_file": code, "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": not cpu, "enable_tpu": False,
        "enable_internet": True, "keywords": [], "dataset_sources": [], "kernel_sources": [f"otdoges/{s}" for s in sources],
        "competition_sources": [], "model_sources": [], "docker_image": DOCKER,
    }
    if not cpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out / code} ({len(b64) / 1e3:.0f} kB payload) and kernel-metadata.json "
          f"(sources={sources}, cpu={cpu}, defines={defines})")


if __name__ == "__main__":
    main()
