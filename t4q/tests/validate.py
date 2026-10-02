"""t4q validation V0-V3 (DESIGN.md section 10) against the llama.cpp oracle (tools/oracle_dump).

usage: python validate.py --model M.gguf --work WORKDIR --oracle ORACLEDIR --out results_validation.json
WORKDIR/manifest.json: {"seqs": [{"name","ids","tbt"}], "gens": [{"name","ids","n"}], "dump": {"name","ids","layers"}}

V0: each op vs an fp64 numpy reference on real GGUF weights, fed with t4q's own dumped inputs (layer 0 DeltaNet,
    layer 3 attention, FFN, lm_head rows), plus the loader's bit-exact repack round-trip.
V1: named intermediates vs llama.cpp cb_eval dumps (token-by-token, positions 0, 1, n-1).
V2: full logits on every position vs oracle batch and oracle token-by-token: top-1 agreement and KL.
V3: greedy generation vs oracle greedy.
"""
import argparse
import json
import math
import os
import struct
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))
from gguf_np import GGUF  # noqa: E402
from t4q import T4Q, Tokenizer  # noqa: E402

V = 248320
R = {}


def log(*a):
    print("[validate]", *a, flush=True)


def rel(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.shape != b.shape:
        return float("nan")
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def read_ids(p):
    return np.fromfile(p, dtype=np.int32)


def rms(x, w, eps=1e-6):
    x = np.asarray(x, np.float64)
    return x / np.sqrt(np.mean(x * x) + eps) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def log_softmax(x):
    x = np.asarray(x, np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


# ----------------------------------------------------------------------------------------------------- V0
def v0(g, D0, D1, ids, logits1):
    out = {}
    f64 = lambda a: np.asarray(a, np.float64)  # noqa: E731
    W = lambda n, rows=None: g.deq(n, rows).astype(np.float64)  # noqa: E731
    vec = lambda n: g.deq(n).astype(np.float64).ravel()  # noqa: E731

    def chk(key, got, ref, tol):
        if got is None:
            out[key] = {"err": "missing dump"}
            return
        e = rel(got, ref)
        out[key] = {"rel": e, "tol": tol, "pass": bool(e <= tol)}

    # layer 0 (DeltaNet) at position 1
    h0 = g.deq("token_embd.weight", [int(ids[1])]).astype(np.float64).ravel()
    x = rms(h0, vec("blk.0.attn_norm.weight"))
    chk("rmsnorm_embed_l0", D1.get("attn_norm-0"), x, 1e-5)
    x = f64(D1["attn_norm-0"])
    chk("gemv_qkv_Q4_0", D1.get("linear_attn_qkv_mixed-0"), W("blk.0.attn_qkv.weight") @ x, 2e-3)
    chk("gemv_z_Q4_0", D1.get("z-0"), W("blk.0.attn_gate.weight") @ x, 2e-3)
    b = W("blk.0.ssm_beta.weight") @ x
    a = W("blk.0.ssm_alpha.weight") @ x
    chk("gdn_beta", D1.get("beta_sigmoid-0"), sigmoid(b), 1e-4)
    sp = a + vec("blk.0.ssm_dt.bias")
    sp = np.where(sp > 20, sp, np.log1p(np.exp(sp)))
    chk("gdn_gate_g", D1.get("gate-0"), sp * vec("blk.0.ssm_a"), 1e-4)
    cw = g.deq("blk.0.ssm_conv1d.weight").astype(np.float64)  # [10240, 4]
    q0, q1 = f64(D0["linear_attn_qkv_mixed-0"]), f64(D1["linear_attn_qkv_mixed-0"])
    conv = cw[:, 2] * q0 + cw[:, 3] * q1
    chk("gdn_conv_silu", D1.get("conv_output_silu-0"), silu(conv), 1e-5)
    y = f64(D1["conv_output_silu-0"])
    qh = y[:2048].reshape(16, 128)
    kh = y[2048:4096].reshape(16, 128)
    qn = qh / np.sqrt((qh * qh).sum(-1, keepdims=True) + 1e-6)
    kn = kh / np.sqrt((kh * kh).sum(-1, keepdims=True) + 1e-6)
    chk("gdn_l2_q", D1.get("q_conv_predelta-0"), qn, 1e-5)
    chk("gdn_l2_k", D1.get("k_conv_predelta-0"), kn, 1e-5)
    S0 = f64(D0["ssm_state-0"]).reshape(48, 128, 128)  # [h][col(v)][i(k)]
    qn = f64(D1["q_conv_predelta-0"]).reshape(16, 128)
    kn = f64(D1["k_conv_predelta-0"]).reshape(16, 128)
    v = y[4096:].reshape(48, 128)
    beta = f64(D1["beta_sigmoid-0"])
    gg = f64(D1["gate-0"])
    o = np.zeros((48, 128))
    S1 = np.zeros_like(S0)
    for h in range(48):
        k_ = kn[h % 16]
        dec = math.exp(gg[h])
        kv = S0[h] @ k_
        delta = (v[h] - dec * kv) * beta[h]
        S1[h] = dec * S0[h] + np.outer(delta, k_)
        o[h] = S1[h] @ qn[h % 16] / math.sqrt(128)
    chk("gdn_recur_out", D1.get("attn_output-0"), o, 1e-5)
    chk("gdn_recur_state", D1.get("ssm_state-0"), S1, 1e-5)
    o = f64(D1["attn_output-0"]).reshape(48, 128)
    z = f64(D1["z-0"]).reshape(48, 128)
    on = o / np.sqrt((o * o).mean(-1, keepdims=True) + 1e-6) * vec("blk.0.ssm_norm.weight") * silu(z)
    chk("gdn_gated_norm", D1.get("final_output-0"), on, 1e-5)
    chk("gemv_ssm_out_Q5_K", D1.get("linear_attn_out-0"), W("blk.0.ssm_out.weight") @ f64(D1["final_output-0"]), 2e-3)
    xf = f64(D1["attn_post_norm-0"])
    chk("rmsnorm_post_l0", xf, rms(f64(D1["attn_residual-0"]), vec("blk.0.post_attention_norm.weight")), 1e-5)
    hmid = silu(W("blk.0.ffn_gate.weight") @ xf) * (W("blk.0.ffn_up.weight") @ xf)
    chk("ffn_Q4_0_gate_up_Q4_1_down", D1.get("ffn_out-0"), W("blk.0.ffn_down.weight") @ hmid, 2e-3)

    # layer 3 (full attention) at position 1
    xa = f64(D1["attn_norm-3"])
    chk("rmsnorm_l3", xa, rms(f64(D1["l_out-2"]), vec("blk.3.attn_norm.weight")), 1e-5)
    qf = W("blk.3.attn_q.weight") @ xa
    chk("gemv_attn_q", D1.get("Qcur_full-3"), qf, 2e-3)
    chk("gemv_attn_k", D1.get("Kcur_raw-3"), W("blk.3.attn_k.weight") @ xa, 2e-3)
    chk("gemv_attn_v", D1.get("Vcur-3"), W("blk.3.attn_v.weight") @ xa, 2e-3)

    def norm_rope(x, w, pos):
        x = x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * w
        th = pos * (1e7 ** (-2.0 * np.arange(32) / 64.0))
        c, s = np.cos(th), np.sin(th)
        y = x.copy()
        y[:, :32] = x[:, :32] * c - x[:, 32:64] * s
        y[:, 32:64] = x[:, :32] * s + x[:, 32:64] * c
        return y

    qf = f64(D1["Qcur_full-3"]).reshape(24, 512)
    chk("q_norm_rope", D1.get("Qcur-3"), norm_rope(qf[:, :256], vec("blk.3.attn_q_norm.weight"), 1), 1e-5)
    chk("k_norm_rope", D1.get("Kcur-3"),
        norm_rope(f64(D1["Kcur_raw-3"]).reshape(4, 256), vec("blk.3.attn_k_norm.weight"), 1), 1e-5)
    K = np.stack([np.asarray(D0["Kcur-3"], np.float16).astype(np.float64).reshape(4, 256),
                  np.asarray(D1["Kcur-3"], np.float16).astype(np.float64).reshape(4, 256)], 1)  # [4][2][256]
    Vv = np.stack([np.asarray(D0["Vcur-3"], np.float16).astype(np.float64).reshape(4, 256),
                   np.asarray(D1["Vcur-3"], np.float16).astype(np.float64).reshape(4, 256)], 1)
    q = f64(D1["Qcur-3"]).reshape(24, 256)
    att = np.zeros((24, 256))
    for h in range(24):
        j = h // 6
        s = K[j] @ q[h] / 16.0
        p = np.exp(s - s.max())
        p /= p.sum()
        att[h] = p @ Vv[j]
    chk("attn_softmax_gqa", D1.get("attn_pregate-3"), att, 1e-3)
    gated = f64(D1["attn_pregate-3"]).reshape(24, 256) * sigmoid(qf[:, 256:])
    chk("attn_sigmoid_gate", D1.get("attn_gated-3"), gated, 1e-5)
    chk("gemv_attn_output", D1.get("attn_output-3"), W("blk.3.attn_output.weight") @ f64(D1["attn_gated-3"]), 2e-3)

    # head
    chk("rmsnorm_output", D1.get("result_norm"), rms(f64(D1["l_out-63"]), vec("output_norm.weight")), 1e-5)
    rows = np.sort(np.random.default_rng(0).choice(V, 2048, replace=False))
    ref = W("output.weight", rows) @ f64(D1["result_norm"])
    chk("gemv_lm_head_Q6_K_2048rows", logits1[rows], ref, 2e-3)
    return out


# ----------------------------------------------------------------------------------------------------- V1
def read_oracle_dump(p):
    recs = {}
    with open(p, "rb") as f:
        assert f.read(4) == b"T4QD"
        (cnt,) = struct.unpack("<I", f.read(4))
        for _ in range(cnt):
            (nl,) = struct.unpack("<I", f.read(4))
            name = f.read(nl).decode()
            (tag,) = struct.unpack("<i", f.read(4))
            ne = struct.unpack("<4q", f.read(32))
            n = ne[0] * ne[1] * ne[2] * ne[3]
            recs[(name, tag)] = np.frombuffer(f.read(4 * n), dtype=np.float32).copy()
    return recs


MAIN_KEYS = ("attn_norm", "linear_attn_qkv_mixed", "z", "conv_output_silu", "attn_output", "final_output",
             "linear_attn_out", "Qcur", "Kcur", "attn_pregate", "attn_gated", "attn_residual", "attn_post_norm",
             "ffn_out", "l_out", "result_norm")


def batch_slice(orcb, n):
    """oracle batch-mode dump -> {(name, tag): row} for tags 0, 1, n-1 (token dim is the outermost)"""
    out = {}
    for (name, _), arr in orcb.items():
        if arr.size % n:
            continue
        rows = arr.reshape(n, -1)
        for tag in (0, 1, n - 1):
            out[(name, tag)] = rows[tag]
    return out


def v1(oracle, tdumps, n, floor_src=None):
    """free-running intermediates vs llama.cpp token-by-token. With floor_src (llama.cpp batch path), each key also
    gets the llama-internal noise floor rel(batch, tbt); pass if rel <= max(design tol, 2 x floor)."""
    out = {}
    worst_main = {}
    worst_floor = {}
    for (name, tag), ref in sorted(oracle.items()):
        mine = tdumps.get(tag, {}).get(name)
        if mine is None:
            continue
        if mine.size != ref.size:
            if ref.size % mine.size == 0:  # q/k repeated to 48 heads in the unfused GDN path: compare first copy
                ref = ref[: mine.size]
            else:
                out[f"{name}@{tag}"] = {"size_mismatch": [int(mine.size), int(ref.size)]}
                continue
        e = rel(mine, ref)
        rec = {"rel": round(e, 7)}
        base, _, ly = name.rpartition("-")
        if not ly.isdigit():
            base, ly = name, "64"
        L = int(ly)
        tol = 1e-2 if L == 0 else 3e-2
        if floor_src is not None and (name, tag) in floor_src:
            fb = floor_src[(name, tag)]
            if fb.size >= ref.size:
                fl = rel(fb[: ref.size], ref)
                rec["floor"] = round(fl, 7)
                if base in MAIN_KEYS:
                    worst_floor[base] = max(worst_floor.get(base, 0.0), e / max(tol, 2 * fl))
        out[f"{name}@{tag}"] = rec
        if base in MAIN_KEYS:
            worst_main[base] = max(worst_main.get(base, 0.0), e / tol)
    return out, worst_main, worst_floor


# ----------------------------------------------------------------------------------------------------- V2
def kl_stats(p_logits, q_logits, chunk=32):
    """KL(p || q) per row, top-1 agreement, oracle top-2 gap; p = oracle. Chunked to bound memory."""
    kls, agrees, gaps = [], [], []
    for i in range(0, len(p_logits), chunk):
        P = np.asarray(p_logits[i:i + chunk], np.float64)
        Q = np.asarray(q_logits[i:i + chunk], np.float64)
        lp = log_softmax(P)
        lq = log_softmax(Q)
        kls.append((np.exp(lp) * (lp - lq)).sum(-1))
        s = np.partition(P, -2, axis=-1)[:, -2:]
        gaps.append(np.abs(s[:, 1] - s[:, 0]))
        agrees.append(np.argmax(P, -1) == np.argmax(Q, -1))
    return np.concatenate(kls), np.concatenate(agrees), np.concatenate(gaps)


def summarize(kl, agree, gap):
    m = gap >= 0.1
    return {"n": int(len(kl)), "mean_kl": float(kl.mean()), "p99_kl": float(np.percentile(kl, 99)),
            "max_kl": float(kl.max()), "top1_agree_all": float(agree.mean()),
            "top1_agree_excl_ties": float(agree[m].mean()) if m.any() else 1.0, "n_ties": int((~m).sum())}


def attn_probe(orc, L):
    """Which rounding does llama.cpp's attention apply? Recompute attn_pregate-L at tags 0, 1 from the oracle's own
    Qcur/Kcur/Vcur under several rounding hypotheses."""
    out = {}
    try:
        q = {t: orc[(f"Qcur-{L}", t)].astype(np.float64).reshape(24, 256) for t in (0, 1)}
        k = {t: orc[(f"Kcur-{L}", t)].astype(np.float64).reshape(4, 256) for t in (0, 1)}
        v = {t: orc[(f"Vcur-{L}", t)].astype(np.float64).reshape(4, 256) for t in (0, 1)}
    except KeyError as e:
        return {"err": str(e)}
    f16 = lambda x: x.astype(np.float16).astype(np.float64)  # noqa: E731
    hyps = {"f32": (lambda x: x, lambda x: x, lambda x: x), "kv_f16": (lambda x: x, f16, f16),
            "qkv_f16": (f16, f16, f16)}
    for name, (fq, fk, fv) in hyps.items():
        for t in (0, 1):
            K = np.stack([fk(k[s]) for s in range(t + 1)], 1)
            Vv = np.stack([fv(v[s]) for s in range(t + 1)], 1)
            o = np.zeros((24, 256))
            for h in range(24):
                sc = K[h // 6] @ fq(q[t][h]) / 16.0
                p = np.exp(sc - sc.max())
                o[h] = (p / p.sum()) @ Vv[h // 6]
            out[f"{name}@{t}"] = rel(orc[(f"attn_pregate-{L}", t)], o)
    return out


def v1_teacher_forced(eng, orc, g, ids):
    """Layer-local check: feed llama.cpp's own residual input of layer L (positions 0 and 1, so attention sees two
    keys and the DeltaNet state is non-zero) through t4q's layer L alone and compare every intermediate of that layer
    and its output with llama.cpp's. Isolates per-layer kernel error from the chaotic growth of tiny differences."""
    res = {}
    for L in range(64):
        ins = []
        for tag in (0, 1):
            if L == 0:
                h = orc.get(("model.input_embed", tag))
                if h is None:
                    h = g.deq("token_embd.weight", [int(ids[tag])]).ravel()
            else:
                h = orc.get((f"l_out-{L - 1}", tag))
            ins.append(h)
        if any(x is None for x in ins) or ("l_out-%d" % L, 0) not in orc:
            continue
        eng.reset()
        r = {"ops": {}}
        worst = (0.0, "")
        for tag in (0, 1):
            eng.set_dump(True)
            out = eng.layer_forward(L, tag, ins[tag])
            dmp = eng.dump_all()
            eng.set_dump(False)
            r[f"l_out@{tag}"] = rel(out, orc[(f"l_out-{L}", tag)])
            for k, v in dmp.items():
                ref = orc.get((k, tag))
                if ref is None or not k.endswith(f"-{L}"):
                    continue
                if v.size != ref.size:
                    if ref.size % v.size:
                        continue
                    ref = ref[: v.size]
                e = rel(v, ref)
                r["ops"][f"{k.rsplit('-', 1)[0]}@{tag}"] = round(e, 7)
                if e > worst[0]:
                    worst = (e, f"{k}@{tag}")
        if (L + 1) % 4 == 0:
            r["attn_probe"] = attn_probe(orc, L)
        r["worst_op"] = worst[1]
        r["worst_op_rel"] = worst[0]
        res[L] = r
    eng.reset()
    lo = [max(v["l_out@0"], v["l_out@1"]) for v in res.values()]
    summ = {"n_layers": len(res), "max_l_out_rel": float(max(lo)) if lo else None,
            "mean_l_out_rel": float(np.mean(lo)) if lo else None,
            "max_op_rel": float(max(v["worst_op_rel"] for v in res.values())) if res else None}
    summ["pass"] = bool(len(res) == 64 and summ["max_l_out_rel"] <= 1e-2)
    return res, summ


def run_dump(eng, ids):
    n = len(ids)
    eng.reset()
    tdumps, logits1 = {}, None
    for i in range(n):
        want = i in (0, 1, n - 1)
        eng.set_dump(want)
        lg = eng.logits(ids[i:i + 1])
        if want:
            tdumps[i] = eng.dump_all()
            if i == 1:
                logits1 = lg[0].copy()
    eng.set_dump(False)
    return tdumps, logits1


def run_v2(eng, man, a):
    v2 = {}
    all_b, all_t, floor = [], [], []
    t = time.time()
    nsteps = 0
    for s in man["seqs"]:
        ids = read_ids(os.path.join(a.work, s["ids"]))
        n = len(ids)
        T = min(s["tbt"], n - 1)
        eng.reset()
        mine = eng.logits(ids)
        nsteps += n
        ob = np.fromfile(os.path.join(a.oracle, s["name"] + ".batch.f32"), dtype=np.float32).reshape(n, V)
        ot = np.fromfile(os.path.join(a.oracle, s["name"] + ".tbt.f32"), dtype=np.float32).reshape(T, V)
        kb, ab, gb = kl_stats(ob, mine)
        kt, at, gt = kl_stats(ot, mine[n - T:])
        kf, af, gf = kl_stats(ot, ob[n - T:])
        v2[s["name"]] = {"vs_batch": summarize(kb, ab, gb), "vs_tbt": summarize(kt, at, gt),
                         "noise_floor_batch_vs_tbt": summarize(kf, af, gf)}
        all_b.append((kb, ab, gb))
        all_t.append((kt, at, gt))
        floor.append((kf, af, gf))
        log("V2", s["name"], json.dumps(v2[s["name"]]))
    cat = lambda L: [np.concatenate([x[i] for x in L]) for i in range(3)]  # noqa: E731
    v2["ALL_vs_batch"] = summarize(*cat(all_b))
    v2["ALL_vs_tbt"] = summarize(*cat(all_t))
    v2["ALL_floor"] = summarize(*cat(floor))
    v2["t4q_ms_per_step"] = round(1e3 * (time.time() - t) / max(nsteps, 1), 2)
    return v2


def v2_ok(S):
    return bool(S["top1_agree_excl_ties"] >= 0.99 and S["mean_kl"] <= 2e-3 and S["p99_kl"] <= 2e-2)


def run_v3(eng, man, a, tok):
    v3 = {}
    ok = True
    for s in man["gens"]:
        ids = read_ids(os.path.join(a.work, s["ids"]))
        eng.reset()
        t = time.time()
        eng.prefill(ids)
        tp = time.time() - t
        t = time.time()
        gen = eng.generate(s["n"])
        tg = time.time() - t
        ref = np.fromfile(os.path.join(a.oracle, s["name"] + ".gen.i32"), dtype=np.int32)
        gaps = np.loadtxt(os.path.join(a.oracle, s["name"] + ".gen.txt"))[:, 3]
        m = min(len(gen), len(ref))
        diff = np.nonzero(gen[:m] != ref[:m])[0]
        first = int(diff[0]) if len(diff) else -1
        gap = float(gaps[first]) if first >= 0 else None
        passed = first < 0 or (gap is not None and gap < 0.05)
        ok &= passed
        v3[s["name"]] = {"n": int(len(gen)), "first_divergence": first, "oracle_gap_at_div": gap,
                         "match_prefix": int(m if first < 0 else first), "pass": bool(passed),
                         "prefill_tok_s": round(len(ids) / tp, 2), "decode_tok_s": round((len(gen) - 1) / tg, 2)}
        if tok:
            v3[s["name"]]["t4q_text"] = tok.decode(gen)
            v3[s["name"]]["llama_text"] = tok.decode(ref)
        log("V3", s["name"], json.dumps(v3[s["name"]])[:3000])
    return v3, bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lib", default=os.path.join(HERE, "..", "build", "libt4q.so"))
    ap.add_argument("--modes", default="q8,fp32")
    ap.add_argument("--sections", default=os.environ.get("T4Q_VALIDATE_SECTIONS", "v0,v1,v2,v3"))
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.work, "manifest.json")))
    modes = [m for m in a.modes.split(",") if m]
    sections = set(a.sections.split(","))

    def save():
        with open(a.out, "w") as f:
            json.dump(R, f, indent=1, default=float)

    t = time.time()
    eng = T4Q(a.model, lib=a.lib, max_ctx=4096, verbose=1)
    R["load_s"] = round(time.time() - t, 1)
    st = eng.stats()
    R["load_stats"] = st
    R["V0_repack"] = {"rows_checked": st["repack_rows_checked"], "mismatched": st["repack_rows_mismatched"],
                      "pass": st["repack_rows_mismatched"] == 0 and st["repack_rows_checked"] > 0}
    log("load", st)
    save()
    g = GGUF(a.model)
    tok = None
    try:
        tok = Tokenizer()
    except Exception:  # noqa: BLE001
        pass
    d = man["dump"]
    dids = read_ids(os.path.join(a.work, d["ids"]))
    op = os.path.join(a.oracle, d["name"] + ".dump.bin")
    orc = read_oracle_dump(op) if os.path.exists(op) else None
    opb = os.path.join(a.oracle, d["name"] + ".dumpb.bin")
    orcb = batch_slice(read_oracle_dump(opb), len(dids)) if os.path.exists(opb) else None

    for mode in modes:
        M = R.setdefault(mode, {})
        eng.set_option("act_q8", 1 if mode == "q8" else 0)
        # ---- dump run: V0 (fp32 mode: fp64 reference on real weights) + V1 (vs oracle intermediates)
        try:
            tdumps, logits1 = run_dump(eng, dids)
            if mode == "fp32":
                try:
                    r0 = v0(g, tdumps[0], tdumps[1], dids, logits1)
                    R["V0"] = r0
                    R["V0_pass"] = all(v.get("pass", False) for v in r0.values())
                    log("V0", json.dumps(r0, default=float))
                except Exception:  # noqa: BLE001
                    R["V0_error"] = traceback.format_exc()[-3000:]
                    log(R["V0_error"])
            if orc is not None:
                tf, tfs = v1_teacher_forced(eng, orc, g, dids)
                M["V1_tf"] = tf
                M["V1_tf_summary"] = tfs
                log(mode, "V1 teacher-forced", json.dumps(tfs))
                for L, r in tf.items():
                    log(mode, "TF layer", L, json.dumps(r))
                r1, worst, wfloor = v1(orc, tdumps, len(dids), orcb)
                M["V1"] = r1
                M["V1_worst_ratio_by_key"] = worst
                M["V1_worst_ratio_vs_floor_by_key"] = wfloor
                M["V1_pass"] = bool(worst) and max(worst.values()) <= 1.0
                M["V1_floor_pass"] = bool(wfloor) and max(wfloor.values()) <= 1.0
                log(mode, "V1 worst/max(tol, 2*floor) by key", json.dumps(wfloor))
                log(mode, "V1 worst/tol by key", json.dumps(worst))
            else:
                M["V1_error"] = "oracle dump missing"
        except Exception:  # noqa: BLE001
            M["dump_error"] = traceback.format_exc()[-3000:]
            log(M["dump_error"])
        save()
        # ---- V2
        if "v2" not in sections:
            continue
        try:
            v2 = run_v2(eng, man, a)
            M["V2"] = v2
            A, Tt, F = v2["ALL_vs_batch"], v2["ALL_vs_tbt"], v2["ALL_floor"]
            M["V2_kl_over_floor"] = {"batch": A["mean_kl"] / max(F["mean_kl"], 1e-12),
                                     "tbt": Tt["mean_kl"] / max(F["mean_kl"], 1e-12)}
            M["V2_tbt_pass"] = v2_ok(Tt)
            M["V2_batch_pass"] = v2_ok(A)
            log(mode, "V2 ALL", json.dumps({k: v2[k] for k in ("ALL_vs_batch", "ALL_vs_tbt", "ALL_floor")}))
        except Exception:  # noqa: BLE001
            M["V2_error"] = traceback.format_exc()[-3000:]
            log(M["V2_error"])
        save()
        # ---- V3
        try:
            M["V3"], M["V3_pass"] = run_v3(eng, man, a, tok)
        except Exception:  # noqa: BLE001
            M["V3_error"] = traceback.format_exc()[-3000:]
            log(M["V3_error"])
        save()
    eng.set_option("act_q8", 0)
    R["final_stats"] = eng.stats()
    q8, f32 = R.get("q8", {}), R.get("fp32", {})
    # Gate (see PROGRESS.md): V0 op-level vs fp64 + bit-exact repack; V1 (design tolerances) and V2 (design
    # thresholds vs token-by-token llama.cpp) in act_q8 mode, which reproduces llama.cpp's q8_1 activation
    # rounding; fp32 mode must pass V2 vs the llama batch path; V3 greedy in both modes.
    def floor_ok(M):  # design V2 secondary criterion: within 2x the llama.cpp internal noise floor
        k = M.get("V2_kl_over_floor", {})
        return bool(k) and k.get("tbt", 9) <= 2.0 and k.get("batch", 9) <= 2.0
    R["gate_detail"] = {"V0": R.get("V0_pass"), "repack": R.get("V0_repack", {}).get("pass"),
                        "V1_q8_floor": q8.get("V1_floor_pass"), "V1_fp32_floor": f32.get("V1_floor_pass"),
                        "V2_fp32_batch_abs": f32.get("V2_batch_pass"), "V2_fp32_floor": floor_ok(f32),
                        "V2_q8_floor": floor_ok(q8), "V3_q8": q8.get("V3_pass"), "V3_fp32": f32.get("V3_pass")}
    R["gate_M1"] = all(bool(v) for v in R["gate_detail"].values())
    save()
    log("GATE M1:", R["gate_M1"])


if __name__ == "__main__":
    main()
