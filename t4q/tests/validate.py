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


def v1(oracle, tdumps, n):
    out = {}
    worst_main = {}
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
        out[f"{name}@{tag}"] = round(e, 6)
        base, _, ly = name.rpartition("-")
        if not ly.isdigit():
            base, ly = name, "64"
        if base in MAIN_KEYS:
            L = int(ly)
            tol = 1e-2 if L == 0 else 3e-2
            k = f"{base}"
            worst_main[k] = max(worst_main.get(k, 0.0), e / tol)
    return out, worst_main


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lib", default=os.path.join(HERE, "..", "build", "libt4q.so"))
    ap.add_argument("--skip", default="")
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.work, "manifest.json")))
    skip = set(a.skip.split(","))

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

    # ---- dump run (V0 + V1)
    try:
        d = man["dump"]
        ids = read_ids(os.path.join(a.work, d["ids"]))
        n = len(ids)
        eng.reset()
        tdumps = {}
        logits1 = None
        for i in range(n):
            want = i in (0, 1, n - 1)
            eng.set_dump(want)
            lg = eng.logits(ids[i:i + 1])
            if want:
                tdumps[i] = eng.dump_all()
                if i == 1:
                    logits1 = lg[0].copy()
        eng.set_dump(False)
        if "V0" not in skip:
            try:
                r0 = v0(g, tdumps[0], tdumps[1], ids, logits1)
                R["V0"] = r0
                R["V0_pass"] = all(v.get("pass", False) for v in r0.values())
                log("V0", json.dumps(r0, default=float))
            except Exception:  # noqa: BLE001
                R["V0_error"] = traceback.format_exc()[-3000:]
                log(R["V0_error"])
        save()
        op = os.path.join(a.oracle, d["name"] + ".dump.bin")
        if os.path.exists(op):
            orc = read_oracle_dump(op)
            r1, worst = v1(orc, tdumps, n)
            R["V1"] = r1
            R["V1_worst_ratio_by_key"] = worst
            R["V1_pass"] = bool(worst) and max(worst.values()) <= 1.0
            R["V1_n_compared"] = len(r1)
            log("V1 worst/tol by key", json.dumps(worst))
        else:
            R["V1_error"] = "oracle dump missing"
        save()
    except Exception:  # noqa: BLE001
        R["dump_error"] = traceback.format_exc()[-3000:]
        log(R["dump_error"])
        save()

    # ---- V2 logits
    try:
        v2 = {}
        all_b, all_t, floor = [], [], []
        t = time.time()
        nsteps = 0
        for s in man["seqs"]:
            ids = read_ids(os.path.join(a.work, s["ids"]))
            n, T = len(ids), s["tbt"]
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
            all_b.append((kb, ab, gb)); all_t.append((kt, at, gt)); floor.append((kf, af, gf))
            log("V2", s["name"], json.dumps(v2[s["name"]]))
        cat = lambda L: [np.concatenate([x[i] for x in L]) for i in range(3)]  # noqa: E731
        v2["ALL_vs_batch"] = summarize(*cat(all_b))
        v2["ALL_vs_tbt"] = summarize(*cat(all_t))
        v2["ALL_floor"] = summarize(*cat(floor))
        v2["t4q_ms_per_step"] = round(1e3 * (time.time() - t) / max(nsteps, 1), 2)
        R["V2"] = v2
        A, Tt, F = v2["ALL_vs_batch"], v2["ALL_vs_tbt"], v2["ALL_floor"]
        R["V2_pass"] = bool(A["top1_agree_excl_ties"] >= 0.99 and A["mean_kl"] <= 2e-3 and A["p99_kl"] <= 2e-2 and
                            Tt["top1_agree_excl_ties"] >= 0.99 and Tt["mean_kl"] <= 2e-3 and Tt["p99_kl"] <= 2e-2)
        R["V2_kl_over_floor"] = {"batch": A["mean_kl"] / max(F["mean_kl"], 1e-12),
                                 "tbt": Tt["mean_kl"] / max(F["mean_kl"], 1e-12)}
        save()
    except Exception:  # noqa: BLE001
        R["V2_error"] = traceback.format_exc()[-3000:]
        log(R["V2_error"])
        save()

    # ---- V3 greedy
    try:
        tok = None
        try:
            tok = Tokenizer()
        except Exception:  # noqa: BLE001
            pass
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
        R["V3"] = v3
        R["V3_pass"] = bool(ok)
        save()
    except Exception:  # noqa: BLE001
        R["V3_error"] = traceback.format_exc()[-3000:]
        log(R["V3_error"])
        save()
    R["final_stats"] = eng.stats()
    R["gate_M1"] = bool(R.get("V0_pass") and R.get("V0_repack", {}).get("pass") and R.get("V1_pass") and
                        R.get("V2_pass") and R.get("V3_pass"))
    save()
    log("GATE M1:", R["gate_M1"])


if __name__ == "__main__":
    main()
