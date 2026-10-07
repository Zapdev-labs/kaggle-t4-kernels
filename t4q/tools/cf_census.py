#!/usr/bin/env python3
"""cf-m2 census analysis: parse a census.bin from tools/cf_run census mode and decide the tier split.

File: "CFC1" u32, u32 NL, u32 TOPK, then per step: NL x TOPK x (u32 expert_id, f32 renormed weight).
Verdicts printed: the per-layer concentration curve (the cumulative routed mass at H resident
experts), the unique-expert counts, the cross-layer overlap, an LRU working-set simulation at
the measured VRAM budget, and the recommended split (hot VRAM / pinned warm / disk tail).
Usage: cf_census.py <census.bin> [--vram-gib 27] [--tok-s 60]
       cf_census.py <census.bin> --hotset <out.hotset> <H>: also write the hot-set file the
       loader's resident tier reads (T4Q_CF_HOTSET=<path>): "CFHS" u32, u32 NL, u32 H, then
       NL x H u32 expert ids, layer-major, the per-layer top-H by accumulated routed mass.
"""
import json
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

# measured platform rates (cf-m0, kaggle/cf0): expert slab bytes per expert-layer
EE = 640          # expert FFN dim
D = 2560
GU_BYTES = 2 * EE * (D // 256) * 84   # gate+up Q2_K per expert-layer: 640 rows x 840 B x 2 = 1075200... computed
DN_BYTES = D * (EE // 32) * 18       # down Q4_0 per expert-layer: 2560 rows x 360 B
SLAB = 2 * 640 * 840 + 2560 * 360    # 1.08 MB + 0.92 MB = ~2.0 MB per expert-layer


def main():
    p = Path(sys.argv[1])
    vram_gib = 27.0
    tok_s = 60.0
    hotset_path = None
    hotset_H = 0
    args = sys.argv[2:]
    for i, a in enumerate(args):
        if a == "--vram-gib":
            vram_gib = float(args[i + 1])
        elif a == "--tok-s":
            tok_s = float(args[i + 1])
        elif a == "--hotset":
            hotset_path = args[i + 1]
            hotset_H = int(args[i + 2])
    data = p.read_bytes()
    magic, nl, topk = struct.unpack_from("<III", data, 0)
    assert magic == 0x31434643, f"bad magic {magic:#x}"
    per = nl * topk * 8
    n_steps = (len(data) - 12) // per
    assert n_steps * per + 12 == len(data), "trailing bytes"
    print(f"census: {n_steps} steps, {nl} layers, top-{topk}", file=sys.stderr)

    layers = [Counter() for _ in range(nl)]
    hits = [Counter() for _ in range(nl)]  # per (layer, step) presence
    uniques = defaultdict(set)
    ids_at = []
    off = 12
    for t in range(n_steps):
        step_ids = []
        for l in range(nl):
            for k in range(topk):
                eid, w = struct.unpack_from("<If", data, off)
                off += 8
                layers[l][eid] += w
                uniques[l].add(eid)
                step_ids.append(eid)
        ids_at.append(step_ids)

    out = {"n_steps": n_steps, "nl": nl, "topk": topk}
    # the concentration curve: cumulative mass of the H hottest experts per layer
    curves = {}
    for l in range(nl):
        masses = sorted(layers[l].values(), reverse=True)
        tot = sum(masses)
        cum = 0.0
        c = []
        for h in range(len(masses)):
            cum += masses[h]
            c.append(round(cum / tot, 4))
        curves[l] = c
    # the coverage at a few H choices, averaged over layers
    cover = {}
    for H in (64, 128, 192, 256, 320, 384, 448, 512):
        vals = []
        for l in range(nl):
            c = curves[l]
            vals.append(c[min(H, len(c)) - 1] if c else 0.0)
        cover[H] = round(sum(vals) / len(vals), 4)
    out["coverage_at_H"] = cover
    uniq_counts = sorted(len(u) for u in uniques.values())
    out["unique_experts_per_layer"] = {"min": uniq_counts[0], "med": uniq_counts[len(uniq_counts) // 2],
                                       "max": uniq_counts[-1], "all_512": sum(1 for u in uniques.values()
                                                                           if len(u) == 512)}
    # cross-layer overlap: the union size of the hottest H per layer (the VRAM residency set)
    sets_at = {}
    for H in (128, 256, 384):
        u = set()
        for l in range(nl):
            top = [e for e, _ in layers[l].most_common(H)]
            u.update(top)
        sets_at[H] = len(u)
    out["hot_union_size"] = sets_at
    # the LRU working set at the VRAM budget: expert-layers are ~2.0 MB; vram_gib carries H
    # resident slots per layer after the ~3.0 GiB core (trunk + lm_head + states + staging)
    budget = int(vram_gib * 1024 / (SLAB / 1024 / 1024)) // nl * nl  # expert-layer slots, per-layer aligned
    per_layer_slots = budget // nl
    miss_mass = [0.0] * nl
    lru = [list() for _ in range(nl)]
    off = 12
    for t in range(n_steps):
        for l in range(nl):
            for k in range(topk):
                eid, w = struct.unpack_from("<If", data, off)
                off += 8
                if eid in lru[l]:
                    lru[l].remove(eid)
                    lru[l].insert(0, eid)
                else:
                    miss_mass[l] += w
                    if per_layer_slots > 0:
                        lru[l].insert(0, eid)
                        if len(lru[l]) > per_layer_slots:
                            lru[l].pop()
    mm = sum(miss_mass) / n_steps / nl
    out["lru"] = {"per_layer_slots": per_layer_slots, "vram_gib": vram_gib,
                  "mean_missed_mass_per_token": round(mm, 4)}
    # the tier verdict at the target rate: the missed mass x slab bytes x tok_s = the byte/s
    miss_bs = mm * nl * SLAB * tok_s
    out["miss_byte_s_at_target"] = round(miss_bs, 0)
    out["warm_tier_share_possible"] = round(11.5e9 / miss_bs, 4) if miss_bs else None
    out["disk_share_needed"] = round(max(0.0, 1 - 11.5e9 / miss_bs), 4) if miss_bs else None
    print(json.dumps(out, indent=1))
    # per-layer detail: the top-8 experts per layer with their mass share
    detail = {}
    for l in range(0, nl, max(1, nl // 8)):
        tot = sum(layers[l].values())
        detail[l] = [(e, round(w / tot, 4)) for e, w in layers[l].most_common(8)]
    out["layer_detail"] = detail
    Path(str(p) + ".json").write_text(json.dumps(out, indent=1))
    print(f"wrote {p}.json", file=sys.stderr)

    if hotset_path:
        # the hot-set file for the loader's resident tier (T4Q_CF_HOTSET): the per-layer
        # top-H expert ids by accumulated routed mass, layer-major, uniform H (capped by
        # the smallest per-layer unique count so every layer emits exactly H ids)
        uniq_min = min(len(u) for u in uniques.values())
        H = hotset_H
        if H > uniq_min:
            print(f"hotset: H {H} capped at the min unique count {uniq_min}", file=sys.stderr)
            H = uniq_min
        with open(hotset_path, "wb") as hf:
            hf.write(struct.pack("<III", 0x53484643, nl, H))
            for l in range(nl):
                ids = [e for e, _ in layers[l].most_common(H)]
                hf.write(struct.pack(f"<{H}I", *ids))
        print(f"hotset: wrote {hotset_path} (H={H})", file=sys.stderr)


if __name__ == "__main__":
    main()
