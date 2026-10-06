#!/usr/bin/env python3
"""The independent check of the r19g q8 fast-path dot formulas (gemv_ref.cu: dot_q8k K2/K4,
dot_q8 Q51, dot_q8_0_p4 + the three quantizes) against the dequant reference.

Path A (reference): the tree's deq32-style dequant (the bit-exact-validated ggml layouts) ->
fp32 weights -> the plain fp32 dot.
Path B (the kernel under test): the ggml activation quantize (q8_K / q8_1 / q8_0) + the
transcribed dot formula, evaluated in plain python exactly as the CUDA body computes it.

The two differ by the activation quantization error (~0.4%/group), so the check is an error
BOUND: rel(B, A) <= 1.5%. A wrong nibble plane, shift, scale index, or bsum/min pairing
produces tens-to-hundreds of percent, so this catches every layout bug the gates would
otherwise spend a quota round on. The exact-integer equivalence with the ggml generics was
verified by construction in the transcription (the integer work is exact and order-free).
"""
import math
import random
import struct

random.seed(0xC0FFEE)


def h2f(h):  # fp16 -> fp32 (round-to-nearest, the numpy-free way)
    return struct.unpack("<e", struct.pack("<e", h))[0]


def f2h(f):  # fp32 -> fp16 with round-to-nearest-even (struct does RN)
    return struct.unpack("<e", struct.pack("<e", f))[0]


def nearest_int(f):  # ggml's magic (fval + 12582912) mantissa trick
    val = f + 12582912.0
    i = struct.unpack("<i", struct.pack("<f", val))[0]
    return (i & 0x007FFFFF) - 0x00400000


def scale_min_k4(j, q):  # dev_scale_min_k4 / the ggml utmp dance, same values
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


# ---------------------------------------------------------------- the ggml quantizes (path B inputs)
def quantize_q8_1(x):  # per 32: d = amax/127 (f16), s = sum (f16), q = round(x/d)
    n = len(x) // 32
    qs, xd, xs = [], [], []
    for b in range(n):
        g = x[32 * b:32 * b + 32]
        amax = max(abs(v) for v in g)
        d = h2f(f2h(amax / 127.0))
        q = [0 if amax == 0.0 else round(v / d) for v in g]
        qs += [max(-127, min(127, v)) for v in q]
        xd.append(d)
        xs.append(h2f(f2h(math.fsum(g))))
    return qs, xd, xs


def quantize_q8_0(x):  # per 32: d = amax/127 (f16), id = 1/d, q = round(x*id)
    n = len(x) // 32
    qs, xd = [], []
    for b in range(n):
        g = x[32 * b:32 * b + 32]
        amax = max(abs(v) for v in g)
        d = amax / 127.0
        idn = (1.0 / d) if d else 0.0
        qs += [int(round(v * idn)) for v in g]
        xd.append(h2f(f2h(d)))
    return qs, xd


def quantize_q8_K(x):  # per 256: signed max-abs, iscale = -127/max, MIN(127,v), 16 bsums, d=1/iscale
    n = len(x) // 256
    qs, bsums, d = [], [], []
    for b in range(n):
        g = x[256 * b:256 * b + 256]
        amax, mx = 0.0, 0.0
        for v in g:
            if abs(v) > amax:
                amax, mx = abs(v), v
        if amax == 0.0:
            qs += [0] * 256
            d.append(0.0)
            bsums += [0] * 16
            continue
        iscale = -127.0 / mx
        q = [min(127, nearest_int(iscale * v)) for v in g]
        qs += q
        for j in range(16):
            bsums.append(sum(q[16 * j:16 * j + 16]))
        d.append(1.0 / iscale)
    return qs, bsums, d


# ---------------------------------------------------------------- random blocks (the M1 packed layouts)
def gen_q2k():  # meta 20 B [d, dmin f16, scales[16]], codes 64 B; the TRUE layout (deq32-validated +
    # the ggml primary): sub s = elems 16s..16s+15 -> bytes [32*(s>>3) + 16*(s&1) .. +16) at ONE
    # shared shift 2*((s>>1)&3); w = d*(sc[s]&15)*q - dmin*(sc[s]>>4), q = (byte>>shift)&3
    d, dmin = random.uniform(-0.02, 0.02), random.uniform(0.0, 0.005)
    sc = [random.randrange(256) for _ in range(16)]
    qs = [random.randrange(256) for _ in range(64)]
    w = []
    for sub in range(16):
        a = d * (sc[sub] & 15)
        m = dmin * (sc[sub] >> 4)
        base = 32 * (sub >> 3) + 16 * (sub & 1)
        sh = 2 * ((sub >> 1) & 3)
        w += [a * ((qs[base + l] >> sh) & 3) - m for l in range(16)]
    return {"meta": [d, dmin] + sc, "codes": qs}, w


def gen_q4k():
    d, dmin = random.uniform(-0.03, 0.03), random.uniform(0.0, 0.006)
    scales = [random.randrange(256) for _ in range(12)]
    qs = [random.randrange(256) for _ in range(128)]
    sm = [scale_min_k4(j, scales) for j in range(8)]
    w = []
    for s in range(8):  # sub s = elems 32s..32s+31; bytes 32*(s>>1), plane s&1
        sc, mi = sm[s]
        base = 32 * (s >> 1)
        d1, m1 = d * sc, dmin * mi
        w += [d1 * ((qs[base + l] >> 4) & 15) - m1 if s & 1 else d1 * (qs[base + l] & 15) - m1
              for l in range(32)]
    return {"meta": [d, dmin] + scales, "codes": qs}, w


def gen_q51():  # the packed (deq32-validated) layout: byte l: lo nibble = elem l, hi = elem 16+l;
    # qh[4] as a LE u32: bit e = elem e's 5th bit. w = d*(nib | qhbit<<4) + m
    d, m = random.uniform(-0.05, 0.05), random.uniform(-0.1, 0.1)
    qs = [random.randrange(256) for _ in range(16)]
    qh = [random.randrange(256) for _ in range(4)]
    qhw = qh[0] | (qh[1] << 8) | (qh[2] << 16) | (qh[3] << 24)
    w = []
    for l in range(16):
        w.append(d * ((qs[l] & 15) + (16 if (qhw >> l) & 1 else 0)) + m)
    for l in range(16):
        w.append(d * ((qs[l] >> 4) + (16 if (qhw >> (16 + l)) & 1 else 0)) + m)
    return {"d": d, "m": m, "codes": qs, "qh": qh}, w


def gen_q40():  # the packed (deq32-validated) layout: byte j: lo = elem j, hi = elem 16+j; w = d*(nib-8)
    d = random.uniform(-0.05, 0.05)
    qs = [random.randrange(256) for _ in range(16)]
    w = []
    for l in range(16):
        w.append(d * ((qs[l] & 15) - 8))
    for l in range(16):
        w.append(d * ((qs[l] >> 4) - 8))
    return {"d": d, "codes": qs}, w


# ---------------------------------------------------------------- the dot bodies UNDER TEST
def dot_k2(blk, x, xv, bs0, bs1, yd, g):
    # the kernel's group form verbatim: window 32*((g&7)>>2), shift 2*(g&3), subs 2g, 2g+1
    meta, codes = blk["meta"], blk["codes"]
    d, dmin = h2f(f2h(meta[0])), h2f(f2h(meta[1]))
    sc = meta[2:]
    qs = codes[32 * ((g & 7) >> 2):]
    sh = (g & 3) << 1
    s0 = 2 * (g & 7)
    a0, m0 = sc[s0] & 15, sc[s0] >> 4
    a1, m1 = sc[s0 + 1] & 15, sc[s0 + 1] >> 4
    is0 = sum(xv[l] * ((qs[l] >> sh) & 3) for l in range(16))
    is1 = sum(xv[l + 16] * ((qs[l + 16] >> sh) & 3) for l in range(16))
    dall, dmin_ = yd * d, yd * dmin
    return dall * float(a0 * is0 + a1 * is1) - dmin_ * float(bs0 * m0 + bs1 * m1)


def sx8(v):  # a byte as a signed int8 lane
    return v - 256 if v >= 128 else v


def dp4a(a, b, c=0):
    """IDP.4A.S8.S8: the 4 signed-int8 lanes of a and b dotted, plus c (int32 exact)."""
    t = c
    for i in range(4):
        t += sx8((a >> (8 * i)) & 0xFF) * sx8((b >> (8 * i)) & 0xFF)
    return t


def lanes4(bytes_):  # 4 values -> one int32 of lanes (each byte masked, like the C++ int8 packing)
    return (bytes_[0] & 0xFF) | ((bytes_[1] & 0xFF) << 8) | ((bytes_[2] & 0xFF) << 16) | (bytes_[3] << 24)


def dot_q51(blk, xv, d8, s8):
    # the kernel's dp4a form verbatim: the packed SPLIT (byte j: lo = elem j, hi = elem 16+j)
    # packs the lanes cleanly; the elems 4q..4q+3's qh bits are the CONSECUTIVE bits (4q..4q+3)
    # - one nibble - spread into the 4 lane positions by (n * 0x00204081 & 0x01010101) << 4
    # (a direct 0x01010101 mask would wrongly pick bits 8 apart)
    c, qh = blk["codes"], blk["qh"]
    qhw = lanes4(qh)
    dx, mx = h2f(f2h(blk["d"])), h2f(f2h(blk["m"]))
    s = 0
    for q in range(4):
        ci = lanes4(c[4 * q:4 * q + 4])
        n0 = (qhw >> (4 * q)) & 0xF
        n1 = (qhw >> (16 + 4 * q)) & 0xF
        lo = (ci & 0x0F0F0F0F) | (((n0 * 0x00204081) & 0x01010101) << 4)
        hi = ((ci >> 4) & 0x0F0F0F0F) | (((n1 * 0x00204081) & 0x01010101) << 4)
        s = dp4a(lo, lanes4(xv[4 * q:4 * q + 4]), s)
        s = dp4a(hi, lanes4(xv[16 + 4 * q:16 + 4 * q + 4]), s)
    return (dx * d8) * float(s) + mx * s8


def dot_q40(blk, xv, dy, s32):
    # the kernel's dp4a form verbatim: lo nibbles = elems 0..15, hi = 16..31 (the packed
    # SPLIT), and the -8 bias FACTORED: sum (nib-8)*x = sum nib*x - 8*sum x (identical int32;
    # no overflow: |s| <= 32*15*127 + 8*32*127 << 2^31). s32 = the block's signed code sum.
    c = blk["codes"]
    d4 = h2f(f2h(blk["d"]))
    s = 0
    for q in range(4):
        ci = lanes4(c[4 * q:4 * q + 4])
        s = dp4a(ci & 0x0F0F0F0F, lanes4(xv[4 * q:4 * q + 4]), s)
        s = dp4a((ci >> 4) & 0x0F0F0F0F, lanes4(xv[16 + 4 * q:16 + 4 * q + 4]), s)
    s -= 8 * s32
    return float(s) * d4 * dy


def dot_k4(blk, x, xv, bs0, bs1, yd, g):
    # the kernel's dp4a form verbatim: the group = one 32-elem sub = the 32 window bytes at
    # ONE shared nibble plane (byte l = elem l), so the lanes pack cleanly - no bias, the
    # mins fold outside the int dot
    meta, codes = blk["meta"], blk["codes"]
    d, dmin = h2f(f2h(meta[0])), h2f(f2h(meta[1]))
    sc, mi = scale_min_k4(g & 7, meta[2:])
    base = 32 * ((g & 7) >> 1)
    qs = codes[base:base + 32]
    hin = g & 1
    s = 0
    for l in range(8):
        qi = lanes4(qs[4 * l:4 * l + 4])
        lo = ((qi >> 4) & 0x0F0F0F0F) if hin else (qi & 0x0F0F0F0F)
        s = dp4a(lo, lanes4(xv[4 * l:4 * l + 4]), s)
    return (d * yd) * float(sc * s) - (dmin * yd) * float((bs0 + bs1) * mi)


# ---------------------------------------------------------------- the harness
def rel(a, b):
    m = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / m


def check(name, trial, a, b, w, x, qds, gshift):
    """The honest bound: |B - A| = |sum w_i e_i| where |e_i| <= d_g(i)/2 (the x-quantization),
    plus the d's f16 storage slack (|f16(d)-d|/d ~ 5e-4 -> 127*that on the dot)."""
    bound = sum(abs(wi) * (abs(qds[i >> gshift]) * 0.5) for i, wi in enumerate(w))
    bound += 0.002 * sum(abs(wi) * abs(xi) for wi, xi in zip(w, x))
    ok = abs(a - b) <= 1.6 * bound + 1e-9
    if not ok:
        print(f"{name} trial {trial}: |B-A|={abs(a-b):.6f} bound={bound:.6f}  A={a:.4f} B={b:.4f}")
    return 0 if ok else 1


def main():
    bad = 0
    # Q2_K / Q4_K: one row = 4 super-blocks (1024 elems), the x = 1024, the Q8_K pairing
    for trial in range(30):
        x = [random.uniform(-1.2, 1.2) for _ in range(1024)]
        for name, gen, dot in (("K2", gen_q2k, dot_k2), ("K4", gen_q4k, dot_k4)):
            gen_results = [gen() for _ in range(4)]
            blocks = [b for b, _ in gen_results]
            w = [wi for _, bw in gen_results for wi in bw]
            a = sum(wi * xi for wi, xi in zip(w, x))
            qs, bsums, ds = quantize_q8_K(x)
            # the dot per group g (32 elems), grouped per lane like the kernel's strided loop
            accs = [[] for _ in range(32)]
            for g in range(32):
                xv = qs[32 * g:32 * g + 32]
                sb = g >> 3
                sub = 2 * (g & 7)
                accs[g % 32].append(dot(blocks[sb], x, xv, bsums[sb * 16 + sub],
                                        bsums[sb * 16 + sub + 1], ds[sb], g & 7))
            bsum = sum(v for acc in accs for v in acc)
            bad += check(name, trial, a, bsum, w, x, ds, 8)
    # Q5_1 / Q4_0: one row = 32 blocks (1024 elems), the q8_1 / q8_0 pairings
    for trial in range(30):
        x = [random.uniform(-1.2, 1.2) for _ in range(1024)]
        for name, gen, dot in (("Q51", gen_q51, dot_q51), ("Q40", gen_q40, dot_q40)):
            gen_results = [gen() for _ in range(32)]
            blocks = [b for b, _ in gen_results]
            w = [wi for _, bw in gen_results for wi in bw]
            a = sum(wi * xi for wi, xi in zip(w, x))
            if name == "Q51":
                qs, xd, xs = quantize_q8_1(x)
                bsum = 0.0
                for g in range(32):
                    xv = qs[32 * g:32 * g + 32]
                    bsum += dot(blocks[g], xv, xd[g], xs[g])
                bad += check(name, trial, a, bsum, w, x, xd, 5)
            else:
                qs, xd = quantize_q8_0(x)
                bsum = 0.0
                for g in range(32):
                    xv = qs[32 * g:32 * g + 32]
                    bsum += dot(blocks[g], xv, xd[g], sum(xv))  # s32 = the block's signed code sum
                bad += check(name, trial, a, bsum, w, x, xd, 5)
    # the quantize round-trips: x ~ d * q (the q8_K's negative-d quirk included)
    x = [random.uniform(-2, 2) for _ in range(512)]
    qs, bsums, ds = quantize_q8_K(x)
    for b in range(2):
        err = max(abs(x[256 * b + i] - ds[b] * qs[256 * b + i]) for i in range(256))
        assert err <= abs(ds[b]) * 0.51 + 1e-9, f"q8_K round-trip err {err}"
        for j in range(16):
            assert bsums[b * 16 + j] == sum(qs[256 * b + 16 * j:256 * b + 16 * j + 16]), "q8_K bsum"
    qs, xd = quantize_q8_0(x)
    for b in range(16):
        err = max(abs(x[32 * b + i] - xd[b] * qs[32 * b + i]) for i in range(32))
        assert err <= abs(xd[b]) * 0.65 + 1e-9, f"q8_0 round-trip err {err}"
    qs, xd, xs = quantize_q8_1(x)
    for b in range(16):
        err = max(abs(x[32 * b + i] - xd[b] * qs[32 * b + i]) for i in range(32))
        assert err <= abs(xd[b]) * 0.51 + 1e-9, f"q8_1 round-trip err {err}"
        assert abs(xs[b] - math.fsum(x[32 * b:32 * b + 32])) <= 1e-3 * max(1.0, abs(xs[b])), "q8_1 sum"
    print("BAD" if bad else "ALL OK", f"({bad} dot failures of 120 trials)")


if __name__ == "__main__":
    main()
