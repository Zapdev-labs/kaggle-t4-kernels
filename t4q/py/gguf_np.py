"""Minimal numpy GGUF reader + dequant (F32, F16, Q4_0, Q4_1, Q8_0, Q5_K, Q6_K) for the fp64 reference checks.

Independent of the C++ loader (written from ggml-quants.c), so it doubles as a cross-check of the CPU port.
"""
import struct

import numpy as np

BLOCK = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210)}
TYPE_NAME = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 8: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K"}


class Tensor:
    def __init__(self, name, typ, ne, offset):
        self.name, self.type, self.ne, self.offset = name, typ, ne, offset

    @property
    def rows(self):
        r = 1
        for x in self.ne[1:]:
            r *= x
        return r

    @property
    def row_bytes(self):
        be, bb = BLOCK[self.type]
        return self.ne[0] // be * bb


class GGUF:
    def __init__(self, path):
        self.path = path
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        buf = self.mm
        p = 0

        def rd(fmt):
            nonlocal p
            v = struct.unpack_from("<" + fmt, buf, p)
            p += struct.calcsize("<" + fmt)
            return v[0]

        def rstr():
            nonlocal p
            n = rd("Q")
            s = bytes(buf[p:p + n]).decode("utf-8", "replace")
            p += n
            return s

        sfmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}
        assert rd("I") == 0x46554747
        self.version = rd("I")
        nt, nkv = rd("Q"), rd("Q")
        self.kv = {}
        for _ in range(nkv):
            k = rstr()
            t = rd("I")
            if t == 8:
                self.kv[k] = rstr()
            elif t == 9:
                et, n = rd("I"), rd("Q")
                if et == 8:
                    for _ in range(n):
                        rstr()
                    self.kv[k] = f"<str array {n}>"
                else:
                    vals = struct.unpack_from("<" + sfmt[et] * n, buf, p) if n <= 64 else None
                    p += struct.calcsize("<" + sfmt[et]) * n
                    self.kv[k] = list(vals) if vals is not None else f"<array {n}>"
            else:
                self.kv[k] = rd(sfmt[t])
        self.tensors = {}
        for _ in range(nt):
            name = rstr()
            nd = rd("I")
            ne = [rd("Q") for _ in range(nd)]
            typ = rd("I")
            off = rd("Q")
            self.tensors[name] = Tensor(name, typ, ne, off)
        align = self.kv.get("general.alignment", 32)
        self.data_start = (p + align - 1) // align * align

    def raw_rows(self, name, rows=None):
        t = self.tensors[name]
        base = self.data_start + t.offset
        rb = t.row_bytes
        if rows is None:
            return np.asarray(self.mm[base:base + rb * t.rows]).reshape(t.rows, rb)
        rows = np.asarray(rows, dtype=np.int64)
        return np.stack([np.asarray(self.mm[base + r * rb: base + (r + 1) * rb]) for r in rows])

    def deq(self, name, rows=None):
        """fp32 [nrows, ne0] dequantized exactly like ggml dequantize_row_*"""
        t = self.tensors[name]
        raw = self.raw_rows(name, rows)
        return dequant(raw, t.type, t.ne[0])


def _f16(a):
    return a.copy().view(np.float16).astype(np.float32)


def _scale_min_k4(sc):  # sc: [nb, 12] uint8 -> (scales [nb, 8], mins [nb, 8])
    d = np.zeros(sc.shape[:-1] + (8,), np.uint8)
    m = np.zeros_like(d)
    for j in range(8):
        if j < 4:
            d[..., j] = sc[..., j] & 63
            m[..., j] = sc[..., j + 4] & 63
        else:
            d[..., j] = (sc[..., j + 4] & 0xF) | ((sc[..., j - 4] >> 6) << 4)
            m[..., j] = (sc[..., j + 4] >> 4) | ((sc[..., j] >> 6) << 4)
    return d, m


def dequant(raw, typ, ne0):
    nr = raw.shape[0]
    be, bb = BLOCK[typ]
    nb = ne0 // be
    b = raw.reshape(nr, nb, bb)
    if typ == 0:
        return raw.copy().view(np.float32).reshape(nr, ne0)
    if typ == 1:
        return raw.copy().view(np.float16).astype(np.float32).reshape(nr, ne0)
    if typ in (2, 3):
        d = _f16(b[..., 0:2])  # [nr, nb, 1]
        off = 2 if typ == 2 else 4
        qs = b[..., off:off + 16]
        lo = (qs & 0xF).astype(np.float32)
        hi = (qs >> 4).astype(np.float32)
        q = np.concatenate([lo, hi], axis=-1)  # [nr, nb, 32]
        if typ == 2:
            y = (q - 8.0).astype(np.float32) * d
        else:
            m = _f16(b[..., 2:4])
            y = q * d + m
        return y.astype(np.float32).reshape(nr, ne0)
    if typ == 8:
        d = _f16(b[..., 0:2])
        q = b[..., 2:34].copy().view(np.int8).astype(np.float32)
        return (q * d).astype(np.float32).reshape(nr, ne0)
    if typ == 13:  # Q5_K
        d = _f16(b[..., 0:2])[..., 0]
        dmin = _f16(b[..., 2:4])[..., 0]
        sc, mn = _scale_min_k4(b[..., 4:16])
        qh = b[..., 16:48]
        ql = b[..., 48:176]
        out = np.empty((nr, nb, 256), np.float32)
        for c in range(4):
            q = ql[..., 32 * c:32 * c + 32]
            for half in range(2):
                s = 2 * c + half
                nib = (q & 0xF) if half == 0 else (q >> 4)
                hb = ((qh >> s) & 1) * 16
                qq = (nib + hb).astype(np.float32)
                d1 = (d * sc[..., s].astype(np.float32)).astype(np.float32)
                m1 = (dmin * mn[..., s].astype(np.float32)).astype(np.float32)
                out[..., 64 * c + 32 * half: 64 * c + 32 * half + 32] = d1[..., None] * qq - m1[..., None]
        return out.reshape(nr, ne0)
    if typ == 14:  # Q6_K
        ql = b[..., 0:128]
        qh = b[..., 128:192]
        sc = b[..., 192:208].copy().view(np.int8).astype(np.float32)
        d = _f16(b[..., 208:210])[..., 0]
        out = np.empty((nr, nb, 256), np.float32)
        for n in range(2):
            l_ = ql[..., 64 * n:64 * n + 64]
            h_ = qh[..., 32 * n:32 * n + 32]
            for qd in range(4):
                src = l_[..., (qd & 1) * 32:(qd & 1) * 32 + 32]
                lo = (src >> 4) if qd >= 2 else (src & 0xF)
                hb = (h_ >> (2 * qd)) & 3
                q = ((lo | (hb << 4)).astype(np.int16) - 32).astype(np.float32)
                for half in range(2):
                    s = sc[..., 8 * n + 2 * qd + half]
                    ds = (d * s).astype(np.float32)
                    e0 = 128 * n + 32 * qd + 16 * half
                    out[..., e0:e0 + 16] = ds[..., None] * q[..., 16 * half:16 * half + 16]
        return out.reshape(nr, ne0)
    raise NotImplementedError(TYPE_NAME.get(typ, typ))
