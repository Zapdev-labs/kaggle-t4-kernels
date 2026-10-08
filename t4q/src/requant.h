// The cf-m6 requant packer core (CF_REQUANT.md section 5): the iq1_s quantizer and the
// tables init, host-only (the Kaggle packer tool and the tests). The MATH is the exact
// ggml quantize_row_iq1_s_impl / iq2xs_init_impl(IQ1_S) form (self-scaled weights for
// the first pack: qw = 1, w = sqrt(sigma2 + x*x)) so the format's tuned quality carries;
// the fudge d = (max_scale/15)*1.125 is part of the tuned format, replicated exactly.
// The neighbour fallback has NO scale fudge on this path (the 1.05 is the IQ2 path's).
// Compile with -ffp-contract=off. The sorts match the ggml forms' observable order
// exactly (the SSD sort compares values only - equal-value ties keep their input
// order, the ggml-inherited reproducibility property, now via a stable insertion
// sort instead of libc qsort; the dist2 sort is a total order on (d2, k)).
#pragma once
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#define T4Q_TABLE_HOST 1
#include "iq1s_table.h"

// ---- fp16 conversions (the standard algorithms; the widening is exact, the narrowing
// is round-to-nearest-even; verified by the all-65536-pattern round-trip in the test) ----
inline float t4q_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff, bits;
    if (e == 0) {
        if (m == 0) bits = sign;  // +-0
        else {
            int p = 9;  // the leading 1 of the 10-bit subnormal mantissa
            while (!(m & (1u << p))) p--;
            bits = sign | ((uint32_t)(p + 103) << 23) | ((m ^ (1u << p)) << (23 - p));
        }
    } else if (e == 0x1f) {
        bits = sign | 0x7f800000u | (m << 13);  // inf / NaN, payload preserved
    } else {
        bits = sign | ((e - 15 + 127) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t t4q_fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000, e = (x >> 23) & 0xff, m = x & 0x7fffff;
    if (e == 0xff) return (uint16_t)(sign | 0x7c00 | (m >> 13));  // inf / NaN, payload preserved
    int E = (int)e - 127 + 15;  // the fp16 exponent field
    if (E >= 0x1f) return (uint16_t)(sign | 0x7c00);  // overflow -> inf
    if (E > 0) {                                       // normal
        uint32_t keep = m >> 13, rem = m & 0x1fff, half = 0x1000;
        if (rem > half || (rem == half && (keep & 1))) keep++;  // RNE
        if (keep == 0x400) {
            keep = 0;
            if (++E >= 0x1f) return (uint16_t)(sign | 0x7c00);
        }
        return (uint16_t)(sign | ((uint32_t)E << 10) | keep);
    }
    if (E < -10) return (uint16_t)sign;  // below half the min subnormal -> +-0
    // subnormal: the value = (0x800000|m) * 2^(E-14-23); round to the 2^-24 quantum, RNE
    int sh = 14 - E;  // in [14, 24]
    uint32_t S = 0x800000u | m, keep = S >> sh, rem = S & ((1u << sh) - 1), half = 1u << (sh - 1);
    if (rem > half || (rem == half && (keep & 1))) keep++;
    if (keep == 0x400) return (uint16_t)(sign | (1u << 10));  // rounded up into the min normal
    return (uint16_t)(sign | keep);
}

namespace t4q_iq1s {

static const int NG = T4Q_IQ1S_NGRID;             // 2048
static const int KMAP_SIZE = T4Q_IQ1S_KMAP_SIZE;  // 43692
static const int BS = 32;                         // IQ1S_BLOCK_SIZE
static const float GROUP_MAX_EPS = 1e-12f;         // GROUP_MAX_EPS_IQ1_S
static const float DELTA = T4Q_IQ1S_DELTA;         // 0.125

// The block (the PackedW planes: codes = qs, hi = qh, d = d). NG = the 32-elem groups per
// block: 8 = the stock iq1_s 256-elem block (50 B, 1.5625 bpw); 4 = the 128-elem half
// block (26 B, 1.625 bpw) the dn tensor needs - its rows are 640 wide and 640 % 256 != 0
// (the r3 pack gate's find: the 256-block form cannot tile the dn; the half block tiles
// it exactly, 5 x 128, with the SAME per-group search/scale/lattice arithmetic and only
// the d/max_scale scope halved - a finer scale grid, never coarser).
template <int NG>
struct BlockT {
    uint16_t d;
    uint8_t qs[4 * NG];
    uint16_t qh[NG];
};
using Block = BlockT<8>;
static_assert(sizeof(BlockT<8>) == 50, "iq1_s block size");
static_assert(sizeof(BlockT<4>) == 26, "iq1_s half-block size");

// ---- the tables (the exact iq2xs_init_impl(IQ1_S) form) ----
struct Tables {
    uint8_t grid[NG][8];       // the {1,3,5} byte form: grid[k][j] = 2*L_j + 1
    int32_t kmap[KMAP_SIZE];    // u -> k for the on-grid vectors, else -(offset+1)
    std::vector<uint16_t> neigh;  // the [count, k...] lists at the negative offsets
};

static int cmp_d2(const void* l, const void* r) {  // iq2_compare_func: (d2, k)
    const int* a = (const int*)l;
    const int* b = (const int*)r;
    return a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : 0;
}
static inline int nearest_int(float fval) {  // the ggml trick
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static void init(Tables& T) {
    for (int k = 0; k < NG; ++k)
        for (int j = 0; j < 8; ++j) T.grid[k][j] = (uint8_t)(2 * ((t4q_kgrid_1bit_2048[k] >> (2 * j)) & 3) + 1);
    for (int i = 0; i < KMAP_SIZE; ++i) T.kmap[i] = -1;
    for (int k = 0; k < NG; ++k) {
        uint16_t u = 0;
        for (int j = 0; j < 8; ++j) u |= (uint16_t)(((T.grid[k][j] - 1) / 2) << (2 * j));
        T.kmap[u] = k;
    }
    // the neighbour lists: for each off-grid u, the grid entries sorted by (d2, k),
    // kept until 3 distinct distances (nwant = 3 for IQ1_S; ties at the 3rd included)
    std::vector<int> n_per_i(KMAP_SIZE, 0);
    long num_neigh = 0, num_not = 0;
    std::vector<int> dist2v(2 * NG);  // RAII: the malloc'd buffer could leak / be null on OOM
    int* dist2 = dist2v.data();
    for (int i = 0; i < KMAP_SIZE; ++i) {
        if (T.kmap[i] >= 0) continue;
        ++num_not;
        uint8_t pos[8];
        for (int j = 0; j < 8; ++j) pos[j] = (uint8_t)(2 * ((i >> (2 * j)) & 3) + 1);
        for (int k = 0; k < NG; ++k) {
            int d2 = 0;
            for (int j = 0; j < 8; ++j) {
                int df = T.grid[k][j] - pos[j];
                d2 += df * df;
            }
            dist2[2 * k + 0] = d2;
            dist2[2 * k + 1] = k;
        }
        qsort(dist2, NG, 2 * sizeof(int), cmp_d2);
        int n = 0, d2 = dist2[0], nhave = 1;
        for (int k = 0; k < NG; ++k) {
            if (dist2[2 * k] > d2) {
                if (nhave == 3) break;
                d2 = dist2[2 * k];
                ++nhave;
            }
            ++n;
        }
        n_per_i[i] = n;
        num_neigh += n;
    }
    T.neigh.assign(num_neigh + num_not, 0);
    std::vector<int> off(KMAP_SIZE);
    long counter = 0;
    for (int i = 0; i < KMAP_SIZE; ++i) {
        if (T.kmap[i] >= 0) {
            off[i] = -1;
            continue;
        }
        off[i] = (int)counter;
        counter += 1 + n_per_i[i];
    }
    for (int i = 0; i < KMAP_SIZE; ++i) {  // pass 3: recompute + write (the ggml form)
        if (T.kmap[i] >= 0) continue;
        uint8_t pos[8];
        for (int j = 0; j < 8; ++j) pos[j] = (uint8_t)(2 * ((i >> (2 * j)) & 3) + 1);
        for (int k = 0; k < NG; ++k) {
            int d2 = 0;
            for (int j = 0; j < 8; ++j) {
                int df = T.grid[k][j] - pos[j];
                d2 += df * df;
            }
            dist2[2 * k + 0] = d2;
            dist2[2 * k + 1] = k;
        }
        qsort(dist2, NG, 2 * sizeof(int), cmp_d2);
        int lc = off[i];
        T.kmap[i] = -(lc + 1);
        int d2 = dist2[0], nhave = 1;
        uint16_t* start = &T.neigh[lc++];
        int n = 0;
        for (int k = 0; k < NG; ++k) {
            if (dist2[2 * k] > d2) {
                if (nhave == 3) break;
                d2 = dist2[2 * k];
                ++nhave;
            }
            T.neigh[lc + n] = (uint16_t)dist2[2 * k + 1];
            ++n;
        }
        *start = (uint16_t)n;
    }
}

// iq1_find_best_neighbour2 (the exact form: the weighted-distance best over the list,
// the full-grid search as the last resort, L from the chosen entry; NO scale fudge)
static int find_best_neighbour2(const uint16_t* neigh, const Tables& T, const float* xval,
        const float* wgt, float scale, const float* xg, int8_t* L) {
    int cnt = neigh[0];
    float best = FLT_MAX;
    int k = -1;
    for (int j = 1; j <= cnt; ++j) {
        const uint8_t* pg = T.grid[neigh[j]];
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            float q = xg[(pg[i] - 1) / 2];
            float w = wgt[i];
            float diff = scale * q - xval[i];
            d2 += w * diff * diff;
        }
        if (d2 < best) {
            best = d2;
            k = neigh[j];
        }
    }
    if (k < 0) {
        for (int i = 0; i < NG; ++i) {
            const uint8_t* pg = T.grid[i];
            float d2 = 0;
            for (int j = 0; j < 8; ++j) {
                float w = wgt[j];
                float q = xg[(pg[j] - 1) / 2];
                float diff = scale * q - xval[j];
                d2 += w * diff * diff;
            }
            if (d2 < best) {
                best = d2;
                k = i;
            }
        }
    }
    const uint8_t* pg = T.grid[k];
    for (int i = 0; i < 8; ++i) L[i] = (int8_t)((pg[i] - 1) / 2);
    return k;
}

// quantize_row_iq1_s_impl (the exact form; qw = nullptr -> the self-scaled weights),
// templated on the groups-per-block: NG=8 the stock 256-elem block, NG=4 the 128-elem
// half block (the dn tiling). One row: n a multiple of 32*NG; y = the row's n/(32*NG)
// blocks.
template <int NG>
static void quant_row_t(const Tables& T, const float* x, int n, BlockT<NG>* y, const float* qw = nullptr) {
    const float x_p[3] = {-1.f + DELTA, DELTA, 1.f + DELTA};
    const float x_m[3] = {-1.f - DELTA, -DELTA, 1.f - DELTA};
    float weight[BS], pairs[2 * BS], scales[NG], sumx[BS + 1], sumw[BS + 1];
    int* idx = (int*)(pairs + 1);
    int8_t L[BS], shifts[NG];
    uint16_t index[4];
    for (int ibl = 0; ibl < n / (32 * NG); ++ibl) {
        const float* xb = x + 32 * NG * ibl;
        BlockT<NG>& B = y[ibl];
        B.d = 0;
        memset(B.qs, 0, 4 * NG);
        memset(B.qh, 0, 2 * NG);
        float max_scale = 0, sumx2 = 0;
        for (int i = 0; i < 32 * NG; ++i) sumx2 += xb[i] * xb[i];
        float sigma2 = 2 * sumx2 / (32 * NG);
        for (int ib = 0; ib < NG; ++ib) {
            const float* xg = xb + BS * ib;
            for (int i = 0; i < BS; ++i) {
                float w = qw ? qw[32 * NG * ibl + BS * ib + i] : 1.f;
                weight[i] = w * sqrtf(sigma2 + xg[i] * xg[i]);
            }
            float max = fabsf(xg[0]);
            for (int i = 1; i < BS; ++i) max = max < fabsf(xg[i]) ? fabsf(xg[i]) : max;
            if (max < GROUP_MAX_EPS) {
                scales[ib] = 0;
                shifts[ib] = 1;
                memset(L, 1, BS);
                continue;
            }
            // the exhaustive 2-boundary SSD split search over the sorted group
            for (int j = 0; j < BS; ++j) {
                pairs[2 * j] = xg[j];
                idx[2 * j] = j;
            }
            // stable insertion sort by value (the qsort contract: equal values keep
            // their input order - the ggml reproducibility property - and the
            // comparator inlines; qsort pays an indirect call per compare)
            for (int a = 1; a < BS; ++a) {
                const float v = pairs[2 * a];
                const int vi = idx[2 * a];
                int b = a;
                while (b > 0 && pairs[2 * (b - 1)] > v) {
                    pairs[2 * b] = pairs[2 * (b - 1)];
                    idx[2 * b] = idx[2 * (b - 1)];
                    --b;
                }
                pairs[2 * b] = v;
                idx[2 * b] = vi;
            }
            sumx[0] = sumw[0] = 0;
            for (int j = 0; j < BS; ++j) {
                int i = idx[2 * j];
                sumx[j + 1] = sumx[j] + weight[i] * xg[i];
                sumw[j + 1] = sumw[j] + weight[i];
            }
            float best_score = -FLT_MAX, scale = max;
            int besti1 = -1, besti2 = -1, best_shift = 0;
            for (int i1 = 0; i1 <= BS; ++i1) {
                for (int i2 = i1; i2 <= BS; ++i2) {
                    for (int sd = 0; sd < 2; ++sd) {
                        const float* xx = sd ? x_m : x_p;
                        float sumqx = (sumx[i1] - sumx[0]) * xx[0] + (sumx[i2] - sumx[i1]) * xx[1] +
                                      (sumx[BS] - sumx[i2]) * xx[2];
                        float sumq2 = (sumw[i1] - sumw[0]) * xx[0] * xx[0] +
                                      (sumw[i2] - sumw[i1]) * xx[1] * xx[1] +
                                      (sumw[BS] - sumw[i2]) * xx[2] * xx[2];
                        if (sumq2 > 0 && sumqx * sumqx > best_score * sumq2) {
                            scale = sumqx / sumq2;
                            best_score = scale * sumqx;
                            besti1 = i1;
                            besti2 = i2;
                            best_shift = sd ? -1 : 1;
                        }
                    }
                }
            }
            if (besti1 < 0 || besti2 < 0 || best_shift == 0) {
                scales[ib] = 0;
                shifts[ib] = 1;
                memset(L, 1, BS);
                continue;
            }
            for (int j = 0; j < besti1; ++j) L[idx[2 * j]] = 0;
            for (int j = besti1; j < besti2; ++j) L[idx[2 * j]] = 1;
            for (int j = besti2; j < BS; ++j) L[idx[2 * j]] = 2;
            if (scale < 0) {
                for (int j = 0; j < BS; ++j) L[j] = 2 - L[j];
                scale = -scale;
                best_shift = -best_shift;
            }
            // the grid mapping (the kmap + the neighbour fallback)
            bool all_on_grid = true;
            const float* xx = best_shift == 1 ? x_p : x_m;
            for (int k = 0; k < BS / 8; ++k) {
                uint16_t u = 0;
                for (int j = 0; j < 8; ++j) u |= (uint16_t)(L[8 * k + j] << (2 * j));
                int gi = T.kmap[u];
                if (gi < 0) {
                    all_on_grid = false;
                    const uint16_t* nb = &T.neigh[-T.kmap[u] - 1];
                    gi = find_best_neighbour2(nb, T, xg + 8 * k, weight + 8 * k, scale, xx, L + 8 * k);
                }
                index[k] = (uint16_t)gi;
            }
            if (!all_on_grid) {
                float sumqx = 0, sumq2 = 0;
                for (int k = 0; k < BS / 8; ++k)
                    for (int j = 0; j < 8; ++j) {
                        float w = weight[8 * k + j];
                        float q = xx[(T.grid[index[k]][j] - 1) / 2];
                        sumqx += w * q * xg[8 * k + j];
                        sumq2 += w * q * q;
                    }
                if (sumqx > 0 && sumq2 > 0) scale = sumqx / sumq2;
            }
            uint16_t h = 0;
            for (int k = 0; k < BS / 8; ++k) {
                B.qs[4 * ib + k] = (uint8_t)(index[k] & 255);
                h |= (uint16_t)((index[k] >> 8) << (3 * k));
            }
            B.qh[ib] = h;
            scales[ib] = scale;
            shifts[ib] = (int8_t)best_shift;
            if (scale > max_scale) max_scale = scale;
        }
        if (!max_scale) continue;
        float d = max_scale / 15;
        B.d = t4q_fp32_to_fp16(d * 1.125f);  // the ggml fudge, part of the tuned format
        float id = 1 / d;
        for (int ib = 0; ib < NG; ++ib) {
            int l = nearest_int(0.5f * (id * scales[ib] - 1));
            l = l < 0 ? 0 : l > 7 ? 7 : l;
            if (shifts[ib] == -1) l |= 8;
            B.qh[ib] |= (uint16_t)(l << 12);
        }
    }
}

// dequantize_row_iq1_s (the exact reference form), templated on the block granularity -
// the round-trip gate's second decode path (independent of deq32: this one walks the
// {1,3,5} byte grid, deq32 walks the u16). The per-group decode is granularity-free.
template <int NG>
static void dequant_row_ref_t(const Tables& T, const BlockT<NG>* y, float* dst, int n) {
    for (int ibl = 0; ibl < n / (32 * NG); ++ibl) {
        const BlockT<NG>& B = y[ibl];
        const float d = t4q_fp16_to_fp32(B.d);
        float* o = dst + 32 * NG * ibl;
        for (int ib = 0; ib < NG; ++ib) {
            const float dl = d * (2 * ((B.qh[ib] >> 12) & 7) + 1);
            const float delta = (B.qh[ib] & 0x8000u) ? -DELTA : DELTA;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* pg = T.grid[B.qs[4 * ib + l] | (((B.qh[ib] >> (3 * l)) & 7) << 8)];
                for (int j = 0; j < 8; ++j) o[32 * ib + 8 * l + j] = dl * (((pg[j] - 1) / 2 - 1) + delta);
            }
        }
    }
}

// the stock-256 callers (the r1 gate's names, unchanged)
static void quant_row(const Tables& T, const float* x, int n, Block* y, const float* qw = nullptr) {
    quant_row_t<8>(T, x, n, y, qw);
}
static void dequant_row_ref(const Tables& T, const Block* y, float* dst, int n) {
    dequant_row_ref_t<8>(T, y, dst, n);
}

}  // namespace t4q_iq1s
