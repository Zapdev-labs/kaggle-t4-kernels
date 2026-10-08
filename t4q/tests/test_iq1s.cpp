// cf-m6 r1 (CF_REQUANT.md section 7, gate 1): the iq1_s round-trip, host-only, no GPU.
// Gates: (1) the fp16 conversions (the all-65536-pattern round-trip + the RNE spot
// checks, including the subnormal and the overflow-boundary ties); (2) the tables
// internal consistency (the kmap round-trip + the byte-grid/u16-table agreement);
// (3) THE ROUND-TRIP: the packer's blocks decoded by the deq32 kernel path (the u16
// lattice walk) must be bit-identical to the reference decode (the {1,3,5} byte-grid
// walk) over the synthetic row classes (normal / all-zero / negative-dominant /
// constant); (4) the RMSE diagnostic (informational - the quality gate is the L4 A/B).
#define T4Q_HOST_SIM 1
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../src/requant.h"
#include "../src/packed.h"
#include "../src/kernels/deq.cuh"

float fp16_to_fp32(uint16_t h) { return t4q_fp16_to_fp32(h); }  // the deq.cuh host-sim shim

static uint32_t rs = 12345;
static uint32_t lcg() {
    rs = rs * 1664525u + 1013904223u;
    return rs >> 8;
}

// ---- r2 gates: the M=1 kernel arithmetic, host-twinned (build-gated locally, the L4 confirms) ----
static int dp4a(int a, int b, int c) {  // the signed-byte dp4a emulation
    for (int i = 0; i < 4; i++) {
        int8_t av, bv;
        memcpy(&av, (const char*)&a + i, 1);
        memcpy(&bv, (const char*)&b + i, 1);
        c += (int)av * (int)bv;
    }
    return c;
}

// the host twin of k_quantize_q8_K (the sequential ggml quantize_row_q8_K_ref form: the
// first-occurrence argmax, iscale = -127/maxv, MIN(127, v), the per-16 bsums, d = 1/iscale)
static void quant_q8_K_row(const float* x, int K, int8_t* qs, int16_t* bsums, float* d) {
    for (int sb = 0; sb < K / 256; sb++) {
        const float* xb = x + sb * 256;
        float amax = 0, maxv = 0;
        for (int i = 0; i < 256; i++)
            if (fabsf(xb[i]) > amax) { amax = fabsf(xb[i]); maxv = xb[i]; }  // the first occurrence wins
        if (amax == 0.f) {
            memset(qs + (size_t)sb * 256, 0, 256);
            memset(bsums + (size_t)sb * 16, 0, 32);
            d[sb] = 0.f;
            continue;
        }
        const float iscale = -127.f / maxv;
        for (int i = 0; i < 256; i++) {
            int q = t4q_iq1s::nearest_int(iscale * xb[i]);
            if (127 < q) q = 127;  // ggml's MIN(127, v)
            qs[(size_t)sb * 256 + i] = (int8_t)q;
        }
        for (int h = 0; h < 16; h++) {
            int s = 0;
            for (int i = 0; i < 16; i++) s += qs[(size_t)sb * 256 + 16 * h + i];
            bsums[(size_t)sb * 16 + h] = (int16_t)s;
        }
        d[sb] = 1.f / iscale;
    }
}

// the host twin of dot_q8k<FMT_IQ1S> (the exact kernel arithmetic, dp4a emulated), plus
// the INDEPENDENT per-elem sum (the u16 table's L fields): the two sumi paths must be
// INTEGER-IDENTICAL - the gate that pins the nibble pairing and the halves packing.
static float dot_iq1s_sim(const PackedW& W, int64_t row, int64_t g, const int8_t* xv, int16_t bs0, int16_t bs1,
                          float yd, int& sumi_mismatch) {
    const int64_t nb = W.cols / 256;
    const int64_t blk = row * nb + (g >> 3);
    const int ib = (int)(g & 7);
    const uint8_t* qs = W.codes + blk * 32 + 4 * ib;
    const int qh = ((const uint16_t*)W.hi + blk * 8)[ib];
    int sumi = 0, ref = 0;
    for (int k = 0; k < 4; ++k) {
        const int idx = qs[k] | (((qh >> (3 * k)) & 0x07) << 8);
        const int grid = t4q_iq1s_grid_gpu[idx];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int* xw = (const int*)xv;
        sumi = dp4a(grid0, xw[2 * k], sumi);
        sumi = dp4a(grid1, xw[2 * k + 1], sumi);
        const uint16_t u16e = t4q_kgrid_1bit_2048[idx];  // the independent walk
        for (int j = 0; j < 8; j++) ref += ((u16e >> (2 * j)) & 3) * (int)xv[8 * k + j];
    }
    if (sumi != ref) sumi_mismatch++;
    const float d1q = t4q_fp16_to_fp32(W.d[blk]) * (float)(((qh >> 11) & 0x0E) + 1);
    const float delta = -1.f + T4Q_IQ1S_DELTA - (float)(qh & 0x8000) * (2.f * T4Q_IQ1S_DELTA / 0x8000);
    return d1q * yd * ((float)sumi + delta * (float)((int)bs0 + (int)bs1));
}

// the host twin of k_quantize_q8_0 (the sequential quantize_row_q8_0_ref form: amax/127,
// the fp16-rounded d, q = roundf(x*id), the per-32 signed int sum)
static void quant_q8_0_row(const float* x, int K, int8_t* qs, float* xd, int* xs) {
    for (int b = 0; b < K / 32; b++) {
        const float* xb = x + b * 32;
        float amax = 0.f;
        for (int i = 0; i < 32; i++) { const float a = fabsf(xb[i]); if (a > amax) amax = a; }
        const float d = amax / 127.f;
        const float id = d ? 1.f / d : 0.f;
        int sum = 0;
        for (int i = 0; i < 32; i++) {
            int q = (int)roundf(xb[i] * id);
            qs[b * 32 + i] = (int8_t)q;
            sum += q;
        }
        xd[b] = t4q_fp16_to_fp32(t4q_fp32_to_fp16(d));  // the __float2half_rn round-trip twin
        xs[b] = sum;
    }
}

// the host twin of dot_q8_0_iq1sh (the FMT_IQ1SH M=1 dot at the q8_0 pairing), with the
// independent u16 per-elem sumi cross-check (integer-exact)
static float dot_iq1sh_sim(const PackedW& W, int64_t row, int64_t g, const int8_t* xv, float dy, int s32,
                           int& sumi_mismatch) {
    const int64_t nb = W.cols / 128;
    const int64_t blk = row * nb + (g >> 2);
    const int ib = (int)(g & 3);
    const uint8_t* qs = W.codes + blk * 16 + 4 * ib;
    const int qh = ((const uint16_t*)W.hi + blk * 4)[ib];
    int sumi = 0, ref = 0;
    for (int k = 0; k < 4; ++k) {
        const int idx = qs[k] | (((qh >> (3 * k)) & 0x07) << 8);
        const int grid = t4q_iq1s_grid_gpu[idx];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int* xw = (const int*)xv;
        sumi = dp4a(grid0, xw[2 * k], sumi);
        sumi = dp4a(grid1, xw[2 * k + 1], sumi);
        const uint16_t u16e = t4q_kgrid_1bit_2048[idx];  // the independent walk
        for (int j = 0; j < 8; j++) ref += ((u16e >> (2 * j)) & 3) * (int)xv[8 * k + j];
    }
    if (sumi != ref) sumi_mismatch++;
    const float d1q = t4q_fp16_to_fp32(W.d[blk]) * (float)(((qh >> 11) & 0x0E) + 1);
    const float delta = -1.f + T4Q_IQ1S_DELTA - (float)(qh & 0x8000) * (2.f * T4Q_IQ1S_DELTA / 0x8000);
    return d1q * dy * ((float)sumi + delta * (float)s32);
}

int main() {
    // ---- gate 1: the fp16 conversions ----
    int bad16 = 0;
    for (uint32_t h = 0; h < 65536; h++)
        if (t4q_fp32_to_fp16(t4q_fp16_to_fp32((uint16_t)h)) != (uint16_t)h) bad16++;
    const float tie_up = 1.0f + ldexpf(1.f, -11);        // the exact 1.0/next midpoint
    struct {
        float f;
        uint16_t h;
    } spot[] = {
        {1.0f, 0x3C00},
        {0.5f, 0x3800},
        {tie_up, 0x3C00},                                // the tie -> even (man 0)
        {nextafterf(tie_up, 2.f), 0x3C01},               // above the tie -> up
        {ldexpf(1.f, -24), 0x0001},                      // the min subnormal
        {ldexpf(1.f, -25), 0x0000},                      // the exact tie -> even (0)
        {ldexpf(1.f, -25) + ldexpf(1.f, -40), 0x0001},   // above the tie -> the min subnormal
        {65504.0f, 0x7BFF},                              // the max fp16
        {65519.9f, 0x7BFF},                              // below the overflow tie
        {65520.0f, 0x7C00},                              // the exact overflow tie -> even (inf)
        {ldexpf(1.f, 16), 0x7C00},                       // 65536 -> inf
    };
    for (const auto& s : spot)
        if (t4q_fp32_to_fp16(s.f) != s.h) {
            printf("fp16 spot %.9g -> %04x != %04x\n", (double)s.f, t4q_fp32_to_fp16(s.f), s.h);
            bad16++;
        }
    printf("fp16 roundtrip bad=%d\n", bad16);
    if (bad16) return 1;

    // ---- gate 2: the tables ----
    t4q_iq1s::Tables T;
    t4q_iq1s::init(T);
    for (int k = 0; k < t4q_iq1s::NG; k++) {
        uint16_t u = 0;
        for (int j = 0; j < 8; j++) {
            if (((t4q_kgrid_1bit_2048[k] >> (2 * j)) & 3) != (T.grid[k][j] - 1) / 2) {
                printf("byte/u16 mismatch k=%d j=%d\n", k, j);
                return 1;
            }
            u |= (uint16_t)(((T.grid[k][j] - 1) / 2) << (2 * j));
        }
        if (T.kmap[u] != k) {
            printf("kmap mismatch k=%d -> %d\n", k, T.kmap[u]);
            return 1;
        }
    }

    // ---- gate 3: the packer/deq32 round-trip + gate 4: the RMSE + the r2 dot gates ----
    // r2 gate A: the u32 nibble table vs the u16 table - the halves-interleave packing
    // identity must hold for every entry (the kernel's dp4a pairing rides on it).
    for (int k = 0; k < t4q_iq1s::NG; k++) {
        uint32_t C = 0;
        for (int b = 0; b < 4; b++) {
            C |= (uint32_t)((t4q_kgrid_1bit_2048[k] >> (2 * b)) & 3) << (8 * b);
            C |= (uint32_t)((t4q_kgrid_1bit_2048[k] >> (2 * (b + 4))) & 3) << (8 * b + 4);
        }
        if (t4q_iq1s_grid_gpu[k] != C) {
            printf("gpu nibble table mismatch k=%d\n", k);
            return 1;
        }
    }
    const int K = 2048, rows = 4096;  // 8 blocks per row
    std::vector<float> x(K), a(K), b(K);
    std::vector<t4q_iq1s::Block> blocks(K / 256);
    PackedW W;
    W.fmt = FMT_IQ1S;
    W.rows = 1;
    W.cols = K;
    std::vector<uint8_t> codes(K / 8), hi(K / 16);
    std::vector<uint16_t> dd(K / 256);
    W.codes = codes.data();
    W.hi = hi.data();
    W.d = dd.data();
    std::vector<int8_t> xq(K);
    std::vector<int16_t> bs(K / 16);
    std::vector<float> xqd(K / 256);
    int bad = 0, sumi_bad = 0, dot_bad = 0;
    double sq = 0, sq0 = 0, dot_err = 0, dot_l1 = 0;
    for (int r = 0; r < rows; r++) {
        const int cls = r & 7;
        for (int i = 0; i < K; i++) {
            float u = (float)((int)(lcg() >> 8) - 32768) / 32768.f;  // [-1, 1)
            if (cls < 5) x[i] = u * (0.02f + 0.15f * (r % 97) / 97.f);
            else if (cls == 5) x[i] = 0.f;                  // the eps path
            else if (cls == 6) x[i] = -fabsf(u) * 0.08f;    // the flip path
            else x[i] = 0.037f;                             // constant
        }
        t4q_iq1s::quant_row(T, x.data(), K, blocks.data());
        for (int bl = 0; bl < K / 256; bl++) {
            memcpy(W.codes + bl * 32, blocks[bl].qs, 32);
            memcpy(W.hi + bl * 16, blocks[bl].qh, 16);
            W.d[bl] = blocks[bl].d;
        }
        for (int g = 0; g < K / 32; g++) deq32<FMT_IQ1S>(W, 0, g, a.data() + g * 32);
        t4q_iq1s::dequant_row_ref(T, blocks.data(), b.data(), K);
        if (memcmp(a.data(), b.data(), K * 4)) {
            bad++;
            if (bad < 4) {
                int d0 = -1;
                for (int i = 0; i < K; i++)
                    if (a[i] != b[i]) { d0 = i; break; }
                printf("row %d MISMATCH (first diff at %d)\n", r, d0);
            }
        }
        for (int i = 0; i < K; i++) {
            sq += (double)(a[i] - x[i]) * (a[i] - x[i]);
            sq0 += (double)x[i] * x[i];
        }
        // r2 gate B: the M=1 dot twin - the exact kernel arithmetic (the nibble dp4a +
        // the bsums correction) vs the deq32-decode fp dot over the SAME x. The two sumi
        // paths (nibble-unpacked vs the u16 per-elem walk) must be INTEGER-IDENTICAL
        // (sumi_bad); the fp dot agrees to the honest q8_K activation error class
        // (checked L1-normalized, the cancellation-safe bound).
        quant_q8_K_row(x.data(), K, xq.data(), bs.data(), xqd.data());
        double dkin = 0, dref = 0, l1 = 0;
        for (int g = 0; g < K / 32; g++) {
            const int sb = g >> 3, sub = 2 * (g & 7);
            dkin += dot_iq1s_sim(W, 0, g, xq.data() + g * 32, bs[sb * 16 + sub], bs[sb * 16 + sub + 1],
                                 xqd[sb], sumi_bad);
        }
        for (int i = 0; i < K; i++) {
            dref += (double)a[i] * x[i];
            l1 += fabs((double)a[i] * x[i]);
        }
        dot_err += fabs(dkin - dref);
        dot_l1 += l1;
        if (l1 > 0 && fabs(dkin - dref) > 0.02 * l1) dot_bad++;
    }
    // ---- r3 gates: the FMT_IQ1SH half-block (the dn tiling) at the real 640-wide rows ----
    const int K2 = 640, rows2 = 2048;  // 5 blocks/row; the 256-block form cannot tile this
    std::vector<float> x2(K2), a2(K2), b2(K2);
    std::vector<t4q_iq1s::BlockT<4>> blocks2(K2 / 128);
    PackedW W2;
    W2.fmt = FMT_IQ1SH;
    W2.rows = 1;
    W2.cols = K2;
    std::vector<uint8_t> codes2(K2 / 8), hi2(K2 / 16);
    std::vector<uint16_t> dd2(K2 / 128);
    W2.codes = codes2.data();
    W2.hi = hi2.data();
    W2.d = dd2.data();
    std::vector<int8_t> xq2(K2);
    std::vector<float> xd2(K2 / 32);
    std::vector<int> xs2(K2 / 32);
    int bad2 = 0, sumi_bad2 = 0, dot_bad2 = 0;
    double sq2 = 0, sq02 = 0, dot_err2 = 0, dot_l12 = 0;
    for (int r = 0; r < rows2; r++) {
        const int cls = r & 7;
        for (int i = 0; i < K2; i++) {
            float u = (float)((int)(lcg() >> 8) - 32768) / 32768.f;
            if (cls < 5) x2[i] = u * (0.02f + 0.15f * (r % 97) / 97.f);
            else if (cls == 5) x2[i] = 0.f;
            else if (cls == 6) x2[i] = -fabsf(u) * 0.08f;
            else x2[i] = 0.037f;
        }
        t4q_iq1s::quant_row_t<4>(T, x2.data(), K2, blocks2.data());
        for (int bl = 0; bl < K2 / 128; bl++) {
            memcpy(W2.codes + bl * 16, blocks2[bl].qs, 16);
            memcpy(W2.hi + bl * 8, blocks2[bl].qh, 8);
            W2.d[bl] = blocks2[bl].d;
        }
        for (int g = 0; g < K2 / 32; g++) deq32<FMT_IQ1SH>(W2, 0, g, a2.data() + g * 32);
        t4q_iq1s::dequant_row_ref_t<4>(T, blocks2.data(), b2.data(), K2);
        if (memcmp(a2.data(), b2.data(), K2 * 4)) {
            bad2++;
            if (bad2 < 4) printf("sh row %d MISMATCH\n", r);
        }
        for (int i = 0; i < K2; i++) {
            sq2 += (double)(a2[i] - x2[i]) * (a2[i] - x2[i]);
            sq02 += (double)x2[i] * x2[i];
        }
        // the M=1 dot twin at the q8_0 pairing (s32 = the per-32 signed sum)
        quant_q8_0_row(x2.data(), K2, xq2.data(), xd2.data(), xs2.data());
        double dkin = 0, dref = 0, l1 = 0;
        for (int g = 0; g < K2 / 32; g++)
            dkin += dot_iq1sh_sim(W2, 0, g, xq2.data() + g * 32, xd2[g], xs2[g], sumi_bad2);
        for (int i = 0; i < K2; i++) {
            dref += (double)a2[i] * x2[i];
            l1 += fabs((double)a2[i] * x2[i]);
        }
        dot_err2 += fabs(dkin - dref);
        dot_l12 += l1;
        if (l1 > 0 && fabs(dkin - dref) > 0.02 * l1) dot_bad2++;
    }
    printf("iq1s round-trip rows=%d %s  rmse=%.6f rel=%.4f\n", rows, bad ? "MISMATCH" : "OK",
           sqrt(sq / ((double)rows * K)), sqrt(sq / sq0));
    printf("iq1s M=1 dot twin: sumi-int %s (%d)  dot %s (%d rows over, err=%.3e l1=%.3e rel=%.2e)\n",
           sumi_bad ? "MISMATCH" : "EXACT", sumi_bad, dot_bad ? "FAIL" : "OK", dot_bad, dot_err, dot_l1,
           dot_l1 > 0 ? dot_err / dot_l1 : 0.0);
    printf("iq1s half-block (dn 640) round-trip rows=%d %s  rmse=%.6f rel=%.4f\n", rows2, bad2 ? "MISMATCH" : "OK",
           sqrt(sq2 / ((double)rows2 * K2)), sqrt(sq2 / sq02));
    printf("iq1s half-block M=1 dot twin: sumi-int %s (%d)  dot %s (%d rows over, rel=%.2e)\n",
           sumi_bad2 ? "MISMATCH" : "EXACT", sumi_bad2, dot_bad2 ? "FAIL" : "OK", dot_bad2,
           dot_l12 > 0 ? dot_err2 / dot_l12 : 0.0);

    // ---- r4 gates: the RESIDENT VIEW strides (host_router's hit-branch math, gated) ----
    // A 2-expert expert-major plane, the expert-1 view built with EXACTLY the loader's
    // offsets (h * rows_per_expert * blocks_per_row * plane units - the d offsets in
    // ELEMENTS, the uint16_t* convention; a byte-vs-element slip here reads expert 2's
    // scales and lands far outside the dot bound), the SH dot twin run on the view vs the
    // deq32 decode of expert 1's own rows.
    int view_bad = 0, view_sumi = 0;
    {
        const int HEX = 2, HROWS = 2, K3 = 640;  // 2 experts x 2 rows of 640 (5 blocks/row)
        PackedW W3;
        W3.fmt = FMT_IQ1SH;
        W3.rows = HEX * HROWS;
        W3.cols = K3;
        std::vector<uint8_t> c3((size_t)W3.rows * (K3 / 128) * 16), h3((size_t)W3.rows * (K3 / 128) * 8);
        std::vector<uint16_t> d3((size_t)W3.rows * (K3 / 128));
        W3.codes = c3.data();
        W3.hi = h3.data();
        W3.d = d3.data();
        std::vector<t4q_iq1s::BlockT<4>> blocks3(W3.rows * (K3 / 128));
        for (int r = 0; r < W3.rows; r++) {
            for (int i = 0; i < K3; i++) {
                float u = (float)((int)(lcg() >> 8) - 32768) / 32768.f;
                x2[i] = u * (0.02f + 0.15f * (r % 97) / 97.f);
            }
            t4q_iq1s::quant_row_t<4>(T, x2.data(), K3, blocks3.data() + (size_t)r * (K3 / 128));
        }
        for (int64_t b = 0; b < (int64_t)W3.rows * (K3 / 128); b++) {
            memcpy(W3.codes + b * 16, blocks3[b].qs, 16);
            memcpy(W3.hi + b * 8, blocks3[b].qh, 8);
            W3.d[b] = blocks3[b].d;
        }
        PackedW V = W3;  // the expert-1 view: host_router's math verbatim
        V.rows = HROWS;
        V.codes = W3.codes + (size_t)1 * HROWS * (K3 / 128) * 16;
        V.hi = W3.hi + (size_t)1 * HROWS * (K3 / 128) * 8;
        V.d = W3.d + (size_t)1 * HROWS * (K3 / 128);
        std::vector<float> xv(K3), av(K3);
        std::vector<int8_t> xqv(K3);
        std::vector<float> xdv(K3 / 32);
        std::vector<int> xsv(K3 / 32);
        for (int rr = 0; rr < HROWS; rr++) {
            for (int i = 0; i < K3; i++) {
                float u = (float)((int)(lcg() >> 8) - 32768) / 32768.f;
                xv[i] = u * (0.05f + 0.1f * (rr % 7) / 7.f);
            }
            quant_q8_0_row(xv.data(), K3, xqv.data(), xdv.data(), xsv.data());
            for (int g = 0; g < K3 / 32; g++) deq32<FMT_IQ1SH>(V, rr, g, av.data() + g * 32);
            double dkin = 0, dref = 0, l1 = 0;
            for (int g = 0; g < K3 / 32; g++)
                dkin += dot_iq1sh_sim(V, rr, g, xqv.data() + g * 32, xdv[g], xsv[g], view_sumi);
            for (int i = 0; i < K3; i++) {
                dref += (double)av[i] * xv[i];
                l1 += fabs((double)av[i] * xv[i]);
            }
            if (l1 <= 0 || fabs(dkin - dref) > 0.02 * l1) view_bad++;
        }
    }
    printf("iq1s resident-view (h=1 of 2, 640) dot twin: sumi-int %s (%d)  dot %s (%d rows over)\n",
           view_sumi ? "MISMATCH" : "EXACT", view_sumi, view_bad ? "FAIL" : "OK", view_bad);

    // ---- r5 gates: the AMORTIZED verify dots (the union decode shared across the rows) ----
    // A 3-expert gu plane (FMT_IQ1S) + a 3-expert dn plane (FMT_IQ1SH), 3 draft rows with
    // OVERLAPPING picks ({0,1}, {1,2}, {0,1,2}), the union + rowmap built with vfy_window's
    // math verbatim (the dedup order, the -1 sweep), and the AMORTIZED sim - the kernel's
    // OWN structure: the 8 grid quads decoded ONCE per (union pick, 32-group), then the
    // per-row dp4a + tail against the shared quads - checked THREE ways: (a) BIT-IDENTICAL
    // to the per-(row, pick) _b form (dot_iq1s_sim / dot_iq1sh_sim over the same view +
    // activation; the same group terms in the same g-order - the claim the kernel's walk
    // makes by construction, pinned deterministically here); (b) the per-(row, pick) sumi
    // INT-exactness vs the independent u16 walk; (c) the rel bound vs the deq32 decode dot.
    // Plus the COVERAGE check: every (row, pick) lands at exactly one (union slot, k), and
    // the uidx sweep leaves no residue for the next layer's dedup.
    int am_bit = 0, am_sumi = 0, am_rel = 0, am_cov = 0;
    {
        const int NRW = 3;                    // the draft rows
        const int NEX = 3, NRG = 4, DG = 512; // gu: 3 experts x 4 rows (2*EEg) of 512 (2 super-blocks)
        const int NRD = 4, EED = 128;         // dn: 3 experts x 4 rows of 128 (1 half-block)
        const int npk[NRW] = {2, 2, 3};       // the rows' pick counts (TOPK=10 at the engine; 3 here)
        const int pks[NRW][3] = {{0, 1, -1}, {1, 2, -1}, {0, 1, 2}};
        // the planes (expert-major, the pack's own layout)
        PackedW G;
        G.fmt = FMT_IQ1S; G.rows = NEX * NRG; G.cols = DG;
        std::vector<uint8_t> gc((size_t)G.rows * (DG / 256) * 32), gh((size_t)G.rows * (DG / 256) * 16);
        std::vector<uint16_t> gd((size_t)G.rows * (DG / 256));
        G.codes = gc.data(); G.hi = gh.data(); G.d = gd.data();
        PackedW DN;
        DN.fmt = FMT_IQ1SH; DN.rows = NEX * NRD; DN.cols = EED;
        std::vector<uint8_t> dc((size_t)DN.rows * (EED / 128) * 16), dh((size_t)DN.rows * (EED / 128) * 8);
        std::vector<uint16_t> dd((size_t)DN.rows * (EED / 128));
        DN.codes = dc.data(); DN.hi = dh.data(); DN.d = dd.data();
        std::vector<float> xs(DG > EED ? DG : EED);
        std::vector<t4q_iq1s::BlockT<8>> gb((size_t)G.rows * (DG / 256));
        std::vector<t4q_iq1s::BlockT<4>> db((size_t)DN.rows * (EED / 128));
        for (int r = 0; r < (int)G.rows; r++) {
            for (int i = 0; i < DG; i++) {
                float u = (float)((int)(lcg() >> 8) - 32768) / 32768.f;
                xs[i] = u * (0.02f + 0.15f * (r % 97) / 97.f);
            }
            t4q_iq1s::quant_row_t<8>(T, xs.data(), DG, gb.data() + (size_t)r * (DG / 256));
        }
        for (int r = 0; r < (int)DN.rows; r++) {
            for (int i = 0; i < EED; i++) {
                float u = (float)((int)(lcg() >> 8) - 32768) / 32768.f;
                xs[i] = u * (0.02f + 0.13f * (r % 89) / 89.f);
            }
            t4q_iq1s::quant_row_t<4>(T, xs.data(), EED, db.data() + (size_t)r * (EED / 128));
        }
        for (int64_t b = 0; b < (int64_t)G.rows * (DG / 256); b++) {
            memcpy(G.codes + b * 32, gb[b].qs, 32); memcpy(G.hi + b * 16, gb[b].qh, 16); G.d[b] = gb[b].d;
        }
        for (int64_t b = 0; b < (int64_t)DN.rows * (EED / 128); b++) {
            memcpy(DN.codes + b * 16, db[b].qs, 16); memcpy(DN.hi + b * 8, db[b].qh, 8); DN.d[b] = db[b].d;
        }
        // the union + rowmap (vfy_window's math verbatim)
        int uidx[NEX], uids[NRW * 3], nu = 0, rowmap[NRW * 3][NRW];
        for (int e = 0; e < NEX; e++) uidx[e] = -1;
        memset(rowmap, -1, sizeof(rowmap));
        for (int r = 0; r < NRW; r++)
            for (int k = 0; k < npk[r]; k++) {
                const int e = pks[r][k];
                if (uidx[e] < 0) { uidx[e] = nu; uids[nu++] = e; }
            }
        for (int r = 0; r < NRW; r++)
            for (int k = 0; k < npk[r]; k++) rowmap[uidx[pks[r][k]]][r] = k;
        for (int r = 0; r < NRW; r++)  // the coverage check: every pick at exactly one (slot, k)
            for (int k = 0; k < npk[r]; k++)
                if (rowmap[uidx[pks[r][k]]][r] != k) am_cov++;
        for (int i = 0; i < nu; i++) uidx[uids[i]] = -1;  // the sweep (the next layer's dedup reset)
        for (int e = 0; e < NEX; e++)
            if (uidx[e] != -1) am_cov++;  // the sweep residue (all -1 after)
        // the rows' activations: gu q8_K over DG, dn q8_0 over EED (SEPARATE originals -
        // the deq32 reference dots read each side's own pre-quantize floats)
        std::vector<float> a0g((size_t)NRW * DG), a0d((size_t)NRW * EED);
        std::vector<int8_t> gq((size_t)NRW * DG);
        std::vector<int16_t> gbs((size_t)NRW * DG / 16);
        std::vector<float> gyd((size_t)NRW * DG / 256);
        std::vector<int8_t> dq((size_t)NRW * EED);
        std::vector<float> dxd((size_t)NRW * EED / 32);
        std::vector<int> dxs((size_t)NRW * EED / 32);
        for (int r = 0; r < NRW; r++) {
            for (int i = 0; i < DG; i++) {
                float u = (float)((int)(lcg() >> 8) - 32768) / 32768.f;
                a0g[(size_t)r * DG + i] = u * (0.03f + 0.11f * (r % 5) / 5.f);
            }
            quant_q8_K_row(a0g.data() + (size_t)r * DG, DG, gq.data() + (size_t)r * DG,
                           gbs.data() + (size_t)r * (DG / 16), gyd.data() + (size_t)r * (DG / 256));
            for (int i = 0; i < EED; i++) {
                float u = (float)((int)(lcg() >> 8) - 32768) / 32768.f;
                a0d[(size_t)r * EED + i] = u * (0.05f + 0.09f * (r % 7) / 7.f);
            }
            quant_q8_0_row(a0d.data() + (size_t)r * EED, EED, dq.data() + (size_t)r * EED,
                           dxd.data() + (size_t)r * (EED / 32), dxs.data() + (size_t)r * (EED / 32));
        }
        // (a) the BIT-IDENTITY + (b) the int sumi: the amortized walk vs the _b walk
        double am_err = 0, am_l1 = 0;
        std::vector<float> dec(DG);
        for (int u = 0; u < nu; u++) {
            const int e = uids[u];
            PackedW Vg = G;  // the union view (vfy_window's offsets verbatim)
            Vg.rows = NRG;
            Vg.codes = G.codes + (size_t)e * NRG * (DG / 256) * 32;
            Vg.hi = G.hi + (size_t)e * NRG * (DG / 256) * 16;
            Vg.d = G.d + (size_t)e * NRG * (DG / 256);
            PackedW Vd = DN;
            Vd.rows = NRD;
            Vd.codes = DN.codes + (size_t)e * NRD * (EED / 128) * 16;
            Vd.hi = DN.hi + (size_t)e * NRD * (EED / 128) * 8;
            Vd.d = DN.d + (size_t)e * NRD * (EED / 128);
            for (int i = 0; i < NRG; i++) {
                double am[NRW] = {0, 0, 0}, bsm[NRW][3] = {{0}, {0}, {0}};
                for (int g = 0; g < DG / 32; g++) {
                    // the decode ONCE (the kernel's structure: the quads held, not consumed)
                    const int64_t blk = (int64_t)i * (DG / 256) + (g >> 3);
                    const int ib = g & 7;
                    const uint8_t* qs = Vg.codes + blk * 32 + 4 * ib;
                    const int qh = ((const uint16_t*)Vg.hi + blk * 8)[ib];
                    int q8[8];
                    for (int k = 0; k < 4; k++) {
                        const int idx = qs[k] | (((qh >> (3 * k)) & 7) << 8);
                        const int grid = t4q_iq1s_grid_gpu[idx];
                        q8[2 * k] = (grid >> 0) & 0x0F0F0F0F;
                        q8[2 * k + 1] = (grid >> 4) & 0x0F0F0F0F;
                    }
                    const float d1q = t4q_fp16_to_fp32(Vg.d[blk]) * (float)(((qh >> 11) & 0x0E) + 1);
                    const float delta =
                        -1.f + T4Q_IQ1S_DELTA - (float)(qh & 0x8000) * (2.f * T4Q_IQ1S_DELTA / 0x8000);
                    // the per-row dots against the shared quads (the masked row loop)
                    for (int r = 0; r < NRW; r++) {
                        if (rowmap[u][r] < 0) continue;
                        const int8_t* xv = gq.data() + (size_t)r * DG + g * 32;
                        int sumi = 0, ref = 0;
                        for (int k = 0; k < 4; k++) {
                            sumi = dp4a(q8[2 * k], *(const int*)(xv + 8 * k), sumi);
                            sumi = dp4a(q8[2 * k + 1], *(const int*)(xv + 8 * k + 4), sumi);
                            const uint16_t u16e = t4q_kgrid_1bit_2048[qs[k] | (((qh >> (3 * k)) & 7) << 8)];
                            for (int j = 0; j < 8; j++) ref += ((u16e >> (2 * j)) & 3) * (int)xv[8 * k + j];
                        }
                        if (sumi != ref) am_sumi++;
                        const int sb = g >> 3, sub = 2 * (g & 7);
                        am[r] += d1q * gyd[(size_t)r * (DG / 256) + sb] *
                                 ((float)sumi +
                                  delta * (float)((int)gbs[(size_t)r * (DG / 16) + sb * 16 + sub] +
                                                  (int)gbs[(size_t)r * (DG / 16) + sb * 16 + sub + 1]));
                    }
                }
                // the _b side: the SAME view + activation through the per-(row, pick) twin
                for (int r = 0; r < NRW; r++) {
                    const int k = rowmap[u][r];
                    if (k < 0) continue;
                    for (int g = 0; g < DG / 32; g++) {
                        const int sb = g >> 3, sub = 2 * (g & 7);
                        bsm[r][k] += dot_iq1s_sim(Vg, i, g, gq.data() + (size_t)r * DG + g * 32,
                                                  gbs[(size_t)r * (DG / 16) + sb * 16 + sub],
                                                  gbs[(size_t)r * (DG / 16) + sb * 16 + sub + 1],
                                                  gyd[(size_t)r * (DG / 256) + sb], am_sumi);
                    }
                }
                // (a) the bit check + (c) the deq32 reference
                for (int g = 0; g < DG / 32; g++) deq32<FMT_IQ1S>(Vg, i, g, dec.data() + g * 32);
                for (int r = 0; r < NRW; r++) {
                    const int k = rowmap[u][r];
                    if (k < 0) continue;
                    if (am[r] != bsm[r][k]) am_bit++;
                    double dref = 0, l1 = 0;
                    for (int ii = 0; ii < DG; ii++) {
                        dref += (double)dec[ii] * a0g[(size_t)r * DG + ii];
                        l1 += fabs((double)dec[ii] * a0g[(size_t)r * DG + ii]);
                    }
                    am_err += fabs(am[r] - dref);
                    am_l1 += l1;
                    if (l1 > 0 && fabs(am[r] - dref) > 0.02 * l1) am_rel++;
                }
            }
            // the dn twin at the q8_0 pairing (the s32 correction)
            for (int i = 0; i < NRD; i++) {
                double am[NRW] = {0, 0, 0}, bsm[NRW][3] = {{0}, {0}, {0}};
                for (int g = 0; g < EED / 32; g++) {
                    const int64_t blk = (int64_t)i * (EED / 128) + (g >> 2);
                    const int ib = g & 3;
                    const uint8_t* qs = Vd.codes + blk * 16 + 4 * ib;
                    const int qh = ((const uint16_t*)Vd.hi + blk * 4)[ib];
                    int q8[8];
                    for (int k = 0; k < 4; k++) {
                        const int idx = qs[k] | (((qh >> (3 * k)) & 7) << 8);
                        const int grid = t4q_iq1s_grid_gpu[idx];
                        q8[2 * k] = (grid >> 0) & 0x0F0F0F0F;
                        q8[2 * k + 1] = (grid >> 4) & 0x0F0F0F0F;
                    }
                    const float d1q = t4q_fp16_to_fp32(Vd.d[blk]) * (float)(((qh >> 11) & 0x0E) + 1);
                    const float delta =
                        -1.f + T4Q_IQ1S_DELTA - (float)(qh & 0x8000) * (2.f * T4Q_IQ1S_DELTA / 0x8000);
                    for (int r = 0; r < NRW; r++) {
                        if (rowmap[u][r] < 0) continue;
                        const int8_t* xv = dq.data() + (size_t)r * EED + g * 32;
                        int sumi = 0, ref = 0;
                        for (int k = 0; k < 4; k++) {
                            sumi = dp4a(q8[2 * k], *(const int*)(xv + 8 * k), sumi);
                            sumi = dp4a(q8[2 * k + 1], *(const int*)(xv + 8 * k + 4), sumi);
                            const uint16_t u16e = t4q_kgrid_1bit_2048[qs[k] | (((qh >> (3 * k)) & 7) << 8)];
                            for (int j = 0; j < 8; j++) ref += ((u16e >> (2 * j)) & 3) * (int)xv[8 * k + j];
                        }
                        if (sumi != ref) am_sumi++;
                        am[r] += d1q * dxd[(size_t)r * (EED / 32) + g] *
                                 ((float)sumi + delta * (float)dxs[(size_t)r * (EED / 32) + g]);
                    }
                }
                for (int r = 0; r < NRW; r++) {
                    const int k = rowmap[u][r];
                    if (k < 0) continue;
                    for (int g = 0; g < EED / 32; g++)
                        bsm[r][k] += dot_iq1sh_sim(Vd, i, g, dq.data() + (size_t)r * EED + g * 32,
                                                   dxd[(size_t)r * (EED / 32) + g], dxs[(size_t)r * (EED / 32) + g],
                                                   am_sumi);
                }
                for (int g = 0; g < EED / 32; g++) deq32<FMT_IQ1SH>(Vd, i, g, dec.data() + g * 32);
                for (int r = 0; r < NRW; r++) {
                    const int k = rowmap[u][r];
                    if (k < 0) continue;
                    if (am[r] != bsm[r][k]) am_bit++;
                    double dref = 0, l1 = 0;
                    for (int ii = 0; ii < EED; ii++) {
                        dref += (double)dec[ii] * a0d[(size_t)r * EED + ii];
                        l1 += fabs((double)dec[ii] * a0d[(size_t)r * EED + ii]);
                    }
                    am_err += fabs(am[r] - dref);
                    am_l1 += l1;
                    if (l1 > 0 && fabs(am[r] - dref) > 0.02 * l1) am_rel++;
                }
            }
        }
        printf("iq1s amortized vfy twin (gu+dn, 3 rows, union %d): bit-vs-_b %s (%d)  sumi-int %s (%d)  "
               "dot %s (%d over, rel=%.2e)  coverage %s (%d)\n",
               nu, am_bit ? "MISMATCH" : "IDENTICAL", am_bit, am_sumi ? "MISMATCH" : "EXACT", am_sumi,
               am_rel ? "FAIL" : "OK", am_rel, am_l1 > 0 ? am_err / am_l1 : 0.0, am_cov ? "FAIL" : "OK", am_cov);
    }

    // ---- r6a gates: the SPLIT-PLANE offsets (the loader's by-ID half reads, gated) ----
    // A 4-expert SH plane; the owner-1 half (the experts [2,4)) read as ONE contiguous
    // byte range per plane - the loader's split form VERBATIM (base + o*HROWS*(K/128)*16
    // with o = NE/2 = 2, the expert-major halves) - then expert 3's view INTO the half
    // (le = 1, the r4 view math), the dot twin on the view vs the deq32 decode of expert
    // 3's OWN rows from the FULL plane. A half-offset slip reads expert 1's rows (the
    // symmetric neighbor) and the dot lands far outside the bound.
    int spl_bad = 0, spl_sumi = 0;
    {
        const int HEX6 = 4, HROWS6 = 2, K6 = 640;
        PackedW F6;
        F6.fmt = FMT_IQ1SH;
        F6.rows = HEX6 * HROWS6;
        F6.cols = K6;
        std::vector<uint8_t> c6((size_t)F6.rows * (K6 / 128) * 16), h6((size_t)F6.rows * (K6 / 128) * 8);
        std::vector<uint16_t> d6((size_t)F6.rows * (K6 / 128));
        F6.codes = c6.data();
        F6.hi = h6.data();
        F6.d = d6.data();
        std::vector<float> x6(K6);
        std::vector<t4q_iq1s::BlockT<4>> b6((size_t)F6.rows * (K6 / 128));
        for (int r = 0; r < (int)F6.rows; r++) {
            for (int i = 0; i < K6; i++) {
                float u = (float)((int)(lcg() >> 8) - 32768) / 32768.f;
                x6[i] = u * (0.02f + 0.15f * (r % 97) / 97.f);
            }
            t4q_iq1s::quant_row_t<4>(T, x6.data(), K6, b6.data() + (size_t)r * (K6 / 128));
        }
        for (int64_t b = 0; b < (int64_t)F6.rows * (K6 / 128); b++) {
            memcpy(F6.codes + b * 16, b6[b].qs, 16);
            memcpy(F6.hi + b * 8, b6[b].qh, 8);
            F6.d[b] = b6[b].d;
        }
        // the owner-1 HALF plane: the contiguous ranges at the split offsets (the loader's
        // split read arithmetic verbatim - o = NE/2 = 2)
        PackedW Hf;
        Hf.fmt = FMT_IQ1SH;
        Hf.rows = (HEX6 / 2) * HROWS6;
        Hf.cols = K6;
        const int o6 = HEX6 / 2;  // the half offset in experts (the loader's o = g * NE/2, g = 1)
        const size_t HC = (size_t)o6 * HROWS6 * (K6 / 128) * 16, HH = (size_t)o6 * HROWS6 * (K6 / 128) * 8,
                     HD = (size_t)o6 * HROWS6 * (K6 / 128);
        std::vector<uint8_t> hc6((size_t)Hf.rows * (K6 / 128) * 16), hh6((size_t)Hf.rows * (K6 / 128) * 8);
        std::vector<uint16_t> hd6((size_t)Hf.rows * (K6 / 128));
        memcpy(hc6.data(), F6.codes + HC, hc6.size());  // the one-range copy (the loader's fseek+fread)
        memcpy(hh6.data(), F6.hi + HH, hh6.size());
        memcpy(hd6.data(), F6.d + HD, hd6.size() * 2);
        Hf.codes = hc6.data();
        Hf.hi = hh6.data();
        Hf.d = hd6.data();
        // expert 3's view INTO the half (le = 1, the r4 view math)
        PackedW V6 = Hf;
        V6.rows = HROWS6;
        V6.codes = Hf.codes + (size_t)1 * HROWS6 * (K6 / 128) * 16;
        V6.hi = Hf.hi + (size_t)1 * HROWS6 * (K6 / 128) * 8;
        V6.d = Hf.d + (size_t)1 * HROWS6 * (K6 / 128);
        std::vector<float> xv6(K6), av6(K6);
        std::vector<int8_t> xqv6(K6);
        std::vector<float> xdv6(K6 / 32);
        std::vector<int> xsv6(K6 / 32);
        for (int rr = 0; rr < HROWS6; rr++) {
            for (int i = 0; i < K6; i++) {
                float u = (float)((int)(lcg() >> 8) - 32768) / 32768.f;
                xv6[i] = u * (0.05f + 0.1f * (rr % 7) / 7.f);
            }
            quant_q8_0_row(xv6.data(), K6, xqv6.data(), xdv6.data(), xsv6.data());
            double dkin = 0, dref = 0, l1 = 0;
            for (int g = 0; g < K6 / 32; g++)
                dkin += dot_iq1sh_sim(V6, rr, g, xqv6.data() + g * 32, xdv6[g], xsv6[g], spl_sumi);
            // the reference: expert 3's OWN rows decoded from the FULL plane
            for (int g = 0; g < K6 / 32; g++) deq32<FMT_IQ1SH>(F6, (size_t)3 * HROWS6 + rr, g, av6.data() + g * 32);
            for (int i = 0; i < K6; i++) {
                dref += (double)av6[i] * xv6[i];
                l1 += fabs((double)av6[i] * xv6[i]);
            }
            if (l1 <= 0 || fabs(dkin - dref) > 0.02 * l1) spl_bad++;
        }
    }
    printf("iq1s split-plane (owner 1 of 2, expert 3, 640) dot twin: sumi-int %s (%d)  dot %s (%d rows over)\n",
           spl_sumi ? "MISMATCH" : "EXACT", spl_sumi, spl_bad ? "FAIL" : "OK", spl_bad);
    // ---- r6b gates: the SPLIT-MOE COMBINE (the owner dispatch + the compact partials +
    // the both-ways final vs the moe_out's own single-loop form) ----
    // The REAL id space (e in [0,512), owner = e>>8, le = e&255) - the same dispatch math
    // host_router's split compose runs: each pick lands in its side's COMPACT slot with
    // its compact we, the per-side partial sums the side's OWN rows in slot order, and
    // the final (p0 + p1 + sigmoid(gate)*ysh) re-expresses the moe_out's arithmetic.
    // The ONLY difference vs the direct form is the ADD ORDER (the ~1e-6 fp32
    // reassociation class - bounded honestly below, the requant's own error dwarfs it);
    // a lost/doubled pick or a compact-slot slip lands ~0.1-1, far outside the bound.
    int tp_bad = 0, tp_n0 = 0, tp_n1 = 0;
    double tp_maxd = 0;
    {
        const int D7 = 64;
        const int TK7 = 10;  // the engine's pick count (10) is not in this TU's includes
        int eid7[TK7], n7[2] = {0, 0};
        float we7[TK7];
        for (int k = 0; k < TK7; k++) {  // 10 distinct picks, BOTH owners forced (a mixed
            // split - a pure-lcg draw can land all 10 on one side and leave side 1 unexercised)
            eid7[k] = ((k & 1) ? 256 : 0) + (int)(lcg() >> 16) % 256;
            for (int j = 0; j < k; j++)
                if (eid7[j] == eid7[k]) eid7[k] = (eid7[k] + 1) % 512;
            we7[k] = 0.05f + 0.9f * (lcg() % 1000) / 1000.f;
        }
        int side7[TK7], slot7[TK7];  // the split dispatch (host_router's compose loop)
        for (int k = 0; k < TK7; k++) {
            const int g = eid7[k] >> 8;
            side7[k] = g;
            slot7[k] = n7[g]++;
        }
        tp_n0 = n7[0];
        tp_n1 = n7[1];
        if (n7[0] + n7[1] != TK7) tp_bad++;  // the dispatch's structure: no pick lost or doubled
        // the pick rows (the k-th pick's dn output): the full plane + the per-side COMPACT
        // planes carry the SAME values at the dispatch's mapping
        std::vector<float> ye_full((size_t)TK7 * D7), ye0((size_t)TK7 * D7), ye1((size_t)TK7 * D7);
        float wec0[TK7], wec1[TK7];
        for (int k = 0; k < TK7; k++)
            for (int c = 0; c < D7; c++)
                ye_full[(size_t)k * D7 + c] = (float)((int)(lcg() >> 8) - 32768) / 32768.f * 0.7f;
        for (int k = 0; k < TK7; k++) {
            const int g = side7[k], kk = slot7[k];
            for (int c = 0; c < D7; c++) (g ? ye1 : ye0)[(size_t)kk * D7 + c] = ye_full[(size_t)k * D7 + c];
            (g ? wec1 : wec0)[kk] = we7[k];
        }
        std::vector<float> ysh7(D7);  // the shared expert's tail + its sigmoid gate
        for (int c = 0; c < D7; c++) ysh7[c] = (float)((int)(lcg() >> 8) - 32768) / 32768.f * 0.4f;
        const float gg7 = (float)((int)(lcg() >> 8) - 32768) / 32768.f;
        const float sig7 = 1.0f / (1.0f + expf(-gg7));
        for (int c = 0; c < D7; c++) {  // the kernels' own op order on both forms
            float p0v = 0, p1v = 0, direct = 0;
            for (int k = 0; k < n7[0]; k++) p0v += wec0[k] * ye0[(size_t)k * D7 + c];
            for (int k = 0; k < n7[1]; k++) p1v += wec1[k] * ye1[(size_t)k * D7 + c];
            for (int k = 0; k < TK7; k++) direct += we7[k] * ye_full[(size_t)k * D7 + c];
            const float splitv = p0v + p1v + sig7 * ysh7[c];
            const float moeoutv = direct + sig7 * ysh7[c];
            const double dd = fabs((double)splitv - moeoutv);
            if (dd > tp_maxd) tp_maxd = dd;
            if (dd > 1e-4) tp_bad++;
        }
    }
    printf("iq1s split-moe combine (owner %d/%d of 512, 10 picks, D 64) vs moe_out: %s (max |d| %.3e)\n", tp_n0,
           tp_n1, tp_bad ? "FAIL" : "OK", tp_maxd);
    // ---- r6c gates: the draft's Q8_0 SPLIT VIEWS (the owner halves at the packed-Q8_0
    // strides, gated) ----
    // A 4-expert Q8_0 pair; the owner-1 half held at the LOCAL row offsets (the loader's
    // split repack: (e0 - NE/2)*rows into the side's own plane), then expert 3's view
    // INTO the half at le = 1 (the emission's compose math: gu codes le*2*EE*D / d
    // le*2*EE*(D/32); dn codes le*D*EE / d le*D*(EE/32)) vs the FULL plane's expert-3
    // rows - BYTE-IDENTICAL (a memcpy twin; an offset slip reads the symmetric neighbor
    // expert 1 and the compare fails on every row).
    int dvp_bad = 0;
    {
        const int HEX8 = 4, EE8 = 32, D8 = 64;  // small twins of EE=640 / D=2560
        const int GROWS = 2 * EE8, DROWS = D8;  // the gu / dn rows per expert
        PackedW F8;                              // the FULL plane (the trunk's form: e*rows offsets)
        F8.fmt = FMT_Q8;
        F8.rows = HEX8 * GROWS;
        F8.cols = D8;
        std::vector<uint8_t> c8((size_t)F8.rows * D8);
        std::vector<uint16_t> d8((size_t)F8.rows * (D8 / 32));  // the fp16-scale words
        for (size_t i = 0; i < c8.size(); i++) c8[i] = (uint8_t)(lcg() >> 24);
        for (size_t i = 0; i < d8.size(); i++) d8[i] = (uint16_t)(lcg() >> 16);
        F8.codes = c8.data();
        F8.d = d8.data();
        // the owner-1 HALF (the loader's split repack form: the local row offsets)
        PackedW H8 = F8;
        H8.rows = (HEX8 / 2) * GROWS;
        const size_t hoff = (size_t)(HEX8 / 2) * GROWS;  // the half's first row in the full plane
        H8.codes = F8.codes + hoff * D8;
        H8.d = F8.d + hoff * (D8 / 32);
        // expert 3's view INTO the half (le = 1, the emission's compose math)
        PackedW V8 = H8;
        V8.rows = GROWS;
        V8.codes = H8.codes + (size_t)1 * GROWS * D8;
        V8.d = H8.d + (size_t)1 * GROWS * (D8 / 32);
        for (int rr = 0; rr < GROWS; rr++) {  // byte-identity vs the full plane's expert 3
            if (memcmp(V8.codes + (size_t)rr * D8, F8.codes + ((size_t)3 * GROWS + rr) * D8, D8) != 0) dvp_bad++;
            if (memcmp(V8.d + (size_t)rr * (D8 / 32), F8.d + ((size_t)3 * GROWS + rr) * (D8 / 32), (D8 / 32) * 2) !=
                       0)
                dvp_bad++;
        }
        // the dn pair at its own strides (rows D, cols EE)
        PackedW FD;
        FD.fmt = FMT_Q8;
        FD.rows = HEX8 * DROWS;
        FD.cols = EE8;
        std::vector<uint8_t> cd((size_t)FD.rows * EE8);
        std::vector<uint16_t> dd((size_t)FD.rows * (EE8 / 32));
        for (size_t i = 0; i < cd.size(); i++) cd[i] = (uint8_t)(lcg() >> 16);
        for (size_t i = 0; i < dd.size(); i++) dd[i] = (uint16_t)(lcg() >> 8);
        FD.codes = cd.data();
        FD.d = dd.data();
        PackedW HD = FD;  // the owner-1 half + expert 3's view (le = 1)
        HD.rows = (HEX8 / 2) * DROWS;
        const size_t doff = (size_t)(HEX8 / 2) * DROWS;
        HD.codes = FD.codes + doff * EE8;
        HD.d = FD.d + doff * (EE8 / 32);
        PackedW VD = HD;
        VD.rows = DROWS;
        VD.codes = HD.codes + (size_t)1 * DROWS * EE8;
        VD.d = HD.d + (size_t)1 * DROWS * (EE8 / 32);
        for (int rr = 0; rr < DROWS; rr++) {
            if (memcmp(VD.codes + (size_t)rr * EE8, FD.codes + ((size_t)3 * DROWS + rr) * EE8, EE8) != 0)
                dvp_bad++;
            if (memcmp(VD.d + (size_t)rr * (EE8 / 32), FD.d + ((size_t)3 * DROWS + rr) * (EE8 / 32),
                       (EE8 / 32) * 2) != 0)
                dvp_bad++;
        }
    }
    printf("draft q8_0 split-views (owner 1 of 2, expert 3, gu+dn): %s (byte-twin %d rows)\n",
           dvp_bad ? "FAIL" : "OK", 2 * (2 * 32 + 64));
    // ---- r6c part 2 gates: the VERIFY'S SPLIT-MOE (the sub-unions + the per-side
    // rowmaps + the per-row owned k-lists + the gather partials + the both-ways finals
    // vs the r5 single-side form) ----
    // The REAL id space + the r5 rowmap semantics (rowmap[u][r] = the row's pick index
    // k, -1 = not picked), 3 rows x 10 picks, BOTH owners forced. THE COVERAGE: every
    // (row, pick k) lands in EXACTLY ONE side's rowmap slot + one ks entry; the two
    // sub-unions together cover the WHOLE deduped union. THE ARITHMETIC: the per-row
    // gather partials over the sides' k-lists + the final (p0 + p1 + sigmoid*ysh) vs
    // the moe_out's own single-loop form over the same values (the add-order class).
    int vsp_bad = 0, vsp_n0 = 0, vsp_n1 = 0, vsp_nu = 0;
    double vsp_maxd = 0;
    {
        const int NRV = 3, TKV = 10, D9 = 64, MAXRV = 8;
        int eid9[NRV][TKV];
        float we9[NRV][TKV];
        for (int r = 0; r < NRV; r++)
            for (int k = 0; k < TKV; k++) {  // distinct picks per row, both owners
                eid9[r][k] = ((k & 1) ? 256 : 0) + (int)(lcg() >> 16) % 256;
                for (int j = 0; j < k; j++)
                    if (eid9[r][j] == eid9[r][k]) eid9[r][k] = (eid9[r][k] + 1) % 512;
                we9[r][k] = 0.05f + 0.9f * (lcg() % 1000) / 1000.f;
            }
        // the union dedup (the r5 form) -> uids[nu] + the union slot per expert
        int uids9[64], nu9 = 0, uidx9[512];
        memset(uidx9, -1, sizeof(uidx9));
        for (int r = 0; r < NRV; r++)
            for (int k = 0; k < TKV; k++) {
                const int e = eid9[r][k];
                if (uidx9[e] < 0) { uidx9[e] = nu9; uids9[nu9++] = e; }
            }
        // the split compose (vfy_window's split branch): the sub-unions + the per-side
        // rowmaps + the per-row owned k-lists
        int uloc[64], nus[2] = {0, 0};
        int rm0[64][MAXRV], rm1[64][MAXRV];
        memset(rm0, -1, sizeof(rm0));
        memset(rm1, -1, sizeof(rm1));
        int ks0[NRV][TKV], ks1[NRV][TKV], nk0[NRV] = {0}, nk1[NRV] = {0};
        for (int u = 0; u < nu9; u++) uloc[u] = nus[uids9[u] >> 8]++;
        for (int r = 0; r < NRV; r++)
            for (int k = 0; k < TKV; k++) {
                const int e = eid9[r][k], g = e >> 8, ug = uloc[uidx9[e]];
                (g ? rm1 : rm0)[ug][r] = k;
                (g ? ks1 : ks0)[r][(g ? nk1 : nk0)[r]++] = k;
            }
        vsp_n0 = nus[0];
        vsp_n1 = nus[1];
        vsp_nu = nu9;
        // THE COVERAGE: each (row, k) in exactly one rowmap slot + one ks entry; the
        // sub-unions cover the whole union; the counts add up
        for (int r = 0; r < NRV; r++)
            for (int k = 0; k < TKV; k++) {
                int hits = 0;
                for (int u = 0; u < nus[0]; u++)
                    if (rm0[u][r] == k) hits++;
                for (int u = 0; u < nus[1]; u++)
                    if (rm1[u][r] == k) hits++;
                if (hits != 1) vsp_bad++;  // a lost or doubled pick
            }
        for (int r = 0; r < NRV; r++)
            if (nk0[r] + nk1[r] != TKV) vsp_bad++;
        for (int r = 0; r < NRV; r++) {  // the ks entries are the row's OWN pick indices
            int seen[TKV] = {0};
            for (int j = 0; j < nk0[r]; j++) seen[ks0[r][j]]++;
            for (int j = 0; j < nk1[r]; j++) seen[ks1[r][j]]++;
            for (int k = 0; k < TKV; k++)
                if (seen[k] != 1) vsp_bad++;
        }
        if (nus[0] + nus[1] != nu9) vsp_bad++;  // the sub-unions cover the union
        // THE ARITHMETIC: the per-row ye planes (the pick-slot layout) + the shared tail
        std::vector<float> ye9((size_t)NRV * TKV * D9), ysh9((size_t)NRV * D9);
        for (size_t i = 0; i < ye9.size(); i++) ye9[i] = (float)((int)(lcg() >> 8) - 32768) / 32768.f * 0.7f;
        for (size_t i = 0; i < ysh9.size(); i++) ysh9[i] = (float)((int)(lcg() >> 8) - 32768) / 32768.f * 0.4f;
        for (int r = 0; r < NRV; r++) {
            const float gg = (float)((int)(lcg() >> 8) - 32768) / 32768.f;
            const float sig = 1.0f / (1.0f + expf(-gg));
            for (int c = 0; c < D9; c++) {
                const float* ye_r = ye9.data() + (size_t)r * TKV * D9;
                float p0 = 0, p1 = 0, direct = 0;
                for (int j = 0; j < nk0[r]; j++) {  // the gather partials (the k-lists)
                    const int k = ks0[r][j];
                    p0 += we9[r][k] * ye_r[(size_t)k * D9 + c];
                }
                for (int j = 0; j < nk1[r]; j++) {
                    const int k = ks1[r][j];
                    p1 += we9[r][k] * ye_r[(size_t)k * D9 + c];
                }
                for (int k = 0; k < TKV; k++) direct += we9[r][k] * ye_r[(size_t)k * D9 + c];
                const float splitv = p0 + p1 + sig * ysh9[(size_t)r * D9 + c];
                const float moeoutv = direct + sig * ysh9[(size_t)r * D9 + c];
                const double dd = fabs((double)splitv - moeoutv);
                if (dd > vsp_maxd) vsp_maxd = dd;
                if (dd > 1e-4) vsp_bad++;
            }
        }
    }
    printf("verify split-moe (3 rows, sub-unions %d+%d of %d, gather partials): %s (coverage %d, max |d| %.3e)\n",
           vsp_n0, vsp_n1, vsp_nu, vsp_bad ? "FAIL" : "OK", vsp_bad, vsp_maxd);
    return (bad || sumi_bad || dot_bad || bad2 || sumi_bad2 || dot_bad2 || view_bad || view_sumi || am_bit ||
            am_sumi || am_rel || am_cov || spl_bad || spl_sumi || tp_bad || dvp_bad || vsp_bad) != 0;
}
