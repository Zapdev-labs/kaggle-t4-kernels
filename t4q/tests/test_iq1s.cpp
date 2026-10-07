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
    return (bad || sumi_bad || dot_bad || bad2 || sumi_bad2 || dot_bad2) != 0;
}
