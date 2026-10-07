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

    // ---- gate 3: the packer/deq32 round-trip + gate 4: the RMSE ----
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
    int bad = 0;
    double sq = 0, sq0 = 0;
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
    }
    printf("iq1s round-trip rows=%d %s  rmse=%.6f rel=%.4f\n", rows, bad ? "MISMATCH" : "OK",
           sqrt(sq / ((double)rows * K)), sqrt(sq / sq0));
    return bad != 0;
}
