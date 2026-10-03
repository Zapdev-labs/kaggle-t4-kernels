// Host-only check of rot::deq32f against gemv::dequant_row_host through the decode layout (P4, P4M, K5; rpl 2/4).
// Build/run (no GPU needed): nvcc -std=c++17 -arch=sm_75 -o test_rot_deq test_rot_deq.cu && ./test_rot_deq
#include "../src/kernels/rot.cuh"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
using namespace t4q;
using namespace t4q::gemv;
int main() {
    std::mt19937 rng(7);
    int bad = 0;
    for (int fmt : {FAST_P4, FAST_P4M, FAST_K5})
        for (int rpl : {2, 4}) {
            const int N = 32, K = 1024;
            const size_t rb = src_row_bytes(fmt, K);
            std::vector<uint8_t> src(rb * N);
            for (auto& b : src) b = rng() & 0xff;
            const int be = src_block_elems(fmt), bb = src_block_bytes(fmt);
            for (int r = 0; r < N; ++r)
                for (int b = 0; b < K / be; ++b) {
                    uint8_t* blk = src.data() + (size_t)r * rb + (size_t)b * bb;
                    uint16_t d = f2h_host(0.002f + 0.02f * (rng() % 1000) / 1000.f);
                    memcpy(blk, &d, 2);
                    if (fmt != FAST_P4) { uint16_t m = f2h_host(0.001f + 0.01f * (rng() % 1000) / 1000.f); memcpy(blk + 2, &m, 2); }
                }
            Layout L = make_layout(fmt, N, K, rpl, 1);
            std::vector<uint8_t> P(L.bytes);
            repack_host(L, src.data(), P.data());
            const uint8_t* codes = P.data() + L.off_codes;
            const uint8_t* sc = P.data() + L.off_sc;
            const uint8_t* qh = P.data() + L.off_qh;
            const uint16_t* dd = (const uint16_t*)(P.data() + L.off_d);
            double maxe = 0;
            std::vector<float> ref(K);
            for (int R = 0; R < N; ++R) {
                dequant_row_host(fmt, src.data() + (size_t)R * rb, K, ref.data());
                for (int kb = 0; kb < K / 32; ++kb) {
                    const int lt = R / (2 * rpl), w = R % (2 * rpl), h = w / rpl, r = w % rpl;
                    const int c = kb >> 4, j = kb & 15, lane = h * 16 + j;
                    const size_t tc = tc_index(lt, c, K >> 9, L.ntiles, L.cm);
                    uint32_t q[4];
                    memcpy(q, codes + (tc * rpl + r) * 512 + lane * 16, 16);
                    uint32_t s0 = 0, s1 = 0, hb = 0;
                    if (fmt == FAST_K5) {
                        uint16_t t; memcpy(&t, sc + ((tc * 32 + lane) * rpl + r) * 2, 2); s0 = t;
                        memcpy(&s1, (const uint8_t*)dd + (((tc * 2 + h) * 2 + (j >> 3)) * rpl + r) * 4, 4);
                        memcpy(&hb, qh + (tc * rpl + r) * 128 + lane * 4, 4);
                    } else {
                        s0 = dd[(tc * 32 + lane) * rpl + r];
                        if (fmt == FAST_P4M) s1 = ((const uint16_t*)sc)[(tc * 32 + lane) * rpl + r];
                    }
                    float A, B;
                    if (fmt == FAST_P4) { A = h2f_host(s0); B = 0; }
                    else if (fmt == FAST_P4M) { A = h2f_host(s0); B = h2f_host(s1); }
                    else { A = h2f_host(s1 & 0xffff) * (float)(s0 & 0xff); B = -h2f_host(s1 >> 16) * (float)((s0 >> 8) & 0xff); }
                    float v[32];
                    if (fmt == FAST_P4M) rot::deq32f<FAST_P4M>(q, A, B, hb, v);
                    if (fmt == FAST_K5) rot::deq32f<FAST_K5>(q, A, B, hb, v);
                    if (fmt == FAST_P4) rot::deq32f<FAST_P4>(q, A, B, hb, v);
                    for (int i = 0; i < 32; ++i) maxe = std::max(maxe, (double)std::fabs(v[i] - ref[kb * 32 + i]) / (1e-6 + std::fabs(A) * 32));
                }
            }
            printf("fmt %d rpl %d max rel err %.3e\n", fmt, rpl, maxe);
            bad += maxe > 1e-5;
        }
    printf(bad ? "FAIL\n" : "OK\n");
    return bad;
}
