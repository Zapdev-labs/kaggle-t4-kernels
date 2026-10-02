// Local (CPU-only) check of the t4q fast-GEMV packed layouts: emulates the device kernel's addressing and
// integer math (dp4a, nibble/high-bit unpack) on the host-repacked buffer and compares against a double
// reference computed from the GGUF blocks. g++ -O2 -std=c++17 gemv_layout_check.cpp -o glc && ./glc
#include "../src/kernels/gemv.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

using namespace t4q::gemv;

static uint64_t s = 88172645463325252ull;
static uint64_t rnd() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }

static int dp4a(int a, int b, int c) {
    for (int i = 0; i < 4; ++i) c += (int)(int8_t)(a >> (8 * i)) * (int)(int8_t)(b >> (8 * i));
    return c;
}
static uint32_t rd32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint16_t rd16(const uint8_t* p) { uint16_t v; memcpy(&v, p, 2); return v; }

static float group_dot(int fmt, const int* q /*4 or 8*/, uint32_t H0, uint32_t H1, uint16_t d16, uint16_t sc16,
                       const int* xl, const int* xh, float xd, int s0, int s1, uint16_t m16 = 0, uint32_t d2 = 0) {
    const int m = 0x0F0F0F0F, mh = 0x30303030, mb = 0x10101010;
    if (fmt == FAST_P4M) {
        int t = 0;
        for (int i = 0; i < 4; ++i) t = dp4a(q[i] & m, xl[i], t);
        for (int i = 0; i < 4; ++i) t = dp4a((q[i] >> 4) & m, xh[i], t);
        return h2f_host(d16) * (xd * (float)t) + h2f_host(m16) * (xd * (float)(s0 + s1));
    } else if (fmt == FAST_K5) {
        const int H = (int)H0;
        int t = 0;
        for (int i = 0; i < 4; ++i) t = dp4a((q[i] & m) | ((int)((unsigned)H << (4 - i)) & mb), xl[i], t);
        for (int i = 0; i < 4; ++i) t = dp4a(((q[i] >> 4) & m) | ((int)((unsigned)H >> i) & mb), xh[i], t);
        int sc = sc16 & 0xff, mn = sc16 >> 8;
        return xd * (h2f_host(d2 & 0xffff) * (float)(sc * t) - h2f_host(d2 >> 16) * (float)(mn * (s0 + s1)));
    } else if (fmt == FAST_P4) {
        int t = 0;
        for (int i = 0; i < 4; ++i) t = dp4a(q[i] & m, xl[i], t);
        for (int i = 0; i < 4; ++i) t = dp4a((q[i] >> 4) & m, xh[i], t);
        t -= 8 * (s0 + s1);
        return h2f_host(d16) * (xd * (float)t);
    } else if (fmt == FAST_Q8) {
        int t = 0;
        for (int i = 0; i < 4; ++i) t = dp4a(q[i], xl[i], t);
        for (int i = 0; i < 4; ++i) t = dp4a(q[4 + i], xh[i], t);
        return h2f_host(d16) * (xd * (float)t);
    } else {
        int sa = dp4a((q[0] & m) | ((int)(H0 << 4) & mh), xl[0], 0);
        sa = dp4a((q[1] & m) | ((int)(H0 << 2) & mh), xl[1], sa);
        sa = dp4a((q[2] & m) | ((int)H0 & mh), xl[2], sa);
        sa = dp4a((q[3] & m) | ((int)(H0 >> 2) & mh), xl[3], sa);
        int sb = dp4a(((q[0] >> 4) & m) | ((int)(H1 << 4) & mh), xh[0], 0);
        sb = dp4a(((q[1] >> 4) & m) | ((int)(H1 << 2) & mh), xh[1], sb);
        sb = dp4a(((q[2] >> 4) & m) | ((int)H1 & mh), xh[2], sb);
        sb = dp4a(((q[3] >> 4) & m) | ((int)(H1 >> 2) & mh), xh[3], sb);
        int scA = (int8_t)(sc16 & 0xff), scB = (int8_t)(sc16 >> 8);
        int si = scA * (sa - 32 * s0) + scB * (sb - 32 * s1);
        return h2f_host(d16) * (xd * (float)si);
    }
}

static int run(int fmt, int N, int K, int rpl, int M, int cm = 0) {
    size_t rb = src_row_bytes(fmt, K);
    std::vector<uint8_t> src(rb * N);
    for (auto& b : src) b = (uint8_t)rnd();
    int be = src_block_elems(fmt), bb = src_block_bytes(fmt);
    for (int r = 0; r < N; ++r)
        for (int b = 0; b < K / be; ++b) {
            uint8_t* blk = &src[r * rb + (size_t)b * bb];
            uint16_t d = f2h_host(0.001f + 0.01f * (rnd() % 1000) / 1000.f);
            memcpy(fmt == FAST_K6 ? blk + 208 : blk, &d, 2);
            if (fmt == FAST_P4M || fmt == FAST_K5) {
                uint16_t m2 = f2h_host(((int)(rnd() % 1000) - 500) * 1e-5f);
                if (fmt == FAST_K5) m2 = f2h_host(0.0005f + 0.002f * (rnd() % 1000) / 1000.f);
                memcpy(blk + 2, &m2, 2);
            }
        }
    Layout L = make_layout(fmt, N, K, rpl, cm);
    std::vector<uint8_t> P(L.bytes);
    repack_host(L, src.data(), P.data());
    std::vector<float> x((size_t)M * K);
    for (auto& v : x) v = ((int)(rnd() % 2001) - 1000) / 250.f;
    std::vector<int8_t> xq((size_t)M * K);
    std::vector<int32_t> xm((size_t)M * K / 32 * 2);
    for (int c = 0; c < M; ++c) quantize_q8_host(&x[(size_t)c * K], K, &xq[(size_t)c * K], &xm[(size_t)c * K / 32 * 2]);
    const int nch = K / 512, NB = K / 32;
    std::vector<float> y((size_t)M * N, 0.f);
    for (int tile = 0; tile < L.ntiles; ++tile)
        for (int r = 0; r < rpl; ++r)
            for (int col = 0; col < M; ++col) {
                float lanesum[32] = {0};
                for (int lane = 0; lane < 32; ++lane) {
                    int h = lane >> 4, j = lane & 15;
                    for (int c = 0; c < nch; ++c) {
                        size_t tc = tc_index(tile, c, nch, L.ntiles, cm);
                        int NP = fmt == FAST_Q8 ? 2 : 1;
                        int q[8];
                        for (int p = 0; p < NP; ++p)
                            for (int i = 0; i < 4; ++i)
                                q[4 * p + i] = (int)rd32(&P[L.off_codes + ((tc * rpl + r) * NP + p) * 512 + lane * 16 + 4 * i]);
                        uint32_t H0 = 0, H1 = 0, d2 = 0; uint16_t d16 = 0, sc16 = 0, m16 = 0;
                        if (fmt == FAST_K5) {
                            H0 = rd32(&P[L.off_qh + (tc * rpl + r) * 128 + lane * 4]);
                            sc16 = rd16(&P[L.off_sc + ((tc * 32 + lane) * rpl + r) * 2]);
                            d2 = rd32(&P[L.off_d + (((tc * 2 + h) * 2 + (j >> 3)) * rpl + r) * 4]);
                        } else if (fmt == FAST_P4M) {
                            d16 = rd16(&P[L.off_d + ((tc * 32 + lane) * rpl + r) * 2]);
                            m16 = rd16(&P[L.off_sc + ((tc * 32 + lane) * rpl + r) * 2]);
                        } else if (fmt == FAST_K6) {
                            const uint8_t* qh = &P[L.off_qh + (tc * rpl + r) * 256 + lane * 8];
                            H0 = rd32(qh); H1 = rd32(qh + 4);
                            sc16 = rd16(&P[L.off_sc + ((tc * 32 + lane) * rpl + r) * 2]);
                            d16 = rd16(&P[L.off_d + (((tc * 2 + h) * 2 + (j >> 3)) * rpl + r) * 2]);
                        } else {
                            d16 = rd16(&P[L.off_d + ((tc * 32 + lane) * rpl + r) * 2]);
                        }
                        int kb = c * 16 + j;
                        int xl[4], xh[4];
                        memcpy(xl, &xq[(size_t)col * K + kb * 32], 16);
                        memcpy(xh, &xq[(size_t)col * K + kb * 32 + 16], 16);
                        float xd; memcpy(&xd, &xm[((size_t)col * NB + kb) * 2], 4);
                        int32_t my = xm[((size_t)col * NB + kb) * 2 + 1];
                        int s0 = (int)(short)(my & 0xffff), s1 = my >> 16;
                        lanesum[lane] += group_dot(fmt, q, H0, H1, d16, sc16, xl, xh, xd, s0, s1, m16, d2);
                    }
                }
                for (int h = 0; h < 2; ++h) {
                    float v = 0; for (int j = 0; j < 16; ++j) v += lanesum[h * 16 + j];
                    int row = tile * 2 * rpl + h * rpl + r;
                    if (row < N) y[(size_t)col * N + row] = v;
                }
            }
    // reference
    double maxe = 0, rms = 0;
    std::vector<float> w(K);
    for (int row = 0; row < N; ++row) {
        dequant_row_host(fmt, &src[row * rb], K, w.data());
        for (int col = 0; col < M; ++col) {
            double acc = 0;
            for (int k = 0; k < K; ++k) {
                float xd; memcpy(&xd, &xm[((size_t)col * NB + k / 32) * 2], 4);
                acc += (double)w[k] * xd * xq[(size_t)col * K + k];
            }
            maxe = std::max(maxe, std::fabs(acc - y[(size_t)col * N + row])); rms += acc * acc;
        }
    }
    rms = std::sqrt(rms / ((double)N * M));
    double ne = maxe / rms;
    printf("%s N=%d K=%d rpl=%d M=%d cm=%d  max_err/rms=%.3e %s\n", fmt_name(fmt), N, K, rpl, M, cm, ne,
           ne < 1e-4 ? "OK" : "FAIL");
    return ne < 1e-4 ? 0 : 1;
}

int main() {
    int bad = 0;
    for (int rpl : {1, 2, 4}) for (int cm : {0, 1}) {
        bad += run(FAST_P4, 37, 1536, rpl, 3, cm);
        bad += run(FAST_P4, 20, 8704, rpl, 2, cm);
        bad += run(FAST_Q8, 21, 1024, rpl, 3, cm);
        bad += run(FAST_K6, 19, 1536, rpl, 3, cm);
        bad += run(FAST_K6, 8, 5120, rpl, 2, cm);
        bad += run(FAST_P4M, 23, 8704, rpl, 2, cm);
        bad += run(FAST_K5, 17, 3072, rpl, 3, cm);
    }
    printf(bad ? "LAYOUT CHECK FAILED\n" : "LAYOUT CHECK PASSED\n");
    return bad != 0;
}
