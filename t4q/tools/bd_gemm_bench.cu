// bd_gemm_bench.cu -- milestone B: gemm9 at batched-decode token tiles (Tp 32 / 64) on the real per-GPU TP shapes,
// synthetic weights (random GGUF blocks repacked to the engine layouts), vs the dp4a GEMV (m = 1) weight-streaming rate.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo bd_gemm_bench.cu -o bd_gemm_bench
// Run:   ./bd_gemm_bench --dev N [--reps R] [--tp 32,64] [--pfk 0,16]
// Output: "R {json}" per (shape, Tp, variant), "CHECK {json}" (variants bit-identical), "SUM {json}" per-step estimate.
#include "../src/kernels/gemm8.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace t4q;
using namespace t4q::gemv;

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) {                                                                     \
            printf("FATAL cuda %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); \
            fflush(stdout);                                                                          \
            exit(2);                                                                                 \
        }                                                                                            \
    } while (0)

struct Shape { const char* name; int fmt, K, N, rpl, count; };
static const Shape SHAPES[] = {
    {"gateup", FAST_P4, 5120, 17408, 4, 64}, {"qkvz", FAST_P4, 5120, 8192, 2, 48}, {"down", FAST_P4, 8704, 5120, 4, 56},
    {"down_q41", FAST_P4M, 8704, 5120, 4, 8}, {"ssm_out", FAST_K5, 3072, 5120, 2, 48},
    {"attn_qkv", FAST_P4, 5120, 7168, 2, 16}, {"wo", FAST_P4, 3072, 5120, 4, 16}, {"lm_head", FAST_K6, 5120, 124160, 2, 1},
};

static uint64_t rng_s = 0x9E3779B97F4A7C15ull;
static inline uint64_t rnd() { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s; }
static inline float rndu() { return (float)((rnd() >> 40) * (1.0 / 16777216.0)); }

static void gen_gguf(int fmt, int N, int K, std::vector<uint8_t>& out) {
    const size_t rb = src_row_bytes(fmt, K);
    out.resize(rb * N);
    uint8_t* p = out.data();
    for (size_t i = 0; i + 8 <= out.size(); i += 8) { uint64_t r = rnd(); memcpy(p + i, &r, 8); }
    const int be = src_block_elems(fmt), bb = src_block_bytes(fmt);
    for (int row = 0; row < N; ++row)
        for (int b = 0; b < K / be; ++b) {
            uint8_t* blk = p + (size_t)row * rb + (size_t)b * bb;
            const uint16_t d = f2h_host(0.002f + 0.02f * rndu());
            if (fmt == FAST_K6) { memcpy(blk + 208, &d, 2); continue; }
            memcpy(blk, &d, 2);
            if (fmt == FAST_P4M) { uint16_t m = f2h_host(-0.01f + 0.02f * rndu()); memcpy(blk + 2, &m, 2); }
            if (fmt == FAST_K5) { uint16_t m = f2h_host(0.001f + 0.01f * rndu()); memcpy(blk + 2, &m, 2); }
        }
}

int main(int argc, char** argv) {
    int dev = 0, reps = 20;
    std::vector<int> tps = {32, 64}, pfks = {0};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto list = [&](std::vector<int>& v) { v.clear(); for (char* t = strtok(argv[++i], ","); t; t = strtok(nullptr, ",")) v.push_back(atoi(t)); };
        if (a == "--dev") dev = atoi(argv[++i]);
        else if (a == "--reps") reps = atoi(argv[++i]);
        else if (a == "--tp") list(tps);
        else if (a == "--pfk") list(pfks);
    }
    CK(cudaSetDevice(dev));
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const int TPMAX = 64;
    int8_t* xq; float* dx; float *y0, *y1;
    CK(cudaMalloc(&xq, (size_t)TPMAX * 8704));
    CK(cudaMalloc(&dx, (size_t)TPMAX * (8704 / 32) * 4));
    CK(cudaMalloc(&y0, (size_t)TPMAX * 124160 * 4));
    CK(cudaMalloc(&y1, (size_t)TPMAX * 124160 * 4));
    {
        std::vector<int8_t> hx((size_t)TPMAX * 8704);
        for (auto& v : hx) v = (int8_t)((int)(rnd() % 255) - 127);
        CK(cudaMemcpy(xq, hx.data(), hx.size(), cudaMemcpyHostToDevice));
        std::vector<float> hd((size_t)TPMAX * (8704 / 32));
        for (auto& v : hd) v = 0.001f + 0.01f * rndu();
        CK(cudaMemcpy(dx, hd.data(), hd.size() * 4, cudaMemcpyHostToDevice));
    }
    std::vector<double> step_ms(16, 0.0);  // per (Tp index, variant) estimated per-step GEMM ms
    for (const Shape& sh : SHAPES) {
        std::vector<uint8_t> g;
        gen_gguf(sh.fmt, sh.N, sh.K, g);
        const Layout L = make_layout(sh.fmt, sh.N, sh.K, sh.rpl, 1);
        uint8_t *src, *w;
        float* invs;
        CK(cudaMalloc(&src, g.size()));
        CK(cudaMalloc(&w, L.bytes));
        CK(cudaMalloc(&invs, (size_t)sh.N * 4));
        CK(cudaMemcpy(src, g.data(), g.size(), cudaMemcpyHostToDevice));
        CK(repack_device(L, src, w, st));
        CK(cudaFree(src));
        {
            gemm8::Args a = gemm8::make_args(L, w, invs, nullptr, nullptr, nullptr, 0, 0, 0);
            CK(gemm8::row_invs(L.fmt, L.rpl, a, invs, st));
        }
        CK(cudaStreamSynchronize(st));
        for (size_t ti = 0; ti < tps.size(); ++ti) {
            const int Tp = tps[ti];
            int vi = 0;
            for (int pfk : pfks)
                for (int ring = 0; ring < 2; ++ring, ++vi) {
                    gemm8::Args a = gemm8::make_args(L, w, invs, xq, dx, ring ? y1 : y0, sh.N, Tp, Tp);
                    a.lbm = ring;
                    a.pfk = pfk;
                    auto run = [&]() { CK(gemm8::launch9(L.fmt, L.rpl, Tp, 64, a, st)); };
                    for (int r = 0; r < 3; ++r) run();
                    CK(cudaEventRecord(e0, st));
                    for (int r = 0; r < reps; ++r) run();
                    CK(cudaEventRecord(e1, st));
                    CK(cudaEventSynchronize(e1));
                    float ms = 0;
                    CK(cudaEventElapsedTime(&ms, e0, e1));
                    const double us = 1e3 * ms / reps;
                    if (ti * 8 + vi < step_ms.size()) step_ms[ti * 8 + vi] += us * sh.count / 1e3;
                    printf("R {\"dev\": %d, \"shape\": \"%s\", \"Tp\": %d, \"ring\": %d, \"pfk\": %d, \"us\": %.1f, \"GBps\": %.1f, "
                           "\"TOPS\": %.2f}\n", dev, sh.name, Tp, ring, pfk, us, L.bytes / us / 1e3,
                           2.0 * sh.N * (double)sh.K * Tp / us / 1e6);
                    fflush(stdout);
                }
            // ring vs no ring: same arithmetic, must be bit-identical
            std::vector<float> h0((size_t)Tp * sh.N), h1((size_t)Tp * sh.N);
            CK(cudaMemcpy(h0.data(), y0, h0.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h1.data(), y1, h1.size() * 4, cudaMemcpyDeviceToHost));
            size_t ne = 0;
            for (size_t i = 0; i < h0.size(); ++i) ne += memcmp(&h0[i], &h1[i], 4) != 0;
            printf("CHECK {\"dev\": %d, \"shape\": \"%s\", \"Tp\": %d, \"ring_vs_plain_mismatch\": %zu, \"n\": %zu}\n", dev,
                   sh.name, Tp, ne, h0.size());
        }
        // dp4a GEMV m = 1 weight-streaming reference (P4 rpl 4 / K6 rpl 2 shapes)
        if ((sh.fmt == FAST_P4 && sh.rpl == 4 && sh.K == 5120) || sh.fmt == FAST_K6) {
            int8_t* gx; int2* gm;
            CK(cudaMalloc(&gx, sh.K));
            CK(cudaMalloc(&gm, (sh.K / 32) * 8));
            CK(cudaMemset(gx, 1, sh.K));
            CK(cudaMemset(gm, 0, (sh.K / 32) * 8));
            GemvArgs ga = gemv::make_args(L, w, gx, gm, y0, sh.N);
            auto run = [&]() {
                if (sh.fmt == FAST_P4) CK((gemv_fast_launch<FAST_P4, 4, 1, 10, 1, false>(ga, 80, 256, st)));
                else CK((gemv_fast_launch<FAST_K6, 2, 1, 10, 1, false>(ga, 80, 256, st)));
            };
            for (int r = 0; r < 3; ++r) run();
            CK(cudaEventRecord(e0, st));
            for (int r = 0; r < reps; ++r) run();
            CK(cudaEventRecord(e1, st));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            const double us = 1e3 * ms / reps;
            printf("R {\"dev\": %d, \"shape\": \"%s\", \"Tp\": 1, \"gemv_dp4a\": 1, \"us\": %.1f, \"GBps\": %.1f}\n", dev,
                   sh.name, us, L.bytes / us / 1e3);
            CK(cudaFree(gx));
            CK(cudaFree(gm));
        }
        CK(cudaFree(w));
        CK(cudaFree(invs));
    }
    for (size_t ti = 0; ti < tps.size(); ++ti)
        for (size_t v = 0; v < pfks.size() * 2 && v < 8; ++v)
            printf("SUM {\"dev\": %d, \"Tp\": %d, \"pfk\": %d, \"ring\": %d, \"gemm_ms_per_step\": %.2f}\n", dev, tps[ti],
                   pfks[v / 2], (int)(v % 2), step_ms[ti * 8 + v]);
    return 0;
}
