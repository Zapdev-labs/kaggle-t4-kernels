// gemm20_bench.cu -- milestone P round 4: W4A8 on int4 tensor cores (kernels/gemm20.cuh) vs gemm9 (production GA64)
// and gemm17 w8 (plain int8 per-token, the R512 GEMM) on the real per-GPU TP shapes.
// Checks: every variant against the exact Q4_0 x GA64-q8 product (fp64 host reference) on sampled rows / tokens.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo gemm20_bench.cu -o gemm20_bench -ldl -lpthread
// Run:   ./gemm20_bench --dev N                                  checks + burst (all shapes, T 512 / 2048)
//        ./gemm20_bench --dev N --sustain SECS --variants 0,2      gateup T=2048 back to back, NVML windows
// Variants: 0 g9 GA64, 1 g17 w8 per-token (KNOB 1), 2 g20 BR256, 3 g20 BR256 FOLD0 (timing only), 4 g20 BR128,
//           5 g20 BR128 FOLD0, 6 d4_from_q8 alone
// Lines: "CHECK {json}", "R {json}" (burst), "S {json}" (sustain windows).
#include "../src/kernels/gemm16.cuh"
#include "../src/kernels/gemm20.cuh"
#include "../src/kernels/gemm21.cuh"
#include "nvml_lite.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
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

struct Shape { const char* name; int fmt, K, N, count, rpl; };
static const Shape SHAPES[] = {
    {"gateup", FAST_P4, 5120, 17408, 64, 4},
    {"qkvz", FAST_P4, 5120, 8192, 48, 2},
    {"down", FAST_P4, 8704, 5120, 56, 4},
    {"attn_qkv", FAST_P4, 5120, 7168, 16, 2},
    {"attn_out", FAST_P4, 3072, 5120, 16, 4},
};
static const char* VN[] = {"g9_ga64", "g17_w8_tok", "g20_br256", "g20_br256_nofold", "g20_br128", "g20_br128_nofold",
                           "d4_conv", "g21_fb0", "g21_fb1", "g21_bt256", "g21_w8", "g21_bt256_w8"};

static uint64_t rng_s = 0x9E3779B97F4A7C15ull;
static inline uint64_t rnd() { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s; }
static inline float rndu() { return (float)((rnd() >> 40) * (1.0 / 16777216.0)); }
static inline float rndn() { float u = rndu() + 1e-7f, v = rndu(); return std::sqrt(-2.f * std::log(u)) * std::cos(6.2831853f * v); }

static void gen_q40(int N, int K, std::vector<uint8_t>& out) {
    const size_t rb = src_row_bytes(FAST_P4, K);
    out.resize(rb * N);
    uint8_t* p = out.data();
    for (size_t i = 0; i + 8 <= out.size(); i += 8) { uint64_t r = rnd(); memcpy(p + i, &r, 8); }
    for (int row = 0; row < N; ++row)
        for (int b = 0; b < K / 32; ++b) {
            uint8_t* blk = p + (size_t)row * rb + (size_t)b * 18;
            uint16_t d = f2h_host(0.002f + 0.02f * rndu());
            memcpy(blk, &d, 2);
        }
}

static NvmlLite g_nvml;
static int g_dev = 0;

struct Bufs {
    Layout L;
    uint8_t* w = nullptr;
    int8_t* w8 = nullptr;
    float* invs = nullptr;
    int8_t* xq0 = nullptr;  // per-token q8
    float* dx0 = nullptr;
    int8_t* xq64 = nullptr; // GA64 q8
    float* dx64 = nullptr;  // [K/64][Tp]
    uint8_t* xd = nullptr;  // digit planes
    float* y = nullptr;
    int N = 0, K = 0, T = 0, Tp = 0, fmt = 0, rpl = 0;
};

static cudaError_t run_var(int v, Bufs& b, cudaStream_t st) {
    gemm8::Args q = gemm8::make_args(b.L, b.w, b.invs, nullptr, nullptr, b.y, b.N, b.T, b.Tp);
    if (v == 0) {
        q.xq = b.xq64; q.dx = b.dx64;
        return gemm8::launch9(b.fmt, b.rpl, 256, 64, q, st);
    }
    if (v == 1) {
        g16::Args a;
        a.w8 = b.w8; a.q = q; a.invs = b.invs; a.xq = b.xq0; a.dx = b.dx0; a.y = b.y; a.ldy = b.N;
        a.N = b.N; a.K = b.K; a.T = b.T; a.Tp = b.Tp;
        return g16::launch17_t<0, FAST_P4, 4, 0, 0, 1>(a, st);
    }
    if (v == 6) return g20::d4_from_q8(b.xq64, b.Tp, b.K, b.xd, st);
    if (v == 7 || v == 8) {
        g21::Args a;
        a.q = q; a.invs = b.invs; a.xq = b.xq64; a.dx = b.dx64; a.y = b.y; a.ldy = b.N;
        a.N = b.N; a.K = b.K; a.T = b.T; a.Tp = b.Tp;
        return g21::launch(b.fmt, b.rpl, v == 8, a, st);
    }
    if (v >= 9 && v <= 11) {
        g21::Args a;
        a.q = q; a.w8 = b.w8; a.invs = b.invs; a.xq = b.xq64; a.dx = b.dx64; a.y = b.y; a.ldy = b.N;
        a.N = b.N; a.K = b.K; a.T = b.T; a.Tp = b.Tp;
        const bool r4 = b.rpl == 4;
        if (v == 9) return r4 ? g21::launch_t<FAST_P4, 4, 1, 256, 1>(a, st) : g21::launch_t<FAST_P4, 2, 1, 256, 1>(a, st);
        if (v == 10) return g21::launch_t<FAST_P4, 4, 1, 128, 0>(a, st);
        return g21::launch_t<FAST_P4, 4, 1, 256, 0>(a, st);
    }
    g20::Args a;
    a.q = q; a.xd = b.xd; a.dx = b.dx64; a.y = b.y; a.ldy = b.N; a.N = b.N; a.K = b.K; a.T = b.T; a.Tp = b.Tp;
    const bool r4 = b.rpl == 4;
#define G20(BR, FOLD) return r4 ? g20::launch_t<FAST_P4, 4, BR, FOLD, 0>(a, st) : g20::launch_t<FAST_P4, 2, BR, FOLD, 0>(a, st)
    if (v == 2) G20(256, 1);
    if (v == 3) G20(256, 0);
    if (v == 4) G20(128, 1);
    if (v == 5) G20(128, 0);
#undef G20
    return cudaErrorInvalidValue;
}

static void setup(const Shape& sh, int T, Bufs& b, std::vector<uint8_t>& src, cudaStream_t st) {
    b.N = sh.N; b.K = sh.K; b.fmt = sh.fmt; b.rpl = sh.rpl; b.T = T; b.Tp = (T + 255) / 256 * 256;
    gen_q40(sh.N, sh.K, src);
    b.L = make_layout(sh.fmt, sh.N, sh.K, sh.rpl, 1);
    std::vector<uint8_t> packed(b.L.bytes);
    repack_host(b.L, src.data(), packed.data());
    CK(cudaMalloc(&b.w, b.L.bytes));
    CK(cudaMemcpy(b.w, packed.data(), b.L.bytes, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&b.w8, (size_t)sh.N * sh.K));
    CK(cudaMalloc(&b.invs, (size_t)sh.N * 4));
    CK(cudaMalloc(&b.xq0, (size_t)b.Tp * sh.K));
    CK(cudaMalloc(&b.dx0, (size_t)b.Tp * 4));
    CK(cudaMalloc(&b.xq64, (size_t)b.Tp * sh.K));
    CK(cudaMalloc(&b.dx64, (size_t)b.Tp * (sh.K / 64) * 4));
    CK(cudaMalloc(&b.xd, (size_t)b.Tp * sh.K));
    CK(cudaMalloc(&b.y, (size_t)b.Tp * sh.N * 4));
    // activations: gaussian, every 397th channel x30 (outliers), so GA64 groups have different scales
    std::vector<float> xk((size_t)T * sh.K);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < sh.K; ++k) xk[(size_t)t * sh.K + k] = rndn() * ((k % 397) == 5 ? 30.f : 1.f);
    float* x;
    CK(cudaMalloc(&x, xk.size() * 4));
    CK(cudaMemcpy(x, xk.data(), xk.size() * 4, cudaMemcpyHostToDevice));
    gemm8::quant8(x, sh.K, T, b.Tp, sh.K, b.xq0, b.dx0, st, 0);
    gemm8::quant8(x, sh.K, T, b.Tp, sh.K, b.xq64, b.dx64, st, 64);
    CK(g20::d4_from_q8(b.xq64, b.Tp, sh.K, b.xd, st));
    CK(cudaGetLastError());
    gemm8::Args q = gemm8::make_args(b.L, b.w, b.invs, nullptr, nullptr, nullptr, 0, 0, 0);
    CK(gemm8::row_invs(sh.fmt, sh.rpl, q, b.invs, st));
    CK(g16::w8_convert(sh.fmt, sh.rpl, q, b.invs, b.w8, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaFree(x));
}
static void release(Bufs& b) {
    for (void* p : {(void*)b.w, (void*)b.w8, (void*)b.invs, (void*)b.xq0, (void*)b.dx0, (void*)b.xq64, (void*)b.dx64,
                    (void*)b.xd, (void*)b.y})
        cudaFree(p);
    b = Bufs();
}

int main(int argc, char** argv) {
    double sustain = 0;
    std::vector<int> vars = {0, 1, 8, 9, 10, 11};
    std::vector<int> Ts = {512, 2048};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--dev") g_dev = atoi(argv[++i]);
        else if (a == "--sustain") sustain = atof(argv[++i]);
        else if (a == "--variants") {
            vars.clear();
            for (char* t = strtok(argv[++i], ","); t; t = strtok(nullptr, ",")) vars.push_back(atoi(t));
        }
    }
    CK(cudaSetDevice(g_dev));
    g_nvml.init();
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    bool all_ok = true;
    if (sustain <= 0) {
        for (const Shape& sh : SHAPES) {
            for (int T : Ts) {
                Bufs b;
                std::vector<uint8_t> src;
                setup(sh, T, b, src, st);
                // digit-plane round trip on the host (lo + 16 hi == q8, nibble order as in the weights)
                std::vector<int8_t> xq((size_t)b.Tp * b.K);
                std::vector<uint8_t> xd((size_t)b.Tp * b.K);
                std::vector<float> dx((size_t)(b.K / 64) * b.Tp), y((size_t)b.T * b.N);
                CK(cudaMemcpy(xq.data(), b.xq64, xq.size(), cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(xd.data(), b.xd, xd.size(), cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(dx.data(), b.dx64, dx.size() * 4, cudaMemcpyDeviceToHost));
                long dmis = 0;
                for (int t = 0; t < b.T; t += 13)
                    for (int kb = 0; kb < b.K / 32; ++kb) {
                        const uint8_t* p = xd.data() + (size_t)t * b.K + kb * 32;
                        for (int i = 0; i < 16; ++i)
                            for (int h = 0; h < 2; ++h) {
                                const int lo = (p[i] >> (4 * h)) & 15, hn = (p[16 + i] >> (4 * h)) & 15;
                                const int hv = hn >= 8 ? hn - 16 : hn;
                                dmis += (16 * hv + lo) != xq[(size_t)t * b.K + kb * 32 + i + 16 * h];
                            }
                    }
                const size_t rb = src_row_bytes(FAST_P4, b.K);
                std::vector<int8_t> w8((size_t)b.N * b.K);
                std::vector<float> invs(b.N);
                CK(cudaMemcpy(w8.data(), b.w8, w8.size(), cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(invs.data(), b.invs, b.N * 4, cudaMemcpyDeviceToHost));
                for (int v : {0, 7, 8, 9, 10, 11}) {
                    CK(cudaMemset(b.y, 0, (size_t)b.T * b.N * 4));
                    cudaError_t e = run_var(v, b, st);
                    if (e != cudaSuccess) {
                        printf("CHECK {\"shape\":\"%s\",\"variant\":\"%s\",\"err\":\"%s\"}\n", sh.name, VN[v], cudaGetErrorString(e));
                        all_ok = false;
                        continue;
                    }
                    CK(cudaStreamSynchronize(st));
                    CK(cudaMemcpy(y.data(), b.y, y.size() * 4, cudaMemcpyDeviceToHost));
                    double e2 = 0, r2 = 0, emax = 0, f2 = 0, q2 = 0;
                    for (int r = 0; r < b.N; r += (r < b.N - 8 ? 37 : 1)) {
                        const uint8_t* row = src.data() + (size_t)r * rb;
                        for (int t = 0; t < b.T; t += (t < b.T - 3 ? 29 : 1)) {
                            double ref = 0;
                            for (int g = 0; g < b.K / 64; ++g) {
                                double sg = 0;
                                for (int bb = 0; bb < 2; ++bb) {
                                    const uint8_t* blk = row + (size_t)(2 * g + bb) * 18;
                                    uint16_t dh; memcpy(&dh, blk, 2);
                                    long long s = 0;
                                    for (int j = 0; j < 16; ++j) {
                                        const int k0 = (2 * g + bb) * 32 + j;
                                        s += (long long)((blk[2 + j] & 15) - 8) * xq[(size_t)t * b.K + k0];
                                        s += (long long)((blk[2 + j] >> 4) - 8) * xq[(size_t)t * b.K + k0 + 16];
                                    }
                                    sg += (double)h2f_host(dh) * (double)s;
                                }
                                ref += sg * dx[(size_t)g * b.Tp + t];
                            }
                            double ref8 = 0;  // per-row int8 weights (w8_convert, same arithmetic as gemm9 / 17 / 21)
                            for (int g = 0; g < b.K / 64; ++g) {
                                long long s8 = 0;
                                for (int k = g * 64; k < g * 64 + 64; ++k) s8 += (long long)w8[(size_t)r * b.K + k] * xq[(size_t)t * b.K + k];
                                ref8 += (double)s8 * dx[(size_t)g * b.Tp + t];
                            }
                            ref8 /= invs[r];
                            const double gv = y[(size_t)t * b.N + r];
                            f2 += (gv - ref8) * (gv - ref8); q2 += ref8 * ref8;
                            e2 += (gv - ref) * (gv - ref); r2 += ref * ref;
                            emax = std::max(emax, std::fabs(gv - ref));
                        }
                    }
                    const double rel = std::sqrt(e2 / r2);
                    // g20 must be exact up to fp32 rounding; g9 / g17 carry their requantization error (info only)
                    const double rel8 = std::sqrt(f2 / q2);
                    // g20 vs the exact Q4 product; g21 vs its own per-row int8 mirror (fp32 accumulation only)
                    const bool ok = (v == 2 || v == 4) ? (rel < 1e-3 && dmis == 0) : v == 7 ? rel8 < 2e-6 : v >= 8 ? rel8 < 3e-4 : true;
                    all_ok &= ok;
                    printf("CHECK {\"shape\":\"%s\",\"variant\":\"%s\",\"T\":%d,\"rel_l2_vs_exact_q4\":%.3e,\"max_abs\":%.3e,"
                           "\"rel_l2_vs_w8\":%.3e,\"digit_mismatch\":%ld,\"ok\":%d}\n", sh.name, VN[v], T, rel, emax, rel8, dmis, (int)ok);
                    fflush(stdout);
                }
                for (int v : vars) {
                    if (run_var(v, b, st) != cudaSuccess) { cudaGetLastError(); continue; }
                    const int R = T >= 2048 ? 6 : 20;
                    CK(cudaEventRecord(e0, st));
                    for (int r = 0; r < R; ++r) CK(run_var(v, b, st));
                    CK(cudaEventRecord(e1, st));
                    auto smp = g_nvml.sample(g_dev);
                    CK(cudaEventSynchronize(e1));
                    float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
                    const double us = ms * 1e3 / R, tops = 2.0 * b.N * b.K * T / (us * 1e-6) / 1e12;
                    printf("R {\"shape\":\"%s\",\"variant\":\"%s\",\"T\":%d,\"us\":%.1f,\"TOPS\":%.2f,\"sm_mhz\":%u,"
                           "\"power_w\":%.1f,\"ops_per_clk_sm\":%.0f}\n", sh.name, VN[v], T, us, tops, smp.sm,
                           smp.mw / 1000.0, smp.sm ? tops * 1e12 / (smp.sm * 1e6) / 40.0 : 0.0);
                    fflush(stdout);
                }
                release(b);
            }
        }
        printf("SUMMARY {\"all_ok\":%d}\n", (int)all_ok);
    } else {
        Bufs b;
        std::vector<uint8_t> src;
        setup(SHAPES[0], 2048, b, src, st);
        for (int v : vars) {
            auto t0 = std::chrono::steady_clock::now();
            int win = 0;
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < sustain) {
                CK(cudaEventRecord(e0, st));
                const int R = 3;
                for (int r = 0; r < R; ++r) CK(run_var(v, b, st));
                CK(cudaEventRecord(e1, st));
                std::this_thread::sleep_for(std::chrono::milliseconds(15));
                auto smp = g_nvml.sample(g_dev);
                CK(cudaEventSynchronize(e1));
                float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
                const double tops = 2.0 * b.N * b.K * b.T * R / (ms * 1e-3) / 1e12;
                char rs[128]; nvml_reason_str(smp.reasons, rs, sizeof rs);
                printf("S {\"dev\":%d,\"variant\":\"%s\",\"win\":%d,\"t\":%.2f,\"TOPS\":%.2f,\"sm_mhz\":%u,\"power_w\":%.1f,\"temp\":%u,\"ops_per_clk_sm\":%.0f,\"reasons\":\"%s\"}\n",
                       g_dev, VN[v], win++, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), tops,
                       smp.sm, smp.mw / 1000.0, smp.temp, smp.sm ? tops * 1e12 / (smp.sm * 1e6) / 40.0 : 0.0, rs);
                fflush(stdout);
            }
        }
        release(b);
    }
    printf("DONE\n");
    return 0;
}
