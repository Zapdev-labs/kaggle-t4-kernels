// gemm16_bench.cu -- milestone P round 3: plain int8 x int8 GEMM (kernels/gemm16.cuh) vs gemm9 / gemm13 on the real
// per-GPU TP shapes; correctness vs an exact int64 host reference; burst table; sustained both-GPU runs with NVML.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo gemm16_bench.cu -o gemm16_bench -ldl -lpthread
// Run:   ./gemm16_bench --dev N                                   burst + checks (all shapes, T 512 / 2048)
//        ./gemm16_bench --dev N --sustain SECS --variants 0,2,3   gateup T=2048 back to back
// Variants: 0 gemm9 GA64 (production), 1 gemm13 (per-token, in-kernel conversion), 2 g16 w8 (pre-converted int8),
//           3 g16 in-place Q4 (early conversion), 4 g16 w8 + w8_convert before every GEMM, 5 w8_convert alone
// Lines: "CHECK {json}", "R {json}" (burst), "S {json}" (sustain windows).
#include "../src/kernels/gemm16.cuh"
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
    {"down_q41", FAST_P4M, 8704, 5120, 8, 4},
    {"ssm_out_k5", FAST_K5, 3072, 5120, 48, 2},
    {"attn_qkv", FAST_P4, 5120, 7168, 16, 2},
    {"attn_out", FAST_P4, 3072, 5120, 16, 4},
};
static const char* VN[] = {"g9_ga64", "g13_tok", "g16_w8", "g16_q4", "g16_w8+conv", "conv_only", "g16_w8_l2",
                           "g16_w8_swap", "g16_w8_swap_l2", "g16_q4_l2", "g17_q4_tok", "g17_q4_gsh", "g17_w8_tok",
                           "g17_w8_gsh"};

static uint64_t rng_s = 0x9E3779B97F4A7C15ull;
static inline uint64_t rnd() { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s; }
static inline float rndu() { return (float)((rnd() >> 40) * (1.0 / 16777216.0)); }
static inline float rndn() { float u = rndu() + 1e-7f, v = rndu(); return std::sqrt(-2.f * std::log(u)) * std::cos(6.2831853f * v); }

static void gen_gguf(int fmt, int N, int K, std::vector<uint8_t>& out) {
    size_t rb = src_row_bytes(fmt, K);
    out.resize(rb * N);
    uint8_t* p = out.data();
    for (size_t i = 0; i + 8 <= out.size(); i += 8) { uint64_t r = rnd(); memcpy(p + i, &r, 8); }
    int be = src_block_elems(fmt), bb = src_block_bytes(fmt);
    for (int row = 0; row < N; ++row)
        for (int b = 0; b < K / be; ++b) {
            uint8_t* blk = p + (size_t)row * rb + (size_t)b * bb;
            uint16_t d = f2h_host(0.002f + 0.02f * rndu());
            memcpy(blk, &d, 2);
            if (fmt == FAST_P4M || fmt == FAST_K5) {
                uint16_t m = f2h_host(-0.01f + 0.02f * rndu());
                if (fmt == FAST_K5) m = f2h_host(0.001f + 0.01f * rndu());
                memcpy(blk + 2, &m, 2);
            }
        }
}

static NvmlLite g_nvml;
static int g_dev = 0;

struct Bufs {
    Layout L;
    uint8_t* w = nullptr;   // decode layout
    int8_t* w8 = nullptr;   // [N][K]
    float* invs = nullptr;
    int8_t* xq0 = nullptr;  // per-token q8
    float* dx0 = nullptr;   // [Tp]
    int8_t* xq64 = nullptr; // GA64 q8
    float* dx64 = nullptr;
    float* y = nullptr;
    int8_t* xqg = nullptr;  // GSH q8
    int8_t* dsh = nullptr;
    float* dxg = nullptr;
    int N = 0, K = 0, T = 0, Tp = 0, fmt = 0, rpl = 0;
};

static cudaError_t run_var(int v, Bufs& b, cudaStream_t st) {
    gemm8::Args q = gemm8::make_args(b.L, b.w, b.invs, nullptr, nullptr, b.y, b.N, b.T, b.Tp);
    if (v == 0 || v == 1) {
        q.xq = v == 0 ? b.xq64 : b.xq0;
        q.dx = v == 0 ? b.dx64 : b.dx0;
        if (v == 0) return gemm8::launch9(b.fmt, b.rpl, 256, 64, q, st);
        if (b.fmt == FAST_P4 && b.rpl == 4) return gemm8::launch13_t<FAST_P4, 4>(q, st);
        if (b.fmt == FAST_P4 && b.rpl == 2) return gemm8::launch13_t<FAST_P4, 2>(q, st);
        return cudaErrorInvalidValue;
    }
    if (v == 5 || v == 4) {
        cudaError_t e = g16::w8_convert(b.fmt, b.rpl, q, b.invs, b.w8, st);
        if (e != cudaSuccess || v == 5) return e;
    }
    g16::Args a;
    a.w8 = b.w8; a.q = q; a.invs = b.invs; a.xq = b.xq0; a.dx = b.dx0; a.y = b.y; a.ldy = b.N;
    a.N = b.N; a.K = b.K; a.T = b.T; a.Tp = b.Tp;
    if (v >= 10 && v <= 13) {
        const int gsh = (v == 11 || v == 13);
        if (gsh) { a.xq = b.xqg; a.dx = b.dxg; a.dsh = b.dsh; }
        if (v == 12) return g16::launch17_t<0, FAST_P4, 4, 0, 0>(a, st);
        if (v == 13) return g16::launch17_t<0, FAST_P4, 4, 0, 1>(a, st);
        return g16::launch17(b.fmt, b.rpl, gsh, a, st);
    }
    if (v == 6) return g16::launch_t<0, FAST_P4, 4, 0, 0, 1>(a, st);
    if (v == 7) return g16::launch_t<0, FAST_P4, 4, 0, 1, 0>(a, st);
    if (v == 8) return g16::launch_t<0, FAST_P4, 4, 0, 1, 1>(a, st);
    if (v == 9) {
        if (b.fmt == FAST_P4 && b.rpl == 4) return g16::launch_t<1, FAST_P4, 4, 0, 0, 1>(a, st);
        if (b.fmt == FAST_P4 && b.rpl == 2) return g16::launch_t<1, FAST_P4, 2, 0, 0, 1>(a, st);
        return cudaErrorInvalidValue;
    }
    return g16::launch(v == 3 ? 1 : 0, b.fmt, b.rpl, a, st);
}

static void setup(const Shape& sh, int T, Bufs& b, std::vector<uint8_t>& src, std::vector<float>& xk, cudaStream_t st) {
    b.N = sh.N; b.K = sh.K; b.fmt = sh.fmt; b.rpl = sh.rpl; b.T = T; b.Tp = (T + 255) / 256 * 256;
    gen_gguf(sh.fmt, sh.N, sh.K, src);
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
    CK(cudaMalloc(&b.y, (size_t)b.Tp * sh.N * 4));
    CK(cudaMalloc(&b.xqg, (size_t)b.Tp * sh.K));
    CK(cudaMalloc(&b.dsh, (size_t)b.Tp * (sh.K / 64)));
    CK(cudaMalloc(&b.dxg, (size_t)b.Tp * 4));
    // activations: gaussian, a few x30 channels
    xk.resize((size_t)T * sh.K);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < sh.K; ++k) xk[(size_t)t * sh.K + k] = rndn() * ((k % 397) == 5 ? 30.f : 1.f);
    float* x;
    CK(cudaMalloc(&x, xk.size() * 4));
    CK(cudaMemcpy(x, xk.data(), xk.size() * 4, cudaMemcpyHostToDevice));
    gemm8::quant8(x, sh.K, T, b.Tp, sh.K, b.xq0, b.dx0, st, 0);
    gemm8::quant8(x, sh.K, T, b.Tp, sh.K, b.xq64, b.dx64, st, 64);
    g16::quant_gsh_kernel<float><<<b.Tp, 256, 0, st>>>(x, sh.K, T, sh.K, 7, b.xqg, b.dsh, b.dxg, b.Tp);
    CK(cudaGetLastError());
    gemm8::Args q = gemm8::make_args(b.L, b.w, b.invs, nullptr, nullptr, nullptr, 0, 0, 0);
    CK(gemm8::row_invs(sh.fmt, sh.rpl, q, b.invs, st));
    CK(g16::w8_convert(sh.fmt, sh.rpl, q, b.invs, b.w8, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaFree(x));
}
static void release(Bufs& b) {
    for (void* p : {(void*)b.w, (void*)b.w8, (void*)b.invs, (void*)b.xq0, (void*)b.dx0, (void*)b.xq64, (void*)b.dx64, (void*)b.y,
                    (void*)b.xqg, (void*)b.dsh, (void*)b.dxg})
        cudaFree(p);
    b = Bufs();
}

// host mirror of the per-row requantization (gemm_bench.cu g8_row_host)
static void g8_row_host(int fmt, const uint8_t* row, int K, float invs, std::vector<int>& w8) {
    w8.resize(K);
    auto rn = [](double v) { return (int)std::nearbyint(v); };
    if (fmt == FAST_P4) {
        for (int b = 0; b < K / 32; ++b) {
            const uint8_t* p = row + 18 * b; uint16_t dh; memcpy(&dh, p, 2);
            const float av = h2f_host(dh) * invs;
            const double r = h2f_host(f2h_host(av)), r16 = h2f_host(f2h_host(av * 0.0625f));
            for (int j = 0; j < 16; ++j) {
                w8[32 * b + j] = rn(((p[2 + j] & 15) - 8) * r);
                w8[32 * b + 16 + j] = rn(16.0 * ((p[2 + j] >> 4) - 8) * r16);
            }
        }
        return;
    }
    if (fmt == FAST_P4M) {
        for (int b = 0; b < K / 32; ++b) {
            const uint8_t* p = row + 20 * b; uint16_t dh, mh; memcpy(&dh, p, 2); memcpy(&mh, p + 2, 2);
            const float A = h2f_host(dh), B = h2f_host(mh);
            const double a = h2f_host(f2h_host(A * invs)), a16 = h2f_host(f2h_host(A * invs * 0.0625f)), bb = h2f_host(f2h_host(B * invs));
            for (int j = 0; j < 16; ++j) {
                w8[32 * b + j] = rn(h2f_host(f2h_host((float)((p[4 + j] & 15) * a + bb))));
                w8[32 * b + 16 + j] = rn(h2f_host(f2h_host((float)(16.0 * (p[4 + j] >> 4) * a16 + bb))));
            }
        }
        return;
    }
    uint8_t q[256]; int sc[8], mn[8];
    for (int sb = 0; sb < K / 256; ++sb) {
        const uint8_t* p = row + 176 * sb; uint16_t dh, mh; memcpy(&dh, p, 2); memcpy(&mh, p + 2, 2);
        q5k_codes(p, q, sc, mn);
        for (int sub = 0; sub < 8; ++sub) {
            const float A = h2f_host(dh) * (float)sc[sub], B = -h2f_host(mh) * (float)mn[sub];
            const double a = h2f_host(f2h_host(A * invs)), a16 = h2f_host(f2h_host(A * invs * 0.0625f)), bb = h2f_host(f2h_host(B * invs));
            for (int e = 0; e < 32; ++e) {
                const int qq = q[32 * sub + e];
                const float t = e < 16 ? h2f_host(f2h_host((float)(qq * a + bb))) : h2f_host(f2h_host((float)(16.0 * qq * a16 + bb)));
                w8[256 * sb + 32 * sub + e] = rn(t);
            }
        }
    }
}

int main(int argc, char** argv) {
    double sustain = 0;
    std::vector<int> vars = {0, 3, 8, 9, 10, 11, 12, 13};
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
                std::vector<float> xk;
                setup(sh, T, b, src, xk, st);
                // ---- checks: w8 vs host mirror; g16 (both sources) vs exact int64 reference on sampled rows / tokens
                std::vector<int8_t> w8((size_t)b.N * b.K), xq((size_t)b.Tp * b.K);
                std::vector<float> invs(b.N), dx(b.Tp), y((size_t)b.T * b.N);
                CK(cudaMemcpy(w8.data(), b.w8, w8.size(), cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(xq.data(), b.xq0, xq.size(), cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(invs.data(), b.invs, b.N * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(dx.data(), b.dx0, b.Tp * 4, cudaMemcpyDeviceToHost));
                long wmis = 0;
                std::vector<int> hw;
                const size_t rb = src_row_bytes(sh.fmt, sh.K);
                for (int r = 0; r < b.N; r += 97) {
                    g8_row_host(sh.fmt, src.data() + (size_t)r * rb, b.K, invs[r], hw);
                    for (int k = 0; k < b.K; ++k) wmis += hw[k] != w8[(size_t)r * b.K + k];
                }
                std::vector<int8_t> xqg((size_t)b.Tp * b.K), dsh((size_t)b.Tp * (b.K / 64));
                std::vector<float> dxg(b.Tp);
                CK(cudaMemcpy(xqg.data(), b.xqg, xqg.size(), cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(dsh.data(), b.dsh, dsh.size(), cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(dxg.data(), b.dxg, b.Tp * 4, cudaMemcpyDeviceToHost));
                // GSH quantization error vs fp32 x (vs per-token and GA64 for scale): relative L2 over sampled tokens
                {
                    std::vector<int8_t> x64((size_t)b.Tp * b.K);
                    std::vector<float> d64((size_t)(b.K / 64) * b.Tp);
                    CK(cudaMemcpy(x64.data(), b.xq64, x64.size(), cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(d64.data(), b.dx64, d64.size() * 4, cudaMemcpyDeviceToHost));
                    double eg = 0, et = 0, e6 = 0, rr = 0;
                    for (int t = 0; t < b.T; t += 7) {
                        auto pos = [&](int tt) { return (tt & ~63) | ((tt & 7) << 3) | ((tt >> 3) & 7); };
                        int E = 0;
                        std::vector<int> Eg(b.K / 64);
                        for (int g = 0; g < b.K / 64; ++g) { E += dsh[(size_t)g * b.Tp + pos(t)]; Eg[g] = E; }
                        for (int k = 0; k < b.K; ++k) {
                            const double xv = xk[(size_t)t * b.K + k];
                            const double vg = xqg[(size_t)t * b.K + k] * (double)dxg[t] * std::ldexp(1.0, E - Eg[k / 64]);
                            const double vt = xq[(size_t)t * b.K + k] * (double)dx[t];
                            const double v6 = x64[(size_t)t * b.K + k] * (double)d64[(size_t)(k / 64) * b.Tp + t];
                            eg += (vg - xv) * (vg - xv); et += (vt - xv) * (vt - xv); e6 += (v6 - xv) * (v6 - xv); rr += xv * xv;
                        }
                    }
                    printf("CHECK {\"shape\":\"%s\",\"T\":%d,\"act_rel_err_gsh\":%.3e,\"act_rel_err_tok\":%.3e,\"act_rel_err_ga64\":%.3e}\n",
                           sh.name, T, std::sqrt(eg / rr), std::sqrt(et / rr), std::sqrt(e6 / rr));
                }
                for (int v : {2, 3, 6, 7, 8, 9, 10, 11, 12, 13}) {
                    if (v == 9 && sh.fmt != FAST_P4) continue;
                    const bool gsh = (v == 11 || v == 13);
                    CK(cudaMemset(b.y, 0, (size_t)b.T * b.N * 4));
                    cudaError_t e = run_var(v, b, st);
                    if (e != cudaSuccess) { printf("CHECK {\"shape\":\"%s\",\"variant\":\"%s\",\"err\":\"%s\"}\n", sh.name, VN[v], cudaGetErrorString(e)); all_ok = false; continue; }
                    CK(cudaStreamSynchronize(st));
                    CK(cudaMemcpy(y.data(), b.y, y.size() * 4, cudaMemcpyDeviceToHost));
                    double e2 = 0, r2 = 0;
                    for (int r = 0; r < b.N; r += (r < b.N - 8 ? 37 : 1))
                        for (int t = 0; t < b.T; t += (t < b.T - 3 ? 29 : 1)) {
                            double ref;
                            if (!gsh) {
                                long long s = 0;
                                for (int k = 0; k < b.K; ++k) s += (long long)w8[(size_t)r * b.K + k] * xq[(size_t)t * b.K + k];
                                ref = (double)s * dx[t] / invs[r];
                            } else {
                                const int pt = (t & ~63) | ((t & 7) << 3) | ((t >> 3) & 7);
                                int E = 0;
                                std::vector<int> Eg(b.K / 64);
                                for (int g = 0; g < b.K / 64; ++g) { E += dsh[(size_t)g * b.Tp + pt]; Eg[g] = E; }
                                double s = 0;
                                for (int g = 0; g < b.K / 64; ++g) {
                                    long long sg = 0;
                                    for (int k = g * 64; k < g * 64 + 64; ++k) sg += (long long)w8[(size_t)r * b.K + k] * xqg[(size_t)t * b.K + k];
                                    s += (double)sg * std::ldexp(1.0, E - Eg[g]);
                                }
                                ref = s * dxg[t] / invs[r];
                            }
                            const double g = y[(size_t)t * b.N + r];
                            e2 += (g - ref) * (g - ref); r2 += ref * ref;
                        }
                    const double rel = std::sqrt(e2 / r2);
                    const bool ok = rel < (gsh ? 1e-5 : 1e-6) && wmis == 0;
                    all_ok &= ok;
                    printf("CHECK {\"shape\":\"%s\",\"variant\":\"%s\",\"T\":%d,\"rel_l2_vs_exact\":%.3e,\"w8_mismatch\":%ld,\"ok\":%d}\n",
                           sh.name, VN[v], T, rel, wmis, (int)ok);
                    fflush(stdout);
                }
                // ---- burst timing
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
                    const double wgb = v == 5 ? ((double)b.L.bytes + (double)b.N * b.K) / (us * 1e-6) / 1e9 : 0;
                    printf("R {\"shape\":\"%s\",\"variant\":\"%s\",\"T\":%d,\"us\":%.1f,\"TOPS\":%.2f,\"conv_GBs\":%.1f,\"sm_mhz\":%u,"
                           "\"power_w\":%.1f,\"ops_per_clk_sm\":%.0f}\n", sh.name, VN[v], T, us, tops, wgb, smp.sm,
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
        std::vector<float> xk;
        setup(SHAPES[0], 2048, b, src, xk, st);
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
