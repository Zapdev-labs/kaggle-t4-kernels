// gemm_bench.cu -- milestone P: W4A8 int8-mma prefill GEMM (gemm.cuh) on the real per-GPU TP shapes.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo gemm_bench.cu -o gemm_bench -ldl -lpthread
// Run:   ./gemm_bench --dev N [--T 512,2048] [--reps R]          burst table + correctness
//        ./gemm_bench --dev N --sustain SECS [--shape gateup]    back-to-back GEMMs, NVML clocks/power per window
// Output lines: "CHECK {json}", "R {json}" (one per shape x T), "S {json}" (sustain windows), "SUMMARY {json}".
#include "../src/kernels/gemm.cuh"
#include "../src/kernels/gemm8.cuh"
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

struct Shape { const char* name; int fmt, K, N, per_layer_count, rpl; };
// per GPU under TP=2; count = layers using it (of 64)
static const Shape SHAPES[] = {
    {"qkvz", FAST_P4, 5120, 8192, 48, 2},       // engine layouts: N-split P4 and K5 use rpl 2
    {"gateup", FAST_P4, 5120, 17408, 64, 4},
    {"down", FAST_P4, 8704, 5120, 56, 4},
    {"down_q41", FAST_P4M, 8704, 5120, 8, 4},
    {"ssm_out_k5", FAST_K5, 3072, 5120, 48, 2},
    {"attn_qkv", FAST_P4, 5120, 7168, 16, 2},
    {"attn_out", FAST_P4, 3072, 5120, 16, 4},
    {"qkvz_r4", FAST_P4, 5120, 8192, 0, 4},     // A/B: same shape with rpl 4
    {"ssm_out_k5_r4", FAST_K5, 3072, 5120, 0, 4},
};

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
            if (fmt == FAST_P4M || fmt == FAST_K5) { uint16_t m = f2h_host(-0.01f + 0.02f * rndu()); if (fmt == FAST_K5) m = f2h_host(0.001f + 0.01f * rndu()); memcpy(blk + 2, &m, 2); }
        }
}

static NvmlLite g_nvml;
static int g_dev = 0;

static const char* VNAME[] = {"i8", "i8_epi1", "i8_epi0", "f16", "i4split", "w4a4_timing", "g8_256", "g8_128", "g8m1", "g8m2", "g8m3", "g8m7", "g8m15", "g8m8", "g8b_256", "g8b_128", "g9_256_32", "g9_256_64", "g9_128_32", "g9_128_64",
                               "g9_256_tok", "g9a_noconv", "g9a_noffma", "g9a_hotld", "g9a_nold", "g9a_nold_nosts", "g9a_tok_nold_nosts",
                               "g9_256_64_il", "g10_64", "g10_128", "g11_256x128", "g11_128x256", "g12_s2", "g12_s3", "g9_64_zero_x", "g9_64_zero_w", "g9_64_const_x", "g9_64_sparse_x", "g13_tok", "g14_64", "g15_64"};
static float* g_invs = nullptr;
static float* g_dx = nullptr;
static const Layout* g_L = nullptr;
static uint8_t* g_w = nullptr;
// variant 0: correct W4A8; 1/2: timing-only epilogue variants (P4 rpl 4); 3: fp16 HMMA (P4)
static cudaError_t launch_var(int var, int fmt, int rpl, const gemm::GemmArgs& a, const __half* xh, cudaStream_t st) {
    if (var == 0) return gemm::gemm_launch_fmt(fmt, rpl, a, st);
    if (var == 1 && fmt == FAST_P4 && rpl == 4) return gemm::gemm_launch<FAST_P4, 4, 1>(a, st);
    if (var == 2 && fmt == FAST_P4 && rpl == 4) return gemm::gemm_launch<FAST_P4, 4, 0>(a, st);
    if (var == 4 && fmt == FAST_P4 && rpl == 4) return gemm::gemm_launch<FAST_P4, 4, 3>(a, st);
    if (var == 4 && fmt == FAST_P4 && rpl == 2) return gemm::gemm_launch<FAST_P4, 2, 3>(a, st);
    if (var == 4 && fmt == FAST_P4M && rpl == 4) return gemm::gemm_launch<FAST_P4M, 4, 3>(a, st);
    if (var == 5 && fmt == FAST_P4 && rpl == 4) return gemm::gemm_launch<FAST_P4, 4, 4>(a, st);
    if (var >= 8 && var <= 15 && fmt == FAST_P4 && rpl == 4) {
        gemm8::Args a8 = gemm8::make_args(*g_L, g_w, g_invs, a.xq, g_dx, a.y, a.ldy, a.T, a.Tp);
        switch (var) {
            case 8: return gemm8::launch_t<FAST_P4, 4, 256, 1>(a8, st);
            case 9: return gemm8::launch_t<FAST_P4, 4, 256, 2>(a8, st);
            case 10: return gemm8::launch_t<FAST_P4, 4, 256, 3>(a8, st);
            case 11: return gemm8::launch_t<FAST_P4, 4, 256, 7>(a8, st);
            case 12: return gemm8::launch_t<FAST_P4, 4, 256, 15>(a8, st);
            case 13: return gemm8::launch_t<FAST_P4, 4, 256, 8>(a8, st);
            case 14: return gemm8::launch_t<FAST_P4, 4, 256, 16>(a8, st);
            case 15: return gemm8::launch_t<FAST_P4, 4, 128, 16>(a8, st);
        }
    }
    if (var == 40) {
        gemm8::Args a8 = gemm8::make_args(*g_L, g_w, g_invs, a.xq, g_dx, a.y, a.ldy, a.T, a.Tp);
        return gemm8::launch15(fmt, rpl, a8, st);
    }
    if (var == 39) {
        gemm8::Args a8 = gemm8::make_args(*g_L, g_w, g_invs, a.xq, g_dx, a.y, a.ldy, a.T, a.Tp);
        return gemm8::launch14(fmt, rpl, a8, st);
    }
    if (var == 38) {
        gemm8::Args a8 = gemm8::make_args(*g_L, g_w, g_invs, a.xq, g_dx, a.y, a.ldy, a.T, a.Tp);
        return (fmt == FAST_P4 && rpl == 4) ? gemm8::launch13_t<FAST_P4, 4>(a8, st) : cudaErrorInvalidValue;
    }
    if (var >= 34 && var <= 37) {  // data-toggle probes: gemm9 GA64 BN256 on modified inputs (set up in the sustain loop)
        gemm8::Args a8 = gemm8::make_args(*g_L, g_w, g_invs, a.xq, g_dx, a.y, a.ldy, a.T, a.Tp);
        return gemm8::launch9(fmt, rpl, 256, 64, a8, st);
    }
    if (var == 32 || var == 33) {
        gemm8::Args a8 = gemm8::make_args(*g_L, g_w, g_invs, a.xq, g_dx, a.y, a.ldy, a.T, a.Tp);
        return gemm8::launch12(fmt, rpl, var == 32 ? 2 : 3, a8, st);
    }
    if (var == 30 || var == 31) {
        gemm8::Args a8 = gemm8::make_args(*g_L, g_w, g_invs, a.xq, g_dx, a.y, a.ldy, a.T, a.Tp);
        return gemm8::launch11(fmt, rpl, var == 30 ? 256 : 128, a8, st);
    }
    if (var >= 27 && var <= 29) {
        gemm8::Args a8 = gemm8::make_args(*g_L, g_w, g_invs, a.xq, g_dx, a.y, a.ldy, a.T, a.Tp);
        if (var == 27) return (fmt == FAST_P4 && rpl == 4) ? gemm8::launch9_t<FAST_P4, 4, 256, 64, 32>(a8, st) : cudaErrorInvalidValue;
        return gemm8::launch10(fmt, rpl, var == 28 ? 64 : 128, a8, st);
    }
    if (var >= 20 && var <= 26) {
        gemm8::Args a8 = gemm8::make_args(*g_L, g_w, g_invs, a.xq, g_dx, a.y, a.ldy, a.T, a.Tp);
        if (fmt != FAST_P4 || rpl != 4) return cudaErrorInvalidValue;
        switch (var) {
            case 20: return gemm8::launch9_t<FAST_P4, 4, 256, 0>(a8, st);
            case 21: return gemm8::launch9_t<FAST_P4, 4, 256, 64, 1>(a8, st);
            case 22: return gemm8::launch9_t<FAST_P4, 4, 256, 64, 2>(a8, st);
            case 23: return gemm8::launch9_t<FAST_P4, 4, 256, 64, 4>(a8, st);
            case 24: return gemm8::launch9_t<FAST_P4, 4, 256, 64, 8>(a8, st);
            case 25: return gemm8::launch9_t<FAST_P4, 4, 256, 64, 8 | 16>(a8, st);
            case 26: return gemm8::launch9_t<FAST_P4, 4, 256, 0, 8 | 16>(a8, st);
        }
    }
    if (var >= 16 && var <= 19) {
        gemm8::Args a8 = gemm8::make_args(*g_L, g_w, g_invs, a.xq, g_dx, a.y, a.ldy, a.T, a.Tp);
        return gemm8::launch9(fmt, rpl, var <= 17 ? 256 : 128, (var & 1) ? 64 : 32, a8, st);
    }
    if (var == 6 || var == 7) {
        gemm8::Args a8 = gemm8::make_args(*g_L, g_w, g_invs, a.xq, g_dx, a.y, a.ldy, a.T, a.Tp);
        return gemm8::launch(fmt, rpl, var == 6 ? 256 : 128, a8, st);
    }
    if (var == 3 && fmt == FAST_P4) {
        static int attr = 0;
        if (!attr) {
            cudaFuncSetAttribute(gemm::gemm_f16_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * gemm::F16Cfg::BYTES);
            cudaFuncSetAttribute(gemm::gemm_f16_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * gemm::F16Cfg::BYTES);
            attr = 1;
        }
        return rpl == 4 ? gemm::gemm_f16_launch<4>(a, xh, st) : gemm::gemm_f16_launch<2>(a, xh, st);
    }
    return cudaErrorInvalidValue;
}

// host mirror of gemm8's per-row requantization of one GGUF row (natural order) -> w8, given invs
static void g8_row_host(int fmt, const uint8_t* row, int K, float invs, std::vector<int>& w8) {
    w8.resize(K);
    auto rnd = [](double v) { return (int)std::nearbyint(v); };
    if (fmt == FAST_P4) {
        for (int b = 0; b < K / 32; ++b) {
            const uint8_t* p = row + 18 * b; uint16_t dh; memcpy(&dh, p, 2);
            const float av = h2f_host(dh) * invs;
            const double r = h2f_host(f2h_host(av)), r16 = h2f_host(f2h_host(av * 0.0625f));
            for (int j = 0; j < 16; ++j) {
                w8[32 * b + j] = rnd(((p[2 + j] & 15) - 8) * r);
                w8[32 * b + 16 + j] = rnd(16.0 * ((p[2 + j] >> 4) - 8) * r16);
            }
        }
        return;
    }
    std::vector<float> wf(K);  // reconstruct per-block (A, B) from the dequantized row: w = A q + B
    if (fmt == FAST_P4M) {
        for (int b = 0; b < K / 32; ++b) {
            const uint8_t* p = row + 20 * b; uint16_t dh, mh; memcpy(&dh, p, 2); memcpy(&mh, p + 2, 2);
            const float A = h2f_host(dh), B = h2f_host(mh);
            const double a = h2f_host(f2h_host(A * invs)), a16 = h2f_host(f2h_host(A * invs * 0.0625f)), bb = h2f_host(f2h_host(B * invs));
            for (int j = 0; j < 16; ++j) {
                const float t0 = h2f_host(f2h_host((float)((p[4 + j] & 15) * a + bb)));
                const float t1 = h2f_host(f2h_host((float)(16.0 * (p[4 + j] >> 4) * a16 + bb)));
                w8[32 * b + j] = rnd(t0);
                w8[32 * b + 16 + j] = rnd(t1);
            }
        }
        return;
    }
    // K5
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
                w8[256 * sb + 32 * sub + e] = rnd(t);
            }
        }
    }
}

struct Dev {
    uint8_t* w = nullptr;
    int8_t* xq = nullptr;
    float2* xs = nullptr;
    float* xsum = nullptr;
    float* x = nullptr;
    float* y = nullptr;
    __half* xh = nullptr;
};


int main(int argc, char** argv) {
    std::vector<int> Ts = {512, 2048};
    int reps = 0;
    double sustain = 0;
    std::string only;
    std::vector<int> svars = {0, 3, 1, 2};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--dev") g_dev = atoi(argv[++i]);
        else if (a == "--reps") reps = atoi(argv[++i]);
        else if (a == "--sustain") sustain = atof(argv[++i]);
        else if (a == "--shape") only = argv[++i];
        else if (a == "--variants") {
            svars.clear();
            for (char* t = strtok(argv[++i], ","); t; t = strtok(nullptr, ",")) svars.push_back(atoi(t));
        }
        else if (a == "--T") {
            Ts.clear();
            std::string s = argv[++i];
            size_t p = 0;
            while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); Ts.push_back(atoi(s.substr(p, q - p).c_str())); p = q + 1; }
        }
    }
    CK(cudaSetDevice(g_dev));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, g_dev));
    g_nvml.init();
    printf("INFO dev=%d name=%s sms=%d clock_khz=%d nvml=%d\n", g_dev, prop.name, prop.multiProcessorCount, prop.clockRate,
           (int)g_nvml.ok);
    fflush(stdout);
    const int Tmax = *std::max_element(Ts.begin(), Ts.end());
    const int Tpmax = (Tmax + 127) / 128 * 128;
    const int Kmax = 8704;
    Dev D;
    CK(cudaMalloc(&D.xq, (size_t)Tpmax * Kmax));
    CK(cudaMalloc(&D.xs, (size_t)Tpmax * (Kmax / 32) * 8));
    CK(cudaMalloc(&D.xsum, (size_t)Tpmax * (Kmax / 32) * 4));
    CK(cudaMalloc(&D.x, (size_t)Tmax * Kmax * 4));
    CK(cudaMalloc(&D.y, (size_t)Tmax * 17408 * 4));
    CK(cudaMalloc(&D.xh, (size_t)Tpmax * Kmax * 2));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    cudaStream_t st;
    CK(cudaStreamCreate(&st));

    // activations: gaussian with a few outlier channels (x30), like real residual-norm outputs
    std::vector<float> hx((size_t)Tmax * Kmax);
    for (int t = 0; t < Tmax; ++t)
        for (int k = 0; k < Kmax; ++k) hx[(size_t)t * Kmax + k] = rndn() * ((k % 397) == 5 ? 30.f : 1.f);

    bool all_ok = true;
    double tot_time[8] = {0}, tot_ops[8] = {0}, tot8_time[8][2] = {{0}};
    for (const Shape& sh : SHAPES) {
        if (!only.empty() && only != sh.name) continue;
        const int N = sh.N, K = sh.K, fmt = sh.fmt;
        std::vector<uint8_t> src;
        gen_gguf(fmt, N, K, src);
        Layout L = make_layout(fmt, N, K, sh.rpl, 1);
        std::vector<uint8_t> packed(L.bytes);
        repack_host(L, src.data(), packed.data());
        CK(cudaMalloc(&D.w, L.bytes));
        CK(cudaMemcpy(D.w, packed.data(), L.bytes, cudaMemcpyHostToDevice));
        // x for this K (row stride K)
        std::vector<float> xk((size_t)Tmax * K);
        for (int t = 0; t < Tmax; ++t) memcpy(&xk[(size_t)t * K], &hx[(size_t)t * Kmax], K * 4);
        CK(cudaMemcpy(D.x, xk.data(), xk.size() * 4, cudaMemcpyHostToDevice));

        for (size_t ti = 0; ti < Ts.size(); ++ti) {
            const int T = Ts[ti], Tp = (T + 127) / 128 * 128;
            gemm::quant_rows(D.x, K, T, Tp, K, D.xq, D.xs, D.xsum, st);
            CK(cudaStreamSynchronize(st));
            gemm::GemmArgs a = gemm::make_args(L, D.w, D.xq, D.xs, D.xsum, D.y, N, T, Tp);
            CK(gemm::gemm_launch_fmt(fmt, sh.rpl, a, st));
            CK(cudaStreamSynchronize(st));
            // ---- correctness (first T only per shape, sampled rows x tokens)
            if (ti == 0 || T == Ts.back()) {
                const int nb = K / 32;
                std::vector<int8_t> xq((size_t)Tp * K);
                std::vector<float2> xs((size_t)nb * Tp);
                std::vector<float> xsum((size_t)nb * Tp);
                CK(cudaMemcpy(xq.data(), D.xq, xq.size(), cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(xs.data(), D.xs, xs.size() * 8, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(xsum.data(), D.xsum, xsum.size() * 4, cudaMemcpyDeviceToHost));
                // quantizer vs host mirror on 64 tokens
                long qmis = 0;
                {
                    std::vector<int8_t> hq(K); std::vector<float> hs(nb), hn(nb), hsx(nb);
                    for (int t = 0; t < std::min(T, 64); ++t) {
                        gemm::quant_row_host(&xk[(size_t)t * K], K, hq.data(), hs.data(), hn.data(), hsx.data());
                        for (int k = 0; k < K; ++k) qmis += hq[k] != xq[(size_t)t * K + k];
                        for (int b = 0; b < nb; ++b) qmis += hs[b] != xs[(size_t)b * Tp + t].x || hn[b] != xs[(size_t)b * Tp + t].y || hsx[b] != xsum[(size_t)b * Tp + t];
                    }
                }
                std::vector<float> y((size_t)T * N);
                CK(cudaMemcpy(y.data(), D.y, y.size() * 4, cudaMemcpyDeviceToHost));
                // reference: rows sampled (every 37th + last 16), tokens sampled (every 61st + last 3)
                std::vector<int> rows, toks;
                for (int r = 0; r < N; r += 37) rows.push_back(r);
                for (int r = N - 16; r < N; ++r) rows.push_back(r);
                for (int t = 0; t < T; t += 61) toks.push_back(t);
                for (int t = T - 3; t < T; ++t) toks.push_back(t);
                const size_t rb = src_row_bytes(fmt, K);
                double e2 = 0, r2 = 0, emax = 0, ex2 = 0;
                std::vector<double> xd((size_t)toks.size() * K);
                for (size_t it = 0; it < toks.size(); ++it) {
                    const int t = toks[it];
                    for (int b = 0; b < nb; ++b) {
                        const double d = (double)xs[(size_t)b * Tp + t].x * 16.0;
                        for (int i = 0; i < 32; ++i) xd[it * K + 32 * b + i] = d * xq[(size_t)t * K + 32 * b + i];
                    }
                }
                std::vector<float> w(K);
                for (int r : rows) {
                    dequant_row_host(fmt, src.data() + (size_t)r * rb, K, w.data());
                    for (size_t it = 0; it < toks.size(); ++it) {
                        const int t = toks[it];
                        double s = 0, sx = 0;
                        for (int k = 0; k < K; ++k) { s += (double)w[k] * xd[it * K + k]; sx += (double)w[k] * xk[(size_t)t * K + k]; }
                        const double g = y[(size_t)t * N + r];
                        e2 += (g - s) * (g - s); r2 += s * s; ex2 += (g - sx) * (g - sx);
                        emax = std::max(emax, std::fabs(g - s));
                    }
                }
                const double rms = std::sqrt(r2 / (rows.size() * toks.size()));
                const double rel = std::sqrt(e2 / r2), relx = std::sqrt(ex2 / r2);
                const bool ok = rel < 1e-4 && qmis == 0 && std::isfinite(rel);
                all_ok = all_ok && ok;
                printf("CHECK {\"shape\":\"%s\",\"fmt\":\"%s\",\"T\":%d,\"rel_l2_vs_q8ref\":%.3e,\"max_err_over_rms\":%.3e,"
                       "\"rel_l2_vs_fp32x\":%.3e,\"quant_mismatch\":%ld,\"n\":%zu,\"ok\":%d}\n",
                       sh.name, fmt_name(fmt), T, rel, emax / rms, relx, qmis, rows.size() * toks.size(), (int)ok);
                fflush(stdout);
            }
            // ---- timing
            const int R = reps > 0 ? reps : (T >= 2048 ? 6 : 20);
            for (int w2 = 0; w2 < 2; ++w2) CK(gemm::gemm_launch_fmt(fmt, sh.rpl, a, st));
            CK(cudaEventRecord(e0, st));
            for (int r = 0; r < R; ++r) CK(gemm::gemm_launch_fmt(fmt, sh.rpl, a, st));
            CK(cudaEventRecord(e1, st));
            // sample NVML while it runs
            auto smp = g_nvml.sample(g_dev);
            CK(cudaEventSynchronize(e1));
            float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
            const double us = ms * 1e3 / R;
            const double tops = 2.0 * N * K * T / (us * 1e-6) / 1e12;
            // quantizer timing
            CK(cudaEventRecord(e0, st));
            for (int r = 0; r < 5; ++r) gemm::quant_rows(D.x, K, T, Tp, K, D.xq, D.xs, D.xsum, st);
            CK(cudaEventRecord(e1, st));
            CK(cudaEventSynchronize(e1));
            float msq = 0; CK(cudaEventElapsedTime(&msq, e0, e1));
            const double ops_per_clk_sm = tops * 1e12 / (smp.sm ? smp.sm * 1e6 : 1) / 40.0;
            printf("R {\"shape\":\"%s\",\"rpl\":%d,\"fmt\":\"%s\",\"N\":%d,\"K\":%d,\"T\":%d,\"us\":%.1f,\"TOPS\":%.2f,\"sm_mhz\":%u,"
                   "\"power_w\":%.1f,\"temp\":%u,\"ops_per_clk_sm\":%.0f,\"pct_of_peak_at_clock\":%.1f,\"quant_us\":%.1f}\n",
                   sh.name, sh.rpl, fmt_name(fmt), N, K, T, us, tops, smp.sm, smp.mw / 1000.0, smp.temp, ops_per_clk_sm,
                   100.0 * ops_per_clk_sm / 2048.0, msq * 1e3 / 5);
            fflush(stdout);
            tot_time[ti] += us * sh.per_layer_count;
            tot_ops[ti] += 2.0 * N * K * T * sh.per_layer_count;
            // ---- gemm8: in-kernel per-row int8 requant (correctness vs its host mirror and vs exact Q4, timing BN 256/128)
            {
                CK(cudaMalloc(&g_invs, (size_t)N * 4));
                CK(cudaMalloc(&g_dx, (size_t)(K / 32) * Tp * 4));
                g_L = &L; g_w = D.w;
                gemm8::Args a8 = gemm8::make_args(L, D.w, g_invs, D.xq, g_dx, D.y, N, T, Tp);
                CK(gemm8::row_invs(fmt, sh.rpl, a8, g_invs, st));
                struct G8c { int kern, bn, ga; };  // kern 8 = gemm8_kernel, 9 = gemm9_kernel
                for (G8c cf : {G8c{9, 256, 64}, G8c{17, 256, 64}}) {
                    if (cf.kern == 11 && !(fmt == FAST_P4 && sh.rpl == 4)) continue;
                    const int bn = cf.bn, ga = cf.ga;
                    if (Tp % bn) continue;
                    char vname[32];
                    snprintf(vname, sizeof vname, "g%d_%d_%d", cf.kern, bn, ga);
                    gemm8::quant8(D.x, K, T, Tp, K, D.xq, g_dx, st, ga);
                    auto run8 = [&]() {
                        if (cf.kern == 8) return gemm8::launch(fmt, sh.rpl, bn, a8, st);
                        if (cf.kern == 10) return gemm8::launch10(fmt, sh.rpl, ga, a8, st);
                        if (cf.kern == 11) return gemm8::launch9_t<FAST_P4, 4, 256, 64, 32>(a8, st);
                        if (cf.kern == 12) return gemm8::launch11(fmt, sh.rpl, 256, a8, st);
                        if (cf.kern == 14) return gemm8::launch12(fmt, sh.rpl, 2, a8, st);
                        if (cf.kern == 16) return gemm8::launch14(fmt, sh.rpl, a8, st);
                        if (cf.kern == 17) return gemm8::launch15(fmt, sh.rpl, a8, st);
                        if (cf.kern == 15) return gemm8::launch12(fmt, sh.rpl, 3, a8, st);
                        if (cf.kern == 13) return gemm8::launch11(fmt, sh.rpl, 128, a8, st);
                        return gemm8::launch9(fmt, sh.rpl, bn, ga, a8, st);
                    };
                    CK(cudaMemset(D.y, 0, (size_t)T * N * 4));
                    CK(run8());
                    CK(cudaStreamSynchronize(st));
                    if (ti == 0 || T == Ts.back()) {
                        const int gq = ga ? ga : K;  // ga 0: one scale per token (dx is [Tp])
                        const int nb = K / gq;
                        std::vector<int8_t> xq((size_t)Tp * K);
                        std::vector<float> dxh((size_t)nb * Tp), invh(N), y((size_t)T * N);
                        CK(cudaMemcpy(xq.data(), D.xq, xq.size(), cudaMemcpyDeviceToHost));
                        CK(cudaMemcpy(dxh.data(), g_dx, dxh.size() * 4, cudaMemcpyDeviceToHost));
                        CK(cudaMemcpy(invh.data(), g_invs, N * 4, cudaMemcpyDeviceToHost));
                        CK(cudaMemcpy(y.data(), D.y, y.size() * 4, cudaMemcpyDeviceToHost));
                        const size_t rb = src_row_bytes(fmt, K);
                        double e2 = 0, r2 = 0, eq2 = 0, emax = 0;
                        long inv_mis = 0;
                        std::vector<int> w8; std::vector<float> w(K);
                        int nrow = 0;
                        for (int r = 0; r < N; r += (r < N - 16 ? 41 : 1)) {
                            dequant_row_host(fmt, src.data() + (size_t)r * rb, K, w.data());
                            const float ih = gemm8::row_invs_host(w.data(), K);
                            // device invs uses the per-block bound (8|d| etc.), the host the actual max: report both
                            if (fmt == FAST_P4 ? false : std::fabs(ih - invh[r]) > 1e-3f * ih) inv_mis++;
                            g8_row_host(fmt, src.data() + (size_t)r * rb, K, invh[r], w8);
                            ++nrow;
                            for (int t = 0; t < T; t += (t < T - 3 ? 53 : 1)) {
                                double s8 = 0, sq = 0;
                                for (int b = 0; b < nb; ++b) {
                                    long si = 0;
                                    double sqb = 0;
                                    for (int i = 0; i < gq; ++i) {
                                        const int xv = xq[(size_t)t * K + gq * b + i];
                                        si += (long)w8[gq * b + i] * xv;
                                        sqb += (double)w[gq * b + i] * xv;
                                    }
                                    s8 += (double)si * dxh[(size_t)b * Tp + t];
                                    sq += sqb * dxh[(size_t)b * Tp + t];
                                }
                                s8 /= invh[r];
                                const double g = y[(size_t)t * N + r];
                                e2 += (g - s8) * (g - s8); r2 += s8 * s8; eq2 += (g - sq) * (g - sq);
                                emax = std::max(emax, std::fabs(g - s8));
                            }
                        }
                        const double rel = std::sqrt(e2 / r2), relq = std::sqrt(eq2 / r2);
                        const bool ok = rel < 2e-3 && relq < 5e-2 && std::isfinite(rel);
                        all_ok = all_ok && ok;
                        printf("CHECK {\"shape\":\"%s\",\"variant\":\"%s\",\"T\":%d,\"rel_l2_vs_mirror\":%.3e,"
                               "\"rel_l2_vs_q4\":%.3e,\"max_err_over_rms\":%.3e,\"inv_mismatch\":%ld,\"rows\":%d,\"ok\":%d}\n",
                               sh.name, vname, T, rel, relq, emax / std::sqrt(r2 / std::max(1, nrow)), inv_mis, nrow, (int)ok);
                        fflush(stdout);
                    }
                    const int R = reps > 0 ? reps : (T >= 2048 ? 6 : 20);
                    for (int w2 = 0; w2 < 2; ++w2) CK(run8());
                    CK(cudaEventRecord(e0, st));
                    for (int r = 0; r < R; ++r) CK(run8());
                    CK(cudaEventRecord(e1, st));
                    auto smp3 = g_nvml.sample(g_dev);
                    CK(cudaEventSynchronize(e1));
                    float ms3 = 0; CK(cudaEventElapsedTime(&ms3, e0, e1));
                    const double us3 = ms3 * 1e3 / R, tops3 = 2.0 * N * K * T / (us3 * 1e-6) / 1e12;
                    printf("RV {\"shape\":\"%s\",\"variant\":\"%s\",\"T\":%d,\"us\":%.1f,\"TOPS\":%.2f,\"sm_mhz\":%u,"
                           "\"power_w\":%.1f,\"ops_per_clk_sm\":%.0f}\n", sh.name, vname, T, us3, tops3, smp3.sm, smp3.mw / 1000.0,
                           smp3.sm ? tops3 * 1e12 / (smp3.sm * 1e6) / 40.0 : 0.0);
                    fflush(stdout);
                    if (cf.kern == 9 && ga == 64) tot8_time[ti][0] += us3 * sh.per_layer_count;
                    if (cf.kern == 17) tot8_time[ti][1] += us3 * sh.per_layer_count;
                }
                CK(cudaFree(g_invs)); CK(cudaFree(g_dx)); g_invs = nullptr; g_dx = nullptr;
                gemm::quant_rows(D.x, K, T, Tp, K, D.xq, D.xs, D.xsum, st);  // restore the gemm.cuh activations
                CK(gemm::gemm_launch_fmt(fmt, sh.rpl, a, st));
                CK(cudaStreamSynchronize(st));
            }
            // ---- other variants: fp16 HMMA (correctness vs fp32 x), timing-only epilogue variants on gateup
            std::vector<float> y8;
            {
                y8.resize((size_t)T * N);
                CK(cudaMemcpy(y8.data(), D.y, y8.size() * 4, cudaMemcpyDeviceToHost));  // timing loop wrote the same y
            }
            for (int var = 1; var <= 5; ++var) {
                if (var == 3 && fmt != FAST_P4) continue;
                if (var == 4 && fmt == FAST_K5) continue;
                const bool gu = fmt == FAST_P4 && sh.rpl == 4 && std::string(sh.name) == "gateup";
                if ((var < 3 || var == 5) && !gu) continue;
                if (var == 4) {
                    gemm::quant_rows_i4_kernel<<<(Tp * (K / 32) + 127) / 128, 128, 0, st>>>(D.x, K, T, Tp, K, D.xq, D.xs, D.xsum);
                    CK(cudaGetLastError());
                }
                if (var == 3) {
                    gemm::to_f16_perm_kernel<<<(Tp * (K / 32) + 127) / 128, 128, 0, st>>>(D.x, K, T, Tp, K, D.xh);
                    CK(cudaGetLastError());
                }
                CK(launch_var(var, fmt, sh.rpl, a, D.xh, st));
                CK(cudaStreamSynchronize(st));
                if (var == 4) {  // int4 path vs the (validated) int8 path output
                    std::vector<float> y((size_t)T * N);
                    CK(cudaMemcpy(y.data(), D.y, y.size() * 4, cudaMemcpyDeviceToHost));
                    double e2 = 0, r2 = 0;
                    for (size_t i2 = 0; i2 < y.size(); ++i2) { const double dd = (double)y[i2] - y8[i2]; e2 += dd * dd; r2 += (double)y8[i2] * y8[i2]; }
                    const double rel = std::sqrt(e2 / r2);
                    const bool ok = rel < 1e-4 && std::isfinite(rel);
                    all_ok = all_ok && ok;
                    printf("CHECK {\"shape\":\"%s\",\"variant\":\"i4split\",\"T\":%d,\"rel_l2_vs_i8\":%.3e,\"ok\":%d}\n",
                           sh.name, T, rel, (int)ok);
                }
                if (var == 3) {  // fp16 GEMM vs fp64 with fp32 x on sampled rows/tokens
                    std::vector<float> y((size_t)T * N);
                    CK(cudaMemcpy(y.data(), D.y, y.size() * 4, cudaMemcpyDeviceToHost));
                    const size_t rb = src_row_bytes(fmt, K);
                    std::vector<float> w(K);
                    double e2 = 0, r2 = 0;
                    for (int r = 0; r < N; r += 97) {
                        dequant_row_host(fmt, src.data() + (size_t)r * rb, K, w.data());
                        for (int t = 0; t < T; t += 67) {
                            double sx = 0;
                            for (int k = 0; k < K; ++k) sx += (double)w[k] * xk[(size_t)t * K + k];
                            const double g = y[(size_t)t * N + r];
                            e2 += (g - sx) * (g - sx); r2 += sx * sx;
                        }
                    }
                    const double rel = std::sqrt(e2 / r2);
                    const bool ok = rel < 2e-3 && std::isfinite(rel);
                    all_ok = all_ok && ok;
                    printf("CHECK {\"shape\":\"%s\",\"variant\":\"f16\",\"T\":%d,\"rel_l2_vs_fp32x\":%.3e,\"ok\":%d}\n",
                           sh.name, T, rel, (int)ok);
                }
                CK(cudaEventRecord(e0, st));
                for (int r = 0; r < R; ++r) CK(launch_var(var, fmt, sh.rpl, a, D.xh, st));
                CK(cudaEventRecord(e1, st));
                auto smp2 = g_nvml.sample(g_dev);
                CK(cudaEventSynchronize(e1));
                float ms2 = 0; CK(cudaEventElapsedTime(&ms2, e0, e1));
                const double us2 = ms2 * 1e3 / R;
                printf("RV {\"shape\":\"%s\",\"variant\":\"%s\",\"T\":%d,\"us\":%.1f,\"TOPS\":%.2f,\"sm_mhz\":%u,"
                       "\"power_w\":%.1f}\n", sh.name, VNAME[var], T, us2, 2.0 * N * K * T / (us2 * 1e-6) / 1e12, smp2.sm,
                       smp2.mw / 1000.0);
                fflush(stdout);
            }
        }
        CK(cudaFree(D.w));
        D.w = nullptr;
    }
    if (only.empty()) {
        printf("SUMMARY {\"all_ok\":%d", (int)all_ok);
        for (size_t ti = 0; ti < Ts.size(); ++ti) {
            const double s = tot_time[ti] * 1e-6;
            const double s8a = tot8_time[ti][0] * 1e-6, s8b = tot8_time[ti][1] * 1e-6;
            printf(",\"T%d\":{\"linear_ms_per_gpu\":%.1f,\"TOPS\":%.2f,\"linear_only_tok_s\":%.0f,\"g9_256_64_ms\":%.1f,"
                   "\"g9_256_64_TOPS\":%.2f,\"g15_ms\":%.1f,\"g15_TOPS\":%.2f}", Ts[ti], s * 1e3,
                   tot_ops[ti] / s / 1e12, Ts[ti] / s, s8a * 1e3, s8a > 0 ? tot_ops[ti] / s8a / 1e12 : 0.0, s8b * 1e3,
                   s8b > 0 ? tot_ops[ti] / s8b / 1e12 : 0.0);
        }
        printf("}\n");
        fflush(stdout);
    }

    if (sustain > 0) {  // gateup at the largest T, back to back, NVML every ~100 ms of GPU time
        const Shape& sh = SHAPES[1];
        const int N = sh.N, K = sh.K, T = Tmax, Tp = Tpmax;
        std::vector<uint8_t> src;
        gen_gguf(sh.fmt, N, K, src);
        Layout L = make_layout(sh.fmt, N, K, sh.rpl, 1);
        std::vector<uint8_t> packed(L.bytes);
        repack_host(L, src.data(), packed.data());
        CK(cudaMalloc(&D.w, L.bytes));
        CK(cudaMemcpy(D.w, packed.data(), L.bytes, cudaMemcpyHostToDevice));
        std::vector<float> xk((size_t)T * K);
        for (int t = 0; t < T; ++t) memcpy(&xk[(size_t)t * K], &hx[(size_t)t * Kmax], K * 4);
        CK(cudaMemcpy(D.x, xk.data(), xk.size() * 4, cudaMemcpyHostToDevice));
        gemm::quant_rows(D.x, K, T, Tp, K, D.xq, D.xs, D.xsum, st);
        gemm::to_f16_perm_kernel<<<(Tp * (K / 32) + 127) / 128, 128, 0, st>>>(D.x, K, T, Tp, K, D.xh);
        gemm::GemmArgs a = gemm::make_args(L, D.w, D.xq, D.xs, D.xsum, D.y, N, T, Tp);
        CK(cudaMalloc(&g_invs, (size_t)N * 4));
        CK(cudaMalloc(&g_dx, (size_t)(K / 32) * Tp * 4));
        g_L = &L; g_w = D.w;
        {
            gemm8::Args a8 = gemm8::make_args(L, D.w, g_invs, D.xq, g_dx, D.y, N, T, Tp);
            CK(gemm8::row_invs(sh.fmt, sh.rpl, a8, g_invs, st));
        }
        for (int var : svars) {
        if (var >= 6) gemm8::quant8(D.x, K, T, Tp, K, D.xq, g_dx, st,
                                    var >= 39 ? 64 : var == 38 ? 0 : var >= 34 ? 64 : var >= 32 ? 32 : var == 29 ? 128 : var >= 30 ? 64 : (var == 20 || var == 26) ? 0 : var >= 21 ? 64 : (var >= 16 && (var & 1)) ? 64 : 32);
        else if (var >= 4) gemm::quant_rows_i4_kernel<<<(Tp * (K / 32) + 127) / 128, 128, 0, st>>>(D.x, K, T, Tp, K, D.xq, D.xs, D.xsum);
        else gemm::quant_rows(D.x, K, T, Tp, K, D.xq, D.xs, D.xsum, st);
        // data-toggle probes
        static uint8_t* wsave = nullptr;
        if (var == 34) CK(cudaMemsetAsync(D.xq, 0, (size_t)Tp * K, st));                 // all-zero activations
        if (var == 36) CK(cudaMemsetAsync(D.xq, 0x11, (size_t)Tp * K, st));              // constant activations
        if (var == 37) {  // 7 of 8 activation bytes zero (keeps every 8th)
            std::vector<int8_t> h((size_t)Tp * K);
            CK(cudaMemcpyAsync(h.data(), D.xq, h.size(), cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            for (size_t i2 = 0; i2 < h.size(); ++i2) if (i2 & 7) h[i2] = 0;
            CK(cudaMemcpyAsync(D.xq, h.data(), h.size(), cudaMemcpyHostToDevice, st));
        }
        if (var == 35) {  // all-zero weights: codes 0x88 (c = 8 in both nibbles -> w8 = 0)
            if (!wsave) CK(cudaMalloc(&wsave, L.off_qh > 0 ? L.off_qh : L.bytes));
            CK(cudaMemcpyAsync(wsave, D.w, L.off_qh, cudaMemcpyDeviceToDevice, st));
            CK(cudaMemsetAsync(D.w, 0x88, L.off_qh, st));
        }
        CK(cudaStreamSynchronize(st));
        auto t0 = std::chrono::steady_clock::now();
        int win = 0;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < sustain) {
            CK(cudaEventRecord(e0, st));
            const int R = 3;
            for (int r = 0; r < R; ++r) CK(launch_var(var, sh.fmt, sh.rpl, a, D.xh, st));
            CK(cudaEventRecord(e1, st));
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
            auto smp = g_nvml.sample(g_dev);
            CK(cudaEventSynchronize(e1));
            float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
            const double tops = 2.0 * N * K * T * R / (ms * 1e-3) / 1e12;
            char rs[128]; nvml_reason_str(smp.reasons, rs, sizeof rs);
            printf("S {\"dev\":%d,\"variant\":\"%s\",\"win\":%d,\"t\":%.2f,\"TOPS\":%.2f,\"sm_mhz\":%u,\"power_w\":%.1f,\"temp\":%u,\"reasons\":\"%s\"}\n",
                   g_dev, VNAME[var], win++, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), tops, smp.sm,
                   smp.mw / 1000.0, smp.temp, rs);
            fflush(stdout);
        }
        if (var == 35) { CK(cudaMemcpyAsync(D.w, wsave, L.off_qh, cudaMemcpyDeviceToDevice, st)); CK(cudaStreamSynchronize(st)); }
        }
    }
    printf("DONE\n");
    return 0;
}
