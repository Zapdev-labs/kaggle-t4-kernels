// gemv_bench.cu -- M0 GEMV bench for t4q fast GEMV (P4 / Q8 / K6, m = 1..8) on real Qwen3.8-27B TP shapes.
//
// Build (parts compile in parallel, each instantiates one format/K):
//   for p in 0 1 2 3 4 5: nvcc -O3 -std=c++17 -arch=sm_75 -DPART=$p -c gemv_bench.cu -o gb$p.o
//   nvcc -arch=sm_75 gb*.o -o gemv_bench -ldl -lpthread
// Run: ./gemv_bench [--dev N] [--quick] [--shape name] | ./gemv_bench --sustain SECS --dev N --shape name --cfg rpl,minb,xsm,thr,gmode,m
// Output: lines "R {json}" per measurement, "CHECK {json}" per correctness check, "SUMMARY {json}" at the end.
#include "../src/kernels/gemv.cuh"
#include "nvml_lite.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace t4q::gemv;

typedef cudaError_t (*LaunchFn)(const GemvArgs&, int, int, cudaStream_t);
typedef int (*OccFn)(int);
struct KEntry { int fmt, nch, m, rpl, D, xsm; LaunchFn fn; OccFn occ; };

template <int FMT, int NCH, int RPL, int D, bool XSM, int... Ms>
static void reg_ms(std::vector<KEntry>& v, std::integer_sequence<int, Ms...>) {
    (v.push_back(KEntry{FMT, NCH, Ms + 1, RPL, D, XSM ? 1 : 0, &gemv_fast_launch<FMT, RPL, Ms + 1, NCH, D, XSM>,
                        &gemv_fast_occupancy<FMT, RPL, Ms + 1, NCH, D, XSM>}),
     ...);
}
template <int FMT, int NCH>
static void reg_fmt(std::vector<KEntry>& v) {
    auto s = std::make_integer_sequence<int, 8>{};
    reg_ms<FMT, NCH, 1, 2, false>(v, s); reg_ms<FMT, NCH, 1, 4, false>(v, s);
    reg_ms<FMT, NCH, 2, 2, false>(v, s); reg_ms<FMT, NCH, 2, 4, false>(v, s);
    reg_ms<FMT, NCH, 1, 2, true>(v, s);  reg_ms<FMT, NCH, 1, 4, true>(v, s);
    reg_ms<FMT, NCH, 2, 2, true>(v, s);  reg_ms<FMT, NCH, 2, 4, true>(v, s);
}

void reg_part1(std::vector<KEntry>& v);
void reg_part2(std::vector<KEntry>& v);
void reg_part3(std::vector<KEntry>& v);
void reg_part4(std::vector<KEntry>& v);
void reg_part5(std::vector<KEntry>& v);

#if PART == 1
void reg_part1(std::vector<KEntry>& v) { reg_fmt<FAST_P4, 6>(v); }
#elif PART == 2
void reg_part2(std::vector<KEntry>& v) { reg_fmt<FAST_P4, 10>(v); }
#elif PART == 3
void reg_part3(std::vector<KEntry>& v) { reg_fmt<FAST_P4, 17>(v); }
#elif PART == 4
void reg_part4(std::vector<KEntry>& v) { reg_fmt<FAST_Q8, 10>(v); }
#elif PART == 5
void reg_part5(std::vector<KEntry>& v) { reg_fmt<FAST_K6, 10>(v); }
#else  // PART 0: main

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) {                                                                     \
            printf("FATAL cuda %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); \
            fflush(stdout);                                                                          \
            exit(2);                                                                                 \
        }                                                                                            \
    } while (0)

struct Shape { const char* name; int fmt, K, N; int count_per_token; };
// per GPU per token under TP=2 (count = how many times this GEMV runs per token)
static const Shape SHAPES[] = {
    {"gateup_tp", FAST_P4, 5120, 17408, 64},     // ffn gate|up interleaved, 8704+8704 rows
    {"qkvzab_full", FAST_P4, 5120, 16480, 0},    // full (non-TP) DeltaNet in-proj, design table shape
    {"qkvzab_tp", FAST_P4, 5120, 8240, 48},      // DeltaNet in-proj per GPU
    {"attn_qkv_tp", FAST_P4, 5120, 7168, 16},    // attn q|k|v per GPU
    {"down_tp", FAST_P4, 8704, 5120, 64},        // ffn_down K-split
    {"out_tp", FAST_P4, 3072, 5120, 64},         // ssm_out (48, Q5_K in file) + attn_output (16), K-split
    {"lmhead_tp_k6", FAST_K6, 5120, 124160, 1},  // output.weight Q6_K, row split
    {"lmhead_tp_q8", FAST_Q8, 5120, 124160, 0},  // Q8_0 variant
};

static uint64_t rng_s = 0x9E3779B97F4A7C15ull;
static inline uint64_t rnd() { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s; }
static inline float rndu() { return (float)((rnd() >> 40) * (1.0 / 16777216.0)); }
static inline float rndn() { float u = rndu() + 1e-7f, v = rndu(); return std::sqrt(-2.f * std::log(u)) * std::cos(6.2831853f * v); }

static void gen_gguf(int fmt, int N, int K, std::vector<uint8_t>& out) {
    size_t rb = src_row_bytes(fmt, K);
    out.resize(rb * N);
    uint8_t* p = out.data();
    const size_t nbytes = out.size();
    for (size_t i = 0; i + 8 <= nbytes; i += 8) { uint64_t r = rnd(); memcpy(p + i, &r, 8); }
    // fix scales to sane values
    int be = src_block_elems(fmt), bb = src_block_bytes(fmt);
    for (int row = 0; row < N; ++row)
        for (int b = 0; b < K / be; ++b) {
            uint8_t* blk = p + (size_t)row * rb + (size_t)b * bb;
            uint16_t d = f2h_host(0.002f + 0.02f * rndu());
            if (fmt == FAST_K6) {
                memcpy(blk + 208, &d, 2);
                for (int s = 0; s < 16; ++s) blk[192 + s] = (uint8_t)(int8_t)((int)(rnd() % 127) - 63);
            } else {
                memcpy(blk, &d, 2);
                if (fmt == FAST_Q8)
                    for (int e = 0; e < 32; ++e) if ((int8_t)blk[2 + e] == -128) blk[2 + e] = (uint8_t)(int8_t)-127;
            }
        }
}

struct Ref { std::vector<int> rows; std::vector<double> y; /* [8][rows] */ std::vector<double> yx; /* float-x ref */ };

static void cpu_reference(int fmt, int N, int K, const std::vector<uint8_t>& src, const std::vector<int8_t>& xq,
                          const std::vector<int32_t>& xm, const std::vector<float>& x, Ref& R) {
    long long work = (long long)N * K;
    int stride = work > 120000000LL ? 23 : 1;
    for (int r = 0; r < N; r += stride) R.rows.push_back(r);
    if (stride > 1) for (int r = std::max(0, N - 64); r < N; ++r) if (R.rows.back() < r) R.rows.push_back(r);
    const int nr = (int)R.rows.size();
    R.y.assign((size_t)8 * nr, 0.0); R.yx.assign((size_t)8 * nr, 0.0);
    const int nb = K / 32;
    std::vector<double> xdq((size_t)8 * K);
    for (int c = 0; c < 8; ++c)
        for (int b = 0; b < nb; ++b) {
            float d; memcpy(&d, &xm[((size_t)c * nb + b) * 2], 4);
            for (int i = 0; i < 32; ++i) xdq[(size_t)c * K + 32 * b + i] = (double)d * xq[(size_t)c * K + 32 * b + i];
        }
    const size_t rb = src_row_bytes(fmt, K);
    int nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t)
        th.emplace_back([&, t]() {
            std::vector<float> w(K);
            for (int i = t; i < nr; i += nt) {
                dequant_row_host(fmt, src.data() + (size_t)R.rows[i] * rb, K, w.data());
                for (int c = 0; c < 8; ++c) {
                    double s = 0, sx = 0;
                    const double* xd = &xdq[(size_t)c * K];
                    const float* xf = &x[(size_t)c * K];
                    for (int k = 0; k < K; ++k) { s += (double)w[k] * xd[k]; sx += (double)w[k] * xf[k]; }
                    R.y[(size_t)c * nr + i] = s; R.yx[(size_t)c * nr + i] = sx;
                }
            }
        });
    for (auto& x_ : th) x_.join();
}

struct Meas { double us = 0, gbps = 0; };

static std::vector<KEntry> g_k;
static NvmlLite g_nvml;
static int g_dev = 0, g_sms = 40;

static const KEntry* find_k(int fmt, int nch, int m, int rpl, int D, int xsm) {
    for (auto& k : g_k)
        if (k.fmt == fmt && k.nch == nch && k.m == m && k.rpl == rpl && k.D == D && k.xsm == xsm) return &k;
    return nullptr;
}

struct Bufs {
    Layout L[3];                      // by rpl index 1,2
    std::vector<uint8_t*> w[3];       // rotation copies
    int8_t* xq = nullptr; int2* xm = nullptr; float* y = nullptr;
};

static int grid_for(const KEntry* k, const Layout& L, int threads, int mode, int* occ_out) {
    int wpb = threads / 32;
    int full = (L.ntiles + wpb - 1) / wpb;
    int occ = k->occ(threads);
    if (occ_out) *occ_out = occ;
    if (occ <= 0) return 0;
    if (mode == 0) return full;
    int pers = occ * g_sms;
    return std::min(pers, full);
}

static Meas time_cfg(const KEntry* k, Bufs& B, int rpl, int threads, int grid, double target_ms) {
    const Layout& L = B.L[rpl];
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    int nc = (int)B.w[rpl].size();
    auto launch = [&](int i) {
        GemvArgs a = make_args(L, B.w[rpl][i % nc], B.xq, B.xm, B.y, L.N);
        CK(k->fn(a, grid, threads, 0));
    };
    for (int i = 0; i < 3; ++i) launch(i);
    CK(cudaEventRecord(e0));
    for (int i = 0; i < 4; ++i) launch(i);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    int iters = (int)std::min(400.0, std::max(10.0, target_ms / (ms / 4)));
    CK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) launch(i);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&ms, e0, e1));
    Meas m; m.us = ms * 1000.0 / iters; m.gbps = (double)L.bytes / (m.us * 1e3);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return m;
}

struct Best { double gbps = 0, us = 0; int rpl = 0, D = 0, xsm = 0, thr = 0, gmode = 0, grid = 0; };

// Sustained load: run one config back-to-back for `secs`, sample NVML every ~100 ms, report windowed GB/s + clocks.
static int run_sustain(const Shape& S, double secs, int rpl, int D, int xsm, int thr, int gmode, int m) {
    const int nch = S.K / 512, K = S.K, N = S.N, nb = K / 32;
    const KEntry* k = find_k(S.fmt, nch, m, rpl, D, xsm);
    if (!k) { printf("SUSTAIN_ERR no kernel\n"); return 1; }
    Layout L = make_layout(S.fmt, N, K, rpl);
    int nc = (int)std::max<size_t>(2, (size_t)(96e6 / L.bytes) + 1);
    std::vector<uint8_t*> w;
    std::vector<uint8_t> h(L.bytes);
    for (size_t i = 0; i + 8 <= h.size(); i += 8) { uint64_t r = rnd(); memcpy(&h[i], &r, 8); }
    // keep fp16 scales finite: overwrite scale planes with 0x2000 (small normal fp16)
    for (size_t i = L.off_d; i + 1 < L.bytes; i += 2) { h[i] = 0x00; h[i + 1] = 0x20; }
    for (int c = 0; c < nc; ++c) {
        uint8_t* p; CK(cudaMalloc(&p, L.bytes)); CK(cudaMemcpy(p, h.data(), L.bytes, cudaMemcpyHostToDevice)); w.push_back(p);
    }
    float* d_x; int8_t* xq; int2* xm; float* y;
    std::vector<float> x((size_t)8 * K); for (auto& v : x) v = rndn();
    CK(cudaMalloc(&d_x, x.size() * 4)); CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&xq, (size_t)8 * K)); CK(cudaMalloc(&xm, (size_t)8 * nb * 8)); CK(cudaMalloc(&y, (size_t)8 * N * 4));
    quantize_q8_kernel<<<(8 * nb * 32 + 255) / 256, 256>>>(d_x, K, 8, xq, xm);
    int grid = grid_for(k, L, thr, gmode, nullptr);
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto t0 = std::chrono::steady_clock::now();
    std::vector<double> gb; std::vector<unsigned> clk, pw; unsigned long long reasons = 0; unsigned tmax = 0;
    int it = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs) {
        // window of ~100 ms
        CK(cudaEventRecord(e0));
        int n = std::max(10, (int)(100e3 / (L.bytes / 250e3)));
        for (int i = 0; i < n; ++i, ++it) {
            GemvArgs a = make_args(L, w[it % nc], xq, xm, y, N);
            CK(k->fn(a, grid, thr, 0));
        }
        CK(cudaEventRecord(e1));
        auto s = g_nvml.sample(g_dev);  // sampled while the window is running
        CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        gb.push_back((double)L.bytes * n / (ms * 1e6));
        clk.push_back(s.sm); pw.push_back(s.mw); reasons |= s.reasons; tmax = std::max(tmax, s.temp);
    }
    auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; };
    std::vector<double> cd(clk.begin(), clk.end()), pd(pw.begin(), pw.end());
    double gmin = gb.empty() ? 0 : *std::min_element(gb.begin(), gb.end());
    double cmin = cd.empty() ? 0 : *std::min_element(cd.begin(), cd.end());
    char rs[128]; nvml_reason_str(reasons, rs, sizeof rs);
    printf("SUSTAIN {\"dev\":%d,\"shape\":\"%s\",\"m\":%d,\"cfg\":\"%d,%d,%d,%d,%d\",\"secs\":%.1f,\"windows\":%zu,"
           "\"GBps_median\":%.1f,\"GBps_min\":%.1f,\"GBps_last\":%.1f,\"sm_mhz_median\":%.0f,\"sm_mhz_min\":%.0f,"
           "\"power_w_median\":%.1f,\"temp_max\":%u,\"reasons_mask\":%llu,\"reasons\":\"%s\"}\n",
           g_dev, S.name, m, rpl, D, xsm, thr, gmode, secs, gb.size(), med(gb), gmin, gb.empty() ? 0 : gb.back(),
           med(cd), cmin, med(pd) / 1000.0, tmax, reasons, rs);
    fflush(stdout);
    for (auto p : w) cudaFree(p);
    cudaFree(d_x); cudaFree(xq); cudaFree(xm); cudaFree(y);
    return 0;
}

int main(int argc, char** argv) {
    bool quick = false; std::string only; double sustain = 0; int cr = 2, cD = 4, cx = 0, ct = 256, cg = 0, cm = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--dev") g_dev = atoi(argv[++i]);
        else if (a == "--quick") quick = true;
        else if (a == "--shape") only = argv[++i];
        else if (a == "--sustain") sustain = atof(argv[++i]);
        else if (a == "--cfg") sscanf(argv[++i], "%d,%d,%d,%d,%d,%d", &cr, &cD, &cx, &ct, &cg, &cm);
    }
    reg_part1(g_k); reg_part2(g_k); reg_part3(g_k); reg_part4(g_k); reg_part5(g_k);
    CK(cudaSetDevice(g_dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, g_dev));
    g_sms = prop.multiProcessorCount;
    g_nvml.init();
    printf("INFO dev=%d name=%s sms=%d kernels=%zu nvml=%d\n", g_dev, prop.name, g_sms, g_k.size(), (int)g_nvml.ok);
    if (sustain > 0) {
        for (const Shape& S : SHAPES)
            if (only == S.name) return run_sustain(S, sustain, cr, cD, cx, ct, cg, cm);
        printf("SUSTAIN_ERR unknown shape\n");
        return 1;
    }

    std::string summary = "{\"dev\":" + std::to_string(g_dev) + ",\"shapes\":[";
    bool first_shape = true, all_ok = true;
    // engine-config aggregates: key "rpl,D,xsm,thr,gmode" -> sum over decode mix of time at m
    struct Agg { std::string key; double t[9] = {0}; double bytes[9] = {0}; int n[9] = {0}; };
    std::vector<Agg> agg;
    double mix_best_t[9] = {0}, mix_bytes[9] = {0};

    for (const Shape& S : SHAPES) {
        if (!only.empty() && only != S.name) continue;
        const int nch = S.K / 512;
        printf("INFO shape %s fmt=%s K=%d N=%d\n", S.name, fmt_name(S.fmt), S.K, S.N); fflush(stdout);
        rng_s = 0x9E3779B97F4A7C15ull ^ (uint64_t)S.N * 7919 ^ (uint64_t)S.K;
        std::vector<uint8_t> src; gen_gguf(S.fmt, S.N, S.K, src);
        Bufs B;
        uint8_t* d_src = nullptr; CK(cudaMalloc(&d_src, src.size()));
        CK(cudaMemcpy(d_src, src.data(), src.size(), cudaMemcpyHostToDevice));
        bool repack_ok = true;
        for (int rpl = 1; rpl <= 2; ++rpl) {
            B.L[rpl] = make_layout(S.fmt, S.N, S.K, rpl);
            const Layout& L = B.L[rpl];
            std::vector<uint8_t> hp(L.bytes);
            repack_host(L, src.data(), hp.data());
            uint8_t* d0; CK(cudaMalloc(&d0, L.bytes));
            CK(repack_device(L, d_src, d0, 0));
            std::vector<uint8_t> dp(L.bytes);
            CK(cudaMemcpy(dp.data(), d0, L.bytes, cudaMemcpyDeviceToHost));
            bool same = memcmp(dp.data(), hp.data(), L.bytes) == 0;
            repack_ok &= same;
            if (!same) CK(cudaMemcpy(d0, hp.data(), L.bytes, cudaMemcpyHostToDevice));  // bench host-packed anyway
            int nc = (int)std::min<size_t>(8, std::max<size_t>(1, (size_t)(96e6 / L.bytes) + 1));
            if (quick) nc = std::min(nc, 2);
            B.w[rpl].push_back(d0);
            for (int c = 1; c < nc; ++c) {
                uint8_t* dc; CK(cudaMalloc(&dc, L.bytes));
                CK(cudaMemcpy(dc, d0, L.bytes, cudaMemcpyDeviceToDevice));
                B.w[rpl].push_back(dc);
            }
        }
        CK(cudaFree(d_src));
        // activations: 8 columns
        const int K = S.K, N = S.N, nb = K / 32;
        std::vector<float> x((size_t)8 * K);
        for (auto& v : x) v = rndn() * (rndu() < 0.01f ? 8.f : 1.f);
        for (int i = 0; i < 64; ++i) x[(size_t)(rnd() % (8 * K))] = 0.f;
        for (int i = 0; i < 32; ++i) x[(size_t)(rnd() % 8) * K + 32 * (rnd() % nb) + i] = 0.f;  // an all-zero group
        float* d_x; CK(cudaMalloc(&d_x, x.size() * 4));
        CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&B.xq, (size_t)8 * K)); CK(cudaMalloc(&B.xm, (size_t)8 * nb * 8));
        CK(cudaMalloc(&B.y, (size_t)8 * N * 4));
        quantize_q8_kernel<<<(8 * nb * 32 + 255) / 256, 256>>>(d_x, K, 8, B.xq, B.xm);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<int8_t> xq((size_t)8 * K), hq((size_t)8 * K);
        std::vector<int32_t> xm((size_t)8 * nb * 2), hm((size_t)8 * nb * 2);
        CK(cudaMemcpy(xq.data(), B.xq, xq.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(xm.data(), B.xm, xm.size() * 4, cudaMemcpyDeviceToHost));
        for (int c = 0; c < 8; ++c) quantize_q8_host(&x[(size_t)c * K], K, &hq[(size_t)c * K], &hm[(size_t)c * nb * 2]);
        long qmis = 0;
        for (size_t i = 0; i < xq.size(); ++i) qmis += xq[i] != hq[i];
        for (size_t i = 0; i < xm.size(); ++i) qmis += xm[i] != hm[i];

        Ref R; cpu_reference(S.fmt, N, K, src, xq, xm, x, R);
        // golden: rpl 2, D 4, no xsm, 256 thr, full grid, m = 8
        const KEntry* gk = find_k(S.fmt, nch, 8, 2, 4, 0);
        {
            GemvArgs a = make_args(B.L[2], B.w[2][0], B.xq, B.xm, B.y, N);
            int grid = grid_for(gk, B.L[2], 256, 0, nullptr);
            CK(gk->fn(a, grid, 256, 0)); CK(cudaDeviceSynchronize());
        }
        std::vector<float> y8((size_t)8 * N);
        CK(cudaMemcpy(y8.data(), B.y, y8.size() * 4, cudaMemcpyDeviceToHost));
        const int nr = (int)R.rows.size();
        double maxerr = 0, rms = 0, l2e = 0, l2r = 0;
        for (int c = 0; c < 8; ++c)
            for (int i = 0; i < nr; ++i) {
                double ref = R.y[(size_t)c * nr + i], g = y8[(size_t)c * N + R.rows[i]];
                maxerr = std::max(maxerr, std::fabs(g - ref)); rms += ref * ref;
                double rx = R.yx[(size_t)c * nr + i]; l2e += (g - rx) * (g - rx); l2r += rx * rx;
            }
        rms = std::sqrt(rms / (8.0 * nr));
        double nerr = maxerr / (rms + 1e-30), qerr = std::sqrt(l2e / (l2r + 1e-30));
        bool ok = repack_ok && qmis == 0 && nerr < 1e-4 && std::isfinite(nerr);
        printf("CHECK {\"shape\":\"%s\",\"repack_dev_eq_host\":%d,\"q8_mismatch\":%ld,\"rows_checked\":%d,"
               "\"max_err_over_rms\":%.3e,\"rel_l2_vs_float_x\":%.3e,\"ok\":%d}\n",
               S.name, (int)repack_ok, qmis, nr, nerr, qerr, (int)ok);
        fflush(stdout);
        all_ok &= ok;

        // sweep
        std::vector<float> yb((size_t)8 * N);
        Best best[9];
        int bitmis_total = 0, nmeas = 0;
        const int ms_list_full[] = {1, 2, 3, 4, 5, 6, 7, 8};
        const int ms_list_quick[] = {1, 4, 8};
        const int* ml = quick ? ms_list_quick : ms_list_full;
        int nm = quick ? 3 : 8;
        const double target_ms = quick ? 10 : 25;
        for (int mi = 0; mi < nm; ++mi) {
            int m = ml[mi];
            for (int rpl = 1; rpl <= 2; ++rpl)
                for (int D : {2, 4})
                    for (int xsm = 0; xsm <= 1; ++xsm)
                        for (int thr : {128, 256})
                            for (int gmode = 0; gmode <= 1; ++gmode) {
                                const KEntry* k = find_k(S.fmt, nch, m, rpl, D, xsm);
                                if (!k) continue;
                                if (xsm && xsm_bytes(m, K) > 64 * 1024) continue;
                                int occ = 0;
                                int grid = grid_for(k, B.L[rpl], thr, gmode, &occ);
                                if (grid <= 0) continue;
                                if (xsm && gmode == 0 && grid > 4 * occ * g_sms) continue;  // xsm only sensible persistent-ish
                                // correctness: bit-identical to golden on all rows
                                CK(cudaMemset(B.y, 0xff, (size_t)8 * N * 4));
                                GemvArgs a = make_args(B.L[rpl], B.w[rpl][0], B.xq, B.xm, B.y, N);
                                CK(k->fn(a, grid, thr, 0)); CK(cudaDeviceSynchronize());
                                CK(cudaMemcpy(yb.data(), B.y, (size_t)m * N * 4, cudaMemcpyDeviceToHost));
                                int bm = memcmp(yb.data(), y8.data(), (size_t)m * N * 4) != 0;
                                bitmis_total += bm;
                                Meas me = time_cfg(k, B, rpl, thr, grid, target_ms);
                                ++nmeas;
                                auto smp = g_nvml.sample(g_dev);
                                printf("R {\"shape\":\"%s\",\"fmt\":\"%s\",\"K\":%d,\"N\":%d,\"m\":%d,\"rpl\":%d,\"minb\":%d,"
                                       "\"xsm\":%d,\"thr\":%d,\"gmode\":%d,\"grid\":%d,\"occ\":%d,\"us\":%.2f,\"GBps\":%.1f,"
                                       "\"bitexact\":%d,\"sm_mhz\":%u}\n",
                                       S.name, fmt_name(S.fmt), K, N, m, rpl, D, xsm, thr, gmode, grid, occ, me.us,
                                       me.gbps, 1 - bm, smp.sm);
                                if (!bm && me.gbps > best[m].gbps) {
                                    best[m].gbps = me.gbps; best[m].us = me.us; best[m].rpl = rpl; best[m].D = D;
                                    best[m].xsm = xsm; best[m].thr = thr; best[m].gmode = gmode; best[m].grid = grid;
                                }
                                if (!bm && S.count_per_token > 0) {
                                    char key[64]; snprintf(key, 64, "%d,%d,%d,%d,%d", rpl, D, xsm, thr, gmode);
                                    Agg* ag = nullptr;
                                    for (auto& g2 : agg) if (g2.key == key) ag = &g2;
                                    if (!ag) { agg.push_back(Agg{}); agg.back().key = key; ag = &agg.back(); }
                                    if (S.fmt == FAST_P4) {
                                        ag->t[m] += me.us * S.count_per_token;
                                        ag->bytes[m] += (double)B.L[rpl].bytes * S.count_per_token;
                                        ag->n[m] += 1;
                                    }
                                }
                            }
            fflush(stdout);
        }
        if (S.fmt == FAST_P4 && S.count_per_token > 0)
            for (int m = 1; m <= 8; ++m)
                if (best[m].gbps > 0) {
                    mix_best_t[m] += best[m].us * S.count_per_token;
                    mix_bytes[m] += (double)B.L[2].bytes * S.count_per_token;
                }
        all_ok &= bitmis_total == 0;
        char buf[512];
        snprintf(buf, sizeof buf, "%s{\"shape\":\"%s\",\"fmt\":\"%s\",\"K\":%d,\"N\":%d,\"bytes\":%zu,\"check_ok\":%d,"
                 "\"bit_mismatch_cfgs\":%d,\"n_meas\":%d,\"best\":[", first_shape ? "" : ",", S.name, fmt_name(S.fmt), K,
                 N, B.L[2].bytes, (int)ok, bitmis_total, nmeas);
        summary += buf; first_shape = false;
        bool fb = true;
        for (int m = 1; m <= 8; ++m) {
            if (best[m].gbps <= 0) continue;
            snprintf(buf, sizeof buf, "%s{\"m\":%d,\"GBps\":%.1f,\"us\":%.2f,\"cfg\":\"%d,%d,%d,%d,%d\",\"grid\":%d}",
                     fb ? "" : ",", m, best[m].gbps, best[m].us, best[m].rpl, best[m].D, best[m].xsm, best[m].thr,
                     best[m].gmode, best[m].grid);
            summary += buf; fb = false;
        }
        summary += "]}";
        printf("BEST %s:", S.name);
        for (int m = 1; m <= 8; ++m) if (best[m].gbps > 0) printf(" m%d=%.1f", m, best[m].gbps);
        printf("\n"); fflush(stdout);
        // free
        for (int r = 1; r <= 2; ++r) for (auto p : B.w[r]) cudaFree(p);
        cudaFree(B.xq); cudaFree(B.xm); cudaFree(B.y); cudaFree(d_x);
    }
    summary += "],\"mix_best_per_shape\":{";
    for (int m = 1; m <= 8; ++m) {
        char b[96]; snprintf(b, 96, "%s\"m%d\":%.1f", m > 1 ? "," : "", m, mix_best_t[m] > 0 ? mix_bytes[m] / (mix_best_t[m] * 1e3) : 0.0);
        summary += b;
    }
    summary += "},\"mix_single_cfg\":[";
    // best single config: maximize m=1 + m=4 mix bandwidth, requiring all P4 mix shapes present
    int nmix = 0; for (const Shape& S : SHAPES) if (S.fmt == FAST_P4 && S.count_per_token > 0 && (only.empty() || only == S.name)) ++nmix;
    std::sort(agg.begin(), agg.end(), [](const Agg& a, const Agg& b) {
        auto sc = [](const Agg& g) { return (g.t[1] > 0 ? g.bytes[1] / g.t[1] : 0) + (g.t[4] > 0 ? g.bytes[4] / g.t[4] : 0); };
        return sc(a) > sc(b);
    });
    int printed = 0;
    for (auto& g : agg) {
        if (g.n[1] != nmix) continue;
        if (printed == 5) break;
        summary += printed ? ",{" : "{";
        char b[96]; snprintf(b, 96, "\"cfg\":\"%s\"", g.key.c_str()); summary += b;
        for (int m = 1; m <= 8; ++m)
            if (g.t[m] > 0 && g.n[m] == nmix) { snprintf(b, 96, ",\"m%d\":%.1f", m, g.bytes[m] / (g.t[m] * 1e3)); summary += b; }
        summary += "}";
        ++printed;
    }
    summary += "],\"all_ok\":" + std::to_string((int)all_ok) + "}";
    printf("SUMMARY %s\n", summary.c_str());
    fflush(stdout);
    return all_ok ? 0 : 1;
}
#endif
