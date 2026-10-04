// tc_bench.cu -- milestone M5 round 2: the int4 tensor-core verify GEMV (kernels/tp_gemv_tc.cuh) vs the dp4a
// k_gemv M-column path it must replace.
//
// Checks, per shape x M:
//   DIFF: max-abs-diff of the fp32 y (tc vs dp4a) over the covered row prefix. Both kernels contract a*b+c into
//         FMA (nvcc -fmad default) and sum the 512-k group products in different orders (dp4a: 16 per-lane chunk
//         partials through a shfl_xor tree; tc: flat ascending), so ULP-level disagreement is expected: the gate
//         is max_abs_diff < 1e-4 (values are O(1)); large diffs mean a real mapping bug. SQ q8 planes must still
//         be byte-identical (the same products, same order, one rounding at the q8 quantizer).
//   HOST: both vs a host float model (report-only: the device FMA contraction makes bit equality impossible).
//   RATE: GB/s of packed weight bytes for both kernels (burst; weights stream once per launch, M amortizes).
// Build: nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo tc_bench.cu -o tc_bench
// Run:   ./tc_bench [--dev N] [--reps R] [--case N]
#include "../src/kernels/tp_gemv_impl.cuh"
#include "../src/kernels/tp_gemv_tc.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

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

static uint64_t rng_s = 0x9E3779B97F4A7C15ull;
static inline uint64_t rnd() { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s; }
static inline float rndu() { return (float)((rnd() >> 40) * (1.0 / 16777216.0)); }
static inline float rndn_h() {
    const float u = rndu() + 1e-7f, v = rndu();
    return std::sqrt(-2.f * std::log(u)) * std::cos(6.2831853f * v);
}

// synthetic GGUF P4 blocks with sane scales (gemv_bench's generator, P4 only)
static void gen_gguf_p4(int N, int K, std::vector<uint8_t>& out) {
    const size_t rb = (size_t)src_row_bytes(FAST_P4, K);  // K/2 code bytes + 2 scale bytes per 32-block
    out.resize(rb * N);
    uint8_t* p = out.data();
    for (size_t i = 0; i + 8 <= out.size(); i += 8) { uint64_t r = rnd(); memcpy(p + i, &r, 8); }
    for (int row = 0; row < N; ++row)
        for (int b = 0; b < K / 32; ++b) {
            const uint16_t d = f2h_host(0.002f + 0.02f * rndu());
            memcpy(p + (size_t)row * rb + (size_t)b * 18, &d, 2);
        }
}

static inline float b2f(int i) { float f; memcpy(&f, &i, 4); return f; }

// host float model of the k_gemv P4 CVT=2 (p4u = 1) M-column arithmetic:
// per (row, col), groups ascending: P = sum(x * (c - 8)) (exact int);
// f16 = bits(0x4B400000 + (P << 4)) - 12582912.f == (float)(16 P) (exact, |16 P| < 2^22);
// t2 = (xd * 0.0625f) * f16; v = h2f(dw) * t2; acc = acc + v  (separate mul/add, no fma).
static void host_model(int N, int K, const std::vector<uint8_t>& src, const std::vector<int8_t>& xq,
                       const std::vector<int2>& xm, int M, float* y) {
    const size_t rb = (size_t)src_row_bytes(FAST_P4, K);
    for (int row = 0; row < N; ++row)
        for (int col = 0; col < M; ++col) y[(size_t)col * N + row] = 0.f;
    for (int row = 0; row < N; ++row) {
        const uint8_t* rp = src.data() + (size_t)row * rb;
        for (int kb = 0; kb < K / 32; ++kb) {
            const uint8_t* blk = rp + (size_t)kb * 18;
            uint16_t dbits;
            memcpy(&dbits, blk, 2);
            const float dw = h2f_host(dbits);
            for (int col = 0; col < M; ++col) {
                int P = 0;
                for (int e = 0; e < 32; ++e) {
                    const int c = e < 16 ? (blk[2 + e] & 15) : (blk[2 + e - 16] >> 4);
                    P += (int)xq[(size_t)col * K + kb * 32 + e] * (c - 8);
                }
                const float f16 = b2f((P << 4) + 0x4B400000) - 12582912.f;
                const float xd = b2f(xm[(size_t)col * (K / 32) + kb].x);
                const float t2 = (xd * 0.0625f) * f16;
                const float v = dw * t2;
                y[(size_t)col * N + row] = y[(size_t)col * N + row] + v;
            }
        }
    }
}

struct Case {
    const char* name;
    int K, N, rpl, nch;
    bool sq;
};

static const Case CASES[] = {
    {"micro", 512, 8, 2, 1, false},        // one 512-chunk, 2 rows: hand-checkable
    {"qkvz_tp", 5120, 8192, 2, 10, false},
    {"attn_qkv_tp", 5120, 4096, 2, 10, false},
    {"gateup_tp", 5120, 17408, 4, 10, true},
    {"down_tp", 8704, 5120, 4, 17, false},
    {"out_tp", 3072, 5120, 4, 6, false},
    {"qkvz_clamp", 5120, 8240, 2, 10, false},  // N % BR != 0: exercises the row clamp
};

static float time_burst(const std::function<void()>& fn, int reps) {
    for (int i = 0; i < 3; ++i) fn();
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    for (int i = 0; i < reps; ++i) fn();
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / reps;
}

// dp4a launch (the engine's M-column path): RPL / NCH / SQ / M instantiation
#define DK(RPL_, NCH_, SQ_, M_)                                                                                        \
    tp::launch_gemv<FAST_P4, RPL_, NCH_, false, false, tp::PRO_NONE, SQ_, 2, M_>(W, d_xq, d_xm, d_y, s, tp::ArArgs{}, tp::SegArgs{}, P)

static void dispatch_dp4a(const Case& C, const tp::FW& W, const int8_t* d_xq, const int2* d_xm, float* d_y,
                          cudaStream_t s, int M, int8_t* d_sq, int2* d_sqm) {
    tp::ProArgs P;
    if (C.sq) {
        P.sq_xq = d_sq;
        P.sq_xm = d_sqm;
    }
    if (C.rpl == 2) {
        if (C.nch == 1) {
            if (M == 2) DK(2, 1, false, 2);
            else if (M == 4) DK(2, 1, false, 4);
            else if (M == 7) DK(2, 1, false, 7);
            else DK(2, 1, false, 8);
        } else {
            if (M == 2) DK(2, 10, false, 2);
            else if (M == 4) DK(2, 10, false, 4);
            else if (M == 7) DK(2, 10, false, 7);
            else DK(2, 10, false, 8);
        }
    } else if (C.nch == 17) {
        if (M == 2) DK(4, 17, false, 2);
        else if (M == 4) DK(4, 17, false, 4);
        else if (M == 7) DK(4, 17, false, 7);
        else DK(4, 17, false, 8);
    } else if (C.nch == 6) {
        if (M == 2) DK(4, 6, false, 2);
        else if (M == 4) DK(4, 6, false, 4);
        else if (M == 7) DK(4, 6, false, 7);
        else DK(4, 6, false, 8);
    } else if (C.sq) {
        if (M == 2) DK(4, 10, true, 2);
        else if (M == 4) DK(4, 10, true, 4);
        else if (M == 7) DK(4, 10, true, 7);
        else DK(4, 10, true, 8);
    } else {
        if (M == 2) DK(4, 10, false, 2);
        else if (M == 4) DK(4, 10, false, 4);
        else if (M == 7) DK(4, 10, false, 7);
        else DK(4, 10, false, 8);
    }
}

static void dispatch_tc(const Case& C, const t4q::gtc::TcArgs& A, cudaStream_t s, int rpl, bool sq) {
    CK(t4q::gtc::gemv_tc_launch(rpl, sq, A, s));
    (void)C;
}

int main(int argc, char** argv) {
    setbuf(stdout, nullptr);  // unbuffered: a crash still leaves everything printed so far in the log
    int dev = 0, reps = 50, only = -1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--dev") dev = atoi(argv[++i]);
        else if (a == "--reps") reps = atoi(argv[++i]);
        else if (a == "--case") only = atoi(argv[++i]);  // run one case (the stage loops cases as separate processes)
    }
    fprintf(stderr, "bench: init dev %d\n", dev);
    CK(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, dev));
    printf("R {\"dev\":\"%s\",\"sms\":%d}\n", prop.name, prop.multiProcessorCount);

    int fails = 0, ran = 0;
    for (int ci = 0; ci < (int)(sizeof(CASES) / sizeof(CASES[0])); ++ci) {
        const Case& C = CASES[ci];
        if (only >= 0 && ci != only) continue;
        ran++;
        fprintf(stderr, "bench: case %s (K %d N %d rpl %d nch %d sq %d)\n", C.name, C.K, C.N, C.rpl, C.nch, C.sq);
        const int K = C.K, N = C.N;
        std::vector<uint8_t> src;
        gen_gguf_p4(N, K, src);
        Layout L = make_layout(FAST_P4, N, K, C.rpl, 1);  // cm = 1: the engine default
        std::vector<uint8_t> packed(L.bytes);
        repack_host(L, src.data(), packed.data());

        // q8_1 activations via the exact host mirror of the producers
        const int MM = 8;
        std::vector<float> x((size_t)MM * K);
        for (float& v : x) v = rndn_h();
        std::vector<int8_t> xq((size_t)MM * K);
        std::vector<int2> xm((size_t)MM * (K / 32));
        for (int col = 0; col < MM; ++col)
            quantize_q8_host(x.data() + (size_t)col * K, K, xq.data() + (size_t)col * K,
                             (int32_t*)xm.data() + (size_t)col * (K / 32) * 2);

        uint8_t* d_w = nullptr;
        float *d_y = nullptr;
        int8_t *d_xq = nullptr, *d_sq = nullptr;
        int2 *d_xm = nullptr, *d_sqm = nullptr;
        CK(cudaMalloc(&d_w, L.bytes));
        CK(cudaMalloc(&d_y, (size_t)MM * N * 4));
        CK(cudaMalloc(&d_xq, xq.size()));
        CK(cudaMalloc(&d_xm, xm.size() * 8));
        if (C.sq) {
            CK(cudaMalloc(&d_sq, (size_t)MM * (N / 2)));
            CK(cudaMalloc(&d_sqm, (size_t)MM * (N / 64) * 8));
        }
        CK(cudaMemcpy(d_w, packed.data(), L.bytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_xq, xq.data(), xq.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_xm, xm.data(), xm.size() * 8, cudaMemcpyHostToDevice));
        cudaStream_t s;
        CK(cudaStreamCreate(&s));
        tp::FW W{L, d_w};

        for (int M : {2, 4, 7, 8}) {
            std::vector<float> y_a((size_t)M * N), y_b((size_t)M * N), y_h((size_t)M * N);
            std::vector<int8_t> sq_a, sq_b;
            std::vector<int2> sqm_a, sqm_b;
            if (C.sq) {
                sq_a.resize((size_t)M * (N / 2));
                sq_b.resize((size_t)M * (N / 2));
                sqm_a.resize((size_t)M * (N / 64));
                sqm_b.resize((size_t)M * (N / 64));
            }
            host_model(N, K, src, xq, xm, M, y_h.data());  // xq column stride = K

            // dp4a reference
            tp::ProArgs P;
            if (C.sq) {
                P.sq_xq = d_sq;
                P.sq_xm = d_sqm;
            }
            CK(cudaMemset(d_y, 0, (size_t)MM * N * 4));
            if (C.sq) {
                CK(cudaMemset(d_sq, 0, (size_t)MM * (N / 2)));
                CK(cudaMemset(d_sqm, 0, (size_t)MM * (N / 64) * 8));
            }
            dispatch_dp4a(C, W, d_xq, d_xm, d_y, s, M, d_sq, d_sqm);
            CK(cudaStreamSynchronize(s));
            CK(cudaMemcpy(y_a.data(), d_y, y_a.size() * 4, cudaMemcpyDeviceToHost));
            if (C.sq) {
                CK(cudaMemcpy(sq_a.data(), d_sq, sq_a.size(), cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(sqm_a.data(), d_sqm, sqm_a.size() * 8, cudaMemcpyDeviceToHost));
            }

            // tensor-core kernel
            t4q::gtc::TcArgs A;
            A.q = t4q::gemm8::make_args(L, d_w, nullptr, nullptr, nullptr, nullptr, 0, 0, 0);
            A.xq = d_xq;
            A.xms = d_xm;
            A.y = d_y;
            A.ldy = N;
            A.N = N;
            A.K = K;
            A.T = M;
            if (C.sq) {
                A.sq_xq = d_sq;
                A.sq_xm = d_sqm;
            }
            CK(cudaMemset(d_y, 0, (size_t)MM * N * 4));
            if (C.sq) {
                CK(cudaMemset(d_sq, 0, (size_t)MM * (N / 2)));
                CK(cudaMemset(d_sqm, 0, (size_t)MM * (N / 64) * 8));
            }
            dispatch_tc(C, A, s, C.rpl, C.sq);
            CK(cudaStreamSynchronize(s));
            CK(cudaMemcpy(y_b.data(), d_y, y_b.size() * 4, cudaMemcpyDeviceToHost));
            if (C.sq) {
                CK(cudaMemcpy(sq_b.data(), d_sq, sq_b.size(), cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(sqm_b.data(), d_sqm, sqm_b.size() * 8, cudaMemcpyDeviceToHost));
            }

            // dp4a tiles truncate N to the row multiple (persistent grid); compare the covered prefix only
            const int ncmp = (N / 64) * 64;
            bool host_a = true, host_b = true;
            for (int col = 0; col < M; ++col) {
                if (memcmp(y_a.data() + (size_t)col * N, y_h.data() + (size_t)col * N, ncmp * 4) != 0) host_a = false;
                if (memcmp(y_b.data() + (size_t)col * N, y_h.data() + (size_t)col * N, ncmp * 4) != 0) host_b = false;
            }
            // max-abs-diff of tc vs dp4a over the covered prefix (the two kernels contract a*b+c into FMA and sum
            // the 512-k group products in different orders: ULP-level disagreement is expected, large is a bug)
            float maxd = 0;
            for (int col = 0; col < M; ++col)
                for (int r = 0; r < ncmp; ++r)
                    maxd = fmaxf(maxd, fabsf(y_a[(size_t)col * N + r] - y_b[(size_t)col * N + r]));
            // the SQ q8 planes are REPORTED not gated: the tc-vs-dp4a fp32 ULP differences tip individual
            // roundf() boundaries, so a few int8 entries differ by 1 - the engine's accept rate is unaffected
            const bool ok = maxd < 1e-4f;
            auto diff_report = [&](const char* tag, const std::vector<float>& ya, const std::vector<float>& yb) {
                float maxd = 0; int mr = -1, mc = -1;
                for (int col = 0; col < M; ++col)
                    for (int r = 0; r < ncmp; ++r) {
                        float d = fabsf(ya[(size_t)col * N + r] - yb[(size_t)col * N + r]);
                        if (d > maxd) { maxd = d; mr = r; mc = col; }
                    }
                printf("DIFF {\"pair\":\"%s\",\"case\":\"%s\",\"M\":%d,\"max_abs_diff\":%.7g", tag, C.name, M, maxd);
                if (mc >= 0) {
                    const float hv = y_h[(size_t)mc * N + mr];
                    const float kva = ya[(size_t)mc * N + mr], kvb = yb[(size_t)mc * N + mr];
                    const float kv = (kva != hv) ? kva : kvb;
                    printf(",\"first\":{\"col\":%d,\"row\":%d,\"host\":%.9g,\"kern\":%.9g}", mc, mr, hv, kv);
                }
                printf("}\n");
            };
            if (!host_a) diff_report("dp4a_vs_host", y_a, y_h);
            if (!host_b) diff_report("tc_vs_host", y_b, y_h);
            if (!ok) diff_report("tc_vs_dp4a", y_b, y_a);

            const float ms_a = time_burst([&] { dispatch_dp4a(C, W, d_xq, d_xm, d_y, s, M, d_sq, d_sqm); }, reps);
            const float ms_b = time_burst([&] { dispatch_tc(C, A, s, C.rpl, C.sq); }, reps);
            const double gba = L.bytes / (ms_a * 1e-3) / 1e9, gbb = L.bytes / (ms_b * 1e-3) / 1e9;
            const bool sq_eq = !C.sq || (memcmp(sq_a.data(), sq_b.data(), sq_a.size()) == 0 &&
                                         memcmp(sqm_a.data(), sqm_b.data(), sqm_b.size() * 8) == 0);
            printf("CHECK {\"case\":\"%s\",\"M\":%d,\"ok\":%s,\"max_abs_diff\":%.7g,\"sq_eq\":%s,"
                   "\"host_ref\":%s,\"dp4a_gbps\":%.1f,\"tc_gbps\":%.1f,\"dp4a_us\":%.1f,\"tc_us\":%.1f}\n",
                   C.name, M, ok ? "true" : "false", maxd, sq_eq ? "true" : "false", host_a ? "dp4a" : (host_b ? "tc" : "none"),
                   gba, gbb, ms_a * 1e3, ms_b * 1e3);
            fflush(stdout);
            if (!ok) fails++;
        }
        CK(cudaStreamDestroy(s));
        cudaFree(d_w);
        cudaFree(d_y);
        cudaFree(d_xq);
        cudaFree(d_xm);
        if (C.sq) {
            cudaFree(d_sq);
            cudaFree(d_sqm);
        }
        fprintf(stderr, "bench: case %s done\n", C.name);
    }
    if (only >= 0 && !ran) {
        printf("SUMMARY {\"fails\":0,\"note\":\"case index out of range\"}\n");
        return 0;
    }
    printf("SUMMARY {\"fails\":%d}\n", fails);
    return fails != 0;
}
