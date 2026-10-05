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
    bool dp4a;  // false: host-model comparison only (the dp4a NCH templates cover only nch 1/6/10/17)
};

static const Case CASES[] = {
    {"micro1", 512, 64, 2, 1, false, false},    // one block, nst=8, no prefetch; ncmp=64 real rows
    {"micro2", 4096, 64, 2, 8, false, false},   // one block, nst=64: the L2 prefetch path fires
    {"qkvz_tp", 5120, 8192, 2, 10, false, true},
    {"attn_qkv_tp", 5120, 4096, 2, 10, false, true},
    {"gateup_tp", 5120, 17408, 4, 10, true, true},
    {"down_tp", 8704, 5120, 4, 17, false, true},
    {"out_tp", 3072, 5120, 4, 6, false, true},
    {"qkvz_clamp", 5120, 8240, 2, 10, false, true},  // N % BR != 0: exercises the row clamp
    // r11 diagnostics (fixed-vs-marginal per-launch cost): n1024/n2048 extend the qkvz-class N sweep
    // (1024, 2048, 4096=attn, 8192=qkvz) at fixed K 5120 rpl 2 - a linear fit over N gives the fixed
    // launch cost and the marginal N cost separately; outk5 extends the out-class K sweep (3072=out,
    // 5120=outk5, 8704=down) at fixed N 5120 rpl 4 - the same split over K (the compute+code-load length).
    {"n1024_tp", 5120, 1024, 2, 10, false, true},   // 16 blocks at BR 64: grid underfill on purpose
    {"n2048_tp", 5120, 2048, 2, 10, false, true},   // 32 blocks at BR 64: partial fill on purpose
    {"outk5_tp", 5120, 5120, 4, 10, false, true},   // out-class at qkvz's K: the K-length effect at fixed N
};

// Timing methodology (r7): 20 warmups (absorb the module load AND the idle->boost clock ramp - v26/v29's
// first-M blocks measured 1.4-2.1x slow while the machine settled), then 4 windows of reps/4 and the MIN
// window's per-rep (the least-contaminated window; a host under memory pressure bloats whole windows, and
// v29's run had the OOM killer reaping the clock monitor and case tails mid-bench).
static float time_burst(const std::function<void()>& fn, int reps) {
    for (int i = 0; i < 20; ++i) fn();
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    float best = 1e30f;
    const int wn = 4, wr = reps / wn >= 1 ? reps / wn : reps;
    for (int w = 0; w < wn; ++w) {
        CK(cudaEventRecord(e0));
        for (int i = 0; i < wr; ++i) fn();
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        if (ms / wr < best) best = ms / wr;
        if (wr == reps) break;  // reps < wn: one window only
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return best;
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

static cudaError_t dispatch_tc(const t4q::gtc::TcArgs& A, cudaStream_t s, int rpl, bool sq, int vreg, int vsmem, int hoist,
                               int bat, int ar, bool br64, int wn) {
    return t4q::gtc::gemv_tc_launch_v(rpl, sq, vreg, vsmem, hoist, bat, ar, br64, wn, A, s);
}

// ring variants A/B'd per shape (the r8 matrix + r11): v = <RREG, RSMEM, HOIST, BAT, AR, force BR 64, WN>.
// HOIST pins the group's A-side global loads up front; BAT turns the per-step 4x LDG.128 W batch into an
// 8-wide pair burst; AR moves the A-side one group early (r7); r8 moved the d words from smem to a shfl'd
// register ring (STAGE -4 KB: the star runs 16 warps/SM at BR 64, was 12). r10's WN 8 twins (one n-tile
// per warp, 2x the total warps) are DROPPED from the matrix - refuted at -25% on every shape (GB/s tracks
// the per-SM in-flight bytes, not the warp count; the engine's gemv_tc_launch comment keeps the verdict)
// - and their dispatch combos freed the instantiation budget the r11 twins need (v33's 30-combo compile
// OOM'd the 13 GB worker). IDS EQUAL ARRAY INDICES (0..7) - v27 passed ids as indices into the table, so
// ids landed on the wrong rows: the loop below resolves ids by LOOKING UP the v field instead. The r11
// RREG-4 twins V8 (star-R4) / V9 (AR-R4) double the per-warp W cover (8 KB) at half the warps (2 blocks =
// 8/SM at BR 64, ~182/206 regs): r8's depth-4 d-ring collision is cleared by the 8-deep (2*RREG) d-ring.
struct VInfo {
    int v, vreg, vsmem, hoist, bat, ar, wn;
    bool br64;
};
static const VInfo VS[] = {
    {0, 2, 1, 0, 0, 0, 16, false},  // control: the engine default path (per-step W issue, per-stage A reads)
    {1, 2, 1, 0, 0, 0, 16, true},   // the same with BR 64 forced for big N (v26's gateup winner)
    {2, 2, 2, 1, 1, 0, 16, false},  // the star: A-hoist + burst-2, r8 16 warps/SM (was 12 at BR 64, 8 at BR 128)
    {3, 2, 2, 1, 1, 0, 16, true},   // the star with BR 64 forced for big N (the ragged-grid A/B)
    {4, 2, 2, 1, 1, 1, 16, false},  // r7 A-ring: the star + next group's A-side one group early
    {5, 2, 2, 1, 1, 1, 16, true},   // the A-ring with BR 64 forced
    {8, 4, 2, 1, 1, 0, 16, true},  // r11 star-R4: the star at RREG 4 (2x the per-warp W cover, 8-deep d-ring)
    {9, 4, 2, 1, 1, 1, 16, true},  // r11 AR-R4: the A-ring at RREG 4 (the promoted-schedule shape at 8 warps/SM)
};

int main(int argc, char** argv) {
    setbuf(stdout, nullptr);  // unbuffered: a crash still leaves everything printed so far in the log
    int dev = 0, reps = 50, only = -1;
    int vrun[8], nvr = 0, mflt[8], nmf = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--dev") dev = atoi(argv[++i]);
        else if (a == "--reps") reps = atoi(argv[++i]);
        else if (a == "--case") only = atoi(argv[++i]);  // run one case (the stage loops cases as separate processes)
        else if (a == "--variants") {  // comma list of variant ids (default 0)
            std::string s = argv[++i];
            size_t pos = 0;
            while (pos < s.size() && nvr < 8) {
                size_t c = s.find(',', pos);
                vrun[nvr++] = atoi(s.substr(pos, c == std::string::npos ? std::string::npos : c - pos).c_str());
                pos = (c == std::string::npos) ? s.size() : c + 1;
            }
        } else if (a == "--mfilter") {  // comma list of M values (default 2,4,7,8; ncu runs pin one)
            std::string s = argv[++i];
            size_t pos = 0;
            while (pos < s.size() && nmf < 8) {
                size_t c = s.find(',', pos);
                mflt[nmf++] = atoi(s.substr(pos, c == std::string::npos ? std::string::npos : c - pos).c_str());
                pos = (c == std::string::npos) ? s.size() : c + 1;
            }
        }
    }
    if (nvr == 0) vrun[nvr++] = 0;
    int mlist[8], nml = 0;  // the M sweep, optionally filtered
    for (int M : {2, 4, 7, 8}) {
        bool want = nmf == 0;
        for (int i = 0; i < nmf; ++i) want = want || mflt[i] == M;
        if (want) mlist[nml++] = M;
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

        for (int M : mlist) {
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

            // dp4a reference (the real shapes only; its NCH templates cover nch 1/6/10/17)
            if (C.dp4a) {
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
            // dp4a tiles truncate N to the row multiple (persistent grid); compare the covered prefix only
            const int ncmp = (N / 64) * 64;
            bool host_a = true;
            if (C.dp4a)
                for (int col = 0; col < M; ++col)
                    if (memcmp(y_a.data() + (size_t)col * N, y_h.data() + (size_t)col * N, ncmp * 4) != 0)
                        host_a = false;
            int cur_br = N >= 56 * 128 ? 128 : 64;  // the running variant's BR (diff_report's decomposition)
            auto diff_report = [&](const char* tag, int vid, const std::vector<float>& ya,
                                   const std::vector<float>& yb) {
                float maxd = 0; int mr = -1, mc = -1;
                int nzero = 0, nnan = 0;
                int zx[3][2]; int nz = 0;
                for (int col = 0; col < M; ++col)
                    for (int r = 0; r < N; ++r) {
                        const float kv = ya[(size_t)col * N + r];
                        if (std::isnan(kv)) nnan++;
                        if (kv == 0.f && y_h[(size_t)col * N + r] != 0.f) {
                            if (nz < 3) { zx[nz][0] = col; zx[nz][1] = r; }
                            nz++;
                        }
                        float d = fabsf(kv - yb[(size_t)col * N + r]);
                        if (!std::isnan(d) && d > maxd) { maxd = d; mr = r; mc = col; }
                    }
                printf("DIFF {\"pair\":\"%s\",\"case\":\"%s\",\"M\":%d,\"v\":%d,\"max_abs_diff\":%.7g,"
                       "\"nzero\":%d,\"nnan\":%d",
                       tag, C.name, M, vid, maxd, nz, nnan);
                for (int i = 0; i < (nz < 3 ? nz : 3); ++i) {
                    const int r = zx[i][1];
                    const int br = cur_br;
                    // decomposition: col, row, blockIdx.y, warp (16-row group), 8-row tile, row in tile
                    printf(",\"z%d\":[%d,%d,%d,%d,%d,%d]", i, zx[i][0], r, r / br, (r % br) / 16, (r % 16) / 8,
                           r & 7);
                }
                if (mc >= 0) {
                    const float hv = y_h[(size_t)mc * N + mr];
                    const float kva = ya[(size_t)mc * N + mr], kvb = yb[(size_t)mc * N + mr];
                    const float kv = (kva != hv) ? kva : kvb;
                    printf(",\"first\":{\"col\":%d,\"row\":%d,\"host\":%.9g,\"kern\":%.9g}", mc, mr, hv, kv);
                }
                printf("}\n");
            };
            if (C.dp4a && !host_a) diff_report("dp4a_vs_host", -1, y_a, y_h);

            const float ms_a = C.dp4a ? time_burst([&] { dispatch_dp4a(C, W, d_xq, d_xm, d_y, s, M, d_sq, d_sqm); }, reps) : 0.f;
            const double gba = C.dp4a ? L.bytes / (ms_a * 1e-3) / 1e9 : 0.0;

            // one run per ring variant: the numeric path is identical across variants (only the ring
            // scheduling and the smem buffer count differ), so the same gates bind each
            for (int vi = 0; vi < nvr; ++vi) {
                const VInfo* Vp = nullptr;  // resolve the id by LOOKUP, never by array position (v27's
                for (int k = 0; k < (int)(sizeof(VS) / sizeof(VS[0])); ++k)  // id/index bug: ids 1,4..9 were
                    if (VS[k].v == vrun[vi]) { Vp = &VS[k]; break; }  // used as indices into 7 rows)
                if (!Vp) {
                    fprintf(stderr, "bench: v%d unknown id, skipped\n", vrun[vi]);
                    continue;
                }
                const VInfo V = *Vp;
                // the vreg-2 BR-64 twins are duplicates below N 7162 (BR is 64 there anyway) - skip them;
                // the r11 vreg-4 twins REQUIRE BR 64 (their only combo), so they run at every N (the N-sweep
                // diagnostics need exactly that: fixed cost vs N at the same BR)
                if (V.br64 && V.vreg == 2 && N < 56 * 128) continue;
                cur_br = V.br64 ? 64 : (N >= 56 * 128 ? 128 : 64);
                CK(cudaMemset(d_y, 0xFF, (size_t)MM * N * 4));  // NaN fill: an unwritten output shows as NaN, not 0
                if (C.sq) {
                    CK(cudaMemset(d_sq, 0, (size_t)MM * (N / 2)));
                    CK(cudaMemset(d_sqm, 0, (size_t)MM * (N / 64) * 8));
                }
                const cudaError_t de =
                    dispatch_tc(A, s, C.rpl, C.sq, V.vreg, V.vsmem, V.hoist, V.bat, V.ar, V.br64, V.wn);
                if (de == cudaErrorInvalidValue) {  // combo invalid for this shape (e.g. the rpl-2-only
                    fprintf(stderr, "bench: v%d invalid combo for %s, skipped\n", V.v, C.name);  // variants on
                    continue;  // an rpl-4 case): SKIP, not abort - v27's rpl-2 cases died at the first
                }  // invalid combo and lost their whole M sweep after it
                CK(de);
                CK(cudaStreamSynchronize(s));
                CK(cudaMemcpy(y_b.data(), d_y, y_b.size() * 4, cudaMemcpyDeviceToHost));
                if (C.sq) {
                    CK(cudaMemcpy(sq_b.data(), d_sq, sq_b.size(), cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(sqm_b.data(), d_sqm, sqm_b.size() * 8, cudaMemcpyDeviceToHost));
                }
                // the primary gate: tc vs the HOST MODEL (the same flat ascending summation order; only the
                // device FMA contraction differs -> ULP). nnan > 0 = an unwritten output = a coverage bug.
                float maxd = 0;
                int nnan = 0;
                for (int col = 0; col < M; ++col)
                    for (int r = 0; r < N; ++r) {
                        const float kv = y_b[(size_t)col * N + r];
                        if (std::isnan(kv)) nnan++;
                        maxd = fmaxf(maxd, fabsf(kv - y_h[(size_t)col * N + r]));
                    }
                const bool ok = maxd < 1e-4f && nnan == 0;
                if (!ok) diff_report("tc_vs_host", V.v, y_b, y_h);
                if (C.dp4a && vi == 0) {  // tc vs dp4a: summation order differs (shfl tree vs flat) -> ULP, not a gate
                    float dx = 0;
                    for (int col = 0; col < M; ++col)
                        for (int r = 0; r < ncmp; ++r) dx = fmaxf(dx, fabsf(y_a[(size_t)col * N + r] - y_b[(size_t)col * N + r]));
                    printf("DP4A_DIFF {\"case\":\"%s\",\"M\":%d,\"max_abs_diff\":%.7g}\n", C.name, M, dx);
                }
                const float ms_b = time_burst([&] { dispatch_tc(A, s, C.rpl, C.sq, V.vreg, V.vsmem, V.hoist, V.bat, V.ar, V.br64, V.wn); }, reps);
                const double gbb = L.bytes / (ms_b * 1e-3) / 1e9;
                const bool sq_eq = !C.sq || (memcmp(sq_a.data(), sq_b.data(), sq_a.size()) == 0 &&
                                             memcmp(sqm_a.data(), sqm_b.data(), sqm_b.size() * 8) == 0);
                printf("CHECK {\"case\":\"%s\",\"M\":%d,\"v\":%d,\"ok\":%s,\"max_abs_diff\":%.7g,\"nnan\":%d,"
                       "\"sq_eq\":%s,\"dp4a_gbps\":%.1f,\"tc_gbps\":%.1f,\"dp4a_us\":%.1f,\"tc_us\":%.1f}\n",
                       C.name, M, V.v, ok ? "true" : "false", maxd, nnan, C.dp4a ? (sq_eq ? "true" : "false") : "na",
                       gba, gbb, ms_a * 1e3, ms_b * 1e3);
                fflush(stdout);
                if (!ok) fails++;
            }
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
