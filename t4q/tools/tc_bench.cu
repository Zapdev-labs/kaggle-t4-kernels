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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <functional>
#include <string>
#include <unistd.h>
#include <vector>

#include <sys/mman.h>
#include <sys/statvfs.h>

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
    // r12 rpl A/B (the packing experiment): the v40 per-SM group-rate decomposition showed the rpl2-packed
    // class at HALF the rpl4 class's rate (qkvz 1.19M groups/s/SM vs down 2.37M, same kernel, same warps;
    // outk5 rpl4-K5120 sits in the same slow regime as out rpl4-K3072) - and the rpl is a PACKING parameter,
    // so the A/B holds the shape fixed and flips only the layout. If TC(qkvz_r4) jumps, the fix is a repack
    // choice (the loader's per-tensor rpl), not a kernel rewrite; the dp4a anchors ride along (DK(4,10) vs
    // DK(2,10)) so the repack's dp4a cost is measured in the same run. down_r2 is the reverse control (the
    // fast shape at rpl2; dp4a's rpl2 nch-17 template does not exist, so it is host-gated only).
    {"qkvz_r4", 5120, 8192, 4, 10, false, true},   // qkvz's exact shape, packed rpl 4
    {"attn_r4", 5120, 4096, 4, 10, false, true},    // attn's exact shape, packed rpl 4
    {"down_r2", 8704, 5120, 2, 17, false, false},  // down's exact shape, packed rpl 2 (TC-only)
    // r13: the engine's TRUE attn shape (selftest: qkv_a N 7168 K 5120 - the r11 attn case's N 4096 was a
    // stale approximation) - the same-shape rpl A/B at the true N, between attn_r4's N 4096 (rpl 4 wins BOTH
    // paths: dp4a 133.7 -> 203.0, TC star 80.9 -> 98.5) and qkvz's N 8192 (rpl 2 wins: dp4a 252.6 vs 244.8,
    // TC 174.3 vs 175.7). If rpl 4 wins at 7168 too, the loader's T4Q_RPL_QKV_A default flips (r14).
    {"attn_e_tp", 5120, 7168, 2, 10, false, true},  // the engine's true qkv_a shape, packed rpl 2 (default)
    {"attn_e_r4", 5120, 7168, 4, 10, false, true},  // the engine's true qkv_a shape, packed rpl 4
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

// r14 AR publish probe (--arprobe): the spec path's k_ar_norm_m publish moves M*20 KB of fp32 partials to the
// peer's mailbox through ONE block per row (M4: 4 blocks x 20 KB; the v42 trace: M4 AR 38 us vs M1 13.9, the
// ~2.5 GB/s marginal says the per-block outstanding-store limit binds, not the PCIe). The M4-round arpub A/Bs
// were M1-only (20 KB from one block; arpub 3's per-slice flags lost on a SLOW-P2P box, v26) - the M > 1 spread
// (every block publishes its own 1 KB slice, one counter + one flag) was never measured on a fast box. This probe
// times the exact publish pattern (fence + counter + flag inclusive, no wait) at M 1/4 x spread 0/1 x dst
// P2P/local so the engine A/B (T4Q_AR_SPREAD) has a mechanism anchor. Needs 2 GPUs; prints SKIP otherwise.
__global__ void __launch_bounds__(256) k_pub_probe(const float* own, float* dst, unsigned* flag, unsigned* cnt,
                                                    unsigned ep, int spread) {
    const int tid = threadIdx.x, row = blockIdx.y;
    const size_t ro = (size_t)row * 5120;
    if (spread || blockIdx.x == 0) {
        if (spread) {
            dst[ro + blockIdx.x * 256 + tid] = own[ro + blockIdx.x * 256 + tid];  // this block's 1 KB slice
        } else {
            const float4* src = (const float4*)(own + ro);
            float4* d = (float4*)(dst + ro);
#pragma unroll
            for (int k = 0; k < 5; k++) d[tid + 256 * k] = src[tid + 256 * k];
        }
        __syncthreads();
        if (tid == 0) {
            __threadfence_system();
            const unsigned old = atomicAdd(cnt, 1u);
            if (old == (spread ? gridDim.x * gridDim.y : gridDim.y) - 1u) {
                atomicExch(cnt, 0u);
                __threadfence_system();
                asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(flag), "r"(ep) : "memory");
            }
        }
    }
}

static int ar_probe(int dev, int reps) {
    int nd = 0;
    CK(cudaGetDeviceCount(&nd));
    if (nd < 2) {
        printf("R {\"arprobe\":\"SKIP\",\"why\":\"one GPU\"}\n");
        return 0;
    }
    const int peer = dev ^ 1;
    int acc = 0;
    CK(cudaDeviceCanAccessPeer(&acc, dev, peer));
    if (!acc) {
        printf("R {\"arprobe\":\"SKIP\",\"why\":\"no peer access\"}\n");
        return 0;
    }
    CK(cudaSetDevice(dev));
    cudaError_t e = cudaDeviceEnablePeerAccess(peer, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CK(e);
    cudaGetLastError();
    float* peer_rx = nullptr;  // allocated on the peer, written from dev: the engine's exact P2P publish direction
    unsigned* peer_flag = nullptr;
    CK(cudaSetDevice(peer));
    CK(cudaMalloc(&peer_rx, 8 * 5120 * 4));
    CK(cudaMalloc(&peer_flag, 4));
    float* own = nullptr;
    float* local_dst = nullptr;
    unsigned *cnt = nullptr, *local_flag = nullptr;
    CK(cudaSetDevice(dev));
    CK(cudaMalloc(&own, 8 * 5120 * 4));
    CK(cudaMalloc(&local_dst, 8 * 5120 * 4));
    CK(cudaMalloc(&cnt, 4));
    CK(cudaMalloc(&local_flag, 4));
    CK(cudaMemset(cnt, 0, 4));
    std::vector<float> h(8 * 5120);
    for (float& v : h) v = rndn_h();
    CK(cudaMemcpy(own, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    unsigned ep = 0;
    printf("R {\"arprobe\":{\"dev\":%d,\"peer\":%d,\"reps\":%d}}\n", dev, peer, reps);
    for (int M : {1, 4}) {
        for (int spread : {0, 1}) {
            for (int p2p : {1, 0}) {
                float* dst = p2p ? peer_rx : local_dst;
                unsigned* flag = p2p ? peer_flag : local_flag;
                float ms = time_burst([&] { k_pub_probe<<<dim3(20, M), 256, 0, s>>>(own, dst, flag, cnt, ++ep, spread); },
                                      reps);
                const double bytes = (double)M * 20 * 1024;  // published per launch, one direction
                printf("R {\"arprobe\":{\"M\":%d,\"spread\":%d,\"p2p\":%d,\"us\":%.2f,\"gbps\":%.1f}}\n", M, spread, p2p,
                       ms * 1000, bytes / (ms * 1e3) / 1e3);
            }
        }
    }
    CK(cudaStreamSynchronize(s));
    printf("R {\"arprobe\":\"done\"}\n");
    return 0;
}

// r15 DRAM ceiling probe (--dramprobe): the dp4a GEMV anchors sit at ~243-253 GB/s and the r4-r13 census
// calls that "the DRAM roof" - but 254 was the best-ever GEMV rate, never a measured stream ceiling. The
// T4's GDDR6 is 320 GB/s theoretical; a STREAM-class coalesced read on a good card reaches 280-300
// (88-94%). The node runs power-capped (66-67 W sustained, SMs ~900-1050 MHz), so the true ceiling could
// sit anywhere. If it is ~254-260 the GEMV is at 97-100% of the ceiling and the closure stands; if it is
// 280+ the GEMV weight streams leave ~10% (2.6 ms/step, ~+5.5% tok/s) and the streams' DRAM efficiency
// becomes the next lever. Read-only (the GEMV's weight traffic is read-only; the y writes are tiny),
// dead-code-guarded accumulation, timed with the same warmup/min-window methodology as the anchors so
// the numbers are directly comparable, on the same node in the same run as the case anchors.
__global__ void __launch_bounds__(256) k_read16(const float4* p, size_t n4, unsigned* sink) {
    unsigned acc = 0;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 v = __ldg(p + i);
        acc ^= __float_as_uint(v.x) ^ __float_as_uint(v.y) ^ __float_as_uint(v.z) ^ __float_as_uint(v.w);
    }
    if (acc == 0x12345678u) sink[0] = acc;  // defeat dead-code elimination
}

static int dram_probe(int dev, int reps) {
    CK(cudaSetDevice(dev));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    unsigned* sink = nullptr;
    float* buf = nullptr;
    const size_t MB = 200;
    CK(cudaMalloc(&buf, MB * 1024 * 1024));  // >> 6 MB L2: pure DRAM stream
    CK(cudaMalloc(&sink, 4));
    CK(cudaMemset(buf, 1, MB * 1024 * 1024));
    printf("R {\"dramprobe\":{\"dev\":%d,\"mb\":%zu,\"reps\":%d}}\n", dev, MB, reps);
    for (int nb : {80, 160, 240}) {
        float ms = time_burst([&] { k_read16<<<nb, 256, 0, s>>>((const float4*)buf, MB * 1024 * 1024 / 16, sink); },
                              reps);
        printf("R {\"dramprobe\":{\"blocks\":%d,\"width\":16,\"us\":%.1f,\"gbps\":%.1f}}\n", nb, ms * 1000,
               MB * 1.048576 / ms);
    }
    CK(cudaStreamSynchronize(s));
    printf("R {\"dramprobe\":\"done\"}\n");
    return 0;
}

// ---------------------------------------------------------------------------
// r17 cf-m0: the CYBER-FROST probes (--cfprobe). The qwen4exp decode pipeline streams Q2_K/Q4_K/Q5_1
// (the 82.85 GB freakyskittle Q2_K_S GGUF) at M=1 on 2xT4, and the tier design (hot VRAM / pinned-RAM
// warm / the 26.85 GiB PLE table + cold tail on disk) plus the honest ceiling ladder hang on rates this
// node has never measured. All GB/s are PACKED weight bytes (the formats as they sit in the file),
// min-window burst (time_burst), every kernel spot-checked against a host model on the same quantized x
// (report-only diff: the device folds integers and sums fp32 in different orders):
//   cfdisk - the filesystem the GGUF will land on: write, sequential read, random 2 MiB (an expert tail
//            row), random 4 KiB (a PLE hash page), and the mmap first-fault cost of a 16-row PLE gather
//            (the PLE's per-token traffic is 16 scattered 90 B rows in a 26.85 GiB file);
//   cfpinned - zero-copy reads of cudaHostAllocMapped memory from each GPU and both concurrently (the
//            warm tier's read ceiling), plus the cudaMemcpyAsync copy leg for comparison;
//   cfq2k - THE cf wall: 84 B/256 elems = 3.05 elems/B vs the P4's 1.78. The Q2_K layout (from the ggml
//            dequant, byte-exact): sub s (16 elems) reads its 16 quants from the 16 consecutive bytes
//            qs[32*(s>>3) + 16*(s&1)] at ONE shared shift 2*((s&7)>>1), so one (w>>sh)&0x03030303 mask
//            per u32 spreads 4 CONSECUTIVE elems - natural x order, the same dp4a shape as q4_0.
//            Probed at the expert gate/up shape [2560->640] and the big-N class [2560->12288];
//   cfq4k - the lm_head class [2560->248320] (probed at N 6208 to fit VRAM; the grid fills either way).
//            Sub s (32 elems) = nibble plane s&1 of the 32 bytes qs[32*(s>>1)], w = d*sc*(nib) - dmin*m
//            with the 6-bit scale/min pairs from scales[12];
//   cfq51 - the hc LoRA shape [10240->320]: w = d*(nib | qhbit<<4) + m, qh as a LE u32 with bit e =
//            elem e's 5th bit (8 masks x 4 bytes), the qh half dp4a'd against an 8-strided x table;
//   cfp4d - the Q4_0 expert-down [640->2560] as 10 separate launches vs ONE grid.z=10 batched gather
//            (48 layers x 10 experts = the launch-count line item). The 18-B rows put the qs u32s at
//            2 mod 4 on odd blocks - assembled with __funnelshift_r from the enclosing aligned words.
// ---------------------------------------------------------------------------

static double now_ms() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void rand_bytes(std::vector<uint8_t>& v) {
    for (size_t i = 0; i + 8 <= v.size(); i += 8) {
        uint64_t r = rnd();
        memcpy(v.data() + i, &r, 8);
    }
    for (size_t i = (v.size() / 8) * 8; i < v.size(); ++i) v[i] = (uint8_t)rnd();
}

// the x side: s8 groups (g = 16 for q2k, 32 for the rest) with an fp16 scale, plus the per-group s8 sums
// (the min folds) and the 8-strided u32 table for the q51 qh masks
struct XQ {
    int K = 0, g = 0;
    std::vector<int8_t> s8;
    std::vector<uint16_t> d;
    std::vector<int32_t> t;
    std::vector<uint8_t> x8;  // [K/32][8] u32: u32 j = the s8 of elems {32g+8c+j, c=0..3}
};
static void xq_build(XQ& q, const float* x, int K, int g) {
    q.K = K;
    q.g = g;
    q.s8.assign(K, 0);
    q.d.assign(K / g, 0);
    q.t.assign(K / g, 0);
    q.x8.assign((size_t)(K / 32) * 32, 0);
    for (int gi = 0; gi < K / g; ++gi) {
        float amx = 1e-9f;
        for (int e = 0; e < g; ++e) amx = fmaxf(amx, fabsf(x[gi * g + e]));
        const uint16_t db = f2h_host(amx / 127.f);
        q.d[gi] = db;
        const float inv = 1.f / h2f_host(db);
        int t = 0;
        for (int e = 0; e < g; ++e) {
            int v = (int)lrintf(x[gi * g + e] * inv);
            v = v < -127 ? -127 : (v > 127 ? 127 : v);
            q.s8[gi * g + e] = (int8_t)v;
            t += v;
        }
        q.t[gi] = t;
    }
    for (int gi = 0; gi < K / 32; ++gi)
        for (int j = 0; j < 8; ++j) {
            uint32_t w = 0;
            for (int c = 0; c < 4; ++c) w |= (uint32_t)(uint8_t)q.s8[gi * 32 + 8 * c + j] << (8 * c);
            memcpy(&q.x8[(size_t)gi * 32 + 4 * j], &w, 4);
        }
}

static void gen_x(int K, std::vector<float>& x) {
    x.resize(K);
    for (int i = 0; i < K; ++i) x[i] = 0.5f * rndn_h();
}

// Q2_K: scales[16] (lo nibble = the d-index, hi = the dmin-index), qs[64], d/dmin fp16. w = a*q - m.
__global__ void __launch_bounds__(256) k_q2k_gemv(const uint8_t* W, const int8_t* xs, const uint16_t* xd, const int* xt,
                                                 float* y, int K, int N) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, NB = K >> 8;
    const uint32_t M = 0x03030303u;
    float acc = 0.f;
    if (row < N) {
        const uint8_t* Wr = W + (size_t)row * NB * 84;
        for (int it = 0; it < (NB + 1) >> 1; ++it) {
            const int qb = (it << 1) + (lane >> 4);
            if (qb < NB) {
                const uint8_t* B = Wr + (size_t)qb * 84;
                const int s = lane & 15, sh = ((s & 7) >> 1) << 1;
                const uint32_t* q = (const uint32_t*)(B + 16 + 32 * (s >> 3) + 16 * (s & 1));
                const uint32_t* xi = (const uint32_t*)(xs + (qb << 8) + (s << 4));
                int S = 0;
#pragma unroll
                for (int c = 0; c < 4; ++c) S = __dp4a((int)((q[c] >> sh) & M), (int)xi[c], S);
                const uint32_t dm = *(const uint32_t*)(B + 80);
                const float d = __half2float(__ushort_as_half((unsigned short)(dm & 0xFFFF)));
                const float dmin = __half2float(__ushort_as_half((unsigned short)(dm >> 16)));
                const float a = d * (float)(B[s] & 15), m = dmin * (float)(B[s] >> 4);
                const int g = (qb << 4) + s;
                acc += __half2float(*(const __half*)(xd + g)) * (a * (float)S - m * (float)xt[g]);
            }
        }
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
    if (!lane && row < N) y[row] = acc;
}

// Q4_K: scales[12] (6-bit packed: s<4 -> sc=B[s]&63, m=B[s+4]&63; s>=4 -> the cross reassembly), qs[128],
// d/dmin fp16 at 140/142. Sub s (32 elems) = nibble plane s&1 of the 32 bytes at 32*(s>>1); natural order.
__global__ void __launch_bounds__(256) k_q4k_gemv(const uint8_t* W, const int8_t* xs, const uint16_t* xd, const int* xt,
                                                 float* y, int K, int N) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, NB = K >> 8;
    const uint32_t M = 0x0F0F0F0Fu;
    float acc = 0.f;
    if (row < N) {
        const uint8_t* Wr = W + (size_t)row * NB * 144;
        for (int it = 0; it < (NB + 3) >> 2; ++it) {
            const int qb = (it << 2) + (lane >> 3);
            if (qb < NB) {
                const uint8_t* B = Wr + (size_t)qb * 144;
                const int s = lane & 7;
                int sc, mi;
                if (s < 4) {
                    sc = B[s] & 63;
                    mi = B[s + 4] & 63;
                } else {
                    sc = (B[s + 4] & 0xF) | ((B[s - 4] >> 6) << 4);
                    mi = (B[s + 4] >> 4) | ((B[s] >> 6) << 4);
                }
                const uint32_t* q = (const uint32_t*)(B + 12 + 32 * (s >> 1));
                const uint32_t* xi = (const uint32_t*)(xs + (qb << 8) + (s << 5));
                int S = 0;
                if (!(s & 1)) {
#pragma unroll
                    for (int c = 0; c < 8; ++c) S = __dp4a((int)(q[c] & M), (int)xi[c], S);
                } else {
#pragma unroll
                    for (int c = 0; c < 8; ++c) S = __dp4a((int)((q[c] >> 4) & M), (int)xi[c], S);
                }
                const uint32_t dm = *(const uint32_t*)(B + 140);
                const float d = __half2float(__ushort_as_half((unsigned short)(dm & 0xFFFF)));
                const float dmin = __half2float(__ushort_as_half((unsigned short)(dm >> 16)));
                const float a = d * (float)sc, m = dmin * (float)mi;
                const int g = (qb << 3) + s;
                acc += __half2float(*(const __half*)(xd + g)) * (a * (float)S - m * (float)xt[g]);
            }
        }
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
    if (!lane && row < N) y[row] = acc;
}

// Q5_1: 24 B/32 (d, m fp16; qh[4]; qs[16]). w = d*(nib | qhbit<<4) + m; qh as a LE u32: bit e = elem e.
__global__ void __launch_bounds__(256) k_q51_gemv(const uint8_t* W, const int8_t* xs, const uint8_t* x8,
                                                 const uint16_t* xd, const int* xt, float* y, int K, int N) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, NG = K >> 5;
    const uint32_t M = 0x0F0F0F0Fu, MB = 0x01010101u;
    float acc = 0.f;
    if (row < N) {
        const uint8_t* Wr = W + (size_t)row * NG * 24;
        for (int it = 0; it < (NG + 31) >> 5; ++it) {
            const int g = (it << 5) + lane;
            if (g < NG) {
                const uint8_t* B = Wr + (size_t)g * 24;
                const uint32_t* q = (const uint32_t*)(B + 8);
                const uint32_t* xi = (const uint32_t*)(xs + (g << 5));
                const uint32_t* xj = (const uint32_t*)(x8 + (size_t)g * 32);
                const uint32_t qhw = *(const uint32_t*)(B + 4);
                int S = 0, Sq = 0;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    S = __dp4a((int)(q[c] & M), (int)xi[c], S);
                    S = __dp4a((int)((q[c] >> 4) & M), (int)xi[4 + c], S);
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) Sq = __dp4a((int)((qhw >> j) & MB), (int)xj[j], Sq);
                const float d = __half2float(*(const __half*)B);
                const float m = __half2float(*(const __half*)(B + 2));
                acc += __half2float(*(const __half*)(xd + g)) *
                       (d * ((float)S + 16.f * (float)Sq) + m * (float)xt[g]);
            }
        }
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
    if (!lane && row < N) y[row] = acc;
}

// Q4_0 (the engine's P4 format's parent): 18 B/32, d fp16 + qs[16]; w = d*(nib-8). The 18-B rows put the
// qs u32s at 2 mod 4 on odd blocks: assemble via __funnelshift_r from the two enclosing aligned words.
static __device__ __forceinline__ uint32_t ld_u32u(const uint8_t* p) {
    const uintptr_t a = (uintptr_t)p;
    const uint32_t* w = (const uint32_t*)(a & ~(uintptr_t)3);
    return __funnelshift_r(w[0], w[1], (uint32_t)((a & 3) * 8));
}
__global__ void __launch_bounds__(256) k_p4d_gemv(const uint8_t* W, const int8_t* xs, const uint16_t* xd, const int* xt,
                                                 float* y, int K, int N, int rows_e) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, NG = K >> 5;
    const uint32_t M = 0x0F0F0F0Fu;
    float acc = 0.f;
    if (row < N) {
        const uint8_t* Wr = W + ((size_t)blockIdx.z * rows_e + row) * NG * 18;
        for (int it = 0; it < (NG + 31) >> 5; ++it) {
            const int g = (it << 5) + lane;
            if (g < NG) {
                const uint8_t* B = Wr + (size_t)g * 18;
                const float d = __half2float(*(const __half*)B);
                int S = 0;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint32_t w = ld_u32u(B + 2 + 4 * c);
                    S = __dp4a((int)(w & M), *(const int*)(xs + (g << 5) + 4 * c), S);
                    S = __dp4a((int)((w >> 4) & M), *(const int*)(xs + (g << 5) + 16 + 4 * c), S);
                }
                acc += d * __half2float(*(const __half*)(xd + g)) * ((float)S - 8.f * (float)xt[g]);
            }
        }
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
    if (!lane && row < N) y[(size_t)blockIdx.z * N + row] = acc;
}

static void gen_q2k(int N, int K, std::vector<uint8_t>& out) {
    const size_t rb = (size_t)(K >> 8) * 84;
    out.resize(rb * N + 8);
    rand_bytes(out);
    for (size_t row = 0; row < (size_t)N; ++row)
        for (int qb = 0; qb < (K >> 8); ++qb) {
            uint8_t* B = out.data() + row * rb + (size_t)qb * 84;
            const uint16_t d = f2h_host(0.002f + 0.02f * rndu()), dm = f2h_host(0.002f + 0.02f * rndu());
            memcpy(B + 80, &d, 2);
            memcpy(B + 82, &dm, 2);
        }
}
static void gen_q4k(int N, int K, std::vector<uint8_t>& out) {
    const size_t rb = (size_t)(K >> 8) * 144;
    out.resize(rb * N + 8);
    rand_bytes(out);
    for (size_t row = 0; row < (size_t)N; ++row)
        for (int qb = 0; qb < (K >> 8); ++qb) {
            uint8_t* B = out.data() + row * rb + (size_t)qb * 144;
            const uint16_t d = f2h_host(0.002f + 0.02f * rndu()), dm = f2h_host(0.002f + 0.02f * rndu());
            memcpy(B + 140, &d, 2);
            memcpy(B + 142, &dm, 2);
        }
}
static void gen_q51(int N, int K, std::vector<uint8_t>& out) {
    const size_t rb = (size_t)(K >> 5) * 24;
    out.resize(rb * N + 8);
    rand_bytes(out);
    for (size_t row = 0; row < (size_t)N; ++row)
        for (int g = 0; g < (K >> 5); ++g) {
            uint8_t* B = out.data() + row * rb + (size_t)g * 24;
            const uint16_t d = f2h_host(0.002f + 0.02f * rndu()), m = f2h_host(0.002f + 0.02f * rndu());
            memcpy(B, &d, 2);
            memcpy(B + 2, &m, 2);
        }
}
static void gen_p4d(int N, int K, std::vector<uint8_t>& out) {
    const size_t rb = (size_t)(K >> 5) * 18;
    out.resize(rb * N + 8);
    rand_bytes(out);
    for (size_t row = 0; row < (size_t)N; ++row)
        for (int g = 0; g < (K >> 5); ++g) {
            uint8_t* B = out.data() + row * rb + (size_t)g * 18;
            const uint16_t d = f2h_host(0.002f + 0.02f * rndu());
            memcpy(B, &d, 2);
        }
}

static void q2k_host(int K, int N, const std::vector<uint8_t>& W, const XQ& xq, float* y) {
    const int NB = K >> 8;
    for (int r = 0; r < N; ++r) {
        const uint8_t* B0 = W.data() + (size_t)r * NB * 84;
        double acc = 0;
        for (int qb = 0; qb < NB; ++qb) {
            const uint8_t* B = B0 + qb * 84;
            const float d = h2f_host(*(const uint16_t*)(B + 80)), dmin = h2f_host(*(const uint16_t*)(B + 82));
            for (int s = 0; s < 16; ++s) {
                const float a = d * (B[s] & 15), m = dmin * (B[s] >> 4);
                const int sh = ((s & 7) >> 1) << 1;
                const uint8_t* q = B + 16 + 32 * (s >> 3) + 16 * (s & 1);
                for (int t = 0; t < 16; ++t) {
                    const int k = (qb << 8) + (s << 4) + t;
                    const float w = a * (float)((q[t] >> sh) & 3) - m;
                    acc += (double)w * h2f_host(xq.d[k / xq.g]) * xq.s8[k];
                }
            }
        }
        y[r] = (float)acc;
    }
}
static void q4k_host(int K, int N, const std::vector<uint8_t>& W, const XQ& xq, float* y) {
    const int NB = K >> 8;
    for (int r = 0; r < N; ++r) {
        const uint8_t* B0 = W.data() + (size_t)r * NB * 144;
        double acc = 0;
        for (int qb = 0; qb < NB; ++qb) {
            const uint8_t* B = B0 + qb * 144;
            const float d = h2f_host(*(const uint16_t*)(B + 140)), dmin = h2f_host(*(const uint16_t*)(B + 142));
            for (int s = 0; s < 8; ++s) {
                int sc, mi;
                if (s < 4) {
                    sc = B[s] & 63;
                    mi = B[s + 4] & 63;
                } else {
                    sc = (B[s + 4] & 0xF) | ((B[s - 4] >> 6) << 4);
                    mi = (B[s + 4] >> 4) | ((B[s] >> 6) << 4);
                }
                const float a = d * sc, m = dmin * mi;
                const uint8_t* q = B + 12 + 32 * (s >> 1);
                for (int t = 0; t < 32; ++t) {
                    const int nib = (s & 1) ? (q[t] >> 4) : (q[t] & 15);
                    const int k = (qb << 8) + (s << 5) + t;
                    acc += (double)(a * nib - m) * h2f_host(xq.d[k / xq.g]) * xq.s8[k];
                }
            }
        }
        y[r] = (float)acc;
    }
}
static void q51_host(int K, int N, const std::vector<uint8_t>& W, const XQ& xq, float* y) {
    const int NG = K >> 5;
    for (int r = 0; r < N; ++r) {
        const uint8_t* B0 = W.data() + (size_t)r * NG * 24;
        double acc = 0;
        for (int g = 0; g < NG; ++g) {
            const uint8_t* B = B0 + g * 24;
            const float d = h2f_host(*(const uint16_t*)B), m = h2f_host(*(const uint16_t*)(B + 2));
            uint32_t qhw;
            memcpy(&qhw, B + 4, 4);
            for (int t = 0; t < 32; ++t) {
                const int nib = t < 16 ? (B[8 + t] & 15) : (B[8 + t - 16] >> 4);
                const float w = d * (float)(nib | (((qhw >> t) & 1) << 4)) + m;
                acc += (double)w * h2f_host(xq.d[g]) * xq.s8[g * 32 + t];
            }
        }
        y[r] = (float)acc;
    }
}
static void p4d_host(int K, int N, int E, const std::vector<uint8_t>& W, const XQ& xq, float* y) {
    const int NG = K >> 5;
    for (int e = 0; e < E; ++e)
        for (int r = 0; r < N; ++r) {
            const uint8_t* B0 = W.data() + ((size_t)e * N + r) * NG * 18;
            double acc = 0;
            for (int g = 0; g < NG; ++g) {
                const uint8_t* B = B0 + g * 18;
                const float d = h2f_host(*(const uint16_t*)B);
                for (int t = 0; t < 32; ++t) {
                    const int nib = t < 16 ? (B[2 + t] & 15) : (B[2 + t - 16] >> 4);
                    acc += (double)(d * (nib - 8)) * h2f_host(xq.d[g]) * xq.s8[g * 32 + t];
                }
            }
            y[(size_t)e * N + r] = (float)acc;
        }
}

static int cf_disk_probe(const char* dir, int gb) {
    std::string path = std::string(dir) + "/cfprobe.bin";
    struct statvfs vfs {};
    if (statvfs(dir, &vfs) != 0) {
        printf("R {\"cfdisk\":{\"err\":\"statvfs\"}}\n");
        return 1;
    }
    const double free_gb = (double)vfs.f_bavail * (double)vfs.f_frsize / 1073741824.0;
    if ((double)gb > free_gb * 0.6) gb = (int)(free_gb * 0.6);
    if (gb < 2) {
        printf("R {\"cfdisk\":{\"free_gb\":%.1f,\"skip\":\"no-space\"}}\n", free_gb);
        return 1;
    }
    printf("R {\"cfdisk\":{\"dir\":\"%s\",\"free_gb\":%.1f,\"file_gb\":%d}}\n", dir, free_gb, gb);
    const size_t CH = 128ull << 20, tot = (size_t)gb << 30;
    void* wb = nullptr;
    if (posix_memalign(&wb, 512, CH) != 0) {
        printf("R {\"cfdisk\":{\"err\":\"memalign\"}}\n");
        return 1;
    }
    for (size_t i = 0; i < CH; i += 8) {
        uint64_t r = rnd();
        memcpy((char*)wb + i, &r, 8);
    }
    int od = 0, fd = -1;
#ifdef O_DIRECT
    fd = open(path.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_DIRECT, 0644);
    if (fd >= 0) od = 1;
#endif
    if (fd < 0) fd = open(path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd < 0) {
        printf("R {\"cfdisk\":{\"err\":\"open-w\"}}\n");
        return 1;
    }
    double t0 = now_ms();
    for (size_t off = 0; off < tot; off += CH)
        if ((size_t)write(fd, wb, CH) != CH) {
            printf("R {\"cfdisk\":{\"err\":\"write\"}}\n");
            return 1;
        }
    fdatasync(fd);
    const double wr_ms = now_ms() - t0;
    printf("R {\"cfdisk\":{\"write\":{\"odirect\":%d,\"ms\":%.0f,\"gbps\":%.2f}}}\n", od, wr_ms, gb / (wr_ms / 1000.0));
    close(fd);
    int rfd = -1;
#ifdef O_DIRECT
    rfd = open(path.c_str(), O_RDONLY | O_DIRECT);
#endif
    if (rfd < 0) rfd = open(path.c_str(), O_RDONLY);
    if (rfd < 0) {
        printf("R {\"cfdisk\":{\"err\":\"open-r\"}}\n");
        return 1;
    }
    if (!od) posix_fadvise(rfd, 0, 0, POSIX_FADV_DONTNEED);  // the write went through the page cache: drop it
    // sequential 1 MiB (the requant read-back / the cold-tail stream)
    t0 = now_ms();
    double rd = 0;
    for (size_t off = 0; off + (1 << 20) <= tot; off += (1 << 20))
        if ((size_t)pread(rfd, wb, 1 << 20, (off_t)off) == (1 << 20)) rd += 1;
    double ms = now_ms() - t0;
    printf("R {\"cfdisk\":{\"seq\":{\"odirect\":%d,\"gb\":%.1f,\"ms\":%.0f,\"gbps\":%.2f}}}\n", od, rd / 1024.0, ms,
           rd / 1024.0 / (ms / 1000.0));
    // random 2 MiB (an expert tail row read at tier-3 granularity)
    const int n2 = 1200;
    t0 = now_ms();
    for (int i = 0; i < n2; ++i) {
        const size_t off = (size_t)(rnd() % (tot >> 21)) << 21;
        if ((size_t)pread(rfd, wb, 2 << 20, (off_t)off) != (2 << 20)) break;
    }
    ms = now_ms() - t0;
    printf("R {\"cfdisk\":{\"rand2m\":{\"n\":%d,\"ms\":%.0f,\"gbps\":%.2f}}}\n", n2, ms,
           (double)n2 * 2 / 1024.0 / (ms / 1000.0));
    // random 4 KiB (a PLE hash page fault through the page cache)
    const int n4 = 2400;
    t0 = now_ms();
    for (int i = 0; i < n4; ++i) {
        const size_t off = (size_t)(rnd() % (tot >> 12)) << 12;
        if ((size_t)pread(rfd, wb, 4096, (off_t)off) != 4096) break;
    }
    ms = now_ms() - t0;
    printf("R {\"cfdisk\":{\"rand4k\":{\"n\":%d,\"ms\":%.0f,\"iops\":%.0f}}}\n", n4, ms, n4 / (ms / 1000.0));
    // mmap first-fault: 16 scattered 90 B rows (the PLE gather shape: 16 rows x 90 B = 1440 B/token)
    if (!od) posix_fadvise(rfd, 0, 0, POSIX_FADV_DONTNEED);
    void* mm = mmap(nullptr, tot, PROT_READ, MAP_SHARED, rfd, 0);
    if (mm == MAP_FAILED) {
        printf("R {\"cfdisk\":{\"mmap\":\"fail\"}}\n");
    } else {
        madvise(mm, tot, MADV_RANDOM);
        const int rounds = 800;
        volatile unsigned sink = 0;
        t0 = now_ms();
        for (int i = 0; i < rounds; ++i) {
            unsigned s = 0;
            for (int r = 0; r < 16; ++r) {
                const size_t off = (size_t)(rnd() % (tot - 90));
                s += ((volatile const unsigned char*)mm)[off] + ((volatile const unsigned char*)mm)[off + 89];
            }
            sink += s;
        }
        ms = now_ms() - t0;
        printf("R {\"cfdisk\":{\"mmap16\":{\"rounds\":%d,\"ms\":%.0f,\"us_per_round\":%.1f,\"us_per_row\":%.2f}}}\n",
               rounds, ms, ms * 1000.0 / rounds, ms * 1000.0 / rounds / 16.0);
        munmap(mm, tot);
    }
    close(rfd);
    unlink(path.c_str());
    free(wb);
    printf("R {\"cfdisk\":\"done\"}\n");
    return 0;
}

static int cf_pinned_probe(int reps) {
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (ndev > 2) ndev = 2;
    const size_t MB = 2048, BY = MB << 20;
    void* hp = nullptr;
    CK(cudaHostAlloc(&hp, BY, cudaHostAllocPortable | cudaHostAllocMapped));
    memset(hp, 0xCD, BY);
    printf("R {\"cfpinned\":{\"mb\":%zu,\"ndev\":%d}}\n", MB, ndev);
    unsigned* sink = nullptr;
    float* dbuf[2] = {nullptr, nullptr};
    for (int d = 0; d < ndev; ++d) {
        CK(cudaSetDevice(d));
        CK(cudaMalloc(&dbuf[d], 1 << 30));
        CK(cudaMemset(dbuf[d], 1, 1 << 30));
    }
    CK(cudaMalloc(&sink, 4));
    for (int d = 0; d < ndev; ++d) {
        CK(cudaSetDevice(d));
        cudaStream_t s;
        CK(cudaStreamCreate(&s));
        void* dp = nullptr;
        CK(cudaHostGetDevicePointer(&dp, hp, 0));
        for (int nb : {80, 240}) {
            const float ms = time_burst([&] { k_read16<<<nb, 256, 0, s>>>((const float4*)dp, BY / 16, sink); }, reps);
            printf("R {\"cfpinned\":{\"dev\":%d,\"mode\":\"zero\",\"blocks\":%d,\"us\":%.1f,\"gbps\":%.2f}}\n", d, nb,
                   ms * 1000, MB / 1024.0 / (ms / 1000.0));
        }
        const float ms = time_burst(
            [&] {
                CK(cudaMemcpyAsync(dbuf[d], dp, 1 << 30, cudaMemcpyHostToDevice, s));
                CK(cudaStreamSynchronize(s));
            },
            12);
        printf("R {\"cfpinned\":{\"dev\":%d,\"mode\":\"memcpy\",\"gbps\":%.2f}}\n", d, 1.0 / (ms / 1000.0));
        CK(cudaStreamSynchronize(s));
    }
    if (ndev == 2) {
        // both concurrently from one host thread: the launches interleave through cudaSetDevice (a ~2 us
        // submission cost, negligible against the ~100 ms+ read bursts) and the kernels overlap on device
        cudaStream_t s[2] = {nullptr, nullptr};
        void* dp[2] = {nullptr, nullptr};
        for (int d = 0; d < 2; ++d) {
            CK(cudaSetDevice(d));
            CK(cudaStreamCreate(&s[d]));
            CK(cudaHostGetDevicePointer(&dp[d], hp, 0));
        }
        for (int r = 0; r < 3; ++r)
            for (int d = 0; d < 2; ++d) {
                CK(cudaSetDevice(d));
                k_read16<<<240, 256, 0, s[d]>>>((const float4*)dp[d], BY / 16, sink);
            }
        for (int d = 0; d < 2; ++d) {
            CK(cudaSetDevice(d));
            CK(cudaStreamSynchronize(s[d]));
        }
        double best = 1e30;
        for (int w = 0; w < 3; ++w) {
            const double t0 = now_ms();
            for (int r = 0; r < reps; ++r)
                for (int d = 0; d < 2; ++d) {
                    CK(cudaSetDevice(d));
                    k_read16<<<240, 256, 0, s[d]>>>((const float4*)dp[d], BY / 16, sink);
                }
            for (int d = 0; d < 2; ++d) {
                CK(cudaSetDevice(d));
                CK(cudaStreamSynchronize(s[d]));
            }
            const double ms = now_ms() - t0;
            if (ms / reps < best) best = ms / reps;
        }
        printf("R {\"cfpinned\":{\"mode\":\"zero-both\",\"blocks\":240,\"gbps_per_gpu\":%.2f,\"gbps_total\":%.2f}}\n",
               MB / 1024.0 / (best / 1000.0), 2.0 * MB / 1024.0 / (best / 1000.0));
    }
    CK(cudaSetDevice(0));
    cudaFreeHost(hp);
    for (int d = 0; d < ndev; ++d) {
        CK(cudaSetDevice(d));
        cudaFree(dbuf[d]);
    }
    printf("R {\"cfpinned\":\"done\"}\n");
    return 0;
}

struct CFB {  // the device-side copies shared by the format runners
    uint8_t* W = nullptr;
    int8_t* xs = nullptr;
    uint8_t* x8 = nullptr;
    uint16_t* xd = nullptr;
    int* xt = nullptr;
    float* y = nullptr;
    size_t wb = 0;
};
static void cf_dev_put(CFB& b, const std::vector<uint8_t>& W, const XQ& xq, size_t yn) {
    b.wb = W.size();
    CK(cudaMalloc(&b.W, W.size()));
    CK(cudaMalloc(&b.xs, xq.K));
    CK(cudaMalloc(&b.xd, xq.d.size() * 2));
    CK(cudaMalloc(&b.xt, xq.t.size() * 4));
    if (!xq.x8.empty()) CK(cudaMalloc(&b.x8, xq.x8.size()));
    CK(cudaMalloc(&b.y, yn * 4));
    CK(cudaMemcpy(b.W, W.data(), W.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.xs, xq.s8.data(), xq.K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.xd, xq.d.data(), xq.d.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.xt, xq.t.data(), xq.t.size() * 4, cudaMemcpyHostToDevice));
    if (!xq.x8.empty()) CK(cudaMemcpy(b.x8, xq.x8.data(), xq.x8.size(), cudaMemcpyHostToDevice));
}
static void cf_dev_free(CFB& b) {
    cudaFree(b.W);
    cudaFree(b.xs);
    cudaFree(b.xd);
    cudaFree(b.xt);
    cudaFree(b.y);
    if (b.x8) cudaFree(b.x8);
}
static float cf_check(const float* y, int n, const float* yh) {
    float mx = 0;
    for (int i = 0; i < n; ++i) mx = fmaxf(mx, fabsf(y[i] - yh[i]));
    return mx;
}

static int cf_formats_probe(int dev, int reps) {
    CK(cudaSetDevice(dev));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    std::vector<float> x;
    std::vector<uint8_t> W;
    std::vector<float> yh;
    CFB b;
    // --- q2k at the expert gate/up shape, then the big-N class ---
    for (int sh = 0; sh < 2; ++sh) {
        const int K = 2560, N = sh ? 12288 : 640;
        gen_q2k(N, K, W);
        gen_x(K, x);
        XQ xq;
        xq_build(xq, x.data(), K, 16);
        cf_dev_put(b, W, xq, N);
        const int NCHK = N < 64 ? N : 64;
        yh.assign(NCHK, 0.f);
        q2k_host(K, NCHK, W, xq, yh.data());
        k_q2k_gemv<<<(NCHK + 7) / 8, 256, 0, s>>>(b.W, b.xs, b.xd, b.xt, b.y, K, NCHK);
        std::vector<float> y(NCHK);
        CK(cudaMemcpy(y.data(), b.y, NCHK * 4, cudaMemcpyDeviceToHost));
        const float ms = time_burst([&] { k_q2k_gemv<<<(N + 7) / 8, 256, 0, s>>>(b.W, b.xs, b.xd, b.xt, b.y, K, N); }, reps);
        printf("R {\"cfq2k\":{\"tag\":\"%s\",\"k\":%d,\"n\":%d,\"us\":%.1f,\"gbps\":%.1f,\"chk\":\"%.2e\"}}\n",
               sh ? "attn_q" : "exp_up", K, N, ms * 1000,
               (double)N * (K >> 8) * 84 / (ms / 1000.0) / 1e9, cf_check(y.data(), NCHK, yh.data()));
        cf_dev_free(b);
    }
    // --- q4k at the lm_head class ---
    {
        const int K = 2560, N = 6208;  // the lm_head's real N is 248320; this fills the grid and fits VRAM
        gen_q4k(N, K, W);
        gen_x(K, x);
        XQ xq;
        xq_build(xq, x.data(), K, 32);
        cf_dev_put(b, W, xq, N);
        const int NCHK = 64;
        yh.assign(NCHK, 0.f);
        q4k_host(K, NCHK, W, xq, yh.data());
        k_q4k_gemv<<<(NCHK + 7) / 8, 256, 0, s>>>(b.W, b.xs, b.xd, b.xt, b.y, K, NCHK);
        std::vector<float> y(NCHK);
        CK(cudaMemcpy(y.data(), b.y, NCHK * 4, cudaMemcpyDeviceToHost));
        const float ms = time_burst([&] { k_q4k_gemv<<<(N + 7) / 8, 256, 0, s>>>(b.W, b.xs, b.xd, b.xt, b.y, K, N); }, reps);
        printf("R {\"cfq4k\":{\"tag\":\"lm_head\",\"k\":%d,\"n\":%d,\"us\":%.1f,\"gbps\":%.1f,\"chk\":\"%.2e\"}}\n", K, N,
               ms * 1000, (double)N * (K >> 8) * 144 / (ms / 1000.0) / 1e9, cf_check(y.data(), NCHK, yh.data()));
        cf_dev_free(b);
    }
    // --- q51 at the hc LoRA shape ---
    {
        const int K = 10240, N = 320;
        gen_q51(N, K, W);
        gen_x(K, x);
        XQ xq;
        xq_build(xq, x.data(), K, 32);
        cf_dev_put(b, W, xq, N);
        const int NCHK = 64;
        yh.assign(NCHK, 0.f);
        q51_host(K, NCHK, W, xq, yh.data());
        k_q51_gemv<<<(NCHK + 7) / 8, 256, 0, s>>>(b.W, b.xs, b.x8, b.xd, b.xt, b.y, K, NCHK);
        std::vector<float> y(NCHK);
        CK(cudaMemcpy(y.data(), b.y, NCHK * 4, cudaMemcpyDeviceToHost));
        const float ms = time_burst([&] { k_q51_gemv<<<(N + 7) / 8, 256, 0, s>>>(b.W, b.xs, b.x8, b.xd, b.xt, b.y, K, N); }, reps);
        printf("R {\"cfq51\":{\"tag\":\"hc_lora\",\"k\":%d,\"n\":%d,\"us\":%.1f,\"gbps\":%.1f,\"chk\":\"%.2e\"}}\n", K, N,
               ms * 1000, (double)N * (K >> 5) * 24 / (ms / 1000.0) / 1e9, cf_check(y.data(), NCHK, yh.data()));
        cf_dev_free(b);
    }
    // --- p4d at the expert-down shape: 10 separate launches vs 1 batched gather ---
    {
        const int K = 640, N = 2560, E = 10;
        const size_t rb = (size_t)(K >> 5) * 18;
        gen_p4d(N * E, K, W);  // E contiguous [N x rb] slabs
        gen_x(K, x);
        XQ xq;
        xq_build(xq, x.data(), K, 32);
        cf_dev_put(b, W, xq, (size_t)N * E);
        const int NCHK = 64;
        yh.assign((size_t)NCHK * E, 0.f);
        p4d_host(K, NCHK, E, W, xq, yh.data());
        for (int e = 0; e < E; ++e)
            k_p4d_gemv<<<(NCHK + 7) / 8, 256, 0, s>>>(b.W + (size_t)e * NCHK * rb, b.xs, b.xd, b.xt,
                                                     b.y + (size_t)e * NCHK, K, NCHK, NCHK);
        std::vector<float> y((size_t)NCHK * E);
        CK(cudaMemcpy(y.data(), b.y, NCHK * E * 4, cudaMemcpyDeviceToHost));
        // 10 separate launches (the rows-per-expert pointer offset stands in for per-expert tensors)
        const float msA = time_burst(
            [&] {
                for (int e = 0; e < E; ++e)
                    k_p4d_gemv<<<(N + 7) / 8, 256, 0, s>>>(b.W + (size_t)e * N * rb, b.xs, b.xd, b.xt, b.y, K, N, N);
            },
            reps);
        const float usA = msA * 1000;
        const float msB = time_burst([&] { k_p4d_gemv<<<dim3((N + 7) / 8, 1, E), 256, 0, s>>>(b.W, b.xs, b.xd, b.xt, b.y, K, N, N); }, reps);
        printf("R {\"cfp4d\":{\"tag\":\"exp_down\",\"k\":%d,\"n\":%d,\"e\":%d,\"sep_us\":%.1f,\"sep_gbps\":%.1f,"
               "\"bat_us\":%.1f,\"bat_gbps\":%.1f,\"chk\":\"%.2e\"}}\n",
               K, N, E, usA, (double)E * N * rb / (msA / 1000.0) / 1e9, msB * 1000,
               (double)E * N * rb / (msB / 1000.0) / 1e9, cf_check(y.data(), NCHK * E, yh.data()));
        cf_dev_free(b);
    }
    printf("R {\"cfformats\":\"done\"}\n");
    return 0;
}

static int cf_probe(const char* dir, int gb, int dev, int reps) {
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    printf("R {\"cfprobe\":{\"gpus\":%d,\"reps\":%d}}\n", ndev, reps);
    cf_disk_probe(dir, gb);
    cf_pinned_probe(reps);
    for (int d = 0; d < ndev && d < 2; ++d) dram_probe(d, reps);
    cf_formats_probe(dev, reps);
    printf("R {\"cfprobe\":\"done\"}\n");
    return 0;
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
    int dev = 0, reps = 50, only = -1, arprobe = 0, dramprobe = 0, cfprobe = 0, cfgb = 32;
    const char* cfdir = ".";
    int vrun[8], nvr = 0, mflt[8], nmf = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--dev") dev = atoi(argv[++i]);
        else if (a == "--reps") reps = atoi(argv[++i]);
        else if (a == "--case") only = atoi(argv[++i]);  // run one case (the stage loops cases as separate processes)
        else if (a == "--arprobe") arprobe = 1;  // r14: the P2P publish writer-count probe (exits before the case loop)
        else if (a == "--dramprobe") dramprobe = 1;  // r15: the DRAM read-stream ceiling probe (exits before the case loop)
        else if (a == "--cfprobe") cfprobe = 1;  // r17 cf-m0: the CYBER-FROST platform+format probes (exits before the case loop)
        else if (a == "--cfdir") cfdir = argv[++i];  // the dir the GGUF will land on (default: the cwd)
        else if (a == "--cfgb") cfgb = atoi(argv[++i]);  // the disk-probe file size, GiB (default 32; capped at 60% free)
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
    if (arprobe) return ar_probe(dev, reps);
    if (dramprobe) return dram_probe(dev, reps);
    if (cfprobe) return cf_probe(cfdir, cfgb, dev, reps);

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
