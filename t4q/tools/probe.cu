// probe.cu -- M0 box probe for t4q (2x T4): mailbox all-reduce paths, NCCL, streaming read, clocks.
// Build: nvcc -O3 -std=c++17 -arch=sm_75 probe.cu -o probe -ldl -lpthread
// Run:   ./probe            (P2P + host-mapped mailboxes, memcpy peer, stream read, clocks under load)
//        ./probe nccl LIB   (NCCL 20 KB all-reduce p50/p99 via dlopen(LIB); env NCCL_* honoured)
// Output: "P {json}" lines + "PROBE_DONE".
#include "nvml_lite.h"

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) {                                                                     \
            printf("FATAL cuda %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); \
            fflush(stdout);                                                                          \
            exit(2);                                                                                 \
        }                                                                                            \
    } while (0)

static constexpr int MAXB = 64;
static constexpr unsigned long long WATCHDOG_NS = 2000000000ull;  // 2 s

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ __forceinline__ unsigned ld_vol_u32(const unsigned* p) {
    unsigned v; asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v;
}
__device__ __forceinline__ float ld_vol_f32(const float* p) {
    float v; asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(v) : "l"(p)); return v;
}
__device__ __forceinline__ void st_vol_u32(unsigned* p, unsigned v) {
    asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__host__ __device__ __forceinline__ float payload_val(unsigned ep, int i, int me) {
    return (float)((ep * 13u + (unsigned)i * 3u + (unsigned)me * 7u) & 0xFFFFFu);
}

// Mailbox exchange. Each GPU runs nb blocks. Per iteration (epoch ep, slot ep&1):
//   produce: block b writes its slice of n floats into peer_rx[slot], fence.sys, then peer_flag[slot][b] = ep
//   consume: wait until my_flag[slot][0..nb) == ep, then read (all n | own slice) of my_rx[slot] and verify.
// send_first = 0 makes this GPU wait first (ping-pong responder).
__global__ void xchg_kernel(float* peer_rx, unsigned* peer_flag, const float* my_rx, const unsigned* my_flag, int n,
                            int iters, int me, int send_first, int read_all, unsigned long long* tstamp, int* err,
                            float* sink) {
    const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x;
    const int per = (n + nb - 1) / nb, lo = min(n, b * per), hi = min(n, lo + per);
    float acc = 0.f;
    __shared__ int s_abort;
    if (tid == 0) s_abort = 0;
    __syncthreads();
    for (int it = 0; it < iters; ++it) {
        const unsigned ep = (unsigned)it + 1u;
        const int slot = ep & 1;
        for (int phase = 0; phase < 2; ++phase) {
            const bool produce = (phase == 0) == (send_first != 0);
            if (produce) {
                for (int i = lo + tid; i < hi; i += blockDim.x) peer_rx[slot * n + i] = payload_val(ep, i, me);
                __syncthreads();
                if (tid == 0) {
                    __threadfence_system();
                    st_vol_u32(peer_flag + slot * MAXB + b, ep);
                }
            } else {
                if (tid < nb) {
                    long long spins = 0; const unsigned long long t0 = gtimer();
                    while (ld_vol_u32(my_flag + slot * MAXB + tid) != ep) {
                        if ((++spins & 255) == 0 && gtimer() - t0 > WATCHDOG_NS) { atomicExch(err, 1000000); s_abort = 1; break; }
                    }
                }
                __syncthreads();
                if (s_abort) return;
                __threadfence();
                const int rlo = read_all ? 0 : lo, rhi = read_all ? n : hi;
                int bad = 0;
                for (int i = rlo + tid; i < rhi; i += blockDim.x) {
                    float v = ld_vol_f32(my_rx + slot * n + i);
                    bad += v != payload_val(ep, i, 1 - me);
                    acc += v;
                }
                if (bad) atomicAdd(err, bad);
                __syncthreads();
            }
        }
        if (b == 0 && tid == 0 && tstamp) tstamp[it] = gtimer();
    }
    if (acc == 1.2345f) sink[0] = acc;
}

__global__ void read_kernel(const float4* __restrict__ p, size_t n, float* out) {
    float4 a = make_float4(0, 0, 0, 0);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float4 v = __ldg(p + i);
        a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
    }
    if (a.x + a.y + a.z + a.w == 1.2345f) out[0] = a.x;
}

static double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t i = (size_t)std::min<double>((double)v.size() - 1, q * (v.size() - 1) + 0.5);
    return v[i];
}

struct Mailbox {  // per destination GPU g: rx[2][n] floats + flags[2][MAXB]
    float* rx[2]; unsigned* flag[2]; bool host = false;
};

static void alloc_mailbox(Mailbox& M, bool host, int n) {
    M.host = host;
    for (int g = 0; g < 2; ++g) {
        if (host) {
            void* p;
            CK(cudaHostAlloc(&p, 2 * n * 4 + 2 * MAXB * 4, cudaHostAllocMapped | cudaHostAllocPortable));
            memset(p, 0, 2 * n * 4 + 2 * MAXB * 4);
            M.rx[g] = (float*)p; M.flag[g] = (unsigned*)((char*)p + 2 * n * 4);
        } else {
            CK(cudaSetDevice(g));
            void* p; CK(cudaMalloc(&p, 2 * n * 4 + 2 * MAXB * 4));
            CK(cudaMemset(p, 0, 2 * n * 4 + 2 * MAXB * 4));
            M.rx[g] = (float*)p; M.flag[g] = (unsigned*)((char*)p + 2 * n * 4);
        }
    }
    CK(cudaSetDevice(0)); CK(cudaDeviceSynchronize()); CK(cudaSetDevice(1)); CK(cudaDeviceSynchronize());
}
static void free_mailbox(Mailbox& M) {
    for (int g = 0; g < 2; ++g) { if (M.host) cudaFreeHost(M.rx[g]); else { cudaSetDevice(g); cudaFree(M.rx[g]); } }
}
static void reset_mailbox(Mailbox& M, int n) {
    for (int g = 0; g < 2; ++g) {
        if (M.host) memset(M.rx[g], 0, 2 * n * 4 + 2 * MAXB * 4);
        else { CK(cudaSetDevice(g)); CK(cudaMemset(M.rx[g], 0, 2 * n * 4 + 2 * MAXB * 4)); CK(cudaDeviceSynchronize()); }
    }
}

// Runs a mailbox test. pingpong: GPU0 sends first, GPU1 responds -> per-iter time = round trip.
static void mailbox_test(const char* kind, Mailbox& M, int nbytes, int nb, bool pingpong, int read_all, int iters) {
    const int n = std::max(1, nbytes / 4);
    reset_mailbox(M, std::max(1, 20480 / 4));
    unsigned long long* ts[2]; int* err[2]; float* sink[2]; cudaStream_t st[2]; cudaEvent_t e0[2], e1[2];
    for (int g = 0; g < 2; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaMalloc(&ts[g], iters * 8)); CK(cudaMalloc(&err[g], 4)); CK(cudaMemset(err[g], 0, 4));
        CK(cudaMalloc(&sink[g], 4)); CK(cudaStreamCreateWithFlags(&st[g], cudaStreamNonBlocking));
        CK(cudaEventCreate(&e0[g])); CK(cudaEventCreate(&e1[g]));
        CK(cudaDeviceSynchronize());
    }
    for (int g = 0; g < 2; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaEventRecord(e0[g], st[g]));
        int send_first = pingpong ? (g == 0) : 1;
        xchg_kernel<<<nb, 256, 0, st[g]>>>(M.rx[1 - g], M.flag[1 - g], M.rx[g], M.flag[g], n, iters, g, send_first,
                                           read_all, ts[g], err[g], sink[g]);
        CK(cudaGetLastError());
        CK(cudaEventRecord(e1[g], st[g]));
    }
    float ms[2]; int errs[2];
    std::vector<unsigned long long> t(iters);
    for (int g = 0; g < 2; ++g) {
        CK(cudaSetDevice(g)); CK(cudaStreamSynchronize(st[g]));
        CK(cudaEventElapsedTime(&ms[g], e0[g], e1[g]));
        CK(cudaMemcpy(&errs[g], err[g], 4, cudaMemcpyDeviceToHost));
        if (g == 0) CK(cudaMemcpy(t.data(), ts[0], iters * 8, cudaMemcpyDeviceToHost));
    }
    std::vector<double> d;
    for (int i = iters / 10 + 1; i < iters; ++i) d.push_back((t[i] - t[i - 1]) * 1e-3);  // us, skip warmup 10%
    double mean = ms[0] * 1e3 / iters;
    // ping-pong: one iteration = round trip, report one-way = rt/2
    double scale = pingpong ? 0.5 : 1.0;
    printf("P {\"test\":\"mailbox\",\"kind\":\"%s\",\"mode\":\"%s\",\"bytes\":%d,\"blocks\":%d,\"read_all\":%d,"
           "\"iters\":%d,\"us_mean\":%.2f,\"us_p50\":%.2f,\"us_p90\":%.2f,\"us_p99\":%.2f,\"us_max\":%.2f,"
           "\"errors\":[%d,%d]}\n",
           kind, pingpong ? "pingpong_oneway" : "exchange", nbytes, nb, read_all, iters, mean * scale,
           pct(d, 0.5) * scale, pct(d, 0.9) * scale, pct(d, 0.99) * scale, pct(d, 1.0) * scale, errs[0], errs[1]);
    fflush(stdout);
    for (int g = 0; g < 2; ++g) {
        CK(cudaSetDevice(g)); cudaFree(ts[g]); cudaFree(err[g]); cudaFree(sink[g]); cudaStreamDestroy(st[g]);
        cudaEventDestroy(e0[g]); cudaEventDestroy(e1[g]);
    }
}

static void memcpy_peer_test(int bytes) {
    void *a, *b;
    CK(cudaSetDevice(0)); CK(cudaMalloc(&a, bytes));
    CK(cudaSetDevice(1)); CK(cudaMalloc(&b, bytes));
    CK(cudaSetDevice(0));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    std::vector<double> d;
    for (int i = 0; i < 2200; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        CK(cudaMemcpyPeerAsync(b, 1, a, 0, bytes, s));
        CK(cudaStreamSynchronize(s));
        if (i >= 200) d.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, s));
    for (int i = 0; i < 2000; ++i) CK(cudaMemcpyPeerAsync(b, 1, a, 0, bytes, s));
    CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    printf("P {\"test\":\"memcpy_peer\",\"bytes\":%d,\"sync_us_p50\":%.2f,\"sync_us_p99\":%.2f,\"pipelined_us\":%.2f}\n",
           bytes, pct(d, 0.5), pct(d, 0.99), ms * 1e3 / 2000);
    fflush(stdout);
    cudaStreamDestroy(s); cudaFree(a); CK(cudaSetDevice(1)); cudaFree(b); CK(cudaSetDevice(0));
}

static void stream_read_test(NvmlLite& nv) {
    const size_t bytes = 512ull << 20, n = bytes / 16;
    float4* p[2]; float* o[2];
    for (int g = 0; g < 2; ++g) {
        CK(cudaSetDevice(g)); CK(cudaMalloc(&p[g], bytes)); CK(cudaMemset(p[g], 0, bytes)); CK(cudaMalloc(&o[g], 4));
    }
    // single-GPU config sweep on GPU0
    CK(cudaSetDevice(0));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    double best = 0; int bg = 0, bt = 0;
    for (int grid : {160, 320, 640, 1280, 2560})
        for (int thr : {256, 512, 1024}) {
            read_kernel<<<grid, thr>>>(p[0], n, o[0]);
            CK(cudaEventRecord(e0));
            for (int i = 0; i < 10; ++i) read_kernel<<<grid, thr>>>(p[0], n, o[0]);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
            double gb = bytes * 10.0 / (ms * 1e6);
            if (gb > best) { best = gb; bg = grid; bt = thr; }
        }
    printf("P {\"test\":\"stream_read\",\"dev\":0,\"bytes\":%zu,\"best_GBps\":%.1f,\"grid\":%d,\"thr\":%d}\n", bytes, best, bg, bt);
    // D2D memcpy
    {
        void* q; CK(cudaMalloc(&q, bytes));
        CK(cudaEventRecord(e0));
        for (int i = 0; i < 5; ++i) CK(cudaMemcpyAsync(q, p[0], bytes, cudaMemcpyDeviceToDevice));
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        printf("P {\"test\":\"memcpy_d2d\",\"dev\":0,\"GBps_rw\":%.1f}\n", 2.0 * bytes * 5 / (ms * 1e6));
        cudaFree(q);
    }
    fflush(stdout);
    // sustained load on both GPUs: 12 s, sample clocks every ~100 ms
    cudaStream_t st[2];
    for (int g = 0; g < 2; ++g) { CK(cudaSetDevice(g)); CK(cudaStreamCreate(&st[g])); }
    auto t0 = std::chrono::steady_clock::now();
    std::vector<double> clk[2], pw[2], gbw[2]; unsigned long long rs[2] = {0, 0}; unsigned tmax[2] = {0, 0};
    cudaEvent_t a0[2], a1[2];
    for (int g = 0; g < 2; ++g) { CK(cudaSetDevice(g)); CK(cudaEventCreate(&a0[g])); CK(cudaEventCreate(&a1[g])); }
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 12.0) {
        for (int g = 0; g < 2; ++g) {
            CK(cudaSetDevice(g)); CK(cudaEventRecord(a0[g], st[g]));
            for (int i = 0; i < 50; ++i) read_kernel<<<bg, bt, 0, st[g]>>>(p[g], n, o[g]);
            CK(cudaEventRecord(a1[g], st[g]));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        for (int g = 0; g < 2; ++g) {
            auto s = nv.sample(g);
            clk[g].push_back(s.sm); pw[g].push_back(s.mw / 1000.0); rs[g] |= s.reasons; tmax[g] = std::max(tmax[g], s.temp);
        }
        for (int g = 0; g < 2; ++g) {
            CK(cudaSetDevice(g)); CK(cudaEventSynchronize(a1[g]));
            float ms; CK(cudaEventElapsedTime(&ms, a0[g], a1[g]));
            gbw[g].push_back(bytes * 50.0 / (ms * 1e6));
        }
    }
    for (int g = 0; g < 2; ++g) {
        char r[128]; nvml_reason_str(rs[g], r, sizeof r);
        printf("P {\"test\":\"load_clocks\",\"dev\":%d,\"load\":\"stream_read_both_gpus\",\"secs\":12,\"GBps_median\":%.1f,"
               "\"GBps_min\":%.1f,\"sm_mhz_median\":%.0f,\"sm_mhz_min\":%.0f,\"power_w_median\":%.1f,\"temp_max\":%u,"
               "\"reasons_mask\":%llu,\"reasons\":\"%s\"}\n",
               g, pct(gbw[g], 0.5), pct(gbw[g], 0.0), pct(clk[g], 0.5), pct(clk[g], 0.0), pct(pw[g], 0.5), tmax[g], rs[g], r);
    }
    fflush(stdout);
    for (int g = 0; g < 2; ++g) { CK(cudaSetDevice(g)); cudaFree(p[g]); cudaFree(o[g]); cudaStreamDestroy(st[g]); }
    CK(cudaSetDevice(0));
}

// ---------------------------------------------------------------------------------------------- NCCL via dlopen
typedef void* ncclComm_t;
typedef int (*f_initall)(ncclComm_t*, int, const int*);
typedef int (*f_allreduce)(const void*, void*, size_t, int, int, ncclComm_t, cudaStream_t);
typedef int (*f_group)();
typedef int (*f_version)(int*);
typedef const char* (*f_errstr)(int);

static int nccl_test(const char* lib) {
    void* h = dlopen(lib, RTLD_NOW);
    if (!h) { printf("P {\"test\":\"nccl\",\"error\":\"dlopen failed: %s\"}\n", dlerror()); return 1; }
    auto initall = (f_initall)dlsym(h, "ncclCommInitAll");
    auto allreduce = (f_allreduce)dlsym(h, "ncclAllReduce");
    auto gstart = (f_group)dlsym(h, "ncclGroupStart");
    auto gend = (f_group)dlsym(h, "ncclGroupEnd");
    auto ver = (f_version)dlsym(h, "ncclGetVersion");
    auto es = (f_errstr)dlsym(h, "ncclGetErrorString");
    int v = 0; if (ver) ver(&v);
    ncclComm_t comms[2]; int devs[2] = {0, 1};
    int r = initall(comms, 2, devs);
    if (r) { printf("P {\"test\":\"nccl\",\"error\":\"init %d %s\"}\n", r, es ? es(r) : ""); return 1; }
    const char* envs[] = {"NCCL_P2P_LEVEL", "NCCL_PROTO", "NCCL_ALGO", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE"};
    std::string env;
    for (auto e : envs) if (getenv(e)) env += std::string(env.empty() ? "" : " ") + e + "=" + getenv(e);
    for (int dtype : {7 /*f32*/, 6 /*f16*/}) {
        const int count = 5120; const size_t esz = dtype == 7 ? 4 : 2;
        void* buf[2]; cudaStream_t st[2];
        for (int g = 0; g < 2; ++g) {
            CK(cudaSetDevice(g)); CK(cudaMalloc(&buf[g], count * esz)); CK(cudaStreamCreate(&st[g]));
            if (dtype == 7) {
                std::vector<float> hv(count, (float)(g + 1));
                CK(cudaMemcpy(buf[g], hv.data(), count * 4, cudaMemcpyHostToDevice));
            } else CK(cudaMemset(buf[g], 0, count * esz));
        }
        auto one = [&]() {
            gstart();
            for (int g = 0; g < 2; ++g) allreduce(buf[g], buf[g], count, dtype, 0, comms[g], st[g]);
            gend();
        };
        // correctness (f32 only): 1 + 2 = 3
        one();
        for (int g = 0; g < 2; ++g) { CK(cudaSetDevice(g)); CK(cudaStreamSynchronize(st[g])); }
        int ok = 1;
        if (dtype == 7) {
            std::vector<float> hv(count);
            for (int g = 0; g < 2; ++g) {
                CK(cudaSetDevice(g)); CK(cudaMemcpy(hv.data(), buf[g], count * 4, cudaMemcpyDeviceToHost));
                for (float x : hv) ok &= x == 3.f;
            }
        }
        std::vector<double> d;
        for (int i = 0; i < 2200; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            one();
            for (int g = 0; g < 2; ++g) { CK(cudaSetDevice(g)); CK(cudaStreamSynchronize(st[g])); }
            if (i >= 200) d.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        }
        cudaEvent_t e0, e1; CK(cudaSetDevice(0)); CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        CK(cudaEventRecord(e0, st[0]));
        for (int i = 0; i < 2000; ++i) one();
        CK(cudaEventRecord(e1, st[0]));
        for (int g = 0; g < 2; ++g) { CK(cudaSetDevice(g)); CK(cudaStreamSynchronize(st[g])); }
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        printf("P {\"test\":\"nccl\",\"version\":%d,\"env\":\"%s\",\"dtype\":\"%s\",\"bytes\":%zu,\"correct\":%d,"
               "\"sync_us_p50\":%.2f,\"sync_us_p99\":%.2f,\"pipelined_us\":%.2f}\n",
               v, env.c_str(), dtype == 7 ? "f32" : "f16", count * esz, ok, pct(d, 0.5), pct(d, 0.99), ms * 1e3 / 2000);
        fflush(stdout);
        for (int g = 0; g < 2; ++g) { CK(cudaSetDevice(g)); cudaFree(buf[g]); cudaStreamDestroy(st[g]); }
    }
    return 0;
}

int main(int argc, char** argv) {
    int ndev = 0; CK(cudaGetDeviceCount(&ndev));
    if (argc >= 3 && !strcmp(argv[1], "nccl")) { int r = nccl_test(argv[2]); printf("PROBE_DONE\n"); return r; }
    NvmlLite nv; nv.init();
    int a01 = 0, a10 = 0;
    if (ndev >= 2) { cudaDeviceCanAccessPeer(&a01, 0, 1); cudaDeviceCanAccessPeer(&a10, 1, 0); }
    int atom01 = 0, perf01 = 0;
    if (ndev >= 2) {
        cudaDeviceGetP2PAttribute(&atom01, cudaDevP2PAttrNativeAtomicSupported, 0, 1);
        cudaDeviceGetP2PAttribute(&perf01, cudaDevP2PAttrPerformanceRank, 0, 1);
    }
    printf("P {\"test\":\"info\",\"ndev\":%d,\"can_access_peer_01\":%d,\"can_access_peer_10\":%d,\"native_atomics_01\":%d,"
           "\"perf_rank_01\":%d,\"nvml\":%d}\n", ndev, a01, a10, atom01, perf01, (int)nv.ok);
    fflush(stdout);
    if (ndev < 2) { printf("PROBE_DONE\n"); return 1; }
    if (a01 && a10) {
        CK(cudaSetDevice(0)); CK(cudaDeviceEnablePeerAccess(1, 0));
        CK(cudaSetDevice(1)); CK(cudaDeviceEnablePeerAccess(0, 0));
        CK(cudaSetDevice(0));
        memcpy_peer_test(10240);
        memcpy_peer_test(20480);
        Mailbox P; alloc_mailbox(P, false, 20480 / 4);
        for (int bytes : {4, 10240, 20480}) mailbox_test("p2p", P, bytes, 1, true, 1, 10000);
        for (int nb : {8, 40}) {  // parallel writers: one-way latency of a 10/20 KB payload split over nb blocks
            mailbox_test("p2p", P, 20480, nb, true, 0, 10000);
            mailbox_test("p2p", P, 10240, nb, true, 0, 10000);
        }
        for (int nb : {1, 8, 20, 40}) mailbox_test("p2p", P, 20480, nb, false, 1, 10000);
        mailbox_test("p2p", P, 20480, 40, false, 0, 10000);
        mailbox_test("p2p", P, 10240, 40, false, 1, 10000);
        free_mailbox(P);
    }
    {
        Mailbox H; alloc_mailbox(H, true, 20480 / 4);
        for (int bytes : {4, 10240, 20480}) mailbox_test("hostmapped", H, bytes, 1, true, 1, 5000);
        for (int nb : {1, 8, 40}) mailbox_test("hostmapped", H, 20480, nb, false, 1, 3000);
        mailbox_test("hostmapped", H, 20480, 40, false, 0, 5000);
        free_mailbox(H);
    }
    stream_read_test(nv);
    printf("PROBE_DONE\n");
    return 0;
}
