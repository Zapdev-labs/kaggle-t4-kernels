// cf-m3 CUDA-graph launch-wall probe (r19s): the G1/G2 win measured before the engine
// work. The r19p census counted the decode step at ~2,578 dependent launches; at the
// ~22 us/launch measured class that is the ~57 ms wall - but that class was inherited
// from the 27B-era measurements, and the graph forms' win (the launch overhead the
// capture removes) has never been measured on this kernel mix. This probe builds the
// SAME-shaped chain - every 4th launch a gemv-ish kernel streaming a ~13 MB weight
// slab (the packed-slab read class of one moe gemv), the rest the norm/add-class tiny
// kernel, all on ONE stream so the chain is strictly serial - and measures it three
// ways, the kernel work identical in all three:
//   (A) launch-by-launch              - the current engine's form
//   (B) ONE captured full-step graph   - the G2 form (~1 replay/token)
//   (C) ~50 per-segment graphs with a cudaStreamSynchronize + a ~30 us host loop
//       between replays (the router's softmax/top-10 class) - the G1 form
// The differential (A)-(B) and (A)-(C) is the pure graph win on this hardware; the
// instantiate time is reported as the one-time load cost of the graph set.
//
//   ./graph_probe [n_launch=2578] [n_seg=50]
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <chrono>

#define CK(x)                                                                              \
    do {                                                                                   \
        cudaError_t e = (x);                                                               \
        if (e != cudaSuccess) {                                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

// the norm/add-class kernel: 41 CTAs x 256 threads over 10240 floats
__global__ void k_tiny(float* __restrict__ y, const float* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * 0.999f + 0.001f;
}

// the gemv-ish kernel: a warp per row over K=2560, 8 rows per CTA (160 CTAs), the whole
// ~13 MB slab streamed per launch (the packed-slab read class of one moe gemv)
__global__ void k_gemvish(const float* __restrict__ w, const float* __restrict__ x,
                          float* __restrict__ y, int rows, int k) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) return;
    const float* wr = w + (size_t)row * k;
    float acc = 0.f;
    for (int i = lane; i < k; i += 32) acc += wr[i] * x[i];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = acc;
}

struct Chain {
    float *x, *y, *gx, *gy, *w;
    static constexpr int N_TINY = 10240, ROWS = 1280, K = 2560;
    // the i-th op of the census-shaped chain: every 4th is the gemv-ish (a fixed
    // input/output pair - the single stream serializes the chain, so the wall is the
    // launch+exec floor), the tiny buffers ping-pong within their own family
    void op(int i, cudaStream_t st) const {
        if ((i & 3) == 3) {
            k_gemvish<<<ROWS / 8, 256, 0, st>>>(w, gx, gy, ROWS, K);
        } else {
            if (i & 1) k_tiny<<<(N_TINY + 255) / 256, 256, 0, st>>>(y, x, N_TINY);
            else k_tiny<<<(N_TINY + 255) / 256, 256, 0, st>>>(x, y, N_TINY);
        }
    }
};

// ~30 us of honest host work: the router's softmax + top-10 over 512 logits
static void host_router_loop(volatile float* h, int n) {
    for (int k = 0; k < 4; k++) {
        float m = h[0];
        for (int i = 1; i < n; i++) if (h[i] > m) m = h[i];
        float s = 0.f;
        for (int i = 0; i < n; i++) { h[i] = expf(h[i] - m); s += h[i]; }
        for (int i = 0; i < n; i++) h[i] /= s;
        for (int t = 0; t < 10; t++) {
            int best = 0;
            for (int i = 1; i < n; i++) if (h[i] > h[best]) best = i;
            h[best] = -1.f;
        }
    }
}

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 2578;
    int nseg = argc > 2 ? atoi(argv[2]) : 50;
    if (n < 8 || nseg < 1 || nseg > n) {
        fprintf(stderr, "usage: %s [n_launch=2578] [n_seg=50]\n", argv[0]);
        return 1;
    }
    int dev = 0;
    CK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));
    printf("[graph] %s, %d SMs | chain %d launches, %d segments\n", prop.name, prop.multiProcessorCount, n, nseg);

    Chain c;
    CK(cudaMalloc(&c.x, Chain::N_TINY * 4));
    CK(cudaMalloc(&c.y, Chain::N_TINY * 4));
    CK(cudaMalloc(&c.gx, Chain::K * 4));
    CK(cudaMalloc(&c.gy, Chain::ROWS * 4));
    CK(cudaMalloc(&c.w, (size_t)Chain::ROWS * Chain::K * 4));
    CK(cudaMemset(c.x, 0, Chain::N_TINY * 4));
    CK(cudaMemset(c.gx, 0, Chain::K * 4));
    CK(cudaMemset(c.w, 0x3c, (size_t)Chain::ROWS * Chain::K * 4));
    volatile float* h = (volatile float*)malloc(512 * 4);
    for (int i = 0; i < 512; i++) h[i] = 0.01f * i;

    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    cudaEvent_t ev[2];
    CK(cudaEventCreate(&ev[0]));
    CK(cudaEventCreate(&ev[1]));

    // warmup (the L2 + the clocks steady)
    for (int i = 0; i < n; i++) c.op(i, st);
    CK(cudaStreamSynchronize(st));

    // (A) launch-by-launch: the current engine's form
    double a_ms = 0;
    for (int pass = 0; pass < 3; pass++) {
        CK(cudaEventRecord(ev[0], st));
        for (int i = 0; i < n; i++) c.op(i, st);
        CK(cudaEventRecord(ev[1], st));
        CK(cudaEventSynchronize(ev[1]));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, ev[0], ev[1]));
        if (!a_ms || ms < a_ms) a_ms = ms;
    }
    printf("[graph] A) launch-by-launch : %7.3f ms (%.2f us/launch over %d)\n", a_ms, a_ms * 1e3 / n, n);

    // (B) ONE captured full-step graph: the G2 form
    cudaGraph_t g = nullptr;
    cudaGraphExec_t gx = nullptr;
    double inst_b = 0;
    {
        auto t0 = std::chrono::steady_clock::now();
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
        for (int i = 0; i < n; i++) c.op(i, st);
        CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&gx, g, 0));
        auto t1 = std::chrono::steady_clock::now();
        inst_b = std::chrono::duration<double>(t1 - t0).count();
    }
    double b_ms = 0;
    for (int pass = 0; pass < 3; pass++) {
        CK(cudaEventRecord(ev[0], st));
        CK(cudaGraphLaunch(gx, st));
        CK(cudaEventRecord(ev[1], st));
        CK(cudaEventSynchronize(ev[1]));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, ev[0], ev[1]));
        if (!b_ms || ms < b_ms) b_ms = ms;
    }
    printf("[graph] B) full-step graph : %7.3f ms (one replay; instantiate %.2f s)\n", b_ms, inst_b);

    // (C) per-segment graphs with a sync + the ~30 us host router loop between: the G1 form
    const int per = (n + nseg - 1) / nseg;
    const int actual_seg = (n + per - 1) / per;
    cudaGraph_t* gs = (cudaGraph_t*)calloc(actual_seg, sizeof(cudaGraph_t));
    cudaGraphExec_t* xs = (cudaGraphExec_t*)calloc(actual_seg, sizeof(cudaGraphExec_t));
    double inst_c = 0;
    {
        auto t0 = std::chrono::steady_clock::now();
        for (int s = 0; s < actual_seg; s++) {
            const int i0 = s * per, i1 = (s + 1) * per < n ? (s + 1) * per : n;
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
            for (int i = i0; i < i1; i++) c.op(i, st);
            CK(cudaStreamEndCapture(st, &gs[s]));
            CK(cudaGraphInstantiate(&xs[s], gs[s], 0));
        }
        auto t1 = std::chrono::steady_clock::now();
        inst_c = std::chrono::duration<double>(t1 - t0).count();
    }
    double c_ms = 0;
    for (int pass = 0; pass < 3; pass++) {
        auto t0 = std::chrono::steady_clock::now();
        for (int s = 0; s < actual_seg; s++) {
            CK(cudaGraphLaunch(xs[s], st));
            CK(cudaStreamSynchronize(st));
            host_router_loop(h, 512);  // the router's host window between segments
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double>(t1 - t0).count() * 1e3;
        if (!c_ms || ms < c_ms) c_ms = ms;
    }
    printf("[graph] C) %3d seg graphs  : %7.3f ms (sync + ~30 us host router loop between; instantiate %.2f s)\n",
           actual_seg, c_ms, inst_c);

    printf("[graph] wins: A-B = %.3f ms (the G2 form's reclamation), A-C = %.3f ms (the G1 form's)\n",
           a_ms - b_ms, a_ms - c_ms);
    printf("[graph] verdict: the launch wall on this host is %.2f us/launch; the graphs reclaim "
           "%.0f%% (G2) / %.0f%% (G1) of it\n",
           a_ms * 1e3 / n, 100.0 * (a_ms - b_ms) / a_ms, 100.0 * (a_ms - c_ms) / a_ms);

    for (int s = 0; s < actual_seg; s++) {
        CK(cudaGraphExecDestroy(xs[s]));
        CK(cudaGraphDestroy(gs[s]));
    }
    CK(cudaGraphExecDestroy(gx));
    CK(cudaGraphDestroy(g));
    free(xs);
    free(gs);
    free((void*)h);
    CK(cudaEventDestroy(ev[0]));
    CK(cudaEventDestroy(ev[1]));
    CK(cudaStreamDestroy(st));
    CK(cudaFree(c.x));
    CK(cudaFree(c.y));
    CK(cudaFree(c.gx));
    CK(cudaFree(c.gy));
    CK(cudaFree(c.w));
    return 0;
}
