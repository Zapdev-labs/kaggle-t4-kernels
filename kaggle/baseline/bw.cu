// T4 bandwidth / P2P / latency microbench. nvcc -O3 -arch=sm_75 bw.cu -o bw
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("ERR %s: %s (line %d)\n", #x, cudaGetErrorString(e), __LINE__); } } while (0)

__global__ void read_kernel(const float4 *__restrict__ p, size_t n, float *out) {
    float acc = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float4 v = __ldg(p + i);
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.678f) out[0] = acc;  // defeat DCE
}
__global__ void read_kernel_peer(const float4 *__restrict__ p, size_t n, float *out) {
    float acc = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float4 v = p[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.678f) out[0] = acc;
}
__global__ void copy_kernel(const float4 *__restrict__ a, float4 *__restrict__ b, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) b[i] = a[i];
}
__global__ void empty_kernel() {}

static float ms_since(cudaEvent_t a, cudaEvent_t b) { float m; cudaEventSynchronize(b); cudaEventElapsedTime(&m, a, b); return m; }

int main() {
    int ndev = 0; CK(cudaGetDeviceCount(&ndev));
    int drv = 0, rt = 0; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
    printf("RES ndev=%d driver_api=%d runtime=%d\n", ndev, drv, rt);
    for (int d = 0; d < ndev; d++) {
        cudaDeviceProp p; cudaGetDeviceProperties(&p, d);
        int clk = 0, memclk = 0, busw = 0;
        cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, d);
        cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, d);
        cudaDeviceGetAttribute(&busw, cudaDevAttrGlobalMemoryBusWidth, d);
        printf("RES dev%d name=%s sm=%d.%d SMs=%d clk_khz=%d memclk_khz=%d buswidth=%d theo_GBps=%.1f l2=%d smem_per_block_optin=%zu regs_per_sm=%d pci=%02x:%02x\n",
               d, p.name, p.major, p.minor, p.multiProcessorCount, clk, memclk, busw,
               2.0 * memclk * 1e3 * (busw / 8) / 1e9, p.l2CacheSize, p.sharedMemPerBlockOptin, p.regsPerMultiprocessor, p.pciBusID, p.pciDeviceID);
    }
    if (ndev >= 2) {
        int a = 0, b = 0; CK(cudaDeviceCanAccessPeer(&a, 0, 1)); CK(cudaDeviceCanAccessPeer(&b, 1, 0));
        printf("RES can_access_peer_0_1=%d can_access_peer_1_0=%d\n", a, b);
        int perf = -1, atom = -1, sup = -1;
        cudaDeviceGetP2PAttribute(&sup, cudaDevP2PAttrAccessSupported, 0, 1);
        cudaDeviceGetP2PAttribute(&perf, cudaDevP2PAttrPerformanceRank, 0, 1);
        cudaDeviceGetP2PAttribute(&atom, cudaDevP2PAttrNativeAtomicSupported, 0, 1);
        printf("RES p2p_attr access_supported=%d perf_rank=%d native_atomic=%d\n", sup, perf, atom);
    }

    // ---- single GPU bandwidth ----
    const size_t BYTES = (size_t)1 << 30;  // 1 GiB
    CK(cudaSetDevice(0));
    float4 *A, *B; float *out;
    CK(cudaMalloc(&A, BYTES)); CK(cudaMalloc(&B, BYTES)); CK(cudaMalloc(&out, 4));
    CK(cudaMemset(A, 0, BYTES)); CK(cudaMemset(B, 0, BYTES));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int IT = 20;
    CK(cudaMemcpy(B, A, BYTES, cudaMemcpyDeviceToDevice));
    cudaEventRecord(e0);
    for (int i = 0; i < IT; i++) cudaMemcpy(B, A, BYTES, cudaMemcpyDeviceToDevice);
    cudaEventRecord(e1);
    float m = ms_since(e0, e1);
    printf("RES memcpy_d2d_GBps=%.1f (counting read+write, 1GiB x%d)\n", 2.0 * BYTES * IT / (m * 1e-3) / 1e9, IT);

    size_t n4 = BYTES / 16;
    int sms = 40; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    int best_grid = 0; double best = 0;
    for (int bpsm = 2; bpsm <= 32; bpsm *= 2) for (int th = 256; th <= 1024; th *= 2) {
        int grid = sms * bpsm;
        read_kernel<<<grid, th>>>(A, n4, out);
        cudaEventRecord(e0);
        for (int i = 0; i < IT; i++) read_kernel<<<grid, th>>>(A, n4, out);
        cudaEventRecord(e1);
        m = ms_since(e0, e1);
        double gbps = (double)BYTES * IT / (m * 1e-3) / 1e9;
        printf("RES read_kernel grid=%d threads=%d GBps=%.1f\n", grid, th, gbps);
        if (gbps > best) { best = gbps; best_grid = grid * 10000 + th; }
    }
    printf("RES read_kernel_best_GBps=%.1f cfg=%d\n", best, best_grid);
    // smaller working sets typical of one weight matrix (e.g. 5120x17408 q4 ~ 45MB)
    for (size_t sz : {(size_t)8 << 20, (size_t)45 << 20, (size_t)128 << 20}) {
        size_t n = sz / 16; int grid = sms * 8;
        read_kernel<<<grid, 512>>>(A, n, out);
        cudaEventRecord(e0);
        for (int i = 0; i < 200; i++) read_kernel<<<grid, 512>>>(A, n, out);
        cudaEventRecord(e1);
        m = ms_since(e0, e1);
        printf("RES read_kernel_size_MB=%zu GBps=%.1f us_per=%.2f\n", sz >> 20, (double)sz * 200 / (m * 1e-3) / 1e9, m * 1e3 / 200);
    }
    cudaEventRecord(e0);
    for (int i = 0; i < IT; i++) copy_kernel<<<sms * 8, 512>>>(A, B, n4);
    cudaEventRecord(e1);
    m = ms_since(e0, e1);
    printf("RES copy_kernel_GBps=%.1f (read+write)\n", 2.0 * BYTES * IT / (m * 1e-3) / 1e9);

    // launch latency
    for (int i = 0; i < 100; i++) empty_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int i = 0; i < 10000; i++) empty_kernel<<<1, 32>>>();
    cudaEventRecord(e1);
    m = ms_since(e0, e1);
    printf("RES empty_kernel_back_to_back_us=%.2f\n", m * 1e3 / 10000);
    {
        cudaStream_t s; cudaStreamCreate(&s);
        cudaGraph_t g; cudaGraphExec_t ge;
        cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        for (int i = 0; i < 1000; i++) empty_kernel<<<1, 32, 0, s>>>();
        cudaStreamEndCapture(s, &g);
        CK(cudaGraphInstantiate(&ge, g, 0));
        cudaGraphLaunch(ge, s); cudaStreamSynchronize(s);
        cudaEventRecord(e0, s);
        for (int r = 0; r < 10; r++) cudaGraphLaunch(ge, s);
        cudaEventRecord(e1, s);
        m = ms_since(e0, e1);
        printf("RES empty_kernel_in_graph_us=%.2f\n", m * 1e3 / 10000);
    }
    // launch+sync roundtrip (host)
    {
        float tot = 0; cudaEvent_t h0, h1; cudaEventCreate(&h0); cudaEventCreate(&h1);
        auto t0 = clock();
        (void)t0;
        cudaEventRecord(h0);
        for (int i = 0; i < 2000; i++) { empty_kernel<<<1, 32>>>(); cudaDeviceSynchronize(); }
        cudaEventRecord(h1);
        tot = ms_since(h0, h1);
        printf("RES launch_sync_roundtrip_us=%.2f\n", tot * 1e3 / 2000);
    }

    if (ndev < 2) { printf("RES done_single\n"); return 0; }

    // ---- cross-GPU copies ----
    const size_t SMALL = 10 * 1024;
    float4 *S0, *S1, *L1; void *H;
    CK(cudaSetDevice(1)); CK(cudaMalloc(&S1, SMALL)); CK(cudaMalloc(&L1, (size_t)256 << 20));
    CK(cudaSetDevice(0)); CK(cudaMalloc(&S0, SMALL));
    CK(cudaMallocHost(&H, (size_t)256 << 20));
    auto peer_lat = [&](const char *tag) {
        CK(cudaSetDevice(0));
        for (int i = 0; i < 50; i++) cudaMemcpyPeer(S1, 1, S0, 0, SMALL);
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        const int N = 2000;
        for (int i = 0; i < N; i++) { cudaMemcpyPeerAsync(S1, 1, S0, 0, SMALL, 0); cudaStreamSynchronize(0); }
        cudaEventRecord(e1);
        float mm = ms_since(e0, e1);
        printf("RES %s peer_copy_10KB_sync_us=%.2f\n", tag, mm * 1e3 / N);
        cudaEventRecord(e0);
        for (int i = 0; i < N; i++) cudaMemcpyPeerAsync(S1, 1, S0, 0, SMALL, 0);
        cudaEventRecord(e1);
        mm = ms_since(e0, e1);
        printf("RES %s peer_copy_10KB_pipelined_us=%.2f\n", tag, mm * 1e3 / N);
        // large peer bandwidth 256MB, dev0->dev1 (source A on dev0)
        cudaMemcpyPeer(L1, 1, A, 0, (size_t)256 << 20);
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for (int i = 0; i < 5; i++) cudaMemcpyPeerAsync(L1, 1, A, 0, (size_t)256 << 20, 0);
        cudaEventRecord(e1);
        mm = ms_since(e0, e1);
        printf("RES %s peer_copy_256MB_GBps=%.2f\n", tag, 5.0 * (256 << 20) / (mm * 1e-3) / 1e9);
    };
    peer_lat("nop2p");
    // explicit host staging (D2H + H2D, pinned), 10KB
    {
        cudaStream_t s0, s1; CK(cudaSetDevice(0)); cudaStreamCreate(&s0); CK(cudaSetDevice(1)); cudaStreamCreate(&s1); CK(cudaSetDevice(0));
        const int N = 2000;
        cudaEventRecord(e0);
        for (int i = 0; i < N; i++) {
            cudaMemcpyAsync(H, S0, SMALL, cudaMemcpyDeviceToHost, s0); cudaStreamSynchronize(s0);
            cudaMemcpyAsync(S1, H, SMALL, cudaMemcpyHostToDevice, s1); cudaStreamSynchronize(s1);
        }
        cudaEventRecord(e1);
        printf("RES host_staged_10KB_us=%.2f\n", ms_since(e0, e1) * 1e3 / N);
        cudaEventRecord(e0);
        for (int i = 0; i < 5; i++) { cudaMemcpyAsync(H, A, (size_t)256 << 20, cudaMemcpyDeviceToHost, s0); cudaStreamSynchronize(s0); }
        cudaEventRecord(e1);
        printf("RES d2h_pinned_256MB_GBps=%.2f\n", 5.0 * (256 << 20) / (ms_since(e0, e1) * 1e-3) / 1e9);
        CK(cudaSetDevice(1));
        cudaEvent_t f0, f1; cudaEventCreate(&f0); cudaEventCreate(&f1);
        cudaEventRecord(f0, s1);
        for (int i = 0; i < 5; i++) cudaMemcpyAsync(L1, H, (size_t)256 << 20, cudaMemcpyHostToDevice, s1);
        cudaEventRecord(f1, s1);
        printf("RES h2d_pinned_256MB_GBps=%.2f\n", 5.0 * (256 << 20) / (ms_since(f0, f1) * 1e-3) / 1e9);
        CK(cudaSetDevice(0));
    }
    int can = 0; cudaDeviceCanAccessPeer(&can, 0, 1);
    if (can) {
        CK(cudaSetDevice(0)); CK(cudaDeviceEnablePeerAccess(1, 0));
        CK(cudaSetDevice(1)); CK(cudaDeviceEnablePeerAccess(0, 0));
        peer_lat("p2p");
        // dev1 kernel directly reading dev0 memory over PCIe
        CK(cudaSetDevice(1));
        float *o1; cudaMalloc(&o1, 4);
        cudaEvent_t f0, f1; cudaEventCreate(&f0); cudaEventCreate(&f1);
        size_t n = ((size_t)256 << 20) / 16;
        read_kernel_peer<<<40 * 8, 512>>>(A, n, o1);
        cudaEventRecord(f0);
        for (int i = 0; i < 5; i++) read_kernel_peer<<<40 * 8, 512>>>(A, n, o1);
        cudaEventRecord(f1);
        printf("RES p2p_remote_read_kernel_GBps=%.2f\n", 5.0 * (256 << 20) / (ms_since(f0, f1) * 1e-3) / 1e9);
        CK(cudaGetLastError());
    } else {
        printf("RES p2p_unavailable=1\n");
    }
    printf("RES done\n");
    return 0;
}
