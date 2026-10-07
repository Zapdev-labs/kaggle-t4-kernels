// cf-m3 UVA feasibility probe (r19o): the mapped-read bandwidth vs the staged path, on the
// real page-cache behavior. The minimal-UVA design (the repack's input pointer swapped to
// the mapped host pages - no pinned staging, no H2D) is only a win if a kernel reading
// cudaHostRegisterMapped pages sustains a large fraction of the PCIe link; the fused raw
// gemv has the same open question. This probe measures BOTH paths on the same bytes:
//
//   ./uva_probe <file> [offset_mb] [gib] [mode]
//     mode u = the mapped path only (register -> read -> unregister)
//     mode b = the A/B: (A) the mapped reads, (B) cudaMemcpyAsync H2D from the same pages
//              into VRAM staging + the kernel reading the staging (the current engine form)
//
// The read kernel is the bandwidth-realistic pattern of the moe gemv/repack input: coalesced
// 16-B loads over contiguous expert-slab-sized regions (the expert gu slab is a contiguous
// 1280 x 840 B = 1.075 MB region), grid-strided to saturation, a trivial ALU (the word sum)
// so the loads dominate. The probe reports the registration time (the pinning cost for the
// range), the steady GB/s of each path over 3 passes (best), and the RSS growth across the
// registration (the pinning's real memory footprint - the Kaggle-host feasibility input).
// The result decides the cf-m3 form: if the mapped reads sustain ~>= 0.6x the staged path's
// effective bytes/s, the minimal UVA stands; if the TLB/mapped-path behavior halves it, the
// fused raw gemv (fewer passes over the wire) or the staged path stays.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <vector>

#define CK(x)                                                                              \
    do {                                                                                   \
        cudaError_t e = (x);                                                               \
        if (e != cudaSuccess) {                                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

static size_t resident_bytes() {
    FILE* f = fopen("/proc/self/statm", "r");
    if (!f) return 0;
    unsigned long long tot = 0, res = 0;
    if (fscanf(f, "%llu %llu", &tot, &res) != 2) res = 0;
    fclose(f);
    return res * 4096ULL;
}

__global__ void k_read_sum(const uint4* __restrict__ p, size_t n16, unsigned long long* sink) {
    unsigned long long acc = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = p[i];
        acc += v.x + v.y + v.z + v.w;
    }
    __shared__ unsigned long long sh[32];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long t = 0;
        for (int i = 0; i < (blockDim.x >> 5); i++) t += sh[i];
        if (t == 0xdeadbeefULL) *sink = t;  // never true; keeps the loads live
    }
}

static double bench(const uint4* p, size_t n16, unsigned long long* sink, cudaStream_t st, int sm) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    double best = 0;
    for (int pass = 0; pass < 3; pass++) {
        CK(cudaEventRecord(a, st));
        for (int rep = 0; rep < 3; rep++)
            k_read_sum<<<sm * 16, 256, 0, st>>>(p, n16, sink);
        CK(cudaEventRecord(b, st));
        CK(cudaEventSynchronize(b));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, a, b));
        double gbps = (double)n16 * 16.0 * 3.0 / (ms / 1e3) / 1e9;
        if (!best || gbps > best) best = gbps;
    }
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    return best;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <file> [offset_mb=0] [gib=2] [mode=u|b]\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];
    long long off_mb = argc > 2 ? atoll(argv[2]) : 0;
    double gib = argc > 3 ? atof(argv[3]) : 2.0;
    const char* mode = argc > 4 ? argv[4] : "b";

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    struct stat sb;
    if (fstat(fd, &sb)) {
        perror("fstat");
        return 1;
    }
    size_t len = (size_t)(gib * (1 << 30));
    if ((size_t)(off_mb << 20) + len > (size_t)sb.st_size) {
        if ((size_t)sb.st_size > (size_t)(off_mb << 20)) len = (size_t)sb.st_size - (size_t)(off_mb << 20);
        else { fprintf(stderr, "offset past EOF\n"); return 1; }
    }
    len &= ~15ULL;  // 16-B aligned
    void* map = mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, (off_t)off_mb << 20);
    if (map == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    int dev = 0;
    CK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));
    size_t rss0 = resident_bytes();
    printf("[probe] %s: %.2f GiB at %lld MB | %s, %d SMs, pcie link via the measured path\n", path, (double)len / (1 << 30),
           off_mb, prop.name, prop.multiProcessorCount);

    // touch the range once so the mapped and staged paths read the SAME (warm) page cache
    {
        volatile unsigned long long acc = 0;
        const unsigned long long* q = (const unsigned long long*)map;
        for (size_t i = 0; i + 8 <= len; i += 8 * 4096) acc += q[i / 8];
        (void)acc;
    }
    size_t rss1 = resident_bytes();

    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    unsigned long long* sink;
    CK(cudaMalloc(&sink, 8));
    CK(cudaMemset(sink, 0, 8));
    const size_t n16 = len / 16;

    auto t0 = std::chrono::steady_clock::now();
    CK(cudaHostRegister(map, len, cudaHostRegisterMapped));
    auto t1 = std::chrono::steady_clock::now();
    void* dev_alias = nullptr;
    CK(cudaHostGetDevicePointer(&dev_alias, map, 0));
    size_t rss2 = resident_bytes();
    printf("[probe] register: %.2f s | RSS mmap-touch %.2f GiB -> after register %.2f GiB\n",
           std::chrono::duration<double>(t1 - t0).count(), (double)(rss1 - rss0) / (1 << 30),
           (double)(rss2 - rss0) / (1 << 30));

    double gbps_map = bench((const uint4*)dev_alias, n16, sink, st, prop.multiProcessorCount);
    printf("[probe] A) mapped reads : %7.2f GB/s (kernel reads over cudaHostRegisterMapped pages)\n", gbps_map);

    if (mode[0] == 'b') {
        void* stage;
        CK(cudaMalloc(&stage, len));
        // B) the staged form: H2D from the same (now warm) page-cache pages + the kernel read
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a));
        CK(cudaEventCreate(&b));
        double best = 0;
        for (int pass = 0; pass < 3; pass++) {
            CK(cudaEventRecord(a, st));
            for (int rep = 0; rep < 3; rep++) {
                CK(cudaMemcpyAsync(stage, map, len, cudaMemcpyHostToDevice, st));
                k_read_sum<<<prop.multiProcessorCount * 16, 256, 0, st>>>((const uint4*)stage, n16, sink);
            }
            CK(cudaEventRecord(b, st));
            CK(cudaEventSynchronize(b));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, a, b));
            double gbps = (double)n16 * 16.0 * 3.0 / (ms / 1e3) / 1e9;  // the wire carries the bytes once
            if (!best || gbps > best) best = gbps;
        }
        printf("[probe] B) staged H2D+read: %7.2f GB/s effective (memcpy H2D + the same kernel over VRAM)\n", best);
        printf("[probe] ratio A/B: %.3f (>= ~0.6 supports the minimal-UVA swap; the fused gemv reads the same wire)\n",
               best > 0 ? gbps_map / best : 0.0);
        CK(cudaFree(stage));
        CK(cudaEventDestroy(a));
        CK(cudaEventDestroy(b));
    }

    auto t2 = std::chrono::steady_clock::now();
    CK(cudaHostUnregister(map));
    auto t3 = std::chrono::steady_clock::now();
    printf("[probe] unregister: %.3f s\n", std::chrono::duration<double>(t3 - t2).count());
    CK(cudaFree(sink));
    CK(cudaStreamDestroy(st));
    munmap(map, len);
    close(fd);
    return 0;
}
