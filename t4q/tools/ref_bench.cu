// ref_bench.cu -- milestone P round 2: vendor GEMM references under the T4's 70 W cap.
//
// Question: is the W4A8 GEMM's ~24 TOPS sustained (both GPUs loaded) an energy wall of the chip, or a property of
// our kernel (31% of tensor peak per clock)? This measures cuBLAS and CUTLASS int8 / int4 / fp16 tensor-core GEMMs on
// the gateup shape (T=2048 tokens, N=17408, K=5120), burst and sustained, with NVML clock / power per window.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_75 ref_bench.cu -o ref_bench -lcublas -lcublasLt -ldl
//        add -DT4Q_CUTLASS -I<cutlass>/include for the CUTLASS variants.
// Run:   ./ref_bench --dev N [--sustain SECS] [--variants a,b,c] [--T 2048] [--N 17408] [--K 5120]
// Output: "B {json}" burst rows, "S {json}" sustain windows, "DONE".
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "nvml_lite.h"

#ifdef T4Q_CUTLASS
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/numeric_types.h"
#endif

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) {                                                                     \
            printf("FATAL cuda %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); \
            fflush(stdout);                                                                          \
            exit(2);                                                                                 \
        }                                                                                            \
    } while (0)
#define CB(x)                                                                         \
    do {                                                                              \
        int s_ = (int)(x);                                                            \
        if (s_ != 0) {                                                                \
            printf("ERR blas status %d at %s:%d: %s\n", s_, __FILE__, __LINE__, #x); \
            fflush(stdout);                                                           \
            return false;                                                             \
        }                                                                             \
    } while (0)

static NvmlLite g_nvml;
static int g_dev = 0;

struct Bufs {
    int8_t* x8 = nullptr;   // [T][K]
    int8_t* w8 = nullptr;   // [N][K]
    int32_t* y32 = nullptr; // [T][N]
    __half* xh = nullptr;
    __half* wh = nullptr;
    __half* yh = nullptr;
    uint8_t* x4 = nullptr;  // [T][K/2]
    uint8_t* w4 = nullptr;  // [N][K/2]
    // cublasLt COL32 / COL4_4R2_8C copies
    int8_t* xc = nullptr;
    int8_t* wc = nullptr;
    int32_t* yc = nullptr;
};

struct Variant {
    std::string name;
    double ops_mult = 1.0;          // 1 = 2*N*K*T
    std::function<bool()> setup;    // returns false if unsupported
    std::function<bool()> run;
};

__global__ void fill_i8(int8_t* p, size_t n, uint32_t seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)i * 2654435761u ^ seed;
        h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
        p[i] = (int8_t)((int)(h & 0xff) - 128);
    }
}
__global__ void fill_h(__half* p, size_t n, uint32_t seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)i * 2654435761u ^ seed;
        h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
        p[i] = __float2half(((int)(h & 0xff) - 128) / 128.f);
    }
}

int main(int argc, char** argv) {
    int T = 2048, N = 17408, K = 5120;
    double sustain = 0;
    std::string vlist = "blas_i8,lt_i8_col32,blas_f16";
#ifdef T4Q_CUTLASS
    vlist += ",cut_i8_128x256,cut_i8_128x128,cut_i4_128x256";
#endif
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--dev") g_dev = atoi(argv[++i]);
        else if (a == "--sustain") sustain = atof(argv[++i]);
        else if (a == "--variants") vlist = argv[++i];
        else if (a == "--T") T = atoi(argv[++i]);
        else if (a == "--N") N = atoi(argv[++i]);
        else if (a == "--K") K = atoi(argv[++i]);
    }
    CK(cudaSetDevice(g_dev));
    g_nvml.init();
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    Bufs B;
    CK(cudaMalloc(&B.x8, (size_t)T * K));
    CK(cudaMalloc(&B.w8, (size_t)N * K));
    CK(cudaMalloc(&B.y32, (size_t)T * N * 4));
    fill_i8<<<1024, 256>>>(B.x8, (size_t)T * K, 1);
    fill_i8<<<1024, 256>>>(B.w8, (size_t)N * K, 2);
    CK(cudaDeviceSynchronize());

    cublasHandle_t hb;
    cublasLtHandle_t hl;
    if (cublasCreate(&hb) || cublasLtCreate(&hl)) { printf("FATAL cublas init\n"); return 2; }
    cublasSetStream(hb, st);
    void* lt_ws = nullptr;
    const size_t lt_ws_bytes = 32u << 20;
    CK(cudaMalloc(&lt_ws, lt_ws_bytes));

    std::vector<Variant> V;
    // ---- cuBLAS GemmEx int8 TN: C[N x T] (col-major) = W^T(op T of [K x N] col-major) * X([K x T] col-major)
    V.push_back({"blas_i8", 1.0, [] { return true; }, [&]() -> bool {
        const int32_t al = 1, be = 0;
        CB(cublasGemmEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &al, B.w8, CUDA_R_8I, K, B.x8, CUDA_R_8I, K, &be, B.y32,
                        CUDA_R_32I, N, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        return true;
    }});
    // ---- cublasLt IMMA in the Turing-native orders: A = W in COL32 ([N rows] x K), B = X in COL4_4R2_8C, C COL32
    cublasLtMatmulDesc_t lt_desc = nullptr;
    cublasLtMatrixLayout_t lA = nullptr, lB = nullptr, lC = nullptr;
    cublasLtMatmulAlgo_t lt_algo;
    bool lt_algo_ok = false;
    V.push_back({"lt_i8_col32", 1.0, [&]() -> bool {
        CK(cudaMalloc(&B.wc, (size_t)N * K));
        CK(cudaMalloc(&B.xc, (size_t)((T + 7) / 8 * 8) * ((K + 31) / 32 * 32)));
        CK(cudaMalloc(&B.yc, (size_t)N * T * 4));
        CK(cudaMemset(B.wc, 1, (size_t)N * K));
        CK(cudaMemset(B.xc, 1, (size_t)((T + 7) / 8 * 8) * ((K + 31) / 32 * 32)));
        // D[m=N][n=T] = A[m=N][k] * B[n=T][k]^T  (B transposed, as cublasLt IMMA requires)
        CB(cublasLtMatmulDescCreate(&lt_desc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
        cublasOperation_t opT = CUBLAS_OP_T;
        CB(cublasLtMatmulDescSetAttribute(lt_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT)));
        cublasLtOrder_t o32 = CUBLASLT_ORDER_COL32, o4 = CUBLASLT_ORDER_COL4_4R2_8C;
        const int64_t ldA = 32LL * N, ldB = 32LL * ((T + 7) / 8 * 8), ldC = 32LL * N;
        CB(cublasLtMatrixLayoutCreate(&lA, CUDA_R_8I, N, K, ldA));
        CB(cublasLtMatrixLayoutSetAttribute(lA, CUBLASLT_MATRIX_LAYOUT_ORDER, &o32, sizeof(o32)));
        CB(cublasLtMatrixLayoutCreate(&lB, CUDA_R_8I, T, K, ldB));
        CB(cublasLtMatrixLayoutSetAttribute(lB, CUBLASLT_MATRIX_LAYOUT_ORDER, &o4, sizeof(o4)));
        CB(cublasLtMatrixLayoutCreate(&lC, CUDA_R_32I, N, T, ldC));
        CB(cublasLtMatrixLayoutSetAttribute(lC, CUBLASLT_MATRIX_LAYOUT_ORDER, &o32, sizeof(o32)));
        cublasLtMatmulPreference_t pref;
        CB(cublasLtMatmulPreferenceCreate(&pref));
        CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &lt_ws_bytes,
                                                sizeof(lt_ws_bytes)));
        cublasLtMatmulHeuristicResult_t res[4];
        int nres = 0;
        CB(cublasLtMatmulAlgoGetHeuristic(hl, lt_desc, lA, lB, lC, lC, pref, 4, res, &nres));
        cublasLtMatmulPreferenceDestroy(pref);
        if (nres <= 0) { printf("ERR lt_i8_col32: no heuristic result\n"); return false; }
        lt_algo = res[0].algo;
        lt_algo_ok = true;
        return true;
    }, [&]() -> bool {
        const int32_t al = 1, be = 0;
        CB(cublasLtMatmul(hl, lt_desc, &al, B.wc, lA, B.xc, lB, &be, B.yc, lC, B.yc, lC, lt_algo_ok ? &lt_algo : nullptr,
                          lt_ws, lt_ws_bytes, st));
        return true;
    }});
    // ---- cuBLAS fp16 tensor-op GEMM (fp32 accumulate, fp16 out)
    V.push_back({"blas_f16", 1.0, [&]() -> bool {
        CK(cudaMalloc(&B.xh, (size_t)T * K * 2));
        CK(cudaMalloc(&B.wh, (size_t)N * K * 2));
        CK(cudaMalloc(&B.yh, (size_t)T * N * 2));
        fill_h<<<1024, 256>>>(B.xh, (size_t)T * K, 3);
        fill_h<<<1024, 256>>>(B.wh, (size_t)N * K, 4);
        CK(cudaDeviceSynchronize());
        return true;
    }, [&]() -> bool {
        const float al = 1.f, be = 0.f;
        CB(cublasGemmEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &al, B.wh, CUDA_R_16F, K, B.xh, CUDA_R_16F, K, &be, B.yh,
                        CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        return true;
    }});
#ifdef T4Q_CUTLASS
    using EpiI32 = cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>;
    using CutI8a = cutlass::gemm::device::Gemm<int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
                                               int32_t, cutlass::layout::RowMajor, int32_t, cutlass::arch::OpClassTensorOp,
                                               cutlass::arch::Sm75, cutlass::gemm::GemmShape<128, 256, 64>,
                                               cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<8, 8, 16>,
                                               EpiI32, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;
    using CutI8b = cutlass::gemm::device::Gemm<int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
                                               int32_t, cutlass::layout::RowMajor, int32_t, cutlass::arch::OpClassTensorOp,
                                               cutlass::arch::Sm75, cutlass::gemm::GemmShape<128, 128, 64>,
                                               cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<8, 8, 16>,
                                               EpiI32, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;
    using CutI4 = cutlass::gemm::device::Gemm<cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::int4b_t,
                                              cutlass::layout::ColumnMajor, int32_t, cutlass::layout::RowMajor, int32_t,
                                              cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
                                              cutlass::gemm::GemmShape<128, 256, 128>, cutlass::gemm::GemmShape<64, 64, 128>,
                                              cutlass::gemm::GemmShape<8, 8, 32>, EpiI32,
                                              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;
    static CutI8a g_i8a;
    static CutI8b g_i8b;
    static CutI4 g_i4;
    // M = T tokens (rows of X), N = output features, K
    V.push_back({"cut_i8_128x256", 1.0, [&]() -> bool {
        typename CutI8a::Arguments args({T, N, K}, {B.x8, K}, {B.w8, K}, {B.y32, N}, {B.y32, N}, {1, 0});
        if (g_i8a.can_implement(args) != cutlass::Status::kSuccess) { printf("ERR cut_i8a can_implement\n"); return false; }
        return g_i8a.initialize(args, nullptr, st) == cutlass::Status::kSuccess;
    }, [&]() -> bool { return g_i8a.run(st) == cutlass::Status::kSuccess; }});
    V.push_back({"cut_i8_128x128", 1.0, [&]() -> bool {
        typename CutI8b::Arguments args({T, N, K}, {B.x8, K}, {B.w8, K}, {B.y32, N}, {B.y32, N}, {1, 0});
        if (g_i8b.can_implement(args) != cutlass::Status::kSuccess) { printf("ERR cut_i8b can_implement\n"); return false; }
        return g_i8b.initialize(args, nullptr, st) == cutlass::Status::kSuccess;
    }, [&]() -> bool { return g_i8b.run(st) == cutlass::Status::kSuccess; }});
    V.push_back({"cut_i4_128x256", 1.0, [&]() -> bool {
        CK(cudaMalloc(&B.x4, (size_t)T * K / 2));
        CK(cudaMalloc(&B.w4, (size_t)N * K / 2));
        fill_i8<<<1024, 256>>>((int8_t*)B.x4, (size_t)T * K / 2, 5);
        fill_i8<<<1024, 256>>>((int8_t*)B.w4, (size_t)N * K / 2, 6);
        CK(cudaDeviceSynchronize());
        typename CutI4::Arguments args({T, N, K}, {(cutlass::int4b_t*)B.x4, K}, {(cutlass::int4b_t*)B.w4, K}, {B.y32, N},
                                       {B.y32, N}, {1, 0});
        if (g_i4.can_implement(args) != cutlass::Status::kSuccess) { printf("ERR cut_i4 can_implement\n"); return false; }
        return g_i4.initialize(args, nullptr, st) == cutlass::Status::kSuccess;
    }, [&]() -> bool { return g_i4.run(st) == cutlass::Status::kSuccess; }});
#endif

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const double ops = 2.0 * N * K * T;
    std::vector<std::string> want;
    for (size_t p = 0; p < vlist.size();) {
        size_t q = vlist.find(',', p);
        if (q == std::string::npos) q = vlist.size();
        want.push_back(vlist.substr(p, q - p));
        p = q + 1;
    }
    std::vector<Variant*> ok;
    for (auto& w : want)
        for (auto& v : V)
            if (v.name == w) {
                if (!v.setup()) { printf("B {\"variant\":\"%s\",\"ok\":0}\n", v.name.c_str()); fflush(stdout); break; }
                // burst: warm 2, then 5 timed
                bool good = v.run() && v.run();
                CK(cudaStreamSynchronize(st));
                if (!good || cudaGetLastError() != cudaSuccess) { printf("B {\"variant\":\"%s\",\"ok\":0}\n", v.name.c_str()); break; }
                CK(cudaEventRecord(e0, st));
                for (int r = 0; r < 5; ++r) v.run();
                CK(cudaEventRecord(e1, st));
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                auto smp = g_nvml.sample(g_dev);
                CK(cudaEventSynchronize(e1));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                const double tops = ops * v.ops_mult * 5 / (ms * 1e-3) / 1e12;
                printf("B {\"dev\":%d,\"variant\":\"%s\",\"ok\":1,\"T\":%d,\"N\":%d,\"K\":%d,\"ms\":%.3f,\"TOPS\":%.2f,"
                       "\"sm_mhz\":%u,\"power_w\":%.1f,\"ops_per_clk_sm\":%.0f}\n",
                       g_dev, v.name.c_str(), T, N, K, ms / 5, tops, smp.sm, smp.mw / 1000.0,
                       smp.sm ? tops * 1e12 / (smp.sm * 1e6) / 40.0 : 0.0);
                fflush(stdout);
                ok.push_back(&v);
                break;
            }
    if (sustain > 0) {
        for (Variant* v : ok) {
            auto t0 = std::chrono::steady_clock::now();
            int win = 0;
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < sustain) {
                CK(cudaEventRecord(e0, st));
                const int R = 3;
                for (int r = 0; r < R; ++r) v->run();
                CK(cudaEventRecord(e1, st));
                std::this_thread::sleep_for(std::chrono::milliseconds(15));
                auto smp = g_nvml.sample(g_dev);
                CK(cudaEventSynchronize(e1));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                const double tops = ops * v->ops_mult * R / (ms * 1e-3) / 1e12;
                char rs[128];
                nvml_reason_str(smp.reasons, rs, sizeof rs);
                printf("S {\"dev\":%d,\"variant\":\"%s\",\"win\":%d,\"t\":%.2f,\"TOPS\":%.2f,\"sm_mhz\":%u,\"power_w\":%.1f,"
                       "\"temp\":%u,\"reasons\":\"%s\"}\n",
                       g_dev, v->name.c_str(), win++,
                       std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), tops, smp.sm,
                       smp.mw / 1000.0, smp.temp, rs);
                fflush(stdout);
            }
        }
    }
    printf("DONE\n");
    return 0;
}
