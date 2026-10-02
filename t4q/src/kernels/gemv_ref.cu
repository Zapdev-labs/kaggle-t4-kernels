// M1 reference GEMV: one warp per output row, each lane dequantizes 32-weight groups (deq32) and dots them with
// fp32 activations. Correctness first; the fast dp4a kernels live in gemv.cuh (M2).
#include "deq.cuh"
#include "kernels.h"

template <int FMT>
__global__ void __launch_bounds__(256) k_gemv(PackedW W, const float* __restrict__ x, float* __restrict__ y) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t row = (int64_t)blockIdx.x * 8 + warp;
    if (row >= W.rows) return;
    const int64_t ng = W.cols / 32;
    float acc = 0.f;
    for (int64_t g = lane; g < ng; g += 32) {
        float w[32];
        deq32<FMT>(W, row, g, w);
        const float4* xv = (const float4*)(x + g * 32);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float4 a = __ldg(xv + j);
            acc += w[4 * j] * a.x + w[4 * j + 1] * a.y + w[4 * j + 2] * a.z + w[4 * j + 3] * a.w;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = acc;
}

void launch_gemv(const PackedW& W, const float* x, float* y, cudaStream_t s) {
    const unsigned G = (unsigned)((W.rows + 7) / 8);
    switch (W.fmt) {
        case FMT_F32: k_gemv<FMT_F32><<<G, 256, 0, s>>>(W, x, y); break;
        case FMT_P4: k_gemv<FMT_P4><<<G, 256, 0, s>>>(W, x, y); break;
        case FMT_P4M: k_gemv<FMT_P4M><<<G, 256, 0, s>>>(W, x, y); break;
        case FMT_Q8: k_gemv<FMT_Q8><<<G, 256, 0, s>>>(W, x, y); break;
        case FMT_K5: k_gemv<FMT_K5><<<G, 256, 0, s>>>(W, x, y); break;
        case FMT_K6: k_gemv<FMT_K6><<<G, 256, 0, s>>>(W, x, y); break;
    }
}
