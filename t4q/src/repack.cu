// On-GPU lossless repack of GGUF blocks into the M1 planar formats, plus the dequant used by the round-trip check.
#include "gguf.h"
#include "kernels/deq.cuh"
#include "kernels/kernels.h"

const char* pack_fmt_name(int f) {
    switch (f) {
        case FMT_F32: return "F32"; case FMT_P4: return "P4"; case FMT_P4M: return "P4M";
        case FMT_Q8: return "Q8"; case FMT_K5: return "K5"; case FMT_K6: return "K6";
        case FMT_K2: return "K2"; case FMT_K4: return "K4"; case FMT_Q51: return "Q51";
        case FMT_IQ1S: return "IQ1S"; case FMT_IQ1SH: return "IQ1SH";
        default: return "?";
    }
}

// one thread per source block
__global__ void k_repack_q4(PackedW W, const uint8_t* raw, int64_t r0, int64_t nr, int bb, int has_m) {
    const int64_t nbr = W.cols / 32;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nr * nbr) return;
    const int64_t r = i / nbr, b = i % nbr;
    repack_q4_block(W, raw + r * nbr * bb + b * bb, (r0 + r) * nbr + b, has_m);
}

__global__ void k_repack_q8(PackedW W, const uint8_t* raw, int64_t r0, int64_t nr) {
    const int64_t nbr = W.cols / 32;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nr * nbr) return;
    const int64_t r = i / nbr, b = i % nbr;
    repack_q8_block(W, raw + r * nbr * 34 + b * 34, (r0 + r) * nbr + b);
}

__global__ void k_repack_q5k(PackedW W, const uint8_t* raw, int64_t r0, int64_t nr) {
    const int64_t nbr = W.cols / 256;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nr * nbr) return;
    const int64_t r = i / nbr, b = i % nbr;
    repack_q5k_block(W, raw + r * nbr * 176 + b * 176, (r0 + r) * nbr + b);
}

__global__ void k_repack_q6k(PackedW W, const uint8_t* raw, int64_t r0, int64_t nr) {
    const int64_t nbr = W.cols / 256;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nr * nbr) return;
    const int64_t r = i / nbr, b = i % nbr;
    repack_q6k_block(W, raw + r * nbr * 210 + b * 210, (r0 + r) * nbr + b);
}
__global__ void k_repack_q2k(PackedW W, const uint8_t* raw, int64_t r0, int64_t nr) {
    const int64_t nbr = W.cols / 256;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nr * nbr) return;
    const int64_t r = i / nbr, b = i % nbr;
    repack_q2k_block(W, raw + r * nbr * 84 + b * 84, (r0 + r) * nbr + b);
}
__global__ void k_repack_q4k(PackedW W, const uint8_t* raw, int64_t r0, int64_t nr) {
    const int64_t nbr = W.cols / 256;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nr * nbr) return;
    const int64_t r = i / nbr, b = i % nbr;
    repack_q4k_block(W, raw + r * nbr * 144 + b * 144, (r0 + r) * nbr + b);
}
__global__ void k_repack_q51(PackedW W, const uint8_t* raw, int64_t r0, int64_t nr) {
    const int64_t nbr = W.cols / 32;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nr * nbr) return;
    const int64_t r = i / nbr, b = i % nbr;
    repack_q51_block(W, raw + r * nbr * 24 + b * 24, (r0 + r) * nbr + b);
}

// cf-m3 (r19u) the UVA pointer-swap scatter repack: the per-pick sources are the picks'
// OWN slabs inside the REGISTERED mmap'd expert tensors, read through the device aliases
// (the coalesced page-span registration at load); the sources are exactly the bytes the
// OFF path's host memcpys stage (cf_engine's staging loop), the same repack_*_block
// decode, so the packed slabs are byte-identical by construction - only the read path
// changes (the mapped pages over PCIe instead of the pinned staging in VRAM).
// gu layout: pick-major, [gate rows EE | up rows EE] per pick; the eid[] maps pick->expert.
__global__ void k_repack_eid_q2k(PackedW W, const uint8_t* gate_alias, const uint8_t* up_alias,
                                 const int* eid, int64_t r0, int64_t nr, int64_t rows_per_pick,
                                 int64_t gate_rows) {
    const int64_t nbr = W.cols / 256;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nr * nbr) return;
    const int64_t r = i / nbr, b = i % nbr;
    const int k = (int)(r / rows_per_pick);
    const int64_t pr = r % rows_per_pick;
    const uint8_t* src =
        (pr < gate_rows)
            ? gate_alias + ((size_t)eid[k] * gate_rows + pr) * (nbr * 84) + b * 84
            : up_alias + ((size_t)eid[k] * gate_rows + (pr - gate_rows)) * (nbr * 84) + b * 84;
    repack_q2k_block(W, src, (r0 + r) * nbr + b);
}
// dn layout: pick-major, D rows of EE per pick (Q4_0: 18 B per 32-elem block)
__global__ void k_repack_eid_q4(PackedW W, const uint8_t* dn_alias, const int* eid,
                                int64_t r0, int64_t nr, int64_t rows_per_pick) {
    const int64_t nbr = W.cols / 32;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nr * nbr) return;
    const int64_t r = i / nbr, b = i % nbr;
    const int k = (int)(r / rows_per_pick);
    const int64_t pr = r % rows_per_pick;
    const uint8_t* src = dn_alias + ((size_t)eid[k] * rows_per_pick + pr) * (nbr * 18) + b * 18;
    repack_q4_block(W, src, (r0 + r) * nbr + b, 0);
}

void launch_repack(const PackedW& W, uint32_t type, const uint8_t* raw, int64_t r0, int64_t nr, cudaStream_t s) {
    const int T = 256;
    switch (type) {
        case GT_F32:
            cudaMemcpyAsync((float*)W.codes + r0 * W.cols, raw, nr * W.cols * 4, cudaMemcpyDeviceToDevice, s);
            break;
        case GT_Q4_0: { int64_t n = nr * (W.cols / 32); k_repack_q4<<<(unsigned)((n + T - 1) / T), T, 0, s>>>(W, raw, r0, nr, 18, 0); break; }
        case GT_Q4_1: { int64_t n = nr * (W.cols / 32); k_repack_q4<<<(unsigned)((n + T - 1) / T), T, 0, s>>>(W, raw, r0, nr, 20, 1); break; }
        case GT_Q8_0: { int64_t n = nr * (W.cols / 32); k_repack_q8<<<(unsigned)((n + T - 1) / T), T, 0, s>>>(W, raw, r0, nr); break; }
        case GT_Q5_K: { int64_t n = nr * (W.cols / 256); k_repack_q5k<<<(unsigned)((n + T - 1) / T), T, 0, s>>>(W, raw, r0, nr); break; }
        case GT_Q6_K: { int64_t n = nr * (W.cols / 256); k_repack_q6k<<<(unsigned)((n + T - 1) / T), T, 0, s>>>(W, raw, r0, nr); break; }
        case GT_Q2_K: { int64_t n = nr * (W.cols / 256); k_repack_q2k<<<(unsigned)((n + T - 1) / T), T, 0, s>>>(W, raw, r0, nr); break; }
        case GT_Q4_K: { int64_t n = nr * (W.cols / 256); k_repack_q4k<<<(unsigned)((n + T - 1) / T), T, 0, s>>>(W, raw, r0, nr); break; }
        case GT_Q5_1: { int64_t n = nr * (W.cols / 32); k_repack_q51<<<(unsigned)((n + T - 1) / T), T, 0, s>>>(W, raw, r0, nr); break; }
        default: break;
    }
}

// cf-m3 (r19u): the UVA pointer-swap scatter repacks - the picks' slabs read through the
// registered aliases + the per-step eid table; the packed targets + W-table views are
// the OFF path's identity ones, byte-identical by construction
void launch_repack_eid_q2k(const PackedW& W, const uint8_t* gate_alias, const uint8_t* up_alias,
                           const int* eid, int64_t r0, int64_t nr, int64_t rows_per_pick, int64_t gate_rows,
                           cudaStream_t s) {
    const int T = 256;
    const int64_t n = nr * (W.cols / 256);
    k_repack_eid_q2k<<<(unsigned)((n + T - 1) / T), T, 0, s>>>(W, gate_alias, up_alias, eid, r0, nr,
                                                              rows_per_pick, gate_rows);
}

void launch_repack_eid_q4(const PackedW& W, const uint8_t* dn_alias, const int* eid, int64_t r0, int64_t nr,
                          int64_t rows_per_pick, cudaStream_t s) {
    const int T = 256;
    const int64_t n = nr * (W.cols / 32);
    k_repack_eid_q4<<<(unsigned)((n + T - 1) / T), T, 0, s>>>(W, dn_alias, eid, r0, nr, rows_per_pick);
}

template <int FMT>
__global__ void k_dequant_rows(PackedW W, const int32_t* rows, int n, float* out) {
    const int64_t ng = W.cols / 32;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)n * ng) return;
    const int ri = (int)(i / ng);
    const int64_t g = i % ng;
    float w[32];
    deq32<FMT>(W, rows[ri], g, w);
    for (int j = 0; j < 32; j++) out[(int64_t)ri * W.cols + g * 32 + j] = w[j];
}

void launch_dequant_rows(const PackedW& W, const int32_t* rows, int n, float* out, cudaStream_t s) {
    const int T = 128;
    const int64_t tot = (int64_t)n * (W.cols / 32);
    const unsigned G = (unsigned)((tot + T - 1) / T);
    switch (W.fmt) {
        case FMT_F32: k_dequant_rows<FMT_F32><<<G, T, 0, s>>>(W, rows, n, out); break;
        case FMT_P4: k_dequant_rows<FMT_P4><<<G, T, 0, s>>>(W, rows, n, out); break;
        case FMT_P4M: k_dequant_rows<FMT_P4M><<<G, T, 0, s>>>(W, rows, n, out); break;
        case FMT_Q8: k_dequant_rows<FMT_Q8><<<G, T, 0, s>>>(W, rows, n, out); break;
        case FMT_K5: k_dequant_rows<FMT_K5><<<G, T, 0, s>>>(W, rows, n, out); break;
        case FMT_K6: k_dequant_rows<FMT_K6><<<G, T, 0, s>>>(W, rows, n, out); break;
        case FMT_K2: k_dequant_rows<FMT_K2><<<G, T, 0, s>>>(W, rows, n, out); break;
        case FMT_K4: k_dequant_rows<FMT_K4><<<G, T, 0, s>>>(W, rows, n, out); break;
        case FMT_Q51: k_dequant_rows<FMT_Q51><<<G, T, 0, s>>>(W, rows, n, out); break;
    }
}
