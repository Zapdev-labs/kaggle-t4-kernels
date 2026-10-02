// Host simulation of the GPU repack + deq32 path (same source as the kernels, compiled with T4Q_HOST_SIM).
// usage: test_deq_sim <gguf> <tensor> <nrows>   -> prints OK / MISMATCH
#define T4Q_HOST_SIM 1
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../src/gguf.h"
#include "../src/kernels/deq.cuh"
#include "../src/quant_cpu.h"

template <int FMT>
static void deq_row(const PackedW& W, int64_t r, float* out) {
    for (int64_t g = 0; g < W.cols / 32; g++) deq32<FMT>(W, r, g, out + g * 32);
}

int main(int argc, char** argv) {
    GgufFile f;
    std::string err;
    if (argc < 4 || !f.open(argv[1], err)) { fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    const GgufTensor* t = f.find(argv[2]);
    const int64_t nr = atoi(argv[3]), K = t->ne[0], n = nr * K;
    PackedW W;
    W.rows = nr; W.cols = K;
    std::vector<uint8_t> codes(n * 4), hi(n), meta(n), dd(n), mm(n);
    W.codes = codes.data(); W.hi = hi.data(); W.meta = meta.data(); W.d = (uint16_t*)dd.data(); W.m = (uint16_t*)mm.data();
    for (int64_t r = 0; r < nr; r++) {
        const uint8_t* row = t->data + r * t->row_bytes;
        switch (t->type) {
            case GT_Q4_0: W.fmt = FMT_P4; for (int64_t b = 0; b < K / 32; b++) repack_q4_block(W, row + b * 18, r * (K / 32) + b, 0); break;
            case GT_Q4_1: W.fmt = FMT_P4M; for (int64_t b = 0; b < K / 32; b++) repack_q4_block(W, row + b * 20, r * (K / 32) + b, 1); break;
            case GT_Q8_0: W.fmt = FMT_Q8; for (int64_t b = 0; b < K / 32; b++) repack_q8_block(W, row + b * 34, r * (K / 32) + b); break;
            case GT_Q5_K: W.fmt = FMT_K5; for (int64_t b = 0; b < K / 256; b++) repack_q5k_block(W, row + b * 176, r * (K / 256) + b); break;
            case GT_Q6_K: W.fmt = FMT_K6; for (int64_t b = 0; b < K / 256; b++) repack_q6k_block(W, row + b * 210, r * (K / 256) + b); break;
            case GT_F32: W.fmt = FMT_F32; memcpy(W.codes + r * K * 4, row, K * 4); break;
            default: fprintf(stderr, "unsupported\n"); return 1;
        }
    }
    std::vector<float> a(K), b(K);
    int bad = 0;
    for (int64_t r = 0; r < nr; r++) {
        switch (W.fmt) {
            case FMT_P4: deq_row<FMT_P4>(W, r, a.data()); break;
            case FMT_P4M: deq_row<FMT_P4M>(W, r, a.data()); break;
            case FMT_Q8: deq_row<FMT_Q8>(W, r, a.data()); break;
            case FMT_K5: deq_row<FMT_K5>(W, r, a.data()); break;
            case FMT_K6: deq_row<FMT_K6>(W, r, a.data()); break;
            case FMT_F32: deq_row<FMT_F32>(W, r, a.data()); break;
        }
        dequant_row_cpu(t->type, t->data + r * t->row_bytes, b.data(), K);
        if (memcmp(a.data(), b.data(), K * 4)) bad++;
    }
    printf("%s %s rows=%lld %s\n", argv[2], pack_fmt_name(W.fmt), (long long)nr, bad ? "MISMATCH" : "OK");
    return bad != 0;
}

const char* pack_fmt_name(int f) {
    static const char* n[] = {"F32", "P4", "P4M", "Q8", "K5", "K6"};
    return f >= 0 && f < 6 ? n[f] : "?";
}
