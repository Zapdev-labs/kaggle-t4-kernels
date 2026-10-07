// cf-m6 r3 (CF_REQUANT.md section 5): the iq1_s expert-slab packer, host-only (g++).
// Reads the BF16 expert tensors (the HF originals range-read to local files by the
// Kaggle driver: mlp.experts.gate_up_proj [ne, 1280, 2560] + mlp.experts.down_proj
// [ne, 2560, 640], BF16, expert-major row-major - pinned from the shard-2 header:
// [512, 1280, 2560], offsets [0, 3355443200], the 160-byte safetensors header),
// quantizes every row with the t4q_iq1s packer core (the exact ggml
// quantize_row_iq1_s_impl port, self-scaled weights), and writes ONE slab file per
// layer in THIS REPO'S OWN layout (not GGUF): a fixed 72-byte header + the 6 PackedW
// planes, expert-major rows preserved so a GPU's expert half is one contiguous byte
// range (the stage-2 TP requirement).
//   cf_requant_pack --gu F --dn F --out SLAB [--ne N] [--threads T]     the pack
//   cf_requant_pack --synthetic --out SLAB [--ne N] [--seed S]          the local gate
//   cf_requant_pack --verify SLAB [--rows N] [--verify-gu F --verify-dn F]
// Plane math (FMT_IQ1S, 50 B / 256 elems): at ne=512, gu -> codes 209.7 MB + hi
// 104.9 MB + d 13.1 MB; dn -> 104.9 + 52.4 + 6.6 MB; ~491.6 MB per layer, 23.6 GB total.
// Compile: g++ -O3 -fopenmp -std=c++17 -ffp-contract=off.
#define T4Q_HOST_SIM 1
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../src/requant.h"
#include "../src/packed.h"
#include "../src/kernels/deq.cuh"

float fp16_to_fp32(uint16_t h) { return t4q_fp16_to_fp32(h); }  // the deq.cuh host-sim shim

// ---- the slab header (frozen v2; all offsets from the file start). v1 pre-dated the
// dn tiling find (the r3 gate): the dn's 640-wide rows cannot tile the 256-elem iq1_s
// block, so the dn planes use the FMT_IQ1SH 128-elem half block - same lattice, same
// per-group arithmetic - and carry their own format field. ----
struct SlabHdr {
    uint32_t magic;      // 0x45513454 'T4QE'
    uint32_t version;    // 2
    uint32_t fmt;        // the gu planes' format: FMT_IQ1S (256-elem blocks)
    uint32_t dn_fmt;     // the dn planes' format: FMT_IQ1SH (128-elem half blocks)
    uint32_t ne;         // the expert count (512 on the real pack; small in the gate)
    uint32_t gu_rows;    // ne * 1280 (the plane row counts)
    uint32_t gu_cols;    // 2560
    uint32_t dn_rows;    // ne * 2560
    uint32_t dn_cols;    // 640
    uint32_t pad0;       // reserved (keeps the u64 fields 8-aligned)
    uint64_t gu_codes, gu_hi, gu_d;  // the plane byte offsets
    uint64_t dn_codes, dn_hi, dn_d;
    uint64_t file_bytes;
};
static_assert(sizeof(SlabHdr) == 96, "slab header v2 (10 u32 + 7 u64, no padding)");
#define SLAB_MAGIC 0x45513454u

// the plane byte sizes: gu at the 256-elem iq1_s block (50 B/256), dn at the 128-elem
// half block (26 B/128) - the tiling the 640-wide dn rows demand
static uint64_t plane_codes(uint64_t rows, uint64_t cols) { return rows * (cols / 256) * 32; }
static uint64_t plane_hi(uint64_t rows, uint64_t cols) { return rows * (cols / 256) * 16; }
static uint64_t plane_d(uint64_t rows, uint64_t cols) { return rows * (cols / 256) * 2; }
static uint64_t plane_codes_h(uint64_t rows, uint64_t cols) { return rows * (cols / 128) * 16; }
static uint64_t plane_hi_h(uint64_t rows, uint64_t cols) { return rows * (cols / 128) * 8; }
static uint64_t plane_d_h(uint64_t rows, uint64_t cols) { return rows * (cols / 128) * 2; }

// ---- bf16 <-> fp32 (the exact top-half widen; the synthetic narrows by truncation so
// the synthetic round-trip through the file is exact) ----
static inline float bf16_to_f32(uint16_t v) {
    uint32_t b = (uint32_t)v << 16;
    float f;
    memcpy(&f, &b, 4);
    return f;
}
static inline uint16_t f32_to_bf16_trunc(float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    return (uint16_t)(b >> 16);
}

// the one-time tables (the init is ~5-15 s; the quantizer threads read them read-only)
static const t4q_iq1s::Tables& the_tables() {
    static t4q_iq1s::Tables T;
    static bool init = false;
    if (!init) {
        t4q_iq1s::init(T);
        init = true;
    }
    return T;
}

// ---- the synthetic source (the gate's input side; deterministic) ----
static uint32_t lcg_state = 12345;
static uint32_t lcg() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return lcg_state >> 8;
}
static void synth_row(float* x, int cols, uint64_t r) {
    const int cls = (int)(r & 7);
    for (int i = 0; i < cols; i++) {
        float u = (float)((int)(lcg() >> 8) - 32768) / 32768.f;  // [-1, 1)
        if (cls < 5) x[i] = u * (0.02f + 0.15f * (r % 97) / 97.f);
        else if (cls == 5) x[i] = 0.f;          // the eps path
        else if (cls == 6) x[i] = -fabsf(u) * 0.08f;
        else x[i] = 0.037f;                     // constant rows
    }
}

// ---- quantize one [rows x cols] bf16 source into the three slab planes ----
// NG = the groups (32-elem) per block: 8 for the gu (256-elem blocks, 32/16/2 B planes),
// 4 for the dn (128-elem half blocks, 16/8/2 B planes - the 640 tiling).
template <int NG>
static void quantize_tensor(const uint16_t* src, uint64_t rows, uint64_t cols, uint8_t* codes, uint8_t* hi,
                            uint16_t* d, int nthreads) {
    const t4q_iq1s::Tables& T = the_tables();
    const uint64_t bpr = cols / (32 * NG);
    std::vector<t4q_iq1s::BlockT<NG>> blocks(rows * bpr);
#pragma omp parallel for num_threads(nthreads) schedule(dynamic, 64)
    for (int64_t r = 0; r < (int64_t)rows; ++r) {
        std::vector<float> x(cols);
        for (uint64_t i = 0; i < cols; i++) x[i] = bf16_to_f32(src[r * cols + i]);
        t4q_iq1s::quant_row_t<NG>(T, x.data(), (int)cols, blocks.data() + r * bpr);
    }
    for (uint64_t b = 0; b < rows * bpr; b++) {  // the plane split
        memcpy(codes + b * (4 * NG), blocks[b].qs, 4 * NG);
        memcpy(hi + b * (2 * NG), blocks[b].qh, 2 * NG);
        d[b] = blocks[b].d;
    }
}

// ---- the verify: read the slab back, deq32 (the kernel's decode) vs the reference ----
// sample_rows <= 0 or >= rows: every row; else a deterministic sample. When the bf16
// sources are given, the deq32 decode is also RMSE-checked against them (the diag).
static int verify_slab(const std::string& path, int sample_rows, const std::string& gu_src,
                       const std::string& dn_src) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path.c_str()); return 1; }
    SlabHdr h;
    if (fread(&h, sizeof(h), 1, f) != 1 || h.magic != SLAB_MAGIC || h.version != 2 ||
        h.fmt != (uint32_t)FMT_IQ1S || h.dn_fmt != (uint32_t)FMT_IQ1SH || h.gu_cols != 2560 ||
        h.dn_cols != 640 || h.gu_rows != (uint64_t)h.ne * 1280 || h.dn_rows != (uint64_t)h.ne * 2560) {
        fprintf(stderr, "bad slab header\n");
        return 1;
    }
    const uint64_t gc = plane_codes(h.gu_rows, h.gu_cols), gh = plane_hi(h.gu_rows, h.gu_cols),
                    gd = plane_d(h.gu_rows, h.gu_cols), dc = plane_codes_h(h.dn_rows, h.dn_cols),
                    dh = plane_hi_h(h.dn_rows, h.dn_cols), dd = plane_d_h(h.dn_rows, h.dn_cols);
    if (h.gu_codes != sizeof(SlabHdr) || h.gu_hi != h.gu_codes + gc || h.gu_d != h.gu_hi + gh ||
        h.dn_codes != h.gu_d + gd || h.dn_hi != h.dn_codes + dc || h.dn_d != h.dn_hi + dh ||
        h.file_bytes != h.dn_d + dd) {
        fprintf(stderr, "bad slab plane offsets\n");
        return 1;
    }
    struct Plane { uint64_t rows, cols; uint64_t codes_off, hi_off, d_off; const char* name; int ng; };
    const Plane pl[2] = {{h.gu_rows, h.gu_cols, h.gu_codes, h.gu_hi, h.gu_d, "gu", 8},
                         {h.dn_rows, h.dn_cols, h.dn_codes, h.dn_hi, h.dn_d, "dn", 4}};
    const t4q_iq1s::Tables& T = the_tables();
    // mmap the bf16 sources when given (the RMSE diag)
    const uint16_t* src[2] = {nullptr, nullptr};
    void* src_map[2] = {nullptr, nullptr};
    size_t src_len[2] = {0, 0};
    const std::string src_path[2] = {gu_src, dn_src};
    for (int p = 0; p < 2; p++) {
        if (src_path[p].empty()) continue;
        struct stat st;
        if (stat(src_path[p].c_str(), &st) || (uint64_t)st.st_size != pl[p].rows * pl[p].cols * 2) {
            fprintf(stderr, "source size mismatch %s\n", src_path[p].c_str());
            return 1;
        }
        int fd = open(src_path[p].c_str(), O_RDONLY);
        if (fd < 0) { fprintf(stderr, "source open failed %s\n", src_path[p].c_str()); return 1; }
        src_len[p] = (size_t)st.st_size;
        src_map[p] = mmap(nullptr, src_len[p], PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (src_map[p] == MAP_FAILED) { fprintf(stderr, "source mmap failed\n"); return 1; }
        src[p] = (const uint16_t*)src_map[p];
    }
    int bad = 0;
    long nchecked = 0;
    double sq = 0, sq0 = 0;
    long long nelem = 0;
    for (int p = 0; p < 2; p++) {
        const Plane& P = pl[p];
        const int nsample = (sample_rows <= 0 || (uint64_t)sample_rows >= P.rows) ? (int)P.rows : sample_rows;
        const int NG = P.ng;  // gu: the 256-elem block (FMT_IQ1S); dn: the 128-elem half (FMT_IQ1SH)
        const int EFMT = (NG == 8) ? FMT_IQ1S : FMT_IQ1SH;  // the runtime W.fmt value
        const uint64_t stride_c = 4 * NG, stride_h = 2 * NG;
        const uint64_t nblocks = P.cols / (32 * NG);
        std::vector<uint8_t> codes(P.rows * nblocks * stride_c), hi(P.rows * nblocks * stride_h);
        std::vector<uint16_t> d(P.rows * nblocks);
        fseeko(f, (off_t)P.codes_off, SEEK_SET);
        if (fread(codes.data(), 1, codes.size(), f) != codes.size() ||
            fread(hi.data(), 1, hi.size(), f) != hi.size() ||
            fread(d.data(), 2, d.size(), f) != d.size()) {
            fprintf(stderr, "short plane read %s\n", P.name);
            return 1;
        }
        PackedW W;
        W.fmt = (PackFmt)EFMT;
        W.rows = 1;
        W.cols = (int64_t)P.cols;
        W.codes = codes.data();
        W.hi = hi.data();
        W.d = d.data();
        std::vector<float> a(P.cols), bref(P.cols);
        std::vector<t4q_iq1s::BlockT<4>> blocks4(nblocks);
        std::vector<t4q_iq1s::BlockT<8>> blocks8(nblocks);
        for (int s = 0; s < nsample; s++) {
            const uint64_t r = (uint64_t)lcg() % P.rows;
            if (NG == 8) {
                for (uint64_t bl = 0; bl < nblocks; bl++) {
                    memcpy(blocks8[bl].qs, &codes[(r * nblocks + bl) * stride_c], stride_c);
                    memcpy(blocks8[bl].qh, &hi[(r * nblocks + bl) * stride_h], stride_h);
                    blocks8[bl].d = d[r * nblocks + bl];
                }
            } else {
                for (uint64_t bl = 0; bl < nblocks; bl++) {
                    memcpy(blocks4[bl].qs, &codes[(r * nblocks + bl) * stride_c], stride_c);
                    memcpy(blocks4[bl].qh, &hi[(r * nblocks + bl) * stride_h], stride_h);
                    blocks4[bl].d = d[r * nblocks + bl];
                }
            }
            for (uint64_t g = 0; g < P.cols / 32; g++) {  // the template arg must be constant
                if (NG == 8) deq32<FMT_IQ1S>(W, (int64_t)r, (int64_t)g, a.data() + g * 32);
                else deq32<FMT_IQ1SH>(W, (int64_t)r, (int64_t)g, a.data() + g * 32);
            }
            if (NG == 8) t4q_iq1s::dequant_row_ref(T, blocks8.data(), bref.data(), (int)P.cols);
            else t4q_iq1s::dequant_row_ref_t<4>(T, blocks4.data(), bref.data(), (int)P.cols);
            if (memcmp(a.data(), bref.data(), P.cols * 4)) {
                bad++;
                if (bad < 4) printf("%s row %llu MISMATCH\n", P.name, (unsigned long long)r);
            }
            nchecked++;
            if (src[p]) {
                for (uint64_t i = 0; i < P.cols; i++) {
                    float xs = bf16_to_f32(src[p][r * P.cols + i]);
                    sq += (double)(a[i] - xs) * (a[i] - xs);
                    sq0 += (double)xs * xs;
                }
                nelem += (long long)P.cols;
            }
        }
    }
    printf("verify %s: rows checked %ld, %s", path.c_str(), nchecked, bad ? "MISMATCH" : "OK");
    if (nelem > 0) printf(", src rmse=%.6f rel=%.4f", sqrt(sq / nelem), sqrt(sq / (sq0 + 1e-30)));
    printf("\n");
    for (int p = 0; p < 2; p++)
        if (src_map[p]) munmap(src_map[p], src_len[p]);
    fclose(f);
    return bad != 0;
}

static void die(const char* m) { fprintf(stderr, "%s\n", m); exit(2); }

int main(int argc, char** argv) {
    std::string gu, dn, out, vgu, vdn;
    int ne = 512, threads = 4, sample = 64;
    bool synthetic = false, do_verify = false;
    uint32_t seed = 12345;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--gu") gu = next();
        else if (a == "--dn") dn = next();
        else if (a == "--out") out = next();
        else if (a == "--ne") ne = atoi(next());
        else if (a == "--threads") threads = atoi(next());
        else if (a == "--seed") seed = (uint32_t)atoi(next());
        else if (a == "--synthetic") synthetic = true;
        else if (a == "--verify") { do_verify = true; out = next(); }
        else if (a == "--rows") sample = atoi(next());
        else if (a == "--verify-gu") vgu = next();
        else if (a == "--verify-dn") vdn = next();
        else die("usage: cf_requant_pack --gu F --dn F --out F [--ne N --threads T] |\n"
                 "                    --synthetic --out F [--ne N --seed S] |\n"
                 "                    --verify F [--rows N --verify-gu F --verify-dn F]");
    }
    if (out.empty()) die("--out (or --verify) required");

    if (synthetic) {  // generate the bf16 sources deterministically
        lcg_state = seed;
        gu = out + ".gu.bf16";
        dn = out + ".dn.bf16";
        const uint64_t gr = (uint64_t)ne * 1280, dr = (uint64_t)ne * 2560;
        FILE* fg = fopen(gu.c_str(), "wb");
        FILE* fd = fopen(dn.c_str(), "wb");
        if (!fg || !fd) die("synthetic source open failed");
        std::vector<float> x(2560);
        std::vector<uint16_t> bf(2560);
        for (uint64_t r = 0; r < gr; r++) {
            synth_row(x.data(), 2560, r);
            for (int i = 0; i < 2560; i++) bf[i] = f32_to_bf16_trunc(x[i]);
            fwrite(bf.data(), 2, 2560, fg);
        }
        x.resize(640);
        bf.resize(640);
        for (uint64_t r = 0; r < dr; r++) {
            synth_row(x.data(), 640, r);
            for (int i = 0; i < 640; i++) bf[i] = f32_to_bf16_trunc(x[i]);
            fwrite(bf.data(), 2, 640, fd);
        }
        fclose(fg);
        fclose(fd);
    }

    if (!gu.empty() && !dn.empty()) {  // the pack path (the sources exist)
        struct stat st;
        const uint64_t gr = (uint64_t)ne * 1280, dr = (uint64_t)ne * 2560;
        if (stat(gu.c_str(), &st) || (uint64_t)st.st_size != gr * 2560 * 2)
            die("gu source size mismatch (expected [ne,1280,2560] bf16)");
        if (stat(dn.c_str(), &st) || (uint64_t)st.st_size != dr * 640 * 2)
            die("dn source size mismatch (expected [ne,2560,640] bf16)");
        int fgu = open(gu.c_str(), O_RDONLY), fdn = open(dn.c_str(), O_RDONLY);
        if (fgu < 0 || fdn < 0) die("source open failed");
        const size_t glen = gr * 2560 * 2, dlen = dr * 640 * 2;
        void* gm = mmap(nullptr, glen, PROT_READ, MAP_PRIVATE, fgu, 0);
        void* dm = mmap(nullptr, dlen, PROT_READ, MAP_PRIVATE, fdn, 0);
        if (gm == MAP_FAILED || dm == MAP_FAILED) die("source mmap failed");

        const uint64_t gc = plane_codes(gr, 2560), gh = plane_hi(gr, 2560), gd = plane_d(gr, 2560);
        const uint64_t dc = plane_codes_h(dr, 640), dh = plane_hi_h(dr, 640), dd = plane_d_h(dr, 640);
        SlabHdr h = {};
        h.magic = SLAB_MAGIC;
        h.version = 2;
        h.fmt = FMT_IQ1S;
        h.dn_fmt = FMT_IQ1SH;  // the dn's 128-elem half block (the 640 tiling)
        h.ne = (uint32_t)ne;
        h.gu_rows = (uint32_t)gr;
        h.gu_cols = 2560;
        h.dn_rows = (uint32_t)dr;
        h.dn_cols = 640;
        h.gu_codes = sizeof(SlabHdr);
        h.gu_hi = h.gu_codes + gc;
        h.gu_d = h.gu_hi + gh;
        h.dn_codes = h.gu_d + gd;
        h.dn_hi = h.dn_codes + dc;
        h.dn_d = h.dn_hi + dh;
        h.file_bytes = h.dn_d + dd;

        FILE* fo = fopen(out.c_str(), "wb");
        if (!fo) die("slab open failed");
        fwrite(&h, sizeof(h), 1, fo);
        std::vector<uint8_t> vgc(gc), vgh(gh), vdc(dc), vdh(dh);
        std::vector<uint16_t> vgd(gd / 2), vdd(dd / 2);
        quantize_tensor<8>((const uint16_t*)gm, gr, 2560, vgc.data(), vgh.data(), vgd.data(), threads);
        quantize_tensor<4>((const uint16_t*)dm, dr, 640, vdc.data(), vdh.data(), vdd.data(), threads);
        fwrite(vgc.data(), 1, gc, fo);
        fwrite(vgh.data(), 1, gh, fo);
        fwrite(vgd.data(), 2, vgd.size(), fo);
        fwrite(vdc.data(), 1, dc, fo);
        fwrite(vdh.data(), 1, dh, fo);
        fwrite(vdd.data(), 2, vdd.size(), fo);
        fclose(fo);
        munmap(gm, glen);
        munmap(dm, dlen);
        close(fgu);
        close(fdn);
        printf("packed %s: ne=%d, %llu bytes (gu %.1f MB, dn %.1f MB)\n", out.c_str(), ne,
               (unsigned long long)h.file_bytes, (gc + gh + gd) / 1048576.0, (dc + dh + dd) / 1048576.0);
    }

    if (synthetic) {  // the local gate: read back + the full deq32-vs-ref + the src RMSE
        int rc = verify_slab(out, 0, gu, dn);  // every row
        remove(gu.c_str());
        remove(dn.c_str());
        if (rc) return rc;
    } else if (do_verify) {
        return verify_slab(out, sample, vgu, vdn);
    }
    return 0;
}
