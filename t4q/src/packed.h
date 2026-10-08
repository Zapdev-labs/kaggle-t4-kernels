// On-device packed weight formats (M1: row-major planar SoA, lossless; see DESIGN.md section 3.3).
#pragma once
#include <cstddef>
#include <cstdint>

enum PackFmt : int {
    FMT_F32 = 0,   // float [rows][cols]
    FMT_P4 = 1,    // Q4_0: codes [rows][cols/2] (ggml nibble order per 32), d fp16 [rows][cols/32]
    FMT_P4M = 2,   // Q4_1: as P4 plus m fp16 [rows][cols/32]
    FMT_Q8 = 3,    // Q8_0: int8 codes [rows][cols], d fp16 [rows][cols/32]
    FMT_K5 = 4,    // Q5_K: qs [rows][cols/2] (128 B / 256), qh [rows][cols/8] (32 B / 256), meta 16 B / 256 (d, dmin, scales[12])
    FMT_K6 = 5,    // Q6_K: ql [rows][cols/2], qh [rows][cols/4], sc int8 [rows][cols/16] (meta), d fp16 [rows][cols/256]
    // r17 CYBER-FROST (qwen4exp) formats
    FMT_K2 = 6,    // Q2_K: qs [rows][cols/4] (64 B / 256), meta 20 B / 256 (d, dmin fp16, scales[16])
    FMT_K4 = 7,    // Q4_K: qs [rows][cols/2] (128 B / 256), meta 16 B / 256 (d, dmin, scales[12]; same meta shape as K5)
    FMT_Q51 = 8,   // Q5_1: codes [rows][cols/2] (16 B / 32), hi [rows][cols/8] (4 B / 32), d/m fp16 [rows][cols/32]
    // cf-m6 requant (CF_REQUANT.md): the iq1_s class, 50 B / 256 = 1.5625 bpw
    FMT_IQ1S = 9,  // codes = qs (32 B / 256: one u8 grid index per 8 elems), hi = qh (16 B / 256: 8 u16
                   // per-32 scale/shift/index-halves), d fp16 [rows][cols/256]; the t4q_kgrid_1bit_2048 lattice
    // cf-m6 r3 (the dn tiling: the dn rows are 640 wide, 640 % 256 != 0): the 128-elem half
    // block - the SAME per-32-group lattice/scale/dp4a arithmetic, the d/max_scale over 4
    // groups; 26 B / 128 = 1.625 bpw; 640 = 5 x 128 tiles exactly. Pairs with the q8_0
    // activation (its 32-blocks tile 640; the q8_K 256-super-blocks do NOT).
    FMT_IQ1SH = 10,  // codes = qs (16 B / 128), hi = qh (8 B / 128: 4 u16), d fp16 [rows][cols/128]
};

struct PackedW {
    int fmt = -1;
    int64_t rows = 0, cols = 0;
    uint8_t* codes = nullptr;
    uint8_t* hi = nullptr;
    uint16_t* d = nullptr;
    uint16_t* m = nullptr;
    uint8_t* meta = nullptr;
    void* base = nullptr;   // single allocation
    size_t bytes = 0;
    int gpu = 0;
};

const char* pack_fmt_name(int fmt);

// cf-m6 r3/r4: the requant SLAB header (the cfreq pack's per-layer file form; all offsets
// from the file start). v2: the dn planes carry their own format (the tiling find - the
// dn's 640-wide rows cannot tile the 256-elem iq1_s block, so the dn is the FMT_IQ1SH
// 128-elem half block, same lattice/arithmetic, 26 B/128). SHARED between the packer
// (t4q/tools/cf_requant_pack.cpp, the writer) and the loader (cf_loader.cu, the reader) -
// ONE definition, the static_assert pins the 96-B no-padding layout.
struct SlabHdr {
    uint32_t magic;      // 0x45513454 'T4QE'
    uint32_t version;    // 2
    uint32_t fmt;        // the gu planes' format: FMT_IQ1S (256-elem blocks)
    uint32_t dn_fmt;     // the dn planes' format: FMT_IQ1SH (128-elem half blocks)
    uint32_t ne;         // the expert count (512 on the real pack)
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
constexpr uint32_t SLAB_MAGIC = 0x45513454u;

// cf-m6 r5 (the spec's frozen AMORTIZED M=nr verify form): the per-row activation/output
// plane table the verify's iq1_s dots read. The pointers are FIXED at the verify alloc
// (the per-row scratch planes never move), so the table uploads ONCE; the per-layer
// varying part is the union W views + the row map (cf_model.h's CfVerify). The gu side
// (xq/bs/yd/y) is the q8_K pairing, the dn side (xq2/xd2/xs2/y2) the q8_0 pairing - the
// SAME planes the per-row _b launches read, so the amortized kernel's per-(row, pick)
// numerics are identical by construction.
constexpr int T4Q_VFY_MAXR = 8;  // the verify's row cap (cf::MAXR; static_assert'd where both are visible)
struct VfyMoeTab {
    const int8_t* xq[8];   // row r's q8_K quads (gu)
    const int16_t* bs[8];  // row r's q8_K per-16 bsums
    const float* yd[8];    // row r's q8_K super-block d
    float* y[8];           // row r's gate|up output ([TOPK, 2n] at s.logits)
    const int8_t* xq2[8];  // row r's q8_0 quads (dn)
    const float* xd2[8];   // row r's q8_0 per-32 d
    const int* xs2[8];     // row r's q8_0 per-32 signed sums
    float* y2[8];          // row r's per-pick down output ([TOPK, D] at ye + r*TOPK*D)
};
static_assert(sizeof(VfyMoeTab) == 512, "the vfy tab: 8 fields x 8 row slots, no padding");
