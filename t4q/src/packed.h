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
