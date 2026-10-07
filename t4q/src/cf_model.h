// cf-m1: the CYBER-FROST-3.8 (qwen4exp) engine structures. research/cf-arch.md + the
// llama.cpp qwen4exp.cpp graph are the spec; every constant below came off the real GGUF.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "gguf.h"
#include "kernels/kernels.h"
#include "packed.h"

#define CK(x)                                                                                              \
    do {                                                                                                   \
        cudaError_t e_ = (x);                                                                              \
        if (e_ != cudaSuccess)                                                                             \
            throw std::runtime_error(std::string("CUDA error ") + cudaGetErrorString(e_) + " at " + __FILE__ + \
                                     ":" + std::to_string(__LINE__) + ": " #x);                            \
    } while (0)

namespace cf {
constexpr int D = 2560;         // hidden
constexpr int NL = 48;          // trunk layers (49th = MTP, cf-m1b)
constexpr int V = 248320;       // vocab
constexpr int HC = 4;           // hyper-connection streams
constexpr int HCD = D * HC;     // 10240, the wide residual
constexpr int LORA = 320;       // hc low-rank
constexpr int NE = 512;         // experts
constexpr int TOPK = 10;
constexpr int EE = 640;        // expert FFN
constexpr int HQ = 24, HKV = 2, HD = 256, NROT = 64;
constexpr int HK = 16, HV = 48, DK = 128, VDIM = 6144, CONV = 10240;
constexpr float EPS = 1e-6f;
constexpr float ROPE_BASE = 1e7f;
constexpr int EOS1 = 248044;    // config eos / ple eos
constexpr int EOS2 = 248046;    // <|im_end|>
constexpr int IMG_TOK = 248056;
// PLE (layer 1): 3-gram, 8 heads per n-gram -> heads 0..7 bigram, 8..15 trigram
constexpr int PLE_LAYER = 1, PLE_DIM = 160, PLE_NHEADS = 16, PLE_NGRAM = 3, PLE_CONV = 4;
constexpr int PLE_HIST = (PLE_CONV - 1) * PLE_NGRAM;  // 9 columns of conv history
inline bool is_attn(int il) { return (il % 4) == 3; }
}  // namespace cf

// One trunk layer. The routed experts stay in the host mmap and are staged per token,
// so the layer holds the GgufTensor* for them instead of a PackedW.
struct CfLayer {
    int il = 0, gpu = 0;
    bool attn = false;
    // hyper-connection mixers: [0] = attn-side, [1] = ffn-side
    PackedW hc_down[2], hc_up[2], hc_inject[2];   // [HCD,320] [320,HCD] [HCD,4]
    float* hc_norm[2] = {nullptr, nullptr};       // [HCD] fp32, (c,s) at s*D+c
    // DeltaNet (the recurrent layers)
    PackedW qkv, z, alpha, beta, ssm_out;
    float *conv_w = nullptr, *ssm_a = nullptr, *ssm_dt = nullptr, *ssm_norm = nullptr;
    float *conv_state = nullptr, *S = nullptr;
    // dense attention
    PackedW wq, wk, wv, wo;
    float *q_norm = nullptr, *k_norm = nullptr;
    uint16_t *kc = nullptr, *vc = nullptr;
    // MoE: router + shared expert in VRAM; the 512 routed experts stay mmap'd
    PackedW router, sh_gate, sh_up, sh_down, sh_ginp;
    const GgufTensor *t_gate_exps = nullptr, *t_up_exps = nullptr, *t_down_exps = nullptr;
    // the resident tier (cf-m3, the hot-set landing): with a hot-set file loaded, the
    // per-layer top-H experts (by routed mass) are packed into VRAM at load in the SAME
    // packed shapes as the staging ([hn*2*EE, D] K2 gate|up + [hn*D, EE] P4 down), so the
    // dual-path moe's hit picks read them through the W table with zero staging. hn = 0
    // (absent file) = OFF = the verbatim full-staging path.
    PackedW res_gu, res_dn;
    int hn = 0;
    int* hot_ids = nullptr;   // [hn] the resident expert ids, the load-time packing order
    int* hot_idx = nullptr;  // [NE] the expert id -> the resident index h, or -1 (a miss)
    // cf-m3 (r19u) the UVA third path: the device aliases of this layer's registered expert
    // tensors (the coalesced page-span registration at load; null when the layer is staged)
    const uint8_t* uva_gate = nullptr;
    const uint8_t* uva_up = nullptr;
    const uint8_t* uva_dn = nullptr;
};

// The PLE module (layer 1 only). The 26.85 GiB hash table stays mmap'd.
struct CfPle {
    PackedW key, value;      // [D,HCD] [D,D]
    float *norm_key = nullptr, *norm_query = nullptr, *norm_conv = nullptr;  // [HCD]
    float* conv_w = nullptr;        // [CONV=4, HCD] fp32, (i*4 + k)
    uint64_t mult[3] = {};          // layer multipliers
    uint64_t head_off[16] = {}, head_vocab[16] = {};
    const GgufTensor* table = nullptr;  // per_layer_token_embd [160, rows]
};

struct CfScratch {
    float *h = nullptr, *xn = nullptr, *lo = nullptr;        // h = res_hc [HCD], xn flat, lo [320]
    float *gate = nullptr, *mixed = nullptr, *inj = nullptr, *block = nullptr;  // [HCD]/[D]/[4]/[D]
    float *qfull = nullptr, *aq = nullptr, *ak = nullptr, *k = nullptr, *v = nullptr, *att = nullptr, *attg = nullptr;
    float *scores = nullptr;
    float *qkv = nullptr, *zz = nullptr, *braw = nullptr, *araw = nullptr, *beta = nullptr, *g = nullptr;
    float *conv = nullptr, *qn = nullptr, *kn = nullptr, *o = nullptr, *on = nullptr, *a = nullptr;
    float *ffg = nullptr, *ffu = nullptr, *ffa = nullptr;    // routed experts: gate/up/act [TOPK*EE]
    float *logits = nullptr;
    // the q8 fast-path activation planes (the r18 repack worklist): the Q2_K/Q4_K tensors dot
    // against the Q8_K quantization, the Q5_1 against the Q8_1 - the exact activation
    // arithmetic the ggml CPU oracle itself computes. Sized for the largest gemv input
    // (HCD = 10240: the hc mixers' down/inject columns; the lm_head's 2560 is smaller).
    int8_t* xq1 = nullptr;               // [HCD] the q8_1 codes
    float *xq1_d = nullptr, *xq1_s = nullptr;  // [HCD/32] the fp16-rounded d and sum
    int8_t* xqk = nullptr;               // [HCD] the q8_K codes
    int16_t* xqk_b = nullptr;            // [HCD/16] the bsums
    float* xqk_d = nullptr;              // [HCD/256] the per-super-block d
    int8_t* xq0 = nullptr;               // [TOPK*EE] the q8_0 codes (the down experts, flat [TOPK][EE])
    float* xd0 = nullptr;                // [TOPK*EE/32] the fp16-rounded d
    int* xs0 = nullptr;                  // [TOPK*EE/32] the per-32-block signed code sum (the dp4a factored -8 bias)
};

struct CfCtx {
    GgufFile f;
    std::vector<CfLayer> layers;
    CfPle ple;
    CfScratch sc;
    PackedW o_down, o_up;            // the final output mixer (the output norm)
    float* o_norm = nullptr;         // [HCD]
    PackedW output;                  // lm_head [V, D]
    const GgufTensor* tok_embd = nullptr;
    // staging (raw GGUF slabs -> repack on device)
    uint8_t *raw_stage = nullptr;     // pinned: one layer's 10 experts (gate+up Q2_K, down Q4_0)
    uint8_t *raw_dev = nullptr;       // device mirror
    PackedW up_stage;                 // [TOPK*2*EE, D]   Q2_K gate|up per expert
    PackedW dn_stage;                // [TOPK*D, EE]     Q4_0 down per expert
    // the per-pick W tables (the cf-m3 tiering mechanism, adopted by the default path): the
    // identity tables = the staged slab views, so the batched gemvs are bit-identical to the
    // old uniform-stride advance; the tiering swaps in per-hit resident views with zero
    // kernel change. Built once at load (the staging bases never move), uploaded once.
    PackedW* wt_gu = nullptr;         // device: [TOPK] the gate|up slab views
    PackedW* wt_dn = nullptr;         // device: [TOPK] the down slab views
    PackedW h_wt_gu[cf::TOPK] = {};   // host staging for the upload
    PackedW h_wt_dn[cf::TOPK] = {};
    // the dual-path moe (cf-m3): with the tier on, the per-step table composes the hit
    // picks' RESIDENT views and the miss picks' staged-slot views and re-uploads per layer
    // (1760 B); with the tier off these stay unused and the load-time identity upload stands
    PackedW h_step_gu[cf::TOPK] = {};  // the per-step composed gate|up table (host)
    PackedW h_step_dn[cf::TOPK] = {};  // the per-step composed down table (host)
    bool tiered = false;              // a hot-set file was loaded (the resident tier is ON)
    bool uva = false;                 // cf-m3 (r19u): T4Q_CF_UVA_LAYERS registered (the alias path is ON for il < uva_n)
    int uva_n = 0;                    // the first uva_n layers read their experts through the aliases
    void* uva_reg = nullptr;          // the coalesced page-span host base (one registration, unregistered at free)
    size_t uva_reg_len = 0;
    int* eid_dev = nullptr;          // device: [TOPK] the per-step expert ids for the scatter repack
    // cf-m3 (r19v) the step params: pos rides a pinned host word uploaded at the step head,
    // and the attention kernels read it from the device word - the same int, the same
    // downstream arithmetic (byte-identical), but the ONLY per-step varying kernel arg
    // becomes capture-constant (the G1/G2 graph forms; the r19p census found no other).
    int* h_params = nullptr;         // pinned host [4]: h_params[0] = pos ([1..3] spare)
    int* d_params = nullptr;         // device [4]
    // r19v: the tiered host window's miss count, feeding the emission's tiered branch (the
    // compose ran in host_router); dead under graphs (the tiered mode is gmode-excluded)
    int tier_nmiss = 0;
    // cf-m3 (r19v) the G1 segment graphs: T4Q_CF_GRAPH=1 captures the NL+1 = 49 sync-bounded
    // segments at the first step and replays them per step (the launch wall -> ~1 replay per
    // segment; the host windows - the router softmax/top-10 + the OFF-path staging memcpys +
    // the PLE gather - stay). OFF (absent env / tiered / dumping) = the verbatim emission
    // path. The graphs survive cf_reset (the buffers are the same; the memsets run outside
    // the graphs); cf_free destroys them after the stream drain.
    std::vector<cudaGraphExec_t> gexec;   // [NL+1] the per-segment instantiated execs
    std::vector<cudaGraph_t> ggraph;      // the captured sources (destroyed at free)
    int gmode = 0;                       // 0 = the direct emission, 1 = the graph driver
    float* ye = nullptr;              // [TOPK*D] per-expert down outputs
    float* we = nullptr;              // [TOPK] renormalized router weights (device)
    float* ysh = nullptr;             // [D] shared expert out
    float* sh_gate_raw = nullptr;    // [1] the shared gate gemv output (device)
    float* h_router = nullptr;        // host: [NE] router logits
    float* we_h = nullptr;            // host: [TOPK] + ids
    int* eid = nullptr;               // host: [TOPK] expert ids
    float* h_emb = nullptr;           // host: [D] embedding row
    float* h_ple = nullptr;           // host: [D] gathered ple rows
    float* h_logits = nullptr;        // host: [V]
    int* toks = nullptr;              // token history (max_ctx)
    // ple conv ring: [PLE_HIST][HCD] fp32 (device)
    float* ple_hist = nullptr;
    float *ple_key = nullptr, *ple_query = nullptr;  // [HCD]
    float *ple_s = nullptr, *ple_gate = nullptr;     // [HC]
    float* ple_gated = nullptr;                      // [HCD]
    cudaStream_t st = nullptr;
    int gpu = 0, pos = 0, max_ctx = 0, steps = 0;
    double step_s = 0.0;
    bool have_logits = false;
    // cf-m2 census: when set, every moe() appends the layer's top-10 (id, renormed weight)
    FILE* census_f = nullptr;
    std::string err;
};

// cf_kernels.cu
void launch_cf_hc_norm(const float* res, const float* w, float* xn, cudaStream_t s);
void launch_cf_hc_lo(const float* y, float* lo, cudaStream_t s);  // lo = silu(y * 1/HC)
void launch_cf_hc_mixed(const float* xn, const float* y_up, float* mixed, cudaStream_t s);
void launch_cf_hc_combine(float* res, const float* block, const float* inj, cudaStream_t s);
void launch_cf_res_init(float* res, const float* emb, cudaStream_t s);
void launch_cf_gdn_gnorm(const float* o, const float* z, const float* w, float* out, float eps, cudaStream_t s);
void launch_cf_qk_norm_rope(const float* qfull, const float* k, const float* qw, const float* kw, float* qn,
                            float* kn, const int* pos_dev, float eps, float freq_base, int n_rot, cudaStream_t s);
void launch_cf_kv_store(const float* k, const float* v, uint16_t* kc, uint16_t* vc, const int* pos_dev, int max_ctx,
                        cudaStream_t s);
void launch_cf_attn_decode(const float* q, const uint16_t* kc, const uint16_t* vc, float* out, float* scores,
                           const int* pos_dev, int max_ctx, float scale, cudaStream_t s);
void launch_cf_ple_sg(const float* key, const float* query, float* s_out, float* gate, cudaStream_t s);
void launch_cf_ple_gated(const float* value, const float* gate, float* gated, cudaStream_t s);
void launch_cf_ple_conv(const float* gnorm, float* hist, const float* w, float* out, cudaStream_t s);
void launch_cf_moe_out(const float* ye, const float* we, const float* ysh, const float* sh_gate_raw, float* out,
                        cudaStream_t s);
void launch_cf_silu_mul_b(const float* gu, float* out, int n_per, int batch, cudaStream_t s);

// cf_engine.cu / cf_loader.cu
CfCtx* cf_load(const char* path, int max_ctx, std::string* err);
void cf_free(CfCtx* c);
bool cf_step(CfCtx* c, int token);       // one decode step; logits land in c->h_logits
void cf_reset(CfCtx* c);
