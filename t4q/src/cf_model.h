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
constexpr int MAXR = 8;        // cf-m4: the verify's row bound (T4Q_CF_K <= 7 -> nr = k+1 rows)
static_assert(MAXR == T4Q_VFY_MAXR, "the r5 vfy tab's fixed row slots must match cf::MAXR");
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
    // cf-m6 r6a (the stage-2 TP split, CF_REQUANT.md section 6's freeze): the GPU1 twin of
    // the requant resident pair - T4Q_CF_IQTP splits the covered layers' pools BY ID
    // (GPU0 owns the experts [0,256), GPU1 [256,512)), each side holding its half's
    // planes. The PAIR form (not an array) keeps every landed r4/r5 read site on res_gu
    // (= GPU0's own) untouched; the r6b emission's per-side loops read both. INERT until
    // the r6b emission lands (cf_step/verify/draft throw under iqtp).
    PackedW res_gu1, res_dn1;
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

// cf-m4 (r19w): the MTP draft block's state. The weights ride a CfLayer (L.il = 48,
// L.attn = true - blk.48 is non-recurrent); the nextn extras (enorm/hnorm/eh_proj +
// the draft's own final mixer hc_head_*) are direct members. The draft's OWN scratch
// (the trunk's is never touched between steps); the ALL-512 resident expert slabs +
// the per-step composed W tables (the tiering's mechanism, all hits - staging-free);
// the chain state hres (res' carried across draft steps) + the draft's own KV position
// (the pos word rides d_params[1], the spare slot of the r19v step-params pair).
struct CfDraft {
    CfLayer L;                        // il = 48, attn = true (the full-attn family + the MoE shell)
    float *enorm = nullptr, *hnorm = nullptr;   // nextn: [2560] plain / [2560,4] grouped F32
    PackedW eh_proj;                  // nextn: [2560 out, 5120 in] Q8_0 (shared by the 4 streams)
    float* hh_norm = nullptr;         // nextn.hc_head_norm [2560,4] F32 (the draft's final mixer)
    PackedW hh_down, hh_up;           // nextn.hc_head_{down,up} Q8_0
    float* zero_h = nullptr;          // device [HCD] the h_{-1} = 0 pair input (the 27B's form)
    float* h_e = nullptr;             // host [D] the draft's own embedding row (pinned)
    float* h_logits = nullptr;        // host [V] the draft's logits (pinned)
    float* h_in = nullptr;            // device [HCD] the h input (D2D from the arg, or zero_h)
    float* hres = nullptr;            // device [HCD] the chain state (the draft's res' output)
    float *e = nullptr, *e_norm = nullptr, *h_norm = nullptr;  // device [D]/[D]/[HCD]
    float* eh_cat = nullptr;          // device [4][5120] the per-stream [e_norm ; h_norm_s] gather
    float* ygu = nullptr;             // device [TOPK*2*EE] the gate|up batched gemv output (own buffer)
    float* ye = nullptr;              // device [TOPK*D] the batched down gemv output (the trunk's is CfCtx-level)
    float *ysh = nullptr, *sh_gate_raw = nullptr;  // device [D]/[1] the shared expert out + its sigmoid gate
    CfScratch sc;                     // the draft's OWN scratch (the trunk's is never touched)
    PackedW res_gu, res_dn;            // the ALL-512 resident slabs: [512*2*EE, D] / [512*D, EE] Q8_0
    // cf-m6 r6c: under T4Q_CF_IQTP the draft's pool splits BY ID (the section 6 freeze,
    // ~1.33 GiB per side): res_gu/res_dn shrink to GPU0's [0,256) half, res_gu1/res_dn1
    // are GPU1's [256,512) half (the same e>>8/e&255 owner map, the local row offsets)
    PackedW res_gu1, res_dn1;
    PackedW h_wt_gu[cf::TOPK] = {}, h_wt_dn[cf::TOPK] = {};  // the per-step composed host tables
    PackedW *wt_gu = nullptr, *wt_dn = nullptr;             // the device W tables [TOPK]
    int pos = 0;                      // the draft's own KV position
};

// cf-m4 (r19x): the MTP verify's state. THE GATE FORM (CF_MTP.md section 5): the k+1
// candidate rows through the WHOLE trunk in one pass, every op the sequential step's op
// with the ROW's slice (the same kernels, the same args, the same per-row order - the
// rows run strictly in row order so the rolling states (KV, GDN S/conv, the PLE ring)
// evolve exactly as the sequential steps would), with the ONE structural change: the
// per-layer MoE host window BATCHES the rows - all the rows' router gemvs + D2Hs, ONE
// sync, the per-row order-exact top-10, the picks' UNION deduped + staged/read ONCE
// (the dedup is the verify's staging win), then the per-row batched gemvs against the
// union slabs. Loaded under the same T4Q_CF_MTP gate (k rides T4Q_CF_K, default 3).
struct CfVerify {
    int nr = 0;                        // the loaded row count (k+1); the guard for cf_verify calls
    CfScratch sc[cf::MAXR];             // the per-row scratch (the trunk's is never touched)
    int *h_pos = nullptr, *d_pos = nullptr;  // the per-row pos words (pinned/host->device; the row's pos_dev = d_pos + r)
    float* h_emb = nullptr;            // pinned [MAXR][D] the per-row embedding rows (the dequant targets)
    float* h_ple = nullptr;            // pinned [MAXR][D] the per-row PLE gather rows (no cross-row H2D race)
    float* h_router = nullptr;         // pinned [MAXR][NE] the per-row router outputs
    int* eid = nullptr;                // host [MAXR][TOPK] the per-row picks
    float* we_h = nullptr;             // host [MAXR][TOPK] the per-row weights
    float* we_dev = nullptr;          // device [MAXR][TOPK] (the row's moe_out reads we_dev + r*TOPK)
    PackedW h_wt_gu[cf::MAXR][cf::TOPK] = {}, h_wt_dn[cf::MAXR][cf::TOPK] = {};  // the per-row composed tables
    PackedW *wt_gu = nullptr, *wt_dn = nullptr;  // device [MAXR][TOPK] the per-row W tables
    PackedW uni_gu, uni_dn;            // the UNION slabs: [nr*TOPK*2EE, D] K2 / [nr*TOPK*D, EE] P4 (the worst case)
    int* uids_dev = nullptr;          // device [MAXR*TOPK] the union's expert ids (the UVA scatter's eid)
    int uidx[cf::NE];                 // the union slot map (expert -> slot, -1 = absent; host, recomposed per layer)
    int uids[cf::MAXR * cf::TOPK];    // the union's expert ids in slot order (the staging order)
    int nu = 0;                        // this layer's union count
    // cf-m6 r5: the union-size accumulator (the mean over the verify's layers) - the
    // amortization's REAL overlap number (nu vs nr*TOPK; the L4 battery's eye)
    long nu_sum = 0, nu_cnt = 0;
    // cf-m6 r5 (the spec's frozen amortized M=nr verify): the iq1_s-covered layer's verify
    // dots decode each UNION pick's W once and dot it against every draft row that picked
    // it (the drop-in _b re-decodes per (row, pick)). The union views + the row map replace
    // the per-row W tables on the RESIDENT path (the uncovered path keeps the _b form);
    // the per-row plane table (vtab) is FIXED at alloc - the scratch pointers never move -
    // and uploads once. rowmap[u][r] = row r's pick index of slot u, -1 = not picked.
    PackedW* uv_gu = nullptr;         // device [MAXR*TOPK] the union's resident gu views (slot order)
    PackedW* uv_dn = nullptr;         // device [MAXR*TOPK] the union's resident dn views
    PackedW h_uv_gu[cf::MAXR * cf::TOPK] = {}, h_uv_dn[cf::MAXR * cf::TOPK] = {};
    int* rowmap_dev = nullptr;        // device [MAXR*TOPK][T4Q_VFY_MAXR]
    int h_rowmap[cf::MAXR * cf::TOPK][T4Q_VFY_MAXR] = {};
    VfyMoeTab* vtab_dev = nullptr;    // the fixed per-row plane table (uploaded once at alloc)
    float* ye = nullptr;              // device [MAXR][TOPK*D] the per-row batched down outputs
    float* ysh = nullptr;             // device [MAXR][D] the per-row shared-expert outputs
    float* sh_gate_raw = nullptr;     // device [MAXR] the per-row shared-expert sigmoid gates
    float* h_logits = nullptr;        // pinned [MAXR][V] the per-row logits (the verify's output)
    // cf-m4 (r19y): the speculative driver's snapshot planes + state. The 27B's tp_spec.cu
    // rollback adapted to the CF rolling states: the GDN S/conv and the PLE ring are SHIFT
    // REGISTERS (not position-indexed like the 27B's CR-slot conv ring), so a partial accept
    // restores the after-row-n state from the verify's per-row captures - stream-ordered D2Ds
    // enqueued at the exact per-(layer,row) boundaries inside cf_verify (after the row's
    // deltanet/PLE roll, before the next row overwrites). The S/conv/PLE snapshots hold the
    // states AFTER rows 0..k-1 (k = nr-1 slots; the accept n = k leaves the rolling L.S
    // itself correct, no capture); pending_h holds every row's pre-final-mixer residual (the
    // catch-up's h inputs - CF_MTP.md section 4's ring; the tail's hc_mix only READS s.h, so
    // the capture rides the row's tail). The attention KV needs NO plane (the cells beyond
    // the rewound pos are invisible). The per-snapshot S set is ngdn*HV*DK*DK f32 ~113 MB,
    // so the 27B's ns = k+2 ring-index form would cost ~566 MB AND a separate-in/out change
    // to the in-place launch_gdn_recur; the captures are zero-kernel-surface (~340 MB at
    // k=3) and the copy cost (~4 ms/verify in the direct-launch form) is the later
    // segment-graph round's to remove.
    int ngdn = 0;                     // the GDN layer count (the gord map below)
    int gord[cf::NL];                 // il -> the GDN ordinal (attn layers -1)
    float* s_snap = nullptr;         // device [ngdn][k][HV*DK*DK] the per-row GDN S snapshots
    float* conv_snap = nullptr;      // device [ngdn][k][CONV*3] the per-row GDN conv snapshots
    float* ple_snap = nullptr;       // device [k][PLE_HIST*HCD] the per-row PLE ring snapshots
    float* pending_h = nullptr;      // device [MAXR][HCD] the per-row pre-final-mixer residuals
    int pending = -1;                 // the pending token (the one at position c->pos)
    double draft_s = 0, verify_s = 0, catch_s = 0;  // the round timers (the k tuning's eyes)
    long rounds = 0;
    // cf-m4 (r19z): the verify's own segment graphs (the G1 pattern applied to cf_verify -
    // the launch wall is the MTP path's dominant direct-form cost: ~2600 launches x nr
    // rows ~ 100-250 ms of pure wall per verify). The same NL+1 = 49 sync-bounded segments
    // (seg 0 = the head emission + L0's rows; seg k = L(k-1)'s moe rest + Lk's rows; seg 48
    // = L47's moe rest + the tail emission), captured at the FIRST full-nr verify call and
    // replayed after; the per-layer host windows (the sync + the top-10s + the union
    // staging + the W-table uploads - the staging sizes VARY per layer, so they stay
    // DIRECT) run between the replays. Partial-nr calls (the gate mode's tail chunks) fall
    // to the direct path (the captured shapes are nr-bound). Rides the same T4Q_CF_GRAPH
    // gate as the step's G1 graphs (the tiered/dump exclusions apply); every varying
    // content rides a pinned-fixed host source the captured memcpy nodes re-carry at each
    // replay, and the r19y snapshot D2Ds are fixed-arg nodes.
    std::vector<cudaGraphExec_t> vgexec;   // [NL+1] the verify's per-segment instantiated execs
    std::vector<cudaGraph_t> vggraph;      // the captured sources (destroyed at free)
};

// cf-m6 r6b (the section 6 freeze, the AMENDED form): the split-moe side planes + the sync
// pair. THE AMENDMENT over the frozen sketch: the core REPLICATION is replaced by the
// ACTIVATION SCATTER - the core (the attention/deltanet family, the hc mixers, the shared
// expert, the router, the rolling states) runs on GPU0 ONLY, the per-layer mixed [D] ships
// 0->1, and GPU1 is a PURE MoE ACCELERATOR (its owned picks' quantize + dots + partial).
// The critical path is the SAME as the replicated core's (the core wall + the split moe -
// the replication ran the same core work in lockstep, buying nothing on the path), and the
// scatter form SAVES the ~2.6 GB core twin, HALVES the launch wall, and removes the
// lockstep-state determinism risk (the rolling states stay GPU0's own - the single source
// of truth). GPU1's VRAM is the half pool + this trivial scratch (<1 MB).
struct CfIqtp {
    cudaStream_t st1 = nullptr;                 // GPU1's moe stream (the events bridge the sides)
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;    // the per-layer sync pair (the mixed 0->1, the partial 1->0)
    float* mixed = nullptr;                     // [D] the shipped mixed (the scatter's landing)
    float *logits = nullptr, *ffa = nullptr;    // [TOPK*2*EE] / [TOPK*EE] the side-1 gu y + the silu out
    float* ye = nullptr;                        // [TOPK*D] the side-1 dn y (the compact k' slots)
    float *partial = nullptr, *partial0 = nullptr;  // [D] side-1's (the 1->0 ship) / [2D] p0 + the shipped p1 at +D
    int8_t* xqk = nullptr;                      // the side-1 q8_K planes (the trunk's own sizes)
    int16_t* xqk_b = nullptr;
    float* xqk_d = nullptr;
    int8_t* xq0 = nullptr;
    float* xd0 = nullptr;
    int* xs0 = nullptr;
    float* we_c[2] = {nullptr, nullptr};        // device [TOPK] the per-side compact we
    float h_we_c[2][cf::TOPK] = {};             // the host staging
    PackedW* wt_gu[2] = {nullptr, nullptr};    // device [TOPK] the per-side W tables
    PackedW* wt_dn[2] = {nullptr, nullptr};
    PackedW h_wt_gu[2][cf::TOPK] = {}, h_wt_dn[2][cf::TOPK] = {};
    int n[2] = {0, 0};                          // the layer's per-side pick counts
    // cf-m6 r6c part 2 (the verify's split-moe, the section 6 freeze's hardest piece):
    // the per-row GPU1 planes + the per-side sub-union structures. The verify's per-row
    // scratch (v->sc[r]) stays GPU0's own; GPU1 gets its OWN per-row planes (the shipped
    // mixed, the q8_K/q8_0 activations, the gu y + the silu out at the PICK-SLOT layout,
    // the dn y) + the side's FIXED tab (the r5 VfyMoeTab form pointing at GPU1's planes)
    // + the sub-union W tables + the rowmap + the per-row owned k-lists + the we copy +
    // the per-row partials (the [2*MAXR*D] pair: p0 at [0,nr*D), the shipped p1 at
    // [nr*D, 2*nr*D) - the greedy's own 2-slot form). The amortized dots run per side
    // over the side's sub-union (the r5 kernel's own walk, the nu0/nu1 counts).
    float* v_mixed[cf::MAXR] = {};              // the per-row shipped mixed
    int8_t* v_xqk[cf::MAXR] = {};
    int16_t* v_xqk_b[cf::MAXR] = {};
    float* v_xqk_d[cf::MAXR] = {};
    float* v_logits[cf::MAXR] = {};            // the per-row gu y (the pick slots)
    float* v_ffa[cf::MAXR] = {};               // the per-row silu out
    int8_t* v_xq0[cf::MAXR] = {};
    float* v_xd0[cf::MAXR] = {};
    int* v_xs0[cf::MAXR] = {};
    float* v_ye[cf::MAXR] = {};                // the per-row dn y (the pick slots)
    float* v_partial = nullptr;                // [MAXR*D] side-1's per-row partials
    float* v_partial0 = nullptr;               // [2*MAXR*D] side-0's + the shipped p1 landing
    PackedW* uv_gu1 = nullptr;                 // GPU1's sub-union W tables [MAXR*TOPK]
    PackedW* uv_dn1 = nullptr;
    PackedW h_uv_gu1[cf::MAXR * cf::TOPK] = {}, h_uv_dn1[cf::MAXR * cf::TOPK] = {};
    int* rowmap1_dev = nullptr;                // GPU1's rowmap [MAXR*TOPK][T4Q_VFY_MAXR]
    int h_rowmap1[cf::MAXR * cf::TOPK][T4Q_VFY_MAXR] = {};
    VfyMoeTab* vtab1_dev = nullptr;            // GPU1's FIXED tab (uploaded once at alloc)
    VfyMoeTab h_vtab1 = {};
    int* ks_dev[2] = {nullptr, nullptr};       // [MAXR*TOPK] the per-row owned-pick lists
    int h_ks[2][cf::MAXR * cf::TOPK] = {};
    int h_nk[2][cf::MAXR] = {};                // the per-row owned counts per side
    float* we_dev1 = nullptr;                  // [MAXR*TOPK] GPU1's we copy
    int nu0 = 0, nu1 = 0;                      // the layer's per-side sub-union counts
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
    // cf-m4 (r19w): the MTP draft block (blk.48, ALL Q8_0, resident - the frozen CF_MTP.md
    // design). Loaded only under T4Q_CF_MTP=1 (absent = not loaded, the VRAM stays free);
    // INERT until cf_draft_step is called (the acceptance smoke now; the speculative
    // verify/rollback driver later). The draft reuses CfLayer (the full-attn family + the hc
    // mixers + the MoE shell; blk.48 is non-recurrent per attention.recurrent_layers[49]);
    // ALL 512 experts pack into VRAM at load (the same chunked raw->pinned->H2D->repack
    // pass as the hot-set tier), so the draft's MoE is STAGING-FREE: the per-step W table
    // composes the 10 picks' resident views and uploads (the tiering's mechanism, all
    // hits). The pair semantics are the 27B's gate-proven form: (x_q, h_{q-1}) at the
    // draft's own KV position q, h_{-1} = 0; the chain state hres carries res' forward.
    CfDraft* draft = nullptr;
    // cf-m4 (r19x): the MTP verify's per-row planes + the union staging (the same
    // T4Q_CF_MTP gate; k rides T4Q_CF_K, default 3 -> nr = k+1 rows). cf_verify runs the
    // k+1 candidate rows through the whole trunk in ONE pass (every op the sequential
    // step's op with the row's slice, the rolling states evolving in row order), with
    // the per-layer MoE host window BATCHED (the rows' routers -> ONE sync -> the
    // per-row order-exact top-10 -> the picks' UNION staged/read once -> the per-row
    // gemvs against the union slabs). INERT until cf_verify is called (the gate mode
    // now; the speculative driver later).
    CfVerify* verify = nullptr;
    // cf-m6 r6b: the split-moe side planes (T4Q_CF_IQTP at load). The greedy emission
    // reads it (the owner dispatch + the scatter + the per-side moe + the combine); the
    // verify/draft TP forms are r6c (those entry points still throw under the iqtp flag).
    CfIqtp* iqp = nullptr;
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
    // cf-m6 r4 (CF_REQUANT.md section 6, stage 1): the iq1_s requant resident tier.
    // T4Q_CF_IQSLAB=<dir> loads the cfreq pack's per-layer slabs for the FIRST iqs_n
    // layers (T4Q_CF_IQN, 0/absent = the free-VRAM auto-fit at ~498.07 MB/layer): the
    // WHOLE expert pool of each covered layer is resident (res_gu FMT_IQ1S + res_dn
    // FMT_IQ1SH, the identity hot map), every pick a HIT with zero staging, the emission
    // taking the r4 batched IQ1S/IQ1SH dots. The covered prefix is excluded from the UVA
    // registration (uva_lo): the resident tier wins the branch order.
    int iqs_n = 0;                    // the covered layer count (0 = off)
    // cf-m6 r6a: the by-ID split flag (T4Q_CF_IQTP=1 + the IQSLAB pair): the covered
    // layers' pools are split by owner (GPU0 [0,256), GPU1 [256,512)) - the full 24.41 GB
    // pool across the two T4s. The emission support (the per-side forward, the dispatch,
    // the both-ways combine) is r6b: the engine entry points THROW under this flag until
    // then (the loader's split tier + its gates land first - the staged-round form).
    bool iqtp = false;
    bool uva = false;                 // cf-m3 (r19u): T4Q_CF_UVA_LAYERS registered (the alias path is ON for il < uva_n)
    int uva_n = 0;                    // the first uva_n layers read their experts through the aliases
    int uva_lo = 0;                   // cf-m6 r4: the alias path is ON for [uva_lo, uva_n) - the resident prefix excluded
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
    // cf-m4 (r19aa) the PLE prefetch (CF_MTP.md section 8, the frozen design): the verify's
    // row gathers fault the mmap'd table (~98 us/row first-fault class on the Kaggle disk);
    // the draft calls' GPU stretches are the only window - the touches warm the row's pages
    // under the drafts, the gather coalesces with any in-flight fault. OFF (absent
    // T4Q_CF_PLE_PRE) = the verbatim round; value-invisible either way (a pure page warm).
    int ple_pre = 0;
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
// cf-m6 r6b (the split-moe combine pair): the owner-side partial over the side's compact
// slots, then the final (p0 + p1 + sigmoid(gate)*ysh) on GPU0 - together the moe_out's own
// arithmetic with the add order split across the sides
void launch_cf_moe_partial(const float* ye, const float* we, int n, float* out, cudaStream_t s);
// cf-m6 r6c part 2 (the verify's split-moe): the per-row gather partial over the row's
// owned pick slots (the pick-slot ye layout, the window-built k-lists)
void launch_cf_moe_partial_k(const float* ye, const float* we, const int* ks, int nk, float* out, cudaStream_t s);
void launch_cf_moe_final(const float* p0, const float* p1, const float* ysh, const float* sh_gate_raw, float* out,
                         cudaStream_t s);
void launch_cf_silu_mul_b(const float* gu, float* out, int n_per, int batch, cudaStream_t s);
// cf-m4 (r19w): the MTP draft's eh_proj input gather - out[s][0:2560] = e_norm (shared),
// out[s][2560:5120] = h_norm[s*2560:(s+1)*2560] (the per-stream half), out flat [4][5120]
void launch_cf_eh_gather(const float* e_norm, const float* h_norm, float* out, cudaStream_t s);

// cf_engine.cu / cf_loader.cu
CfCtx* cf_load(const char* path, int max_ctx, std::string* err);
void cf_free(CfCtx* c);
bool cf_step(CfCtx* c, int token);       // one decode step; logits land in c->h_logits
// cf-m4 (r19w): one MTP draft forward at the draft's own KV position (the pair
// (token, h): h = the trunk's pre-final-mixer wide residual, or nullptr = h_{-1} = 0).
// The draft's logits land in c->draft->h_logits (the prediction for the NEXT position);
// the chain state (res') carries into the next draft call. Requires T4Q_CF_MTP=1 at load.
bool cf_draft_step(CfCtx* c, int token, const float* h);
// cf-m4 (r19x): the MTP verify - nr <= k+1 candidate rows through the whole trunk in ONE
// pass (CF_MTP.md section 5): every op the sequential step's op with the row's slice
// (the same kernels, the same args, the same per-row order - the rolling states evolve
// exactly as the sequential steps), the ONE structural change: the per-layer MoE host
// window batches the rows (the routers D2H, ONE sync, the per-row order-exact top-10,
// the picks' UNION deduped + staged/read ONCE, the per-row gemvs against the union).
// The rows' logits land in c->verify->h_logits [nr][V] (row r = the prediction for the
// position pos+r+1); the trunk's pos/states advance by nr. Requires T4Q_CF_MTP=1 at load.
bool cf_verify(CfCtx* c, const int* toks, int nr);
// cf-m4 (r19y): the speculative driver's prompt pass (CF_MTP.md section 4) - the trunk
// over the prompt (one cf_step per token, the same path the reference modes run) with the
// draft paired one step behind (the pairs (ids[j], h_{j-1}), h_{-1} = 0; the h source
// c->sc.h is stable until the next trunk step overwrites it - the draft's D2D is enqueued
// first, stream-ordered), then the PENDING pair (the greedy argmax after the prompt,
// h_{np-1}) at the draft's position np. Requires a fresh context (pos 0, cf_reset first).
// Returns the pending token (the first generated token, the position-np prediction; the
// draft's h_logits hold its prediction of position np+1) or -1 on error (c->err set).
int cf_spec_prime(CfCtx* c, const int* ids, int np);
// cf-m4 (r19y): ONE speculative round - the 27B's gate-proven tp_spec.cu arithmetic
// adapted to the CF engine (CF_MTP.md sections 1/5/6). IN: the loop invariant (the draft
// processed 0..c->pos so its h_logits predict position c->pos+1; v->pending = the token
// at c->pos). (1) the k drafts: vt[0] = the pending, vt[1] = the pending's draft
// prediction (in hand), vt[2..k] chained on the draft's own hres (section 1's chain form);
// (2) the verify over the k+1 rows (the r19x GATE form + the r19y per-row captures);
// (3) the argmaxes yv[r] (the same first-max pick as the reference modes) + the accept
// scan n = the longest prefix with vt[n+1] == yv[n]; (4) the rollback (n < k): the GDN
// S/conv + PLE snapshots restored to the after-row-n state (stream-ordered D2Ds; the
// attention KV cells beyond the rewound pos are invisible - no copy), c->pos = p+n+1;
// (5) the emission yv[0..n] (n+1 tokens, the new pending = yv[n]); (6) the catch-up: the
// draft over the accepted tokens (yv[t], pending_h[t]) at the draft positions p+1..p+n+1
// (d->pos rewound to p+1 first - the chain's speculative draft KV slots are re-consumed
// in order), restoring the invariant. Returns the emitted count (n+1 >= 1), 0 on error.
int cf_spec_step(CfCtx* c, int* out);
void cf_reset(CfCtx* c);
