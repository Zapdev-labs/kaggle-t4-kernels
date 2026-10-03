// TP=2 fast decode engine (M2-M4): fast SoA GEMVs (gemv.cuh), fused small kernels, P2P mailbox all-reduce,
// device-side StepState (graph-replayable), optional CUDA graphs. Both GPUs hold every layer's shard and an identical
// fp32 residual stream. See DESIGN.md sections 4-6.
#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "kernels/gemv.cuh"

namespace tp {

constexpr int NAR = 256;     // epoch stride per step (2 ARs per layer = 128 used)
constexpr int NSPLIT = 40;   // attention split-K blocks per kv head (2 x 40 = one wave at 2 blocks/SM)
constexpr int RING = 4096;   // host-mapped token ring
constexpr int MMAX = 7;      // spec: max verify columns (k <= 6 drafts)
constexpr int NSNAP = MMAX + 1;  // spec: DeltaNet state snapshot buffers
constexpr int CR = 16;       // spec: conv ring slots (positions pos-3 .. pos+MMAX-1 must not alias)

// device step state (one per GPU, identical contents on both)
struct StepState {
    int pos;        // position of the token being processed
    int token;      // token to embed at this step (written by the previous step's argmax, or from the prompt)
    uint32_t step;  // monotonic step counter (never reset: AR epochs derive from it)
    int err;        // watchdog / error code
    int n_prompt;   // positions < n_prompt take their token from the prompt buffer
    int last_tok;   // argmax of this step
    float last_val;
    int pad;
    // speculative decoding (tp_spec.cu)
    int vpos;       // position of the first token of the last verify
    int nacc;       // drafts accepted by the last verify (0..k)
    int sidx;       // DeltaNet snapshot buffer holding the state after the accepted tokens
    int nemit;      // tokens emitted by spec steps (GPU0 also mirrors it to host memory)
    int pad2[4];
};

struct FW {  // fast packed weight on one GPU
    t4q::gemv::Layout L;
    uint8_t* base = nullptr;
    float* invs = nullptr;  // prefill gemm8: per-row 127 / max|w| (computed on first batched prefill)
    float* invr = nullptr;  // prefill R512: per-row 127 / max|T(w)| of the rotated row (kernels/rot.cuh)
    int8_t* w8c = nullptr;  // prefill R512: persistent rotated int8 rows (pf_wcache), qkvz with the alpha/beta rows appended
    float* invc = nullptr;  // their scales (qkvz: 8448 rows)
    bool ok() const { return base != nullptr; }
};

struct ArArgs {
    float* y_peer = nullptr;        // peer's rx[slot]
    unsigned* cnt = nullptr;        // local block counter
    unsigned* peer_flag = nullptr;  // peer's flag[slot]
    const StepState* st = nullptr;
    int idx = 0;                    // AR index within the step
    int fence = 2;  // per-block fence before counting: 2 = system (correct), 1 = gpu, 0 = none; -1: staged remote rows
                    // only (no fence, counter or flag; the consumer publishes the flag); -2: staging only (tests);
                    // -3: LL rows; -4: AR tail (below)
    // fence -4 (option tail): the GEMV grid has tb extra tail blocks after the work blocks. Work blocks write their
    // rows locally and count; the tail blocks copy the partial to the peer (one slice and flag each), wait for the
    // peer's tb flags and do the AR + RMSNorm + q8 for the next GEMV (what k_ar_norm_mb does), so no ar_norm kernel.
    int tb = 0;
    unsigned* cnt2 = nullptr;          // local: tail blocks past the wait
    const float* h_in = nullptr;       // residual in / out
    float* h_out = nullptr;
    const float* own = nullptr;        // this GPU's partial (slot), = the GEMV's y
    const float* rx = nullptr;         // peer-written partial (slot)
    const unsigned* tflag = nullptr;   // local tail flags [2][TFLAGS] (peer-written)
    unsigned* peer_tflag = nullptr;    // peer's tail flags
    const float* nw = nullptr;         // next norm weight
    float* xn = nullptr;
    int8_t* xq = nullptr;
    int2* xm = nullptr;
};
constexpr int TFLAGS = 64;

struct SegArgs {  // extra fp32 rows (K = 5120) appended to a GEMV launch: y[i] = w[i] . x
    const float* w = nullptr;
    const float* x = nullptr;
    float* y = nullptr;
    int nrows = 0;
};

// L2 prefetch of the next GEMV's first weight bytes, issued by extra blocks of the small kernel before it (the memory
// system is otherwise idle during all-reduce waits and the small glue kernels)
struct Pf {
    const uint8_t* p[4] = {nullptr, nullptr, nullptr, nullptr};
    unsigned n[4] = {0, 0, 0, 0};
    int blocks = 0;  // extra blocks to launch (0: no prefetch)
};

// GEMV prologues (x is built per block in shared memory, after the first weight chunks are already in flight)
enum ProKind { PRO_NONE = 0, PRO_ARNORM = 1, PRO_SILU = 2, PRO_GNORM = 3, PRO_LEADER = 4 };
struct ProArgs {
    // PRO_ARNORM: [wait flag >= epoch(idx)] x = h_in (+ own + rx); block 0 writes h_out = x and xn_out = norm(x) * nw
    const float* h_in = nullptr;
    float* h_out = nullptr;
    const float* own = nullptr;     // already offset to the slot
    const float* rx = nullptr;      // already offset to the slot
    const unsigned* flag = nullptr; // already offset to the slot; nullptr = no wait (layer 0, or pulled already)
    int add = 0;                    // add own + rx (0 for layer 0)
    StepState* st = nullptr;
    int idx = 0;
    const float* nw = nullptr;
    float* xn_out = nullptr;
    // PRO_SILU: x = silu(gu[i]) * gu[K + i]
    const float* gu = nullptr;
    // PRO_GNORM: x = rmsnorm128(o) * gw * silu(z)
    const float *o = nullptr, *z = nullptr, *gw = nullptr;
    // PRO_LEADER (= ARNORM done once): block 0 [publishes this GPU's partial,] waits, builds x and writes it to
    // gxq/gxm (+ xn_out fp32), then sets *xflag = epoch; the other blocks (co-resident, first weight chunks already in
    // flight) wait on xflag and copy x from L2
    unsigned* xflag = nullptr;
    int8_t* gxq = nullptr;
    int2* gxm = nullptr;
    float* pub_peer_rx = nullptr;     // slot base; nullptr with pub_peer_flag set: flag only
    unsigned* pub_peer_flag = nullptr;  // slot-offset peer flag; nullptr: no publish
    // silu-quant epilogue (gate|up rows interleaved by RPL per tile): q8(silu(g) * u) -> sq_xq / sq_xm
    int8_t* sq_xq = nullptr;
    int2* sq_xm = nullptr;
    // extra blocks after the work blocks (dispatched last, i.e. in the kernel's tail) prefetch the next GEMV's
    // first-wave chunks into L2 (option pf_kb with the chunk-major layout)
    Pf pf_next;
    int pull = 0;  // host only: the fallback pull kernel is deferred to the GEMV launch (so it can prefetch)
};

// persistent per-layer kernels (option mega): phases separated by grid barriers on a co-resident grid
struct MegaCommon {
    StepState* st = nullptr;
    unsigned* bar = nullptr;   // [2] grid barrier counter + generation (local)
    unsigned* xrdy = nullptr;  // x-ready flag written by the leader block (local)
    int xid = 0;               // phase ordinal within the step for x-ready values (layer * 8 + k)
    ProArgs in;                // AR-in + norm for the first GEMV (leader block 0); flag/publish as in PRO_LEADER
    int8_t* gxq = nullptr;     // global x staging (q8)
    int2* gxm = nullptr;
    float* xn = nullptr;       // global fp32 normalized x
};
struct MegaDN {                // DeltaNet layer: [AR+norm] qkvz(+ab) | gdn | gated norm -> ssm_out (+rows to peer)
    MegaCommon c;
    t4q::gemv::GemvArgs qkvz, ssm_out;
    SegArgs ab;
    const float *ring_w, *ssm_a, *ssm_dt, *ssm_norm;
    float *ring, *S, *y, *yab, *o;
    float* y_peer;             // peer rx slot for ssm_out rows
};
struct MegaFFN {               // [AR+norm] gate|up (+silu q8 epilogue) | down (+rows to peer)
    MegaCommon c;
    t4q::gemv::GemvArgs gateup, down;
    int8_t* xq2;
    int2* xm2;
    float* y_peer;
};
struct MegaAttn {              // [AR+norm] q|k|v | prep | split | combine -> attn_output (+rows to peer)
    MegaCommon c;
    t4q::gemv::GemvArgs qkv, wo;
    const float *qw, *kw;
    float *ya, *qa, *ws;
    uint16_t *kc, *vc;
    int max_ctx;
    float theta_scale;
    float* y_peer;
};
struct MegaHead {              // [AR+norm] lm_head | argmax partials | final + exchange + state update
    MegaCommon c;
    t4q::gemv::GemvArgs lm;
    float *logits, *apart, *amb, *peer_amb;
    const unsigned* aflag;
    unsigned* peer_aflag;
    int row0;
    int* ring;
};

struct Layer {
    bool attn = false;
    float *attn_norm = nullptr, *post_norm = nullptr;
    // DeltaNet: qkvz rows = q (8 k-heads x 128) | k (1024) | v (24 v-heads x 128) | z (3072)
    FW qkvz, ssm_out;
    float* ab = nullptr;      // F32 [48][5120]: alpha rows of the 24 local heads, then beta rows
    int8_t* ab8r = nullptr;   // prefill R512: rotated int8 ab rows padded to [256][5120] (pf_abq)
    float* ab8i = nullptr;    // their invr [256]
    float* conv_w = nullptr;  // [5120][4]
    float *ssm_a = nullptr, *ssm_dt = nullptr, *ssm_norm = nullptr;
    float* conv_ring = nullptr;  // [4][5120] raw conv inputs, slot = pos & 3
    float* S = nullptr;          // [24][128 v col][128 k]
    // attention: qkv_a rows = q|gate of 12 local heads (6144) | k (2 heads, 512) | v (512)
    FW qkv_a, wo;
    float *q_norm = nullptr, *k_norm = nullptr;
    uint16_t *kc = nullptr, *vc = nullptr;  // [2][max_ctx][256] fp16
    // FFN: gateup rows = gate shard (8704) | up shard (8704)
    FW gateup, down;
};

struct Mtp {  // MTP block (blk.64 of the main GGUF) sharded like a full-attention layer, plus the draft lm_head
    Layer L;      // attn_norm, post_norm, qkv_a, wo, q_norm, k_norm, kc/vc (MTP KV cache), gateup, down
    FW eh;        // nextn.eh_proj K-split: GPU g multiplies its 5120 input columns (GPU0: enorm(embed), GPU1: hnorm(h))
    float *enorm = nullptr, *hnorm = nullptr, *shnorm = nullptr;
    FW lmd;       // draft head: output.weight rows [dv/2 * g, +dv/2) (the first dv token ids)
    int dv = 0;   // draft vocab (0: draft with the full lm_head)
};

struct Gpu {
    int g = 0;
    cudaStream_t s = nullptr;
    Layer L[64];
    float* output_norm = nullptr;
    FW lm;
    Mtp mtp;
    uint8_t* embd = nullptr;  // raw Q4_0 token_embd rows
    // activations
    float* hb[2];  // residual double buffer: after AR idx the residual is in hb[(idx + 1) & 1]
    float *xn, *y, *yab, *o, *qa, *attn_ws, *logits;
    int8_t* xq;
    int2* xm;
    int8_t* xq2;  // q8 x for ffn_down (written by the gate|up epilogue)
    int2* xm2;
    float* part;     // [2][5120] own AR partials
    float* rx;       // [2][5120] peer writes its partials here
    unsigned* flag;  // [2] peer-written AR flags, [2..3] argmax flags
    unsigned* cnt;   // [4] block counters
    unsigned* xflag; // leader-prologue x-ready flag (local)
    unsigned* mbar;  // [8] mega kernels: grid barrier counter/generation, x-ready flag
    float* amb;      // [2 slots][2] argmax mailbox (val, idx bits), peer-written
    float* apart;    // argmax partials [2][NB]
    StepState* st;
    int* prompt;     // [max_ctx]
    // peer pointers
    float* peer_rx = nullptr;
    unsigned* peer_flag = nullptr;
    float* peer_amb = nullptr;
    // host-mapped mailbox (fallback when the GPUs have no P2P): the peer writes here, a pull kernel copies to rx
    float2* rxl = nullptr;       // [2][5120] LL mailbox {value, epoch tag} (option ll), peer-written
    float2* peer_rxl = nullptr;
    unsigned* gcnt = nullptr;    // [32] per-head block counters of the fused gdn + gated norm kernel
    unsigned* tflag = nullptr;   // [2][TFLAGS] AR tail flags (peer-written), option tail
    unsigned* htflag = nullptr;  // host-mapped [2][TFLAGS] slice flags (no-P2P pull_norm), peer-written
    unsigned* peer_htflag = nullptr;
    float2* ssb = nullptr;       // [2][32] tagged per-block sums of squares (pull_norm)
    unsigned* peer_tflag = nullptr;
    float* hrx = nullptr;       // [2][5120]
    unsigned* hflag = nullptr;  // [8]: [0..1] AR, [2..3] argmax
    float* hamb = nullptr;      // [4]
    float* scratch = nullptr;   // [5120 + 64] AR test target (self-test timing), local VRAM
    // graphs
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t gexec = nullptr;
    // profiling
    std::vector<cudaEvent_t> ev;
    std::vector<const char*> ev_name;
};

struct ShardSpec {  // how a FW was gathered from GGUF tensors (kept for the self-test)
    std::string what;
    std::vector<std::pair<const void*, std::pair<int64_t, int64_t>>> rows;  // (GgufTensor*, (row0, nrows))
    std::vector<std::pair<int64_t, int64_t>> cols;                        // (col0, ncols) in elements
    int interleave = 0;  // > 0: two equal row pieces interleaved in groups of this many rows
};

struct State {
    Gpu G[2];
    int max_ctx = 4096;
    bool graphs = false;
    bool profile = false;
    int* h_ring = nullptr;  // host-mapped [RING] (written by GPU0 argmax_final)
    int* d_ring = nullptr;  // device view of h_ring
    std::vector<std::pair<int, ShardSpec>> specs;  // (gpu, spec) of weights to self-test
    std::vector<const FW*> spec_fw;
    std::string selftest_json;
    std::string prof_json;
    std::string trace_json;
    std::string dbg_json;
    int max_blocks = 80;  // co-resident GEMV blocks (2 per SM)
    bool p2p = true;      // false: host-mapped mailbox fallback
    float* hscratch[2] = {nullptr, nullptr};  // host-mapped AR test targets (self-test timing without P2P)
    int fuse = 0;         // 0: separate ar_norm kernel; 2: AR + RMSNorm prologue redundantly in every GEMV block;
                          // 3: leader-block prologue (block 0 does AR + norm, the others prefetch and wait)
    int mega = 0;         // 1: persistent per-layer kernels (MegaDN/FFN/Attn/Head) instead of per-op kernels
    int mega_grid = 80;
    int arpub = 2;        // 1: the consumer kernel publishes this GPU's partial (plain K-split GEMVs; default);
                          // 0: the K-split GEMV epilogue publishes (M2-M4 v5). fuse != 0 forces 0.
    int ll = 0;           // 1: LL all-reduce (tagged 8-byte rows from the K-split GEMVs, no flags; P2P, fuse 0 only)
    int gdnf = 1;         // 1: gdn and the gated norm q8 in one kernel (default since M4 round 2)
    int spin_ns = 0;      // spin-wait backoff
    int attnf = 0;        // 1: fused attention kernel (prep + split + combine)
    int attn2 = 0;        // 1: split attention v2 (slower than v1 in M4 v20: 27.6 vs 20.9 us short, 48 vs 48 at 3.6k)
    int sqt = 256;        // gate|up silu-quant GEMV block size (256 since M4 v21: gate|up 199 -> 195 us)
    int pn = 0;           // no P2P: 0 = pull + ar_norm kernels (default); 1 = pull_norm (slice publish; 34 vs 25 us
                          // per AR in v22); 2 = pull_arn (transport block + norm blocks; 30.6 vs 27.3 us in v26)
    int p4u = 1;          // 1: P4 GEMVs with the unsigned high-nibble dp4a path (default; bit-identical)
    int tail = 0;         // 1: AR + norm in tail blocks of the K-split GEMVs (no ar_norm kernels; P2P only)
    int arn = 0;          // ar_norm kernel: 0 = multi-block (20 x 256), 1 = single block (round 1)
    int arpub_auto = 2;   // load-time choice from the rows-to-peer cost (option arpub -1 restores it)
    double ar_rows_cost_us = 0;
    int pf_gemv = 0;      // 1: gate|up's tail blocks prefetch ffn_down's first wave (needs pf_kb > 0)
    int pf_kb = 0;        // L2 prefetch of the next GEMV during small kernels (0 = off; no gain in M4 v4)
    double ms_graph_capture = 0;
    // batched prefill (tp_prefill.cu)
    void* pf = nullptr;   // buffers
    int pf_on = 1;        // 1: t4q_prefill uses the batched path for n >= 2 (0: decode steps)
    int pf_ub = 2048;     // ubatch tokens
    int pf_i4 = 0;        // 1: int4 tensor-core path (m8n8k32, split int8 activations) for P4/P4M GEMMs
    int pf_fuse = 1;      // 1: norm / silu / gated norm quantize straight into the GEMM activation layout
    int pf_fa = 1;        // 1: tensor-core flash attention for prefill (0: SIMT online-softmax kernel)
    int pf_nsub = 2;      // sub-batches per ubatch (AR copy of one overlaps the other's compute); 1 = off
    int pf_arc = 1;       // K-split GEMMs (ssm_out, attn_out, down) in this many token chunks, each chunk's all-reduce
                          // copy starting as soon as it is done (1 = off; gemm8 path)
    int pf_nsub_min = 1024; // ubatches shorter than this run as one sub-batch (round 4: nsub 1 is faster at pp512)
    int pf_g8 = 1;        // 1: gemm8 (in-kernel per-row int8 requant, one FFMA per output per block); 0: gemm.cuh W4A8
    int pf_ga = 64;       // gemm8 activation scale group: 32 (exact q8 blocks), 64, or 0 (one scale per token)
    int pf_silu = 1;      // 1: gate|up GEMM epilogue writes q8(silu(gate) * up) for down (gemm8 GA 64 only)
    int pf_gdn2 = 0;
    int pf_ar16 = 1;      // 1: fp16 all-reduce partials (gemm8 path)
    int pf_gdnc = 1;      // 1: chunked DeltaNet scan on fp16 tensor cores (k_pf_gdnc); 2: + state-independent part (gates, P, T)
                          //    for all chunks in parallel first (k_pf_gdnp), so the sequential scan skips it
    int pf_gdnc_chk = 0;  // 1: first DeltaNet call of each batched prefill also runs the sequential scan and compares
    std::string pf_gdnc_json;      // 1: DeltaNet scan computes o_t and kv_{t+1} in one pass over the state (0: two passes)
    int pf_bn = 0;        // gemm8 token tile: 128 / 256, 0 = auto (256 when the sub-batch has >= 512 padded tokens)
    int pf_prof = 0;      // 1: per-op event profile of GPU0 (ms per op class) in stats "pf_profile"
    std::string pf_json;
    // activation-format accuracy study (tp_prefill.cu k_fq_*): emulate a GEMM input format on the GA64 int8 input
    int pf_h16 = 0;       // R512: silu(gate) * up written as fp16 (0: fp32; fp16 can overflow on massive-activation tokens)
    int pf_rgb = 0;       // R512 activations with one scale per (token, 512-block), folded by exact accumulator rescaling
    int pf_rot_mask = 63; // R512 GEMM types: 1 qkvz, 2 qkv_a, 4 gateup, 8 down, 16 ssm_out, 32 attn_out (others: GA64)
    int pf_rot_min = 0;   // R512 only for ubatches of at least this many tokens (smaller ones take the GA64 path)
    int pf_wcache = 0;    // R512: MB of spare VRAM per GPU for persistent rotated int8 weights (no per-ubatch conversion)
    long long pf_wcache_used = 0;
    int pf_rot_chk = 0;   // R512: first conversion per weight slot also runs the fp32 converter and compares (stats pf_gdnc_check)
    int pf_rcf = 1;       // R512: fp16x2 weight rotation for P4 / P4M (0: fp32 converter)
    int pf_abq = 1;       // R512: alpha/beta projection as a rotated int8 tensor-core GEMM (rows padded to 256)
    int pf_rot = 0;       // 1: R512 path: block-Hadamard rotated per-token activations x rotated int8 weights (gemm17)
    int pf_g17 = 0;       // 1: gemm17 (CUTLASS-style int8 pipeline, shift-folded per-64 activation groups, unfused producers)
    int pf_emax = 7;      // gemm17: max group exponent (group step = D_t 2^-e / 127)
    int pf_fq = 0;        // 0 off; 1 per-token; 2 per-token clip a*rms + exact residual; 3 top-n channels exact + per-token;
                          // 4 = 3 + clip; 5 per-group of pf_fq_n elements; 6 random signs + block Hadamard of
                          // pf_fq_n, then per-token (clip if pf_fq_a); 8 per-channel smoothing (sqrt amax) + per-token;
                          // 9 per-token scale x 2^-e per pf_fq_n group, e <= pf_fq_a (shift-foldable)
    int pf_fq_a = 0;      // clip multiple of the token rms, in tenths (mode 2/4)
    int pf_fq_n = 0;      // top-n channels (mode 3/4) or group size (mode 5)
    int pf_fq_mask = 63;  // GEMM types: 1 qkvz, 2 attn_qkv, 4 gateup, 8 down, 16 ssm_out, 32 attn_out
    int pf_head = 1;      // 1: the last prompt token also goes through the batch path; only the head (norm, lm_head,
                          //    argmax) runs decode-style afterwards (0: the last token runs a full decode step)
    int pf_keep_h = 0;    // 1: copy GPU0's final residual of every batch token into dumps["pf_h"] (accuracy studies)
    std::string pf_fq_json;
    double pf_last_batch_s = 0, pf_last_total_s = 0;
    int pf_h_pos0 = 0, pf_h_n = 0;  // rows of the last prefill ubatch's final residuals (tp_prefill_hrows)
    int pf_last_n = 0;
    // batched decode (milestone B, tp_batch.cuh): B concurrent sequences in slots, each with its own DeltaNet state,
    // conv ring and KV cache; GEMMs at m = B through gemm9 (64-token tile up to B = 64)
    void* bd = nullptr;   // slot buffers (tp_batch_init)
    int bd_head = 1;      // lm_head: 1 = gemm9 on Q6_K (per-row int8 requant, GA64 activations); 0 = dp4a GEMV, 8 columns per pass
    int bd_ch = 128;      // attention positions per split block (fixed, so results do not depend on B)
    int bd_prof = 0;      // 1: per-op event profile of GPU0 per step, accumulated in stats "bd_profile"
    int bd_pfk = 0;       // gemm9 L2 prefetch distance of the weight planes in 32-blocks (0 = off) for batched decode
    int bd_gemmr = 0;     // P4 GEMMs at 32 / 64-token tiles via gemm_r.cuh (register-direct A fragments, bit-identical to gemm9)
    int bd_lbm = 1;       // gemm9 at 32/64-token tiles: ring of 2-3 stages of global loads in flight (0: one stage)
    int bd_ksplit = 1;    // split-K slices for the K-split GEMMs (down, ssm_out, attn_output: N = 5120 gives 40 blocks)
    int bd_p2p = 0;       // 1: K-split GEMM epilogues store the fp16 all-reduce partials into the peer's mailbox (P2P)
                          //    instead of a copy-engine transfer after the GEMM (needs P2P)
    std::string bd_json;
    // speculative decoding (milestone M5, tp_spec.cu)
    void* spec = nullptr; // buffers / graphs (spec_init)
    int spec_k = 0;       // drafts per step (0: plain decode in t4q_generate; 1..MMAX-1)
    int spec_dv = 1;      // 1: drafts use the truncated head (Mtp::lmd), 0: the full lm_head
    int spec_force = 0;   // debug: drafts are the continuation set with t4q_spec_force (acceptance 100% if correct)
    int spec_dbg = 0;     // debug: copy the accepted verify columns' logits to dumps["spec_logits"] every step
    int spec_prof = 0;    // 1: one iteration in flight, per-graph GPU0 event times (stats spec.ms_draft / ms_verify)
    int spec_ahead = 4;   // spec iterations enqueued before the host reads the token ring
    std::string spec_json;
};

}  // namespace tp
