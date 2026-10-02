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
};

struct FW {  // fast packed weight on one GPU
    t4q::gemv::Layout L;
    uint8_t* base = nullptr;
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

struct Gpu {
    int g = 0;
    cudaStream_t s = nullptr;
    Layer L[64];
    float* output_norm = nullptr;
    FW lm;
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
    int p4u = 1;          // 1: P4 GEMVs with the unsigned high-nibble dp4a path (default; bit-identical)
    int tail = 0;         // 1: AR + norm in tail blocks of the K-split GEMVs (no ar_norm kernels; P2P only)
    int arn = 0;          // ar_norm kernel: 0 = multi-block (20 x 256), 1 = single block (round 1)
    int pf_kb = 0;        // L2 prefetch of the next GEMV during small kernels (0 = off; no gain in M4 v4)
    double ms_graph_capture = 0;
};

}  // namespace tp
