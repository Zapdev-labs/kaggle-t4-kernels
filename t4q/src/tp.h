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
    int fence = 2;                  // per-block fence before counting: 2 = system (correct), 1 = gpu, 0 = none (tests)
};

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
enum ProKind { PRO_NONE = 0, PRO_ARNORM = 1, PRO_SILU = 2, PRO_GNORM = 3 };
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
    float* part;     // [2][5120] own AR partials
    float* rx;       // [2][5120] peer writes its partials here
    unsigned* flag;  // [2] peer-written AR flags, [2..3] argmax flags
    unsigned* cnt;   // [4] block counters
    float* amb;      // [2 slots][2] argmax mailbox (val, idx bits), peer-written
    float* apart;    // argmax partials [2][NB]
    StepState* st;
    int* prompt;     // [max_ctx]
    // peer pointers
    float* peer_rx = nullptr;
    unsigned* peer_flag = nullptr;
    float* peer_amb = nullptr;
    // host-mapped mailbox (fallback when the GPUs have no P2P): the peer writes here, a pull kernel copies to rx
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
    int max_blocks = 80;  // co-resident GEMV blocks (2 per SM)
    bool p2p = true;      // false: host-mapped mailbox fallback
    float* hscratch[2] = {nullptr, nullptr};  // host-mapped AR test targets (self-test timing without P2P)
    int fuse = 0;         // 0: separate ar_norm / gnorm_q8 / silu_q8 kernels (fastest in M4 v3); 1: all prologues fused;
                          // 2: only the AR + RMSNorm prologue fused
    int pf_kb = 0;        // L2 prefetch of the next GEMV during small kernels (0 = off; no gain in M4 v4)
    double ms_graph_capture = 0;
};

}  // namespace tp
