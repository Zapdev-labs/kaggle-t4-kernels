// Internal engine state (M1: layer split, GPU0 = layers 0..31, GPU1 = layers 32..63 + head).
#pragma once
#include <cuda_runtime.h>

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/t4q.h"
#include "gguf.h"
#include "packed.h"

#define CK(x)                                                                                              \
    do {                                                                                                   \
        cudaError_t e_ = (x);                                                                              \
        if (e_ != cudaSuccess)                                                                             \
            throw std::runtime_error(std::string("CUDA error ") + cudaGetErrorString(e_) + " at " + __FILE__ + \
                                     ":" + std::to_string(__LINE__) + ": " #x);                            \
    } while (0)

namespace hp {
constexpr int D = 5120, NL = 64, FF = 17408, V = 248320;
constexpr int HQ = 24, HKV = 4, HD = 256, NROT = 64;
constexpr int HK = 16, HV = 48, DK = 128, CONV = 10240, VDIM = 6144;
constexpr float EPS = 1e-6f, ROPE_BASE = 1e7f;
inline bool is_attn(int il) { return (il + 1) % 4 == 0; }
}  // namespace hp

struct Layer {
    int il = 0, gpu = 0;
    bool attn = false;
    float* attn_norm = nullptr;
    float* post_norm = nullptr;
    // DeltaNet
    PackedW qkv, z, alpha, beta, ssm_out;
    float *conv_w = nullptr, *ssm_a = nullptr, *ssm_dt = nullptr, *ssm_norm = nullptr;
    float *conv_state = nullptr, *S = nullptr;
    // attention
    PackedW wq, wk, wv, wo;
    float *q_norm = nullptr, *k_norm = nullptr;
    uint16_t *kc = nullptr, *vc = nullptr;
    // FFN
    PackedW gate, up, down;
};

struct Scratch {
    float *h, *xn, *a, *qkv, *z, *braw, *araw, *beta, *g, *conv, *qn, *kn, *o, *on;
    float *qfull, *k, *v, *aq, *ak, *att, *attg, *ffg, *ffu, *ffa, *scores, *logits;
    int8_t* xq;          // q8_1 activations (act_q8 mode)
    float *xd, *xs;
};

struct RepackStats {
    int tensors = 0, checked = 0, mismatched = 0;
    double max_abs_diff = 0;
    std::string first_bad;
};

struct t4q_ctx {
    GgufFile f;
    t4q_params params{};
    int split = 32;  // first layer on GPU1
    Layer layers[hp::NL];
    float* output_norm = nullptr;
    PackedW output;
    const GgufTensor* tok_embd = nullptr;
    cudaStream_t st[2] = {nullptr, nullptr};
    Scratch sc[2];
    int pos = 0;
    int max_ctx = 4096;
    float* h_emb = nullptr;      // pinned [D]
    float* h_logits = nullptr;   // pinned [V], logits of the last step
    bool have_logits = false;
    bool dump_on = false;
    bool act_q8 = false;  // llama.cpp-style q8_1 activation quantization in the GEMVs (validation mode)
    std::map<std::string, std::vector<float>> dumps;
    RepackStats rstats;
    double load_s = 0, gen_s = 0;
    long gen_tokens = 0, steps = 0;
    double step_s = 0;
    size_t vram_used[2] = {0, 0};
};

// loader.cu
void load_model(t4q_ctx* c, const char* path);
void free_model(t4q_ctx* c);
// engine.cu
void engine_step(t4q_ctx* c, int token);   // one token at c->pos; leaves logits in c->h_logits; pos++
void engine_reset(t4q_ctx* c);
// debug/validation: run one layer at `pos` on a host residual input (uses and updates that layer's state)
void engine_layer(t4q_ctx* c, int il, int pos, const float* h_in, float* h_out);
