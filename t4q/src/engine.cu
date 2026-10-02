// M1 decode step: unfused kernels, layer split across the two T4s, eager launches, host sync per token.
#include <chrono>
#include <cmath>
#include <cstring>

#include "kernels/kernels.h"
#include "model.h"
#include "quant_cpu.h"

using namespace hp;

namespace {

void dump(t4q_ctx* c, int gpu, const char* name, int il, const float* dptr, size_t n) {
    if (!c->dump_on) return;
    std::string key = il >= 0 ? std::string(name) + "-" + std::to_string(il) : std::string(name);
    std::vector<float>& v = c->dumps[key];
    v.resize(n);
    CK(cudaSetDevice(gpu));
    CK(cudaStreamSynchronize(c->st[gpu]));
    CK(cudaMemcpy(v.data(), dptr, n * 4, cudaMemcpyDeviceToHost));
}

// y = W x; in act_q8 mode quantized weights see llama.cpp-style q8_1 activations
void gemv(t4q_ctx* c, const PackedW& W, const float* x, float* y, Scratch& s, cudaStream_t st) {
    if (c->act_q8 && W.fmt != FMT_F32) {
        launch_quantize_q8_1(x, (int)W.cols, s.xq, s.xd, s.xs, st);
        launch_gemv_q8(W, s.xq, s.xd, s.xs, y, st);
    } else {
        launch_gemv(W, x, y, st);
    }
}

void check_launch(const char* what) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("launch failed: ") + what + ": " + cudaGetErrorString(e));
}

void deltanet(t4q_ctx* c, Layer& L, Scratch& s, cudaStream_t st) {
    const int il = L.il, g = L.gpu;
    gemv(c, L.qkv, s.xn, s.qkv, s, st);
    gemv(c, L.z, s.xn, s.z, s, st);
    gemv(c, L.beta, s.xn, s.braw, s, st);
    gemv(c, L.alpha, s.xn, s.araw, s, st);
    check_launch("gdn proj");
    dump(c, g, "linear_attn_qkv_mixed", il, s.qkv, CONV);
    dump(c, g, "z", il, s.z, VDIM);
    launch_gdn_gates(s.braw, s.araw, L.ssm_a, L.ssm_dt, s.beta, s.g, HV, st);
    dump(c, g, "beta_sigmoid", il, s.beta, HV);
    dump(c, g, "gate", il, s.g, HV);
    launch_gdn_conv(s.qkv, L.conv_state, L.conv_w, s.conv, CONV, st);
    dump(c, g, "conv_output_silu", il, s.conv, CONV);
    launch_gdn_l2(s.conv, s.qn, s.kn, EPS, st);
    dump(c, g, "q_conv_predelta", il, s.qn, HK * DK);
    dump(c, g, "k_conv_predelta", il, s.kn, HK * DK);
    launch_gdn_recur(L.S, s.qn, s.kn, s.conv + 2 * HK * DK, s.beta, s.g, s.o, 1.0f / sqrtf((float)DK), st);
    check_launch("gdn recur");
    dump(c, g, "attn_output", il, s.o, VDIM);
    if (c->dump_on && il == 0) dump(c, g, "ssm_state", il, L.S, (size_t)HV * DK * DK);
    launch_gdn_gnorm(s.o, s.z, L.ssm_norm, s.on, HV, EPS, st);
    dump(c, g, "final_output", il, s.on, VDIM);
    gemv(c, L.ssm_out, s.on, s.a, s, st);
    check_launch("ssm_out");
    dump(c, g, "linear_attn_out", il, s.a, D);
}

void attention(t4q_ctx* c, Layer& L, Scratch& s, cudaStream_t st) {
    const int il = L.il, g = L.gpu;
    gemv(c, L.wq, s.xn, s.qfull, s, st);
    gemv(c, L.wk, s.xn, s.k, s, st);
    gemv(c, L.wv, s.xn, s.v, s, st);
    check_launch("attn proj");
    dump(c, g, "Qcur_full", il, s.qfull, HQ * HD * 2);
    dump(c, g, "Kcur_raw", il, s.k, HKV * HD);
    dump(c, g, "Vcur", il, s.v, HKV * HD);
    launch_qk_norm_rope(s.qfull, s.k, L.q_norm, L.k_norm, s.aq, s.ak, c->pos, EPS, ROPE_BASE, NROT, st);
    dump(c, g, "Qcur", il, s.aq, HQ * HD);
    dump(c, g, "Kcur", il, s.ak, HKV * HD);
    launch_kv_store(s.ak, s.v, L.kc, L.vc, c->pos, c->max_ctx, st);
    launch_attn_decode(s.aq, L.kc, L.vc, s.att, s.scores, c->pos + 1, c->max_ctx, 1.0f / 16.0f, st);
    check_launch("attn");
    dump(c, g, "attn_pregate", il, s.att, HQ * HD);
    launch_gate_sigmoid(s.att, s.qfull, s.attg, st);
    dump(c, g, "attn_gated", il, s.attg, HQ * HD);
    gemv(c, L.wo, s.attg, s.a, s, st);
    check_launch("attn out");
    dump(c, g, "attn_output", il, s.a, D);
}

void run_layer(t4q_ctx* c, Layer& L) {
    const int g = L.gpu, il = L.il;
    Scratch& s = c->sc[g];
    cudaStream_t st = c->st[g];
    launch_rmsnorm(s.h, L.attn_norm, s.xn, D, EPS, st);
    dump(c, g, "attn_norm", il, s.xn, D);
    if (L.attn) attention(c, L, s, st);
    else deltanet(c, L, s, st);
    launch_add(s.h, s.a, D, st);
    dump(c, g, "attn_residual", il, s.h, D);
    launch_rmsnorm(s.h, L.post_norm, s.xn, D, EPS, st);
    dump(c, g, "attn_post_norm", il, s.xn, D);
    gemv(c, L.gate, s.xn, s.ffg, s, st);
    gemv(c, L.up, s.xn, s.ffu, s, st);
    launch_silu_mul(s.ffg, s.ffu, s.ffa, FF, st);
    gemv(c, L.down, s.ffa, s.a, s, st);
    check_launch("ffn");
    dump(c, g, "ffn_out", il, s.a, D);
    launch_add(s.h, s.a, D, st);
    dump(c, g, "l_out", il, s.h, D);
}

}  // namespace

void engine_step(t4q_ctx* c, int token) {
    if (token < 0 || token >= V) throw std::runtime_error("token id out of range");
    if (c->pos >= c->max_ctx) throw std::runtime_error("context full");
    auto t0 = std::chrono::steady_clock::now();
    if (c->dump_on) c->dumps.clear();
    // embedding: dequantize the row on the host (exact), upload to GPU0
    const GgufTensor* te = c->tok_embd;
    if (!dequant_row_cpu(te->type, te->data + (size_t)token * te->row_bytes, c->h_emb, D))
        throw std::runtime_error("embedding dequant failed");
    CK(cudaSetDevice(0));
    CK(cudaMemcpyAsync(c->sc[0].h, c->h_emb, D * 4, cudaMemcpyHostToDevice, c->st[0]));
    for (int il = 0; il < NL; il++) {
        Layer& L = c->layers[il];
        if (il == c->split) {
            CK(cudaSetDevice(0));
            CK(cudaMemcpyPeerAsync(c->sc[1].h, 1, c->sc[0].h, 0, D * 4, c->st[0]));
            CK(cudaStreamSynchronize(c->st[0]));
        }
        CK(cudaSetDevice(L.gpu));
        run_layer(c, L);
    }
    Scratch& s = c->sc[1];
    cudaStream_t st = c->st[1];
    CK(cudaSetDevice(1));
    launch_rmsnorm(s.h, c->output_norm, s.xn, D, EPS, st);
    dump(c, 1, "result_norm", -1, s.xn, D);
    gemv(c, c->output, s.xn, s.logits, s, st);
    check_launch("lm_head");
    CK(cudaMemcpyAsync(c->h_logits, s.logits, (size_t)V * 4, cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaSetDevice(0));
    CK(cudaStreamSynchronize(c->st[0]));
    c->have_logits = true;
    c->pos++;
    c->steps++;
    c->step_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void engine_reset(t4q_ctx* c) {
    for (int il = 0; il < NL; il++) {
        Layer& L = c->layers[il];
        CK(cudaSetDevice(L.gpu));
        if (L.attn) {
            // KV beyond pos is never read; no clear needed
        } else {
            CK(cudaMemset(L.conv_state, 0, (size_t)CONV * 3 * 4));
            CK(cudaMemset(L.S, 0, (size_t)HV * DK * DK * 4));
        }
    }
    for (int g = 0; g < 2; g++) { CK(cudaSetDevice(g)); CK(cudaDeviceSynchronize()); }
    c->pos = 0;
    c->have_logits = false;
}
