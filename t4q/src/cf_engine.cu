// cf-m1: the CYBER-FROST decode step. Exact decode math, one host sync per token
// (the router top-k comes back host-side, as do the logits). Single GPU for M1.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "cf_model.h"
#include "quant_cpu.h"

using namespace cf;

namespace {

void gemv(const PackedW& W, const float* x, float* y, cudaStream_t st) { launch_gemv(W, x, y, st); }

void check_launch(const char* what) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("launch failed: ") + what + ": " + cudaGetErrorString(e));
}

// res_hc -> xn (grouped norm), lo, gate(y_up), mixed, inj. side 0 = attn, 1 = ffn.
// The final output mixer passes with_inject = false (no scatter weights).
void hc_mix(CfCtx* c, const float* res, const float* w_norm, const PackedW& down, const PackedW& up,
            const PackedW* inject, CfScratch& s) {
    cudaStream_t st = c->st;
    launch_cf_hc_norm(res, w_norm, s.xn, st);
    gemv(down, s.xn, s.lo, st);
    launch_cf_hc_lo(s.lo, s.lo, st);  // in-place: reads y[i] writes lo[i], identity-safe
    gemv(up, s.lo, s.gate, st);       // gate buffer = y_up [HCD]
    launch_cf_hc_mixed(s.xn, s.gate, s.mixed, st);
    if (inject) gemv(*inject, s.xn, s.inj, st);  // [HC]
}

// res += 2*sigmoid(inj/HC) * block
void hc_combine(CfCtx* c, float* res, const float* block, const float* inj) {
    launch_cf_hc_combine(res, block, inj, c->st);
}

void deltanet(CfCtx* c, CfLayer& L, CfScratch& s) {
    cudaStream_t st = c->st;
    gemv(L.qkv, s.xn, s.qkv, st);
    gemv(L.z, s.xn, s.zz, st);
    gemv(L.beta, s.xn, s.braw, st);
    gemv(L.alpha, s.xn, s.araw, st);
    check_launch("gdn proj");
    launch_gdn_gates(s.braw, s.araw, L.ssm_a, L.ssm_dt, s.beta, s.g, HV, st);
    launch_gdn_conv(s.qkv, L.conv_state, L.conv_w, s.conv, CONV, st);
    launch_gdn_l2(s.conv, s.qn, s.kn, EPS, st);
    launch_gdn_recur(L.S, s.qn, s.kn, s.conv + 2 * HK * DK, s.beta, s.g, s.o, 1.0f / sqrtf((float)DK), st);
    check_launch("gdn recur");
    launch_cf_gdn_gnorm(s.o, s.zz, L.ssm_norm, s.on, EPS, st);
    gemv(L.ssm_out, s.on, s.block, st);
    check_launch("ssm_out");
}

void attention(CfCtx* c, CfLayer& L, CfScratch& s, int pos) {
    cudaStream_t st = c->st;
    gemv(L.wq, s.xn, s.qfull, st);
    gemv(L.wk, s.xn, s.k, st);
    gemv(L.wv, s.xn, s.v, st);
    check_launch("attn proj");
    launch_cf_qk_norm_rope(s.qfull, s.k, L.q_norm, L.k_norm, s.aq, s.ak, pos, EPS, ROPE_BASE, NROT, st);
    launch_cf_kv_store(s.ak, s.v, L.kc, L.vc, pos, c->max_ctx, st);
    launch_cf_attn_decode(s.aq, L.kc, L.vc, s.att, s.scores, pos + 1, c->max_ctx, 1.0f / 16.0f, st);
    check_launch("attn");
    launch_gate_sigmoid(s.att, s.qfull, s.attg, st);  // 24 blocks, [24][512]: same layout
    gemv(L.wo, s.attg, s.block, st);
    check_launch("attn out");
}

// The PLE n-gram hash gather + key/value/norms/s-gate/conv. Runs before layer 1's hc mix.
void ple(CfCtx* c, int token) {
    CfScratch& s = c->sc;
    cudaStream_t st = c->st;
    // host: the u64 hash. ctx[0] = the token; predecessors cut at EOS/missing (missing reads EOS).
    const int pos = c->pos;
    int64_t ctx[PLE_NGRAM];
    ctx[0] = token;
    bool cut = false;
    for (int k = 1; k < PLE_NGRAM; k++) {
        const int t = pos - k >= 0 ? c->toks[pos - k] : -1;
        cut = cut || t < 0 || t == EOS1;
        ctx[k] = cut ? EOS1 : t;
    }
    for (int n = 2; n <= PLE_NGRAM; n++) {
        uint64_t mixed = (uint64_t)ctx[0] * c->ple.mult[0];
        for (int j = 1; j < n; j++) mixed ^= (uint64_t)ctx[j] * c->ple.mult[j];
        for (int g = 0; g < PLE_NHEADS / 2; g++) {
            const int h = (n - 2) * (PLE_NHEADS / 2) + g;
            const uint64_t row = mixed % c->ple.head_vocab[h] + c->ple.head_off[h];
            if (!dequant_row_cpu(GT_Q4_0, c->ple.table->data + (size_t)row * c->ple.table->row_bytes,
                                 c->h_ple + (size_t)h * PLE_DIM, PLE_DIM))
                throw std::runtime_error("ple row dequant failed");
        }
    }
    CK(cudaMemcpyAsync(s.mixed, c->h_ple, (size_t)D * 4, cudaMemcpyHostToDevice, st));
    gemv(c->ple.key, s.mixed, c->ple_key, st);
    gemv(c->ple.value, s.mixed, s.block, st);
    // grouped norms: key in place, query from the current wide residual
    launch_cf_hc_norm(c->ple_key, c->ple.norm_key, c->ple_key, st);
    launch_cf_hc_norm(s.h, c->ple.norm_query, c->ple_query, st);
    launch_cf_ple_sg(c->ple_key, c->ple_query, c->ple_s, c->ple_gate, st);
    launch_cf_ple_gated(s.block, c->ple_gate, c->ple_gated, st);  // gated = value * gate[s]
    // conv over the normed gated, then res += gated + conv (exact add order)
    launch_cf_hc_norm(c->ple_gated, c->ple.norm_conv, c->ple_key, st);  // reuse the dead key buffer
    launch_cf_ple_conv(c->ple_key, c->ple_hist, c->ple.conv_w, c->ple_query, st);
    launch_add(c->ple_query, c->ple_gated, HCD, st);  // t = gated + conv
    launch_add(s.h, c->ple_query, HCD, st);           // res += t
}

// The MoE: router -> host softmax/top-10/renorm -> stage 10 experts -> repack -> gemv -> combine
void moe(CfCtx* c, CfLayer& L, CfScratch& s) {
    cudaStream_t st = c->st;
    gemv(L.router, s.xn, s.logits, st);  // borrow the logits scratch [512 of V]
    CK(cudaMemcpyAsync(c->h_router, s.logits, (size_t)NE * 4, cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    // host softmax over 512 (fp32, exact formula), then top-10 renormalized
    float m = c->h_router[0];
    for (int i = 1; i < NE; i++) m = std::max(m, c->h_router[i]);
    float sum = 0.f;
    for (int i = 0; i < NE; i++) {
        c->h_router[i] = std::exp(c->h_router[i] - m);
        sum += c->h_router[i];
    }
    for (int i = 0; i < NE; i++) c->h_router[i] /= sum;
    for (int k = 0; k < TOPK; k++) c->eid[k] = -1;
    for (int k = 0; k < TOPK; k++) {
        int best = -1;
        for (int i = 0; i < NE; i++) {
            if (c->h_router[i] < 0.f) continue;  // already picked
            if (best < 0 || c->h_router[i] > c->h_router[best]) best = i;
        }
        c->eid[k] = best;
        c->we_h[k] = c->h_router[best];
        c->h_router[best] = -1.f;
    }
    float wsum = 0.f;
    for (int k = 0; k < TOPK; k++) wsum += c->we_h[k];
    for (int k = 0; k < TOPK; k++) c->we_h[k] /= wsum;
    if (c->census_f) {  // cf-m2: the router concentration census, same round as the gate
        fwrite(c->eid, 4, TOPK, c->census_f);
        fwrite(c->we_h, 4, TOPK, c->census_f);
    }
    // stage the 10 experts' slabs: [gate 640 | up 640] per expert (Q2_K), then 2560 down rows (Q4_0)
    const size_t gu_row = L.t_gate_exps->row_bytes;   // 840
    const size_t dn_row = L.t_down_exps->row_bytes;   // 360
    const size_t up_bytes = (size_t)TOPK * 2 * EE * gu_row;
    for (int k = 0; k < TOPK; k++) {
        const int64_t e = c->eid[k];
        uint8_t* dst = c->raw_stage + (size_t)k * 2 * EE * gu_row;
        memcpy(dst, L.t_gate_exps->data + (size_t)e * EE * gu_row, (size_t)EE * gu_row);
        memcpy(dst + (size_t)EE * gu_row, L.t_up_exps->data + (size_t)e * EE * gu_row, (size_t)EE * gu_row);
        memcpy(c->raw_stage + up_bytes + (size_t)k * D * dn_row, L.t_down_exps->data + (size_t)e * D * dn_row,
               (size_t)D * dn_row);
    }
    CK(cudaMemcpyAsync(c->raw_dev, c->raw_stage, up_bytes + (size_t)TOPK * D * dn_row, cudaMemcpyHostToDevice, st));
    launch_repack(c->up_stage, GT_Q2_K, c->raw_dev, 0, (int64_t)TOPK * 2 * EE, st);
    launch_repack(c->dn_stage, GT_Q4_0, c->raw_dev + up_bytes, 0, (int64_t)TOPK * D, st);
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(c->we, c->we_h, (size_t)TOPK * 4, cudaMemcpyHostToDevice, st));
    // gate|up gemv over the stacked staging, then the batched silu*up and the batched down gemv
    // (the r18 verdict: the batched launches are mandatory - one launch each instead of 10
    // underfilled ones, the measured 112.2 -> 144.2 GB/s family)
    gemv(c->up_stage, s.xn, s.logits, st);  // y_gu [TOPK*2*EE = 12800], borrowing the logits scratch
    launch_cf_silu_mul_b(s.logits, s.ffa, EE, TOPK, st);  // [gate n | up n] per expert, stride 2n
    {  // the down gemv as one batched launch over the 10-expert SoA staging
        PackedW W = c->dn_stage;  // the per-expert view: rows = D, planes stay at the staging base
        W.rows = D;
        launch_gemv_batched(W, s.ffa, c->ye, EE, D, TOPK, st);
    }
    // shared expert + its sigmoid gate, then the weighted combine
    gemv(L.sh_gate, s.xn, s.ffg, st);
    gemv(L.sh_up, s.xn, s.ffu, st);
    launch_silu_mul(s.ffg, s.ffu, s.ffa, EE, st);
    gemv(L.sh_down, s.ffa, c->ysh, st);
    gemv(L.sh_ginp, s.xn, c->sh_gate_raw, st);
    launch_cf_moe_out(c->ye, c->we, c->ysh, c->sh_gate_raw, s.block, st);
    check_launch("moe");
}

}  // namespace

bool cf_step(CfCtx* c, int token) {
    try {
        if (token < 0 || token >= V) throw std::runtime_error("token id out of range");
        if (c->pos >= c->max_ctx) throw std::runtime_error("context full");
        auto t0 = std::chrono::steady_clock::now();
        CfScratch& s = c->sc;
        cudaStream_t st = c->st;
        CK(cudaSetDevice(c->gpu));
        // embedding row (Q4_K) dequantized on the host, then the 4 streams start as 4 copies
        if (!dequant_row_cpu(c->tok_embd->type, c->tok_embd->data + (size_t)token * c->tok_embd->row_bytes, c->h_emb,
                             D))
            throw std::runtime_error("embedding dequant failed");
        CK(cudaMemcpyAsync(s.h, c->h_emb, (size_t)D * 4, cudaMemcpyHostToDevice, st));
        launch_cf_res_init(s.h, s.h, st);  // in-place: first-D writes are identities, safe
        c->toks[c->pos] = token;
        for (int il = 0; il < NL; il++) {
            CfLayer& L = c->layers[il];
            if (il == PLE_LAYER) ple(c, token);
            hc_mix(c, s.h, L.hc_norm[0], L.hc_down[0], L.hc_up[0], &L.hc_inject[0], s);
            if (L.attn) attention(c, L, s, c->pos);
            else deltanet(c, L, s);
            hc_combine(c, s.h, s.block, s.inj);
            hc_mix(c, s.h, L.hc_norm[1], L.hc_down[1], L.hc_up[1], &L.hc_inject[1], s);
            moe(c, L, s);
            hc_combine(c, s.h, s.block, s.inj);
            // live progress: the stage log shows the rate even when a run never finishes
            if ((il & 15) == 15) fprintf(stderr, "[cf] step %d: layer %d done\n", c->pos, il);
        }
        // the final mixer is the output norm, then the lm_head
        hc_mix(c, s.h, c->o_norm, c->o_down, c->o_up, nullptr, s);
        gemv(c->output, s.mixed, s.logits, st);
        check_launch("lm_head");
        CK(cudaMemcpyAsync(c->h_logits, s.logits, (size_t)V * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        c->have_logits = true;
        c->pos++;
        c->steps++;
        c->step_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return true;
    } catch (const std::exception& e) {
        c->err = e.what();
        return false;
    }
}

void cf_reset(CfCtx* c) {
    CK(cudaSetDevice(c->gpu));
    for (int il = 0; il < NL; il++) {
        CfLayer& L = c->layers[il];
        if (L.attn) continue;  // KV beyond pos is never read
        CK(cudaMemset(L.conv_state, 0, (size_t)CONV * 3 * 4));
        CK(cudaMemset(L.S, 0, (size_t)HV * DK * DK * 4));
    }
    CK(cudaMemset(c->ple_hist, 0, (size_t)PLE_HIST * HCD * 4));
    memset(c->toks, -1, (size_t)c->max_ctx * sizeof(int));
    CK(cudaDeviceSynchronize());
    c->pos = 0;
    c->have_logits = false;
}

void cf_free(CfCtx* c) {
    if (!c) return;
    if (c->st) { cudaStreamSynchronize(c->st); cudaStreamDestroy(c->st); }
    if (c->raw_stage) cudaFreeHost(c->raw_stage);
    if (c->raw_dev) cudaFree(c->raw_dev);
    if (c->h_router) cudaFreeHost(c->h_router);
    if (c->we_h) cudaFreeHost(c->we_h);
    if (c->h_emb) cudaFreeHost(c->h_emb);
    if (c->h_ple) cudaFreeHost(c->h_ple);
    if (c->h_logits) cudaFreeHost(c->h_logits);
    delete[] c->eid;
    delete[] c->toks;
    CK(cudaSetDevice(c->gpu));
    CK(cudaDeviceReset());
    delete c;
}
