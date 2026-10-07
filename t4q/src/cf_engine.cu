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

// the r18 repack worklist fast path: the Q2_K/Q4_K/Q5_1 tensors dot against the ggml's own
// activation quantizations (Q8_K for the K-quants, Q8_1 for the Q5_1) - the exact arithmetic
// the CPU oracle itself computes, so the gates compare the same quantization, and the dots run
// at the dp4a-class rates the r18 probes measured instead of the fp32 reference path. The
// quantize (~5 us) rides the same stream before the dot. The traffic it covers: the Q2_K trunk
// (~1.5 GB/token: attn_qkv/gate, ssm_out, the routers, the shexp, the ple key/value), the Q5_1
// hc mixers (~0.44 GB/token), the Q4_K lm_head (~0.34 GB/token).
void gemv(CfScratch& sc, const PackedW& W, const float* x, float* y, cudaStream_t st) {
    // T4Q_CF_NOFAST=1: bypass the r18 q8 fast paths, run the float-reference gemv (debug bisect)
    static int nofast = -1;
    if (nofast < 0) { const char* e = getenv("T4Q_CF_NOFAST"); nofast = (e && atoi(e)) ? 1 : 0; }
    if (nofast) { launch_gemv(W, x, y, st); return; }
    if (W.fmt == FMT_K2 || W.fmt == FMT_K4) {
        launch_quantize_q8_K(x, (int)W.cols, sc.xqk, sc.xqk_b, sc.xqk_d, st);
        launch_gemv_q8k(W, sc.xqk, sc.xqk_b, sc.xqk_d, y, st);
    } else if (W.fmt == FMT_Q51) {
        launch_quantize_q8_1(x, (int)W.cols, sc.xq1, sc.xq1_d, sc.xq1_s, st);
        launch_gemv_q8(W, sc.xq1, sc.xq1_d, sc.xq1_s, y, st);
    } else if (W.fmt == FMT_P4) {
        launch_quantize_q8_0(x, (int)W.cols, sc.xq0, sc.xd0, sc.xs0, st);
        launch_gemv_q8_0(W, sc.xq0, sc.xd0, sc.xs0, y, st);
    } else {
        launch_gemv(W, x, y, st);
    }
}

void check_launch(const char* what) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("launch failed: ") + what + ": " + cudaGetErrorString(e));
}

// T4Q_CF_DUMP bisect support: file-scope so the sub-op functions can capture internals.
static FILE* cfdump = (FILE*)1;
static int cfdump_last = -2;
static int cfdump_tag = -1;
// the dump copies MUST go on the engine's (non-blocking) stream: a NULL-stream copy races the
// pending kernels and reads pre-write garbage (the "z ~ 0" phantom was this race)
static cudaStream_t cfdump_st = nullptr;
static bool cfdump_active() {
    return cfdump && cfdump_tag >= 0;
}
static void cfdump_rec(const char* name, const float* dev, int n) {
    if (!cfdump_active()) return;
    std::vector<float> tmp(n);
    cudaMemcpyAsync(tmp.data(), dev, (size_t)n * 4, cudaMemcpyDeviceToHost, cfdump_st);
    cudaStreamSynchronize(cfdump_st);
    uint32_t nl = (uint32_t)strlen(name);
    fwrite(&nl, 4, 1, cfdump); fwrite(name, 1, nl, cfdump);
    fwrite(&cfdump_tag, 4, 1, cfdump);
    int64_t ne[4] = {n, 1, 1, 1}; fwrite(ne, 8, 4, cfdump);
    fwrite(tmp.data(), 4, n, cfdump);
}

// res_hc -> xn (grouped norm), lo, gate(y_up), mixed, inj. side 0 = attn, 1 = ffn.
// The final output mixer passes with_inject = false (no scatter weights).
void hc_mix(CfCtx* c, const float* res, const float* w_norm, const PackedW& down, const PackedW& up,
            const PackedW* inject, CfScratch& s) {
    cudaStream_t st = c->st;
    launch_cf_hc_norm(res, w_norm, s.xn, st);
    gemv(s, down, s.xn, s.lo, st);
    launch_cf_hc_lo(s.lo, s.lo, st);  // in-place: reads y[i] writes lo[i], identity-safe
    gemv(s, up, s.lo, s.gate, st);       // gate buffer = y_up [HCD]
    launch_cf_hc_mixed(s.xn, s.gate, s.mixed, st);
    if (inject) gemv(s, *inject, s.xn, s.inj, st);  // [HC]
}

// res += 2*sigmoid(inj/HC) * block
void hc_combine(CfCtx* c, float* res, const float* block, const float* inj) {
    launch_cf_hc_combine(res, block, inj, c->st);
}

void deltanet(CfCtx* c, CfLayer& L, CfScratch& s) {
    cudaStream_t st = c->st;
    // the arch feeds every block projection the hc_mix OUTPUT (the sigmoid-gated, stream-mean
    // collapsed mixed), not the raw grouped-norm xn (build_layer_attn_linear(cur) where cur =
    // build_hc_mix(...) and build_qkvz(cur)): the r18 fast-path bisect phantom was this input.
    gemv(s, L.qkv, s.mixed, s.qkv, st);
    gemv(s, L.z, s.mixed, s.zz, st);
    { std::string nm = "linear_attn_qkv_mixed-" + std::to_string(L.il); cfdump_rec(nm.c_str(), s.qkv, CONV); }
    { std::string nm = "z-" + std::to_string(L.il); cfdump_rec(nm.c_str(), s.zz, VDIM); }
    gemv(s, L.beta, s.mixed, s.braw, st);
    gemv(s, L.alpha, s.mixed, s.araw, st);
    check_launch("gdn proj");
    launch_gdn_gates(s.braw, s.araw, L.ssm_a, L.ssm_dt, s.beta, s.g, HV, st);
    { std::string nm = "beta_sigmoid-" + std::to_string(L.il); cfdump_rec(nm.c_str(), s.beta, HV); }
    { std::string nm = "gate-" + std::to_string(L.il); cfdump_rec(nm.c_str(), s.g, HV); }
    launch_gdn_conv(s.qkv, L.conv_state, L.conv_w, s.conv, CONV, st);
    { std::string nm = "conv_output_silu-" + std::to_string(L.il); cfdump_rec(nm.c_str(), s.conv, CONV); }
    launch_gdn_l2(s.conv, s.qn, s.kn, EPS, st);
    launch_gdn_recur(L.S, s.qn, s.kn, s.conv + 2 * HK * DK, s.beta, s.g, s.o, 1.0f / sqrtf((float)DK), st);
    check_launch("gdn recur");
    launch_cf_gdn_gnorm(s.o, s.zz, L.ssm_norm, s.on, EPS, st);
    { std::string nm = "linear_attn_out_norm-" + std::to_string(L.il); cfdump_rec(nm.c_str(), s.on, VDIM); }
    gemv(s, L.ssm_out, s.on, s.block, st);
    check_launch("ssm_out");
}

void attention(CfCtx* c, CfLayer& L, CfScratch& s) {
    cudaStream_t st = c->st;
    gemv(s, L.wq, s.mixed, s.qfull, st);
    gemv(s, L.wk, s.mixed, s.k, st);
    gemv(s, L.wv, s.mixed, s.v, st);
    check_launch("attn proj");
    launch_cf_qk_norm_rope(s.qfull, s.k, L.q_norm, L.k_norm, s.aq, s.ak, c->d_params, EPS, ROPE_BASE, NROT, st);
    launch_cf_kv_store(s.ak, s.v, L.kc, L.vc, c->d_params, c->max_ctx, st);
    launch_cf_attn_decode(s.aq, L.kc, L.vc, s.att, s.scores, c->d_params, c->max_ctx, 1.0f / 16.0f, st);
    check_launch("attn");
    launch_gate_sigmoid(s.att, s.qfull, s.attg, st);  // 24 blocks, [24][512]: same layout
    gemv(s, L.wo, s.attg, s.block, st);
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
    if (cfdump_active()) {  // host gather result, no race
        uint32_t nl = 7;
        fwrite(&nl, 4, 1, cfdump); fwrite("ple_emb", 1, 7, cfdump);
        fwrite(&cfdump_tag, 4, 1, cfdump);
        int64_t ne[4] = {D, 1, 1, 1}; fwrite(ne, 8, 4, cfdump);
        fwrite(c->h_ple, 4, D, cfdump);
    }
    gemv(s, c->ple.key, s.mixed, c->ple_key, st);
    gemv(s, c->ple.value, s.mixed, s.block, st);
    if (cfdump_active()) { cfdump_rec("ple_key_raw", c->ple_key, HCD); cfdump_rec("ple_value_raw", s.block, D); }
    // grouped norms: key in place, query from the current wide residual
    launch_cf_hc_norm(c->ple_key, c->ple.norm_key, c->ple_key, st);
    launch_cf_hc_norm(s.h, c->ple.norm_query, c->ple_query, st);
    if (cfdump_active()) { cfdump_rec("ple_key_norm", c->ple_key, HCD); cfdump_rec("ple_query_norm", c->ple_query, HCD); }
    launch_cf_ple_sg(c->ple_key, c->ple_query, c->ple_s, c->ple_gate, st);
    if (cfdump_active()) { cfdump_rec("ple_s", c->ple_s, HC); cfdump_rec("ple_gate", c->ple_gate, HC); }
    launch_cf_ple_gated(s.block, c->ple_gate, c->ple_gated, st);  // gated = value * gate[s]
    if (cfdump_active()) cfdump_rec("ple_gated", c->ple_gated, HCD);
    // conv over the normed gated, then res += gated + conv (exact add order)
    launch_cf_hc_norm(c->ple_gated, c->ple.norm_conv, c->ple_key, st);  // reuse the dead key buffer
    if (cfdump_active()) cfdump_rec("ple_normed", c->ple_key, HCD);
    launch_cf_ple_conv(c->ple_key, c->ple_hist, c->ple.conv_w, c->ple_query, st);
    if (cfdump_active()) cfdump_rec("ple_conv_out", c->ple_query, HCD);
    launch_add(c->ple_query, c->ple_gated, HCD, st);  // t = gated + conv
    launch_add(s.h, c->ple_query, HCD, st);           // res += t
}

// The MoE: router -> host softmax/top-10/renorm -> stage 10 experts -> repack -> gemv -> combine
void moe(CfCtx* c, CfLayer& L, CfScratch& s) {
    cudaStream_t st = c->st;
    { std::string nm = "moe_mixed-" + std::to_string(L.il); cfdump_rec(nm.c_str(), s.mixed, D); }
    // the router and every expert/shared input is the ffn-side hc mixed (build_layer_ffn(cur))
    gemv(s, L.router, s.mixed, s.logits, st);  // borrow the logits scratch [512 of V]
    CK(cudaMemcpyAsync(c->h_router, s.logits, (size_t)NE * 4, cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    { std::string nm = "moe_router-" + std::to_string(L.il); cfdump_rec(nm.c_str(), s.logits, NE); }
    if (cfdump_active())
        fprintf(stderr, "[moe] il %d pos ? router[0..3] %.4f %.4f %.4f %.4f\n", L.il, c->h_router[0],
                c->h_router[1], c->h_router[2], c->h_router[3]);
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
    const size_t gu_row = L.t_gate_exps->row_bytes;   // 840
    const size_t dn_row = L.t_down_exps->row_bytes;   // 360
    const size_t up_bytes = (size_t)TOPK * 2 * EE * gu_row;
    if (c->uva && L.il < c->uva_n) {
        // cf-m3 (r19u) the UVA pointer-swap path: the picks' raw slabs are read from the
        // REGISTERED mmap'd expert pages through the device aliases - NO host memcpys,
        // NO H2D, NO raw staging; the scatter repack's address math is exactly the OFF
        // path's memcpy sources, the same block decode, the same identity W table + the
        // same gemvs, so the packed slabs are byte-identical by construction (only the
        // read path changes: the mapped pages over PCIe instead of the pinned VRAM copy)
        CK(cudaMemcpyAsync(c->eid_dev, c->eid, (size_t)TOPK * 4, cudaMemcpyHostToDevice, st));
        launch_repack_eid_q2k(c->up_stage, L.uva_gate, L.uva_up, c->eid_dev, 0,
                              (int64_t)TOPK * 2 * EE, (int64_t)2 * EE, (int64_t)EE, st);
        launch_repack_eid_q4(c->dn_stage, L.uva_dn, c->eid_dev, 0, (int64_t)TOPK * D, (int64_t)D, st);
    } else if (c->tiered) {
        // the dual-path moe (cf-m3, r19l part 2): a HIT pick reads its resident slab with ZERO
        // staging (the load-time repack already made it byte-identical to what the staging would
        // produce); only the MISS picks pay the MMAP->pinned->H2D path, each for its OWN rows,
        // packed compactly at the staging front (slot m = the miss ordinal). The per-pick W
        // table is rebuilt on the host per step and uploaded - the same table the r19k batched
        // gemvs read; the kernels are untouched.
        int nmiss = 0;
        for (int k = 0; k < TOPK; k++) {
            const int64_t e = c->eid[k];
            const int h = L.hot_idx[e];
            PackedW& v = c->h_wt_gu[k];
            PackedW& w = c->h_wt_dn[k];
            if (h >= 0) {  // hit: the resident slab view, the same offsets a staged slab would have
                v = L.res_gu;
                v.rows = 2 * EE;
                v.codes = L.res_gu.codes + (size_t)h * 2 * EE * (D / 4);
                v.meta = L.res_gu.meta + (size_t)h * 2 * EE * (size_t)(D / 256) * 20;
                w = L.res_dn;
                w.rows = D;
                w.codes = L.res_dn.codes + (size_t)h * D * (EE / 2);
                w.d = L.res_dn.d + (size_t)h * D * (EE / 32);
            } else {  // miss: stage this expert's own rows at the compact slot m
                const int m = nmiss++;
                uint8_t* dst = c->raw_stage + (size_t)m * 2 * EE * gu_row;
                memcpy(dst, L.t_gate_exps->data + (size_t)e * EE * gu_row, (size_t)EE * gu_row);
                memcpy(dst + (size_t)EE * gu_row, L.t_up_exps->data + (size_t)e * EE * gu_row, (size_t)EE * gu_row);
                memcpy(c->raw_stage + up_bytes + (size_t)m * D * dn_row,
                       L.t_down_exps->data + (size_t)e * D * dn_row, (size_t)D * dn_row);
                v = c->up_stage;
                v.rows = 2 * EE;
                v.codes = c->up_stage.codes + (size_t)m * 2 * EE * (D / 4);
                v.meta = c->up_stage.meta + (size_t)m * 2 * EE * (size_t)(D / 256) * 20;
                w = c->dn_stage;
                w.rows = D;
                w.codes = c->dn_stage.codes + (size_t)m * D * (EE / 2);
                w.d = c->dn_stage.d + (size_t)m * D * (EE / 32);
            }
        }
        if (nmiss) {  // stage + repack ONLY the miss rows (an all-hit layer pays neither)
            CK(cudaMemcpyAsync(c->raw_dev, c->raw_stage, (size_t)nmiss * 2 * EE * gu_row, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(c->raw_dev + up_bytes, c->raw_stage + up_bytes, (size_t)nmiss * D * dn_row,
                               cudaMemcpyHostToDevice, st));
            launch_repack(c->up_stage, GT_Q2_K, c->raw_dev, 0, (int64_t)nmiss * 2 * EE, st);
            launch_repack(c->dn_stage, GT_Q4_0, c->raw_dev + up_bytes, 0, (int64_t)nmiss * D, st);
        }
        CK(cudaMemcpyAsync(c->wt_gu, c->h_wt_gu, (size_t)TOPK * sizeof(PackedW), cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(c->wt_dn, c->h_wt_dn, (size_t)TOPK * sizeof(PackedW), cudaMemcpyHostToDevice, st));
    } else {
        // OFF (no hot-set file): the verbatim full-staging path over the load-time identity table
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
    }
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(c->we, c->we_h, (size_t)TOPK * 4, cudaMemcpyHostToDevice, st));
    // gate|up gemv over the stacked staging, then the batched silu*up and the batched down gemv
    // (the r18 verdict: the batched launches are mandatory - one launch each instead of 10
    // underfilled ones, the measured 112.2 -> 144.2 GB/s family). The r19k form: the gemv runs
    // the per-pick W table (the identity views of the staged slabs, bit-identical to the old
    // single launch's stride math; the tiering later swaps in resident views with zero kernel
    // change); the xn quantizes ONCE and is shared by all 10 picks (strides 0).
    launch_quantize_q8_K(s.mixed, D, s.xqk, s.xqk_b, s.xqk_d, st);
    launch_gemv_q8k_b(c->wt_gu, FMT_K2, 2 * EE, s.xqk, s.xqk_b, s.xqk_d, s.logits, 0, 0, 0, (int64_t)2 * EE, TOPK,
                      st);
    launch_cf_silu_mul_b(s.logits, s.ffa, EE, TOPK, st);  // [gate n | up n] per expert, stride 2n
    {  // the down gemv as one batched launch over the 10-expert SoA staging, on the Q8_0
        // pairing (the oracle's own arithmetic for the Q4_0 down experts); the ffa [TOPK][EE]
        // is flat and the expert boundaries are the 32-group boundaries, so one flat quantize
        launch_quantize_q8_0(s.ffa, TOPK * EE, s.xq0, s.xd0, s.xs0, st);
        launch_gemv_q8_0_b(c->wt_dn, D, s.xq0, s.xd0, s.xs0, c->ye, EE, D, EE / 32, TOPK, st);
    }
    // shared expert + its sigmoid gate, then the weighted combine
    gemv(s, L.sh_gate, s.mixed, s.ffg, st);
    gemv(s, L.sh_up, s.mixed, s.ffu, st);
    launch_silu_mul(s.ffg, s.ffu, s.ffa, EE, st);
    gemv(s, L.sh_down, s.ffa, c->ysh, st);
    gemv(s, L.sh_ginp, s.mixed, c->sh_gate_raw, st);
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
        // T4Q_CF_DUMP=<path> T4Q_CF_DUMP_LAST=<n-1>: T4QD-format bisect capture
        // (header count is a placeholder; the reader goes to EOF).
        if (cfdump == (FILE*)1) {
            const char* p = getenv("T4Q_CF_DUMP");
            cfdump = (p && *p) ? fopen(p, "wb") : nullptr;
            cfdump_st = st;
            if (cfdump) {
                const char* lp = getenv("T4Q_CF_DUMP_LAST");
                cfdump_last = lp ? atoi(lp) : -1;
                uint32_t magic_hdr = 0x44513454;  // "T4QD"
                uint32_t cnt = 0xFFFFFFFFu;
                fwrite(&magic_hdr, 4, 1, cfdump);
                fwrite(&cnt, 4, 1, cfdump);
            }
        }
        cfdump_tag = (cfdump && (c->pos == 0 || c->pos == 1 || c->pos == cfdump_last)) ? c->pos : -1;
        // embedding row (Q4_K) dequantized on the host, then the 4 streams start as 4 copies
        if (!dequant_row_cpu(c->tok_embd->type, c->tok_embd->data + (size_t)token * c->tok_embd->row_bytes, c->h_emb,
                             D))
            throw std::runtime_error("embedding dequant failed");
        CK(cudaMemcpyAsync(s.h, c->h_emb, (size_t)D * 4, cudaMemcpyHostToDevice, st));
        // r19v: the step params (pos) ride the pinned word -> the device word; the attention
        // kernels read it from there (the same int, the same arithmetic - capture-constant)
        c->h_params[0] = c->pos;
        CK(cudaMemcpyAsync(c->d_params, c->h_params, 4 * sizeof(int), cudaMemcpyHostToDevice, st));
        launch_cf_res_init(s.h, s.h, st);  // in-place: first-D writes are identities, safe
        if (cfdump_active()) {  // model.input_embed: the raw [D] host embedding, matching the oracle's cb name
            uint32_t nl = 17;
            fwrite(&nl, 4, 1, cfdump); fwrite("model.input_embed", 1, 17, cfdump);
            fwrite(&cfdump_tag, 4, 1, cfdump);
            int64_t ne[4] = {D, 1, 1, 1}; fwrite(ne, 8, 4, cfdump);
            fwrite(c->h_emb, 4, D, cfdump);
        }
        c->toks[c->pos] = token;
        for (int il = 0; il < NL; il++) {
            CfLayer& L = c->layers[il];
            if (il == PLE_LAYER) ple(c, token);
            hc_mix(c, s.h, L.hc_norm[0], L.hc_down[0], L.hc_up[0], &L.hc_inject[0], s);
            // hc_norm: the FIRST (attn-side) norm output; the oracle's first-capture matches
            { std::string nm = "hc_norm-" + std::to_string(il); cfdump_rec(nm.c_str(), s.xn, HCD); }
            { std::string nm = "hc_gate-" + std::to_string(il); cfdump_rec(nm.c_str(), s.gate, HCD); }
            { std::string nm = "hc_mixed-" + std::to_string(il); cfdump_rec(nm.c_str(), s.mixed, D); }
            { std::string nm = "hc_inject-" + std::to_string(il); cfdump_rec(nm.c_str(), s.inj, HC); }
            if (L.attn) attention(c, L, s);
            else deltanet(c, L, s);
            {
                std::string bn = (L.attn ? "attn_output-" : "linear_attn_out-") + std::to_string(il);
                cfdump_rec(bn.c_str(), s.block, D);
            }
            hc_combine(c, s.h, s.block, s.inj);
            { std::string nm = "res_mid-" + std::to_string(il); cfdump_rec(nm.c_str(), s.h, HCD); }
            { std::string nm = "hc_combine-" + std::to_string(il); cfdump_rec(nm.c_str(), s.h, HCD); }
            hc_mix(c, s.h, L.hc_norm[1], L.hc_down[1], L.hc_up[1], &L.hc_inject[1], s);
            { std::string nm = "ffn_hc_norm-" + std::to_string(il); cfdump_rec(nm.c_str(), s.xn, HCD); }
            { std::string nm = "ffn_mixed-" + std::to_string(il); cfdump_rec(nm.c_str(), s.mixed, D); }
            moe(c, L, s);
            { std::string nm = "ffn_out-" + std::to_string(il); cfdump_rec(nm.c_str(), s.block, D); }
            hc_combine(c, s.h, s.block, s.inj);
            { std::string nm = "l_out-" + std::to_string(il); cfdump_rec(nm.c_str(), s.h, HCD); }
            // live progress: the stage log shows the rate even when a run never finishes
            if ((il & 15) == 15) fprintf(stderr, "[cf] step %d: layer %d done\n", c->pos, il);
        }
        // the final mixer is the output norm, then the lm_head
        hc_mix(c, s.h, c->o_norm, c->o_down, c->o_up, nullptr, s);
        cfdump_rec("result_norm", s.mixed, D);
        gemv(s, c->output, s.mixed, s.logits, st);
        check_launch("lm_head");
        CK(cudaMemcpyAsync(c->h_logits, s.logits, (size_t)V * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        if (cfdump_active()) {
            uint32_t nl = 13;
            fwrite(&nl, 4, 1, cfdump); fwrite("result_output", 1, 13, cfdump);
            fwrite(&cfdump_tag, 4, 1, cfdump);
            int64_t ne[4] = {V, 1, 1, 1}; fwrite(ne, 8, 4, cfdump);
            fwrite(c->h_logits, 4, V, cfdump);
        }
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
    if (c->uva_reg) cudaHostUnregister(c->uva_reg);  // cf-m3 (r19u): the stream is drained
    if (c->raw_stage) cudaFreeHost(c->raw_stage);
    if (c->raw_dev) cudaFree(c->raw_dev);
    if (c->eid_dev) cudaFree(c->eid_dev);
    if (c->d_params) cudaFree(c->d_params);   // r19v: the step-params device word
    if (c->h_params) cudaFreeHost(c->h_params);
    if (c->wt_gu) cudaFree(c->wt_gu);
    if (c->wt_dn) cudaFree(c->wt_dn);
    if (c->h_router) cudaFreeHost(c->h_router);
    if (c->we_h) cudaFreeHost(c->we_h);
    if (c->h_emb) cudaFreeHost(c->h_emb);
    if (c->h_ple) cudaFreeHost(c->h_ple);
    if (c->h_logits) cudaFreeHost(c->h_logits);
    if (c->eid) cudaFreeHost(c->eid);  // r19v: pinned (the captured eid H2D reads it)
    delete[] c->toks;
    for (int il = 0; il < NL; il++) {  // the resident tier (cf-m3): the device bases + the host maps
        CfLayer& L = c->layers[il];
        if (L.res_gu.base) cudaFree(L.res_gu.base);
        if (L.res_dn.base) cudaFree(L.res_dn.base);
        delete[] L.hot_ids;
        delete[] L.hot_idx;
    }
    CK(cudaSetDevice(c->gpu));
    CK(cudaDeviceReset());
    delete c;
}
