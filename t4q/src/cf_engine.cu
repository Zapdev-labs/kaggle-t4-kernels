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
    } else if (W.fmt == FMT_P4 || W.fmt == FMT_Q8) {
        // FMT_Q8 (cf-m4, r19w): the Q8_0 weights pair with the SAME q8_0-form activation
        // (amax/127, no s-term) - launch_quantize_q8_0 already produces it; the dot is the
        // gate-proven FAST_Q8 int8-dp4a family (the 27B's own MTP graft passed the
        // byte-identical spec gates on this exact arithmetic)
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
// r19v split: ple_host (the u64 hash + the 16-row host dequant) is the driver's host
// window work; emit_ple_kernels (the H2D + the 8 launches) is the segment emission.
// r19x: the core takes (token, pos, dst) - the verify's per-row gather runs it at the
// ROW's position into the row's pinned slice (no cross-row H2D race); ple_host is the
// trunk's single-row instance.
void ple_host_core(CfCtx* c, int token, int pos, float* dst) {
    // host: the u64 hash. ctx[0] = the token; predecessors cut at EOS/missing (missing reads EOS).
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
                                 dst + (size_t)h * PLE_DIM, PLE_DIM))
                throw std::runtime_error("ple row dequant failed");
        }
    }
}

void ple_host(CfCtx* c, int token) { ple_host_core(c, token, c->pos, c->h_ple); }

void emit_ple_kernels(CfCtx* c) {
    CfScratch& s = c->sc;
    cudaStream_t st = c->st;
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

// The MoE: router -> host softmax/top-10/renorm -> stage 10 experts -> repack -> gemv -> combine.
// r19v split (one emission source, two drivers): emit_router (the router gemv + the D2H, the
// segment's last ops) / host_router (the sync + the order-exact host loops + the staging HOST
// work) / emit_moe_rest (the staging/repack branch's stream ops + the we upload + the batched
// gemv family + the shared expert + moe_out). The direct driver emits them in exactly the old
// inline order; the graph driver captures them into the per-segment graphs.
void emit_router(CfCtx* c, CfLayer& L) {
    CfScratch& s = c->sc;
    cudaStream_t st = c->st;
    { std::string nm = "moe_mixed-" + std::to_string(L.il); cfdump_rec(nm.c_str(), s.mixed, D); }
    // the router and every expert/shared input is the ffn-side hc mixed (build_layer_ffn(cur))
    gemv(s, L.router, s.mixed, s.logits, st);  // borrow the logits scratch [512 of V]
    CK(cudaMemcpyAsync(c->h_router, s.logits, (size_t)NE * 4, cudaMemcpyDeviceToHost, st));
}

// the order-exact host router loops (r19t: the fp accumulation ORDER is the correctness bar
// for the G2 device router and every other consumer; ONE source, shared by the trunk's host
// window, the draft's forward, and the verify's per-row window - the routing MUST be the
// same order-exact form everywhere). The row variant takes the row's buffers (the router
// row is scratch, softmaxed in place exactly as the trunk's does); host_top10 is the
// trunk's single-row instance.
void host_top10_row(float* router, int* eid, float* we_h) {
    // host softmax over 512 (fp32, exact formula), then top-10 renormalized
    float m = router[0];
    for (int i = 1; i < NE; i++) m = std::max(m, router[i]);
    float sum = 0.f;
    for (int i = 0; i < NE; i++) {
        router[i] = std::exp(router[i] - m);
        sum += router[i];
    }
    for (int i = 0; i < NE; i++) router[i] /= sum;
    for (int k = 0; k < TOPK; k++) eid[k] = -1;
    for (int k = 0; k < TOPK; k++) {
        int best = -1;
        for (int i = 0; i < NE; i++) {
            if (router[i] < 0.f) continue;  // already picked
            if (best < 0 || router[i] > router[best]) best = i;
        }
        eid[k] = best;
        we_h[k] = router[best];
        router[best] = -1.f;
    }
    float wsum = 0.f;
    for (int k = 0; k < TOPK; k++) wsum += we_h[k];
    for (int k = 0; k < TOPK; k++) we_h[k] /= wsum;
}

void host_top10(CfCtx* c) { host_top10_row(c->h_router, c->eid, c->we_h); }

// the host window between segments: the sync (the router D2H has landed), then the r19t
// order-exact softmax/top-10/we renorm (the host loops verbatim), the census, and the staging
// HOST work (the OFF-path mmap->pinned memcpys; the tiered compose - the tiered mode never
// runs under graphs) - the pinned sources (eid/we_h/raw_stage) the next segment's captured
// H2D nodes re-carry at its replay
void host_router(CfCtx* c, CfLayer& L) {
    CfScratch& s = c->sc;
    cudaStream_t st = c->st;
    CK(cudaStreamSynchronize(st));
    { std::string nm = "moe_router-" + std::to_string(L.il); cfdump_rec(nm.c_str(), s.logits, NE); }
    if (cfdump_active())
        fprintf(stderr, "[moe] il %d pos ? router[0..3] %.4f %.4f %.4f %.4f\n", L.il, c->h_router[0],
                c->h_router[1], c->h_router[2], c->h_router[3]);
    host_top10(c);
    if (c->census_f) {  // cf-m2: the router concentration census, same round as the gate
        fwrite(c->eid, 4, TOPK, c->census_f);
        fwrite(c->we_h, 4, TOPK, c->census_f);
    }
    const size_t gu_row = L.t_gate_exps->row_bytes;   // 840
    const size_t dn_row = L.t_down_exps->row_bytes;  // 360
    const size_t up_bytes = (size_t)TOPK * 2 * EE * gu_row;
    if (c->uva && L.il < c->uva_n) {
        // the UVA path stages NOTHING on the host (the picks' raw slabs stay in the
        // registered pages; the next segment's captured eid H2D node carries the fresh ids)
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
        c->tier_nmiss = nmiss;  // r19v: the emission's tiered branch reads it (never captured)
    } else {
        // OFF (no hot-set file): the verbatim full-staging host memcpys over the load-time identity table
        for (int k = 0; k < TOPK; k++) {
            const int64_t e = c->eid[k];
            uint8_t* dst = c->raw_stage + (size_t)k * 2 * EE * gu_row;
            memcpy(dst, L.t_gate_exps->data + (size_t)e * EE * gu_row, (size_t)EE * gu_row);
            memcpy(dst + (size_t)EE * gu_row, L.t_up_exps->data + (size_t)e * EE * gu_row, (size_t)EE * gu_row);
            memcpy(c->raw_stage + up_bytes + (size_t)k * D * dn_row, L.t_down_exps->data + (size_t)e * D * dn_row,
                   (size_t)D * dn_row);
        }
    }
}

// the moe's post-router emission: the staging/repack branch's stream ops (the UVA scatters /
// the tiered miss upload (never captured) / the OFF full upload), the we upload, the batched
// gemv family, the shared expert, moe_out (the verbatim moved bodies)
void emit_moe_rest(CfCtx* c, CfLayer& L) {
    CfScratch& s = c->sc;
    cudaStream_t st = c->st;
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
        // the tiered stream ops (never captured - the miss-count-varying H2D sizes; the
        // compose ran in host_router and left c->tier_nmiss)
        if (c->tier_nmiss) {  // stage + repack ONLY the miss rows (an all-hit layer pays neither)
            CK(cudaMemcpyAsync(c->raw_dev, c->raw_stage, (size_t)c->tier_nmiss * 2 * EE * gu_row,
                               cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(c->raw_dev + up_bytes, c->raw_stage + up_bytes, (size_t)c->tier_nmiss * D * dn_row,
                               cudaMemcpyHostToDevice, st));
            launch_repack(c->up_stage, GT_Q2_K, c->raw_dev, 0, (int64_t)c->tier_nmiss * 2 * EE, st);
            launch_repack(c->dn_stage, GT_Q4_0, c->raw_dev + up_bytes, 0, (int64_t)c->tier_nmiss * D, st);
        }
        CK(cudaMemcpyAsync(c->wt_gu, c->h_wt_gu, (size_t)TOPK * sizeof(PackedW), cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(c->wt_dn, c->h_wt_dn, (size_t)TOPK * sizeof(PackedW), cudaMemcpyHostToDevice, st));
    } else {
        // OFF (no hot-set file): the verbatim full-staging upload + repack over the identity table
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

// the post-router emission of one layer: the moe rest + the ffn_out rec + the combine + the
// l_out rec (the verbatim tail of the old loop body; the recs no-op when the dump is off)
void emit_post(CfCtx* c, CfLayer& L) {
    CfScratch& s = c->sc;
    emit_moe_rest(c, L);
    { std::string nm = "ffn_out-" + std::to_string(L.il); cfdump_rec(nm.c_str(), s.block, D); }
    hc_combine(c, s.h, s.block, s.inj);
    { std::string nm = "l_out-" + std::to_string(L.il); cfdump_rec(nm.c_str(), s.h, HCD); }
}

// the step head emission: the embedding row H2D (the host dequant + the h_params[0] = pos
// write ran in the DRIVER - the per-step host state must NOT live in the emission, which the
// graph driver calls only at capture time) + the step-params upload + the residual init
void emit_head(CfCtx* c) {
    CfScratch& s = c->sc;
    cudaStream_t st = c->st;
    CK(cudaMemcpyAsync(s.h, c->h_emb, (size_t)D * 4, cudaMemcpyHostToDevice, st));
    // r19v: the step params (pos) ride the pinned word -> the device word; the attention
    // kernels read it from there (the same int, the same arithmetic - capture-constant)
    CK(cudaMemcpyAsync(c->d_params, c->h_params, 4 * sizeof(int), cudaMemcpyHostToDevice, st));
    launch_cf_res_init(s.h, s.h, st);  // in-place: first-D writes are identities, safe
}

// the pre-router emission of one layer: hc_mix -> (attention | deltanet) -> combine -> hc_mix
void emit_pre(CfCtx* c, CfLayer& L) {
    CfScratch& s = c->sc;
    hc_mix(c, s.h, L.hc_norm[0], L.hc_down[0], L.hc_up[0], &L.hc_inject[0], s);
    // hc_norm: the FIRST (attn-side) norm output; the oracle's first-capture matches
    { std::string nm = "hc_norm-" + std::to_string(L.il); cfdump_rec(nm.c_str(), s.xn, HCD); }
    { std::string nm = "hc_gate-" + std::to_string(L.il); cfdump_rec(nm.c_str(), s.gate, HCD); }
    { std::string nm = "hc_mixed-" + std::to_string(L.il); cfdump_rec(nm.c_str(), s.mixed, D); }
    { std::string nm = "hc_inject-" + std::to_string(L.il); cfdump_rec(nm.c_str(), s.inj, HC); }
    if (L.attn) attention(c, L, s);
    else deltanet(c, L, s);
    {
        std::string bn = (L.attn ? "attn_output-" : "linear_attn_out-") + std::to_string(L.il);
        cfdump_rec(bn.c_str(), s.block, D);
    }
    hc_combine(c, s.h, s.block, s.inj);
    { std::string nm = "res_mid-" + std::to_string(L.il); cfdump_rec(nm.c_str(), s.h, HCD); }
    { std::string nm = "hc_combine-" + std::to_string(L.il); cfdump_rec(nm.c_str(), s.h, HCD); }
    hc_mix(c, s.h, L.hc_norm[1], L.hc_down[1], L.hc_up[1], &L.hc_inject[1], s);
    { std::string nm = "ffn_hc_norm-" + std::to_string(L.il); cfdump_rec(nm.c_str(), s.xn, HCD); }
    { std::string nm = "ffn_mixed-" + std::to_string(L.il); cfdump_rec(nm.c_str(), s.mixed, D); }
}

// the tail emission: the final mixer (the output norm) + the lm_head + the logits D2H
void emit_tail(CfCtx* c) {
    CfScratch& s = c->sc;
    // the final mixer is the output norm, then the lm_head
    hc_mix(c, s.h, c->o_norm, c->o_down, c->o_up, nullptr, s);
    cfdump_rec("result_norm", s.mixed, D);
    gemv(s, c->output, s.mixed, s.logits, c->st);
    check_launch("lm_head");
    CK(cudaMemcpyAsync(c->h_logits, s.logits, (size_t)V * 4, cudaMemcpyDeviceToHost, c->st));
}

}  // namespace

bool cf_step(CfCtx* c, int token) {
    try {
        if (token < 0 || token >= V) throw std::runtime_error("token id out of range");
        if (c->pos >= c->max_ctx) throw std::runtime_error("context full");
        auto t0 = std::chrono::steady_clock::now();
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
        // r19v: the per-step host state (pos) - written EVERY step in the driver, BEFORE the
        // emission (the params H2D's pinned source is read at execution time, so the write
        // must precede the enqueue; and it must never ride the emission itself, which the
        // graph driver calls only at capture time)
        c->h_params[0] = c->pos;
        emit_head(c);
        if (cfdump_active()) {  // model.input_embed: the raw [D] host embedding, matching the oracle's cb name
            uint32_t nl = 17;
            fwrite(&nl, 4, 1, cfdump); fwrite("model.input_embed", 1, 17, cfdump);
            fwrite(&cfdump_tag, 4, 1, cfdump);
            int64_t ne[4] = {D, 1, 1, 1}; fwrite(ne, 8, 4, cfdump);
            fwrite(c->h_emb, 4, D, cfdump);
        }
        c->toks[c->pos] = token;
        if (!c->gmode) {
            // ---- the direct driver: the verbatim op sequence via the emission functions ----
            for (int il = 0; il < NL; il++) {
                CfLayer& L = c->layers[il];
                if (il == PLE_LAYER) { ple_host(c, token); emit_ple_kernels(c); }
                emit_pre(c, L);
                emit_router(c, L);
                host_router(c, L);  // the sync + the order-exact host loops + the staging host work
                emit_post(c, L);
                // live progress: the stage log shows the rate even when a run never finishes
                if ((il & 15) == 15) fprintf(stderr, "[cf] step %d: layer %d done\n", c->pos, il);
            }
            emit_tail(c);
            CK(cudaStreamSynchronize(st));
        } else {
            // ---- the G1 segment-graph driver (r19v) ----
            // NL+1 = 49 sync-bounded segments (the 48 router syncs + the final logits sync):
            //   seg 0 = head + L0 pre + L0 router | seg k (1..47) = L(k-1) post + [the PLE
            //   kernels if k == PLE_LAYER] + Lk pre + Lk router | seg 48 = L47 post + tail.
            // The FIRST step captures each segment right before its first replay (the ops are
            // RECORDED, not executed; every arg is a fixed steady-state buffer - the census
            // found no varying arg after the pos fix), later steps replay only. The host
            // windows between replays are exactly the direct path's host work (the sync + the
            // router softmax/top-10/we + the census + the OFF staging memcpys + the PLE
            // gather), refreshing the pinned sources the replayed memcpy nodes re-carry.
            // The RELAXED capture mode keeps the emission's diagnostic queries
            // (CK(cudaGetLastError)/check_launch) legal mid-capture (single-threaded engine).
            for (int k = 0; k <= NL; k++) {
                if ((int)c->gexec.size() <= k) {
                    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed));
                    if (k == 0) {
                        emit_head(c);
                        emit_pre(c, c->layers[0]);
                        emit_router(c, c->layers[0]);
                    } else if (k < NL) {
                        emit_post(c, c->layers[k - 1]);
                        if (k == PLE_LAYER) emit_ple_kernels(c);  // ple_host ran in the window
                        emit_pre(c, c->layers[k]);
                        emit_router(c, c->layers[k]);
                    } else {
                        emit_post(c, c->layers[NL - 1]);
                        emit_tail(c);
                    }
                    cudaGraph_t g;
                    CK(cudaStreamEndCapture(st, &g));
                    cudaGraphExec_t ex;
                    CK(cudaGraphInstantiate(&ex, g, 0));
                    c->ggraph.push_back(g);
                    c->gexec.push_back(ex);
                }
                CK(cudaGraphLaunch(c->gexec[k], st));
                if (k < NL) {
                    host_router(c, c->layers[k]);  // the sync + the host window work
                    if (k + 1 == PLE_LAYER) ple_host(c, token);  // the PLE gather for seg 1
                    if ((k & 15) == 15) fprintf(stderr, "[cf] step %d: layer %d routed\n", c->pos, k);
                }
            }
            CK(cudaStreamSynchronize(st));
        }
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

// cf-m4 (r19w): one MTP draft forward (the frozen CF_MTP.md section 1 - the exact block
// forward of cf-arch.md section 6). The pair (token, h) at the draft's OWN KV position:
// h = the trunk's pre-final-mixer wide residual (a device pointer, valid until the next
// trunk step), or nullptr = the h_{-1} = 0 pair (the 27B's gate-proven form). The draft's
// logits land in d->h_logits (the prediction for the NEXT position); the chain state
// d->hres holds the final res' (the next chained draft's h). STAGING-FREE: the per-step W
// table composes the 10 picks' ALL-RESIDENT Q8_0 views (the tiering's mechanism, all hits,
// zero host staging) and the router runs the SAME order-exact host loops (host_top10, the
// r19t bar). The draft's pos word rides d_params[1] (the spare slot), written in this
// driver BEFORE the upload (the r19v discipline: the write precedes the enqueue).
bool cf_draft_step(CfCtx* c, int token, const float* h) {
    try {
        if (!c->draft) throw std::runtime_error("cf_draft_step: no draft block (T4Q_CF_MTP=1 at load)");
        if (token < 0 || token >= V) throw std::runtime_error("draft token id out of range");
        CfDraft* d = c->draft;
        CfLayer& L = d->L;
        CfScratch& s = d->sc;
        cudaStream_t st = c->st;
        if (d->pos >= c->max_ctx) throw std::runtime_error("draft context full");
        CK(cudaSetDevice(c->gpu));
        // the pair's h input: the arg (a D2D so the source is stable across the chain) or
        // the h_{-1} = 0 zeros
        CK(cudaMemcpyAsync(d->h_in, h ? h : d->zero_h, (size_t)HCD * 4, cudaMemcpyDeviceToDevice, st));
        // e = token_embd[token] (the host dequant into the draft's own host row)
        if (!dequant_row_cpu(c->tok_embd->type, c->tok_embd->data + (size_t)token * c->tok_embd->row_bytes, d->h_e, D))
            throw std::runtime_error("draft embedding dequant failed");
        CK(cudaMemcpyAsync(d->e, d->h_e, (size_t)D * 4, cudaMemcpyHostToDevice, st));
        // the draft's pos word (the spare slot): the write PRECEDES the upload's enqueue
        c->h_params[1] = d->pos;
        CK(cudaMemcpyAsync(c->d_params + 1, c->h_params + 1, 4, cudaMemcpyHostToDevice, st));
        // e_norm = RMSNorm(e, enorm) [D]; h_norm = the grouped RMSNorm(h, hnorm) [HCD];
        // the per-stream [e_norm ; h_norm_s] gather [4][2D]
        launch_rmsnorm(d->e, d->enorm, d->e_norm, D, EPS, st);
        launch_cf_hc_norm(d->h_in, d->hnorm, d->h_norm, st);
        launch_cf_eh_gather(d->e_norm, d->h_norm, d->eh_cat, st);
        // res' = [u_0..u_3]: the eh_proj gemv per stream (the shared [2560,5120] Q8_0)
        for (int sd = 0; sd < HC; sd++)
            gemv(s, d->eh_proj, d->eh_cat + (size_t)sd * 2 * D, d->hres + (size_t)sd * D, st);
        check_launch("draft eh_proj");
        // a = FullAttn(blk.48, hc_attn_mix(res'), the draft's own pos/KV): the attention()
        // twin with the draft's own weights + KV + scratch + pos word slot
        hc_mix(c, d->hres, L.hc_norm[0], L.hc_down[0], L.hc_up[0], &L.hc_inject[0], s);
        gemv(s, L.wq, s.mixed, s.qfull, st);
        gemv(s, L.wk, s.mixed, s.k, st);
        gemv(s, L.wv, s.mixed, s.v, st);
        check_launch("draft attn proj");
        launch_cf_qk_norm_rope(s.qfull, s.k, L.q_norm, L.k_norm, s.aq, s.ak, c->d_params + 1, EPS, ROPE_BASE, NROT,
                              st);
        launch_cf_kv_store(s.ak, s.v, L.kc, L.vc, c->d_params + 1, c->max_ctx, st);
        launch_cf_attn_decode(s.aq, L.kc, L.vc, s.att, s.scores, c->d_params + 1, c->max_ctx, 1.0f / 16.0f, st);
        check_launch("draft attn");
        launch_gate_sigmoid(s.att, s.qfull, s.attg, st);  // 24 blocks, [24][512]: same layout
        gemv(s, L.wo, s.attg, s.block, st);
        check_launch("draft attn out");
        hc_combine(c, d->hres, s.block, s.inj);
        // m = MoE(blk.48, hc_ffn_mix(res')) + the gated shared expert
        hc_mix(c, d->hres, L.hc_norm[1], L.hc_down[1], L.hc_up[1], &L.hc_inject[1], s);
        gemv(s, L.router, s.mixed, s.logits, st);  // borrow the draft's logits scratch [512 of V]
        CK(cudaMemcpyAsync(c->h_router, s.logits, (size_t)NE * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        host_top10(c);  // the SAME order-exact loops as the trunk's host window
        // the W table: the 10 picks' ALL-RESIDENT views (the tiering's mechanism, all hits;
        // the Q8_0 packed plane strides: codes D/EE bytes per row, d D/32 per row)
        for (int k = 0; k < TOPK; k++) {
            const int64_t e = c->eid[k];
            PackedW& v = d->h_wt_gu[k];
            v = d->res_gu;
            v.rows = 2 * EE;
            v.codes = d->res_gu.codes + (size_t)e * 2 * EE * D;
            v.d = d->res_gu.d + (size_t)e * 2 * EE * (D / 32);
            PackedW& w = d->h_wt_dn[k];
            w = d->res_dn;
            w.rows = D;
            w.codes = d->res_dn.codes + (size_t)e * D * EE;
            w.d = d->res_dn.d + (size_t)e * D * (EE / 32);
        }
        CK(cudaMemcpyAsync(d->wt_gu, d->h_wt_gu, (size_t)TOPK * sizeof(PackedW), cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d->wt_dn, d->h_wt_dn, (size_t)TOPK * sizeof(PackedW), cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(c->we, c->we_h, (size_t)TOPK * 4, cudaMemcpyHostToDevice, st));
        // the batched Q8_0 gemvs (the gate|up on the shared mixed - strides 0; the down on
        // the per-pick ffa) - the trunk's moe family with the draft's own buffers
        launch_quantize_q8_0(s.mixed, D, s.xq0, s.xd0, s.xs0, st);
        launch_gemv_q8_0_b(d->wt_gu, 2 * EE, s.xq0, s.xd0, s.xs0, d->ygu, 0, 2 * EE, 0, TOPK, st);
        launch_cf_silu_mul_b(d->ygu, s.ffa, EE, TOPK, st);  // [gate n | up n] per expert, stride 2n
        launch_quantize_q8_0(s.ffa, TOPK * EE, s.xq0, s.xd0, s.xs0, st);
        launch_gemv_q8_0_b(d->wt_dn, D, s.xq0, s.xd0, s.xs0, d->ye, EE, D, EE / 32, TOPK, st);
        // shared expert + its sigmoid gate, then the weighted combine
        gemv(s, L.sh_gate, s.mixed, s.ffg, st);
        gemv(s, L.sh_up, s.mixed, s.ffu, st);
        launch_silu_mul(s.ffg, s.ffu, s.ffa, EE, st);
        gemv(s, L.sh_down, s.ffa, d->ysh, st);
        gemv(s, L.sh_ginp, s.mixed, d->sh_gate_raw, st);
        launch_cf_moe_out(d->ye, c->we, d->ysh, d->sh_gate_raw, s.block, st);
        check_launch("draft moe");
        hc_combine(c, d->hres, s.block, s.inj);
        // the draft's own final mixer (hc_head), then the SHARED lm_head (full vocab)
        hc_mix(c, d->hres, d->hh_norm, d->hh_down, d->hh_up, nullptr, s);
        gemv(s, c->output, s.mixed, s.logits, st);
        check_launch("draft lm_head");
        CK(cudaMemcpyAsync(d->h_logits, s.logits, (size_t)V * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        d->pos++;
        return true;
    } catch (const std::exception& e) {
        c->err = e.what();
        return false;
    }
}

// cf-m4 (r19x): the MTP verify - nr candidate rows through the WHOLE trunk in ONE pass
// (the frozen CF_MTP.md section 5, THE GATE form). Every op is the sequential step's op
// with the ROW's slice (the same kernels, the same args, the same per-row order - the
// rows run strictly in row order, so the rolling states (KV, GDN S/conv, the PLE ring)
// evolve exactly as the sequential steps would; the per-row attention is inherently
// causal - row r's n_kv = pos_r+1 never reaches the later rows' slots). The ONE
// structural change: the per-layer MoE host window BATCHES the rows - all the rows'
// router gemvs + D2Hs (the [nr][NE] pinned slices, no cross-row race), ONE sync, the
// per-row order-exact top-10 (host_top10_row, the r19t bar), the picks' UNION deduped +
// staged/read ONCE (the dedup is the verify's staging win), the per-row W-table views
// into the union slabs, then the per-row batched gemvs + the shared expert + moe_out.
// The rows' logits land in v->h_logits [nr][V] (row r = the prediction for pos+r+1); the
// trunk's pos advances by nr. The per-row pos words ride the [nr] device array (the
// row's pos_dev = d_pos + r) - the single-word form would race the per-row async
// uploads (the write-after-enqueue trap, the r19v class).
bool cf_verify(CfCtx* c, const int* toks, int nr) {
    try {
        if (!c->verify) throw std::runtime_error("cf_verify: no verify block (T4Q_CF_MTP=1 at load)");
        if (nr < 1 || nr > c->verify->nr) throw std::runtime_error("cf_verify: nr out of range");
        if (c->pos + nr > c->max_ctx) throw std::runtime_error("context full");
        CfVerify* v = c->verify;
        cudaStream_t st = c->st;
        CK(cudaSetDevice(c->gpu));
        // ---- the head (per row): the embedding dequants into the pinned [nr][D] slices
        // (all rows up front - the per-row H2Ds read disjoint slices, no race), the toks
        // records (the PLE predecessors + the record order match the sequential), the pos
        // words (the writes precede the ONE upload's enqueue), then the per-row emb H2D +
        // the wide-residual identity init
        for (int r = 0; r < nr; r++) {
            if (toks[r] < 0 || toks[r] >= V) throw std::runtime_error("verify token id out of range");
            if (!dequant_row_cpu(c->tok_embd->type, c->tok_embd->data + (size_t)toks[r] * c->tok_embd->row_bytes,
                                 v->h_emb + (size_t)r * D, D))
                throw std::runtime_error("verify embedding dequant failed");
            c->toks[c->pos + r] = toks[r];
            v->h_pos[r] = c->pos + r;
        }
        CK(cudaMemcpyAsync(v->d_pos, v->h_pos, (size_t)nr * 4, cudaMemcpyHostToDevice, st));
        for (int r = 0; r < nr; r++) {
            CfScratch& s = v->sc[r];
            CK(cudaMemcpyAsync(s.h, v->h_emb + (size_t)r * D, (size_t)D * 4, cudaMemcpyHostToDevice, st));
            launch_cf_res_init(s.h, s.h, st);  // in-place: first-D writes are identities, safe
        }
        check_launch("verify head");
        // ---- the layers: per row [the PLE at layer 1] pre -> router; then the ONE
        // batched host window; then per row the moe rest + the combine
        for (int il = 0; il < NL; il++) {
            CfLayer& L = c->layers[il];
            for (int r = 0; r < nr; r++) {
                CfScratch& s = v->sc[r];
                const int* pos_dev = v->d_pos + r;  // the row's pos word (the array form)
                if (il == PLE_LAYER) {
                    // the PLE twin per row: the hash + the 16-row dequant into the row's
                    // pinned slice (no cross-row H2D race), then emit_ple_kernels verbatim
                    // with the row's scratch (the ple_key/ple_query/ple_s/ple_gate/
                    // ple_gated buffers are shared - the stream order serializes the rows,
                    // the same device-side reuse the sequential steps make) + the ring
                    ple_host_core(c, toks[r], v->h_pos[r], v->h_ple + (size_t)r * D);
                    CK(cudaMemcpyAsync(s.mixed, v->h_ple + (size_t)r * D, (size_t)D * 4, cudaMemcpyHostToDevice, st));
                    gemv(s, c->ple.key, s.mixed, c->ple_key, st);
                    gemv(s, c->ple.value, s.mixed, s.block, st);
                    launch_cf_hc_norm(c->ple_key, c->ple.norm_key, c->ple_key, st);
                    launch_cf_hc_norm(s.h, c->ple.norm_query, c->ple_query, st);
                    launch_cf_ple_sg(c->ple_key, c->ple_query, c->ple_s, c->ple_gate, st);
                    launch_cf_ple_gated(s.block, c->ple_gate, c->ple_gated, st);
                    launch_cf_hc_norm(c->ple_gated, c->ple.norm_conv, c->ple_key, st);  // reuse the dead key buffer
                    launch_cf_ple_conv(c->ple_key, c->ple_hist, c->ple.conv_w, c->ple_query, st);
                    launch_add(c->ple_query, c->ple_gated, HCD, st);  // t = gated + conv
                    launch_add(s.h, c->ple_query, HCD, st);           // res += t (the row's h)
                }
                // the pre twin (emit_pre verbatim with the row's slices)
                hc_mix(c, s.h, L.hc_norm[0], L.hc_down[0], L.hc_up[0], &L.hc_inject[0], s);
                if (L.attn) {
                    gemv(s, L.wq, s.mixed, s.qfull, st);
                    gemv(s, L.wk, s.mixed, s.k, st);
                    gemv(s, L.wv, s.mixed, s.v, st);
                    check_launch("verify attn proj");
                    launch_cf_qk_norm_rope(s.qfull, s.k, L.q_norm, L.k_norm, s.aq, s.ak, pos_dev, EPS, ROPE_BASE,
                                          NROT, st);
                    launch_cf_kv_store(s.ak, s.v, L.kc, L.vc, pos_dev, c->max_ctx, st);
                    launch_cf_attn_decode(s.aq, L.kc, L.vc, s.att, s.scores, pos_dev, c->max_ctx, 1.0f / 16.0f, st);
                    check_launch("verify attn");
                    launch_gate_sigmoid(s.att, s.qfull, s.attg, st);  // 24 blocks, [24][512]: same layout
                    gemv(s, L.wo, s.attg, s.block, st);
                    check_launch("verify attn out");
                } else {
                    deltanet(c, L, s);  // the row's scratch; L.S/L.conv_state evolve in row order
                }
                hc_combine(c, s.h, s.block, s.inj);
                hc_mix(c, s.h, L.hc_norm[1], L.hc_down[1], L.hc_up[1], &L.hc_inject[1], s);
                // the router twin (emit_router verbatim with the row's slices): the gemv
                // borrows the row's logits scratch [512 of V], the D2H lands in the row's
                // pinned router slice
                gemv(s, L.router, s.mixed, s.logits, st);
                CK(cudaMemcpyAsync(v->h_router + (size_t)r * NE, s.logits, (size_t)NE * 4, cudaMemcpyDeviceToHost,
                                   st));
            }
            // ---- the ONE batched host window: the sync (the layer's routers have landed),
            // the per-row order-exact top-10, the union dedup, the union staging (the OFF
            // chunked passes through the trunk's raw_stage / the UVA scatters), the per-row
            // W-table compose + the uploads
            CK(cudaStreamSynchronize(st));
            v->nu = 0;
            for (int r = 0; r < nr; r++)
                host_top10_row(v->h_router + (size_t)r * NE, v->eid + (size_t)r * TOPK, v->we_h + (size_t)r * TOPK);
            for (int r = 0; r < nr; r++)
                for (int k = 0; k < TOPK; k++) {
                    const int e = v->eid[(size_t)r * TOPK + k];
                    if (v->uidx[e] < 0) { v->uidx[e] = v->nu; v->uids[v->nu++] = e; }
                }
            const size_t gu_row = L.t_gate_exps->row_bytes;   // 840
            const size_t dn_row = L.t_down_exps->row_bytes;   // 360
            const size_t up_bytes = (size_t)TOPK * 2 * EE * gu_row;  // the raw_stage layout
            if (c->uva && L.il < c->uva_n) {
                // the UVA union scatters (cf-m3): each union expert's OWN slab read once
                // through the registered aliases (the dedup preserved), the same address
                // math as the OFF passes' memcpy sources, the same repack block decode
                // into the union slabs - byte-identical by construction, no host staging
                CK(cudaMemcpyAsync(v->uids_dev, v->uids, (size_t)v->nu * 4, cudaMemcpyHostToDevice, st));
                launch_repack_eid_q2k(v->uni_gu, L.uva_gate, L.uva_up, v->uids_dev, 0, (int64_t)v->nu * 2 * EE,
                                      (int64_t)2 * EE, (int64_t)EE, st);
                launch_repack_eid_q4(v->uni_dn, L.uva_dn, v->uids_dev, 0, (int64_t)v->nu * D, (int64_t)D, st);
            } else {
                // the OFF chunked passes (the hot-set tier's pass form): the union's experts
                // through the trunk's raw_stage (TOPK experts per pass), the repacks into
                // the union slabs at the slot row offsets
                for (int p0 = 0; p0 < v->nu; p0 += TOPK) {
                    const int ch = std::min(TOPK, v->nu - p0);
                    CK(cudaStreamSynchronize(st));  // the previous pass's H2D must drain before the refill
                    for (int j = 0; j < ch; j++) {
                        const int64_t e = v->uids[p0 + j];
                        uint8_t* dst = c->raw_stage + (size_t)j * 2 * EE * gu_row;
                        memcpy(dst, L.t_gate_exps->data + (size_t)e * EE * gu_row, (size_t)EE * gu_row);
                        memcpy(dst + (size_t)EE * gu_row, L.t_up_exps->data + (size_t)e * EE * gu_row,
                               (size_t)EE * gu_row);
                        memcpy(c->raw_stage + up_bytes + (size_t)j * D * dn_row,
                               L.t_down_exps->data + (size_t)e * D * dn_row, (size_t)D * dn_row);
                    }
                    CK(cudaMemcpyAsync(c->raw_dev, c->raw_stage, (size_t)ch * 2 * EE * gu_row, cudaMemcpyHostToDevice,
                                       st));
                    CK(cudaMemcpyAsync(c->raw_dev + up_bytes, c->raw_stage + up_bytes, (size_t)ch * D * dn_row,
                                       cudaMemcpyHostToDevice, st));
                    launch_repack(v->uni_gu, GT_Q2_K, c->raw_dev, (int64_t)p0 * 2 * EE, (int64_t)ch * 2 * EE, st);
                    launch_repack(v->uni_dn, GT_Q4_0, c->raw_dev + up_bytes, (int64_t)p0 * D, (int64_t)ch * D, st);
                    CK(cudaGetLastError());
                }
            }
            // the per-row W tables: the row's pick k -> the union slot of eid[r][k] (the
            // tiering's hit-branch view math verbatim), then the map sweep for the next
            // layer's dedup
            for (int r = 0; r < nr; r++)
                for (int k = 0; k < TOPK; k++) {
                    const int e = v->eid[(size_t)r * TOPK + k];
                    const int64_t sl = v->uidx[e];
                    PackedW& vg = v->h_wt_gu[r][k];
                    vg = v->uni_gu;
                    vg.rows = 2 * EE;
                    vg.codes = v->uni_gu.codes + (size_t)sl * 2 * EE * (D / 4);
                    vg.meta = v->uni_gu.meta + (size_t)sl * 2 * EE * (size_t)(D / 256) * 20;
                    PackedW& wd = v->h_wt_dn[r][k];
                    wd = v->uni_dn;
                    wd.rows = D;
                    wd.codes = v->uni_dn.codes + (size_t)sl * D * (EE / 2);
                    wd.d = v->uni_dn.d + (size_t)sl * D * (EE / 32);
                }
            for (int i = 0; i < v->nu; i++) v->uidx[v->uids[i]] = -1;
            CK(cudaMemcpyAsync(v->wt_gu, v->h_wt_gu, (size_t)nr * TOPK * sizeof(PackedW), cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(v->wt_dn, v->h_wt_dn, (size_t)nr * TOPK * sizeof(PackedW), cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(v->we_dev, v->we_h, (size_t)nr * TOPK * 4, cudaMemcpyHostToDevice, st));
            // ---- the per-row moe rest + the combine (emit_moe_rest/emit_post verbatim with
            // the row's slices): the gu gemv borrows the row's logits scratch (the router's
            // D2H long since drained - stream order), the down gemv's y is the row's ye
            // slice, the moe_out reads the row's we slice
            for (int r = 0; r < nr; r++) {
                CfScratch& s = v->sc[r];
                float* ye_r = v->ye + (size_t)r * TOPK * D;
                launch_quantize_q8_K(s.mixed, D, s.xqk, s.xqk_b, s.xqk_d, st);
                launch_gemv_q8k_b(v->wt_gu + (size_t)r * TOPK, FMT_K2, 2 * EE, s.xqk, s.xqk_b, s.xqk_d, s.logits, 0,
                                  0, 0, (int64_t)2 * EE, TOPK, st);
                launch_cf_silu_mul_b(s.logits, s.ffa, EE, TOPK, st);  // [gate n | up n] per expert, stride 2n
                launch_quantize_q8_0(s.ffa, TOPK * EE, s.xq0, s.xd0, s.xs0, st);
                launch_gemv_q8_0_b(v->wt_dn + (size_t)r * TOPK, D, s.xq0, s.xd0, s.xs0, ye_r, EE, D, EE / 32, TOPK,
                                   st);
                gemv(s, L.sh_gate, s.mixed, s.ffg, st);
                gemv(s, L.sh_up, s.mixed, s.ffu, st);
                launch_silu_mul(s.ffg, s.ffu, s.ffa, EE, st);
                gemv(s, L.sh_down, s.ffa, v->ysh + (size_t)r * D, st);
                gemv(s, L.sh_ginp, s.mixed, v->sh_gate_raw + r, st);
                launch_cf_moe_out(ye_r, v->we_dev + (size_t)r * TOPK, v->ysh + (size_t)r * D, v->sh_gate_raw + r,
                                  s.block, st);
                check_launch("verify moe");
                hc_combine(c, s.h, s.block, s.inj);
            }
        }
        // ---- the tail (per row): the final mixer + the lm_head + the logits D2H into the
        // row's pinned slice (emit_tail verbatim with the row's slices)
        for (int r = 0; r < nr; r++) {
            CfScratch& s = v->sc[r];
            hc_mix(c, s.h, c->o_norm, c->o_down, c->o_up, nullptr, s);
            gemv(s, c->output, s.mixed, s.logits, st);
            check_launch("verify lm_head");
            CK(cudaMemcpyAsync(v->h_logits + (size_t)r * V, s.logits, (size_t)V * 4, cudaMemcpyDeviceToHost, st));
        }
        CK(cudaStreamSynchronize(st));
        c->have_logits = false;  // the trunk's h_logits is stale (the rows' logits are in v->h_logits)
        c->pos += nr;
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
    if (c->draft) c->draft->pos = 0;  // cf-m4: the draft's own KV pos (cells beyond it are never read)
    CK(cudaDeviceSynchronize());
    c->pos = 0;
    c->have_logits = false;
}

void cf_free(CfCtx* c) {
    if (!c) return;
    if (c->st) { cudaStreamSynchronize(c->st); cudaStreamDestroy(c->st); }
    for (auto& e : c->gexec) cudaGraphExecDestroy(e);  // r19v: the G1 segment graphs
    for (auto& g : c->ggraph) cudaGraphDestroy(g);
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
    if (c->draft) {  // cf-m4 (r19w): the draft's pinned host rows (the device buffers fall to
        // the cudaDeviceReset below, the same as the trunk's scratch)
        cudaFreeHost(c->draft->h_e);
        cudaFreeHost(c->draft->h_logits);
        delete c->draft;
    }
    if (c->verify) {  // cf-m4 (r19x): the verify's pinned host planes (the per-row scratch
        // and the union slabs fall to the cudaDeviceReset, the trunk's own style)
        cudaFreeHost(c->verify->h_pos);
        cudaFreeHost(c->verify->h_emb);
        cudaFreeHost(c->verify->h_ple);
        cudaFreeHost(c->verify->h_router);
        cudaFreeHost(c->verify->we_h);
        cudaFreeHost(c->verify->h_logits);
        delete c->verify;
    }
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
