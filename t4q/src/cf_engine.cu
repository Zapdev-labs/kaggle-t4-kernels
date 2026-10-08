// cf-m1: the CYBER-FROST decode step. Exact decode math, one host sync per token
// (the router top-k comes back host-side, as do the logits). Single GPU for M1.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>

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
    if (W.fmt == FMT_K2 || W.fmt == FMT_K4 || W.fmt == FMT_IQ1S) {
        // cf-m6 r2 (CF_REQUANT.md section 4): FMT_IQ1S rides the same q8_K pairing - the
        // launch_gemv_q8k FMT_IQ1S dot (the nibble-grid dp4a + the bsums correction); no new
        // activation format, the same quantize the K2/K4 trunk uses.
        launch_quantize_q8_K(x, (int)W.cols, sc.xqk, sc.xqk_b, sc.xqk_d, st);
        launch_gemv_q8k(W, sc.xqk, sc.xqk_b, sc.xqk_d, y, st);
    } else if (W.fmt == FMT_Q51) {
        launch_quantize_q8_1(x, (int)W.cols, sc.xq1, sc.xq1_d, sc.xq1_s, st);
        launch_gemv_q8(W, sc.xq1, sc.xq1_d, sc.xq1_s, y, st);
    } else if (W.fmt == FMT_P4 || W.fmt == FMT_Q8 || W.fmt == FMT_IQ1SH) {
        // FMT_Q8 (cf-m4, r19w): the Q8_0 weights pair with the SAME q8_0-form activation
        // (amax/127, no s-term) - launch_quantize_q8_0 already produces it; the dot is the
        // gate-proven FAST_Q8 int8-dp4a family (the 27B's own MTP graft passed the
        // byte-identical spec gates on this exact arithmetic)
        // FMT_IQ1SH (cf-m6 r3): the dn's iq1_s half-block rides the same q8_0 pairing - its
        // 32-elem activation blocks tile the dn's 640-wide rows (the q8_K 256-blocks do
        // not); the launch_gemv_q8_0 FMT_IQ1SH dot carries the per-32 signed-sum correction.
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
// cf-m4 (r19aa): ONE source for the PLE row enumeration (the r19t bar) - the gather
// (ple_host_core) and the prefetch's page touch (ple_touch_rows) must walk the
// IDENTICAL (n, g, h, row) sequence or the prefetch warms the wrong pages (a silent
// perf loss, no correctness hit - still a bug class). The walk is the verbatim hash
// loop extracted; the caller supplies the per-row action.
template <class F>
void ple_walk_rows(CfCtx* c, const int64_t* ctx, F&& f) {
    for (int n = 2; n <= PLE_NGRAM; n++) {
        uint64_t mixed = (uint64_t)ctx[0] * c->ple.mult[0];
        for (int j = 1; j < n; j++) mixed ^= (uint64_t)ctx[j] * c->ple.mult[j];
        for (int g = 0; g < PLE_NHEADS / 2; g++) {
            const int h = (n - 2) * (PLE_NHEADS / 2) + g;
            const uint64_t row = mixed % c->ple.head_vocab[h] + c->ple.head_off[h];
            f(h, row);
        }
    }
}

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
    ple_walk_rows(c, ctx, [&](int h, uint64_t row) {
        if (!dequant_row_cpu(GT_Q4_0, c->ple.table->data + (size_t)row * c->ple.table->row_bytes,
                             dst + (size_t)h * PLE_DIM, PLE_DIM))
            throw std::runtime_error("ple row dequant failed");
    });
}

// cf-m4 (r19aa): the PLE prefetch (CF_MTP.md section 8, the frozen design). The
// verify's row gathers fault the mmap'd 26.85 GiB table (~98 us/row first-fault class
// on the Kaggle disk); the draft calls' GPU stretches (the host blocked at each call's
// drains) are the only window this engine has. The touch reads the row's span (one
// byte per 4 KiB step + the last byte - the span's pages) so the gather's read lands
// warm. VOLATILE: a dead load can be elided - the volatile read cannot, and the page
// fault is the whole point. Pure host reads of the immutable table: NO CUDA API (the
// single-threaded-engine capture discipline holds - a non-CUDA thread is invisible to
// the stream capture), no value risk (the gather re-reads the same bytes), no
// exception path (the touch cannot fail).
void ple_touch_rows(CfCtx* c, const int64_t* ctx) {
    const size_t rb = c->ple.table->row_bytes;
    ple_walk_rows(c, ctx, [&](int, uint64_t row) {
        const volatile uint8_t* p = c->ple.table->data + (size_t)row * rb;
        for (size_t o = 0; o < rb; o += 4096) (void)p[o];
        (void)p[rb - 1];
    });
}

// the prefetch's ctx: ple_host_core's exact construction with the verify's
// to-be-written records substituted - the verify's driver head writes
// c->toks[pos + r] = vt[r] for EVERY r before the gather reads them, so the gather's
// c->toks[pos + r - k] IS vt[r - k] whenever r >= k; below that it is the trunk's own
// rolling entry (the accepted stream, valid at the spawn) or the pre-context -1 (the
// cut). The EOS/missing cut logic verbatim. Runs on the ENGINE thread at the spawn
// point (during the draft chain, before cf_verify writes the records - race-free by
// ordering, the thread body itself reads no c->toks).
void ple_pre_ctx(CfCtx* c, const int* vt, int r, int pos, int64_t* ctx) {
    ctx[0] = vt[r];
    bool cut = false;
    for (int k = 1; k < PLE_NGRAM; k++) {
        const int t = r - k >= 0 ? vt[r - k] : (pos + r - k >= 0 ? c->toks[pos + r - k] : -1);
        cut = cut || t < 0 || t == EOS1;
        ctx[k] = cut ? EOS1 : t;
    }
}

// the per-round prefetch threads: joined on EVERY exit path (an unjoined std::thread
// terminates the process at destruction - the returns inside the draft chain
// included); the threads run through the verify (the gather coalesces with any
// in-flight page fault - concurrent faults on the same page wait for the first, so a
// lagging thread never ADDS wall, it only warms).
struct PlePreThreads {
    std::vector<std::thread> ts;
    ~PlePreThreads() {
        for (auto& t : ts)
            if (t.joinable()) t.join();
    }
};

void ple_pre_spawn(PlePreThreads& pt, CfCtx* c, const int* vt, int r, int pos) {
    std::array<int64_t, PLE_NGRAM> ctx;
    ple_pre_ctx(c, vt, r, pos, ctx.data());
    pt.ts.emplace_back([c, ctx]() { ple_touch_rows(c, ctx.data()); });  // no CUDA, no throw
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
    if (c->uva && L.il >= c->uva_lo && L.il < c->uva_n) {  // cf-m6 r4: [uva_lo, uva_n) - the iq1_s prefix excluded
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
                if (L.res_gu.fmt == FMT_IQ1S) {
                    // cf-m6 r4 (the iq1_s tier): the FMT_IQ1S/FMT_IQ1SH plane strides - the
                    // expert h's rows at h*(2*EE) into the resident planes, per row (D/256)
                    // blocks of 32/16 B + (D/256) d ELEMENTS (gu; d is uint16_t* - the
                    // element count, no byte factor, the K2 form's own convention) and
                    // (EE/128) blocks of 16/8 B + (EE/128) d elements (dn); the fmt rides
                    // the PackedW copy so the emission's launch picks the r4 dots.
                    v = L.res_gu;
                    v.rows = 2 * EE;
                    v.codes = L.res_gu.codes + (size_t)h * 2 * EE * (size_t)(D / 256) * 32;
                    v.hi = L.res_gu.hi + (size_t)h * 2 * EE * (size_t)(D / 256) * 16;
                    v.d = L.res_gu.d + (size_t)h * 2 * EE * (size_t)(D / 256);
                    w = L.res_dn;
                    w.rows = D;
                    w.codes = L.res_dn.codes + (size_t)h * D * (size_t)(EE / 128) * 16;
                    w.hi = L.res_dn.hi + (size_t)h * D * (size_t)(EE / 128) * 8;
                    w.d = L.res_dn.d + (size_t)h * D * (size_t)(EE / 128);
                } else {
                    v = L.res_gu;
                    v.rows = 2 * EE;
                    v.codes = L.res_gu.codes + (size_t)h * 2 * EE * (D / 4);
                    v.meta = L.res_gu.meta + (size_t)h * 2 * EE * (size_t)(D / 256) * 20;
                    w = L.res_dn;
                    w.rows = D;
                    w.codes = L.res_dn.codes + (size_t)h * D * (EE / 2);
                    w.d = L.res_dn.d + (size_t)h * D * (EE / 32);
                }
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
    if (c->uva && L.il >= c->uva_lo && L.il < c->uva_n) {  // cf-m6 r4: [uva_lo, uva_n) - the iq1_s prefix excluded
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
    // cf-m6 r4: an iq1_s-covered layer's picks are ALL hits on the FMT_IQ1S/FMT_IQ1SH
    // resident planes - the SAME quantizes (the pairings coincide: the gu rides q8_K, the
    // dn rides q8_0) with the r4 batched IQ1S/IQ1SH dots; no within-layer format mix
    // exists (whole-layer residency), so ONE branch per layer, not per pick.
    const bool iqs = c->tiered && L.res_gu.fmt == FMT_IQ1S;
    launch_quantize_q8_K(s.mixed, D, s.xqk, s.xqk_b, s.xqk_d, st);
    if (iqs)
        launch_gemv_q8k_b(c->wt_gu, FMT_IQ1S, 2 * EE, s.xqk, s.xqk_b, s.xqk_d, s.logits, 0, 0, 0, (int64_t)2 * EE,
                          TOPK, st);
    else
        launch_gemv_q8k_b(c->wt_gu, FMT_K2, 2 * EE, s.xqk, s.xqk_b, s.xqk_d, s.logits, 0, 0, 0, (int64_t)2 * EE,
                          TOPK, st);
    launch_cf_silu_mul_b(s.logits, s.ffa, EE, TOPK, st);  // [gate n | up n] per expert, stride 2n
    {  // the down gemv as one batched launch over the 10-expert SoA staging, on the Q8_0
        // pairing (the oracle's own arithmetic for the Q4_0 down experts); the ffa [TOPK][EE]
        // is flat and the expert boundaries are the 32-group boundaries, so one flat quantize
        launch_quantize_q8_0(s.ffa, TOPK * EE, s.xq0, s.xd0, s.xs0, st);
        if (iqs)
            launch_gemv_iq1sh_b(c->wt_dn, D, s.xq0, s.xd0, s.xs0, c->ye, EE, D, EE / 32, TOPK, st);
        else
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

// ---- cf-m4 (r19z): the verify's emission split (the r19v pattern - ONE source for the
// direct driver and the segment graphs; the bodies are the r19x inline blocks verbatim).
// The two moves, both value-safe: (1) the per-row PLE host gathers hoist out of the
// emission into the driver's window work (vfy_ple_host - the G1 form: host work cannot
// run mid-capture; the gathers write disjoint pinned slices and the row's H2D reads its
// slice at execution, so batching them changes no value); (2) the we upload moves from
// the window's tail to vfy_moe_em's head (the pinned source is capture-legal there, the
// r19v emit_moe_rest precedent; a memcpy among disjoint buffers, written by the window's
// host_top10_row before and read by the moe_out after in both orders).
void vfy_ple_host(CfCtx* c, const int* toks, int nr) {
    CfVerify* v = c->verify;
    for (int r = 0; r < nr; r++) ple_host_core(c, toks[r], v->h_pos[r], v->h_ple + (size_t)r * D);
}

// the head emission: the pos-word upload (the pinned source the captured node re-carries
// at each replay; the driver writes h_pos before the enqueue) + the per-row embedding
// H2Ds (the pinned slices) + the wide-residual identity inits
void vfy_head_em(CfCtx* c, int nr) {
    CfVerify* v = c->verify;
    cudaStream_t st = c->st;
    CK(cudaMemcpyAsync(v->d_pos, v->h_pos, (size_t)nr * 4, cudaMemcpyHostToDevice, st));
    for (int r = 0; r < nr; r++) {
        CfScratch& s = v->sc[r];
        CK(cudaMemcpyAsync(s.h, v->h_emb + (size_t)r * D, (size_t)D * 4, cudaMemcpyHostToDevice, st));
        launch_cf_res_init(s.h, s.h, st);  // in-place: first-D writes are identities, safe
    }
    check_launch("verify head");
}

// the per-layer pre emission - the r19x per-row block verbatim: per row [the PLE twin's
// stream ops (layer 1: the H2D from the row's pinned gather slice + the 8 launches with
// the row's scratch + the r19y PLE ring snapshot) + the pre twin (the hc mix, the
// attention twin with the row's pos word, or deltanet with the r19y S/conv snapshots) +
// the router gemv + the D2H into the row's pinned router slice]
void vfy_pre_em(CfCtx* c, CfLayer& L, int nr) {
    CfVerify* v = c->verify;
    cudaStream_t st = c->st;
    for (int r = 0; r < nr; r++) {
        CfScratch& s = v->sc[r];
        const int* pos_dev = v->d_pos + r;  // the row's pos word (the array form)
        if (L.il == PLE_LAYER) {
            // the PLE twin per row: the 8 launches verbatim with the row's scratch (the
            // ple_key/ple_query/ple_s/ple_gate/ple_gated buffers are shared - the stream
            // order serializes the rows, the same device-side reuse the sequential steps
            // make) + the ring (the gather itself ran in the driver's window)
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
            if (r < nr - 1)  // r19y: the PLE ring snapshot (the shift register, not pos-indexed)
                CK(cudaMemcpyAsync(v->ple_snap + (size_t)r * PLE_HIST * HCD, c->ple_hist,
                                   (size_t)PLE_HIST * HCD * 4, cudaMemcpyDeviceToDevice, st));
        }
        // the pre twin (emit_pre verbatim with the row's slices)
        hc_mix(c, s.h, L.hc_norm[0], L.hc_down[0], L.hc_up[0], &L.hc_inject[0], s);
        if (L.attn) {
            gemv(s, L.wq, s.mixed, s.qfull, st);
            gemv(s, L.wk, s.mixed, s.k, st);
            gemv(s, L.wv, s.mixed, s.v, st);
            check_launch("verify attn proj");
            launch_cf_qk_norm_rope(s.qfull, s.k, L.q_norm, L.k_norm, s.aq, s.ak, pos_dev, EPS, ROPE_BASE, NROT, st);
            launch_cf_kv_store(s.ak, s.v, L.kc, L.vc, pos_dev, c->max_ctx, st);
            launch_cf_attn_decode(s.aq, L.kc, L.vc, s.att, s.scores, pos_dev, c->max_ctx, 1.0f / 16.0f, st);
            check_launch("verify attn");
            launch_gate_sigmoid(s.att, s.qfull, s.attg, st);  // 24 blocks, [24][512]: same layout
            gemv(s, L.wo, s.attg, s.block, st);
            check_launch("verify attn out");
        } else {
            deltanet(c, L, s);  // the row's scratch; L.S/L.conv_state evolve in row order
            if (r < nr - 1) {  // r19y: the GDN S/conv snapshots for the partial-accept
                // rollback: stream-ordered D2Ds after the row's gdn_recur/conv writes,
                // before the next row overwrites L.S/L.conv_state
                const size_t ro = (size_t)v->gord[L.il] * (v->nr - 1) + r;
                CK(cudaMemcpyAsync(v->s_snap + ro * (size_t)HV * DK * DK, L.S, (size_t)HV * DK * DK * 4,
                                   cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpyAsync(v->conv_snap + ro * (size_t)CONV * 3, L.conv_state, (size_t)CONV * 3 * 4,
                                   cudaMemcpyDeviceToDevice, st));
            }
        }
        hc_combine(c, s.h, s.block, s.inj);
        hc_mix(c, s.h, L.hc_norm[1], L.hc_down[1], L.hc_up[1], &L.hc_inject[1], s);
        // the router twin (emit_router verbatim with the row's slices): the gemv borrows
        // the row's logits scratch [512 of V], the D2H lands in the row's pinned slice
        gemv(s, L.router, s.mixed, s.logits, st);
        CK(cudaMemcpyAsync(v->h_router + (size_t)r * NE, s.logits, (size_t)NE * 4, cudaMemcpyDeviceToHost, st));
    }
}

// the per-layer host window - the r19x block verbatim: the sync (the layer's routers have
// landed), the per-row order-exact top-10, the union dedup, the union staging (DIRECT -
// the union count varies per layer, not capture-constant: the OFF chunked passes through
// the trunk's raw_stage / the UVA scatters + the uids upload), the per-row W-table compose
// + the uploads (pageable sources, direct)
void vfy_window(CfCtx* c, CfLayer& L, int nr) {
    CfVerify* v = c->verify;
    cudaStream_t st = c->st;
    CK(cudaStreamSynchronize(st));
    v->nu = 0;
    for (int r = 0; r < nr; r++)
        host_top10_row(v->h_router + (size_t)r * NE, v->eid + (size_t)r * TOPK, v->we_h + (size_t)r * TOPK);
    // cf-m6 r4: an iq1_s-covered layer's picks ALL hit the FMT_IQ1S/FMT_IQ1SH residents -
    // NO union staging (nothing staged), the views point at the resident planes (the
    // tiering's hit-branch math verbatim, e = the resident index), and vfy_moe_em takes
    // the r5 AMORTIZED dots. The verify MUST read the SAME weights the greedy path reads
    // or the MTP acceptance compares two different models (the byte-identity gate's class
    // - the originals-staged verify would diverge from the resident-reading greedy at the
    // covered layers).
    // cf-m6 r5: the resident path feeds the AMORTIZED dots - the picks' UNION (the dedup
    // the uncovered path stages by) + the row map (row r's pick index of slot u) + the
    // union's resident views. Each union pick's W decode is shared across every row that
    // picked it (the spec's frozen M=nr form); the per-(row, pick) numerics stay
    // bit-identical to the _b form's (the same walk, the same per-lane group order).
    if (c->tiered && L.res_gu.fmt == FMT_IQ1S) {
        v->nu = 0;
        for (int r = 0; r < nr; r++)
            for (int k = 0; k < TOPK; k++) {
                const int e = v->eid[(size_t)r * TOPK + k];
                if (v->uidx[e] < 0) { v->uidx[e] = v->nu; v->uids[v->nu++] = e; }
            }
        for (int u = 0; u < v->nu; u++) {
            const int e = v->uids[u];
            PackedW& vg = v->h_uv_gu[u];
            vg = L.res_gu;  // fmt FMT_IQ1S rides the copy
            vg.rows = 2 * EE;
            vg.codes = L.res_gu.codes + (size_t)e * 2 * EE * (size_t)(D / 256) * 32;
            vg.hi = L.res_gu.hi + (size_t)e * 2 * EE * (size_t)(D / 256) * 16;
            vg.d = L.res_gu.d + (size_t)e * 2 * EE * (size_t)(D / 256);
            PackedW& wd = v->h_uv_dn[u];
            wd = L.res_dn;  // fmt FMT_IQ1SH rides the copy
            wd.rows = D;
            wd.codes = L.res_dn.codes + (size_t)e * D * (size_t)(EE / 128) * 16;
            wd.hi = L.res_dn.hi + (size_t)e * D * (size_t)(EE / 128) * 8;
            wd.d = L.res_dn.d + (size_t)e * D * (size_t)(EE / 128);
        }
        memset(v->h_rowmap, -1, sizeof(v->h_rowmap));
        for (int r = 0; r < nr; r++)
            for (int k = 0; k < TOPK; k++)
                v->h_rowmap[v->uidx[v->eid[(size_t)r * TOPK + k]]][r] = k;
        for (int i = 0; i < v->nu; i++) v->uidx[v->uids[i]] = -1;  // the map sweep (the next layer's dedup reset)
        CK(cudaMemcpyAsync(v->uv_gu, v->h_uv_gu, (size_t)v->nu * sizeof(PackedW), cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(v->uv_dn, v->h_uv_dn, (size_t)v->nu * sizeof(PackedW), cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(v->rowmap_dev, v->h_rowmap, (size_t)v->nu * T4Q_VFY_MAXR * 4, cudaMemcpyHostToDevice, st));
        return;
    }
    for (int r = 0; r < nr; r++)
        for (int k = 0; k < TOPK; k++) {
            const int e = v->eid[(size_t)r * TOPK + k];
            if (v->uidx[e] < 0) { v->uidx[e] = v->nu; v->uids[v->nu++] = e; }
        }
    const size_t gu_row = L.t_gate_exps->row_bytes;   // 840
    const size_t dn_row = L.t_down_exps->row_bytes;   // 360
    const size_t up_bytes = (size_t)TOPK * 2 * EE * gu_row;  // the raw_stage layout
    if (c->uva && L.il >= c->uva_lo && L.il < c->uva_n) {  // cf-m6 r4: [uva_lo, uva_n) - the iq1_s prefix excluded
        // the UVA union scatters (cf-m3): each union expert's OWN slab read once through
        // the registered aliases (the dedup preserved), the same address math as the OFF
        // passes' memcpy sources, the same repack block decode into the union slabs -
        // byte-identical by construction, no host staging
        CK(cudaMemcpyAsync(v->uids_dev, v->uids, (size_t)v->nu * 4, cudaMemcpyHostToDevice, st));
        launch_repack_eid_q2k(v->uni_gu, L.uva_gate, L.uva_up, v->uids_dev, 0, (int64_t)v->nu * 2 * EE,
                              (int64_t)2 * EE, (int64_t)EE, st);
        launch_repack_eid_q4(v->uni_dn, L.uva_dn, v->uids_dev, 0, (int64_t)v->nu * D, (int64_t)D, st);
    } else {
        // the OFF chunked passes (the hot-set tier's pass form): the union's experts
        // through the trunk's raw_stage (TOPK experts per pass), the repacks into the
        // union slabs at the slot row offsets
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
            CK(cudaMemcpyAsync(c->raw_dev, c->raw_stage, (size_t)ch * 2 * EE * gu_row, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(c->raw_dev + up_bytes, c->raw_stage + up_bytes, (size_t)ch * D * dn_row,
                               cudaMemcpyHostToDevice, st));
            launch_repack(v->uni_gu, GT_Q2_K, c->raw_dev, (int64_t)p0 * 2 * EE, (int64_t)ch * 2 * EE, st);
            launch_repack(v->uni_dn, GT_Q4_0, c->raw_dev + up_bytes, (int64_t)p0 * D, (int64_t)ch * D, st);
            CK(cudaGetLastError());
        }
    }
    // the per-row W tables: the row's pick k -> the union slot of eid[r][k] (the tiering's
    // hit-branch view math verbatim), then the map sweep for the next layer's dedup
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
}

// the per-layer moe emission - the we upload (the pinned source, capture-legal at the
// emission head - the r19v emit_moe_rest precedent) + the r19x per-row block verbatim
// (the batched gemv family on the per-row W tables, the shared expert, the moe_out on the
// row's we slice, the combine)
void vfy_moe_em(CfCtx* c, CfLayer& L, int nr) {
    CfVerify* v = c->verify;
    cudaStream_t st = c->st;
    CK(cudaMemcpyAsync(v->we_dev, v->we_h, (size_t)nr * TOPK * 4, cudaMemcpyHostToDevice, st));
    // cf-m6 r4: an iq1_s-covered layer's rows read the FMT_IQ1S/FMT_IQ1SH residents (the
    // vfy_window resident views) - the SAME quantizes with the r5 AMORTIZED dots, so the
    // verify's numerics are the greedy path's own (the MTP agreement bar). cf-m6 r5: the
    // phases split so ONE launch covers the whole (union pick x row) set per side - the
    // nr quantizes, the gu dot (the W decode shared across the rows that picked the
    // expert), the per-row silu + q8_0 quantizes, the dn dot, then the per-row tail
    // verbatim (the shared expert, the moe_out on the row's we slice, the combine).
    const bool iqs = c->tiered && L.res_gu.fmt == FMT_IQ1S;
    if (iqs) {
        for (int r = 0; r < nr; r++) {
            CfScratch& s = v->sc[r];
            launch_quantize_q8_K(s.mixed, D, s.xqk, s.xqk_b, s.xqk_d, st);
        }
        launch_gemv_iq1s_vfy(v->uv_gu, v->rowmap_dev, v->vtab_dev, 2 * EE, v->nu, st);
        for (int r = 0; r < nr; r++) {
            CfScratch& s = v->sc[r];
            launch_cf_silu_mul_b(s.logits, s.ffa, EE, TOPK, st);  // [gate n | up n] per expert, stride 2n
            launch_quantize_q8_0(s.ffa, TOPK * EE, s.xq0, s.xd0, s.xs0, st);
        }
        launch_gemv_iq1sh_vfy(v->uv_dn, v->rowmap_dev, v->vtab_dev, D, v->nu, st);
        for (int r = 0; r < nr; r++) {
            CfScratch& s = v->sc[r];
            float* ye_r = v->ye + (size_t)r * TOPK * D;
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
        return;
    }
    for (int r = 0; r < nr; r++) {
        CfScratch& s = v->sc[r];
        float* ye_r = v->ye + (size_t)r * TOPK * D;
        launch_quantize_q8_K(s.mixed, D, s.xqk, s.xqk_b, s.xqk_d, st);
        if (iqs)
            launch_gemv_q8k_b(v->wt_gu + (size_t)r * TOPK, FMT_IQ1S, 2 * EE, s.xqk, s.xqk_b, s.xqk_d, s.logits, 0, 0,
                              0, (int64_t)2 * EE, TOPK, st);
        else
            launch_gemv_q8k_b(v->wt_gu + (size_t)r * TOPK, FMT_K2, 2 * EE, s.xqk, s.xqk_b, s.xqk_d, s.logits, 0, 0, 0,
                              (int64_t)2 * EE, TOPK, st);
        launch_cf_silu_mul_b(s.logits, s.ffa, EE, TOPK, st);  // [gate n | up n] per expert, stride 2n
        launch_quantize_q8_0(s.ffa, TOPK * EE, s.xq0, s.xd0, s.xs0, st);
        if (iqs)
            launch_gemv_iq1sh_b(v->wt_dn + (size_t)r * TOPK, D, s.xq0, s.xd0, s.xs0, ye_r, EE, D, EE / 32, TOPK, st);
        else
            launch_gemv_q8_0_b(v->wt_dn + (size_t)r * TOPK, D, s.xq0, s.xd0, s.xs0, ye_r, EE, D, EE / 32, TOPK, st);
        gemv(s, L.sh_gate, s.mixed, s.ffg, st);
        gemv(s, L.sh_up, s.mixed, s.ffu, st);
        launch_silu_mul(s.ffg, s.ffu, s.ffa, EE, st);
        gemv(s, L.sh_down, s.ffa, v->ysh + (size_t)r * D, st);
        gemv(s, L.sh_ginp, s.mixed, v->sh_gate_raw + r, st);
        launch_cf_moe_out(ye_r, v->we_dev + (size_t)r * TOPK, v->ysh + (size_t)r * D, v->sh_gate_raw + r, s.block,
                          st);
        check_launch("verify moe");
        hc_combine(c, s.h, s.block, s.inj);
    }
}

// the tail emission - the r19x per-row block verbatim: the r19y pending_h capture (the
// row's pre-final-mixer residual; the tail's hc_mix only READS s.h), the final mixer, the
// shared lm_head, the logits D2H into the row's pinned slice
void vfy_tail_em(CfCtx* c, int nr) {
    CfVerify* v = c->verify;
    cudaStream_t st = c->st;
    for (int r = 0; r < nr; r++) {
        CfScratch& s = v->sc[r];
        CK(cudaMemcpyAsync(v->pending_h + (size_t)r * HCD, s.h, (size_t)HCD * 4, cudaMemcpyDeviceToDevice, st));
        hc_mix(c, s.h, c->o_norm, c->o_down, c->o_up, nullptr, s);
        gemv(s, c->output, s.mixed, s.logits, st);
        check_launch("verify lm_head");
        CK(cudaMemcpyAsync(v->h_logits + (size_t)r * V, s.logits, (size_t)V * 4, cudaMemcpyDeviceToHost, st));
    }
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
        // ---- the driver's head (host): the embedding dequants into the pinned [nr][D]
        // slices (all rows up front - the per-row H2D nodes read disjoint slices, no
        // race), the toks records (the PLE predecessors + the record order match the
        // sequential), the pos words (the writes precede the emission's ONE upload
        // enqueue - the r19v discipline, so the captured node re-carries them at each
        // replay)
        for (int r = 0; r < nr; r++) {
            if (toks[r] < 0 || toks[r] >= V) throw std::runtime_error("verify token id out of range");
            if (!dequant_row_cpu(c->tok_embd->type, c->tok_embd->data + (size_t)toks[r] * c->tok_embd->row_bytes,
                                 v->h_emb + (size_t)r * D, D))
                throw std::runtime_error("verify embedding dequant failed");
            c->toks[c->pos + r] = toks[r];
            v->h_pos[r] = c->pos + r;
        }
        if (c->gmode && nr == v->nr) {
            // ---- the V1 segment-graph driver (r19z): the G1 pattern applied to the
            // verify. The NL+1 = 49 sync-bounded segments (seg 0 = the head emission +
            // L0's rows; seg k = L(k-1)'s moe emission + Lk's pre emission; seg 48 =
            // L47's moe emission + the tail emission), captured at the FIRST full-nr
            // call and replayed after - the direct form's launch wall (~2600 launches
            // x nr rows) collapses to ~1 graph launch + the window's DIRECT staging
            // per layer. Every varying content rides a pinned-fixed host source the
            // captured memcpy nodes re-carry at each replay (the pos words, the emb
            // rows, the PLE gather rows, the router D2Hs, the logits D2Hs, the we
            // rows); the r19y snapshot D2Ds are fixed-arg nodes; the union staging +
            // the W-table/uids uploads stay DIRECT in the windows (the union count
            // varies per layer, not capture-constant). The sync before each capture is
            // the idle-stream invariant (the G1 captures always began on a drained
            // stream; the verify's windows, unlike the step's, enqueue the staging) -
            // it runs only during the build, the steady state just replays.
            // Partial-nr calls (the gate mode's tail chunks) never enter here: the
            // captured shapes are nr-bound, they take the direct path below.
            for (int k = 0; k <= NL; k++) {
                if ((int)v->vgexec.size() <= k) {
                    CK(cudaStreamSynchronize(st));  // the capture's idle-stream invariant
                    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed));
                    if (k == 0) {
                        vfy_head_em(c, nr);
                        vfy_pre_em(c, c->layers[0], nr);
                    } else if (k < NL) {
                        vfy_moe_em(c, c->layers[k - 1], nr);
                        vfy_pre_em(c, c->layers[k], nr);
                    } else {
                        vfy_moe_em(c, c->layers[NL - 1], nr);
                        vfy_tail_em(c, nr);
                    }
                    cudaGraph_t g;
                    CK(cudaStreamEndCapture(st, &g));
                    cudaGraphExec_t ex;
                    CK(cudaGraphInstantiate(&ex, g, 0));
                    v->vggraph.push_back(g);
                    v->vgexec.push_back(ex);
                }
                CK(cudaGraphLaunch(v->vgexec[k], st));
                if (k < NL) {
                    vfy_window(c, c->layers[k], nr);  // the sync + the host work + the DIRECT staging
                    if (k + 1 == PLE_LAYER) vfy_ple_host(c, toks, nr);  // the PLE gathers for seg 1
                }
            }
            CK(cudaStreamSynchronize(st));
        } else {
            // ---- the direct driver (the r19x form): the emission functions in the
            // r19x order - the gmode-off hosts and the partial-nr calls (the gate
            // mode's tail chunks) run here; byte-identical to the graph path (the
            // same ops, the same args, the same order) and to the r19x-gate-proven
            // form up to the two value-safe moves (the PLE gathers batched before
            // the row loop, the we upload at the moe emission head)
            vfy_head_em(c, nr);
            for (int il = 0; il < NL; il++) {
                CfLayer& L = c->layers[il];
                if (il == PLE_LAYER) vfy_ple_host(c, toks, nr);
                vfy_pre_em(c, L, nr);
                vfy_window(c, L, nr);
                vfy_moe_em(c, L, nr);
            }
            vfy_tail_em(c, nr);
            CK(cudaStreamSynchronize(st));
        }
        c->have_logits = false;  // the trunk's h_logits is stale (the rows' logits are in v->h_logits)
        c->pos += nr;
        return true;
    } catch (const std::exception& e) {
        c->err = e.what();
        return false;
    }
}

// cf-m4 (r19y): the greedy pick - the FIRST max (the lowest index on ties), cf_run's
// argmax verbatim, so the spec emissions pick identically to the reference modes on the
// byte-identical logits (a differing tie-break could split an exact tie).
static int pick(const float* x, int n) {
    int b = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[b]) b = i;
    return b;
}

// cf-m4 (r19y): the speculative driver's prompt pass (CF_MTP.md section 4) - the trunk
// over the prompt (one cf_step per token, the same path the reference modes run) with the
// draft paired one step behind (the pairs (ids[j], h_{j-1}), h_{-1} = 0 - the catch-up
// form; the h source c->sc.h is stable until the next trunk step overwrites it, and the
// draft's D2D is enqueued before that step's launches, stream-ordered), then the PENDING
// pair (the greedy argmax after the prompt, h_{np-1}) at the draft's position np. This
// establishes the spec loop's invariant: the draft has processed every position 0..np (its
// own KV slots), its h_logits predict position np+1, and v->pending = the token at np.
// Each cf_step/cf_draft_step call drains the stream before returning, so the pairing never
// interleaves a trunk op with a draft op mid-flight (the r19w rule).
int cf_spec_prime(CfCtx* c, const int* ids, int np) {
    try {
        if (!c->draft || !c->verify) throw std::runtime_error("cf_spec_prime: no MTP block (T4Q_CF_MTP=1 at load)");
        if (np < 1) throw std::runtime_error("cf_spec_prime: empty prompt");
        if (c->pos != 0 || c->draft->pos != 0)
            throw std::runtime_error("cf_spec_prime: not at pos 0 (cf_reset first)");
        for (int j = 0; j < np; j++) {
            if (!cf_draft_step(c, ids[j], j ? c->sc.h : nullptr)) return -1;
            if (!cf_step(c, ids[j])) return -1;
        }
        const int pending = pick(c->h_logits, V);  // the trunk's greedy pick after the prompt
        if (!cf_draft_step(c, pending, c->sc.h)) return -1;
        c->verify->pending = pending;
        return pending;
    } catch (const std::exception& e) {
        c->err = e.what();
        return -1;
    }
}

// cf-m4 (r19y): ONE speculative round - the 27B's gate-proven tp_spec.cu arithmetic
// adapted to the CF engine (CF_MTP.md sections 1/5/6). IN: the loop invariant (the draft
// processed 0..c->pos, its h_logits predicting position c->pos+1; v->pending = the token at
// c->pos - cf_spec_prime or the previous round's catch-up left it so). The round:
// (1) the k drafts: vt[0] = the pending, vt[1] = the pending's draft prediction (in hand),
//     vt[2..k] chained (the draft's own hres as the h input - section 1's chain form; the
//     D2D in cf_draft_step decouples hres from h_in, so the forward's hres overwrite is
//     stream-ordered after the copy);
// (2) the verify over the k+1 rows (the r19x GATE form, now with the r19y per-row
//     snapshot captures + pending_h);
// (3) the argmaxes yv[r] = pick(row r) + the accept scan n = the longest prefix with
//     vt[n+1] == yv[n] (the 27B's scan verbatim; row 0's argmax is always emitted - the
//     pending token's consumption is never rejected);
// (4) the rollback (n < k): the GDN S/conv + PLE snapshots restored to the after-row-n
//     state (stream-ordered D2Ds on the drained stream; the verify's own end sync makes
//     them land before the next round's first op), c->pos = p+n+1 (the attention KV cells
//     and the toks entries beyond are stale-but-invisible - every read is at a position
//     whose entry was written by an accepted consumption or rewritten first);
// (5) the emission yv[0..n] (n+1 tokens; the new pending = yv[n]);
// (6) the catch-up: the draft over the accepted tokens (yv[t], pending_h[t] = the verify
//     row t's pre-final-mixer residual, the same pairing the acceptance smoke ran) at the
//     draft positions p+1..p+n+1 (d->pos rewound to p+1 first - the chain's speculative
//     draft KV slots are re-consumed in order), restoring the invariant (the last call's
//     h_logits predict the new pending's successor = the next round's vt[1]; its hres is
//     the next round's chain state).
// The emitted prefix is the sequential greedy's stream by construction: the verify's rows
// reproduce the sequential logits byte-exactly (the r19x gate), so yv[0..n] are the
// sequential's tokens and the restored state is the sequential's state - the round is the
// reference's own arithmetic, re-batched.
int cf_spec_step(CfCtx* c, int* out) {
    try {
        if (!c->draft || !c->verify) throw std::runtime_error("cf_spec_step: no MTP block (T4Q_CF_MTP=1 at load)");
        CfDraft* d = c->draft;
        CfVerify* v = c->verify;
        const int k = v->nr - 1;
        if (k < 1) throw std::runtime_error("cf_spec_step: k < 1 (T4Q_CF_K >= 1 for the spec driver)");
        const int p = c->pos;  // the pending token's position (v->pending sits here)
        int vt[cf::MAXR];
        vt[0] = v->pending;
        if (vt[0] < 0 || vt[0] >= V) throw std::runtime_error("cf_spec_step: no pending token (cf_spec_prime first)");
        // ---- (1) the drafts: vt[1] from the pending's prediction in hand, vt[2..k] chained
        // r19aa: the PLE prefetch threads (T4Q_CF_PLE_PRE=1, absent = the verbatim round)
        // - rows 0/1 are known before the chain (the pending + its draft prediction), each
        // later row the moment its producing draft call lands; the touches run under the
        // draft calls' GPU stretches + the verify's seg-0 drain, the gather coalesces with
        // any in-flight fault, the holder joins on every exit path
        auto t0 = std::chrono::steady_clock::now();
        vt[1] = pick(d->h_logits, V);
        PlePreThreads pre;
        if (c->ple_pre) {
            ple_pre_spawn(pre, c, vt, 0, p);
            ple_pre_spawn(pre, c, vt, 1, p);
        }
        for (int i = 2; i <= k; i++) {
            if (!cf_draft_step(c, vt[i - 1], d->hres)) return 0;
            vt[i] = pick(d->h_logits, V);
            if (c->ple_pre) ple_pre_spawn(pre, c, vt, i, p);
        }
        v->draft_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        // ---- (2) the verify over the k+1 rows (its own captures ride the per-(layer,row)
        // boundaries; its end sync drains the stream for the rollback below)
        auto t1 = std::chrono::steady_clock::now();
        if (!cf_verify(c, vt, k + 1)) return 0;
        v->verify_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        // ---- (3) the argmaxes + the accept scan (the 27B's arithmetic)
        int yv[cf::MAXR];
        for (int r = 0; r <= k; r++) yv[r] = pick(v->h_logits + (size_t)r * V, V);
        int n = 0;
        while (n < k && vt[n + 1] == yv[n]) n++;
        // ---- (4) the rollback (n < k): the after-row-n snapshots + the pos rewind
        if (n < k) {
            for (int il = 0; il < NL; il++) {
                CfLayer& L = c->layers[il];
                if (L.attn) continue;  // the KV cells beyond the rewound pos are invisible
                const size_t ro = (size_t)v->gord[il] * k + n;
                CK(cudaMemcpyAsync(L.S, v->s_snap + ro * (size_t)HV * DK * DK, (size_t)HV * DK * DK * 4,
                                   cudaMemcpyDeviceToDevice, c->st));
                CK(cudaMemcpyAsync(L.conv_state, v->conv_snap + ro * (size_t)CONV * 3, (size_t)CONV * 3 * 4,
                                   cudaMemcpyDeviceToDevice, c->st));
            }
            CK(cudaMemcpyAsync(c->ple_hist, v->ple_snap + (size_t)n * PLE_HIST * HCD, (size_t)PLE_HIST * HCD * 4,
                               cudaMemcpyDeviceToDevice, c->st));
        }
        c->pos = p + n + 1;  // the trunk's new pending position (the verify advanced it by nr)
        // ---- (5) the emission (the new pending rides yv[n])
        for (int i = 0; i <= n; i++) out[i] = yv[i];
        v->pending = yv[n];
        // ---- (6) the catch-up: the draft over the accepted tokens, then the invariant
        auto t2 = std::chrono::steady_clock::now();
        d->pos = p + 1;
        for (int t = 0; t <= n; t++)
            if (!cf_draft_step(c, yv[t], v->pending_h + (size_t)t * HCD)) return 0;
        v->catch_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
        v->rounds++;
        return n + 1;
    } catch (const std::exception& e) {
        c->err = e.what();
        return 0;
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
    if (c->verify) c->verify->pending = -1;  // r19y: the spec driver's pending (cf_spec_prime re-arms it)
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
        for (auto& e : c->verify->vgexec) cudaGraphExecDestroy(e);  // r19z: the verify segment graphs
        for (auto& g : c->verify->vggraph) cudaGraphDestroy(g);
        cudaFreeHost(c->verify->h_pos);
        cudaFreeHost(c->verify->h_emb);
        cudaFreeHost(c->verify->h_ple);
        cudaFreeHost(c->verify->h_router);
        cudaFreeHost(c->verify->eid);  // r19z sweep: the per-row picks plane
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
