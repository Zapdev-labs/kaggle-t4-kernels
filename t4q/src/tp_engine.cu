// TP=2 engine: shard repack loader, load-time GEMV self-test, step enqueue (eager or CUDA graph), host loop.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <cstdlib>
#include <random>

#include "cupti_trace.h"
#include "kernels/tp_kernels.h"
#include "model.h"
#include "quant_cpu.h"
#include "tp.h"
#include "tp_api.h"

using namespace t4q::gemv;
using hp::D;

namespace {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

int fast_fmt_of(uint32_t t) {
    switch (t) {
        case GT_Q4_0: return FAST_P4;
        case GT_Q4_1: return FAST_P4M;
        case GT_Q5_K: return FAST_K5;
        case GT_Q6_K: return FAST_K6;
        case GT_Q8_0: return FAST_Q8;
        default: return -1;
    }
}

struct Piece {
    const GgufTensor* t;
    int64_t r0, nr;
};
using Cols = std::vector<std::pair<int64_t, int64_t>>;

struct Stage {
    uint8_t* pin = nullptr;
    size_t cap = 0;
    uint8_t* dev[2] = {nullptr, nullptr};
};

const GgufTensor* need(t4q_ctx* c, const std::string& n) {
    const GgufTensor* t = c->f.find(n);
    if (!t) throw std::runtime_error("missing tensor " + n);
    return t;
}

// raw GGUF bytes of local row `lr` (rows are the concatenation of pieces, columns gathered from col ranges)
size_t gather_row(const std::vector<Piece>& rows, const Cols& cols, int64_t lr, uint8_t* dst, int il = 0) {
    if (il > 0) {  // two equal pieces interleaved in groups of il rows
        const int pi = (int)((lr / il) % 2);
        lr = (lr / (2 * il)) * il + lr % il;
        std::vector<Piece> one = {rows[pi]};
        return gather_row(one, cols, lr, dst, 0);
    }
    for (const Piece& p : rows) {
        if (lr >= p.nr) { lr -= p.nr; continue; }
        int be, bb;
        ggml_block_info(p.t->type, be, bb);
        const uint8_t* src = p.t->data + (size_t)(p.r0 + lr) * p.t->row_bytes;
        size_t o = 0;
        for (auto& cr : cols) {
            const size_t n = (size_t)(cr.second / be) * bb;
            memcpy(dst + o, src + (size_t)(cr.first / be) * bb, n);
            o += n;
        }
        return o;
    }
    throw std::runtime_error("gather_row out of range");
}

void build_fw(t4q_ctx* c, Stage& sg, int g, tp::FW& W, const std::vector<Piece>& rows, const Cols& cols, int rpl,
              const char* what, int il = 0) {
    tp::Gpu& G = c->tps->G[g];
    const uint32_t type = rows[0].t->type;
    int64_t N = 0, K = 0;
    for (const Piece& p : rows) {
        if (p.t->type != type || p.t->ne[0] != rows[0].t->ne[0])
            throw std::runtime_error(std::string("mixed types/widths in ") + what);
        N += p.nr;
    }
    for (auto& cr : cols) K += cr.second;
    const int ff = fast_fmt_of(type);
    if (ff < 0) throw std::runtime_error(std::string("no fast format for ") + ggml_type_name(type) + " in " + what);
    const int be = src_block_elems(ff), bb = src_block_bytes(ff);
    for (auto& cr : cols)
        if (cr.first % be || cr.second % be) throw std::runtime_error(std::string("unaligned column split in ") + what);
    if (K % 512) throw std::runtime_error(std::string("K % 512 != 0 in ") + what);
    if ((ff == FAST_P4 || ff == FAST_P4M) && !il) {  // A/B knobs: all P4, N-split (qkvz, qkv_a), K-split (down, wo)
        const std::string w(what);
        const bool nsplit = w.find("qkvz") != std::string::npos || w.find("qkv_a") != std::string::npos;
        const bool ksplit = w.find("ffn_down") != std::string::npos || w.find("attn_output") != std::string::npos;
        if (nsplit) rpl = 2;  // M4 v11 selftest: qkvz 99.6 -> 94.3 us, qkv_a 84.3 -> 83.8 us with 128-thread blocks
        if (getenv("T4Q_RPL_P4")) rpl = atoi(getenv("T4Q_RPL_P4"));
        if (nsplit && getenv("T4Q_RPL_N")) rpl = atoi(getenv("T4Q_RPL_N"));
        if (ksplit && getenv("T4Q_RPL_K")) rpl = atoi(getenv("T4Q_RPL_K"));
    }
    if (ff == FAST_K5 && getenv("T4Q_RPL_K5")) rpl = atoi(getenv("T4Q_RPL_K5"));
    if (il && (il != rpl || rows.size() != 2 || rows[0].nr != rows[1].nr))
        throw std::runtime_error(std::string("bad interleave in ") + what);
    W.L = make_layout(ff, (int)N, (int)K, rpl, getenv("T4Q_CM") ? atoi(getenv("T4Q_CM")) : 0);
    CK(cudaSetDevice(g));
    CK(cudaMalloc(&W.base, W.L.bytes));
    CK(cudaMemsetAsync(W.base, 0, W.L.bytes, G.s));
    const size_t lrb = (size_t)(K / be) * bb;
    const int64_t per = std::max<int64_t>(1, (int64_t)(sg.cap / lrb));
    for (int64_t r0 = 0; r0 < N; r0 += per) {
        const int64_t n = std::min(per, N - r0);
        CK(cudaStreamSynchronize(G.s));
        for (int64_t i = 0; i < n; i++) gather_row(rows, cols, r0 + i, sg.pin + (size_t)i * lrb, il);
        CK(cudaMemcpyAsync(sg.dev[g], sg.pin, (size_t)n * lrb, cudaMemcpyHostToDevice, G.s));
        CK(repack_device_rows(W.L, sg.dev[g], W.base, (int)r0, (int)n, G.s));
    }
    CK(cudaStreamSynchronize(G.s));
    // keep the spec for the self-test
    tp::ShardSpec spec;
    spec.what = std::string(what) + "@gpu" + std::to_string(g);
    for (const Piece& p : rows) spec.rows.push_back({p.t, {p.r0, p.nr}});
    spec.cols = cols;
    spec.interleave = il;
    c->tps->specs.push_back({g, spec});
    c->tps->spec_fw.push_back(&W);
}

template <class T>
T* dmalloc(size_t n) {
    T* p;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

float* upload_f32_rows(t4q_ctx* c, int g, const GgufTensor* t, const std::vector<std::pair<int64_t, int64_t>>& rr) {
    if (t->type != GT_F32) throw std::runtime_error("expected F32: " + t->name);
    std::vector<float> buf;
    for (auto& r : rr) {
        const float* src = (const float*)(t->data + (size_t)r.first * t->row_bytes);
        buf.insert(buf.end(), src, src + (size_t)r.second * t->ne[0]);
    }
    CK(cudaSetDevice(g));
    float* p = dmalloc<float>(buf.size());
    CK(cudaMemcpy(p, buf.data(), buf.size() * 4, cudaMemcpyHostToDevice));
    return p;
}
float* upload_f32(t4q_ctx* c, int g, const std::string& n) {
    const GgufTensor* t = need(c, n);
    return upload_f32_rows(c, g, t, {{0, t->nrows()}});
}

void load_gpu_layer(t4q_ctx* c, Stage& sg, int g, int il) {
    tp::Layer& L = c->tps->G[g].L[il];
    const std::string p = "blk." + std::to_string(il) + ".";
    L.attn = hp::is_attn(il);
    L.attn_norm = upload_f32(c, g, p + "attn_norm.weight");
    L.post_norm = upload_f32(c, g, p + "post_attention_norm.weight");
    const bool st = (il == 0 || il == 3 || il == 8 || il == 63);  // keep self-test specs for a few layers only
    const size_t nspec = c->tps->specs.size();
    if (!L.attn) {
        const GgufTensor* qkv = need(c, p + "attn_qkv.weight");
        const GgufTensor* z = need(c, p + "attn_gate.weight");
        std::vector<Piece> rows = {{qkv, 1024 * g, 1024}, {qkv, 2048 + 1024 * g, 1024}};
        std::vector<std::pair<int64_t, int64_t>> vheads, vrows, zr;
        for (int r = 0; r < 3; r++) {
            rows.push_back({qkv, 4096 + (r * 16 + 8 * g) * 128, 1024});
            vheads.push_back({r * 16 + 8 * g, 8});
        }
        for (int r = 0; r < 3; r++) rows.push_back({z, (r * 16 + 8 * g) * 128, 1024});
        build_fw(c, sg, g, L.qkvz, rows, {{0, 5120}}, 4, (p + "qkvz").c_str());
        // alpha rows then beta rows for the 24 local heads
        {
            std::vector<float> buf;
            for (const char* nm : {"ssm_alpha.weight", "ssm_beta.weight"}) {
                const GgufTensor* t = need(c, p + nm);
                for (auto& vh : vheads) {
                    const float* src = (const float*)(t->data + (size_t)vh.first * t->row_bytes);
                    buf.insert(buf.end(), src, src + (size_t)vh.second * 5120);
                }
            }
            CK(cudaSetDevice(g));
            L.ab = dmalloc<float>(buf.size());
            CK(cudaMemcpy(L.ab, buf.data(), buf.size() * 4, cudaMemcpyHostToDevice));
        }
        // conv weights for the local channels in qkv-local order
        {
            std::vector<std::pair<int64_t, int64_t>> ch = {{1024 * g, 1024}, {2048 + 1024 * g, 1024}};
            for (int r = 0; r < 3; r++) ch.push_back({4096 + (r * 16 + 8 * g) * 128, 1024});
            L.conv_w = upload_f32_rows(c, g, need(c, p + "ssm_conv1d.weight"), ch);
        }
        {
            auto heads = [&](const std::string& n) {
                const GgufTensor* t = need(c, n);
                std::vector<float> buf;
                for (auto& vh : vheads)
                    for (int i = 0; i < vh.second; i++) buf.push_back(((const float*)t->data)[vh.first + i]);
                CK(cudaSetDevice(g));
                float* d = dmalloc<float>(buf.size());
                CK(cudaMemcpy(d, buf.data(), buf.size() * 4, cudaMemcpyHostToDevice));
                return d;
            };
            L.ssm_a = heads(p + "ssm_a");
            L.ssm_dt = heads(p + "ssm_dt.bias");
        }
        L.ssm_norm = upload_f32(c, g, p + "ssm_norm.weight");
        Cols oc;
        for (int r = 0; r < 3; r++) oc.push_back({(r * 16 + 8 * g) * 128, 1024});
        build_fw(c, sg, g, L.ssm_out, {{need(c, p + "ssm_out.weight"), 0, 5120}}, oc, 2, (p + "ssm_out").c_str());
        CK(cudaSetDevice(g));
        L.conv_ring = dmalloc<float>(4 * 5120);
        L.S = dmalloc<float>((size_t)24 * 128 * 128);
    } else {
        build_fw(c, sg, g, L.qkv_a,
                 {{need(c, p + "attn_q.weight"), 6144 * g, 6144},
                  {need(c, p + "attn_k.weight"), 512 * g, 512},
                  {need(c, p + "attn_v.weight"), 512 * g, 512}},
                 {{0, 5120}}, 4, (p + "qkv_a").c_str());
        build_fw(c, sg, g, L.wo, {{need(c, p + "attn_output.weight"), 0, 5120}}, {{3072 * g, 3072}}, 4,
                 (p + "attn_output").c_str());
        L.q_norm = upload_f32(c, g, p + "attn_q_norm.weight");
        L.k_norm = upload_f32(c, g, p + "attn_k_norm.weight");
        CK(cudaSetDevice(g));
        L.kc = dmalloc<uint16_t>((size_t)2 * c->tps->max_ctx * 256);
        L.vc = dmalloc<uint16_t>((size_t)2 * c->tps->max_ctx * 256);
    }
    // gate and up rows interleaved by 4 per tile: one GEMV block owns 32 outputs = one q8 group of silu(g) * u
    build_fw(c, sg, g, L.gateup,
             {{need(c, p + "ffn_gate.weight"), 8704 * g, 8704}, {need(c, p + "ffn_up.weight"), 8704 * g, 8704}},
             {{0, 5120}}, 4, (p + "gateup").c_str(), 4);
    build_fw(c, sg, g, L.down, {{need(c, p + "ffn_down.weight"), 0, 5120}}, {{8704 * g, 8704}}, 4,
             (p + "ffn_down").c_str());
    if (!st) {  // drop the specs of this layer
        c->tps->specs.resize(nspec);
        c->tps->spec_fw.resize(nspec);
    }
}

// GEMV self-test against the CPU ggml dequant (fp64 dot with the same q8 activations)
void selftest(t4q_ctx* c) {
    tp::State& S = *c->tps;
    std::string js = "[";
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    double worst = 0;
    for (size_t k = 0; k < S.specs.size(); k++) {
        const int g = S.specs[k].first;
        const tp::ShardSpec& sp = S.specs[k].second;
        const tp::FW& W = *S.spec_fw[k];
        tp::Gpu& G = S.G[g];
        const int K = W.L.K, N = W.L.N;
        std::vector<float> x(K);
        for (auto& v : x) v = nd(rng);
        std::vector<int8_t> xq(K);
        std::vector<int32_t> xm(K / 32 * 2);
        quantize_q8_host(x.data(), K, xq.data(), xm.data());
        CK(cudaSetDevice(g));
        CK(cudaMemcpy(G.xq, xq.data(), K, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(G.xm, xm.data(), xm.size() * 4, cudaMemcpyHostToDevice));
        tp::gemv(W, G.xq, G.xm, G.logits, G.s);
        CK(cudaGetLastError());
        std::vector<float> y(N);
        CK(cudaMemcpyAsync(y.data(), G.logits, (size_t)N * 4, cudaMemcpyDeviceToHost, G.s));
        CK(cudaStreamSynchronize(G.s));
        std::vector<Piece> rows;
        for (auto& r : sp.rows) rows.push_back({(const GgufTensor*)r.first, r.second.first, r.second.second});
        std::vector<uint8_t> raw(16 * K + 1024);
        std::vector<float> w(K);
        double maxe = 0, ss = 0;
        const int nchk = 24;
        for (int i = 0; i < nchk; i++) {
            const int64_t r = i == 0 ? 0 : i == 1 ? N - 1 : (int64_t)(rng() % N);
            gather_row(rows, sp.cols, r, raw.data(), sp.interleave);
            if (!dequant_row_cpu(rows[0].t->type, raw.data(), w.data(), K)) throw std::runtime_error("selftest dequant");
            double ref = 0;
            for (int e = 0; e < K; e++) {
                float xd;
                memcpy(&xd, &xm[(e / 32) * 2], 4);
                ref += (double)w[e] * ((double)xd * xq[e]);
            }
            maxe = std::max(maxe, std::fabs(ref - y[r]));
            ss += ref * ref;
        }
        const double rel = maxe / std::max(1e-30, std::sqrt(ss / nchk));
        worst = std::max(worst, rel);
        // bandwidth of this weight's GEMV (burst, back-to-back, no AR)
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        for (int it = 0; it < 3; it++) tp::gemv(W, G.xq, G.xm, G.logits, G.s);
        const int NIT = 20;
        CK(cudaEventRecord(e0, G.s));
        for (int it = 0; it < NIT; it++) tp::gemv(W, G.xq, G.xm, G.logits, G.s);
        CK(cudaEventRecord(e1, G.s));
        CK(cudaEventSynchronize(e1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        const double us = 1e3 * ms / NIT, gbs = W.L.bytes / (us * 1e3);
        // K-split weights: cost of the AR publish epilogue (remote float4 rows + fence + counter + flag)
        std::string arj;
        if (sp.what.find("ssm_out") != std::string::npos || sp.what.find("ffn_down") != std::string::npos ||
            sp.what.find("attn_output") != std::string::npos) {
            float* remote = S.p2p ? S.G[1 - g].scratch : S.hscratch[1 - g];
            const struct { const char* nm; float* dst; int fence; } V[4] = {
                {"remote_sys", remote, 2}, {"remote_gpu", remote, 1}, {"remote_rowsonly", remote, -1},
                {"staging_only", remote, -2}};
            for (auto& v : V) {
                tp::ArArgs a;
                a.y_peer = v.dst; a.cnt = G.cnt; a.peer_flag = (unsigned*)(v.dst + 5120); a.st = G.st; a.idx = 0;
                a.fence = v.fence;
                for (int it = 0; it < 3; it++) tp::gemv(W, G.xq, G.xm, G.logits, G.s, &a);
                CK(cudaEventRecord(e0, G.s));
                for (int it = 0; it < NIT; it++) tp::gemv(W, G.xq, G.xm, G.logits, G.s, &a);
                CK(cudaEventRecord(e1, G.s));
                CK(cudaEventSynchronize(e1));
                float ms2 = 0;
                CK(cudaEventElapsedTime(&ms2, e0, e1));
                char bb[96];
                snprintf(bb, sizeof bb, ",\"ar_%s_us\":%.1f", v.nm, 1e3 * ms2 / NIT);
                arj += bb;
            }
        }
        // L2 warm-up experiment (gateup only): GEMV time after flushing L2, after touching its first 2 MB with real
        // loads, and after prefetch.global.L2 of the same range
        if (sp.what.find("gateup") != std::string::npos && g == 0 && sp.what.find("blk.0.") == 0) {
            const size_t flush_n = 64ull << 20;
            const uint8_t* other = S.G[g].lm.base ? S.G[g].lm.base : W.base;  // unrelated 64 MB to evict L2
            double tt[3] = {0, 0, 0};
            for (int rep = 0; rep < 5; rep++)
                for (int mode = 0; mode < 3; mode++) {
                    tp::touch(other, flush_n, 0, G.logits + 124000, G.s);
                    if (mode == 1) tp::touch(W.base, 2u << 20, 0, G.logits + 124000, G.s);
                    if (mode == 2) tp::touch(W.base, 2u << 20, 1, G.logits + 124000, G.s);
                    CK(cudaEventRecord(e0, G.s));
                    tp::gemv(W, G.xq, G.xm, G.logits, G.s);
                    CK(cudaEventRecord(e1, G.s));
                    CK(cudaEventSynchronize(e1));
                    float m3 = 0;
                    CK(cudaEventElapsedTime(&m3, e0, e1));
                    tt[mode] += 1e3 * m3 / 5;
                }
            char bb[200];
            snprintf(bb, sizeof bb, ",\"l2exp_cold_us\":%.1f,\"l2exp_ld2MB_us\":%.1f,\"l2exp_pf2MB_us\":%.1f", tt[0], tt[1],
                     tt[2]);
            arj += bb;
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        char b[900];
        snprintf(b, sizeof b,
                 "%s{\"w\":\"%s\",\"fmt\":\"%s\",\"N\":%d,\"K\":%d,\"max_err_over_rms\":%.3e,\"us\":%.1f,"
                 "\"GBps\":%.1f%s}",
                 k ? "," : "", sp.what.c_str(), fmt_name(W.L.fmt), N, K, rel, us, gbs, arj.c_str());
        js += b;
    }
    char b[128];
    snprintf(b, sizeof b, "], \"worst\": %.3e, \"pass\": %s}", worst, worst < 1e-4 ? "true" : "false");
    S.selftest_json = "{\"tests\": " + js + b;
    if (c->params.verbose) fprintf(stderr, "[t4q-tp] selftest %s\n", S.selftest_json.c_str());
}

// ------------------------------------------------------------------------------------------------ step enqueue
struct Enq {
    t4q_ctx* c;
    bool prof = false;
    void mark(tp::Gpu& G, const char* name) {
        if (!prof) return;
        cudaEvent_t e;
        CK(cudaEventCreate(&e));
        CK(cudaEventRecord(e, G.s));
        G.ev.push_back(e);
        G.ev_name.push_back(name);
    }
    bool arpub() const { return c->tps->arpub && (c->tps->fuse == 0 || c->tps->fuse == 3); }
    tp::ProArgs lead;  // storage for the leader-prologue args of the current GEMV
    bool rows_in_gemv() const { return arpub() && c->tps->arpub == 2; }  // arpub 2: GEMV writes rows, consumer flags
    // AR args for a K-split GEMV: nullptr (arpub 1: plain GEMV), rows-only (arpub 2) or full epilogue publish
    bool ll() const { return c->tps->ll && c->tps->p2p && c->tps->fuse == 0 && !mega(); }
    const tp::ArArgs* ksplit(tp::Gpu& G, int idx, tp::ArArgs& a) {
        a = ar(G, idx);
        if (ll()) {
            a.y_peer = (float*)(G.peer_rxl + (idx & 1) * D);
            a.fence = -3;
            return &a;
        }
        if (!arpub()) return &a;
        if (!rows_in_gemv()) return nullptr;
        a.fence = -1;
        return &a;
    }
    tp::ArArgs ar(tp::Gpu& G, int idx) {
        tp::ArArgs a;
        a.y_peer = G.peer_rx + (idx & 1) * D;
        a.cnt = G.cnt;
        a.peer_flag = G.peer_flag + (idx & 1);
        a.st = G.st;
        a.idx = idx;
        return a;
    }
    void start(int g) {
        tp::Gpu& G = c->tps->G[g];
        mark(G, "start");
        const tp::Pf pf = pf_for(G.L[0].attn ? G.L[0].qkv_a : G.L[0].qkvz);
        tp::embed(G.embd, G.st, G.prompt, G.hb[0], G.s, &pf);
        mark(G, "embed");
    }
    // prologue consuming AR idx (idx < 0: layer 0, no all-reduce): residual in hb[idx & 1] -> hb[(idx + 1) & 1]
    tp::ProArgs arnorm(tp::Gpu& G, int idx, const float* nw) {
        tp::ProArgs p;
        if (idx < 0) {
            p.h_in = G.hb[0];
        } else {
            p.add = 1;
            p.h_in = G.hb[idx & 1];
            p.h_out = G.hb[(idx + 1) & 1];
            p.own = G.part + (idx & 1) * D;
            p.rx = G.rx + (idx & 1) * D;
            if (c->tps->p2p) {
                p.flag = G.flag + (idx & 1);
            } else {  // fallback: a pull kernel (publishes first if arpub) waits on the host flag, copies into rx
                tp::pull(G.hflag + (idx & 1), G.hrx + (idx & 1) * D, G.rx + (idx & 1) * D, G.st, idx, G.s, G.part,
                         arpub() && !rows_in_gemv() ? G.peer_rx : nullptr, arpub() ? G.peer_flag : nullptr);
                mark(G, "pull");
            }
        }
        p.st = G.st;
        p.idx = idx;
        p.nw = nw;
        p.xn_out = G.xn;
        return p;
    }
    // L2 prefetch spec for the first pf_kb of W (each plane gets the same fraction, i.e. the first tiles)
    tp::Pf pf_for(const tp::FW& W) {
        tp::Pf f;
        const int kb = c->tps->pf_kb;
        if (kb <= 0) return f;
        const double frac = std::min(1.0, (double)kb * 1024.0 / (double)W.L.bytes);
        const size_t start[4] = {0, W.L.off_qh, W.L.off_sc, W.L.off_d};
        const size_t end[4] = {W.L.off_qh, W.L.off_sc, W.L.off_d, W.L.bytes};
        for (int r = 0; r < 4; r++) {
            const size_t sz = end[r] - start[r];
            if (!sz) continue;
            f.p[r] = W.base + start[r];
            f.n[r] = (unsigned)std::min(sz, ((size_t)(sz * frac) + 127) & ~(size_t)127);
        }
        f.blocks = 32;
        return f;
    }
    // unfused path: the ARNORM prologue as its own kernel (q8 x to G.xq/G.xm, fp32 to G.xn); returns nullptr so the
    // GEMV reads x from global memory. Its extra blocks prefetch the next GEMV (nxt) into L2.
    const tp::ProArgs* pre(tp::Gpu& G, const tp::ProArgs& p, const tp::FW& nxt) {
        if (c->tps->fuse == 3) {  // leader block: [publish,] wait, residual, norm, q8 -> G.xq/G.xm, then xflag
            lead = p;
            lead.xflag = G.xflag;
            lead.gxq = G.xq;
            lead.gxm = G.xm;
            lead.xn_out = G.xn;
            if (arpub() && p.add && c->tps->p2p) {
                const int sl = p.idx & 1;
                lead.pub_peer_flag = G.peer_flag + sl;
                lead.pub_peer_rx = rows_in_gemv() ? nullptr : G.peer_rx + sl * D;
            }
            return &lead;
        }
        if (c->tps->fuse) return &p;  // fuse 2: redundant AR + norm prologue in every block
        if (ll() && p.add) {
            tp::ar_norm_ll(p.h_in, p.h_out, G.part, G.rxl, G.st, p.idx, p.nw, G.xn, G.xq, G.xm, G.s);
            mark(G, "ar_norm");
            return nullptr;
        }
        // in fallback mode arnorm() already enqueued the pull and cleared p.flag
        const tp::Pf pf = pf_for(nxt);
        const bool pub = arpub() && p.add && c->tps->p2p;  // fallback mode published in pull()
        tp::ar_norm(p.h_in, p.h_out, p.add ? G.part : nullptr, G.rx, p.flag ? G.flag : nullptr, G.st, p.idx, p.nw,
                    G.xn, G.xq, G.xm, G.s, &pf, pub && !rows_in_gemv() ? G.peer_rx : nullptr,
                    pub ? G.peer_flag : nullptr);
        mark(G, "ar_norm");
        return nullptr;
    }
    // ---- persistent per-layer kernels (mega): rows go to the peer from the K-split GEMVs (arpub 2 semantics) and the
    // next kernel's leader block publishes the flag
    bool mega() const { return c->tps->mega && !c->dump_on; }
    tp::MegaCommon mcommon(tp::Gpu& G, int idx, const float* nw, int xid) {
        tp::MegaCommon m;
        m.st = G.st;
        m.bar = G.mbar;
        m.xrdy = G.mbar + 2;
        m.xid = xid;
        m.in = arnorm(G, idx, nw);  // enqueues the pull kernel in fallback mode
        if (m.in.add && c->tps->p2p) {
            m.in.pub_peer_flag = G.peer_flag + (idx & 1);
            m.in.pub_peer_rx = nullptr;
        }
        m.gxq = G.xq;
        m.gxm = G.xm;
        m.xn = G.xn;
        return m;
    }
    static t4q::gemv::GemvArgs gargs(const tp::FW& W, float* y) {
        return t4q::gemv::make_args(W.L, W.base, nullptr, nullptr, y, W.L.N);
    }
    void layer_mega(int g, int il, int part) {
        tp::State& S = *c->tps;
        tp::Gpu& G = S.G[g];
        tp::Layer& L = G.L[il];
        const int grid = S.mega_grid;
        if (part == 0) {
            const int idx = 2 * il;
            if (!L.attn) {
                tp::MegaDN m;
                m.c = mcommon(G, idx - 1, L.attn_norm, il * 8);
                m.qkvz = gargs(L.qkvz, G.y);
                m.ab.w = L.ab; m.ab.x = nullptr; m.ab.y = G.yab; m.ab.nrows = 48;
                m.ssm_out = gargs(L.ssm_out, G.part + (idx & 1) * D);
                m.ring_w = L.conv_w; m.ssm_a = L.ssm_a; m.ssm_dt = L.ssm_dt; m.ssm_norm = L.ssm_norm;
                m.ring = L.conv_ring; m.S = L.S; m.y = G.y; m.yab = G.yab; m.o = G.o;
                m.y_peer = G.peer_rx + (idx & 1) * D;
                tp::mega_dn(m, grid, G.s);
                mark(G, "mega_dn");
            } else {
                tp::MegaAttn m;
                m.c = mcommon(G, idx - 1, L.attn_norm, il * 8);
                m.qkv = gargs(L.qkv_a, G.y);
                m.wo = gargs(L.wo, G.part + (idx & 1) * D);
                m.qw = L.q_norm; m.kw = L.k_norm; m.ya = G.y; m.qa = G.qa; m.ws = G.attn_ws;
                m.kc = L.kc; m.vc = L.vc; m.max_ctx = S.max_ctx;
                m.theta_scale = powf(hp::ROPE_BASE, -2.0f / hp::NROT);
                m.y_peer = G.peer_rx + (idx & 1) * D;
                tp::mega_attn(m, grid, G.s);
                mark(G, "mega_attn");
            }
        } else {
            const int idx = 2 * il + 1;
            tp::MegaFFN m;
            m.c = mcommon(G, idx - 1, L.post_norm, il * 8 + 2);
            m.gateup = gargs(L.gateup, G.y);
            m.down = gargs(L.down, G.part + (idx & 1) * D);
            m.xq2 = G.xq2; m.xm2 = G.xm2;
            m.y_peer = G.peer_rx + (idx & 1) * D;
            tp::mega_ffn(m, L.down.L.fmt, grid, G.s);
            mark(G, "mega_ffn");
        }
    }
    void head_mega(int g) {
        tp::State& S = *c->tps;
        tp::Gpu& G = S.G[g];
        tp::MegaHead m;
        m.c = mcommon(G, 127, G.output_norm, 512);
        m.lm = gargs(G.lm, G.logits);
        m.logits = G.logits; m.apart = G.apart;
        m.amb = S.p2p ? G.amb : G.hamb;
        m.aflag = S.p2p ? G.flag + 2 : G.hflag + 2;
        m.peer_amb = G.peer_amb; m.peer_aflag = G.peer_flag + 2;
        m.row0 = 124160 * g;
        m.ring = g == 0 ? S.d_ring : nullptr;
        tp::mega_head(m, S.mega_grid, G.s);
        mark(G, "mega_head");
    }
    // part 0: [AR + attn_norm] mixer, publishes AR 2il; part 1: [AR + post_norm] FFN, publishes AR 2il+1
    void layer(int g, int il, int part) {
        if (mega()) return layer_mega(g, il, part);
        tp::State& S = *c->tps;
        tp::Gpu& G = S.G[g];
        tp::Layer& L = G.L[il];
        cudaStream_t s = G.s;
        if (part == 0) {
            const int idx = 2 * il;
            const tp::ProArgs pa = arnorm(G, 2 * il - 1, L.attn_norm);
            if (!L.attn) {
                tp::SegArgs sg;
                sg.w = L.ab; sg.x = G.xn; sg.y = G.yab; sg.nrows = 48;
                tp::gemv(L.qkvz, G.xq, G.xm, G.y, s, nullptr, &sg, pre(G, pa, L.qkvz));
                mark(G, "gemv_qkvz");
                if (S.gdnf) {
                    tp::gdn_gn(G.y, G.yab, L.conv_ring, L.conv_w, L.ssm_a, L.ssm_dt, L.S, G.o, G.st, G.gcnt, G.y + 5120,
                               L.ssm_norm, G.xq, G.xm, s);
                    mark(G, "gdn+gnorm");
                } else {
                    tp::gdn(G.y, G.yab, L.conv_ring, L.conv_w, L.ssm_a, L.ssm_dt, L.S, G.o, G.st, s);
                    mark(G, "gdn");
                }
                tp::ProArgs pg;
                pg.o = G.o; pg.z = G.y + 5120; pg.gw = L.ssm_norm;
                if (S.fuse != 1 && !S.gdnf) {
                    const tp::Pf pf = pf_for(L.ssm_out);
                    tp::gnorm_q8(G.o, G.y + 5120, L.ssm_norm, G.xq, G.xm, s, &pf);
                    mark(G, "gnorm_q8");
                }
                tp::ArArgs a;
                tp::gemv(L.ssm_out, G.xq, G.xm, G.part + (idx & 1) * D, s, ksplit(G, idx, a), nullptr,
                         S.fuse == 1 ? &pg : nullptr);
                mark(G, "gemv_ssm_out");
            } else {
                tp::gemv(L.qkv_a, G.xq, G.xm, G.y, s, nullptr, nullptr, pre(G, pa, L.qkv_a));
                mark(G, "gemv_attn_qkv");
                tp::attn_prep(G.y, L.q_norm, L.k_norm, G.qa, L.kc, L.vc, S.max_ctx, G.st,
                              powf(hp::ROPE_BASE, -2.0f / hp::NROT), s);
                mark(G, "attn_prep");
                tp::attn_split(G.qa, L.kc, L.vc, G.attn_ws, S.max_ctx, G.st, s);
                mark(G, "attn_split");
                const tp::Pf pf = pf_for(L.wo);
                tp::attn_combine_q8(G.attn_ws, G.y, G.st, G.xq, G.xm, s, &pf);
                mark(G, "attn_combine");
                tp::ArArgs a;
                tp::gemv(L.wo, G.xq, G.xm, G.part + (idx & 1) * D, s, ksplit(G, idx, a));
                mark(G, "gemv_attn_out");
            }
        } else {
            const int idx = 2 * il + 1;
            const tp::ProArgs pa = arnorm(G, 2 * il, L.post_norm);
            // gate|up GEMV with the silu * up + q8 epilogue (x for ffn_down lands in G.xq2 / G.xm2)
            const tp::ProArgs* pp = pre(G, pa, L.gateup);
            tp::ProArgs pq = pp ? *pp : tp::ProArgs{};
            pq.sq_xq = G.xq2;
            pq.sq_xm = G.xm2;
            tp::gemv(L.gateup, G.xq, G.xm, G.y, s, nullptr, nullptr, &pq);
            mark(G, "gemv_gateup+silu_q8");
            tp::ArArgs a;
            tp::gemv(L.down, G.xq2, G.xm2, G.part + (idx & 1) * D, s, ksplit(G, idx, a));
            mark(G, "gemv_down");
        }
    }
    void head(int g) {
        if (mega()) return head_mega(g);
        tp::State& S = *c->tps;
        tp::Gpu& G = S.G[g];
        const tp::ProArgs pa = arnorm(G, 127, G.output_norm);
        tp::gemv(G.lm, G.xq, G.xm, G.logits, G.s, nullptr, nullptr, pre(G, pa, G.lm));
        mark(G, "gemv_lm_head");
        tp::argmax_step(G.logits, 124160, 124160 * g, G.apart, S.p2p ? G.amb : G.hamb,
                        S.p2p ? G.flag + 2 : G.hflag + 2, G.peer_amb, G.peer_flag + 2, G.st,
                        g == 0 ? S.d_ring : nullptr, G.s);
        mark(G, "argmax");
    }
    void gpu_step(int g) {  // whole step on one GPU (graph capture)
        start(g);
        for (int il = 0; il < 64; il++)
            for (int p = 0; p < 2; p++) layer(g, il, p);
        head(g);
    }
};

void check_launch(const char* w) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("launch failed: ") + w + ": " + cudaGetErrorString(e));
}

void sync_both(t4q_ctx* c) {
    for (int g = 0; g < 2; g++) {
        CK(cudaSetDevice(g));
        CK(cudaStreamSynchronize(c->tps->G[g].s));
    }
}

void check_err(t4q_ctx* c) {
    for (int g = 0; g < 2; g++) {
        tp::StepState st;
        CK(cudaSetDevice(g));
        CK(cudaMemcpy(&st, c->tps->G[g].st, sizeof st, cudaMemcpyDeviceToHost));
        if (st.err) throw std::runtime_error("device error code " + std::to_string(st.err) + " on gpu " + std::to_string(g));
    }
}

void dumpv(t4q_ctx* c, const std::string& key, const float* d, size_t n) {
    std::vector<float>& v = c->dumps[key];
    v.resize(n);
    CK(cudaSetDevice(0));
    CK(cudaMemcpy(v.data(), d, n * 4, cudaMemcpyDeviceToHost));
}

// one eager step, interleaving the two GPUs per layer part; optional dumps and profiling
void eager_step(t4q_ctx* c, bool prof) {
    tp::State& S = *c->tps;
    Enq q{c, prof};
    const bool dmp = c->dump_on;
    if (dmp) c->dumps.clear();
    for (int g = 0; g < 2; g++) { CK(cudaSetDevice(g)); q.start(g); }
    for (int il = 0; il < 64; il++)
        for (int p = 0; p < 2; p++) {
            for (int g = 0; g < 2; g++) { CK(cudaSetDevice(g)); q.layer(g, il, p); }
            check_launch("layer");
            if (dmp) {
                sync_both(c);
                tp::Gpu& G = S.G[0];
                if (p == 0) {  // first kernel consumed AR 2il-1: residual in hb[0], xn = attn_norm
                    if (il > 0) dumpv(c, "l_out-" + std::to_string(il - 1), G.hb[0], D);
                    dumpv(c, "attn_norm-" + std::to_string(il), G.xn, D);
                } else {       // consumed AR 2il: residual in hb[1], xn = post norm
                    dumpv(c, "attn_residual-" + std::to_string(il), G.hb[1], D);
                    dumpv(c, "attn_post_norm-" + std::to_string(il), G.xn, D);
                }
            }
        }
    for (int g = 0; g < 2; g++) { CK(cudaSetDevice(g)); q.head(g); }
    check_launch("head");
    if (dmp || prof) sync_both(c);
    if (dmp) {
        dumpv(c, "l_out-63", S.G[0].hb[0], D);
        dumpv(c, "result_norm", S.G[0].xn, D);
        // both GPUs must hold the same residual bits
        std::vector<float> h1(D);
        CK(cudaSetDevice(1));
        CK(cudaMemcpy(h1.data(), S.G[1].hb[0], D * 4, cudaMemcpyDeviceToHost));
        c->dumps["tp_h_mismatch"] = {(float)(memcmp(h1.data(), c->dumps["l_out-63"].data(), D * 4) != 0)};
    }
}

void capture_graphs(t4q_ctx* c) {
    tp::State& S = *c->tps;
    auto t0 = Clock::now();
    for (int g = 0; g < 2; g++) {
        tp::Gpu& G = S.G[g];
        CK(cudaSetDevice(g));
        if (G.gexec) { cudaGraphExecDestroy(G.gexec); G.gexec = nullptr; }
        if (G.graph) { cudaGraphDestroy(G.graph); G.graph = nullptr; }
        CK(cudaStreamBeginCapture(G.s, cudaStreamCaptureModeThreadLocal));
        Enq q{c, false};
        q.gpu_step(g);
        CK(cudaStreamEndCapture(G.s, &G.graph));
        CK(cudaGraphInstantiate(&G.gexec, G.graph, 0));
    }
    S.ms_graph_capture = 1e3 * secs(t0);
}

void run_step(t4q_ctx* c) {
    tp::State& S = *c->tps;
    if (S.graphs) {
        if (!S.G[0].gexec) capture_graphs(c);
        for (int g = 0; g < 2; g++) {
            CK(cudaSetDevice(g));
            CK(cudaGraphLaunch(S.G[g].gexec, S.G[g].s));
        }
    } else {
        eager_step(c, false);
    }
    c->pos++;
    c->steps++;
}

void set_prompt(t4q_ctx* c, const int32_t* ids, int n) {
    tp::State& S = *c->tps;
    if (c->pos + n > S.max_ctx) throw std::runtime_error("context full");
    for (int g = 0; g < 2; g++) {
        tp::Gpu& G = S.G[g];
        CK(cudaSetDevice(g));
        CK(cudaMemcpyAsync(G.prompt + c->pos, ids, (size_t)n * 4, cudaMemcpyHostToDevice, G.s));
        const int np = c->pos + n;
        CK(cudaMemcpyAsync(&G.st->n_prompt, &np, 4, cudaMemcpyHostToDevice, G.s));
        CK(cudaStreamSynchronize(G.s));  // np is a stack value
    }
}

// CUPTI timeline of n steps: per-position kernel durations / launch gaps (file T4Q_TRACE_OUT) and a per-kernel-name
// summary (stats "trace"). Positions repeat every step, so kernel k of every step is the same graph node.
void trace_steps(t4q_ctx* c, int n) {
    tp::State& S = *c->tps;
    if (c->pos + n > S.max_ctx) throw std::runtime_error("trace: context full");
    sync_both(c);
    if (S.graphs && !S.G[0].gexec) {  // capture outside the trace window
        capture_graphs(c);
    }
    std::string err;
    if (!trace::begin(err)) {
        S.trace_json = "{\"error\": \"" + err + "\"}";
        return;
    }
    for (int i = 0; i < n; i++) run_step(c);
    sync_both(c);
    std::vector<trace::Rec> R = trace::end();
    check_err(c);
    std::vector<trace::Rec> dv[2];
    for (auto& r : R)
        if (r.dev >= 0 && r.dev < 2) dv[r.dev].push_back(r);
    std::string js = "{\"steps\": " + std::to_string(n) + ", \"mode\": \"" + (S.graphs ? "graphs" : "eager") + "\"";
    std::string fj = "{\"steps\": " + std::to_string(n);
    const int skip = 1;  // first step may include warm-up effects
    std::vector<std::string> names;
    int per = 0;
    for (int g = 0; g < 2; g++) {
        auto& v = dv[g];
        if (v.empty() || v.size() % n) {
            js += ", \"gpu" + std::to_string(g) + "\": {\"error\": \"" + std::to_string(v.size()) + " records for " +
                  std::to_string(n) + " steps\"}";
            continue;
        }
        per = (int)(v.size() / n);
        if (names.empty())
            for (int k = 0; k < per; k++) names.push_back(v[k].name);
        std::vector<double> dur(per, 0), gap(per, 0), rel(per, 0);
        double step_us = 0, busy = 0, gaps = 0;
        std::map<std::string, std::pair<double, int>> agg, agg_gap;
        for (int s = skip; s < n; s++) {
            const uint64_t s0 = v[(size_t)s * per].start;
            if (s + 1 < n) step_us += (v[(size_t)(s + 1) * per].start - s0) * 1e-3;
            for (int k = 0; k < per; k++) {
                const auto& r = v[(size_t)s * per + k];
                const double d = (r.end - r.start) * 1e-3;
                const double gp = ((int64_t)r.start - (int64_t)v[(size_t)s * per + k - 1].end) * 1e-3;
                dur[k] += d;
                gap[k] += gp;
                rel[k] += (r.start - s0) * 1e-3;
                busy += d;
                gaps += gp;
                agg[r.name].first += d;
                agg[r.name].second += 1;
                agg_gap[r.name].first += gp;
            }
        }
        const int m = n - skip;
        char b[256];
        snprintf(b, sizeof b, ", \"gpu%d\": {\"kernels_per_step\": %d, \"step_us\": %.1f, \"busy_us\": %.1f, \"gap_us\": %.1f",
                 g, per, step_us / std::max(1, m - 1), busy / m, gaps / m);
        js += b;
        js += ", \"by_name\": {";
        bool first = true;
        for (auto& kv : agg) {
            snprintf(b, sizeof b, "%s\"%s\": [%.1f, %d, %.1f]", first ? "" : ", ", kv.first.c_str(), kv.second.first / m,
                     kv.second.second / m, agg_gap[kv.first].first / m);
            js += b;
            first = false;
        }
        js += "}}";
        fj += ", \"gpu" + std::to_string(g) + "\": {\"dur\": [";
        for (int k = 0; k < per; k++) { snprintf(b, sizeof b, "%s%.2f", k ? "," : "", dur[k] / m); fj += b; }
        fj += "], \"gap\": [";
        for (int k = 0; k < per; k++) { snprintf(b, sizeof b, "%s%.2f", k ? "," : "", gap[k] / m); fj += b; }
        fj += "], \"rel\": [";
        for (int k = 0; k < per; k++) { snprintf(b, sizeof b, "%s%.2f", k ? "," : "", rel[k] / m); fj += b; }
        fj += "]}";
    }
    // cross-GPU offset of step starts (common CUPTI timebase)
    if (!dv[0].empty() && dv[0].size() == dv[1].size() && per) {
        double off = 0;
        for (int s = skip; s < n; s++) off += ((int64_t)dv[1][(size_t)s * per].start - (int64_t)dv[0][(size_t)s * per].start) * 1e-3;
        char b[96];
        snprintf(b, sizeof b, ", \"gpu1_minus_gpu0_step_start_us\": %.2f", off / (n - skip));
        js += b;
        fj += b;
    }
    fj += ", \"names\": [";
    for (size_t k = 0; k < names.size(); k++) fj += (k ? ",\"" : "\"") + names[k] + "\"";
    fj += "]}";
    js += "}";
    S.trace_json = js;
    if (const char* p = getenv("T4Q_TRACE_OUT")) {
        if (FILE* f = fopen(p, "w")) {
            fputs(fj.c_str(), f);
            fclose(f);
        }
    }
}

}  // namespace

// ================================================================================================ API
void tp_load(t4q_ctx* c, const char* path) {
    auto t0 = Clock::now();
    std::string err;
    if (!c->f.open(path, err)) throw std::runtime_error("gguf: " + err);
    if (c->f.num("qwen35.embedding_length", -1) != 5120 || c->f.num("qwen35.block_count", -1) != 65)
        throw std::runtime_error("unexpected model hparams");
    c->tps = new tp::State();
    tp::State& S = *c->tps;
    S.max_ctx = c->max_ctx = c->params.max_ctx > 0 ? c->params.max_ctx : 4096;
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (ndev < 2) throw std::runtime_error("need 2 GPUs");
    int a01 = 0, a10 = 0;
    CK(cudaDeviceCanAccessPeer(&a01, 0, 1));
    CK(cudaDeviceCanAccessPeer(&a10, 1, 0));
    S.p2p = a01 && a10 && getenv("T4Q_NO_P2P") == nullptr;
    if (!S.p2p && c->params.verbose) fprintf(stderr, "[t4q-tp] no P2P: host-mapped mailbox fallback\n");
    for (int g = 0; g < 2; g++) {
        tp::Gpu& G = S.G[g];
        G.g = g;
        CK(cudaSetDevice(g));
        if (S.p2p) {
            cudaError_t e = cudaDeviceEnablePeerAccess(1 - g, 0);
            if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CK(e);
            cudaGetLastError();
        }
        CK(cudaStreamCreateWithFlags(&G.s, cudaStreamNonBlocking));
        G.hb[0] = dmalloc<float>(D); G.hb[1] = dmalloc<float>(D); G.xn = dmalloc<float>(D); G.y = dmalloc<float>(17408); G.yab = dmalloc<float>(64);
        G.o = dmalloc<float>(3072); G.qa = dmalloc<float>(12 * 256);
        G.attn_ws = dmalloc<float>((size_t)2 * tp::NSPLIT * 6 * 258);
        G.logits = dmalloc<float>(124160);
        G.xq = dmalloc<int8_t>(17408); G.xm = dmalloc<int2>(17408 / 32);
        G.xq2 = dmalloc<int8_t>(8704); G.xm2 = dmalloc<int2>(8704 / 32);
        G.part = dmalloc<float>(2 * D); G.rx = dmalloc<float>(2 * D);
        G.flag = dmalloc<unsigned>(8); G.cnt = dmalloc<unsigned>(8); G.xflag = dmalloc<unsigned>(8);
        G.mbar = dmalloc<unsigned>(8);
        G.amb = dmalloc<float>(8); G.apart = dmalloc<float>(2 * 160);
        G.st = dmalloc<tp::StepState>(1);
        G.scratch = dmalloc<float>(5120 + 64);
        G.rxl = dmalloc<float2>(2 * D);
        G.gcnt = dmalloc<unsigned>(32);
        G.prompt = dmalloc<int>(S.max_ctx);
    }
    for (int g = 0; g < 2; g++) {
        tp::Gpu& G = S.G[g];
        CK(cudaHostAlloc(&G.hrx, 2 * D * 4, cudaHostAllocMapped | cudaHostAllocPortable));
        CK(cudaHostAlloc(&G.hflag, 8 * 4, cudaHostAllocMapped | cudaHostAllocPortable));
        CK(cudaHostAlloc(&G.hamb, 4 * 4, cudaHostAllocMapped | cudaHostAllocPortable));
        memset(G.hrx, 0, 2 * D * 4);
        memset(G.hflag, 0, 8 * 4);
        memset(G.hamb, 0, 4 * 4);
        CK(cudaHostAlloc(&S.hscratch[g], (5120 + 64) * 4, cudaHostAllocMapped | cudaHostAllocPortable));
    }
    for (int g = 0; g < 2; g++) {
        tp::Gpu& G = S.G[g];
        tp::Gpu& P = S.G[1 - g];
        G.peer_rx = S.p2p ? P.rx : P.hrx;
        G.peer_flag = S.p2p ? P.flag : P.hflag;
        G.peer_amb = S.p2p ? P.amb : P.hamb;
        G.peer_rxl = S.p2p ? P.rxl : nullptr;
    }
    {
        int nsm = 40;
        CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
        S.max_blocks = 2 * nsm;
        tp::set_max_blocks(S.max_blocks);
        if (getenv("T4Q_THREADS")) tp::set_threads(atoi(getenv("T4Q_THREADS")));  // A/B knob
    }
    CK(cudaSetDevice(0));
    CK(cudaHostAlloc(&S.h_ring, tp::RING * 4, cudaHostAllocMapped | cudaHostAllocPortable));
    memset(S.h_ring, 0xff, tp::RING * 4);
    CK(cudaHostGetDevicePointer((void**)&S.d_ring, S.h_ring, 0));

    Stage sg;
    sg.cap = 64ull << 20;
    CK(cudaMallocHost(&sg.pin, sg.cap));
    for (int g = 0; g < 2; g++) { CK(cudaSetDevice(g)); CK(cudaMalloc(&sg.dev[g], sg.cap)); }
    // embeddings: raw Q4_0 rows on both GPUs
    const GgufTensor* te = need(c, "token_embd.weight");
    if (te->type != GT_Q4_0 || te->row_bytes != 2880) throw std::runtime_error("TP engine expects Q4_0 token_embd");
    for (int g = 0; g < 2; g++) {
        CK(cudaSetDevice(g));
        CK(cudaMalloc(&S.G[g].embd, te->nbytes));
        for (size_t o = 0; o < te->nbytes; o += sg.cap) {
            const size_t n = std::min(sg.cap, te->nbytes - o);
            memcpy(sg.pin, te->data + o, n);
            CK(cudaMemcpy(S.G[g].embd + o, sg.pin, n, cudaMemcpyHostToDevice));
        }
    }
    for (int il = 0; il < 64; il++) {
        for (int g = 0; g < 2; g++) load_gpu_layer(c, sg, g, il);
        if (c->params.verbose && il % 16 == 15) fprintf(stderr, "[t4q-tp] layers 0..%d loaded (%.1f s)\n", il, secs(t0));
    }
    for (int g = 0; g < 2; g++) {
        S.G[g].output_norm = upload_f32(c, g, "output_norm.weight");
        build_fw(c, sg, g, S.G[g].lm, {{need(c, "output.weight"), 124160 * g, 124160}}, {{0, 5120}}, 2, "lm_head");
    }
    for (int g = 0; g < 2; g++) { CK(cudaSetDevice(g)); CK(cudaFree(sg.dev[g])); }
    CK(cudaFreeHost(sg.pin));
    selftest(c);
    for (int g = 0; g < 2; g++) {
        CK(cudaSetDevice(g));
        size_t fr, tot;
        CK(cudaMemGetInfo(&fr, &tot));
        c->vram_used[g] = tot - fr;
    }
    CK(cudaMallocHost(&c->h_logits, (size_t)hp::V * 4));
    c->load_s = secs(t0);
    if (c->params.verbose)
        fprintf(stderr, "[t4q-tp] load %.1f s, vram %.0f / %.0f MiB\n", c->load_s, c->vram_used[0] / 1048576.0,
                c->vram_used[1] / 1048576.0);
}

void tp_reset(t4q_ctx* c) {
    tp::State& S = *c->tps;
    sync_both(c);
    for (int g = 0; g < 2; g++) {
        tp::Gpu& G = S.G[g];
        CK(cudaSetDevice(g));
        for (int il = 0; il < 64; il++)
            if (!G.L[il].attn) {
                CK(cudaMemset(G.L[il].conv_ring, 0, 4 * 5120 * 4));
                CK(cudaMemset(G.L[il].S, 0, (size_t)24 * 128 * 128 * 4));
            }
        tp::StepState st;
        CK(cudaMemcpy(&st, G.st, sizeof st, cudaMemcpyDeviceToHost));
        st.pos = 0; st.n_prompt = 0; st.token = 0; st.err = 0;
        CK(cudaMemcpy(G.st, &st, sizeof st, cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());
    }
    c->pos = 0;
    c->have_logits = false;
}

int tp_logits(t4q_ctx* c, const int32_t* ids, int n, float* out) {
    tp::State& S = *c->tps;
    set_prompt(c, ids, n);
    for (int i = 0; i < n; i++) {
        auto t0 = Clock::now();
        if (c->dump_on) {  // dumps need the eager path
            eager_step(c, false);
            c->pos++;
            c->steps++;
        } else {
            run_step(c);
        }
        if (out || i == n - 1) {
            for (int g = 0; g < 2; g++) {
                CK(cudaSetDevice(g));
                CK(cudaMemcpyAsync(c->h_logits + (size_t)124160 * g, S.G[g].logits, 124160 * 4, cudaMemcpyDeviceToHost,
                                   S.G[g].s));
            }
            sync_both(c);
            if (out) memcpy(out + (size_t)i * hp::V, c->h_logits, (size_t)hp::V * 4);
        }
        c->step_s += secs(t0);
    }
    sync_both(c);
    check_err(c);
    c->have_logits = true;
    return 0;
}

int tp_prefill(t4q_ctx* c, const int32_t* ids, int n) {
    set_prompt(c, ids, n);
    auto t0 = Clock::now();
    for (int i = 0; i < n; i++) run_step(c);
    sync_both(c);
    c->step_s += secs(t0);
    check_err(c);
    c->have_logits = true;
    return 0;
}

int tp_generate(t4q_ctx* c, int32_t* out, int max_new, const int32_t* stop, int n_stop) {
    tp::State& S = *c->tps;
    if (!c->have_logits) throw std::runtime_error("generate needs a prefill first");
    auto t0 = Clock::now();
    tp::StepState st;
    CK(cudaSetDevice(0));
    CK(cudaMemcpy(&st, S.G[0].st, sizeof st, cudaMemcpyDeviceToHost));
    uint32_t step0 = st.step;  // ring[step0 - 1] holds the current argmax
    int n = 0;
    auto is_stop = [&](int t) {
        for (int j = 0; j < n_stop; j++)
            if (stop[j] == t) return true;
        return false;
    };
    out[n++] = st.last_tok;
    bool done = is_stop(st.last_tok) || max_new <= 1;
    const int AHEAD = 8;
    int launched = 0;  // steps launched beyond step0
    while (!done) {
        const int want = std::min(max_new - 1, launched + AHEAD);
        while (launched < want) { run_step(c); launched++; }
        CK(cudaSetDevice(0));
        CK(cudaStreamSynchronize(S.G[0].s));
        for (; n <= launched && !done; ) {
            const int t = S.h_ring[(step0 + n - 1) % tp::RING];
            out[n++] = t;
            if (is_stop(t) || n >= max_new) done = true;
        }
        if (launched >= max_new - 1) done = true;
    }
    sync_both(c);
    check_err(c);
    c->gen_s += secs(t0);
    c->gen_tokens += n;
    return n;
}

int tp_set_option(t4q_ctx* c, const std::string& k, int v) {
    tp::State& S = *c->tps;
    if (k == "graphs") { S.graphs = v != 0; return 0; }
    if (k == "spin_ns") {
        S.spin_ns = v;
        for (int g = 0; g < 2; g++) { CK(cudaSetDevice(g)); tp::set_spin_ns(v); }
        sync_both(c);
        return 0;
    }
    if (k == "arn") {  // host-side launch choice: graphs are re-captured
        sync_both(c);
        tp::set_arn(v);
        S.arn = v;
        for (int g = 0; g < 2; g++) {
            CK(cudaSetDevice(g));
            if (S.G[g].gexec) { cudaGraphExecDestroy(S.G[g].gexec); S.G[g].gexec = nullptr; }
            if (S.G[g].graph) { cudaGraphDestroy(S.G[g].graph); S.G[g].graph = nullptr; }
        }
        return 0;
    }
    if (k == "ll" || k == "gdnf") {
        sync_both(c);
        if (k == "ll") S.ll = v;
        else S.gdnf = v;
        for (int g = 0; g < 2; g++) {
            CK(cudaSetDevice(g));
            if (S.G[g].gexec) { cudaGraphExecDestroy(S.G[g].gexec); S.G[g].gexec = nullptr; }
            if (S.G[g].graph) { cudaGraphDestroy(S.G[g].graph); S.G[g].graph = nullptr; }
        }
        return 0;
    }
    if (k == "fuse" || k == "pf_kb" || k == "arpub" || k == "mega") {  // graphs are re-captured on the next step
        sync_both(c);
        if (k == "mega" && v) {
            int cap = 1 << 30;
            for (int g = 0; g < 2; g++) {
                CK(cudaSetDevice(g));
                cap = std::min(cap, tp::mega_capacity());
            }
            if (cap < 80) throw std::runtime_error("mega: co-resident capacity " + std::to_string(cap) + " < 80");
            if (S.arpub != 2 || S.fuse != 0) throw std::runtime_error("mega needs arpub=2, fuse=0");
            S.mega_grid = 80;
        }
        if (k == "mega") S.mega = v;
        if (k == "fuse" && v == 1) throw std::runtime_error("fuse=1 (silu/gnorm prologues) retired in M4 v8");
        if (k == "fuse") S.fuse = v;
        else if (k == "arpub") S.arpub = v;
        else S.pf_kb = v;
        for (int g = 0; g < 2; g++) {
            CK(cudaSetDevice(g));
            if (S.G[g].gexec) { cudaGraphExecDestroy(S.G[g].gexec); S.G[g].gexec = nullptr; }
            if (S.G[g].graph) { cudaGraphDestroy(S.G[g].graph); S.G[g].graph = nullptr; }
        }
        return 0;
    }
    if (k == "trace") {  // CUPTI timeline of v steps in the current mode (state advances by v tokens)
        trace_steps(c, std::max(2, v));
        return 0;
    }
    if (k == "profile") {  // run one eager step with per-kernel events (state advances by one token)
        for (int g = 0; g < 2; g++) {
            for (auto e : S.G[g].ev) cudaEventDestroy(e);
            S.G[g].ev.clear();
            S.G[g].ev_name.clear();
        }
        if (true) {  // event nodes in captured graphs did not give elapsed times (M4 v5): profile eagerly
            eager_step(c, true);
        } else {  // profile the real graph: capture a copy with event-record nodes after every kernel
            cudaGraph_t gr[2];
            cudaGraphExec_t ge[2];
            for (int g = 0; g < 2; g++) {
                CK(cudaSetDevice(g));
                CK(cudaStreamBeginCapture(S.G[g].s, cudaStreamCaptureModeRelaxed));
                Enq q{c, true};
                q.gpu_step(g);
                CK(cudaStreamEndCapture(S.G[g].s, &gr[g]));
                CK(cudaGraphInstantiate(&ge[g], gr[g], 0));
            }
            for (int g = 0; g < 2; g++) {
                CK(cudaSetDevice(g));
                CK(cudaGraphLaunch(ge[g], S.G[g].s));
            }
            sync_both(c);
            for (int g = 0; g < 2; g++) {
                CK(cudaSetDevice(g));
                cudaGraphExecDestroy(ge[g]);
                cudaGraphDestroy(gr[g]);
            }
        }
        c->pos++;
        c->steps++;
        std::string js = "{\"mode\": \"eager\", ";
        for (int g = 0; g < 2; g++) {
            tp::Gpu& G = S.G[g];
            std::map<std::string, std::pair<double, int>> agg;
            double tot = 0;
            for (size_t i = 1; i < G.ev.size(); i++) {
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, G.ev[i - 1], G.ev[i]));
                agg[G.ev_name[i]].first += ms;
                agg[G.ev_name[i]].second += 1;
                tot += ms;
            }
            char b[256];
            snprintf(b, sizeof b, "%s\"gpu%d\": {\"total_ms\": %.3f", g ? ", " : "", g, tot);
            js += b;
            for (auto& kv : agg) {
                snprintf(b, sizeof b, ", \"%s\": [%.3f, %d]", kv.first.c_str(), kv.second.first, kv.second.second);
                js += b;
            }
            js += "}";
        }
        js += "}";
        S.prof_json = js;
        check_err(c);
        return 0;
    }
    return -1;
}

std::string tp_stats_json(t4q_ctx* c) {
    tp::State& S = *c->tps;
    char b[512];
    snprintf(b, sizeof b,
             ", \"tp\": 1, \"p2p\": %d, \"fuse\": %d, \"arpub\": %d, \"mega\": %d, \"pf_kb\": %d, \"graphs\": %d, "
             "\"ll\": %d, \"gdnf\": %d, \"spin_ns\": %d, \"arn\": %d, \"graph_capture_ms\": %.1f",
             (int)S.p2p, S.fuse, S.arpub, S.mega, S.pf_kb, (int)S.graphs, S.ll, S.gdnf, S.spin_ns, S.arn,
             S.ms_graph_capture);
    std::string s = b;
    if (!S.selftest_json.empty()) s += ", \"selftest\": " + S.selftest_json;
    if (!S.prof_json.empty()) s += ", \"profile\": " + S.prof_json;
    if (!S.trace_json.empty()) s += ", \"trace\": " + S.trace_json;
    return s;
}
