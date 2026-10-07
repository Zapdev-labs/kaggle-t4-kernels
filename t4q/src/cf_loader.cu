// cf-m1: the CYBER-FROST loader. Trunk weights repack into VRAM (Q2_K / Q4_0 / Q5_1 / F32),
// the 512 routed experts and the 26.85 GiB PLE table stay in the host mmap and are staged
// per token. Every repacked matrix is sampled bit-exactly against the CPU dequant.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "cf_model.h"
#include "quant_cpu.h"

using namespace cf;

namespace {

constexpr size_t STAGE = 64ull << 20;

size_t al256(size_t x) { return (x + 255) & ~(size_t)255; }

int fmt_for(uint32_t t) {
    switch (t) {
        case GT_F32: return FMT_F32;
        case GT_Q4_0: return FMT_P4;
        case GT_Q4_1: return FMT_P4M;
        case GT_Q8_0: return FMT_Q8;
        case GT_Q5_K: return FMT_K5;
        case GT_Q6_K: return FMT_K6;
        case GT_Q2_K: return FMT_K2;
        case GT_Q4_K: return FMT_K4;
        case GT_Q5_1: return FMT_Q51;
        default: return -1;
    }
}

struct RStats {
    int checked = 0, mismatched = 0;
    double max_abs_diff = 0;
    std::string first_bad;
};

void alloc_packed(PackedW& W, int gpu, int fmt, int64_t rows, int64_t cols) {
    W.fmt = fmt; W.rows = rows; W.cols = cols; W.gpu = gpu;
    const size_t n = (size_t)rows * cols;
    size_t sz[5] = {0, 0, 0, 0, 0};  // codes, hi, d, m, meta
    switch (fmt) {
        case FMT_F32: sz[0] = n * 4; break;
        case FMT_P4: sz[0] = n / 2; sz[2] = n / 32 * 2; break;
        case FMT_P4M: sz[0] = n / 2; sz[2] = n / 32 * 2; sz[3] = n / 32 * 2; break;
        case FMT_Q8: sz[0] = n; sz[2] = n / 32 * 2; break;
        case FMT_K5: sz[0] = n / 2; sz[1] = n / 8; sz[4] = n / 256 * 16; break;
        case FMT_K6: sz[0] = n / 2; sz[1] = n / 4; sz[2] = n / 256 * 2; sz[4] = n / 16; break;
        case FMT_K2: sz[0] = n / 4; sz[4] = n / 256 * 20; break;
        case FMT_K4: sz[0] = n / 2; sz[4] = n / 256 * 16; break;
        case FMT_Q51: sz[0] = n / 2; sz[1] = n / 8; sz[2] = n / 32 * 2; sz[3] = n / 32 * 2; break;
    }
    size_t tot = 0;
    for (size_t s : sz) tot += al256(s);
    CK(cudaSetDevice(gpu));
    CK(cudaMalloc(&W.base, tot));
    W.bytes = tot;
    uint8_t* p = (uint8_t*)W.base;
    if (sz[0]) { W.codes = p; p += al256(sz[0]); }
    if (sz[1]) { W.hi = p; p += al256(sz[1]); }
    if (sz[2]) { W.d = (uint16_t*)p; p += al256(sz[2]); }
    if (sz[3]) { W.m = (uint16_t*)p; p += al256(sz[3]); }
    if (sz[4]) { W.meta = p; p += al256(sz[4]); }
}

uint64_t lcg(uint64_t& s) { s = s * 6364136223846793005ull + 1442695040888963407ull; return s >> 33; }

void check_rows(CfCtx* c, RStats* rs, const GgufTensor* t, const PackedW& W, cudaStream_t st) {
    const int nchk = (int)std::min<int64_t>(W.rows, 64);
    uint64_t seed = 0x1234567ull ^ (uint64_t)W.rows * 31 ^ (uint64_t)W.cols;
    std::vector<int32_t> rows(nchk);
    for (int i = 0; i < nchk; i++)
        rows[i] = (i == 0) ? 0 : (i == 1 ? (int32_t)(W.rows - 1) : (int32_t)(lcg(seed) % W.rows));
    int32_t* drows;
    float* dout;
    CK(cudaMalloc(&drows, nchk * 4));
    CK(cudaMalloc(&dout, (size_t)nchk * W.cols * 4));
    CK(cudaMemcpyAsync(drows, rows.data(), nchk * 4, cudaMemcpyHostToDevice, st));
    launch_dequant_rows(W, drows, nchk, dout, st);
    CK(cudaGetLastError());
    std::vector<float> g((size_t)nchk * W.cols), ref(W.cols);
    CK(cudaMemcpyAsync(g.data(), dout, g.size() * 4, cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaFree(drows));
    CK(cudaFree(dout));
    for (int i = 0; i < nchk; i++) {
        if (!dequant_row_cpu(t->type, t->data + (size_t)rows[i] * t->row_bytes, ref.data(), W.cols))
            throw std::runtime_error("cpu dequant missing for " + t->name);
        const float* gr = g.data() + (size_t)i * W.cols;
        rs->checked++;
        if (memcmp(gr, ref.data(), W.cols * 4) != 0) {
            double md = 0;
            for (int64_t j = 0; j < W.cols; j++) md = std::max(md, (double)std::fabs(gr[j] - ref[j]));
            rs->mismatched++;
            rs->max_abs_diff = std::max(rs->max_abs_diff, md);
            if (rs->first_bad.empty()) rs->first_bad = t->name + " row " + std::to_string(rows[i]);
        }
    }
}

void upload_matrix(CfCtx* c, uint8_t* pin, uint8_t* dev, RStats* rs, const std::string& name, int gpu, PackedW& W) {
    const GgufTensor* t = c->f.find(name);
    if (!t) throw std::runtime_error("missing tensor " + name);
    const int fmt = fmt_for(t->type);
    if (fmt < 0) throw std::runtime_error(std::string("unsupported type ") + ggml_type_name(t->type) + " for " + name);
    if (t->ne[2] != 1) throw std::runtime_error("unexpected expert rank for " + name);
    alloc_packed(W, gpu, fmt, t->nrows(), t->ne[0]);
    cudaStream_t st = c->st;
    const int64_t chunk = std::max<int64_t>(1, (int64_t)(STAGE / t->row_bytes));
    for (int64_t r0 = 0; r0 < W.rows; r0 += chunk) {
        const int64_t nr = std::min(chunk, W.rows - r0);
        const size_t bytes = (size_t)nr * t->row_bytes;
        CK(cudaStreamSynchronize(st));
        memcpy(pin, t->data + (size_t)r0 * t->row_bytes, bytes);
        CK(cudaMemcpyAsync(dev, pin, bytes, cudaMemcpyHostToDevice, st));
        launch_repack(W, t->type, dev, r0, nr, st);
        CK(cudaGetLastError());
    }
    CK(cudaStreamSynchronize(st));
    check_rows(c, rs, t, W, st);
}

float* upload_vec(CfCtx* c, const std::string& name, int64_t expect) {
    const GgufTensor* t = c->f.find(name);
    if (!t) throw std::runtime_error("missing tensor " + name);
    if (t->type != GT_F32) throw std::runtime_error("expected F32 for " + name);
    const int64_t n = t->ne[0] * t->nrows();
    if (expect > 0 && n != expect) throw std::runtime_error("bad size for " + name);
    float* p;
    CK(cudaMalloc(&p, n * 4));
    CK(cudaMemcpy(p, t->data, n * 4, cudaMemcpyHostToDevice));
    return p;
}

// F16 [4, N] conv kernel -> fp32 device [N*4] laid out (channel*4 + tap), matching k_gdn_conv
float* upload_conv(CfCtx* c, const std::string& name, int64_t channels) {
    const GgufTensor* t = c->f.find(name);
    if (!t) throw std::runtime_error("missing tensor " + name);
    if (t->type != GT_F16 && t->type != GT_F32) throw std::runtime_error("expected F16/F32 for " + name);
    if (t->ne[0] != 4 || t->ne[1] != channels) throw std::runtime_error("bad shape for " + name);
    const size_t n = (size_t)channels * 4;
    std::vector<float> h(n);
    if (t->type == GT_F32) {
        memcpy(h.data(), t->data, n * 4);
    } else {
        const uint16_t* f16 = (const uint16_t*)t->data;
        for (size_t i = 0; i < channels; i++)
            for (int k = 0; k < 4; k++) h[i * 4 + k] = fp16_to_fp32(f16[i * 4 + k]);
    }
    float* p;
    CK(cudaMalloc(&p, n * 4));
    CK(cudaMemcpy(p, h.data(), n * 4, cudaMemcpyHostToDevice));
    return p;
}

template <class T>
T* dalloc(size_t n, bool zero = true) {
    T* p;
    CK(cudaMalloc(&p, n * sizeof(T)));
    if (zero) CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

}  // namespace

CfCtx* cf_load(const char* path, int max_ctx, std::string* err_out) {
    std::unique_ptr<CfCtx> up(new CfCtx());
    CfCtx* c = up.get();
    RStats rs;
    auto t0 = std::chrono::steady_clock::now();
    try {
        std::string err;
        if (!c->f.open(path, err)) throw std::runtime_error("gguf: " + err);
        auto need = [&](const char* k, double v) {
            double got = c->f.num(k, -1);
            if (std::fabs(got - v) > 1e-3 * std::max(1.0, std::fabs(v)))
                throw std::runtime_error(std::string("hparam mismatch ") + k + " got " + std::to_string(got));
        };
        need("qwen4exp.embedding_length", D);
        need("qwen4exp.block_count", 49);
        need("qwen4exp.attention.head_count", HQ);
        need("qwen4exp.attention.head_count_kv", HKV);
        need("qwen4exp.attention.key_length", HD);
        need("qwen4exp.rope.dimension_count", NROT);
        need("qwen4exp.rope.freq_base", ROPE_BASE);
        need("qwen4exp.ssm.group_count", HK);
        need("qwen4exp.ssm.time_step_rank", HV);
        need("qwen4exp.ssm.state_size", DK);
        need("qwen4exp.ssm.conv_kernel", 4);
        need("qwen4exp.expert_count", NE);
        need("qwen4exp.expert_used_count", TOPK);
        need("qwen4exp.expert_feed_forward_length", EE);
        need("qwen4exp.expert_shared_feed_forward_length", EE);
        need("qwen4exp.hyper_connection.count", HC);
        need("qwen4exp.hyper_connection.low_rank", LORA);
        need("qwen4exp.attention.layer_norm_rms_epsilon", EPS);
        need("qwen4exp.ple.ngram_size", PLE_NGRAM);
        need("qwen4exp.ple.heads_per_ngram", PLE_NHEADS / 2);
        need("qwen4exp.ple.conv_kernel", PLE_CONV);
        need("qwen4exp.ple.eos_token_id", EOS1);
        need("qwen4exp.ple.image_token_id", IMG_TOK);
        if (c->f.num("general.file_type", -1) < 0) throw std::runtime_error("no general.file_type");
        // the PLE hash constants must be exact u64
        const auto& mult = c->f.u64_arr("qwen4exp.ple.layer_multipliers");
        const auto& hoff = c->f.u64_arr("qwen4exp.ple.head_offsets");
        const auto& hvoc = c->f.u64_arr("qwen4exp.ple.head_vocab_sizes");
        if (mult.size() != 3 || hoff.size() != PLE_NHEADS || hvoc.size() != PLE_NHEADS)
            throw std::runtime_error("PLE KV arrays missing or short");
        memcpy(c->ple.mult, mult.data(), 24);
        memcpy(c->ple.head_off, hoff.data(), 128);
        memcpy(c->ple.head_vocab, hvoc.data(), 128);

        int ndev = 0;
        CK(cudaGetDeviceCount(&ndev));
        c->gpu = 0;
        CK(cudaSetDevice(0));
        CK(cudaStreamCreateWithFlags(&c->st, cudaStreamNonBlocking));
        cudaStream_t st = c->st;
        c->max_ctx = max_ctx > 0 ? max_ctx : 4096;

        uint8_t *pin, *dev;
        CK(cudaMallocHost(&pin, STAGE));
        CK(cudaMalloc(&dev, STAGE));

        c->tok_embd = c->f.find("token_embd.weight");
        if (!c->tok_embd || c->tok_embd->type != GT_Q4_K)
            throw std::runtime_error("token_embd.weight missing or not Q4_K");
        c->layers.resize(NL);
        for (int il = 0; il < NL; il++) {
            CfLayer& L = c->layers[il];
            L.il = il; L.gpu = 0; L.attn = is_attn(il);
            const std::string p = "blk." + std::to_string(il) + ".";
            for (int side = 0; side < 2; side++) {
                const std::string tag = side == 0 ? "hc_attn_" : "hc_ffn_";
                L.hc_norm[side] = upload_vec(c, p + tag + "norm.weight", HCD);
                upload_matrix(c, pin, dev, &rs, p + tag + "down.weight", 0, L.hc_down[side]);
                upload_matrix(c, pin, dev, &rs, p + tag + "up.weight", 0, L.hc_up[side]);
                upload_matrix(c, pin, dev, &rs, p + tag + "inject.weight", 0, L.hc_inject[side]);
            }
            if (L.attn) {
                upload_matrix(c, pin, dev, &rs, p + "attn_q.weight", 0, L.wq);
                upload_matrix(c, pin, dev, &rs, p + "attn_k.weight", 0, L.wk);
                upload_matrix(c, pin, dev, &rs, p + "attn_v.weight", 0, L.wv);
                upload_matrix(c, pin, dev, &rs, p + "attn_output.weight", 0, L.wo);
                L.q_norm = upload_vec(c, p + "attn_q_norm.weight", HD);
                L.k_norm = upload_vec(c, p + "attn_k_norm.weight", HD);
                L.kc = dalloc<uint16_t>((size_t)HKV * c->max_ctx * HD);
                L.vc = dalloc<uint16_t>((size_t)HKV * c->max_ctx * HD);
            } else {
                upload_matrix(c, pin, dev, &rs, p + "attn_qkv.weight", 0, L.qkv);
                upload_matrix(c, pin, dev, &rs, p + "attn_gate.weight", 0, L.z);
                upload_matrix(c, pin, dev, &rs, p + "ssm_alpha.weight", 0, L.alpha);
                upload_matrix(c, pin, dev, &rs, p + "ssm_beta.weight", 0, L.beta);
                upload_matrix(c, pin, dev, &rs, p + "ssm_out.weight", 0, L.ssm_out);
                L.conv_w = upload_conv(c, p + "ssm_conv1d.weight", CONV);
                L.ssm_a = upload_vec(c, p + "ssm_a", HV);
                L.ssm_dt = upload_vec(c, p + "ssm_dt.bias", HV);
                L.ssm_norm = upload_vec(c, p + "ssm_norm.weight", DK);
                L.conv_state = dalloc<float>((size_t)CONV * 3);
                L.S = dalloc<float>((size_t)HV * DK * DK);
            }
            // MoE: router + shared expert in VRAM, the routed experts stay mmap'd
            upload_matrix(c, pin, dev, &rs, p + "ffn_gate_inp.weight", 0, L.router);
            upload_matrix(c, pin, dev, &rs, p + "ffn_gate_shexp.weight", 0, L.sh_gate);
            upload_matrix(c, pin, dev, &rs, p + "ffn_up_shexp.weight", 0, L.sh_up);
            upload_matrix(c, pin, dev, &rs, p + "ffn_down_shexp.weight", 0, L.sh_down);
            upload_matrix(c, pin, dev, &rs, p + "ffn_gate_inp_shexp.weight", 0, L.sh_ginp);
            L.t_gate_exps = c->f.find(p + "ffn_gate_exps.weight");
            L.t_up_exps = c->f.find(p + "ffn_up_exps.weight");
            L.t_down_exps = c->f.find(p + "ffn_down_exps.weight");
            if (!L.t_gate_exps || !L.t_up_exps || !L.t_down_exps)
                throw std::runtime_error("missing expert tensors for layer " + std::to_string(il));
            if (il == PLE_LAYER) {
                upload_matrix(c, pin, dev, &rs, p + "ple_key.weight", 0, c->ple.key);
                upload_matrix(c, pin, dev, &rs, p + "ple_value.weight", 0, c->ple.value);
                c->ple.norm_key = upload_vec(c, p + "ple_norm_key.weight", HCD);
                c->ple.norm_query = upload_vec(c, p + "ple_norm_query.weight", HCD);
                c->ple.norm_conv = upload_vec(c, p + "ple_norm_conv.weight", HCD);
                c->ple.conv_w = upload_conv(c, p + "ple_conv1d.weight", HCD);
                c->ple.table = c->f.find("per_layer_token_embd.weight");
                if (!c->ple.table) throw std::runtime_error("missing per_layer_token_embd.weight");
                if (c->ple.table->ne[0] != PLE_DIM || c->ple.table->type != GT_Q4_0)
                    throw std::runtime_error("bad PLE table shape/type");
            }
            if (il % 16 == 15) {
                double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                fprintf(stderr, "[cf] loaded layers 0..%d (%.1f s)\n", il, s);
            }
        }
        // the final mixer is the output norm; then the lm_head
        c->o_norm = upload_vec(c, "output_hc_norm.weight", HCD);
        upload_matrix(c, pin, dev, &rs, "output_hc_down.weight", 0, c->o_down);
        upload_matrix(c, pin, dev, &rs, "output_hc_up.weight", 0, c->o_up);
        upload_matrix(c, pin, dev, &rs, "output.weight", 0, c->output);

        // decode scratch + staging (the expert staging areas are per token, allocated once)
        CfScratch& s = c->sc;
        s.h = dalloc<float>(HCD, false); s.xn = dalloc<float>(HCD, false); s.lo = dalloc<float>(LORA);
        s.gate = dalloc<float>(HCD); s.mixed = dalloc<float>(D); s.inj = dalloc<float>(HC);
        s.block = dalloc<float>(D, false);
        s.qfull = dalloc<float>(HQ * HD * 2, false); s.aq = dalloc<float>(HQ * HD, false);
        s.ak = dalloc<float>(HKV * HD, false); s.k = dalloc<float>(HKV * HD, false); s.v = dalloc<float>(HKV * HD, false);
        s.att = dalloc<float>(HQ * HD, false); s.attg = dalloc<float>(HQ * HD, false);
        s.scores = dalloc<float>((size_t)HQ * c->max_ctx);
        s.qkv = dalloc<float>(CONV, false); s.zz = dalloc<float>(VDIM, false);
        s.braw = dalloc<float>(HV, false); s.araw = dalloc<float>(HV, false);
        s.beta = dalloc<float>(HV, false); s.g = dalloc<float>(HV, false);
        s.conv = dalloc<float>(CONV, false); s.qn = dalloc<float>(HK * DK, false); s.kn = dalloc<float>(HK * DK, false);
        s.o = dalloc<float>(VDIM, false); s.on = dalloc<float>(VDIM, false); s.a = dalloc<float>(D, false);
        s.ffg = dalloc<float>((size_t)TOPK * EE, false); s.ffu = dalloc<float>((size_t)TOPK * EE, false);
        s.ffa = dalloc<float>((size_t)TOPK * EE, false);
        s.logits = dalloc<float>(V, false);
        // the q8 fast-path activation planes (the oracle's own activation quantizations)
        s.xq1 = dalloc<int8_t>(HCD, false); s.xq1_d = dalloc<float>(HCD / 32, false);
        s.xq1_s = dalloc<float>(HCD / 32, false);
        s.xqk = dalloc<int8_t>(HCD, false); s.xqk_b = dalloc<int16_t>(HCD / 16, false);
        s.xqk_d = dalloc<float>(HCD / 256, false);
        // the q8_0 activation planes sized for the LARGEST gemv K (HCD): the trunk's Q4_0 gemv
        // (hc_ffn_down, K=10240) overflowed the old TOPK*EE=6400 sizing and NaN'd the neighbors
        s.xq0 = dalloc<int8_t>(HCD, false); s.xd0 = dalloc<float>(HCD / 32, false);
        s.xs0 = dalloc<int>(HCD / 32, false);
        c->ye = dalloc<float>((size_t)TOPK * D, false);
        c->we = dalloc<float>(TOPK, false);
        c->ysh = dalloc<float>(D, false);
        c->sh_gate_raw = dalloc<float>(1, false);
        c->ple_key = dalloc<float>(HCD, false); c->ple_query = dalloc<float>(HCD, false);
        c->ple_s = dalloc<float>(HC, false); c->ple_gate = dalloc<float>(HC, false);
        c->ple_gated = dalloc<float>(HCD, false);
        c->ple_hist = dalloc<float>((size_t)PLE_HIST * HCD);
        // expert staging: [10 experts][gate 640 | up 640] Q2_K rows of D=2560, down 2560 rows of EE=640 Q4_0
        alloc_packed(c->up_stage, 0, fmt_for(GT_Q2_K), (int64_t)TOPK * 2 * EE, D);
        alloc_packed(c->dn_stage, 0, fmt_for(GT_Q4_0), (int64_t)TOPK * D, EE);
        {
            // one layer's worth of raw slabs: 10*(640+640) Q2_K rows of 840 B + 10*2560 Q4_0 rows of 360 B
            const size_t up_bytes = (size_t)TOPK * 2 * EE * c->layers[0].t_gate_exps->row_bytes;
            const size_t dn_bytes = (size_t)TOPK * D * c->layers[0].t_down_exps->row_bytes;
            CK(cudaMallocHost(&c->raw_stage, up_bytes + dn_bytes));
            CK(cudaMalloc(&c->raw_dev, up_bytes + dn_bytes));
        }
        // the per-pick W tables: the IDENTITY views of the staged slabs (the cf-m3 tiering
        // swaps in per-hit resident views later with zero kernel change; the staging bases
        // never move, so this is built + uploaded ONCE). The K2 gate|up slab view: rows 2*EE,
        // cols D - the codes plane 640 B/row, the meta 200 B/row; the P4 down slab: rows D,
        // cols EE - the codes 320 B/row, the d 20 fp16/row (element offsets).
        for (int p = 0; p < TOPK; p++) {
            PackedW& v = c->h_wt_gu[p];
            v = c->up_stage;
            v.rows = 2 * EE;
            v.codes = c->up_stage.codes + (size_t)p * 2 * EE * (D / 4);
            v.meta = c->up_stage.meta + (size_t)p * 2 * EE * (size_t)(D / 256) * 20;
            PackedW& w = c->h_wt_dn[p];
            w = c->dn_stage;
            w.rows = D;
            w.codes = c->dn_stage.codes + (size_t)p * D * (EE / 2);
            w.d = c->dn_stage.d + (size_t)p * D * (EE / 32);
        }
        CK(cudaMalloc(&c->wt_gu, (size_t)TOPK * sizeof(PackedW)));
        CK(cudaMalloc(&c->wt_dn, (size_t)TOPK * sizeof(PackedW)));
        CK(cudaMemcpy(c->wt_gu, c->h_wt_gu, (size_t)TOPK * sizeof(PackedW), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(c->wt_dn, c->h_wt_dn, (size_t)TOPK * sizeof(PackedW), cudaMemcpyHostToDevice));
        // the resident tier (cf-m3): T4Q_CF_HOTSET=<path>, produced by cf_census.py --hotset
        // from the round's census.bin. ABSENT = OFF = the verbatim full-staging path (the
        // identity W table above stands). Present = ON: the per-layer top-H experts (by
        // routed mass) are packed into VRAM here at load, in the SAME packed shapes as the
        // staging, through the SAME chunked raw->pinned->H2D->repack path the moe uses per
        // step - so the resident slabs are byte-identical to what the staging would produce
        // for the same experts, and the dual-path moe's hit picks read them with zero staging.
        if (const char* hs_path = getenv("T4Q_CF_HOTSET")) {
            FILE* hf = fopen(hs_path, "rb");
            if (!hf) throw std::runtime_error(std::string("hot-set open failed: ") + hs_path);
            uint32_t magic = 0, nl = 0, h = 0;
            if (fread(&magic, 4, 1, hf) != 1 || fread(&nl, 4, 1, hf) != 1 || fread(&h, 4, 1, hf) != 1 ||
                magic != 0x53484643u /* "CFHS" */ || (int)nl != NL || h == 0 || h > (uint32_t)NE)
                throw std::runtime_error("bad hot-set header (want CFHS, NL, 0 < H <= NE)");
            // the VRAM check before any alloc: H resident slabs/layer at ~2.0 MB each
            {
                const double per_layer_mb = h * (2.0 * EE * 840.0 / 1e6 + (double)D * 360.0 / 1e6);
                const double need_gib = per_layer_mb * NL / 1024.0;
                size_t free_b = 0, total_b = 0;
                CK(cudaMemGetInfo(&free_b, &total_b));
                if ((double)free_b < need_gib * 1024 * 1024 * 1024 * 1.02)
                    throw std::runtime_error("hot set needs " + std::to_string(need_gib) + " GiB, only " +
                                             std::to_string((double)free_b / (1 << 30)) + " GiB free");
                fprintf(stderr, "[cf] resident tier: H=%u/layer, %.2f GiB\n", h, need_gib);
            }
            const size_t gu_row = c->layers[0].t_gate_exps->row_bytes;  // 840
            const size_t dn_row = c->layers[0].t_down_exps->row_bytes;  // 360
            const size_t up_bytes = (size_t)TOPK * 2 * EE * gu_row;    // the fixed raw_stage layout
            for (int il = 0; il < NL; il++) {
                CfLayer& L = c->layers[il];
                L.hn = (int)h;
                L.hot_ids = new int[h];
                L.hot_idx = new int[NE];
                memset(L.hot_idx, -1, (size_t)NE * sizeof(int));
                if (fread(L.hot_ids, 4, h, hf) != h)
                    throw std::runtime_error("hot-set truncated at layer " + std::to_string(il));
                for (uint32_t j = 0; j < h; j++) {
                    if (L.hot_ids[j] < 0 || L.hot_ids[j] >= NE || L.hot_idx[L.hot_ids[j]] >= 0)
                        throw std::runtime_error("bad/duplicate hot-set id at layer " + std::to_string(il));
                    L.hot_idx[L.hot_ids[j]] = (int)j;
                }
                alloc_packed(L.res_gu, 0, fmt_for(GT_Q2_K), (int64_t)L.hn * 2 * EE, D);
                alloc_packed(L.res_dn, 0, fmt_for(GT_Q4_0), (int64_t)L.hn * D, EE);
                for (int h0 = 0; h0 < L.hn; h0 += TOPK) {  // TOPK slabs per pass through raw_stage
                    const int ch = std::min(TOPK, L.hn - h0);
                    CK(cudaStreamSynchronize(st));  // the previous pass's H2D must drain before the refill
                    for (int j = 0; j < ch; j++) {
                        const int64_t e = L.hot_ids[h0 + j];
                        uint8_t* dst = c->raw_stage + (size_t)j * 2 * EE * gu_row;
                        memcpy(dst, L.t_gate_exps->data + (size_t)e * EE * gu_row, (size_t)EE * gu_row);
                        memcpy(dst + (size_t)EE * gu_row, L.t_up_exps->data + (size_t)e * EE * gu_row,
                               (size_t)EE * gu_row);
                        memcpy(c->raw_stage + up_bytes + (size_t)j * D * dn_row,
                               L.t_down_exps->data + (size_t)e * D * dn_row, (size_t)D * dn_row);
                    }
                    CK(cudaMemcpyAsync(c->raw_dev, c->raw_stage, (size_t)ch * 2 * EE * gu_row,
                                       cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(c->raw_dev + up_bytes, c->raw_stage + up_bytes, (size_t)ch * D * dn_row,
                                       cudaMemcpyHostToDevice, st));
                    launch_repack(L.res_gu, GT_Q2_K, c->raw_dev, (int64_t)h0 * 2 * EE, (int64_t)ch * 2 * EE, st);
                    launch_repack(L.res_dn, GT_Q4_0, c->raw_dev + up_bytes, (int64_t)h0 * D, (int64_t)ch * D, st);
                    CK(cudaGetLastError());
                }
            }
            fclose(hf);
            CK(cudaStreamSynchronize(st));
            c->tiered = true;
        }
        // cf-m3 (r19u) the UVA third path: T4Q_CF_UVA_LAYERS=<n> registers the first n
        // layers' expert tensors as cudaHostRegisterMapped (ONE coalesced page-aligned
        // span - the per-tensor page-spans of the file-adjacent tensors overlap on the
        // shared boundary pages and would double-register). ABSENT/0 = OFF = the
        // verbatim full-staging path. The full 47.9 GiB region fits only a big-RAM host;
        // the Kaggle host's ~29 GB caps n at ~28-29 (the r19r RAM cap): the registered
        // layers' picks read their raw slabs through the aliases over PCIe, the rest stay
        // staged. Byte-identity by construction: the aliases read the SAME mmap pages the
        // OFF path's memcpys stage, through the same repack block decode (the scatter
        // kernels' address math is exactly the OFF path's memcpy sources).
        if (const char* un = getenv("T4Q_CF_UVA_LAYERS")) {
            c->uva_n = atoi(un);
            if (c->uva_n <= 0 || c->uva_n > NL)
                throw std::runtime_error("T4Q_CF_UVA_LAYERS must be in 1..48");
            const uintptr_t PG = 4095;
            uintptr_t lo = ~(uintptr_t)0, hi = 0;
            for (int il = 0; il < c->uva_n; il++) {
                CfLayer& L = c->layers[il];
                for (const GgufTensor* t : {L.t_gate_exps, L.t_up_exps, L.t_down_exps}) {
                    const uintptr_t b = (uintptr_t)t->data;
                    const uintptr_t e = b + (uintptr_t)t->nrows() * t->row_bytes;
                    lo = std::min(lo, b & ~PG);
                    hi = std::max(hi, (e + PG) & ~PG);
                }
            }
            auto tr0 = std::chrono::steady_clock::now();
            CK(cudaHostRegister((void*)lo, (size_t)(hi - lo), cudaHostRegisterMapped));
            void* alias = nullptr;
            CK(cudaHostGetDevicePointer(&alias, (void*)lo, 0));
            auto tr1 = std::chrono::steady_clock::now();
            for (int il = 0; il < c->uva_n; il++) {
                CfLayer& L = c->layers[il];
                L.uva_gate = (const uint8_t*)alias + ((uintptr_t)L.t_gate_exps->data - lo);
                L.uva_up = (const uint8_t*)alias + ((uintptr_t)L.t_up_exps->data - lo);
                L.uva_dn = (const uint8_t*)alias + ((uintptr_t)L.t_down_exps->data - lo);
            }
            c->uva_reg = (void*)lo;
            c->uva_reg_len = (size_t)(hi - lo);
            c->uva = true;
            CK(cudaMalloc(&c->eid_dev, (size_t)TOPK * 4));
            fprintf(stderr, "[cf] UVA: %d/%d layers, %.2f GiB registered in %.2f s (the alias path ON)\n",
                    c->uva_n, NL, (double)(hi - lo) / (1ull << 30),
                    std::chrono::duration<double>(tr1 - tr0).count());
        }
        // cf-m4 (r19w): the MTP draft block (blk.48, ALL Q8_0, resident - the frozen
        // CF_MTP.md design). T4Q_CF_MTP=1 loads it (absent = not loaded, the ~2.5 GiB
        // stays free for the tiering); INERT until cf_draft_step is called (the acceptance
        // smoke now, the speculative verify/rollback driver later). The pair semantics are
        // the 27B's gate-proven form: (x_q, h_{q-1}) at the draft's own KV position q,
        // h_{-1} = 0. blk.48 is NON-RECURRENT (attention.recurrent_layers[49]) despite
        // 48 % 4 != 3, so L.attn is forced true. ALL 512 experts pack into VRAM (the same
        // chunked raw->pinned->H2D->repack pass as the hot-set tier; the trunk-sized
        // raw_stage takes 3 draft experts per pass - the Q8_0 rows are 2720/680 B vs the
        // trunk's 840/360), so the draft's MoE is STAGING-FREE: the per-step W table
        // composes the 10 picks' resident views (the tiering's mechanism, all hits).
        if (getenv("T4Q_CF_MTP") && atoi(getenv("T4Q_CF_MTP"))) {
            CfDraft* d = new CfDraft();
            c->draft = d;
            CfLayer& L = d->L;
            L.il = NL; L.gpu = 0; L.attn = true;
            const std::string p = "blk." + std::to_string(NL) + ".";
            for (int side = 0; side < 2; side++) {
                const std::string tag = side == 0 ? "hc_attn_" : "hc_ffn_";
                L.hc_norm[side] = upload_vec(c, p + tag + "norm.weight", HCD);
                upload_matrix(c, pin, dev, &rs, p + tag + "down.weight", 0, L.hc_down[side]);
                upload_matrix(c, pin, dev, &rs, p + tag + "up.weight", 0, L.hc_up[side]);
                upload_matrix(c, pin, dev, &rs, p + tag + "inject.weight", 0, L.hc_inject[side]);
            }
            upload_matrix(c, pin, dev, &rs, p + "attn_q.weight", 0, L.wq);
            upload_matrix(c, pin, dev, &rs, p + "attn_k.weight", 0, L.wk);
            upload_matrix(c, pin, dev, &rs, p + "attn_v.weight", 0, L.wv);
            upload_matrix(c, pin, dev, &rs, p + "attn_output.weight", 0, L.wo);
            L.q_norm = upload_vec(c, p + "attn_q_norm.weight", HD);
            L.k_norm = upload_vec(c, p + "attn_k_norm.weight", HD);
            L.kc = dalloc<uint16_t>((size_t)HKV * c->max_ctx * HD);
            L.vc = dalloc<uint16_t>((size_t)HKV * c->max_ctx * HD);
            upload_matrix(c, pin, dev, &rs, p + "ffn_gate_inp.weight", 0, L.router);
            upload_matrix(c, pin, dev, &rs, p + "ffn_gate_shexp.weight", 0, L.sh_gate);
            upload_matrix(c, pin, dev, &rs, p + "ffn_up_shexp.weight", 0, L.sh_up);
            upload_matrix(c, pin, dev, &rs, p + "ffn_down_shexp.weight", 0, L.sh_down);
            upload_matrix(c, pin, dev, &rs, p + "ffn_gate_inp_shexp.weight", 0, L.sh_ginp);
            L.t_gate_exps = c->f.find(p + "ffn_gate_exps.weight");
            L.t_up_exps = c->f.find(p + "ffn_up_exps.weight");
            L.t_down_exps = c->f.find(p + "ffn_down_exps.weight");
            if (!L.t_gate_exps || !L.t_up_exps || !L.t_down_exps)
                throw std::runtime_error("missing draft expert tensors (blk.48)");
            d->enorm = upload_vec(c, p + "nextn.enorm.weight", D);
            d->hnorm = upload_vec(c, p + "nextn.hnorm.weight", HCD);
            upload_matrix(c, pin, dev, &rs, p + "nextn.eh_proj.weight", 0, d->eh_proj);
            d->hh_norm = upload_vec(c, p + "nextn.hc_head_norm.weight", HCD);
            upload_matrix(c, pin, dev, &rs, p + "nextn.hc_head_down.weight", 0, d->hh_down);
            upload_matrix(c, pin, dev, &rs, p + "nextn.hc_head_up.weight", 0, d->hh_up);
            // the draft's own scratch (the trunk's is never touched); the planes sized for
            // the largest gemv K (HCD: the hc mixers' down/inject columns)
            CfScratch& ds = d->sc;
            d->zero_h = dalloc<float>(HCD);
            CK(cudaMallocHost(&d->h_e, (size_t)D * 4));
            CK(cudaMallocHost(&d->h_logits, (size_t)V * 4));
            d->h_in = dalloc<float>(HCD, false);
            d->hres = dalloc<float>(HCD, false);
            d->e = dalloc<float>(D, false); d->e_norm = dalloc<float>(D, false);
            d->h_norm = dalloc<float>(HCD, false);
            d->eh_cat = dalloc<float>((size_t)HC * 2 * D, false);
            d->ygu = dalloc<float>((size_t)TOPK * 2 * EE, false);
            ds.h = dalloc<float>(HCD, false); ds.xn = dalloc<float>(HCD, false); ds.lo = dalloc<float>(LORA);
            ds.gate = dalloc<float>(HCD); ds.mixed = dalloc<float>(D); ds.inj = dalloc<float>(HC);
            ds.block = dalloc<float>(D, false);
            ds.qfull = dalloc<float>(HQ * HD * 2, false); ds.aq = dalloc<float>(HQ * HD, false);
            ds.ak = dalloc<float>(HKV * HD, false); ds.k = dalloc<float>(HKV * HD, false);
            ds.v = dalloc<float>(HKV * HD, false); ds.att = dalloc<float>(HQ * HD, false);
            ds.attg = dalloc<float>(HQ * HD, false); ds.scores = dalloc<float>((size_t)HQ * c->max_ctx);
            ds.ffg = dalloc<float>((size_t)TOPK * EE, false); ds.ffu = dalloc<float>((size_t)TOPK * EE, false);
            ds.ffa = dalloc<float>((size_t)TOPK * EE, false);
            d->ye = dalloc<float>((size_t)TOPK * D, false); d->ysh = dalloc<float>(D, false);
            d->sh_gate_raw = dalloc<float>(1, false);
            ds.xq0 = dalloc<int8_t>(HCD, false); ds.xd0 = dalloc<float>(HCD / 32, false);
            ds.xs0 = dalloc<int>(HCD / 32, false);
            ds.xqk = dalloc<int8_t>(HCD, false); ds.xqk_b = dalloc<int16_t>(HCD / 16, false);
            ds.xqk_d = dalloc<float>(HCD / 256, false);
            // the Q8_1 trio (the FMT_Q51 branch's planes; the graft is all-Q8_0 so these
            // are dead weight, but the shared lm_head is Q4_K and the engine's gemv() must
            // never deref a NULL plane whatever the file's draft tensors turn out to be)
            ds.xq1 = dalloc<int8_t>(HCD, false); ds.xq1_d = dalloc<float>(HCD / 32, false);
            ds.xq1_s = dalloc<float>(HCD / 32, false);
            ds.logits = dalloc<float>(V, false);
            // the ALL-512 resident slabs (the VRAM check first: ~2.67 GB packed)
            {
                const double need_gib =
                    (NE * 2.0 * EE * D + (double)NE * D * EE) * 1.0625 / (1ull << 30) * 1.0;
                size_t free_b = 0, total_b = 0;
                CK(cudaMemGetInfo(&free_b, &total_b));
                if ((double)free_b < need_gib * (1ull << 30) * 1.02)
                    throw std::runtime_error("the draft block needs " + std::to_string(need_gib) +
                                             " GiB resident, only " + std::to_string((double)free_b / (1ull << 30)) +
                                             " GiB free");
                fprintf(stderr, "[cf] draft block: resident tier %.2f GiB\n", need_gib);
            }
            alloc_packed(d->res_gu, 0, FMT_Q8, (int64_t)NE * 2 * EE, D);
            alloc_packed(d->res_dn, 0, FMT_Q8, (int64_t)NE * D, EE);
            CK(cudaMalloc(&d->wt_gu, (size_t)TOPK * sizeof(PackedW)));
            CK(cudaMalloc(&d->wt_dn, (size_t)TOPK * sizeof(PackedW)));
            {
                const size_t gu_row_d = L.t_gate_exps->row_bytes;   // Q8_0 rows of D (2720)
                const size_t dn_row_d = L.t_down_exps->row_bytes;   // Q8_0 rows of EE (680)
                const size_t up_bytes =
                    (size_t)TOPK * 2 * EE * c->layers[0].t_gate_exps->row_bytes;  // the raw_stage layout
                const size_t dn_bytes = (size_t)TOPK * D * c->layers[0].t_down_exps->row_bytes;
                const int ch = std::max(
                    1, (int)std::min(up_bytes / (2 * EE * gu_row_d), dn_bytes / ((size_t)D * dn_row_d)));
                for (int e0 = 0; e0 < NE; e0 += ch) {
                    const int n = std::min(ch, NE - e0);
                    CK(cudaStreamSynchronize(st));  // the previous pass's H2D must drain before the refill
                    for (int j = 0; j < n; j++) {
                        const int64_t e = e0 + j;
                        uint8_t* dst = c->raw_stage + (size_t)j * 2 * EE * gu_row_d;
                        memcpy(dst, L.t_gate_exps->data + (size_t)e * EE * gu_row_d, (size_t)EE * gu_row_d);
                        memcpy(dst + (size_t)EE * gu_row_d, L.t_up_exps->data + (size_t)e * EE * gu_row_d,
                               (size_t)EE * gu_row_d);
                        memcpy(c->raw_stage + up_bytes + (size_t)j * D * dn_row_d,
                               L.t_down_exps->data + (size_t)e * D * dn_row_d, (size_t)D * dn_row_d);
                    }
                    CK(cudaMemcpyAsync(c->raw_dev, c->raw_stage, (size_t)n * 2 * EE * gu_row_d, cudaMemcpyHostToDevice,
                                       st));
                    CK(cudaMemcpyAsync(c->raw_dev + up_bytes, c->raw_stage + up_bytes, (size_t)n * D * dn_row_d,
                                       cudaMemcpyHostToDevice, st));
                    launch_repack(d->res_gu, GT_Q8_0, c->raw_dev, (int64_t)e0 * 2 * EE, (int64_t)n * 2 * EE, st);
                    launch_repack(d->res_dn, GT_Q8_0, c->raw_dev + up_bytes, (int64_t)e0 * D, (int64_t)n * D, st);
                    CK(cudaGetLastError());
                }
                CK(cudaStreamSynchronize(st));
            }
            const double d_dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "[cf] draft block loaded (all-512 resident) %.1f s\n", d_dur);
            // cf-m4 (r19x): the MTP verify's per-row planes + the union staging (the same
            // T4Q_CF_MTP gate; k rides T4Q_CF_K, default 3 -> nr = k+1 rows, the MAXR=8
            // bound). The per-row scratch = the trunk's plane list verbatim (the rolling
            // states - KV, GDN S/conv, the PLE ring - stay the TRUNK's buffers, evolving in
            // row order exactly as the sequential steps); the UNION slabs are the same
            // packed formats as the staging (K2 gu / P4 dn), sized for the WORST case
            // nr*TOPK experts (the dedup usually ~1.5x), fed by the same repack path.
            {
                const char* ke = getenv("T4Q_CF_K");
                const int k = ke ? atoi(ke) : 3;
                if (k < 0 || k + 1 > MAXR) throw std::runtime_error("T4Q_CF_K out of range (0..7)");
                CfVerify* v = new CfVerify();
                c->verify = v;
                v->nr = k + 1;
                const int nr = v->nr;
                CK(cudaMallocHost(&v->h_pos, MAXR * 4));
                CK(cudaMalloc(&v->d_pos, MAXR * 4));
                CK(cudaMallocHost(&v->h_emb, (size_t)MAXR * D * 4));
                CK(cudaMallocHost(&v->h_ple, (size_t)MAXR * D * 4));
                CK(cudaMallocHost(&v->h_router, (size_t)MAXR * NE * 4));
                // r19z sweep: the per-row picks plane - written by the window's host_top10_row,
                // read by the dedup + the W-table compose (host-only, no captured node reads
                // it; pinned to match the sibling planes' discipline)
                CK(cudaMallocHost(&v->eid, (size_t)MAXR * TOPK * 4));
                CK(cudaMallocHost(&v->we_h, (size_t)MAXR * TOPK * 4));
                CK(cudaMalloc(&v->we_dev, (size_t)MAXR * TOPK * 4));
                CK(cudaMallocHost(&v->h_logits, (size_t)MAXR * V * 4));
                CK(cudaMalloc(&v->wt_gu, (size_t)MAXR * TOPK * sizeof(PackedW)));
                CK(cudaMalloc(&v->wt_dn, (size_t)MAXR * TOPK * sizeof(PackedW)));
                CK(cudaMalloc(&v->uids_dev, (size_t)MAXR * TOPK * 4));
                CK(cudaMalloc(&v->ye, (size_t)MAXR * TOPK * D * 4));
                CK(cudaMalloc(&v->ysh, (size_t)MAXR * D * 4));
                CK(cudaMalloc(&v->sh_gate_raw, (size_t)MAXR * 4));
                for (int i = 0; i < NE; i++) v->uidx[i] = -1;  // the union slot map (recomposed per layer)
                for (int r = 0; r < MAXR; r++) {  // the per-row scratch: the trunk's plane list verbatim
                    CfScratch& s = v->sc[r];
                    s.h = dalloc<float>(HCD, false); s.xn = dalloc<float>(HCD, false); s.lo = dalloc<float>(LORA);
                    s.gate = dalloc<float>(HCD); s.mixed = dalloc<float>(D); s.inj = dalloc<float>(HC);
                    s.block = dalloc<float>(D, false);
                    s.qfull = dalloc<float>(HQ * HD * 2, false); s.aq = dalloc<float>(HQ * HD, false);
                    s.ak = dalloc<float>(HKV * HD, false); s.k = dalloc<float>(HKV * HD, false);
                    s.v = dalloc<float>(HKV * HD, false);
                    s.att = dalloc<float>(HQ * HD, false); s.attg = dalloc<float>(HQ * HD, false);
                    s.scores = dalloc<float>((size_t)HQ * c->max_ctx);
                    s.qkv = dalloc<float>(CONV, false); s.zz = dalloc<float>(VDIM, false);
                    s.braw = dalloc<float>(HV, false); s.araw = dalloc<float>(HV, false);
                    s.beta = dalloc<float>(HV, false); s.g = dalloc<float>(HV, false);
                    s.conv = dalloc<float>(CONV, false); s.qn = dalloc<float>(HK * DK, false);
                    s.kn = dalloc<float>(HK * DK, false);
                    s.o = dalloc<float>(VDIM, false); s.on = dalloc<float>(VDIM, false); s.a = dalloc<float>(D, false);
                    s.ffg = dalloc<float>((size_t)TOPK * EE, false); s.ffu = dalloc<float>((size_t)TOPK * EE, false);
                    s.ffa = dalloc<float>((size_t)TOPK * EE, false);
                    s.logits = dalloc<float>(V, false);
                    s.xq1 = dalloc<int8_t>(HCD, false); s.xq1_d = dalloc<float>(HCD / 32, false);
                    s.xq1_s = dalloc<float>(HCD / 32, false);
                    s.xqk = dalloc<int8_t>(HCD, false); s.xqk_b = dalloc<int16_t>(HCD / 16, false);
                    s.xqk_d = dalloc<float>(HCD / 256, false);
                    s.xq0 = dalloc<int8_t>(HCD, false); s.xd0 = dalloc<float>(HCD / 32, false);
                    s.xs0 = dalloc<int>(HCD / 32, false);
                }
                // cf-m4 (r19y): the speculative driver's snapshot planes (the 27B's tp_spec.cu
                // rollback adapted to the CF rolling states - the GDN S/conv and the PLE ring
                // are SHIFT REGISTERS, not position-indexed, so a partial accept restores the
                // after-row-n state from the verify's per-row captures): the S/conv snapshots
                // [ngdn][k] (the state AFTER row r, r = 0..k-1; the accept n = k leaves the
                // rolling L.S itself correct, no capture), the PLE ring snapshots [k], and
                // the pending_h ring [MAXR] (the per-row pre-final-mixer residuals - the
                // catch-up's h inputs, CF_MTP.md section 4). The attention KV needs NO plane
                // (the cells beyond the rewound pos are invisible). Pure scratch: every plane
                // is written before any read (the verify captures rows 0..k-1 before the
                // restore reads slot n <= k-1), so no zeroing.
                v->ngdn = 0;
                for (int il = 0; il < NL; il++) v->gord[il] = is_attn(il) ? -1 : v->ngdn++;
                v->pending_h = dalloc<float>((size_t)MAXR * HCD, false);
                if (k > 0) {
                    const double snap_gib =
                        (double)v->ngdn * k * ((double)HV * DK * DK * 4 + (double)CONV * 3 * 4) / (1ull << 30) +
                        (double)k * PLE_HIST * HCD * 4.0 / (1ull << 30);
                    size_t free_b = 0, total_b = 0;
                    CK(cudaMemGetInfo(&free_b, &total_b));
                    if ((double)free_b < snap_gib * (1ull << 30) * 1.02)
                        throw std::runtime_error("the spec snapshots need " + std::to_string(snap_gib) +
                                                 " GiB, only " + std::to_string((double)free_b / (1ull << 30)) +
                                                 " GiB free");
                    fprintf(stderr, "[cf] verify block: spec snapshots (GDN S/conv + PLE + pending_h) %.2f GiB\n",
                            snap_gib);
                    v->s_snap = dalloc<float>((size_t)v->ngdn * k * HV * DK * DK, false);
                    v->conv_snap = dalloc<float>((size_t)v->ngdn * k * CONV * 3, false);
                    v->ple_snap = dalloc<float>((size_t)k * PLE_HIST * HCD, false);
                }
                {
                    const double slab_gib = nr * (double)TOPK * 2.0 * EE * 840.0 / (1ull << 30) +
                                            nr * (double)TOPK * (double)D * 360.0 / (1ull << 30);
                    size_t free_b = 0, total_b = 0;
                    CK(cudaMemGetInfo(&free_b, &total_b));
                    if ((double)free_b < slab_gib * (1ull << 30) * 1.02)
                        throw std::runtime_error("the verify union slabs need " + std::to_string(slab_gib) +
                                                 " GiB, only " + std::to_string((double)free_b / (1ull << 30)) +
                                                 " GiB free");
                    fprintf(stderr, "[cf] verify block: nr=%d, union slabs (worst case) %.2f GiB\n", nr, slab_gib);
                }
                alloc_packed(v->uni_gu, 0, fmt_for(GT_Q2_K), (int64_t)nr * TOPK * 2 * EE, D);
                alloc_packed(v->uni_dn, 0, fmt_for(GT_Q4_0), (int64_t)nr * TOPK * D, EE);
                const double v_dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                fprintf(stderr, "[cf] verify block loaded (nr=%d) %.1f s\n", nr, v_dur);
            }
        }
        CK(cudaMallocHost(&c->h_router, (size_t)NE * 4));
        CK(cudaMallocHost(&c->we_h, (size_t)TOPK * 4));
        // r19v: eid pinned (the UVA path's per-step eid H2D rides it; a captured memcpy node
        // must read pinned host memory - a pageable source is not capture-legal)
        CK(cudaMallocHost(&c->eid, (size_t)TOPK * 4));
        // r19v: the step params - pos rides the pinned word, uploaded at every step head
        CK(cudaMallocHost(&c->h_params, 4 * sizeof(int)));
        memset(c->h_params, 0, 4 * sizeof(int));
        c->d_params = dalloc<int>(4, false);
        CK(cudaMallocHost(&c->h_emb, (size_t)D * 4));
        CK(cudaMallocHost(&c->h_ple, (size_t)D * 4));
        CK(cudaMallocHost(&c->h_logits, (size_t)V * 4));
        c->toks = new int[c->max_ctx];
        memset(c->toks, -1, (size_t)c->max_ctx * sizeof(int));

        // cf-m3 (r19v) the G1 segment graphs: T4Q_CF_GRAPH=1 captures the 49 sync-bounded
        // segments at the first step and replays them (the launch wall -> ~1 replay per
        // segment). Requires the capture-constant form: NOT tiered (the miss-count-varying
        // H2D sizes are not capture-constant; the census verdict says the tiering pays
        // ~nothing at this routing entropy anyway) and NOT dumping (the T4Q_CF_DUMP mid-step
        // D2H probes are not capture-legal). The OFF/UVA moe branches are per-layer constant,
        // pos rides the device step-params word, and every varying memcpy content rides a
        // pinned-fixed host source the captured H2D nodes re-carry at each replay.
        if (const char* gv = getenv("T4Q_CF_GRAPH")) c->gmode = atoi(gv) ? 1 : 0;
        if (c->gmode && c->tiered) {
            c->gmode = 0;
            fprintf(stderr, "[cf] graph OFF: the tiered moe's miss-varying H2D sizes are not capture-constant\n");
        }
        if (c->gmode && getenv("T4Q_CF_DUMP")) {
            c->gmode = 0;
            fprintf(stderr, "[cf] graph OFF: the T4Q_CF_DUMP mid-step probes are not capture-legal\n");
        }
        if (c->gmode) fprintf(stderr, "[cf] graph ON: %d segment graphs, captured at the first step\n", NL + 1);
        // cf-m4 (r19aa): the PLE prefetch (CF_MTP.md section 8) - the spec rounds' verify
        // gathers warm their table pages under the draft calls; OFF absent the env (the
        // L4 A/B decides: the warm-cache class pays the spawn overhead for nothing)
        if (const char* pp = getenv("T4Q_CF_PLE_PRE")) c->ple_pre = atoi(pp) ? 1 : 0;
        if (c->ple_pre) fprintf(stderr, "[cf] PLE prefetch ON: the verify's rows warm under the draft steps\n");

        CK(cudaFree(dev));
        CK(cudaFreeHost(pin));
        if (rs.mismatched > 0)
            throw std::runtime_error("repack check failed: " + std::to_string(rs.mismatched) + "/" +
                                     std::to_string(rs.checked) + " rows, first " + rs.first_bad);
        double s1 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[cf] load done %.1f s; repack check %d rows, 0 mismatched\n", s1, rs.checked);
        return up.release();
    } catch (const std::exception& e) {
        if (err_out) *err_out = e.what();
        return nullptr;
    }
}
