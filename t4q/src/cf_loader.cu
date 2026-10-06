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
        CK(cudaMallocHost(&c->h_router, (size_t)NE * 4));
        CK(cudaMallocHost(&c->we_h, (size_t)TOPK * 4));
        c->eid = new int[TOPK];
        CK(cudaMallocHost(&c->h_emb, (size_t)D * 4));
        CK(cudaMallocHost(&c->h_ple, (size_t)D * 4));
        CK(cudaMallocHost(&c->h_logits, (size_t)V * 4));
        c->toks = new int[c->max_ctx];
        memset(c->toks, -1, (size_t)c->max_ctx * sizeof(int));

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
