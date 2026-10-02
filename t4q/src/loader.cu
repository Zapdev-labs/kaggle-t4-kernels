// GGUF -> GPU loader: uploads raw blocks through pinned staging, repacks on the GPU, then checks a sample of rows
// bit-exactly against the CPU port of ggml's dequant.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "kernels/kernels.h"
#include "model.h"
#include "quant_cpu.h"

namespace {

constexpr size_t STAGE = 64ull << 20;

struct Stager {
    uint8_t* pin = nullptr;
    uint8_t* dev[2] = {nullptr, nullptr};
};

size_t al256(size_t x) { return (x + 255) & ~(size_t)255; }

int fmt_for(uint32_t t) {
    switch (t) {
        case GT_F32: return FMT_F32;
        case GT_Q4_0: return FMT_P4;
        case GT_Q4_1: return FMT_P4M;
        case GT_Q8_0: return FMT_Q8;
        case GT_Q5_K: return FMT_K5;
        case GT_Q6_K: return FMT_K6;
        default: return -1;
    }
}

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

void check_rows(t4q_ctx* c, const GgufTensor* t, const PackedW& W, cudaStream_t st) {
    const int nchk = (int)std::min<int64_t>(W.rows, 64);
    uint64_t seed = 0x1234567ull ^ (uint64_t)W.rows * 31 ^ (uint64_t)W.cols;
    std::vector<int32_t> rows(nchk);
    for (int i = 0; i < nchk; i++) rows[i] = (i == 0) ? 0 : (i == 1 ? (int32_t)(W.rows - 1) : (int32_t)(lcg(seed) % W.rows));
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
        dequant_row_cpu(t->type, t->data + (size_t)rows[i] * t->row_bytes, ref.data(), W.cols);
        const float* gr = g.data() + (size_t)i * W.cols;
        c->rstats.checked++;
        if (memcmp(gr, ref.data(), W.cols * 4) != 0) {
            double md = 0;
            for (int64_t j = 0; j < W.cols; j++) md = std::max(md, (double)std::fabs(gr[j] - ref[j]));
            c->rstats.mismatched++;
            c->rstats.max_abs_diff = std::max(c->rstats.max_abs_diff, md);
            if (c->rstats.first_bad.empty()) c->rstats.first_bad = t->name + " row " + std::to_string(rows[i]);
        }
    }
}

void upload_matrix(t4q_ctx* c, Stager& sg, const std::string& name, int gpu, PackedW& W) {
    const GgufTensor* t = c->f.find(name);
    if (!t) throw std::runtime_error("missing tensor " + name);
    const int fmt = fmt_for(t->type);
    if (fmt < 0) throw std::runtime_error(std::string("unsupported type ") + ggml_type_name(t->type) + " for " + name);
    alloc_packed(W, gpu, fmt, t->nrows(), t->ne[0]);
    cudaStream_t st = c->st[gpu];
    const int64_t chunk = std::max<int64_t>(1, (int64_t)(STAGE / t->row_bytes));
    for (int64_t r0 = 0; r0 < W.rows; r0 += chunk) {
        const int64_t nr = std::min(chunk, W.rows - r0);
        const size_t bytes = (size_t)nr * t->row_bytes;
        CK(cudaStreamSynchronize(st));  // staging reuse
        memcpy(sg.pin, t->data + (size_t)r0 * t->row_bytes, bytes);
        CK(cudaMemcpyAsync(sg.dev[gpu], sg.pin, bytes, cudaMemcpyHostToDevice, st));
        launch_repack(W, t->type, sg.dev[gpu], r0, nr, st);
        CK(cudaGetLastError());
    }
    CK(cudaStreamSynchronize(st));
    c->rstats.tensors++;
    check_rows(c, t, W, st);
}

float* upload_vec(t4q_ctx* c, const std::string& name, int gpu, int64_t expect) {
    const GgufTensor* t = c->f.find(name);
    if (!t) throw std::runtime_error("missing tensor " + name);
    if (t->type != GT_F32) throw std::runtime_error("expected F32 for " + name);
    const int64_t n = t->ne[0] * t->nrows();
    if (expect > 0 && n != expect) throw std::runtime_error("bad size for " + name);
    float* p;
    CK(cudaSetDevice(gpu));
    CK(cudaMalloc(&p, n * 4));
    CK(cudaMemcpy(p, t->data, n * 4, cudaMemcpyHostToDevice));
    return p;
}

template <class T>
T* dalloc(size_t n, bool zero = true) {
    T* p;
    CK(cudaMalloc(&p, n * sizeof(T)));
    if (zero) CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

void alloc_scratch(Scratch& s, bool head) {
    using namespace hp;
    s.h = dalloc<float>(D); s.xn = dalloc<float>(D); s.a = dalloc<float>(D);
    s.qkv = dalloc<float>(CONV); s.z = dalloc<float>(VDIM); s.braw = dalloc<float>(64); s.araw = dalloc<float>(64);
    s.beta = dalloc<float>(64); s.g = dalloc<float>(64); s.conv = dalloc<float>(CONV);
    s.qn = dalloc<float>(HK * DK); s.kn = dalloc<float>(HK * DK); s.o = dalloc<float>(VDIM); s.on = dalloc<float>(VDIM);
    s.qfull = dalloc<float>(HQ * HD * 2); s.k = dalloc<float>(HKV * HD); s.v = dalloc<float>(HKV * HD);
    s.aq = dalloc<float>(HQ * HD); s.ak = dalloc<float>(HKV * HD); s.att = dalloc<float>(HQ * HD);
    s.attg = dalloc<float>(HQ * HD);
    s.ffg = dalloc<float>(FF); s.ffu = dalloc<float>(FF); s.ffa = dalloc<float>(FF);
    s.scores = nullptr;
    s.xq = dalloc<int8_t>(FF); s.xd = dalloc<float>(FF / 32); s.xs = dalloc<float>(FF / 32);
    s.logits = head ? dalloc<float>(V) : nullptr;
}

}  // namespace

void load_model(t4q_ctx* c, const char* path) {
    using namespace hp;
    auto t0 = std::chrono::steady_clock::now();
    std::string err;
    if (!c->f.open(path, err)) throw std::runtime_error("gguf: " + err);
    // hyperparameter asserts (arch.md section 0)
    auto need = [&](const char* k, double v) {
        double got = c->f.num(k, -1);
        if (std::fabs(got - v) > 1e-3 * std::max(1.0, std::fabs(v)))
            throw std::runtime_error(std::string("hparam mismatch ") + k + " got " + std::to_string(got));
    };
    need("qwen35.embedding_length", D); need("qwen35.block_count", 65); need("qwen35.feed_forward_length", FF);
    need("qwen35.attention.head_count", HQ); need("qwen35.attention.head_count_kv", HKV);
    need("qwen35.attention.key_length", HD); need("qwen35.rope.dimension_count", NROT);
    need("qwen35.rope.freq_base", ROPE_BASE); need("qwen35.ssm.group_count", HK); need("qwen35.ssm.time_step_rank", HV);
    need("qwen35.ssm.state_size", DK); need("qwen35.ssm.conv_kernel", 4); need("qwen35.full_attention_interval", 4);
    c->tok_embd = c->f.find("token_embd.weight");
    if (!c->tok_embd || c->tok_embd->type != GT_Q4_0 && c->tok_embd->type != GT_F32 && c->tok_embd->type != GT_Q8_0 &&
                            c->tok_embd->type != GT_Q4_1 && c->tok_embd->type != GT_Q6_K && c->tok_embd->type != GT_Q5_K)
        throw std::runtime_error("token_embd missing or unsupported type");

    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (ndev < 2) throw std::runtime_error("need 2 GPUs");
    c->max_ctx = c->params.max_ctx > 0 ? c->params.max_ctx : 4096;
    for (int g = 0; g < 2; g++) {
        CK(cudaSetDevice(g));
        CK(cudaStreamCreateWithFlags(&c->st[g], cudaStreamNonBlocking));
        size_t fr, tot;
        CK(cudaMemGetInfo(&fr, &tot));
        c->vram_used[g] = tot - fr;
    }
    Stager sg;
    CK(cudaMallocHost(&sg.pin, STAGE));
    for (int g = 0; g < 2; g++) { CK(cudaSetDevice(g)); CK(cudaMalloc(&sg.dev[g], STAGE)); }

    for (int il = 0; il < NL; il++) {
        Layer& L = c->layers[il];
        L.il = il;
        L.gpu = il < c->split ? 0 : 1;
        L.attn = is_attn(il);
        const int g = L.gpu;
        const std::string p = "blk." + std::to_string(il) + ".";
        L.attn_norm = upload_vec(c, p + "attn_norm.weight", g, D);
        L.post_norm = upload_vec(c, p + "post_attention_norm.weight", g, D);
        if (L.attn) {
            upload_matrix(c, sg, p + "attn_q.weight", g, L.wq);
            upload_matrix(c, sg, p + "attn_k.weight", g, L.wk);
            upload_matrix(c, sg, p + "attn_v.weight", g, L.wv);
            upload_matrix(c, sg, p + "attn_output.weight", g, L.wo);
            L.q_norm = upload_vec(c, p + "attn_q_norm.weight", g, HD);
            L.k_norm = upload_vec(c, p + "attn_k_norm.weight", g, HD);
            CK(cudaSetDevice(g));
            L.kc = dalloc<uint16_t>((size_t)HKV * c->max_ctx * HD);
            L.vc = dalloc<uint16_t>((size_t)HKV * c->max_ctx * HD);
        } else {
            upload_matrix(c, sg, p + "attn_qkv.weight", g, L.qkv);
            upload_matrix(c, sg, p + "attn_gate.weight", g, L.z);
            upload_matrix(c, sg, p + "ssm_alpha.weight", g, L.alpha);
            upload_matrix(c, sg, p + "ssm_beta.weight", g, L.beta);
            upload_matrix(c, sg, p + "ssm_out.weight", g, L.ssm_out);
            L.conv_w = upload_vec(c, p + "ssm_conv1d.weight", g, 4 * CONV);
            L.ssm_a = upload_vec(c, p + "ssm_a", g, HV);
            L.ssm_dt = upload_vec(c, p + "ssm_dt.bias", g, HV);
            L.ssm_norm = upload_vec(c, p + "ssm_norm.weight", g, DK);
            CK(cudaSetDevice(g));
            L.conv_state = dalloc<float>((size_t)CONV * 3);
            L.S = dalloc<float>((size_t)HV * DK * DK);
        }
        upload_matrix(c, sg, p + "ffn_gate.weight", g, L.gate);
        upload_matrix(c, sg, p + "ffn_up.weight", g, L.up);
        upload_matrix(c, sg, p + "ffn_down.weight", g, L.down);
        if (c->params.verbose && il % 8 == 7) {
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "[t4q] loaded layers 0..%d (%.1f s)\n", il, s);
        }
    }
    c->output_norm = upload_vec(c, "output_norm.weight", 1, D);
    upload_matrix(c, sg, "output.weight", 1, c->output);

    for (int g = 0; g < 2; g++) {
        CK(cudaSetDevice(g));
        alloc_scratch(c->sc[g], g == 1);
        c->sc[g].scores = dalloc<float>((size_t)HQ * c->max_ctx);
        CK(cudaFree(sg.dev[g]));
    }
    CK(cudaFreeHost(sg.pin));
    CK(cudaMallocHost(&c->h_emb, D * 4));
    CK(cudaMallocHost(&c->h_logits, (size_t)V * 4));
    for (int g = 0; g < 2; g++) {
        CK(cudaSetDevice(g));
        size_t fr, tot;
        CK(cudaMemGetInfo(&fr, &tot));
        c->vram_used[g] = tot - fr;
    }
    c->load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (c->params.verbose)
        fprintf(stderr, "[t4q] load done %.1f s; repack check %d rows, %d mismatched (max diff %g) %s\n", c->load_s,
                c->rstats.checked, c->rstats.mismatched, c->rstats.max_abs_diff, c->rstats.first_bad.c_str());
}

void free_model(t4q_ctx* c) {
    // process exit frees device memory; explicit teardown kept minimal for M1
    for (int g = 0; g < 2; g++)
        if (c->st[g]) { cudaSetDevice(g); cudaStreamSynchronize(c->st[g]); cudaStreamDestroy(c->st[g]); }
    if (c->h_emb) cudaFreeHost(c->h_emb);
    if (c->h_logits) cudaFreeHost(c->h_logits);
    for (int g = 0; g < 2; g++) { cudaSetDevice(g); cudaDeviceReset(); }
}
