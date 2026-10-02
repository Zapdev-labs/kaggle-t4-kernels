// oracle_dump: llama.cpp (a4cb4c61, sm_75 libllama from otdoges/t4-qwen38-baseline) reference for t4q validation.
//
// usage: oracle_dump <model.gguf> <jobs.txt> <outdir> [layer|tensor]
// jobs.txt lines (token-id files are raw little-endian int32):
//   seq  <name> <ids.i32> <tbt_last>   -> <name>.batch.f32 (n x V logits, one batch), <name>.tbt.f32 (last tbt_last
//                                         positions decoded one token at a time after a batch prefix)
//   gen  <name> <ids.i32> <n_gen>      -> <name>.gen.i32 (greedy tokens), <name>.gen.txt (tok top1 top2 gap)
//   dump <name> <ids.i32> <layers>     -> <name>.dump.bin: named intermediates (cb_eval) for layers (comma list),
//                                         token-by-token, captured at positions 0, 1 and n-1
//   tok  <name> <text.txt> <ids.i32>   -> compares llama_tokenize(text, parse_special) with the id file
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

static std::vector<int32_t> read_ids(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<int32_t> v;
    int32_t x;
    while (f.read((char*)&x, 4)) v.push_back(x);
    return v;
}

static void write_f32(const std::string& p, const std::vector<float>& v) {
    FILE* f = fopen(p.c_str(), "wb");
    fwrite(v.data(), 4, v.size(), f);
    fclose(f);
}

struct Rec { std::string name; int tag; int64_t ne[4]; std::vector<float> data; };
static int g_tag = -1;
static std::set<int> g_layers;
static std::map<std::pair<std::string, int>, Rec> g_recs;

static bool wanted(const char* name) {
    std::string n(name);
    if (n == "result_norm" || n == "result_output" || n == "h_nextn" || n == "model.input_embed") return true;
    size_t d = n.rfind('-');
    if (d == std::string::npos || d + 1 >= n.size()) return false;
    for (size_t i = d + 1; i < n.size(); i++) if (n[i] < '0' || n[i] > '9') return false;
    return g_layers.count(atoi(n.c_str() + d + 1)) > 0;
}

static bool cb_eval(struct ggml_tensor* t, bool ask, void*) {
    if (g_tag < 0) return false;
    if (ask) return wanted(t->name) && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) && ggml_nelements(t) <= 2000000;
    if (!wanted(t->name)) return true;
    const size_t nb = ggml_nbytes(t);
    std::vector<uint8_t> raw(nb);
    ggml_backend_tensor_get(t, raw.data(), 0, nb);
    Rec r;
    r.name = t->name;
    r.tag = g_tag;
    for (int i = 0; i < 4; i++) r.ne[i] = t->ne[i];
    r.data.resize(ggml_nelements(t));
    size_t k = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++)
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++)
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++)
                for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                    const size_t off = i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                    float v;
                    if (t->type == GGML_TYPE_F32) memcpy(&v, raw.data() + off, 4);
                    else v = ggml_fp16_to_fp32(*(const ggml_fp16_t*)(raw.data() + off));
                    r.data[k++] = v;
                }
    g_recs[{r.name, r.tag}] = std::move(r);
    return true;
}

static int decode(llama_context* ctx, const std::vector<int32_t>& toks, int pos0, bool all_logits) {
    const int n = (int)toks.size();
    llama_batch b = llama_batch_init(n, 0, 1);
    b.n_tokens = n;
    for (int i = 0; i < n; i++) {
        b.token[i] = toks[i];
        b.pos[i] = pos0 + i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = all_logits || i == n - 1;
    }
    int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc;
}

static void clear(llama_context* ctx) { llama_memory_clear(llama_get_memory(ctx), true); }

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model jobs outdir [layer|tensor]\n", argv[0]); return 2; }
    const std::string out = argv[3];
    const bool tensor = argc > 4 && std::string(argv[4]) == "tensor";
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.split_mode = tensor ? LLAMA_SPLIT_MODE_TENSOR : LLAMA_SPLIT_MODE_LAYER;
    auto t0 = std::chrono::steady_clock::now();
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "ORACLE_ERROR model load failed\n"); return 1; }
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = nullptr;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ORACLE_ERROR context init failed\n"); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int V = llama_vocab_n_tokens(vocab);
    printf("ORACLE load_s=%.1f n_vocab=%d split=%s\n",
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), V, tensor ? "tensor" : "layer");
    fflush(stdout);

    std::ifstream jf(argv[2]);
    std::string line;
    while (std::getline(jf, line)) {
        std::istringstream ss(line);
        std::string kind, name, file, arg;
        ss >> kind >> name >> file >> arg;
        if (kind.empty() || kind[0] == '#') continue;
        auto tj = std::chrono::steady_clock::now();
        if (kind == "seq") {
            std::vector<int32_t> ids = read_ids(file);
            const int n = (int)ids.size(), T = std::min(atoi(arg.c_str()), n - 1);
            clear(ctx);
            if (n > 512) { fprintf(stderr, "ORACLE_ERROR seq > 512\n"); return 1; }
            if (decode(ctx, ids, 0, true)) { fprintf(stderr, "ORACLE_ERROR decode failed (%s)\n", name.c_str()); return 1; }
            std::vector<float> lg((size_t)n * V);
            for (int i = 0; i < n; i++) memcpy(lg.data() + (size_t)i * V, llama_get_logits_ith(ctx, i), V * 4);
            write_f32(out + "/" + name + ".batch.f32", lg);
            clear(ctx);
            std::vector<int32_t> pre(ids.begin(), ids.end() - T);
            if (decode(ctx, pre, 0, false)) { fprintf(stderr, "ORACLE_ERROR decode prefix failed\n"); return 1; }
            std::vector<float> tb((size_t)T * V);
            for (int i = 0; i < T; i++) {
                const int p = n - T + i;
                if (decode(ctx, {ids[p]}, p, true)) { fprintf(stderr, "ORACLE_ERROR decode tbt failed\n"); return 1; }
                memcpy(tb.data() + (size_t)i * V, llama_get_logits_ith(ctx, 0), V * 4);
            }
            write_f32(out + "/" + name + ".tbt.f32", tb);
        } else if (kind == "gen") {
            std::vector<int32_t> ids = read_ids(file);
            const int ng = atoi(arg.c_str());
            clear(ctx);
            if (decode(ctx, ids, 0, false)) { fprintf(stderr, "ORACLE_ERROR decode failed\n"); return 1; }
            std::vector<int32_t> gen;
            FILE* tf = fopen((out + "/" + name + ".gen.txt").c_str(), "w");
            int pos = (int)ids.size();
            for (int i = 0; i < ng; i++) {
                const float* lg = llama_get_logits_ith(ctx, -1);
                int b1 = 0, b2 = -1;
                for (int j = 1; j < V; j++) {
                    if (lg[j] > lg[b1]) { b2 = b1; b1 = j; }
                    else if (b2 < 0 || lg[j] > lg[b2]) b2 = j;
                }
                fprintf(tf, "%d %.6f %.6f %.6f\n", b1, lg[b1], lg[b2], lg[b1] - lg[b2]);
                gen.push_back(b1);
                if (i + 1 < ng) {
                    if (decode(ctx, {b1}, pos++, true)) { fprintf(stderr, "ORACLE_ERROR gen decode failed\n"); return 1; }
                }
            }
            fclose(tf);
            FILE* f = fopen((out + "/" + name + ".gen.i32").c_str(), "wb");
            fwrite(gen.data(), 4, gen.size(), f);
            fclose(f);
        } else if (kind == "dump") {
            std::vector<int32_t> ids = read_ids(file);
            g_layers.clear();
            std::stringstream ls(arg);
            std::string x;
            while (std::getline(ls, x, ',')) g_layers.insert(atoi(x.c_str()));
            g_recs.clear();
            clear(ctx);
            const int n = (int)ids.size();
            for (int i = 0; i < n; i++) {
                g_tag = (i == 0 || i == 1 || i == n - 1) ? i : -1;
                if (decode(ctx, {ids[i]}, i, true)) { fprintf(stderr, "ORACLE_ERROR dump decode failed\n"); return 1; }
            }
            g_tag = -1;
            FILE* f = fopen((out + "/" + name + ".dump.bin").c_str(), "wb");
            uint32_t cnt = (uint32_t)g_recs.size();
            fwrite("T4QD", 1, 4, f);
            fwrite(&cnt, 4, 1, f);
            for (auto& kv : g_recs) {
                const Rec& r = kv.second;
                uint32_t nl = (uint32_t)r.name.size();
                fwrite(&nl, 4, 1, f);
                fwrite(r.name.data(), 1, nl, f);
                fwrite(&r.tag, 4, 1, f);
                fwrite(r.ne, 8, 4, f);
                fwrite(r.data.data(), 4, r.data.size(), f);
            }
            fclose(f);
            printf("ORACLE dump %s records=%u\n", name.c_str(), cnt);
        } else if (kind == "tok") {
            std::ifstream tf(file, std::ios::binary);
            std::string text((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>());
            std::vector<int32_t> ref = read_ids(arg);
            std::vector<llama_token> toks(text.size() + 16);
            int nt = llama_tokenize(vocab, text.data(), (int32_t)text.size(), toks.data(), (int32_t)toks.size(), false, true);
            toks.resize(nt > 0 ? nt : 0);
            int first_diff = -1;
            for (size_t i = 0; i < std::max(toks.size(), ref.size()); i++) {
                if (i >= toks.size() || i >= ref.size() || toks[i] != ref[i]) { first_diff = (int)i; break; }
            }
            printf("ORACLE tok %s llama_n=%zu hf_n=%zu match=%d first_diff=%d\n", name.c_str(), toks.size(), ref.size(),
                   first_diff < 0, first_diff);
        }
        printf("ORACLE job %s %s done %.1f s\n", kind.c_str(), name.c_str(),
               std::chrono::duration<double>(std::chrono::steady_clock::now() - tj).count());
        fflush(stdout);
    }
    llama_free(ctx);
    llama_model_free(model);
    printf("ORACLE_OK\n");
    return 0;
}
