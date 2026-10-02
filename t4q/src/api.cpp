// C ABI (include/t4q.h).
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

#include "model.h"
#include "tp_api.h"

static thread_local std::string g_err;

#define API_TRY try {
#define API_CATCH(ret)                                   \
    }                                                    \
    catch (const std::exception& e) {                    \
        g_err = e.what();                                \
        fprintf(stderr, "[t4q] error: %s\n", e.what()); \
        return ret;                                      \
    }

extern "C" {

const char* t4q_last_error(void) { return g_err.c_str(); }

t4q_ctx* t4q_load(const char* gguf, const t4q_params* p) {
    t4q_ctx* c = new t4q_ctx();
    try {
        if (p) c->params = *p;
        if (c->params.tp) tp_load(c, gguf);
        else load_model(c, gguf);
        return c;
    } catch (const std::exception& e) {
        g_err = e.what();
        fprintf(stderr, "[t4q] load error: %s\n", e.what());
        delete c;
        return nullptr;
    }
}

int t4q_n_vocab(t4q_ctx*) { return hp::V; }
int t4q_pos(t4q_ctx* c) { return c->pos; }

int t4q_logits(t4q_ctx* c, const int32_t* ids, int n, float* out) {
    API_TRY
    if (c->tps) return tp_logits(c, ids, n, out);
    for (int i = 0; i < n; i++) {
        engine_step(c, ids[i]);
        if (out) memcpy(out + (size_t)i * hp::V, c->h_logits, (size_t)hp::V * 4);
    }
    return 0;
    API_CATCH(-1)
}

int t4q_last_logits(t4q_ctx* c, float* out) {
    API_TRY
    if (c->tps) return tp_last_logits(c, out);
    memcpy(out, c->h_logits, (size_t)hp::V * 4);
    return 0;
    API_CATCH(-1)
}

int t4q_prefill(t4q_ctx* c, const int32_t* ids, int n) {
    if (c->tps) {
        API_TRY
        return tp_prefill(c, ids, n);
        API_CATCH(-1)
    }
    return t4q_logits(c, ids, n, nullptr);
}

static int argmax_host(const float* x, int n) {
    int bi = 0;
    float bv = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > bv) { bv = x[i]; bi = i; }
    return bi;
}

int t4q_generate(t4q_ctx* c, int32_t* out, int max_new, const t4q_sampling* smp, const int32_t* stop, int n_stop) {
    API_TRY
    (void)smp;  // greedy only
    if (c->tps) return tp_generate(c, out, max_new, stop, n_stop);
    if (!c->have_logits) throw std::runtime_error("generate needs a prefill first");
    auto t0 = std::chrono::steady_clock::now();
    int n = 0;
    for (; n < max_new; n++) {
        const int tok = argmax_host(c->h_logits, hp::V);
        out[n] = tok;
        bool is_stop = false;
        for (int j = 0; j < n_stop; j++) is_stop |= (stop[j] == tok);
        if (is_stop) { n++; break; }
        if (n + 1 < max_new) engine_step(c, tok);
    }
    c->gen_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    c->gen_tokens += n;
    return n;
    API_CATCH(-1)
}

int t4q_layer_forward(t4q_ctx* c, int il, int pos, const float* h_in, float* h_out) {
    API_TRY
    if (c->tps) throw std::runtime_error("layer_forward is not available in the TP engine");
    engine_layer(c, il, pos, h_in, h_out);
    return 0;
    API_CATCH(-1)
}

int t4q_set_option(t4q_ctx* c, const char* key, int value) {
    const std::string k(key);
    if (k == "act_q8") { c->act_q8 = value != 0; return 0; }
    if (c->tps) {
        try {
            if (tp_set_option(c, k, value) == 0) return 0;
        } catch (const std::exception& e) {
            g_err = e.what();
            fprintf(stderr, "[t4q] error: %s\n", e.what());
            return -1;
        }
    }
    g_err = "unknown option " + k;
    return -1;
}

void t4q_set_dump(t4q_ctx* c, int on) { c->dump_on = on != 0; if (!on) c->dumps.clear(); }

int t4q_dump(t4q_ctx* c, const char* name, int layer, float* out, size_t cap) {
    std::string key = layer >= 0 ? std::string(name) + "-" + std::to_string(layer) : std::string(name);
    auto it = c->dumps.find(key);
    if (it == c->dumps.end()) return -1;
    const size_t n = it->second.size();
    if (out) memcpy(out, it->second.data(), std::min(n, cap) * 4);
    return (int)n;
}

int t4q_dump_keys(t4q_ctx* c, char* buf, int cap) {
    std::string s;
    for (auto& kv : c->dumps) s += kv.first + "\n";
    if (buf && cap > 0) {
        const int n = std::min<int>((int)s.size(), cap - 1);
        memcpy(buf, s.data(), n);
        buf[n] = 0;
    }
    return (int)s.size();
}

void t4q_stats(t4q_ctx* c, char* json, int cap) {
    char tmp[4096];
    snprintf(tmp, sizeof tmp,
             "{\"load_s\": %.2f, \"steps\": %ld, \"step_ms\": %.3f, \"gen_tokens\": %ld, \"gen_tok_s\": %.3f, "
             "\"repack_tensors\": %d, \"repack_rows_checked\": %d, \"repack_rows_mismatched\": %d, "
             "\"repack_max_abs_diff\": %g, \"repack_first_bad\": \"%s\", \"vram_used_mib\": [%.0f, %.0f], "
             "\"max_ctx\": %d, \"split\": %d, \"act_q8\": %d}",
             c->load_s, c->steps, c->steps ? 1e3 * c->step_s / c->steps : 0.0, c->gen_tokens,
             c->gen_s > 0 ? c->gen_tokens / c->gen_s : 0.0, c->rstats.tensors, c->rstats.checked,
             c->rstats.mismatched, c->rstats.max_abs_diff, c->rstats.first_bad.c_str(),
             c->vram_used[0] / 1048576.0, c->vram_used[1] / 1048576.0, c->max_ctx, c->split, (int)c->act_q8);
    std::string s(tmp);
    if (c->tps) {
        s.pop_back();
        try { s += tp_stats_json(c); } catch (...) {}
        s += "}";
    }
    if (json && cap > 0) {
        strncpy(json, s.c_str(), cap - 1);
        json[cap - 1] = 0;
    }
}

void t4q_reset(t4q_ctx* c) {
    try { if (c->tps) tp_reset(c); else engine_reset(c); } catch (const std::exception& e) { g_err = e.what(); }
}

void t4q_free(t4q_ctx* c) {
    if (!c) return;
    free_model(c);
    delete c;
}

}  // extern "C"
