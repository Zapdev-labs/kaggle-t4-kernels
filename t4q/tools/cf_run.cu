// cf-m1: the CYBER-FROST engine driver. Modes:
//   seq  <model> <ids.i32> <oracle.tbt.f32> <n_last>  -> step every id, compare the per-position logits with the
//                                                          oracle's token-by-token dump (max rel diff, top1 agree)
//   gen  <model> <ids.i32> <n> <oracle.gen.i32>        -> feed the ids, then n greedy tokens (byte-compare)
//   time <model> <ids.i32> <n>                        -> the timed run: prompt + n greedy, tok/s + step stats
// Prints CF {json} summary lines; exits 0 on pass, 3 on mismatch.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cf_model.h"

static std::vector<int32_t> read_i32(const char* p) {
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "CF_ERROR open %s\n", p); exit(2); }
    std::vector<int32_t> v;
    int32_t x;
    while (fread(&x, 4, 1, f) == 1) v.push_back(x);
    fclose(f);
    return v;
}

static std::vector<float> read_f32(const char* p, size_t expect) {
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "CF_ERROR open %s\n", p); exit(2); }
    std::vector<float> v(expect);
    if (fread(v.data(), 4, expect, f) != expect) { fprintf(stderr, "CF_ERROR short read %s\n", p); exit(2); }
    fclose(f);
    return v;
}

static int argmax(const float* x, int n) {
    int b = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[b]) b = i;
    return b;
}

int main(int argc, char** argv) {
    setbuf(stdout, nullptr);
    if (argc < 4) {
        fprintf(stderr, "usage: %s seq|gen|time <model> <ids.i32> [args]\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    const char* model = argv[2];
    std::vector<int32_t> ids = read_i32(argv[3]);
    fprintf(stderr, "cf_run: mode %s, %zu prompt tokens\n", mode.c_str(), ids.size());

    std::string err;
    auto t0 = std::chrono::steady_clock::now();
    CfCtx* c = cf_load(model, 4096, &err);
    if (!c) { printf("CF {\"error\":\"load: %s\"}\n", err.c_str()); return 1; }
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("CF {\"load_s\":%.1f}\n", load_s);

    if (mode == "seq") {
        // the oracle's .tbt.f32 covers the last T positions: feed the whole prompt, compare the tail
        const int n = (int)ids.size();
        const int T = argc > 5 ? atoi(argv[5]) : n;
        std::vector<float> oracle = read_f32(argv[4], (size_t)T * cf::V);
        double worst_rel = 0, worst_abs = 0;
        int worst_pos = -1, agree = 0, first_top1_diff = -1;
        std::vector<float> got((size_t)T * cf::V);
        for (int i = 0; i < n; i++) {
            if (!cf_step(c, ids[i])) { printf("CF {\"error\":\"step %d: %s\"}\n", i, c->err.c_str()); return 1; }
            if (i >= n - T) memcpy(got.data() + (size_t)(i - (n - T)) * cf::V, c->h_logits, (size_t)cf::V * 4);
        }
        for (int i = 0; i < T; i++) {
            const float* mine = got.data() + (size_t)i * cf::V;
            const float* ref = oracle.data() + (size_t)i * cf::V;
            for (int j = 0; j < cf::V; j++) {
                const double a = mine[j], b = ref[j];
                const double ad = std::fabs(a - b);
                if (ad > worst_abs) worst_abs = ad;
                const double rel = ad / (std::fabs(b) + 1e-6);
                if (rel > worst_rel) { worst_rel = rel; worst_pos = i; }
            }
            const int t1 = argmax(mine, cf::V), r1 = argmax(ref, cf::V);
            if (t1 == r1) agree++;
            else if (first_top1_diff < 0) first_top1_diff = i;
        }
        const bool pass = worst_rel < 1e-3 && agree == T;
        printf("CF {\"mode\":\"seq\",\"n\":%d,\"T\":%d,\"worst_rel\":%.3e,\"worst_abs\":%.3e,\"worst_pos\":%d,"
               "\"top1_agree\":%d,\"first_top1_diff\":%d,\"mean_ms\":%.2f,\"pass\":%s}\n",
               n, T, worst_rel, worst_abs, worst_pos, agree, first_top1_diff,
               c->step_s * 1000 / std::max(1, c->steps), pass ? "true" : "false");
        cf_free(c);
        return pass ? 0 : 3;
    }

    if (mode == "gen") {
        const int n = atoi(argv[4]);
        std::vector<int32_t> oracle = argc > 5 ? read_i32(argv[5]) : std::vector<int32_t>();
        for (int t : ids)
            if (!cf_step(c, t)) { printf("CF {\"error\":\"prompt step: %s\"}\n", c->err.c_str()); return 1; }
        std::vector<int32_t> gen;
        int first_diff = -1;
        for (int i = 0; i < n; i++) {
            int tok = argmax(c->h_logits, cf::V);
            gen.push_back(tok);
            if (!oracle.empty() && (size_t)i >= oracle.size()) break;
            if (!oracle.empty() && tok != oracle[i] && first_diff < 0) first_diff = i;
            if (i + 1 < n && !cf_step(c, tok)) {
                printf("CF {\"error\":\"gen step %d: %s\"}\n", i, c->err.c_str());
                return 1;
            }
        }
        int match = 0;
        for (size_t i = 0; i < std::min(gen.size(), oracle.size()); i++) match += gen[i] == oracle[i];
        const bool pass = oracle.empty() ? true : (first_diff < 0 && gen.size() == oracle.size());
        printf("CF {\"mode\":\"gen\",\"n\":%d,\"match\":%d,\"oracle_n\":%zu,\"first_diff\":%d,"
               "\"mean_ms\":%.2f,\"pass\":%s}\n",
               n, match, oracle.size(), first_diff, c->step_s * 1000 / std::max(1, c->steps),
               pass ? "true" : "false");
        if (!oracle.empty() && first_diff >= 0) {
            printf("CF_GEN t4q:");
            for (size_t i = 0; i < std::min(gen.size(), (size_t)32); i++) printf(" %d", gen[i]);
            printf("\nCF_GEN orc:");
            for (size_t i = 0; i < std::min(oracle.size(), (size_t)32); i++) printf(" %d", oracle[i]);
            printf("\n");
        }
        cf_free(c);
        return pass ? 0 : 3;
    }

    if (mode == "time") {
        const int n = atoi(argv[4]);
        for (int t : ids)
            if (!cf_step(c, t)) { printf("CF {\"error\":\"prompt step: %s\"}\n", c->err.c_str()); return 1; }
        const double prompt_s = c->step_s;
        const int prompt_n = c->steps;
        const int eos = cf::EOS2;
        std::vector<int32_t> gen;
        auto tg0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; i++) {
            const int tok = argmax(c->h_logits, cf::V);
            gen.push_back(tok);
            if (tok == eos) break;
            if (i + 1 < n && !cf_step(c, tok)) { printf("CF {\"error\":\"gen step: %s\"}\n", c->err.c_str()); return 1; }
        }
        const double gen_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tg0).count();
        const int gn = (int)gen.size() - 1;
        printf("CF {\"mode\":\"time\",\"prompt_ms\":%.1f,\"prompt_n\":%d,\"gen_ms_per_tok\":%.2f,\"gen_n\":%d,"
               "\"tok_s\":%.2f,\"load_s\":%.1f}\n",
               prompt_s * 1000 / std::max(1, prompt_n), prompt_n, gen_s * 1000 / std::max(1, gn), gn,
               gn / gen_s, load_s);
        printf("CF_TOK:");
        for (int t : gen) printf(" %d", t);
        printf("\n");
        cf_free(c);
        return 0;
    }

    if (mode == "census") {
        // cf-m2: the router concentration census. Header: "CFC1" u32 NL u32 TOPK u32 n_steps,
        // then per step per layer the top-10 (u32 id, f32 renormed weight). Analysis is local.
        const int n = atoi(argv[4]);
        FILE* cf = fopen(argv[5], "wb");
        if (!cf) { printf("CF {\"error\":\"census open %s\"}\n", argv[5]); return 1; }
        uint32_t hdr[3] = {(uint32_t)'C' | ((uint32_t)'F' << 8) | ((uint32_t)'C' << 16) | ((uint32_t)'1' << 24),
                           (uint32_t)cf::NL, (uint32_t)cf::TOPK};
        fwrite(hdr, 4, 3, cf);
        c->census_f = cf;
        for (int t : ids)
            if (!cf_step(c, t)) { printf("CF {\"error\":\"census prompt step: %s\"}\n", c->err.c_str()); return 1; }
        const int eos = cf::EOS2;
        std::vector<int32_t> gen;
        for (int i = 0; i < n; i++) {
            const int tok = argmax(c->h_logits, cf::V);
            gen.push_back(tok);
            if (tok == eos) break;
            if (i + 1 < n && !cf_step(c, tok)) {
                printf("CF {\"error\":\"census gen step: %s\"}\n", c->err.c_str());
                return 1;
            }
        }
        c->census_f = nullptr;
        fclose(cf);
        printf("CF {\"mode\":\"census\",\"prompt_n\":%d,\"gen_n\":%d,\"steps\":%d,\"mean_ms\":%.2f}\n",
               (int)ids.size(), (int)gen.size() - 1, c->steps, c->step_s * 1000 / std::max(1, c->steps));
        cf_free(c);
        return 0;
    }

    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
