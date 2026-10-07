// cf-m1: the CYBER-FROST engine driver. Modes:
//   seq  <model> <ids.i32> <oracle.tbt.f32> <n_last>  -> step every id, compare the per-position logits with the
//                                                          oracle's token-by-token dump (max rel diff, top1 agree)
//   gen  <model> <ids.i32> <n> <oracle.gen.i32>        -> feed the ids, then n greedy tokens (byte-compare)
//   time <model> <ids.i32> <n>                        -> the timed run: prompt + n greedy, tok/s + step stats
//   draft <model> <ids.i32> <n>                       -> cf-m4: the MTP acceptance smoke (needs T4Q_CF_MTP=1 at
//                                                          load): the prompt + n greedy with the per-pair draft
//                                                          forward (x_q, h_{q-1}); prints alpha1 (the 1-step
//                                                          acceptance the MTP speed math rides on) + draft/trunk ms
//   verify <model> <ids.i32> <n>                      -> cf-m4: THE GATE (needs T4Q_CF_MTP=1): run A = the prompt +
//                                                          n-1 greedy steps (the reference logits); cf_reset; run B =
//                                                          the same prompt + the verify rounds over the SAME tokens
//                                                          (nr = T4Q_CF_K+1 rows per round) - every row's logits
//                                                          must BYTE-MATCH the sequential reference
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
        fprintf(stderr, "usage: %s seq|gen|time|census|draft|verify <model> <ids.i32> [args]\n", argv[0]);
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

    if (mode == "draft") {
        // cf-m4 (r19w): the MTP acceptance smoke - the 1-step acceptance (alpha1) the whole
        // MTP speed math rides on. The pair semantics (x_q, h_{q-1}), h_{-1} = 0: the
        // pre-loop pair (ids[0], 0) fills the draft's slot 0 and predicts position 1;
        // after each trunk step at i, the compare (the draft's prediction from its pair at
        // i vs the run's ACTUAL token at i+1 - the prompt id or the trunk's own greedy
        // argmax) and the next pair (x_{i+1}, h_i = the trunk's pre-final-mixer residual).
        const int n = atoi(argv[4]);
        const int P = (int)ids.size();
        if (P < 1) { printf("CF {\"error\":\"draft needs >= 1 prompt token\"}\n"); return 1; }
        if (!c->draft) { printf("CF {\"error\":\"no draft block (T4Q_CF_MTP=1 at load)\"}\n"); return 1; }
        double draft_s = 0;
        int pairs = 0, agree1 = 0, first_dis = -1;
        auto tdd = [&](int tok, const float* h) {
            const auto ta = std::chrono::steady_clock::now();
            if (!cf_draft_step(c, tok, h)) {
                printf("CF {\"error\":\"draft step: %s\"}\n", c->err.c_str());
                exit(1);
            }
            draft_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - ta).count();
        };
        tdd(ids[0], nullptr);  // the position-0 pair: (x_0, h_{-1} = 0)
        int pred = argmax(c->draft->h_logits, cf::V);
        for (int i = 0; i < P + n - 1; i++) {
            const int x = (i < P) ? ids[i] : argmax(c->h_logits, cf::V);
            if (!cf_step(c, x)) { printf("CF {\"error\":\"trunk step %d: %s\"}\n", i, c->err.c_str()); return 1; }
            const int next = (i + 1 < P) ? ids[i + 1] : argmax(c->h_logits, cf::V);
            pairs++;
            if (pred == next) agree1++;
            else if (first_dis < 0) first_dis = i;
            if (i + 1 < P + n - 1) {  // the pair at i+1; the final, compare-less pair is skipped
                tdd(next, c->sc.h);
                pred = argmax(c->draft->h_logits, cf::V);
            }
        }
        printf("CF {\"mode\":\"draft\",\"pairs\":%d,\"agree1\":%d,\"alpha1\":%.4f,\"first_dis\":%d,"
               "\"draft_ms\":%.2f,\"trunk_ms\":%.2f,\"load_s\":%.1f}\n",
               pairs, agree1, (double)agree1 / std::max(1, pairs), first_dis,
               draft_s * 1000 / std::max(1, pairs), c->step_s * 1000 / std::max(1, c->steps), load_s);
        cf_free(c);
        return 0;
    }

    if (mode == "verify") {
        // cf-m4 (r19x): THE GATE (CF_MTP.md section 5) - the batched verify rows MUST
        // reproduce the sequential decode bit-exactly (the near-tie argmaxes flip
        // otherwise). Run A: the prompt + the greedy tail, ref[i] = the logits AFTER the
        // step consuming tok_i (the prediction of tok_{i+1}). Run B (after cf_reset): the
        // same prompt + the verify rounds over the SAME tail tokens (chunked by nr = k+1)
        // - the row i's logits must byte-match ref[i] at every position.
        const int n = atoi(argv[4]);
        if (!c->verify) { printf("CF {\"error\":\"no verify block (T4Q_CF_MTP=1 at load)\"}\n"); return 1; }
        const int nr = c->verify->nr;
        if (n < 2) { printf("CF {\"error\":\"verify needs n >= 2\"}\n"); return 1; }
        for (int t : ids)
            if (!cf_step(c, t)) { printf("CF {\"error\":\"prompt step: %s\"}\n", c->err.c_str()); return 1; }
        std::vector<int32_t> tail;                      // run A's consumed tail tokens
        std::vector<float> ref((size_t)(n - 1) * cf::V);
        for (int i = 0; i < n; i++) {
            const int tok = argmax(c->h_logits, cf::V);
            if (i + 1 < n) {
                tail.push_back(tok);
                if (!cf_step(c, tok)) { printf("CF {\"error\":\"gen step: %s\"}\n", c->err.c_str()); return 1; }
                memcpy(ref.data() + (size_t)i * cf::V, c->h_logits, (size_t)cf::V * 4);
            }
        }
        const double seq_ms = c->step_s * 1000 / std::max(1, c->steps);
        cf_reset(c);
        for (int t : ids)
            if (!cf_step(c, t)) { printf("CF {\"error\":\"prompt step 2: %s\"}\n", c->err.c_str()); return 1; }
        int rows = 0, bytematch = 0, top1 = 0, first_diff = -1;
        double worst_rel = 0, v_s = 0;
        for (int i = 0; i < (int)tail.size();) {
            const int m = std::min(nr, (int)tail.size() - i);
            const auto tv = std::chrono::steady_clock::now();
            if (!cf_verify(c, tail.data() + i, m)) {
                printf("CF {\"error\":\"verify round: %s\"}\n", c->err.c_str());
                return 1;
            }
            v_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - tv).count();
            for (int r = 0; r < m; r++) {
                const float* mine = c->verify->h_logits + (size_t)r * cf::V;
                const float* want = ref.data() + (size_t)(i + r) * cf::V;
                rows++;
                const bool bm = memcmp(mine, want, (size_t)cf::V * 4) == 0;
                if (bm) bytematch++;
                if (argmax(mine, cf::V) == argmax(want, cf::V)) top1++;
                if ((!bm || argmax(mine, cf::V) != argmax(want, cf::V)) && first_diff < 0) first_diff = i + r;
                if (!bm)
                    for (int j = 0; j < cf::V; j++) {
                        const double a = mine[j], b = want[j];
                        const double rel = std::fabs(a - b) / (std::fabs(b) + 1e-6);
                        if (rel > worst_rel) worst_rel = rel;
                    }
            }
            i += m;
        }
        const bool pass = bytematch == rows;
        printf("CF {\"mode\":\"verify\",\"n\":%d,\"nr\":%d,\"rows\":%d,\"byte_match\":%d,\"top1_agree\":%d,"
               "\"first_diff\":%d,\"worst_rel\":%.3e,\"seq_ms\":%.2f,\"verify_ms\":%.2f,"
               "\"verify_ms_per_tok\":%.2f,\"pass\":%s}\n",
               n, nr, rows, bytematch, top1, first_diff, worst_rel, seq_ms, v_s * 1000,
               v_s * 1000 / std::max(1, rows), pass ? "true" : "false");
        cf_free(c);
        return pass ? 0 : 3;
    }

    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
