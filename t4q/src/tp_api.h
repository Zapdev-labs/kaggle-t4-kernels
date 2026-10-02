// TP engine entry points (tp_engine.cu), dispatched from api.cpp when t4q_params.tp != 0.
#pragma once
#include <cstdint>
#include <string>

struct t4q_ctx;
void tp_load(t4q_ctx* c, const char* path);
void tp_reset(t4q_ctx* c);
int tp_logits(t4q_ctx* c, const int32_t* ids, int n, float* out);
int tp_prefill(t4q_ctx* c, const int32_t* ids, int n);
int tp_last_logits(t4q_ctx* c, float* out);
int tp_generate(t4q_ctx* c, int32_t* out, int max_new, const int32_t* stop, int n_stop);
int tp_set_option(t4q_ctx* c, const std::string& key, int value);  // -1: unknown key
std::string tp_stats_json(t4q_ctx* c);
