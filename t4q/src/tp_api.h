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
// batched decode (tp_batch.cuh, compiled into tp_prefill.cu)
int tp_batch_init(t4q_ctx* c, int n_slots, int slot_ctx, int sf16);
void tp_batch_free(t4q_ctx* c);
int tp_batch_prefill(t4q_ctx* c, int slot, const int32_t* ids, int n);
int tp_batch_clone(t4q_ctx* c, int src, int dst);
int tp_batch_set_token(t4q_ctx* c, int slot, int token);
int tp_batch_pos(t4q_ctx* c, int slot);
int tp_batch_step(t4q_ctx* c, int n, const int32_t* slots, int32_t* out);
int tp_batch_logits(t4q_ctx* c, int row, float* out);
std::string tp_batch_stats(t4q_ctx* c);
void tp_batch_reset_stats(t4q_ctx* c);
