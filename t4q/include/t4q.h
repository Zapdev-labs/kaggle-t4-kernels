/* t4q: Qwen3.8-27B decode engine for 2x T4 (sm_75). Plain C ABI, loaded from Python with ctypes. */
#ifndef T4Q_H
#define T4Q_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct t4q_ctx t4q_ctx;
typedef struct { int n_gpu; int tp; int max_ctx; int kv_q8; int spec_k; int draft_vocab; int verbose; } t4q_params;
typedef struct { float temp; int top_k; float top_p; uint64_t seed; } t4q_sampling;

t4q_ctx* t4q_load(const char* gguf, const t4q_params*);
int  t4q_prefill(t4q_ctx*, const int32_t* ids, int n);                 /* returns 0 or error */
/* returns number of generated tokens (stops after a stop token, which is included), or <0 on error */
int  t4q_generate(t4q_ctx*, int32_t* out, int max_new, const t4q_sampling*, const int32_t* stop, int n_stop);
/* debug: feed n tokens at the current position, write n x n_vocab fp32 logits (out may be NULL) */
int  t4q_logits(t4q_ctx*, const int32_t* ids, int n, float* out);
/* logits of the last processed position (after t4q_prefill / t4q_logits); out holds n_vocab floats */
int  t4q_last_logits(t4q_ctx*, float* out);
/* named intermediate from the last token step while dump mode is on; returns element count or -1 */
int  t4q_dump(t4q_ctx*, const char* name, int layer, float* out, size_t cap);
void t4q_set_dump(t4q_ctx*, int on);
/* runtime options: "act_q8" (1 = llama.cpp-style q8_1 activations in quantized GEMVs; validation mode) */
int  t4q_set_option(t4q_ctx*, const char* key, int value);
/* validation: run layer il alone at position pos on a residual input h_in[5120]; writes the layer output */
int  t4q_layer_forward(t4q_ctx*, int il, int pos, const float* h_in, float* h_out);
int  t4q_dump_keys(t4q_ctx*, char* buf, int cap); /* newline separated "name-layer" keys */
void t4q_stats(t4q_ctx*, char* json, int cap);   /* tok/s, load info, repack check */
int  t4q_n_vocab(t4q_ctx*);
int  t4q_pos(t4q_ctx*);
const char* t4q_last_error(void);
/* batched decode (TP engine only). Slots hold independent sequences (DeltaNet state, conv ring, KV cache).
   init: (re)allocates n_slots slots of slot_ctx positions (state_f16: fp16 DeltaNet state storage, fp32 math).
   prefill: runs the prompt through the single-stream engine and copies its state into the slot; returns the first
   generated token (greedy) or <0. step: one greedy decode step for n slots (each consumes its pending token, which
   the step replaces with the argmax written to out[i]); returns n or <0. set_token overrides a slot's pending token
   (teacher forcing). logits: row `row` of the last step (n_vocab floats). */
int  t4q_batch_init(t4q_ctx*, int n_slots, int slot_ctx, int state_f16);
int  t4q_batch_prefill(t4q_ctx*, int slot, const int32_t* ids, int n);
int  t4q_batch_clone(t4q_ctx*, int src, int dst);
int  t4q_batch_set_token(t4q_ctx*, int slot, int token);
int  t4q_batch_pos(t4q_ctx*, int slot);
int  t4q_batch_step(t4q_ctx*, int n, const int32_t* slots, int32_t* out);
int  t4q_batch_logits(t4q_ctx*, int row, float* out);
void t4q_batch_free(t4q_ctx*);
void t4q_reset(t4q_ctx*);
void t4q_free(t4q_ctx*);

#ifdef __cplusplus
}
#endif
#endif
