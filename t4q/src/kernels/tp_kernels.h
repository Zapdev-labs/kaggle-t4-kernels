// Launchers for the TP decode kernels (tp_kernels.cu).
#pragma once
#include "../tp.h"

namespace tp {

// y = W x (q8 activations xq/xm). With ar: also stores rows to ar.y_peer and publishes the AR flag (two-level).
// With seg: extra fp32 rows. m columns (1 for decode).
// pro: optional prologue (x built in shared memory, xq/xm unused). Grid: <= max_blocks co-resident blocks, each warp
// owns a contiguous run of tiles.
void gemv(const FW& W, const int8_t* xq, const int2* xm, float* y, cudaStream_t s, const ArArgs* ar = nullptr,
          const SegArgs* seg = nullptr, const ProArgs* pro = nullptr);
void set_max_blocks(int n);

// fallback (no P2P): wait for the host-mapped flag of AR idx, copy the 20 KB payload to the local rx slot
void pull(const unsigned* hflag, const float* hrx, float* rx, StepState* st, int idx, cudaStream_t s);
// h = embed(token) (Q4_0 row dequant, bit-exact with ggml)
void embed(const uint8_t* embd, StepState* st, const int* prompt, float* h, cudaStream_t s);
// [wait for AR idx] h += own + rx; xn = rmsnorm(h) * w; q8(xn) -> xq, xm
// own == nullptr: no add (layer 0); flag == nullptr: no wait. own/rx are slot bases ([2][5120]); idx & 1 = slot
void ar_norm(const float* h, float* h_out, const float* own, const float* rx, const unsigned* flag,
             const StepState* st, int idx, const float* w, float* xn, int8_t* xq, int2* xm, cudaStream_t s);
// DeltaNet step for 24 local heads: conv (ring), SiLU, L2 q/k, gates, recurrence; o [24][128]
void gdn(const float* y, const float* yab, float* ring, const float* conv_w, const float* ssm_a, const float* ssm_dt,
         float* S, float* o, const StepState* st, cudaStream_t s);
// gated RMSNorm (o, z) -> q8 for ssm_out
void gnorm_q8(const float* o, const float* z, const float* w, int8_t* xq, int2* xm, cudaStream_t s);
// silu(g) * u -> q8 (n = 8704)
void silu_q8(const float* gu, int n, int8_t* xq, int2* xm, cudaStream_t s);
// q/k RMSNorm + RoPE, KV append (fp16) at st->pos; qa [12][256]
void attn_prep(const float* ya, const float* qw, const float* kw, float* qa, uint16_t* kc, uint16_t* vc, int max_ctx,
               const StepState* st, float theta_scale, cudaStream_t s);
void attn_split(const float* qa, const uint16_t* kc, const uint16_t* vc, float* ws, int max_ctx, const StepState* st,
                cudaStream_t s);
void attn_combine_q8(const float* ws, const float* ya, const StepState* st, int8_t* xq, int2* xm, cudaStream_t s);
// argmax over the local logits shard, exchange with the peer, update StepState (token, pos, step), ring
void argmax_step(const float* logits, int n, int row0, float* apart, float* amb, const unsigned* aflag,
                 float* peer_amb, unsigned* peer_aflag, StepState* st, int* ring, cudaStream_t s);

}  // namespace tp
