// Host-side launchers for the M1 (straightforward, unfused) kernels.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#include "../packed.h"

// repack.cu
// raw: device staging holding GGUF rows [r0, r0+nr) of `ggml_type`; writes them into W at the same rows
void launch_repack(const PackedW& W, uint32_t ggml_type, const uint8_t* raw, int64_t r0, int64_t nr, cudaStream_t s);
// dequant rows (list of row ids on device) into out [n][cols] fp32, bit-exact with the CPU port
void launch_dequant_rows(const PackedW& W, const int32_t* rows, int n, float* out, cudaStream_t s);

// gemv_ref.cu: y[rows] = W x (fp32 activations, fp32 accumulation)
void launch_gemv(const PackedW& W, const float* x, float* y, cudaStream_t s);
// llama.cpp-style q8_1 activations (validation mode "act_q8"): per 32 block d = amax/127 and the sum of x, both
// rounded to fp16 (stored here as floats), q = round(x/d). gemv_q8 reproduces ggml's vec_dot_*_q8_1 formulas.
void launch_quantize_q8_1(const float* x, int K, int8_t* xq, float* xd, float* xs, cudaStream_t s);
void launch_gemv_q8(const PackedW& W, const int8_t* xq, const float* xd, const float* xs, float* y, cudaStream_t s);

// misc.cu
void launch_rmsnorm(const float* x, const float* w, float* y, int n, float eps, cudaStream_t s);
// per-head RMSNorm: head h reads x + h*in_stride (hdim values), writes y + h*out_stride; w shared [hdim]
void launch_rmsnorm_heads(const float* x, const float* w, float* y, int nheads, int hdim, int in_stride,
                          int out_stride, float eps, cudaStream_t s);
void launch_add(float* h, const float* a, int n, cudaStream_t s);
void launch_silu_mul(const float* g, const float* u, float* out, int n, cudaStream_t s);
// embedding: dequant Q4_0 row from a host-or-device pointer is done on host in M1
void launch_argmax(const float* x, int n, int* out_idx, float* out_val, cudaStream_t s);

// gdn.cu
// conv1d (kernel 4) + SiLU over nch channels; conv_state [nch][3] holds raw inputs, updated in place
void launch_gdn_conv(const float* qkv, float* conv_state, const float* conv_w, float* y, int nch, cudaStream_t s);
// L2-normalize q and k heads (16 each, dim 128): qn, kn [16][128]
void launch_gdn_l2(const float* y, float* qn, float* kn, float eps, cudaStream_t s);
// beta = sigmoid(b_raw); g = ssm_a * softplus(a_raw + dt)
void launch_gdn_gates(const float* b_raw, const float* a_raw, const float* ssm_a, const float* dt, float* beta,
                      float* g, int nv, cudaStream_t s);
// recurrence: S [48][128 v][128 k] (ggml layout: row = value column, contiguous over k), o [48][128]
void launch_gdn_recur(float* S, const float* qn, const float* kn, const float* v, const float* beta, const float* g,
                      float* o, float scale, cudaStream_t s);
// gated RMSNorm: out = rmsnorm(o_h) * w * silu(z_h), per head of 128
void launch_gdn_gnorm(const float* o, const float* z, const float* w, float* out, int nv, float eps, cudaStream_t s);

// attn.cu
// qfull [24][512] (q | gate per head); qn [24][256] = RoPE(RMSNorm(q)); kn [4][256] = RoPE(RMSNorm(k))
void launch_qk_norm_rope(const float* qfull, const float* k, const float* qw, const float* kw, float* qn, float* kn,
                         int pos, float eps, float theta_base_freq, int n_rot, cudaStream_t s);
// append k, v (fp32 [4][256]) as fp16 at position pos: cache layout [kvh][max_ctx][256]
void launch_kv_store(const float* k, const float* v, uint16_t* kc, uint16_t* vc, int pos, int max_ctx, cudaStream_t s);
// out [24][256] = softmax(q K^T * scale) V over n_kv positions; scores scratch [24][max_ctx]
void launch_attn_decode(const float* q, const uint16_t* kc, const uint16_t* vc, float* out, float* scores, int n_kv,
                        int max_ctx, float scale, cudaStream_t s);
// out[h*256+d] = att[h*256+d] * sigmoid(qfull[h*512+256+d])
void launch_gate_sigmoid(const float* att, const float* qfull, float* out, cudaStream_t s);
