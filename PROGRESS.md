# t4q progress log

Handoff log for agents working on t4q. Append a dated section per session, newest last. Plan: `research/DESIGN.md`.

## 2026-10-02 - M0-probe (round 1), GEMV fast path + box probe

**Gate: PASS (burst).** Measured on Kaggle `otdoges/t4q-m0` v1, v2 and v3. Full tables are in `research/m0_results.md`.

Built (I own these; the M1 agent owns the rest of `t4q/`):
- `t4q/src/kernels/gemv.cuh` (namespace `t4q::gemv`, enum `FAST_P4/FAST_Q8/FAST_K6`, which does not clash with `packed.h`'s `FMT_*`):
  - Lossless SoA repack for P4 (Q4_0), Q8 (Q8_0) and K6 (Q6_K), with a host version (`repack_host`) and a device version (`repack_device`).
  - A q8_1 activation quantizer (`quantize_q8_kernel`) and its host mirror.
  - `gemv_fast_kernel<FMT,RPL,M,NCH,CVT,XSM>`, a dp4a GEMV for m = 1..8 whose per-column bits are identical for every m and every knob.
  - GGUF dequant helpers for CPU references.
  - The recommended engine config is in the header comment.
- `t4q/tools/gemv_bench.cu` builds as 6 TUs in parallel. It covers the real TP shapes, runs a CPU fp64 reference check, a device-vs-host repack check and a cross-config bit-identity check, a config sweep, and a sustained mode with NVML clock and power sampling.
- `t4q/tools/probe.cu` covers:
  - P2P and host-mapped mailboxes (ping-pong one-way and symmetric exchange; per-block flags or two-level flag).
  - memcpy peer, stream read, and clocks under load.
  - NCCL through dlopen: `./probe nccl <libnccl.so>`, with `NCCL_*` env vars honoured.
- `t4q/tools/nvml_lite.h` (NVML via dlopen) and `t4q/tools/gemv_layout_check.cpp` (CPU emulation of the kernel's addressing and dp4a math; run locally with g++).
- `t4q/tools/mkkernel.py <stage>` packs `t4q/` into `kaggle/<stage>/`. The M1 agent uses it too. The stage drivers are `t4q/tools/stage_<stage>.py`, and `stage_m0.py` is mine.
- Local compile check without a GPU: `podman run --rm -v $PWD/t4q:/src:ro,Z docker.io/nvidia/cuda:12.8.1-devel-ubuntu22.04 nvcc ...`. This is the same nvcc 12.8 and gcc 11 as Kaggle, so it gives ptxas register and spill reports and SASS through cuobjdump.

Verified numbers (burst, GB/s of packed weight bytes):
- P4 decode-mix, one launch config: m=1 258.3 / 258.5 / 258.4 and m=4 247.6 / 251.4 / 250.5 (v1/v2/v3). The gate needs >= 250 and >= 230.
- Per shape at m=1: gateup 262, qkvzab_tp 259, attn_qkv 258, down 261, out_tp 247-253 (the small 8.8 MB shape is marginal). K6 lm_head 265, Q8 lm_head 266.
- Correctness:
  - GEMV max error / rms is 4.6e-7 to 6.1e-7 against the fp64 reference on the same q8 activations.
  - Device repack equals host repack byte for byte, and the q8 quantizer equals its CPU mirror exactly.
  - Every config at every m is bit-identical to the m=8 golden run.

Verified probe and AR results:
- P2P works for stores with zero ordering errors. Flag-only one-way latency is 3.6-4.1 us.
- One-way 20 KB takes 5.8 us with 8 writer blocks; 10 KB fp16 takes 4.5 us.
- Two-level flag (local atomic counter, one remote flag) takes 7.3-8.1 us with 8-64 producer blocks.
- Symmetric 20 KB exchange read by 40 blocks: 14.7 us p50.
- Host-mapped pinned memory: 24 us one-way for 20 KB. Rejected.
- NCCL 2.25.1: 28 us host-synced and 15.4 us pipelined at best (`NCCL_P2P_LEVEL=SYS`); 36-39 us with defaults.
- **Chosen: P2P remote-store mailbox.** NCCL `P2P_LEVEL=SYS` is the fallback.
- Streaming read: 282 GB/s.

Important finding (affects M4/M5):
- Under sustained load the 70 W software power cap drops the SM clock to 300-900 MHz while the dp4a GEMV runs. A plain streaming read holds 1350-1590 MHz at the same power. GPU0 is about 20 C hotter on every box and throttles hardest.
- m=1 P4 stays memory-bound: 251-259 GB/s sustained even at 420 MHz.
- m=4 drops to about 160-200 GB/s with RPL=2 and about 190-250 with RPL=4.
- K6 lm_head at m=1 drops to 152-207 GB/s at about 300-340 MHz on GPU0.
- Hence RPL=4 for all P4 tensors (sustained m=4 +20-25%, m=1 -1%). Plan for verify (m=4) costing about 1.1-1.3x a decode step on a hot GPU, not 1.0x.

Broken or open:
- `out_tp` (3072 x 5120) m=1 is about 247 GB/s standalone, a small-kernel ramp effect. It should be fused with its prologue in M2 anyway.
- No llama.cpp mmvq per-kernel comparison was run. The baseline only has end-to-end numbers.
- Q4_1 (8 ffn_down), Q5_K (ssm_out), IQ4/Q4_K are not in the fast path yet. Q4_1 is P4 plus an m plane, and Q5_K follows K6's pattern (the design calls it K5).

Next steps:
1. M2 should call `gemv_fast_launch<FAST_P4, 4, M, K/512, 1, false>` with 256 threads and a persistent grid. Add Q4_1 (P4 + fp16 m plane, using `sumx * m` from the q8 meta) and K5 (Q5_K: K6-style 1-bit high plane plus native 12-byte scales) to `gemv.cuh`, and validate both with `gemv_layout_check.cpp` and gemv_bench.
2. Fuse the prologues (norm and q8 quantize) and epilogues (swiglu, AR publish, argmax). The kernel already takes q8 activations from global memory.
3. To reduce verify cost on a throttled GPU (ALU-bound at m=4), try an int8 `mma.m8n8k16` swap-AB path. Integer group sums are exact, so it can stay bit-identical to dp4a if the float epilogue order is kept. Also try cheaper K6 unpacking.
4. M4 AR: use the P2P mailbox with weight prefetch before the flag wait, and measure the exposed AR cost inside the real graph.
