// CPU ports of ggml dequantize_row_* (bit-exact transcription; compile with -ffp-contract=off).
#pragma once
#include <cstdint>

float fp16_to_fp32(uint16_t h);
// Dequantize n elements (one or more whole rows) of a ggml type. Returns false for unsupported types.
bool dequant_row_cpu(uint32_t type, const uint8_t* src, float* dst, int64_t n);
