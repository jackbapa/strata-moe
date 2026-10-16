#pragma once

#include "ggml.h"

namespace strata_a3b {
// Opaque, reusable CUDA scratch for the original Strata grouped expert kernel.
struct StrataNativeExperts;
bool strata_native_experts_compatible(const ggml_tensor * gate, const ggml_tensor * up,
    const ggml_tensor * down);
StrataNativeExperts * strata_native_experts_create(int max_entries, int hidden, int ffn);
void strata_native_experts_destroy(StrataNativeExperts * state);
uint64_t strata_native_experts_calls(StrataNativeExperts * state);
void strata_native_experts_compute(StrataNativeExperts * state, void * stream,
    const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down,
    const ggml_tensor * input, const ggml_tensor * ids, ggml_tensor * output);
}
