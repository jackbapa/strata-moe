#pragma once

#include "ggml.h"
#include <cstdint>

namespace strata_a3b {

// Opt-in, GPU-resident, long-prefill adapter to original Strata's grouped IQ
// MMQ runtime. A3B's IQ GGUF stores gate/up/down as separate tensors, so the
// adapter issues three MMQ products without making a second weight copy.
struct StrataPrefillMmq;

// output may be null while building the graph; compute requires a fully
// allocated output tensor. Compatibility checks only layout and MMQ support.
bool strata_prefill_mmq_compatible(const ggml_tensor * gate,
    const ggml_tensor * up, const ggml_tensor * down,
    const ggml_tensor * input, const ggml_tensor * ids,
    const ggml_tensor * output);

StrataPrefillMmq * strata_prefill_mmq_create(
    int max_entries, int hidden, int ffn, int experts);
void strata_prefill_mmq_destroy(StrataPrefillMmq * state);

// These getters synchronize the device and belong outside timed inference.
uint64_t strata_prefill_mmq_calls(StrataPrefillMmq * state);
uint64_t strata_prefill_mmq_rows(StrataPrefillMmq * state);

void strata_prefill_mmq_compute(StrataPrefillMmq * state, void * stream,
    const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down,
    const ggml_tensor * input, const ggml_tensor * ids, ggml_tensor * output);

} // namespace strata_a3b
