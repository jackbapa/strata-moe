#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "ggml.h"

namespace strata_a3b {

// Original Strata's native expert pool shares each token's quantized input,
// groups work by expert, and splits Gate/Up and Down by rows across cores.
// This opt-in executor applies that design to quantized MoE geometry.
// It owns activation/scheduling buffers, never a second copy of the weights.
struct CpuExpertWeightView {
    const void * data = nullptr;
    size_t row_stride = 0;
    size_t expert_stride = 0;
    ggml_type type = GGML_TYPE_Q8_0;
};
struct CpuExpertWeights {
    CpuExpertWeightView gate, up, down;
    int hidden = 0, ffn = 0, experts = 0;
};
struct CpuExpertInput {
    const float * data = nullptr;
    int lanes = 1, tokens = 1;
    // The hidden dimension consists of contiguous F32 values. Zero strides
    // select the natural [hidden, lanes, tokens] layout.
    size_t lane_stride = 0, token_stride = 0;
};
struct CpuExpertJob { int expert = 0, token = 0, lane = 0; };
struct CpuExpertPoolStats {
    uint64_t calls = 0, jobs = 0, unique_inputs = 0, unique_experts = 0;
    double input_quant_ms = 0, gate_up_ms = 0, hidden_quant_ms = 0, down_ms = 0;
};

class CpuExpertPool {
public:
    struct Config {
        // Total participating threads, including the caller. Zero selects
        // one thread per available physical core. The caller also drains rows.
        int threads = 0;
        bool pin = true;
        int tasks = 0;       // Zero: three row tasks per participating thread.
        int spin_us = 2000;  // Then workers sleep until a phase is published.
    };
    CpuExpertPool();
    explicit CpuExpertPool(Config config);
    ~CpuExpertPool();
    CpuExpertPool(const CpuExpertPool &) = delete;
    CpuExpertPool & operator=(const CpuExpertPool &) = delete;

    // Single caller; tokens <= 256. Each result occupies weights.hidden floats
    // at the corresponding job index. lane broadcasts modulo input.lanes,
    // like the existing routed FFN. Empty jobs produce an empty result.
    // Arithmetic: ggml-cpu quantizers/dots for each GGUF weight type, scalar std::exp SwiGLU.
    // CPU and GPU arithmetic are not promised to be bit-identical.
    // Short-prefill sharing supplies sparse low-fanout jobs; activation and
    // result buffers scale with the jobs actually requested, not all tokens.
    void execute(const CpuExpertWeights & weights, const CpuExpertInput & input,
                 const std::vector<CpuExpertJob> & jobs, std::vector<float> & outputs);
    int threads() const noexcept;
    int physical_cores() const noexcept;
    CpuExpertPoolStats stats() const noexcept;
    void reset_stats() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata_a3b
