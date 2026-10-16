#include "strata_prefill_mmq_adapter.hpp"

#include "strata/prefill/moe_mmq.hpp"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "mmid.cuh"
#include "quantize.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata_a3b {
namespace {
namespace mmq = strata::prefill::mmq;

void ck(cudaError_t e, const char * operation) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("A3B Strata prefill MMQ ") + operation + ": " + cudaGetErrorString(e));
}

__global__ void count_prefill_call(unsigned long long * calls,
                                  unsigned long long * rows, int entries) {
    if (threadIdx.x == 0) {
        atomicAdd(calls, 1ULL);
        atomicAdd(rows, static_cast<unsigned long long>(entries));
    }
}

int64_t pad512(int64_t n) { return (n + 511) / 512 * 512; }

template<class T> T * allocate(size_t n, const char * name) {
    T * p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), name);
    return p;
}

} // namespace

struct StrataPrefillMmq {
    int max_entries = 0, hidden = 0, ffn = 0, experts = 0;
    int32_t * ids_src1 = nullptr, * ids_dst = nullptr, * bounds = nullptr, * identity = nullptr;
    void * xq = nullptr, * hq = nullptr;
    float * gu = nullptr, * h = nullptr;
    unsigned long long * calls = nullptr, * rows = nullptr;
    std::unique_ptr<mmq::Context> context;
};

bool strata_prefill_mmq_compatible(const ggml_tensor * gate,
    const ggml_tensor * up, const ggml_tensor * down,
    const ggml_tensor * input, const ggml_tensor * ids,
    const ggml_tensor * output) {
    if (!gate || !up || !down || !input || !ids ||
        gate->type != up->type || input->type != GGML_TYPE_F32 ||
        ids->type != GGML_TYPE_I32 ||
        gate->ne[0] != 2048 || gate->ne[1] != 512 || gate->ne[2] != 256 || gate->ne[3] != 1 ||
        up->ne[0] != gate->ne[0] || up->ne[1] != gate->ne[1] ||
        up->ne[2] != gate->ne[2] || up->ne[3] != 1 ||
        down->ne[0] != 512 || down->ne[1] != 2048 ||
        down->ne[2] != gate->ne[2] || down->ne[3] != 1 ||
        input->ne[0] != 2048 || input->ne[2] < 9 || input->ne[2] > 2048 || input->ne[3] != 1 ||
        ids->ne[0] != 8 || ids->ne[1] != input->ne[2] || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        (input->ne[1] != 1 && input->ne[1] != ids->ne[0]) ||
        !ggml_is_contiguous(gate) || !ggml_is_contiguous(up) || !ggml_is_contiguous(down) ||
        !ggml_is_contiguous(input) || !ggml_is_contiguous(ids) ||
        (output && (output->type != GGML_TYPE_F32 ||
                    output->ne[0] != 2048 || output->ne[1] != ids->ne[0] ||
                    output->ne[2] != ids->ne[1] || output->ne[3] != 1 ||
                    !ggml_is_contiguous(output))))
        return false;
    // The downloaded IQ3_S GGUF uses IQ2_S gate/up, IQ3_S or IQ4_XS down.
    // Limiting the adapter to these formats also keeps unsupported draft or
    // alternative model weights on the unchanged GGML path.
    if (gate->type != GGML_TYPE_IQ2_S ||
        (down->type != GGML_TYPE_IQ3_S && down->type != GGML_TYPE_IQ4_XS))
        return false;
    // The geometry and types above are fixed for this GGUF. Cache the tile
    // checks so a 50K prompt does not repeat them for every layer/chunk.
    static const bool iq2_fit = mmq::supported(GGML_TYPE_IQ2_S) &&
                                mmq::fits(GGML_TYPE_IQ2_S, 512);
    static const bool iq3_fit = mmq::supported(GGML_TYPE_IQ3_S) &&
                                mmq::fits(GGML_TYPE_IQ3_S, 2048);
    static const bool iq4_fit = mmq::supported(GGML_TYPE_IQ4_XS) &&
                                mmq::fits(GGML_TYPE_IQ4_XS, 2048);
    return iq2_fit && (down->type == GGML_TYPE_IQ3_S ? iq3_fit : iq4_fit);
}

StrataPrefillMmq * strata_prefill_mmq_create(int max_entries, int hidden, int ffn, int experts) {
    if (max_entries < 8 * 9 || max_entries > 8 * 2048 ||
        hidden != 2048 || ffn != 512 || experts != 256)
        throw std::runtime_error("invalid A3B Strata prefill MMQ geometry");
    auto * s = new StrataPrefillMmq{};
    s->max_entries = max_entries;
    s->hidden = hidden;
    s->ffn = ffn;
    s->experts = experts;
    try {
        s->ids_src1 = allocate<int32_t>(max_entries, "route source map");
        s->ids_dst = allocate<int32_t>(max_entries, "route destination map");
        s->bounds = allocate<int32_t>(experts + 1, "route expert bounds");
        s->identity = allocate<int32_t>(max_entries, "identity rows");
        ck(cudaMalloc(&s->xq, mmq::q8_bytes(max_entries, hidden)), "Q8 activation scratch");
        ck(cudaMalloc(&s->hq, mmq::q8_bytes(max_entries, ffn)), "Q8 hidden scratch");
        s->gu = allocate<float>(size_t(max_entries) * 2 * ffn, "gate/up scratch");
        s->h = allocate<float>(size_t(max_entries) * ffn, "hidden scratch");
        s->calls = allocate<unsigned long long>(1, "call counter");
        s->rows = allocate<unsigned long long>(1, "row counter");
        ck(cudaMemset(s->calls, 0, sizeof(unsigned long long)), "reset call counter");
        ck(cudaMemset(s->rows, 0, sizeof(unsigned long long)), "reset row counter");
        std::vector<int32_t> identity(max_entries);
        for (int i = 0; i < max_entries; ++i) identity[i] = i;
        ck(cudaMemcpy(s->identity, identity.data(), identity.size() * sizeof(int32_t),
                      cudaMemcpyHostToDevice), "initialize identity rows");
        s->context = std::make_unique<mmq::Context>();
        return s;
    } catch (...) { strata_prefill_mmq_destroy(s); throw; }
}

void strata_prefill_mmq_destroy(StrataPrefillMmq * s) {
    if (!s) return;
    s->context.reset();
    cudaFree(s->rows); cudaFree(s->calls); cudaFree(s->h); cudaFree(s->gu);
    cudaFree(s->hq); cudaFree(s->xq); cudaFree(s->identity); cudaFree(s->bounds);
    cudaFree(s->ids_dst); cudaFree(s->ids_src1);
    delete s;
}

uint64_t strata_prefill_mmq_calls(StrataPrefillMmq * s) {
    if (!s) return 0;
    unsigned long long count = 0;
    ck(cudaDeviceSynchronize(), "wait call counter");
    ck(cudaMemcpy(&count, s->calls, sizeof(count), cudaMemcpyDeviceToHost), "read call counter");
    return count;
}

uint64_t strata_prefill_mmq_rows(StrataPrefillMmq * s) {
    if (!s) return 0;
    unsigned long long count = 0;
    ck(cudaDeviceSynchronize(), "wait row counter");
    ck(cudaMemcpy(&count, s->rows, sizeof(count), cudaMemcpyDeviceToHost), "read row counter");
    return count;
}

void strata_prefill_mmq_compute(StrataPrefillMmq * s, void * stream,
    const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down,
    const ggml_tensor * input, const ggml_tensor * ids, ggml_tensor * output) {
    if (!s || !stream || !output ||
        !strata_prefill_mmq_compatible(gate, up, down, input, ids, output) ||
        input->ne[0] != s->hidden || gate->ne[1] != s->ffn || gate->ne[2] != s->experts ||
        ids->ne[0] * ids->ne[1] > s->max_entries ||
        !gate->data || !up->data || !down->data || !input->data || !ids->data || !output->data)
        throw std::runtime_error("invalid A3B Strata prefill MMQ call");
    const int tokens = static_cast<int>(ids->ne[1]);
    const int routes = static_cast<int>(ids->ne[0]);
    const int lanes = static_cast<int>(input->ne[1]);
    const int entries = routes * tokens;
    const auto cs = static_cast<cudaStream_t>(stream);
    // GGML's route helper is already the GPU implementation of Strata's
    // host-side expert grouping. Like GGML, it assumes top-k IDs are distinct
    // within a token; the Qwen graph obtains them from argsort_top_k.
    const bool broadcast = lanes == 1;
    ggml_cuda_launch_mm_ids_helper(static_cast<const int32_t *>(ids->data),
        s->ids_src1, s->ids_dst, s->bounds, s->experts, tokens, routes,
        lanes, static_cast<int>(ids->nb[1] / sizeof(int32_t)), lanes,
        broadcast, cs);
    ck(cudaGetLastError(), "group routes");
    if (broadcast) {
        // GGML's scatter quantizer encodes a broadcast token once before
        // placing it into every routed row. This saves seven redundant Q8_1
        // encodes while preserving the original Strata MMQ input format.
        quantize_scatter_mmq_q8_1_cuda(static_cast<const float *>(input->data),
            s->ids_src1, s->xq, gate->type, s->hidden, s->hidden,
            pad512(s->hidden), tokens, entries, routes, cs);
        ck(cudaGetLastError(), "quantize and scatter activations");
    } else {
        mmq::quantize(static_cast<const float *>(input->data), s->ids_src1,
                      s->xq, gate->type, s->hidden, s->hidden, entries, stream);
    }
    mmq::Product p;
    p.type = gate->type;
    p.w_rows = s->ffn;
    p.w_cols = s->hidden;
    p.n = s->experts;
    p.xq = s->xq;
    p.bounds = s->bounds;
    p.ids = s->identity;
    p.total_rows = entries;
    // For a top-k route each expert occurs at most once per token, exactly
    // the same per-expert grid bound used by GGML's MMID path on sm_75.
    p.max_rows = tokens;
    p.dst = s->gu;
    p.ld_dst = 2 * s->ffn;
    p.w = gate->data;
    p.expert_bytes = gate->nb[2];
    s->context->run(p, stream);
    p.w = up->data;
    p.expert_bytes = up->nb[2];
    p.dst = s->gu + s->ffn;
    s->context->run(p, stream);
    mmq::swiglu(s->gu, s->h, entries, s->ffn, false, stream);
    mmq::quantize(s->h, nullptr, s->hq, down->type, s->ffn, s->ffn, entries, stream);
    p.w = down->data;
    p.type = down->type;
    p.w_rows = s->hidden;
    p.w_cols = s->ffn;
    p.expert_bytes = down->nb[2];
    p.xq = s->hq;
    p.ids = s->ids_dst;
    p.dst = static_cast<float *>(output->data);
    p.ld_dst = s->hidden;
    s->context->run(p, stream);
    count_prefill_call<<<1, 1, 0, cs>>>(s->calls, s->rows, entries);
    ck(cudaGetLastError(), "count call");
}

} // namespace strata_a3b
