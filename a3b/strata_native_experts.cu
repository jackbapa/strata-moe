#include "strata_native_experts.hpp"
#include "strata_expert_plan_parallel.cuh"
#include "strata/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata_a3b {
namespace {
void checked(cudaError_t result, const char * operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

// Match the original Strata grouping: every routed expert appears once, with
// all of its token rows consecutive so one weight decode serves the group.
// The original A3B adapter used one CUDA thread for at most 64 routes.
// Keep that default unchanged; the opt-in parallel plan below adapts the
// original Strata MTP resident grouping without a host read or sync.
__global__ void plan_expert_groups(const int32_t * ids, size_t ids_token_stride,
    int routes, int tokens, int input_lanes, const uint8_t * gate,
    const uint8_t * up, const uint8_t * down, size_t gate_stride,
    size_t up_stride, size_t down_stride, int experts,
    unsigned long long * group_ptr, int32_t * starts,
    int32_t * destinations, int32_t * input_rows, int32_t * count,
    unsigned long long * calls) {
    if (threadIdx.x != 0) return;
    const int n = routes * tokens;
    int32_t expert_ids[64];
    bool consumed[64]{};
    for (int i = 0; i < n; ++i) {
        const int t = i / routes, k = i % routes;
        expert_ids[i] = *(const int32_t *) ((const uint8_t *) ids + t * ids_token_stride + k * sizeof(int32_t));
        assert(expert_ids[i] >= 0 && expert_ids[i] < experts);
    }
    int groups = 0, entry = 0;
    for (int i = 0; i < n; ++i) {
        if (consumed[i]) continue;
        const int e = expert_ids[i];
        starts[groups] = entry;
        group_ptr[3 * groups]     = (unsigned long long) (gate + size_t(e) * gate_stride);
        group_ptr[3 * groups + 1] = (unsigned long long) (up   + size_t(e) * up_stride);
        group_ptr[3 * groups + 2] = (unsigned long long) (down + size_t(e) * down_stride);
        for (int j = i; j < n; ++j) if (expert_ids[j] == e) {
            consumed[j] = true;
            const int t = j / routes, k = j % routes;
            destinations[entry] = j;
            input_rows[entry] = t * input_lanes + (input_lanes == 1 ? 0 : k);
            ++entry;
        }
        ++groups;
    }
    starts[groups] = entry;
    *count = groups;
    atomicAdd(calls, 1ULL);
}
}

struct StrataNativeExperts {
    int max_entries, hidden, ffn;
    unsigned long long * group_ptr = nullptr;
    int32_t * starts = nullptr, * destinations = nullptr, * input_rows = nullptr, * count = nullptr;
    unsigned long long * calls = nullptr;
    void * input_q8 = nullptr, * scratch = nullptr;
};

bool strata_native_experts_compatible(const ggml_tensor * gate, const ggml_tensor * up,
    const ggml_tensor * down) {
    return gate && up && down && gate->type == up->type &&
        gate->ne[0] == up->ne[0] && gate->ne[1] == up->ne[1] &&
        strata::kernels::native_expert_supported(gate->type, down->type,
            gate->ne[0], gate->ne[1]);
}

StrataNativeExperts * strata_native_experts_create(int max_entries, int hidden, int ffn) {
    auto * s = new StrataNativeExperts{};
    s->max_entries = max_entries; s->hidden = hidden; s->ffn = ffn;
    try {
        checked(cudaMalloc(&s->group_ptr, size_t(3 * max_entries) * sizeof(unsigned long long)), "native group pointers");
        checked(cudaMalloc(&s->starts, size_t(max_entries + 1) * sizeof(int32_t)), "native group starts");
        checked(cudaMalloc(&s->destinations, size_t(max_entries) * sizeof(int32_t)), "native destinations");
        checked(cudaMalloc(&s->input_rows, size_t(max_entries) * sizeof(int32_t)), "native input rows");
        checked(cudaMalloc(&s->count, sizeof(int32_t)), "native group count");
        checked(cudaMalloc(&s->calls, sizeof(unsigned long long)), "native call count");
        checked(cudaMemset(s->calls, 0, sizeof(unsigned long long)), "clear native call count");
        checked(cudaMalloc(&s->input_q8, size_t(max_entries) * size_t(hidden / 32) * 36), "native q8 inputs");
        checked(cudaMalloc(&s->scratch, strata::kernels::native_expert_scratch_bytes(max_entries, ffn)), "native expert scratch");
        return s;
    } catch (...) { strata_native_experts_destroy(s); throw; }
}

void strata_native_experts_destroy(StrataNativeExperts * s) {
    if (!s) return;
    cudaFree(s->scratch); cudaFree(s->input_q8); cudaFree(s->calls); cudaFree(s->count);
    cudaFree(s->input_rows); cudaFree(s->destinations); cudaFree(s->starts); cudaFree(s->group_ptr);
    delete s;
}

uint64_t strata_native_experts_calls(StrataNativeExperts * s) {
    if (!s) return 0;
    unsigned long long count = 0;
    checked(cudaMemcpy(&count, s->calls, sizeof(count), cudaMemcpyDeviceToHost), "native call count");
    return count;
}

void strata_native_experts_compute(StrataNativeExperts * s, void * stream,
    const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down,
    const ggml_tensor * input, const ggml_tensor * ids, ggml_tensor * output) {
    const int routes = (int) ids->ne[0], tokens = (int) ids->ne[1];
    const int entries = routes * tokens, input_lanes = (int) input->ne[1];
    if (!s || !stream || entries < 1 || entries > 64 || entries > s->max_entries || input->ne[0] != s->hidden ||
        gate->ne[1] != s->ffn || input->nb[0] != sizeof(float) ||
        input->nb[1] != size_t(s->hidden) * sizeof(float) ||
        input->nb[2] != size_t(input_lanes) * input->nb[1] ||
        ids->nb[0] != sizeof(int32_t))
        throw std::runtime_error("invalid Strata native A3B expert geometry");
    auto L = strata::kernels::native_expert_layout(gate->type, down->type, s->hidden, s->ffn);
    L.split_storage = true;
    auto cuda_stream = (cudaStream_t) stream;
    static const bool parallel_plan = [] {
        const char * flag = std::getenv("STRATA_A3B_PARALLEL_PLAN");
        return flag && flag[0] == '1' && flag[1] == '\0';
    }();
    // The original Strata verifier gives every VRAM-resident expert group its
    // own block row (grid_groups=0).  The A3B adapter used eight rows, which
    // makes later groups run serially within a block row during decode.  Keep
    // the old launch geometry unless this is explicitly enabled and measured.
    static const bool full_group_grid = [] {
        const char * flag = std::getenv("STRATA_A3B_FULL_GROUP_GRID");
        return flag && flag[0] == '1' && flag[1] == '\0';
    }();
    if (parallel_plan)
        detail::plan_expert_groups_parallel<<<1, 64, 0, cuda_stream>>>(
            (const int32_t *) ids->data, ids->nb[1], routes, tokens, input_lanes,
            (const uint8_t *) gate->data, (const uint8_t *) up->data, (const uint8_t *) down->data,
            gate->nb[2], up->nb[2], down->nb[2], (int) gate->ne[2],
            s->group_ptr, s->starts, s->destinations, s->input_rows, s->count, s->calls);
    else
        plan_expert_groups<<<1, 1, 0, cuda_stream>>>(
            (const int32_t *) ids->data, ids->nb[1], routes, tokens, input_lanes,
            (const uint8_t *) gate->data, (const uint8_t *) up->data, (const uint8_t *) down->data,
            gate->nb[2], up->nb[2], down->nb[2], (int) gate->ne[2],
            s->group_ptr, s->starts, s->destinations, s->input_rows, s->count, s->calls);
    checked(cudaGetLastError(), "native route plan");
    strata::kernels::quantize_q8_1_rows((const float *) input->data,
        input_lanes * tokens, s->hidden, s->input_q8, stream);
    strata::kernels::native_expert_grouped(L, s->group_ptr, s->starts, s->count,
        s->destinations, s->input_rows, entries, entries, s->input_q8,
        s->scratch, (float *) output->data, stream, full_group_grid ? 0 : 8);
}
}
