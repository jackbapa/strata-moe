#pragma once

#include <cuda_runtime.h>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace strata_a3b::detail {

// Adapted from the original Strata MTP resident grouping in
// src/kernels/cuda/s2_expert_grouped.cu. Each route gets a CUDA lane, while
// first-appearance group order and within-group route order remain unchanged.
// A3B stores gate/up/down in separate GGUF tensors, unlike Strata's blobs.
__global__ void plan_expert_groups_parallel(const int32_t * ids, size_t ids_token_stride,
    int routes, int tokens, int input_lanes, const uint8_t * gate,
    const uint8_t * up, const uint8_t * down, size_t gate_stride,
    size_t up_stride, size_t down_stride, int experts,
    unsigned long long * group_ptr, int32_t * starts,
    int32_t * destinations, int32_t * input_rows, int32_t * count,
    unsigned long long * calls) {
    __shared__ int32_t expert_ids[64];
    __shared__ int first[64], group_of_first[64], group_start[64], group_size[64];
    const int i = threadIdx.x;
    const int n = routes * tokens;
    if (i < n) {
        const int t = i / routes, k = i % routes;
        const int e = *(const int32_t *) ((const uint8_t *) ids + t * ids_token_stride + k * sizeof(int32_t));
        assert(e >= 0 && e < experts);
        expert_ids[i] = e;
    }
    __syncthreads();

    int first_i = i, rank = 0, size = 0;
    if (i < n) {
        const int e = expert_ids[i];
        for (int j = 0; j < i; ++j) {
            if (expert_ids[j] == e) {
                if (first_i == i) first_i = j;
                ++rank;
            }
        }
        if (first_i == i)
            for (int j = i; j < n; ++j) size += expert_ids[j] == e;
        first[i] = first_i;
        group_size[i] = first_i == i ? size : 0;
    }
    __syncthreads();

    if (i == 0) {
        int groups = 0, entry = 0;
        for (int j = 0; j < n; ++j) {
            if (first[j] != j) continue;
            const int e = expert_ids[j];
            group_of_first[j] = groups;
            group_start[groups] = entry;
            starts[groups] = entry;
            group_ptr[3 * groups]     = (unsigned long long) (gate + size_t(e) * gate_stride);
            group_ptr[3 * groups + 1] = (unsigned long long) (up   + size_t(e) * up_stride);
            group_ptr[3 * groups + 2] = (unsigned long long) (down + size_t(e) * down_stride);
            entry += group_size[j];
            ++groups;
        }
        starts[groups] = entry;
        *count = groups;
        atomicAdd(calls, 1ULL);
    }
    __syncthreads();

    if (i < n) {
        const int at = group_start[group_of_first[first_i]] + rank;
        destinations[at] = i;
        input_rows[at] = (i / routes) * input_lanes + (input_lanes == 1 ? 0 : i % routes);
    }
}

}  // namespace strata_a3b::detail
