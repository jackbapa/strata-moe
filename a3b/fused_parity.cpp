#include "backend.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<float> oracle(ggml_backend_t backend, ggml_backend_t cpu,
                                 ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down,
                                 const std::vector<float> & input, const std::vector<int32_t> & routes,
                                 int lanes, int nt, int mode) {
    // 0: the original token/lane axes; 1: independent flat rows; 2: decode
    // lanes of one pseudo token. Every reference uses ordinary native ops.
    const int ni = (int) gate->ne[0], nk = (int) routes.size() / nt;
    auto ctx = ggml_init({ggml_tensor_overhead() * 12 + ggml_graph_overhead_custom(32, false), nullptr, true});
    auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ni, mode == 2 ? nk * nt : mode == 1 ? 1 : lanes,
                                  mode == 2 ? 1 : mode == 1 ? nk * nt : nt);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, mode == 2 ? nk * nt : mode == 1 ? 1 : nk,
                                    mode == 2 ? 1 : mode == 1 ? nk * nt : nt);
    auto * g = ggml_mul_mat_id(ctx, gate, x, ids);
    auto * u = ggml_mul_mat_id(ctx, up, x, ids);
    auto * h = ggml_swiglu_split(ctx, g, u);
    auto * y = ggml_mul_mat_id(ctx, down, h, ids);
    auto * graph = ggml_new_graph_custom(ctx, 32, false); ggml_build_forward_expand(graph, y);
    auto buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    std::vector<float> flat;
    if (mode) {
        flat.resize(ni * nk * nt);
        for (int t = 0; t < nt; ++t) for (int k = 0; k < nk; ++k)
            std::copy_n(input.data() + (t * lanes + k % lanes) * ni, ni, flat.data() + (t * nk + k) * ni);
    }
    const auto & values = mode ? flat : input;
    ggml_backend_tensor_set(x, values.data(), 0, values.size() * sizeof(float));
    ggml_backend_tensor_set(ids, routes.data(), 0, routes.size() * sizeof(int32_t));
    std::vector<float> result(ggml_nelements(y));
    if (backend == cpu) {
        if (ggml_backend_graph_compute(cpu, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("native CPU FFN failed");
        ggml_backend_tensor_get(y, result.data(), 0, result.size() * sizeof(float));
    } else {
        auto copy = ggml_backend_graph_copy(backend, graph);
        if (!copy.graph || ggml_backend_graph_compute(backend, copy.graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("native GPU FFN failed");
        ggml_backend_tensor_get(ggml_graph_node(copy.graph, -1), result.data(), 0, result.size() * sizeof(float));
        ggml_backend_graph_copy_free(copy);
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    return result;
}

// Fill slots deterministically with experts [0, slots), without depending on
// test-case history. The complete expert route set protects every used expert
// during admission. Transient PCIe staging must not count as cache admission.
static void warm_hybrid_cache(ggml_backend_t gpu, ggml_backend_t cpu, ggml_backend_t hybrid,
                              ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down, int slots, double pcie_fraction) {
    const int ni = (int) gate->ne[0], ne = (int) gate->ne[2], nk = 8, nt = (ne + nk - 1) / nk;
    auto ctx = ggml_init({ggml_tensor_overhead() * 12 + ggml_graph_overhead_custom(32, false), nullptr, true});
    auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ni, 1, nt);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, nk, nt);
    auto input_buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    std::vector<float> inputs(ni * nt, 0.125f);
    std::vector<int32_t> routes(nk * nt);
    for (int i = 0; i < nk * nt; ++i) routes[i] = i % ne;
    ggml_backend_tensor_set(x, inputs.data(), 0, inputs.size() * sizeof(float));
    ggml_backend_tensor_set(ids, routes.data(), 0, routes.size() * sizeof(int32_t));
    auto * output = strata_a3b::build_fused_moe(ctx, x, gate, up, down, ids);
    if (!output) throw std::runtime_error("hybrid cache warmup builder returned nullptr");
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(ctx, 32, false); ggml_build_forward_expand(graph, output);
    ggml_backend_t backends[] = {gpu, hybrid, cpu};
    auto sched = ggml_backend_sched_new(backends, nullptr, 3, 32, false, true);
    const auto before = strata_a3b::stats();
    if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("hybrid cache warmup failed");
    const auto after = strata_a3b::stats();
    const uint64_t bundle_bytes = gate->nb[2] + up->nb[2] + down->nb[2];
    const uint64_t upload_bytes = after.upload_bytes - before.upload_bytes;
    const uint64_t staging_bytes = after.staging_upload_bytes - before.staging_upload_bytes;
    const uint64_t gpu_misses = pcie_fraction == 1.0 ? uint64_t(ne - slots) :
        uint64_t(std::floor((ne - slots) * pcie_fraction));
    if (upload_bytes < staging_bytes || upload_bytes - staging_bytes != uint64_t(slots) * bundle_bytes ||
        staging_bytes != gpu_misses * bundle_bytes ||
        after.uploads - before.uploads != 3 * (uint64_t(slots) + gpu_misses))
        throw std::runtime_error("hybrid fixture did not seed the requested complete cache slots");
    ggml_backend_sched_free(sched); ggml_backend_buffer_free(input_buf); ggml_free(ctx);
}

static uint64_t raw_bytes_hash(const void * data, size_t size) {
    uint64_t hash = 14695981039346656037ull;
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) { hash ^= bytes[i]; hash *= 1099511628211ull; }
    return hash;
}

static std::vector<float> prefill_native_axis_diagnostic(
        ggml_backend_t gpu, ggml_backend_t cpu, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down,
        const std::vector<float> & inputs, const std::vector<int32_t> & routes, int lanes, int nt, bool & supported) {
    const int ni = (int) gate->ne[0], nk = (int) routes.size() / nt;
    bool repeated_lane = false;
    for (int t = 0; t < nt; ++t) {
        std::set<int32_t> seen;
        for (int lane = 0; lane < nk; ++lane) repeated_lane |= !seen.insert(routes[t * nk + lane]).second;
    }
    supported = !repeated_lane;
    if (supported) return oracle(gpu, cpu, gate, up, down, inputs, routes, lanes, nt, 0);
    // CUDA MMQ's mm_ids_helper stores one occurrence of an expert per token;
    // passing duplicate top-k IDs races its store and leaves compact-index
    // holes. Duplicate routing is intentional fixture coverage, so keep that
    // engine test and explicitly mark this optional diagnostic unsupported.
    // Flat/singleton-lane diagnostics remain valid, and strict GPU correctness
    // still uses only the separate task-layout reference below. NaN means no
    // diagnostic result; it must never be mistaken for a zero error or a pass.
    std::fprintf(stderr, "prefill native-axis diagnostic UNSUPPORTED: repeated top-k IDs violate CUDA MMQ compact-index uniqueness (T=%d lanes=%d)\n", nt, lanes);
    return std::vector<float>(size_t(ni) * routes.size(), NAN);
}

// The ordinary GPU MMQ stream-K decomposition depends on both the compact
// expert count and the gathered row count. A full E32/native-axis reference
// therefore need not round like a staged E8 task. Build the task's mathematical
// reference independently: CPU gather, original Q8-byte copies, ordinary CUDA
// FFN operators, then CPU scatter. No Task, scheduler, backend plan or private
// executor is used. Cache and staged references have their actual capacities,
// including unused stage slots, so the CUDA launch geometry also agrees.
static std::vector<float> prefill_task_layout_oracle(
        ggml_backend_t gpu, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down,
        const std::vector<float> & inputs, const std::vector<int32_t> & routes,
        int lanes, int nt, int slots, int stage_slots,
        const std::vector<uint64_t> & fanout, uint64_t & reference_hash) {
    const int ni = (int) gate->ne[0], ne = (int) gate->ne[2], nk = (int) routes.size() / nt;
    auto require = [](bool value, const char * message) { if (!value) throw std::runtime_error(message); };
    require(ni == 2048 && gate->ne[1] == 512 && up->ne[0] == ni && up->ne[1] == 512 &&
            down->ne[0] == 512 && down->ne[1] == ni && ne == 32 && nk == 8 &&
            stage_slots > 0 && slots >= 0 && slots <= ne,
            "prefill task reference has unexpected model/staging metadata");
    std::vector<float> result(size_t(ni) * routes.size(), NAN);
    std::vector<bool> written(routes.size(), false);
    reference_hash = 14695981039346656037ull;
    auto run = [&](const std::vector<int> & experts, const std::vector<size_t> & channels, int capacity) {
        if (channels.empty()) return;
        require(capacity > 0 && (int) experts.size() <= capacity, "prefill reference capacity is invalid");
        auto wctx = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
        ggml_tensor * source[] = {gate, up, down};
        ggml_tensor * weights[3] = {};
        for (int role = 0; role < 3; ++role) {
            weights[role] = ggml_new_tensor_3d(wctx, GGML_TYPE_Q8_0, source[role]->ne[0], source[role]->ne[1], capacity);
            require(weights[role]->ne[2] == capacity && weights[role]->ne[3] == 1 &&
                    weights[role]->nb[2] == source[role]->nb[2] && ggml_is_contiguous(weights[role]),
                    "prefill reference compact weight metadata differs from Q8 source");
        }
        auto wbuf = ggml_backend_alloc_ctx_tensors(wctx, gpu);
        require(wbuf != nullptr, "prefill task reference CUDA weights allocation failed");
        ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int role = 0; role < 3; ++role) {
            const size_t stride = source[role]->nb[2];
            std::vector<uint8_t> packed(ggml_nbytes(weights[role]), 0), downloaded(packed.size());
            for (size_t slot = 0; slot < experts.size(); ++slot) {
                require(experts[slot] >= 0 && experts[slot] < ne, "prefill reference expert ID is invalid");
                std::memcpy(packed.data() + slot * stride,
                    static_cast<const uint8_t *>(source[role]->data) + size_t(experts[slot]) * stride, stride);
            }
            const uint64_t expected_hash = raw_bytes_hash(packed.data(), packed.size());
            ggml_backend_tensor_set(weights[role], packed.data(), 0, packed.size());
            ggml_backend_tensor_get(weights[role], downloaded.data(), 0, downloaded.size());
            require(raw_bytes_hash(downloaded.data(), downloaded.size()) == expected_hash && downloaded == packed,
                    "prefill reference CUDA upload changed the raw Q8 bytes");
            for (size_t slot = 0; slot < experts.size(); ++slot)
                require(std::memcmp(downloaded.data() + slot * stride,
                    static_cast<const uint8_t *>(source[role]->data) + size_t(experts[slot]) * stride, stride) == 0,
                    "prefill reference compact expert bytes differ from original RAM weights");
            reference_hash ^= expected_hash; reference_hash *= 1099511628211ull;
        }
        const int rows = (int) channels.size();
        std::vector<float> gathered(size_t(ni) * rows);
        std::vector<int32_t> indices(rows);
        for (int row = 0; row < rows; ++row) {
            const size_t channel = channels[row];
            require(channel < routes.size(), "prefill reference channel is out of range");
            const auto found = std::find(experts.begin(), experts.end(), routes[channel]);
            require(found != experts.end(), "prefill reference route is absent from compact weights");
            indices[row] = (int32_t) (found - experts.begin());
            const size_t input_row = channel / nk * lanes + channel % nk % lanes;
            std::copy_n(inputs.data() + input_row * ni, ni, gathered.data() + size_t(row) * ni);
        }
        auto ctx = ggml_init({ggml_tensor_overhead() * 12 + ggml_graph_overhead_custom(32, false), nullptr, true});
        auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ni, 1, rows);
        auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, rows);
        auto * g = ggml_mul_mat_id(ctx, weights[0], x, ids);
        auto * u = ggml_mul_mat_id(ctx, weights[1], x, ids);
        auto * h = ggml_swiglu_split(ctx, g, u);
        auto * y = ggml_mul_mat_id(ctx, weights[2], h, ids);
        auto * graph = ggml_new_graph_custom(ctx, 32, false); ggml_build_forward_expand(graph, y);
        require(x->ne[1] == 1 && x->ne[2] == rows && ids->ne[0] == 1 && ids->ne[1] == rows &&
                y->ne[0] == ni && y->ne[1] == 1 && y->ne[2] == rows,
                "prefill reference does not use gathered live-row task layout");
        auto buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
        require(buf != nullptr, "prefill task reference CUDA graph allocation failed");
        ggml_backend_tensor_set(x, gathered.data(), 0, gathered.size() * sizeof(float));
        ggml_backend_tensor_set(ids, indices.data(), 0, indices.size() * sizeof(int32_t));
        require(ggml_backend_graph_compute(gpu, graph) == GGML_STATUS_SUCCESS,
                "prefill independent native CUDA task reference failed");
        std::vector<float> output(size_t(ni) * rows);
        ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
        for (int row = 0; row < rows; ++row) {
            const size_t channel = channels[row];
            require(!written[channel], "prefill task reference wrote one GPU channel twice");
            written[channel] = true;
            std::copy_n(output.data() + size_t(row) * ni, ni, result.data() + channel * ni);
        }
        ggml_backend_buffer_free(buf); ggml_free(ctx);
        ggml_backend_buffer_free(wbuf); ggml_free(wctx);
    };
    std::vector<int> cached;
    for (int expert = 0; expert < slots; ++expert) cached.push_back(expert);
    std::vector<size_t> hits;
    std::vector<int> staging;
    for (size_t j = 0; j < routes.size(); ++j) {
        const int expert = routes[j];
        require(expert >= 0 && expert < ne, "prefill reference received an invalid source route");
        if (expert < slots) hits.push_back(j);
        else if (fanout[expert] > 2 && std::find(staging.begin(), staging.end(), expert) == staging.end()) staging.push_back(expert);
    }
    run(cached, hits, slots);
    for (size_t first = 0; first < staging.size(); first += stage_slots) {
        const size_t last = std::min(staging.size(), first + size_t(stage_slots));
        const std::vector<int> experts(staging.begin() + first, staging.begin() + last);
        std::vector<size_t> channels;
        for (size_t j = 0; j < routes.size(); ++j)
            if (std::find(experts.begin(), experts.end(), routes[j]) != experts.end()) channels.push_back(j);
        run(experts, channels, stage_slots);
    }
    for (size_t j = 0; j < routes.size(); ++j)
        require(written[j] == (routes[j] < slots || fanout[routes[j]] > 2),
                "prefill task reference did not partition each GPU/CPU channel exactly once");
    return result;
}

// Opt-in short-prefill coverage uses the model's actual Q8 matrix geometry.
// Its oracle graphs remain independent native CPU/GPU FFNs; the expected
// placement is derived from the fixture routes and seeded cache, not backend
// decisions or counters. Keep the existing small-shape fixture path unchanged.
static unsigned prefill_share_cases(ggml_backend_t gpu, ggml_backend_t cpu,
                                    ggml_backend_reg_t reg, const strata_a3b::Config & cfg,
                                    int requested_slots) {
    constexpr int ni = 2048, ff = 512, ne = 32, nk = 8;
    constexpr uint64_t role_bytes = uint64_t(ni) * ff / 32 * 34;
    constexpr uint64_t bundle_bytes = 3 * role_bytes;
    auto require = [](bool value, const char * message) { if (!value) throw std::runtime_error(message); };
    auto weight_ctx = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
    auto * gate = ggml_new_tensor_3d(weight_ctx, GGML_TYPE_Q8_0, ni, ff, ne);
    auto * up = ggml_new_tensor_3d(weight_ctx, GGML_TYPE_Q8_0, ni, ff, ne);
    auto * down = ggml_new_tensor_3d(weight_ctx, GGML_TYPE_Q8_0, ff, ni, ne);
    auto weight_buf = ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx, strata_a3b::expert_buffer_type());
    require(weight_buf != nullptr, "prefill CPU share host weights allocation failed");
    ggml_backend_buffer_set_usage(weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    int seed = 1;
    for (auto * weight : {gate, up, down}) {
        require(weight->nb[2] == role_bytes && ggml_nbytes(weight) == ne * role_bytes,
                "prefill fixture is not complete canonical Q8_0 expert bytes");
        std::vector<float> values(ggml_nelements(weight));
        for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(float(i) * (0.017f + seed * 0.003f)) * 0.05f;
        std::vector<char> packed(ggml_nbytes(weight));
        ggml_quantize_chunk(weight->type, values.data(), packed.data(), 0, weight->ne[1] * ne, weight->ne[0], nullptr);
        ggml_backend_tensor_set(weight, packed.data(), 0, packed.size());
        ++seed;
    }
    auto weights_hash = [&]() {
        uint64_t hash = 14695981039346656037ull;
        for (auto * weight : {gate, up, down}) {
            const auto * bytes = static_cast<const uint8_t *>(weight->data);
            for (size_t i = 0; i < ggml_nbytes(weight); ++i) { hash ^= bytes[i]; hash *= 1099511628211ull; }
        }
        return hash;
    };
    const uint64_t immutable_weights = weights_hash();
    const std::vector<int> slot_cases = requested_slots >= 0 ? std::vector<int>{requested_slots} : std::vector<int>{0, 4, 16};
    unsigned checks = 0;
    for (int slots : slot_cases) {
        strata_a3b::set_cache_budget((cfg.expert_bytes / ne) * size_t(slots));
        auto * hybrid = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
        require(hybrid != nullptr, "prefill CPU share backend init failed");
        strata_a3b::set_cache_admission(true);
        warm_hybrid_cache(gpu, cpu, hybrid, gate, up, down, slots, cfg.pcie_fraction);
        strata_a3b::set_cache_admission(false);
        for (int nt : {9, 32, 256}) for (int lanes : {1, nk}) {
            auto phase = [&](const char * text) {
                std::fprintf(stderr, "prefill fixture T=%d lanes=%d slots=%d phase=%s\n", nt, lanes, slots, text);
                std::fflush(stderr);
            };
            phase("inputs-and-independent-counts");
            std::vector<float> inputs(size_t(ni) * lanes * nt);
            for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = std::cos(float(i) * 0.071f) * 0.25f;
            std::vector<int32_t> routes(nk * nt);
            constexpr int base[nk] = {0, 16, 17, 18, 19, 20, 21, 16};
            for (size_t j = 0; j < routes.size(); ++j) routes[j] = base[j % nk];
            // Two scarce missed groups: one job and exactly two jobs. The
            // repeated ID 31 shares one input when lanes=1, and uses two
            // independent input lanes when lanes=8. Expert 27 is just above
            // the CPU threshold, so all three of its jobs must stay on GPU.
            routes[0] = 30; routes[1] = 31; routes[2] = 31;
            routes[3] = routes[nk + 3] = routes[2 * nk + 3] = 27;
            std::vector<uint64_t> fanout(ne, 0);
            for (int expert : routes) ++fanout[expert];
            require(fanout[30] == 1 && fanout[31] == 2 && fanout[27] == 3,
                    "prefill fixture lost the fanout boundary cases");
            uint64_t cpu_jobs = 0, gpu_jobs = 0, hit_jobs = 0, staged_jobs = 0, cpu_experts = 0, staged_experts = 0;
            std::set<std::pair<int, int>> cpu_inputs;
            for (int e = 0; e < ne; ++e) if (fanout[e]) {
                if (e < slots) hit_jobs += fanout[e];
                else if (fanout[e] <= 2) { cpu_jobs += fanout[e]; ++cpu_experts; }
                else { staged_jobs += fanout[e]; ++staged_experts; }
            }
            gpu_jobs = hit_jobs + staged_jobs;
            for (size_t j = 0; j < routes.size(); ++j)
                if (routes[j] >= slots && fanout[routes[j]] <= 2)
                    cpu_inputs.emplace(int(j) / nk, (int(j) % nk) % lanes);
            require(cpu_jobs == 3 && cpu_experts == 2 && staged_jobs > 0 && staged_experts > 0 &&
                    cpu_jobs + gpu_jobs == routes.size() && (!slots || hit_jobs > 0),
                    "prefill fixture failed to cover complete CPU/staged/hit groups");
            phase("CPU-native-oracle");
            const auto cpu_ref = oracle(cpu, cpu, gate, up, down, inputs, routes, lanes, nt, 0);
            phase("GPU-native-axis-diagnostic");
            bool native_axis_supported = false;
            const auto gpu_ref = prefill_native_axis_diagnostic(gpu, cpu, gate, up, down, inputs, routes, lanes, nt, native_axis_supported);
            phase("GPU-flat-diagnostic");
            const auto flat_ref = oracle(gpu, cpu, gate, up, down, inputs, routes, lanes, nt, 1);
            phase("GPU-singleton-lanes-diagnostic");
            const auto lanes_ref = oracle(gpu, cpu, gate, up, down, inputs, routes, lanes, nt, 2);
            phase("GPU-independent-task-layout-oracle");
            uint64_t reference_hash = 0;
            const auto task_ref = prefill_task_layout_oracle(gpu, gate, up, down, inputs, routes, lanes, nt,
                slots, cfg.expert_staging_slots, fanout, reference_hash);
            auto ctx = ggml_init({ggml_tensor_overhead() * 12 + ggml_graph_overhead_custom(32, false), nullptr, true});
            auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ni, lanes, nt);
            auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, nk, nt);
            auto input_buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
            require(input_buf != nullptr, "prefill CPU share input allocation failed");
            auto uploaded_inputs = inputs;
            for (auto & value : uploaded_inputs) value *= 2.f;
            ggml_backend_tensor_set(x, uploaded_inputs.data(), 0, uploaded_inputs.size() * sizeof(float));
            ggml_backend_tensor_set(ids, routes.data(), 0, routes.size() * sizeof(int32_t));
            auto * routed_x = ggml_scale(ctx, x, 0.5f);
            auto * routed_ids = ggml_dup(ctx, ids);
            auto * fused = strata_a3b::build_fused_moe(ctx, routed_x, gate, up, down, routed_ids);
            require(fused != nullptr, "prefill CPU share builder returned nullptr");
            auto * output = ggml_scale(ctx, fused, 2.f); ggml_set_output(output);
            auto * graph = ggml_new_graph_custom(ctx, 32, false); ggml_build_forward_expand(graph, output);
            ggml_backend_t backends[] = {gpu, hybrid, cpu};
            auto sched = ggml_backend_sched_new(backends, nullptr, 3, 32, false, true);
            phase("actual-prefill-share-engine");
            const auto before = strata_a3b::stats();
            require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
                    "prefill CPU share scheduler failed");
            const auto after = strata_a3b::stats();
            phase("strict-counters-and-numerical-comparison");
            require(ggml_backend_sched_get_tensor_backend(sched, fused) == hybrid &&
                    ggml_backend_sched_get_tensor_backend(sched, routed_x) == gpu &&
                    ggml_backend_sched_get_tensor_backend(sched, routed_ids) == gpu &&
                    ggml_backend_sched_get_tensor_backend(sched, output) == gpu,
                    "prefill fixture did not retain CUDA/custom/CUDA placement");
            require(after.fused_calls - before.fused_calls == 1 && after.hybrid_calls - before.hybrid_calls == 1 &&
                    after.prefill_share_calls - before.prefill_share_calls == 1 &&
                    after.prefill_share_cpu_jobs - before.prefill_share_cpu_jobs == cpu_jobs &&
                    after.prefill_share_cpu_experts - before.prefill_share_cpu_experts == cpu_experts &&
                    after.route_reads - before.route_reads == 1,
                    "prefill complete-group dispatch/call accounting mismatch");
            require(after.fused_cpu_rows - before.fused_cpu_rows == cpu_jobs &&
                    after.fused_gpu_rows - before.fused_gpu_rows == gpu_jobs &&
                    after.cpu_rows - before.cpu_rows == 3 * cpu_jobs && after.gpu_rows - before.gpu_rows == 3 * gpu_jobs &&
                    after.hits - before.hits == 3 * hit_jobs && after.misses - before.misses == 3 * (cpu_jobs + staged_jobs) &&
                    after.gpu_only_rows - before.gpu_only_rows == gpu_jobs &&
                    after.hybrid_gpu_miss_rows - before.hybrid_gpu_miss_rows == staged_jobs &&
                    after.hybrid_gpu_miss_experts - before.hybrid_gpu_miss_experts == staged_experts,
                    "prefill CPU/GPU/cache-hit partition lost or duplicated expert rows");
            require(after.staged_rows - before.staged_rows == staged_jobs &&
                    after.staged_experts - before.staged_experts == staged_experts &&
                    after.staging_batches - before.staging_batches ==
                        (staged_experts + cfg.expert_staging_slots - 1) / cfg.expert_staging_slots &&
                    after.uploads - before.uploads == 3 * staged_experts &&
                    after.upload_bytes - before.upload_bytes == staged_experts * bundle_bytes &&
                    after.staging_upload_bytes - before.staging_upload_bytes == staged_experts * bundle_bytes &&
                    after.cache_admissions == before.cache_admissions,
                    "prefill staged uploads are not the original complete three-role Q8 bundles");
            require(after.cpu_pool_calls - before.cpu_pool_calls == 1 &&
                    after.cpu_pool_jobs - before.cpu_pool_jobs == cpu_jobs &&
                    after.cpu_pool_unique_inputs - before.cpu_pool_unique_inputs == cpu_inputs.size() &&
                    after.cpu_pool_unique_experts - before.cpu_pool_unique_experts == cpu_experts,
                    "prefill sparse CPU pool jobs/input sharing/group accounting mismatch");
            std::vector<float> actual(cpu_ref.size());
            ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
            size_t gpu_bitwise_differences = 0;
            float max_native_axis_delta = native_axis_supported ? 0.0f : NAN;
            float max_native_flat_delta = 0.0f, max_native_lane_delta = 0.0f;
            for (size_t i = 0; i < actual.size(); ++i) {
                const int expert = routes[i / ni];
                const bool on_cpu = expert >= slots && fanout[expert] <= 2;
                const float expected = 2 * (on_cpu ? cpu_ref[i] : task_ref[i]);
                const float error = std::abs(actual[i] - expected);
                if (!on_cpu) {
                    gpu_bitwise_differences += std::memcmp(&actual[i], &expected, sizeof(float)) != 0;
                    if (native_axis_supported) max_native_axis_delta = std::max(max_native_axis_delta, std::abs(actual[i] - 2 * gpu_ref[i]));
                    max_native_flat_delta = std::max(max_native_flat_delta, std::abs(actual[i] - 2 * flat_ref[i]));
                    max_native_lane_delta = std::max(max_native_lane_delta, std::abs(actual[i] - 2 * lanes_ref[i]));
                }
                const float tolerance = 4e-5f * (1 + std::abs(cpu_ref[i]));
                if (!std::isfinite(actual[i]) || !std::isfinite(expected) || error > tolerance) {
                    std::fprintf(stderr, "prefill mismatch T=%d lanes=%d slots=%d expert=%d CPUowned=%d index=%zu actual=%.9g expected=%.9g CPU=%.9g nativeGPU=%.9g flat=%.9g lanes=%.9g taskGPU=%.9g error=%.9g\n",
                        nt, lanes, slots, expert, on_cpu, i, actual[i], expected, 2 * cpu_ref[i], 2 * gpu_ref[i], 2 * flat_ref[i], 2 * lanes_ref[i], 2 * task_ref[i], error);
                    throw std::runtime_error("prefill CPU share differs from its independent native executor oracle");
                }
            }
            std::printf("prefill reference T=%d lanes=%d slots=%d raw_GPU_weight_FNV64=%016llx task_GPU_bitwise_differences=%zu native_axes_supported=%d native_layout_diagnostics_max_abs: axes=%.9g flat=%.9g singleton_lanes=%.9g\n",
                nt, lanes, slots, (unsigned long long) reference_hash, gpu_bitwise_differences,
                native_axis_supported, max_native_axis_delta, max_native_flat_delta, max_native_lane_delta);
            std::printf("prefill CPU share Q8_0 H2048 F512 E32 T=%d lanes=%d slots=%d CPUjobs=%llu CPUgroups=%llu GPUhits=%llu GPUstaged=%llu stagedgroups=%llu raw_expert_bytes=%llu CPU_unique_inputs=%zu native_oracles/complete_groups/quantized_bytes PASS\n",
                nt, lanes, slots, (unsigned long long) cpu_jobs, (unsigned long long) cpu_experts,
                (unsigned long long) hit_jobs, (unsigned long long) staged_jobs, (unsigned long long) staged_experts,
                (unsigned long long) (staged_experts * bundle_bytes), cpu_inputs.size());
            ++checks;
            ggml_backend_sched_free(sched); ggml_backend_buffer_free(input_buf); ggml_free(ctx);
        }
        ggml_backend_free(hybrid);
    }
    require(weights_hash() == immutable_weights, "prefill CPU share changed the original host Q8 bytes");
    strata_a3b::set_cache_admission(true);
    ggml_backend_buffer_free(weight_buf); ggml_free(weight_ctx);
    return checks;
}

int main(int argc, char ** argv) {
    try {
        bool cpu_only = false, pageable = false, hybrid_experts = false, cpu_expert_pool = false;
        bool q8_only = false, async_events = false, prefill_cpu_share = false;
        int threads = 4, requested_slots = -1;
        double pcie_fraction = 0.0;
        for (int i = 1; i < argc; ++i) {
            auto value = [&]() -> const char * {
                if (++i >= argc) throw std::runtime_error("missing fused parity argument value");
                return argv[i];
            };
            if (std::strcmp(argv[i], "--cpu") == 0) cpu_only = true;
            else if (std::strcmp(argv[i], "--pageable") == 0) pageable = true;
            else if (std::strcmp(argv[i], "--hybrid-experts") == 0) hybrid_experts = true;
            else if (std::strcmp(argv[i], "--cpu-expert-pool") == 0) cpu_expert_pool = true;
            else if (std::strcmp(argv[i], "--prefill-cpu-share") == 0) prefill_cpu_share = hybrid_experts = cpu_expert_pool = true;
            else if (std::strcmp(argv[i], "--q8-only") == 0) q8_only = true;
            else if (std::strcmp(argv[i], "--async-expert-events") == 0) async_events = true;
            else if (std::strcmp(argv[i], "--pcie-frac") == 0) {
                const std::string text = value(); size_t parsed = 0;
                pcie_fraction = std::stod(text, &parsed);
                if (parsed != text.size()) throw std::runtime_error("invalid --pcie-frac");
            } else if (std::strcmp(argv[i], "--threads") == 0 || std::strcmp(argv[i], "--cache-slots") == 0) {
                const bool is_threads = std::strcmp(argv[i], "--threads") == 0;
                const std::string text = value(); size_t parsed = 0;
                const int n = std::stoi(text, &parsed);
                if (parsed != text.size()) throw std::runtime_error("invalid fused parity integer");
                if (is_threads) threads = n; else requested_slots = n;
            }
            else throw std::runtime_error("unknown fused parity argument");
        }
        if (threads < 1 || threads > 4096 || !std::isfinite(pcie_fraction) || pcie_fraction < 0 || pcie_fraction > 1 ||
            requested_slots < -1 || requested_slots > 16 ||
            (cpu_only && hybrid_experts) || ((cpu_expert_pool || pcie_fraction > 0 || requested_slots >= 0 || async_events) && !hybrid_experts))
            throw std::runtime_error("hybrid fixture requires GPU hybrid mode, threads >= 1, PCIe fraction 0..1 and cache slots 0..16");
        ggml_backend_load_all();
        auto device = cpu_only ? nullptr : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (hybrid_experts && !device) { std::puts("SKIP: hybrid experts require a GPU"); return 77; }
        auto gpu = device ? ggml_backend_dev_init(device, nullptr) : nullptr;
        auto cpu = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(cpu, threads);
        strata_a3b::Config cfg;
        cfg.gpu = device; cfg.expert_bytes = 1ull << 30; cfg.threads = threads; cfg.pageable_experts = pageable;
        cfg.fused_moe = true; cfg.exact_verify = true; cfg.cache_graph_routes = true; cfg.preserve_verify_layout = true;
        cfg.hybrid_experts = hybrid_experts; cfg.cpu_expert_pool = cpu_expert_pool;
        cfg.pcie_fraction = pcie_fraction; cfg.async_expert_events = async_events;
        cfg.prefill_cpu_share = prefill_cpu_share;
        if (hybrid_experts) cfg.native_exact_max_tokens = 8;
        auto reg = strata_a3b::register_backend(cfg);
        auto hybrid = hybrid_experts ? nullptr : ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
        if (!cpu || (!hybrid_experts && !hybrid)) throw std::runtime_error("fused backend init failed");
        if (prefill_cpu_share) {
            const auto checks = prefill_share_cases(gpu, cpu, reg, cfg, requested_slots);
            ggml_backend_free(cpu); ggml_backend_free(gpu);
            std::printf("PASS prefill CPU share: %u cases, original Q8 bytes, rare CPU/frequent GPU complete groups, native CPU/GPU/flat/lane oracles\n", checks);
            return 0;
        }
        unsigned checks = 0;
        struct Types { ggml_type gu, down; };
        for (auto types : {Types{GGML_TYPE_F32, GGML_TYPE_F32}, Types{GGML_TYPE_Q4_K, GGML_TYPE_Q6_K},
                           Types{GGML_TYPE_Q5_K, GGML_TYPE_Q6_K}, Types{GGML_TYPE_Q6_K, GGML_TYPE_Q6_K},
                           Types{GGML_TYPE_Q8_0, GGML_TYPE_Q8_0}}) {
            if (q8_only && types.gu != GGML_TYPE_Q8_0) continue;
            constexpr int ni = 512, ff = 256, ne = 16, nk = 8;
            auto weight_ctx = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
            auto * gate = ggml_new_tensor_3d(weight_ctx, types.gu, ni, ff, ne);
            auto * up = ggml_new_tensor_3d(weight_ctx, types.gu, ni, ff, ne);
            auto * down = ggml_new_tensor_3d(weight_ctx, types.down, ff, ni, ne);
            auto weight_buf = ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx, strata_a3b::expert_buffer_type());
            ggml_backend_buffer_set_usage(weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            int seed = 1;
            for (auto * weight : {gate, up, down}) {
                std::vector<float> values(ggml_nelements(weight));
                for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(float(i) * (0.017f + seed * 0.003f)) * 0.05f;
                std::vector<char> packed(ggml_nbytes(weight));
                if (weight->type == GGML_TYPE_F32) std::memcpy(packed.data(), values.data(), packed.size());
                else ggml_quantize_chunk(weight->type, values.data(), packed.data(), 0,
                                         weight->ne[1] * ne, weight->ne[0], nullptr);
                ggml_backend_tensor_set(weight, packed.data(), 0, packed.size()); ++seed;
            }
            const std::vector<int> cache_slots = hybrid_experts ?
                (requested_slots >= 0 ? std::vector<int>{requested_slots} : std::vector<int>{0, 4, ne}) : std::vector<int>{4};
            for (int slots : cache_slots) {
                strata_a3b::set_cache_budget((cfg.expert_bytes / ne) * size_t(slots));
                if (hybrid_experts) {
                    hybrid = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
                    if (!hybrid) throw std::runtime_error("hybrid cohort backend init failed");
                    strata_a3b::set_cache_admission(true);
                    warm_hybrid_cache(gpu, cpu, hybrid, gate, up, down, slots, pcie_fraction);
                }
            for (int pass = 0; pass < (hybrid_experts ? 16 : 6); ++pass) {
                static constexpr int hybrid_tokens[] = {1, 2, 7, 8};
                const int nt = hybrid_experts ? hybrid_tokens[pass % 4] : pass == 1 ? 2 : pass == 2 ? 4 : pass == 3 ? 9 : 1;
                const int lanes = hybrid_experts ? (pass / 4 % 2 ? nk : 1) : pass % 2 ? nk : 1;
                const bool duplicates = hybrid_experts && pass >= 8;
                const bool frozen = hybrid_experts || pass == 2;
                strata_a3b::set_cache_admission(!frozen);
                std::vector<float> inputs(ni * lanes * nt);
                for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = std::cos(float(i) * 0.071f) * 0.25f;
                std::vector<int32_t> routes(nk * nt);
                for (int t = 0; t < nt; ++t) for (int lane = 0; lane < nk; ++lane)
                    routes[size_t(t * nk + lane)] = hybrid_experts ?
                        (t % 3 * 4 + (duplicates && lane == nk - 1 ? lane - 1 : lane)) % ne : (t * nk + lane + pass * 3) % ne;
                auto cpu_ref = oracle(cpu, cpu, gate, up, down, inputs, routes, lanes, nt, 0);
                auto gpu_ref = gpu ? oracle(gpu, cpu, gate, up, down, inputs, routes, lanes, nt, 0) : cpu_ref;
                auto flat_ref = gpu ? oracle(gpu, cpu, gate, up, down, inputs, routes, lanes, nt, 1) : cpu_ref;
                auto lanes_ref = gpu && types.gu != GGML_TYPE_F32 ?
                    oracle(gpu, cpu, gate, up, down, inputs, routes, lanes, nt, 2) : cpu_ref;
                auto ctx = ggml_init({ggml_tensor_overhead() * 12 + ggml_graph_overhead_custom(32, false), nullptr, true});
                auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ni, lanes, nt);
                auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, nk, nt);
                auto input_buf = ggml_backend_alloc_ctx_tensors(ctx, gpu ? gpu : cpu);
                auto uploaded_inputs = inputs;
                if (hybrid_experts) for (auto & value : uploaded_inputs) value *= 2.f;
                ggml_backend_tensor_set(x, uploaded_inputs.data(), 0, uploaded_inputs.size() * sizeof(float));
                ggml_backend_tensor_set(ids, routes.data(), 0, routes.size() * sizeof(int32_t));
                // Real CUDA producers exercise the input/route event boundary.
                // Multiplication by powers of two preserves the oracle inputs.
                auto * routed_x = hybrid_experts ? ggml_scale(ctx, x, 0.5f) : x;
                auto * routed_ids = hybrid_experts ? ggml_dup(ctx, ids) : ids;
                auto * fused = strata_a3b::build_fused_moe(ctx, routed_x, gate, up, down, routed_ids);
                if (!fused) throw std::runtime_error("compatible fused builder returned nullptr");
                auto * output = ggml_scale(ctx, fused, 2.f); ggml_set_output(output);
                auto * graph = ggml_new_graph_custom(ctx, 32, false); ggml_build_forward_expand(graph, output);
                ggml_backend_t backends[] = {gpu ? gpu : hybrid, gpu ? hybrid : cpu, cpu};
                auto sched = ggml_backend_sched_new(backends, nullptr, gpu ? 3 : 2, 32, false, true);
                const auto before = strata_a3b::stats();
                if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("fused scheduler failed");
                const auto after = strata_a3b::stats();
                if (ggml_backend_sched_get_tensor_backend(sched, fused) != hybrid)
                    throw std::runtime_error("scheduler bypassed fused backend");
                if (hybrid_experts && (ggml_backend_sched_get_tensor_backend(sched, routed_x) != gpu ||
                    ggml_backend_sched_get_tensor_backend(sched, routed_ids) != gpu ||
                    ggml_backend_sched_get_tensor_backend(sched, output) != gpu))
                    throw std::runtime_error("hybrid fixture did not retain CUDA/custom/CUDA placement");
                if (after.fused_calls != before.fused_calls + 1 ||
                    after.fused_cpu_rows - before.fused_cpu_rows + after.fused_gpu_rows - before.fused_gpu_rows != routes.size())
                    throw std::runtime_error("complete routed experts were not partitioned once");
                if (gpu && after.route_reads != before.route_reads + 1)
                    throw std::runtime_error("fused experts did not read routes exactly once");
                if (frozen && (after.upload_bytes - before.upload_bytes !=
                               after.staging_upload_bytes - before.staging_upload_bytes ||
                               after.cache_admissions != before.cache_admissions))
                    throw std::runtime_error("frozen fused placement admitted persistent weights");
                if (!hybrid_experts && gpu && nt <= 4 && (after.fused_cpu_rows == before.fused_cpu_rows || after.fused_gpu_rows == before.fused_gpu_rows))
                    throw std::runtime_error("fused CPU/GPU split was not exercised");
                if (!hybrid_experts && gpu && nt > 4 && after.fused_cpu_rows != before.fused_cpu_rows)
                    throw std::runtime_error("fused long prefill did not stream");
                if (hybrid_experts) {
                    std::vector<int> missing;
                    for (int expert : routes) if (expert >= slots && std::find(missing.begin(), missing.end(), expert) == missing.end())
                        missing.push_back(expert);
                    const size_t offloaded = pcie_fraction == 1.0 ? missing.size() : size_t(std::floor(missing.size() * pcie_fraction));
                    std::vector<bool> gpu_miss(ne, false);
                    for (size_t j = missing.size() - offloaded; j < missing.size(); ++j) gpu_miss[size_t(missing[j])] = true;
                    uint64_t expected_cpu = 0, expected_staged = 0;
                    std::set<std::pair<int, int>> cpu_inputs;
                    for (size_t j = 0; j < routes.size(); ++j) {
                        if (routes[j] < slots) continue;
                        if (gpu_miss[size_t(routes[j])]) ++expected_staged;
                        else { ++expected_cpu; cpu_inputs.emplace(int(j) / nk, (int(j) % nk) % lanes); }
                    }
                    const bool pool_expected = cpu_expert_pool && types.gu == GGML_TYPE_Q8_0 && types.down == GGML_TYPE_Q8_0 && expected_cpu > 0;
                    if (after.hybrid_calls - before.hybrid_calls != 1 ||
                        after.fused_cpu_rows - before.fused_cpu_rows != expected_cpu ||
                        after.fused_gpu_rows - before.fused_gpu_rows != routes.size() - expected_cpu ||
                        after.hybrid_gpu_miss_rows - before.hybrid_gpu_miss_rows != expected_staged ||
                        after.hybrid_gpu_miss_experts - before.hybrid_gpu_miss_experts != offloaded ||
                        after.staged_rows - before.staged_rows != expected_staged ||
                        after.staged_experts - before.staged_experts != offloaded ||
                        after.uploads - before.uploads != 3 * offloaded ||
                        after.staging_upload_bytes - before.staging_upload_bytes !=
                            offloaded * (gate->nb[2] + up->nb[2] + down->nb[2]))
                        throw std::runtime_error("hybrid expert groups/rows did not follow cache and PCIe fraction exactly once");
                    if (after.cpu_pool_calls - before.cpu_pool_calls != uint64_t(pool_expected) ||
                        after.cpu_pool_jobs - before.cpu_pool_jobs != (pool_expected ? expected_cpu : 0) ||
                        after.cpu_pool_unique_inputs - before.cpu_pool_unique_inputs != (pool_expected ? cpu_inputs.size() : 0))
                        throw std::runtime_error("hybrid CPU pool input sharing/job accounting mismatch");
                }
                std::vector<float> actual(cpu_ref.size());
                ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
                for (size_t i = 0; i < actual.size(); ++i) {
                    const float error = std::min({std::abs(actual[i] - 2 * cpu_ref[i]), std::abs(actual[i] - 2 * gpu_ref[i]),
                                                 std::abs(actual[i] - 2 * flat_ref[i]), std::abs(actual[i] - 2 * lanes_ref[i])});
                    const float tolerance = gpu ? 4e-5f * (1 + std::abs(cpu_ref[i])) : 1e-6f;
                    if (!std::isfinite(actual[i]) || error > tolerance) {
                        std::fprintf(stderr, "fused mismatch index=%zu actual=%.9g CPU=%.9g GPU=%.9g flat=%.9g lanes=%.9g error=%.9g\n",
                                     i, actual[i], 2 * cpu_ref[i], 2 * gpu_ref[i], 2 * flat_ref[i], 2 * lanes_ref[i], error);
                        throw std::runtime_error("fused complete FFN differs from native oracles");
                    }
                }
                std::printf("%s/%s nt=%d lanes=%d frozen=%d slots=%d duplicate=%d pcie_frac=%.2f CPU_rows=%llu GPU_rows=%llu staged_rows=%llu pool_jobs=%llu PASS\n",
                            ggml_type_name(types.gu), ggml_type_name(types.down), nt, lanes, frozen, slots, duplicates, pcie_fraction,
                            (unsigned long long) (after.fused_cpu_rows - before.fused_cpu_rows),
                            (unsigned long long) (after.fused_gpu_rows - before.fused_gpu_rows),
                            (unsigned long long) (after.staged_rows - before.staged_rows),
                            (unsigned long long) (after.cpu_pool_jobs - before.cpu_pool_jobs));
                ++checks;
                ggml_backend_sched_free(sched); ggml_backend_buffer_free(input_buf); ggml_free(ctx);
            }
                if (hybrid_experts) { ggml_backend_free(hybrid); hybrid = nullptr; }
            }
            ggml_backend_buffer_free(weight_buf); ggml_free(weight_ctx);
        }
        strata_a3b::set_cache_admission(true);
        if (hybrid) ggml_backend_free(hybrid); ggml_backend_free(cpu); if (gpu) ggml_backend_free(gpu);
        std::printf("PASS fused complete experts: %u cases, native CPU/GPU oracles, split, streaming, frozen placement%s\n",
                    checks, hybrid_experts ? "; hybrid PCIe groups, CPU pool sharing, zero/partial/full cache, T1/2/7/8 and duplicates" : "");
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
