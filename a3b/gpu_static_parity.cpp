// Independent real-geometry Q8 oracles for opt-in GPU and CPU mailbox experts.
// This fixture does not load a model or bypass the production GGUF/hash checks;
// its manual cache plan is confined to one synthetic blk.0 layer.
#include "backend.hpp"
#include "cache_profile.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cuda-strata-moe.h"
#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int hidden = 2048, ffn = 512, experts = 256, top_k = 8;
struct Context {
    ggml_context * p = nullptr;
    explicit Context(size_t tensors = 64) {
        p = ggml_init({ggml_tensor_overhead() * tensors + 2 * ggml_graph_overhead_custom(128, false), nullptr, true});
        if (!p) throw std::bad_alloc();
    }
    ~Context() { if (p) ggml_free(p); }
};
struct Buffer {
    ggml_backend_buffer_t p = nullptr;
    ~Buffer() { if (p) ggml_backend_buffer_free(p); }
};
struct Backend {
    ggml_backend_t p = nullptr;
    ~Backend() { if (p) ggml_backend_free(p); }
};
struct Scheduler {
    ggml_backend_sched_t p = nullptr;
    ~Scheduler() { if (p) { ggml_backend_sched_synchronize(p); ggml_backend_sched_free(p); } }
};

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

// Every expected token uses a separate ordinary CUDA graph and full resident
// expert weights. It never calls the custom factory, mapped-miss kernel, cache
// or CPU expert implementation, and there is no tolerance-based alternative.
std::vector<float> singleton_oracle(ggml_backend_t gpu, const std::array<ggml_tensor *, 3> & weights,
        const std::vector<float> & values, const std::vector<int32_t> & ids, int lanes, int tokens) {
    std::vector<float> expected(hidden * top_k * tokens);
    for (int token = 0; token < tokens; ++token) {
        Context context;
        auto * x = ggml_new_tensor_3d(context.p, GGML_TYPE_F32, hidden, lanes, 1);
        auto * routed = ggml_new_tensor_2d(context.p, GGML_TYPE_I32, top_k, 1);
        auto * gate = ggml_mul_mat_id(context.p, weights[0], x, routed);
        auto * up = ggml_mul_mat_id(context.p, weights[1], x, routed);
        auto * out = ggml_mul_mat_id(context.p, weights[2], ggml_swiglu_split(context.p, gate, up), routed);
        auto * graph = ggml_new_graph_custom(context.p, 128, false);
        ggml_build_forward_expand(graph, out);
        Buffer buffer; buffer.p = ggml_backend_alloc_ctx_tensors(context.p, gpu);
        require(buffer.p != nullptr, "singleton CUDA oracle allocation failed");
        ggml_backend_tensor_set(x, values.data() + token * hidden * lanes, 0, hidden * lanes * sizeof(float));
        ggml_backend_tensor_set(routed, ids.data() + token * top_k, 0, top_k * sizeof(int32_t));
        require(ggml_backend_graph_compute(gpu, graph) == GGML_STATUS_SUCCESS, "singleton CUDA oracle failed");
        ggml_backend_tensor_get(out, expected.data() + token * hidden * top_k, 0, hidden * top_k * sizeof(float));
    }
    return expected;
}

// Independent CPU arithmetic reference: one complete expert job at a time.
// It uses no CpuExpertPool, worker dispatch, production callback or CPU graph.
void serial_cpu_expert(const std::array<ggml_tensor *, 3> & weights,
        const float * input, int expert, float * output) {
    const auto * traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    require(traits && traits->from_float && traits->vec_dot, "CPU Q8 oracle traits unavailable");
    std::vector<uint8_t> activation(ggml_row_size(GGML_TYPE_Q8_0, hidden));
    std::vector<uint8_t> hidden_q(ggml_row_size(GGML_TYPE_Q8_0, ffn));
    std::vector<float> ff(ffn);
    traits->from_float(input, activation.data(), hidden);
    const auto * gate = static_cast<const uint8_t *>(weights[0]->data) + size_t(expert) * weights[0]->nb[2];
    const auto * up = static_cast<const uint8_t *>(weights[1]->data) + size_t(expert) * weights[1]->nb[2];
    const auto * down = static_cast<const uint8_t *>(weights[2]->data) + size_t(expert) * weights[2]->nb[2];
    for (int row = 0; row < ffn; ++row) {
        float g = 0, u = 0;
        traits->vec_dot(hidden, &g, 0, gate + size_t(row) * weights[0]->nb[1], 0, activation.data(), 0, 1);
        traits->vec_dot(hidden, &u, 0, up + size_t(row) * weights[1]->nb[1], 0, activation.data(), 0, 1);
        ff[row] = (g / (1.f + std::exp(-g))) * u;
    }
    traits->from_float(ff.data(), hidden_q.data(), ffn);
    for (int row = 0; row < hidden; ++row)
        traits->vec_dot(ffn, output + row, 0, down + size_t(row) * weights[2]->nb[1], 0, hidden_q.data(), 0, 1);
}

void replace_cpu_miss_oracle(std::vector<float> & expected,
        const std::array<ggml_tensor *, 3> & weights, const std::vector<float> & input,
        const std::vector<int32_t> & ids, const strata_a3b::ExpertCachePlan & plan, int lanes, int tokens) {
    std::array<bool, experts> cached{};
    for (int expert : plan.expert_ids[0]) cached[expert] = true;
    for (int token = 0; token < tokens; ++token) for (int lane = 0; lane < top_k; ++lane) {
        const int expert = ids[token * top_k + lane];
        require(expert >= 0 && expert < experts, "hybrid oracle requires valid IDs");
        if (!cached[expert]) serial_cpu_expert(weights,
            input.data() + (token * lanes + lane % lanes) * hidden, expert,
            expected.data() + (token * top_k + lane) * hidden);
    }
}

strata_a3b::ExpertCachePlan manual_plan(int slots) {
    strata_a3b::ExpertCachePlan plan;
    plan.budget_bytes = plan.planned_bytes = uint64_t(slots) * strata_a3b::expert_profile_bundle_bytes;
    plan.selected_experts = slots;
    plan.total_score = plan.covered_score = slots;
    plan.coverage = slots ? 1 : 0;
    for (int slot = 0; slot < slots; ++slot)
        plan.expert_ids[0].push_back((slot * 73 + 19) % experts);
    // The other 39 layers deliberately remain empty. No fake model digest is
    // needed: set_cache_plan consumes an already validated plan, while real
    // model/file identity validation belongs to main's whole-model tests.
    return plan;
}

void initialize_weights(ggml_backend_t gpu, const std::array<ggml_tensor *, 3> & host,
        const std::array<ggml_tensor *, 3> & resident) {
    for (int projection = 0; projection < 3; ++projection) {
        auto * tensor = host[projection];
        // Quantize one expert at a time: the 256-expert fixture does not need
        // a temporary 1 GiB float tensor. Every expert has distinct data.
        std::vector<float> values(tensor->ne[0] * tensor->ne[1]);
        std::vector<unsigned char> packed(tensor->nb[2]);
        for (int expert = 0; expert < experts; ++expert) {
            uint32_t state = 0x9e3779b9u ^ uint32_t(projection * 65537 + expert * 9973 + 1);
            for (auto & value : values) {
                state ^= state << 13; state ^= state >> 17; state ^= state << 5;
                value = (int(state & 1023) - 512) * (1.0f / 8192);
            }
            const auto written = ggml_quantize_chunk(GGML_TYPE_Q8_0, values.data(), packed.data(), 0,
                                                     tensor->ne[1], tensor->ne[0], nullptr);
            require(written == packed.size(), "Q8 oracle fixture packing size mismatch");
            const size_t offset = expert * tensor->nb[2];
            ggml_backend_tensor_set(tensor, packed.data(), offset, packed.size());
            ggml_backend_tensor_set(resident[projection], packed.data(), offset, packed.size());
        }
    }
    ggml_backend_synchronize(gpu);
}

std::vector<int32_t> route_pattern(const strata_a3b::ExpertCachePlan & plan, int tokens, int shift, bool duplicates) {
    const auto & hits = plan.expert_ids[0];
    std::array<bool, experts> cached{};
    for (int expert : hits) cached[expert] = true;
    std::vector<int> misses;
    for (int expert = 0; expert < experts; ++expert) if (!cached[expert]) misses.push_back(expert);
    std::vector<int32_t> routes(top_k * tokens);
    for (int token = 0; token < tokens; ++token) for (int lane = 0; lane < top_k; ++lane) {
        const bool use_hit = !hits.empty() && (misses.empty() || lane % 2 == 0);
        const auto & pool = use_hit ? hits : misses;
        routes[token * top_k + lane] = pool[(token * 29 + lane * 7 + shift) % pool.size()];
        if (duplicates && lane == top_k - 1) routes[token * top_k + lane] = routes[token * top_k + lane - 1];
    }
    // Both valid bounds occur even with a sparse, permuted cache mapping.
    if (!duplicates) { routes.front() = 0; routes.back() = experts - 1; }
    return routes;
}

void assert_exact(const std::vector<float> & actual, const std::vector<float> & expected,
        int slots, int lanes, int tokens, bool duplicates, int repeat) {
    require(actual.size() == expected.size(), "static GPU output size differs from oracle");
    for (size_t index = 0; index < actual.size(); ++index)
        if (!std::isfinite(actual[index]) || std::memcmp(&actual[index], &expected[index], sizeof(float))) {
            std::fprintf(stderr, "static GPU mismatch slots=%d lanes=%d T=%d duplicates=%d repeat=%d index=%zu expected=%.9g actual=%.9g\n",
                slots, lanes, tokens, int(duplicates), repeat, index, expected[index], actual[index]);
            throw std::runtime_error("expert FFN is not bit-identical to its independent arithmetic reference");
        }
}

void check_case(ggml_backend_t gpu, ggml_backend_t cpu, ggml_backend_t hybrid,
        const std::array<ggml_tensor *, 3> & host, const std::array<ggml_tensor *, 3> & resident,
        const strata_a3b::ExpertCachePlan & plan, int lanes, int tokens, bool duplicates, bool invalid_tests,
        bool strided_ids, bool dynamic_cache, bool hybrid_doorbell) {
    Context input_context;
    auto * input = ggml_new_tensor_3d(input_context.p, GGML_TYPE_F32, hidden, lanes, tokens);
    auto * ids_storage = ggml_new_tensor_2d(input_context.p, GGML_TYPE_I32, strided_ids ? experts : top_k, tokens);
    auto * ids = strided_ids ? ggml_view_2d(input_context.p, ids_storage, top_k, tokens, ids_storage->nb[1], 0) : ids_storage;
    ggml_set_input(input); ggml_set_input(ids_storage);
    Buffer input_buffer; input_buffer.p = ggml_backend_alloc_ctx_tensors(input_context.p, gpu);
    require(input_buffer.p != nullptr, "static GPU input allocation failed");
    Context graph_context;
    // Real CUDA producer/consumer nodes verify that this custom op stays in
    // one CUDA scheduler split rather than joining via a host/backend bridge.
    auto * scaled_input = ggml_scale(graph_context.p, input, 0.5f);
    // Keep the [8,T] view with its 256-ID outer stride intact. Production's
    // factory must insert its own CUDA contiguity pack, without CPU ID reads.
    auto * gpu_ids = strided_ids ? ids : ggml_dup(graph_context.p, ids);
    auto * experts_out = strata_a3b::build_fused_moe(graph_context.p, scaled_input, host[0], host[1], host[2], gpu_ids);
    require(experts_out != nullptr, "static GPU factory refused real Q8 MoE geometry");
    const auto * packed_ids = experts_out->src[1];
    const bool route_source_matches = packed_ids == gpu_ids ||
        (strided_ids && packed_ids && packed_ids->op == GGML_OP_CONT && packed_ids->src[0] == gpu_ids);
    require(experts_out->src[0] == scaled_input && route_source_matches,
            "static GPU op must depend only on GPU input and routed IDs");
    for (int source = 2; source < GGML_MAX_SRC; ++source)
        require(experts_out->src[source] == nullptr, "host expert weights unexpectedly became scheduler graph inputs");
    auto * output = ggml_scale(graph_context.p, experts_out, 2.0f);
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(graph_context.p, 128, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_t backends[] = {gpu, hybrid, cpu};
    Scheduler scheduler; scheduler.p = ggml_backend_sched_new(backends, nullptr, 3, 128, false, true);
    require(scheduler.p != nullptr, "static GPU scheduler initialization failed");
    require(ggml_backend_sched_alloc_graph(scheduler.p, graph), "static GPU scheduler allocation failed");
    require(ggml_backend_sched_get_n_splits(scheduler.p) == 1, "static MoE introduced a non-CUDA scheduler split");
    for (int node = 0; node < ggml_graph_n_nodes(graph); ++node)
        require(ggml_backend_sched_get_tensor_backend(scheduler.p, ggml_graph_node(graph, node)) == gpu,
                "static MoE graph contains a non-CUDA compute node");
    for (int source = 0; source < 2; ++source) {
        const auto * owner = experts_out->src[source];
        while (owner && !owner->buffer && owner->view_src) owner = owner->view_src;
        require(owner && owner->buffer && !ggml_backend_buffer_is_host(owner->buffer),
                "static MoE input or routed IDs reside in a host buffer");
        require(ggml_backend_sched_get_tensor_backend(scheduler.p, experts_out->src[source]) == gpu,
                "static MoE packed input or IDs were not assigned to CUDA");
    }

    std::vector<float> values(hidden * lanes * tokens), actual(hidden * top_k * tokens);
    std::vector<int32_t> routes;
    std::vector<int32_t> storage_routes((strided_ids ? experts : top_k) * tokens);
    auto upload_routes = [&]() {
        // A wrong stride/packing path must expose an invalid-ID error rather
        // than accidentally reading another valid expert from padding.
        std::fill(storage_routes.begin(), storage_routes.end(), std::numeric_limits<int32_t>::min());
        const int stride = strided_ids ? experts : top_k;
        for (int token = 0; token < tokens; ++token)
            std::copy_n(routes.data() + token * top_k, top_k, storage_routes.data() + token * stride);
        ggml_backend_tensor_set(ids_storage, storage_routes.data(), 0, storage_routes.size() * sizeof(int32_t));
    };
    auto set_inputs = [&](int shift) {
        for (size_t index = 0; index < values.size(); ++index) {
            const size_t token = index / (hidden * lanes);
            const float value = std::cos(float(index + shift * 31) * 0.071f) * 0.25f;
            values[index] = token == 1 ? 0.0f : std::ldexp(value, int((index / 32) % 9) - 4);
        }
        routes = route_pattern(plan, tokens, shift * 11, duplicates);
        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
        upload_routes();
    };
    auto submit = [&]() { return ggml_backend_sched_graph_compute_async(scheduler.p, graph); };
    auto read_output = [&]() {
        ggml_backend_tensor_get_async(gpu, output, actual.data(), 0, actual.size() * sizeof(float));
        ggml_backend_synchronize(gpu);
    };
    auto expected_for_inputs = [&]() {
        auto scaled = values;
        for (auto & value : scaled) value *= 0.5f;
        auto expected = singleton_oracle(gpu, resident, scaled, routes, lanes, tokens);
        if (hybrid_doorbell) replace_cpu_miss_oracle(expected, host, scaled, routes, plan, lanes, tokens);
        for (auto & value : expected) value *= 2.0f;
        return expected;
    };
    // Materialize the fixed profile and CUDA capture paths before testing
    // request counters or the no-miss fully-resident upload invariant.
    set_inputs(0);
    auto expected = expected_for_inputs();
    require(submit() == GGML_STATUS_SUCCESS, "static GPU warmup failed");
    read_output();
    strata_a3b::collect_gpu_static_stats();
    assert_exact(actual, expected, int(plan.selected_experts), lanes, tokens, duplicates, -1);
    const auto before = strata_a3b::stats();
    std::array<bool, experts> cached{};
    for (int expert : plan.expert_ids[0]) cached[expert] = true;
    uint64_t expected_miss_rows = 0;
    constexpr int repeats = 3;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        set_inputs(repeat + 1);
        for (int expert : routes) if (!cached[expert]) ++expected_miss_rows;
        expected = expected_for_inputs();
        require(submit() == GGML_STATUS_SUCCESS, "static GPU expert execution failed");
        read_output();
        strata_a3b::collect_gpu_static_stats();
        assert_exact(actual, expected, int(plan.selected_experts), lanes, tokens, duplicates, repeat);
    }
    const auto after = strata_a3b::stats();
    const uint64_t rows = uint64_t(repeats) * top_k * tokens;
    require(after.route_reads == 0, "static experts downloaded routed IDs to the CPU");
    require(after.gpu_static_calls - before.gpu_static_calls == repeats, "static GPU call counter did not cover the requests");
    if (hybrid_doorbell) {
        const uint64_t gpu_rows = rows - expected_miss_rows;
        require(after.cpu_rows - before.cpu_rows == 3 * expected_miss_rows &&
                after.fused_cpu_rows - before.fused_cpu_rows == expected_miss_rows &&
                after.gpu_rows - before.gpu_rows == 3 * gpu_rows &&
                after.fused_gpu_rows - before.fused_gpu_rows == gpu_rows &&
                after.gpu_only_rows - before.gpu_only_rows == gpu_rows,
                "doorbell CPU/GPU row accounting differs from fixed cache map");
        require(after.hits - before.hits == 3 * gpu_rows && after.misses - before.misses == 3 * expected_miss_rows,
                "doorbell hit/miss accounting differs from fixed cache map");
        require(after.hybrid_doorbell_calls - before.hybrid_doorbell_calls == repeats &&
                after.cpu_pool_jobs - before.cpu_pool_jobs == expected_miss_rows,
                "doorbell execution or CPU pool job counter mismatch");
        const uint64_t requests = expected_miss_rows ? repeats : 0;
        require(after.hybrid_doorbell_requests - before.hybrid_doorbell_requests == requests &&
                after.cpu_pool_calls - before.cpu_pool_calls == requests,
                "doorbell CPU service request counter mismatch");
        require(after.hybrid_mailbox_to_host_bytes - before.hybrid_mailbox_to_host_bytes ==
                    requests * uint64_t(tokens) * (lanes * hidden + top_k) * sizeof(float) &&
                after.hybrid_mailbox_to_device_bytes - before.hybrid_mailbox_to_device_bytes ==
                    expected_miss_rows * hidden * sizeof(float), "doorbell small mailbox traffic counter mismatch");
        require(after.upload_bytes == before.upload_bytes && after.uploads == before.uploads &&
                after.staged_rows == before.staged_rows && after.cache_admissions == before.cache_admissions &&
                after.decays == before.decays, "doorbell uploaded expert misses or mutated its fixed cache");
        std::printf("hybrid doorbell Q8 slots=%llu lanes=%d T=%d duplicateIDs=%d stridedIDs=%d single_CUDA_split=1 calls=%d CPU_rows=%llu GPU_rows=%llu route_reads=0 expert_upload_bytes=0 dual_oracle_bitwise_PASS\n",
            (unsigned long long) plan.selected_experts, lanes, tokens, int(duplicates), int(strided_ids), repeats,
            (unsigned long long) expected_miss_rows, (unsigned long long) gpu_rows);
    } else {
        require(after.cpu_rows == 0 && after.fused_cpu_rows == 0, "static experts performed CPU matrix computation");
        require(after.gpu_only_rows - before.gpu_only_rows == rows && after.fused_gpu_rows - before.fused_gpu_rows == rows &&
                after.gpu_rows - before.gpu_rows == 3 * rows, "static GPU row accounting mismatch");
        const uint64_t observed_misses = after.staged_rows - before.staged_rows;
        require(after.hits - before.hits + after.misses - before.misses == 3 * rows &&
                after.misses - before.misses == 3 * observed_misses && observed_misses <= rows,
                "GPU cache actual hit/miss accounting mismatch");
        if (!dynamic_cache)
            require(observed_misses == expected_miss_rows, "static cache hit/miss accounting mismatch");
        if (plan.selected_experts == experts)
            require(after.upload_bytes == before.upload_bytes && after.misses == before.misses,
                    "fully resident static experts uploaded a miss");
        else if (!dynamic_cache || !plan.selected_experts)
            require(after.upload_bytes > before.upload_bytes && observed_misses > 0,
                    "mapped-miss route was not exercised");
        if (!dynamic_cache && plan.selected_experts && plan.selected_experts < experts)
            require(after.hits > before.hits && after.misses > before.misses, "mixed cache fixture did not exercise both paths");
        if (!dynamic_cache) require(after.cache_admissions == before.cache_admissions, "static cache mapping changed during requests");
        const uint64_t uploads = after.uploads - before.uploads;
        require(uploads % 3 == 0 && after.upload_bytes - before.upload_bytes ==
                uploads / 3 * strata_a3b::expert_profile_bundle_bytes, "GPU cache uploaded bundle accounting mismatch");
        std::printf("%s GPU Q8 slots=%llu lanes=%d T=%d duplicateIDs=%d stridedIDs=%d single_CUDA_split=1 calls=%llu rows=%llu miss_rows=%llu CPU_rows=0 route_reads=0 bitwise_PASS\n",
            dynamic_cache ? "dynamic" : "static",
            (unsigned long long) plan.selected_experts, lanes, tokens, int(duplicates), int(strided_ids),
            (unsigned long long) (after.gpu_static_calls - before.gpu_static_calls),
            (unsigned long long) rows, (unsigned long long) observed_misses);
    }

    if (dynamic_cache) {
        // Reuse the final inputs, not the initial profile's now-stale mapping.
        // The LFU admission hysteresis may require more than one encounter.
        // With enough capacity for this active set, repeated routes must
        // eventually stop uploading; under-capacity requests remain correct.
        std::array<bool, experts> active{};
        for (int expert : routes) active[expert] = true;
        const auto unique = std::count(active.begin(), active.end(), true);
        const bool fits = unique <= int(plan.selected_experts);
        bool all_hit = false;
        int replay = 0;
        for (; replay < (fits ? 32 : 2); ++replay) {
            const auto replay_before = strata_a3b::stats();
            require(submit() == GGML_STATUS_SUCCESS, "dynamic GPU route replay failed");
            read_output(); strata_a3b::collect_gpu_static_stats();
            assert_exact(actual, expected, int(plan.selected_experts), lanes, tokens, duplicates, 100 + replay);
            const auto replay_after = strata_a3b::stats();
            const uint64_t replay_rows = uint64_t(top_k) * tokens;
            const auto replay_misses = replay_after.staged_rows - replay_before.staged_rows;
            require(replay_after.gpu_static_calls - replay_before.gpu_static_calls == 1 &&
                    replay_after.gpu_only_rows - replay_before.gpu_only_rows == replay_rows &&
                    replay_after.hits - replay_before.hits + replay_after.misses - replay_before.misses == 3 * replay_rows &&
                    replay_after.misses - replay_before.misses == 3 * replay_misses && replay_misses <= replay_rows &&
                    replay_after.cpu_rows == 0 && replay_after.route_reads == 0,
                    "dynamic route replay accounting mismatch or CPU work");
            all_hit = replay_misses == 0;
            if (all_hit) {
                require(replay_after.upload_bytes == replay_before.upload_bytes, "dynamic all-hit replay uploaded weights");
                break;
            }
        }
        require(!fits || all_hit, "dynamic repeated active set did not converge to an all-hit GPU cache");
        if (!plan.selected_experts)
            require(strata_a3b::stats().cache_admissions == before.cache_admissions,
                    "zero-capacity dynamic cache admitted an expert");
        std::printf("dynamic GPU replay active_unique=%d fits=%d attempts=%d all_hit=%d oracle_bitwise_PASS\n",
                    int(unique), int(fits), replay + 1, int(all_hit));
    }

    if (invalid_tests) for (bool all_invalid : {true, false}) for (int32_t invalid : {-1, experts}) {
        set_inputs(9);
        const auto valid_expected = expected_for_inputs();
        const auto before_error = strata_a3b::stats();
        if (all_invalid) std::fill(routes.begin(), routes.end(), invalid);
        else routes[0] = invalid;
        upload_routes();
        // Device-side bounds checks flag an error but deliberately avoid an
        // immediate host synchronization in every layer's graph_compute.
        submit();
        read_output();
        bool rejected = false;
        try { strata_a3b::collect_gpu_static_stats(); }
        catch (const std::exception &) { rejected = true; }
        require(rejected, "static GPU collector accepted an invalid routed expert ID");
        for (size_t index = 0; index < actual.size(); ++index) {
            if (all_invalid || index < hidden)
                require(std::isnan(actual[index]), "invalid static expert ID did not trigger the non-finite output guard");
            else require(std::isfinite(actual[index]) && !std::memcmp(&actual[index], &valid_expected[index], sizeof(float)),
                         "mixed invalid routing changed a valid channel's singleton CUDA result");
        }
        const auto rejected_stats = strata_a3b::stats();
        require(rejected_stats.gpu_static_errors > before_error.gpu_static_errors, "static GPU device error was not accounted for");
        require(rejected_stats.route_reads == 0 && (hybrid_doorbell || rejected_stats.cpu_rows == 0),
                "invalid routing used unexpected CPU work or ordinary ID readback");
        if (hybrid_doorbell) {
            uint64_t valid_cpu = 0, valid_gpu = 0;
            for (int expert : routes) if (expert >= 0 && expert < experts) {
                if (cached[expert]) ++valid_gpu; else ++valid_cpu;
            }
            require(rejected_stats.cpu_rows - before_error.cpu_rows == 3 * valid_cpu &&
                    rejected_stats.gpu_rows - before_error.gpu_rows == 3 * valid_gpu,
                    "invalid doorbell routing lost or computed an invalid expert row");
        }
        // reset=1 at the collector is also required on its error path. Verify
        // normal execution can resume on this same graph/handle afterwards.
        set_inputs(9);
        expected = expected_for_inputs();
        require(submit() == GGML_STATUS_SUCCESS, "static GPU graph did not recover after invalid routed ID");
        read_output(); strata_a3b::collect_gpu_static_stats();
        assert_exact(actual, expected, int(plan.selected_experts), lanes, tokens, duplicates, 9);
        std::printf("%s invalidID=%d all_invalid=%d NaN_guard/valid_channel_oracle/collector_rejection/same_graph_recovery PASS\n",
                    hybrid_doorbell ? "hybrid doorbell" : "static GPU", invalid, int(all_invalid));
    }
}

struct FailingCpuService {
    std::array<ggml_tensor *, 3> weights{};
    std::atomic<int> failure_mode{0};
};

int failure_test_prepare(void * user, ggml_cuda_strata_moe_layer_v1 * layer, char *, size_t) {
    auto & service = *static_cast<FailingCpuService *>(user);
    *layer = {};
    for (int role = 0; role < 3; ++role) layer->host_weights[role] = service.weights[role];
    for (auto & slot : layer->slot_of_expert) slot = -1;
    layer->cpu_user = user;
    layer->cpu_callback = [](void * payload, const float * input, int lanes, int tokens,
            const int32_t * ids, const int32_t *, float * output, char * error, size_t size) -> int {
        auto & injected = *static_cast<FailingCpuService *>(payload);
        const int mode = injected.failure_mode.load(std::memory_order_acquire);
        if (mode == 1) {
            if (error && size) std::snprintf(error, size, "injected CPU callback return failure");
            return 0;
        }
        if (mode == 2) throw std::runtime_error("injected CPU callback exception");
        for (int token = 0; token < tokens; ++token) for (int lane = 0; lane < top_k; ++lane)
            serial_cpu_expert(injected.weights, input + (token * lanes + lane % lanes) * hidden,
                              ids[token * top_k + lane], output + (token * top_k + lane) * hidden);
        return 1;
    };
    return 1;
}

struct RawHandle {
    const ggml_cuda_strata_moe_api_v1 * api;
    void * p;
    ~RawHandle() { if (p) api->destroy(p); }
};

// Run after the production providers have been released: a callback failure
// deliberately releases every mailbox on its device, including idle layers.
void check_callback_failure(ggml_backend_t gpu, ggml_backend_t cpu,
        const ggml_cuda_strata_moe_api_v1 * api, int device,
        const std::array<ggml_tensor *, 3> & host) {
    FailingCpuService service; service.weights = host;
    char error[512]{};
    RawHandle handle{api, api->create(device, failure_test_prepare, &service, error, sizeof(error))};
    require(handle.p != nullptr, "failure-test CUDA mailbox creation failed");
    Context input_context;
    auto * input = ggml_new_tensor_3d(input_context.p, GGML_TYPE_F32, hidden, 1, 1);
    auto * ids = ggml_new_tensor_2d(input_context.p, GGML_TYPE_I32, top_k, 1);
    ggml_set_input(input); ggml_set_input(ids);
    Buffer input_buffer; input_buffer.p = ggml_backend_alloc_ctx_tensors(input_context.p, gpu);
    require(input_buffer.p != nullptr, "failure-test input allocation failed");
    Context graph_context;
    auto * scaled = ggml_scale(graph_context.p, input, 0.5f);
    auto * expert_output = api->build(handle.p, graph_context.p, scaled, ids);
    require(expert_output != nullptr, "failure-test mailbox op creation failed");
    auto * output = ggml_scale(graph_context.p, expert_output, 2.0f);
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(graph_context.p, 128, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_t backends[] = {gpu, cpu};
    Scheduler scheduler; scheduler.p = ggml_backend_sched_new(backends, nullptr, 2, 128, false, true);
    require(scheduler.p && ggml_backend_sched_alloc_graph(scheduler.p, graph), "failure-test graph allocation failed");
    require(ggml_backend_sched_get_n_splits(scheduler.p) == 1, "failure-test mailbox introduced a non-CUDA split");
    std::vector<float> values(hidden), actual(hidden * top_k), expected(hidden * top_k);
    std::vector<int32_t> routes(top_k);
    for (int index = 0; index < hidden; ++index) values[index] = std::cos(float(index) * .031f) * .125f;
    for (int lane = 0; lane < top_k; ++lane) routes[lane] = lane;
    ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    ggml_backend_tensor_set(ids, routes.data(), 0, routes.size() * sizeof(int32_t));
    for (auto & value : values) value *= .5f;
    for (int lane = 0; lane < top_k; ++lane)
        serial_cpu_expert(host, values.data(), routes[lane], expected.data() + lane * hidden);
    for (auto & value : expected) value *= 2.f;
    auto run = [&]() {
        require(ggml_backend_sched_graph_compute_async(scheduler.p, graph) == GGML_STATUS_SUCCESS,
                "failure-test CUDA graph submission failed");
        ggml_backend_tensor_get_async(gpu, output, actual.data(), 0, actual.size() * sizeof(float));
        ggml_backend_synchronize(gpu);
    };
    for (int mode : {1, 2}) {
        service.failure_mode.store(mode, std::memory_order_release);
        const auto start = std::chrono::steady_clock::now();
        run();
        require(std::chrono::steady_clock::now() - start < std::chrono::seconds(10),
                "CPU callback failure did not release GPU waits promptly");
        ggml_cuda_strata_moe_stats_v1 rejected{};
        error[0] = '\0'; // A successful get_stats need not overwrite old error text.
        const int ok = api->get_stats(handle.p, 1, &rejected, error, sizeof(error));
        const auto nan_count = std::count_if(actual.begin(), actual.end(), [](float value) { return std::isnan(value); });
        std::printf("hybrid doorbell failure diagnostic mode=%d nan_count=%llu/%zu cpu_requests=%llu device_error=%u get_stats_ok=%d error=%s\n",
            mode, (unsigned long long) nan_count, actual.size(), (unsigned long long) rejected.host_callback_requests,
            rejected.device_error, ok, error);
        std::fflush(stdout);
        for (float value : actual) require(std::isnan(value), "CPU service failure did not poison GPU output");
        require(!ok && rejected.device_error == 2 && rejected.host_callback_requests == 1 &&
                std::strstr(error, "injected") != nullptr, "CPU callback failure did not reach/reset request error boundary");
        service.failure_mode.store(0, std::memory_order_release);
        run();
        ggml_cuda_strata_moe_stats_v1 recovered{};
        require(api->get_stats(handle.p, 1, &recovered, error, sizeof(error)) &&
                recovered.device_error == 0 && recovered.cpu_rows == top_k && recovered.gpu_rows == 0 &&
                recovered.host_callback_requests == 1, "CPU callback failure did not permit same-graph recovery");
        assert_exact(actual, expected, 0, 1, 1, false, 200 + mode);
        std::printf("hybrid doorbell injected_callback_%s NaN_guard/error2/reset/same_graph_serial_CPU_oracle_recovery PASS\n",
                    mode == 1 ? "return0" : "throw");
    }
} // scheduler/graphs precede raw handle, and callback userdata remains alive

struct StaticCleanup {
    bool active = true;
    void release() { strata_a3b::release_gpu_static_experts(); active = false; }
    ~StaticCleanup() {
        if (active) try { strata_a3b::release_gpu_static_experts(); }
        catch (const std::exception & error) { std::fprintf(stderr, "static cleanup: %s\n", error.what()); }
    }
};
}

int main(int argc, char ** argv) {
    try {
        int slots = 128, device_index = 0;
        bool strided_ids = false, dynamic_cache = false, hybrid_doorbell = false;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--cache-slots" && i + 1 < argc) slots = std::stoi(argv[++i]);
            else if (option == "--device" && i + 1 < argc) device_index = std::stoi(argv[++i]);
            else if (option == "--strided-ids") strided_ids = true;
            else if (option == "--dynamic-cache") dynamic_cache = true;
            else if (option == "--hybrid-doorbell") hybrid_doorbell = true;
            else if (option == "--help") {
                std::puts("usage: strata-moe-gpu-static-parity [--cache-slots 0|16|128|256] [--device N] [--strided-ids] [--dynamic-cache | --hybrid-doorbell]"); return 0;
            } else throw std::runtime_error("usage: strata-moe-gpu-static-parity [--cache-slots 0|16|128|256] [--device N] [--strided-ids] [--dynamic-cache | --hybrid-doorbell]");
        }
        require(slots == 0 || slots == 16 || slots == 128 || slots == 256, "cache slots must be 0, 16, 128 or 256");
        require(device_index >= 0, "device must be nonnegative");
        require(!(dynamic_cache && hybrid_doorbell), "dynamic cache and hybrid doorbell fixture modes are mutually exclusive");
        ggml_backend_load_all();
        ggml_backend_dev_t device = nullptr;
        for (size_t index = 0, gpu_number = 0; index < ggml_backend_dev_count(); ++index) {
            auto * candidate = ggml_backend_dev_get(index);
            if (ggml_backend_dev_type(candidate) == GGML_BACKEND_DEVICE_TYPE_GPU && int(gpu_number++) == device_index) {
                device = candidate; break;
            }
        }
        if (!device) { std::puts("SKIP static GPU parity: requested GPU unavailable"); return 77; }
        auto * cuda = ggml_backend_dev_backend_reg(device);
        auto get_api = reinterpret_cast<ggml_cuda_strata_moe_get_api_v1>(
            ggml_backend_reg_get_proc_address(cuda, "ggml_backend_cuda_strata_moe_v1"));
        if (!get_api || !get_api() || get_api()->abi_version != GGML_CUDA_STRATA_MOE_ABI_V1) {
            std::puts("SKIP static GPU parity: native CUDA static-MoE ABI unavailable"); return 77;
        }
        Backend gpu; gpu.p = ggml_backend_dev_init(device, nullptr);
        Backend cpu; cpu.p = ggml_backend_cpu_init();
        require(gpu.p != nullptr && cpu.p != nullptr, "static parity backend initialization failed");
        ggml_backend_cpu_set_n_threads(cpu.p, 1);
        const auto plan = manual_plan(slots);
        strata_a3b::Config config;
        config.gpu = device; config.gpu_only_experts = !hybrid_doorbell; config.gpu_static_experts = !hybrid_doorbell;
        config.gpu_dynamic_experts = dynamic_cache;
        config.hybrid_experts = config.cpu_expert_pool = config.hybrid_doorbell = hybrid_doorbell;
        if (hybrid_doorbell) config.pageable_experts = true;
        config.fused_moe = true; config.exact_verify = true; config.native_exact_max_tokens = 8;
        config.expert_bytes = strata_a3b::expert_profile_bundle_bytes * experts;
        config.cache_bytes = plan.planned_bytes; config.cache_decay_interval = 0;
        auto * registration = strata_a3b::register_backend(config);
        strata_a3b::set_cache_budget(plan.planned_bytes);
        strata_a3b::set_cache_plan(plan);
        strata_a3b::set_cache_admission(false);
        Context host_context, resident_context;
        std::array<ggml_tensor *, 3> host{}, resident{};
        for (int projection = 0; projection < 3; ++projection) {
            host[projection] = ggml_new_tensor_3d(host_context.p, GGML_TYPE_Q8_0,
                projection == 2 ? ffn : hidden, projection == 2 ? hidden : ffn, experts);
            ggml_format_name(host[projection], "blk.0.ffn_%s_exps.weight", projection == 0 ? "gate" : projection == 1 ? "up" : "down");
            resident[projection] = ggml_dup_tensor(resident_context.p, host[projection]);
        }
        Buffer host_buffer, resident_buffer;
        host_buffer.p = ggml_backend_alloc_ctx_tensors_from_buft(host_context.p, strata_a3b::expert_buffer_type());
        resident_buffer.p = ggml_backend_alloc_ctx_tensors(resident_context.p, gpu.p);
        require(host_buffer.p != nullptr && resident_buffer.p != nullptr, "real-geometry expert weights did not fit");
        ggml_backend_buffer_set_usage(host_buffer.p, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_set_usage(resident_buffer.p, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        initialize_weights(gpu.p, host, resident);
        StaticCleanup cleanup; // destroys handles before immutable host weights
        std::printf("%s GPU oracle device=%s Q8_0 hidden=2048 FFN=512 experts=256 top8 cache_slots=%d permuted_IDs=1 stridedIDs=%d\n",
            hybrid_doorbell ? "hybrid doorbell" : dynamic_cache ? "dynamic" : "static", ggml_backend_dev_name(device), slots, int(strided_ids));
        {
            Backend hybrid; hybrid.p = ggml_backend_dev_init(ggml_backend_reg_dev_get(registration, 0), nullptr);
            require(hybrid.p != nullptr, "static cache owner backend initialization failed");
            for (int tokens : {1, 2, 3, 7, 8}) for (int lanes : {1, top_k}) for (bool duplicates : {false, true})
                check_case(gpu.p, cpu.p, hybrid.p, host, resident, plan, lanes, tokens, duplicates,
                           tokens == 8 && lanes == top_k && duplicates, strided_ids, dynamic_cache, hybrid_doorbell);
            strata_a3b::collect_gpu_static_stats();
            const auto stats = strata_a3b::stats();
            require(stats.route_reads == 0 && stats.gpu_static_calls > 0,
                    "fixture did not retain native CUDA execution and routing");
            if (hybrid_doorbell) {
                require(stats.hybrid_doorbell_calls > 0 && stats.cache_admissions == 0 && stats.decays == 0,
                        "hybrid doorbell fixture bypassed mailbox or changed fixed cache");
                require((slots == experts) == (stats.cpu_rows == 0), "hybrid doorbell CPU ownership was not exercised");
                require((slots == 0) == (stats.gpu_rows == 0), "hybrid doorbell GPU cache-hit ownership was not exercised");
            } else require(stats.cpu_rows == 0 && stats.fused_cpu_rows == 0,
                           "static fixture unexpectedly performed CPU expert math");
            require(stats.gpu_static_errors == 4, "fixture did not account for all four deliberately invalid ID calls exactly once");
            if (dynamic_cache && slots > 0 && slots < experts)
                require(stats.cache_admissions > 0 && stats.decays > 0,
                        "dynamic fixture did not exercise actual GPU admissions and policy decay");
        } // graphs/schedulers and the custom cache backend precede ABI cleanup
        cleanup.release();
        if (hybrid_doorbell) check_callback_failure(gpu.p, cpu.p, get_api(), device_index, host);
        std::puts(hybrid_doorbell ?
            "PASS hybrid doorbell Q8 MoE: 20 shape/layout/duplicate cases, independent serial CPU misses plus resident CUDA singleton hits bitwise, one CUDA split, fixed map/counts/mailbox traffic, route_reads=0, invalid -1/256 all/mixed NaN guards and recovery" :
            "PASS static Q8 MoE: 20 shape/layout/duplicate cases, exact resident CUDA singleton oracle, one CUDA split, CPU_rows=0, route_reads=0, invalid -1/256 all/mixed NaN guards and recovery");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL static GPU expert parity: %s\n", error.what()); return 1;
    }
}
