#include "backend.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void check_complete_windows(ggml_backend_t gpu, ggml_backend_t cpu, ggml_backend_t hybrid,
                                   ggml_tensor * weights, int ni, int nk, bool preserve_layout, bool cache_routes, bool exact_verify) {
    // A fresh weight address gets sixteen slots, so every routed row stays on
    // GPU. Two independent projections share routes in one hybrid graph call.
    strata_a3b::set_cache_budget(1ull << 30);
    for (int nt : {1, 2, 4}) for (int lanes : {1, nk}) {
        auto ctx = ggml_init({ggml_tensor_overhead() * 16 + 2 * ggml_graph_overhead_custom(32, false), nullptr, true});
        auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ni, lanes, nt);
        auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, nk, nt);
        const bool lanes_oracle = exact_verify && ggml_is_quantized(weights->type) && nt > 1;
        const bool flat_oracle = !lanes_oracle && !preserve_layout && nt > 1;
        auto * oracle_x = lanes_oracle ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ni, nk * nt, 1) :
                          flat_oracle ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ni, 1, nk * nt) : x;
        auto * oracle_ids = lanes_oracle ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, nk * nt, 1) :
                            flat_oracle ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, nk * nt) : ids;
        auto * a = ggml_mul_mat_id(ctx, weights, oracle_x, oracle_ids);
        auto * b = ggml_mul_mat_id(ctx, weights, oracle_x, oracle_ids);
        auto * sum = ggml_add(ctx, a, b);
        auto * graph = ggml_new_graph_custom(ctx, 32, false); ggml_build_forward_expand(graph, sum);
        auto host_buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        std::vector<float> inputs(ni * lanes * nt);
        for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = std::cos(float(i) * 0.071f) * 0.25f;
        std::vector<int32_t> routes(nk * nt);
        for (int t = 0; t < nt; ++t) for (int k = 0; k < nk; ++k) routes[t * nk + k] = (t + k) % weights->ne[2];
        ggml_backend_tensor_set(x, inputs.data(), 0, inputs.size() * sizeof(float));
        ggml_backend_tensor_set(ids, routes.data(), 0, routes.size() * sizeof(int32_t));
        if (flat_oracle || lanes_oracle) {
            std::vector<float> flat_inputs(ni * nk * nt);
            for (int t = 0; t < nt; ++t) for (int k = 0; k < nk; ++k)
                std::copy_n(inputs.data() + (t * lanes + k % lanes) * ni, ni,
                            flat_inputs.data() + (t * nk + k) * ni);
            ggml_backend_tensor_set(oracle_x, flat_inputs.data(), 0, flat_inputs.size() * sizeof(float));
            ggml_backend_tensor_set(oracle_ids, routes.data(), 0, routes.size() * sizeof(int32_t));
        }
        auto copy = ggml_backend_graph_copy(gpu, graph);
        if (!copy.graph || ggml_backend_graph_compute(gpu, copy.graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("complete-window GPU oracle failed");
        std::vector<float> reference(ggml_nelements(sum));
        ggml_backend_tensor_get(ggml_graph_node(copy.graph, -1), reference.data(), 0, reference.size() * sizeof(float));
        ggml_backend_graph_copy_free(copy);

        auto input_ctx = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
        auto * gx = ggml_dup_tensor(input_ctx, x); auto * gi = ggml_dup_tensor(input_ctx, ids);
        auto input_buf = ggml_backend_alloc_ctx_tensors(input_ctx, gpu);
        ggml_backend_tensor_copy(x, gx); ggml_backend_tensor_copy(ids, gi);
        auto * ga = ggml_mul_mat_id(ctx, weights, gx, gi);
        auto * gb = ggml_mul_mat_id(ctx, weights, gx, gi);
        auto * total = ggml_add(ctx, ga, gb); ggml_set_output(total);
        auto * mixed = ggml_new_graph_custom(ctx, 32, false); ggml_build_forward_expand(mixed, total);
        ggml_backend_t backends[] = {gpu, hybrid, cpu};
        auto sched = ggml_backend_sched_new(backends, nullptr, 3, 32, false, true);
        const auto before = strata_a3b::stats();
        if (ggml_backend_sched_graph_compute(sched, mixed) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("complete-window scheduler failed");
        const auto after = strata_a3b::stats();
        if (ggml_backend_sched_get_tensor_backend(sched, ga) != hybrid ||
            ggml_backend_sched_get_tensor_backend(sched, gb) != hybrid)
            throw std::runtime_error("complete-window scheduler bypassed custom backend");
        if (after.cpu_rows != before.cpu_rows)
            throw std::runtime_error("complete-window cache did not hold every route");
        if (after.preserved_layout_calls - before.preserved_layout_calls != (preserve_layout && !lanes_oracle ? 2u : 0u))
            throw std::runtime_error("complete-window layout fast path selection mismatch");
        if (after.exact_verify_calls - before.exact_verify_calls != (lanes_oracle ? 2u : 0u))
            throw std::runtime_error("complete-window exact lane path selection mismatch");
        if (after.route_reads - before.route_reads != (cache_routes ? 1u : 2u))
            throw std::runtime_error("graph routing snapshot reuse mismatch");
        std::vector<float> actual(reference.size());
        ggml_backend_tensor_get(total, actual.data(), 0, actual.size() * sizeof(float));
        // Compare against the native GPU graph with the exact selected layout;
        // no CPU/other-kernel alternative or relaxed tolerance is needed here.
        for (size_t i = 0; i < actual.size(); ++i)
            if (!std::isfinite(actual[i]) || std::abs(actual[i] - reference[i]) > 2e-5f * (1 + std::abs(reference[i])))
                throw std::runtime_error("preserved window differs from native GPU layout oracle");
        std::printf("%s complete tokens=%d lanes=%d layout=%d exact=%d route_reads=%llu PASS\n", ggml_type_name(weights->type), nt, lanes,
                    preserve_layout, exact_verify, (unsigned long long) (after.route_reads - before.route_reads));
        ggml_backend_sched_free(sched);
        ggml_backend_buffer_free(input_buf); ggml_free(input_ctx);
        ggml_backend_buffer_free(host_buf); ggml_free(ctx);
    }
}

int main(int argc, char ** argv) {
    try {
        ggml_backend_load_all();
        bool cpu_only = false, pageable = false, stable_tasks = false, preserve_layout = false, cache_routes = false, exact_verify = false;
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--cpu") == 0) cpu_only = true;
            else if (std::strcmp(argv[i], "--pageable") == 0) pageable = true;
            else if (std::strcmp(argv[i], "--stable-tasks") == 0) stable_tasks = true;
            else if (std::strcmp(argv[i], "--preserve-layout") == 0) preserve_layout = true;
            else if (std::strcmp(argv[i], "--cache-routes") == 0) cache_routes = true;
            else if (std::strcmp(argv[i], "--exact-verify") == 0) exact_verify = true;
            else throw std::runtime_error("unknown parity argument");
        }
        auto gpu_dev = cpu_only ? nullptr :
                       ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        auto reg = strata_a3b::register_backend({gpu_dev, 0, 1ull << 30, 4, true, pageable, stable_tasks, preserve_layout, cache_routes, exact_verify});
        auto hybrid = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
        auto gpu = gpu_dev ? ggml_backend_dev_init(gpu_dev, nullptr) : nullptr;
        auto cpu = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(cpu, 4);
        if (!hybrid || !cpu) throw std::runtime_error("backend init failed");
        for (auto qtype : {GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0}) {
            const int ni = 256, no = 64, ne = 16, nk = 8;
            std::vector<float> values(ni * no * ne);
            for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(float(i) * 0.17f) * 0.1f;
            const size_t row_bytes = ggml_row_size(qtype, ni);
            std::vector<char> packed(row_bytes * no * ne);
            if (qtype == GGML_TYPE_F32) std::memcpy(packed.data(), values.data(), packed.size());
            else ggml_quantize_chunk(qtype, values.data(), packed.data(), 0, no * ne, ni, nullptr);
            auto weight_ctx = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
            auto * wa = ggml_new_tensor_3d(weight_ctx, qtype, ni, no, ne);
            auto * wb = ggml_new_tensor_3d(weight_ctx, qtype, ni, no, ne);
            auto * wc = ggml_new_tensor_3d(weight_ctx, qtype, ni, no, ne);
            auto weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx, strata_a3b::expert_buffer_type());
            ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_tensor_set(wa, packed.data(), 0, packed.size());
            ggml_backend_tensor_set(wc, packed.data(), 0, packed.size());
            for (auto & v : values) v = -v;
            if (qtype == GGML_TYPE_F32) std::memcpy(packed.data(), values.data(), packed.size());
            else ggml_quantize_chunk(qtype, values.data(), packed.data(), 0, no * ne, ni, nullptr);
            ggml_backend_tensor_set(wb, packed.data(), 0, packed.size());
            // Four slots force both CPU misses and GPU hits, and repeated routes force eviction.
            strata_a3b::set_cache_budget(1ull << 28);
            for (int pass = 0; pass < 11; ++pass) {
                // Check both sides of the small verification-window boundary,
                // then return to decode to exercise cache and task reuse.
                const int nt = pass == 2 ? 2 : pass == 3 ? 4 : pass == 4 ? 5 : pass == 5 ? 9 : 1;
                const int lanes = pass % 2 ? nk : 1;
                const bool frozen = pass == 3 || pass == 4;
                strata_a3b::set_cache_admission(!frozen);
                auto ctx = ggml_init({ggml_tensor_overhead() * 16 + 2 * ggml_graph_overhead_custom(32, false), nullptr, true});
                auto * w = pass % 2 ? wb : wa;
                auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ni, lanes, nt);
                auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, nk, nt);
                auto * y = ggml_mul_mat_id(ctx, w, x, ids);
                auto * graph = ggml_new_graph_custom(ctx, 32, false); ggml_build_forward_expand(graph, y);
                auto buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, strata_a3b::expert_buffer_type());
                if (!buffer) throw std::runtime_error("test buffer allocation failed");
                std::vector<float> inputs(ni * lanes * nt);
                for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = std::cos(float(i) * 0.07f) * 0.25f;
                std::vector<int32_t> routes(nk * nt);
                for (size_t i = 0; i < routes.size(); ++i) routes[i] = (i + pass * 3) % ne;
                ggml_backend_tensor_set(x, inputs.data(), 0, inputs.size() * sizeof(float));
                ggml_backend_tensor_set(ids, routes.data(), 0, routes.size() * sizeof(int32_t));
                if (ggml_backend_graph_compute(cpu, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("reference failed");
                std::vector<float> reference(ggml_nelements(y));
                ggml_backend_tensor_get(y, reference.data(), 0, reference.size() * sizeof(float));
                std::vector<float> gpu_reference(reference);
                std::vector<float> gpu_flat_reference(reference);
                std::vector<float> gpu_lanes_reference(reference);
                if (gpu) {
                    auto copy = ggml_backend_graph_copy(gpu, graph);
                    if (!copy.graph || ggml_backend_graph_compute(gpu, copy.graph) != GGML_STATUS_SUCCESS)
                        throw std::runtime_error("GPU reference failed");
                    auto * out = ggml_graph_node(copy.graph, -1);
                    ggml_backend_tensor_get(out, gpu_reference.data(), 0, gpu_reference.size() * sizeof(float));
                    ggml_backend_graph_copy_free(copy);
                    // Gathering routed rows can cross CUDA's MMVQ/MMQ batch
                    // threshold. Their activation quantizers differ, so test
                    // against a second native GPU graph with independent flat
                    // rows as well, without relaxing the numerical tolerance.
                    auto flat_ctx = ggml_init({ggml_tensor_overhead() * 8 + 2 * ggml_graph_overhead_custom(32, false), nullptr, true});
                    auto * flat_x = ggml_new_tensor_3d(flat_ctx, GGML_TYPE_F32, ni, 1, nk * nt);
                    auto * flat_ids = ggml_new_tensor_2d(flat_ctx, GGML_TYPE_I32, 1, nk * nt);
                    auto * flat_y = ggml_mul_mat_id(flat_ctx, w, flat_x, flat_ids);
                    auto * flat_graph = ggml_new_graph_custom(flat_ctx, 32, false);
                    ggml_build_forward_expand(flat_graph, flat_y);
                    ggml_tensor * lanes_x = nullptr, * lanes_ids = nullptr;
                    ggml_cgraph * lanes_graph = nullptr;
                    if (exact_verify && ggml_is_quantized(qtype)) {
                        lanes_x = ggml_new_tensor_3d(flat_ctx, GGML_TYPE_F32, ni, nk * nt, 1);
                        lanes_ids = ggml_new_tensor_2d(flat_ctx, GGML_TYPE_I32, nk * nt, 1);
                        auto * lanes_y = ggml_mul_mat_id(flat_ctx, w, lanes_x, lanes_ids);
                        lanes_graph = ggml_new_graph_custom(flat_ctx, 32, false);
                        ggml_build_forward_expand(lanes_graph, lanes_y);
                    }
                    auto flat_buf = ggml_backend_alloc_ctx_tensors(flat_ctx, cpu);
                    if (!flat_buf) throw std::runtime_error("flat oracle allocation failed");
                    std::vector<float> flat_inputs(ni * nk * nt);
                    for (int t = 0; t < nt; ++t) for (int k = 0; k < nk; ++k)
                        std::copy_n(inputs.data() + (t * lanes + k % lanes) * ni, ni,
                                    flat_inputs.data() + (t * nk + k) * ni);
                    ggml_backend_tensor_set(flat_x, flat_inputs.data(), 0, flat_inputs.size() * sizeof(float));
                    ggml_backend_tensor_set(flat_ids, routes.data(), 0, routes.size() * sizeof(int32_t));
                    if (lanes_graph) {
                        ggml_backend_tensor_set(lanes_x, flat_inputs.data(), 0, flat_inputs.size() * sizeof(float));
                        ggml_backend_tensor_set(lanes_ids, routes.data(), 0, routes.size() * sizeof(int32_t));
                    }
                    auto flat_copy = ggml_backend_graph_copy(gpu, flat_graph);
                    if (!flat_copy.graph || ggml_backend_graph_compute(gpu, flat_copy.graph) != GGML_STATUS_SUCCESS)
                        throw std::runtime_error("flat GPU reference failed");
                    ggml_backend_tensor_get(ggml_graph_node(flat_copy.graph, -1), gpu_flat_reference.data(), 0,
                                            gpu_flat_reference.size() * sizeof(float));
                    ggml_backend_graph_copy_free(flat_copy);
                    if (lanes_graph) {
                        auto lanes_copy = ggml_backend_graph_copy(gpu, lanes_graph);
                        if (!lanes_copy.graph || ggml_backend_graph_compute(gpu, lanes_copy.graph) != GGML_STATUS_SUCCESS)
                            throw std::runtime_error("single-column lanes GPU reference failed");
                        ggml_backend_tensor_get(ggml_graph_node(lanes_copy.graph, -1), gpu_lanes_reference.data(), 0,
                                                gpu_lanes_reference.size() * sizeof(float));
                        ggml_backend_graph_copy_free(lanes_copy);
                    }
                    ggml_backend_buffer_free(flat_buf); ggml_free(flat_ctx);
                }
                const auto before = strata_a3b::stats();
                if (ggml_backend_graph_compute(hybrid, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("hybrid failed");
                const auto after = strata_a3b::stats();
                if (gpu_dev && nt <= 4 && after.cpu_rows == before.cpu_rows)
                    throw std::runtime_error("small verification window streamed instead of splitting");
                if (gpu_dev && nt > 4 && !frozen && after.cpu_rows != before.cpu_rows)
                    throw std::runtime_error("long prefill did not stream through GPU slots");
                if (gpu_dev && nt > 4 && frozen && after.cpu_rows == before.cpu_rows)
                    throw std::runtime_error("frozen long window streamed instead of splitting");
                if (frozen && after.uploads != before.uploads)
                    throw std::runtime_error("frozen expert placement admitted a new upload");
                if (gpu_dev && exact_verify && ggml_is_quantized(qtype) && nt > 1 && nt <= 4 && after.exact_verify_calls == before.exact_verify_calls)
                    throw std::runtime_error("partial-hit window did not use exact single-column lanes");
                std::vector<float> result(reference.size());
                ggml_backend_tensor_get(y, result.data(), 0, result.size() * sizeof(float));
                float max_error = 0;
                for (size_t i = 0; i < result.size(); ++i) {
                    float error = std::abs(reference[i] - result[i]); max_error = std::max(max_error, error);
                    // A row must match a native CPU or native CUDA kernel result.
                    const float oracle_error = std::min({error, std::abs(gpu_reference[i] - result[i]),
                                                         std::abs(gpu_flat_reference[i] - result[i]),
                                                         std::abs(gpu_lanes_reference[i] - result[i])});
                    const float tolerance = gpu_dev ? 2e-5f * (1 + std::abs(reference[i])) : 1e-6f;
                    if (!std::isfinite(result[i]) || oracle_error > tolerance) {
                        std::fprintf(stderr, "mismatch index=%zu reference=%.9g gpu_reference=%.9g flat_gpu_reference=%.9g lanes_gpu_reference=%.9g actual=%.9g error=%.9g tolerance=%.9g\n", i, reference[i], gpu_reference[i], gpu_flat_reference[i], gpu_lanes_reference[i], result[i], error, tolerance);
                        throw std::runtime_error(std::string(ggml_type_name(qtype)) + " result mismatch, pass " + std::to_string(pass));
                    }
                }
                std::printf("%s pass=%d tokens=%d lanes=%d max_abs_error=%.7g\n", ggml_type_name(qtype), pass, nt, lanes, max_error);
                if (gpu && (pass == 3 || pass == 4 || pass == 5 || pass == 10)) {
                    auto inp_ctx = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
                    auto * gx = ggml_dup_tensor(inp_ctx, x);
                    auto * gi = ggml_dup_tensor(inp_ctx, ids);
                    auto inp_buf = ggml_backend_alloc_ctx_tensors(inp_ctx, gpu);
                    ggml_backend_tensor_copy(x, gx); ggml_backend_tensor_copy(ids, gi);
                    auto * device_y = ggml_mul_mat_id(ctx, w, gx, gi);
                    auto * z = ggml_scale(ctx, device_y, 2.0f);
                    ggml_set_output(z);
                    auto * mixed = ggml_new_graph_custom(ctx, 32, false); ggml_build_forward_expand(mixed, z);
                    ggml_backend_t backends[] = {gpu, hybrid, cpu};
                    auto sched = ggml_backend_sched_new(backends, nullptr, 3, 32, false, true);
                    if (ggml_backend_sched_graph_compute(sched, mixed) != GGML_STATUS_SUCCESS)
                        throw std::runtime_error("scheduler crossing failed");
                    if (ggml_backend_sched_get_tensor_backend(sched, device_y) != hybrid)
                        throw std::runtime_error("scheduler bypassed StrataMoE");
                    std::vector<float> scaled(result.size());
                    ggml_backend_tensor_get(z, scaled.data(), 0, scaled.size() * sizeof(float));
                    for (size_t i = 0; i < scaled.size(); ++i) {
                        const float error = std::min({std::abs(scaled[i] - 2 * reference[i]),
                                                      std::abs(scaled[i] - 2 * gpu_reference[i]),
                                                      std::abs(scaled[i] - 2 * gpu_flat_reference[i]),
                                                      std::abs(scaled[i] - 2 * gpu_lanes_reference[i])});
                        if (!std::isfinite(scaled[i]) || error > 4e-5f * (1 + std::abs(reference[i])))
                            throw std::runtime_error("GPU/RAM scheduler result mismatch");
                    }
                    std::printf("%s scheduler GPU -> RAM experts -> GPU PASS\n", ggml_type_name(qtype));
                    ggml_backend_sched_free(sched);
                    ggml_backend_buffer_free(inp_buf); ggml_free(inp_ctx);
                }
                if (frozen && strata_a3b::stats().uploads != before.uploads)
                    throw std::runtime_error("frozen scheduler placement admitted a new upload");
                ggml_backend_buffer_free(buffer); ggml_free(ctx);
            }
            if (gpu) check_complete_windows(gpu, cpu, hybrid, wc, ni, nk, preserve_layout, cache_routes, exact_verify);
            ggml_backend_buffer_free(weight_buffer); ggml_free(weight_ctx);
        }
        auto s = strata_a3b::stats();
        std::printf("PASS expert_host=%s stable_tasks=%d gpu_rows=%llu cpu_rows=%llu uploaded_bytes=%llu\n", pageable ? "pageable" : "pinned", stable_tasks, (unsigned long long) s.gpu_rows,
                    (unsigned long long) s.cpu_rows, (unsigned long long) s.upload_bytes);
        if (gpu_dev && (!s.gpu_rows || !s.cpu_rows)) throw std::runtime_error("split path not exercised");
        ggml_backend_free(hybrid); ggml_backend_free(cpu);
        if (gpu) ggml_backend_free(gpu);
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
