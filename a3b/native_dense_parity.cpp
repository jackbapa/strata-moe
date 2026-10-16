#include "backend.hpp"
#include "ggml.h"
#ifdef STRATA_A3B_DENSE_CUDA_EVENTS
#include "ggml-cuda.h"
#include <cuda_runtime.h>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static std::vector<float> compute(ggml_backend_t gpu, ggml_context * ctx, ggml_tensor * out,
                                  ggml_tensor * input, const float * data, size_t count,
                                  int repeats = 0, double * mean_us = nullptr, double * event_us = nullptr) {
    auto * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, out);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    if (!buffer) throw std::runtime_error("dense parity tensor allocation failed");
    ggml_backend_tensor_set(input, data, 0, count * sizeof(float));
    if (ggml_backend_graph_compute(gpu, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("dense parity computation failed");
    if (repeats && mean_us) {
        for (int i = 0; i < 5; ++i)
            if (ggml_backend_graph_compute(gpu, graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("dense benchmark warmup failed");
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i)
            if (ggml_backend_graph_compute(gpu, graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("dense benchmark computation failed");
        ggml_backend_synchronize(gpu);
        *mean_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / repeats;
    }
#ifdef STRATA_A3B_DENSE_CUDA_EVENTS
    if (repeats && event_us) {
        auto stream = static_cast<cudaStream_t>(ggml_backend_cuda_stream(gpu));
        if (!stream) throw std::runtime_error("CUDA event benchmark needs a CUDA backend");
        auto check = [](cudaError_t e, const char * what) {
            if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
        };
        cudaEvent_t start = nullptr, stop = nullptr;
        check(cudaEventCreate(&start), "create start event");
        check(cudaEventCreate(&stop), "create stop event");
        for (int i = 0; i < 5; ++i)
            if (ggml_backend_graph_compute_async(gpu, graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("dense event benchmark warmup failed");
        ggml_backend_synchronize(gpu);
        check(cudaEventRecord(start, stream), "record start event");
        for (int i = 0; i < repeats; ++i)
            if (ggml_backend_graph_compute_async(gpu, graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("dense event benchmark computation failed");
        check(cudaEventRecord(stop, stream), "record stop event");
        check(cudaEventSynchronize(stop), "synchronize stop event");
        float elapsed_ms = 0.0f;
        check(cudaEventElapsedTime(&elapsed_ms, start, stop), "read CUDA events");
        *event_us = 1000.0 * elapsed_ms / repeats;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
#else
    if (event_us) throw std::runtime_error("CUDA event benchmark unavailable in this backend");
#endif
    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    return result;
}

int main(int argc, char ** argv) {
    try {
        bool q8_only = false, large_head = false, batched_q6_only = false, grouped_q6_hook = false;
        bool q6_real_shapes = false, benchmark_events = false;
        int benchmark = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--q8-reuse-only") q8_only = true;
            else if (key == "--batched-q6-only") batched_q6_only = true;
            else if (key == "--grouped-q6-hook") batched_q6_only = grouped_q6_hook = true;
            else if (key == "--q6-real-shapes") q6_real_shapes = true;
            else if (key == "--large-head") { large_head = true; q8_only = true; }
            else if (key == "--benchmark" && i + 1 < argc) { benchmark = std::stoi(argv[++i]); q8_only = true; }
            else if (key == "--benchmark-events" && i + 1 < argc) { benchmark = std::stoi(argv[++i]); benchmark_events = q8_only = true; }
            else throw std::runtime_error("usage: strata-moe-native-dense-parity [--q8-reuse-only] [--large-head] [--batched-q6-only | --grouped-q6-hook [--q6-real-shapes]] [--benchmark N | --benchmark-events N]");
        }
        if (benchmark < 0 || benchmark > 1000000) throw std::runtime_error("benchmark must be 0..1000000");
        if (q6_real_shapes && !grouped_q6_hook) throw std::runtime_error("--q6-real-shapes requires --grouped-q6-hook");
        ggml_backend_load_all();
        auto device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!device) { std::puts("SKIP: native dense parity requires a GPU"); return 77; }
        const char * reuse = std::getenv("STRATA_A3B_DENSE_Q8_REUSE");
        std::printf("native dense device=%s STRATA_A3B_DENSE_Q8_REUSE=%s Q8_only=%d large_head=%d\n",
                    ggml_backend_dev_name(device), reuse ? reuse : "unset", (int) q8_only, (int) large_head);
        auto gpu = ggml_backend_dev_init(device, nullptr);
        strata_a3b::Config cfg;
        cfg.gpu = device; cfg.native_dense_exact = true; cfg.native_exact_max_tokens = 9;
        auto reg = strata_a3b::register_backend(cfg);
        using Hook = ggml_tensor * (*)(ggml_context *, ggml_tensor *, ggml_tensor *);
        auto hook = reinterpret_cast<Hook>(ggml_backend_reg_get_proc_address(reg, "strata_a3b_build_mm_v1"));
        if (!hook) throw std::runtime_error("native dense hook was not registered");
        const std::vector<ggml_type> types = batched_q6_only ? std::vector<ggml_type>{GGML_TYPE_Q6_K} :
            q8_only ? std::vector<ggml_type>{GGML_TYPE_Q8_0} :
            std::vector<ggml_type>{GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
                                   GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16};
        std::vector<std::pair<int, int>> shapes;
        if (batched_q6_only) shapes = q6_real_shapes ?
            std::vector<std::pair<int, int>>{{2048, 512}, {2048, 2048}, {2048, 4096}, {2048, 8192}, {2048, 248320}} :
            std::vector<std::pair<int, int>>{{2048, 2048}};
        else if (large_head) shapes = {{2048, 248320}};
        else if (q8_only) shapes = {{256, 256}, {512, 2048}, {768, 256}, {1024, 256},
                                   {2048, 512}, {2048, 8192}, {4096, 2048}, {8192, 256}};
        else for (int ni : {256, 2048, 8192}) for (int no : {1, 32, 256}) shapes.emplace_back(ni, no);
        for (auto type : types) for (const auto & shape : shapes) {
            const int ni = shape.first, no = shape.second;
            if (!q8_only && !batched_q6_only && ggml_is_quantized(type) && no != 256) continue;
            auto weight_ctx = ggml_init({ggml_tensor_overhead() * 2, nullptr, true});
            auto * weight = ggml_new_tensor_2d(weight_ctx, type, ni, no);
            auto weight_buffer = ggml_backend_alloc_ctx_tensors(weight_ctx, gpu);
            ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            // The optional vocabulary-sized fixture repeats 128 independently
            // quantized rows, avoiding a 1.9 GiB temporary float matrix.
            const int packed_rows = large_head || no >= 100000 ? std::min(no, 128) : no;
            std::vector<float> values(ni * packed_rows);
            for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(float(i) * 0.017f) * 0.1f;
            const size_t row_bytes = ggml_row_size(type, ni);
            std::vector<char> packed(row_bytes * packed_rows);
            ggml_quantize_chunk(type, values.data(), packed.data(), 0, packed_rows, ni, nullptr);
            for (int row = 0; row < no; row += packed_rows)
                ggml_backend_tensor_set(weight, packed.data(), row * row_bytes,
                                        std::min(packed_rows, no - row) * row_bytes);
            for (bool padded : {false, true}) {
                const int stride = ni + (padded ? 256 : 0);
                std::vector<float> inputs(stride * 10), expected(no * 9);
                for (size_t i = 0; i < inputs.size(); ++i) {
                    inputs[i] = std::cos(float(i) * 0.071f) * 0.25f;
                    if (q8_only) {
                        const size_t token = i / stride;
                        inputs[i] = token == 1 ? 0.0f : std::ldexp(inputs[i], int((i / 32) % 9) - 4);
                    }
                }
                // Each oracle call is a separate ordinary one-token graph;
                // retain its results to avoid retesting the same prefix rows.
                for (int t = 0; t < 9; ++t) {
                    auto seq_ctx = ggml_init({ggml_tensor_overhead() * 4 + ggml_graph_overhead_custom(32, false), nullptr, true});
                    auto * sx = ggml_new_tensor_2d(seq_ctx, GGML_TYPE_F32, ni, 1);
                    auto * sy = ggml_mul_mat(seq_ctx, weight, sx);
                    auto reference = compute(gpu, seq_ctx, sy, sx, inputs.data() + t * stride, ni);
                    std::copy(reference.begin(), reference.end(), expected.begin() + t * no);
                    ggml_free(seq_ctx);
                }
                for (int nt : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
                    auto ctx = ggml_init({ggml_tensor_overhead() * 12 + ggml_graph_overhead_custom(32, false), nullptr, true});
                    auto * storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, stride, nt);
                    auto * input = padded ? ggml_view_2d(ctx, storage, ni, nt, storage->nb[1], 0) : storage;
                    auto * out = (batched_q6_only && !grouped_q6_hook) ? nullptr : hook(ctx, weight, input);
                    if (!batched_q6_only && (out != nullptr) != (nt >= 2 && nt <= 9))
                        throw std::runtime_error("native dense hook selection mismatch");
                    if (grouped_q6_hook && (out != nullptr) != (nt >= 2 && nt <= 9))
                        throw std::runtime_error("grouped Q6 hook selection mismatch");
                    const bool exact = out != nullptr;
                    const bool grouped = grouped_q6_hook && exact && nt >= 2 && nt <= 4;
                    if (exact && !grouped && (out->src[0]->op != GGML_OP_MUL_MAT || out->src[0]->ne[1] != 1 || out->src[0]->ne[2] != nt))
                        throw std::runtime_error("native dense hook did not preserve one-column CUDA layout");
                    if (grouped && (out->op != GGML_OP_MUL_MAT || out->ne[1] != nt))
                        throw std::runtime_error("grouped Q6 hook did not make one batched CUDA matrix product");
                    if (!out) out = ggml_mul_mat(ctx, weight, input);
                    double mean_us = 0, event_us = 0;
                    const int repeats = benchmark && !padded && nt <= 8 ? benchmark : 0;
                    auto actual = compute(gpu, ctx, out, storage, inputs.data(), stride * nt, repeats,
                                          benchmark_events ? nullptr : &mean_us,
                                          benchmark_events ? &event_us : nullptr);
                    if (exact || nt == 1 || (batched_q6_only && nt <= 4)) for (int t = 0; t < nt; ++t) {
                        for (int r = 0; r < no; ++r) if (!std::isfinite(actual[t * no + r]) ||
                                std::memcmp(&actual[t * no + r], &expected[t * no + r], sizeof(float)) != 0) {
                            std::fprintf(stderr, "dense parity %s ni=%d no=%d nt=%d padded=%d token=%d row=%d expected=%.9g actual=%.9g\n",
                                         ggml_type_name(type), ni, no, nt, padded, t, r, expected[t * no + r], actual[t * no + r]);
                            throw std::runtime_error("native dense exact graph differs from sequential GPU arithmetic");
                        }
                    }
                    std::printf("native dense %s ni=%d no=%d nt=%d padded=%d exact=%d PASS\n", ggml_type_name(type), ni, no, nt, padded, exact);
                    if (repeats) std::printf("BENCH native dense %s ni=%d no=%d nt=%d repeats=%d %s_us=%.3f (%s, not model throughput)\n",
                                            ggml_type_name(type), ni, no, nt, repeats,
                                            benchmark_events ? "cuda_event" : "host_graph", benchmark_events ? event_us : mean_us,
                                            benchmark_events ? "async graph submissions bracketed by CUDA events" : "synchronous graph API");
                    ggml_free(ctx);
                }
            }
            ggml_backend_buffer_free(weight_buffer); ggml_free(weight_ctx);
        }
        ggml_backend_free(gpu);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "native dense parity: %s\n", error.what());
        return 1;
    }
}
