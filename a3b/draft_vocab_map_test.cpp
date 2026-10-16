#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "draft_vocab_map.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>

int main() {
    try {
        strata_a3b::detail::validate_draft_vocab_ids({0, 3, 7}, 8, {3});
        for (const auto & bad : {std::vector<int32_t>{0, 0, 3},
                                 std::vector<int32_t>{0, 3, 8},
                                 std::vector<int32_t>{3, 0, 7}}) {
            bool rejected = false;
            try { strata_a3b::detail::validate_draft_vocab_ids(bad, 8, {3}); }
            catch (const std::runtime_error &) { rejected = true; }
            if (!rejected) throw std::runtime_error("corrupt compact draft map accepted");
        }
        bool missing_special = false;
        try { strata_a3b::detail::validate_draft_vocab_ids({0, 7}, 8, {3}); }
        catch (const std::runtime_error &) { missing_special = true; }
        if (!missing_special) throw std::runtime_error("missing end token accepted");
        constexpr int n_vocab = 8;
        constexpr int n_draft = 3;
        constexpr int n_outputs = 2;
        const std::array<int64_t, n_draft> ids = {0, 3, 7};
        const std::array<float, n_draft * n_outputs> compact = {1, 2, 4, 6, 9, 7};
        const std::array<float, 2 * n_vocab> embedding = {
             0,  1, 10, 11, 20, 21, 30, 31,
            40, 41, 50, 51, 70, 71, 80, 81,
        };
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
            ggml_init({1024 * 1024, nullptr, true}), ggml_free);
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            ggml_backend_cpu_init(), ggml_backend_free);
        if (!ctx || !backend) throw std::runtime_error("CPU GGML init failed");
        ggml_tensor * subset = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_draft, n_outputs);
        ggml_tensor * d2t = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, n_draft);
        ggml_tensor * d2t_i32 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_draft);
        ggml_tensor * embd = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 2, n_vocab);
        ggml_tensor * filled = ggml_fill(ctx.get(),
            ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, n_vocab, n_outputs),
            std::numeric_limits<float>::lowest());
        ggml_tensor * full = ggml_set_rows(ctx.get(), filled,
            ggml_reshape_3d(ctx.get(), subset, 1, n_draft, n_outputs),
            ggml_reshape_3d(ctx.get(), d2t, n_draft, 1, 1));
        full = ggml_reshape_2d(ctx.get(), full, n_vocab, n_outputs);
        ggml_tensor * filled_i32 = ggml_fill(ctx.get(),
            ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, n_vocab, n_outputs),
            std::numeric_limits<float>::lowest());
        ggml_tensor * full_i32 = ggml_set_rows(ctx.get(), filled_i32,
            ggml_reshape_3d(ctx.get(), subset, 1, n_draft, n_outputs),
            ggml_reshape_3d(ctx.get(), d2t_i32, n_draft, 1, 1));
        full_i32 = ggml_reshape_2d(ctx.get(), full_i32, n_vocab, n_outputs);
        ggml_tensor * second = ggml_cont(ctx.get(), ggml_view_1d(ctx.get(), full, n_vocab,
                                                                   n_vocab * sizeof(float)));
        ggml_tensor * sampled = ggml_argmax(ctx.get(), second);
        ggml_tensor * next_embd = ggml_get_rows(ctx.get(), embd, sampled);
        ggml_tensor * second_compact = ggml_cont(ctx.get(), ggml_view_1d(ctx.get(), subset,
                                                         n_draft, n_draft * sizeof(float)));
        ggml_tensor * compact_index = ggml_argmax(ctx.get(), second_compact);
        ggml_tensor * compact_token = ggml_get_rows(ctx.get(),
            ggml_reshape_2d(ctx.get(), d2t_i32, 1, n_draft), compact_index);
        ggml_tensor * compact_next_embd = ggml_get_rows(ctx.get(), embd, compact_token);
        ggml_cgraph * graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, full);
        ggml_build_forward_expand(graph, full_i32);
        ggml_build_forward_expand(graph, sampled);
        ggml_build_forward_expand(graph, next_embd);
        ggml_build_forward_expand(graph, compact_next_embd);
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
        if (!buffer) throw std::runtime_error("CPU graph allocation failed");
        ggml_backend_tensor_set(subset, compact.data(), 0, sizeof(compact));
        ggml_backend_tensor_set(d2t, ids.data(), 0, sizeof(ids));
        const std::array<int32_t, n_draft> ids_i32 = {0, 3, 7};
        ggml_backend_tensor_set(d2t_i32, ids_i32.data(), 0, sizeof(ids_i32));
        ggml_backend_tensor_set(embd, embedding.data(), 0, sizeof(embedding));
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("CPU scatter graph failed");
        std::array<float, n_vocab * n_outputs> actual{};
        std::array<float, n_vocab * n_outputs> actual_i32{};
        ggml_backend_tensor_get(full, actual.data(), 0, sizeof(actual));
        ggml_backend_tensor_get(full_i32, actual_i32.data(), 0, sizeof(actual_i32));
        if (actual_i32 != actual)
            throw std::runtime_error("I32 full-vocabulary fallback differs from I64 scatter");
        for (int row = 0; row < n_outputs; ++row) {
            for (int id = 0; id < n_vocab; ++id) {
                const float value = actual[row * n_vocab + id];
                const int sub = id == 0 ? 0 : id == 3 ? 1 : id == 7 ? 2 : -1;
                const float expected = sub >= 0 ? compact[row * n_draft + sub]
                                                : std::numeric_limits<float>::lowest();
                if (!std::isfinite(value) || value != expected)
                    throw std::runtime_error("full-vocabulary scatter or finite sentinel mismatch");
            }
        }
        int32_t token = -1;
        std::array<float, 2> next{};
        ggml_backend_tensor_get(sampled, &token, 0, sizeof(token));
        ggml_backend_tensor_get(next_embd, next.data(), 0, sizeof(next));
        if (token != 3 || next != std::array<float, 2>{30, 31})
            throw std::runtime_error("sampled full token ID did not index the next-step embedding");
        int32_t compact_id = -1;
        std::array<float, 2> compact_embedding{};
        ggml_backend_tensor_get(compact_token, &compact_id, 0, sizeof(compact_id));
        ggml_backend_tensor_get(compact_next_embd, compact_embedding.data(), 0, sizeof(compact_embedding));
        if (compact_id != token || compact_embedding != next)
            throw std::runtime_error("compact I32 map did not preserve the full token/embedding chain");
        std::printf("PASS draft subset 3/8 -> full ID %d -> next embedding, finite CPU logits\n", token);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL draft vocabulary map: %s\n", error.what());
        return 1;
    }
}
