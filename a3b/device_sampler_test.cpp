#include "speculative.hpp"
#include "ggml-cpu.h"

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    try {
        int gpu_index = -1, benchmark = 0;
        bool direct_first = false;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--device" && i + 1 < argc) gpu_index = std::stoi(argv[++i]);
            else if (key == "--benchmark" && i + 1 < argc) benchmark = std::stoi(argv[++i]);
            else if (key == "--assert-direct-first") direct_first = true;
            else throw std::runtime_error("usage: strata-moe-device-sampler-test [--device N] [--benchmark N | --assert-direct-first]");
        }
        if (benchmark < 0 || benchmark > 1000000 || (benchmark && direct_first))
            throw std::runtime_error("benchmark must be 0..1000000 and separate from direct-op validation");
        ggml_backend_load_all();
        ggml_backend_dev_t device = nullptr;
        if (gpu_index < 0) device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        else {
            int index = 0;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto d = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU && index++ == gpu_index) {
                    device = d; break;
                }
            }
        }
        if (!device) throw std::runtime_error("requested backend unavailable");
        const char * multi = std::getenv("STRATA_A3B_ARGMAX_MULTI");
        if (direct_first && (!multi || std::atoi(multi) == 0 ||
                            std::string(ggml_backend_dev_name(device)).rfind("CUDA", 0) != 0))
            throw std::runtime_error("direct first-maximum assertions require CUDA and STRATA_A3B_ARGMAX_MULTI=1 in a fresh process");
        std::printf("sampler test device=%s STRATA_A3B_ARGMAX_MULTI=%s direct_first=%d\n",
                    ggml_backend_dev_name(device), multi ? multi : "unset", (int) direct_first);
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(device, nullptr), ggml_backend_free);

        for (const int vocab : {4097, 16383, 16384, 16385, 248320, 248321}) {
            std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(strata_a3b::create_device_greedy_sampler(), llama_sampler_free);
            if (!sampler->iface->backend_init(sampler.get(), ggml_backend_dev_buffer_type(device), 9))
                throw std::runtime_error("device sampler graph unsupported");
            std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
            auto * logits = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, vocab);
            auto * graph = ggml_new_graph_custom(ctx.get(), 64, false);
            llama_sampler_data outputs{logits, nullptr, nullptr, nullptr};
            sampler->iface->backend_apply(sampler.get(), ctx.get(), graph, &outputs);
            if (outputs.logits || outputs.candidates || !outputs.probs || !outputs.sampled ||
                ggml_nelements(outputs.probs) != 1 || ggml_nelements(outputs.sampled) != 1)
                throw std::runtime_error("device sampler must expose only token plus scalar guard");
            ggml_build_forward_expand(graph, outputs.sampled);
            ggml_build_forward_expand(graph, outputs.probs);
            // The separate raw op is excluded from timing and from the
            // sampler's eight-byte output. It validates the new tie rule.
            auto * direct = direct_first && vocab >= 16384 ? ggml_argmax(ctx.get(), logits) : nullptr;
            if (direct) ggml_build_forward_expand(graph, direct);
            std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
                ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
            if (!buffer) throw std::runtime_error("sampler graph allocation failed");
            std::vector<float> values(vocab);
            for (size_t i = 0; i < values.size(); ++i) values[i] = -20.0f + float(i % 997) * 0.01f;
            for (int fixture = 0; fixture < 11; ++fixture) {
                auto input = values;
                input[17] = 100.0f;
                int expected = 17, expected_direct = 17;
                if (fixture == 1) input[vocab - 1] = input[4096] = 100.0f;
                if (fixture == 2) { input[vocab / 2] = std::numeric_limits<float>::infinity(); expected_direct = vocab / 2; }
                if (fixture == 3) input[vocab / 2] = -std::numeric_limits<float>::infinity();
                if (fixture == 4) input[vocab / 2] = std::numeric_limits<float>::quiet_NaN();
                if (fixture == 5) { std::fill(input.begin(), input.end(), -std::numeric_limits<float>::infinity()); expected_direct = 0; }
                if (fixture == 6) { std::fill(input.begin(), input.end(), std::numeric_limits<float>::quiet_NaN()); expected_direct = -1; }
                if (fixture == 7) { std::fill(input.begin(), input.end(), std::numeric_limits<float>::lowest()); expected = expected_direct = 0; }
                if (fixture == 8) { std::fill(input.begin(), input.end(), -0.0f); input[vocab - 1] = 0.0f; expected = expected_direct = 0; }
                if (fixture == 9) { input[17] = -5.0f; input[4095] = input[4096] = 100.0f; expected = expected_direct = 4095; }
                if (fixture == 10) { input[17] = -5.0f; input[vocab - 1] = 100.0f; expected = expected_direct = vocab - 1; }
                ggml_backend_tensor_set(logits, input.data(), 0, input.size() * sizeof(float));
                if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("device sampler graph execution failed");
                int32_t token = -1;
                float guard = -1;
                ggml_backend_tensor_get(outputs.sampled, &token, 0, sizeof(token));
                ggml_backend_tensor_get(outputs.probs, &guard, 0, sizeof(guard));
                const bool finite = fixture < 2 || fixture >= 7;
                if (finite ? (token != expected || guard != 0) : std::isfinite(guard))
                    throw std::runtime_error("device greedy fixture failed: vocab=" + std::to_string(vocab) + " fixture=" + std::to_string(fixture));
                if (direct) {
                    int32_t raw = -2;
                    ggml_backend_tensor_get(direct, &raw, 0, sizeof(raw));
                    if (raw != expected_direct) throw std::runtime_error("two-stage direct argmax first-maximum fixture failed");
                }
                std::printf("vocab=%d fixture=%d token=%d finite_guard=%d sampler_download_bytes=8 direct_checked=%d PASS\n",
                            vocab, fixture, token, (int) std::isfinite(guard), (int) (direct != nullptr));
            }
            if (benchmark && vocab == 248320) {
                // Warm the identical finite graph. Uploads and result reads
                // stay outside the measured synchronous backend-call loop.
                for (int i = 0; i < 10; ++i)
                    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS)
                        throw std::runtime_error("sampler benchmark warmup failed");
                const auto start = std::chrono::steady_clock::now();
                for (int i = 0; i < benchmark; ++i)
                    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS)
                        throw std::runtime_error("sampler benchmark failed");
                ggml_backend_synchronize(backend.get());
                const auto us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
                std::printf("BENCH sampler_graph vocab=%d repeats=%d mean_us=%.3f (synchronous backend API; not model throughput)\n",
                            vocab, benchmark, us / benchmark);
            }
        }
        {
            // The production compact branch must emit a full tokenizer ID
            // directly from the GPU sampler, before K2 reads t_sampled.
            constexpr int compact_vocab = 8;
            std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
                strata_a3b::create_device_greedy_sampler(), llama_sampler_free);
            if (!sampler->iface->backend_init(sampler.get(), ggml_backend_dev_buffer_type(device), 1))
                throw std::runtime_error("compact device sampler unsupported");
            std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
                ggml_init({1024 * 1024, nullptr, true}), ggml_free);
            auto * logits = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, compact_vocab);
            auto * map = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, compact_vocab);
            auto * graph = ggml_new_graph_custom(ctx.get(), 64, false);
            llama_sampler_data outputs{logits, nullptr, nullptr, map};
            sampler->iface->backend_apply(sampler.get(), ctx.get(), graph, &outputs);
            if (!outputs.sampled || outputs.candidates || outputs.logits || !outputs.probs ||
                outputs.sampled->type != GGML_TYPE_I32 || ggml_nelements(outputs.sampled) != 1)
                throw std::runtime_error("compact sampler must emit one mapped I32 token and one guard");
            ggml_build_forward_expand(graph, outputs.sampled);
            ggml_build_forward_expand(graph, outputs.probs);
            std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
                ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
            if (!buffer) throw std::runtime_error("compact sampler graph allocation failed");
            const std::array<int32_t, compact_vocab> ids = {0, 3, 7, 10, 15, 30, 41, 50};
            const std::array<float, compact_vocab> values = {-5, 9, 2, 9, 1, 0, -1, -2};
            ggml_backend_tensor_set(map, ids.data(), 0, sizeof(ids));
            ggml_backend_tensor_set(logits, values.data(), 0, sizeof(values));
            if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("compact sampler graph execution failed");
            int32_t token = -1;
            float guard = NAN;
            ggml_backend_tensor_get(outputs.sampled, &token, 0, sizeof(token));
            ggml_backend_tensor_get(outputs.probs, &guard, 0, sizeof(guard));
            if (token != 3 || guard != 0)
                throw std::runtime_error("compact sampler full-ID remap or first-ID tie failed");
            std::printf("compact sampler: subset 8 -> full token %d, finite_guard=%d PASS\n", token, (int) std::isfinite(guard));
        }
        std::printf("PASS device greedy sampler: %s, 66 threshold/tail/cross-tile/non-finite fixtures\n", ggml_backend_dev_name(device));
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
