#include "ggml.h"
#include "ggml-backend.h"
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

// The reference is an ordinary one-channel GPU MMID/FFN graph. Its one-channel
// shape cannot dispatch to the duplicate-ID reuse kernel, even when enabled.
// Do not use the new grouped kernel or CPU quantized arithmetic as the oracle.
static std::vector<float> compute(ggml_backend_t gpu, ggml_tensor * gate,
                                  ggml_tensor * up, ggml_tensor * down,
                                  const std::vector<float> & values,
                                  const std::vector<int32_t> & routes,
                                  int input_lanes, bool padded, bool ffn,
                                  int benchmark = 0) {
    const int ni = (int) gate->ne[0], channels = (int) routes.size();
    const int stride = ni + (padded ? 256 : 0);
    auto * ctx = ggml_init({ggml_tensor_overhead() * 12 + ggml_graph_overhead_custom(32, false), nullptr, true});
    if (!ctx) throw std::runtime_error("expert Q8 fixture context allocation failed");
    auto * storage = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, stride, input_lanes, 1);
    auto * x = padded ? ggml_view_3d(ctx, storage, ni, input_lanes, 1, storage->nb[1], storage->nb[2], 0) : storage;
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, channels, 1);
    auto * out = ggml_mul_mat_id(ctx, gate, x, ids);
    if (ffn) {
        auto * u = ggml_mul_mat_id(ctx, up, x, ids);
        auto * h = ggml_swiglu_split(ctx, out, u);
        out = ggml_mul_mat_id(ctx, down, h, ids);
    }
    auto * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, out);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    if (!buffer) throw std::runtime_error("expert Q8 fixture GPU allocation failed");
    ggml_backend_tensor_set(storage, values.data(), 0, values.size() * sizeof(float));
    ggml_backend_tensor_set(ids, routes.data(), 0, routes.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(gpu, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("expert Q8 fixture computation failed");
    if (benchmark) {
        for (int repeat = 0; repeat < 5; ++repeat)
            if (ggml_backend_graph_compute(gpu, graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("expert Q8 benchmark warmup failed");
        const auto start = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < benchmark; ++repeat)
            if (ggml_backend_graph_compute(gpu, graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("expert Q8 benchmark failed");
        ggml_backend_synchronize(gpu);
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / benchmark;
        std::printf("BENCH expert Q8 K=%d M=%lld channels=%d lanes=%d padded=%d ffn=%d repeats=%d mean_us=%.3f (synchronous graph API, not model throughput)\n",
                    ni, (long long) gate->ne[1], channels, input_lanes, (int) padded, (int) ffn, benchmark, us);
    }
    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
    return result;
}

int main(int argc, char ** argv) {
    try {
        int benchmark = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--benchmark" && i + 1 < argc) benchmark = std::stoi(argv[++i]);
            else throw std::runtime_error("usage: strata-moe-expert-q8-reuse-test [--benchmark N]");
        }
        if (benchmark < 0 || benchmark > 1000000) throw std::runtime_error("benchmark must be 0..1000000");
        ggml_backend_load_all();
        auto device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!device) { std::puts("SKIP: expert Q8 duplicate-ID parity requires a GPU"); return 77; }
        auto gpu = ggml_backend_dev_init(device, nullptr);
        if (!gpu) throw std::runtime_error("GPU backend initialization failed");
        const char * env = std::getenv("STRATA_A3B_EXPERT_Q8_REUSE");
        std::printf("expert Q8 duplicate-ID parity device=%s STRATA_A3B_EXPERT_Q8_REUSE=%s\n",
                    ggml_backend_dev_name(device), env ? env : "unset");
        constexpr int experts = 67;
        int checked = 0;
        for (auto shape : {std::pair<int, int>{2048, 512}, {512, 2048}, {256, 32}, {768, 32}}) {
            const int ni = shape.first, ff = shape.second;
            auto * weights = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
            auto * gate = ggml_new_tensor_3d(weights, GGML_TYPE_Q8_0, ni, ff, experts);
            auto * up = ggml_new_tensor_3d(weights, GGML_TYPE_Q8_0, ni, ff, experts);
            auto * down = ggml_new_tensor_3d(weights, GGML_TYPE_Q8_0, ff, ni, experts);
            auto buffer = ggml_backend_alloc_ctx_tensors(weights, gpu);
            if (!buffer) throw std::runtime_error("expert Q8 weight allocation failed");
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            int role = 0;
            for (auto * weight : {gate, up, down}) {
                std::vector<float> data(weight->ne[0] * weight->ne[1]);
                std::vector<char> packed(weight->nb[2]);
                for (int e = 0; e < experts; ++e) {
                    for (size_t i = 0; i < data.size(); ++i) {
                        data[i] = std::sin(float(i) * (0.017f + role * 0.003f) + float(e)) * 0.05f;
                        data[i] = std::ldexp(data[i], int((i / 32 + e) % 9) - 4);
                    }
                    ggml_quantize_chunk(GGML_TYPE_Q8_0, data.data(), packed.data(), 0, weight->ne[1], weight->ne[0], nullptr);
                    ggml_backend_tensor_set(weight, packed.data(), e * weight->nb[2], packed.size());
                }
                ++role;
            }
            const std::vector<int> counts = ni >= 512 && ff >= 512 ?
                std::vector<int>{1, 2, 3, 8, 24, 64, 65} : std::vector<int>{3, 24};
            for (int channels : counts) for (int pattern = 0; pattern < 3; ++pattern)
                for (bool broadcast : {false, true}) for (bool padded : {false, true}) {
                    const int lanes = broadcast ? 1 : channels, stride = ni + (padded ? 256 : 0);
                    std::vector<float> input(stride * lanes);
                    for (size_t i = 0; i < input.size(); ++i) {
                        input[i] = std::cos(float(i) * 0.071f) * 0.25f;
                        input[i] = std::ldexp(input[i], int((i / 32) % 11) - 5);
                        if (i / stride == 1) input[i] = 0.0f;
                    }
                    std::vector<int32_t> ids(channels);
                    for (int c = 0; c < channels; ++c)
                        ids[c] = pattern == 0 ? (c * 19 + 7) % experts : pattern == 1 ? 13 : (c * 7 + c / 3) % 11;
                    for (bool ffn : {false, true}) {
                        const int no = ffn ? ni : ff;
                        std::vector<float> expected(channels * no);
                        for (int c = 0; c < channels; ++c) {
                            const float * row = input.data() + (c % lanes) * stride;
                            std::vector<float> singleton(row, row + ni);
                            auto ref = compute(gpu, gate, up, down, singleton, {ids[c]}, 1, false, ffn);
                            std::copy(ref.begin(), ref.end(), expected.begin() + c * no);
                        }
                        const int repeats = benchmark && !padded && !broadcast && pattern != 0 && channels <= 64 ? benchmark : 0;
                        auto actual = compute(gpu, gate, up, down, input, ids, lanes, padded, ffn, repeats);
                        for (size_t i = 0; i < actual.size(); ++i)
                            if (!std::isfinite(actual[i]) || std::memcmp(&actual[i], &expected[i], sizeof(float))) {
                                std::fprintf(stderr, "Q8 expert mismatch K=%d M=%d channels=%d lanes=%d padded=%d pattern=%d ffn=%d index=%zu expected=%.9g actual=%.9g\n",
                                             ni, ff, channels, lanes, (int) padded, pattern, (int) ffn, i, expected[i], actual[i]);
                                throw std::runtime_error("Q8 expert reuse differs from independent singleton GPU oracle");
                            }
                        ++checked;
                    }
                    std::printf("expert Q8 K=%d M=%d channels=%d lanes=%d padded=%d pattern=%d projection+fusedFFN strict PASS\n",
                                ni, ff, channels, lanes, (int) padded, pattern);
                }
            ggml_backend_buffer_free(buffer); ggml_free(weights);
        }
        ggml_backend_free(gpu);
        std::printf("PASS %d Q8 expert projection/fused FFN cases against independent one-channel GPU oracle\n", checked);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "expert Q8 duplicate-ID parity: %s\n", e.what());
        return 1;
    }
}
