// Real IQ3_S-layer parity for the opt-in original-Strata long-prefill MMQ
// adapter. The independent oracle is GGML's normal three-MMID CUDA graph.
#include "strata_prefill_mmq_adapter.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const char * what) { if (!ok) throw std::runtime_error(what); }
void check_cuda(cudaError_t status, const char * what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}
template<class Fn> float time_cuda(cudaStream_t stream, int repeats, Fn && fn) {
    cudaEvent_t start = nullptr, end = nullptr;
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&end), "create end event");
    try {
        for (int i = 0; i < 3; ++i) fn();
        check_cuda(cudaStreamSynchronize(stream), "synchronize warmup");
        check_cuda(cudaEventRecord(start, stream), "record start");
        for (int i = 0; i < repeats; ++i) fn();
        check_cuda(cudaEventRecord(end, stream), "record end");
        check_cuda(cudaEventSynchronize(end), "synchronize end");
        float elapsed = 0;
        check_cuda(cudaEventElapsedTime(&elapsed, start, end), "elapsed time");
        cudaEventDestroy(start); cudaEventDestroy(end);
        return elapsed / repeats;
    } catch (...) {
        cudaEventDestroy(start); cudaEventDestroy(end);
        throw;
    }
}
struct Context {
    ggml_context * p = nullptr;
    explicit Context(int n = 32) {
        p = ggml_init({ggml_tensor_overhead() * n + ggml_graph_overhead_custom(64, false), nullptr, true});
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
const strata::TensorInfo * find(const strata::GgufFile & f, int layer, const char * role) {
    const std::string name = "blk." + std::to_string(layer) + ".ffn_" + role + "_exps.weight";
    for (const auto & t : f.tensors()) if (t.name == name) return &t;
    return nullptr;
}
void run(const char * path, int layer, int tokens, int lanes, int repeats) {
    require(tokens >= 9 && tokens <= 2048 && (lanes == 1 || lanes == 8) &&
            repeats >= 0 && repeats <= 1000, "invalid token count, lanes, or repeats");
    constexpr int H = 2048, FF = 512, E = 256, K = 8;
    const strata::GgufFile file(path);
    const auto * gt = find(file, layer, "gate"), * ut = find(file, layer, "up"), * dt = find(file, layer, "down");
    require(gt && ut && dt, "missing routed expert tensor");
    require(gt->type == ut->type && gt->shape == ut->shape &&
            gt->shape[0] == H && gt->shape[1] == FF && gt->shape[2] == E &&
            dt->shape[0] == FF && dt->shape[1] == H && dt->shape[2] == E,
            "unexpected GGUF tensor layout");
    Backend gpu; ggml_backend_load_all(); gpu.p = ggml_backend_cuda_init(0);
    require(gpu.p != nullptr, "CUDA GPU0 unavailable");

    Context wc;
    auto * gate = ggml_new_tensor_3d(wc.p, static_cast<ggml_type>(gt->type), H, FF, E);
    auto * up = ggml_new_tensor_3d(wc.p, static_cast<ggml_type>(ut->type), H, FF, E);
    auto * down = ggml_new_tensor_3d(wc.p, static_cast<ggml_type>(dt->type), FF, H, E);
    Buffer wb; wb.p = ggml_backend_alloc_ctx_tensors(wc.p, gpu.p);
    require(wb.p != nullptr, "GPU weights allocation failed");
    ggml_backend_buffer_set_usage(wb.p, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(gate, file.tensor_data(*gt), 0, ggml_nbytes(gate));
    ggml_backend_tensor_set(up, file.tensor_data(*ut), 0, ggml_nbytes(up));
    ggml_backend_tensor_set(down, file.tensor_data(*dt), 0, ggml_nbytes(down));

    Context ic;
    auto * x = ggml_new_tensor_3d(ic.p, GGML_TYPE_F32, H, lanes, tokens);
    auto * ids = ggml_new_tensor_2d(ic.p, GGML_TYPE_I32, K, tokens);
    auto * actual = ggml_new_tensor_3d(ic.p, GGML_TYPE_F32, H, K, tokens);
    Buffer ib; ib.p = ggml_backend_alloc_ctx_tensors(ic.p, gpu.p);
    require(ib.p != nullptr, "GPU activation allocation failed");
    std::vector<float> xv(size_t(H) * lanes * tokens);
    std::vector<int32_t> rv(size_t(K) * tokens);
    for (size_t i = 0; i < xv.size(); ++i) xv[i] = std::sin(float(i % 65521) * 0.017f) * 0.1f;
    for (int t = 0; t < tokens; ++t) for (int k = 0; k < K; ++k)
        rv[size_t(t) * K + k] = (t * 31 + k * 29) % E;
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
    ggml_backend_tensor_set(ids, rv.data(), 0, rv.size() * sizeof(int32_t));

    Context gc;
    auto * g = ggml_mul_mat_id(gc.p, gate, x, ids);
    auto * u = ggml_mul_mat_id(gc.p, up, x, ids);
    auto * h = ggml_swiglu_split(gc.p, g, u);
    auto * expected = ggml_mul_mat_id(gc.p, down, h, ids);
    auto * graph = ggml_new_graph_custom(gc.p, 64, false);
    ggml_build_forward_expand(graph, expected);
    Buffer gb; gb.p = ggml_backend_alloc_ctx_tensors(gc.p, gpu.p);
    require(gb.p != nullptr, "GPU oracle graph allocation failed");
    require(ggml_backend_graph_compute(gpu.p, graph) == GGML_STATUS_SUCCESS, "GGML oracle failed");
    std::vector<float> want(size_t(H) * K * tokens), got(want.size());
    ggml_backend_tensor_get(expected, want.data(), 0, want.size() * sizeof(float));

    require(strata_a3b::strata_prefill_mmq_compatible(gate, up, down, x, ids, actual),
            "this IQ3_S layer is not compatible with Strata prefill MMQ");
    auto * state = strata_a3b::strata_prefill_mmq_create(K * tokens, H, FF, E);
    try {
        strata_a3b::strata_prefill_mmq_compute(state, ggml_backend_cuda_stream(gpu.p),
                                                gate, up, down, x, ids, actual);
        ggml_backend_synchronize(gpu.p);
        ggml_backend_tensor_get(actual, got.data(), 0, got.size() * sizeof(float));
        require(strata_a3b::strata_prefill_mmq_calls(state) == 1 &&
                strata_a3b::strata_prefill_mmq_rows(state) == uint64_t(K * tokens),
                "Strata prefill MMQ counter does not match the CUDA execution");
        if (repeats) {
            const auto stream = static_cast<cudaStream_t>(ggml_backend_cuda_stream(gpu.p));
            auto oracle = [&] {
                require(ggml_backend_graph_compute_async(gpu.p, graph) == GGML_STATUS_SUCCESS,
                        "timed GGML oracle failed");
            };
            auto adapter = [&] {
                strata_a3b::strata_prefill_mmq_compute(state, stream,
                                                        gate, up, down, x, ids, actual);
            };
            const float ggml_a = time_cuda(stream, repeats, oracle);
            const float strata_a = time_cuda(stream, repeats, adapter);
            const float strata_b = time_cuda(stream, repeats, adapter);
            const float ggml_b = time_cuda(stream, repeats, oracle);
            require(strata_a3b::strata_prefill_mmq_calls(state) == uint64_t(1 + 2 * (repeats + 3)),
                    "timed Strata prefill MMQ CUDA call counter mismatch");
            std::printf("layer=%d T=%d lanes=%d GGML three-MMID %.4f / %.4f ms; "
                        "original Strata split-MMQ %.4f / %.4f ms "
                        "(CUDA events, %d repeats each, warmed)\n",
                        layer, tokens, lanes, ggml_a, ggml_b, strata_a, strata_b, repeats);
        }
        strata_a3b::strata_prefill_mmq_destroy(state);
    } catch (...) { strata_a3b::strata_prefill_mmq_destroy(state); throw; }
    size_t bitwise = 0, outside = 0;
    float max_absolute = 0, max_relative = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        bitwise += std::memcmp(&got[i], &want[i], sizeof(float)) != 0;
        const float abs = std::abs(got[i] - want[i]);
        const float rel = abs / std::max(1e-5f, std::abs(want[i]));
        max_absolute = std::max(max_absolute, abs);
        max_relative = std::max(max_relative, rel);
        outside += !std::isfinite(got[i]) || abs > 1e-3f + 1e-3f * std::abs(want[i]);
    }
    std::printf("A3B original Strata MMQ layer=%d gate/up=%s down=%s T=%d lanes=%d "
                "bitwise_diff=%zu/%zu outside_1e-3=%zu max_abs=%.9g max_rel=%.9g "
                "prefill_calls=1 prefill_rows=%d\n",
                layer, ggml_type_name(gate->type), ggml_type_name(down->type), tokens, lanes,
                bitwise, got.size(), outside, max_absolute, max_relative, K * tokens);
    require(outside == 0, "Strata prefill MMQ exceeds numerical parity tolerance");
}
}

int main(int argc, char ** argv) {
    try {
        if (argc < 2 || argc > 6)
            throw std::runtime_error("usage: strata-moe-prefill-mmq-adapter-parity <IQ3_S.gguf> [layer=0] [tokens=128] [lanes=1] [bench_repeats=0]");
        run(argv[1], argc > 2 ? std::stoi(argv[2]) : 0,
            argc > 3 ? std::stoi(argv[3]) : 128,
            argc > 4 ? std::stoi(argv[4]) : 1,
            argc > 5 ? std::stoi(argv[5]) : 0);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "Strata prefill MMQ parity: %s\n", e.what());
        return 1;
    }
}
