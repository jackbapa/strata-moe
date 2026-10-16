// One real IQ3_S layer: measure the useful part of Strata's long-prefill
// arrangement (one grouped gate/up MMQ) against GGML's separate MMIDs.
// This probe deliberately excludes packing cost, so its speedup is an upper
// bound for GPU-resident A3B, whose GGUF stores gate and up separately.
#include "strata/artifact/gguf_reader.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
void require(bool ok, const char * what) { if (!ok) throw std::runtime_error(what); }
struct Context {
    ggml_context * p = nullptr;
    explicit Context(int tensors = 32) {
        const size_t bytes = ggml_tensor_overhead() * tensors + ggml_graph_overhead_custom(64, false);
        p = ggml_init({bytes, nullptr, true});
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

const strata::TensorInfo * tensor(const strata::GgufFile & f, int layer, const char * role) {
    const std::string name = "blk." + std::to_string(layer) + ".ffn_" + role + "_exps.weight";
    for (const auto & t : f.tensors()) if (t.name == name) return &t;
    return nullptr;
}

float time_graph(ggml_backend_t gpu, ggml_cgraph * graph, int repeats, cudaStream_t stream) {
    cudaEvent_t a = nullptr, b = nullptr;
    ck(cudaEventCreate(&a), "event start"); ck(cudaEventCreate(&b), "event end");
    for (int i = 0; i < 3; ++i)
        require(ggml_backend_graph_compute_async(gpu, graph) == GGML_STATUS_SUCCESS, "warmup failed");
    ck(cudaStreamSynchronize(stream), "warmup synchronization");
    ck(cudaEventRecord(a, stream), "record start");
    for (int i = 0; i < repeats; ++i)
        require(ggml_backend_graph_compute_async(gpu, graph) == GGML_STATUS_SUCCESS, "graph failed");
    ck(cudaEventRecord(b, stream), "record end"); ck(cudaEventSynchronize(b), "wait event");
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, a, b), "event elapsed");
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / repeats;
}

void run(const char * path, int layer, int tokens, int repeats) {
    require(repeats > 0 && repeats <= 1000, "repeats must be 1..1000");
    const strata::GgufFile file(path);
    const auto * tg = tensor(file, layer, "gate"), * tu = tensor(file, layer, "up");
    require(tg && tu, "layer has no routed gate/up weights");
    require(tg->type == tu->type && tg->shape == tu->shape, "gate/up type or shape differs");
    const int H = int(tg->shape[0]), FF = int(tg->shape[1]), E = int(tg->shape[2]);
    constexpr int K = 8;
    require(H == 2048 && FF == 512 && E > K && tokens >= 1 && tokens <= 2048,
            "unexpected IQ3_S geometry or token count");
    const auto type = static_cast<ggml_type>(tg->type);
    require(type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_S,
            "this IQ3_S layer's gate/up is outside the MMQ probe");
    const size_t expert_bytes = ggml_row_size(type, H) * FF;
    const size_t role_bytes = expert_bytes * E;
    std::printf("one-layer packed gate+up residency: %.2f MiB extra (two separate GGUF roles already resident)\n",
                double(2 * role_bytes) / 1048576.0);
    Backend gpu; ggml_backend_load_all();
    gpu.p = ggml_backend_cuda_init(0);
    require(gpu.p != nullptr && ggml_backend_is_cuda(gpu.p), "CUDA backend unavailable");

    Context wc;
    auto * gate = ggml_new_tensor_3d(wc.p, type, H, FF, E);
    auto * up = ggml_new_tensor_3d(wc.p, type, H, FF, E);
    auto * packed = ggml_new_tensor_3d(wc.p, type, H, FF * 2, E);
    Buffer wb; wb.p = ggml_backend_alloc_ctx_tensors(wc.p, gpu.p);
    require(wb.p != nullptr, "GPU weight allocation failed");
    ggml_backend_buffer_set_usage(wb.p, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const auto * gh = file.tensor_data(*tg), * uh = file.tensor_data(*tu);
    ggml_backend_tensor_set(gate, gh, 0, role_bytes);
    ggml_backend_tensor_set(up, uh, 0, role_bytes);
    // One extra packed copy is the optimistic setup cost Strata's native
    // model packs avoid. GPU-resident A3B must pay or reserve this memory.
    std::vector<uint8_t> both(2 * role_bytes);
    for (int e = 0; e < E; ++e) {
        std::memcpy(both.data() + e * 2 * expert_bytes, gh + e * expert_bytes, expert_bytes);
        std::memcpy(both.data() + e * 2 * expert_bytes + expert_bytes, uh + e * expert_bytes, expert_bytes);
    }
    ggml_backend_tensor_set(packed, both.data(), 0, both.size());

    Context ic;
    auto * x = ggml_new_tensor_3d(ic.p, GGML_TYPE_F32, H, 1, tokens);
    auto * ids = ggml_new_tensor_2d(ic.p, GGML_TYPE_I32, K, tokens);
    Buffer ib; ib.p = ggml_backend_alloc_ctx_tensors(ic.p, gpu.p);
    require(ib.p != nullptr, "GPU input allocation failed");
    std::vector<float> xv(size_t(H) * tokens);
    std::vector<int32_t> idv(size_t(K) * tokens);
    for (size_t i = 0; i < xv.size(); ++i) xv[i] = std::sin(float(i % 65521) * 0.017f) * 0.1f;
    for (int t = 0; t < tokens; ++t) for (int k = 0; k < K; ++k)
        idv[size_t(t) * K + k] = (t * 31 + k * 29) % E;
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
    ggml_backend_tensor_set(ids, idv.data(), 0, idv.size() * sizeof(int32_t));

    Context sc;
    auto * gout = ggml_mul_mat_id(sc.p, gate, x, ids);
    auto * uout = ggml_mul_mat_id(sc.p, up, x, ids);
    auto * sep = ggml_new_graph_custom(sc.p, 64, false);
    ggml_build_forward_expand(sep, gout); ggml_build_forward_expand(sep, uout);
    Buffer sb; sb.p = ggml_backend_alloc_ctx_tensors(sc.p, gpu.p);
    require(sb.p != nullptr, "separate MMID allocation failed");

    Context pc;
    auto * pout = ggml_mul_mat_id(pc.p, packed, x, ids);
    auto * one = ggml_new_graph_custom(pc.p, 64, false);
    ggml_build_forward_expand(one, pout);
    Buffer pb; pb.p = ggml_backend_alloc_ctx_tensors(pc.p, gpu.p);
    require(pb.p != nullptr, "packed MMID allocation failed");

    require(ggml_backend_graph_compute(gpu.p, sep) == GGML_STATUS_SUCCESS, "separate MMID failed");
    require(ggml_backend_graph_compute(gpu.p, one) == GGML_STATUS_SUCCESS, "packed MMID failed");
    const size_t nr = size_t(K) * tokens;
    std::vector<float> gv(nr * FF), uv(nr * FF), pv(nr * 2 * FF);
    ggml_backend_tensor_get(gout, gv.data(), 0, gv.size() * sizeof(float));
    ggml_backend_tensor_get(uout, uv.data(), 0, uv.size() * sizeof(float));
    ggml_backend_tensor_get(pout, pv.data(), 0, pv.size() * sizeof(float));
    size_t different = 0;
    for (size_t row = 0; row < nr; ++row) for (int j = 0; j < FF; ++j) {
        different += std::memcmp(&gv[row * FF + j], &pv[row * 2 * FF + j], sizeof(float)) != 0;
        different += std::memcmp(&uv[row * FF + j], &pv[row * 2 * FF + FF + j], sizeof(float)) != 0;
    }
    std::printf("A3B layer=%d type=%s experts=%d T=%d routes=%d packed_gate_up_diff=%zu/%zu\n",
                layer, ggml_type_name(type), E, tokens, K, different, nr * 2 * FF);
    require(different == 0, "packed gate/up differs from the original two-MMID outputs");
    const auto stream = static_cast<cudaStream_t>(ggml_backend_cuda_stream(gpu.p));
    const float a = time_graph(gpu.p, sep, repeats, stream);
    const float b = time_graph(gpu.p, one, repeats, stream);
    const float c = time_graph(gpu.p, one, repeats, stream);
    const float d = time_graph(gpu.p, sep, repeats, stream);
    std::printf("separate gate+up MMQ %.3f / %.3f ms, packed %.3f / %.3f ms (CUDA events, %d repeats each; packing excluded)\n",
                a, d, b, c, repeats);
}
}

int main(int argc, char ** argv) {
    try {
        if (argc < 2 || argc > 5)
            throw std::runtime_error("usage: strata-moe-prefill-mmq-probe <IQ3_S.gguf> [layer=0] [tokens=2048] [repeats=5]");
        run(argv[1], argc > 2 ? std::stoi(argv[2]) : 0,
            argc > 3 ? std::stoi(argv[3]) : 2048,
            argc > 4 ? std::stoi(argv[4]) : 5);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "prefill MMQ probe: %s\n", e.what());
        return 1;
    }
}
