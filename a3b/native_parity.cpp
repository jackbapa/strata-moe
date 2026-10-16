#include "backend.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static ggml_tensor * native_ffn(ggml_context * ctx, ggml_tensor * x, ggml_tensor * ids,
                                ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down) {
    auto * g = ggml_mul_mat_id(ctx, gate, x, ids);
    auto * u = ggml_mul_mat_id(ctx, up, x, ids);
    return ggml_mul_mat_id(ctx, down, ggml_swiglu_split(ctx, g, u), ids);
}

static void check_native_ffn(ggml_backend_t gpu, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down) {
    const int ni = (int) gate->ne[0], nk = 8;
    for (int nt : {1, 2, 4}) for (int lanes : {1, nk}) {
        auto ctx = ggml_init({ggml_tensor_overhead() * 24 + ggml_graph_overhead_custom(64, false), nullptr, true});
        auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ni, lanes, nt);
        auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, nk, nt);
        auto * out = strata_a3b::build_fused_moe(ctx, x, gate, up, down, ids);
        if ((out != nullptr) != (nt > 1)) throw std::runtime_error("native whole-FFN hook selection mismatch");
        if (!out) out = native_ffn(ctx, x, ids, gate, up, down);
        auto * graph = ggml_new_graph_custom(ctx, 64, false); ggml_build_forward_expand(graph, out);
        bool adjacent_gate_up = false;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
            if (ggml_graph_node(graph, i)->op == GGML_OP_CUSTOM) throw std::runtime_error("native FFN introduced a custom op");
            else if (i >= 2 && ggml_graph_node(graph, i)->op == GGML_OP_GLU &&
                     ggml_graph_node(graph, i - 1)->op == GGML_OP_MUL_MAT_ID &&
                     ggml_graph_node(graph, i - 2)->op == GGML_OP_MUL_MAT_ID) adjacent_gate_up = true;
        if (!adjacent_gate_up) throw std::runtime_error("native whole FFN interrupted the gate/up/GLU fusion pattern");
        auto buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
        std::vector<float> inputs(ni * lanes * nt);
        for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = std::cos(float(i) * 0.071f) * 0.25f;
        std::vector<int32_t> routes(nk * nt);
        for (int t = 0; t < nt; ++t) for (int k = 0; k < nk; ++k) routes[t * nk + k] = (t * 3 + k) % gate->ne[2];
        ggml_backend_tensor_set(x, inputs.data(), 0, inputs.size() * sizeof(float));
        ggml_backend_tensor_set(ids, routes.data(), 0, routes.size() * sizeof(int32_t));
        if (ggml_backend_graph_compute(gpu, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("native whole FFN failed");
        std::vector<float> actual(ggml_nelements(out));
        ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
        for (int t = 0; t < nt; ++t) {
            auto seq_ctx = ggml_init({ggml_tensor_overhead() * 12 + ggml_graph_overhead_custom(32, false), nullptr, true});
            auto * sx = ggml_new_tensor_3d(seq_ctx, GGML_TYPE_F32, ni, lanes, 1);
            auto * si = ggml_new_tensor_2d(seq_ctx, GGML_TYPE_I32, nk, 1);
            auto * sy = native_ffn(seq_ctx, sx, si, gate, up, down);
            auto * seq = ggml_new_graph_custom(seq_ctx, 32, false); ggml_build_forward_expand(seq, sy);
            auto seq_buf = ggml_backend_alloc_ctx_tensors(seq_ctx, gpu);
            ggml_backend_tensor_set(sx, inputs.data() + t * ni * lanes, 0, ni * lanes * sizeof(float));
            ggml_backend_tensor_set(si, routes.data() + t * nk, 0, nk * sizeof(int32_t));
            if (ggml_backend_graph_compute(gpu, seq) != GGML_STATUS_SUCCESS) throw std::runtime_error("native sequential FFN failed");
            std::vector<float> reference(ggml_nelements(sy));
            ggml_backend_tensor_get(sy, reference.data(), 0, reference.size() * sizeof(float));
            for (size_t i = 0; i < reference.size(); ++i)
                if (!std::isfinite(actual[t * reference.size() + i]) || actual[t * reference.size() + i] != reference[i]) {
                    std::fprintf(stderr, "native whole FFN %s/%s ni=%d hidden=%lld nt=%d lanes=%d token=%d index=%zu expected=%.9g actual=%.9g\n",
                        ggml_type_name(gate->type), ggml_type_name(down->type), ni, (long long) gate->ne[1],
                        nt, lanes, t, i, reference[i], actual[t * reference.size() + i]);
                    throw std::runtime_error("native whole-FFN exact graph differs from sequential GPU arithmetic");
                }
            ggml_backend_buffer_free(seq_buf); ggml_free(seq_ctx);
        }
        std::printf("native whole FFN %s/%s ni=%d hidden=%lld nt=%d lanes=%d max_abs=0 adjacent_gate_up=1 PASS\n", ggml_type_name(gate->type),
                    ggml_type_name(down->type), ni, (long long) gate->ne[1], nt, lanes);
        ggml_backend_buffer_free(buf); ggml_free(ctx);
    }
}

int main() {
    try {
        ggml_backend_load_all();
        auto device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!device) { std::puts("SKIP: native GPU parity requires a GPU"); return 77; }
        auto gpu = ggml_backend_dev_init(device, nullptr);
        strata_a3b::Config cfg;
        cfg.gpu = device; cfg.threads = 4; cfg.native_exact_verify = true; cfg.fused_moe = true;
        auto reg = strata_a3b::register_backend(cfg);
        using Hook = ggml_tensor * (*)(ggml_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *);
        auto hook = reinterpret_cast<Hook>(ggml_backend_reg_get_proc_address(reg, "strata_a3b_build_mmid_v1"));
        if (!hook) throw std::runtime_error("native graph hook was not registered");
        for (auto type : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0})
        for (int ni : {256, 2048}) {
            if (ni == 2048 && type != GGML_TYPE_Q4_K) continue;
            const int no = ni == 2048 ? 512 : 256;
            constexpr int ne = 16, nk = 8;
            std::vector<float> floats(ni * no * ne);
            for (size_t i = 0; i < floats.size(); ++i) floats[i] = std::sin(float(i) * 0.017f) * 0.1f;
            std::vector<char> packed(ggml_row_size(type, ni) * no * ne);
            ggml_quantize_chunk(type, floats.data(), packed.data(), 0, no * ne, ni, nullptr);
            auto weight_ctx = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
            auto * weights = ggml_new_tensor_3d(weight_ctx, type, ni, no, ne);
            auto * up_weights = ggml_new_tensor_3d(weight_ctx, type, ni, no, ne);
            auto * down_weights = ggml_new_tensor_3d(weight_ctx, type == GGML_TYPE_Q5_K ? GGML_TYPE_Q6_K : type, no, ni, ne);
            auto weight_buf = ggml_backend_alloc_ctx_tensors(weight_ctx, gpu);
            ggml_backend_buffer_set_usage(weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_tensor_set(weights, packed.data(), 0, packed.size());
            for (auto & value : floats) value = -value;
            ggml_quantize_chunk(type, floats.data(), packed.data(), 0, no * ne, ni, nullptr);
            ggml_backend_tensor_set(up_weights, packed.data(), 0, packed.size());
            for (auto & value : floats) value *= -0.5f;
            std::vector<char> down_packed(ggml_nbytes(down_weights));
            ggml_quantize_chunk(down_weights->type, floats.data(), down_packed.data(), 0, ni * ne, no, nullptr);
            ggml_backend_tensor_set(down_weights, down_packed.data(), 0, down_packed.size());
            for (int nt : {1, 2, 4, 5}) for (int lanes : {1, nk}) for (bool padded : {false, true}) {
                auto ctx = ggml_init({ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(64, false), nullptr, true});
                const int storage_lanes = lanes + (padded ? 1 : 0), storage_ids = nk + (padded ? 3 : 0);
                auto * xs = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ni, storage_lanes, nt);
                auto * is = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, storage_ids, nt);
                auto * x = padded ? ggml_view_3d(ctx, xs, ni, lanes, nt, ni * sizeof(float),
                                                ni * storage_lanes * sizeof(float), 0) : xs;
                auto * ids = padded ? ggml_view_2d(ctx, is, nk, nt, storage_ids * sizeof(int32_t), 0) : is;
                auto * output = hook(ctx, weights, x, ids);
                if ((output != nullptr) != (nt > 1 && nt <= 4))
                    throw std::runtime_error("native graph hook window selection mismatch");
                if (!output) output = ggml_mul_mat_id(ctx, weights, x, ids);
                auto * graph = ggml_new_graph_custom(ctx, 64, false); ggml_build_forward_expand(graph, output);
                auto buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
                std::vector<float> inputs(ni * storage_lanes * nt);
                for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = std::cos(float(i) * 0.071f) * 0.25f;
                std::vector<int32_t> routes(storage_ids * nt, -1);
                for (int t = 0; t < nt; ++t) for (int k = 0; k < nk; ++k)
                    routes[t * storage_ids + k] = (t * 3 + k) % ne;
                ggml_backend_tensor_set(xs, inputs.data(), 0, inputs.size() * sizeof(float));
                ggml_backend_tensor_set(is, routes.data(), 0, routes.size() * sizeof(int32_t));
                if (ggml_backend_graph_compute(gpu, graph) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("native exact graph failed");
                std::vector<float> actual(ggml_nelements(output));
                ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
                float max_error = 0;
                for (int t = 0; t < nt; ++t) {
                    auto seq_ctx = ggml_init({ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(32, false), nullptr, true});
                    auto * sx = ggml_new_tensor_3d(seq_ctx, GGML_TYPE_F32, ni, lanes, 1);
                    auto * si = ggml_new_tensor_2d(seq_ctx, GGML_TYPE_I32, nk, 1);
                    auto * sy = ggml_mul_mat_id(seq_ctx, weights, sx, si);
                    auto * seq = ggml_new_graph_custom(seq_ctx, 32, false); ggml_build_forward_expand(seq, sy);
                    auto seq_buf = ggml_backend_alloc_ctx_tensors(seq_ctx, gpu);
                    ggml_backend_tensor_set(sx, inputs.data() + t * ni * storage_lanes, 0, ni * lanes * sizeof(float));
                    ggml_backend_tensor_set(si, routes.data() + t * storage_ids, 0, nk * sizeof(int32_t));
                    if (ggml_backend_graph_compute(gpu, seq) != GGML_STATUS_SUCCESS)
                        throw std::runtime_error("independent sequential GPU oracle failed");
                    std::vector<float> reference(no * nk);
                    ggml_backend_tensor_get(sy, reference.data(), 0, reference.size() * sizeof(float));
                    for (size_t i = 0; i < reference.size(); ++i) {
                        const float value = actual[t * reference.size() + i];
                        if (!std::isfinite(value)) throw std::runtime_error("native exact output is not finite");
                        const float error = std::abs(value - reference[i]); max_error = std::max(max_error, error);
                        if (nt <= 4 && error != 0.f)
                            throw std::runtime_error("native exact graph differs from sequential GPU arithmetic");
                    }
                    ggml_backend_buffer_free(seq_buf); ggml_free(seq_ctx);
                }
                std::printf("%s ni=%d no=%d nt=%d lanes=%d padded=%d max_abs=%.9g PASS\n", ggml_type_name(type), ni, no, nt, lanes, padded, max_error);
                ggml_backend_buffer_free(buf); ggml_free(ctx);
            }
            check_native_ffn(gpu, weights, up_weights, down_weights);
            ggml_backend_buffer_free(weight_buf); ggml_free(weight_ctx);
        }
        ggml_backend_free(gpu);
        std::puts("PASS native graph hook: exact sequential GPU arithmetic for 1/2/4 tokens");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1;
    }
}
