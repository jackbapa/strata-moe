#include "backend.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static ggml_context * context() {
    return ggml_init({ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(64, false), nullptr, true});
}
static void attention_masks(ggml_tensor * result, std::vector<ggml_tensor *> & masks) {
    if (result->op == GGML_OP_FLASH_ATTN_EXT) {
        masks.push_back(result->src[3]);
    } else if (result->op == GGML_OP_CONCAT) {
        attention_masks(result->src[0], masks);
        attention_masks(result->src[1], masks);
    } else throw std::runtime_error("unexpected exact attention result topology");
}
static std::vector<float> run(ggml_backend_t gpu, ggml_context * ctx, ggml_tensor * out,
    ggml_tensor * q, const std::vector<float> & queries, ggml_tensor * mask,
    const std::vector<ggml_fp16_t> & masks) {
    auto * graph = ggml_new_graph_custom(ctx, 64, false); ggml_build_forward_expand(graph, out);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    if (!buffer) throw std::runtime_error("attention allocation failed");
    ggml_backend_tensor_set(q, queries.data(), 0, queries.size() * sizeof(float));
    ggml_backend_tensor_set(mask, masks.data(), 0, masks.size() * sizeof(ggml_fp16_t));
    if (ggml_backend_graph_compute(gpu, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("attention computation failed");
    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    return result;
}
int main() {
    try {
        ggml_backend_load_all();
        auto device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!device) { std::puts("SKIP: native attention parity requires a GPU"); return 77; }
        auto gpu = ggml_backend_dev_init(device, nullptr);
        strata_a3b::Config cfg;
        cfg.gpu = device; cfg.native_exact_verify = true; cfg.native_attention_exact = true;
        cfg.native_exact_max_tokens = 9;
        auto reg = strata_a3b::register_backend(cfg);
        using Hook = ggml_tensor * (*)(ggml_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, float, float, float, int32_t, ggml_tensor **);
        auto hook = reinterpret_cast<Hook>(ggml_backend_reg_get_proc_address(reg, "strata_a3b_build_attention_v1"));
        if (!hook) throw std::runtime_error("attention hook was not registered");
        const char * mask_reuse_env = std::getenv("STRATA_A3B_ATTENTION_MASK_REUSE");
        const bool mask_reuse = mask_reuse_env && std::strcmp(mask_reuse_env, "1") == 0;
        constexpr int d = 256, hq = 16, hk = 2, mask_rows = 32;
        const float scale = 1.0f / std::sqrt(float(d));
        for (int nk : {128, 256, 512, 2048, 4096}) {
            auto cache_ctx = context();
            auto * k = ggml_new_tensor_4d(cache_ctx, GGML_TYPE_F16, d, nk, hk, 1);
            auto * v = ggml_new_tensor_4d(cache_ctx, GGML_TYPE_F16, d, nk, hk, 1);
            auto cache_buffer = ggml_backend_alloc_ctx_tensors(cache_ctx, gpu);
            std::vector<float> floats(d * nk * hk);
            std::vector<ggml_fp16_t> halves(floats.size());
            for (size_t i = 0; i < floats.size(); ++i) floats[i] = std::sin(float(i) * 0.013f) * 0.4f;
            ggml_fp32_to_fp16_row(floats.data(), halves.data(), halves.size());
            ggml_backend_tensor_set(k, halves.data(), 0, halves.size() * sizeof(ggml_fp16_t));
            for (size_t i = 0; i < floats.size(); ++i) floats[i] = std::cos(float(i) * 0.017f) * 0.3f;
            ggml_fp32_to_fp16_row(floats.data(), halves.data(), halves.size());
            ggml_backend_tensor_set(v, halves.data(), 0, halves.size() * sizeof(ggml_fp16_t));
            for (int nt : {1, 2, 4, 5, 9}) for (int prefix : {8, nk - nt - 1}) {
                auto ctx = context();
                auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, nt, hq, 1);
                auto * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, nk, mask_rows, 1, 1);
                ggml_tensor * probe = nullptr;
                auto * fa = hook(ctx, q, k, v, mask, nullptr, scale, 0.0f, 0.0f, 0, &probe);
                if ((fa != nullptr) != (nt > 1)) throw std::runtime_error("attention hook selection mismatch");
                if (!fa) fa = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
                if (nt > 1 && (!probe || probe->op != GGML_OP_FLASH_ATTN_EXT ||
                    probe->src[0]->ne[1] != 1 || probe->src[0]->ne[3] != 1 ||
                    probe->src[1] != k || probe->src[2] != v))
                    throw std::runtime_error("attention hook did not preserve singleton queries/shared cache");
                if (nt > 1) {
                    ggml_tensor * second_probe = nullptr;
                    auto * second = hook(ctx, q, k, v, mask, nullptr, scale, 0.0f, 0.0f, 0, &second_probe);
                    if (!second || second == fa || second_probe == probe)
                        throw std::runtime_error("mask reuse unexpectedly shared an attention computation");
                    std::vector<ggml_tensor *> first_masks, second_masks;
                    attention_masks(fa, first_masks); attention_masks(second, second_masks);
                    if (first_masks.size() != (size_t) nt || second_masks.size() != (size_t) nt)
                        throw std::runtime_error("exact attention mask row count changed");
                    for (int t = 0; t < nt; ++t) {
                        const auto * row = first_masks[t];
                        if ((row == second_masks[t]) != mask_reuse)
                            throw std::runtime_error("attention mask sharing disagrees with its opt-in environment");
                        if (row->type != GGML_TYPE_F16 || row->src[0] != mask || row->view_src != mask ||
                            row->view_offs != (size_t) t * mask->nb[1] || row->ne[0] != nk ||
                            row->ne[1] != 1 || row->ne[2] != 1 || row->ne[3] != 1)
                            throw std::runtime_error("attention mask reuse changed its view geometry");
                    }
                }
                if (nt == 1) ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
                auto * out = ggml_reshape_4d(ctx, fa, d, hq, nt, 1);
                std::vector<float> queries(d * nt * hq);
                for (size_t i = 0; i < queries.size(); ++i) queries[i] = std::cos(float(i) * 0.031f) * 0.5f;
                std::vector<float> mf(nk * mask_rows, -INFINITY);
                for (int t = 0; t < nt; ++t) std::fill(mf.begin() + t * nk, mf.begin() + t * nk + prefix + t + 1, 0.0f);
                std::vector<ggml_fp16_t> masks(mf.size());
                ggml_fp32_to_fp16_row(mf.data(), masks.data(), masks.size());
                auto actual = run(gpu, ctx, out, q, queries, mask, masks);
                for (int t = 0; t < nt; ++t) {
                    auto seq_ctx = context();
                    auto * sq = ggml_new_tensor_4d(seq_ctx, GGML_TYPE_F32, d, 1, hq, 1);
                    auto * sm = ggml_new_tensor_4d(seq_ctx, GGML_TYPE_F16, nk, mask_rows, 1, 1);
                    auto * sy = ggml_flash_attn_ext(seq_ctx, sq, k, v, sm, scale, 0.0f, 0.0f);
                    ggml_flash_attn_ext_set_prec(sy, GGML_PREC_F32);
                    std::vector<float> query(d * hq);
                    for (int h = 0; h < hq; ++h)
                        std::copy_n(queries.data() + d * (t + nt * h), d, query.data() + d * h);
                    std::vector<ggml_fp16_t> row_mask(masks.size(), ggml_fp32_to_fp16(-INFINITY));
                    std::copy_n(masks.data() + t * nk, nk, row_mask.data());
                    auto reference = run(gpu, seq_ctx, sy, sq, query, sm, row_mask);
                    for (size_t i = 0; i < reference.size(); ++i)
                        if (!std::isfinite(actual[t * reference.size() + i]) || actual[t * reference.size() + i] != reference[i]) {
                            std::fprintf(stderr, "attention nk=%d nt=%d prefix=%d token=%d index=%zu expected=%.9g actual=%.9g\n",
                                nk, nt, prefix, t, i, reference[i], actual[t * reference.size() + i]);
                            throw std::runtime_error("attention differs from independent sequential GPU oracle");
                        }
                    ggml_free(seq_ctx);
                }
                std::printf("native attention head=256 GQA=8 nk=%d nt=%d prefix=%d mask_reuse=%d max_abs=0 PASS\n", nk, nt, prefix, (int) mask_reuse);
                ggml_free(ctx);
            }
            ggml_backend_buffer_free(cache_buffer); ggml_free(cache_ctx);
        }
        const char * q8_verify_env = std::getenv("STRATA_A3B_Q8_KV_VERIFY_VEC");
        if (q8_verify_env && std::strcmp(q8_verify_env, "1") == 0) {
            for (int nk : {512, 4096}) {
                auto cache_ctx = context();
                auto * k = ggml_new_tensor_4d(cache_ctx, GGML_TYPE_Q8_0, d, nk, hk, 1);
                auto * v = ggml_new_tensor_4d(cache_ctx, GGML_TYPE_Q8_0, d, nk, hk, 1);
                auto cache_buffer = ggml_backend_alloc_ctx_tensors(cache_ctx, gpu);
                std::vector<float> floats(d * nk * hk);
                std::vector<uint8_t> packed(ggml_nbytes(k));
                for (size_t i = 0; i < floats.size(); ++i) floats[i] = std::sin(float(i) * 0.013f) * 0.4f;
                if (ggml_quantize_chunk(GGML_TYPE_Q8_0, floats.data(), packed.data(), 0, nk * hk, d, nullptr) != packed.size())
                    throw std::runtime_error("Q8 K fixture packing size mismatch");
                ggml_backend_tensor_set(k, packed.data(), 0, packed.size());
                for (size_t i = 0; i < floats.size(); ++i) floats[i] = std::cos(float(i) * 0.017f) * 0.3f;
                if (ggml_quantize_chunk(GGML_TYPE_Q8_0, floats.data(), packed.data(), 0, nk * hk, d, nullptr) != packed.size())
                    throw std::runtime_error("Q8 V fixture packing size mismatch");
                ggml_backend_tensor_set(v, packed.data(), 0, packed.size());
                for (int nt : {2, 3, 4}) for (int prefix : {8, nk - nt - 1}) {
                    auto ctx = context();
                    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, nt, hq, 1);
                    auto * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, nk, mask_rows, 1, 1);
                    ggml_tensor * probe = nullptr;
                    auto * fa = hook(ctx, q, k, v, mask, nullptr, scale, 0.0f, 0.0f, 0, &probe);
                    if (!fa || !probe || probe->src[0]->ne[1] != 1 || probe->src[1] != k || probe->src[2] != v)
                        throw std::runtime_error("Q8 verify failed to select direct singleton FlashAttention");
                    auto * out = ggml_reshape_4d(ctx, fa, d, hq, nt, 1);
                    std::vector<float> queries(d * nt * hq);
                    for (size_t i = 0; i < queries.size(); ++i) queries[i] = std::cos(float(i) * 0.031f) * 0.5f;
                    std::vector<float> mf(nk * mask_rows, -INFINITY);
                    for (int t = 0; t < nt; ++t) std::fill(mf.begin() + t * nk, mf.begin() + t * nk + prefix + t + 1, 0.0f);
                    std::vector<ggml_fp16_t> masks(mf.size());
                    ggml_fp32_to_fp16_row(mf.data(), masks.data(), masks.size());
                    auto actual = run(gpu, ctx, out, q, queries, mask, masks);
                    for (int t = 0; t < nt; ++t) {
                        auto seq_ctx = context();
                        auto * sq = ggml_new_tensor_4d(seq_ctx, GGML_TYPE_F32, d, 1, hq, 1);
                        auto * sm = ggml_new_tensor_4d(seq_ctx, GGML_TYPE_F16, nk, mask_rows, 1, 1);
                        auto * sy = ggml_flash_attn_ext(seq_ctx, sq, k, v, sm, scale, 0.0f, 0.0f);
                        ggml_flash_attn_ext_set_prec(sy, GGML_PREC_F32);
                        std::vector<float> query(d * hq);
                        for (int h = 0; h < hq; ++h)
                            std::copy_n(queries.data() + d * (t + nt * h), d, query.data() + d * h);
                        std::vector<ggml_fp16_t> row_mask(masks.size(), ggml_fp32_to_fp16(-INFINITY));
                        std::copy_n(masks.data() + t * nk, nk, row_mask.data());
                        auto reference = run(gpu, seq_ctx, sy, sq, query, sm, row_mask);
                        for (size_t i = 0; i < reference.size(); ++i)
                            if (!std::isfinite(actual[t * reference.size() + i]) || actual[t * reference.size() + i] != reference[i]) {
                                std::fprintf(stderr, "Q8 attention nk=%d nt=%d prefix=%d token=%d index=%zu expected=%.9g actual=%.9g\n",
                                    nk, nt, prefix, t, i, reference[i], actual[t * reference.size() + i]);
                                throw std::runtime_error("Q8 verify differs from independent singleton GPU oracle");
                            }
                        ggml_free(seq_ctx);
                    }
                    std::printf("Q8 singleton verify head=256 GQA=8 nk=%d nt=%d prefix=%d max_abs=0 PASS\n", nk, nt, prefix);
                    ggml_free(ctx);
                }
                ggml_backend_buffer_free(cache_buffer); ggml_free(cache_ctx);
            }
        }
        if (!strata_a3b::stats().native_attention_graphs) throw std::runtime_error("attention graph hook never accepted");
        ggml_backend_free(gpu); return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "native attention parity: %s\n", error.what()); return 1;
    }
}
