#include "shared_weights.hpp"
#include "llama-model.h"
#include "gguf.h"
extern "C" {
#include "hash/sha256/sha256.h"
}

#include <array>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace strata_a3b {
namespace {
struct File {
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata{nullptr, gguf_free};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors{nullptr, ggml_free};
    std::ifstream stream;
    explicit File(const std::string & path) : stream(std::filesystem::u8path(path), std::ios::binary) {
        ggml_context * ctx = nullptr;
        metadata.reset(gguf_init_from_file(path.c_str(), {true, &ctx}));
        tensors.reset(ctx);
        if (!metadata || !stream) throw std::runtime_error("cannot inspect shared-weight GGUF: " + path);
        const int64_t split = gguf_find_key(metadata.get(), "split.count");
        if (split >= 0 && gguf_get_val_u16(metadata.get(), split) > 1)
            throw std::runtime_error("shared MTP weights require single-file GGUFs");
    }
    std::string digest(const char * name, const ggml_tensor * loaded) {
        auto * header = ggml_get_tensor(tensors.get(), name);
        const int64_t index = gguf_find_tensor(metadata.get(), name);
        if (!header || index < 0 || !loaded || header->type != loaded->type)
            throw std::runtime_error(std::string("shared-weight file/model type mismatch: ") + name);
        for (int i = 0; i < GGML_MAX_DIMS; ++i)
            if (header->ne[i] != loaded->ne[i])
                throw std::runtime_error(std::string("shared-weight file/model shape mismatch: ") + name);
        size_t remaining = ggml_nbytes(header);
        if (remaining != ggml_nbytes(loaded))
            throw std::runtime_error(std::string("shared-weight file/model byte mismatch: ") + name);
        const size_t offset = gguf_get_data_offset(metadata.get()) + gguf_get_tensor_offset(metadata.get(), index);
        stream.clear();
        stream.seekg((std::streamoff) offset);
        sha256_t hash;
        sha256_init(&hash);
        std::vector<unsigned char> bytes(4 * 1024 * 1024);
        while (remaining) {
            const size_t count = std::min(remaining, bytes.size());
            stream.read(reinterpret_cast<char *>(bytes.data()), (std::streamsize) count);
            if (!stream) throw std::runtime_error(std::string("short shared-weight tensor read: ") + name);
            sha256_update(&hash, bytes.data(), count);
            remaining -= count;
        }
        std::array<unsigned char, SHA256_DIGEST_SIZE> digest;
        sha256_final(&hash, digest.data());
        std::ostringstream text;
        text << std::hex << std::setfill('0');
        for (const auto byte : digest) text << std::setw(2) << (int) byte;
        return text.str();
    }
};

std::string buffer_name(const ggml_tensor * tensor) {
    if (!tensor || !tensor->buffer) return "none";
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(tensor->buffer));
}
} // namespace

SharedWeights share_mtp_weights(llama_model * target, llama_model * draft,
                               const std::string & target_file, const std::string & draft_file) {
    if (!target || !draft || target == draft || target->arch != LLM_ARCH_QWEN35MOE ||
        draft->arch != LLM_ARCH_QWEN35MOE || draft->hparams.n_layer_nextn != 1)
        throw std::runtime_error("shared weights require separate matching Qwen MoE target/MTP models");
    const auto & nextn = draft->layers.at(draft->hparams.n_layer()).nextn;
    if (nextn.embed_tokens || nextn.shared_head_head)
        throw std::runtime_error("shared MTP weights require the draft block to use its model token/output tables");
    if (target->output_s || draft->output_s)
        throw std::runtime_error("shared MTP output tables with separate scale tensors are not supported");
    std::array<const char *, 3> names = {"token_embd.weight", "output.weight", "output_norm.weight"};
    std::array<ggml_tensor *, 3> target_tensors = {target->tok_embd, target->output, target->output_norm};
    std::array<ggml_tensor **, 3> draft_tensors = {&draft->tok_embd, &draft->output, &draft->output_norm};
    File target_gguf(target_file), draft_gguf(draft_file);
    SharedWeights result;
    for (size_t i = 0; i < names.size(); ++i) {
        auto * t = target_tensors[i];
        auto * d = *draft_tensors[i];
        if (!t || !d || !t->buffer || !d->buffer || t->type != d->type)
            throw std::runtime_error(std::string("shared-weight loaded type/buffer mismatch: ") + names[i]);
        if (!ggml_backend_buffer_is_host(d->buffer))
            throw std::runtime_error(std::string("load draft shared tensor into CPU memory before sharing: ") + names[i]);
        for (int j = 0; j < GGML_MAX_DIMS; ++j)
            if (t->ne[j] != d->ne[j] || t->nb[j] != d->nb[j])
                throw std::runtime_error(std::string("shared-weight loaded shape/stride mismatch: ") + names[i]);
        const auto target_hash = target_gguf.digest(names[i], t);
        const auto draft_hash = draft_gguf.digest(names[i], d);
        if (target_hash != draft_hash)
            throw std::runtime_error(std::string("shared-weight SHA-256 mismatch; keeping separate weights: ") + names[i]);
        SharedTensor entry;
        entry.name = names[i]; entry.type = ggml_type_name(t->type); entry.sha256 = target_hash;
        entry.bytes = ggml_nbytes(t); entry.target_on_host = ggml_backend_buffer_is_host(t->buffer);
        entry.target_buffer = buffer_name(t); entry.previous_draft_buffer = buffer_name(d);
        if (entry.target_on_host) result.reused_host_bytes += entry.bytes;
        else result.reused_gpu_bytes += entry.bytes;
        result.retained_draft_host_bytes += ggml_nbytes(d);
        result.tensors.push_back(std::move(entry));
    }
    // Validate every table before changing any graph-facing pointer. Tensor
    // ownership stays with the original model contexts and is not transferred.
    for (size_t i = 0; i < names.size(); ++i) *draft_tensors[i] = target_tensors[i];
    return result;
}

} // namespace strata_a3b
