#include "cache_profile.hpp"
#include "ggml.h"
#include "gguf.h"
extern "C" {
#include "hash/sha256/sha256.h"
}
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#endif

namespace strata_a3b {
namespace {
constexpr size_t row_count = expert_profile_layers * expert_profile_experts;
bool hex_digest(const std::string & value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}
uint64_t integer(const std::string & value) {
    if (value.empty() || !std::all_of(value.begin(), value.end(), [](char c) { return c >= '0' && c <= '9'; }))
        throw std::runtime_error("cache profile contains an invalid nonnegative integer");
    try { return std::stoull(value); }
    catch (...) { throw std::runtime_error("cache profile integer overflow"); }
}
std::string field(std::istream & input, const char * key) {
    std::string actual, value;
    if (!(input >> actual >> value) || actual != key)
        throw std::runtime_error(std::string("cache profile expected field: ") + key);
    return value;
}
void fixed(std::istream & input, const char * key, uint64_t expected) {
    if (integer(field(input, key)) != expected)
        throw std::runtime_error(std::string("cache profile geometry mismatch: ") + key);
}
bool earlier(const ExpertScore & a, const ExpertScore & b) {
    if (a.score != b.score) return a.score > b.score;
    if (a.layer != b.layer) return a.layer < b.layer;
    return a.expert < b.expert;
}
void validate(const ExpertCacheProfile & profile) {
    if (!hex_digest(profile.model_sha256) || !hex_digest(profile.source_fingerprint) || !profile.model_bytes ||
        profile.bundle_bytes != expert_profile_bundle_bytes || profile.scores.size() != row_count)
        throw std::runtime_error("cache profile identity, bundle size or row count invalid");
    std::array<bool, row_count> seen{};
    uint64_t total = 0;
    for (size_t i = 0; i < profile.scores.size(); ++i) {
        const auto & s = profile.scores[i];
        if (s.score > std::numeric_limits<uint64_t>::max() - total)
            throw std::runtime_error("cache profile total score overflow");
        total += s.score;
        if (s.layer < 0 || s.layer >= expert_profile_layers || s.expert < 0 || s.expert >= expert_profile_experts)
            throw std::runtime_error("cache profile expert ID out of range");
        const auto index = (size_t) s.layer * expert_profile_experts + s.expert;
        if (seen[index]) throw std::runtime_error("cache profile contains duplicate layer/expert rows");
        seen[index] = true;
        if (i && earlier(s, profile.scores[i - 1]))
            throw std::runtime_error("cache profile rows are not in canonical global score order");
    }
}
std::string digest_string(const std::array<unsigned char, SHA256_DIGEST_SIZE> & digest) {
    std::ostringstream text; text << std::hex << std::setfill('0');
    for (auto byte : digest) text << std::setw(2) << (int) byte;
    return text.str();
}
std::string digest_text(sha256_t & hash) {
    std::array<unsigned char, SHA256_DIGEST_SIZE> digest;
    sha256_final(&hash, digest.data());
    return digest_string(digest);
}
#ifdef _WIN32
void check_hash_status(NTSTATUS status, const char * operation) {
    if (status < 0) {
        std::ostringstream message;
        message << "cache-profile SHA-256 " << operation << " failed: NTSTATUS 0x"
                << std::hex << static_cast<unsigned long>(status);
        throw std::runtime_error(message.str());
    }
}
struct WindowsHash {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    std::vector<unsigned char> object;
    WindowsHash() = default;
    WindowsHash(const WindowsHash &) = delete;
    WindowsHash & operator=(const WindowsHash &) = delete;
    ~WindowsHash() {
        if (hash) BCryptDestroyHash(hash);
        if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
    }
    void init() {
        check_hash_status(BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0), "open");
        DWORD object_bytes = 0, digest_bytes = 0, written = 0;
        check_hash_status(BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
            reinterpret_cast<PUCHAR>(&object_bytes), sizeof(object_bytes), &written, 0), "object length");
        if (written != sizeof(object_bytes) || !object_bytes)
            throw std::runtime_error("cache-profile SHA-256 invalid BCrypt object length");
        check_hash_status(BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH,
            reinterpret_cast<PUCHAR>(&digest_bytes), sizeof(digest_bytes), &written, 0), "digest length");
        if (written != sizeof(digest_bytes) || digest_bytes != SHA256_DIGEST_SIZE)
            throw std::runtime_error("cache-profile SHA-256 invalid BCrypt digest length");
        object.resize(object_bytes);
        check_hash_status(BCryptCreateHash(algorithm, &hash, object.data(), object_bytes, nullptr, 0, 0), "create");
    }
};
#endif
std::string file_hash(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open file for cache-profile SHA-256: " + path.u8string());
#ifdef _WIN32
    WindowsHash hash; hash.init();
#else
    sha256_t hash; sha256_init(&hash);
#endif
    std::vector<unsigned char> buffer(16 * 1024 * 1024);
    while (input) {
        input.read(reinterpret_cast<char *>(buffer.data()), (std::streamsize) buffer.size());
        const auto count = input.gcount();
        if (count) {
#ifdef _WIN32
            check_hash_status(BCryptHashData(hash.hash, buffer.data(), static_cast<ULONG>(count), 0), "update");
#else
            sha256_update(&hash, buffer.data(), (size_t) count);
#endif
        }
    }
    if (!input.eof()) throw std::runtime_error("cache-profile SHA-256 read failed");
#ifdef _WIN32
    std::array<unsigned char, SHA256_DIGEST_SIZE> digest;
    check_hash_status(BCryptFinishHash(hash.hash, digest.data(), static_cast<ULONG>(digest.size()), 0), "finish");
    return digest_string(digest);
#else
    return digest_text(hash);
#endif
}
void validate_model(const std::string & path, uint64_t bytes) {
    ggml_context * raw = nullptr;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> uf(gguf_init_from_file(path.c_str(), {true, &raw}), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(raw, ggml_free);
    if (!uf || !tensors) throw std::runtime_error("cannot inspect cache-profile model GGUF");
    const auto architecture = gguf_find_key(uf.get(), "general.architecture");
    if (architecture < 0 || gguf_get_kv_type(uf.get(), architecture) != GGUF_TYPE_STRING ||
        std::string(gguf_get_val_str(uf.get(), architecture)) != "qwen35moe")
        throw std::runtime_error("cache profile requires qwen35moe");
    const auto split = gguf_find_key(uf.get(), "split.count");
    if (split >= 0 && (gguf_get_kv_type(uf.get(), split) != GGUF_TYPE_UINT16 || gguf_get_val_u16(uf.get(), split) != 1))
        throw std::runtime_error("cache profile requires one complete GGUF file");
    const std::pair<const char *, uint32_t> geometry[] = {
        {"block_count", 40}, {"embedding_length", 2048}, {"expert_count", 256}, {"expert_used_count", 8},
        {"expert_feed_forward_length", 512}, {"attention.head_count", 16}, {"attention.head_count_kv", 2},
        {"ssm.state_size", 128}, {"ssm.group_count", 16}, {"ssm.time_step_rank", 32},
        {"ssm.conv_kernel", 4}, {"full_attention_interval", 4},
    };
    for (const auto & item : geometry) {
        const auto key = std::string("qwen35moe.") + item.first;
        const auto index = gguf_find_key(uf.get(), key.c_str());
        if (index < 0 || gguf_get_kv_type(uf.get(), index) != GGUF_TYPE_UINT32 || gguf_get_val_u32(uf.get(), index) != item.second)
            throw std::runtime_error("cache-profile model geometry mismatch: " + key);
    }
    const std::regex routed("^blk\\.([0-9]+)\\.ffn_(gate|up|down)_exps\\.weight$");
    std::array<std::array<bool, 3>, expert_profile_layers> seen{};
    std::array<uint64_t, expert_profile_layers> bundle{};
    size_t experts = 0;
    for (auto * tensor = ggml_get_first_tensor(tensors.get()); tensor; tensor = ggml_get_next_tensor(tensors.get(), tensor)) {
        const auto index = gguf_find_tensor(uf.get(), tensor->name);
        const uint64_t base = gguf_get_data_offset(uf.get());
        const uint64_t offset = gguf_get_tensor_offset(uf.get(), index);
        if (base > bytes || offset > bytes - base || ggml_nbytes(tensor) > bytes - base - offset)
            throw std::runtime_error("cache-profile model tensor payload is truncated");
        std::smatch match; const std::string name = tensor->name;
        if (!std::regex_match(name, match, routed)) {
            if (name.find("_exps.weight") != std::string::npos)
                throw std::runtime_error("cache profile does not support merged or unknown expert tensors");
            continue;
        }
        const int layer = std::stoi(match[1]);
        const std::string kind = match[2];
        const int projection = kind == "gate" ? 0 : kind == "up" ? 1 : 2;
        const int64_t columns = projection == 2 ? 512 : 2048;
        const int64_t rows = projection == 2 ? 2048 : 512;
        if (layer < 0 || layer >= expert_profile_layers || seen[layer][projection] || tensor->type != GGML_TYPE_Q8_0 ||
            tensor->ne[0] != columns || tensor->ne[1] != rows || tensor->ne[2] != 256 || tensor->ne[3] != 1 ||
            tensor->nb[2] != expert_profile_bundle_bytes / 3)
            throw std::runtime_error("cache profile requires 120 canonical Q8_0 routed tensors");
        seen[layer][projection] = true; ++experts;
        bundle[layer] += tensor->nb[2];
    }
    if (experts != 120 || !std::all_of(bundle.begin(), bundle.end(), [](uint64_t n) { return n == expert_profile_bundle_bytes; }))
        throw std::runtime_error("cache-profile Q8_0 expert bundle mismatch");
}
}

ExpertCacheProfile read_expert_cache_profile(std::istream & input) {
    std::string magic;
    if (!(input >> magic) || magic != "STRATA_A3B_EXPERT_PROFILE_V1")
        throw std::runtime_error("unsupported expert cache profile format");
    ExpertCacheProfile profile;
    profile.model_sha256 = field(input, "model_sha256");
    profile.model_bytes = integer(field(input, "model_bytes"));
    fixed(input, "layers", 40); fixed(input, "experts", 256);
    fixed(input, "embedding", 2048); fixed(input, "ffn", 512); fixed(input, "experts_used", 8);
    profile.bundle_bytes = integer(field(input, "bundle_bytes"));
    profile.source_fingerprint = field(input, "source_fingerprint");
    fixed(input, "scores", row_count);
    profile.scores.reserve(row_count);
    for (size_t i = 0; i < row_count; ++i) {
        std::string l, e, s;
        if (!(input >> l >> e >> s)) throw std::runtime_error("truncated cache profile scores");
        const auto layer = integer(l), expert = integer(e), score = integer(s);
        if (layer >= expert_profile_layers || expert >= expert_profile_experts)
            throw std::runtime_error("cache profile expert ID out of range");
        profile.scores.push_back({(int) layer, (int) expert, score});
    }
    std::string extra;
    if (input >> extra) throw std::runtime_error("unexpected trailing cache profile data");
    validate(profile);
    return profile;
}

ExpertCachePlan plan_expert_cache(const ExpertCacheProfile & profile, uint64_t budget_bytes) {
    validate(profile);
    ExpertCachePlan plan;
    plan.budget_bytes = budget_bytes; plan.bundle_bytes = profile.bundle_bytes;
    plan.model_sha256 = profile.model_sha256; plan.source_fingerprint = profile.source_fingerprint;
    const auto capacity = budget_bytes / profile.bundle_bytes;
    for (const auto & score : profile.scores) {
        if (score.score > std::numeric_limits<uint64_t>::max() - plan.total_score)
            throw std::runtime_error("cache profile total score overflow");
        plan.total_score += score.score;
        if (score.score && plan.selected_experts < capacity) {
            plan.expert_ids[score.layer].push_back(score.expert);
            plan.covered_score += score.score; ++plan.selected_experts;
        }
    }
    plan.planned_bytes = plan.selected_experts * profile.bundle_bytes;
    plan.coverage = plan.total_score ? double(plan.covered_score) / double(plan.total_score) : 0;
    return plan;
}

ExpertCachePlan load_expert_cache_profile(const std::string & profile_path, const std::string & model_path,
                                         uint64_t budget_bytes) {
    const auto profile_file = std::filesystem::u8path(profile_path);
    if (std::filesystem::file_size(profile_file) > 1024 * 1024)
        throw std::runtime_error("expert cache profile exceeds 1 MiB format limit");
    std::ifstream input(profile_file, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open expert cache profile");
    std::ostringstream contents; contents << input.rdbuf();
    if (input.bad()) throw std::runtime_error("cannot read expert cache profile");
    const auto text = contents.str();
    std::istringstream parsed(text);
    const auto profile = read_expert_cache_profile(parsed);
    const auto model = std::filesystem::u8path(model_path);
    if (std::filesystem::file_size(model) != profile.model_bytes)
        throw std::runtime_error("cache profile model byte size mismatch");
    const auto began = std::chrono::steady_clock::now();
    const auto sha = file_hash(model);
    const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count();
    if (sha != profile.model_sha256) throw std::runtime_error("cache profile model SHA-256 mismatch");
    validate_model(model_path, profile.model_bytes);
    auto plan = plan_expert_cache(profile, budget_bytes);
    plan.model_hash_ms = elapsed;
    sha256_t profile_hash; sha256_init(&profile_hash);
    sha256_update(&profile_hash, reinterpret_cast<const unsigned char *>(text.data()), text.size());
    plan.profile_sha256 = digest_text(profile_hash);
    return plan;
}
}
