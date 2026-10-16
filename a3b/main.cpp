#include "backend.hpp"
#include "speculative.hpp"
#include "shared_weights.hpp"
#include "cache_profile.hpp"
#include "kv_tier.hpp"
#include "profile_ranges.hpp"
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr size_t MiB = 1024 * 1024;
void log_message(enum ggml_log_level level, const char * message, void *) {
    if (level >= GGML_LOG_LEVEL_INFO) std::fputs(message, stderr);
}
struct GdnDebugDump {
    std::FILE * file = nullptr;
    bool saw_layer_zero = false;
    bool finished = false;
    ~GdnDebugDump() { if (file) std::fclose(file); }
};
bool gdn_debug_eval(struct ggml_tensor * tensor, bool ask, void * user_data) {
    auto & dump = *static_cast<GdnDebugDump *>(user_data);
    if (dump.finished || std::strncmp(tensor->name, "final_output-", 13) != 0) return false;
    const int layer = std::atoi(tensor->name + 13);
    if (layer == 0 && dump.saw_layer_zero) {
        dump.finished = true;
        return false;
    }
    if (ask) return true;
    if (layer == 0) dump.saw_layer_zero = true;
    const uint32_t bytes = static_cast<uint32_t>(ggml_nbytes(tensor));
    std::vector<unsigned char> data(bytes);
    ggml_backend_tensor_get(tensor, data.data(), 0, bytes);
    std::fwrite(&layer, sizeof(layer), 1, dump.file);
    std::fwrite(&bytes, sizeof(bytes), 1, dump.file);
    std::fwrite(data.data(), 1, bytes, dump.file);
    return true;
}
const char * expert_pattern = "blk\\.[0-9]+\\.ffn_(gate|up|down|gate_up)_exps\\.weight";
struct Options {
    std::string model, prompt = "Explain mixture of experts briefly.", prompt_file, logits, mtp_model, route_dump, expert_profile;
    int ctx = 4096, batch = 128, predict = 128, device = 0;
    int threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    int cache_mib = -1, vram_mib = 0;
    int cache_reserve_mib = 640;
    int mtp_draft = 3, mtp_ubatch = 512;
    bool cpu = false, raw = false, stream = true, repl = false, serve = false;
    bool mmap = false, pageable_experts = false;
    bool stable_graphs = false, preserve_verify_layout = false, cache_graph_routes = false;
    bool exact_verify = false;
    bool native_gpu = false, native_dense_exact = false, native_attention_exact = false;
    bool gpu_only_experts = false;
    bool hybrid_experts = false, cpu_expert_pool = false;
    bool hybrid_doorbell = false;
    bool prefill_cpu_share = false;
    double pcie_fraction = 0.0;
    bool pcie_explicit = false;
    bool async_expert_events = false;
    bool mtp_adaptive = false;
    bool mtp_fused_rounds = false;
    bool gpu_static_experts = false;
    bool gpu_dynamic_experts = false;
    int expert_staging_slots = 8;
    int expert_ring_banks = 2;
    bool prefill_copy_ahead = false;
    bool prefill_independent_issuer = false;
    bool prefill_packed_gate_up = false;
    int cache_decay_interval = 0;
    bool share_mtp_weights = false;
    bool gpu_sampling = false, deferred_catchup = false, gpu_embedding = false;
    bool kv_q8 = false, kv_q4 = false;
    bool kv_tiered = false, kv_tiered_f16_short = false, kv_tier_low_vram = false;
    bool fused_moe = false;
    bool strata_native_experts = false;
    bool strata_prefill_mmq = false;
};
struct Sampling {
    float temperature = 0, top_p = 1, min_p = 0, repeat = 1, frequency = 0, presence = 0;
    int top_k = 20, last_n = 64;
    uint32_t seed = LLAMA_DEFAULT_SEED;
};
struct StaticExpertScope {
    bool enabled = false;
    void close() {
        if (enabled) { strata_a3b::release_gpu_static_experts(); enabled = false; }
    }
    ~StaticExpertScope() {
        try { close(); }
        catch (const std::exception & e) { std::fprintf(stderr, "static expert cleanup: %s\n", e.what()); }
    }
};
void usage() {
    std::puts("strata-moe --model MODEL.gguf [--prompt TEXT | --prompt-file FILE]\n"
              "  --ctx 4096 --batch 128 --predict 128 --threads N --device 0\n"
              "  --vram-mib 0 (all available VRAM) --cache-mib N (auto after context allocation)\n"
              "  --native-gpu (all experts directly on GPU; requires a fitting model)\n"
              "  --gpu-only-experts (RAM expert cache; all routed matrix computation on GPU)\n"
              "  --expert-staging-slots 8 (bounded GPU slots for nonresident experts)\n"
              "  --expert-ring-banks 2 (copy-ahead ring depth, independent of 8-expert compute groups)\n"
              "  --prefill-copy-ahead (1024+ token chunks: host worker stages pageable experts)\n"
              "  --prefill-independent-issuer (experimental separate expert H2D issuer)\n"
              "  --prefill-packed-gate-up (Q8 long prefill: one MMQ for gate/up; requires independent issuer)\n"
              "  --gpu-static-experts (experimental CUDA graph routing and UVA miss fetch; fixed profile)\n"
              "  --gpu-dynamic-experts (experimental GPU-managed expert cache; implies static path)\n"
              "  --dump-routing FILE (expert route counts after each request)\n"
              "  --expert-profile FILE (verified Q8 global hot-expert plan; startup warmup)\n"
              "  --async-expert-events (GPU-only event pipeline; opt-in)\n"
              "  --cache-reserve-mib 640 (workspace outside the expert cache)\n"
              "  --cache-decay-interval N (GPU-only LFU decay every N windows; 0 keeps existing policy)\n"
              "  --native-dense-exact (single-column dense arithmetic for short GPU verification)\n"
              "  --native-attention-exact (single-query GPU arithmetic for short verification)\n"
              "  --share-mtp-weights (verify identical tensors before sharing the embedding/head)\n"
              "  --gpu-sampling --defer-mtp-catchup --gpu-embedding (opt-in resident GPU tuning)\n"
              "  --fused-moe (whole CPU/GPU routed expert execution)\n"
              "  --strata-native-experts (opt-in original Strata grouped GPU IQ experts for short windows)\n"
              "  --strata-prefill-mmq (opt-in original Strata IQ MMQ for 9..2048-token GPU-resident A3B prompts)\n"
              "  --hybrid-experts (opt-in concurrent CPU/GPU experts, profile and exact GPU trunk)\n"
              "  --hybrid-doorbell (experimental original-style CUDA graph/CPU mailbox; requires profile, PCIe 0, decay 0)\n"
              "  --prefill-cpu-share (opt-in: CPU computes rare missed quantized experts for 9..256-token prompt chunks)\n"
              "  --cpu-expert-pool (original-style shared-activation Q8 CPU pool; requires hybrid)\n"
              "  --pcie-frac F (hybrid: fraction of missed experts sent to GPU, 0..1)\n"
              "  --cpu (CPU reference) --raw (preformatted prompt) --no-stream-prefill\n"
              "  --logits FILE (first-token float32 logits) --repl (one raw prompt per line)\n"
              "  --serve (resident token protocol for the existing Python web/API server)\n"
              "  --mtp-model MTP.gguf --mtp-draft 3 (optional target-verified NextN acceleration)\n"
              "  --mtp-ubatch 512 (independent draft microbatch; main prefill may be larger)\n"
              "  --kv-q8 (opt-in Q8_0 K/V cache for long contexts; measure quality and speed)\n"
              "  --kv-q4 (opt-in Q4_0 K/V cache for long contexts; measure quality and speed)\n"
              "  --kv-tiered (serve: resize target/MTP contexts in 64K tiers, max --ctx)\n"
              "  --kv-tier-low-vram (2G: 1K/GPU KV through 8K; 6G+: 64K/GPU KV, then host RAM)\n"
              "  --kv-tiered-f16-short (with --kv-tiered: F16 K/V through 131K, Q8_0 above 131K)\n"
              "  --mtp-adaptive (choose 0..mtp-draft from local acceptance and round cost)\n"
              "  --mtp-fused-rounds (opt-in accepted catch-up + first proposal; GPU sampling/defer/exact trunk)\n"
              "  --mmap (opt in to mapped weights) --pageable-experts (ordinary host RAM)\n"
              "  --stable-expert-graphs --preserve-verify-layout --cache-graph-routes (opt-in tuning)\n"
              "  --exact-verify (short expert windows use single-column GPU arithmetic)\n"
              "Qwen3.5/3.6-35B-A3B text inference. Sampling: greedy.");
}
Options parse(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto value = [&]() -> std::string { if (++i >= argc) throw std::runtime_error("missing value: " + key); return argv[i]; };
        if (key == "--help" || key == "-h") { usage(); std::exit(0); }
        else if (key == "--model" || key == "-m") o.model = value();
        else if (key == "--mtp-model") o.mtp_model = value();
        else if (key == "--mtp-draft") o.mtp_draft = std::stoi(value());
        else if (key == "--mtp-ubatch") o.mtp_ubatch = std::stoi(value());
        else if (key == "--mtp-adaptive") o.mtp_adaptive = true;
        else if (key == "--mtp-fused-rounds") o.mtp_fused_rounds = true;
        else if (key == "--prompt") o.prompt = value();
        else if (key == "--prompt-file") o.prompt_file = value();
        else if (key == "--logits") o.logits = value();
        else if (key == "--ctx" || key == "--max-context") o.ctx = std::stoi(value());
        else if (key == "--batch") o.batch = std::stoi(value());
        else if (key == "--predict" || key == "-n") o.predict = std::stoi(value());
        else if (key == "--threads") o.threads = std::stoi(value());
        else if (key == "--device") o.device = std::stoi(value());
        else if (key == "--cache-mib") o.cache_mib = std::stoi(value());
        else if (key == "--cache-reserve-mib") o.cache_reserve_mib = std::stoi(value());
        else if (key == "--cache-decay-interval") o.cache_decay_interval = std::stoi(value());
        else if (key == "--vram-mib") o.vram_mib = std::stoi(value());
        else if (key == "--native-gpu") o.native_gpu = true;
        else if (key == "--gpu-only-experts") o.gpu_only_experts = true;
        else if (key == "--gpu-static-experts") o.gpu_static_experts = true;
        else if (key == "--gpu-dynamic-experts") o.gpu_dynamic_experts = o.gpu_static_experts = true;
        else if (key == "--async-expert-events") o.async_expert_events = true;
        else if (key == "--expert-staging-slots") o.expert_staging_slots = std::stoi(value());
        else if (key == "--expert-ring-banks") o.expert_ring_banks = std::stoi(value());
        else if (key == "--prefill-copy-ahead") o.prefill_copy_ahead = true;
        else if (key == "--prefill-independent-issuer") o.prefill_independent_issuer = true;
        else if (key == "--prefill-packed-gate-up") o.prefill_packed_gate_up = true;
        else if (key == "--dump-routing") o.route_dump = value();
        else if (key == "--expert-profile") o.expert_profile = value();
        else if (key == "--native-dense-exact") o.native_dense_exact = true;
        else if (key == "--native-attention-exact") o.native_attention_exact = true;
        else if (key == "--share-mtp-weights") o.share_mtp_weights = true;
        else if (key == "--gpu-sampling") o.gpu_sampling = true;
        else if (key == "--defer-mtp-catchup") o.deferred_catchup = true;
        else if (key == "--gpu-embedding") o.gpu_embedding = true;
        else if (key == "--kv-q8") o.kv_q8 = true;
        else if (key == "--kv-q4") o.kv_q4 = true;
        else if (key == "--kv-tiered") o.kv_tiered = true;
        else if (key == "--kv-tiered-f16-short") o.kv_tiered_f16_short = true;
        else if (key == "--kv-tier-low-vram") o.kv_tier_low_vram = true;
        else if (key == "--fused-moe") o.fused_moe = true;
        else if (key == "--strata-native-experts") o.strata_native_experts = o.fused_moe = true;
        else if (key == "--strata-prefill-mmq") o.strata_prefill_mmq = o.fused_moe = true;
        else if (key == "--hybrid-experts") o.hybrid_experts = true;
        else if (key == "--hybrid-doorbell") o.hybrid_doorbell = o.hybrid_experts = o.cpu_expert_pool = true;
        else if (key == "--cpu-expert-pool") o.cpu_expert_pool = true;
        else if (key == "--prefill-cpu-share") o.prefill_cpu_share = true;
        else if (key == "--pcie-frac") { o.pcie_fraction = std::stod(value()); o.pcie_explicit = true; }
        else if (key == "--cpu") o.cpu = true;
        else if (key == "--mmap") o.mmap = true;
        else if (key == "--pageable-experts") o.pageable_experts = true;
        else if (key == "--stable-expert-graphs") o.stable_graphs = true;
        else if (key == "--preserve-verify-layout") o.preserve_verify_layout = true;
        else if (key == "--cache-graph-routes") o.cache_graph_routes = true;
        else if (key == "--exact-verify") o.exact_verify = true;
        else if (key == "--raw") o.raw = true;
        else if (key == "--no-stream-prefill") o.stream = false;
        else if (key == "--repl") { o.repl = true; o.raw = true; }
        else if (key == "--serve") { o.serve = true; o.raw = true; }
        else throw std::runtime_error("unknown argument: " + key);
    }
    if (o.kv_q8 && o.kv_q4) throw std::runtime_error("--kv-q8 and --kv-q4 are mutually exclusive");
    if (!strata_a3b::kv_tier_f16_short_options_valid(o.kv_tiered_f16_short, o.kv_tiered, o.kv_q8, o.kv_q4))
        throw std::runtime_error("--kv-tiered-f16-short requires --kv-tiered and excludes --kv-q8/--kv-q4");
    if (o.kv_tiered && (!o.serve || o.ctx <= 65536))
        throw std::runtime_error("--kv-tiered requires --serve and --ctx > 65536");
    if (o.kv_tier_low_vram && (!o.kv_tiered || o.cpu))
        throw std::runtime_error("--kv-tier-low-vram requires GPU and --kv-tiered");
    if (o.model.empty() || o.ctx < 32 || o.ctx > 262144 || o.batch < 1 || o.predict < 0 || o.predict > o.ctx || o.threads < 1 ||
        o.device < 0 || o.cache_mib < -1 || o.vram_mib < 0 || o.mtp_draft < 1 || o.mtp_draft > 8 || o.mtp_ubatch < 1 ||
        o.expert_staging_slots < 1 || o.expert_staging_slots > 64 || o.expert_ring_banks < 2 || o.expert_ring_banks > 8 ||
        o.cache_reserve_mib < 64 || o.cache_reserve_mib > 4096 ||
        o.cache_decay_interval < 0 || o.cache_decay_interval > 1024 ||
        (!o.mtp_model.empty() && o.batch <= o.mtp_draft))
        throw std::runtime_error("invalid arguments (see --help)");
    // The trained draft head is opt-in. Its target verification defaults to
    // decode's single-column expert arithmetic, validated against accepted
    // sequential prefixes, rather than the faster multi-column exploration.
    if (!o.mtp_model.empty()) o.exact_verify = true;
    if ((o.cpu && (o.native_gpu || o.gpu_only_experts || o.hybrid_experts)) ||
        (o.hybrid_experts && (o.native_gpu || o.gpu_only_experts)) || (o.native_gpu && o.gpu_only_experts) ||
        ((o.native_dense_exact || o.native_attention_exact) && !(o.native_gpu || o.gpu_only_experts || o.hybrid_experts)))
        throw std::runtime_error("select one GPU expert mode; exact native options require --native-gpu or --gpu-only-experts");
    if (o.strata_native_experts && (!o.native_gpu || o.device != 0))
        throw std::runtime_error("--strata-native-experts currently requires --native-gpu on device 0");
    if (o.strata_prefill_mmq && (!o.native_gpu || o.device != 0))
        throw std::runtime_error("--strata-prefill-mmq currently requires --native-gpu on device 0");
    if (o.gpu_only_experts || o.hybrid_experts) {
        o.fused_moe = true;
        o.gpu_embedding = true;
    }
    if (o.cpu_expert_pool && !o.hybrid_experts)
        throw std::runtime_error("--cpu-expert-pool requires --hybrid-experts");
    if (o.prefill_copy_ahead && !(o.gpu_only_experts || o.hybrid_experts))
        throw std::runtime_error("--prefill-copy-ahead requires managed GPU experts");
    if (o.prefill_independent_issuer && (!o.prefill_copy_ahead || !o.pageable_experts))
        throw std::runtime_error("--prefill-independent-issuer requires --prefill-copy-ahead and --pageable-experts");
    if (o.prefill_packed_gate_up && !o.prefill_independent_issuer)
        throw std::runtime_error("--prefill-packed-gate-up requires --prefill-independent-issuer");
    if (o.prefill_cpu_share && (!o.hybrid_experts || !o.cpu_expert_pool))
        throw std::runtime_error("--prefill-cpu-share requires hybrid experts and the native CPU pool");
    if (!std::isfinite(o.pcie_fraction) || o.pcie_fraction < 0 || o.pcie_fraction > 1 ||
        (o.pcie_explicit && !o.hybrid_experts))
        throw std::runtime_error("--pcie-frac requires --hybrid-experts and a finite value in 0..1");
    if (o.async_expert_events && !(o.gpu_only_experts || o.hybrid_experts))
        throw std::runtime_error("--async-expert-events requires --gpu-only-experts");
    if (o.cache_decay_interval && !(o.gpu_only_experts || o.hybrid_experts))
        throw std::runtime_error("--cache-decay-interval requires --gpu-only-experts");
    if (!o.expert_profile.empty() && !(o.gpu_only_experts || o.hybrid_experts))
        throw std::runtime_error("--expert-profile requires --gpu-only-experts");
    if (o.mtp_adaptive && o.mtp_model.empty())
        throw std::runtime_error("--mtp-adaptive requires --mtp-model");
    if (o.mtp_fused_rounds && (o.mtp_model.empty() || !o.gpu_sampling || !o.deferred_catchup ||
        !o.native_dense_exact || !o.native_attention_exact || o.cpu || o.device != 0 || o.mtp_draft > 7 ||
        !(o.native_gpu || o.gpu_only_experts || o.hybrid_experts)))
        throw std::runtime_error("--mtp-fused-rounds requires MTP, GPU 0 sampling, deferred catch-up, native dense/attention exact and at most seven drafts");
    if (o.gpu_static_experts && (!o.gpu_only_experts || o.expert_profile.empty() || o.cache_decay_interval))
        throw std::runtime_error("--gpu-static-experts requires GPU-only, a verified expert profile and decay interval 0");
    if (o.hybrid_doorbell && (o.gpu_static_experts || o.expert_profile.empty() || o.cache_decay_interval ||
                              o.pcie_fraction != 0 || o.device != 0 || o.mtp_draft > 7))
        throw std::runtime_error("--hybrid-doorbell requires device 0, a verified profile, PCIe fraction 0, decay 0 and draft count <=7");
    if (!o.prompt_file.empty()) {
        std::ifstream f(o.prompt_file, std::ios::binary);
        if (!f) throw std::runtime_error("cannot read prompt file");
        o.prompt.assign(std::istreambuf_iterator<char>(f), {});
    }
    return o;
}
size_t inspect(const std::string & path) {
    ggml_context * ctx = nullptr;
    gguf_context * uf = gguf_init_from_file(path.c_str(), {true, &ctx});
    if (!uf) throw std::runtime_error("cannot read GGUF");
    std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(uf, gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(ctx, ggml_free);
    auto integer = [&](const char * key, uint32_t expected) {
        const auto i = gguf_find_key(uf, key);
        if (i < 0 || gguf_get_kv_type(uf, i) != GGUF_TYPE_UINT32 || gguf_get_val_u32(uf, i) != expected)
            throw std::runtime_error(std::string("unsupported A3B geometry: ") + key);
    };
    const auto a = gguf_find_key(uf, "general.architecture");
    if (a < 0 || gguf_get_kv_type(uf, a) != GGUF_TYPE_STRING || std::string(gguf_get_val_str(uf, a)) != "qwen35moe")
        throw std::runtime_error("requires a qwen35moe GGUF (Qwen3.5/3.6-35B-A3B)");
    const auto split = gguf_find_key(uf, "split.count");
    if (split >= 0 && gguf_get_val_u16(uf, split) > 1)
        throw std::runtime_error("use a single merged GGUF for the A3B path");
    integer("qwen35moe.block_count", 40);
    integer("qwen35moe.embedding_length", 2048);
    integer("qwen35moe.expert_count", 256);
    integer("qwen35moe.expert_used_count", 8);
    integer("qwen35moe.attention.head_count", 16);
    integer("qwen35moe.attention.head_count_kv", 2);
    integer("qwen35moe.expert_feed_forward_length", 512);
    integer("qwen35moe.ssm.state_size", 128);
    integer("qwen35moe.ssm.group_count", 16);
    integer("qwen35moe.ssm.time_step_rank", 32);
    integer("qwen35moe.ssm.conv_kernel", 4);
    integer("qwen35moe.full_attention_interval", 4);
    size_t bytes = 0; int count = 0;
    const std::regex pattern(expert_pattern);
    for (auto * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (!std::regex_match(t->name, pattern)) continue;
        if (t->ne[2] != 256 || t->ne[3] != 1) throw std::runtime_error("invalid routed tensor shape");
        bytes += ggml_nbytes(t); ++count;
    }
    if (count < 80 || !bytes) throw std::runtime_error("incomplete routed expert tensors");
    std::fprintf(stderr, "A3B: %d routed tensors, %.1f MiB expert weights, original GGUF quantization retained\n", count, bytes / double(MiB));
    return bytes;
}
std::string format_prompt(llama_model * model, const Options & o, const std::string & text) {
    if (o.raw) return text;
    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) throw std::runtime_error("no chat template; supply --raw");
    llama_chat_message msg{"user", text.c_str()};
    int n = llama_chat_apply_template(tmpl, &msg, 1, true, nullptr, 0);
    if (n < 0) throw std::runtime_error("unsupported chat template; supply --raw");
    std::string prompt((size_t) n, '\0');
    if (llama_chat_apply_template(tmpl, &msg, 1, true, prompt.data(), n) != n)
        throw std::runtime_error("chat template failed");
    return prompt;
}
void generate(llama_model * model, llama_context * ctx, const Options & o, const std::string & text,
              const std::vector<llama_token> * provided = nullptr, const Sampling & sampling = {},
              strata_a3b::MtpDraft * mtp = nullptr) {
    strata_a3b::profile::Range request_range("a3b/request");
    if (strata_a3b::profile::requested()) std::fprintf(stderr, "A3B NVTX ranges: %s\n",
        strata_a3b::profile::available() ? "enabled (diagnostic only)" : "unavailable in this build");
    // The opt-in prefill bridge may still have a D2D last-row copy queued on
    // the target stream from the previous request. Finish it before clearing
    // that request's recurrent/KV state; the default path is unchanged.
    if (mtp && mtp->device_prefill_enabled()) llama_synchronize(ctx);
    llama_memory_clear(llama_get_memory(ctx), true);
    if (mtp) mtp->reset();
    const auto * vocab = llama_model_get_vocab(model);
    const auto prompt = provided ? std::string() : format_prompt(model, o, text);
    int n = provided ? (int) provided->size() : -llama_tokenize(vocab, prompt.data(), (int) prompt.size(), nullptr, 0, true, true);
    if (n <= 0 || n + o.predict > o.ctx) throw std::runtime_error("prompt + output exceeds --ctx");
    std::vector<llama_token> tokens = provided ? *provided : std::vector<llama_token>((size_t) n);
    if (!provided && llama_tokenize(vocab, prompt.data(), (int) prompt.size(), tokens.data(), n, true, true) != n)
        throw std::runtime_error("tokenization failed");
    for (auto id : tokens) if (id < 0 || id >= llama_vocab_n_tokens(vocab)) throw std::runtime_error("invalid input token id");
    const bool device_sampling = o.gpu_sampling && !o.cpu && o.logits.empty() && sampling.temperature == 0 &&
                                 sampling.repeat == 1 && sampling.frequency == 0 && sampling.presence == 0;
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
        device_sampling ? strata_a3b::create_device_greedy_sampler() : llama_sampler_chain_init(llama_sampler_chain_default_params()),
        llama_sampler_free);
    if (!device_sampling) {
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), sampling.last_n, sampling.repeat, sampling.frequency, sampling.presence));
        if (sampling.temperature > 0) {
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(sampling.top_k));
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(sampling.top_p, 1));
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_min_p(sampling.min_p, 1));
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(sampling.temperature));
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(sampling.seed));
        } else llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
    }
    for (auto id : tokens) llama_sampler_accept(sampler.get(), id);
    struct SamplerBinding {
        llama_context * context; bool active;
        ~SamplerBinding() { if (active) llama_set_sampler(context, 0, nullptr); }
    } binding{ctx, device_sampling};
    if (device_sampling && !llama_set_sampler(ctx, 0, sampler.get())) throw std::runtime_error("target GPU sampler unavailable");
    if (o.gpu_static_experts || o.hybrid_doorbell) strata_a3b::collect_gpu_static_stats();
    const auto cache_begin = strata_a3b::stats();
    const int64_t start = ggml_time_us();
    struct MtpPrefillHostCopyScope {
        strata_a3b::MtpDraft * draft;
        MtpPrefillHostCopyScope(strata_a3b::MtpDraft * d, int count) : draft(d) {
            if (draft) draft->arm_prefill_host_copy(count);
        }
        ~MtpPrefillHostCopyScope() { if (draft) draft->disarm_prefill_host_copy(); }
    };
    {
        strata_a3b::profile::Range prefill_range("a3b/prefill");
        for (int pos = 0; pos < n; pos += o.batch) {
            const int count = std::min(o.batch, n - pos);
            auto batch = llama_batch_get_one(tokens.data() + pos, count);
            {
                strata_a3b::profile::Range target_range("a3b/prefill/target-decode");
                MtpPrefillHostCopyScope host_copy(mtp, count);
                if (llama_decode(ctx, batch)) throw std::runtime_error("prompt decode failed");
            }
            if (mtp) {
                strata_a3b::profile::Range mtp_range("a3b/prefill/mtp-observe");
                mtp->observe(tokens.data() + pos, count, pos);
            }
            if (o.serve) { std::printf("PP %d %d %.1f\n", std::min(pos + o.batch, n), n, (ggml_time_us() - start) / 1000.0); std::fflush(stdout); }
        }
    }
    const int64_t prefill_end = ggml_time_us();
    if (o.gpu_static_experts || o.hybrid_doorbell) strata_a3b::collect_gpu_static_stats();
    const auto cache_prefill = strata_a3b::stats();
    if (!o.logits.empty()) {
        std::ofstream f(o.logits, std::ios::binary);
        f.write((const char *) llama_get_logits_ith(ctx, -1), llama_vocab_n_tokens(vocab) * sizeof(float));
        if (!f) throw std::runtime_error("cannot write logits");
    }
    int generated = 0;
    const char * finish = "length";
    const bool timing_phases = std::getenv("STRATA_A3B_TIMING") && std::getenv("STRATA_A3B_TIMING")[0] == '1';
    int64_t proposal_us = 0, verify_us = 0, observe_us = 0, sample_emit_us = 0, commit_us = 0;
    int verify_windows = 0;
    auto sample = [&](int row) {
        if (device_sampling) {
            strata_a3b::device_greedy_token(ctx, row);
            return llama_sampler_sample(sampler.get(), ctx, row);
        }
        const auto * logits = llama_get_logits_ith(ctx, row);
        if (!logits) throw std::runtime_error("missing target logits");
        for (int i = 0; i < llama_vocab_n_tokens(vocab); ++i)
            if (!std::isfinite(logits[i])) throw std::runtime_error("non-finite model logits");
        return llama_sampler_sample(sampler.get(), ctx, row);
    };
    auto emit = [&](llama_token token) {
        if (llama_vocab_is_eog(vocab, token)) {
            if (o.serve) { std::printf("T %d\n", token); std::fflush(stdout); }
            finish = "stop"; return false;
        }
        if (o.serve) { std::printf("T %d\n", token); std::fflush(stdout); }
        else {
        std::vector<char> piece(256);
        int len = llama_token_to_piece(vocab, token, piece.data(), (int) piece.size(), 0, false);
        if (len < 0) { piece.resize((size_t) -len); len = llama_token_to_piece(vocab, token, piece.data(), (int) piece.size(), 0, false); }
        if (len < 0) throw std::runtime_error("token decoding failed");
        std::fwrite(piece.data(), 1, len, stdout); std::fflush(stdout);
        }
        return true;
    };
    {
    strata_a3b::profile::Range output_range("a3b/output");
    if (mtp && o.predict > 0) {
        // Prefill is identical to the non-speculative path. Its final logits
        // provide the first real token; subsequent windows verify that token
        // plus proposals, and only emit samples from the target distribution.
        llama_token last;
        bool running;
        {
            strata_a3b::profile::Range sample_range("a3b/output/initial-sample-emit");
            last = sample(-1);
            running = emit(last);
        }
        generated = running || o.serve ? 1 : 0;
        llama_pos pos = n;
        while (running && generated < o.predict) {
            const int64_t phase_start = timing_phases ? ggml_time_us() : 0;
            const auto proposals = [&] {
                strata_a3b::profile::Range propose_range("a3b/output/mtp-propose");
                return mtp->propose(last, pos, o.predict - generated - 1);
            }();
            if (timing_phases) proposal_us += ggml_time_us() - phase_start;
            std::vector<llama_token> window{last};
            window.insert(window.end(), proposals.begin(), proposals.end());
            auto batch = llama_batch_init((int) window.size(), 0, 1);
            batch.n_tokens = (int) window.size();
            for (int i = 0; i < batch.n_tokens; ++i) {
                batch.token[i] = window[i]; batch.pos[i] = pos + i;
                batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = true;
            }
            const int64_t verify_start = timing_phases ? ggml_time_us() : 0;
            int rc;
            {
                strata_a3b::profile::Range verify_range("a3b/output/target-verify");
                rc = llama_decode(ctx, batch);
            }
            if (timing_phases) { verify_us += ggml_time_us() - verify_start; ++verify_windows; }
            llama_batch_free(batch);
            if (rc) throw std::runtime_error("MTP target verification failed");
            const int64_t observe_start = timing_phases ? ggml_time_us() : 0;
            {
                strata_a3b::profile::Range observe_range("a3b/output/target-hidden-observe");
                mtp->observe(window.data(), (int) window.size(), pos, true);
            }
            if (timing_phases) observe_us += ggml_time_us() - observe_start;
            int consumed = 0, matched = 0;
            const int64_t sample_start = timing_phases ? ggml_time_us() : 0;
            {
                strata_a3b::profile::Range sample_range("a3b/output/target-sample-emit");
                for (int i = 0; i < (int) window.size(); ++i) {
                    last = sample(i);
                    if (i < (int) proposals.size() && last == proposals[i]) ++matched;
                    running = emit(last);
                    if (running || o.serve) ++generated;
                    ++consumed;
                    if (!running || generated >= o.predict || i == (int) proposals.size() || last != proposals[i]) break;
                }
            }
            if (timing_phases) sample_emit_us += ggml_time_us() - sample_start;
            const int64_t commit_start = timing_phases ? ggml_time_us() : 0;
            {
                strata_a3b::profile::Range commit_range("a3b/output/mtp-commit");
                mtp->commit(consumed, matched);
            }
            if (timing_phases) commit_us += ggml_time_us() - commit_start;
            pos += consumed;
        }
    } else for (; generated < o.predict; ++generated) {
        llama_token token;
        {
            strata_a3b::profile::Range sample_range("a3b/output/target-sample-emit");
            token = sample(-1);
            if (!emit(token)) { if (o.serve) ++generated; break; }
        }
        if (generated + 1 < o.predict) {
            strata_a3b::profile::Range decode_range("a3b/output/target-decode");
            if (llama_decode(ctx, llama_batch_get_one(&token, 1)))
                throw std::runtime_error("generation failed");
        }
    }
    if (o.mtp_fused_rounds && mtp) {
        strata_a3b::profile::Range finish_range("a3b/output/mtp-finish");
        mtp->finish();
    }
    }
    if (!o.serve) std::puts("");
    const double pp = (prefill_end - start) / 1e6, tg = (ggml_time_us() - prefill_end) / 1e6;
    if (o.gpu_static_experts || o.hybrid_doorbell) strata_a3b::collect_gpu_static_stats();
    auto s = strata_a3b::stats();
    if (o.gpu_static_experts || o.hybrid_doorbell) {
        std::fprintf(stderr, "A3B static: calls %llu; mapped host %.1f MiB; shared GPU scratch %.1f MiB; prefill route reads %llu decode route reads %llu\n",
            (unsigned long long) (s.gpu_static_calls - cache_begin.gpu_static_calls),
            s.gpu_static_mapped_host_bytes / double(MiB), s.gpu_static_shared_gpu_scratch_bytes / double(MiB),
            (unsigned long long) (cache_prefill.route_reads - cache_begin.route_reads),
            (unsigned long long) (s.route_reads - cache_prefill.route_reads));
    }
    if (o.serve) {
        const auto accepted = mtp ? mtp->stats().accepted : 0;
        const auto offered = mtp ? mtp->stats().offered : 0;
        std::printf("DONE %d %d %.1f %.1f %s %llu %llu 0\n", generated, n, pp * 1000, tg * 1000, finish,
                    (unsigned long long) accepted, (unsigned long long) offered);
        std::fflush(stdout);
    }
    std::fprintf(stderr, "A3B: prompt %d tokens %.3fs %.2f tok/s; output %d tokens %.3fs %.2f tok/s\n",
                 n, pp, n / pp, generated, tg, generated / std::max(tg, 1e-9));
    if (timing_phases && mtp) std::fprintf(stderr,
        "A3B decode phases: windows %d; propose %.3f ms verify %.3f ms observe %.3f ms sample/emit %.3f ms commit %.3f ms\n",
        verify_windows, proposal_us / 1000.0, verify_us / 1000.0, observe_us / 1000.0,
        sample_emit_us / 1000.0, commit_us / 1000.0);
    std::fprintf(stderr, "A3B cache: %.1f MiB; GPU rows %llu CPU rows %llu; hit %llu miss %llu; uploaded %.1f MiB\n",
                 s.cache_bytes / double(MiB), (unsigned long long) s.gpu_rows, (unsigned long long) s.cpu_rows,
                 (unsigned long long) s.hits, (unsigned long long) s.misses, s.upload_bytes / double(MiB));
    std::fprintf(stderr, "A3B backend: route reads %llu; native layout calls %llu; single-column calls %llu\n",
                 (unsigned long long) s.route_reads, (unsigned long long) s.preserved_layout_calls,
                 (unsigned long long) s.exact_verify_calls);
    if (o.strata_native_experts) std::fprintf(stderr,
        "A3B Strata grouped CUDA expert calls: %llu; request prefill %llu decode %llu\n",
        (unsigned long long) s.strata_native_calls,
        (unsigned long long) (cache_prefill.strata_native_calls - cache_begin.strata_native_calls),
        (unsigned long long) (s.strata_native_calls - cache_prefill.strata_native_calls));
    if (const char * fused_gdn = std::getenv("STRATA_A3B_GDN_FUSED_STEP"); fused_gdn && fused_gdn[0] == '1')
        std::fprintf(stderr,
            "A3B Strata fused GDN graph matches: lifetime %llu; request prefill builds %llu decode builds %llu (not CUDA graph replays)\n",
            (unsigned long long) s.gdn_fused_graph_nodes,
            (unsigned long long) (cache_prefill.gdn_fused_graph_nodes - cache_begin.gdn_fused_graph_nodes),
            (unsigned long long) (s.gdn_fused_graph_nodes - cache_prefill.gdn_fused_graph_nodes));
    if (o.strata_prefill_mmq) std::fprintf(stderr,
        "A3B Strata original prefill MMQ: CUDA calls %llu rows %llu; request prefill calls %llu rows %llu; "
        "decode calls %llu; graph builds %llu fallback graph builds %llu\n",
        (unsigned long long) s.strata_prefill_mmq_calls,
        (unsigned long long) s.strata_prefill_mmq_rows,
        (unsigned long long) (cache_prefill.strata_prefill_mmq_calls - cache_begin.strata_prefill_mmq_calls),
        (unsigned long long) (cache_prefill.strata_prefill_mmq_rows - cache_begin.strata_prefill_mmq_rows),
        (unsigned long long) (s.strata_prefill_mmq_calls - cache_prefill.strata_prefill_mmq_calls),
        (unsigned long long) s.strata_prefill_mmq_graphs,
        (unsigned long long) s.strata_prefill_mmq_fallback_graphs);
    std::fprintf(stderr, "A3B transfer: prefill %.1f MiB; decode %.1f MiB; decode GPU rows %llu CPU rows %llu; decode hit %llu miss %llu\n",
                 (cache_prefill.upload_bytes - cache_begin.upload_bytes) / double(MiB),
                 (s.upload_bytes - cache_prefill.upload_bytes) / double(MiB),
                 (unsigned long long) (s.gpu_rows - cache_prefill.gpu_rows),
                 (unsigned long long) (s.cpu_rows - cache_prefill.cpu_rows),
                 (unsigned long long) (s.hits - cache_prefill.hits),
                 (unsigned long long) (s.misses - cache_prefill.misses));
    std::fprintf(stderr, "A3B prefill host profile: route read %.3f ms; GPU-only scheduling %.3f ms (bank reuse wait %.3f ms, task submit %.3f ms); staging batches %llu\n",
                 cache_prefill.route_read_ms - cache_begin.route_read_ms,
                 cache_prefill.gpu_only_host_ms - cache_begin.gpu_only_host_ms,
                 cache_prefill.staging_reuse_wait_ms - cache_begin.staging_reuse_wait_ms,
                 cache_prefill.expert_task_submit_ms - cache_begin.expert_task_submit_ms,
                 (unsigned long long) (cache_prefill.staging_batches - cache_begin.staging_batches));
    std::fprintf(stderr, "A3B native graph builds: dense %llu; experts %llu; whole FFN %llu; attention %llu\n",
                 (unsigned long long) s.native_dense_graphs, (unsigned long long) s.native_expert_graphs,
                 (unsigned long long) s.native_fused_graphs, (unsigned long long) s.native_attention_graphs);
    std::fprintf(stderr, "A3B cache decay: prefill %llu; decode %llu\n",
                 (unsigned long long) (cache_prefill.decays - cache_begin.decays),
                 (unsigned long long) (s.decays - cache_prefill.decays));
    if (o.prefill_cpu_share) std::fprintf(stderr,
        "A3B prefill CPU share: calls %llu; CPU jobs %llu experts %llu\n",
        (unsigned long long) (cache_prefill.prefill_share_calls - cache_begin.prefill_share_calls),
        (unsigned long long) (cache_prefill.prefill_share_cpu_jobs - cache_begin.prefill_share_cpu_jobs),
        (unsigned long long) (cache_prefill.prefill_share_cpu_experts - cache_begin.prefill_share_cpu_experts));
    if (o.hybrid_experts) {
        if (o.hybrid_doorbell) std::fprintf(stderr,
            "A3B doorbell: calls %llu CPU requests %llu callback %.3f ms; mailbox to host %llu bytes to GPU %llu bytes\n",
            (unsigned long long) (s.hybrid_doorbell_calls - cache_begin.hybrid_doorbell_calls),
            (unsigned long long) (s.hybrid_doorbell_requests - cache_begin.hybrid_doorbell_requests),
            s.hybrid_doorbell_callback_ms - cache_begin.hybrid_doorbell_callback_ms,
            (unsigned long long) (s.hybrid_mailbox_to_host_bytes - cache_begin.hybrid_mailbox_to_host_bytes),
            (unsigned long long) (s.hybrid_mailbox_to_device_bytes - cache_begin.hybrid_mailbox_to_device_bytes));
        std::fprintf(stderr, "A3B hybrid: calls %llu; PCIe fraction %.4f; GPU miss rows %llu experts %llu; input allocations %llu bytes %llu fallbacks %llu\n",
            (unsigned long long) (s.hybrid_calls - cache_begin.hybrid_calls), o.pcie_fraction,
            (unsigned long long) (s.hybrid_gpu_miss_rows - cache_begin.hybrid_gpu_miss_rows),
            (unsigned long long) (s.hybrid_gpu_miss_experts - cache_begin.hybrid_gpu_miss_experts),
            (unsigned long long) s.hybrid_input_allocations, (unsigned long long) s.hybrid_input_bytes,
            (unsigned long long) s.hybrid_input_fallbacks);
        if (o.cpu_expert_pool) std::fprintf(stderr,
            "A3B CPU pool: calls %llu jobs %llu unique inputs %llu experts %llu; input quant %.3f ms gate/up %.3f ms hidden quant %.3f ms down %.3f ms\n",
            (unsigned long long) (s.cpu_pool_calls - cache_prefill.cpu_pool_calls),
            (unsigned long long) (s.cpu_pool_jobs - cache_prefill.cpu_pool_jobs),
            (unsigned long long) (s.cpu_pool_unique_inputs - cache_prefill.cpu_pool_unique_inputs),
            (unsigned long long) (s.cpu_pool_unique_experts - cache_prefill.cpu_pool_unique_experts),
            s.cpu_pool_input_quant_ms - cache_prefill.cpu_pool_input_quant_ms,
            s.cpu_pool_gate_up_ms - cache_prefill.cpu_pool_gate_up_ms,
            s.cpu_pool_hidden_quant_ms - cache_prefill.cpu_pool_hidden_quant_ms,
            s.cpu_pool_down_ms - cache_prefill.cpu_pool_down_ms);
    }
    if (mtp) {
        const auto & m = mtp->stats();
        std::fprintf(stderr, "A3B MTP: windows %llu offered %llu accepted %llu; draft %.1f ms catchup %.1f ms\n",
                     (unsigned long long) m.windows, (unsigned long long) m.offered,
                     (unsigned long long) m.accepted, m.draft_ms, m.catchup_ms);
        std::fprintf(stderr, "A3B MTP sampling: GPU samples %llu; token/guard download %llu bytes; hidden %.1f ms\n",
                     (unsigned long long) m.gpu_draft_samples,
                     (unsigned long long) m.draft_sampling_download_bytes, m.hidden_ms);
        std::fprintf(stderr, "A3B MTP compact GPU sampler: graph sampled rows %llu\n",
                     (unsigned long long) mtp->compact_gpu_sample_rows());
        std::fprintf(stderr, "A3B MTP device prefill: calls %llu rows %llu; avoided hidden D2H %llu bytes\n",
                     (unsigned long long) m.device_prefill_calls,
                     (unsigned long long) m.device_prefill_rows,
                     (unsigned long long) m.device_prefill_avoided_d2h_bytes);
        if (mtp->device_chain_k2_enabled()) std::fprintf(stderr,
            "A3B MTP device K2 chain: calls %llu fallback %llu; token/hidden D2D %llu bytes; inter-step context syncs %llu; CUDA graph replays not instrumented\n",
            (unsigned long long) m.device_chain_k2_calls,
            (unsigned long long) m.device_chain_k2_fallbacks,
            (unsigned long long) m.device_chain_k2_d2d_bytes,
            (unsigned long long) m.device_chain_k2_inter_step_syncs);
        if (o.mtp_fused_rounds) std::fprintf(stderr,
            "A3B MTP fused rounds: calls %llu; accepted catch-up rows %llu; fused %.3f ms (excluded from draft/catchup)\n",
            (unsigned long long) m.fused_rounds, (unsigned long long) m.fused_catchup_rows, m.fused_round_ms);
        if (o.mtp_adaptive) {
            std::fprintf(stderr, "A3B MTP adaptive:");
            for (int k = 0; k <= o.mtp_draft; ++k)
                std::fprintf(stderr, " K%d=%llu/%.1fms", k,
                    (unsigned long long) m.adaptive_windows_by_k[k], m.adaptive_round_ms_by_k[k]);
            std::fputc('\n', stderr);
        }
    }
    if (o.gpu_only_experts) std::fprintf(stderr,
        "A3B GPU-only: rows %llu; staged rows %llu experts %llu batches %llu; staging GPU %.1f MiB host %.1f MiB allocations %llu\n",
        (unsigned long long) s.gpu_only_rows, (unsigned long long) s.staged_rows,
        (unsigned long long) s.staged_experts, (unsigned long long) s.staging_batches,
        s.staging_gpu_bytes / double(MiB), s.staging_host_bytes / double(MiB),
        (unsigned long long) s.staging_allocations);
    if (!o.route_dump.empty()) strata_a3b::dump_route_frequencies(o.route_dump.c_str());
}
using ContextPtr = std::unique_ptr<llama_context, decltype(&llama_free)>;

void apply_short_f16_kv_tier(const Options & options, uint32_t capacity, llama_context_params & cp) {
    if (!options.kv_tiered_f16_short) return;
    const auto format = strata_a3b::kv_tier_f16_short_format(capacity);
    cp.type_k = cp.type_v = format == strata_a3b::KvTierFormat::F16 ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
}

// A3B currently uses llama's fixed-size GGML KV tensors. Rebuilding an idle context
// changes their size without moving model weights or changing the advertised limit.
// Every GEN clears its target and draft state, so no conversation state is lost here.
void set_kv_tier(llama_model * model, llama_model * draft_model, const Options & options,
                 llama_context_params base_params, uint32_t needed, ContextPtr & ctx,
                 ContextPtr & draft_ctx, std::unique_ptr<strata_a3b::MtpDraft> & mtp) {
    const bool tiny_vram = options.kv_tier_low_vram && options.vram_mib > 0 && options.vram_mib <= 2048;
    const uint32_t initial = tiny_vram ? 1024 : 65536;
    const uint32_t target = strata_a3b::kv_tier_capacity((uint32_t) options.ctx, needed, initial);
    if (!target) throw std::runtime_error("request exceeds --ctx");
    if (ctx && llama_n_ctx(ctx.get()) == target) return;
    const uint32_t previous = ctx ? llama_n_ctx(ctx.get()) :
        strata_a3b::kv_tier_capacity((uint32_t) options.ctx, 0, initial);
    if (draft_ctx) llama_synchronize(draft_ctx.get());
    if (ctx) llama_synchronize(ctx.get());
    mtp.reset();
    draft_ctx.reset();
    ctx.reset();

    auto rebuild = [&](uint32_t capacity) {
        auto cp = base_params;
        cp.n_ctx = capacity;
        if (options.kv_tier_low_vram) cp.offload_kqv = capacity <= (tiny_vram ? 8192 : 65536);
        apply_short_f16_kv_tier(options, capacity, cp);
        std::fprintf(stderr, "A3B tiered KV: rebuilding %u cells with K=%s V=%s\n",
                     capacity, ggml_type_name(cp.type_k), ggml_type_name(cp.type_v));
        ContextPtr next_ctx(llama_init_from_model(model, cp), llama_free);
        if (!next_ctx) throw std::runtime_error("target context allocation failed at KV tier " + std::to_string(capacity));
        ContextPtr next_draft(nullptr, llama_free);
        std::unique_ptr<strata_a3b::MtpDraft> next_mtp;
        if (draft_model) {
            auto dc = cp;
            dc.n_rs_seq = 0;
            dc.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            dc.ctx_other = next_ctx.get();
            dc.n_ubatch = std::min(cp.n_ubatch, (uint32_t) options.mtp_ubatch);
            dc.n_outputs_max = dc.n_outputs_max_per_seq = 1;
            next_draft.reset(llama_init_from_model(draft_model, dc));
            if (!next_draft) throw std::runtime_error("MTP context allocation failed at KV tier " + std::to_string(capacity));
            next_mtp = std::make_unique<strata_a3b::MtpDraft>(next_ctx.get(), next_draft.get(), options.mtp_draft,
                0.0f, options.gpu_sampling && !options.cpu, options.deferred_catchup,
                options.mtp_adaptive, options.mtp_fused_rounds);
        }
        ctx = std::move(next_ctx);
        draft_ctx = std::move(next_draft);
        mtp = std::move(next_mtp);
    };

    try {
        rebuild(target);
    } catch (...) {
        const auto failure = std::current_exception();
        try {
            rebuild(previous);
            auto restored = base_params;
            apply_short_f16_kv_tier(options, previous, restored);
            std::fprintf(stderr, "A3B tiered KV: restored %u cells with K=%s V=%s after failed resize\n",
                         previous, ggml_type_name(restored.type_k), ggml_type_name(restored.type_v));
        }
        catch (const std::exception & e) {
            auto restored = base_params;
            apply_short_f16_kv_tier(options, previous, restored);
            std::fprintf(stderr, "A3B tiered KV: failed to restore %u cells with K=%s V=%s: %s\n",
                         previous, ggml_type_name(restored.type_k), ggml_type_name(restored.type_v), e.what());
        }
        std::rethrow_exception(failure);
    }
    auto allocated = base_params;
    apply_short_f16_kv_tier(options, target, allocated);
    std::fprintf(stderr, "A3B tiered KV: allocated %u cells with K=%s V=%s (limit %d; previous %u)\n",
                 target, ggml_type_name(allocated.type_k), ggml_type_name(allocated.type_v), options.ctx, previous);
}

void serve(llama_model * model, ContextPtr & ctx, const Options & options, llama_model * draft_model,
           ContextPtr & draft_ctx, std::unique_ptr<strata_a3b::MtpDraft> & mtp, llama_context_params base_params) {
    std::printf("INFO engine=strata-moe-a3b context=%d\nREADY %d\n", options.ctx, options.ctx); std::fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "QUIT") break;
        if (line == "STOP") continue;
        try {
            Options o = options;
            Sampling s;
            std::istringstream in(line); std::string command, field, csv;
            if (!(in >> command >> o.predict) || command != "GEN" || o.predict < 0 || o.predict > o.ctx)
                throw std::runtime_error("expected GEN <max_new> [sampling keys] <ids>");
            while (in >> field) {
                const auto eq = field.find('=');
                if (eq == std::string::npos) { if (!csv.empty()) throw std::runtime_error("extra token list"); csv = field; continue; }
                const auto k = field.substr(0, eq), v = field.substr(eq + 1);
                if (k == "temperature") s.temperature = std::stof(v);
                else if (k == "top_p") s.top_p = std::stof(v);
                else if (k == "top_k") s.top_k = std::stoi(v);
                else if (k == "min_p") s.min_p = std::stof(v);
                else if (k == "seed") s.seed = (uint32_t) std::stoul(v);
                else if (k == "penalty_repeat") s.repeat = std::stof(v);
                else if (k == "penalty_freq") s.frequency = std::stof(v);
                else if (k == "penalty_present") s.presence = std::stof(v);
                else if (k == "penalty_last_n") s.last_n = std::stoi(v);
                else if (k != "ckpt") throw std::runtime_error("unsupported A3B sampling key: " + k);
            }
            if (!std::isfinite(s.temperature) || s.temperature < 0 || !std::isfinite(s.top_p) || s.top_p <= 0 || s.top_p > 1 ||
                !std::isfinite(s.min_p) || s.min_p < 0 || s.min_p > 1 || !std::isfinite(s.repeat) || s.repeat <= 0 ||
                !std::isfinite(s.frequency) || !std::isfinite(s.presence) || s.top_k < 0 || s.last_n < 0)
                throw std::runtime_error("invalid sampling parameters");
            std::vector<llama_token> tokens; std::istringstream ids(csv);
            while (std::getline(ids, field, ',')) {
                size_t used = 0; int id = std::stoi(field, &used);
                if (used != field.size()) throw std::runtime_error("invalid token list");
                tokens.push_back(id);
            }
            if (tokens.empty() || tokens.size() >= (size_t) o.ctx) throw std::runtime_error("prompt does not fit context");
            if (!o.predict) o.predict = o.ctx - (int) tokens.size();
            if (tokens.size() + (size_t) o.predict > (size_t) o.ctx)
                throw std::runtime_error("prompt + output exceeds --ctx");
            if (o.kv_tiered)
                set_kv_tier(model, draft_model, o, base_params, (uint32_t) tokens.size() + o.predict,
                            ctx, draft_ctx, mtp);
            if (!ctx) throw std::runtime_error("target context unavailable after KV tier allocation failure");
            generate(model, ctx.get(), o, {}, &tokens, s, mtp.get());
        } catch (const std::exception & e) {
            std::printf("ERR %s\n", e.what()); std::fflush(stdout);
        }
    }
}
}
int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    try {
        llama_log_set(log_message, nullptr); ggml_log_set(log_message, nullptr);
        const auto o = parse(argc, argv);
        const size_t expert_bytes = inspect(o.model);
        ggml_backend_load_all(); llama_backend_init();
        ggml_backend_dev_t gpu = nullptr;
        if (!o.cpu) {
            int index = 0;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto d = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU && index++ == o.device) { gpu = d; break; }
            }
            if (!gpu) throw std::runtime_error("GPU unavailable; build a GPU backend or use --cpu");
        }
        size_t free_before = 0, total = 0;
        if (gpu) ggml_backend_dev_memory(gpu, &free_before, &total);
        if (gpu) {
            strata_a3b::Config backend;
            backend.gpu = gpu; backend.expert_bytes = expert_bytes; backend.threads = o.threads;
            backend.stream_prefill = o.stream; backend.pageable_experts = o.pageable_experts;
            backend.stable_tasks = o.stable_graphs; backend.preserve_verify_layout = o.preserve_verify_layout;
            backend.cache_graph_routes = o.cache_graph_routes; backend.exact_verify = o.exact_verify;
            backend.native_exact_verify = ((o.native_gpu || o.gpu_only_experts) && o.exact_verify) || o.mtp_fused_rounds;
            backend.native_dense_exact = o.native_dense_exact;
            backend.native_attention_exact = o.native_attention_exact;
            backend.native_exact_max_tokens = o.mtp_fused_rounds ? 8 : o.mtp_draft + 1;
            backend.fused_moe = o.fused_moe;
            backend.strata_native_experts = o.strata_native_experts;
            backend.strata_prefill_mmq = o.strata_prefill_mmq;
            backend.gpu_only_experts = o.gpu_only_experts;
            backend.hybrid_experts = o.hybrid_experts;
            backend.hybrid_doorbell = o.hybrid_doorbell;
            backend.cpu_expert_pool = o.cpu_expert_pool;
            backend.prefill_cpu_share = o.prefill_cpu_share;
            backend.pcie_fraction = o.pcie_fraction;
            backend.async_expert_events = o.async_expert_events;
            backend.cache_decay_interval = o.cache_decay_interval;
            backend.gpu_static_experts = o.gpu_static_experts;
            backend.gpu_dynamic_experts = o.gpu_dynamic_experts;
            backend.expert_staging_slots = o.expert_staging_slots;
            backend.expert_ring_banks = o.expert_ring_banks;
            backend.prefill_copy_ahead = o.prefill_copy_ahead;
            backend.prefill_independent_issuer = o.prefill_independent_issuer;
            backend.prefill_packed_gate_up = o.prefill_packed_gate_up;
            strata_a3b::register_backend(backend);
        }
        std::vector<llama_model_tensor_buft_override> overrides;
        if (gpu && !o.native_gpu) overrides.push_back({expert_pattern, strata_a3b::expert_buffer_type()});
        if (gpu && o.gpu_embedding) overrides.push_back({"^token_embd\\.weight$", ggml_backend_dev_buffer_type(gpu)});
        overrides.push_back({nullptr, nullptr});
        ggml_backend_dev_t devices[] = {gpu, nullptr};
        auto mp = llama_model_default_params();
        // Loading copies routed weights into our host buffer. Keeping the source
        // mapping resident as well roughly doubles physical RAM on Windows.
        mp.load_mode = o.mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
        mp.devices = devices; mp.n_gpu_layers = gpu ? 999 : 0; mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.tensor_buft_overrides = gpu ? overrides.data() : nullptr;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(o.model.c_str(), mp), llama_model_free);
        if (!model) throw std::runtime_error("model load failed");
        StaticExpertScope static_scope{o.gpu_static_experts || o.hybrid_doorbell};
        GdnDebugDump gdn_debug_dump;
        if (const char * path = std::getenv("STRATA_A3B_GDN_DEBUG_DUMP")) {
            gdn_debug_dump.file = std::fopen(path, "wb");
            if (!gdn_debug_dump.file) throw std::runtime_error("cannot open GDN debug dump");
        }
        auto cp = llama_context_default_params();
        if (gdn_debug_dump.file) {
            cp.cb_eval = gdn_debug_eval;
            cp.cb_eval_user_data = &gdn_debug_dump;
        }
        const bool tiny_vram = o.kv_tier_low_vram && o.vram_mib > 0 && o.vram_mib <= 2048;
        cp.n_ctx = o.kv_tiered ? std::min(o.ctx, tiny_vram ? 1024 : 65536) : o.ctx;
        cp.n_batch = o.batch; cp.n_ubatch = o.batch;
        cp.n_threads = cp.n_threads_batch = o.threads; cp.no_perf = false;
        if (o.kv_q8) cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
        if (o.kv_q4) cp.type_k = cp.type_v = GGML_TYPE_Q4_0;
        apply_short_f16_kv_tier(o, cp.n_ctx, cp);
        cp.offload_kqv = gpu != nullptr;
        cp.n_rs_seq = o.mtp_model.empty() ? 0 : o.mtp_draft;
        cp.n_outputs_max = cp.n_outputs_max_per_seq = o.mtp_model.empty() ? 1 : o.mtp_draft + 1;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) throw std::runtime_error("context allocation failed; reduce --ctx or --batch");
        if (o.kv_tiered)
            std::fprintf(stderr, "A3B tiered KV: allocated %u cells with K=%s V=%s at startup; context limit %d\n",
                         llama_n_ctx(ctx.get()), ggml_type_name(cp.type_k), ggml_type_name(cp.type_v), o.ctx);
        std::unique_ptr<llama_model, decltype(&llama_model_free)> draft_model(nullptr, llama_model_free);
        std::unique_ptr<llama_context, decltype(&llama_free)> draft_ctx(nullptr, llama_free);
        std::unique_ptr<strata_a3b::MtpDraft> mtp;
        if (!o.mtp_model.empty()) {
            llama_model_tensor_buft_override shared_overrides[] = {
                {"^(token_embd|output|output_norm)\\.weight$", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
            // K2 feeds a GPU-sampled token directly into the next draft step.
            // Its Q8 token table must also be on that GPU; otherwise GET_ROWS
            // becomes a CPU split and the vendor preflight falls back before
            // changing the draft KV. Keep this extra 515 MiB opt-in.
            const char * chain_env = std::getenv("STRATA_A3B_MTP_DEVICE_CHAIN_K2");
            const bool chain_gpu_embedding = gpu && o.gpu_embedding && chain_env &&
                std::strcmp(chain_env, "1") == 0 && !o.share_mtp_weights;
            llama_model_tensor_buft_override chain_overrides[] = {
                {"^token_embd\\.weight$", gpu ? ggml_backend_dev_buffer_type(gpu) : nullptr},
                {nullptr, nullptr}};
            auto dm = mp;
            dm.tensor_buft_overrides = o.share_mtp_weights ? shared_overrides :
                (chain_gpu_embedding ? chain_overrides : nullptr);
            dm.load_mtp = true;
            draft_model.reset(llama_model_load_from_file(o.mtp_model.c_str(), dm));
            if (!draft_model) throw std::runtime_error("MTP model load failed");
            if (o.share_mtp_weights) {
                const auto shared = strata_a3b::share_mtp_weights(model.get(), draft_model.get(), o.model, o.mtp_model);
                std::fprintf(stderr, "A3B shared MTP: reused GPU %.1f MiB, host %.1f MiB; SHA-256 verified\n",
                             shared.reused_gpu_bytes / double(MiB), shared.reused_host_bytes / double(MiB));
            }
            auto dc = cp; dc.n_rs_seq = 0; dc.ctx_type = LLAMA_CONTEXT_TYPE_MTP; dc.ctx_other = ctx.get();
            dc.cb_eval = nullptr; dc.cb_eval_user_data = nullptr;
            dc.n_ubatch = std::min(cp.n_ubatch, (uint32_t) o.mtp_ubatch);
            dc.n_outputs_max = dc.n_outputs_max_per_seq = 1;
            draft_ctx.reset(llama_init_from_model(draft_model.get(), dc));
            if (!draft_ctx) throw std::runtime_error("MTP context allocation failed");
            mtp = std::make_unique<strata_a3b::MtpDraft>(ctx.get(), draft_ctx.get(), o.mtp_draft, 0.0f,
                                                      o.gpu_sampling && !o.cpu, o.deferred_catchup, o.mtp_adaptive,
                                                      o.mtp_fused_rounds);
        }
        size_t budget = 0;
        if (gpu && !o.native_gpu) {
            const size_t static_scratch = (o.gpu_static_experts || o.hybrid_doorbell) ? strata_a3b::gpu_static_scratch_reserve_bytes() : 0;
            const size_t cache_reserve = (size_t) o.cache_reserve_mib * MiB + static_scratch;
            if (o.gpu_static_experts || o.hybrid_doorbell)
                std::fprintf(stderr, "A3B static scratch reserve: %.1f MiB\n", static_scratch / double(MiB));
            size_t free_now = 0; ggml_backend_dev_memory(gpu, &free_now, &total);
            const size_t spent = free_before > free_now ? free_before - free_now : 0;
            const size_t cap = o.vram_mib ? (size_t) o.vram_mib * MiB : free_before;
            if (spent + cache_reserve > cap) throw std::runtime_error("trunk/context exceeds --vram-mib; reduce --ctx or --batch");
            const size_t available = std::min(free_now, cap > spent ? cap - spent : 0);
            budget = available > cache_reserve ? available - cache_reserve : 0;
            if (o.cache_mib >= 0) budget = std::min(budget, (size_t) o.cache_mib * MiB);
            budget = std::min(budget, expert_bytes);
            std::fprintf(stderr, "A3B: %s, GPU allocated %.1f MiB, cache budget %.1f MiB, reserve %d MiB\n",
                         ggml_backend_dev_name(gpu), spent / double(MiB), budget / double(MiB), o.cache_reserve_mib);
        }
        strata_a3b::set_cache_budget(budget);
        if (!o.expert_profile.empty()) {
            const auto plan = strata_a3b::load_expert_cache_profile(o.expert_profile, o.model, budget);
            strata_a3b::set_cache_plan(plan);
            std::fprintf(stderr, "A3B expert profile: %llu experts %.1f MiB; training coverage %.4f; model SHA %s; hash %.1f ms\n",
                (unsigned long long) plan.selected_experts, plan.planned_bytes / double(MiB),
                plan.coverage, plan.model_sha256.c_str(), plan.model_hash_ms);
            // Populate the selected raw Q8 rows before READY. The temporary
            // token and recurrent state are cleared before any user request.
            const auto warm_start = ggml_time_us();
            llama_token warm_token = 0;
            if (llama_decode(ctx.get(), llama_batch_get_one(&warm_token, 1)))
                throw std::runtime_error("profile expert cache warmup failed");
            llama_synchronize(ctx.get());
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            if (mtp) mtp->reset();
            if (o.gpu_static_experts || o.hybrid_doorbell) strata_a3b::collect_gpu_static_stats();
            std::fprintf(stderr, "A3B expert profile: startup warmup %.1f ms; user KV cache cleared\n",
                         (ggml_time_us() - warm_start) / 1000.0);
        }
        if (o.serve) serve(model.get(), ctx, o, draft_model.get(), draft_ctx, mtp, cp);
        else if (o.repl) {
            std::string line;
            std::fprintf(stderr, "A3B ready (one raw prompt per line)\n");
            while (std::getline(std::cin, line)) generate(model.get(), ctx.get(), o, line, nullptr, {}, mtp.get());
        } else generate(model.get(), ctx.get(), o, o.prompt, nullptr, {}, mtp.get());
        mtp.reset(); draft_ctx.reset(); draft_model.reset(); ctx.reset();
        static_scope.close(); model.reset(); llama_backend_free();
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "strata-moe: %s\n", e.what()); return 1; }
}
