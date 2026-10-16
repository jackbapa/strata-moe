#include "speculative.hpp"
#include "shared_weights.hpp"
#include "llama-ext.h"
#include "llama-model.h"
#include "backend.hpp"
#include "cache_profile.hpp"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr size_t MiB = 1024 * 1024;
strata_a3b::Stats read_backend_stats() {
    strata_a3b::collect_gpu_static_stats();
    return strata_a3b::stats();
}
struct StaticScope {
    bool enabled;
    ~StaticScope() {
        if (enabled) try { strata_a3b::release_gpu_static_experts(); }
        catch (const std::exception & e) { std::fprintf(stderr, "static expert cleanup: %s\n", e.what()); }
    }
};
const char * expert_pattern = "blk\\.[0-9]+\\.ffn_(gate|up|down|gate_up)_exps\\.weight";

size_t routed_bytes(const std::string & path) {
    ggml_context * ctx = nullptr;
    gguf_context * uf = gguf_init_from_file(path.c_str(), {true, &ctx});
    if (!uf) throw std::runtime_error("cannot inspect expert GGUF");
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(uf, gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(ctx, ggml_free);
    const std::regex pattern(expert_pattern);
    size_t bytes = 0;
    for (auto * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t))
        if (std::regex_match(t->name, pattern)) bytes += ggml_nbytes(t);
    if (!bytes) throw std::runtime_error("GGUF contains no routed expert tensors");
    return bytes;
}

void print_stats(const char * phase, const strata_a3b::Stats & previous, const strata_a3b::Stats & current) {
    std::printf("cache %s bytes=%.1fMiB GPU_rows=%llu CPU_rows=%llu hits=%llu misses=%llu uploads=%llu upload=%.1fMiB route_reads=%llu preserved_layout_calls=%llu exact_verify_calls=%llu\n",
                phase, current.cache_bytes / double(MiB),
                (unsigned long long) (current.gpu_rows - previous.gpu_rows),
                (unsigned long long) (current.cpu_rows - previous.cpu_rows),
                (unsigned long long) (current.hits - previous.hits),
                (unsigned long long) (current.misses - previous.misses),
                (unsigned long long) (current.uploads - previous.uploads),
                (current.upload_bytes - previous.upload_bytes) / double(MiB),
                (unsigned long long) (current.route_reads - previous.route_reads),
                (unsigned long long) (current.preserved_layout_calls - previous.preserved_layout_calls),
                (unsigned long long) (current.exact_verify_calls - previous.exact_verify_calls));
    if (current.gpu_only_rows || current.staged_rows) {
        std::printf("GPU-only %s rows=%llu staged_rows=%llu staged_experts=%llu staging_upload=%.1fMiB staging_batches=%llu staging_allocations=%llu staging_GPU=%.1fMiB staging_host=%.1fMiB\n",
                    phase, (unsigned long long) (current.gpu_only_rows - previous.gpu_only_rows),
                    (unsigned long long) (current.staged_rows - previous.staged_rows),
                    (unsigned long long) (current.staged_experts - previous.staged_experts),
                    (current.staging_upload_bytes - previous.staging_upload_bytes) / double(MiB),
                    (unsigned long long) (current.staging_batches - previous.staging_batches),
                    (unsigned long long) (current.staging_allocations - previous.staging_allocations),
                    current.staging_gpu_bytes / double(MiB), current.staging_host_bytes / double(MiB));
    }
}

void print_native_graphs() {
    const auto s = read_backend_stats();
    std::printf("native hook graph-builder acceptances dense=%llu expert=%llu fused=%llu attention=%llu (includes reserve/probe; not execution counts)\n",
                (unsigned long long) s.native_dense_graphs, (unsigned long long) s.native_expert_graphs,
                (unsigned long long) s.native_fused_graphs, (unsigned long long) s.native_attention_graphs);
}

void warm(llama_context * ctx, const std::vector<llama_token> & tokens) {
    strata_a3b::collect_gpu_static_stats();
    llama_memory_clear(llama_get_memory(ctx), true);
    auto batch = llama_batch_get_one(const_cast<llama_token *>(tokens.data()), (int) tokens.size());
    if (llama_decode(ctx, batch)) throw std::runtime_error("cached expert warmup failed");
    llama_synchronize(ctx);
    llama_memory_clear(llama_get_memory(ctx), true);
}

struct TestSampler {
    llama_context * ctx;
    bool device;
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler{nullptr, llama_sampler_free};
    TestSampler(llama_context * context, bool gpu) : ctx(context), device(gpu) {
        sampler.reset(gpu ? strata_a3b::create_device_greedy_sampler() :
                      llama_sampler_chain_init(llama_sampler_chain_default_params()));
        if (!gpu) llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
        if (gpu && !llama_set_sampler(ctx, 0, sampler.get()))
            throw std::runtime_error("acceptance diagnostic device sampler unavailable");
    }
    ~TestSampler() { if (device) llama_set_sampler(ctx, 0, nullptr); }
    llama_token sample(int row) {
        if (device) strata_a3b::device_greedy_token(ctx, row);
        else {
            const auto * values = llama_get_logits_ith(ctx, row);
            const int n = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
            if (!values) throw std::runtime_error("acceptance diagnostic missing logits");
            for (int i = 0; i < n; ++i)
                if (!std::isfinite(values[i])) throw std::runtime_error("acceptance diagnostic non-finite logits");
        }
        return llama_sampler_sample(sampler.get(), ctx, row);
    }
};

void decode_inputs(llama_context * ctx, const std::vector<llama_token> & tokens,
                   llama_pos pos, bool all_outputs) {
    strata_a3b::collect_gpu_static_stats();
    auto batch = llama_batch_init((int) tokens.size(), 0, 1);
    batch.n_tokens = (int) tokens.size();
    for (int i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[i]; batch.pos[i] = pos + i;
        batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
        batch.logits[i] = all_outputs || i + 1 == batch.n_tokens;
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc) throw std::runtime_error("acceptance diagnostic decode failed: " + std::to_string(rc));
}

struct BatchTrace {
    struct Tensor {
        int64_t shape[GGML_MAX_DIMS];
        size_t stride[GGML_MAX_DIMS];
        std::vector<uint8_t> data;
    };
    bool active = false, recording = true;
    int rows = 0, row = 0;
    bool row_has_difference = false;
    std::map<std::string, std::vector<Tensor>> batch;
    std::map<std::string, size_t> occurrence;

    static bool callback(ggml_tensor * t, bool ask, void * userdata) {
        auto & trace = *static_cast<BatchTrace *>(userdata);
        const std::string name = t->name;
        const auto split = name.find_last_of('-');
        const bool layer = split != std::string::npos && name.size() == split + 2 &&
                           name[split + 1] >= '0' && name[split + 1] <= '3';
        if (ask && trace.active && trace.recording &&
            (t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID) && t->src[0] && t->src[1]) {
            const std::string weight = t->src[0]->name;
            if (weight.rfind("blk.", 0) == 0 && weight.size() > 5 && weight[5] == '.' &&
                weight[4] >= '0' && weight[4] <= '3') {
                const auto * w = t->src[0]; const auto * x = t->src[1];
                std::printf("TRACE graph op=%s weight=%s type=%s buffer=%s host=%d input=[%lld,%lld,%lld,%lld]\n",
                            ggml_op_name(t->op), w->name, ggml_type_name(w->type),
                            w->buffer ? ggml_backend_buffer_name(w->buffer) : "none",
                            w->buffer ? (int) ggml_backend_buffer_is_host(w->buffer) : -1,
                            (long long) x->ne[0], (long long) x->ne[1], (long long) x->ne[2], (long long) x->ne[3]);
            }
        }
        const bool selected = trace.active && t->type == GGML_TYPE_F32 &&
                              (layer || name == "model.input_embed") && name.find("state") == std::string::npos;
        if (ask) return selected;
        if (!selected) return true;
        Tensor current;
        std::copy(t->ne, t->ne + GGML_MAX_DIMS, current.shape);
        std::copy(t->nb, t->nb + GGML_MAX_DIMS, current.stride);
        current.data.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, current.data.data(), 0, current.data.size());
        if (trace.recording) {
            trace.batch[name].push_back(std::move(current));
            return true;
        }
        const auto found = trace.batch.find(name);
        if (found == trace.batch.end()) return true;
        const size_t occurrence = trace.occurrence[name]++;
        if (occurrence >= found->second.size()) return true;
        const auto & reference = found->second[occurrence];
        int axis = -1;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (current.shape[i] == reference.shape[i]) continue;
            if (axis >= 0 || current.shape[i] != 1 || reference.shape[i] != trace.rows) return true;
            axis = i;
        }
        if (axis < 0) return true; // per-block tensors have no token axis
        double maximum = 0, mean = 0;
        bool finite = true;
        const size_t elements = (size_t) ggml_nelements(t);
        for (size_t index = 0; index < elements; ++index) {
            size_t rest = index, src = 0, dst = 0;
            for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
                const size_t coordinate = rest % current.shape[dim];
                rest /= current.shape[dim];
                src += coordinate * current.stride[dim];
                dst += (dim == axis ? (size_t) trace.row : coordinate) * reference.stride[dim];
            }
            float a, b;
            std::memcpy(&a, current.data.data() + src, sizeof(a));
            std::memcpy(&b, reference.data.data() + dst, sizeof(b));
            finite &= std::isfinite(a) && std::isfinite(b);
            const double difference = std::abs(double(a) - b);
            maximum = std::max(maximum, difference);
            mean += difference;
        }
        const bool first = maximum > 0 && !trace.row_has_difference;
        if (maximum > 0) trace.row_has_difference = true;
        std::printf("TRACE row=%d tensor=%s occurrence=%zu op=%s axis=%d max_abs=%.9g mean_abs=%.9g finite=%d%s\n",
                    trace.row, name.c_str(), occurrence, ggml_op_name(t->op), axis, maximum, mean / elements,
                    (int) finite, first ? " FIRST_DIFFERENCE" : "");
        return true;
    }
};

void trace_batch_difference(llama_context * ctx, BatchTrace & trace,
                            const std::vector<llama_token> & prefix, const std::vector<llama_token> & window) {
    std::puts("TRACE diagnostic: synchronized callback observations can inhibit fusion; these measurements are not throughput or normal-graph parity evidence.");
    llama_memory_clear(llama_get_memory(ctx), true);
    for (size_t i = 0; i < prefix.size(); ++i) decode_inputs(ctx, {prefix[i]}, (llama_pos) i, false);
    trace.rows = (int) window.size(); trace.recording = true; trace.active = true;
    decode_inputs(ctx, window, (llama_pos) prefix.size(), true);
    trace.active = false;
    llama_memory_clear(llama_get_memory(ctx), true);
    for (size_t i = 0; i < prefix.size(); ++i) decode_inputs(ctx, {prefix[i]}, (llama_pos) i, false);
    trace.recording = false;
    for (size_t i = 0; i < window.size(); ++i) {
        trace.row = (int) i; trace.row_has_difference = false; trace.occurrence.clear(); trace.active = true;
        decode_inputs(ctx, {window[i]}, (llama_pos) (prefix.size() + i), false);
        trace.active = false;
        std::printf("TRACE row=%zu observed_difference=%d within first four layers\n", i, (int) trace.row_has_difference);
    }
}

std::vector<llama_token> generate_greedy(llama_context * ctx, const std::vector<llama_token> & prompt,
                                        int limit, bool gpu_sampler, strata_a3b::MtpDraft * mtp,
                                        std::vector<float> * first_logits = nullptr) {
    llama_memory_clear(llama_get_memory(ctx), true);
    if (mtp) mtp->reset();
    TestSampler sampler(ctx, gpu_sampler);
    for (auto token : prompt) llama_sampler_accept(sampler.sampler.get(), token);
    decode_inputs(ctx, prompt, 0, false);
    if (first_logits) {
        if (gpu_sampler) throw std::runtime_error("prefill diagnostic requires raw CPU logits, not a backend sampler");
        const auto * logits = llama_get_logits_ith(ctx, -1);
        const auto count = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        if (!logits) throw std::runtime_error("prefill diagnostic logits missing");
        first_logits->assign(logits, logits + count);
    }
    if (mtp) mtp->observe(prompt.data(), (int) prompt.size(), 0);
    const auto * vocab = llama_model_get_vocab(llama_get_model(ctx));
    std::vector<llama_token> result;
    uint64_t matched_total = 0;
    llama_token last = sampler.sample(-1);
    result.push_back(last);
    llama_pos pos = (llama_pos) prompt.size();
    while ((int) result.size() < limit && !llama_vocab_is_eog(vocab, last)) {
        if (!mtp) {
            decode_inputs(ctx, {last}, pos++, false);
            last = sampler.sample(-1);
            result.push_back(last);
            continue;
        }
        const auto proposals = mtp->propose(last, pos, limit - (int) result.size() - 1);
        std::vector<llama_token> window{last};
        window.insert(window.end(), proposals.begin(), proposals.end());
        decode_inputs(ctx, window, pos, true);
        mtp->observe(window.data(), (int) window.size(), pos, true);
        int consumed = 0, matched = 0;
        for (int row = 0; row < (int) window.size(); ++row) {
            last = sampler.sample(row);
            if (row < (int) proposals.size() && last == proposals[row]) ++matched;
            result.push_back(last);
            ++consumed;
            if (llama_vocab_is_eog(vocab, last) || (int) result.size() == limit ||
                row == (int) proposals.size() || last != proposals[row]) break;
        }
        mtp->commit(consumed, matched);
        matched_total += matched;
        pos += consumed;
    }
    if (mtp) mtp->finish();
    if (mtp && mtp->stats().accepted != matched_total)
        throw std::runtime_error("MTP accepted metric differs from target-matching output proposals");
    return result;
}

void dump_final_sequence_state(llama_context * ctx, const std::filesystem::path & directory,
                               int limit, const char * role) {
    llama_synchronize(ctx);
    const size_t expected = llama_state_seq_get_size(ctx, 0);
    if (!expected) throw std::runtime_error("final sequence state has zero serialized size");
    std::vector<uint8_t> state(expected);
    const size_t written = llama_state_seq_get_data(ctx, state.data(), state.size(), 0);
    if (written != expected) throw std::runtime_error("final sequence state serialization size mismatch");
    uint64_t hash = 14695981039346656037ull;
    for (const uint8_t byte : state) { hash ^= byte; hash *= 1099511628211ull; }
    std::filesystem::create_directories(directory);
    const auto path = directory / ("limit-" + std::to_string(limit) + "." + role + "-seq0.bin");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char *>(state.data()), (std::streamsize) state.size());
    output.close();
    if (!output || std::filesystem::file_size(path) != written)
        throw std::runtime_error("final sequence state dump write failed");
    std::printf("MTP final %s state limit=%d seq=0 expected_bytes=%zu serialized_bytes=%zu file_bytes=%llu fnv1a=%016llx path=%s\n",
                role, limit, expected, written, (unsigned long long) std::filesystem::file_size(path),
                (unsigned long long) hash, path.string().c_str());
}

void validate_mtp_class(llama_model * target_model, llama_context * target, const llama_model_params & model_params,
                        const llama_context_params & context_params, const std::string & target_path,
                        const std::string & draft_path, int drafts, const std::vector<llama_token> & prompt,
                         bool gpu_sampling, bool deferred, bool share_weights, int draft_ubatch, bool adaptive,
                         bool fused_rounds, const std::string & state_directory) {
    llama_model_tensor_buft_override shared_overrides[] = {
        {"^token_embd\\.weight$", ggml_backend_cpu_buffer_type()},
        {"^output\\.weight$", ggml_backend_cpu_buffer_type()},
        {"^output_norm\\.weight$", ggml_backend_cpu_buffer_type()},
        {nullptr, nullptr},
    };
    auto mp = model_params;
    mp.load_mtp = true;
    // Preserve the target fixture's GPU token-table placement for the MTP
    // fixture unless the shared-weight fixture explicitly overrides it.  The
    // device K2 preflight must see its token table on CUDA; dropping this
    // override silently turns every attempted chain into a fallback.
    mp.tensor_buft_overrides = share_weights ? shared_overrides : model_params.tensor_buft_overrides;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(draft_path.c_str(), mp), llama_model_free);
    if (!model) throw std::runtime_error("acceptance diagnostic MTP model load failed");
    if (share_weights) {
        const auto shared = strata_a3b::share_mtp_weights(target_model, model.get(), target_path, draft_path);
        std::printf("acceptance diagnostic shared weights reused_GPU=%.3fMiB reused_host=%.3fMiB retained_draft_host=%.3fMiB\n",
                    shared.reused_gpu_bytes / double(MiB), shared.reused_host_bytes / double(MiB),
                    shared.retained_draft_host_bytes / double(MiB));
    }
    auto cp = context_params;
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cp.n_rs_seq = 0;
    cp.n_outputs_max = cp.n_outputs_max_per_seq = 1;
    if (draft_ubatch) cp.n_ubatch = draft_ubatch;
    std::unique_ptr<llama_context, decltype(&llama_free)> draft(llama_init_from_model(model.get(), cp), llama_free);
    if (!draft) throw std::runtime_error("acceptance diagnostic MTP context allocation failed");
    std::printf("acceptance diagnostic draft batch=%u ubatch=%u; prefix=%zu rows\n",
                llama_n_batch(draft.get()), llama_n_ubatch(draft.get()), prompt.size());
    strata_a3b::MtpDraft mtp(target, draft.get(), drafts, 0, gpu_sampling, deferred, adaptive, fused_rounds);
    const char * diagnostic_env = std::getenv("STRATA_A3B_PREFILL_DIAGNOSTIC");
    const bool diagnostic = diagnostic_env && diagnostic_env[0] && diagnostic_env[0] != '0';
    std::vector<float> baseline_logits;
    const auto baseline = generate_greedy(target, prompt, 32, false, nullptr, diagnostic ? &baseline_logits : nullptr);
    auto hash_logits = [](const std::vector<float> & logits) {
        uint64_t hash = 14695981039346656037ull;
        const auto * bytes = reinterpret_cast<const unsigned char *>(logits.data());
        for (size_t i = 0; i < logits.size() * sizeof(float); ++i) { hash ^= bytes[i]; hash *= 1099511628211ull; }
        return hash;
    };
    if (diagnostic) std::printf("PREFILL baseline first_token=%d logits_fnv1a=%016llx\n", baseline.front(),
                                (unsigned long long) hash_logits(baseline_logits));
    for (const int limit : {1, 2, 3, 4, 5, 8, 16, 32}) {
        if (diagnostic) {
            std::vector<float> repeated_logits;
            const auto repeated = generate_greedy(target, prompt, 1, false, nullptr, &repeated_logits);
            double maximum = 0;
            for (size_t i = 0; i < baseline_logits.size(); ++i)
                maximum = std::max(maximum, std::abs(double(repeated_logits[i]) - baseline_logits[i]));
            const bool identical = repeated_logits.size() == baseline_logits.size() &&
                std::memcmp(repeated_logits.data(), baseline_logits.data(), baseline_logits.size() * sizeof(float)) == 0;
            std::printf("PREFILL repeat limit=%d first_token=%d expected=%d logits_fnv1a=%016llx bitwise=%d max_abs=%.9g\n",
                        limit, repeated.front(), baseline.front(), (unsigned long long) hash_logits(repeated_logits),
                        (int) identical, maximum);
        }
        const auto actual = generate_greedy(target, prompt, limit, gpu_sampling, &mtp);
        const auto count = std::min((size_t) limit, baseline.size());
        const std::vector<llama_token> expected(baseline.begin(), baseline.begin() + count);
        if (actual != expected) {
            size_t index = 0;
            while (index < actual.size() && index < expected.size() && actual[index] == expected[index]) ++index;
            std::fprintf(stderr, "MTP output mismatch limit=%d index=%zu actual=%d expected=%d first_actual=%d first_expected=%d\n",
                         limit, index, index < actual.size() ? actual[index] : LLAMA_TOKEN_NULL,
                         index < expected.size() ? expected[index] : LLAMA_TOKEN_NULL,
                         actual.empty() ? LLAMA_TOKEN_NULL : actual.front(), expected.empty() ? LLAMA_TOKEN_NULL : expected.front());
            throw std::runtime_error("actual MTP class differs from sequential CPU greedy at limit=" +
                                     std::to_string(limit) + " token_index=" + std::to_string(index));
        }
        // generate_greedy has finished pending accepted catch-up and checked
        // acceptance accounting; only dump after target-output parity passes.
        // Sequence state excludes the transient logits/NextN output buffers.
        if (!state_directory.empty()) {
            dump_final_sequence_state(draft.get(), state_directory, limit, "draft");
            dump_final_sequence_state(target, state_directory, limit, "target");
        }
        const auto & s = mtp.stats();
        std::printf("MTP class limit=%d output=%zu offered=%llu accepted=%llu GPU_draft_samples=%llu draft_sampling_download_bytes=%llu target_GPU_sampler=%d deferred=%d PASS\n",
                    limit, actual.size(), (unsigned long long) s.offered, (unsigned long long) s.accepted,
                    (unsigned long long) s.gpu_draft_samples, (unsigned long long) s.draft_sampling_download_bytes,
                    (int) gpu_sampling, (int) deferred);
        if (fused_rounds) std::printf(
            "MTP fused class limit=%d calls=%llu catchup_rows=%llu fused_ms=%.3f draft_ms=%.3f catchup_ms=%.3f exclusive_phase_times=1 PASS\n",
            limit, (unsigned long long) s.fused_rounds, (unsigned long long) s.fused_catchup_rows,
            s.fused_round_ms, s.draft_ms, s.catchup_ms);
    }
    const auto compact_rows = mtp.compact_gpu_sample_rows();
    const char * compact_flag = std::getenv("STRATA_A3B_DRAFT_COMPACT_GPU");
    const bool compact_expected = gpu_sampling && model->d2t && model->d2t->type == GGML_TYPE_I32 &&
        compact_flag && std::strcmp(compact_flag, "1") == 0;
    if ((compact_expected && compact_rows == 0) || (!compact_expected && compact_rows != 0))
        throw std::runtime_error("compact GPU sampler graph row count does not match the selected path");
    std::printf("MTP compact GPU sampler graph sampled rows=%llu expected=%d PASS\n",
                (unsigned long long) compact_rows, (int) compact_expected);
}
} // namespace

// Native CPU experts are the default. --cached-experts instead warms the real
// custom cache before comparing prefixes; row/upload counters make remaining
// device-placement changes visible. This loads the base GGUF, not an MTP head.
int main(int argc, char ** argv) {
    try {
        std::string path, mtp_path, expert_profile, draft_state_directory;
        int gpu_index = -1;
        int draft_count = 3;
        int draft_ubatch = 0;
        int cache_mib = -1;
        bool cached = false, assert_sequential = false, pageable = false;
        bool freeze_cache = false, native_gpu = false, fused_moe = false, native_dense_exact = false;
        bool strata_native_experts = false;
        bool gpu_only_experts = false, staging_slots_explicit = false;
        bool async_expert_events = false;
        bool gpu_static_experts = false;
        bool hybrid_experts = false, cpu_expert_pool = false;
        bool hybrid_doorbell = false;
        bool prefill_cpu_share = false;
        double pcie_fraction = 0.0;
        bool pcie_explicit = false;
        bool gpu_dynamic_experts = false;
        int cache_decay_interval = 0;
        int expert_staging_slots = 8;
        bool native_attention_exact = false;
        bool gpu_embedding = false, nextn_hidden = false;
        bool mtp_gpu_sampling = false, mtp_deferred = false, share_weights = false, mtp_adaptive = false;
        bool mtp_fused_rounds = false;
        bool trace_difference = false;
        bool stable_tasks = false, preserve_verify_layout = false, cache_graph_routes = false, exact_verify = false;
        int threads = std::max(1u, std::thread::hardware_concurrency() / 2);
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if ((key == "--model" || key == "-m") && i + 1 < argc) path = argv[++i];
            else if (key == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
            else if (key == "--device" && i + 1 < argc) gpu_index = std::stoi(argv[++i]);
            else if (key == "--cached-experts") cached = true;
            else if (key == "--native-gpu") native_gpu = true;
            else if (key == "--strata-native-experts") strata_native_experts = fused_moe = true;
            else if (key == "--gpu-only-experts") gpu_only_experts = true;
            else if (key == "--async-expert-events") async_expert_events = true;
            else if (key == "--gpu-static-experts") gpu_static_experts = true;
            else if (key == "--hybrid-experts") hybrid_experts = true;
            else if (key == "--hybrid-doorbell") hybrid_doorbell = hybrid_experts = cpu_expert_pool = true;
            else if (key == "--cpu-expert-pool") cpu_expert_pool = true;
            else if (key == "--prefill-cpu-share") prefill_cpu_share = true;
            else if (key == "--pcie-frac" && i + 1 < argc) { pcie_fraction = std::stod(argv[++i]); pcie_explicit = true; }
            else if (key == "--gpu-dynamic-experts") gpu_dynamic_experts = gpu_static_experts = true;
            else if (key == "--cache-decay-interval" && i + 1 < argc) cache_decay_interval = std::stoi(argv[++i]);
            else if (key == "--expert-profile" && i + 1 < argc) expert_profile = argv[++i];
            else if (key == "--mtp-adaptive") mtp_adaptive = true;
            else if (key == "--mtp-fused-rounds") mtp_fused_rounds = true;
            else if (key == "--expert-staging-slots" && i + 1 < argc) {
                expert_staging_slots = std::stoi(argv[++i]); staging_slots_explicit = true;
            }
            else if (key == "--fused-moe") fused_moe = true;
            else if (key == "--native-dense-exact") native_dense_exact = true;
            else if (key == "--native-attention-exact") native_attention_exact = true;
            else if (key == "--gpu-embedding") gpu_embedding = true;
            else if (key == "--nextn-hidden") nextn_hidden = true;
            else if (key == "--mtp-model" && i + 1 < argc) mtp_path = argv[++i];
            else if (key == "--gpu-sampling") mtp_gpu_sampling = true;
            else if (key == "--defer-mtp-catchup") mtp_deferred = true;
            else if (key == "--share-mtp-weights") share_weights = true;
            else if (key == "--trace-batch-difference") trace_difference = true;
            else if (key == "--draft-count" && i + 1 < argc) draft_count = std::stoi(argv[++i]);
            else if (key == "--draft-ubatch" && i + 1 < argc) draft_ubatch = std::stoi(argv[++i]);
            else if (key == "--draft-state-dir" && i + 1 < argc) draft_state_directory = argv[++i];
            else if (key == "--cache-mib" && i + 1 < argc) cache_mib = std::stoi(argv[++i]);
            else if (key == "--freeze-cache") freeze_cache = true;
            else if (key == "--assert-sequential") assert_sequential = true;
            else if (key == "--pageable-experts") pageable = true;
            else if (key == "--stable-tasks") stable_tasks = true;
            else if (key == "--preserve-verify-layout") preserve_verify_layout = true;
            else if (key == "--cache-graph-routes") cache_graph_routes = true;
            else if (key == "--exact-verify") exact_verify = true;
            else if (key == "--help") {
                std::puts("strata-moe-speculative-test --model MODEL.gguf [--threads N] [--device GPU_INDEX]\n"
                          "  --cached-experts (requires --device 0; 16 GiB GPU budget; warms the fixture)\n"
                          "  --native-gpu --draft-count N (all weights on GPU; exact native verification; N=1..8)\n"
                          "  --strata-native-experts (execute original Strata IQ expert hook; requires --native-gpu on GPU 0)\n"
                          "  --gpu-only-experts (RAM expert storage, bounded GPU miss staging; no CPU expert computation)\n"
                          "  --async-expert-events (GPU-only device-event pipeline)\n"
                          "  --expert-profile FILE (verified global Q8 cache placement) --mtp-adaptive\n"
                          "  --expert-staging-slots N (GPU-only transient experts, 1..64, default 8)\n"
                          "  --fused-moe (whole-chain exact verification; requires --native-gpu or --gpu-only-experts)\n"
                          "  --native-dense-exact (dense one-column verification; requires a GPU-only mode)\n"
                          "  --native-attention-exact (attention single-query verification; requires a GPU-only mode)\n"
                          "  --gpu-embedding --nextn-hidden (match the native MTP target configuration)\n"
                          "  --mtp-model MODEL.gguf --gpu-sampling --defer-mtp-catchup --share-mtp-weights\n"
                          "  --mtp-fused-rounds (accepted catch-up + first proposal; GPU sampling/defer/exact trunk)\n"
                          "    (after strict state PASS, compare actual MTP class at output limits 1/2/3/4/5/8/16/32)\n"
                          "  --draft-ubatch N (MTP microbatch size 1..32; logical batch stays 32)\n"
                          "  --draft-state-dir DIR (dump final draft/target seq0 bytes after each passing MTP class limit)\n"
                          "  --trace-batch-difference (diagnostic-only synchronized intermediate comparison, then exit)\n"
                          "  --cache-mib N --freeze-cache (fixed persistent placement; GPU-only misses still stage)\n"
                          "  --assert-sequential (also require top1 agreement and sequential KL < 1e-4)\n"
                          "  --pageable-experts --stable-tasks --preserve-verify-layout --cache-graph-routes --exact-verify\n"
                          "GDN rollback restoration and batch arithmetic diagnostics, independent of draft accuracy.");
                return 0;
            } else throw std::runtime_error("unknown or incomplete argument: " + key);
        }
        if (path.empty() || threads < 1 || cache_mib < -1 || draft_count < 1 || draft_count > 8 ||
            draft_ubatch < 0 || draft_ubatch > 32)
            throw std::runtime_error("supply --model MODEL.gguf, positive --threads and a nonnegative cache limit");
        if (expert_staging_slots < 1 || expert_staging_slots > 64)
            throw std::runtime_error("--expert-staging-slots must be 1..64");
        if (gpu_only_experts || hybrid_experts) {
            if (gpu_index != 0 || native_gpu || (gpu_only_experts && hybrid_experts))
                throw std::runtime_error("--gpu-only-experts requires --device 0 and cannot use --native-gpu");
            cached = fused_moe = gpu_embedding = exact_verify = true;
            if (!mtp_path.empty()) nextn_hidden = true;
        }
        if (cpu_expert_pool && !hybrid_experts)
            throw std::runtime_error("--cpu-expert-pool requires --hybrid-experts");
        if (prefill_cpu_share && (!hybrid_experts || !cpu_expert_pool))
            throw std::runtime_error("--prefill-cpu-share requires hybrid experts and the native CPU pool");
        if (!std::isfinite(pcie_fraction) || pcie_fraction < 0 || pcie_fraction > 1 ||
            (pcie_explicit && !hybrid_experts))
            throw std::runtime_error("--pcie-frac requires --hybrid-experts and a finite value in 0..1");
        if (staging_slots_explicit && !(gpu_only_experts || hybrid_experts))
            throw std::runtime_error("--expert-staging-slots requires --gpu-only-experts");
        if (async_expert_events && !(gpu_only_experts || hybrid_experts))
            throw std::runtime_error("--async-expert-events requires --gpu-only-experts");
        if (cache_decay_interval < 0 || cache_decay_interval > 1024 || (cache_decay_interval && !(gpu_only_experts || hybrid_experts)))
            throw std::runtime_error("--cache-decay-interval requires GPU-only and a value in 0..1024");
        if (!expert_profile.empty() && !(gpu_only_experts || hybrid_experts))
            throw std::runtime_error("--expert-profile requires --gpu-only-experts");
        if (gpu_static_experts && (!gpu_only_experts || expert_profile.empty() || cache_decay_interval))
            throw std::runtime_error("--gpu-static-experts requires GPU-only, a verified profile and decay 0");
        if (hybrid_doorbell && (gpu_static_experts || expert_profile.empty() || cache_decay_interval ||
                                pcie_fraction != 0 || gpu_index != 0 || draft_count > 7))
            throw std::runtime_error("--hybrid-doorbell requires device 0, a verified profile, PCIe fraction 0, decay 0 and draft count <=7");
        const bool gpu_computation = native_gpu || gpu_only_experts || hybrid_experts;
        if (cached && gpu_index != 0) throw std::runtime_error("--cached-experts requires --device 0");
        if (native_gpu && (gpu_index != 0 || cached))
            throw std::runtime_error("--native-gpu requires --device 0 and cannot use --cached-experts");
        if (strata_native_experts && !native_gpu)
            throw std::runtime_error("--strata-native-experts requires --native-gpu on device 0");
        if (fused_moe && !gpu_computation) throw std::runtime_error("--fused-moe requires --native-gpu or --gpu-only-experts");
        if (native_dense_exact && !gpu_computation) throw std::runtime_error("--native-dense-exact requires a GPU-only mode");
        if (native_attention_exact && !gpu_computation) throw std::runtime_error("--native-attention-exact requires a GPU-only mode");
        if ((gpu_embedding || nextn_hidden) && !gpu_computation)
            throw std::runtime_error("--gpu-embedding and --nextn-hidden require a GPU-only mode in this diagnostic");
        if (!mtp_path.empty() && (!gpu_computation || !assert_sequential))
            throw std::runtime_error("--mtp-model requires a GPU-only mode and --assert-sequential");
        if (mtp_path.empty() && (mtp_gpu_sampling || mtp_deferred || share_weights || draft_ubatch || mtp_adaptive || mtp_fused_rounds))
            throw std::runtime_error("MTP execution options require --mtp-model");
        if (mtp_path.empty() && !draft_state_directory.empty())
            throw std::runtime_error("--draft-state-dir requires --mtp-model");
        if (mtp_fused_rounds && (!mtp_gpu_sampling || !mtp_deferred || !native_dense_exact ||
            !native_attention_exact || !gpu_computation || gpu_index != 0 || draft_count > 7))
            throw std::runtime_error("--mtp-fused-rounds requires GPU 0 sampling, deferred catch-up, native dense/attention exact and at most seven drafts");
        if (!cached && (pageable || stable_tasks || preserve_verify_layout || cache_graph_routes || exact_verify ||
                        freeze_cache || cache_mib >= 0))
            throw std::runtime_error("custom expert options require --cached-experts");
        llama_backend_init();
        ggml_backend_dev_t gpu = nullptr;
        if (gpu_index >= 0) {
            ggml_backend_load_all();
            int index = 0;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto d = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU && index++ == gpu_index) {
                    gpu = d;
                    break;
                }
            }
            if (!gpu) throw std::runtime_error("requested GPU unavailable in this build");
        }
        size_t free_before = 0, total = 0, expert_bytes = 0;
        if (gpu) ggml_backend_dev_memory(gpu, &free_before, &total);
        if (cached || native_gpu) {
            expert_bytes = routed_bytes(path);
            strata_a3b::Config config;
            config.gpu = gpu;
            config.expert_bytes = expert_bytes;
            config.threads = threads;
            config.pageable_experts = pageable;
            config.stable_tasks = stable_tasks;
            config.preserve_verify_layout = preserve_verify_layout;
            config.cache_graph_routes = cache_graph_routes;
            config.exact_verify = exact_verify;
            config.native_exact_verify = native_gpu || gpu_only_experts || mtp_fused_rounds;
            config.native_exact_max_tokens = mtp_fused_rounds ? 8 : draft_count + 1;
            config.fused_moe = fused_moe;
            config.strata_native_experts = strata_native_experts;
            config.native_dense_exact = native_dense_exact;
            config.native_attention_exact = native_attention_exact;
            config.gpu_only_experts = gpu_only_experts;
            config.async_expert_events = async_expert_events;
            config.gpu_static_experts = gpu_static_experts;
            config.hybrid_experts = hybrid_experts;
            config.hybrid_doorbell = hybrid_doorbell;
            config.cpu_expert_pool = cpu_expert_pool;
            config.prefill_cpu_share = prefill_cpu_share;
            config.pcie_fraction = pcie_fraction;
            config.gpu_dynamic_experts = gpu_dynamic_experts;
            config.cache_decay_interval = cache_decay_interval;
            config.expert_staging_slots = expert_staging_slots;
            strata_a3b::register_backend(config);
            std::printf("custom cache config expert_bytes=%zu pageable=%d stable_tasks=%d preserve_verify_layout=%d cache_graph_routes=%d exact_verify=%d\n",
                        expert_bytes, (int) pageable, (int) stable_tasks,
                        (int) preserve_verify_layout, (int) cache_graph_routes, (int) exact_verify);
            if (native_gpu) std::printf("native GPU verification enabled, max_tokens=%d fused_moe=%d native_dense_exact=%d native_attention_exact=%d, no routed tensor CPU override\n", config.native_exact_max_tokens, (int) fused_moe, (int) native_dense_exact, (int) native_attention_exact);
            if (gpu_only_experts) std::printf("GPU-only expert verification enabled, max_tokens=%d staging_slots=%d pageable=%d; experts stored in RAM, every routed row computed on GPU\n", config.native_exact_max_tokens, expert_staging_slots, (int) pageable);
        }
        ggml_backend_dev_t devices[] = {gpu, nullptr};
        std::vector<llama_model_tensor_buft_override> overrides;
        if (gpu && !native_gpu)
            overrides.push_back({expert_pattern, cached ? strata_a3b::expert_buffer_type() : ggml_backend_cpu_buffer_type()});
        if (gpu_embedding) overrides.push_back({"token_embd\\.weight", ggml_backend_dev_buffer_type(gpu)});
        overrides.push_back({nullptr, nullptr});
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = gpu ? 999 : 0;
        mp.devices = devices;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.tensor_buft_overrides = overrides.size() > 1 ? overrides.data() : nullptr;
        mp.load_mtp = false;
        if (cached || native_gpu) mp.load_mode = LLAMA_LOAD_MODE_NONE;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(path.c_str(), mp), llama_model_free);
        if (!model) throw std::runtime_error("model load failed");
        StaticScope static_scope{gpu_static_experts || hybrid_doorbell};
        auto cp = llama_context_default_params();
        BatchTrace trace;
        if (trace_difference) { cp.cb_eval = BatchTrace::callback; cp.cb_eval_user_data = &trace; }
        cp.n_ctx = 128;
        cp.n_batch = cp.n_ubatch = 32;
        cp.n_seq_max = 1;
        cp.n_rs_seq = draft_count;
        if (!mtp_path.empty()) cp.n_outputs_max = cp.n_outputs_max_per_seq = draft_count + 1;
        cp.n_threads = cp.n_threads_batch = threads;
        cp.offload_kqv = cp.op_offload = gpu != nullptr;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
            llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) throw std::runtime_error("context allocation failed");
        if (nextn_hidden) llama_set_embeddings_nextn(ctx.get(), true, false);
        if (gpu_computation) std::printf("GPU target configuration gpu_embedding=%d nextn_hidden=%d; raw target logits retained for independent reference\n",
                                    (int) gpu_embedding, (int) nextn_hidden);
        if (cached) {
            size_t free_now = 0;
            ggml_backend_dev_memory(gpu, &free_now, &total);
            const size_t spent = free_before > free_now ? free_before - free_now : 0;
            const size_t cap = 16384 * MiB, reserve = 512 * MiB;
            if (spent + reserve >= cap) throw std::runtime_error("custom test trunk/context exceeds 16 GiB budget");
            const size_t available = std::min(free_now, cap - spent);
            size_t budget = std::min(expert_bytes, available > reserve ? available - reserve : 0);
            if (cache_mib >= 0) budget = std::min(budget, (size_t) cache_mib * MiB);
            strata_a3b::set_cache_budget(budget);
            if (!expert_profile.empty()) {
                const auto plan = strata_a3b::load_expert_cache_profile(expert_profile, path, budget);
                strata_a3b::set_cache_plan(plan);
                std::printf("profile validation selected=%llu bytes=%llu\n", (unsigned long long) plan.selected_experts,
                    (unsigned long long) plan.planned_bytes);
            }
            std::printf("custom cache budget=%.1fMiB GPU_allocated=%.1fMiB reserve=512MiB cap=16384MiB\n",
                        budget / double(MiB), spent / double(MiB));
        }

        std::string text = "The capital of France is Paris. Explain why careful state restoration matters when a model rejects speculative tokens.";
        if (prefill_cpu_share)
            text += " Test a longer prefix so rare expert sharing is exercised before each rollback and output boundary.";
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        const int n = -llama_tokenize(vocab, text.data(), (int) text.size(), nullptr, 0, true, true);
        std::vector<llama_token> tokens(n);
        const int prefix_size = prefill_cpu_share ? 16 : 8;
        const int fixture_size = prefix_size + draft_count + 2;
        if (n < fixture_size || llama_tokenize(vocab, text.data(), (int) text.size(), tokens.data(), n, true, true) != n)
            throw std::runtime_error("rollback test tokenization failed");
        const std::vector<llama_token> prefix(tokens.begin(), tokens.begin() + prefix_size);
        const std::vector<llama_token> window(tokens.begin() + prefix_size, tokens.begin() + prefix_size + draft_count + 1);
        if (trace_difference) {
            trace_batch_difference(ctx.get(), trace, prefix, window);
            print_native_graphs();
            return 0;
        }
        if (cached) {
            const std::vector<llama_token> fixture(tokens.begin(), tokens.begin() + fixture_size);
            auto before = read_backend_stats();
            warm(ctx.get(), fixture);
            auto after = read_backend_stats();
            print_stats("warmup1", before, after);
            before = after;
            warm(ctx.get(), fixture);
            print_stats("warmup2", before, read_backend_stats());
            if (freeze_cache) {
                strata_a3b::set_cache_admission(false);
                std::puts(gpu_only_experts ?
                    "persistent cache admission frozen after warmup; validation must still stage GPU misses, with no CPU expert rows" :
                    "custom cache admission frozen after warmup; validation must have GPU and CPU rows with no uploads");
            }
        }
        const auto stats_before = read_backend_stats();
        const auto checks = strata_a3b::validate_recurrent_rollback(ctx.get(), prefix, window, tokens[fixture_size - 1], prefill_cpu_share);
        const auto stats_after = read_backend_stats();
        if (gpu_computation) print_native_graphs();
        if (cached) print_stats("validation", stats_before, stats_after);
        bool passed = true;
        if (prefill_cpu_share) {
            const auto calls = stats_after.prefill_share_calls - stats_before.prefill_share_calls;
            const auto jobs = stats_after.prefill_share_cpu_jobs - stats_before.prefill_share_cpu_jobs;
            const auto experts = stats_after.prefill_share_cpu_experts - stats_before.prefill_share_cpu_experts;
            std::printf("prefill CPU share validation prefix=%d calls=%llu CPU_jobs=%llu experts=%llu\n",
                        prefix_size, (unsigned long long) calls, (unsigned long long) jobs,
                        (unsigned long long) experts);
            passed &= calls > 0 && jobs > 0 && experts > 0 && jobs <= 2 * experts;
        }
        if (freeze_cache && hybrid_experts) {
            const auto uploaded = stats_after.upload_bytes - stats_before.upload_bytes;
            const auto staged = stats_after.staging_upload_bytes - stats_before.staging_upload_bytes;
            const bool gpu_used = stats_after.gpu_rows > stats_before.gpu_rows;
            const bool cpu_ok = pcie_fraction < 1 || prefill_cpu_share ? stats_after.cpu_rows > stats_before.cpu_rows :
                                                   stats_after.cpu_rows == stats_before.cpu_rows;
            std::printf("frozen hybrid placement GPU_rows_positive=%d CPU_rows_expected=%d staged_uploads_only=%d PCIe_fraction=%.4f\n",
                        (int) gpu_used, (int) cpu_ok, (int) (uploaded == staged), pcie_fraction);
            passed &= gpu_used && cpu_ok && uploaded == staged;
        } else if (freeze_cache && gpu_only_experts) {
            const auto uploaded = stats_after.upload_bytes - stats_before.upload_bytes;
            const auto staged = stats_after.staging_upload_bytes - stats_before.staging_upload_bytes;
            const bool staged_fixed = stats_after.gpu_only_rows > stats_before.gpu_only_rows &&
                stats_after.staged_rows > stats_before.staged_rows && staged > 0 && uploaded == staged &&
                stats_after.cpu_rows == stats_before.cpu_rows;
            std::printf("frozen GPU-only misses GPU_rows_positive=%d staged_rows_positive=%d CPU_rows_zero=%d GPU_only_upload_bytes_accounted=%d\n",
                        (int) (stats_after.gpu_only_rows > stats_before.gpu_only_rows),
                        (int) (stats_after.staged_rows > stats_before.staged_rows),
                        (int) (stats_after.cpu_rows == stats_before.cpu_rows), (int) (staged > 0 && uploaded == staged));
            passed &= staged_fixed;
        } else if (freeze_cache) {
            const bool mixed_fixed = stats_after.gpu_rows > stats_before.gpu_rows &&
                stats_after.cpu_rows > stats_before.cpu_rows && stats_after.uploads == stats_before.uploads;
            std::printf("fixed mixed placement GPU_rows_positive=%d CPU_rows_positive=%d uploads_zero=%d\n",
                        (int) (stats_after.gpu_rows > stats_before.gpu_rows),
                        (int) (stats_after.cpu_rows > stats_before.cpu_rows),
                        (int) (stats_after.uploads == stats_before.uploads));
            passed &= mixed_fixed;
        }
        for (const auto & check : checks) {
            std::printf("rollback keep=%d/%zu drafts_accepted=%d max_abs=%.9g mean_abs=%.9g KL=%.9g top1=%d finite=%d\n",
                        check.kept_rows, window.size(), check.kept_rows - 1, check.max_abs_logit_error,
                        check.mean_abs_logit_error, check.kl_reference_to_rollback,
                        (int) check.top1_agrees, (int) check.finite);
            std::printf("snapshot keep=%d/%zu max_abs=%.9g KL=%.9g top1=%d finite=%d\n",
                        check.kept_rows, window.size(), check.snapshot_max_abs_logit_error,
                        check.snapshot_kl_reference_to_rollback, (int) check.snapshot_top1_agrees, (int) check.snapshot_finite);
            passed &= check.finite && check.snapshot_finite && check.snapshot_top1_agrees &&
                      check.snapshot_max_abs_logit_error < 1e-5;
            if (assert_sequential)
                passed &= check.top1_agrees && check.kl_reference_to_rollback < 1e-4;
        }
        if (!passed) throw std::runtime_error("recurrent rollback or requested sequential-reference assertion failed");
        std::puts("PASS: pending GDN rollback matches materialized accepted-state continuation (max_abs < 1e-5).\n"
                  "The separately reported sequential comparison measures batch arithmetic differences; it is not a parity assertion.");
        if (assert_sequential) std::puts("PASS: independent sequential-prefix continuation top1 agrees and KL < 1e-4.");
        if (!mtp_path.empty()) validate_mtp_class(model.get(), ctx.get(), mp, cp, path, mtp_path, draft_count,
                                                 prefix, mtp_gpu_sampling, mtp_deferred, share_weights, draft_ubatch, mtp_adaptive,
                                                 mtp_fused_rounds, draft_state_directory);
        if (strata_native_experts) {
            const auto calls = read_backend_stats().strata_native_calls - stats_before.strata_native_calls;
            if (!calls) throw std::runtime_error("Strata native expert hook was not executed");
            const char * parallel = std::getenv("STRATA_A3B_PARALLEL_PLAN");
            const bool parallel_on = parallel && std::strcmp(parallel, "1") == 0;
            std::printf("Strata native expert plan actual calls=%llu parallel=%d\n",
                        (unsigned long long) calls, (int) parallel_on);
        }
        if (gpu_only_experts && read_backend_stats().cpu_rows != 0)
            throw std::runtime_error("GPU-only expert validation observed CPU expert computation");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
