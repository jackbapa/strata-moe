#include "speculative.hpp"
#include "draft_vocab_map.hpp"
#include "llama-ext.h"
#include "llama-model.h"
#include "profile_ranges.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace strata_a3b {
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

const char * device_greedy_name(const llama_sampler *) { return "strata-device-greedy-guard"; }
void device_greedy_apply(llama_sampler *, llama_token_data_array * data) {
    if (!data->size) throw std::runtime_error("empty greedy candidate list");
    data->selected = 0;
    for (size_t i = 0; i < data->size; ++i) {
        if (!std::isfinite(data->data[i].logit)) throw std::runtime_error("non-finite greedy logits");
        if (data->data[i].logit > data->data[data->selected].logit) data->selected = (int64_t) i;
    }
}
void device_greedy_backend_apply(llama_sampler *, ggml_context * ctx, ggml_cgraph *, llama_sampler_data * data) {
    auto * logits = ggml_reshape_1d(ctx, data->logits, ggml_nelements(data->logits));
    const int64_t count = ggml_nelements(logits);
    if (count > 16777216) throw std::runtime_error("device greedy vocabulary exceeds exact F32 index range");
    auto * index = ggml_argmax(ctx, logits);
    // GGML CPU/CUDA argmax tie rules differ from llama_sampler_greedy's first
    // maximum. Gather the maximum value, then reduce unique integer scores on
    // exactly-equal maxima so the emitted token preserves the CPU tie rule.
    // Clamp the preliminary index so an all-NaN/-inf row cannot cause an invalid
    // gather before its finite guard is checked on the host.
    index = ggml_cast(ctx, ggml_clamp(ctx, ggml_cast(ctx, index, GGML_TYPE_F32), 0, float(count - 1)), GGML_TYPE_I32);
    auto * maximum = ggml_get_rows(ctx, ggml_reshape_2d(ctx, logits, 1, count), index);
    maximum = ggml_reshape_1d(ctx, maximum, 1);
    auto * below = ggml_step(ctx, ggml_neg(ctx, ggml_sub(ctx, logits, ggml_repeat(ctx, maximum, logits))));
    auto * equal = ggml_scale_bias(ctx, below, -1, 1);
    auto * priority = ggml_scale_bias(ctx, ggml_arange(ctx, 0, float(count), 1), -1, float(count));
    auto * subset_index = ggml_argmax(ctx, ggml_mul(ctx, equal, priority));
    if (data->candidates) {
        if (data->candidates->type != GGML_TYPE_I32 || ggml_nelements(data->candidates) != count)
            throw std::runtime_error("compact draft token map must be I32 and match the head");
        auto * map_rows = ggml_reshape_2d(ctx, data->candidates, 1, count);
        data->sampled = ggml_get_rows(ctx, map_rows, subset_index);
    } else {
        data->sampled = subset_index;
    }
    ggml_set_name(data->sampled, "strata_greedy_token");
    // Finite x - x is exactly zero; an infinity or NaN produces NaN. Preserve
    // the finite-logit guard while downloading one float instead of the vocab.
    data->probs = ggml_sum(ctx, ggml_sub(ctx, logits, logits));
    ggml_set_name(data->probs, "strata_greedy_finite_guard");
    data->logits = nullptr;
    data->candidates = nullptr;
}
bool device_greedy_backend_supports(llama_sampler * sampler, ggml_backend_buffer_type_t buft, bool compact) {
    auto * device = ggml_backend_buft_get_device(buft);
    if (!device) return !compact; // CPU buffer type has no device association.
    auto * ctx = ggml_init({64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false), nullptr, true});
    if (!ctx) return false;
    auto * graph = ggml_new_graph_custom(ctx, 64, false);
    // Probe only the selected path so ordinary MTP keeps its original init.
    // The explicit compact preflight must fail before any K2 decode if the
    // backend lacks the I32 GET_ROWS remap.
    llama_sampler_data data{ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024), nullptr, nullptr,
                            compact ? ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1024) : nullptr};
    device_greedy_backend_apply(sampler, ctx, graph, &data);
    ggml_build_forward_expand(graph, data.sampled);
    ggml_build_forward_expand(graph, data.probs);
    bool supported = true;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
        supported = supported && ggml_backend_dev_supports_op(device, ggml_graph_node(graph, i));
    ggml_free(ctx);
    return supported;
}
bool device_greedy_backend_init(llama_sampler * sampler, ggml_backend_buffer_type_t buft, uint32_t) {
    return device_greedy_backend_supports(sampler, buft, false);
}

// llama_batch_init allocates token OR embedding input; MTP requires both.
// Keep the extra embedding array in a vector and detach it before batch_free.
struct Batch {
    llama_batch value;
    std::vector<float> embeddings;
    Batch(const llama_token * tokens, int count, llama_pos start, int width = 0,
          bool all_logits = false) : value(llama_batch_init(count, 0, 1)) {
        if (!value.token || !value.pos || !value.n_seq_id || !value.seq_id || !value.logits)
            throw std::runtime_error("MTP batch allocation failed");
        value.n_tokens = count;
        for (int i = 0; i < count; ++i) {
            value.token[i] = tokens[i];
            value.pos[i] = start + i;
            value.n_seq_id[i] = 1;
            value.seq_id[i][0] = 0;
            value.logits[i] = all_logits || i == count - 1;
        }
        if (width) {
            embeddings.resize((size_t) count * width);
            value.embd = embeddings.data();
        }
    }
    ~Batch() { value.embd = nullptr; llama_batch_free(value); }
    Batch(const Batch &) = delete;
    Batch & operator=(const Batch &) = delete;
};

void decode(llama_context * ctx, Batch & batch, const char * phase) {
    const int rc = llama_decode(ctx, batch.value);
    if (rc) throw std::runtime_error(std::string("MTP ") + phase + " decode failed: " + std::to_string(rc));
}

void trim(llama_context * ctx, llama_pos first_removed, const char * phase) {
    if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, first_removed, -1))
        throw std::runtime_error(std::string("MTP ") + phase + " rollback failed at " + std::to_string(first_removed));
}

llama_token greedy(const float * logits, int count, float min_probability, bool & confident) {
    if (!logits) throw std::runtime_error("MTP draft logits missing");
    llama_token best = 0;
    for (int i = 0; i < count; ++i) {
        if (!std::isfinite(logits[i])) throw std::runtime_error("MTP draft has non-finite logits");
        if (logits[i] > logits[best]) best = i;
    }
    confident = true;
    if (min_probability > 0) {
        double total = 0;
        for (int i = 0; i < count; ++i) total += std::exp(double(logits[i]) - logits[best]);
        confident = 1.0 / total >= min_probability;
    }
    return best;
}

std::vector<float> logits_copy(llama_context * ctx) {
    const int n = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const float * p = llama_get_logits_ith(ctx, -1);
    if (!p) throw std::runtime_error("rollback diagnostic logits missing");
    return std::vector<float>(p, p + n);
}

void decode_sequence(llama_context * ctx, const std::vector<llama_token> & tokens, llama_pos start) {
    for (size_t i = 0; i < tokens.size(); ++i) {
        Batch b(&tokens[i], 1, start + (llama_pos) i);
        decode(ctx, b, "rollback reference");
    }
}

RollbackCheck compare(const std::vector<float> & reference, const std::vector<float> & actual, int kept) {
    RollbackCheck result;
    result.kept_rows = kept;
    result.finite = true;
    int arg_ref = 0, arg_actual = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(actual[i])) result.finite = false;
        const double error = std::abs(double(reference[i]) - actual[i]);
        result.max_abs_logit_error = std::max(result.max_abs_logit_error, error);
        result.mean_abs_logit_error += error;
        if (reference[i] > reference[arg_ref]) arg_ref = (int) i;
        if (actual[i] > actual[arg_actual]) arg_actual = (int) i;
    }
    result.mean_abs_logit_error /= reference.size();
    result.top1_agrees = arg_ref == arg_actual;
    if (!result.finite) {
        result.max_abs_logit_error = result.mean_abs_logit_error =
            result.kl_reference_to_rollback = std::numeric_limits<double>::infinity();
        return result;
    }
    double ref_z = 0, actual_z = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        ref_z += std::exp(double(reference[i]) - reference[arg_ref]);
        actual_z += std::exp(double(actual[i]) - actual[arg_actual]);
    }
    const double ref_log_z = reference[arg_ref] + std::log(ref_z);
    const double actual_log_z = actual[arg_actual] + std::log(actual_z);
    for (size_t i = 0; i < reference.size(); ++i) {
        const double lp = reference[i] - ref_log_z;
        const double lq = actual[i] - actual_log_z;
        result.kl_reference_to_rollback += std::exp(lp) * (lp - lq);
    }
    // An identical distribution can sum to a tiny negative value from rounding.
    result.kl_reference_to_rollback = std::max(0.0, result.kl_reference_to_rollback);
    return result;
}
} // namespace

llama_sampler * create_device_greedy_sampler() {
    static llama_sampler_i sampler_iface{
        device_greedy_name, nullptr, device_greedy_apply, nullptr, nullptr, nullptr,
        device_greedy_backend_init, nullptr, device_greedy_backend_apply, nullptr, nullptr, nullptr,
    };
    auto * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init(&sampler_iface, nullptr));
    return chain;
}

llama_token device_greedy_token(llama_context * ctx, int row) {
    const llama_token token = llama_get_sampled_token_ith(ctx, row);
    const float * guard = llama_get_sampled_probs_ith(ctx, row);
    if (token == LLAMA_TOKEN_NULL || !guard)
        throw std::runtime_error("device greedy sampler did not produce a token/finite guard");
    if (!std::isfinite(guard[0])) throw std::runtime_error("device greedy logits are non-finite");
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    if (token < 0 || token >= n_vocab) throw std::runtime_error("device greedy sampler produced an invalid token");
    return token;
}

bool detail::can_fuse_mtp_round(int accepted, uint32_t batch, uint32_t ubatch) {
    return accepted >= 1 && accepted < 8 && (uint32_t) accepted + 1 <= batch &&
           (uint32_t) accepted + 1 <= ubatch;
}

detail::MtpRoundInputs detail::make_mtp_round_inputs(const std::vector<llama_token> & observed_tokens,
    const std::vector<float> & observed_h, const std::vector<float> & pending_h,
    int accepted, llama_pos start, llama_token last, llama_pos next_pos) {
    if (accepted < 1 || accepted >= 8 || start < 0 || last < 0 || pending_h.empty() ||
        (int64_t) start + accepted != next_pos || observed_tokens.size() < (size_t) accepted ||
        observed_h.size() / pending_h.size() < (size_t) accepted)
        throw std::runtime_error("invalid fused MTP accepted-prefix inputs");
    MtpRoundInputs result;
    result.start = start;
    result.tokens.assign(observed_tokens.begin(), observed_tokens.begin() + accepted);
    result.tokens.push_back(last);
    result.hidden = pending_h;
    result.hidden.insert(result.hidden.end(), observed_h.begin(),
                         observed_h.begin() + (size_t) accepted * pending_h.size());
    return result;
}

MtpDraft::MtpDraft(llama_context * target, llama_context * draft, int max_draft, float min_probability,
                   bool gpu_sampling, bool deferred_catchup, bool adaptive, bool fused_rounds)
    : target_(target), draft_(draft), max_draft_(max_draft), min_probability_(min_probability),
      gpu_sampling_(gpu_sampling), deferred_catchup_(deferred_catchup), fused_rounds_(fused_rounds),
      adaptive_(adaptive), policy_(max_draft) {
    if (!target || !draft || target == draft || max_draft < 1 || max_draft > 16 ||
        !std::isfinite(min_probability) || min_probability < 0 || min_probability > 1)
        throw std::runtime_error("invalid MTP contexts or options");
    const llama_model * tm = llama_get_model(target), * dm = llama_get_model(draft);
    width_ = llama_model_n_embd_out(tm);
    if (llama_model_n_layer_nextn(dm) != 1 || width_ != llama_model_n_embd_out(dm) ||
        llama_vocab_n_tokens(llama_model_get_vocab(tm)) != llama_vocab_n_tokens(llama_model_get_vocab(dm)))
        throw std::runtime_error("MTP requires a matching single-head Qwen NextN model");
    if (dm->d2t && dm->d2t->type == GGML_TYPE_I32) {
        const auto * vocab = llama_model_get_vocab(dm);
        const int full_vocab = llama_vocab_n_tokens(vocab);
        if (ggml_n_dims(dm->d2t) != 1 || dm->d2t->ne[0] <= 0 ||
            dm->d2t->ne[0] >= full_vocab)
            throw std::runtime_error("invalid compact draft map tensor shape");
        std::vector<int32_t> ids((size_t) dm->d2t->ne[0]);
        ggml_backend_tensor_get(dm->d2t, ids.data(), 0, ids.size() * sizeof(int32_t));
        std::vector<int32_t> required;
        for (int id = 0; id < full_vocab; ++id) {
            if (!(llama_vocab_get_attr(vocab, id) & LLAMA_TOKEN_ATTR_NORMAL)) required.push_back(id);
            if (llama_vocab_is_eog(vocab, id)) required.push_back(id);
        }
        required.push_back(llama_vocab_bos(vocab));
        detail::validate_draft_vocab_ids(ids, full_vocab, required);
    }
    if (llama_n_rs_seq(target) < (uint32_t) max_draft ||
        llama_n_ubatch(target) < (uint32_t) max_draft + 1 ||
        llama_n_batch(target) < (uint32_t) max_draft + 1 ||
        llama_n_seq_max(target) != 1 || llama_n_seq_max(draft) != 1)
        throw std::runtime_error("MTP requires one sequence, n_rs_seq >= drafts and batch/ubatch >= drafts + 1");
    if (gpu_sampling_ && min_probability_ != 0)
        throw std::runtime_error("device draft sampling currently requires min_probability=0");
    if (fused_rounds_ && (!gpu_sampling_ || !deferred_catchup_ || max_draft > 7))
        throw std::runtime_error("fused MTP rounds require GPU sampling, deferred catch-up and at most seven drafts");
    if (gpu_sampling_) {
        draft_sampler_ = create_device_greedy_sampler();
        const char * compact = std::getenv("STRATA_A3B_DRAFT_COMPACT_GPU");
        if (compact && std::strcmp(compact, "1") == 0 && dm->d2t && dm->d2t->type == GGML_TYPE_I32) {
            auto * output_device = dm->dev_output();
            if (!output_device || ggml_backend_dev_type(output_device) != GGML_BACKEND_DEVICE_TYPE_GPU ||
                !dm->d2t->buffer ||
                ggml_backend_buft_get_device(ggml_backend_buffer_get_type(dm->d2t->buffer)) != output_device) {
                llama_sampler_free(draft_sampler_); draft_sampler_ = nullptr;
                throw std::runtime_error("compact draft I32 map is not resident on the output GPU");
            }
            auto * child = llama_sampler_chain_get(draft_sampler_, 0);
            auto * buft = ggml_backend_dev_buffer_type(output_device);
            if (!child || !buft || !device_greedy_backend_supports(child, buft, true)) {
                llama_sampler_free(draft_sampler_); draft_sampler_ = nullptr;
                throw std::runtime_error("compact draft GPU sampler does not support I32 map gather");
            }
        }
        if (!llama_set_sampler(draft_, 0, draft_sampler_)) {
            llama_sampler_free(draft_sampler_); draft_sampler_ = nullptr;
            throw std::runtime_error("draft device sampler could not be attached");
        }
    }
    llama_set_embeddings_nextn(target_, true, false);
    // graph_mtp retains all h_nextn rows before selecting its vocabulary head.
    // A batch whose only output is its last row must read that raw hidden row;
    // masked extraction would copy row zero and corrupt the second proposal.
    llama_set_embeddings_nextn(draft_, true, !fused_rounds_);
    const char * device_prefill = std::getenv("STRATA_A3B_MTP_DEVICE_PREFILL");
    device_prefill_enabled_ = device_prefill && std::strcmp(device_prefill, "1") == 0;
    const char * device_chain_k2 = std::getenv("STRATA_A3B_MTP_DEVICE_CHAIN_K2");
    device_chain_k2_enabled_ = device_chain_k2 && std::strcmp(device_chain_k2, "1") == 0;
    const char * proposal_trace = std::getenv("STRATA_A3B_MTP_PROPOSAL_TRACE");
    proposal_trace_enabled_ = proposal_trace && std::strcmp(proposal_trace, "1") == 0;
    if (device_chain_k2_enabled_ && (!gpu_sampling_ || fused_rounds_ || max_draft_ != 2))
        throw std::runtime_error("MTP device K2 chain requires GPU greedy sampling, two drafts and no fused rounds");
    pending_h_.resize(width_);
    reset();
}

MtpDraft::~MtpDraft() {
    if (draft_sampler_) {
        llama_set_sampler(draft_, 0, nullptr);
        llama_sampler_free(draft_sampler_);
    }
}

void MtpDraft::trace_proposal(llama_pos pos, llama_token input,
                              const std::vector<llama_token> & offered) const {
    if (!proposal_trace_enabled_) return;
    // One complete line per window allows exact K2/on-versus-off comparison.
    // An output hash alone cannot reveal a changed draft corrected by verify.
    std::string line = "A3B MTP proposal trace: window " + std::to_string(stats_.windows) +
        " pos " + std::to_string(pos) + " input " + std::to_string(input) +
        " offered " + std::to_string(offered.size());
    for (const auto token : offered) line += " " + std::to_string(token);
    std::fprintf(stderr, "%s\n", line.c_str());
}

void MtpDraft::reset() {
    compact_gpu_sample_rows_baseline_ = llama_mtp_compact_gpu_sample_rows(draft_);
    llama_mtp_device_prefill_reset(draft_);
    llama_mtp_device_prefill_host_arm(target_, 0, 0);
    llama_memory_clear(llama_get_memory(draft_), true);
    std::fill(pending_h_.begin(), pending_h_.end(), 0.0f);
    observed_h_.clear();
    observed_tokens_.clear();
    observed_deferred_ = false;
    device_prefill_pending_ = false;
    device_prefill_legacy_ = false;
    device_prefill_last_rows_ = 0;
    observed_start_ = 0;
    observed_count_ = last_offered_ = accepted_pending_ = 0;
    stats_ = {};
    if (adaptive_) { policy_.reset(); adaptive_window_pending_ = false; adaptive_choice_ = 0; }
    if (draft_sampler_) llama_sampler_reset(draft_sampler_);
}

uint64_t MtpDraft::compact_gpu_sample_rows() const {
    return llama_mtp_compact_gpu_sample_rows(draft_) - compact_gpu_sample_rows_baseline_;
}

void MtpDraft::arm_prefill_host_copy(int count) {
    if (count < 1) throw std::runtime_error("invalid MTP prompt block size");
    if (device_prefill_enabled_ && !device_prefill_legacy_)
        llama_mtp_device_prefill_host_arm(target_, (uint32_t) count, (uint32_t) width_);
}

void MtpDraft::disarm_prefill_host_copy() {
    llama_mtp_device_prefill_host_disarm(target_);
}

void MtpDraft::observe(const llama_token * tokens, int count, llama_pos start, bool verification) {
    profile::Range observe_range(verification ? "a3b/mtp/verify-observe" : "a3b/mtp/prefill-observe");
    if (!tokens || count < 1 || start < 0 || count > (int) llama_n_batch(draft_) ||
        count > (int) llama_n_ubatch(target_))
        throw std::runtime_error("invalid MTP observed block");
    if (fused_rounds_ && (int64_t) start + count > std::numeric_limits<llama_pos>::max())
        throw std::runtime_error("MTP observed positions exceed their representable range");
    if (fused_rounds_ && observed_deferred_)
        throw std::runtime_error("MTP observed verify prefix must be committed before replacing it");
    if (accepted_pending_) {
        if ((int64_t) observed_start_ + accepted_pending_ != start)
            throw std::runtime_error("MTP observed block does not follow its pending accepted prefix");
        finish();
    }
    if (!verification && device_prefill_enabled_ && !device_prefill_legacy_) {
        const auto time = Clock::now();
        Batch batch(tokens, count, start);
        // llama.cpp still requires a non-null embd pointer to build the MTP
        // input graph. The bridge never reads it; keep one reusable host
        // allocation instead of clearing a whole hidden block each call.
        device_prefill_dummy_.resize(std::max(device_prefill_dummy_.size(), (size_t) count * width_));
        batch.value.embd = device_prefill_dummy_.data();
        std::fill(batch.value.logits, batch.value.logits + count, 0);
        int rc;
        {
            profile::Range bridge_range("a3b/mtp/prefill-device-bridge");
            rc = llama_mtp_device_prefill_decode(target_, draft_, batch.value, (uint32_t) width_);
        }
        if (rc == 1) {
            device_prefill_pending_ = true;
            device_prefill_last_rows_ = count;
            ++stats_.device_prefill_calls;
            stats_.device_prefill_rows += count;
            stats_.device_prefill_avoided_d2h_bytes +=
                llama_mtp_device_prefill_host_saved_bytes(target_);
            stats_.catchup_ms += milliseconds(time);
            return;
        }
        if (rc == 2) {
            // The draft has already consumed this block through the safe
            // host-input fallback. Preserve its final hidden row for the
            // next block and never run the catch-up a second time.
            const float * last_hidden = llama_get_embeddings_nextn_ith(target_, count - 1);
            if (!last_hidden) throw std::runtime_error("MTP prefill fallback final hidden row missing");
            std::copy(last_hidden, last_hidden + width_, pending_h_.begin());
            device_prefill_pending_ = false;
            device_prefill_legacy_ = true;
            stats_.catchup_ms += milliseconds(time);
            return;
        }
        if (rc < 0 || device_prefill_pending_)
            throw std::runtime_error("MTP CUDA prefill bridge failed after draft state advanced");
        if (llama_mtp_device_prefill_host_restore(target_) < 0)
            throw std::runtime_error("MTP CUDA prefill cannot restore target hidden rows for host fallback");
        // No compatible CUDA source/destination: preserve the original host
        // hidden path before any bridged block has advanced draft KV.
        device_prefill_legacy_ = true;
    }
    const auto time = Clock::now();
    const float * hidden = llama_get_embeddings_nextn(target_);
    if (!hidden) throw std::runtime_error("target nextn hidden rows missing");
    observed_h_.assign(hidden, hidden + (size_t) count * width_);
    observed_start_ = start;
    observed_count_ = count;
    observed_tokens_.assign(tokens, tokens + count);
    observed_deferred_ = verification && deferred_catchup_;
    stats_.hidden_ms += milliseconds(time);
    if (!observed_deferred_) process_observed(count);
}

void MtpDraft::process_observed(int count) {
    profile::Range catchup_range("a3b/mtp/catchup");
    if (count < 1 || observed_tokens_.size() < (size_t) count ||
        observed_h_.size() / (size_t) width_ < (size_t) count)
        throw std::runtime_error("invalid MTP catch-up prefix");
    const auto time = Clock::now();
    Batch batch(observed_tokens_.data(), count, observed_start_, width_);
    std::copy(pending_h_.begin(), pending_h_.end(), batch.embeddings.begin());
    if (count > 1)
        std::copy(observed_h_.begin(), observed_h_.begin() + (size_t) (count - 1) * width_, batch.embeddings.begin() + width_);
    // The catch-up only populates KV; it never needs the vocabulary projection.
    std::fill(batch.value.logits, batch.value.logits + count, 0);
    {
        profile::Range decode_range("a3b/mtp/catchup/draft-decode");
        decode(draft_, batch, "catch-up");
    }
    std::copy(observed_h_.begin() + (size_t) (count - 1) * width_,
              observed_h_.begin() + (size_t) count * width_, pending_h_.begin());
    stats_.catchup_ms += milliseconds(time);
}

void MtpDraft::finish() {
    if (fused_rounds_ && observed_deferred_)
        throw std::runtime_error("cannot finish an uncommitted MTP verification window");
    if (!accepted_pending_) return;
    const int count = accepted_pending_;
    process_observed(count);
    trim(draft_, observed_start_ + count, "pending catch-up");
    accepted_pending_ = 0;
}

std::vector<llama_token> MtpDraft::propose(llama_token last, llama_pos pos, int limit) {
    profile::Range propose_range("a3b/mtp/propose");
    if (pos < 0 || limit < 0) throw std::runtime_error("invalid MTP draft position or limit");
    if (observed_deferred_) throw std::runtime_error("MTP verify prefix must be committed before drafting");
    if (accepted_pending_ && (int64_t) observed_start_ + accepted_pending_ != pos)
        throw std::runtime_error("MTP proposal position does not follow its pending accepted prefix");
    if (device_prefill_pending_) {
        // The first proposal still uses the legacy host-packed one-token
        // graph. Fetch only the last row once at the prefill/decode boundary.
        const auto time = Clock::now();
        profile::Range last_hidden_range("a3b/mtp/prefill-last-hidden");
        const float * last_hidden = llama_get_embeddings_nextn_ith(target_, device_prefill_last_rows_ - 1);
        if (!last_hidden) throw std::runtime_error("MTP CUDA prefill final hidden row missing");
        std::copy(last_hidden, last_hidden + width_, pending_h_.begin());
        stats_.hidden_ms += milliseconds(time);
        device_prefill_pending_ = false;
    }
    int count = std::min(max_draft_, limit);
    if (fused_rounds_ && (int64_t) pos + count > std::numeric_limits<llama_pos>::max())
        throw std::runtime_error("MTP proposal positions exceed their representable range");
    if (adaptive_) {
        if (adaptive_window_pending_) throw std::runtime_error("adaptive draft window was not committed");
        window_started_ = Clock::now();
        count = policy_.choose(count);
        adaptive_choice_ = count;
        adaptive_window_pending_ = true;
    }
    std::vector<llama_token> result;
    result.reserve(count);
    last_offered_ = 0;
    if (!count) {
        if (fused_rounds_) finish();
        if (adaptive_) ++stats_.windows;
        return result;
    }
    const auto legacy_time = Clock::now();
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(draft_));
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (last < 0 || last >= n_vocab) throw std::runtime_error("invalid MTP last token");
    const llama_token input_token = last;
    if (!fused_rounds_) {
        // Preserve the existing proposal/synchronization path when the new
        // option is disabled, including masked hidden extraction and timing.
        if (device_chain_k2_enabled_ && count == 2) {
            if ((int64_t) pos + 1 > std::numeric_limits<llama_pos>::max())
                throw std::runtime_error("MTP device K2 position exceeds representable range");
            Batch first(&last, 1, pos, width_);
            Batch second(&last, 1, pos + 1, width_);
            std::copy(pending_h_.begin(), pending_h_.end(), first.embeddings.begin());
            llama_token sampled[2] = {LLAMA_TOKEN_NULL, LLAMA_TOKEN_NULL};
            float finite_guards[2] = {NAN, NAN};
            uint32_t inter_step_syncs = 0;
            int rc = 0;
            {
                profile::Range chain_range("a3b/mtp/proposal/device-chain-k2");
                rc = llama_mtp_device_chain_k2(draft_, first.value, second.value,
                                               (uint32_t) width_, sampled, finite_guards,
                                               &inter_step_syncs);
            }
            if (rc < 0) throw std::runtime_error("MTP device K2 chain failed after draft state advanced");
            if (rc == 1) {
                ++stats_.device_chain_k2_calls;
                // Output -> stable buffer, then stable buffer -> second graph input.
                stats_.device_chain_k2_d2d_bytes += 2 * (sizeof(llama_token) + (uint64_t) width_ * sizeof(float));
                stats_.device_chain_k2_inter_step_syncs += inter_step_syncs;
                stats_.gpu_draft_samples += 2;
                stats_.draft_sampling_download_bytes += 2 * (sizeof(llama_token) + sizeof(float));
                for (int step = 0; step < 2; ++step) {
                    if (!std::isfinite(finite_guards[step]) || sampled[step] < 0 || sampled[step] >= n_vocab)
                        throw std::runtime_error("MTP device K2 chain produced invalid sample or logits");
                    result.push_back(sampled[step]);
                    if (llama_vocab_is_eog(vocab, sampled[step])) break;
                }
                trim(draft_, pos, "device K2 proposal");
                last_offered_ = (int) result.size();
                ++stats_.windows;
                stats_.offered += result.size();
                stats_.draft_ms += milliseconds(legacy_time);
                trace_proposal(pos, input_token, result);
                return result;
            }
            ++stats_.device_chain_k2_fallbacks;
        }
        std::vector<float> hidden = pending_h_;
        for (int step = 0; step < count; ++step) {
            Batch batch(&last, 1, pos + step, width_);
            std::copy(hidden.begin(), hidden.end(), batch.embeddings.begin());
            {
                profile::Range decode_range("a3b/mtp/proposal/draft-decode");
                decode(draft_, batch, "proposal");
            }
            bool confident = true;
            const llama_token token = [&] {
                profile::Range sample_range("a3b/mtp/proposal/draft-sample");
                return gpu_sampling_ ? device_greedy_token(draft_) :
                    greedy(llama_get_logits_ith(draft_, -1), n_vocab, min_probability_, confident);
            }();
            stats_.draft_sampling_download_bytes += gpu_sampling_ ? 8 : (uint64_t) n_vocab * sizeof(float);
            if (gpu_sampling_) ++stats_.gpu_draft_samples;
            if (!confident) break;
            result.push_back(token);
            if (llama_vocab_is_eog(vocab, token)) break;
            if (step + 1 < count) {
                profile::Range hidden_range("a3b/mtp/proposal/draft-hidden");
                const float * h = llama_get_embeddings_nextn_ith(draft_, -1);
                if (!h) throw std::runtime_error("MTP proposal hidden row missing");
                std::copy(h, h + width_, hidden.begin());
                last = token;
            }
        }
        trim(draft_, pos, "proposal");
        last_offered_ = (int) result.size();
        ++stats_.windows;
        stats_.offered += result.size();
        stats_.draft_ms += milliseconds(legacy_time);
        trace_proposal(pos, input_token, result);
        return result;
    }
    const bool fuse_first = accepted_pending_ &&
        detail::can_fuse_mtp_round(accepted_pending_, llama_n_batch(draft_), llama_n_ubatch(draft_));
    if (accepted_pending_ && !fuse_first) finish();
    const auto time = Clock::now();
    double fused_ms = 0;
    std::vector<float> hidden = pending_h_;
    for (int step = 0; step < count; ++step) {
        const bool fused = step == 0 && fuse_first;
        const auto fused_start = fused ? Clock::now() : Clock::time_point{};
        int hidden_row = 0;
        if (fused) {
            const int accepted = accepted_pending_;
            auto inputs = detail::make_mtp_round_inputs(observed_tokens_, observed_h_, pending_h_,
                                                        accepted, observed_start_, last, pos);
            Batch batch(inputs.tokens.data(), (int) inputs.tokens.size(), inputs.start);
            batch.embeddings = std::move(inputs.hidden);
            batch.value.embd = batch.embeddings.data();
            {
                profile::Range decode_range("a3b/mtp/fused-catchup-proposal/draft-decode");
                decode(draft_, batch, "fused catch-up/proposal");
            }
            std::copy(observed_h_.begin() + (size_t) (accepted - 1) * width_,
                      observed_h_.begin() + (size_t) accepted * width_, pending_h_.begin());
            hidden_row = accepted;
            accepted_pending_ = 0;
            ++stats_.fused_rounds;
            stats_.fused_catchup_rows += accepted;
        } else {
            Batch batch(&last, 1, pos + step, width_);
            std::copy(hidden.begin(), hidden.end(), batch.embeddings.begin());
            {
                profile::Range decode_range("a3b/mtp/proposal/draft-decode");
                decode(draft_, batch, "proposal");
            }
        }
        bool confident = true;
        const llama_token token = [&] {
            profile::Range sample_range("a3b/mtp/proposal/draft-sample");
            return gpu_sampling_ ? device_greedy_token(draft_) :
                greedy(llama_get_logits_ith(draft_, -1), n_vocab, min_probability_, confident);
        }();
        stats_.draft_sampling_download_bytes += gpu_sampling_ ? 8 : (uint64_t) n_vocab * sizeof(float);
        if (gpu_sampling_) ++stats_.gpu_draft_samples;
        if (confident) result.push_back(token);
        const bool stop = !confident || llama_vocab_is_eog(vocab, token);
        if (!stop && step + 1 < count) {
            profile::Range hidden_range("a3b/mtp/proposal/draft-hidden");
            const float * h = llama_get_embeddings_nextn_ith(draft_, fused_rounds_ ? hidden_row : -1);
            if (!h) throw std::runtime_error("MTP proposal hidden row missing");
            std::copy(h, h + width_, hidden.begin());
            last = token;
        }
        if (fused) {
            fused_ms = milliseconds(fused_start);
            stats_.fused_round_ms += fused_ms;
        }
        if (stop) break;
    }
    trim(draft_, pos, "proposal");
    last_offered_ = (int) result.size();
    ++stats_.windows;
    stats_.offered += result.size();
    stats_.draft_ms += milliseconds(time) - fused_ms;
    trace_proposal(pos, input_token, result);
    return result;
}

void MtpDraft::commit(int accepted_rows, int matched_drafts) {
    profile::Range commit_range("a3b/mtp/commit");
    if (accepted_rows < 1 || accepted_rows > observed_count_)
        throw std::runtime_error("invalid MTP accepted prefix");
    if (matched_drafts < -1 || matched_drafts > std::min(last_offered_, accepted_rows))
        throw std::runtime_error("invalid MTP matched draft count");
    if (adaptive_ && (!adaptive_window_pending_ || matched_drafts < 0))
        throw std::runtime_error("adaptive MTP commit requires a proposal window and explicit output match count");
    const llama_pos end = observed_start_ + accepted_rows;
    trim(target_, end, "target acceptance");
    const bool defer_round = observed_deferred_ && fused_rounds_;
    if (defer_round) accepted_pending_ = accepted_rows;
    else if (observed_deferred_) process_observed(accepted_rows);
    trim(draft_, end, "draft acceptance");
    if (!defer_round) {
        const auto first = observed_h_.begin() + (size_t) (accepted_rows - 1) * width_;
        std::copy(first, first + width_, pending_h_.begin());
    }
    stats_.accepted += matched_drafts < 0 ? std::min(last_offered_, accepted_rows - 1) : matched_drafts;
    if (adaptive_) {
        const double round_ms = milliseconds(window_started_);
        // Keep the policy's actual propose-to-commit wall-cost convention. In
        // fused mode this includes the preceding accepted prefix when it is
        // materialized now, rather than assigning the same work twice.
        policy_.observe(last_offered_, matched_drafts, accepted_rows, round_ms, adaptive_choice_);
        ++stats_.adaptive_windows_by_k[last_offered_];
        stats_.adaptive_round_ms_by_k[last_offered_] += round_ms;
        adaptive_window_pending_ = false;
    }
    last_offered_ = 0;
    observed_count_ = 0;
    observed_deferred_ = false;
}

std::vector<RollbackCheck> validate_recurrent_rollback(
    llama_context * target, const std::vector<llama_token> & prefix,
    const std::vector<llama_token> & window, llama_token continuation, bool batch_prefix) {
    if (!target || prefix.empty() || window.size() < 2 ||
        window.size() > llama_n_ubatch(target) || window.size() > llama_n_batch(target) ||
        window.size() - 1 > llama_n_rs_seq(target))
        throw std::runtime_error("rollback diagnostic requires a prefix and a bounded verify window");
    if (batch_prefix && (prefix.size() > llama_n_ubatch(target) || prefix.size() > llama_n_batch(target)))
        throw std::runtime_error("rollback diagnostic batched prefix exceeds context capacity");
    auto restore_prefix = [&] {
        if (batch_prefix) {
            Batch batch(prefix.data(), (int) prefix.size(), 0);
            decode(target, batch, "rollback batched prefix");
        } else decode_sequence(target, prefix, 0);
    };
    std::vector<RollbackCheck> results;
    for (size_t keep = 1; keep <= window.size(); ++keep) {
        llama_memory_clear(llama_get_memory(target), true);
        restore_prefix();
        Batch batch(window.data(), (int) window.size(), (llama_pos) prefix.size(), 0, true);
        decode(target, batch, "rollback window");
        trim(target, (llama_pos) (prefix.size() + keep), "rollback diagnostic");
        const size_t state_size = llama_state_seq_get_size(target, 0);
        std::vector<uint8_t> state(state_size);
        if (!state_size || llama_state_seq_get_data(target, state.data(), state.size(), 0) != state_size)
            throw std::runtime_error("rollback diagnostic could not serialize accepted state");
        Batch tail(&continuation, 1, (llama_pos) (prefix.size() + keep));
        decode(target, tail, "rollback continuation");
        const auto actual = logits_copy(target);

        llama_memory_clear(llama_get_memory(target), true);
        if (llama_state_seq_set_data(target, state.data(), state.size(), 0) != state_size)
            throw std::runtime_error("rollback diagnostic could not materialize accepted state");
        Batch restored_tail(&continuation, 1, (llama_pos) (prefix.size() + keep));
        decode(target, restored_tail, "snapshot continuation");
        const auto snapshot = compare(logits_copy(target), actual, (int) keep);

        llama_memory_clear(llama_get_memory(target), true);
        restore_prefix();
        decode_sequence(target, std::vector<llama_token>(window.begin(), window.begin() + keep), (llama_pos) prefix.size());
        Batch ref_tail(&continuation, 1, (llama_pos) (prefix.size() + keep));
        decode(target, ref_tail, "rollback reference continuation");
        auto result = compare(logits_copy(target), actual, (int) keep);
        result.snapshot_max_abs_logit_error = snapshot.max_abs_logit_error;
        result.snapshot_kl_reference_to_rollback = snapshot.kl_reference_to_rollback;
        result.snapshot_top1_agrees = snapshot.top1_agrees;
        result.snapshot_finite = snapshot.finite;
        results.push_back(result);
    }
    llama_memory_clear(llama_get_memory(target), true);
    return results;
}

} // namespace strata_a3b
