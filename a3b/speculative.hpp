#pragma once

#include "llama.h"
#include "draft_policy.hpp"

#include <chrono>
#include <cstdint>
#include <vector>

namespace strata_a3b {

struct MtpStats {
    uint64_t windows = 0;
    uint64_t offered = 0;
    uint64_t accepted = 0;
    double draft_ms = 0;
    double catchup_ms = 0;
    double hidden_ms = 0;
    // Opt-in fused accepted-prefix catch-up + first proposal. This wall time
    // includes its first sample/hidden wait and is excluded from draft_ms and
    // catchup_ms, so the three phase totals do not double count work.
    uint64_t fused_rounds = 0, fused_catchup_rows = 0;
    double fused_round_ms = 0;
    uint64_t gpu_draft_samples = 0;
    uint64_t draft_sampling_download_bytes = 0;
    // Prompt-only CUDA hidden bridge. These are direct device-to-device draft
    // catch-up calls/rows; they do not imply MTP round graph capture.
    uint64_t device_prefill_calls = 0, device_prefill_rows = 0;
    uint64_t device_prefill_avoided_d2h_bytes = 0;
    // Two successive draft decodes connected through stable CUDA token/hidden
    // buffers. This is not a captured cross-step CUDA graph.
    uint64_t device_chain_k2_calls = 0, device_chain_k2_fallbacks = 0;
    uint64_t device_chain_k2_d2d_bytes = 0, device_chain_k2_inter_step_syncs = 0;
    // Adaptive-only accounting by actual offered proposal count. Includes
    // terminal rounds for reporting; the policy excludes their censored cost.
    std::array<uint64_t, AdaptiveDraftPolicy::capacity + 1> adaptive_windows_by_k{};
    std::array<double, AdaptiveDraftPolicy::capacity + 1> adaptive_round_ms_by_k{};
};

// A sampler chain that emits only GPU argmax and a scalar finite-logit guard.
// Unlike the built-in greedy backend sampler, it does not retain full logits
// as a backend output. Caller owns the chain and attaches it once with
// llama_set_sampler (or context_params.samplers). Neutral greedy only: using it
// in place of a temperature/penalty chain would change the target distribution.
llama_sampler * create_device_greedy_sampler();
// Reads the backend token and finite guard, throwing if unavailable/non-finite.
// This does not accept into sampler history; the caller handles that separately.
llama_token device_greedy_token(llama_context * ctx, int row = -1);

namespace detail {
// Pure input packing shared by production and the small CPU boundary test.
bool can_fuse_mtp_round(int accepted, uint32_t batch, uint32_t ubatch);
struct MtpRoundInputs {
    llama_pos start = 0;
    std::vector<llama_token> tokens;
    std::vector<float> hidden;
};
MtpRoundInputs make_mtp_round_inputs(const std::vector<llama_token> & observed_tokens,
    const std::vector<float> & observed_h, const std::vector<float> & pending_h,
    int accepted, llama_pos start, llama_token last, llama_pos next_pos);
} // namespace detail

// Single sequence, independent Qwen3.5/3.6 NextN context. The caller owns both
// contexts and loads the draft model with load_mtp=true and ctx_type=MTP.
// The target needs n_rs_seq >= max_draft, n_batch/n_ubatch >= max_draft + 1.
// Draft tokens are proposals only: sample every emitted token from the target
// logits, accepting a proposal only when that sampled token matches it. This
// also preserves the target sampling distribution at temperature > 0.
class MtpDraft {
public:
    MtpDraft(llama_context * target, llama_context * draft,
             int max_draft = 3, float min_probability = 0,
             bool gpu_sampling = false, bool deferred_catchup = false, bool adaptive = false,
             bool fused_rounds = false);
    ~MtpDraft();
    MtpDraft(const MtpDraft &) = delete;
    MtpDraft & operator=(const MtpDraft &) = delete;

    // Call at the start of every independent request. The target is cleared by
    // the caller; this resets the draft context and the cross-block hidden row.
    void reset();

    // Scope the opt-in NextN host-copy deferral around a target prompt decode.
    // The caller must disarm even if llama_decode fails or throws; observe()
    // then either bridges directly or restores the full host block on fallback.
    void arm_prefill_host_copy(int count);
    void disarm_prefill_host_copy();

    // After the target decodes a continuous block at [start, start + count),
    // pair each token with the target hidden row immediately BEFORE that token
    // and advance the MTP KV. Call after every prefill block as well as verify.
    // The target's nextn output must still be from this exact decode call.
    // verification=true permits the opt-in deferred catch-up path to wait for
    // commit(), then teacher-force only accepted rows. With fused_rounds they
    // may remain pending until the next propose()/finish(). Prefill stays immediate.
    void observe(const llama_token * tokens, int count, llama_pos start, bool verification = false);

    // Prefix [0, pos) has been observed; last is the unprocessed token at pos.
    // Returns at most min(limit,max_draft) greedy proposals AFTER last. Draft KV
    // is restored to the prefix before return, ready for observe(verify batch).
    std::vector<llama_token> propose(llama_token last, llama_pos pos, int limit);

    // Keep this many INPUT rows from the most recent observe(), including row 0
    // (the last token). The final sampled correction/bonus token is not an input
    // row yet. Trims BOTH target and draft memory and selects the accepted h-row.
    // Failure throws; continuing after a failed recurrent rollback is unsafe.
    // matched_drafts counts matching OUTPUT proposals, including a terminal EOS.
    // The default preserves the legacy input-prefix-based diagnostic count.
    // Adaptive mode requires an explicit count to distinguish censored EOS and
    // output limits from tested rejections; it does not sample target tokens.
    void commit(int accepted_rows, int matched_drafts = -1);

    // Materialize any accepted draft KV left for the next fused round. Call at
    // EOS/output-limit/request end; it computes no proposal or vocabulary head.
    // reset() can instead discard the previous request's pending draft state.
    void finish();

    int max_draft() const { return max_draft_; }
    bool device_prefill_enabled() const { return device_prefill_enabled_; }
    bool device_chain_k2_enabled() const { return device_chain_k2_enabled_; }
    const MtpStats & stats() const { return stats_; }
    // Counts completed-request graph output rows that actually used the
    // compact I32 GPU map, excluding any scatter fallback.
    uint64_t compact_gpu_sample_rows() const;
    const AdaptiveDraftPolicy & draft_policy() const { return policy_; }

private:
    llama_context * target_;
    llama_context * draft_;
    int width_ = 0;
    int max_draft_ = 0;
    float min_probability_ = 0;
    bool gpu_sampling_ = false, deferred_catchup_ = false, observed_deferred_ = false;
    bool fused_rounds_ = false;
    bool device_prefill_enabled_ = false, device_prefill_pending_ = false, device_prefill_legacy_ = false;
    bool device_chain_k2_enabled_ = false;
    bool proposal_trace_enabled_ = false;
    int device_prefill_last_rows_ = 0;
    int accepted_pending_ = 0;
    bool adaptive_ = false, adaptive_window_pending_ = false;
    int adaptive_choice_ = 0;
    llama_sampler * draft_sampler_ = nullptr;
    uint64_t compact_gpu_sample_rows_baseline_ = 0;
    std::vector<float> pending_h_;
    std::vector<float> device_prefill_dummy_;
    std::vector<float> observed_h_;
    std::vector<llama_token> observed_tokens_;
    llama_pos observed_start_ = 0;
    int observed_count_ = 0;
    int last_offered_ = 0;
    AdaptiveDraftPolicy policy_;
    std::chrono::steady_clock::time_point window_started_{};
    MtpStats stats_;
    void process_observed(int count);
    void trace_proposal(llama_pos pos, llama_token input,
                        const std::vector<llama_token> & offered) const;
};

struct RollbackCheck {
    int kept_rows = 0;
    double max_abs_logit_error = 0;
    double mean_abs_logit_error = 0;
    double kl_reference_to_rollback = 0;
    bool top1_agrees = false;
    bool finite = false;
    double snapshot_max_abs_logit_error = 0;
    double snapshot_kl_reference_to_rollback = 0;
    bool snapshot_top1_agrees = false;
    bool snapshot_finite = false;
};

// Independent numerical diagnostic: compare a batch followed by partial
// recurrent rollback with sequential decoding of ONLY its accepted prefix.
// Covers zero/partial/full draft acceptance (keep 1..window.size()) and checks
// the NEXT token's logits, so stale GDN convolution/history state is observable.
// This clears and changes target memory; reset before starting a real request.
// Also compares continuation from a pending rollback with continuation after
// materializing that same accepted-state snapshot through sequence save/load.
// This isolates snapshot restoration from batch-vs-sequential arithmetic.
// Batched and sequential kernels may round differently; return measured errors
// rather than declaring byte equality or hiding them behind a wide tolerance.
std::vector<RollbackCheck> validate_recurrent_rollback(
    llama_context * target, const std::vector<llama_token> & prefix,
    const std::vector<llama_token> & window, llama_token continuation,
    bool batch_prefix = false);

} // namespace strata_a3b
