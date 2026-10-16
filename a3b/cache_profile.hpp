#pragma once
#include <array>
#include <cstdint>
#include <istream>
#include <string>
#include <vector>

namespace strata_a3b {
constexpr int expert_profile_layers = 40;
constexpr int expert_profile_experts = 256;
constexpr uint64_t expert_profile_bundle_bytes = 3342336;

struct ExpertScore {
    int layer = 0, expert = 0;
    uint64_t score = 0;
};
struct ExpertCacheProfile {
    std::string model_sha256, source_fingerprint;
    uint64_t model_bytes = 0, bundle_bytes = expert_profile_bundle_bytes;
    // Exactly 40*256 globally ordered entries, including zero scores.
    std::vector<ExpertScore> scores;
};
struct ExpertCachePlan {
    // Selected IDs per layer, in descending global score order; ties use ID.
    std::array<std::vector<int>, expert_profile_layers> expert_ids;
    uint64_t budget_bytes = 0, planned_bytes = 0, bundle_bytes = expert_profile_bundle_bytes;
    uint64_t total_score = 0, covered_score = 0, selected_experts = 0;
    double coverage = 0, model_hash_ms = 0;
    std::string model_sha256, profile_sha256, source_fingerprint;
};
// Parsing/selection are exposed for offline tests. Both reject malformed rows.
ExpertCacheProfile read_expert_cache_profile(std::istream & input);
ExpertCachePlan plan_expert_cache(const ExpertCacheProfile & profile, uint64_t budget_bytes);
// Production entry point: validates complete Q8 GGUF geometry and the actual
// whole-file SHA-256 exactly once per invocation before returning a plan.
ExpertCachePlan load_expert_cache_profile(const std::string & profile_path,
                                         const std::string & model_path,
                                         uint64_t budget_bytes);
}
