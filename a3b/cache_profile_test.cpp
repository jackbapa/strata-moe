#include "cache_profile.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace strata_a3b;
namespace {
int checks = 0;
void require(bool condition, const char * message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void rejected(const std::function<void()> & action, const char * message) {
    try { action(); }
    catch (const std::exception & error) {
        require(std::string(error.what()).find(message) != std::string::npos, error.what());
        return;
    }
    throw std::runtime_error(std::string("expected rejection: ") + message);
}
ExpertCacheProfile fixture() {
    ExpertCacheProfile profile;
    profile.model_sha256 = std::string(64, 'a'); profile.source_fingerprint = std::string(64, 'b');
    profile.model_bytes = 3;
    for (int layer = 0; layer < expert_profile_layers; ++layer)
        for (int expert = 0; expert < expert_profile_experts; ++expert) {
            uint64_t score = layer == 0 && expert == 9 ? 100 :
                             layer == 0 && expert == 8 ? 90 :
                             layer == 1 && expert == 7 ? 80 : 0;
            profile.scores.push_back({layer, expert, score});
        }
    std::sort(profile.scores.begin(), profile.scores.end(), [](const auto & a, const auto & b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.layer != b.layer) return a.layer < b.layer;
        return a.expert < b.expert;
    });
    return profile;
}
std::string serialized(const ExpertCacheProfile & profile) {
    std::ostringstream text;
    text << "STRATA_A3B_EXPERT_PROFILE_V1\nmodel_sha256 " << profile.model_sha256
         << "\nmodel_bytes " << profile.model_bytes
         << "\nlayers 40\nexperts 256\nembedding 2048\nffn 512\nexperts_used 8\nbundle_bytes "
         << profile.bundle_bytes << "\nsource_fingerprint " << profile.source_fingerprint << "\nscores 10240\n";
    for (const auto & score : profile.scores)
        text << score.layer << '\t' << score.expert << '\t' << score.score << '\n';
    return text.str();
}
ExpertCacheProfile parse(const std::string & text) {
    std::istringstream input(text);
    return read_expert_cache_profile(input);
}
std::string replace_first(std::string text, const std::string & from, const std::string & to) {
    const auto at = text.find(from);
    if (at == std::string::npos) throw std::runtime_error("test replacement not found");
    text.replace(at, from.size(), to);
    return text;
}
void parser_and_planner() {
    const auto original = fixture();
    const auto text = serialized(original);
    const auto profile = parse(text);
    require(profile.scores.size() == 10240, "all 40x256 rows must be present");
    const auto plan = plan_expert_cache(profile, 3 * expert_profile_bundle_bytes - 1);
    require(plan.selected_experts == 2 && plan.planned_bytes == 2 * expert_profile_bundle_bytes, "budget floor");
    require(plan.expert_ids[0] == std::vector<int>({9, 8}) && plan.expert_ids[1].empty(), "global selection is uneven");
    require(plan.total_score == 270 && plan.covered_score == 190, "coverage counts are exact");
    require(plan.coverage == double(190) / 270, "training coverage ratio");
    const auto all = plan_expert_cache(profile, std::numeric_limits<uint64_t>::max());
    require(all.selected_experts == 3 && all.coverage == 1, "never cache zero scores");
    const auto empty = plan_expert_cache(profile, expert_profile_bundle_bytes - 1);
    require(empty.selected_experts == 0 && empty.coverage == 0, "sub-bundle budget selects nothing");
    auto tie = profile;
    tie.scores[0].score = 90;
    std::swap(tie.scores[0], tie.scores[1]);
    require(plan_expert_cache(parse(serialized(tie)), expert_profile_bundle_bytes).expert_ids[0] == std::vector<int>({8}),
            "equal score selection uses ascending expert ID");
    std::swap(tie.scores[0], tie.scores[1]);
    rejected([&] { parse(serialized(tie)); }, "canonical");
    auto zero = profile;
    for (auto & row : zero.scores) row.score = 0;
    std::sort(zero.scores.begin(), zero.scores.end(), [](auto a, auto b) { return a.layer * 256 + a.expert < b.layer * 256 + b.expert; });
    const auto no_scores = plan_expert_cache(zero, 100 * expert_profile_bundle_bytes);
    require(no_scores.selected_experts == 0 && no_scores.total_score == 0 && no_scores.coverage == 0, "all-zero profile");
    for (const auto & entry : std::vector<std::pair<std::string, std::string>>{
             {"STRATA_A3B_EXPERT_PROFILE_V1", "STRATA_A3B_EXPERT_PROFILE_V2"},
             {"layers 40", "layers 39"}, {"experts 256", "experts 128"},
             {"embedding 2048", "embedding 1024"}, {"ffn 512", "ffn 1024"},
             {"experts_used 8", "experts_used 4"}, {"scores 10240", "scores 3"},
             {"bundle_bytes 3342336", "bundle_bytes 1"},
             {"model_bytes 3", "model_bytes -3"}, {"model_bytes 3", "model_bytes 18446744073709551616"},
             {"0\t9\t100", "40\t9\t100"}, {"0\t9\t100", "0\t256\t100"},
             {"0\t9\t100", "0\t9\t-1"}, {"0\t9\t100", "0\t9\t1.0"}}) {
        bool did_reject = false;
        try { parse(replace_first(text, entry.first, entry.second)); }
        catch (const std::exception &) { did_reject = true; }
        require(did_reject, "malformed fixed field/row must fail");
    }
    rejected([&] { parse(text + "unexpected\n"); }, "trailing");
    rejected([&] { parse(text.substr(0, text.find("0\t8\t90"))); }, "truncated");
    rejected([&] { parse(replace_first(text, "0\t8\t90", "0\t9\t90")); }, "duplicate");
    rejected([&] { parse(replace_first(text, "0\t9\t100", "0\t9\t89")); }, "canonical");
    rejected([&] { parse(replace_first(text, std::string(64, 'a'), std::string(64, 'A'))); }, "identity");
    auto overflow = profile;
    overflow.scores[0].score = std::numeric_limits<uint64_t>::max();
    rejected([&] { parse(serialized(overflow)); }, "total score overflow");
    rejected([&] { plan_expert_cache(overflow, 0); }, "total score overflow");
}
void loader_identity() {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path() / ("a3b-cache-profile-test-" + std::to_string(unique));
    std::filesystem::create_directory(directory);
    struct Cleanup {
        std::filesystem::path directory;
        ~Cleanup() {
            std::error_code ignored;
            // Only the two explicitly created files; never recurse through a computed path.
            std::filesystem::remove(directory / "model.gguf", ignored);
            std::filesystem::remove(directory / "profile.txt", ignored);
            std::filesystem::remove(directory, ignored);
        }
    } cleanup{directory};
    const auto model = directory / "model.gguf", profile_path = directory / "profile.txt";
    std::ofstream(model, std::ios::binary) << "abc";
    std::ofstream(profile_path, std::ios::binary) << serialized(fixture());
    rejected([&] { load_expert_cache_profile(profile_path.u8string(), model.u8string(), 0); }, "SHA-256 mismatch");
    auto correct_hash = fixture();
    correct_hash.model_sha256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    std::ofstream(profile_path, std::ios::binary) << serialized(correct_hash);
    // Known whole-file SHA succeeds, but the non-GGUF fixture still fails
    // geometry inspection. No GPU backend is created by this test.
    rejected([&] { load_expert_cache_profile(profile_path.u8string(), model.u8string(), 0); }, "cannot inspect");
    auto wrong_bytes = fixture(); wrong_bytes.model_bytes = 4;
    std::ofstream(profile_path, std::ios::binary) << serialized(wrong_bytes);
    rejected([&] { load_expert_cache_profile(profile_path.u8string(), model.u8string(), 0); }, "byte size mismatch");
}
}
int main() {
    try {
        parser_and_planner(); loader_identity();
        std::cout << "PASS " << checks << " offline cache-profile checks; no GPU or network used\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n'; return 1;
    }
}
