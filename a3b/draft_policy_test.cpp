#include "draft_policy.hpp"
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace {
void require(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "invalid policy input was accepted");
}
void train(strata_a3b::AdaptiveDraftPolicy & p, int maximum, bool all_matched) {
    for (int repeat = 0; repeat < 80; ++repeat)
        for (int k = 0; k <= maximum; ++k)
            p.observe(k, all_matched ? k : 0, all_matched ? k + 1 : 1, 10.0 + 1.0 * k);
}
} // namespace

int main() {
    try {
        using strata_a3b::AdaptiveDraftPolicy;
        AdaptiveDraftPolicy p(6);
        for (int k = 0; k <= 6; ++k) {
            require(p.choose(6) == k, "initial local-cost exploration skipped a size");
            p.observe(k, k, k + 1, 10 + k);
        }
        const double old_failure = p.failures(2), old_success = p.successes(2);
        p.observe(6, 2, 2, 30); // EOS/output limit after two matches.
        require(p.failures(2) == old_failure && p.successes(2) == old_success,
                "censored untested next proposal changed acceptance counts");
        require(p.failures(3) == 0 && p.successes(3) > 0,
                "censored deep suffix was marked rejected");
        const auto cost_samples = p.observations(6);
        p.observe(6, 3, 3, 1);
        require(p.observations(6) == cost_samples, "terminal round biased cost samples");
        p.observe(6, 2, 3, 30); // Next proposal actually sampled and rejected.
        require(p.failures(2) > old_failure, "tested rejected proposal did not update conditional failure");
        require(p.failures(3) == 0, "failure updated an unreached deeper position");
        p.reset();
        train(p, 6, false);
        int zero = 0, nonzero = 0;
        for (int n = 0; n < 96; ++n) { const int k = p.choose(6); if (k == 0) ++zero; else ++nonzero; }
        require(zero > 70 && nonzero > 0, "low acceptance did not prefer decode with bounded exploration");
        p.reset(); train(p, 6, true);
        int large = 0, probes = 0;
        for (int n = 0; n < 96; ++n) { const int k = p.choose(6); if (k >= 5) ++large; else ++probes; }
        require(large > 70 && probes > 0, "high acceptance did not grow the draft window or revisit stale sizes");
        for (int cap = 0; cap <= 6; ++cap) for (int n = 0; n < 16; ++n)
            require(p.choose(cap) <= cap, "remaining-token limit was exceeded");
        p.reset();
        require(p.choose(6) == 0 && p.observations(6) == 0 && p.successes(0) == 0,
                "request reset retained costs or acceptance history");
        p.reset();
        for (int k = 0; k <= 6; ++k) {
            require(p.choose(6) == k, "truncated proposal stalled initial cost exploration");
            p.observe(std::min(k, 1), 0, 1, 10 + k, k);
        }
        require(p.observations(6) == 1, "early draft EOS lost planned-window cost");
        require(std::abs(p.expected_tokens(6) - p.expected_tokens(1)) < 1e-12,
                "unoffered truncated proposals inflated expected committed tokens");
        rejects([] { AdaptiveDraftPolicy invalid(0); });
        rejects([&] { p.choose(-1); });
        rejects([&] { p.observe(7, 0, 1, 1); });
        rejects([&] { p.observe(2, 2, 1, 1); });
        rejects([&] { p.observe(2, 1, 2, std::nan("")); });
        std::puts("PASS adaptive draft policy: measured costs; conditional acceptance; censored EOS/limits; neighbour/stale probes; bounds; per-request reset");
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "FAIL adaptive draft policy: %s\n", e.what()); return 1; }
}
