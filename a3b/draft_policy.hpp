#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata_a3b {

// Pure CPU scheduling: proposals still undergo the ordinary full target
// verification. No target sampling, RNG, routes or weights change here.
class AdaptiveDraftPolicy {
public:
    static constexpr int capacity = 16;
    explicit AdaptiveDraftPolicy(int maximum = 3) : maximum_(maximum) {
        if (maximum < 1 || maximum > capacity) throw std::runtime_error("invalid adaptive draft maximum");
        reset();
    }

    void reset() {
        successes_.fill(0); failures_.fill(0); cost_.fill(0);
        for (auto & row : availability_) row.fill(0);
        observations_.fill(0); last_seen_.fill(0); choices_.fill(0);
        selections_ = observations_total_ = 0;
    }

    int choose(int limit) {
        if (limit < 0) throw std::runtime_error("negative adaptive draft limit");
        const int end = std::min(limit, maximum_);
        ++selections_;
        // Obtain a local cost for each permitted size, including plain decode.
        // Terminal/censored observations do not mark an unmeasured size warm.
        for (int k = 0; k <= end; ++k) if (!observations_[k]) return picked(k);
        int best = 0;
        double rate = expected_tokens(0) / cost_[0];
        for (int k = 1; k <= end; ++k) {
            const double candidate = expected_tokens(k) / cost_[k];
            if (candidate > rate * 1.03) { best = k; rate = candidate; }
        }
        // Probe a neighbour periodically, so success on code can grow a short
        // window and misses on prose can shrink a long one. Revisit stale
        // sizes less often as cache placement/context cost can change too.
        if (end && selections_ % 32 == 0) {
            int oldest = 0;
            for (int k = 1; k <= end; ++k) if (last_seen_[k] < last_seen_[oldest]) oldest = k;
            return picked(oldest);
        }
        if (end && selections_ % 8 == 0) {
            int neighbour = best + ((selections_ / 8) % 2 ? 1 : -1);
            if (neighbour < 0 || neighbour > end) neighbour = best + (best == 0 ? 1 : -1);
            return picked(neighbour);
        }
        return picked(best);
    }

    // matched is the number of matching OUTPUT proposals; consumed is the
    // number of accepted INPUT rows. A terminal EOS/output limit can match the
    // last observed proposal with consumed==matched: the next position was not
    // tested. Never turn that censored suffix into rejection observations.
    void observe(int offered, int matched, int consumed, double round_ms, int planned = -1) {
        if (planned < 0) planned = offered;
        if (offered < 0 || offered > maximum_ || matched < 0 || matched > offered ||
            consumed < 1 || consumed > offered + 1 || matched > consumed ||
            planned < offered || planned > maximum_ ||
            !std::isfinite(round_ms) || round_ms <= 0)
            throw std::runtime_error("invalid adaptive draft observation");
        for (int i = 0; i < matched; ++i) update_position(i, true);
        const bool rejected = matched < offered && consumed > matched;
        if (rejected) update_position(matched, false);
        // Terminal rounds have no later decision in this request. Their shorter
        // catch-up/sample work would bias an ordinary full-round cost downward.
        if (consumed == matched) return;
        for (int i = 0; i < planned; ++i) {
            const double exists = i < offered ? 1.0 : 0.0;
            if (!observations_[planned]) availability_[planned][i] = exists;
            else availability_[planned][i] += 0.2 * (exists - availability_[planned][i]);
        }
        // A draft EOS (or an existing CPU confidence cutoff) can offer fewer
        // tokens than planned. Attribute its measured cost to that scheduling
        // choice, so initial exploration cannot get stuck on an unseen size.
        if (!observations_[planned]) cost_[planned] = round_ms;
        else cost_[planned] += 0.2 * (round_ms - cost_[planned]);
        ++observations_[planned];
        last_seen_[planned] = ++observations_total_;
    }

    double acceptance(int position) const {
        if (position < 0 || position >= maximum_) throw std::runtime_error("invalid adaptive acceptance position");
        // Weak prior only. Counts at an unvisited deeper position remain
        // unobserved, rather than inventing successes from shallower matches.
        return (successes_[position] + 1.5) / (successes_[position] + failures_[position] + 2.0);
    }
    double expected_tokens(int k) const {
        if (k < 0 || k > maximum_) throw std::runtime_error("invalid adaptive expected window");
        double result = 1, prefix = 1;
        for (int i = 0; i < k; ++i) {
            prefix *= acceptance(i);
            result += prefix * (observations_[k] ? availability_[k][i] : 1.0);
        }
        return result;
    }
    double cost_ms(int k) const { return cost_.at(k); }
    uint64_t observations(int k) const { return observations_.at(k); }
    uint64_t choices(int k) const { return choices_.at(k); }
    double successes(int i) const { return successes_.at(i); }
    double failures(int i) const { return failures_.at(i); }

private:
    int picked(int k) { ++choices_[k]; return k; }
    void update_position(int i, bool matched) {
        successes_[i] *= 0.96; failures_[i] *= 0.96;
        (matched ? successes_[i] : failures_[i]) += 1.0;
    }
    int maximum_;
    uint64_t selections_ = 0, observations_total_ = 0;
    std::array<double, capacity> successes_{}, failures_{};
    std::array<std::array<double, capacity>, capacity + 1> availability_{};
    std::array<double, capacity + 1> cost_{};
    std::array<uint64_t, capacity + 1> observations_{}, last_seen_{}, choices_{};
};
} // namespace strata_a3b
