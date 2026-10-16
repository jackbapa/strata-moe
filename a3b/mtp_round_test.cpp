#include "speculative.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F action, const char * message) {
    bool rejected = false;
    try { action(); } catch (const std::runtime_error &) { rejected = true; }
    require(rejected, message);
}

void check_packing() {
    using strata_a3b::detail::make_mtp_round_inputs;
    constexpr size_t width = 4;
    constexpr llama_pos start = 13;
    const std::vector<float> pending{-101.0f, -102.0f, -103.0f, -104.0f};
    std::vector<llama_token> tokens;
    std::vector<float> hidden;
    for (int row = 0; row < 8; ++row) {
        tokens.push_back(100 + row);
        for (size_t col = 0; col < width; ++col)
            hidden.push_back(float(10 * (row + 1) + col));
    }
    for (int accepted = 1; accepted <= 7; ++accepted) {
        const llama_token last = 900 + accepted;
        const auto packed = make_mtp_round_inputs(tokens, hidden, pending, accepted, start, last, start + accepted);
        require(packed.start == start, "fused round changed its start position");
        require(packed.tokens.size() == size_t(accepted + 1), "fused round has the wrong token count");
        require(packed.hidden.size() == size_t(accepted + 1) * width, "fused round has the wrong hidden row count");
        for (int row = 0; row < accepted; ++row)
            require(packed.tokens[size_t(row)] == tokens[size_t(row)], "accepted input token was changed or shifted");
        require(packed.tokens.back() == last, "first proposal input token is missing");
        for (size_t col = 0; col < width; ++col)
            require(packed.hidden[col] == pending[col], "first input lost the previous round's pending hidden row");
        for (int row = 1; row <= accepted; ++row)
            for (size_t col = 0; col < width; ++col)
                require(packed.hidden[size_t(row) * width + col] == hidden[size_t(row - 1) * width + col],
                        "token/hidden pairing is not shifted by one row");

        // Rejected rows must not influence either the catch-up or the appended proposal.
        auto poisoned_tokens = tokens;
        auto poisoned_hidden = hidden;
        for (size_t row = size_t(accepted); row < poisoned_tokens.size(); ++row)
            poisoned_tokens[row] = -10000 - llama_token(row);
        for (size_t i = size_t(accepted) * width; i < poisoned_hidden.size(); ++i)
            poisoned_hidden[i] = -10000.0f - float(i);
        const auto poisoned = make_mtp_round_inputs(poisoned_tokens, poisoned_hidden, pending,
                                                    accepted, start, last, start + accepted);
        require(poisoned.tokens == packed.tokens && poisoned.hidden == packed.hidden,
                "unaccepted suffix leaked into fused round inputs");

        // Exactly A observed rows are enough; the appended token uses observed row A-1.
        const std::vector<llama_token> exact_tokens(tokens.begin(), tokens.begin() + accepted);
        const std::vector<float> exact_hidden(hidden.begin(), hidden.begin() + accepted * width);
        const auto exact = make_mtp_round_inputs(exact_tokens, exact_hidden, pending,
                                                 accepted, start, last, start + accepted);
        require(exact.tokens == packed.tokens && exact.hidden == packed.hidden,
                "packing depends on unavailable unaccepted rows");
    }

    constexpr llama_token eos = 248046; // An opaque token ID to the packing helper.
    const auto terminal = make_mtp_round_inputs(tokens, hidden, pending, 2, 0, eos, 2);
    require(terminal.start == 0 && terminal.tokens.size() == 3 && terminal.tokens.back() == eos,
            "EOS proposal input was dropped or changed");
    auto eos_tokens = tokens;
    eos_tokens[0] = eos;
    const auto observed_eos = make_mtp_round_inputs(eos_tokens, hidden, pending, 2, start, 901, start + 2);
    require(observed_eos.tokens.size() == 3 && observed_eos.tokens[0] == eos && observed_eos.tokens[1] == tokens[1],
            "EOS in observed inputs truncated accepted rows");
}

void check_invalid_inputs() {
    using strata_a3b::detail::make_mtp_round_inputs;
    const std::vector<llama_token> tokens{10, 11, 12, 13, 14, 15, 16, 17};
    const std::vector<float> hidden(32, 1.0f), pending{2.0f, 3.0f, 4.0f, 5.0f};
    rejects([&] { make_mtp_round_inputs(tokens, hidden, pending, 0, 5, 20, 5); },
            "empty accepted prefix was packed");
    rejects([&] { make_mtp_round_inputs(tokens, hidden, pending, -1, 5, 20, 4); },
            "negative accepted prefix was packed");
    rejects([&] { make_mtp_round_inputs(tokens, hidden, pending, 8, 5, 20, 13); },
            "eight accepted rows exceeded the fused window limit");
    rejects([&] { make_mtp_round_inputs(tokens, hidden, pending, 2, -1, 20, 1); },
            "negative observed start was accepted");
    rejects([&] { make_mtp_round_inputs(tokens, hidden, pending, 2, 5, 20, 6); },
            "proposal position overlapped the accepted prefix");
    rejects([&] { make_mtp_round_inputs(tokens, hidden, pending, 2, 5, 20, 8); },
            "proposal position skipped a cell after the accepted prefix");
    rejects([&] { make_mtp_round_inputs(tokens, hidden, {}, 2, 5, 20, 7); },
            "empty pending hidden row was accepted");
    rejects([&] { make_mtp_round_inputs({10}, hidden, pending, 2, 5, 20, 7); },
            "short observed token prefix was accepted");
    rejects([&] { make_mtp_round_inputs(tokens, std::vector<float>(7, 1.0f), pending, 2, 5, 20, 7); },
            "short observed hidden prefix was accepted");
}

void check_fusion_gate() {
    using strata_a3b::detail::can_fuse_mtp_round;
    for (int accepted = 1; accepted <= 7; ++accepted) {
        const uint32_t rows = uint32_t(accepted + 1);
        require(can_fuse_mtp_round(accepted, rows, rows), "valid exact-capacity fused round was refused");
        require(!can_fuse_mtp_round(accepted, rows - 1, rows), "fused round exceeded logical batch capacity");
        require(!can_fuse_mtp_round(accepted, rows, rows - 1), "fused round exceeded microbatch capacity");
    }
    require(!can_fuse_mtp_round(0, 32, 32), "empty round was eligible for fusion");
    require(!can_fuse_mtp_round(-1, 32, 32), "negative accepted count was eligible for fusion");
    require(!can_fuse_mtp_round(8, 32, 32), "eight accepted rows were eligible for fusion");
    require(!can_fuse_mtp_round(1, 0, 32) && !can_fuse_mtp_round(1, 32, 0),
            "zero batch capacity was eligible for fusion");
    require(can_fuse_mtp_round(1, 32, 2) && !can_fuse_mtp_round(2, 32, 2),
            "ubatch=2 did not preserve the one-accepted-row boundary and larger-round fallback");
}
} // namespace

int main() {
    try {
        check_packing();
        check_invalid_inputs();
        check_fusion_gate();
        std::puts("PASS MTP round inputs: accepted 1..7; shifted hidden rows; rejected suffix isolation; positions; bounds; EOS passthrough; batch fallback");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL MTP round inputs: %s\n", e.what());
        return 1;
    }
}
