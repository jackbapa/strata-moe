#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strata_a3b::detail {

// Validate a loaded I32 map before K2 can feed its GPU-sampled token to the
// next embedding lookup. Sorted IDs also preserve first-token tie behavior.
inline void validate_draft_vocab_ids(const std::vector<int32_t> & ids, int32_t full_vocab,
                                     const std::vector<int32_t> & required) {
    if (full_vocab <= 1 || ids.empty() || ids.size() >= (size_t) full_vocab)
        throw std::runtime_error("invalid compact draft vocabulary size");
    int32_t previous = -1;
    for (int32_t id : ids) {
        if (id <= previous || id >= full_vocab)
            throw std::runtime_error("compact draft token IDs must be sorted, unique and in range");
        previous = id;
    }
    for (int32_t id : required) {
        if (id >= 0 && id < full_vocab && !std::binary_search(ids.begin(), ids.end(), id))
            throw std::runtime_error("compact draft vocabulary omits a required token");
    }
}

} // namespace strata_a3b::detail
