#pragma once

#include <cstdint>

namespace strata_a3b {

enum class KvTierFormat { F16, Q8_0 };

// Reserve the compact KV format for the largest physical tier. The advertised
// context limit is independent of this per-request allocation choice.
constexpr KvTierFormat kv_tier_f16_short_format(uint32_t capacity) {
    return capacity > 131072 ? KvTierFormat::Q8_0 : KvTierFormat::F16;
}

constexpr bool kv_tier_f16_short_options_valid(bool enabled, bool tiered,
                                               bool explicit_q8, bool explicit_q4) {
    return !enabled || (tiered && !explicit_q8 && !explicit_q4);
}

// Physical K/V capacity for a request; the advertised context remains max_context.
constexpr uint32_t kv_tier_capacity(uint32_t max_context, uint32_t needed, uint32_t initial = 65536) {
    if (max_context == 0 || needed > max_context) return 0;
    uint32_t tier = max_context < initial ? max_context : initial;
    while (tier < needed) {
        tier = tier > max_context / 2 ? max_context : tier * 2;
    }
    return tier;
}

static_assert(kv_tier_capacity(262144, 65536) == 65536);
static_assert(kv_tier_capacity(262144, 65537) == 131072);
static_assert(kv_tier_capacity(262144, 131072) == 131072);
static_assert(kv_tier_capacity(262144, 131073) == 262144);
static_assert(kv_tier_capacity(262144, 262144) == 262144);
static_assert(kv_tier_capacity(262144, 262145) == 0);
static_assert(kv_tier_capacity(262144, 7 + 128, 1024) == 1024);
static_assert(kv_tier_capacity(262144, 1024 + 128, 1024) == 2048);
static_assert(kv_tier_capacity(262144, 8192 + 128, 1024) == 16384);
static_assert(kv_tier_f16_short_format(kv_tier_capacity(262144, 50000)) == KvTierFormat::F16);
static_assert(kv_tier_f16_short_format(kv_tier_capacity(262144, 65537)) == KvTierFormat::F16);
static_assert(kv_tier_f16_short_format(kv_tier_capacity(262144, 131073)) == KvTierFormat::Q8_0);

} // namespace strata_a3b
