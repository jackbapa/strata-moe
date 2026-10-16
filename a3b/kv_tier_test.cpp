#include "kv_tier.hpp"

#include <cstdio>

int main() {
    using strata_a3b::KvTierFormat;
    struct Case { uint32_t needed, capacity; KvTierFormat format; };
    constexpr Case cases[] = {
        {50000, 65536, KvTierFormat::F16},
        {65536, 65536, KvTierFormat::F16},
        {65537, 131072, KvTierFormat::F16},
        {131072, 131072, KvTierFormat::F16},
        {131073, 262144, KvTierFormat::Q8_0},
        {262144, 262144, KvTierFormat::Q8_0},
    };
    for (const auto & test : cases) {
        const auto capacity = strata_a3b::kv_tier_capacity(262144, test.needed);
        if (capacity != test.capacity || strata_a3b::kv_tier_f16_short_format(capacity) != test.format) {
            std::fprintf(stderr, "KV tier choice failed at %u tokens: got capacity %u\n", test.needed, capacity);
            return 1;
        }
    }
    constexpr struct { uint32_t needed, capacity; } low_vram_cases[] = {
        {7, 1024}, {1024, 1024}, {1025, 2048}, {8192, 8192},
        {8193, 16384}, {32768, 32768}, {262144, 262144},
    };
    for (const auto & test : low_vram_cases) {
        const auto capacity = strata_a3b::kv_tier_capacity(262144, test.needed, 1024);
        if (capacity != test.capacity) {
            std::fprintf(stderr, "Low-VRAM KV tier failed at %u tokens: got capacity %u\n", test.needed, capacity);
            return 1;
        }
    }
    if (strata_a3b::kv_tier_capacity(262144, 262145) != 0 ||
        !strata_a3b::kv_tier_f16_short_options_valid(false, false, true, false) ||
        !strata_a3b::kv_tier_f16_short_options_valid(true, true, false, false) ||
        strata_a3b::kv_tier_f16_short_options_valid(true, false, false, false) ||
        strata_a3b::kv_tier_f16_short_options_valid(true, true, true, false) ||
        strata_a3b::kv_tier_f16_short_options_valid(true, true, false, true)) {
        std::fputs("KV tier parameter compatibility failed\n", stderr);
        return 1;
    }
    std::puts("A3B F16-short/Q8-long KV tier logic PASS");
    return 0;
}
