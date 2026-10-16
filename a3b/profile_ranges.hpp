#pragma once

#include <cstdlib>
#include <cstring>

// CUDA's NVTX header can include Windows.h, whose min/max macros break the
// existing std::min/std::max calls in this translation unit on MSVC.
#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX 1
#  endif
#  ifdef min
#    undef min
#  endif
#  ifdef max
#    undef max
#  endif
#endif

// NVTX v3 is header-only. The CUDA build supplies its include directory;
// CPU-only builds and CUDA toolkits without NVTX retain an inert diagnostic.
#if defined(STRATA_A3B_NVTX_CUDA) && defined(__has_include)
#  if __has_include(<nvtx3/nvToolsExt.h>)
#    include <nvtx3/nvToolsExt.h>
#    define STRATA_A3B_NVTX_HEADER 1
#  endif
#endif

#if defined(_WIN32)
#  ifdef min
#    undef min
#  endif
#  ifdef max
#    undef max
#  endif
#endif

namespace strata_a3b::profile {

inline bool requested() {
    static const bool value = [] {
        const char * env = std::getenv("STRATA_A3B_NVTX");
        return env && std::strcmp(env, "1") == 0;
    }();
    return value;
}

inline bool available() {
#if defined(STRATA_A3B_NVTX_HEADER)
    return true;
#else
    return false;
#endif
}

class Range {
public:
    explicit Range(const char * label) : active_(requested() && available()) {
#if defined(STRATA_A3B_NVTX_HEADER)
        if (active_) nvtxRangePushA(label);
#else
        (void) label;
#endif
    }
    ~Range() {
#if defined(STRATA_A3B_NVTX_HEADER)
        if (active_) nvtxRangePop();
#endif
    }
    Range(const Range &) = delete;
    Range & operator=(const Range &) = delete;

private:
    bool active_;
};

} // namespace strata_a3b::profile
