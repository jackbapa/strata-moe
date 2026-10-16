// Standalone host-memory capability probe. No model, GPU kernel, matrix math,
// engine registration, or inference configuration is involved.
// Build separately, e.g. nvcc -std=c++17 -O2 probe_pinned_uva.cu -o probe-pinned-uva
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

namespace {
constexpr uint64_t MiB = 1024 * 1024;
using Clock = std::chrono::steady_clock;
struct Options {
    uint64_t mib = 16000, chunk_mib = 1024;
    bool whole = false, use_register = false, help = false, no_map_flag = false;
};
struct Range {
    void * host = nullptr, * device = nullptr;
    size_t bytes = 0;
    bool cuda_owned = false, registered = false;
};
double elapsed_ms(Clock::time_point since) {
    return std::chrono::duration<double, std::milli>(Clock::now() - since).count();
}
uint64_t integer(const char * text) {
    if (!text || !*text) throw std::runtime_error("empty size argument");
    uint64_t value = 0;
    for (const char * p = text; *p; ++p) {
        if (*p < '0' || *p > '9') throw std::runtime_error("sizes must be positive integer MiB values");
        const uint64_t digit = uint64_t(*p - '0');
        if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10)
            throw std::runtime_error("size argument overflows uint64");
        value = value * 10 + digit;
    }
    if (!value || value > std::numeric_limits<size_t>::max() / MiB)
        throw std::runtime_error("size is zero or exceeds the host address range");
    return value;
}
Options parse(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") o.help = true;
        else if (arg == "--whole") o.whole = true;
        else if (arg == "--no-map-flag") o.no_map_flag = true;
        else if (arg == "--mib" && i + 1 < argc) o.mib = integer(argv[++i]);
        else if (arg == "--chunk-mib" && i + 1 < argc) o.chunk_mib = integer(argv[++i]);
        else if (arg == "--mode" && i + 1 < argc) {
            const std::string mode = argv[++i];
            if (mode == "register") o.use_register = true;
            else if (mode == "alloc") o.use_register = false;
            else throw std::runtime_error("--mode must be alloc or register");
        } else throw std::runtime_error("unknown or incomplete argument: " + arg);
    }
    return o;
}
void cuda_check(cudaError_t error, const char * operation) {
    if (error == cudaSuccess) return;
    const std::string detail = std::string(operation) + ": " + cudaGetErrorName(error) +
                               " (" + cudaGetErrorString(error) + ")";
    (void) cudaGetLastError(); // Do not let an expected allocation refusal poison the cleanup checks.
    throw std::runtime_error(detail);
}
void * page_allocate(size_t bytes) {
#ifdef _WIN32
    void * p = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!p) throw std::runtime_error("VirtualAlloc failed, Win32 error " + std::to_string(GetLastError()));
#else
    // All sizes are integer MiB, therefore multiples of this page alignment.
    void * p = std::aligned_alloc(4096, bytes);
    if (!p) throw std::runtime_error("aligned_alloc failed");
#endif
    return p;
}
void touch(void * memory, size_t bytes) {
    volatile unsigned char * p = static_cast<volatile unsigned char *>(memory);
    // Populate every CPU page before cudaHostRegister, as requested by this arm.
    for (size_t at = 0; at < bytes; at += 4096) p[at] = 0;
    p[bytes - 1] = 0;
}
int cleanup(std::vector<Range> & ranges) {
    int errors = 0;
    for (auto it = ranges.rbegin(); it != ranges.rend(); ++it) {
        if (!it->host) continue;
        if (it->cuda_owned) {
            const auto e = cudaFreeHost(it->host);
            if (e != cudaSuccess) {
                std::fprintf(stderr, "CLEANUP_ERROR cudaFreeHost: %s (%s)\n", cudaGetErrorName(e), cudaGetErrorString(e));
                (void) cudaGetLastError(); ++errors;
            }
        } else {
            if (it->registered) {
                const auto e = cudaHostUnregister(it->host);
                if (e != cudaSuccess) {
                    std::fprintf(stderr, "CLEANUP_ERROR cudaHostUnregister: %s (%s)\n", cudaGetErrorName(e), cudaGetErrorString(e));
                    (void) cudaGetLastError(); ++errors;
                }
            }
#ifdef _WIN32
            if (!VirtualFree(it->host, 0, MEM_RELEASE)) {
                std::fprintf(stderr, "CLEANUP_ERROR VirtualFree: Win32 error %lu\n", GetLastError()); ++errors;
            }
#else
            std::free(it->host);
#endif
        }
        it->host = it->device = nullptr;
    }
    ranges.clear();
    return errors;
}
}

int main(int argc, char ** argv) {
    Options o;
    std::vector<Range> ranges;
    uint64_t mapped_bytes = 0;
    size_t mapped_chunks = 0;
    int result = 0;
    const auto program_start = Clock::now();
    auto test_start = program_start;
    double test_ms = 0;
    try {
        o = parse(argc, argv);
        if (o.help) {
            std::puts("usage: probe-pinned-uva [--mib 16000] [--chunk-mib 1024] [--mode alloc|register] [--whole]\n"
                      "alloc: cudaHostAllocMapped; register: page-aligned CPU allocation, page touch, cudaHostRegisterMapped.\n"
                      "All chunks remain allocated/mapped until the target is reached or a call fails, then are released.\n"
                      "--whole uses one allocation/registration of --mib instead of chunks. GPU 0 is forced.\n"
                      "No GPU kernels or memory-copy throughput tests are run. Successful mapping is not an access/speed test.");
            return 0;
        }
        // Set visibility before the first CUDA runtime call. Device ordinal 0
        // below then selects the user's physical GPU 0 only.
#ifdef _WIN32
        if (_putenv_s("CUDA_VISIBLE_DEVICES", "0")) throw std::runtime_error("cannot set CUDA_VISIBLE_DEVICES=0");
#else
        if (setenv("CUDA_VISIBLE_DEVICES", "0", 1)) throw std::runtime_error("cannot set CUDA_VISIBLE_DEVICES=0");
#endif
        if (!o.no_map_flag) cuda_check(cudaSetDeviceFlags(cudaDeviceMapHost), "cudaSetDeviceFlags(cudaDeviceMapHost)");
        cuda_check(cudaSetDevice(0), "cudaSetDevice(0)");
        size_t free_bytes = 0, total_bytes = 0;
        cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo before host mapping");
        std::printf("CONTEXT map_flag_requested=%d memory_query_before_mapping=1\n", (int) !o.no_map_flag);
        cudaDeviceProp props{};
        cuda_check(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties(0)");
        if (!props.canMapHostMemory) throw std::runtime_error("GPU 0 does not support mapped host memory");
        const uint64_t total = o.mib * MiB;
        const uint64_t chunk = (o.whole ? o.mib : std::min(o.mib, o.chunk_mib)) * MiB;
        ranges.reserve(static_cast<size_t>((total - 1) / chunk + 1));
        std::printf("PROBE device=0 name=\"%s\" sm=%d%d pci=%04x:%02x:%02x.0 CUDA_VISIBLE_DEVICES=0 mode=%s requested_mib=%llu chunk_mib=%llu whole=%d\n",
                    props.name, props.major, props.minor, props.pciDomainID, props.pciBusID, props.pciDeviceID, o.use_register ? "register" : "alloc",
                    (unsigned long long) o.mib, (unsigned long long) (chunk / MiB), (int) o.whole);
        std::fflush(stdout);
        test_start = Clock::now();
        for (uint64_t offset = 0; offset < total; ) {
            // Install the record before acquiring resources, so every failure
            // (including an alias refusal) is cleaned up by the common tail.
            ranges.emplace_back();
            auto & r = ranges.back();
            r.bytes = static_cast<size_t>(std::min(chunk, total - offset));
            const auto start = Clock::now();
            double touch_ms = 0, register_ms = 0;
            if (o.use_register) {
                r.host = page_allocate(r.bytes);
                const auto touch_start = Clock::now(); touch(r.host, r.bytes); touch_ms = elapsed_ms(touch_start);
                const auto register_start = Clock::now();
                cuda_check(cudaHostRegister(r.host, r.bytes, cudaHostRegisterMapped), "cudaHostRegisterMapped");
                r.registered = true; register_ms = elapsed_ms(register_start);
            } else {
                const auto allocated = cudaHostAlloc(&r.host, r.bytes, cudaHostAllocMapped);
                r.cuda_owned = r.host != nullptr;
                cuda_check(allocated, "cudaHostAllocMapped");
            }
            cuda_check(cudaHostGetDevicePointer(&r.device, r.host, 0), "cudaHostGetDevicePointer");
            if (!r.device) throw std::runtime_error("cudaHostGetDevicePointer returned a null alias");
            mapped_bytes += r.bytes; ++mapped_chunks; offset += r.bytes;
            std::printf("CHUNK index=%zu bytes=%zu cumulative_mapped_mib=%.3f elapsed_ms=%.3f touch_ms=%.3f register_ms=%.3f alias=ok\n",
                        mapped_chunks, r.bytes, mapped_bytes / double(MiB), elapsed_ms(start), touch_ms, register_ms);
            std::fflush(stdout);
        }
        test_ms = elapsed_ms(test_start);
    } catch (const std::exception & e) {
        test_ms = elapsed_ms(test_start);
        result = 1;
        std::fprintf(stderr, "PROBE_ERROR after_mapped_mib=%.3f chunks=%zu: %s\n", mapped_bytes / double(MiB), mapped_chunks, e.what());
        std::fflush(stderr);
    } catch (...) {
        test_ms = elapsed_ms(test_start); result = 1;
        std::fprintf(stderr, "PROBE_ERROR unknown exception after_mapped_mib=%.3f\n", mapped_bytes / double(MiB));
    }
    const auto cleanup_start = Clock::now();
    const int cleanup_errors = cleanup(ranges);
    if (cleanup_errors) result = 1;
    std::printf("SUMMARY mode=%s whole=%d requested_mib=%llu mapped_mib=%.3f mapped_chunks=%zu test_ms=%.3f cleanup_ms=%.3f total_ms=%.3f cleanup_errors=%d status=%s\n",
                o.use_register ? "register" : "alloc", (int) o.whole, (unsigned long long) o.mib,
                mapped_bytes / double(MiB), mapped_chunks, test_ms, elapsed_ms(cleanup_start), elapsed_ms(program_start),
                cleanup_errors, result == 0 ? "success" : "failed");
    std::fflush(stdout);
    return result;
}
