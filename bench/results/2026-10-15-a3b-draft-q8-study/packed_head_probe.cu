// Isolated sm75 probe of the UNMODIFIED copied Strata native_mmvq.cu.
// Baseline here is Strata's GGUF-layout single-column Q8 kernel, not the
// currently running GGML kernel. A favorable result is only a preflight for
// an integrated GGML real-model parity test and same-binary HTTP A/B.
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// The original native_mmvq.cu exposes generic IQ dispatch that this Q8-only
// preflight never calls. Its launch_pdl helper also needs host PDL symbols.
// Keep the unmodified Strata source and provide probe-only symbols.
namespace strata::kernels {
std::size_t iq_row_bytes(int, std::int64_t) noexcept { std::terminate(); }
void iq_mmvq(int, const void *, const void *, float *, int, int, int, void *) {
    throw std::logic_error("IQ dispatch is outside the Q8-only probe");
}
bool & pdl_scope() { static bool disabled = false; return disabled; }
bool pdl_launch_ok(const void *, cudaStream_t) { return false; }
} // namespace strata::kernels

namespace {
constexpr int kIn = 2048, kOut = 248320, kRowsPerChunk = 8192;
constexpr std::size_t kRowBytes = std::size_t(kIn / 32) * 34;
constexpr std::size_t kWeightBytes = std::size_t(kOut) * kRowBytes;
constexpr std::size_t kQ81Bytes = std::size_t(kIn / 32) * 36;

void check(cudaError_t status, const char * where) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(status));
}

struct DeviceBuffer {
    void * ptr = nullptr;
    explicit DeviceBuffer(std::size_t bytes) { check(cudaMalloc(&ptr, bytes), "cudaMalloc"); }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer & operator=(const DeviceBuffer &) = delete;
};

struct Events {
    cudaEvent_t start = nullptr, end = nullptr;
    Events() { check(cudaEventCreate(&start), "event start"); check(cudaEventCreate(&end), "event end"); }
    ~Events() { if (start) cudaEventDestroy(start); if (end) cudaEventDestroy(end); }
    Events(const Events &) = delete;
    Events & operator=(const Events &) = delete;
};

void read_and_pack(std::ifstream & gguf, std::uint64_t file_offset, void * raw_gpu, void * packed_gpu) {
    gguf.seekg(static_cast<std::streamoff>(file_offset));
    if (!gguf) throw std::runtime_error("invalid GGUF tensor file offset");
    auto * raw = static_cast<std::uint8_t *>(raw_gpu);
    auto * packed = static_cast<std::uint8_t *>(packed_gpu);
    const std::size_t qs_plane = std::size_t(kIn) * kOut;
    for (int row = 0; row < kOut; row += kRowsPerChunk) {
        const int rows = std::min(kRowsPerChunk, kOut - row);
        const std::size_t bytes = std::size_t(rows) * kRowBytes;
        const std::size_t qs_bytes = std::size_t(rows) * kIn;
        const std::size_t d_bytes = bytes - qs_bytes;
        std::vector<std::uint8_t> source(bytes), transformed(bytes);
        gguf.read(reinterpret_cast<char *>(source.data()), static_cast<std::streamsize>(bytes));
        if (gguf.gcount() != static_cast<std::streamsize>(bytes))
            throw std::runtime_error("short read from real GGUF Q8 output tensor");
        strata::kernels::native_q8_0_pack_host(source.data(), transformed.data(), kIn, rows);
        check(cudaMemcpy(raw + std::size_t(row) * kRowBytes, source.data(), bytes,
                         cudaMemcpyHostToDevice), "upload real GGUF blocks");
        check(cudaMemcpy(packed + std::size_t(row) * kIn, transformed.data(), qs_bytes,
                         cudaMemcpyHostToDevice), "upload Q8 qs plane");
        check(cudaMemcpy(packed + qs_plane + std::size_t(row) * kIn / 32 * 2,
                         transformed.data() + qs_bytes, d_bytes,
                         cudaMemcpyHostToDevice), "upload Q8 d plane");
    }
}

std::vector<float> activation(int case_id) {
    std::vector<float> x(kIn);
    for (int i = 0; i < kIn; ++i) {
        if (case_id == 0) x[i] = 0.0f;
        else {
            const float scale = std::ldexp(1.0f, ((i / 32 + case_id) % 13) - 7);
            x[i] = scale * (std::sin(float(i * 17 + case_id) * 0.013f) +
                            0.3f * std::cos(float(i * 3 - case_id) * 0.071f));
        }
    }
    return x;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: packed_head_probe MTP.gguf OUTPUT_WEIGHT_FILE_OFFSET");
        check(cudaSetDevice(0), "select GPU 0");
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, 0), "GPU 0 properties");
        if (prop.major != 7 || prop.minor != 5)
            throw std::runtime_error("this isolated benchmark is restricted to the measured sm75 card");
        std::size_t free_before = 0, total = 0, free_after = 0;
        check(cudaMemGetInfo(&free_before, &total), "memory before");
        DeviceBuffer weights(kWeightBytes), packed(kWeightBytes), input_f32(kIn * sizeof(float)),
                     input_q81(kQ81Bytes), baseline(kOut * sizeof(float)), candidate(kOut * sizeof(float));
        check(cudaMemGetInfo(&free_after, &total), "memory after");
        std::ifstream gguf(std::filesystem::path(argv[1]), std::ios::binary);
        if (!gguf) throw std::runtime_error("cannot open matching MTP GGUF");
        const std::uint64_t offset = std::stoull(argv[2]);
        const auto pack_start = std::chrono::steady_clock::now();
        read_and_pack(gguf, offset, weights.ptr, packed.ptr);
        const auto startup_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - pack_start).count();
        std::printf("GPU=%s raw_bytes=%zu extra_packed_bytes=%zu free_before_mib=%.2f free_after_mib=%.2f init_ms=%.3f\n",
                    prop.name, kWeightBytes, kWeightBytes, free_before / 1048576.0,
                    free_after / 1048576.0, startup_ms);

        auto * stream = static_cast<void *>(nullptr);
        const auto invoke = [&] (void * out) {
            strata::kernels::native_q8_0_mmvq(weights.ptr, input_q81.ptr,
                                               static_cast<float *>(out), kIn, kOut, 1, stream);
        };
        // native_mmvq requires a non-null stream pointer; use an explicit stream.
        cudaStream_t cuda_stream = nullptr;
        check(cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking), "create stream");
        stream = cuda_stream;
        for (int c = 0; c < 12; ++c) {
            const auto x = activation(c);
            check(cudaMemcpyAsync(input_f32.ptr, x.data(), x.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, cuda_stream), "upload input");
            strata::kernels::native_quantize_q8_1(static_cast<const float *>(input_f32.ptr),
                                                   input_q81.ptr, kIn, 1, stream);
            strata::kernels::native_q8_0_packed_unregister(weights.ptr);
            invoke(baseline.ptr);
            strata::kernels::native_q8_0_packed_register(weights.ptr, packed.ptr, kIn, kOut);
            invoke(candidate.ptr);
            check(cudaStreamSynchronize(cuda_stream), "parity synchronize");
            std::vector<std::uint32_t> a(kOut), b(kOut);
            check(cudaMemcpy(a.data(), baseline.ptr, kOut * sizeof(float), cudaMemcpyDeviceToHost), "download baseline");
            check(cudaMemcpy(b.data(), candidate.ptr, kOut * sizeof(float), cudaMemcpyDeviceToHost), "download candidate");
            auto mismatch = std::mismatch(a.begin(), a.end(), b.begin());
            if (mismatch.first != a.end()) {
                const auto row = mismatch.first - a.begin();
                std::fprintf(stderr, "FAIL case=%d first_row=%td baseline_bits=%08x packed_bits=%08x\n",
                             c, row, *mismatch.first, *mismatch.second);
                return 2;
            }
            std::printf("case=%d real_Q8_output_rows=%d bitwise_PASS\n", c, kOut);
        }

        Events events;
        const auto measure = [&] (bool use_packed) {
            if (use_packed) strata::kernels::native_q8_0_packed_register(weights.ptr, packed.ptr, kIn, kOut);
            else strata::kernels::native_q8_0_packed_unregister(weights.ptr);
            check(cudaEventRecord(events.start, cuda_stream), "record start");
            invoke(use_packed ? candidate.ptr : baseline.ptr);
            check(cudaEventRecord(events.end, cuda_stream), "record end");
            check(cudaEventSynchronize(events.end), "wait event");
            float milliseconds = 0.0f;
            check(cudaEventElapsedTime(&milliseconds, events.start, events.end), "event time");
            return double(milliseconds) * 1000.0;
        };
        for (int i = 0; i < 8; ++i) { measure(false); measure(true); }
        std::vector<double> off, on;
        for (int i = 0; i < 51; ++i) {
            if (i % 2) { on.push_back(measure(true)); off.push_back(measure(false)); }
            else { off.push_back(measure(false)); on.push_back(measure(true)); }
        }
        std::printf("Strata native standalone, not GGML service: Q8 GGUF_layout median_us=%.3f range_us=%.3f..%.3f; packed median_us=%.3f range_us=%.3f..%.3f; paired_samples=%zu\n",
                    median(off), *std::min_element(off.begin(), off.end()), *std::max_element(off.begin(), off.end()),
                    median(on), *std::min_element(on.begin(), on.end()), *std::max_element(on.begin(), on.end()), off.size());
        strata::kernels::native_q8_0_packed_unregister(weights.ptr);
        check(cudaStreamDestroy(cuda_stream), "destroy stream");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "packed_head_probe: %s\n", e.what());
        return 1;
    }
}
