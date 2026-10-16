// Isolated Q6_K micro parity probe. This executable is never linked into strata-moe.
// Both dot products consume the same Q6_K bytes from an IQ3_S GGUF and the same
// GGML-generated Q8_1 activation buffer. No weight or activation requantization.

#include "ggml.h"
#include "gguf.h"
#include "ggml-cuda.h"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t result, const char * what) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(result));
    }
}

struct DeviceBuffer {
    void * ptr = nullptr;
    explicit DeviceBuffer(size_t bytes) { check(cudaMalloc(&ptr, bytes), "cudaMalloc"); }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer & operator=(const DeviceBuffer &) = delete;
};

struct EventPair {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    EventPair() {
        check(cudaEventCreate(&start), "cudaEventCreate(start)");
        check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    }
    ~EventPair() {
        if (stop) cudaEventDestroy(stop);
        if (start) cudaEventDestroy(start);
    }
    double time_us(cudaStream_t stream, const std::function<void()> & launch) {
        check(cudaEventRecord(start, stream), "cudaEventRecord(start)");
        launch();
        check(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
        check(cudaEventSynchronize(stop), "cudaEventSynchronize");
        float ms = 0;
        check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        return 1000.0 * ms;
    }
};

double median(std::vector<double> xs) {
    std::sort(xs.begin(), xs.end());
    return xs[xs.size() / 2];
}

struct TensorBytes {
    std::string name;
    int n_in = 0;
    int n_out = 0;
    std::vector<char> bytes;
};

TensorBytes load_q6_tensor(const char * path, const std::string & requested) {
    gguf_init_params params{};
    params.no_alloc = true;
    gguf_context * raw = gguf_init_from_file(path, params);
    if (!raw) throw std::runtime_error("cannot open GGUF metadata");
    std::unique_ptr<gguf_context, decltype(&gguf_free)> header(raw, gguf_free);

    int64_t choice = -1;
    for (int64_t id = 0; id < gguf_get_n_tensors(header.get()); ++id) {
        if (gguf_get_tensor_type(header.get(), id) != GGML_TYPE_Q6_K) continue;
        const int64_t * ne = gguf_get_tensor_ne(header.get(), id);
        if (ne[0] <= 0 || ne[1] <= 0 || ne[2] != 1 || ne[3] != 1 ||
            ne[0] > std::numeric_limits<int>::max() || ne[1] > std::numeric_limits<int>::max()) continue;
        if (!requested.empty()) {
            if (requested == gguf_get_tensor_name(header.get(), id)) { choice = id; break; }
        } else if (ne[0] == 2048 && ne[1] >= 512 && ne[1] <= 8192) {
            choice = id;
            break;
        }
    }
    if (choice < 0) throw std::runtime_error("no requested Q6_K 2D tensor (default: 2048-wide, 512..8192 rows)");

    const int64_t * ne = gguf_get_tensor_ne(header.get(), choice);
    const int n_in = static_cast<int>(ne[0]);
    const int n_out = static_cast<int>(ne[1]);
    if (n_in % 512 != 0) {
        throw std::runtime_error("this fixture requires n_in divisible by 512 so GGML Q8_1 padding equals native's tight stride");
    }
    const size_t bytes = gguf_get_tensor_size(header.get(), choice);
    const size_t expected = ggml_row_size(GGML_TYPE_Q6_K, n_in) * size_t(n_out);
    if (bytes != expected) throw std::runtime_error("unexpected Q6_K tensor byte size");
    const uint64_t offset = uint64_t(gguf_get_data_offset(header.get())) +
                            uint64_t(gguf_get_tensor_offset(header.get(), choice));
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open GGUF payload");
    file.seekg(static_cast<std::streamoff>(offset));
    if (!file) throw std::runtime_error("cannot seek to GGUF tensor");
    TensorBytes result{gguf_get_tensor_name(header.get(), choice), n_in, n_out, std::vector<char>(bytes)};
    file.read(result.bytes.data(), static_cast<std::streamsize>(bytes));
    if (file.gcount() != static_cast<std::streamsize>(bytes)) throw std::runtime_error("short GGUF tensor read");
    return result;
}

uint32_t bits(float x) {
    uint32_t value;
    std::memcpy(&value, &x, sizeof(value));
    return value;
}

void compare(const std::vector<float> & expected, const std::vector<float> & actual,
             int n_out, int ncols) {
    size_t bit_mismatch = 0, nonfinite = 0;
    double max_abs = 0.0;
    int first_col = -1, first_row = -1;
    for (int col = 0; col < ncols; ++col) {
        for (int row = 0; row < n_out; ++row) {
            const size_t i = size_t(col) * n_out + row;
            if (!std::isfinite(expected[i]) || !std::isfinite(actual[i])) ++nonfinite;
            if (bits(expected[i]) != bits(actual[i])) {
                ++bit_mismatch;
                if (first_col < 0) { first_col = col; first_row = row; }
            }
            if (std::isfinite(expected[i]) && std::isfinite(actual[i])) {
                max_abs = std::max(max_abs, std::abs(double(expected[i]) - double(actual[i])));
            }
        }
    }
    std::printf("ncols=%d outputs=%zu bit_mismatch=%zu nonfinite=%zu max_abs_delta=%.9g",
                ncols, expected.size(), bit_mismatch, nonfinite, max_abs);
    if (first_col >= 0) {
        const size_t i = size_t(first_col) * n_out + first_row;
        std::printf(" first=(col=%d,row=%d,ggml=%.9g,native=%.9g)", first_col, first_row,
                    expected[i], actual[i]);
    }
    std::puts("");
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc < 2 || argc > 4) {
            throw std::runtime_error("usage: native_q6_gguf_parity <IQ3_S.gguf> [Q6_K tensor name] [timing repeats, default 40]");
        }
        const std::string tensor_name = argc >= 3 ? argv[2] : "";
        const int repeats = argc >= 4 ? std::stoi(argv[3]) : 40;
        if (repeats < 1 || repeats > 10000) throw std::runtime_error("timing repeats must be 1..10000");
        const TensorBytes tensor = load_q6_tensor(argv[1], tensor_name);
        std::printf("tensor=%s type=Q6_K n_in=%d n_out=%d bytes=%zu\n",
                    tensor.name.c_str(), tensor.n_in, tensor.n_out, tensor.bytes.size());
        std::printf("STRATA_A3B_Q6_GROUPED_EXACT=%s STRATA_A3B_Q6_REUSE_EXACT=%s\n",
                    std::getenv("STRATA_A3B_Q6_GROUPED_EXACT") ? std::getenv("STRATA_A3B_Q6_GROUPED_EXACT") : "unset",
                    std::getenv("STRATA_A3B_Q6_REUSE_EXACT") ? std::getenv("STRATA_A3B_Q6_REUSE_EXACT") : "unset");
        check(cudaSetDevice(0), "cudaSetDevice(0)");
        ggml_backend_t backend = ggml_backend_cuda_init(0);
        if (!backend) throw std::runtime_error("ggml_backend_cuda_init(0) failed");
        ggml_backend_cuda_context cuda_ctx(0);
        const cudaStream_t stream = cuda_ctx.stream();

        ggml_init_params params{ggml_tensor_overhead() * 4 + 1024, nullptr, true};
        ggml_context * desc_ctx = ggml_init(params);
        if (!desc_ctx) throw std::runtime_error("ggml_init failed");
        ggml_tensor * weight = ggml_new_tensor_2d(desc_ctx, GGML_TYPE_Q6_K, tensor.n_in, tensor.n_out);
        ggml_tensor * input = ggml_new_tensor_2d(desc_ctx, GGML_TYPE_F32, tensor.n_in, 3);
        ggml_tensor * output = ggml_new_tensor_2d(desc_ctx, GGML_TYPE_F32, tensor.n_out, 3);

        const size_t x_count = size_t(tensor.n_in) * 3;
        const size_t y_count = size_t(tensor.n_out) * 3;
        const size_t q8_bytes = x_count / 32 * 36; // n_in % 512 == 0: no GGML column padding.
        DeviceBuffer d_weight(tensor.bytes.size());
        DeviceBuffer d_input(x_count * sizeof(float));
        DeviceBuffer d_q8(q8_bytes);
        DeviceBuffer d_ggml(y_count * sizeof(float));
        DeviceBuffer d_native(y_count * sizeof(float));
        std::vector<float> host_input(x_count);
        for (size_t i = 0; i < x_count; ++i) {
            host_input[i] = 0.1f * std::sin(float(i) * 0.017f) +
                            0.03f * std::cos(float(i) * 0.071f);
        }
        check(cudaMemcpyAsync(d_weight.ptr, tensor.bytes.data(), tensor.bytes.size(), cudaMemcpyHostToDevice, stream), "copy Q6_K");
        check(cudaMemcpyAsync(d_input.ptr, host_input.data(), x_count * sizeof(float), cudaMemcpyHostToDevice, stream), "copy F32 input");

        EventPair events;
        for (int ncols : {1, 2, 3}) {
            const auto quantize = [&] {
                quantize_row_q8_1_cuda(static_cast<const float *>(d_input.ptr), nullptr, d_q8.ptr,
                    GGML_TYPE_Q6_K, tensor.n_in, tensor.n_in, tensor.n_in * ncols,
                    tensor.n_in * ncols, tensor.n_in, ncols, 1, 1, stream);
            };
            const auto ggml = [&] {
                ggml_cuda_op_mul_mat_vec_q(cuda_ctx, weight, input, output,
                    static_cast<const char *>(d_weight.ptr), static_cast<const float *>(d_input.ptr),
                    static_cast<const char *>(d_q8.ptr), static_cast<float *>(d_ggml.ptr),
                    0, tensor.n_out, ncols, tensor.n_in, stream);
            };
            const auto native = [&] {
                strata::kernels::native_q6_k_mmvq(d_weight.ptr, d_q8.ptr,
                    static_cast<float *>(d_native.ptr), tensor.n_in, tensor.n_out, ncols, stream);
            };
            quantize();
            ggml();
            native();
            check(cudaStreamSynchronize(stream), "initial parity sync");
            const size_t output_bytes = size_t(tensor.n_out) * ncols * sizeof(float);
            std::vector<float> reference(size_t(tensor.n_out) * ncols);
            std::vector<float> candidate(reference.size());
            check(cudaMemcpy(reference.data(), d_ggml.ptr, output_bytes, cudaMemcpyDeviceToHost), "copy GGML output");
            check(cudaMemcpy(candidate.data(), d_native.ptr, output_bytes, cudaMemcpyDeviceToHost), "copy native output");
            compare(reference, candidate, tensor.n_out, ncols);

            for (int i = 0; i < 10; ++i) { quantize(); ggml(); native(); }
            check(cudaStreamSynchronize(stream), "warmup sync");
            std::vector<double> q_us, ggml_us, native_us;
            q_us.reserve(repeats); ggml_us.reserve(repeats); native_us.reserve(repeats);
            for (int i = 0; i < repeats; ++i) {
                q_us.push_back(events.time_us(stream, quantize));
                if (i & 1) {
                    native_us.push_back(events.time_us(stream, native));
                    ggml_us.push_back(events.time_us(stream, ggml));
                } else {
                    ggml_us.push_back(events.time_us(stream, ggml));
                    native_us.push_back(events.time_us(stream, native));
                }
            }
            std::printf("ncols=%d cuda_event_median_us q8_quant=%.3f ggml_dot=%.3f native_dot=%.3f (repeats=%d)\n",
                        ncols, median(q_us), median(ggml_us), median(native_us), repeats);
        }
        ggml_free(desc_ctx);
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "native_q6_gguf_parity: %s\n", e.what());
        return 1;
    }
}
