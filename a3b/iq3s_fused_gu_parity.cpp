#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace K = strata::kernels;

static void check(cudaError_t result, const char * what) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(result));
}

template <class T> static T * device_copy(const std::vector<T> & host) {
    T * result = nullptr;
    check(cudaMalloc(&result, host.size() * sizeof(T)), "allocate");
    check(cudaMemcpy(result, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "copy to GPU");
    return result;
}

static void layer_parity(const strata::GgufFile & file, int layer, bool benchmark, bool full_grid_parity) {
    const strata::TensorInfo * t[3]{};
    const char * role[3] = {"gate", "up", "down"};
    for (const auto & item : file.tensors()) {
        for (int r = 0; r < 3; ++r)
            if (item.name == "blk." + std::to_string(layer) + ".ffn_" + role[r] + "_exps.weight") t[r] = &item;
    }
    if (!t[0] || !t[1] || !t[2]) throw std::runtime_error("missing expert tensors");
    const int H = static_cast<int>(t[0]->shape[0]), FF = static_cast<int>(t[0]->shape[1]);
    if (H != 2048 || FF != 512 || t[0]->type != 22 || (t[2]->type != 21 && t[2]->type != 23))
        throw std::runtime_error("unexpected IQ3_S expert geometry");
    const auto layout = K::native_expert_layout(t[0]->type, t[2]->type, H, FF);
    const int groups = full_grid_parity ? 24 : 8;
    constexpr int input_tokens = 8;
    const int tokens = full_grid_parity ? 1 + layer % 2 : 1 + layer % 4;
    // The grid test uses 24 groups and at most 48 entries, within the A3B
    // adapter's 64-route limit.  The existing eight-group test keeps 1..4.
    const int entries = groups * tokens;
    const size_t gu_bytes = static_cast<size_t>(FF) * layout.gu_row;
    const size_t down_bytes = static_cast<size_t>(H) * layout.d_row;
    std::vector<uint8_t> gate(groups * gu_bytes), up(groups * gu_bytes), down(groups * down_bytes);
    std::vector<uint8_t> packed(groups * layout.bytes);
    for (int g = 0; g < groups; ++g) {
        const size_t e = static_cast<size_t>((g * 37 + 5) % t[0]->shape[2]);
        std::memcpy(gate.data() + g * gu_bytes, file.tensor_data(*t[0]) + e * gu_bytes, gu_bytes);
        std::memcpy(up.data() + g * gu_bytes, file.tensor_data(*t[1]) + e * gu_bytes, gu_bytes);
        std::memcpy(down.data() + g * down_bytes, file.tensor_data(*t[2]) + e * down_bytes, down_bytes);
        auto * dst = packed.data() + g * layout.bytes;
        std::memcpy(dst, gate.data() + g * gu_bytes, gu_bytes);
        std::memcpy(dst + layout.up_off, up.data() + g * gu_bytes, gu_bytes);
        std::memcpy(dst + layout.down_off, down.data() + g * down_bytes, down_bytes);
    }
    auto * dgate = device_copy(gate);
    auto * dup = device_copy(up);
    auto * ddown = device_copy(down);
    auto * dpacked = device_copy(packed);
    std::vector<unsigned long long> packed_ptr(groups), split_ptr(3 * groups);
    std::vector<int32_t> starts(groups + 1), destinations(entries), input_rows(entries);
    for (int g = 0; g < groups; ++g) {
        packed_ptr[g] = reinterpret_cast<unsigned long long>(dpacked + g * layout.bytes);
        split_ptr[3 * g] = reinterpret_cast<unsigned long long>(dgate + g * gu_bytes);
        split_ptr[3 * g + 1] = reinterpret_cast<unsigned long long>(dup + g * gu_bytes);
        split_ptr[3 * g + 2] = reinterpret_cast<unsigned long long>(ddown + g * down_bytes);
        starts[g] = g * tokens;
        for (int j = 0; j < tokens; ++j) {
            const int entry = g * tokens + j;
            destinations[entry] = entry;
            input_rows[entry] = (g + j * 3) % input_tokens;
        }
    }
    starts[groups] = entries;
    auto * dp = device_copy(packed_ptr);
    auto * ds = device_copy(split_ptr);
    auto * dstarts = device_copy(starts);
    auto * ddst = device_copy(destinations);
    auto * drows = device_copy(input_rows);
    const std::vector<int32_t> count{groups};
    auto * dcount = device_copy(count);
    std::vector<float> x(input_tokens * H);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(static_cast<float>(i) * 0.017f);
    auto * dx = device_copy(x);
    void * dxq = nullptr, * scratch = nullptr;
    float * dy = nullptr;
    check(cudaMalloc(&dxq, static_cast<size_t>(input_tokens) * H / 32 * 36), "q8 inputs");
    check(cudaMalloc(&scratch, K::native_expert_scratch_bytes(entries, FF)), "scratch");
    check(cudaMalloc(&dy, static_cast<size_t>(entries) * H * sizeof(float)), "outputs");
    cudaStream_t stream;
    check(cudaStreamCreate(&stream), "stream");
    K::quantize_q8_1_rows(dx, input_tokens, H, dxq, stream);
    auto split = layout;
    split.split_storage = true;
    // The grid test holds the weights, routes, and arithmetic path constant:
    // only eight serialized group rows versus one block row per resident group.
    K::native_expert_grouped(full_grid_parity ? split : layout, full_grid_parity ? ds : dp,
                             dstarts, dcount, ddst, drows, groups, entries,
                             dxq, scratch, dy, stream, full_grid_parity ? 8 : groups);
    std::vector<float> reference(static_cast<size_t>(entries) * H), actual(reference.size());
    check(cudaMemcpyAsync(reference.data(), dy, reference.size() * sizeof(float), cudaMemcpyDeviceToHost, stream), "read baseline");
    K::native_expert_grouped(split, ds, dstarts, dcount, ddst, drows, groups, entries,
                             dxq, scratch, dy, stream, full_grid_parity ? 0 : groups);
    check(cudaMemcpyAsync(actual.data(), dy, actual.size() * sizeof(float), cudaMemcpyDeviceToHost, stream), "read fused");
    check(cudaStreamSynchronize(stream), "finish");
    size_t differences = 0;
    for (size_t i = 0; i < actual.size(); ++i)
        differences += std::memcmp(&actual[i], &reference[i], sizeof(float)) != 0;
    std::printf("layer %d %s/%s%s: %zu/%zu differing floats\n", layer,
                 t[0]->type_name(), t[2]->type_name(),
                 full_grid_parity ? " full-grid" : "", differences, actual.size());
    if (benchmark) {
        cudaEvent_t start, end;
        check(cudaEventCreate(&start), "start event");
        check(cudaEventCreate(&end), "end event");
        const auto measure = [&](const K::NativeExpertLayout & use, const unsigned long long * pointers) {
            for (int i = 0; i < 20; ++i)
                K::native_expert_grouped(use, pointers, dstarts, dcount, ddst, drows, groups, entries,
                                         dxq, scratch, dy, stream, groups);
            check(cudaEventRecord(start, stream), "record start");
            for (int i = 0; i < 200; ++i)
                K::native_expert_grouped(use, pointers, dstarts, dcount, ddst, drows, groups, entries,
                                         dxq, scratch, dy, stream, groups);
            check(cudaEventRecord(end, stream), "record end");
            check(cudaEventSynchronize(end), "wait for benchmark");
            float milliseconds = 0;
            check(cudaEventElapsedTime(&milliseconds, start, end), "elapsed time");
            return 1000.0f * milliseconds / 200.0f;
        };
        float packed_us[3], split_us[3];
        for (int round = 0; round < 3; ++round) {
            if (round & 1) {
                split_us[round] = measure(split, ds);
                packed_us[round] = measure(layout, dp);
            } else {
                packed_us[round] = measure(layout, dp);
                split_us[round] = measure(split, ds);
            }
        }
        std::sort(packed_us, packed_us + 3);
        std::sort(split_us, split_us + 3);
        std::printf("layer %d tokens/group %d: packed %.2f, split %.2f us/call "
                    "(paired median, 3 x 200 calls each, CUDA events)\n",
                    layer, tokens, packed_us[1], split_us[1]);
        cudaEventDestroy(end); cudaEventDestroy(start);
    }
    check(cudaStreamDestroy(stream), "destroy stream");
    cudaFree(dy); cudaFree(scratch); cudaFree(dxq); cudaFree(dx); cudaFree(dcount);
    cudaFree(drows); cudaFree(ddst); cudaFree(dstarts); cudaFree(ds); cudaFree(dp);
    cudaFree(dpacked); cudaFree(ddown); cudaFree(dup); cudaFree(dgate);
    if (differences) throw std::runtime_error("split fused expert differs from original grouped expert");
}

int main(int argc, char ** argv) {
    try {
        const bool benchmark = argc == 3 && std::strcmp(argv[2], "--bench") == 0;
        const bool full_grid_parity = argc == 3 && std::strcmp(argv[2], "--full-grid-parity") == 0;
        if (argc != 2 && !benchmark && !full_grid_parity)
            throw std::runtime_error("usage: parity <IQ3_S.gguf> [--bench|--full-grid-parity]");
        if (!std::filesystem::is_regular_file(argv[1])) {
            std::puts("SKIP: local IQ3_S GGUF unavailable");
            return 77;
        }
        const char * enabled = std::getenv("STRATA_A3B_FUSED_GU");
        const bool fused = enabled && enabled[0] == '1';
        const char * down = std::getenv("STRATA_A3B_DOWN_MODE");
        std::printf("STRATA_A3B_FUSED_GU=%d STRATA_A3B_DOWN_MODE=%d\n",
                    fused ? 1 : 0, down ? std::atoi(down) : 0);
        const strata::GgufFile file(argv[1]);
        for (int layer = 0; layer < 40; ++layer)
            layer_parity(file, layer, benchmark && (layer == 0 || layer == 1 || layer == 19 || layer == 39),
                         full_grid_parity);
        if (full_grid_parity)
            std::puts("PASS: eight-row and full-grid Strata expert launches match bitwise");
        else
            std::printf("PASS: packed original and separate-tensor %s experts match bitwise\n",
                        fused ? "fused" : "original");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
