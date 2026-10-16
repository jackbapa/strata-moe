#include "strata_expert_plan_parallel.cuh"
#include <array>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t result, const char * operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

template<class T> struct Device {
    T * p = nullptr;
    explicit Device(size_t n) { check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc"); }
    ~Device() { if (p) cudaFree(p); }
    Device(const Device &) = delete;
    Device & operator=(const Device &) = delete;
};

struct Plan {
    std::array<unsigned long long, 3 * 64> pointers{};
    std::array<int32_t, 65> starts{};
    std::array<int32_t, 64> destinations{};
    std::array<int32_t, 64> input_rows{};
    int32_t count = 0;
};

Plan reference(const std::vector<int32_t> & ids, int routes, int tokens,
               int stride, int input_lanes, const uint8_t * gate,
               const uint8_t * up, const uint8_t * down,
               size_t gate_stride, size_t up_stride, size_t down_stride) {
    Plan plan;
    bool consumed[64]{};
    const int n = routes * tokens;
    int entry = 0;
    for (int i = 0; i < n; ++i) {
        if (consumed[i]) continue;
        const int e = ids[(i / routes) * stride + i % routes];
        const int group = plan.count++;
        plan.starts[group] = entry;
        plan.pointers[3 * group]     = (unsigned long long) (gate + size_t(e) * gate_stride);
        plan.pointers[3 * group + 1] = (unsigned long long) (up   + size_t(e) * up_stride);
        plan.pointers[3 * group + 2] = (unsigned long long) (down + size_t(e) * down_stride);
        for (int j = i; j < n; ++j) {
            if (ids[(j / routes) * stride + j % routes] != e) continue;
            consumed[j] = true;
            plan.destinations[entry] = j;
            plan.input_rows[entry] = (j / routes) * input_lanes + (input_lanes == 1 ? 0 : j % routes);
            ++entry;
        }
        plan.starts[group + 1] = entry;
    }
    return plan;
}

template<class T, size_t N> void equal(const std::array<T, N> & expected,
                                      const std::array<T, N> & actual, int used,
                                      const char * field, int case_id) {
    for (int i = 0; i < used; ++i)
        if (expected[i] != actual[i])
            throw std::runtime_error(std::string(field) + " mismatch in case " +
                                     std::to_string(case_id) + " index " + std::to_string(i));
}
}

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
            std::puts("SKIP: parallel expert plan needs a CUDA GPU");
            return 77;
        }
        check(cudaSetDevice(0), "cudaSetDevice");
        Device<int32_t> d_ids(8 * 11), d_starts(65), d_destinations(64), d_input_rows(64), d_count(1);
        Device<unsigned long long> d_pointers(3 * 64), d_calls(1);
        constexpr size_t gate_stride = 97, up_stride = 113, down_stride = 127;
        Device<uint8_t> gate(256 * gate_stride), up(256 * up_stride), down(256 * down_stride);
        check(cudaMemset(d_calls.p, 0, sizeof(unsigned long long)), "clear calls");
        int cases = 0;
        for (int tokens : {1, 2, 3, 4, 8})
        for (int input_lanes : {1, 8})
        for (int stride : {8, 11})
        for (int experts : {16, 256})
        for (int pattern = 0; pattern < 5; ++pattern) {
            std::vector<int32_t> ids(size_t(tokens * stride), -1);
            uint32_t state = 0x9E3779B9u + uint32_t(cases);
            for (int i = 0; i < tokens * 8; ++i) {
                const int t = i / 8, k = i % 8;
                state ^= state << 13; state ^= state >> 17; state ^= state << 5;
                int e = 0;
                switch (pattern) {
                    case 0: e = 3; break;
                    case 1: e = i % experts; break;
                    case 2: e = (i * 7 + t * 3) % experts; break;
                    case 3: e = ((i / 2) * 3) % experts; break;
                    default: e = int(state % uint32_t(experts)); break;
                }
                ids[t * stride + k] = e;
            }
            check(cudaMemcpy(d_ids.p, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "upload ids");
            check(cudaMemset(d_pointers.p, 0xA5, 3 * 64 * sizeof(unsigned long long)), "clear pointers");
            check(cudaMemset(d_starts.p, 0xA5, 65 * sizeof(int32_t)), "clear starts");
            check(cudaMemset(d_destinations.p, 0xA5, 64 * sizeof(int32_t)), "clear destinations");
            check(cudaMemset(d_input_rows.p, 0xA5, 64 * sizeof(int32_t)), "clear input rows");
            check(cudaMemset(d_count.p, 0xA5, sizeof(int32_t)), "clear count");
            strata_a3b::detail::plan_expert_groups_parallel<<<1, 64>>>(
                d_ids.p, size_t(stride) * sizeof(int32_t), 8, tokens, input_lanes,
                gate.p, up.p, down.p, gate_stride, up_stride, down_stride, experts,
                d_pointers.p, d_starts.p, d_destinations.p, d_input_rows.p, d_count.p, d_calls.p);
            check(cudaGetLastError(), "launch parallel plan");
            check(cudaDeviceSynchronize(), "run parallel plan");
            Plan actual;
            check(cudaMemcpy(actual.pointers.data(), d_pointers.p, sizeof(actual.pointers), cudaMemcpyDeviceToHost), "read pointers");
            check(cudaMemcpy(actual.starts.data(), d_starts.p, sizeof(actual.starts), cudaMemcpyDeviceToHost), "read starts");
            check(cudaMemcpy(actual.destinations.data(), d_destinations.p, sizeof(actual.destinations), cudaMemcpyDeviceToHost), "read destinations");
            check(cudaMemcpy(actual.input_rows.data(), d_input_rows.p, sizeof(actual.input_rows), cudaMemcpyDeviceToHost), "read input rows");
            check(cudaMemcpy(&actual.count, d_count.p, sizeof(actual.count), cudaMemcpyDeviceToHost), "read count");
            const Plan expected = reference(ids, 8, tokens, stride, input_lanes,
                                            gate.p, up.p, down.p, gate_stride, up_stride, down_stride);
            if (actual.count != expected.count)
                throw std::runtime_error("group count mismatch in case " + std::to_string(cases));
            equal(expected.pointers, actual.pointers, 3 * expected.count, "pointer", cases);
            equal(expected.starts, actual.starts, expected.count + 1, "start", cases);
            equal(expected.destinations, actual.destinations, tokens * 8, "destination", cases);
            equal(expected.input_rows, actual.input_rows, tokens * 8, "input row", cases);
            ++cases;
        }
        unsigned long long calls = 0;
        check(cudaMemcpy(&calls, d_calls.p, sizeof(calls), cudaMemcpyDeviceToHost), "read calls");
        if (calls != (unsigned long long) cases) throw std::runtime_error("call counter mismatch");
        std::printf("PASS parallel expert plan: %d exact CPU/GPU cases, call count %llu\n", cases, calls);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL parallel expert plan: %s\n", error.what());
        return 1;
    }
}
