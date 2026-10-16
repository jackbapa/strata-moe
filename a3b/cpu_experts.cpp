// Design adapted from Strata's MIT-licensed native_expert.cpp and pool.cpp:
// shared quantized input, expert grouping, and Gate/Up -> quantize -> Down
// row phases. Geometry and three weight views are independent of its pack.
#include "cpu_experts.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#elif defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace strata_a3b {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}
void pause_thread() {
#if defined(_M_X64) || defined(_M_IX86) || defined(__i386__) || defined(__x86_64__)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}
struct Core { unsigned group = 0, processor = 0; };

// Select a single allowed logical processor per physical core. In particular,
// processor IDs 0..N-1 must not be mistaken for N independent physical cores.
std::vector<Core> physical_cores() {
    std::vector<Core> result;
#if defined(_WIN32)
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    if (bytes) {
        std::vector<unsigned char> storage(bytes);
        auto * first = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(storage.data());
        if (GetLogicalProcessorInformationEx(RelationProcessorCore, first, &bytes)) {
            GROUP_AFFINITY caller_group{};
            GetThreadGroupAffinity(GetCurrentThread(), &caller_group);
            DWORD_PTR process_mask = 0, system_mask = 0;
            const bool have_mask = GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) != 0;
            for (size_t offset = 0; offset < bytes;) {
                auto * info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(storage.data() + offset);
                if (!info->Size || info->Size > bytes - offset) break;
                if (info->Relationship == RelationProcessorCore) {
                    bool found = false;
                    for (WORD g = 0; g < info->Processor.GroupCount && !found; ++g) {
                        const auto & group = info->Processor.GroupMask[g];
                        KAFFINITY mask = group.Mask;
                        if (have_mask && group.Group == caller_group.Group) mask &= process_mask;
                        for (unsigned p = 0; p < sizeof(KAFFINITY) * 8; ++p) {
                            if (mask & (KAFFINITY(1) << p)) {
                                result.push_back({group.Group, p}); found = true; break;
                            }
                        }
                    }
                }
                offset += info->Size;
            }
        }
    }
#elif defined(__linux__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
        std::map<std::pair<int, int>, unsigned> selected;
        for (unsigned p = 0; p < CPU_SETSIZE; ++p) if (CPU_ISSET(p, &allowed)) {
            int package = 0, core = int(p);
            const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(p) + "/topology/";
            std::ifstream(base + "physical_package_id") >> package;
            std::ifstream(base + "core_id") >> core;
            selected.emplace(std::make_pair(package, core), p);
        }
        for (const auto & entry : selected) result.push_back({0, entry.second});
    }
#endif
    if (result.empty()) {
        const unsigned count = std::max(1u, std::thread::hardware_concurrency());
        for (unsigned p = 0; p < count; ++p) result.push_back({0, p});
    }
    return result;
}

// Restore the caller's affinity after the complete expert phase. Workers use
// the same helper for their lifetime; no affinity calls occur for each row.
class Affinity {
public:
    explicit Affinity(const Core * core) {
        if (!core) return;
#if defined(_WIN32)
        GROUP_AFFINITY desired{};
        desired.Group = WORD(core->group);
        desired.Mask = KAFFINITY(1) << core->processor;
        valid_ = SetThreadGroupAffinity(GetCurrentThread(), &desired, &previous_) != 0;
#elif defined(__linux__)
        cpu_set_t desired;
        CPU_ZERO(&desired);
        if (core->processor >= CPU_SETSIZE) return;
        CPU_SET(core->processor, &desired);
        if (pthread_getaffinity_np(pthread_self(), sizeof(previous_), &previous_) == 0)
            valid_ = pthread_setaffinity_np(pthread_self(), sizeof(desired), &desired) == 0;
#endif
    }
    ~Affinity() {
        if (!valid_) return;
#if defined(_WIN32)
        SetThreadGroupAffinity(GetCurrentThread(), &previous_, nullptr);
#elif defined(__linux__)
        pthread_setaffinity_np(pthread_self(), sizeof(previous_), &previous_);
#endif
    }
private:
    bool valid_ = false;
#if defined(_WIN32)
    GROUP_AFFINITY previous_{};
#elif defined(__linux__)
    cpu_set_t previous_{};
#endif
};

size_t checked_product(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw std::runtime_error("CPU expert buffer size overflow");
    return a * b;
}
} // namespace

struct CpuExpertPool::Impl {
    struct Group { int expert; std::vector<size_t> jobs; };
    struct InputKey { int token, lane; };
    Config config;
    std::vector<Core> cores;
    int count = 1;
    std::vector<std::thread> workers;
    std::mutex wake_mutex, finish_mutex, caller_mutex;
    std::condition_variable wake_cv, finish_cv;
    alignas(64) std::atomic<uint64_t> epoch{0};
    alignas(64) std::atomic<size_t> next{0};
    alignas(64) std::atomic<int> remaining{0};
    std::atomic<bool> stop{false};
    int phase = 0;
    size_t rows = 0, tasks = 0;
    CpuExpertWeights weights{};
    float * output = nullptr;
    ggml_from_float_t quantize_input = nullptr, quantize_hidden = nullptr;
    ggml_vec_dot_t dot_gate = nullptr, dot_up = nullptr, dot_down = nullptr;
    size_t input_bytes = 0, hidden_bytes = 0;
    std::vector<uint8_t> quantized_input, quantized_hidden;
    std::vector<float> hidden;
    std::vector<size_t> input_index;
    std::vector<Group> groups;
    size_t group_count = 0;
    std::vector<InputKey> input_keys;
    std::vector<int> group_index;
    CpuExpertPoolStats counters{};

    explicit Impl(Config value) : config(value), cores(::strata_a3b::physical_cores()) {
        if (config.threads < 0 || config.threads > 4096 || config.tasks < 0 || config.tasks > 4096 || config.spin_us < 0)
            throw std::runtime_error("invalid CPU expert pool configuration");
        // Do not silently put two compute workers on one physical core.
        count = config.threads ? config.threads : int(cores.size());
        if (config.pin) count = std::min(count, int(cores.size()));
        count = std::max(count, 1);
        ggml_cpu_init();
        try {
            for (int i = 1; i < count; ++i) workers.emplace_back([this, i] { worker(i); });
        } catch (...) {
            shutdown(); throw;
        }
    }
    ~Impl() { shutdown(); }
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(wake_mutex);
            stop.store(true, std::memory_order_release);
        }
        wake_cv.notify_all();
        for (auto & worker_thread : workers) if (worker_thread.joinable()) worker_thread.join();
    }
    void worker(int index) {
        Affinity affinity(config.pin ? &cores[size_t(index)] : nullptr);
        uint64_t seen = 0;
        while (!stop.load(std::memory_order_acquire)) {
            const auto deadline = Clock::now() + std::chrono::microseconds(config.spin_us);
            uint64_t observed = epoch.load(std::memory_order_acquire);
            while (observed == seen && !stop.load(std::memory_order_acquire) && Clock::now() < deadline) {
                pause_thread(); observed = epoch.load(std::memory_order_acquire);
            }
            if (observed == seen) {
                std::unique_lock<std::mutex> lock(wake_mutex);
                wake_cv.wait(lock, [&] { return stop.load(std::memory_order_acquire) || epoch.load(std::memory_order_acquire) != seen; });
                observed = epoch.load(std::memory_order_acquire);
            }
            if (stop.load(std::memory_order_acquire)) break;
            seen = observed;
            drain();
            // A phase is complete only after every worker has stopped reading
            // its description. This also covers late wakeups and empty claims.
            if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lock(finish_mutex);
                finish_cv.notify_one();
            }
        }
    }
    void run_phase(int which, int width) {
        phase = which;
        rows = checked_product(group_count, size_t(width));
        tasks = std::min(rows, size_t(config.tasks ? config.tasks : 3 * count));
        next.store(0, std::memory_order_relaxed);
        remaining.store(int(workers.size()), std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(wake_mutex);
            epoch.fetch_add(1, std::memory_order_release);
        }
        wake_cv.notify_all();
        drain();
        if (!workers.empty()) {
            std::unique_lock<std::mutex> lock(finish_mutex);
            finish_cv.wait(lock, [&] { return remaining.load(std::memory_order_acquire) == 0; });
        }
    }
    void drain() {
        for (;;) {
            const size_t task = next.fetch_add(1, std::memory_order_relaxed);
            if (task >= tasks) return;
            const size_t begin = rows * task / tasks, end = rows * (task + 1) / tasks;
            const int width = phase == 1 ? weights.ffn : weights.hidden;
            for (size_t row = begin; row < end;) {
                const Group & group = groups[row / size_t(width)];
                const size_t first = row % size_t(width);
                const size_t last = std::min(size_t(width), first + end - row);
                const auto * gate = static_cast<const uint8_t *>(weights.gate.data) + size_t(group.expert) * weights.gate.expert_stride;
                const auto * up = static_cast<const uint8_t *>(weights.up.data) + size_t(group.expert) * weights.up.expert_stride;
                const auto * down = static_cast<const uint8_t *>(weights.down.data) + size_t(group.expert) * weights.down.expert_stride;
                for (size_t r = first; r < last; ++r) for (size_t job : group.jobs) {
                    if (phase == 1) {
                        const auto * act = quantized_input.data() + input_index[job] * input_bytes;
                        float g = 0.f, u = 0.f;
                        dot_gate(weights.hidden, &g, 0, gate + r * weights.gate.row_stride, 0, act, 0, 1);
                        dot_up(weights.hidden, &u, 0, up + r * weights.up.row_stride, 0, act, 0, 1);
                        hidden[job * size_t(weights.ffn) + r] = (g / (1.f + std::exp(-g))) * u;
                    } else {
                        const auto * act = quantized_hidden.data() + job * hidden_bytes;
                        dot_down(weights.ffn, &output[job * size_t(weights.hidden) + r], 0,
                            down + r * weights.down.row_stride, 0, act, 0, 1);
                    }
                }
                row += last - first;
            }
        }
    }
    void execute(CpuExpertWeights w, const CpuExpertInput & input,
                 const std::vector<CpuExpertJob> & requested, std::vector<float> & outputs) {
        std::lock_guard<std::mutex> serial(caller_mutex);
        if (requested.empty()) { outputs.clear(); return; }
        if (w.hidden < 1 || w.ffn < 1 || w.experts < 1 ||
            input.tokens < 1 || input.tokens > 256 || input.lanes < 1)
            throw std::runtime_error("CPU expert pool requires valid geometry and 1..256 tokens");
        const auto * gt = ggml_get_type_traits_cpu(w.gate.type);
        const auto * ut = ggml_get_type_traits_cpu(w.up.type);
        const auto * dt = ggml_get_type_traits_cpu(w.down.type);
        if (!gt || !ut || !dt || !gt->vec_dot || !ut->vec_dot || !dt->vec_dot ||
            gt->vec_dot_type != ut->vec_dot_type ||
            w.hidden % ggml_blck_size(w.gate.type) || w.hidden % ggml_blck_size(w.up.type) ||
            w.ffn % ggml_blck_size(w.down.type))
            throw std::runtime_error("CPU expert pool requires supported quantized types and shared gate/up activation type");
        const auto * input_traits = ggml_get_type_traits_cpu(gt->vec_dot_type);
        const auto * hidden_traits = ggml_get_type_traits_cpu(dt->vec_dot_type);
        if (!input_traits || !hidden_traits || !input_traits->from_float || !hidden_traits->from_float)
            throw std::runtime_error("CPU expert pool has no activation quantizer for GGUF types");
        dot_gate = gt->vec_dot; dot_up = ut->vec_dot; dot_down = dt->vec_dot;
        quantize_input = input_traits->from_float; quantize_hidden = hidden_traits->from_float;
        const size_t gate_row = ggml_row_size(w.gate.type, w.hidden);
        const size_t up_row = ggml_row_size(w.up.type, w.hidden);
        const size_t down_row = ggml_row_size(w.down.type, w.ffn);
        for (auto * projection : {&w.gate, &w.up, &w.down}) {
            const bool is_down = projection == &w.down;
            const size_t row_bytes = is_down ? down_row : projection == &w.up ? up_row : gate_row;
            const int nrows = is_down ? w.hidden : w.ffn;
            if (!projection->data) throw std::runtime_error("CPU expert pool has null weights");
            if (!projection->row_stride) projection->row_stride = row_bytes;
            if (!projection->expert_stride) projection->expert_stride = checked_product(projection->row_stride, size_t(nrows));
            if (projection->row_stride < row_bytes || projection->expert_stride < checked_product(projection->row_stride, size_t(nrows)))
                throw std::runtime_error("CPU expert weight strides overlap rows or experts");
            checked_product(projection->expert_stride, size_t(w.experts));
        }
        if (!input.data) throw std::runtime_error("CPU expert pool has null input");
        const size_t lane_stride = input.lane_stride ? input.lane_stride : size_t(w.hidden) * sizeof(float);
        const size_t token_stride = input.token_stride ? input.token_stride : checked_product(lane_stride, size_t(input.lanes));
        if (lane_stride < size_t(w.hidden) * sizeof(float) || token_stride < checked_product(lane_stride, size_t(input.lanes)) ||
            lane_stride % alignof(float) || token_stride % alignof(float))
            throw std::runtime_error("CPU expert input strides overlap or misalign rows");
        checked_product(token_stride, size_t(input.tokens));
        group_count = 0; input_keys.clear(); input_index.resize(requested.size());
        group_index.resize(size_t(w.experts));
        std::fill(group_index.begin(), group_index.end(), -1);
        for (size_t j = 0; j < requested.size(); ++j) {
            const auto & job = requested[j];
            if (job.expert < 0 || job.expert >= w.experts || job.token < 0 || job.token >= input.tokens || job.lane < 0)
                throw std::runtime_error("CPU expert job is outside model/input geometry");
            const InputKey key{job.token, job.lane % input.lanes};
            size_t index = 0;
            while (index < input_keys.size() &&
                   (input_keys[index].token != key.token || input_keys[index].lane != key.lane)) ++index;
            if (index == input_keys.size()) input_keys.push_back(key);
            input_index[j] = index;
            int & group = group_index[size_t(job.expert)];
            if (group < 0) {
                group = int(group_count++);
                if (groups.size() < group_count) groups.push_back({job.expert, {}});
                groups[size_t(group)].expert = job.expert;
                groups[size_t(group)].jobs.clear();
            }
            groups[size_t(group)].jobs.push_back(j);
        }
        weights = w;
        input_bytes = ggml_row_size(gt->vec_dot_type, w.hidden);
        hidden_bytes = ggml_row_size(dt->vec_dot_type, w.ffn);
        quantized_input.resize(checked_product(input_keys.size(), input_bytes));
        quantized_hidden.resize(checked_product(requested.size(), hidden_bytes));
        hidden.resize(checked_product(requested.size(), size_t(w.ffn)));
        outputs.resize(checked_product(requested.size(), size_t(w.hidden))); output = outputs.data();
        Affinity affinity(config.pin ? &cores.front() : nullptr);
        const auto a = Clock::now();
        for (size_t index = 0; index < input_keys.size(); ++index) {
            const auto & key = input_keys[index];
            const auto * source = reinterpret_cast<const float *>(reinterpret_cast<const uint8_t *>(input.data) +
                size_t(key.token) * token_stride + size_t(key.lane) * lane_stride);
            quantize_input(source, quantized_input.data() + index * input_bytes, w.hidden);
        }
        const auto b = Clock::now();
        run_phase(1, w.ffn);
        const auto c = Clock::now();
        for (size_t j = 0; j < requested.size(); ++j)
            quantize_hidden(hidden.data() + j * size_t(w.ffn), quantized_hidden.data() + j * hidden_bytes, w.ffn);
        const auto d = Clock::now();
        run_phase(2, w.hidden);
        const auto e = Clock::now();
        ++counters.calls; counters.jobs += requested.size();
        counters.unique_inputs += input_keys.size(); counters.unique_experts += group_count;
        counters.input_quant_ms += elapsed_ms(a, b); counters.gate_up_ms += elapsed_ms(b, c);
        counters.hidden_quant_ms += elapsed_ms(c, d); counters.down_ms += elapsed_ms(d, e);
        output = nullptr;
    }
};

CpuExpertPool::CpuExpertPool() : CpuExpertPool(Config{}) {}
CpuExpertPool::CpuExpertPool(Config config) : impl_(std::make_unique<Impl>(config)) {}
CpuExpertPool::~CpuExpertPool() = default;
void CpuExpertPool::execute(const CpuExpertWeights & weights, const CpuExpertInput & input,
                            const std::vector<CpuExpertJob> & jobs, std::vector<float> & outputs) {
    impl_->execute(weights, input, jobs, outputs);
}
int CpuExpertPool::threads() const noexcept { return impl_->count; }
int CpuExpertPool::physical_cores() const noexcept { return int(impl_->cores.size()); }
CpuExpertPoolStats CpuExpertPool::stats() const noexcept { return impl_->counters; }
void CpuExpertPool::reset_stats() noexcept { impl_->counters = {}; }
} // namespace strata_a3b
