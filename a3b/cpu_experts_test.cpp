#include "cpu_experts.hpp"
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace strata_a3b;
namespace {
using Clock = std::chrono::steady_clock;
const ggml_type_traits_cpu * traits(ggml_type type = GGML_TYPE_Q8_0) { return ggml_get_type_traits_cpu(type); }
void require(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
struct Fixture {
    CpuExpertWeights weights;
    std::vector<uint8_t> gate, up, down;
    Fixture(int hidden, int ffn, int experts, bool padded,
            ggml_type gate_type = GGML_TYPE_Q8_0, ggml_type down_type = GGML_TYPE_Q8_0) {
        weights.hidden = hidden; weights.ffn = ffn; weights.experts = experts;
        weights.gate.type = weights.up.type = gate_type;
        weights.down.type = down_type;
        std::mt19937 random(0x51a7);
        std::uniform_real_distribution<float> values(-0.08f, 0.08f);
        auto fill = [&](std::vector<uint8_t> & storage, CpuExpertWeightView & view, int cols, int rows) {
            view.row_stride = size_t(ggml_row_size(view.type, cols)) + (padded ? 32 : 0);
            view.expert_stride = view.row_stride * size_t(rows) + (padded ? 64 : 0);
            storage.resize(view.expert_stride * size_t(experts), 0xd3);
            view.data = storage.data();
            std::vector<float> row(size_t(cols), 0.f);
            for (int e = 0; e < experts; ++e) for (int r = 0; r < rows; ++r) {
                for (float & v : row) v = values(random);
                auto * dst = storage.data() + size_t(e) * view.expert_stride + size_t(r) * view.row_stride;
                if (traits(view.type)->from_float) traits(view.type)->from_float(row.data(), dst, cols);
                else {
                    std::vector<float> importance(size_t(cols), 1.f);
                    require(ggml_quantize_chunk(view.type, row.data(), dst, 0, 1, cols, importance.data()) ==
                            size_t(ggml_row_size(view.type, cols)), "IQ fixture quantization failed");
                }
            }
        };
        fill(gate, weights.gate, hidden, ffn);
        fill(up, weights.up, hidden, ffn);
        fill(down, weights.down, ffn, hidden);
    }
};
struct Input {
    CpuExpertInput view;
    std::vector<float> data;
    Input(int hidden, int lanes, int tokens, bool padded) {
        view.lanes = lanes; view.tokens = tokens;
        view.lane_stride = size_t(hidden + (padded ? 8 : 0)) * sizeof(float);
        view.token_stride = view.lane_stride * size_t(lanes) + (padded ? 32 : 0);
        data.resize(view.token_stride * size_t(tokens) / sizeof(float), -999.f);
        view.data = data.data();
        std::mt19937 random(0x8237);
        std::uniform_real_distribution<float> values(-1.f, 1.f);
        for (int t = 0; t < tokens; ++t) for (int lane = 0; lane < lanes; ++lane) {
            float * row = data.data() + (size_t(t) * view.token_stride + size_t(lane) * view.lane_stride) / sizeof(float);
            for (int c = 0; c < hidden; ++c) row[c] = values(random);
        }
    }
};
std::vector<CpuExpertJob> jobs_for(int experts, int tokens, int topk, int round) {
    std::vector<CpuExpertJob> jobs;
    for (int t = 0; t < tokens; ++t) for (int lane = 0; lane < topk; ++lane)
        jobs.push_back({(round * 17 + lane * 3 + t / 2) % experts, t, lane});
    // Deliberately unordered: execution must preserve the original job order.
    std::reverse(jobs.begin(), jobs.end());
    return jobs;
}
std::vector<float> oracle(const CpuExpertWeights & w, const CpuExpertInput & x, const std::vector<CpuExpertJob> & jobs) {
    const auto * g = traits(w.gate.type), * u = traits(w.up.type), * d = traits(w.down.type);
    require(g->vec_dot_type == u->vec_dot_type, "oracle gate/up activation types differ");
    const auto * q_in = traits(g->vec_dot_type), * q_hidden = traits(d->vec_dot_type);
    const size_t input_bytes = size_t(ggml_row_size(g->vec_dot_type, w.hidden));
    const size_t hidden_bytes = size_t(ggml_row_size(d->vec_dot_type, w.ffn));
    const size_t lane_stride = x.lane_stride ? x.lane_stride : size_t(w.hidden) * sizeof(float);
    const size_t token_stride = x.token_stride ? x.token_stride : lane_stride * size_t(x.lanes);
    std::vector<uint8_t> a(input_bytes), hq(hidden_bytes);
    std::vector<float> hidden(size_t(w.ffn)), output(jobs.size() * size_t(w.hidden));
    for (size_t j = 0; j < jobs.size(); ++j) {
        const auto & job = jobs[j];
        const auto * in = reinterpret_cast<const float *>(reinterpret_cast<const uint8_t *>(x.data) +
            size_t(job.token) * token_stride + size_t(job.lane % x.lanes) * lane_stride);
        q_in->from_float(in, a.data(), w.hidden);
        const auto * gate = static_cast<const uint8_t *>(w.gate.data) + size_t(job.expert) * w.gate.expert_stride;
        const auto * up = static_cast<const uint8_t *>(w.up.data) + size_t(job.expert) * w.up.expert_stride;
        const auto * down = static_cast<const uint8_t *>(w.down.data) + size_t(job.expert) * w.down.expert_stride;
        for (int row = 0; row < w.ffn; ++row) {
            float gv = 0.f, uv = 0.f;
            g->vec_dot(w.hidden, &gv, 0, gate + size_t(row) * w.gate.row_stride, 0, a.data(), 0, 1);
            u->vec_dot(w.hidden, &uv, 0, up + size_t(row) * w.up.row_stride, 0, a.data(), 0, 1);
            hidden[size_t(row)] = (gv / (1.f + std::exp(-gv))) * uv;
        }
        q_hidden->from_float(hidden.data(), hq.data(), w.ffn);
        for (int row = 0; row < w.hidden; ++row)
            d->vec_dot(w.ffn, &output[j * size_t(w.hidden) + size_t(row)], 0,
                       down + size_t(row) * w.down.row_stride, 0, hq.data(), 0, 1);
    }
    return output;
}
void exact(const std::vector<float> & actual, const std::vector<float> & expected) {
    require(actual.size() == expected.size(), "CPU expert result shape mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::memcmp(&actual[i], &expected[i], sizeof(float)) || !std::isfinite(actual[i]))
            throw std::runtime_error("CPU expert oracle mismatch at " + std::to_string(i));
    }
}
template<class F> void rejected(F && action) {
    try { action(); } catch (const std::runtime_error &) { return; }
    throw std::runtime_error("invalid CPU expert input was accepted");
}
void validate() {
    int cases = 0;
    for (int geometry = 0; geometry < 2; ++geometry) for (bool padded : {false, true}) {
        const int hidden = geometry ? 2048 : 128, ffn = geometry ? 512 : 64;
        Fixture fixture(hidden, ffn, 16, padded);
        for (int threads : {1, 4}) {
            CpuExpertPool::Config cfg; cfg.threads = threads; cfg.spin_us = 100;
            CpuExpertPool pool(cfg);
            for (int tokens : {1, 2, 8}) for (int lanes : {1, 8}) {
                Input input(hidden, lanes, tokens, padded);
                auto jobs = jobs_for(16, tokens, 8, cases);
                jobs.push_back(jobs.front()); // Duplicate expert/token/lane must produce another output row.
                const auto expected = oracle(fixture.weights, input.view, jobs);
                std::vector<float> actual;
                const auto before = pool.stats();
                pool.execute(fixture.weights, input.view, jobs, actual);
                exact(actual, expected);
                const auto after = pool.stats();
                require(after.jobs - before.jobs == jobs.size(), "CPU expert job counter mismatch");
                require(after.unique_inputs - before.unique_inputs == size_t(tokens * lanes), "input quantization was not shared");
                require(after.calls - before.calls == 1, "CPU expert call counter mismatch");
                // A lone job has the same bits as it did in the grouped MTP window.
                std::vector<float> single;
                pool.execute(fixture.weights, input.view, {jobs.front()}, single);
                exact(single, std::vector<float>(expected.begin(), expected.begin() + hidden));
                ++cases;
            }
            if (geometry && padded && threads == 4) {
                // Sparse short-prefill jobs span the full token extent, while
                // each complete expert has only two jobs. Include broadcast,
                // per-lane input and lane modulo without allocating dense work.
                for (int tokens : {9, 32, 256}) for (int lanes : {1, 8}) {
                    Input input(hidden, lanes, tokens, true);
                    const std::vector<CpuExpertJob> sparse{
                        {7, tokens - 1, 7}, {3, 0, 1},
                        {7, tokens / 2, 0}, {3, tokens - 1, 9}};
                    const auto expected = oracle(fixture.weights, input.view, sparse);
                    const auto before = pool.stats();
                    std::vector<float> actual;
                    pool.execute(fixture.weights, input.view, sparse, actual);
                    exact(actual, expected);
                    const auto after = pool.stats();
                    require(after.calls - before.calls == 1 && after.jobs - before.jobs == 4,
                            "sparse prefill CPU expert counters mismatch");
                    require(after.unique_experts - before.unique_experts == 2,
                            "sparse prefill CPU experts were not grouped");
                    require(after.unique_inputs - before.unique_inputs == size_t(lanes == 1 ? 3 : 4),
                            "sparse prefill quantized unused inputs or failed sharing");
                    // Group scheduling and high token indices preserve each
                    // job's independent scalar arithmetic and output order.
                    std::vector<float> single;
                    pool.execute(fixture.weights, input.view, {sparse.front()}, single);
                    exact(single, std::vector<float>(expected.begin(), expected.begin() + hidden));
                    ++cases;
                }
            }
            if (!geometry) {
                Input input(hidden, 1, 1, padded);
                const auto jobs = jobs_for(16, 1, 8, 0);
                std::vector<float> result;
                for (int repeat = 0; repeat < 32; ++repeat) {
                    if (repeat % 8 == 0) std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    pool.execute(fixture.weights, input.view, jobs, result);
                }
                exact(result, oracle(fixture.weights, input.view, jobs));
                pool.execute(fixture.weights, input.view, {}, result);
                require(result.empty(), "empty CPU expert batch produced output");
                rejected([&] { pool.execute(fixture.weights, input.view, {{-1, 0, 0}}, result); });
                rejected([&] { pool.execute(fixture.weights, input.view, {{16, 0, 0}}, result); });
                rejected([&] { pool.execute(fixture.weights, input.view, {{0, 1, 0}}, result); });
                auto invalid = input.view; invalid.tokens = 257;
                rejected([&] { pool.execute(fixture.weights, invalid, jobs, result); });
                auto bad_weights = fixture.weights; bad_weights.gate.row_stride = 1;
                rejected([&] { pool.execute(bad_weights, input.view, jobs, result); });
                // Recovery after validation errors also exercises a sleeping worker.
                pool.execute(fixture.weights, input.view, jobs, result);
                exact(result, oracle(fixture.weights, input.view, jobs));
            }
        }
    }
    for (auto down_type : {GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS}) {
        Fixture fixture(256, 256, 4, true, GGML_TYPE_IQ2_S, down_type);
        Input input(256, 8, 2, true);
        CpuExpertPool::Config cfg; cfg.threads = 4;
        CpuExpertPool pool(cfg);
        const auto jobs = jobs_for(4, 2, 8, 1);
        std::vector<float> actual;
        pool.execute(fixture.weights, input.view, jobs, actual);
        exact(actual, oracle(fixture.weights, input.view, jobs));
        ++cases;
    }
    std::cout << "PASS CPU Q8/IQ2_S/IQ3_S/IQ4_XS expert pool: " << cases << " bitwise oracle cases; single/group identity; "
                 "2048/512 geometry; broadcast/8 lanes; padded strides; duplicate jobs; sparse T9/32/256; idle wakeup; validation recovery\n";
}

// This mirrors the old Task CPU layout (gather each routed input row, then
// ordinary MUL_MAT_ID, MUL_MAT_ID, SWIGLU, MUL_MAT_ID). It reuses its graph and
// ggml threadpool. Its vector exp can differ from the std::exp pool contract.
struct GenericGraph {
    ggml_context * ctx = nullptr;
    ggml_threadpool_t pool = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_cplan plan{};
    ggml_tensor * input = nullptr, * ids = nullptr, * output = nullptr;
    std::vector<float> gathered, g, u, h, result;
    std::vector<int32_t> routes;
    std::vector<uint8_t> scratch;
    GenericGraph(const CpuExpertWeights & w, int jobs, int tokens, int threads) {
        ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
        if (!ctx) throw std::runtime_error("generic CPU fixture context allocation failed");
        auto make_weights = [&](const CpuExpertWeightView & view, int cols, int rows) {
            auto * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, cols, rows, w.experts);
            tensor->data = const_cast<void *>(view.data);
            tensor->nb[1] = view.row_stride; tensor->nb[2] = view.expert_stride;
            tensor->nb[3] = view.expert_stride * size_t(w.experts);
            return tensor;
        };
        auto * gate = make_weights(w.gate, w.hidden, w.ffn);
        auto * up = make_weights(w.up, w.hidden, w.ffn);
        auto * down = make_weights(w.down, w.ffn, w.hidden);
        const bool decode = tokens == 1;
        input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w.hidden, decode ? jobs : 1, decode ? 1 : jobs);
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, decode ? jobs : 1, decode ? 1 : jobs);
        auto * gate_result = ggml_mul_mat_id(ctx, gate, input, ids);
        auto * up_result = ggml_mul_mat_id(ctx, up, input, ids);
        auto * activation = ggml_swiglu_split(ctx, gate_result, up_result);
        output = ggml_mul_mat_id(ctx, down, activation, ids);
        gathered.resize(size_t(jobs) * size_t(w.hidden)); routes.resize(size_t(jobs));
        g.resize(size_t(jobs) * size_t(w.ffn)); u.resize(g.size()); h.resize(g.size());
        result.resize(size_t(jobs) * size_t(w.hidden));
        input->data = gathered.data(); ids->data = routes.data();
        gate_result->data = g.data(); up_result->data = u.data(); activation->data = h.data(); output->data = result.data();
        graph = ggml_new_graph_custom(ctx, 32, false); ggml_build_forward_expand(graph, output);
        auto parameters = ggml_threadpool_params_default(threads);
        pool = ggml_threadpool_new(&parameters);
        if (!pool) throw std::runtime_error("generic CPU fixture pool allocation failed");
        plan = ggml_graph_plan(graph, threads, pool);
        scratch.resize(plan.work_size); plan.work_data = scratch.data();
    }
    ~GenericGraph() { if (pool) ggml_threadpool_free(pool); if (ctx) ggml_free(ctx); }
    void run(const CpuExpertInput & x, const std::vector<CpuExpertJob> & jobs, int hidden) {
        for (size_t j = 0; j < jobs.size(); ++j) {
            const auto & job = jobs[j];
            const auto * row = reinterpret_cast<const uint8_t *>(x.data) + size_t(job.token) * x.token_stride + size_t(job.lane % x.lanes) * x.lane_stride;
            std::memcpy(gathered.data() + j * size_t(hidden), row, size_t(hidden) * sizeof(float));
            routes[j] = job.expert;
        }
        if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("generic CPU fixture graph compute failed");
    }
};

void benchmark(int threads, int iterations, int tokens, int experts, int topk, bool pin, int spin_us, bool generic) {
    require(iterations > 0 && tokens > 0 && tokens <= 8 && experts > 0 && topk > 0, "invalid benchmark arguments");
    Fixture fixture(2048, 512, experts, false);
    Input input(2048, 1, tokens, false);
    CpuExpertPool::Config cfg; cfg.threads = threads; cfg.pin = pin; cfg.spin_us = spin_us;
    CpuExpertPool pool(cfg);
    auto jobs = jobs_for(experts, tokens, topk, 0);
    std::vector<float> output;
    pool.execute(fixture.weights, input.view, jobs, output);
    exact(output, oracle(fixture.weights, input.view, jobs));
    for (int warm = 0; warm < 5; ++warm) {
        jobs = jobs_for(experts, tokens, topk, warm); pool.execute(fixture.weights, input.view, jobs, output);
    }
    pool.reset_stats();
    const auto begin = Clock::now();
    for (int round = 0; round < iterations; ++round) {
        jobs = jobs_for(experts, tokens, topk, round); pool.execute(fixture.weights, input.view, jobs, output);
    }
    const double wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    const auto stats = pool.stats();
    const size_t bundle = fixture.weights.gate.expert_stride + fixture.weights.up.expert_stride + fixture.weights.down.expert_stride;
    std::cout << std::fixed << std::setprecision(6)
              << "{\"kind\":\"cpu_q8_expert_pool\",\"hidden\":2048,\"ffn\":512,\"experts\":" << experts
              << ",\"topk\":" << topk << ",\"tokens\":" << tokens << ",\"threads\":" << pool.threads()
              << ",\"physical_cores\":" << pool.physical_cores() << ",\"pin\":" << (pin ? "true" : "false")
              << ",\"spin_us\":" << spin_us << ",\"iterations\":" << iterations << ",\"wall_ms\":" << wall_ms
              << ",\"ms_per_call\":" << wall_ms / iterations << ",\"jobs_per_s\":" << stats.jobs * 1000.0 / wall_ms
              << ",\"unique_expert_effective_gbps\":" << double(stats.unique_experts) * double(bundle) / (wall_ms * 1e6)
              << ",\"input_quant_ms\":" << stats.input_quant_ms << ",\"gate_up_ms\":" << stats.gate_up_ms
              << ",\"hidden_quant_ms\":" << stats.hidden_quant_ms << ",\"down_ms\":" << stats.down_ms
              << ",\"unique_inputs\":" << stats.unique_inputs << ",\"jobs\":" << stats.jobs
              << ",\"oracle\":\"bitwise_pass\",\"gpu_calls\":0}\n";
    if (generic) {
        GenericGraph baseline(fixture.weights, int(jobs.size()), tokens, pool.threads());
        for (int warm = 0; warm < 5; ++warm) {
            jobs = jobs_for(experts, tokens, topk, warm); baseline.run(input.view, jobs, 2048);
        }
        const auto generic_begin = Clock::now();
        for (int round = 0; round < iterations; ++round) {
            jobs = jobs_for(experts, tokens, topk, round); baseline.run(input.view, jobs, 2048);
        }
        const double generic_ms = std::chrono::duration<double, std::milli>(Clock::now() - generic_begin).count();
        float max_abs = 0.f;
        for (size_t i = 0; i < output.size(); ++i) max_abs = std::max(max_abs, std::abs(output[i] - baseline.result[i]));
        std::cout << "{\"kind\":\"generic_ggml_cpu_ffn\",\"threads\":" << pool.threads()
                  << ",\"iterations\":" << iterations << ",\"wall_ms\":" << generic_ms
                  << ",\"ms_per_call\":" << generic_ms / iterations << ",\"pool_speed_ratio\":" << generic_ms / wall_ms
                  << ",\"pool_max_abs_difference\":" << max_abs << ",\"gpu_calls\":0}\n";
    }
}
} // namespace

int main(int argc, char ** argv) {
    try {
        ggml_cpu_init();
        bool bench = false, generic = false, pin = true;
        int threads = 0, iterations = 100, tokens = 3, experts = 64, topk = 8, spin_us = 2000;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto number = [&] {
                if (++i >= argc) throw std::runtime_error("missing numeric argument");
                size_t parsed = 0; int value = std::stoi(argv[i], &parsed);
                if (argv[i][parsed]) throw std::runtime_error("invalid numeric argument");
                return value;
            };
            if (arg == "--bench") bench = true;
            else if (arg == "--generic") generic = true;
            else if (arg == "--threads") threads = number();
            else if (arg == "--iterations") iterations = number();
            else if (arg == "--tokens") tokens = number();
            else if (arg == "--experts") experts = number();
            else if (arg == "--topk") topk = number();
            else if (arg == "--pin") pin = number() != 0;
            else if (arg == "--spin-us") spin_us = number();
            else throw std::runtime_error("unknown option: " + arg);
        }
        if (bench) benchmark(threads, iterations, tokens, experts, topk, pin, spin_us, generic);
        else validate();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL CPU expert pool: " << error.what() << '\n'; return 1;
    }
}
