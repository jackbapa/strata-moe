#include "backend.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int ni = 512, ff = 256, ne = 24, nk = 8;
struct Context {
    ggml_context * p;
    explicit Context(size_t tensors = 32) : p(ggml_init({ggml_tensor_overhead() * tensors +
            2 * ggml_graph_overhead_custom(64, false), nullptr, true})) {
        if (!p) throw std::bad_alloc();
    }
    ~Context() { ggml_free(p); }
};
struct Buffer {
    ggml_backend_buffer_t p = nullptr;
    ~Buffer() { if (p) ggml_backend_buffer_free(p); }
};
struct Backend {
    ggml_backend_t p = nullptr;
    ~Backend() { if (p) ggml_backend_free(p); }
};
struct Scheduler {
    ggml_backend_sched_t p = nullptr;
    ~Scheduler() { if (p) ggml_backend_sched_free(p); }
};

// Each oracle is a separate ordinary one-token GPU graph, with every expert
// weight resident on that GPU. No custom cache, host expert math or alternative
// arithmetic result is used as a tolerance escape.
std::vector<float> oracle(ggml_backend_t gpu, const std::vector<ggml_tensor *> & weights,
                          const std::vector<float> & inputs, const std::vector<int32_t> & routes,
                          int lanes, int nt, bool fused) {
    std::vector<float> expected((fused ? ni : ff) * nk * nt);
    for (int token = 0; token < nt; ++token) {
        Context ctx;
        auto * x = ggml_new_tensor_3d(ctx.p, GGML_TYPE_F32, ni, lanes, 1);
        auto * ids = ggml_new_tensor_2d(ctx.p, GGML_TYPE_I32, nk, 1);
        auto * out = ggml_mul_mat_id(ctx.p, weights[0], x, ids);
        if (fused) {
            auto * up = ggml_mul_mat_id(ctx.p, weights[1], x, ids);
            out = ggml_mul_mat_id(ctx.p, weights[2], ggml_swiglu_split(ctx.p, out, up), ids);
        }
        auto * graph = ggml_new_graph_custom(ctx.p, 64, false);
        ggml_build_forward_expand(graph, out);
        Buffer buffer; buffer.p = ggml_backend_alloc_ctx_tensors(ctx.p, gpu);
        if (!buffer.p) throw std::runtime_error("native oracle allocation failed");
        ggml_backend_tensor_set(x, inputs.data() + token * ni * lanes, 0, ni * lanes * sizeof(float));
        ggml_backend_tensor_set(ids, routes.data() + token * nk, 0, nk * sizeof(int32_t));
        if (ggml_backend_graph_compute(gpu, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("native one-token oracle failed");
        ggml_backend_tensor_get(out, expected.data() + token * (fused ? ni : ff) * nk, 0,
                                (fused ? ni : ff) * nk * sizeof(float));
    }
    return expected;
}

void check_case(ggml_backend_t gpu, ggml_backend_t cpu, ggml_backend_reg_t reg,
                const std::vector<ggml_tensor *> & host_weights,
                const std::vector<ggml_tensor *> & gpu_weights,
                size_t bundle_bytes, int slots, int lanes, int nt, bool fused, bool freeze,
                const std::string & profile, bool async_events, bool boundary_copies,
                const std::string & dump_output = {}) {
    strata_a3b::set_cache_budget(bundle_bytes * slots);
    strata_a3b::set_cache_admission(true);
    Backend hybrid; hybrid.p = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
    if (!hybrid.p) throw std::runtime_error("GPU-only expert backend initialization failed");
    Context ctx;
    auto * x = ggml_new_tensor_3d(ctx.p, GGML_TYPE_F32, ni, lanes, nt);
    auto * ids = ggml_new_tensor_2d(ctx.p, GGML_TYPE_I32, nk, nt);
    Buffer inputs_buffer; inputs_buffer.p = ggml_backend_alloc_ctx_tensors(ctx.p, gpu);
    if (!inputs_buffer.p) throw std::runtime_error("GPU-only input allocation failed");
    // Actual CUDA producers on both sides exercise the stream bridge, not
    // merely a direct expert backend call followed by a global scheduler join.
    auto * routed_x = async_events ? ggml_scale(ctx.p, x, 0.5f) : x;
    auto * routed_ids = async_events ? ggml_dup(ctx.p, ids) : ids;
    auto * experts = fused ? strata_a3b::build_fused_moe(ctx.p, routed_x, host_weights[0], host_weights[1], host_weights[2], routed_ids) :
                            ggml_mul_mat_id(ctx.p, host_weights[0], routed_x, routed_ids);
    if (!experts) throw std::runtime_error("fused GPU-only builder refused compatible experts");
    auto * output = ggml_scale(ctx.p, experts, async_events ? 2.0f : 1.0f);
    // A CPU branch between custom producer B and CUDA consumer D reads only
    // earlier CUDA producer A. C's input copy must not hide a missing B tail
    // dependency; D also directly shares B's GPU output buffer.
    auto * cpu_interlude = boundary_copies && fused && lanes == 1 ? ggml_scale(ctx.p, routed_x, 2.0f) : nullptr;
    if (cpu_interlude) output = ggml_add(ctx.p, output, cpu_interlude);
    Context cpu_copy_ctx;
    Buffer cpu_copy_buffer;
    ggml_tensor * cpu_copy_input = nullptr;
    std::vector<float> cpu_copy_values;
    if (boundary_copies && !cpu_interlude) {
        const int width = fused ? ni : ff;
        cpu_copy_input = ggml_new_tensor_3d(cpu_copy_ctx.p, GGML_TYPE_F32, width, 1, nt);
        if (freeze) ggml_set_input(cpu_copy_input);
        cpu_copy_buffer.p = ggml_backend_alloc_ctx_tensors(cpu_copy_ctx.p, cpu);
        if (!cpu_copy_buffer.p) throw std::runtime_error("async CPU copy-input allocation failed");
        cpu_copy_values.resize(width * nt);
        for (size_t i = 0; i < cpu_copy_values.size(); ++i) cpu_copy_values[i] = 0.125f * (1 + i % 8);
        ggml_backend_tensor_set(cpu_copy_input, cpu_copy_values.data(), 0, cpu_copy_values.size() * sizeof(float));
        output = ggml_add(ctx.p, output, cpu_copy_input);
    }
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(ctx.p, 64, false);
    if (cpu_interlude) {
        ggml_build_forward_expand(graph, experts);
        ggml_build_forward_expand(graph, cpu_interlude);
    }
    ggml_build_forward_expand(graph, output);
    // The scheduler API requires a final CPU backend even when no expert math
    // is assigned there. Expert/output placement and zero CPU-row counters are
    // asserted below; this backend supplies only the mandatory fallback entry.
    ggml_backend_t backends[] = {gpu, hybrid.p, cpu};
    Scheduler sched; sched.p = ggml_backend_sched_new(backends, nullptr, 3, 64, false, true);
    if (!sched.p) throw std::runtime_error("GPU-only scheduler initialization failed");
    // set_tensor_backend itself clears is_reset. Explicit allocation after the
    // assignments is essential: compute_async otherwise resets a not-yet-
    // allocated scheduler and discards all those assignments.
    ggml_backend_sched_reset(sched.p);
    if (cpu_interlude) {
        ggml_backend_sched_set_tensor_backend(sched.p, routed_x, gpu);
        ggml_backend_sched_set_tensor_backend(sched.p, routed_ids, gpu);
        ggml_backend_sched_set_tensor_backend(sched.p, experts, hybrid.p);
        ggml_backend_sched_set_tensor_backend(sched.p, cpu_interlude, cpu);
        ggml_backend_sched_set_tensor_backend(sched.p, output, gpu);
    }
    if (cpu_copy_input) {
        ggml_backend_sched_set_tensor_backend(sched.p, cpu_copy_input, cpu);
        ggml_backend_sched_set_tensor_backend(sched.p, output, gpu);
    }
    if ((cpu_interlude || cpu_copy_input) && !ggml_backend_sched_alloc_graph(sched.p, graph))
        throw std::runtime_error("async boundary graph allocation failed");
    std::vector<float> inputs(ni * lanes * nt);
    for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = std::cos(float(i) * 0.071f) * 0.25f;
    std::vector<int32_t> routes(nk * nt);
    auto fill_routes = [&](int shift) {
        for (int t = 0; t < nt; ++t) for (int lane = 0; lane < nk; ++lane)
            routes[t * nk + lane] = (t * nk + (lane == nk - 1 ? lane - 1 : lane) + shift) % ne;
    };
    fill_routes(0);
    ggml_backend_tensor_set(x, inputs.data(), 0, inputs.size() * sizeof(float));
    auto submit = [&]() {
        ggml_backend_tensor_set(ids, routes.data(), 0, routes.size() * sizeof(int32_t));
        return async_events ? ggml_backend_sched_graph_compute_async(sched.p, graph) :
                              ggml_backend_sched_graph_compute(sched.p, graph);
    };
    auto read_output = [&](std::vector<float> & values) {
        if (async_events) {
            // Wait only the final CUDA consumer stream. The custom expert
            // stream and its copy stream must be ordered through events.
            ggml_backend_tensor_get_async(gpu, output, values.data(), 0, values.size() * sizeof(float));
            ggml_backend_synchronize(gpu);
        } else ggml_backend_tensor_get(output, values.data(), 0, values.size() * sizeof(float));
    };
    // Warm both persistent-hit and bank-specific task paths before asserting
    // bounded allocation reuse. A shifted frozen route set introduces misses
    // where a warm cache has not seen all experts.
    if (submit() != GGML_STATUS_SUCCESS || submit() != GGML_STATUS_SUCCESS)
        throw std::runtime_error("GPU-only warmup failed");
    if (freeze) { strata_a3b::set_cache_admission(false); fill_routes(3); }
    auto oracle_inputs = inputs;
    if (async_events) for (auto & value : oracle_inputs) value *= 0.5f;
    // The long case compares byte-identical output over repeated submits and
    // across separate copy-ahead/control processes. The singleton oracle above
    // has decode arithmetic; it is only valid for the existing short cases.
    auto expected = nt >= 1024 ? std::vector<float>((fused ? ni : ff) * nk * nt) :
        oracle(gpu, gpu_weights, oracle_inputs, routes, lanes, nt, fused);
    if (async_events && nt < 1024) for (auto & value : expected) value *= 2.0f;
    if (cpu_interlude) for (int token = 0; token < nt; ++token) for (int lane = 0; lane < nk; ++lane)
        for (int row = 0; row < ni; ++row)
            expected[row + ni * (lane + nk * token)] += inputs[row + ni * token];
    if (cpu_copy_input) {
        const int width = fused ? ni : ff;
        for (int token = 0; token < nt; ++token) for (int lane = 0; lane < nk; ++lane)
            for (int row = 0; row < width; ++row)
                expected[row + width * (lane + nk * token)] += cpu_copy_values[row + width * token];
    }
    if (submit() != GGML_STATUS_SUCCESS || submit() != GGML_STATUS_SUCCESS)
        throw std::runtime_error("GPU-only frozen/route-shift warmup failed");
    if (async_events) ggml_backend_synchronize(gpu);
    const auto before = strata_a3b::stats();
    for (int repeat = 0; repeat < 3; ++repeat) {
        if (submit() != GGML_STATUS_SUCCESS) throw std::runtime_error("GPU-only expert computation failed");
        if (ggml_backend_sched_get_tensor_backend(sched.p, experts) != hybrid.p)
            throw std::runtime_error("scheduler bypassed the custom GPU-only expert backend");
        if (cpu_interlude && (ggml_backend_sched_get_n_splits(sched.p) < 4 ||
            ggml_backend_sched_get_tensor_backend(sched.p, cpu_interlude) != cpu ||
            ggml_backend_sched_get_tensor_backend(sched.p, routed_x) != gpu ||
            ggml_backend_sched_get_tensor_backend(sched.p, routed_ids) != gpu ||
            ggml_backend_sched_get_tensor_backend(sched.p, output) != gpu)) {
            std::fprintf(stderr, "CPU-interlude placements: splits=%d cpu=%s x=%s ids=%s output=%s\n",
                ggml_backend_sched_get_n_splits(sched.p),
                ggml_backend_name(ggml_backend_sched_get_tensor_backend(sched.p, cpu_interlude)),
                ggml_backend_name(ggml_backend_sched_get_tensor_backend(sched.p, routed_x)),
                ggml_backend_name(ggml_backend_sched_get_tensor_backend(sched.p, routed_ids)),
                ggml_backend_name(ggml_backend_sched_get_tensor_backend(sched.p, output)));
            throw std::runtime_error("async CPU interlude fixture did not retain CUDA/custom/CPU/CUDA splits");
        }
        // The custom backend also supports ordinary host buffers, so the
        // scheduler may attribute this leaf to StrataMoE. Its physical CPU
        // buffer still forces a blocking host-to-CUDA copy at this boundary.
        if (cpu_copy_input && (ggml_backend_sched_get_n_splits(sched.p) < 3 ||
            !ggml_backend_buffer_is_host(cpu_copy_input->buffer) ||
            ggml_backend_sched_get_tensor_backend(sched.p, output) != gpu)) {
            std::fprintf(stderr, "CPU-copy fixture placements: splits=%d cpu_input=%s output=%s\n",
                ggml_backend_sched_get_n_splits(sched.p),
                ggml_backend_sched_get_tensor_backend(sched.p, cpu_copy_input) ? ggml_backend_name(ggml_backend_sched_get_tensor_backend(sched.p, cpu_copy_input)) : "null",
                ggml_backend_sched_get_tensor_backend(sched.p, output) ? ggml_backend_name(ggml_backend_sched_get_tensor_backend(sched.p, output)) : "null");
            throw std::runtime_error("async adjacent CPU-copy fixture did not retain its copy boundary");
        }
        std::vector<float> actual(expected.size());
        read_output(actual);
        if (nt >= 1024 && repeat == 0) expected = actual;
        for (size_t i = 0; i < actual.size(); ++i)
            if (!std::isfinite(actual[i]) || std::memcmp(&actual[i], &expected[i], sizeof(float)) != 0) {
                std::fprintf(stderr, "GPU-only mismatch fused=%d slots=%d nt=%d lanes=%d freeze=%d repeat=%d index=%zu expected=%.9g actual=%.9g\n",
                             (int) fused, slots, nt, lanes, (int) freeze, repeat, i, expected[i], actual[i]);
                throw std::runtime_error("GPU-only experts differ bitwise from independent singleton GPU oracle");
            }
    }
    const auto after = strata_a3b::stats();
    const uint64_t rows = 3ull * routes.size();
    const uint64_t projections = fused ? 3 : 1;
    if (after.cpu_rows != before.cpu_rows || after.fused_cpu_rows != before.fused_cpu_rows ||
        after.gpu_only_rows - before.gpu_only_rows != rows ||
        after.gpu_rows - before.gpu_rows != projections * rows)
        throw std::runtime_error("GPU-only routing counts include missing rows or CPU expert execution");
    if (after.staging_allocations != before.staging_allocations ||
        after.expert_task_allocations != before.expert_task_allocations)
        throw std::runtime_error("identical warmed route windows reallocated staging banks or expert tasks");
    if (freeze && after.cache_admissions != before.cache_admissions)
        throw std::runtime_error("frozen cache placement admitted a routed expert");
    if (!slots && (after.staging_upload_bytes == before.staging_upload_bytes ||
                   after.staging_batches - before.staging_batches < 3 * 3))
        throw std::runtime_error("zero-cache fixture did not exercise multiple staged waves");
    if (slots == ne && !freeze && after.upload_bytes != before.upload_bytes)
        throw std::runtime_error("fully resident warmed experts were uploaded again");
    if (after.hits - before.hits + after.misses - before.misses != projections * rows)
        throw std::runtime_error("GPU-only hit/miss counts did not cover every projection row");
    if (async_events && (after.async_expert_calls - before.async_expert_calls != 3 ||
                         after.async_route_waits - before.async_route_waits != 3 ||
                         after.async_event_waits == before.async_event_waits)) {
        std::fprintf(stderr, "async fixture counters expert_calls=%llu route_waits=%llu event_waits=%llu task_reuse_waits=%llu\n",
            (unsigned long long) (after.async_expert_calls - before.async_expert_calls),
            (unsigned long long) (after.async_route_waits - before.async_route_waits),
            (unsigned long long) (after.async_event_waits - before.async_event_waits),
            (unsigned long long) (after.async_task_reuse_waits - before.async_task_reuse_waits));
        throw std::runtime_error("async expert/event/route bridge was not exercised");
    }
    // Invalid routes are rejected, then the same context must remain usable.
    const int32_t saved = routes[0]; routes[0] = ne;
    if (submit() == GGML_STATUS_SUCCESS) throw std::runtime_error("out-of-range GPU-only route was accepted");
    routes[0] = saved;
    if (submit() != GGML_STATUS_SUCCESS) throw std::runtime_error("GPU-only backend could not recover after a rejected route");
    std::vector<float> recovered(expected.size());
    read_output(recovered);
    for (size_t i = 0; i < recovered.size(); ++i)
        if (!std::isfinite(recovered[i]) || std::memcmp(&recovered[i], &expected[i], sizeof(float)) != 0)
            throw std::runtime_error("GPU-only recovery changed expert results");
    if (!dump_output.empty()) {
        std::ofstream file(dump_output, std::ios::binary);
        if (!file.write((const char *) expected.data(), expected.size() * sizeof(float)))
            throw std::runtime_error("failed to write Q8 expert comparison output");
    }
    uint64_t checksum = 14695981039346656037ull;
    const auto * bytes = reinterpret_cast<const unsigned char *>(expected.data());
    for (size_t i = 0; i < expected.size() * sizeof(float); ++i)
        checksum = (checksum ^ bytes[i]) * 1099511628211ull;
    std::printf("GPU-only %s slots=%d nt=%d lanes=%d frozen=%d CPU_interlude=%d CPU_copy=%d user_input=%d GPU_rows=%llu CPU_rows=0 staged_rows=%llu waves=%llu allocations_delta=0 task_allocations_delta=0 checksum=%016llx bitwise_PASS\n",
                fused ? "FFN" : "projection", slots, nt, lanes, (int) freeze, (int) (cpu_interlude != nullptr),
                (int) (cpu_copy_input != nullptr), (int) (cpu_copy_input && freeze),
                (unsigned long long) (after.gpu_rows - before.gpu_rows),
                (unsigned long long) (after.staged_rows - before.staged_rows),
                (unsigned long long) (after.staging_batches - before.staging_batches),
                (unsigned long long) checksum);
    // Export while this backend and its observed route counts are still live.
    // Repeated cases replace the diagnostic file with the final complete case.
    if (!profile.empty()) strata_a3b::dump_route_frequencies(profile.c_str());
    strata_a3b::set_cache_admission(true);
}

void check_async_host_lifetime(ggml_backend_t gpu, ggml_backend_t cpu, ggml_backend_reg_t reg,
        const std::vector<ggml_tensor *> & host_weights, const std::vector<ggml_tensor *> & gpu_weights,
        size_t bundle_bytes, int slots) {
    constexpr int nt = 3, repeats = 8;
    std::vector<std::vector<float>> inputs(repeats, std::vector<float>(ni * nt));
    std::vector<std::vector<int32_t>> routes(repeats, std::vector<int32_t>(nk * nt));
    std::vector<std::vector<float>> expected, actual(repeats, std::vector<float>(ni * nk * nt));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        for (size_t i = 0; i < inputs[repeat].size(); ++i)
            inputs[repeat][i] = std::sin(float(i + repeat * 31) * 0.043f) * 0.2f;
        for (size_t i = 0; i < routes[repeat].size(); ++i) routes[repeat][i] = (i + repeat * 5) % ne;
        expected.push_back(oracle(gpu, gpu_weights, inputs[repeat], routes[repeat], 1, nt, true));
    }
    strata_a3b::set_cache_budget(bundle_bytes * slots);
    strata_a3b::set_cache_admission(true);
    Backend hybrid; hybrid.p = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
    if (!hybrid.p) throw std::runtime_error("async lifetime backend initialization failed");
    Context ctx;
    auto * x = ggml_new_tensor_3d(ctx.p, GGML_TYPE_F32, ni, 1, nt);
    auto * ids = ggml_new_tensor_2d(ctx.p, GGML_TYPE_I32, nk, nt);
    Buffer host_input; host_input.p = ggml_backend_alloc_ctx_tensors(ctx.p, cpu);
    auto * out = strata_a3b::build_fused_moe(ctx.p, x, host_weights[0], host_weights[1], host_weights[2], ids);
    auto * graph = ggml_new_graph_custom(ctx.p, 64, false); ggml_build_forward_expand(graph, out);
    Buffer output; output.p = ggml_backend_alloc_ctx_tensors(ctx.p, gpu);
    if (!host_input.p || !output.p) throw std::runtime_error("async lifetime tensor allocation failed");
    for (int repeat = 0; repeat < repeats; ++repeat) {
        ggml_backend_tensor_set(x, inputs[repeat].data(), 0, inputs[repeat].size() * sizeof(float));
        ggml_backend_tensor_set(ids, routes[repeat].data(), 0, routes[repeat].size() * sizeof(int32_t));
        if (ggml_backend_graph_compute_async(hybrid.p, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("async direct host-input submit failed");
        ggml_backend_tensor_get_async(hybrid.p, out, actual[repeat].data(), 0, actual[repeat].size() * sizeof(float));
        // The custom task must own its queued host sources. Poison the caller's
        // host input immediately, then reuse the same task and staging banks.
        std::memset(x->data, 0x7f, ggml_nbytes(x));
        std::memset(ids->data, 0x7f, ggml_nbytes(ids));
    }
    ggml_backend_synchronize(hybrid.p);
    for (int repeat = 0; repeat < repeats; ++repeat)
        if (std::memcmp(actual[repeat].data(), expected[repeat].data(), actual[repeat].size() * sizeof(float)))
            throw std::runtime_error("async task host-source lifetime/reuse changed a direct GPU result");
    std::printf("async host lifetime slots=%d repeated_submit=%d caller_input_poisoned=1 bitwise_PASS\n", slots, repeats);
}
} // namespace

int main(int argc, char ** argv) {
    try {
        bool pageable = false, async_events = false, boundary_copies = false, copy_ahead_long = false,
             independent_issuer = false, packed_gate_up_long = false, long_only = false;
        int cache_decay_interval = 0;
        std::string profile, dump_output;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--pageable") pageable = true;
            else if (key == "--async-events") async_events = true;
            else if (key == "--boundary-copies") boundary_copies = true;
            else if (key == "--copy-ahead-long") copy_ahead_long = true;
            else if (key == "--independent-issuer") independent_issuer = true;
            else if (key == "--packed-gate-up-long") packed_gate_up_long = true;
            else if (key == "--long-only") long_only = true;
            else if (key == "--cache-decay-interval" && i + 1 < argc) cache_decay_interval = std::stoi(argv[++i]);
            else if (key == "--profile" && i + 1 < argc) profile = argv[++i];
            else if (key == "--dump-output" && i + 1 < argc) dump_output = argv[++i];
            else throw std::runtime_error("usage: strata-moe-gpu-expert-parity [--pageable] [--async-events] [--boundary-copies] [--copy-ahead-long] [--long-only] [--profile FILE]");
        }
        if (boundary_copies && !async_events) throw std::runtime_error("--boundary-copies requires --async-events");
        if (copy_ahead_long && !pageable) throw std::runtime_error("--copy-ahead-long requires --pageable");
        if (independent_issuer && !copy_ahead_long)
            throw std::runtime_error("--independent-issuer requires --copy-ahead-long");
        if (packed_gate_up_long && !independent_issuer)
            throw std::runtime_error("--packed-gate-up-long requires --independent-issuer");
        if (cache_decay_interval < 0 || cache_decay_interval > 1024) throw std::runtime_error("decay interval must be 0..1024");
        ggml_backend_load_all();
        auto device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!device) { std::puts("SKIP GPU-only expert parity: GPU unavailable"); return 77; }
        Backend gpu; gpu.p = ggml_backend_dev_init(device, nullptr);
        if (!gpu.p) throw std::runtime_error("native GPU initialization failed");
        Backend cpu; cpu.p = ggml_backend_cpu_init();
        if (!cpu.p) throw std::runtime_error("scheduler fallback initialization failed");
        ggml_backend_cpu_set_n_threads(cpu.p, 1);
        const size_t expert_bytes = ggml_row_size(GGML_TYPE_Q8_0, ni) * ff;
        const size_t bundle_bytes = expert_bytes * 3;
        strata_a3b::Config cfg;
        cfg.gpu = device; cfg.gpu_only_experts = true; cfg.pageable_experts = pageable;
        cfg.async_expert_events = async_events;
        cfg.prefill_copy_ahead = copy_ahead_long;
        cfg.prefill_independent_issuer = independent_issuer;
        cfg.prefill_packed_gate_up = packed_gate_up_long;
        cfg.cache_decay_interval = cache_decay_interval;
        cfg.expert_staging_slots = 2; cfg.expert_bytes = bundle_bytes * ne;
        cfg.stable_tasks = true; cfg.fused_moe = true; cfg.exact_verify = true;
        cfg.native_exact_max_tokens = 8; cfg.cache_graph_routes = true;
        auto reg = strata_a3b::register_backend(cfg);
        Context host_ctx, gpu_ctx;
        std::vector<ggml_tensor *> host_weights, gpu_weights;
        for (int p = 0; p < 3; ++p) {
            auto * host = ggml_new_tensor_3d(host_ctx.p, GGML_TYPE_Q8_0, p == 2 ? ff : ni, p == 2 ? ni : ff, ne);
            ggml_format_name(host, "blk.0.ffn_%s_exps.weight", p == 0 ? "gate" : p == 1 ? "up" : "down");
            host_weights.push_back(host); gpu_weights.push_back(ggml_dup_tensor(gpu_ctx.p, host));
        }
        Buffer host_buffer, gpu_buffer;
        host_buffer.p = ggml_backend_alloc_ctx_tensors_from_buft(host_ctx.p, strata_a3b::expert_buffer_type());
        gpu_buffer.p = ggml_backend_alloc_ctx_tensors(gpu_ctx.p, gpu.p);
        if (!host_buffer.p || !gpu_buffer.p) throw std::runtime_error("expert test weights did not fit");
        ggml_backend_buffer_set_usage(host_buffer.p, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_set_usage(gpu_buffer.p, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (size_t p = 0; p < host_weights.size(); ++p) {
            auto * w = host_weights[p];
            std::vector<float> values(ggml_nelements(w));
            for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(float(i) * (0.017f + p * 0.003f)) * 0.05f;
            std::vector<char> packed(ggml_nbytes(w));
            ggml_quantize_chunk(w->type, values.data(), packed.data(), 0, w->ne[1] * ne, w->ne[0], nullptr);
            ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
            ggml_backend_tensor_set(gpu_weights[p], packed.data(), 0, packed.size());
        }
        std::printf("GPU-only expert parity device=%s pageable=%d async_events=%d boundary_copies=%d staging_slots=2\n", ggml_backend_dev_name(device), (int) pageable, (int) async_events, (int) boundary_copies);
        if (!long_only) for (bool fused : {false, true}) for (int slots : {0, 2, ne})
        for (int nt : {1, 3, 8}) for (int lanes : {1, nk}) for (bool freeze : {false, true})
            check_case(gpu.p, cpu.p, reg, host_weights, gpu_weights, bundle_bytes, slots, lanes, nt, fused, freeze, profile, async_events, boundary_copies);
        if (async_events && !long_only) for (int slots : {0, 2, ne})
            check_async_host_lifetime(gpu.p, cpu.p, reg, host_weights, gpu_weights, bundle_bytes, slots);
        if (copy_ahead_long || long_only)
            check_case(gpu.p, cpu.p, reg, host_weights, gpu_weights, bundle_bytes,
                       0, nk, 1024, true, true, profile, async_events, false, dump_output);
        const auto s = strata_a3b::stats();
        if (cache_decay_interval && !s.decays) throw std::runtime_error("LFU decay fixture did not trigger its policy");
        const bool effective_pageable = pageable || s.pinned_arena_fallbacks;
        if ((effective_pageable && !s.staging_host_bytes) || (!effective_pageable && s.staging_host_bytes))
            throw std::runtime_error("pinned direct-DMA/pageable upload-bank policy was not respected");
        std::puts(long_only ?
            "PASS long Q8 expert staging: repeated outputs bitwise stable; compare control/copy-ahead checksums across processes" :
            "PASS GPU-only Q8 experts: independent GPU singleton oracle; zero/tiny/full cache; frozen placement; multiwave banks; bounded allocation reuse");
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "FAIL GPU-only expert parity: %s\n", e.what()); return 1; }
}
