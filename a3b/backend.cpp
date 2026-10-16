#include "backend.hpp"
#include "cache_profile.hpp"
#include "cpu_experts.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-cuda-strata-moe.h"
#if defined(STRATA_A3B_STRATA_NATIVE)
#include "ggml-cuda.h"
#include "strata_native_experts.hpp"
#include <cuda_runtime_api.h>
extern "C" uint64_t ggml_backend_cuda_a3b_gdn_fused_graph_nodes();
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace strata_a3b {
namespace {
Config config;
Stats counters;
bool cache_admission = true;
std::unique_ptr<ExpertCachePlan> cache_plan;
ggml_backend_device device;
ggml_backend_buffer_type host_type;
struct Backend;
std::vector<Backend *> live_backends;
bool shared_expert_store_enabled() { return config.gpu_static_experts || config.hybrid_doorbell; }

void trace_native_hook(const char * hook, const char * decision, const ggml_tensor * w,
                       const ggml_tensor * x, int token_axis) {
    const char * enabled = std::getenv("STRATA_A3B_NATIVE_HOOK_TRACE");
    if (!enabled || !enabled[0] || enabled[0] == '0' || !x ||
        x->ne[token_axis] < 2 || x->ne[token_axis] > 9) return;
    static std::map<std::string, unsigned> printed;
    if (printed[hook]++ >= 96) return;
    const ggml_tensor * owner = w;
    while (owner && !owner->buffer && owner->view_src) owner = owner->view_src;
    auto buffer = w ? w->buffer : nullptr;
    auto owner_buffer = owner ? owner->buffer : nullptr;
    std::fprintf(stderr,
        "StrataMoE native hook %s %s weight=%s type=%s buffer=%p(%s) view_src=%p owner_buffer=%p(%s) "
        "input=[%lld,%lld,%lld,%lld] limit=%d flags[dense=%d expert=%d fused=%d]\n",
        hook, decision, w ? w->name : "<null>", w ? ggml_type_name(w->type) : "<null>",
        (void *) buffer, buffer ? ggml_backend_buffer_name(buffer) : "none",
        w ? (void *) w->view_src : nullptr, (void *) owner_buffer,
        owner_buffer ? ggml_backend_buffer_name(owner_buffer) : "none",
        (long long) x->ne[0], (long long) x->ne[1], (long long) x->ne[2], (long long) x->ne[3],
        config.native_exact_max_tokens, (int) config.native_dense_exact,
        (int) config.native_exact_verify, (int) config.fused_moe);
}

const char * name(ggml_backend_dev_t) { return "StrataMoE"; }
const char * description(ggml_backend_dev_t) { return "RAM experts + adaptive GPU cache"; }
const char * buffer_name(ggml_backend_buffer_type_t) { return "StrataMoE_Host"; }
const char * backend_name(ggml_backend_t) { return "StrataMoE"; }
void memory(ggml_backend_dev_t, size_t * f, size_t * t) { *f = *t = 0; }
enum ggml_backend_dev_type type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_ACCEL; }
bool is_view(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}
void props(ggml_backend_dev_t d, ggml_backend_dev_props * p) {
    *p = {}; p->name = name(d); p->description = description(d); p->type = type(d);
    p->caps.async = config.async_expert_events || config.strata_native_experts || config.strata_prefill_mmq;
    p->caps.events = config.async_expert_events || config.strata_native_experts;
}
ggml_backend_buffer_t host_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto parent = config.gpu && !config.pageable_experts ? ggml_backend_dev_host_buffer_type(config.gpu) : nullptr;
    const auto requested_pinned = parent;
    const bool wants_pinned = (config.gpu_only_experts || config.hybrid_experts) && !config.pageable_experts;
    if (!parent) parent = ggml_backend_cpu_buffer_type();
    auto b = ggml_backend_buft_alloc_buffer(parent, size);
    if (b) {
        // CUDA's pinned allocator can return a CPU buffer after allocation
        // failure. Detect it before replacing the public buffer type: raw
        // pageable DMA would otherwise silently bypass bounded upload banks.
        if (wants_pinned && (!requested_pinned || ggml_backend_buffer_get_type(b) != requested_pinned)) {
            config.pageable_experts = true;
            ++counters.pinned_arena_fallbacks;
            std::fprintf(stderr, "StrataMoE GPU-only: %.1f MiB expert arena fell back to pageable RAM; enabling bounded host upload banks\n",
                         size / (1024.0 * 1024.0));
        }
        b->buft = buft;
    }
    return b;
}
ggml_backend_buffer_type_t buffer_type(ggml_backend_dev_t) {
    return config.gpu ? ggml_backend_dev_buffer_type(config.gpu) : &host_type;
}
bool supports_buffer(ggml_backend_dev_t, ggml_backend_buffer_type_t b) {
    return ggml_backend_buft_is_host(b) || (config.gpu && ggml_backend_dev_supports_buft(config.gpu, b));
}
void fused_marker(ggml_tensor *, int, int, void *) {
    GGML_ABORT("StrataMoE fused experts require the StrataMoE backend");
}
bool is_fused_op(const ggml_tensor * t) {
    if (t->op != GGML_OP_CUSTOM) return false;
    ggml_custom_op_params p;
    std::memcpy(&p, t->op_params, sizeof(p));
    return p.fun == fused_marker;
}
bool supports_op(ggml_backend_dev_t, const ggml_tensor * t) {
    if (t->op == GGML_OP_NONE || is_view(t->op)) return true;
    if (is_fused_op(t)) return true;
    if (t->op != GGML_OP_MUL_MAT_ID) return false;
    const auto * w = t->src[0];
    return w && w->buffer && w->buffer->buft == &host_type &&
           ggml_is_contiguous(w) && t->src[1]->type == GGML_TYPE_F32 &&
           t->src[2]->type == GGML_TYPE_I32 && t->src[1]->ne[3] == 1 && t->ne[3] == 1;
}

struct Context {
    ggml_context * p = nullptr;
    Context() { p = ggml_init({ggml_tensor_overhead() * 20 + 3 * ggml_graph_overhead_custom(32, false), nullptr, true});
        if (!p) throw std::bad_alloc(); }
    ~Context() { ggml_free(p); }
};
struct Buffer {
    ggml_backend_buffer_t p = nullptr;
    ~Buffer() { if (p) ggml_backend_buffer_free(p); }
};
struct Event {
    ggml_backend_event_t p = nullptr;
    ~Event() { if (p) ggml_backend_event_free(p); }
};
struct Cache {
    Context ctx;
    Buffer buf;
    ggml_tensor * weights = nullptr;
    std::vector<int> slot, owner;
    std::vector<uint64_t> frequency;
    std::vector<uint64_t> route_count;
    int decay_windows = 0;
    std::string tensor_name;
    int capacity = 0;
};
void count_frequency(Cache & cache, int expert) {
    if (shared_expert_store_enabled()) return;
    auto & value = cache.frequency[expert];
    if (config.cache_decay_interval > 0) {
        constexpr uint64_t unit = 256;
        value = value > std::numeric_limits<uint64_t>::max() - unit ?
            std::numeric_limits<uint64_t>::max() : value + unit;
    } else ++value;
}
bool frequency_wins(uint64_t challenger, uint64_t incumbent) {
    return config.cache_decay_interval > 0 ?
        challenger > incumbent && challenger - incumbent > 256 : challenger > incumbent + 1;
}
void decay_window(Cache & cache) {
    if (++cache.decay_windows >= config.cache_decay_interval) {
        for (auto & frequency : cache.frequency)
            frequency = (frequency / 10) * 7 + ((frequency % 10) * 7) / 10;
        cache.decay_windows = 0;
        ++counters.decays;
    }
}

using StaticCacheKey = std::tuple<const void *, int, int64_t, int64_t, int64_t>;
using StaticProviderKey = std::tuple<const void *, const void *, const void *>;
struct StaticProvider;
struct StaticCacheStore {
    StaticCacheStore();
    const ggml_cuda_strata_moe_api_v1 * api = nullptr;
    ggml_backend_t allocator = nullptr;
    std::map<StaticCacheKey, std::unique_ptr<Cache>> caches;
    std::map<StaticProviderKey, std::unique_ptr<StaticProvider>> providers;
    std::unique_ptr<CpuExpertPool> doorbell_pool;
    std::vector<CpuExpertJob> doorbell_jobs;
    std::vector<float> doorbell_outputs;
    CpuExpertPoolStats collected_pool_stats;
    ~StaticCacheStore();
    Cache & cache(ggml_tensor * source);
};
std::unique_ptr<StaticCacheStore> static_store;

const ggml_cuda_strata_moe_api_v1 * static_api(ggml_backend_dev_t gpu) {
    auto reg = gpu ? ggml_backend_dev_backend_reg(gpu) : nullptr;
    if (!reg || std::strcmp(ggml_backend_reg_name(reg), "CUDA") != 0 ||
        ggml_backend_reg_dev_count(reg) < 1 || ggml_backend_reg_dev_get(reg, 0) != gpu)
        throw std::runtime_error("GPU-static experts require native CUDA device 0");
    const auto get_api = reinterpret_cast<ggml_cuda_strata_moe_get_api_v1>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_strata_moe_v1"));
    const auto * api = get_api ? get_api() : nullptr;
    if (!api || api->abi_version != GGML_CUDA_STRATA_MOE_ABI_V1 || !api->create ||
        !api->build || !api->get_stats || !api->destroy || !api->scratch_bytes)
        throw std::runtime_error("CUDA GPU-static expert ABI v1 is unavailable");
    return api;
}
StaticCacheStore & static_cache_store() {
    if (!static_store) {
        auto pending = std::make_unique<StaticCacheStore>();
        pending->api = static_api(config.gpu);
        static_store = std::move(pending);
    }
    return *static_store;
}
Cache & StaticCacheStore::cache(ggml_tensor * source) {
    if (!cache_plan) throw std::runtime_error("set a fixed expert profile before GPU-static expert execution");
    const StaticCacheKey key{source->data, (int) source->type, source->ne[0], source->ne[1], source->ne[2]};
    auto & entry = caches[key];
    if (entry) return *entry;
    int layer = -1;
    if (std::sscanf(source->name, "blk.%d.", &layer) != 1 || layer < 0 || layer >= expert_profile_layers ||
        source->type != GGML_TYPE_Q8_0 || source->ne[2] != expert_profile_experts || source->ne[3] != 1 ||
        source->ne[0] * source->ne[1] != 2048 * 512 || !ggml_is_contiguous(source))
        throw std::runtime_error("GPU-static profile requires complete Q8 A3B expert tensors");
    auto pending = std::make_unique<Cache>();
    auto & c = *pending;
    c.tensor_name = source->name;
    c.capacity = (int) cache_plan->expert_ids[layer].size();
    c.slot.assign(expert_profile_experts, -1); c.owner.assign(c.capacity, -1);
    c.frequency.assign(expert_profile_experts, 0); c.route_count.assign(expert_profile_experts, 0);
    if (c.capacity) {
        if (!allocator) allocator = ggml_backend_dev_init(config.gpu, nullptr);
        if (!allocator) throw std::runtime_error("GPU-static cache allocator initialization failed");
        c.weights = ggml_new_tensor_3d(c.ctx.p, source->type, source->ne[0], source->ne[1], c.capacity);
        c.buf.p = ggml_backend_alloc_ctx_tensors(c.ctx.p, allocator);
        if (!c.buf.p) throw std::runtime_error("GPU-static expert cache does not fit its budget");
        ggml_backend_buffer_set_usage(c.buf.p, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_clear(c.buf.p, 0);
        counters.cache_bytes += ggml_backend_buffer_get_size(c.buf.p);
        for (int slot = 0; slot < c.capacity; ++slot) {
            const int expert = cache_plan->expert_ids[layer][slot];
            ggml_backend_tensor_set(c.weights, (const char *) source->data + expert * source->nb[2],
                slot * c.weights->nb[2], source->nb[2]);
            c.owner[slot] = expert; c.slot[expert] = slot; c.frequency[expert] = 1;
            ++counters.uploads; counters.upload_bytes += source->nb[2];
        }
    }
    entry = std::move(pending);
    return *entry;
}
struct StaticProvider {
    StaticCacheStore * store = nullptr;
    std::array<ggml_tensor, 3> source{};
    std::array<Cache *, 3> cache{};
    void * handle = nullptr;
    ~StaticProvider() { if (handle) store->api->destroy(handle); }
};
StaticCacheStore::StaticCacheStore() = default;
StaticCacheStore::~StaticCacheStore() {
    // CUDA handles retain both source metadata and cache tensor pointers.
    // Destroy them before releasing either the registry or its allocator.
    providers.clear(); doorbell_pool.reset(); caches.clear();
    if (allocator) ggml_backend_free(allocator);
}
int compute_doorbell_cpu(void * user, const float * input, int input_lanes, int tokens,
                        const int32_t * ids, const int32_t * slots, float * output,
                        char * error, size_t error_size);
int prepare_static_layer(void * user, ggml_cuda_strata_moe_layer_v1 * output,
                         char * error, size_t error_size) {
    try {
        auto & provider = *(StaticProvider *) user;
        int layer = -1;
        for (size_t role = 0; role < provider.source.size(); ++role) {
            int current = -1;
            if (std::sscanf(provider.source[role].name, "blk.%d.", &current) != 1 ||
                (role && current != layer))
                throw std::runtime_error("GPU-static complete-expert roles belong to different layers");
            layer = current;
        }
        for (size_t role = 0; role < provider.source.size(); ++role) {
            provider.cache[role] = &provider.store->cache(&provider.source[role]);
            output->host_weights[role] = &provider.source[role];
            output->cached_weights[role] = provider.cache[role]->weights;
        }
        const auto & gate = *provider.cache[0];
        if (provider.cache[1]->capacity != gate.capacity || provider.cache[2]->capacity != gate.capacity ||
            provider.cache[1]->slot != gate.slot || provider.cache[2]->slot != gate.slot)
            throw std::runtime_error("GPU-static complete-expert cache roles have different slot maps");
        output->cache_slots = gate.capacity;
        output->dynamic_cache = config.gpu_dynamic_experts ? 1 : 0;
        if (config.hybrid_doorbell) {
            output->cpu_callback = compute_doorbell_cpu;
            output->cpu_user = &provider;
        }
        for (int expert = 0; expert < expert_profile_experts; ++expert) output->slot_of_expert[expert] = gate.slot[expert];
        return 1;
    } catch (const std::exception & e) {
        if (error && error_size) std::snprintf(error, error_size, "%s", e.what());
        return 0;
    } catch (...) {
        if (error && error_size) std::snprintf(error, error_size, "unknown GPU-static cache preparation error");
        return 0;
    }
}
StaticProvider & static_provider(ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down) {
    auto & store = static_cache_store();
    auto & entry = store.providers[{gate->data, up->data, down->data}];
    if (entry) return *entry;
    auto pending = std::make_unique<StaticProvider>();
    pending->store = &store;
    pending->source = {*gate, *up, *down};
    char error[512]{};
    pending->handle = store.api->create(0, prepare_static_layer, pending.get(), error, sizeof(error));
    if (!pending->handle) throw std::runtime_error(std::string("GPU-static layer creation failed: ") + error);
    entry = std::move(pending);
    return *entry;
}
struct StagingBank {
    Context ctx, host_ctx, packed_ctx;
    Buffer buf, host_buf, packed_buf;
    std::vector<ggml_tensor *> weights, host_weights;
    ggml_tensor * packed_gate_up = nullptr;
    Event copied, consumed;
    bool live = false;
    uint64_t consumed_serial = 0;
    char cache_task_tag = 0, staging_task_tag = 0, packed_task_tag = 0;
    void ensure_packed_gate_up(ggml_backend_t gpu) {
        if (packed_gate_up) return;
        if (weights.size() != 3 || weights[0]->type != GGML_TYPE_Q8_0 ||
            weights[1]->type != GGML_TYPE_Q8_0 || weights[2]->type != GGML_TYPE_Q8_0 ||
            weights[0]->ne[0] != weights[1]->ne[0] || weights[0]->ne[1] != weights[1]->ne[1])
            throw std::runtime_error("packed gate/up requires matching Q8 expert projections");
        packed_gate_up = ggml_new_tensor_3d(packed_ctx.p, GGML_TYPE_Q8_0,
            weights[0]->ne[0], 2 * weights[0]->ne[1], config.expert_staging_slots);
        packed_buf.p = ggml_backend_alloc_ctx_tensors(packed_ctx.p, gpu);
        if (!packed_buf.p) throw std::runtime_error("packed gate/up staging does not fit GPU memory");
        ggml_backend_buffer_set_usage(packed_buf.p, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        counters.staging_gpu_bytes += ggml_backend_buffer_get_size(packed_buf.p);
    }
    void ensure_host_bank() {
        if (host_buf.p) return;
        if (host_weights.empty()) for (auto * w : weights)
            host_weights.push_back(ggml_dup_tensor(host_ctx.p, w));
        auto parent = ggml_backend_dev_host_buffer_type(config.gpu);
        if (!parent) parent = ggml_backend_cpu_buffer_type();
        host_buf.p = ggml_backend_alloc_ctx_tensors_from_buft(host_ctx.p, parent);
        if (!host_buf.p) throw std::runtime_error("expert upload staging allocation failed");
        if (ggml_backend_buffer_get_type(host_buf.p) != parent || parent == ggml_backend_cpu_buffer_type()) {
            ++counters.pinned_bank_fallbacks;
            std::fprintf(stderr, "StrataMoE GPU-only: bounded upload bank is pageable; transfers remain correct but may synchronize\n");
        }
        counters.staging_host_bytes += ggml_backend_buffer_get_size(host_buf.p);
    }
    StagingBank(ggml_backend_t gpu, const std::vector<ggml_tensor *> & source) {
        for (auto * w : source) {
            weights.push_back(ggml_new_tensor_3d(ctx.p, w->type, w->ne[0], w->ne[1], config.expert_staging_slots));
        }
        buf.p = ggml_backend_alloc_ctx_tensors(ctx.p, gpu);
        if (!buf.p) throw std::runtime_error("GPU expert staging does not fit; lower cache budget or staging slots");
        ggml_backend_buffer_set_usage(buf.p, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if (config.pageable_experts) ensure_host_bank();
        copied.p = ggml_backend_event_new(config.gpu);
        consumed.p = ggml_backend_event_new(config.gpu);
        if (!copied.p || !consumed.p) throw std::runtime_error("GPU-only expert staging requires backend events");
        ++counters.staging_allocations;
        counters.staging_gpu_bytes += ggml_backend_buffer_get_size(buf.p);
    }
};
struct StagingPool {
    std::vector<std::unique_ptr<StagingBank>> banks =
        std::vector<std::unique_ptr<StagingBank>>(2);
};
struct Job { int expert; int slot; int64_t token, lane; };
struct RouteSnapshot {
    ggml_tensor tensor{};
    std::vector<char> bytes;
};
using RouteSnapshots = std::map<const ggml_tensor *, RouteSnapshot>;
bool on_host(const ggml_tensor * t) { return !t->buffer || ggml_backend_buffer_is_host(t->buffer); }
ggml_tensor row_view(const ggml_tensor * t, size_t offset, int64_t width) {
    ggml_tensor v = *t;
    v.ne[0] = width; v.ne[1] = v.ne[2] = v.ne[3] = 1;
    v.nb[1] = v.nb[2] = v.nb[3] = width * sizeof(float);
    v.data = (char *) t->data + offset;
    v.view_src = nullptr; v.view_offs = 0; v.op = GGML_OP_NONE;
    return v;
}

// Gather independent routed rows. No router weights or expert contributions are changed.
struct Task {
    Context ctx;
    Buffer buf;
    Buffer row_buf;
    ggml_tensor * w = nullptr, * x = nullptr, * ids = nullptr, * y = nullptr;
    ggml_tensor * w_up = nullptr, * w_down = nullptr;
    std::vector<ggml_tensor *> hidden;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * gather_from = nullptr, * gather_ids = nullptr, * gathered = nullptr;
    ggml_tensor * scatter_to = nullptr, * scatter_ids = nullptr, * scatter_values = nullptr, * scattered = nullptr;
    ggml_cgraph * gather_graph = nullptr, * scatter_graph = nullptr;
    std::unique_ptr<Context> return_ctx;
    Buffer return_buf;
    ggml_tensor * return_values = nullptr, * return_ids = nullptr, * return_to = nullptr, * returned = nullptr;
    ggml_cgraph * return_graph = nullptr;
    std::vector<float> inputs, outputs;
    std::vector<int32_t> indices;
    std::vector<int32_t> gather_rows;
    std::vector<int64_t> scatter_rows;
    std::vector<Job> jobs;
    size_t capacity = 0;
    void * own_x = nullptr, * own_y = nullptr;
    ggml_backend_buffer_t own_x_buffer = nullptr, own_y_buffer = nullptr;
    bool direct_output = false;
    bool external_cpu_outputs = false;
    // Host vectors are sources of asynchronous index/input transfers. A task
    // cannot resize or overwrite them until this compute-stream serial ends.
    uint64_t pending_serial = 0;
    Task(ggml_backend_t backend, ggml_tensor * weights, size_t rows,
         ggml_tensor * weight_up = nullptr, ggml_tensor * weight_down = nullptr,
         bool packed_gate_up = false) : capacity(rows) {
        w = ggml_dup_tensor(ctx.p, weights);
        *w = *weights;
        x = ggml_new_tensor_3d(ctx.p, GGML_TYPE_F32, weights->ne[0], 1, rows);
        ids = ggml_new_tensor_2d(ctx.p, GGML_TYPE_I32, 1, rows);
        if (packed_gate_up && weight_down) {
            w_down = ggml_dup_tensor(ctx.p, weight_down); *w_down = *weight_down;
            auto * gate_up = ggml_mul_mat_id(ctx.p, w, x, ids);
            auto * activation = ggml_swiglu(ctx.p, gate_up);
            hidden = {gate_up, activation};
            y = ggml_mul_mat_id(ctx.p, w_down, activation, ids);
        } else if (weight_up && weight_down) {
            w_up = ggml_dup_tensor(ctx.p, weight_up); *w_up = *weight_up;
            w_down = ggml_dup_tensor(ctx.p, weight_down); *w_down = *weight_down;
            auto * gate = ggml_mul_mat_id(ctx.p, w, x, ids);
            auto * up = ggml_mul_mat_id(ctx.p, w_up, x, ids);
            auto * activation = ggml_swiglu_split(ctx.p, gate, up);
            hidden = {gate, up, activation};
            y = ggml_mul_mat_id(ctx.p, w_down, activation, ids);
        } else y = ggml_mul_mat_id(ctx.p, w, x, ids);
        graph = ggml_new_graph_custom(ctx.p, 32, false);
        ggml_build_forward_expand(graph, y);
        buf.p = ggml_backend_alloc_ctx_tensors(ctx.p, backend);
        if (!buf.p) throw std::runtime_error("expert task allocation failed");
        own_x = x->data; own_y = y->data;
        own_x_buffer = x->buffer; own_y_buffer = y->buffer;
        gather_from = ggml_new_tensor_2d(ctx.p, GGML_TYPE_F32, weights->ne[0], rows);
        gather_from->data = own_x; gather_from->buffer = own_x_buffer;
        gather_ids = ggml_new_tensor_1d(ctx.p, GGML_TYPE_I32, rows);
        gathered = ggml_get_rows(ctx.p, gather_from, gather_ids);
        gathered->data = own_x; gathered->buffer = own_x_buffer;
        gather_graph = ggml_new_graph_custom(ctx.p, 32, false); ggml_build_forward_expand(gather_graph, gathered);
        scatter_to = ggml_new_tensor_2d(ctx.p, GGML_TYPE_F32, y->ne[0], rows);
        scatter_to->data = own_y; scatter_to->buffer = own_y_buffer;
        scatter_ids = ggml_new_tensor_1d(ctx.p, GGML_TYPE_I64, rows);
        scatter_values = ggml_new_tensor_2d(ctx.p, GGML_TYPE_F32, y->ne[0], rows);
        scatter_values->data = own_y; scatter_values->buffer = own_y_buffer;
        scattered = ggml_set_rows(ctx.p, scatter_to, scatter_values, scatter_ids);
        scatter_graph = ggml_new_graph_custom(ctx.p, 32, false); ggml_build_forward_expand(scatter_graph, scattered);
        row_buf.p = ggml_backend_alloc_ctx_tensors(ctx.p, backend);
        if (!row_buf.p) throw std::runtime_error("expert row-index allocation failed");
        ++counters.expert_task_allocations;
    }
    static void matrix(ggml_tensor * t, int64_t rows) {
        t->ne[1] = rows; t->ne[2] = t->ne[3] = 1;
        t->nb[1] = t->ne[0] * sizeof(float); t->nb[2] = t->nb[3] = t->nb[1] * rows;
    }
    static void vector(ggml_tensor * t, int64_t count) {
        t->ne[0] = count; t->ne[1] = t->ne[2] = t->ne[3] = 1;
        t->nb[1] = t->nb[2] = t->nb[3] = count * t->nb[0];
    }
    void run(ggml_backend_t backend, ggml_tensor * weights, const ggml_tensor * source,
             std::vector<Job> work, bool cached, ggml_tensor * dst,
             ggml_tensor * weight_up = nullptr, ggml_tensor * weight_down = nullptr) {
        jobs = std::move(work);
        external_cpu_outputs = false;
        const int64_t n = (int64_t) jobs.size();
        if ((size_t) n > capacity) throw std::runtime_error("expert task exceeds its buffer");
        *w = *weights;
        if (w_up) { *w_up = *weight_up; *w_down = *weight_down; }
        else if (w_down) *w_down = *weight_down;
        // CUDA's multi-column MMVQ changes its reduction layout. Exact short
        // verification uses independent expert lanes of one pseudo token, so
        // every row keeps decode's single-column arithmetic.
        const bool source_decode = source->ne[2] == 1;
        const bool quantized_graph = ggml_is_quantized(weights->type) &&
            (!w_up || ggml_is_quantized(weight_up->type)) &&
            (!w_down || ggml_is_quantized(weight_down->type));
        const bool exact_window = cached && config.exact_verify && quantized_graph &&
            source->ne[2] > 1 && source->ne[2] <= config.native_exact_max_tokens;
        const bool decode = source_decode || exact_window;
        if (exact_window) ++counters.exact_verify_calls;
        x->ne[1] = decode ? n : 1; x->ne[2] = decode ? 1 : n;
        x->nb[2] = x->nb[1] * x->ne[1]; x->nb[3] = x->nb[2] * x->ne[2];
        ids->ne[0] = decode ? n : 1; ids->ne[1] = decode ? 1 : n;
        ids->nb[1] = sizeof(int32_t) * ids->ne[0]; ids->nb[2] = ids->nb[3] = ids->nb[1] * ids->ne[1];
        y->ne[1] = decode ? n : 1; y->ne[2] = decode ? 1 : n;
        y->nb[2] = y->nb[1] * y->ne[1]; y->nb[3] = y->nb[2] * y->ne[2];
        x->data = own_x; x->buffer = own_x_buffer;
        y->data = own_y; y->buffer = own_y_buffer;
        bool full_lanes = source_decode;
        for (size_t j = 0; j < jobs.size(); ++j) full_lanes = full_lanes && jobs[j].lane == (int64_t) j;
        const bool device_input = cached && !on_host(source);
        bool native_layout = !exact_window && config.preserve_verify_layout && device_input && !on_host(dst) &&
            source->ne[2] <= config.native_exact_max_tokens && source->ne[2] == dst->ne[2] &&
            (source->ne[1] == 1 || source->ne[1] == dst->ne[1]) &&
            n == dst->ne[1] * dst->ne[2] && ggml_is_contiguous(source) && ggml_is_contiguous(dst);
        for (size_t j = 0; native_layout && j < jobs.size(); ++j)
            native_layout = jobs[j].token * dst->ne[1] + jobs[j].lane == (int64_t) j;
        const bool direct_input = native_layout || (device_input && source_decode &&
            (source->ne[1] == 1 || (full_lanes && source->ne[1] == n)) && ggml_is_contiguous(source));
        const bool gather_device = device_input && !direct_input && ggml_is_contiguous(source);
        if (direct_input) {
            x->data = source->data; x->buffer = source->buffer;
            std::copy(source->ne, source->ne + GGML_MAX_DIMS, x->ne);
            std::copy(source->nb, source->nb + GGML_MAX_DIMS, x->nb);
        }
        direct_output = native_layout || (cached && !on_host(dst) && full_lanes && dst->ne[1] == n && ggml_is_contiguous(dst));
        if (direct_output) { y->data = dst->data; y->buffer = dst->buffer; }
        if (native_layout) {
            ids->ne[0] = dst->ne[1]; ids->ne[1] = dst->ne[2];
            ids->nb[1] = sizeof(int32_t) * ids->ne[0]; ids->nb[2] = ids->nb[3] = ids->nb[1] * ids->ne[1];
            std::copy(dst->ne, dst->ne + GGML_MAX_DIMS, y->ne);
            std::copy(dst->nb, dst->nb + GGML_MAX_DIMS, y->nb);
            ++counters.preserved_layout_calls;
        }
        for (auto * h : hidden) {
            h->ne[1] = ids->ne[0]; h->ne[2] = ids->ne[1]; h->ne[3] = 1;
            h->nb[1] = h->ne[0] * sizeof(float); h->nb[2] = h->nb[1] * h->ne[1]; h->nb[3] = h->nb[2] * h->ne[2];
        }
        inputs.resize((size_t) n * source->ne[0]); indices.resize((size_t) n);
        outputs.resize((size_t) n * y->ne[0]);
        for (size_t j = 0; j < jobs.size(); ++j) {
            const auto & job = jobs[j];
            const size_t offset = job.token * source->nb[2] + (job.lane % source->ne[1]) * source->nb[1];
            const auto * row = (const char *) source->data + offset;
            if (device_input && !direct_input && !gather_device) {
                auto from = row_view(source, offset, source->ne[0]);
                auto to = row_view(x, j * source->ne[0] * sizeof(float), source->ne[0]);
                ggml_backend_tensor_copy(&from, &to);
            } else if (!device_input && source->nb[0] == sizeof(float))
                std::memcpy(&inputs[j * source->ne[0]], row, source->ne[0] * sizeof(float));
            else if (!device_input) for (int64_t k = 0; k < source->ne[0]; ++k)
                std::memcpy(&inputs[j * source->ne[0] + k], row + k * source->nb[0], sizeof(float));
            indices[j] = cached ? job.slot : job.expert;
        }
        if (gather_device) {
            gather_from->data = source->data; gather_from->buffer = source->buffer;
            matrix(gather_from, source->ne[1] * source->ne[2]);
            vector(gather_ids, n); matrix(gathered, n);
            gather_rows.resize(jobs.size());
            for (size_t j = 0; j < jobs.size(); ++j)
                gather_rows[j] = (int32_t) (jobs[j].token * source->ne[1] + jobs[j].lane % source->ne[1]);
            ggml_backend_tensor_set_async(backend, gather_ids, gather_rows.data(), 0, gather_rows.size() * sizeof(int32_t));
            if (ggml_backend_graph_compute_async(backend, gather_graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("GPU expert gather failed");
        }
        // Task-owned host vectors remain valid until finish(). Submit transfers
        // on the compute stream instead of synchronizing the per-thread stream
        // separately for each tiny route/index tensor.
        if (!device_input) ggml_backend_tensor_set_async(backend, x, inputs.data(), 0, inputs.size() * sizeof(float));
        ggml_backend_tensor_set_async(backend, ids, indices.data(), 0, indices.size() * sizeof(int32_t));
        if (ggml_backend_graph_compute_async(backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("expert graph failed");
    }
    void finish(ggml_backend_t backend, ggml_tensor * dst, ggml_backend_t return_backend = nullptr, bool synchronize = true) {
        if (direct_output) { if (synchronize) ggml_backend_synchronize(backend); return; }
        if (!on_host(y) && !on_host(dst) && ggml_is_contiguous(dst)) {
            scatter_to->data = dst->data; scatter_to->buffer = dst->buffer;
            matrix(scatter_to, dst->ne[1] * dst->ne[2]);
            scatter_values->data = y->data; scatter_values->buffer = y->buffer;
            matrix(scatter_values, (int64_t) jobs.size()); vector(scatter_ids, (int64_t) jobs.size());
            matrix(scattered, scatter_to->ne[1]); scattered->data = dst->data; scattered->buffer = dst->buffer;
            scatter_rows.resize(jobs.size());
            for (size_t j = 0; j < jobs.size(); ++j) scatter_rows[j] = jobs[j].token * dst->ne[1] + jobs[j].lane;
            ggml_backend_tensor_set_async(backend, scatter_ids, scatter_rows.data(), 0, scatter_rows.size() * sizeof(int64_t));
            if (ggml_backend_graph_compute_async(backend, scatter_graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("GPU expert scatter failed");
            if (synchronize) ggml_backend_synchronize(backend);
            return;
        }
        if (!external_cpu_outputs) {
            ggml_backend_synchronize(backend);
            ggml_backend_tensor_get(y, outputs.data(), 0, outputs.size() * sizeof(float));
        }
        if (return_backend && !on_host(dst)) {
            if (!return_ctx) {
                return_ctx = std::make_unique<Context>();
                return_values = ggml_new_tensor_2d(return_ctx->p, GGML_TYPE_F32, y->ne[0], capacity);
                return_ids = ggml_new_tensor_1d(return_ctx->p, GGML_TYPE_I64, capacity);
                return_to = ggml_new_tensor_2d(return_ctx->p, GGML_TYPE_F32, y->ne[0], capacity);
                return_to->data = dst->data; return_to->buffer = dst->buffer;
                returned = ggml_set_rows(return_ctx->p, return_to, return_values, return_ids);
                returned->data = dst->data; returned->buffer = dst->buffer;
                return_graph = ggml_new_graph_custom(return_ctx->p, 32, false);
                ggml_build_forward_expand(return_graph, returned);
                return_buf.p = ggml_backend_alloc_ctx_tensors(return_ctx->p, return_backend);
                if (!return_buf.p) throw std::runtime_error("CPU expert return staging allocation failed");
            }
            matrix(return_values, (int64_t) jobs.size()); vector(return_ids, (int64_t) jobs.size());
            matrix(return_to, dst->ne[1] * dst->ne[2]); return_to->data = dst->data; return_to->buffer = dst->buffer;
            matrix(returned, return_to->ne[1]); returned->data = dst->data; returned->buffer = dst->buffer;
            scatter_rows.resize(jobs.size());
            for (size_t j = 0; j < jobs.size(); ++j) scatter_rows[j] = jobs[j].token * dst->ne[1] + jobs[j].lane;
            ggml_backend_tensor_set_async(return_backend, return_values, outputs.data(), 0, outputs.size() * sizeof(float));
            ggml_backend_tensor_set_async(return_backend, return_ids, scatter_rows.data(), 0, scatter_rows.size() * sizeof(int64_t));
            if (ggml_backend_graph_compute_async(return_backend, return_graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("CPU expert return scatter failed");
            if (synchronize) ggml_backend_synchronize(return_backend);
            return;
        }
        bool whole = jobs.size() == (size_t) (dst->ne[1] * dst->ne[2]) && ggml_is_contiguous(dst);
        for (size_t j = 0; j < jobs.size(); ++j) whole = whole &&
            jobs[j].token * dst->ne[1] + jobs[j].lane == (int64_t) j;
        if (whole) { ggml_backend_tensor_set(dst, outputs.data(), 0, outputs.size() * sizeof(float)); return; }
        for (size_t j = 0; j < jobs.size(); ++j) {
            const auto & job = jobs[j];
            auto * row = (char *) dst->data + job.token * dst->nb[2] + job.lane * dst->nb[1];
            if (!on_host(dst)) {
                auto to = row_view(dst, job.token * dst->nb[2] + job.lane * dst->nb[1], dst->ne[0]);
                ggml_backend_tensor_set(&to, &outputs[j * dst->ne[0]], 0, dst->ne[0] * sizeof(float));
            } else if (dst->nb[0] == sizeof(float))
                std::memcpy(row, &outputs[j * dst->ne[0]], dst->ne[0] * sizeof(float));
            else for (int64_t k = 0; k < dst->ne[0]; ++k)
                std::memcpy(row + k * dst->nb[0], &outputs[j * dst->ne[0] + k], sizeof(float));
        }
    }
};

struct Backend {
    ggml_backend_t cpu = nullptr, gpu = nullptr, copy_gpu = nullptr;
#if defined(STRATA_A3B_STRATA_NATIVE)
    StrataNativeExperts * strata_native = nullptr;
#endif
    uint64_t submitted_serial = 0, completed_serial = 0;
    ggml_threadpool_t cpu_pool = nullptr;
    std::unique_ptr<CpuExpertPool> native_cpu_pool;
    Buffer hybrid_input;
    Event hybrid_input_ready;
    size_t hybrid_input_capacity = 0;
    std::vector<CpuExpertJob> native_cpu_jobs;
    char native_cpu_return_tag = 0;
    using CacheKey = std::tuple<const void *, int, int64_t, int64_t, int64_t>;
    std::map<CacheKey, std::unique_ptr<Cache>> caches;
    using TaskKey = std::tuple<int, int64_t, int64_t, bool, const void *, const void *>;
    std::map<TaskKey, std::unique_ptr<Task>> tasks;
    using FusedTaskKey = std::tuple<int, int, int, int64_t, int64_t, bool, const void *, const void *, const void *, size_t, const void *>;
    std::map<FusedTaskKey, std::unique_ptr<Task>> fused_tasks;
    using StagingKey = std::tuple<int, int, int, int64_t, int64_t, int64_t, int64_t>;
    std::map<StagingKey, std::unique_ptr<StagingPool>> staging;
    Backend() {
        if (!config.gpu_only_experts &&
            !(config.hybrid_experts && config.pcie_fraction == 1.0 && !config.prefill_cpu_share)) {
            cpu = ggml_backend_cpu_init();
            if (!cpu) throw std::runtime_error("CPU backend init failed");
            ggml_backend_cpu_set_n_threads(cpu, config.threads);
            auto pool_params = ggml_threadpool_params_default(config.threads);
            if (config.cpu_expert_pool) pool_params.paused = true;
            cpu_pool = ggml_threadpool_new(&pool_params);
            if (!cpu_pool) {
                ggml_backend_free(cpu); cpu = nullptr;
                throw std::runtime_error("CPU expert threadpool init failed");
            }
            ggml_backend_cpu_set_threadpool(cpu, cpu_pool);
        }
        if (config.gpu) {
            gpu = ggml_backend_dev_init(config.gpu, nullptr);
            if (!gpu) {
                if (cpu) { ggml_backend_free(cpu); cpu = nullptr; }
                if (cpu_pool) { ggml_threadpool_free(cpu_pool); cpu_pool = nullptr; }
                throw std::runtime_error("GPU backend init failed");
            }
            if (config.gpu_only_experts || config.hybrid_experts) {
                copy_gpu = ggml_backend_dev_init(config.gpu, nullptr);
                if (!copy_gpu) {
                    ggml_backend_free(gpu); gpu = nullptr;
                    if (cpu) { ggml_backend_free(cpu); cpu = nullptr; }
                    if (cpu_pool) { ggml_threadpool_free(cpu_pool); cpu_pool = nullptr; }
                    throw std::runtime_error("GPU expert copy backend init failed");
                }
            }
        }
        live_backends.push_back(this);
    }
    void synchronize_gpu() {
        if (gpu) ggml_backend_synchronize(gpu);
        completed_serial = submitted_serial;
    }
    void drain() {
        synchronize_gpu();
        if (copy_gpu) ggml_backend_synchronize(copy_gpu);
    }
    void await_task(Task * task) {
        if (config.async_expert_events && task && task->pending_serial > completed_serial) {
            // A host-route caller or two ops sharing one cached snapshot can
            // reuse a task without the usual route-read synchronization.
            synchronize_gpu();
            ++counters.async_task_reuse_waits;
        }
    }
    void read_routes(const ggml_tensor * ids, RouteSnapshot & snapshot) {
        const auto started = std::chrono::steady_clock::now();
        snapshot.bytes.resize(ggml_nbytes(ids));
        if (config.async_expert_events) {
            // The scheduler has queued the producer's event on this stream.
            // One wait makes both the routes and all earlier task host sources
            // safe, instead of joining the producer then a per-thread copy.
            ggml_backend_tensor_get_async(gpu, ids, snapshot.bytes.data(), 0, snapshot.bytes.size());
            synchronize_gpu();
            ++counters.async_route_waits;
        } else ggml_backend_tensor_get(ids, snapshot.bytes.data(), 0, snapshot.bytes.size());
        snapshot.tensor = *ids; snapshot.tensor.data = snapshot.bytes.data();
        ++counters.route_reads;
        counters.route_read_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
    }
    ~Backend() {
        drain();
#if defined(STRATA_A3B_STRATA_NATIVE)
        strata_native_experts_destroy(strata_native);
#endif
        tasks.clear();
        fused_tasks.clear();
        staging.clear();
        native_cpu_pool.reset();
        if (hybrid_input_ready.p) { ggml_backend_event_free(hybrid_input_ready.p); hybrid_input_ready.p = nullptr; }
        if (hybrid_input.p) {
            counters.hybrid_input_bytes -= std::min(counters.hybrid_input_bytes,
                (uint64_t) ggml_backend_buffer_get_size(hybrid_input.p));
            ggml_backend_buffer_free(hybrid_input.p); hybrid_input.p = nullptr;
        }
        caches.clear();
        live_backends.erase(std::remove(live_backends.begin(), live_backends.end(), this), live_backends.end());
        if (copy_gpu) ggml_backend_free(copy_gpu);
        if (gpu) ggml_backend_free(gpu);
        if (cpu) ggml_backend_free(cpu);
        if (cpu_pool) ggml_threadpool_free(cpu_pool);
    }
    Task & run(ggml_backend_t backend, ggml_tensor * w, const ggml_tensor * x,
               std::vector<Job> work, bool cached, ggml_tensor * dst, const void * task_tag = nullptr) {
        const void * graph_weights = config.stable_tasks && x->ne[2] <= config.native_exact_max_tokens ? w->data : nullptr;
        auto & task = tasks[{(int) w->type, w->ne[0], w->ne[1], cached, graph_weights, task_tag}];
        await_task(task.get());
        if (!task || task->capacity < work.size()) {
            size_t capacity = 8;
            while (capacity < work.size()) capacity *= 2;
            task = std::make_unique<Task>(backend, w, capacity);
        }
        task->run(backend, w, x, std::move(work), cached, dst);
        return *task;
    }
    Task & run_fused(ggml_backend_t backend, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down,
                     const ggml_tensor * x, std::vector<Job> work, bool cached, ggml_tensor * dst,
                     const void * task_tag = nullptr) {
        const bool short_window = x->ne[2] <= config.native_exact_max_tokens;
        size_t capacity = 8; while (capacity < work.size()) capacity *= 2;
        // Capacity buckets separate decode and larger verification captures
        // without retaining every possible hit/miss count over a long chat.
        // Large prompt groups share bounded scratch across layers.
        auto key = FusedTaskKey{(int) gate->type, (int) up->type, (int) down->type, gate->ne[0], gate->ne[1], cached,
            short_window ? gate->data : nullptr, short_window ? up->data : nullptr, short_window ? down->data : nullptr,
            short_window ? capacity : 0, task_tag};
        auto & task = fused_tasks[key];
        await_task(task.get());
        if (!task || task->capacity < work.size()) {
            task = std::make_unique<Task>(backend, gate, capacity, up, down);
        }
        task->run(backend, gate, x, std::move(work), cached, dst, up, down);
        return *task;
    }
    Task & run_packed_gate_up(ggml_backend_t backend, ggml_tensor * gate_up, ggml_tensor * down,
                             const ggml_tensor * x, std::vector<Job> work, ggml_tensor * dst,
                             const void * task_tag) {
        size_t capacity = 8; while (capacity < work.size()) capacity *= 2;
        auto key = FusedTaskKey{(int) gate_up->type, -2, (int) down->type,
            gate_up->ne[0], gate_up->ne[1], true, nullptr, nullptr, nullptr, 0, task_tag};
        auto & task = fused_tasks[key];
        await_task(task.get());
        if (!task || task->capacity < work.size())
            task = std::make_unique<Task>(backend, gate_up, capacity, nullptr, down, true);
        task->run(backend, gate_up, x, std::move(work), true, dst, nullptr, down);
        return *task;
    }
    ggml_tensor queue_hybrid_input(const ggml_tensor * source) {
        ggml_tensor host = *source;
        if (on_host(source)) return host;
        const size_t bytes = ggml_nbytes(source);
        if (!hybrid_input.p || hybrid_input_capacity < bytes) {
            auto parent = ggml_backend_dev_host_buffer_type(config.gpu);
            if (!parent) parent = ggml_backend_cpu_buffer_type();
            auto allocated = ggml_backend_buft_alloc_buffer(parent, bytes);
            if (!allocated) throw std::runtime_error("hybrid CPU input allocation failed");
            if (ggml_backend_buffer_get_type(allocated) != parent || parent == ggml_backend_cpu_buffer_type()) {
                ++counters.hybrid_input_fallbacks;
                std::fprintf(stderr, "StrataMoE hybrid: CPU input buffer is pageable; DMA may synchronize\n");
            }
            if (hybrid_input.p) {
                counters.hybrid_input_bytes -= ggml_backend_buffer_get_size(hybrid_input.p);
                ggml_backend_buffer_free(hybrid_input.p);
            }
            hybrid_input.p = allocated; hybrid_input_capacity = bytes;
            counters.hybrid_input_bytes += ggml_backend_buffer_get_size(allocated);
            ++counters.hybrid_input_allocations;
        }
        if (!hybrid_input_ready.p) hybrid_input_ready.p = ggml_backend_event_new(config.gpu);
        if (!hybrid_input_ready.p) throw std::runtime_error("hybrid CPU input event allocation failed");
        // read_routes already joined the producer on the compute stream. This
        // copy stream reads the immutable input without joining subsequent hits.
        ggml_backend_tensor_get_async(copy_gpu, source, ggml_backend_buffer_get_base(hybrid_input.p), 0, bytes);
        ggml_backend_event_record(hybrid_input_ready.p, copy_gpu);
        host.data = ggml_backend_buffer_get_base(hybrid_input.p);
        host.buffer = hybrid_input.p; host.view_src = nullptr; host.view_offs = 0;
        return host;
    }
    Task & run_native_cpu_pool(ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down,
                               const ggml_tensor * source, std::vector<Job> work) {
        if (!native_cpu_pool) {
            CpuExpertPool::Config params;
            params.threads = config.threads;
            native_cpu_pool = std::make_unique<CpuExpertPool>(params);
            std::fprintf(stderr, "StrataMoE hybrid CPU pool: %d participants, %d physical cores; original GGUF quantization, shared input quantization\n",
                native_cpu_pool->threads(), native_cpu_pool->physical_cores());
        }
        // The generic fallback pool can have been awakened by a long prompt;
        // it must not spin beside the persistent native row workers.
        if (cpu_pool) ggml_threadpool_pause(cpu_pool);
        size_t capacity = 8; while (capacity < work.size()) capacity *= 2;
        auto key = FusedTaskKey{(int) gate->type, (int) up->type, (int) down->type, gate->ne[0], gate->ne[1], false,
            nullptr, nullptr, nullptr, 0, &native_cpu_return_tag};
        auto & task = fused_tasks[key];
        if (!task || task->capacity < work.size()) task = std::make_unique<Task>(cpu, gate, capacity, up, down);
        task->jobs = std::move(work); task->direct_output = false; task->external_cpu_outputs = true;
        native_cpu_jobs.resize(task->jobs.size());
        for (size_t j = 0; j < task->jobs.size(); ++j) {
            const auto & job = task->jobs[j];
            native_cpu_jobs[j] = {job.expert, (int) job.token, (int) job.lane};
        }
        CpuExpertWeights weights;
        weights.gate = {gate->data, gate->nb[1], gate->nb[2], gate->type};
        weights.up = {up->data, up->nb[1], up->nb[2], up->type};
        weights.down = {down->data, down->nb[1], down->nb[2], down->type};
        weights.hidden = (int) gate->ne[0]; weights.ffn = (int) gate->ne[1]; weights.experts = (int) gate->ne[2];
        CpuExpertInput input{(const float *) source->data, (int) source->ne[1], (int) source->ne[2], source->nb[1], source->nb[2]};
        const auto before = native_cpu_pool->stats();
        native_cpu_pool->execute(weights, input, native_cpu_jobs, task->outputs);
        const auto after = native_cpu_pool->stats();
        counters.cpu_pool_calls += after.calls - before.calls; counters.cpu_pool_jobs += after.jobs - before.jobs;
        counters.cpu_pool_unique_inputs += after.unique_inputs - before.unique_inputs;
        counters.cpu_pool_unique_experts += after.unique_experts - before.unique_experts;
        counters.cpu_pool_input_quant_ms += after.input_quant_ms - before.input_quant_ms;
        counters.cpu_pool_gate_up_ms += after.gate_up_ms - before.gate_up_ms;
        counters.cpu_pool_hidden_quant_ms += after.hidden_quant_ms - before.hidden_quant_ms;
        counters.cpu_pool_down_ms += after.down_ms - before.down_ms;
        return *task;
    }
    Cache & cache(ggml_tensor * w) {
        if (shared_expert_store_enabled()) return static_cache_store().cache(w);
        auto & entry = caches[{w->data, (int) w->type, w->ne[0], w->ne[1], w->ne[2]}];
        if (entry) return *entry;
        auto pending = std::make_unique<Cache>();
        auto & c = *pending;
        const int ne = (int) w->ne[2];
        c.tensor_name = w->name;
        c.slot.assign(ne, -1); c.frequency.assign(ne, 0); c.route_count.assign(ne, 0);
        c.capacity = gpu ? (int) std::min<size_t>(ne, (size_t)
            ((long double) config.cache_bytes * ggml_nbytes(w) / config.expert_bytes) / w->nb[2]) : 0;
        int profile_layer = -1;
        if (cache_plan) {
            if (std::sscanf(w->name, "blk.%d.", &profile_layer) != 1 || profile_layer < 0 ||
                profile_layer >= expert_profile_layers || w->type != GGML_TYPE_Q8_0 ||
                w->ne[2] != expert_profile_experts || w->ne[0] * w->ne[1] != 2048 * 512)
                throw std::runtime_error("cache profile requires matching complete Q8 expert tensors");
            c.capacity = (int) cache_plan->expert_ids[profile_layer].size();
        }
        c.owner.assign(c.capacity, -1);
        if (c.capacity) {
            c.weights = ggml_new_tensor_3d(c.ctx.p, w->type, w->ne[0], w->ne[1], c.capacity);
            c.buf.p = ggml_backend_alloc_ctx_tensors(c.ctx.p, gpu);
            if (!c.buf.p) throw std::runtime_error("GPU expert cache does not fit; lower --cache-mib");
            ggml_backend_buffer_set_usage(c.buf.p, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_buffer_clear(c.buf.p, 0);
            counters.cache_bytes += ggml_backend_buffer_get_size(c.buf.p);
            if (cache_plan) for (int slot = 0; slot < c.capacity; ++slot) {
                const int expert = cache_plan->expert_ids[profile_layer][slot];
                upload(c, w, expert, slot);
                c.frequency[expert] = config.cache_decay_interval > 0 ? 256 : 1;
            }
        }
        entry = std::move(pending);
        return *entry;
    }
    void upload(Cache & c, ggml_tensor * w, int e, int slot) {
        if (c.owner[slot] >= 0) c.slot[c.owner[slot]] = -1;
        c.owner[slot] = e; c.slot[e] = slot;
        ggml_backend_tensor_set(c.weights, (const char *) w->data + e * w->nb[2],
                                slot * c.weights->nb[2], w->nb[2]);
        ++counters.uploads; counters.upload_bytes += w->nb[2];
    }
    void compute_gpu_only(ggml_tensor * dst, const ggml_tensor * x, const std::vector<Job> & work,
                          const std::vector<bool> & used, const std::vector<ggml_tensor *> & source,
                          const std::vector<Cache *> & cache_set, bool defer_final_join = false) {
        if (!gpu || !copy_gpu) throw std::runtime_error("GPU-only experts have no GPU copy backend");
        const auto host_started = std::chrono::steady_clock::now();
        const bool fused = source.size() == 3;
        auto & primary = *cache_set[0];
        int capacity = primary.capacity;
        for (auto * c : cache_set) capacity = std::min(capacity, c->capacity);
        auto resident = [&](int e) {
            const int slot = primary.slot[e];
            if (slot < 0 || slot >= capacity) return -1;
            for (auto * c : cache_set) if (c->slot[e] != slot) return -1;
            return slot;
        };
        auto execute = [&](const std::vector<ggml_tensor *> & weights, std::vector<Job> jobs,
                           const void * task_tag) {
            if (jobs.empty()) return;
            const auto submit_started = std::chrono::steady_clock::now();
            auto & task = fused ? (weights.size() == 2 ?
                                     run_packed_gate_up(gpu, weights[0], weights[1], x,
                                         std::move(jobs), dst, task_tag) :
                                     run_fused(gpu, weights[0], weights[1], weights[2], x,
                                         std::move(jobs), true, dst, task_tag)) :
                                 run(gpu, weights[0], x, std::move(jobs), true, dst, task_tag);
            // Scatter on the same GPU stream now. A bank's consumed event
            // protects task-owned async inputs and weights until reuse.
            task.finish(gpu, dst, nullptr, false);
            if (config.async_expert_events) task.pending_serial = ++submitted_serial;
            counters.expert_task_submit_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - submit_started).count();
        };
        std::vector<ggml_tensor *> cached_weights;
        for (auto * c : cache_set) cached_weights.push_back(c->weights);
        std::vector<Job> hits, misses;
        std::vector<int> experts;
        std::vector<bool> missing(used.size(), false);
        for (auto job : work) {
            job.slot = resident(job.expert);
            if (job.slot >= 0) hits.push_back(job);
            else {
                misses.push_back(job);
                if (!missing[job.expert]) { missing[job.expert] = true; experts.push_back(job.expert); }
            }
        }
        const uint64_t projections = fused ? 3 : 1;
        counters.hits += projections * hits.size(); counters.misses += projections * misses.size();
        counters.gpu_rows += projections * work.size(); counters.gpu_only_rows += work.size();
        if (fused) counters.fused_gpu_rows += work.size();
        // Submit resident work before host packing or any expert upload.
        execute(cached_weights, std::move(hits), nullptr);
        if (!experts.empty()) {
            auto key = StagingKey{(int) source[0]->type,
                fused ? (int) source[1]->type : -1, fused ? (int) source[2]->type : -1,
                source[0]->ne[0], source[0]->ne[1],
                fused ? source[2]->ne[0] : 0, fused ? source[2]->ne[1] : 0};
            auto & pool = staging[key];
            if (!pool) pool = std::make_unique<StagingPool>();
            const size_t ring_banks = x->ne[2] > 8 ? (size_t) config.expert_ring_banks : 2;
            if (pool->banks.size() < ring_banks) pool->banks.resize(ring_banks);
            if (cache_admission && !defer_final_join && x->ne[2] > config.native_exact_max_tokens) {
                // Fill an empty persistent cache with the hottest prompt
                // experts first. Live hits remain protected during uploads.
                std::stable_sort(experts.begin(), experts.end(), [&](int a, int b) {
                    return primary.frequency[a] > primary.frequency[b];
                });
            }
            // Strata's stager prepares pageable expert bytes on a host worker
            // while earlier groups run on the GPU. The source GGUF arenas here
            // cannot be pinned as one 32 GiB allocation on this Windows host.
            // Host-bank reuse waits only for its previous DMA; GPU-bank reuse
            // remains protected separately by bank.consumed below.
            struct HostStager {
                std::mutex mu;
                std::condition_variable cv;
                std::vector<int64_t> prepared, copied, issued, consumed;
                bool stop = false;
                std::exception_ptr failure;
                std::thread worker, issuer;
                explicit HostStager(size_t n) : prepared(n, -1), copied(n, -1),
                    issued(n, -1), consumed(n, -1) {}
                ~HostStager() {
                    { std::lock_guard<std::mutex> lock(mu); stop = true; }
                    cv.notify_all();
                    if (worker.joinable()) worker.join();
                    if (issuer.joinable()) issuer.join();
                }
            };
            const bool copy_ahead = config.prefill_copy_ahead && config.pageable_experts &&
                !cache_admission && x->ne[2] >= 1024;
            const bool issuer_enabled = copy_ahead && config.prefill_independent_issuer;
            const bool packed_gate_up = issuer_enabled && config.prefill_packed_gate_up && fused;
            std::unique_ptr<HostStager> stager;
            if (copy_ahead) {
                for (size_t i = 0; i < ring_banks; ++i) {
                    auto & pending = pool->banks[i];
                    if (!pending) pending = std::make_unique<StagingBank>(gpu, source);
                    pending->ensure_host_bank();
                    if (packed_gate_up) pending->ensure_packed_gate_up(gpu);
                }
                stager = std::make_unique<HostStager>(ring_banks);
                auto * stage = stager.get();
                auto * stage_pool = pool.get();
                stage->worker = std::thread([&, stage, stage_pool] {
                    try {
                        const size_t groups = (experts.size() + config.expert_staging_slots - 1) /
                            config.expert_staging_slots;
                        for (size_t group = 0; group < groups; ++group) {
                            const size_t i = group % ring_banks;
                            auto & bank = *stage_pool->banks[i];
                            if (group >= ring_banks) {
                                std::unique_lock<std::mutex> lock(stage->mu);
                                stage->cv.wait(lock, [&] {
                                    return stage->stop || stage->copied[i] >= (int64_t) (group - ring_banks);
                                });
                                if (stage->stop) return;
                                lock.unlock();
                                ggml_backend_event_synchronize(bank.copied.p);
                            }
                            const size_t first = group * config.expert_staging_slots;
                            const size_t end = std::min(experts.size(), first + config.expert_staging_slots);
                            for (size_t j = first; j < end; ++j) {
                                const int e = experts[j];
                                const size_t slot = j - first;
                                for (size_t p = 0; p < source.size(); ++p) {
                                    const auto * src = source[p];
                                    auto * host = bank.host_weights[p];
                                    std::memcpy((char *) host->data + slot * host->nb[2],
                                        (const char *) src->data + e * src->nb[2], src->nb[2]);
                                }
                            }
                            { std::lock_guard<std::mutex> lock(stage->mu); stage->prepared[i] = (int64_t) group; }
                            stage->cv.notify_all();
                        }
                    } catch (...) {
                        { std::lock_guard<std::mutex> lock(stage->mu);
                          stage->failure = std::current_exception(); stage->stop = true; }
                        stage->cv.notify_all();
                    }
                });
                // Keep all long-prefill H2D submissions on a dedicated
                // backend thread. The host worker may refill a pinned bank
                // after its previous DMA, whereas this issuer may reuse its
                // GPU weights only after the compute stream consumes them.
                if (issuer_enabled) stage->issuer = std::thread([&, stage, stage_pool] {
                    try {
                        const size_t groups = (experts.size() + config.expert_staging_slots - 1) /
                            config.expert_staging_slots;
                        for (size_t group = 0; group < groups; ++group) {
                            const size_t i = group % ring_banks;
                            auto & bank = *stage_pool->banks[i];
                            {
                                std::unique_lock<std::mutex> lock(stage->mu);
                                stage->cv.wait(lock, [&] {
                                    return stage->stop || stage->prepared[i] == (int64_t) group || stage->failure;
                                });
                                if (stage->failure) std::rethrow_exception(stage->failure);
                                if (stage->stop) return;
                            }
                            if (group >= ring_banks) {
                                std::unique_lock<std::mutex> lock(stage->mu);
                                stage->cv.wait(lock, [&] {
                                    return stage->stop || stage->consumed[i] >= (int64_t) (group - ring_banks);
                                });
                                if (stage->stop) return;
                                lock.unlock();
                                // Mirror Strata's copy stream: enqueue the GPU
                                // dependency and continue issuing. A host wait
                                // here serialized the issuer behind compute.
                                ggml_backend_event_wait(copy_gpu, bank.consumed.p);
                            }
                            const size_t first = group * config.expert_staging_slots;
                            const size_t end = std::min(experts.size(), first + config.expert_staging_slots);
                            for (size_t j = first; j < end; ++j) {
                                const size_t slot = j - first;
                                for (size_t p = 0; p < source.size(); ++p) {
                                    auto * host = bank.host_weights[p];
                                    auto * target = packed_gate_up && p < 2 ?
                                        bank.packed_gate_up : bank.weights[p];
                                    const size_t bytes = source[p]->nb[2];
                                    const char * row = (const char *) host->data + slot * host->nb[2];
                                    const size_t offset = slot * target->nb[2] +
                                        (packed_gate_up && p == 1 ? bytes : 0);
                                    ggml_backend_tensor_set_async(copy_gpu, target, row, offset, bytes);
                                    ++counters.uploads; counters.upload_bytes += bytes;
                                    counters.staging_upload_bytes += bytes;
                                }
                                ++counters.staged_experts;
                            }
                            ++counters.staging_batches;
                            ggml_backend_event_record(bank.copied.p, copy_gpu);
                            { std::lock_guard<std::mutex> lock(stage->mu);
                              stage->copied[i] = (int64_t) group;
                              stage->issued[i] = (int64_t) group; }
                            stage->cv.notify_all();
                        }
                    } catch (...) {
                        { std::lock_guard<std::mutex> lock(stage->mu);
                          stage->failure = std::current_exception(); stage->stop = true; }
                        stage->cv.notify_all();
                    }
                });
            }
            size_t batch = 0;
            for (size_t first = 0; first < experts.size(); first += config.expert_staging_slots, ++batch) {
                auto & pending = pool->banks[batch % ring_banks];
                if (!pending) pending = std::make_unique<StagingBank>(gpu, source);
                auto & bank = *pending;
                if (bank.live && !issuer_enabled) {
                    const auto wait_started = std::chrono::steady_clock::now();
                    ggml_backend_event_synchronize(bank.consumed.p);
                    counters.staging_reuse_wait_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - wait_started).count();
                    completed_serial = std::max(completed_serial, bank.consumed_serial);
                }
                if (config.pageable_experts) bank.ensure_host_bank();
                if (stager) {
                    std::unique_lock<std::mutex> lock(stager->mu);
                    stager->cv.wait(lock, [&] {
                        return stager->stop ||
                            (issuer_enabled ? stager->issued[batch % ring_banks] :
                                              stager->prepared[batch % ring_banks]) == (int64_t) batch;
                    });
                    if (stager->failure) std::rethrow_exception(stager->failure);
                    if (stager->stop) throw std::runtime_error("expert host stager stopped");
                }
                const size_t end = std::min(experts.size(), first + config.expert_staging_slots);
                std::vector<int> persistent(used.size(), -1), transient(used.size(), -1);
                for (size_t j = first; j < end; ++j) {
                    const int e = experts[j];
                    const int stage_slot = (int) (j - first);
                    int victim = -1;
                    if (cache_admission && !defer_final_join) {
                        for (int s = 0; s < capacity; ++s) {
                            const int old = primary.owner[s];
                            if (old < 0) { victim = s; break; }
                            if (!used[old] && (victim < 0 || primary.frequency[old] < primary.frequency[primary.owner[victim]]))
                                victim = s;
                        }
                        if (victim >= 0 && primary.owner[victim] >= 0) {
                            const auto challenger = primary.frequency[e];
                            const auto incumbent = primary.frequency[primary.owner[victim]];
                            const bool keep = config.cache_decay_interval > 0 ?
                                (challenger <= incumbent || challenger - incumbent <= 256) :
                                challenger <= incumbent + 1;
                            if (keep) victim = -1;
                        }
                    }
                    if (victim >= 0) { persistent[e] = victim; ++counters.cache_admissions; }
                    else transient[e] = stage_slot;
                    for (size_t p = 0; p < source.size(); ++p) {
                        auto * src = source[p];
                        auto * target = victim >= 0 ? cache_set[p]->weights : bank.weights[p];
                        const int target_slot = victim >= 0 ? victim : stage_slot;
                        const size_t bytes = src->nb[2];
                        const char * upload_source = (const char *) src->data + e * src->nb[2];
                        if (config.pageable_experts) {
                            auto * host = bank.host_weights[p];
                            auto * host_row = (char *) host->data + stage_slot * host->nb[2];
                            if (!stager) std::memcpy(host_row, upload_source, bytes);
                            upload_source = host_row;
                        }
                        // Pinned model arenas live until backend synchronize
                        // or destruction, so DMA them directly. Only pageable
                        // arenas need the bounded, event-protected host bank.
                        if (!issuer_enabled) {
                            ggml_backend_tensor_set_async(copy_gpu, target, upload_source, target_slot * target->nb[2], bytes);
                            ++counters.uploads; counters.upload_bytes += bytes;
                            counters.staging_upload_bytes += bytes;
                        }
                        if (victim >= 0) {
                            auto & c = *cache_set[p];
                            if (c.owner[victim] >= 0) c.slot[c.owner[victim]] = -1;
                            c.owner[victim] = e; c.slot[e] = victim;
                        }
                    }
                    if (!issuer_enabled) ++counters.staged_experts;
                }
                if (!issuer_enabled) {
                    ++counters.staging_batches;
                    ggml_backend_event_record(bank.copied.p, copy_gpu);
                }
                if (stager && !issuer_enabled) {
                    { std::lock_guard<std::mutex> lock(stager->mu);
                      stager->copied[batch % ring_banks] = (int64_t) batch; }
                    stager->cv.notify_all();
                }
                ggml_backend_event_wait(gpu, bank.copied.p);
                std::vector<Job> admitted_jobs, staged_jobs;
                for (auto job : misses) {
                    if (persistent[job.expert] >= 0) {
                        job.slot = persistent[job.expert]; admitted_jobs.push_back(job);
                    } else if (transient[job.expert] >= 0) {
                        job.slot = transient[job.expert]; staged_jobs.push_back(job);
                    }
                }
                counters.staged_rows += staged_jobs.size();
                execute(cached_weights, std::move(admitted_jobs), &bank.cache_task_tag);
                if (packed_gate_up)
                    execute({bank.packed_gate_up, bank.weights[2]}, std::move(staged_jobs), &bank.packed_task_tag);
                else execute(bank.weights, std::move(staged_jobs), &bank.staging_task_tag);
                ggml_backend_event_record(bank.consumed.p, gpu);
                bank.consumed_serial = submitted_serial;
                bank.live = true;
                if (issuer_enabled) {
                    { std::lock_guard<std::mutex> lock(stager->mu);
                      stager->consumed[batch % ring_banks] = (int64_t) batch; }
                    stager->cv.notify_all();
                }
            }
        }
        // The opt-in backend publishes its tail event to the scheduler. The
        // following trunk can enqueue a device wait without a CPU round trip.
        // A hybrid caller owns the one final join and the one LFU sweep after
        // its CPU result has also been queued. Bank reuse remains event guarded.
        if (defer_final_join) {
            counters.gpu_only_host_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - host_started).count();
            return;
        }
        if (config.async_expert_events) ++counters.async_expert_calls;
        else ggml_backend_synchronize(gpu);
        if (shared_expert_store_enabled()) {
            // Fixed profile: routing observations do not change LFU or slots.
        } else if (config.cache_decay_interval > 0) {
            decay_window(primary);
        } else if (!primary.frequency.empty() && *std::max_element(primary.frequency.begin(), primary.frequency.end()) > 4096)
            for (auto & f : primary.frequency) f = (f + 1) / 2;
        counters.gpu_only_host_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - host_started).count();
    }
    void compute_hybrid_fused(ggml_tensor * dst, const ggml_tensor * x,
                               ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down,
                               Cache & cg, Cache & cu, Cache & cd,
                               const std::vector<Job> & work, bool prefill_share = false) {
        ++counters.hybrid_calls;
        const int capacity = std::min({cg.capacity, cu.capacity, cd.capacity});
        auto resident = [&](int expert) {
            const int slot = cg.slot[expert];
            return slot >= 0 && slot < capacity && cu.slot[expert] == slot && cd.slot[expert] == slot;
        };
        std::vector<int> missed;
        std::vector<bool> seen(cg.slot.size(), false), gpu_miss(cg.slot.size(), false);
        std::vector<size_t> fanout;
        if (prefill_share) fanout.assign(cg.slot.size(), 0);
        for (const auto & job : work) if (!resident(job.expert)) {
            if (prefill_share) ++fanout[job.expert];
            if (!seen[job.expert]) { seen[job.expert] = true; missed.push_back(job.expert); }
        }
        size_t offloaded = 0;
        if (prefill_share) {
            // Like Strata's prefill CPU share, choose complete low-fanout
            // nonresident experts. A repeated expert never crosses executors
            // within this window. GPU misses use bounded transient banks,
            // preserving existing cache slots even when admission is enabled.
            for (int expert : missed) if (fanout[expert] > 2) {
                gpu_miss[expert] = true; ++offloaded;
            }
        } else {
            offloaded = config.pcie_fraction == 1.0 ? missed.size() :
                (size_t) std::floor(missed.size() * config.pcie_fraction);
            for (size_t j = missed.size() - offloaded; j < missed.size(); ++j) gpu_miss[missed[j]] = true;
        }
        std::vector<Job> gpu_work, cpu_work;
        std::vector<bool> gpu_used(cg.slot.size(), false);
        for (auto job : work) {
            if (resident(job.expert) || gpu_miss[job.expert]) {
                gpu_work.push_back(job); gpu_used[job.expert] = true;
                if (gpu_miss[job.expert]) ++counters.hybrid_gpu_miss_rows;
            } else cpu_work.push_back(job);
        }
        counters.hybrid_gpu_miss_experts += offloaded;
        if (prefill_share) {
            ++counters.prefill_share_calls;
            counters.prefill_share_cpu_jobs += cpu_work.size();
            counters.prefill_share_cpu_experts += missed.size() - offloaded;
        }
        counters.cpu_rows += 3 * cpu_work.size(); counters.fused_cpu_rows += cpu_work.size();
        counters.misses += 3 * cpu_work.size();
        ggml_tensor host_x = *x;
        if (!cpu_work.empty()) host_x = queue_hybrid_input(x);
        if (!gpu_work.empty())
            compute_gpu_only(dst, x, gpu_work, gpu_used, {gate, up, down}, {&cg, &cu, &cd}, true);
        if (!cpu_work.empty()) {
            // Only the small input copy must finish before CPU execution. GPU
            // experts keep running on their separate compute stream.
            if (!on_host(x)) ggml_backend_event_synchronize(hybrid_input_ready.p);
            const bool native = config.cpu_expert_pool && (x->ne[2] <= 8 || prefill_share) && x->nb[0] == sizeof(float) &&
                ggml_is_quantized(gate->type) && ggml_is_quantized(up->type) && ggml_is_quantized(down->type);
            auto & task = native ? run_native_cpu_pool(gate, up, down, &host_x, std::move(cpu_work)) :
                run_fused(cpu, gate, up, down, &host_x, std::move(cpu_work), false, dst);
            task.finish(cpu, dst, gpu, false);
        }
        // Both hit/staged GPU scatter and CPU-result H2D/scatter are ordered
        // on this stream. This join also protects all task-owned host sources.
        synchronize_gpu();
        if (config.async_expert_events) ++counters.async_expert_calls;
        if (config.cache_decay_interval > 0) decay_window(cg);
        else if (!cg.frequency.empty() && *std::max_element(cg.frequency.begin(), cg.frequency.end()) > 4096)
            for (auto & f : cg.frequency) f = (f + 1) / 2;
    }
    void compute_fused(ggml_tensor * dst, RouteSnapshots * routes) {
        auto * gate = dst->src[0]; auto * up = dst->src[1]; auto * down = dst->src[2];
        const auto * x = dst->src[3]; const auto * ids = dst->src[4];
#if defined(STRATA_A3B_STRATA_NATIVE)
        if (config.strata_native_experts && gpu && ggml_backend_is_cuda(gpu) &&
            x->ne[2] <= config.native_exact_max_tokens &&
            gate->buffer && !ggml_backend_buffer_is_host(gate->buffer) &&
            strata_native_experts_compatible(gate, up, down)) {
            const int entries = (int) (ids->ne[0] * ids->ne[1]);
            if (!strata_native) strata_native = strata_native_experts_create(
                (int) (ids->ne[0] * config.native_exact_max_tokens),
                (int) gate->ne[0], (int) gate->ne[1]);
            strata_native_experts_compute(strata_native, ggml_backend_cuda_stream(gpu),
                gate, up, down, x, ids, dst);
            ++counters.fused_calls;
            ++counters.strata_native_calls;
            counters.strata_native_rows += entries;
            counters.fused_gpu_rows += entries;
            counters.gpu_rows += 3 * entries;
            return;
        }
#endif
        RouteSnapshot local_routes;
        if (!on_host(ids)) {
            auto & snapshot = routes ? (*routes)[ids] : local_routes;
            if (snapshot.bytes.empty()) {
                read_routes(ids, snapshot);
            }
            ids = &snapshot.tensor;
        } else if (config.async_expert_events) synchronize_gpu();
        auto & cg = cache(gate); auto & cu = cache(up); auto & cd = cache(down);
        const int capacity = std::min({cg.capacity, cu.capacity, cd.capacity});
        auto resident = [&](int e) {
            const int s = cg.slot[e];
            return s >= 0 && s < capacity && cu.slot[e] == s && cd.slot[e] == s ? s : -1;
        };
        auto admit = [&](int e, int s) {
            upload(cg, gate, e, s); upload(cu, up, e, s); upload(cd, down, e, s);
        };
        auto victim_for = [&](const std::vector<bool> & protect) {
            int victim = -1;
            for (int s = 0; s < capacity; ++s) {
                const int old = cg.owner[s];
                if (old < 0) return s;
                if (!protect[old] && (victim < 0 || cg.frequency[old] < cg.frequency[cg.owner[victim]])) victim = s;
            }
            return victim;
        };
        std::vector<Job> work;
        std::vector<bool> used((size_t) gate->ne[2], false);
        for (int64_t t = 0; t < ids->ne[1]; ++t) for (int64_t k = 0; k < ids->ne[0]; ++k) {
            int32_t e;
            std::memcpy(&e, (const char *) ids->data + t * ids->nb[1] + k * ids->nb[0], sizeof(e));
            if (e < 0 || e >= gate->ne[2]) throw std::runtime_error("invalid fused expert id");
            work.push_back({e, -1, t, k}); used[e] = true; count_frequency(cg, e); ++cg.route_count[e];
        }
        ++counters.fused_calls;
        const bool prefill_share = config.prefill_cpu_share && config.hybrid_experts &&
            ids->ne[1] >= 9 && ids->ne[1] <= 256 && x->ne[2] == ids->ne[1] &&
            x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float) &&
            ggml_is_quantized(gate->type) && ggml_is_quantized(up->type) && ggml_is_quantized(down->type);
        if (prefill_share) {
            compute_hybrid_fused(dst, x, gate, up, down, cg, cu, cd, work, true);
            return;
        }
        if (config.gpu_only_experts || (config.hybrid_doorbell && ids->ne[1] > 8)) {
            compute_gpu_only(dst, x, work, used, {gate, up, down}, {&cg, &cu, &cd});
            return;
        }
        if (cache_admission && config.stream_prefill && ids->ne[1] > config.native_exact_max_tokens && capacity) {
            std::vector<int> experts;
            for (int e = 0; e < gate->ne[2]; ++e) if (used[e]) experts.push_back(e);
            std::stable_sort(experts.begin(), experts.end(), [&](int a, int b) { return cg.frequency[a] < cg.frequency[b]; });
            for (size_t first = 0; first < experts.size();) {
                const size_t remainder = experts.size() % capacity;
                const size_t count = first == 0 && remainder ? remainder : (size_t) capacity;
                const size_t end = std::min(experts.size(), first + count);
                std::vector<bool> protect(used.size(), false);
                for (size_t j = first; j < end; ++j) protect[experts[j]] = true;
                for (size_t j = first; j < end; ++j) {
                    const int e = experts[j];
                    if (resident(e) >= 0) continue;
                    const int s = cg.slot[e] >= 0 && cg.slot[e] < capacity ? cg.slot[e] : victim_for(protect);
                    if (s < 0) throw std::runtime_error("no slot for streamed fused expert");
                    admit(e, s);
                }
                std::vector<Job> group;
                for (auto job : work) if (protect[job.expert]) {
                    job.slot = resident(job.expert); group.push_back(job);
                }
                counters.gpu_rows += 3 * group.size(); counters.fused_gpu_rows += group.size();
                auto & task = run_fused(gpu, cg.weights, cu.weights, cd.weights, x, std::move(group), true, dst);
                task.finish(gpu, dst);
                first = end;
            }
            if (config.hybrid_experts && config.cache_decay_interval > 0) decay_window(cg);
            return;
        }
        if (cache_admission && capacity) for (int e = 0; e < gate->ne[2]; ++e) if (used[e] && resident(e) < 0) {
            if (cg.slot[e] >= 0 && cg.slot[e] < capacity) { admit(e, cg.slot[e]); continue; }
            const int s = victim_for(used);
            if (s >= 0 && (cg.owner[s] < 0 || frequency_wins(cg.frequency[e], cg.frequency[cg.owner[s]]))) admit(e, s);
        }
        if (config.hybrid_experts) {
            compute_hybrid_fused(dst, x, gate, up, down, cg, cu, cd, work);
            return;
        }
        std::vector<Job> hits, misses;
        for (auto job : work) {
            job.slot = resident(job.expert);
            if (job.slot >= 0) hits.push_back(job); else misses.push_back(job);
        }
        counters.hits += 3 * hits.size(); counters.misses += 3 * misses.size();
        counters.gpu_rows += 3 * hits.size(); counters.cpu_rows += 3 * misses.size();
        counters.fused_gpu_rows += hits.size(); counters.fused_cpu_rows += misses.size();
        Task * gpu_task = nullptr;
        if (!hits.empty()) gpu_task = &run_fused(gpu, cg.weights, cu.weights, cd.weights, x, std::move(hits), true, dst);
        ggml_tensor host_x = *x;
        std::vector<char> x_data;
        if (!misses.empty() && !on_host(x)) {
            x_data.resize(ggml_nbytes(x)); ggml_backend_tensor_get(x, x_data.data(), 0, x_data.size());
            host_x.data = x_data.data();
        }
        if (!misses.empty()) {
            auto & task = run_fused(cpu, gate, up, down, &host_x, std::move(misses), false, dst);
            task.finish(cpu, dst, gpu);
        }
        if (gpu_task) gpu_task->finish(gpu, dst);
        if (!cg.frequency.empty() && *std::max_element(cg.frequency.begin(), cg.frequency.end()) > 4096)
            for (auto & f : cg.frequency) f = (f + 1) / 2;
    }
    void compute(ggml_tensor * dst, RouteSnapshots * routes) {
        auto * w = dst->src[0]; const auto * x = dst->src[1]; const auto * ids = dst->src[2];
        RouteSnapshot local_routes;
        if (!on_host(ids)) {
            auto & snapshot = routes ? (*routes)[ids] : local_routes;
            if (snapshot.bytes.empty()) {
                read_routes(ids, snapshot);
            }
            ids = &snapshot.tensor;
        } else if (config.async_expert_events) synchronize_gpu();
        auto & c = cache(w);
        std::vector<Job> work;
        std::vector<bool> used((size_t) w->ne[2], false);
        for (int64_t t = 0; t < ids->ne[1]; ++t) for (int64_t k = 0; k < ids->ne[0]; ++k) {
            int32_t e;
            std::memcpy(&e, (const char *) ids->data + t * ids->nb[1] + k * ids->nb[0], sizeof(e));
            if (e < 0 || e >= w->ne[2]) throw std::runtime_error("invalid routed expert id");
            work.push_back({e, -1, t, k}); used[e] = true; count_frequency(c, e); ++c.route_count[e];
        }
        if (config.gpu_only_experts || (config.hybrid_experts && config.pcie_fraction == 1.0)) {
            compute_gpu_only(dst, x, work, used, {w}, {&c});
            return;
        }
        // Long prompts stream groups through bounded slots instead of computing every cold expert on CPU.
        // A speculative verification window is a few tokens. Keep its CPU/GPU
        // split and resident cache instead of streaming every routed expert.
        if (cache_admission && config.stream_prefill && ids->ne[1] > config.native_exact_max_tokens && c.capacity) {
            std::vector<int> experts;
            for (int e = 0; e < w->ne[2]; ++e) if (used[e]) experts.push_back(e);
            std::stable_sort(experts.begin(), experts.end(), [&](int a, int b) { return c.frequency[a] < c.frequency[b]; });
            for (size_t first = 0; first < experts.size();) {
                // End with a full group of the hottest experts, ready for decode.
                const size_t remainder = experts.size() % c.capacity;
                const size_t count = first == 0 && remainder ? remainder : (size_t) c.capacity;
                const size_t end = std::min(experts.size(), first + count);
                std::vector<bool> protected_ids(c.slot.size(), false);
                for (size_t j = first; j < end; ++j) protected_ids[experts[j]] = true;
                for (size_t j = first; j < end; ++j) {
                    const int expert = experts[j];
                    if (c.slot[expert] >= 0) continue;
                    int victim = -1;
                    for (int s = 0; s < c.capacity; ++s) {
                        const int old = c.owner[s];
                        if (old < 0) { victim = s; break; }
                        if (!protected_ids[old] && (victim < 0 || c.frequency[old] < c.frequency[c.owner[victim]])) victim = s;
                    }
                    if (victim < 0) throw std::runtime_error("no slot for streamed expert");
                    upload(c, w, expert, victim);
                }
                std::vector<Job> group;
                for (auto job : work) if (protected_ids[job.expert]) {
                    job.slot = c.slot[job.expert]; group.push_back(job);
                }
                counters.gpu_rows += group.size();
                auto & task = run(gpu, c.weights, x, std::move(group), true, dst);
                task.finish(gpu, dst);
                first = end;
            }
            if (config.hybrid_experts && config.cache_decay_interval > 0) decay_window(c);
            return;
        }
        // Protect all live hits. Promote a miss only over an unused, less frequent expert.
        if (cache_admission) for (int e = 0; e < w->ne[2]; ++e) if (used[e] && c.slot[e] < 0 && c.capacity) {
            int victim = -1;
            for (int s = 0; s < c.capacity; ++s) {
                const int old = c.owner[s];
                if (old < 0) { victim = s; break; }
                if (!used[old] && (victim < 0 || c.frequency[old] < c.frequency[c.owner[victim]])) victim = s;
            }
            if (victim >= 0 && (c.owner[victim] < 0 || frequency_wins(c.frequency[e], c.frequency[c.owner[victim]])))
                upload(c, w, e, victim);
        }
        std::vector<Job> hits, misses;
        for (auto job : work) {
            job.slot = c.slot[job.expert];
            if (job.slot >= 0) hits.push_back(job); else misses.push_back(job);
        }
        counters.hits += hits.size(); counters.misses += misses.size();
        counters.gpu_rows += hits.size(); counters.cpu_rows += misses.size();
        Task * gpu_task = nullptr;
        if (!hits.empty()) gpu_task = &run(gpu, c.weights, x, std::move(hits), true, dst);
        ggml_tensor host_x = *x;
        std::vector<char> x_data;
        if (!misses.empty() && !on_host(x)) {
            x_data.resize(ggml_nbytes(x));
            ggml_backend_tensor_get(x, x_data.data(), 0, x_data.size());
            host_x.data = x_data.data();
        }
        if (!misses.empty()) {
            auto & cpu_task = run(cpu, w, &host_x, std::move(misses), false, dst);
            cpu_task.finish(cpu, dst);
        }
        if (gpu_task) gpu_task->finish(gpu, dst);
        if (config.hybrid_experts && config.cache_decay_interval > 0) decay_window(c);
        else if (!c.frequency.empty() && *std::max_element(c.frequency.begin(), c.frequency.end()) > 4096)
            for (auto & f : c.frequency) f = (f + 1) / 2;
    }
};

int compute_doorbell_cpu(void * user, const float * input, int input_lanes, int tokens,
                        const int32_t * ids, const int32_t * slots, float * output,
                        char * error, size_t error_size) {
    try {
        auto & provider = *static_cast<StaticProvider *>(user);
        auto & store = *provider.store;
        if (!config.hybrid_doorbell || !input || !ids || !slots || !output ||
            (input_lanes != 1 && input_lanes != 8) || tokens < 1 || tokens > 8 || !provider.cache[0])
            throw std::runtime_error("invalid hybrid doorbell CPU request");
        store.doorbell_jobs.clear();
        for (int token = 0; token < tokens; ++token) for (int lane = 0; lane < 8; ++lane) {
            const int expert = ids[token * 8 + lane];
            // CUDA rejects invalid IDs and leaves their result NaN. The CPU
            // service computes only the valid misses; hit rows are untouched.
            if (expert < 0 || expert >= expert_profile_experts) continue;
            if (slots[expert] < -1 || slots[expert] >= provider.cache[0]->capacity)
                throw std::runtime_error("hybrid doorbell has an invalid cache slot");
            if (slots[expert] < 0) store.doorbell_jobs.push_back({expert, token, lane});
        }
        if (store.doorbell_jobs.empty()) return 1;
        if (!store.doorbell_pool) {
            CpuExpertPool::Config params;
            params.threads = config.threads;
            store.doorbell_pool = std::make_unique<CpuExpertPool>(params);
            std::fprintf(stderr, "StrataMoE doorbell CPU pool: %d participants, %d physical cores; one shared GPU service caller\n",
                store.doorbell_pool->threads(), store.doorbell_pool->physical_cores());
        }
        const auto & gate = provider.source[0];
        const auto & up = provider.source[1];
        const auto & down = provider.source[2];
        CpuExpertWeights weights;
        weights.gate = {gate.data, gate.nb[1], gate.nb[2]};
        weights.up = {up.data, up.nb[1], up.nb[2]};
        weights.down = {down.data, down.nb[1], down.nb[2]};
        weights.hidden = (int) gate.ne[0]; weights.ffn = (int) gate.ne[1]; weights.experts = (int) gate.ne[2];
        CpuExpertInput cpu_input{input, input_lanes, tokens, size_t(2048) * sizeof(float),
            size_t(input_lanes) * 2048 * sizeof(float)};
        store.doorbell_pool->execute(weights, cpu_input, store.doorbell_jobs, store.doorbell_outputs);
        for (size_t j = 0; j < store.doorbell_jobs.size(); ++j) {
            const auto & job = store.doorbell_jobs[j];
            std::copy_n(store.doorbell_outputs.data() + j * 2048, 2048,
                output + (job.token * 8 + job.lane) * 2048);
        }
        return 1;
    } catch (const std::exception & e) {
        if (error && error_size) std::snprintf(error, error_size, "%s", e.what());
        return 0;
    } catch (...) {
        if (error && error_size) std::snprintf(error, error_size, "unknown hybrid doorbell CPU failure");
        return 0;
    }
}

// This registry proc is also consumed by the narrow scheduler patch. Returning
// nullptr keeps all non-opted-in backends on the original scheduler path.
ggml_backend_t async_gpu_backend(ggml_backend_t b) {
    if (!(config.async_expert_events || config.strata_native_experts) || !b || b->device != &device) return nullptr;
    return ((Backend *) b->context)->gpu;
}
void synchronize_backend(ggml_backend_t b) { ((Backend *) b->context)->drain(); }
void set_tensor_async(ggml_backend_t b, ggml_tensor * t, const void * data, size_t offset, size_t size) {
    auto * state = (Backend *) b->context;
    if (on_host(t)) { state->drain(); ggml_backend_tensor_set(t, data, offset, size); }
    else ggml_backend_tensor_set_async(state->gpu, t, data, offset, size);
}
void get_tensor_async(ggml_backend_t b, const ggml_tensor * t, void * data, size_t offset, size_t size) {
    auto * state = (Backend *) b->context;
    if (on_host(t)) { state->synchronize_gpu(); ggml_backend_tensor_get(t, data, offset, size); }
    else ggml_backend_tensor_get_async(state->gpu, t, data, offset, size);
}
bool copy_tensor_async(ggml_backend_t src, ggml_backend_t dst, const ggml_tensor * from, ggml_tensor * to) {
    auto real_src = async_gpu_backend(src); if (!real_src) real_src = src;
    auto real_dst = async_gpu_backend(dst); if (!real_dst) real_dst = dst;
    if (real_src->device != real_dst->device || !real_dst->iface.cpy_tensor_async) return false;
    return real_dst->iface.cpy_tensor_async(real_src, real_dst, from, to);
}
void record_event(ggml_backend_t b, ggml_backend_event_t event) {
    ggml_backend_event_record(event, ((Backend *) b->context)->gpu);
}
void wait_event(ggml_backend_t b, ggml_backend_event_t event) {
    ggml_backend_event_wait(((Backend *) b->context)->gpu, event);
    ++counters.async_event_waits;
}
ggml_backend_event_t new_event(ggml_backend_dev_t) { return ggml_backend_event_new(config.gpu); }
void free_backend(ggml_backend_t b) { delete (Backend *) b->context; delete b; }
ggml_status compute_graph(ggml_backend_t b, ggml_cgraph * graph) {
    try {
        // This backend writes only F32 expert outputs. I32 routes are immutable
        // for this call; never retain the snapshot across graph invocations,
        // since the scheduler reuses the same I32 tensor for the next token.
        RouteSnapshots routes;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * t = ggml_graph_node(graph, i);
            if (t->op == GGML_OP_NONE || is_view(t->op)) continue;
            auto * state = (Backend *) b->context;
            if (is_fused_op(t)) state->compute_fused(t, config.cache_graph_routes ? &routes : nullptr);
            else if (t->op == GGML_OP_MUL_MAT_ID) state->compute(t, config.cache_graph_routes ? &routes : nullptr);
            else throw std::runtime_error("unsupported StrataMoE op");
        }
        return GGML_STATUS_SUCCESS;
    } catch (const std::exception & e) {
        if (config.gpu_only_experts || config.hybrid_experts || config.strata_native_experts) {
            auto * state = (Backend *) b->context;
            state->drain();
        }
        std::fprintf(stderr, "StrataMoE: %s\n", e.what());
        return GGML_STATUS_FAILED;
    } catch (...) {
        // Host vectors, pinned upload banks and task buffers must not be freed
        // or reused while either stream still has work from a failed submit.
        if (config.gpu_only_experts || config.hybrid_experts || config.strata_native_experts) ((Backend *) b->context)->drain();
        std::fprintf(stderr, "StrataMoE: unknown expert backend failure\n");
        return GGML_STATUS_FAILED;
    }
}
ggml_backend_t init(ggml_backend_dev_t dev, const char *) {
    static ggml_guid guid = {0x53,0x74,0x72,0x61,0x74,0x61,0x4d,0x6f,0x45,0x41,0x33,0x42,0,0,0,1};
    try {
        auto state = std::make_unique<Backend>();
        auto * b = new ggml_backend{};
        b->guid = &guid; b->device = dev; b->context = state.release();
        b->iface.get_name = backend_name; b->iface.free = free_backend; b->iface.graph_compute = compute_graph;
        if (config.async_expert_events || config.strata_native_experts) {
            b->iface.synchronize = synchronize_backend;
            b->iface.set_tensor_async = set_tensor_async; b->iface.get_tensor_async = get_tensor_async;
            b->iface.cpy_tensor_async = copy_tensor_async;
            b->iface.event_record = record_event; b->iface.event_wait = wait_event;
        }
        return b;
    } catch (...) { return nullptr; }
}
const char * reg_name(ggml_backend_reg_t) { return "StrataMoE"; }
size_t device_count(ggml_backend_reg_t) { return 1; }
ggml_backend_dev_t get_device(ggml_backend_reg_t, size_t i) { GGML_ASSERT(i == 0); return &device; }
void * reg_proc(ggml_backend_reg_t, const char * proc) {
    if (std::strcmp(proc, "strata_a3b_async_gpu_backend_v1") == 0)
        return reinterpret_cast<void *>(async_gpu_backend);
    if (std::strcmp(proc, "strata_a3b_build_attention_v1") == 0)
        return reinterpret_cast<void *>(build_native_exact_attention);
    if (std::strcmp(proc, "strata_a3b_build_mm_v1") == 0)
        return reinterpret_cast<void *>(build_native_exact_mm);
    if (std::strcmp(proc, "strata_a3b_build_mmid_v1") == 0)
        return reinterpret_cast<void *>(build_native_exact_mmid);
    if (std::strcmp(proc, "strata_a3b_build_moe_v1") == 0)
        return reinterpret_cast<void *>(build_fused_moe);
    return nullptr;
}
}
ggml_backend_reg_t register_backend(const Config & cfg) {
    static bool registered = false;
    if (registered) throw std::runtime_error("StrataMoE already registered");
#if !defined(STRATA_A3B_STRATA_NATIVE)
    if (cfg.strata_native_experts) throw std::runtime_error("Strata native IQ experts require a CUDA build");
#endif
    if (!cfg.expert_bytes || cfg.threads < 1 || cfg.native_exact_max_tokens < 1 ||
        cfg.expert_staging_slots < 1 || cfg.expert_staging_slots > 64 ||
        cfg.expert_ring_banks < 2 || cfg.expert_ring_banks > 8 ||
        (cfg.prefill_copy_ahead && !(cfg.gpu_only_experts || cfg.hybrid_experts)) ||
        (cfg.prefill_independent_issuer && (!cfg.prefill_copy_ahead || !cfg.pageable_experts)) ||
        (cfg.prefill_packed_gate_up && !cfg.prefill_independent_issuer) ||
        (cfg.gpu_only_experts && !cfg.gpu) ||
        (cfg.hybrid_experts && (!cfg.gpu || cfg.gpu_only_experts)) ||
        (cfg.cpu_expert_pool && !cfg.hybrid_experts) ||
        (cfg.prefill_cpu_share && (!cfg.hybrid_experts || !cfg.cpu_expert_pool)) ||
        !std::isfinite(cfg.pcie_fraction) || cfg.pcie_fraction < 0.0 || cfg.pcie_fraction > 1.0 ||
        (cfg.pcie_fraction > 0.0 && !cfg.hybrid_experts) ||
        (cfg.async_expert_events && !(cfg.gpu_only_experts || cfg.hybrid_experts)) || cfg.cache_decay_interval < 0 ||
        (cfg.cache_decay_interval > 0 && !(cfg.gpu_only_experts || cfg.hybrid_experts)) ||
        (cfg.gpu_static_experts && (!cfg.gpu_only_experts || cfg.cache_decay_interval != 0)) ||
        (cfg.gpu_dynamic_experts && !cfg.gpu_static_experts) ||
        (cfg.hybrid_doorbell && (!cfg.hybrid_experts || !cfg.cpu_expert_pool || cfg.pcie_fraction != 0.0 ||
            cfg.cache_decay_interval != 0 || cfg.gpu_static_experts || cfg.gpu_dynamic_experts)))
        throw std::runtime_error("invalid expert budget, thread count or short-window limit");
    if (cfg.gpu_static_experts || cfg.hybrid_doorbell) static_api(cfg.gpu);
    if (cfg.async_expert_events) {
        ggml_backend_dev_props gpu_props{};
        ggml_backend_dev_get_props(cfg.gpu, &gpu_props);
        if (!gpu_props.caps.async || !gpu_props.caps.events)
            throw std::runtime_error("asynchronous expert events require an asynchronous GPU with events");
    }
    config = cfg; counters = {}; cache_admission = !shared_expert_store_enabled(); cache_plan.reset();
    if (config.gpu_static_experts || config.hybrid_experts) config.fused_moe = true;
    host_type = *ggml_backend_cpu_buffer_type();
    host_type.iface.get_name = buffer_name; host_type.iface.alloc_buffer = host_alloc;
    host_type.device = &device;
    static ggml_backend_reg reg = {GGML_BACKEND_API_VERSION, {reg_name, device_count, get_device, reg_proc}, nullptr};
    device = {};
    device.reg = &reg; device.iface.get_name = name; device.iface.get_description = description;
    device.iface.get_memory = memory; device.iface.get_type = type; device.iface.get_props = props;
    device.iface.init_backend = init; device.iface.get_buffer_type = buffer_type;
    device.iface.supports_op = supports_op; device.iface.supports_buft = supports_buffer;
    if (config.async_expert_events || config.strata_native_experts || config.strata_prefill_mmq)
        device.iface.event_new = new_event;
    ggml_backend_register(&reg); registered = true;
    return &reg;
}
ggml_backend_buffer_type_t expert_buffer_type() { return &host_type; }
Stats stats() {
    auto result = counters;
#if defined(STRATA_A3B_STRATA_NATIVE)
    if (config.strata_native_experts) result.strata_native_calls += ggml_backend_cuda_strata_iq_calls();
    if (config.strata_prefill_mmq) {
        result.strata_prefill_mmq_calls += ggml_backend_cuda_strata_prefill_mmq_calls();
        result.strata_prefill_mmq_rows += ggml_backend_cuda_strata_prefill_mmq_rows();
    }
    result.gdn_fused_graph_nodes = ggml_backend_cuda_a3b_gdn_fused_graph_nodes();
#endif
    return result;
}
uint64_t gpu_static_scratch_reserve_bytes() {
    return shared_expert_store_enabled() ? static_api(config.gpu)->scratch_bytes() : 0;
}
void collect_gpu_static_stats() {
    if (!shared_expert_store_enabled() || !static_store) return;
    ++counters.gpu_static_collections;
    uint64_t mapped = 0, scratch = 0;
    double prepare_ms = 0;
    std::string failure;
    for (auto & entry : static_store->providers) {
        if (!entry.second) continue;
        auto & provider = *entry.second;
        ggml_cuda_strata_moe_stats_v1 observed{};
        char error[512]{};
        const int ok = static_store->api->get_stats(provider.handle, 1, &observed, error, sizeof(error));
        mapped += observed.mapped_host_bytes;
        scratch = std::max(scratch, observed.shared_gpu_scratch_bytes);
        prepare_ms += observed.prepare_ms;
        // Consume every handle even when one reports an invalid route. This
        // keeps later retries from double-counting histograms collected before
        // the exception and preserves the other layers' observations.
        counters.gpu_static_calls += observed.calls;
        counters.fused_calls += observed.calls;
        if (config.hybrid_doorbell) {
            const bool valid_rows = observed.cpu_rows <= observed.rows &&
                observed.gpu_rows <= observed.rows - observed.cpu_rows;
            if ((!valid_rows || (ok && !observed.device_error &&
                 (observed.cpu_rows + observed.gpu_rows != observed.rows || observed.miss_rows != observed.cpu_rows))) && failure.empty())
                failure = "hybrid doorbell CPU/GPU row accounting mismatch";
            if ((observed.upload_bytes || observed.cache_admissions || observed.cache_decays) && failure.empty())
                failure = "hybrid doorbell changed its fixed expert cache";
            if (observed.cpu_rows && !observed.host_callback_requests && failure.empty())
                failure = "hybrid doorbell CPU rows have no service request";
            counters.hybrid_doorbell_calls += observed.calls;
            counters.hybrid_doorbell_requests += observed.host_callback_requests;
            counters.hybrid_doorbell_callback_ms += observed.host_callback_ms;
            counters.hybrid_mailbox_to_host_bytes += observed.mailbox_to_host_bytes;
            counters.hybrid_mailbox_to_device_bytes += observed.mailbox_to_device_bytes;
            counters.cpu_rows += 3 * observed.cpu_rows; counters.fused_cpu_rows += observed.cpu_rows;
            counters.gpu_rows += 3 * observed.gpu_rows; counters.fused_gpu_rows += observed.gpu_rows;
            counters.gpu_only_rows += observed.gpu_rows;
            counters.hits += 3 * observed.gpu_rows; counters.misses += 3 * observed.cpu_rows;
        } else {
            counters.gpu_only_rows += observed.rows;
            counters.gpu_rows += 3 * observed.rows; counters.fused_gpu_rows += observed.rows;
            if (observed.miss_rows <= observed.rows) {
                counters.hits += 3 * (observed.rows - observed.miss_rows);
                counters.misses += 3 * observed.miss_rows;
            } else if (failure.empty()) failure = "GPU-static invalid row accounting";
            counters.staged_rows += observed.miss_rows; counters.staged_experts += observed.unique_misses;
            counters.uploads += 3 * observed.unique_misses; counters.upload_bytes += observed.upload_bytes;
            counters.staging_upload_bytes += observed.upload_bytes; counters.staging_batches += observed.miss_calls;
        }
        counters.cache_admissions += observed.cache_admissions; counters.decays += observed.cache_decays;
        if (provider.cache[0]) {
            for (int expert = 0; expert < expert_profile_experts; ++expert)
                provider.cache[0]->route_count[expert] += observed.routes[expert];
        } else if (observed.rows && failure.empty()) failure = "GPU-static histogram has no prepared cache";
        // CUDA replaces complete three-role bundles. Before another long
        // prompt uses the host backend, mirror its quiescent device slot map.
        // Unprepared handles return dynamic_cache=0 and must not erase a map.
        if (observed.dynamic_cache) {
            bool valid = config.gpu_dynamic_experts && observed.dynamic_cache == 1 &&
                provider.cache[0] && provider.cache[1] && provider.cache[2];
            const int capacity = observed.cache_slots;
            valid = valid && capacity >= 0 && capacity <= expert_profile_experts;
            if (valid) for (const auto * cache : provider.cache)
                valid = valid && cache->capacity == capacity &&
                    cache->slot.size() == expert_profile_experts && cache->owner.size() == (size_t) capacity &&
                    cache->slot == provider.cache[0]->slot && cache->owner == provider.cache[0]->owner;
            std::array<int, expert_profile_experts> slots, owners;
            slots.fill(-1); owners.fill(-1);
            if (valid) for (int expert = 0; expert < expert_profile_experts; ++expert) {
                const int slot = observed.slot_of_expert[expert];
                if (slot < -1 || slot >= capacity || (slot >= 0 && owners[slot] >= 0)) {
                    valid = false;
                    break;
                }
                slots[expert] = slot;
                if (slot >= 0) owners[slot] = expert;
            }
            if (valid) valid = std::none_of(owners.begin(), owners.begin() + capacity, [](int expert) { return expert < 0; });
            if (valid) for (auto * cache : provider.cache) {
                std::copy(slots.begin(), slots.end(), cache->slot.begin());
                std::copy_n(owners.begin(), capacity, cache->owner.begin());
            }
            else if (failure.empty()) failure = "GPU-dynamic expert slot map is incomplete or inconsistent";
        } else if (config.gpu_dynamic_experts && observed.calls && failure.empty())
            failure = "GPU-dynamic executed without returning its slot map";
        if (!ok || observed.device_error) {
            ++counters.gpu_static_errors;
            if (failure.empty()) failure = error[0] ? error : "GPU-static device error";
        }
    }
    counters.gpu_static_mapped_host_bytes = mapped;
    counters.gpu_static_shared_gpu_scratch_bytes = scratch;
    counters.gpu_static_prepare_ms = prepare_ms;
    // API collection joined the graph and parked the service before this CPU
    // snapshot. Callback threads never mutate the public request counters.
    if (config.hybrid_doorbell && static_store->doorbell_pool) {
        const auto after = static_store->doorbell_pool->stats();
        const auto & before = static_store->collected_pool_stats;
        counters.cpu_pool_calls += after.calls - before.calls; counters.cpu_pool_jobs += after.jobs - before.jobs;
        counters.cpu_pool_unique_inputs += after.unique_inputs - before.unique_inputs;
        counters.cpu_pool_unique_experts += after.unique_experts - before.unique_experts;
        counters.cpu_pool_input_quant_ms += after.input_quant_ms - before.input_quant_ms;
        counters.cpu_pool_gate_up_ms += after.gate_up_ms - before.gate_up_ms;
        counters.cpu_pool_hidden_quant_ms += after.hidden_quant_ms - before.hidden_quant_ms;
        counters.cpu_pool_down_ms += after.down_ms - before.down_ms;
        static_store->collected_pool_stats = after;
    }
    if (!failure.empty()) throw std::runtime_error(failure);
}
void release_gpu_static_experts() {
    if (!shared_expert_store_enabled() || !static_store) return;
    if (!live_backends.empty()) throw std::runtime_error("destroy GPU-static contexts/backends before releasing their cache");
    uint64_t bytes = 0;
    for (const auto & entry : static_store->caches)
        if (entry.second && entry.second->buf.p) bytes += ggml_backend_buffer_get_size(entry.second->buf.p);
    static_store.reset();
    counters.cache_bytes -= std::min(counters.cache_bytes, bytes);
    counters.gpu_static_mapped_host_bytes = counters.gpu_static_shared_gpu_scratch_bytes = 0;
    counters.gpu_static_prepare_ms = 0;
}
void set_cache_budget(size_t bytes) {
    if (shared_expert_store_enabled() && static_store && !static_store->caches.empty() && bytes != config.cache_bytes)
        throw std::runtime_error("GPU-static cache budget is fixed after materialization");
    config.cache_bytes = bytes;
}
void set_cache_plan(const ExpertCachePlan & plan) {
    if (!(config.gpu_only_experts || config.hybrid_experts) || plan.planned_bytes > config.cache_bytes ||
        plan.bundle_bytes != expert_profile_bundle_bytes)
        throw std::runtime_error("cache profile exceeds the GPU expert budget");
    for (const auto * backend : live_backends) if (!backend->caches.empty())
        throw std::runtime_error("set cache profile before expert cache allocation");
    if (static_store && !static_store->caches.empty())
        throw std::runtime_error("set cache profile before GPU-static cache materialization");
    uint64_t selected = 0;
    for (const auto & experts : plan.expert_ids) {
        std::vector<bool> seen(expert_profile_experts, false);
        for (const int expert : experts) {
            if (expert < 0 || expert >= expert_profile_experts || seen[expert])
                throw std::runtime_error("cache profile contains an invalid or duplicate expert");
            seen[expert] = true;
            ++selected;
        }
    }
    if (selected != plan.selected_experts || selected * plan.bundle_bytes != plan.planned_bytes)
        throw std::runtime_error("cache profile byte accounting mismatch");
    cache_plan = std::make_unique<ExpertCachePlan>(plan);
}
void set_cache_admission(bool enabled) {
    if (shared_expert_store_enabled() && enabled) throw std::runtime_error("GPU graph expert cache requires fixed profile admission");
    cache_admission = enabled;
}
void dump_route_frequencies(const char * path) {
    if (!path || !path[0]) throw std::runtime_error("route profile path is empty");
    auto * out = std::fopen(path, "wb");
    if (!out) throw std::runtime_error("cannot open route profile output");
    std::fprintf(out, "backend\ttensor\texpert\troutes\tlfu_frequency\n");
    for (size_t b = 0; b < live_backends.size(); ++b)
        for (const auto & entry : live_backends[b]->caches) {
            const auto & c = *entry.second;
            for (size_t e = 0; e < c.frequency.size(); ++e) if (c.route_count[e])
                std::fprintf(out, "%zu\t%s\t%zu\t%llu\t%llu\n", b, c.tensor_name.c_str(), e,
                    (unsigned long long) c.route_count[e], (unsigned long long) c.frequency[e]);
        }
    if (shared_expert_store_enabled() && static_store)
        for (const auto & entry : static_store->caches) {
            if (!entry.second) continue;
            const auto & c = *entry.second;
            for (size_t e = 0; e < c.route_count.size(); ++e) if (c.route_count[e])
                std::fprintf(out, "%zu\t%s\t%zu\t%llu\t%llu\n", size_t(0), c.tensor_name.c_str(), e,
                    (unsigned long long) c.route_count[e], (unsigned long long) c.frequency[e]);
        }
    const bool failed = std::ferror(out) != 0;
    const int closed = std::fclose(out);
    if (failed || closed != 0) throw std::runtime_error("route profile write failed");
}
ggml_tensor * build_native_exact_attention(ggml_context * ctx, ggml_tensor * q,
    ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask, ggml_tensor * sinks,
    float scale, float max_bias, float logit_softcap, int32_t n_kv_max, ggml_tensor ** probe) {
    static const bool q8_singleton_verify = [] {
        const char * value = std::getenv("STRATA_A3B_Q8_KV_VERIFY_VEC");
        if (!value || std::strcmp(value, "1") != 0) return false;
#if defined(STRATA_A3B_STRATA_NATIVE)
        const char * name = config.gpu ? ggml_backend_dev_name(config.gpu) : nullptr;
        int device = -1, major = 0, minor = 0;
        return name && std::sscanf(name, "CUDA%d", &device) == 1 && device >= 0 &&
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess &&
            major == 7 && minor == 5;
#else
        return false;
#endif
    }();
    // With Q8 K/V on Turing, batched FlashAttention converts the entire KV
    // cache to F16 before MMA. The existing singleton-query path selects the
    // compiled Q8_0-Q8_0 vector kernel and reads the quantized cache directly.
    // Keep this opt-in until long-context latency and decode parity are measured.
    const bool supported_kv = k && v &&
        ((k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16) ||
         (q8_singleton_verify && q && q->ne[0] == 256 && q->ne[1] <= 4 &&
          q->ne[2] == 16 && k->ne[2] == 2 &&
          k->type == GGML_TYPE_Q8_0 && v->type == GGML_TYPE_Q8_0));
    if (!config.native_attention_exact || !(config.native_exact_verify || config.hybrid_experts) || !config.gpu ||
        !q || !k || !v || !mask || q->type != GGML_TYPE_F32 ||
        !supported_kv || mask->type != GGML_TYPE_F16 ||
        q->ne[1] < 2 || q->ne[1] > config.native_exact_max_tokens ||
        q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || mask->ne[3] != 1 ||
        q->ne[0] != k->ne[0] || k->ne[1] != v->ne[1] || k->ne[2] != v->ne[2] ||
        q->ne[2] % k->ne[2] != 0 || mask->ne[0] != k->ne[1] ||
        mask->ne[1] < q->ne[1] || mask->ne[2] != 1 || !ggml_is_contiguous(mask) ||
        max_bias != 0.0f || logit_softcap != 0.0f) return nullptr;
    ggml_tensor * result = nullptr;
    if (probe) *probe = nullptr;
    static const bool reuse_mask_rows = [] {
        const char * value = std::getenv("STRATA_A3B_ATTENTION_MASK_REUSE");
        return value && std::strcmp(value, "1") == 0;
    }();
    for (int64_t t = 0; t < q->ne[1]; ++t) {
        auto * query = ggml_view_4d(ctx, q, q->ne[0], 1, q->ne[2], 1,
                                   q->nb[1], q->nb[2], q->nb[3], t * q->nb[1]);
        ggml_tensor * row_mask = nullptr;
        char mask_name[GGML_MAX_NAME] = {};
        const size_t offset = (size_t) t * mask->nb[1];
        if (reuse_mask_rows) {
            // Scheduler input copies are keyed by tensor identity, so ten
            // layers can share one host-to-device copy per mask row. Search
            // only this live context; no pointers survive a graph reset.
            const int length = std::snprintf(mask_name, sizeof(mask_name), "strata_a3b_mask_%p_r%lld",
                                              (void *) mask, (long long) t);
            if (length < 0 || (size_t) length >= sizeof(mask_name))
                throw std::runtime_error("exact attention mask cache name overflow");
            row_mask = ggml_get_tensor(ctx, mask_name);
            if (row_mask) {
                const auto * owner = mask->view_src ? mask->view_src : mask;
                const size_t absolute_offset = offset + (mask->view_src ? mask->view_offs : 0);
                size_t operation_offset = 0;
                std::memcpy(&operation_offset, row_mask->op_params, sizeof(operation_offset));
                if (row_mask->op != GGML_OP_VIEW || row_mask->src[0] != mask ||
                    row_mask->view_src != owner || row_mask->view_offs != absolute_offset ||
                    operation_offset != offset || row_mask->type != mask->type ||
                    row_mask->ne[0] != mask->ne[0] || row_mask->ne[1] != 1 ||
                    row_mask->ne[2] != 1 || row_mask->ne[3] != 1 ||
                    row_mask->nb[0] != mask->nb[0] || row_mask->nb[1] != mask->nb[1] ||
                    row_mask->nb[2] != mask->nb[1] || row_mask->nb[3] != mask->nb[1])
                    throw std::runtime_error("exact attention mask cache metadata conflict");
            }
        }
        if (!row_mask) {
            row_mask = ggml_view_4d(ctx, mask, mask->ne[0], 1, 1, 1,
                mask->nb[1], mask->nb[1], mask->nb[1], offset);
            if (reuse_mask_rows) ggml_set_name(row_mask, mask_name);
        }
        auto * fa = ggml_flash_attn_ext(ctx, query, k, v, row_mask, scale, max_bias, logit_softcap);
        ggml_flash_attn_ext_add_sinks(fa, sinks);
        ggml_flash_attn_ext_set_n_kv_max(fa, n_kv_max);
        ggml_prec_set_acc(fa, GGML_PREC_F32);
        if (probe && !*probe) *probe = fa;
        result = result ? ggml_concat(ctx, result, fa, 2) : fa;
    }
    ++counters.native_attention_graphs;
    trace_native_hook("attention", "accepted", k, q, 1);
    return result;
}
ggml_tensor * build_native_exact_mm(ggml_context * ctx, ggml_tensor * weights, ggml_tensor * input) {
    if (!config.native_dense_exact || !weights || !input || !weights->buffer ||
        ggml_backend_buffer_is_host(weights->buffer) ||
        !(ggml_is_quantized(weights->type) || weights->type == GGML_TYPE_F32 ||
          weights->type == GGML_TYPE_F16 || weights->type == GGML_TYPE_BF16) ||
        (!ggml_is_quantized(weights->type) && weights->ne[0] % 2 != 0) ||
        input->type != GGML_TYPE_F32 || weights->ne[2] != 1 || weights->ne[3] != 1 ||
        input->ne[2] != 1 || input->ne[3] != 1 || weights->ne[0] != input->ne[0] ||
        input->ne[1] < 2 || input->ne[1] > config.native_exact_max_tokens) {
        trace_native_hook("dense", !config.native_dense_exact ? "skip-disabled" :
            (!weights || !weights->buffer) ? "skip-no-buffer" :
            ggml_backend_buffer_is_host(weights->buffer) ? "skip-host-buffer" : "skip-type-or-shape",
            weights, input, 1);
        return nullptr;
    }
    ++counters.native_dense_graphs;
    trace_native_hook("dense", "accepted", weights, input, 1);
    // On sm_75, the opt-in Q6_K multi-column MMVQ variant keeps the
    // one-column two-warp reduction while sharing each weight load across
    // verification tokens. Other dense tensors retain their exact layout.
    const char * grouped_q6 = std::getenv("STRATA_A3B_Q6_GROUPED_EXACT");
    bool turing_cuda = false;
    if (grouped_q6 && grouped_q6[0] == '1') {
#if defined(STRATA_A3B_STRATA_NATIVE)
    static const bool available = [] {
        const char * name = config.gpu ? ggml_backend_dev_name(config.gpu) : nullptr;
        int device = -1, major = 0, minor = 0;
        return name && std::sscanf(name, "CUDA%d", &device) == 1 && device >= 0 &&
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess &&
            major == 7 && minor == 5;
    }();
    turing_cuda = available;
#endif
    }
    if (weights->type == GGML_TYPE_Q6_K && input->ne[1] <= 4 &&
        turing_cuda && grouped_q6 && grouped_q6[0] == '1')
        return ggml_mul_mat(ctx, weights, input);
    auto * rows = ggml_is_contiguous(input) ? input : ggml_cont(ctx, input);
    // CUDA MMVQ/MMVF treats axis 1 as columns and axis 2 as independent channels.
    // The weight's singleton channel broadcasts without copying the matrix.
    // Thus every channel uses decode's one-column quantization and reduction.
    rows = ggml_reshape_3d(ctx, rows, input->ne[0], 1, input->ne[1]);
    auto * result = ggml_mul_mat(ctx, weights, rows);
    return ggml_reshape_2d(ctx, result, weights->ne[1], input->ne[1]);
}
ggml_tensor * build_native_exact_mmid(ggml_context * ctx, ggml_tensor * weights,
                                     ggml_tensor * input, ggml_tensor * ids) {
    if (!config.native_exact_verify || !weights || !input || !ids || !weights->buffer ||
        ggml_backend_buffer_is_host(weights->buffer) || !ggml_is_quantized(weights->type) ||
        input->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || input->ne[3] != 1 ||
        input->ne[2] < 2 || input->ne[2] > config.native_exact_max_tokens ||
        ids->ne[1] != input->ne[2] || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        (input->ne[1] != 1 && input->ne[1] != ids->ne[0])) {
        trace_native_hook("expert", !config.native_exact_verify ? "skip-disabled" :
            (!weights || !weights->buffer) ? "skip-no-buffer" :
            ggml_backend_buffer_is_host(weights->buffer) ? "skip-host-buffer" : "skip-type-or-shape",
            weights, input, 2);
        return nullptr;
    }
    ++counters.native_expert_graphs;
    trace_native_hook("expert", "accepted", weights, input, 2);
    const int64_t lanes = ids->ne[0], tokens = ids->ne[1];
    auto * rows = input;
    if (input->ne[1] == 1) rows = ggml_repeat_4d(ctx, input, input->ne[0], lanes, tokens, 1);
    if (!ggml_is_contiguous(rows)) rows = ggml_cont(ctx, rows);
    rows = ggml_reshape_3d(ctx, rows, input->ne[0], lanes * tokens, 1);
    auto * routes = ggml_is_contiguous(ids) ? ids : ggml_cont(ctx, ids);
    routes = ggml_reshape_2d(ctx, routes, lanes * tokens, 1);
    auto * result = ggml_mul_mat_id(ctx, weights, rows, routes);
    return ggml_reshape_3d(ctx, result, weights->ne[1], lanes, tokens);
}
ggml_tensor * build_fused_moe(ggml_context * ctx, ggml_tensor * input,
                             ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down, ggml_tensor * ids) {
    if (!config.fused_moe || !input || !gate || !up || !down || !ids ||
        input->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || input->ne[3] != 1 ||
        ids->ne[2] != 1 || ids->ne[3] != 1 || ids->ne[1] != input->ne[2] ||
        (input->ne[1] != 1 && input->ne[1] != ids->ne[0])) {
        trace_native_hook("fused", !config.fused_moe ? "skip-disabled" : "skip-input-or-ids-shape", gate, input, 2);
        return nullptr;
    }
    const bool prefill_candidate = config.strata_prefill_mmq && input->ne[2] >= 9;
    if (gate->ne[0] != input->ne[0] || up->ne[0] != gate->ne[0] || up->ne[1] != gate->ne[1] ||
        down->ne[0] != gate->ne[1] || down->ne[1] != gate->ne[0] ||
        up->ne[2] != gate->ne[2] || down->ne[2] != gate->ne[2]) {
        if (prefill_candidate) ++counters.strata_prefill_mmq_fallback_graphs;
        trace_native_hook("fused", "skip-weight-shape", gate, input, 2);
        return nullptr;
    }
    bool native = config.native_exact_verify && input->ne[2] > 1 && input->ne[2] <= config.native_exact_max_tokens;
    bool strata_native = config.strata_native_experts && input->ne[2] <= config.native_exact_max_tokens;
    bool strata_prefill_resident = prefill_candidate && input->ne[2] <= 2048;
    for (auto * w : {gate, up, down}) {
        if (!w->buffer || !ggml_is_contiguous(w)) {
            if (prefill_candidate) ++counters.strata_prefill_mmq_fallback_graphs;
            trace_native_hook("fused", !w->buffer ? "skip-no-buffer" : "skip-weight-stride", w, input, 2);
            return nullptr;
        }
        native = native && !ggml_backend_buffer_is_host(w->buffer) && ggml_is_quantized(w->type);
        strata_native = strata_native && !ggml_backend_buffer_is_host(w->buffer) && ggml_is_quantized(w->type);
        strata_prefill_resident = strata_prefill_resident &&
            !ggml_backend_buffer_is_host(w->buffer) && ggml_is_quantized(w->type);
    }
#if defined(STRATA_A3B_STRATA_NATIVE)
    if (strata_native && strata_native_experts_compatible(gate, up, down)) {
        auto * rows = ggml_is_contiguous(input) ? input : ggml_cont(ctx, input);
        auto * routes = ggml_is_contiguous(ids) ? ids : ggml_cont(ctx, ids);
        return ggml_backend_cuda_strata_iq_build(ctx, gate, up, down, rows, routes);
    }
    if (strata_prefill_resident) {
        auto * rows = ggml_is_contiguous(input) ? input : ggml_cont(ctx, input);
        auto * routes = ggml_is_contiguous(ids) ? ids : ggml_cont(ctx, ids);
        if (auto * op = ggml_backend_cuda_strata_iq_build(ctx, gate, up, down, rows, routes)) {
            ++counters.strata_prefill_mmq_graphs;
            return op;
        }
    }
#endif
    if (prefill_candidate) ++counters.strata_prefill_mmq_fallback_graphs;
    if (native) {
        ++counters.native_fused_graphs;
        trace_native_hook("fused", "accepted", gate, input, 2);
        const int64_t lanes = ids->ne[0], tokens = ids->ne[1];
        auto * rows = input;
        if (input->ne[1] == 1) rows = ggml_repeat_4d(ctx, input, input->ne[0], lanes, tokens, 1);
        if (!ggml_is_contiguous(rows)) rows = ggml_cont(ctx, rows);
        rows = ggml_reshape_3d(ctx, rows, input->ne[0], lanes * tokens, 1);
        auto * routes = ggml_is_contiguous(ids) ? ids : ggml_cont(ctx, ids);
        routes = ggml_reshape_2d(ctx, routes, lanes * tokens, 1);
        // Keep the three native projections and SwiGLU adjacent: CUDA may fuse
        // gate/up/activation while every row retains decode's column count.
        auto * g = ggml_mul_mat_id(ctx, gate, rows, routes);
        auto * u = ggml_mul_mat_id(ctx, up, rows, routes);
        auto * h = ggml_swiglu_split(ctx, g, u);
        auto * y = ggml_mul_mat_id(ctx, down, h, routes);
        return ggml_reshape_3d(ctx, y, down->ne[1], lanes, tokens);
    }
    for (auto * w : {gate, up, down}) if (w->buffer->buft != &host_type) {
        trace_native_hook("fused", "skip-native-type-or-window", w, input, 2);
        return nullptr;
    }
    if (shared_expert_store_enabled() && input->ne[0] == 2048 &&
        (input->ne[1] == 1 || input->ne[1] == 8) && input->ne[2] >= 1 && input->ne[2] <= 8 &&
        ids->ne[0] == 8 && gate->type == GGML_TYPE_Q8_0 && up->type == GGML_TYPE_Q8_0 &&
        down->type == GGML_TYPE_Q8_0 && gate->ne[0] == 2048 && gate->ne[1] == 512 &&
        gate->ne[2] == 256 && gate->ne[3] == 1 && up->ne[3] == 1 && down->ne[3] == 1) {
        auto & provider = static_provider(gate, up, down);
        // top_k IDs can be a [8,T] view with a 256-ID outer stride. Pack that
        // view on CUDA inside the same graph, without any CPU route snapshot.
        auto * rows = ggml_is_contiguous(input) ? input : ggml_cont(ctx, input);
        auto * routes = ggml_is_contiguous(ids) ? ids : ggml_cont(ctx, ids);
        auto * result = provider.store->api->build(provider.handle, ctx, rows, routes);
        if (!result) throw std::runtime_error("CUDA rejected a compatible GPU-static expert graph");
        ++counters.gpu_static_graphs;
        return result;
    }
    ggml_tensor * args[] = {gate, up, down, input, ids};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, down->ne[1], ids->ne[0], ids->ne[1], 1,
                          args, 5, fused_marker, 1, nullptr);
}
}
