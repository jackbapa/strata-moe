#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>

namespace strata_a3b {
struct ExpertCachePlan;
struct Config {
    ggml_backend_dev_t gpu = nullptr;
    size_t cache_bytes = 0;
    size_t expert_bytes = 1;
    int threads = 4;
    bool stream_prefill = true;
    // Pageable RAM avoids locking the entire expert arena; GPU copies still use
    // bounded task/cache buffers. Keep pinned experts as the existing default.
    bool pageable_experts = false;
    // Keep short-window task graphs per weight so CUDA can capture them once
    // their input/output shapes stabilize. Long prefill keeps bounded sharing.
    bool stable_tasks = false;
    // Preserve native token/lane axes for complete GPU-resident short windows.
    bool preserve_verify_layout = false;
    // Reuse an immutable routing snapshot only within one backend graph call.
    bool cache_graph_routes = false;
    // Keep short verification rows on decode's single-column GPU arithmetic.
    bool exact_verify = false;
    // Native all-GPU MMID graph hook; uses only ordinary GGML operations.
    bool native_exact_verify = false;
    // Shared limit for native and cached short verification windows.
    int native_exact_max_tokens = 4;
    // Whole routed experts: one plan, complete CPU/GPU FFN graphs.
    bool fused_moe = false;
    // Short-window GPU-resident IQ experts use Strata's grouped Q8 MMQ kernel.
    bool strata_native_experts = false;
    // Long GPU-resident IQ3_S prompts use original Strata grouped IQ MMQ.
    // Explicitly opt in; short windows and default GGML prompt path stay put.
    bool strata_prefill_mmq = false;
    // Dense GPU projections use decode's one-column arithmetic in
    // short verification windows, including attention, shared FFN and heads.
    bool native_dense_exact = false;
    // Short single-sequence attention uses singleton FA nodes in one GPU graph.
    bool native_attention_exact = false;
    // RAM holds expert weights only. Every cache miss executes on the GPU.
    bool gpu_only_experts = false;
    // Complete expert slots per transient bank (the GPU compute group).
    int expert_staging_slots = 8;
    // Independent copy-ahead ring depth. Keep compute groups small while more
    // banks are in flight, as in Strata's streamed prefill path.
    int expert_ring_banks = 2;
    // Stage pageable expert bytes on a host worker while prior GPU groups run.
    bool prefill_copy_ahead = false;
    // Experimental independent H2D issuer. Keep the measured host-only
    // copy-ahead schedule as the default until a paired full-model win.
    bool prefill_independent_issuer = false;
    // Long Q8 prefill: upload gate and up as consecutive rows, so one MMQ
    // product quantizes the routed input once for both projections.
    bool prefill_packed_gate_up = false;
    // Forward same-device scheduler dependencies to GPU events. Expert routing
    // still waits once for the CPU plan; no expert math runs on the CPU.
    bool async_expert_events = false;
    // Opt-in GPU-only LFU: Q8.8 route weights and floor(0.7 * frequency)
    // after this many windows per primary cache. Zero keeps legacy LFU.
    int cache_decay_interval = 0;
    // Profile-seeded Q8 experts: short windows route/copy/compute entirely on
    // CUDA; long prefill shares the same cache, fixed unless dynamic is enabled.
    bool gpu_static_experts = false;
    // Experimental CUDA-owned slot replacement in static-op short windows.
    // Request-boundary collection mirrors its map for the long-prefill path.
    bool gpu_dynamic_experts = false;
    // Opt-in mixed CPU/GPU expert scheduling. GPU hit scatter is queued before
    // the CPU miss graph, whose final return joins the GPU once.
    bool hybrid_experts = false;
    // Reuse per-token Q8 CPU activations and a persistent expert-row pool.
    bool cpu_expert_pool = false;
    // Fraction of distinct cache-missed experts computed via GPU staging in
    // hybrid mode. Choose the last floor(unique_misses * fraction) route groups.
    double pcie_fraction = 0.0;
    // Fixed-profile original-style mailbox: short CUDA graphs publish CPU
    // misses to one service thread while cached experts run on the GPU.
    bool hybrid_doorbell = false;
    // Short prefill (9..256 tokens): complete Q8 cache-missed experts with
    // at most two routed jobs use the native CPU pool; other misses are staged
    // on the GPU. Decode/short verification and long prefill keep their paths.
    bool prefill_cpu_share = false;
};
struct Stats {
    uint64_t hits = 0, misses = 0, uploads = 0, upload_bytes = 0;
    uint64_t cpu_rows = 0, gpu_rows = 0, cache_bytes = 0;
    uint64_t route_reads = 0, preserved_layout_calls = 0;
    uint64_t exact_verify_calls = 0;
    uint64_t strata_native_calls = 0, strata_native_rows = 0;
    uint64_t strata_prefill_mmq_calls = 0, strata_prefill_mmq_rows = 0;
    // Builder decisions; only the two counters above denote CUDA execution.
    uint64_t strata_prefill_mmq_graphs = 0, strata_prefill_mmq_fallback_graphs = 0;
    // Number of A3B GDN graph nodes accepted by the optional Strata fused
    // CUDA path. CUDA Graph replays are not counted again.
    uint64_t gdn_fused_graph_nodes = 0;
    uint64_t fused_calls = 0, fused_cpu_rows = 0, fused_gpu_rows = 0;
    // Graph-builder acceptances, including reserve/probe graphs. These are not
    // execution counts; native GPU graphs bypass the custom expert backend.
    uint64_t native_dense_graphs = 0, native_expert_graphs = 0, native_fused_graphs = 0;
    uint64_t native_attention_graphs = 0;
    // Complete expert rows/bundles for fused execution; projection rows for
    // ordinary MMID. staged_rows counts transient GPU consumers only.
    uint64_t gpu_only_rows = 0, staged_rows = 0, staged_experts = 0;
    uint64_t staging_upload_bytes = 0, staging_batches = 0, staging_allocations = 0;
    uint64_t expert_task_allocations = 0, cache_admissions = 0;
    uint64_t pinned_arena_fallbacks = 0, pinned_bank_fallbacks = 0;
    // Cumulative allocated bytes; use deltas for a newly initialized backend.
    uint64_t staging_gpu_bytes = 0, staging_host_bytes = 0;
    uint64_t async_expert_calls = 0, async_route_waits = 0;
    uint64_t async_task_reuse_waits = 0, async_event_waits = 0;
    double route_read_ms = 0, gpu_only_host_ms = 0;
    double staging_reuse_wait_ms = 0, expert_task_submit_ms = 0;
    // Primary-cache decay sweeps, not individual expert/projection counts.
    uint64_t decays = 0;
    uint64_t gpu_static_graphs = 0, gpu_static_calls = 0, gpu_static_collections = 0;
    uint64_t gpu_static_errors = 0;
    // Current extra mapped RAM (sum across layers) and shared GPU scratch (max,
    // not sum across layers). Preparation time sums each handle's lifetime.
    uint64_t gpu_static_mapped_host_bytes = 0, gpu_static_shared_gpu_scratch_bytes = 0;
    double gpu_static_prepare_ms = 0;
    uint64_t hybrid_calls = 0, hybrid_gpu_miss_rows = 0, hybrid_gpu_miss_experts = 0;
    uint64_t hybrid_input_allocations = 0, hybrid_input_bytes = 0, hybrid_input_fallbacks = 0;
    uint64_t cpu_pool_calls = 0, cpu_pool_jobs = 0, cpu_pool_unique_inputs = 0, cpu_pool_unique_experts = 0;
    double cpu_pool_input_quant_ms = 0, cpu_pool_gate_up_ms = 0;
    double cpu_pool_hidden_quant_ms = 0, cpu_pool_down_ms = 0;
    uint64_t hybrid_doorbell_calls = 0, hybrid_doorbell_requests = 0;
    uint64_t hybrid_mailbox_to_host_bytes = 0, hybrid_mailbox_to_device_bytes = 0;
    double hybrid_doorbell_callback_ms = 0;
    // Actual short-prefill sharing calls and complete CPU expert jobs/groups.
    uint64_t prefill_share_calls = 0, prefill_share_cpu_jobs = 0, prefill_share_cpu_experts = 0;
};
// One registration per process. Default contexts own their caches. Static mode
// shares one fixed profile cache until explicit release; weights stay in RAM.
ggml_backend_reg_t register_backend(const Config & config);
ggml_backend_buffer_type_t expert_buffer_type();
Stats stats();
// Static mode only: call once at an idle/request boundary before stats()/dump.
// Consumes GPU histograms, including error flags; throws after collecting all
// handles on a device/runtime error. A consumed invalid-ID flag permits retry.
// Dynamic mode also mirrors GPU slots; collect before every host-planned
// long-prefill entry after short GPU windows may have replaced expert bundles.
void collect_gpu_static_stats();
// Extra shared CUDA scratch reservation before selecting a cache profile.
uint64_t gpu_static_scratch_reserve_bytes();
// All graphs/schedulers/contexts referring to static ops must be destroyed,
// while their original model host buffers are still alive. Other modes no-op.
void release_gpu_static_experts();
void set_cache_budget(size_t bytes);
// Must be set before any expert cache is allocated. The default has no plan.
void set_cache_plan(const ExpertCachePlan & plan);
// Call while inference is idle. Disabling admission freezes existing placement
// for comparisons without changing which routed experts are computed.
void set_cache_admission(bool enabled);
// Export lifetime route counts and decayed LFU frequency while inference is
// idle. Does not move weights or reset the observations.
// Positive cache_decay_interval exports raw Q8.8 lfu_frequency units;
// routes always retains its ordinary integer count.
// In GPU-static mode call collect_gpu_static_stats() first while inference is idle.
void dump_route_frequencies(const char * path);
// Returns nullptr when the optional native GPU hook does not apply.
ggml_tensor * build_native_exact_mmid(ggml_context * ctx, ggml_tensor * weights,
                                     ggml_tensor * input, ggml_tensor * ids);
ggml_tensor * build_native_exact_mm(ggml_context * ctx, ggml_tensor * weights,
                                   ggml_tensor * input);
// Returns [Dv,H,T,1] from independently configured singleton FA nodes.
// probe receives a raw FA node for the caller's backend feature check.
ggml_tensor * build_native_exact_attention(ggml_context * ctx, ggml_tensor * q,
    ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask, ggml_tensor * sinks,
    float scale, float max_bias, float logit_softcap, int32_t n_kv_max, ggml_tensor ** probe);
ggml_tensor * build_fused_moe(ggml_context * ctx, ggml_tensor * input,
                             ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down,
                             ggml_tensor * ids);
}
