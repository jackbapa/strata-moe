# IQ3_S A3B 50K/256 Nsight trace analysis

Source: `explicit-graph.sqlite`, exported from Nsight Systems 2024.4.2 with
CUDA graph-level activity, CUDA API and NVTX tracing. The profiling config is
`profile-config.json` (GPU 0: RTX 2080 Ti, 22,528 MiB; Qwen3.6-35B-A3B
UD-IQ3_S target, 262,144 context, Q8 K/V, Q8_0 MTP K2). The executable
SHA-256 recorded by the associated harness run is
`05802fa88d8b9a7e865b24bed82f15828eb52ad15d4609e76f91c1f4617fc390`.
This is an instrumented diagnostic run, **not** a throughput A/B result.

| NVTX phase | Instances | Host range time | `cudaGraphLaunch` | `cudaLaunchKernel` | `cudaStreamSynchronize` | Synchronize API time | Associated GPU activity busy / span |
|---|---:|---:|---:|---:|---:|---:|---:|
| `a3b/output` | 1 | 3.920 s | 367 | 16,159 | 11,149 | 3.431 s | 3.277 / 3.906 s |
| `a3b/output/mtp-propose` | 114 | 0.743 s | 224 | 396 | 4,964 | 0.625 s | 0.443 / 0.588 s |
| `a3b/output/target-verify` | 114 | 0.260 s | 111 | 10,368 | 2,398 | 0.050 s | 2.716 / 2.906 s |
| `a3b/output/target-hidden-observe` | 114 | 2.707 s | 0 | 0 | 114 | 2.705 s | no GPU work submitted |
| `a3b/output/mtp-commit` | 114 | 0.206 s | 32 | 5,395 | 2,137 | 0.049 s | 0.117 / 0.214 s |

The target verification range submits work asynchronously. Its associated GPU
work occupies about **2.716 s** across the 114 windows, including **2.647 s**
recorded as 111 whole CUDA Graph executions. The next `target-hidden-observe`
range spends **2.705 s** in exactly 114 `cudaStreamSynchronize` calls. Thus its
host duration represents waiting for target GPU verification, not CPU expert
inference. MTP proposal has 224 graph launches and about **0.443 s** associated
GPU work; its 227 `draft-sample` ranges spend **0.447 s** in 454 stream syncs.

For the full output range, the union of CUDA graph/kernel/memory activities is
3.277 s over a 3.906 s span, leaving about **0.629 s** without a traced GPU
activity. This is timeline coverage, **not SM occupancy**. GPU execution may
continue after a host NVTX range ends, so host range times and associated GPU
times are not additive. Nested ranges also double-count activity if summed.

The 256-token profiled output lasts 3.920 s. Reaching 100 token/s requires at
most 2.560 s, a 1.360 s reduction. Even removing the entire 0.743 s proposal
host range would leave 3.177 s (about 80.6 token/s) with the current sequential
schedule. A faster draft chain alone cannot meet the target; target verification
GPU time must shrink or overlap with other useful work. The output trace also
contains 18 graph instantiations and 26 begin/end captures. Of these, 15
instantiations and 21 captures occur in MTP commit/catch-up. Their count should
be reduced where safe, but the entire commit host range is only 0.206 s.

Prefill was included in this explicit capture: 25 target-decode calls enqueue
59,830 CUDA kernel launches, with about 25.059 s of associated GPU activity
union. Their host ranges total only 3.179 s. The 25 MTP observe host ranges
total 24.815 s, including 23.990 s in stream synchronization while previously
queued GPU work finishes. This again is GPU execution/host wait, not CPU expert
computation. Leading prefill kernel-duration sums are `flash_attn_ext_f16`
7.195 s, three `mul_mat_q` specializations 10.216 s combined, and
`gated_delta_net_cuda` 3.146 s. These duration sums can overlap across streams.

At graph granularity, kernels *inside* the 2.647 s target verification graphs
are opaque. A node-granularity trace over steady-state output is needed before
attributing their time to experts, GDN, attention, or Q6 dense work. The
profiled numbers above must not replace unprofiled first-byte, first-model-token,
VRAM, prefill, and decode measurements.

## Reproduction and next capture

The phase API counts can be reproduced from the SQLite export using the
following query; the profiler's default NVTX range implementation uses
`eventType` 59 or 70 for completed push/pop ranges.

```sql
WITH ranges AS (
  SELECT e.rowid AS range_id, e.start, e.end, e.globalTid,
         COALESCE(text_id.value, e.text) AS phase
  FROM NVTX_EVENTS AS e
  LEFT JOIN StringIds AS text_id ON text_id.id = e.textId
  WHERE e.eventType IN (59, 70) AND e.end IS NOT NULL
), api AS (
  SELECT ranges.phase, ranges.range_id, runtime.start, runtime.end,
         name_id.value AS api_name
  FROM ranges
  JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
    ON runtime.globalTid = ranges.globalTid
   AND runtime.start >= ranges.start AND runtime.end <= ranges.end
  JOIN StringIds AS name_id ON name_id.id = runtime.nameId
  WHERE ranges.phase LIKE 'a3b/output%'
)
SELECT phase, COUNT(DISTINCT range_id) AS ranges,
       SUM(api_name LIKE 'cudaGraphLaunch%') AS graph_launches,
       SUM(api_name LIKE 'cudaGraphInstantiate%') AS graph_instantiations,
       SUM(api_name LIKE 'cudaStreamBeginCapture%') AS graph_captures,
       SUM(api_name LIKE 'cudaLaunchKernel%') AS kernel_launches,
       SUM(api_name LIKE 'cudaStreamSynchronize%') AS stream_syncs,
       ROUND(SUM(CASE WHEN api_name LIKE 'cudaStreamSynchronize%'
                      THEN end - start ELSE 0 END) / 1e6, 3) AS stream_sync_ms
FROM api GROUP BY phase ORDER BY phase;
```

The A/B harness now accepts a diagnostic-only `--nsys-session SESSION` option
for a profiler already launched with `--session-new=SESSION --start-later=true
--trace=cuda,nvtx --cuda-graph-trace=node`. On the first nonempty model SSE
delta it calls `nsys start`; after the response it calls `nsys stop` and
`nsys shutdown --kill=false`. Use `--repeats 1` and a separate suite/arm. The
per-trial `*.nsys-control.json` records command latency. Since the SSE trigger
arrives after model generation began and `nsys start` takes time, the node
trace covers only **some remaining output tokens**, not necessarily all 255.
The harness refuses to include that run in a normal unprofiled A/B comparison.
