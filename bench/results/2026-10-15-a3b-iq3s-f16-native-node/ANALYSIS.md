# Partial output-node trace: F16-tier KV + original Strata IQ experts

Nsight Systems 2024.4.2 captured CUDA/NVTX node activity after the first nonempty model SSE delta of one 50K/256 request on GPU 0 (RTX 2080 Ti). The executable SHA-256 and model identities are in `capture/arms/traced/metadata.json`. It used the single Qwen3.6-35B-A3B UD-IQ3_S mixed GGUF, Q8_0 MTP K2 head, 262,144-token configured limit, 65,536-cell **F16** physical KV tier and `--strata-native-experts`. Profiling began **1.813 s after the trigger** and covers only **61 complete** target-verification and MTP-proposal NVTX ranges, not the full 256-token output. Its observed 68.7 tok/s is profiler-perturbed and must not replace the separate unprofiled 79.35 tok/s two-run median.

Runtime-to-kernel correlation inside those 61 target-verification NVTX ranges gives the following CUDA **kernel-duration sums**:

| Family | Calls | Captured GPU kernel duration |
|---|---:|---:|
| Q6_K `mul_mat_vec_q` (GGML type 14) | 15,311 | 266.0 ms |
| Original Strata `native_down_multi_kernel` | 2,440 | 156.0 ms |
| F16 `flash_attn_ext_f16` | 610 | 155.7 ms |
| Original Strata `native_gu_multi_kernel` | 2,440 | 144.1 ms |
| Remaining kernels | 94,916 | 375.6 ms |
| **All target kernels** | **115,717** | **1,097.5 ms** |

The first four rows account for **721.8 ms (65.8%)** of the 1,097.5 ms target kernel sum. In this F16 tier, `dequantize_block_q8_0_f16` appears **zero** times in the captured target-verification windows. The earlier Q8-KV/standard-GGML-expert node trace captured 1,320 such conversions in 66 windows, taking 238.4 ms; that old run also differed in experts and output, so subtraction of the two traces is **not** an isolated speedup measurement.

The 61 MTP-proposal windows contained 7,686 kernels totaling 187.1 ms: **Q8_0**, GGML type 8, `mul_mat_vec_q` 130.393 ms (976 calls), F16 attention 23.5 ms, and the remainder about 33.2 ms. An earlier revision mislabeled this type-8 proposal sum as Q6_K. The proposal ranges contain **zero** Q6_K type-14 kernels. There were 60 `cudaGraphLaunch` calls inside 61 target-verification ranges and 89 in the 61 proposal ranges. Target verification was submitted asynchronously; the following 61 `target-hidden-observe` ranges spent about 1,197.0 ms in stream synchronization waiting for the target GPU work. That is **not** CPU expert inference.

Across the **entire captured SQLite trace**, Q6_K type-14 `mul_mat_vec_q` has 15,472 unique kernel records and 269.230 ms of total GPU kernel duration. The 61 complete target-verification ranges account for 15,311 of those records and 266.029 ms; the other 161 records/3.201 ms precede the first complete range. The only Q6_K template in this trace has `ncols_dst=3` and the exact-grouped flag. Thus an ideal, physically impossible elimination of all Q6_K arithmetic could save at most 269.230 ms of **captured kernel duration**. This is neither a full-256-token output bound nor a wall-time speedup: the capture omits early output, profiling perturbs timing, and streams can overlap. The default-off Q6_K cross-column experiment can affect target verification but not the Q8_0 proposal kernel sum.

The trace narrows future kernel work to quantized dense products (Q6_K target and Q8_0 MTP proposal), native grouped IQ expert gate/up/down and long-context F16 attention. The original Strata grouped expert kernel really executes here. Its presence does not imply that the original Flash-Next target/commit graph, adaptive expert cache, or cross-layer copy scheduler has been ported. Within a single GPU, another CUDA graph wrapper is unlikely to help: GGML already replays target and individual draft graphs. A later kernel change needs a same-binary, unprofiled 50K/256 A/B with output hash, MTP acceptance, first HTTP byte, first model token, prefill, decode and whole-card VRAM reported.

The SQL attribution joins `NVTX_EVENTS` range `globalTid` and bounds to `CUPTI_ACTIVITY_KIND_RUNTIME`, then matches `correlationId` to `CUPTI_ACTIVITY_KIND_KERNEL` and kernel names in `StringIds`. The following SQLite query reproduces the target Q6_K and proposal Q8_0 counts and durations directly from `f16-native-output.sqlite`:

```sql
WITH launches AS (
    SELECT n.text AS phase, s.value AS kernel_name, k.start, k.end
    FROM NVTX_EVENTS AS n
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
      ON r.globalTid = n.globalTid AND r.start >= n.start AND r.end <= n.end
    JOIN CUPTI_ACTIVITY_KIND_KERNEL AS k ON k.correlationId = r.correlationId
    JOIN StringIds AS s ON s.id = k.demangledName
    WHERE n.text IN ('a3b/output/target-verify', 'a3b/output/mtp-propose')
      AND n.end IS NOT NULL
)
SELECT phase,
    CASE WHEN kernel_name LIKE 'void mul_mat_vec_q<(ggml_type)14,%' THEN 'Q6_K (type 14)'
         WHEN kernel_name LIKE 'void mul_mat_vec_q<(ggml_type)8,%' THEN 'Q8_0 (type 8)' END AS family,
    COUNT(*) AS kernels, ROUND(SUM(end-start)/1e6, 3) AS duration_ms
FROM launches
WHERE family IS NOT NULL
GROUP BY phase, family ORDER BY phase, family;
```

For the unique Q6_K whole-capture total, query the kernel table without an NVTX/range join:

```sql
SELECT COUNT(*) AS kernels, ROUND(SUM(k.end-k.start)/1e6, 3) AS duration_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN StringIds AS s ON s.id = k.demangledName
WHERE s.value LIKE 'void mul_mat_vec_q<(ggml_type)14,%';
```

Kernel-duration sums may overlap across streams and cannot be added to host phase durations or treated as exclusive wall time. Raw data are `f16-native-output.nsys-rep`, `f16-native-output.sqlite` and `capture/arms/traced/run001.nsys-control.json`.
