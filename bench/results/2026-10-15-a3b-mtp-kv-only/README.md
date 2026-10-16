# A3B MTP KV-only catch-up: correctness and exploratory A/B

The opt-in `STRATA_A3B_MTP_KV_ONLY=1` stops the Qwen3.5/3.6 MoE MTP catch-up graph after its original Q/K/V and K/V-cache writes, before attention, FFN and the output head. It applies only to zero-output, masked NextN inference without ordinary embeddings or pooling. Other MTP steps and target inference keep the original graph. The environment switch is fixed per process; restart the experiment process to change it. Default is off, including the restored production service.

GPU 0 was the RTX 2080 Ti with 22,528 MiB. Both API arms used the same executable `bf92cad55095f9909671eabd57b4c36bcda98579761a2e20642b8a7302b69225`, the same IQ3_S mixed-quantization target, Q8_0 MTP model, MTP K2, target batch 2,048, draft microbatch 512, configured context 262,144 and actual 65,536-cell F16 KV tier. Model SHA-256: `66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66`; draft SHA-256: `f6cb995fab78e7284097b2e13508e8daad4ffc6a747b433aef399f4e48524b5a`. Each request used the identical uncached 50,000-token prompt and generated 256 tokens at temperature 0 / seed 12345. Two off runs preceded two on runs; this small sequential experiment is not randomized and does not establish a stable speedup.

| Unprofiled API metric | Off | On |
|---|---:|---:|
| Output, two individual runs | 81.7 / 80.5 tok/s | 83.8 / 79.1 tok/s |
| Output median | 81.10 tok/s | 81.45 tok/s |
| First model token median | 27.616 s | 27.428 s |
| Prefill median | 1,819.65 tok/s | 1,837.30 tok/s |
| HTTP first byte median | 14.99 ms | 14.86 ms |
| Whole GPU 0 sampled peak | 17,119 MiB | 17,103 MiB |
| Actual native expert calls / request decode | 5,116 | 4,998 |
| MTP accepted / offered | 136 / 238 | 136 / 238 |

The observed output difference is only +0.43%, and the individual runs cross in direction. First-token median improved by 0.188 s, also within this small sample's variability. The feature remains experimental and default off. These measurements do not meet the 100 tok/s target. Whole-card GPU memory is sampled total usage, not process-attributed allocation. Peak process RSS medians were 1.644 / 1.626 GiB; private committed bytes were 20.389 / 20.413 GiB and must not be read as physical RAM. CPU expert rows and expert upload bytes were reported as zero in both native GPU arms.

All four complete outputs had SHA-256 `03fc938322f153aa7f2b93b062ffc6b820b034322ac86a5b8fceb7d7758a1bf7`, matching the prior default-off native F16 arm. The 238 raw proposal-trace lines from the two requests in each arm compare byte-for-byte equal, with identical acceptance. `api-proposal-parity.json` records this independent check. The corrected `comparison.json` reports `mtp_proposal_trace_identical=null` (unavailable) because the per-run JSON did not capture these traces. The older `api-comparison.log` printed `false` for that missing evidence; it is retained as the original console record, not as evidence of a mismatch. The raw engine logs are retained.

The real-model state test used the same IQ3_S target and Q8_0 draft, K2, GPU sampling, deferred catch-up, native dense/attention exact reference paths and draft microbatch **2** to force prompt catch-up splitting. Independent env 0 and env 1 processes passed output limits 1/2/3/4/5/8/16/32. All **16 final sequence-state pairs** (eight draft KV and eight target KV/GDN) compare byte-for-byte equal. Each dump occurs after `finish()` and output/acceptance assertions, and validates serialized and written sizes. State excludes transient logits, hidden-output buffers and sampler history; the additional 29-line proposal trace also matches exactly. Rollback keep 1/2/3 and snapshot checks all had max absolute error 0 and KL 0. See `state-parity.json`, `strict-off.log`, `strict-on.log` and the raw `state-off/` / `state-on/` bytes. The on diagnostic reached 50 graph builds, while off had none; this is explicitly a graph-build counter, not an execution counter.

Actual execution was checked separately, not inferred from graph builds. The native expert counter is incremented on the GPU in `plan_expert_groups` and downloaded at the request boundary; the same 50K proposal/output paths made 118 fewer expert FFN invocations per request. Two short node-level Nsight fixture captures then correlated CUDA runtime launches inside the same named NVTX phases to kernel activity:

| Short profiled fixture phase | Off | On |
|---|---:|---:|
| Catch-up decode NVTX ranges | 41 | 41 |
| Catch-up GPU kernel invocations | 3,767 | 1,422 |
| Catch-up FA-related kernel invocations | 254 | 0 |
| Proposal decode NVTX ranges | 55 | 55 |
| Proposal GPU kernel invocations | 3,410 | 3,410 |
| Proposal FA-related kernel invocations | 110 | 110 |

FA-related counts include attention and its combine kernels. The unchanged proposal phase verifies that capture/attribution remained active. `nsys-kernel-path.json`, both `.nsys-rep` files and both `.sqlite` exports preserve the evidence. These short fixtures have 128-cell contexts and target exact-attention test hooks; their timings are not 50K throughput measurements and are excluded from the API medians.

The complete CUDA-build CTest suite passed **18/18**, serially, in 66.33 s (`ctest.log`). No pure-CPU/HIP/SYCL build was run in this test window. After testing, port 8096 was restored with `strata-a3b-iq3s-q6-kv-tier-f16-auto-native.json`; `/v1/status` reported loaded=true, correct model and engine. Restored engine PID was 37728. No production configuration was edited and the original `G:/Strata` source was not modified.
