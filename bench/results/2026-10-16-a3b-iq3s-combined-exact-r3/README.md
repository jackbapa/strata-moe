# IQ3_S four-switch exact A/B on GPU 0

This is an **unprofiled, same-binary, interleaved** 50,000-token-input / 256-token-output test on the local RTX 2080 Ti (22,528 MiB), Windows, with a 262,144-token configured context. The target is `Qwen3.6-35B-A3B-UD-IQ3_S.gguf`; the matching MTP draft is `mtp-Qwen3.6-35B-A3B-Q8_0.gguf`, K2. The request had zero cached prompt tokens. The target GGUF's Q6_K dense tensors are part of this **same IQ3_S file**, not another target model.

Both arms used the same engine SHA-256 `a0a3deaec132200616e7db2c0a18e3cfd719575e660ce6a7d29af58f6c4d9c04`, same target and MTP SHA-256, sampling, context, GPU, F16 short KV tier and original Strata grouped IQ expert path. The baseline retained `STRATA_A3B_Q6_GROUPED_EXACT=1`. The combined arm enabled four additional default-off switches: `STRATA_A3B_FUSED_GU=1` and `STRATA_A3B_DOWN_MODE=1` select adapted original Strata grouped IQ expert phases, `STRATA_A3B_Q6_REUSE_EXACT=1` selects a local GGML Q6_K dense matvec reuse optimization, and `STRATA_A3B_MTP_KV_ONLY=1` avoids unused MTP catch-up graph roots. The Q6 reuse path is **not** an original Strata kernel call.

The separate-process strict gate passed 16 byte-identical target/draft state pairs, 29 identical proposal lines and eight matching output/acceptance classes. The six API runs followed off/combined/off/combined/off/combined; every run produced the same full output SHA-256, 119-window proposal trace SHA-256, 238 offered and 136 accepted draft tokens. The runner stopped each arm before auditing its buffered proposal log, and restored the selected service on completion. The earlier `...combined-exact/` and `...combined-exact-r2/` folders are failed harness attempts and are **not** performance observations.

| Metric (three-run median) | Baseline | Combined | Difference |
|---|---:|---:|---:|
| HTTP response first byte | 19.25 ms | 19.89 ms | +0.64 ms |
| First nonempty model token | 29.385 s | 28.575 s | -0.810 s |
| Engine prompt speed | 1,708.9 tok/s | 1,765.0 tok/s | +56.1 tok/s |
| Engine output speed | 75.2 tok/s | 83.2 tok/s | +8.0 tok/s (+10.6%) |
| GPU 0 observed whole-card used peak | 17,119 MiB | 17,103 MiB | -16 MiB |

Output speed for the three baseline runs was 77.9, 75.2 and 73.4 tok/s; combined was 86.7, 82.7 and 83.2 tok/s. First-model-token times were 28.555, 29.385 and 29.668 s baseline; 28.094, 28.575 and 28.638 s combined. This supports a combined-configuration speed benefit on this workload, **not** a measured benefit for each switch separately or a 100 tok/s result. The 16 MiB whole-card VRAM difference is too small to call a meaningful memory saving. Native expert calls in output fell from 5,116 to 4,998 with the MTP KV-only route; none of these calls were in prefill. These counters establish actual grouped-kernel use in decode, but this test does not establish that the original Strata cross-layer prefill stager/issuer or CPU expert pool executed. The resident IQ3_S path reports zero CPU expert rows and zero expert uploads.

The engine's sampled physical RSS peak was 1.627 versus 1.621 GiB; sampled whole-system used memory was 24.455 versus 24.418 GiB. Windows private committed bytes were about 20.36 GiB in both arms and must not be added to physical RSS or GPU VRAM as another RAM allocation. These are sampled observations, not a process-exclusive system-RAM benchmark.

Machine-readable evidence: [exact-ab-report.json](exact-ab-report.json), [comparison.json](comparison.json), [identity.json](identity.json), [runner-state.json](runner-state.json), `strict/`, and per-run records in `arms/`. The large binary strict-state dumps remain local and are ignored by Git; their hashes and pass/fail results are in the JSON reports. The trial JSON records the measured first-byte and first-model-token separately; GPU memory is sampled for the entire GPU 0, not attributed to this process. CUDA/HIP/SYCL are not all built by this benchmark.
