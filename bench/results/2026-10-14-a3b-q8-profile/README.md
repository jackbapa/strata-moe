# Q8 global expert-cache profile, 2026-10-14

`actual-model-training-v2.profile` is the valid profile. Its [provenance](actual-model-training-v2.profile.provenance.json) records the exact source files, whole-model SHA-256, benchmark JSON fingerprints and preview allocation. The model is the complete official `ggml-org/Qwen3.6-35B-A3B-GGUF` Q8_0 file, 36,903,139,328 bytes, SHA-256 `d8d7842cc657d720f39546878e431937c84473aa130c36371669ac17c80c7361`. Python hashed the complete file in 28.490 seconds on this machine; runtime C++ hashing is measured separately in startup logs.

Only [q8-first/routes.tsv](../2026-10-14-a3b-q8-first/routes.tsv) and [q8-async-k2/routes.tsv](../2026-10-14-a3b-q8-async-k2/routes.tsv) train this profile. They are separate real-model executions of the **same prompt**, each generating 256 output tokens. Both traces have the same SHA-256 and 6,242 layer/expert observations. Combining them doubles that task's weight; it does not introduce task diversity. This profile has not yet been validated on independent prompts.

The earlier `INVALID-synthetic-contaminated-v1.profile` and its sidecar are retained as an invalid experiment and must not be enabled. They incorrectly included the small GPU expert parity fixture's pinned/pageable TSVs. Those synthetic 24-expert fixture routes do not describe the real model.

## Format and selection

The text starts with `STRATA_A3B_EXPERT_PROFILE_V1`. It binds a complete target model SHA-256 and byte count to fixed geometry: 40 layers, 256 experts per layer, embedding 2,048, routed FFN 512, eight selected experts, and 3,342,336 bytes per gate/up/down expert bundle. All 10,240 `(layer, expert, score)` rows follow, descending by score with ascending layer/expert tie breaks. Zero scores are present in the file but do not consume cache slots.

Each input trace contributes the maximum route count across gate/up/down and backend rows for a given layer/expert. Fused gate-only counts are therefore not multiplied by three. The generator then sums independent trace executions. `lfu_frequency` is not the training count. The recorded fingerprint identifies the exact TSV snapshots parsed and the supplied benchmark JSON files.

Runtime selection uses its actual available budget, rather than the generator's preview budget. It chooses positive-score global top N and returns explicit ordered expert IDs per layer, real allocated bytes, score totals and training coverage. The C++ loader rejects wrong model byte size or whole-file hash, malformed geometry, non-Q8_0 routed tensors, duplicate/unsorted/missing score rows, score overflow and unsupported format. Model hashing reads the whole file once per load; its time is recorded as `model_hash_ms`. No GPU inference is required to generate or validate the profile.

## Training-placement comparison

The [machine-readable comparison](placement-training-comparison.json) evaluates placement against these same training traces. At a preview budget of 16,704 MiB, 5,240 full bundles fit, using 17,513,840,640 bytes. The remaining 1,572,864 bytes cannot hold another full bundle. The allocation is uneven: the first three layers receive 243/235/239 experts; later layers receive 84–181 in this trace.

| Placement | Slots | Covered training route score | Total score | Training coverage |
| --- | ---: | ---: | ---: | ---: |
| Best 131 experts independently in every layer | 5,240 | 221,314 | 225,920 | 97.961225% |
| Global top 5,240 | 5,240 | 223,916 | 225,920 | 99.112960% |

These are theoretical placement counts on the supplied training task. They are not live cache hit rates, measured throughput, independent generalization evidence, or evidence that 100 output token/s has been achieved. Current adaptive LFU admission and static profile placement also have different startup and migration costs; those require model measurements.

## Reproduce

Run in `G:\Strata-Moe`:

```powershell
.venv\Scripts\python.exe tools/test_a3b_cache_profile.py
.venv\Scripts\python.exe tools/a3b_cache_profile.py --model models/a3b/Qwen3.6-35B-A3B-Q8_0.gguf --trace bench/results/2026-10-14-a3b-q8-first/routes.tsv --trace bench/results/2026-10-14-a3b-q8-async-k2/routes.tsv --benchmark-json bench/results/2026-10-14-a3b-q8-first/results.json --benchmark-json bench/results/2026-10-14-a3b-q8-async-k2/results.json --out NEW-profile.txt --budget-mib 16704
```

The generator refuses to overwrite an existing profile or sidecar. Python's 14 offline regressions passed, covering projection/backend deduplication, independent-trace summation, invalid TSV values/columns, overflow, global allocation, Q8 geometry/payload bounds and exact output/provenance SHA-256 bytes. Both the CUDA-linked and CPU-only builds passed all 34 checks in `a3b/cache_profile_test.cpp` after integration. Their `failed to read magic` message is the expected negative fixture: the known full-file SHA of `abc` passes, then GGUF inspection correctly rejects it. These tests create no GPU backend. The [offline verification manifest](offline-verification.json) records test commands, source/executable hashes and the successful download/protocol regressions. No new GPU workload was run for profile preparation.

The current engine preloads the selected experts and then permits adaptive LFU admission, so runtime placement can depart from this static plan. Training coverage must not be substituted for observed hit/miss counters. Exported route TSVs are process-lifetime counts, including the profile startup dummy token when enabled; successive cumulative snapshots from one process must not be summed as independent traces.
