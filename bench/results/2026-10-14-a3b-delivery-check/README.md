# Final executable delivery check, 2026-10-14

This checks the final executable after reporting real MTP acceptance in the serving protocol, enabling exact verification automatically when MTP is requested, and adding the cache-admission freeze test hook. It preserves the earlier six-run benchmark in `../2026-10-14-a3b-optimized-final/`; these final edits do not change the computation used there.

Hardware: Threadripper 3960X, 128 GiB RAM, RTX 2080 Ti GPU 0 reporting 22 GiB, Windows WDDM. The unrelated WSL service on GPU 1 remained running. Model: Qwen3.6-35B-A3B-UD-Q5_K_M plus Q8_0 MTP weights for the MTP arm. Both fresh-process runs use the same 23-token prompt, 256 generated tokens, context 4,096, batch 128, 24 CPU threads, 16 GiB GPU allocation budget and all four expert/verification tuning options. Executable SHA-256 is `12eb4fd1fba161de94a05487696a937ce91485b791d0899c2e0ddd1a647ffb6a`.

| Metric | Cached experts, no MTP | MTP, three draft tokens |
|---|---:|---:|
| Decode tokens/s | 26.50 | 31.84 |
| Decode seconds | 9.659 | 8.040 |
| Prefill seconds | 2.150 | 1.923 |
| Peak GPU usage including background | 16,034 MiB | 16,258 MiB |
| Single-column verification calls | 0 | 11,760 |
| Output hash matches corresponding earlier arm | yes | yes |

The earlier three-trial medians were 26.59 and 31.92 tokens/s, consistent with this final check. MTP again accepted 157/294 proposals in 98 windows. First target-token logits match exactly between configurations. Each final output hash also matches its corresponding earlier benchmark arm, confirming the metadata/test-hook edits preserve those generated outputs. Cross-configuration full outputs differ because expert-cache space and CPU/GPU placement differ; this check does not claim otherwise.

Separate strict continuation tests passed for both [full GPU-cache placement](../2026-10-14-a3b-mtp-validation/custom-cache-exact-rollback.log) and [frozen mixed CPU/GPU placement](../2026-10-14-a3b-mtp-validation/custom-cache-mixed-frozen-rollback.log), covering zero, partial and complete draft acceptance. All tested independent-prefix and state-snapshot continuation comparisons report maximum absolute logit error 0 and KL 0. The mixed test exercises 65,454 GPU rows and 32,466 CPU rows with no cache uploads during comparison.

`results.json`, `summary.json`, and stdout/stderr preserve commands, counters, timing, memory and output comparisons. `provenance.json` records current source/executable hashes and references the unchanged model hashes fully recomputed after the six-run benchmark. Multi-GB model hashing was not repeated during this check. The two engine processes exited before the web service was restarted.
