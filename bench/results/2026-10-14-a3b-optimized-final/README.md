# Final candidate: repeated MTP benchmark, 2026-10-14

Hardware: Threadripper 3960X, 128 GiB RAM, RTX 2080 Ti GPU 0 reporting 22 GiB, Windows WDDM. The unrelated WSL service on GPU 1 remained running. Model: Qwen3.6-35B-A3B-UD-Q5_K_M, with separate Q8_0 MTP weights for the MTP arm. All six runs use executable SHA-256 `5efc81275af3d150242c4c72342f0e52394c6e29c30a6bf8cb7df9b24dd92753`.

Each arm ran in three fresh processes, with the three cached-expert runs followed by the three MTP runs. Each process uses the same 23-token prompt, 256 generated tokens, greedy decoding, context 4,096, batch 128, 24 CPU threads and a 16 GiB GPU allocation budget. All four options are enabled in both arms: `--stable-expert-graphs --preserve-verify-layout --cache-graph-routes --exact-verify`. The candidate reserves 640 MiB outside the GPU expert-cache budget. GPU 0 returned to 100 MiB background use after all runs.

| Metric | Cached experts, no MTP | MTP, three draft tokens |
|---|---:|---:|
| Decode tokens/s, trials 1/2/3 | 26.59 / 26.57 / 26.65 | 31.89 / 32.24 / 31.92 |
| Median decode tokens/s | **26.59** | **31.92** |
| Median decode seconds | 9.627 | 8.019 |
| Median prefill seconds | 2.152 | 1.915 |
| Maximum process working set | 23.202 GiB | 23.841 GiB |
| Maximum private commitment | 39.281 GiB | 40.249 GiB |
| Maximum GPU usage including background | 16,034 MiB | 16,258 MiB |
| Maximum GPU increment above background | 15,934 MiB | 16,158 MiB |
| Lowest system available RAM | 27.151 GiB | 26.727 GiB |
| Expert cache | 13,369.4 MiB | 11,687.2 MiB |
| Single-column verification calls per run | 0 | 11,760 |

MTP improves the median decode rate by **20.05%** on this prompt. All MTP runs accepted 157/294 proposals (53.40%) across 98 verification windows, producing 2.612 output tokens per window. Draft computation took 0.9757–1.0198 s and draft-state catchup took 0.2826–0.2896 s. The single-column counter confirms the corrected verification arithmetic was exercised. The earlier 35.66 tokens/s exploratory result used different verification arithmetic and is superseded by this correctness-oriented candidate.

Every run within each arm produces the same output hash. First target-token logits match exactly across all six runs. Output hashes differ between the two arms. Loading MTP reduces the target expert cache, which changes CPU/GPU expert placement and can change numerical rounding; this benchmark does not claim byte-identical full outputs across configurations. The strict rollback validation separately checks continuation against independently decoded prefixes. Quality over a corpus and Qwen3.5 weights have not been evaluated here.

The [strict full-cache rollback log](../2026-10-14-a3b-mtp-validation/custom-cache-exact-rollback.log) covers zero, partial and complete draft acceptance with independent sequential-prefix and state-snapshot continuations. The [strict mixed CPU/GPU rollback log](../2026-10-14-a3b-mtp-validation/custom-cache-mixed-frozen-rollback.log) additionally freezes cache placement at 2,036.4 MiB, exercising 65,454 GPU rows and 32,466 CPU rows. For all four retained-prefix lengths in each test, continuation logits have maximum absolute error 0 and KL divergence 0, with matching top-1 tokens. Frozen placement isolates verification and rollback from cache-admission changes.

The private-commitment figures include nonresident CUDA/WDDM allocations and must not be presented as resident RAM. The working-set improvement was separately isolated in `../2026-10-14-a3b-memory-fixed/`: disabling the source GGUF mmap reduced working set from 46.800 to 23.178 GiB with the same executable and weights.

`results.json` contains all six commands, executable hashes, output hashes, first-token comparisons, counters and memory measurements. `summary.json` contains medians and maxima. `provenance.json` records source, executable, prompt, full model and MTP SHA-256 values; both model files were rehashed after the performance runs. Stdout/stderr preserve the full outputs and engine logs. Float32 logits are ignored by Git.
