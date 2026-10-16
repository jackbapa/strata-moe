# Expert graph tuning and MTP, first run, 2026-10-14

One exploratory trial per arm. Hardware: Threadripper 3960X, 128 GiB RAM, RTX 2080 Ti GPU 0 reporting 22 GiB, Windows WDDM. Qwen3.6-35B-A3B-UD-Q5_K_M, with separate Q8_0 MTP weights for the MTP arm. Both arms share one executable (`9b8480d5...`), the same 23-token prompt, 256-token output limit, 4,096 context and 16 GiB GPU allocation budget. All three optional tuning flags are enabled: `--stable-expert-graphs --preserve-verify-layout --cache-graph-routes`. The unrelated GPU 1 WSL service remained running; GPU 0 was released afterward.

| Metric | Cached experts | MTP, three draft tokens |
|---|---:|---:|
| Decode seconds | 9.598 | 7.179 |
| Decode tokens/s | 26.67 | 35.66 |
| Prefill seconds | 2.145 | 1.963 |
| GPU usage including background | 16,274 MiB | 16,418 MiB |
| GPU increment above 100 MiB background | 16,174 MiB | 16,318 MiB |
| Router reads | 20,480 | 7,280 |
| Native expert layout calls | 28,497 | 8,055 |

MTP accepted 165/267 proposals (61.80%) across 89 windows, producing 2.876 output tokens per window. Draft computation took 0.9107 s and draft-state catchup took 0.2674 s. MTP was 33.7% faster than the same-build cached-expert arm for this prompt. Compared with the earlier executable in `../2026-10-14-a3b-mtp-first/`, the single-trial rates increased from 25.09 to 26.67 tokens/s without MTP and from 30.90 to 35.66 with MTP. Individual flags and run variation have not yet been isolated.

The tuned no-MTP generated-output hash matches the earlier no-MTP output exactly. The MTP output differs. First target-token logits still match exactly between the two current arms; this is not proof of token-for-token speculative equivalence. Separate rollback and batch-layout correctness checks remain necessary.

Peak total GPU use in this MTP trial exceeds 16,384 MiB by 34 MiB because the allocation budget excludes the existing background. The measured increment is below 16 GiB. Report both figures when describing this result, and leave a larger reserve before promising operation on a physical 16 GiB card under the same desktop conditions.

Commands, full executable SHA-256, memory samples, output hashes and counters are in `results.json`. Raw stdout/stderr preserve generated text and engine logs. Float32 logits are ignored by Git.
