# Q8 GPU expert execution and dense weight reuse, 2026-10-14

Hardware: GPU0 NVIDIA RTX 2080 Ti, SM75, driver 617.42, approximately 22 GiB available VRAM. Every GPU test used `CUDA_VISIBLE_DEVICES=0`. GPU1 and GPU2 did not run these tests. Original `G:\Strata` remained read-only. llama.cpp stays pinned to `3cf03257f219afbe7334045ff7c6a06ac68c627d` with the recorded independent A3B patches.

## GPU-only RAM experts

The expert arena stays in system RAM. Persistent cache hits execute on the GPU. Cold experts transfer through two bounded GPU banks, then execute the complete routed projection or FFN on that same GPU. Copied/consumed events protect bank reuse and asynchronous task inputs. Pinned arenas transfer directly; pageable arenas use bounded host upload banks. Exception exits drain both internal GPU streams before task or upload memory can be released/reused. No experts or router contributions are omitted.

The [pinned](gpu-only-pinned-final.log), [pageable](gpu-only-pageable-final.log), and [forced pinned-allocation fallback](gpu-only-auto-pageable-fallback.log) fixtures each passed 72 combinations: Q8 projection and complete FFN, zero/two/all-24 persistent slots, 1/3/8 input tokens, one/eight input lanes, frozen/adaptive admission, duplicate expert IDs and multiple staging waves. Each result was compared bit for bit with separate ordinary one-token GPU graphs holding all expert weights on the GPU. There was no CPU-math oracle alternative or widened tolerance. Warm repeated routes allocated neither additional banks nor additional expert task graphs; every routed row was accounted for and CPU expert rows stayed zero. Invalid expert IDs were rejected, followed by a correct retry on the same context. This does not inject an asynchronous device failure or out-of-memory failure after submission.

The fallback fixture sets `GGML_CUDA_NO_PINNED=1` in a fresh process. It verifies that a returned CPU arena is detected before the custom buffer type hides its real type, automatically enabling bounded banks. The intentionally pageable small banks also log that DMA may synchronize. These checks establish correctness, not transfer/compute overlap when pinning is unavailable. GPU-only execution is opt-in; the existing Q4/CPU expert path is retained.

The first [CTest run](ctest-cuda.log) passed the six existing tests and failed the new test's scheduler construction: GGML requires a CPU backend as the final fallback entry. The fixture was corrected to provide that mandatory entry while asserting custom expert placement and zero CPU expert rows. The subsequent [complete CUDA CTest run](ctest-cuda-final-env1.log), with both dense reuse and multi-stage argmax enabled, passed all seven tests. These CTest logs precede the later frozen-order and automatic arena-fallback fix; the three final GPU-only fixture logs above use the later build.

## Canonical Q8 dense multi-token weight reuse

`STRATA_A3B_DENSE_Q8_REUSE=1` is a separate opt-in switch, cached once per process. It applies to SM75, Q8_0, non-ID, non-fused dense matrices whose input uses the existing `--native-dense-exact` layout `[K,1,T]`, T=2..8. Each CTA loads a weight block once into registers and uses it for independent token accumulators. The activation quantizer, four-warp singleton K partition, existing Q8 dot helper, ascending cross-warp additions and XOR reduction tree are preserved. Singleton, fused/ID, other-type, other-shape and other-architecture paths retain their original implementation. No weights are requantized.

Both [env0](dense-q8-reuse-env0.log) and [env1](dense-q8-reuse-env1.log) passed bitwise comparison for eight dense geometries, contiguous/padded inputs, zero input rows and varying magnitudes. Tokens 1..9 were checked against nine independent singleton GPU oracle graphs; the new kernel covers 2..8 and token 9 retains the old exact layout. Token 10 checks that the exact graph hook declines the unsupported window; it is not an independent-reference assertion.

The 248,320-row Q8 head fixture also passed both layouts and every checked token count: [env0](dense-q8-head-env0.log), [env1](dense-q8-head-env1.log). It has the real `[2048,248320]` head geometry, approximately 515 MiB of Q8 weight storage, and generated repeated quantized rows. It is not a model accuracy evaluation.

One paired fresh-process measurement, 100 synchronous graph calls after warmup:

| Verification tokens | Original exact layout, microseconds | Weight reuse, microseconds |
| --- | --- | --- |
| 1 (unchanged kernel) | 1087.122 | 1086.774 |
| 2 | 2158.220 | 1351.370 |
| 3 | 3207.746 | 1677.943 |
| 4 | 4218.036 | 2002.038 |
| 5 | 5257.517 | 2339.116 |
| 6 | 6297.067 | 2676.756 |
| 7 | 7374.238 | 3031.337 |
| 8 | 8363.788 | 3262.909 |

[Machine-readable timing](dense-head-benchmark.json). These numbers include the synchronous graph API and input quantization, with uploads and output reads outside the measured loop. They are not isolated kernel times or model token throughput. The dense fixture measurements preceded the later GPU-only frozen-order/fallback fix; the CUDA reuse patch and dense fixture source were unchanged by that fix.

The [reproducible patch](../../../a3b/patches/llama-dense-q8-reuse.patch) passed clean forward application and exact source reproduction against the official fixed pin, plus reverse application check. CUDA was built and tested. HIP/SYCL execution was not tested; the added CUDA-only kernels and dispatcher are excluded for HIP/MUSA.

## Full Q8 model validation

The initial [real Q8 frozen-cache run](q8-mtp-k2-frozen-dense-reuse.log) passed every K2 rollback/sequential/snapshot comparison with max abs / KL zero, then failed the actual MTP class at output limit 8, token index 0. Its expert arena's 32 GiB pinned allocation also fell back to ordinary RAM. Both findings are retained in the log.

The frozen-cache failure exposed a separate issue: long-prefill misses were sorted by evolving LFU counts even when placement was frozen. Prior generations could therefore change staging groups and their multi-column/MMQ arithmetic, altering the next request's first token. Frozen admission now keeps the miss group's route order fixed. Adaptive long-prefill grouping can still round differently as placement/frequencies change; short-window exact rollback alone does not prove all prefill layouts bitwise equivalent.

Revalidation **passed** with the full `Qwen3.6-35B-A3B-Q8_0.gguf`, frozen 2048 MiB persistent cache, K2 MTP, exact dense/attention, GPU greedy sampling, shared exact Q8 weights, deferred catch-up, draft microbatch 2 and explicit pageable experts. All keep counts 1..3 had max abs / KL zero for both independent sequential-prefix and snapshot continuation. Actual MTP output limits 1/2/3/4/5/8/16/32 matched the sequential CPU-greedy baseline token for token. [Complete later diagnostic log](q8-mtp-k2-frozen-pageable-dense-reuse-fixed.log).

`STRATA_A3B_PREFILL_DIAGNOSTIC=1` also repeated each CPU-logit prefill before the actual MTP boundary. Every full-vocabulary logit comparison was bitwise identical to baseline, with max abs zero, first token 279, and FNV1a `377220aa6c7cb2c9`. This supports the frozen-group-order diagnosis and verifies the corrected frozen/pageable configuration; it does not claim adaptive long-prefill groups are numerically invariant. Model throughput is measured separately and is not filled in from the synthetic timing above. [Final source/executable/log hashes](final-sha256.json).

## Optional asynchronous expert events

`--async-expert-events` requires GPU-only experts. The custom backend forwards same-device GPU events and asynchronous copies; the scheduler queues a device wait before the following CUDA trunk instead of joining the expert tail on the CPU. Routing still downloads the actual IDs and waits once on the expert compute stream before the CPU plan. Task-owned input/index vectors retain a submission serial until reuse is safe; upload banks retain their copied/consumed events. Errors and destruction still drain both streams. CPU/different-device interludes and blocking input copies keep a conservative host barrier when a device wait cannot protect their buffers.

The initial [trigger-count failure](gpu-only-async-pinned.log) revealed that upstream creates scheduler events only for parallel input copies. The corrected opt-in scheduler creates events for the custom backend and its same-device CUDA trunk even with one graph copy. It keeps the original behavior when the option is off. [Reproducible scheduler patch](../../../a3b/patches/llama-async-expert-events.patch) is the exact diff against official pin `3cf03257`; reverse application also passes.

[Pinned](gpu-only-async-pinned-events.log) and [pageable](gpu-only-async-pageable-events.log) pure-GPU fixtures each passed all 72 independent singleton GPU comparisons. They submit the scheduler asynchronously and synchronize only the final CUDA consumer stream; event/route counters must prove the bridge ran. Each arm also passed three host-input lifetime tests with eight repeated submissions and immediate poisoning of the caller's host input and routing tensors.

Independent static review then identified CPU-interlude and per-thread blocking-copy boundaries requiring conservative synchronization. The hardened [pinned](gpu-expert-async-boundaries-allocated.log) and [pageable](gpu-expert-async-boundaries-pageable-allocated.log) fixtures each passed 18 forced CUDA/custom/CPU/CUDA cases plus 54 adjacent CPU-copy cases, including user-input and ordinary fallback copies, as well as the lifetime tests. The earlier [fixture-placement failure](gpu-expert-async-boundaries.log) remains recorded: `set_tensor_backend` marks the scheduler unreset, so the test must explicitly allocate after manual assignments to retain the intended CPU interlude. It was not a numerical tolerance failure.

The [full Q8 K2 model run](q8-mtp-k2-frozen-pageable-async-events.log), with frozen 2048 MiB cache, exact dense/attention, GPU sampling, shared weights, deferred catch-up and draft microbatch 2, passed every independent sequential/snapshot continuation with max abs and KL zero. Actual MTP output boundaries 1/2/3/4/5/8/16/32 matched baseline. This run preceded the later conservative CPU-boundary hardening; those later boundaries have their own synthetic checks above.

One paired cold-process model measurement on the 23-token prompt/256-token output improved from 32.71 to 34.32 decode tokens/s (4.9%), with identical output SHA, acceptance and expert upload bytes. [Async measurement and executable hash](../2026-10-14-a3b-q8-async-k2/results.json), [baseline](../2026-10-14-a3b-q8-first/results.json). This is one pair, not a repeated timing study, and it does not meet the 100-token/s goal.

## Global cache profile and adaptive draft length

The optional CPU draft policy explores locally measured round costs for 0..the configured maximum, then uses observed conditional prefix acceptance to choose expected committed tokens per millisecond. It probes neighbouring and stale sizes periodically, resets on each request, and treats terminal EOS/output limits as censored rather than invented rejections. Planned windows that stop early at a draft EOS retain their measured cost, while unoffered proposals do not inflate the expected benefit. It adds no GPU confidence/softmax operation or target RNG calls. Both fixed and adaptive proposals still undergo full target verification.

The pure CPU policy test passed. The [CPU ctest run](ctest-adaptive-profile-cpu.log) reports 10 registered tests, five passed and five GPU-specific skips, with zero failures. Policy cases include low/high acceptance, cost exploration, neighbour/stale probes, censoring, early draft EOS, request reset and remaining-token bounds. These checks establish scheduling logic; they do not predict throughput.

The [full Q8 global-profile/adaptive validation](q8-profile-adaptive-k6-strict.log) passed with a maximum of six drafts. Its verified profile selected 642 complete experts, 2046.375 MiB within a 2048 MiB budget. Placement was frozen after warmup; both GPU hits and staged GPU misses were exercised, with CPU expert rows zero and no additional staging-bank allocation during validation. Dense and attention graph hooks both recorded acceptances; these counters include reserve/probe graphs and are not execution counts.

All independent sequential-prefix and snapshot comparisons at keep counts 1..7 had max abs zero, KL zero and matching top1. Actual adaptive MTP output limits 1/2/3/4/5/8/16/32 matched the sequential CPU-greedy baseline, with target/draft GPU sampling, shared weights and deferred catch-up. This run used draft microbatch **8**; the earlier K2 validation separately covered forced draft microbatch **2**. The configured six-draft maximum is not a claim that every adaptive round drafted six tokens. No adaptive/profile throughput result is inferred from this correctness run.

## Experimental static GPU experts

The [full Q8 static/MTP validation](q8-gpu-static-mtp-k6-strict.log) completed with exit code 0 on GPU0. Its fixed profile selected 3,137 complete experts, 9,999.2 MiB within a 10,000 MiB cache budget. Forty static CUDA layer providers were prepared. Short windows routed, fetched misses and computed experts on CUDA; the longer prompt path retained the existing backend and shared the same fixed cache.

Every keep count 1..7, representing zero through six accepted draft input rows, reported max abs error and KL zero, matching top1 and finite logits for both the independent sequential-prefix continuation and snapshot restoration. Validation accounted for 67,200 complete expert rows / 201,600 projection rows, with positive cache hits and GPU misses, `CPU_rows=0`, and `route_reads=0`. The program's final assertion also rejects any CPU expert row after the MTP class checks.

Actual MTP output limits 1/2/3/4/5/8/16/32 all matched the token IDs of the same GPU target decoded sequentially with CPU greedy sampling. This covered shared Q8 weights, target/draft GPU sampling, deferred catch-up, a six-draft maximum and draft microbatch 8. The logits checks report numerical error rather than a byte comparison; the output token comparison is exact. Limits 1/2/3 offered no proposals, while the later limits exercised both accepted proposals and mismatches.

This is a single-sequence correctness check with an eight-token prefix, not a speed measurement or coverage of arbitrary prompts, long contexts, stochastic sampling, cancellation or concurrent requests. Static execution remains experimental and opt-in; the production configuration is unchanged. `tools/a3b.py config --gpu-static-experts` requires GPU-only experts, a local expert profile, device 0 and decay interval 0. With MTP it also selects exact dense verification. The [offline configuration selection tests](config-static-options.log) passed all 14 cases, including unchanged defaults, preserved existing config files and rejection before tokenizer export for invalid combinations.
