# A3B speculative state validation, 2026-10-14

This is a target-model state check, not an MTP speed or acceptance benchmark.
The draft driver is in `a3b/speculative.hpp` / `a3b/speculative.cpp`.
The independent diagnostic executable is `a3b/speculative_test.cpp`.
No original `G:\Strata` files were changed.

## Model and machine

- Existing base model: `unsloth/Qwen3.6-35B-A3B-GGUF`, revision `a483e9e6cbd595906af30beda3187c2663a1118c`, `Qwen3.6-35B-A3B-UD-Q5_K_M.gguf`.
- Model SHA-256: `c13ce26253ea334df472bd8fbd2d6da66d8a41195c17f6fcbf44c4d20ece0932` (previous download verification; this diagnostic did not download or requantize the model).
- Windows, Threadripper 3960X, 24 inference threads, 128 GiB RAM.
- GPU test: GPU 0, RTX 2080 Ti with reported 22 GiB VRAM, CUDA 12.6, compute capability 7.5.
- Fixed llama.cpp revision: `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
- Context 128, batch and ubatch 32, one sequence, three recurrent rollback slots.
- GPU arm puts the trunk and attention state on GPU and overrides all routed expert tensors to the native CPU buffer type. It does not use Strata-Moe's custom expert cache, so cache admissions cannot change the comparison's device assignments.

## Procedure

The fixture tokenizes one fixed English sentence, uses its first eight tokens as the prefix, and the next four as a verification window. It tests keeping 1, 2, 3 and 4 input rows. These correspond to accepting zero, one, two and three draft tokens, with row zero always being the already sampled target token.

For every retained prefix, the diagnostic evaluates a subsequent token and reports two comparisons:

1. **Sequential reference:** full-window batch, trim to the retained prefix, then continuation versus evaluating only the retained prefix token by token, then the same continuation. This checks the resulting next-token distribution while also exposing differences between batched and single-token arithmetic.
2. **Materialized state reference:** the continuation from the pending rollback versus serialization and restoration of that same accepted sequence state, then the same continuation. This isolates the pending GDN snapshot restoration from batched-versus-sequential arithmetic. It uses the standard sequence-state save/load APIs as an independent materialization path.

The second comparison is required to have finite logits, the same top token, and maximum absolute logit error below `1e-5`. The first comparison's measured errors are reported without an expanded tolerance or a parity claim.

The state-materialization comparison is a consistency test: serialization and pending restoration can both select the same incorrect snapshot index. By itself it cannot prove that the selected snapshot is the intended accepted prefix. The separately evaluated sequential prefix is the independent oracle for that question; its GPU-trunk results below are exactly equal on this fixture.

## Results

| Backend | Kept input rows | Sequential-reference max absolute logit error | Sequential-reference KL | State-restore max absolute logit error |
| --- | ---: | ---: | ---: | ---: |
| Native CPU | 1 | 0.603254199 | 0.0171164958 | 0 |
| Native CPU | 2 | 0.510450840 | 0.00195055033 | 0 |
| Native CPU | 3 | 0.748961449 | 0.00236637964 | 0 |
| Native CPU | 4 | 0.416027129 | 0.00245153831 | 0 |
| GPU trunk, native CPU experts | 1 | 0 | 0 | 0 |
| GPU trunk, native CPU experts | 2 | 0 | 0 | 0 |
| GPU trunk, native CPU experts | 3 | 0 | 0 | 0 |
| GPU trunk, native CPU experts | 4 | 0 | 0 | 0 |

All rows have finite logits and matching top-1 tokens in both comparisons. Materialized-state KL is zero in all rows. Both diagnostic runs pass the state-restoration assertion.

The CPU-only arm has measurable batched-versus-sequential differences even when all four rows are kept and no speculative suffix is rejected. Those differences therefore cannot be treated as evidence of a partial-rollback failure. Their exact source within the CPU matrix path was not isolated. The GPU-trunk arm is exactly equal on this fixture; that does not establish exact equality for every prompt, cache placement, quantization or backend.

## Commands and logs

The native CPU executable was compiled with MSVC `/std:c++17 /EHsc /MD /O2 /DNDEBUG`, linking the existing `build-a3b-cpu` llama/ggml static libraries. The CUDA diagnostic target was built through the repository CMake configuration.

```powershell
# Native CPU build of the diagnostic; temporary executable from the existing CPU libraries.
& "$env:TEMP/strata-a3b-speculative-check.exe" --model models/a3b/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf --threads 24

# CUDA-linked diagnostic, GPU trunk and CPU expert override.
cmake --build build-a3b --config Release --target strata-moe-speculative-test -j 4
build-a3b/bin/strata-moe-speculative-test.exe --model models/a3b/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf --threads 24 --device 0
```

Complete logs: `native-cpu-rollback.log`, `gpu-trunk-cpu-experts-rollback.log`. `sha256.json` records the diagnostic source and executable hashes at validation time.

## Draft weights and limits

The existing base GGUF contains neither NextN metadata nor appended `blk.40.*` MTP tensors. An MTP switch alone cannot supply the missing trained head.

The [ggml-org Qwen3.6 GGUF repository](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/tree/baec3ebee244827cda0f4557eafa8b28f7545fa6) publishes a separate `mtp-Qwen3.6-35B-A3B-Q8_0.gguf`: 1,990,649,632 bytes, SHA-256 `f6cb995fab78e7284097b2e13508e8daad4ffc6a747b433aef399f4e48524b5a`. This validation does not load that draft file and does not measure its prediction quality.

The MTP driver follows the fixed llama.cpp [NextN graph](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/models/qwen35moe.cpp) and [MTP speculative driver](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/common/speculative.cpp): shifted target hidden rows, a separate draft attention cache, target verification, and accepted-prefix state trimming. At positive temperature, every emitted token must still be sampled from the target; a drafted token is accepted only if that target sample matches it. This retains the target sampling rule, subject to the numerical behavior of the target verification graph.

The fixed version's `llama_sampler_sample` automatically accepts the sampled token into its sampler history. The generation loop must call it exactly once for each emitted token, including matched proposals, and must stop sampling at the first mismatch. Calling `llama_sampler_accept` again would incorrectly duplicate penalty history. Proposals use their own greedy picks and do not consume the target sampler's random state. Per-request target clearing and draft reset also reset cross-block hidden carryover.

The custom expert cache, real MTP acceptance rate, stochastic-output quality, long contexts, HIP and SYCL are not covered by these two runs.

## Custom-cache candidate check

An additional run uses the actual Strata-Moe expert backend, with `stable_tasks`, `preserve_verify_layout` and `cache_graph_routes` enabled. Routed tensors were counted from the GGUF header: 23,767,023,616 bytes. Model loading used `LLAMA_LOAD_MODE_NONE`, pinned expert RAM, and a 16,384 MiB GPU budget with 512 MiB reserved after the context allocation. The requested expert-cache budget was 13,542.0 MiB; allocated cache buffers totalled 13,457.9 MiB.

The complete 13-token prefix/window/continuation fixture was processed twice before validation. The first warmup copied 4,888.9 MiB into the cache; the second copied zero. During the subsequent independent comparisons, all 97,920 routed rows ran on GPU and zero ran on CPU. There were 561 additional uploads (412.2 MiB), 7,200 routing reads and 10,800 preserved-layout calls. Small numerical changes can alter later routing, so warmup does not prove that every later route was already resident.

| Kept input rows | Sequential-reference max absolute logit error | Sequential-reference KL | State-restore max absolute logit error |
| ---: | ---: | ---: | ---: |
| 1 | 1.16415238 | 0.0205387650 | 0 |
| 2 | 0.413010597 | 0.000981252162 | 0 |
| 3 | 0.401244402 | 0.00305166853 | 0 |
| 4 | 0.490708351 | 0.00105753549 | 0 |

All rows have finite logits and matching top-1 tokens. The strict `--assert-sequential` check **fails** because these KL values exceed `1e-4`; the threshold was not widened. State-materialization comparisons remain exactly equal. Keeping the entire four-row window also differs from the sequential reference, so the result does not isolate a partial-rollback error. It does show that this custom-cache candidate cannot claim an identical sequential target distribution on the fixture.

```powershell
build-a3b/bin/strata-moe-speculative-test.exe --model models/a3b/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf --threads 24 --device 0 --cached-experts --stable-tasks --preserve-verify-layout --cache-graph-routes --assert-sequential
```

Full log: `custom-cache-candidate-rollback.log`; source and executable hashes: `custom-cache-candidate-sha256.json`. This run did not load the MTP draft head or measure its acceptance rate. The fixed CUDA MMVQ implementation selects launch layouts using its token-column count, so preserving natural token/lane dimensions by itself does not guarantee single-token arithmetic. The exact contribution of those layouts to these measured differences has not yet been isolated.

## Custom-cache single-column verification

The next candidate enables `exact_verify` for quantized GPU expert tasks in verification windows of two through four tokens. It gathers the routed rows into expert lanes of one pseudo token, uses the same single-column MMVQ layout as decode, then scatters outputs back to their original token/lane positions. Single-token decode and long prefill keep their existing layouts. This follows the original Strata verifier's requirement that each verification column use single-token arithmetic.

The same strict diagnostic was repeated with all earlier flags retained plus `--exact-verify`. Cache budget and warmup were unchanged. During validation, all 97,920 routed rows ran on GPU and zero on CPU. The backend recorded 480 exact-verification calls, 10,320 preserved-layout calls, 7,200 routing reads, and 546 uploads (401.0 MiB).

| Kept input rows | Sequential-reference max absolute logit error | Sequential-reference KL | State-restore max absolute logit error |
| ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 |

All logits are finite and all top-1 tokens agree. Both the state-restoration assertion and the independent sequential-prefix assertion pass, with no tolerance change. For this fixture, switching to single-column expert verification eliminates all differences measured in the preceding custom-cache candidate. This supports the MMVQ layout explanation; it does not establish identical output for other prompts, cache assignments or devices.

```powershell
build-a3b/bin/strata-moe-speculative-test.exe --model models/a3b/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf --threads 24 --device 0 --cached-experts --stable-tasks --preserve-verify-layout --cache-graph-routes --exact-verify --assert-sequential
```

Full log: `custom-cache-exact-rollback.log`; source and executable hashes: `custom-cache-exact-sha256.json`. The test retains its 512 MiB reserve; later serving benchmarks may reserve more headroom. This run checks target verification and state restoration, without an MTP draft model, long-context or stochastic-output quality evaluation.

## Fixed mixed CPU/GPU expert placement

The final strict check caps the requested expert cache at 2,048 MiB (2,036.4 MiB allocated), warms the complete fixture twice, then disables cache admission before validation. Freezing placement prevents speculative grouping from changing which device computes an expert between the candidate and reference. Every expert remains available; cache misses run on CPU.

The comparison processed **65,454 GPU rows and 32,466 CPU rows**, with **zero uploads** after admission was frozen. The diagnostic explicitly requires both device row counts to increase and upload count to stay unchanged. It recorded 480 exact-verification calls, 975 complete preserved-layout calls and 7,200 routing reads. The small cache's warmups stream 4,882.8 MiB and 4,880.4 MiB respectively; those copies are outside the frozen validation phase.

For every kept input length from one to four, the independent sequential-prefix comparison has maximum and mean absolute logit error **0**, KL **0**, and matching top-1 tokens. The materialized-state comparison is also exactly equal. Both strict assertions pass, without changing either threshold.

```powershell
build-a3b/bin/strata-moe-speculative-test.exe --model models/a3b/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf --threads 24 --device 0 --cached-experts --cache-mib 2048 --freeze-cache --stable-tasks --preserve-verify-layout --cache-graph-routes --exact-verify --assert-sequential
```

Full log: `custom-cache-mixed-frozen-rollback.log`; source and executable hashes: `custom-cache-mixed-frozen-sha256.json`. This validates the real mixed hit/miss computation and accepted-state restoration on the fixture. Adaptive placement changes, long contexts, sampled-output quality, HIP and SYCL remain outside this diagnostic's scope.
