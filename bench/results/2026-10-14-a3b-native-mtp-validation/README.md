# Native GPU MTP validation, 2026-10-14

Hardware: one NVIDIA RTX 2080 Ti, compute capability 7.5, driver-reported 22,527 MiB. Tests used CUDA device 0 only. Original `G:\Strata` was read-only. The fixed llama.cpp pin is `3cf03257f219afbe7334045ff7c6a06ac68c627d`, with the repository's recorded native/fused GGML graph patches.

## Final native GPU state and actual MTP class: PASS

The final singleton FlashAttention graph hook removed the sequential-reference differences documented below. Each fresh process used the same eight-token prefix, a `K+1`-token verification window and a continuation token. Every keep count `1..K+1` was compared both with an independently decoded sequential prefix and with a materialized accepted-state snapshot. The criteria stayed top-1 agreement and KL below `1e-4` for the independent reference, and maximum absolute logit error below `1e-5` for the snapshot. Every final comparison was exactly equal, exceeding these criteria.

| Draft K | Dense exact | Independent reference max abs / KL, all keep counts | Snapshot max abs / KL | Actual MTP output boundaries | Log |
| --- | --- | --- | --- | --- | --- |
| 1 | Off | 0 / 0 | 0 / 0 | All pass | [K1](native-attention-k1-mtp-class.log) |
| 2 | Off | 0 / 0 | 0 / 0 | All pass | [K2](native-attention-k2-mtp-class.log) |
| 3 | Off | 0 / 0 | 0 / 0 | All pass | [K3](native-attention-k3-mtp-class.log) |
| 4 | On | 0 / 0 | 0 / 0 | All pass | [K4](native-attention-dense-k4-mtp-class.log) |
| 6 | On | 0 / 0 | 0 / 0 | All pass | [K6](native-attention-dense-k6-mtp-class.log) |

Every run also exercised the actual `MtpDraft` class, resetting the request for output limits 1, 2, 3, 4, 5, 8, 16 and 32. All emitted token IDs matched the sequential CPU-greedy sampler baseline. These runs enabled GPU draft and target sampling, deferred draft catch-up, verified shared weights, GPU target embedding, NextN hidden output, native fused routed FFN, exact singleton attention, and `STRATA_A3B_ARGMAX_MULTI=1`. The target retained full logits for the independent numerical comparison. The tests cover this finite greedy fixture and these output boundaries; stochastic sampling and API behavior require separate service tests.

A final placement review found that the test's initial shared-table pattern `output\\.weight` also matched `blk.40.attn_output.weight`, placing that 8.5 MiB draft matrix on the host. The main program already used anchored patterns and was unaffected. All independent target-state checks above remain valid; the earlier class runs establish output equality with that draft placement, rather than fully resident draft execution. The test patterns were anchored and [K2 was rerun with the actual all-GPU draft and draft microbatch 2](native-attention-k2-draftubatch2-mtp-class.log): all strict comparisons again had max abs / KL zero and every output boundary passed. The new load log shows 857.36 MiB of draft GPU weights and only the three retained shared copies, 1,030.63 MiB, in the host model buffer.

That final K2 run keeps draft logical batch 32 but uses microbatch 2, forcing the eight-row prompt catch-up and accepted verification prefixes larger than two rows through the decoder's microbatch split. Token IDs and hidden rows use the same split indices. It also checks the explicit accepted-output metric against matched proposals. The reporting implementation now counts a matching terminal EOS even though it is never fed back as a subsequent input; this fixture does not force an EOS match. Recurrent-state trim still uses consumed input rows. [Final split-test sources, executable and log hashes](native-draft-split-final.sha256.json) record this later test revision.

The shared-weight helper verified file tensor type, shape and SHA256 before borrowing target pointers. In this GPU-embedding configuration it reported 1,030.633 MiB of reused GPU weights, zero reused host weights, and 1,030.633 MiB of retained CPU draft copies. GPU saving depends on where the target tables reside. The MTP sampler download counter was eight bytes per GPU draft sample; hidden-state downloads remain separate.

Native hook graph-builder acceptances, including reserve/probe graphs, were respectively dense/fused/attention: K1 `0/80/20`, K2 `0/120/30`, K3 `0/160/40`, K4 `1957/200/50`, K6 `2739/280/70`. These establish hook selection, not execution counts. K8 was not tested.

Command template for K1, K2 and K3 (K4 and K6 additionally use `--native-dense-exact`):

```powershell
$env:STRATA_A3B_ARGMAX_MULTI='1'
build-a3b/bin/strata-moe-speculative-test.exe --model models/a3b/Qwen3.6-35B-A3B-Strata-Q4_K_S-Q8Head.gguf --mtp-model models/a3b/mtp-Qwen3.6-35B-A3B-Q8_0.gguf --device 0 --native-gpu --fused-moe --native-attention-exact --gpu-embedding --nextn-hidden --draft-count 2 --threads 4 --assert-sequential --gpu-sampling --defer-mtp-catchup --share-mtp-weights
```

[Final sources, executables, patch and log hashes](native-attention-final.sha256.json). The main model SHA256 is `9a91f467e1dc65f40487cc8f495b29a294a8afdbde7eb22dc4bf746ad350161e`; the draft model SHA256 is `f6cb995fab78e7284097b2e13508e8daad4ffc6a747b433aef399f4e48524b5a`.

## Device greedy sampler and optional two-stage argmax

The final CUDA fixture suite passed 66 cases with the default argmax path and another 66 with `STRATA_A3B_ARGMAX_MULTI=1`. Vocabularies 4097, 16383, 16384, 16385, 248320 and 248321 cover the dispatch threshold and tail tiles. Fixtures cover first-maximum ties across tiles, unique final elements, signed zero, finite `-FLT_MAX`, positive/negative infinity, NaNs and all-invalid rows. A separate opt-in run also directly asserted the raw two-stage argmax's earliest-index tie rule. The device sampler's guard rejects all rows containing a non-finite logit. Logs: [default](argmax-legacy-cuda.log), [two-stage sampler](argmax-multi-cuda.log), [two-stage direct raw-op assertions](argmax-multi-direct-cuda.log).

On this RTX 2080 Ti, one paired fresh-process synthetic measurement of 500 synchronous sampler-graph calls at vocabulary 248,320 took 223.169 microseconds per call with the default path and 136.168 microseconds with the opt-in path. This includes the sampler graph and synchronous backend API overhead; it is neither isolated CUDA kernel time nor model throughput. The opt-in path uses 256 threads per CTA, 4096 elements per tile and a second merge kernel with pool scratch, without a host transfer between stages. The environment switch is cached on first call and requires a fresh process. Shapes outside 16,384..16,777,216 elements and 1..64 rows retain the original kernel; HIP and MUSA retain their original paths. The existing default kernel body is unchanged. CUDA was built and measured; HIP/SYCL builds were not available in this environment.

The reproducible [argmax patch](../../../a3b/patches/llama-argmax-multi.patch) is against official llama.cpp pin `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Both a clean forward application and a reverse application check against the tested source passed.

### Earlier sampler fixtures

`strata-moe-device-sampler-test.exe` passed all seven synthetic fixtures on the CPU and CUDA0: unique maximum, equal maxima with the CPU sampler's first-maximum rule, positive infinity, negative infinity, one NaN, all negative infinity and all NaN. Invalid rows produce a non-finite guard; the caller rejects them. The graph exposes one token and one float guard, eight bytes per row, rather than a 248,320-element F32 vocabulary output. This is a structural transfer-size test, not an end-to-end throughput measurement.

Logs: [CPU](device-sampler-cpu.log), [CUDA](device-sampler-cuda.log). Corresponding `*.sha256.json` files record sources and executable hashes.

## Initial native GPU revisions: strict reference failed

Model: `models/a3b/Qwen3.6-35B-A3B-Strata-Q4_K_S-Q8Head.gguf`, a recorded raw Q8 output-head transplant into the downloaded Q4_K_S GGUF. The original downloaded model is preserved. No new quantization was performed by this validation.

Each fresh process used the same eight-token prefix, a `K+1`-token verification window and a continuation token. For every keep count `1..K+1`, the test compared the continuation against independently decoding only the accepted prefix, one token at a time. The strict criterion remained top-1 agreement and KL below `1e-4`; it was not widened.

| Draft K | Verify rows | Maximum sequential-reference KL | All top-1 agree | Pending rollback vs materialized snapshot | Strict result |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0.00555089948 | Yes | All logits exactly equal | FAIL |
| 2 | 3 | 0.00141319744 | Yes | All logits exactly equal | FAIL |
| 3 | 4 | 0.000811941908 | Yes | All logits exactly equal | FAIL |
| 4 | 5 | 0.0112680259 | Yes | All logits exactly equal | FAIL |
| 6 | 7 | 0.0468023423 | Yes | All logits exactly equal | FAIL |

Full logs: [K1](native-fused-k1-rollback.log), [K2](native-fused-k2-rollback.log), [K3](native-fused-k3-rollback.log), [K4](native-fused-k4-rollback.log), [K6](native-fused-k6-rollback.log). [Source/executable hashes](native-fused-rollback.sha256.json). K8 was not run after these failures.

The additional [unfused K3 run](native-unfused-k3-rollback.log) produced exactly the same maximum/mean errors and KL values as fused K3 for every keep count. The whole routed-FFN hook did not introduce an additional difference in this fixture. At that revision, no tested draft count passed the strict criterion.

The subsequent [dense one-column K3 run](native-dense-k3-mtp-class.log), with both `--gpu-embedding` and `--nextn-hidden` enabled, again reproduced exactly the same sequential errors and KL values. Snapshot continuation was still exactly equal. Its state check failed before the optional actual `MtpDraft` class and shared-weight boundary tests could execute. [Dense revision hashes](native-dense-k3.sha256.json) record the executable and engine sources; the test source was subsequently extended with an intermediate-tensor diagnostic and is not included in that record.

After fixing CUDA's expert-fusion eligibility to inspect the token axis of `MUL_MAT_ID`, the [normal K3 state run](native-fusionfixed-k3-mtp-class.log) still reproduced the same errors. Its native fused graph-builder count was 160, confirming that this graph hook applied. This count includes reserve/probe graphs; it is not an execution count.

The corresponding [synchronized intermediate trace](native-fusionfixed-k3-trace.log) found the first nonzero difference at `attn_pregate-3`, the first full-attention layer, with maximum absolute errors of `2.38e-7` to `4.77e-7`. All compared named activations in layers 0–2 were exactly equal. For verification row 2, the difference grew to `1.27e-5` at the attention output projection, `2.46e-4` at the post-attention norm and approximately `1e-3` in shared-FFN projections. This provides a specific attention-batch arithmetic source to investigate; it does not prove every later discrepancy has the same cause. The callback synchronizes observed nodes and can inhibit fusion, so a normal-graph strict test is still required after a fix.

The first trace version incorrectly paired repeated generic names such as `norm-0` by retaining only the last tensor with that name. Its `FIRST_DIFFERENCE` labels for those generic names are invalid; ignore those rows. The unique attention names above were unaffected. The source was subsequently corrected to pair each name by occurrence. [Fusion revision hashes](native-fusionfixed-k3.sha256.json) record the engine/executable; the evolving diagnostic source is not included in that record.

Command template (replace `K` with 1, 2, 3, 4 or 6):

```powershell
build-a3b/bin/strata-moe-speculative-test.exe --model models/a3b/Qwen3.6-35B-A3B-Strata-Q4_K_S-Q8Head.gguf --device 0 --native-gpu --fused-moe --draft-count K --threads 4 --assert-sequential
```

Exact snapshot continuation alone does not prove that the accepted recurrent state equals an independent reference. Both comparisons are retained. In these initial revisions K4 and K6 already differed when only the first verification input was retained, indicating batch arithmetic differences affected the first row. The later singleton FlashAttention hook and, for K4/K6, dense one-column hook passed the independent reference in the final runs above. The initial failing logs are retained to distinguish the earlier arithmetic issue from rollback restoration.
