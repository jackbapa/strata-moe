# IQ3_S four-switch leave-one-out measurement

Local Windows RTX 2080 Ti, GPU 0 (22,528 MiB), same engine executable SHA-256 `a0a3deaec132200616e7db2c0a18e3cfd719575e660ce6a7d29af58f6c4d9c04`, same Qwen3.6-35B-A3B UD-IQ3_S target and matching Q8_0 MTP GGUF. Every unprofiled API trial used the same uncached 50,000-token input, 256 outputs, configured 262,144-token context, MTP K2, batch 2,048 and GPU 0. The target GGUF's dense Q6_K tensors remain part of **that same IQ3_S file**. Every arm enabled the original Strata grouped IQ expert path and `STRATA_A3B_Q6_GROUPED_EXACT=1`.

The five real-model short strict fixtures each passed sequential continuation and rollback; all four leave-one-out arms had the same eight output/acceptance classes, 29 draft-proposal lines, and **16 byte-identical serialized target/draft state files** as all-on. One state-summary log line in `no_down` was interleaved with a CUDA diagnostic, so the experiment's parser validates the complete numeric prefix plus exact file set, sizes and SHA-256 rather than requiring its path tail to occupy one line. The original combined A/B parser was not changed. All 13 long requests also shared the full-output SHA-256 `03fc9383...d7758a1bf7`, 119-window proposal SHA-256 `caaeb4a5...6a086b4ea8a`, and 136 accepted of 238 offered draft tokens. The runner restored and checked the all-on 8096 service after testing.

The sequence was `all_on, no_gu, no_down, all_on, no_q6, no_kv, all_on, no_kv, no_q6, all_on, no_down, no_gu, all_on`. These are conditional leave-one-out contrasts with the **other three switches on**; they are not independent main effects.

| Arm | Runs | Output median tok/s | First model token median | Prefill median tok/s | GPU0 whole-card sampled peak |
|---|---:|---:|---:|---:|---:|
| All four on | 5 | 83.0 | 28.606 s | 1,755.8 | 17,103 MiB |
| No Strata fused gate/up | 2 | 82.15 | 28.341 s | 1,776.85 | 17,103 MiB |
| No Strata 16-lane down | 2 | 80.7 | 28.514 s | 1,764.95 | 17,103 MiB |
| No GGML Q6_K reuse | 2 | 82.15 | 28.653 s | 1,754.6 | 17,103 MiB |
| No MTP KV-only catch-up | 2 | 79.5 | 29.639 s | 1,693.55 | 17,119 MiB |

The five all-on output measurements were 87.5, 83.9, 82.7, 82.8 and 83.0 tok/s, showing start-of-suite drift. To avoid crediting a switch for this drift, linearly interpolating between the **adjacent all-on anchors** around each leave-one-out trial gives all-on-minus-without output differences of **+2.10/+2.83 tok/s** for fused gate/up, **+2.60/+3.97** for 16-lane down, **+0.80/+1.17** for GGML Q6_K reuse, and **+3.20/+3.63** for MTP KV-only. This local interpolation is exploratory with two observations per leave-one-out arm; it is not a fitted factorial model and the effects should not be summed as a predicted total gain. In particular, the Q6_K improvement is small relative to run variation, and Q6 reuse is a **local GGML optimization, not an original Strata kernel**.

MTP KV-only reduced original Strata native expert decode calls from 5,116 to 4,998, because the draft catch-up no longer computes unused downstream graph roots. It also reduced sampled whole-GPU peak by 16 MiB, too little to call meaningful VRAM saving. The HTTP first byte denotes early response headers, not model latency; medians ranged from 9.54 to 19.00 ms across arms. The selected 50K path had zero CPU expert rows and zero expert uploads, so the original cross-layer host copy ring and CPU miss pool were not exercised. The result supports keeping the four-switch combination for this workload, with output below the 100 tok/s target.

Primary evidence: [exact-ablation-report.json](exact-ablation-report.json), [runner-state.json](runner-state.json), `strict/`, and 13 `arms/*/run*.json` files. GPU memory is whole-card usage, not process-attributed. The prior [six-run all-off-versus-all-on A/B](../2026-10-16-a3b-iq3s-combined-exact-r3/README.md) measured 75.2 versus 83.2 output tok/s and 29.385 versus 28.575 s first model token; its timings are a separate experiment and were not mixed into this ablation.
