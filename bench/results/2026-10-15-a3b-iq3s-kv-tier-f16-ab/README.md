# IQ3_S A3B: tiered Q8 versus F16 KV at 50K

This exploratory same-executable comparison used GPU 0 (RTX 2080 Ti, 22,528 MiB VRAM), the same Qwen3.6-35B-A3B UD-IQ3_S mixed GGUF, its Q8_0 MTP head at draft K2, batch 2,048, and the same uncached 50,000-token prompt with 256 output tokens. Both configurations advertised a 262,144-token limit and allocated the existing 65,536-cell physical context tier for this request. `--native-attention-exact` was disabled in both arms to keep the batched GGML attention graph; the only intentional KV-format difference was Q8_0 versus F16. The Q6_K dense tensors remain in the **same** IQ3_S model. Each arm ran twice sequentially, not alternated.

| Arm (two-run median) | HTTP first byte | First model token | Prefill | Output | Whole GPU 0 peak | Output SHA-256 prefix |
|---|---:|---:|---:|---:|---:|---|
| 65K physical Q8 KV tier | 19.55 ms | 28.487 s | 1,768.4 tok/s | 64.85 tok/s | 16,591 MiB | `c1d52a6c` |
| 65K physical F16 KV tier | 15.00 ms | 29.288 s | 1,722.9 tok/s | 72.25 tok/s | 17,119 MiB | `3ee09940` |

The F16 arm gained 7.4 output tok/s (11.4%) and used 528 MiB more sampled GPU memory than the Q8 tier. Its first model token was 0.80 s later and prefill 45.5 tok/s slower in this sequential sample. The answers and MTP acceptance differed (Q8 141/227; F16 139/232), so the end-to-end speed change cannot be assigned solely to an attention kernel. Both arms had stable output hashes within the arm. Additional alternating runs would be needed to separate run-order drift from small first-token/prefill differences.

The earlier full-256K Q8 service on the same card/binary peaked at roughly 20,499–20,528 MiB for the same 50K request, with a 64.9 tok/s two-run output median and the same output hash as the tiered Q8 arm. Thus the 65K Q8 physical tier saved about 3,900 MiB without an observed output penalty in these separate sequential runs; the F16 tier remained about 3,400 MiB below the full-256K Q8 peak. The tier reconstructs the idle target and MTP contexts between independent requests; it is not the original Strata CUDA VMM `--kv-grow` mechanism. The F16-only configuration in this experiment has **not** been validated at 256K physical capacity on this card and should not be used as a 256K-wide service without a Q8 fallback.

Raw per-run timing, GPU/RAM samples, identities and generated-output hashes are in `arms/{q8-tier,f16-tier}/run*.json`, with resource time series in `run*.resources.jsonl`. The engine binary SHA-256 is recorded in both arms' metadata. GPU memory is whole-card sampled usage, not process-attributed. HTTP first byte is protocol response latency, not first generated token.
