# Initial A3B validation, 2026-10-14

This records the initial implementation before the memory and MTP optimization pass.
The current all-GPU implementation and results are in [A3B-FULL-GPU.md](A3B-FULL-GPU.md).
Earlier memory and speculative-state evidence is in
[`2026-10-14-a3b-memory-fixed`](../bench/results/2026-10-14-a3b-memory-fixed/README.md) and
[`2026-10-14-a3b-mtp-validation`](../bench/results/2026-10-14-a3b-mtp-validation/README.md).

This record covers the independent `a3b/` engine in Strata-Moe. The original `G:\Strata` checkout was read only.
During this initial validation, both checkouts' existing tracked files had no diff and all implementation changes were new files.
The later A3B-only Python protocol adapter changes Strata-Moe's server; the original checkout remains untouched.

## Model and machine

- `unsloth/Qwen3.6-35B-A3B-GGUF`, revision `a483e9e6cbd595906af30beda3187c2663a1118c`.
- `Qwen3.6-35B-A3B-UD-Q5_K_M.gguf`, 26,456,194,016 bytes; SHA-256 checked against the repository's LFS metadata: `c13ce26253ea334df472bd8fbd2d6da66d8a41195c17f6fcbf44c4d20ece0932`.
- 120 routed tensors: 80 Q5_K, 37 Q6_K, 3 Q8_0. Routed weights occupy 22,666 MiB in RAM. The backend retains their quantized bytes.
- Windows, Threadripper 3960X, 24 physical cores / 48 logical CPUs, 128 GiB RAM; 24 inference threads.
- GPU 0: RTX 2080 Ti, reported VRAM 22 GiB; CUDA 12.6, MSVC 19.38, compute capability 7.5. One GPU used for A3B.
- A pre-existing model remained resident on GPU 1. It was not stopped or changed. GPU 2 was not used for A3B.
- The A3B process was given a 16,384 MiB planning budget. This is a budgeted run on a 22 GiB card, not a test on a physical 16 GiB card.
- llama.cpp: `3cf03257f219afbe7334045ff7c6a06ac68c627d`, the existing Strata setup pin. No third-party source was edited.

## Expert and scheduler checks

CUDA and CPU-only Release builds pass `ctest`. The expert test compares the split implementation to separate native CPU and native GPU graphs for F32, Q4_K, Q5_K, Q6_K and Q8_0.
It checks one/many tokens, shared/per-route activations, separate weight addresses, cache eviction, streamed groups, and actual scheduler placement with GPU inputs and outputs.
CPU results are identical to the native CPU reference. Each mixed row matches the CPU or GPU oracle within `2e-5 * (1 + abs(reference))`.
CPU and GPU activation quantization differ, so mixed output is not promised bit-identical to either complete-device run.

HIP configuration stopped because the HIP SDK is absent. SYCL configuration stopped because the installed compiler lacks SYCL support.
Neither backend has been built or runtime-validated. Linux has not been runtime-validated.

## Same-day, same-GGUF comparison

Commands, stdout, stderr, binary/source hashes and model provenance are saved under
[`bench/results/2026-10-14-a3b-2080ti-final`](../bench/results/2026-10-14-a3b-2080ti-final/results.json).

Each arm starts a fresh process. Same Chinese prompt, 23 tokens, 96 greedy output tokens, context 4096, prompt block 128.
The prompt supplies the completed empty thinking block. OS file cache was not cleared; these are empty expert-cache runs, not cold-SSD measurements.
One measurement per arm is recorded. The generated continuations can differ after rounding differences, so these numbers do not establish a general speed ratio.

| Path | Prompt time | Output time | Output tok/s | Peak GPU memory, including background |
| --- | ---: | ---: | ---: | ---: |
| Native CPU reference | 1.658 s | 9.414 s | 10.20 | 99 MiB |
| GPU trunk, CPU routed experts | 0.803 s | 5.044 s | 19.03 | 2,598 MiB |
| GPU trunk + adaptive expert cache, 16 GiB budget | 2.028 s | 4.108 s | 23.37 | 16,248 MiB |

The cached arm's measured GPU increment is 16,149 MiB over its 99 MiB background, about 15.77 GiB.
The expert cache occupies 13,546.5 MiB. During decode, 90,387 routed rows ran on GPU and 813 on CPU, counted after cache admission.
This does not count a newly uploaded expert as a pre-existing cache hit.

The cache improves decode in this prompt, but its initial upload makes the first prompt slower.
Prompt plus output is about 6.14 s with cache versus 5.85 s with CPU experts. Startup/loading time is excluded from those two figures.
No claim is made that using more cache is always faster on this CPU/GPU combination.

First-token distribution against the same GGUF's native CPU run:

| Path | KL(CPU reference || path) | Top-1 agrees |
| --- | ---: | --- |
| GPU trunk, CPU experts | 0.0001177171 | yes |
| GPU trunk + expert cache | 0.0000277945 | yes |

These are one-position numerical checks, not a perplexity or model-quality benchmark. BF16, task accuracy and long-context retrieval were not compared.

## Real API checks

`tools/test_a3b_http.py` passes against the running model on `127.0.0.1:8096`:

- `/health`, `/v1/models`, and the existing web page.
- OpenAI non-stream and SSE stream, both with normal `finish_reason=stop`.
- Two identical requests in the same process, both answering `17 * 23` with `391`.
- Anthropic Messages text request, answering in Chinese.
- Exported tokenizer round trips Chinese, English and control-token spans.

The first math request took 2.141 s; its repeat took 0.203 s. The engine recomputed the prompt in both cases (`cached_tokens=0`), while keeping experts resident.
Prompt processing dropped from 1,909.9 ms to 94.3 ms. These tiny answers are a reuse check, not a sustained throughput benchmark.
Full responses are in [`api-smoke.json`](../bench/results/2026-10-14-a3b-2080ti-final/api-smoke.json).

## Not tested or not implemented

Actual Qwen3.5 weights were not downloaded. Its official geometry matches this path, but only the named Qwen3.6 GGUF was run end to end.
At this initial checkpoint, images/video, MTP, multiple GPUs, conversation-state reuse, RAM-complement residency, expert mmap, KV streaming, elastic VRAM, mid-request STOP, logit bias and grammars were not implemented. MTP was added in the subsequent optimization pass linked above.
The original Strata engine and defaults remain available separately and were not performance-tested as part of this A3B change.
