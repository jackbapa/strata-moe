# A3B IQ3_S, one RTX 2080 Ti, 256K context, 50K input

Hardware: Threadripper 3960X, 128 GiB RAM, RTX 2080 Ti 22,528 MiB on
GPU 0, Windows. The target was Unsloth Qwen3.6-35B-A3B UD-IQ3_S
(SHA-256 `66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66`).
The separate MTP head was ggml-org Qwen3.6 Q8_0. The service used
`strata-a3b-iq3s.json`: all target experts on GPU 0, 262144 context,
2048 batch, Q8_0 K/V, MTP K2, no embedding/head pointer sharing.

`tools/bench_a3b_latency.py` generated the same synthetic Chinese MoE
prompt as the Q8 50K baseline and confirmed 50,000 input tokens through
both the count endpoint and response usage. Each request used temperature
0, seed 12345, 256 output tokens, no input-cache reuse. Model load is
excluded. HTTP first byte measures the initial response headers; first
model token is the first nonempty streamed answer text.

| Variant | First model token | Prefill | Output (engine) | HTTP first byte |
|---|---:|---:|---:|---:|
| Selected Q8 MTP K2, first | 27.57 s | 1837.9 tok/s | 56.9 tok/s | 18.82 ms |
| Selected Q8 MTP K2, repeat 1 | 28.80 s | 1749.1 tok/s | 54.2 tok/s | 19.93 ms |
| Selected Q8 MTP K2, repeat 2 | 31.03 s | 1614.1 tok/s | 52.5 tok/s | 19.12 ms |
| No MTP, one run | 25.72 s | 2013.8 tok/s | 50.6 tok/s | 9.57 ms |
| Q8 MTP K1, one run | 28.22 s | 1794.0 tok/s | 56.7 tok/s | 19.91 ms |
| Q8 MTP K2 fused rounds, one run | 28.64 s | 1770.8 tok/s | 55.5 tok/s | 18.89 ms |
| Q8 MTP K3, one run | 28.73 s | 1763.3 tok/s | 49.2 tok/s | 18.61 ms |
| Q4_0 MTP K2, one run | 27.99 s | 1804.4 tok/s | 55.7 tok/s | 19.23 ms |
| Q4_0 MTP K2, FP16 K/V, one run | 27.43 s | 1845.1 tok/s | 56.9 tok/s | 19.63 ms |

The three selected K2 runs give **median 28.80 s first model token,
1749.1 prefill tok/s, 54.2 output tok/s**. Their spread matters: one
run does not establish a stable maximum. The earlier official Q8_0 model
on this machine recorded 111.38 s, 449.9 prefill tok/s and 39.5 output
tok/s for a 50K/256 request. The IQ3_S result is 3.9 times faster to the
first model token and 37% faster in output by these medians, but model
quantization changed and output text is not a quality comparison.
The [Q8 measurement](../2026-10-15-a3b-q8-packed-gu/README.md) is the
baseline. The original Strata checkout was not modified.

Raw API plans, event streams, output files, timing JSON and summaries are
under the `api-50k*` subfolders. The selected K2 output hash was identical
across the initial, fused, repeat and Q4 MTP runs. K1, K3 and no-MTP output
text differed under their different verification layouts. The target model
verified speculative tokens, but this is not bitwise parity across layouts.

The [retrieval test](needle-result.json) placed `海蓝-73192` near the middle
of a 50,017-token prompt. The model returned exactly that marker in nine
output tokens, with zero cached prompt tokens. This is one success, not a
long-context accuracy rate.

At 256K, the full GPU target without MTP ran; adding FP16 K/V plus the
Q8_0 MTP head failed at MTP context compute-buffer allocation. Q8 K/V
plus Q8 MTP loaded and generated. The smaller Q4_0 MTP head also let
FP16 K/V load and finish one 50K/256 trial. That trial had no speed
advantage beyond run-to-run variation and left only 562 MiB GPU memory
free, so it remains optional; the selected config uses Q8 K/V and Q8 MTP.
GPU 0 showed 20,513 MiB used during the first selected 50K run;
this was a point sample, not a sampled or lifetime peak. No expert bytes
were uploaded in the selected native-GPU path. A separate 4 GiB-cache
hybrid smoke run with real IQ2_S/IQ3_S/IQ4_XS experts and MTP completed,
with CPU pool calls and GPU rows in `hybrid-mtp-smoke.log`; its short-input
17.7 tok/s is not comparable to the 50K output rate.

Build: `python tools/a3b.py build --jobs 8` passed for CUDA SM75.
`ctest --test-dir build-a3b --output-on-failure -j 4` passed 16/16.
Python option/download/profile tests passed. A Q8_0 hybrid smoke run also
passed after the CPU pool was generalized. HIP and SYCL builds were not run
on this Windows CUDA machine.
