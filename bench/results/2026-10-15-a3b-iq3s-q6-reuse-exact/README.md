# A3B Q6_K cross-column load/apply probe (15 October 2026)

This is a default-off adaptation of original Strata's Q6_K weight `load` plus
per-column `apply` idea in the A3B clone. It changes only the GGML CUDA exact
grouped Q6_K verification specialization on sm_75, with 2–4 columns. The
existing two-warp K partition, each output's accumulation order, and final
reduction remain in place. There is no extra packed weight copy. Original
`G:/Strata` was not modified. The A3B patch is
`a3b/patches/llama-zzzzz-a3b-q6-reuse-exact.patch` (SHA-256
`1a5b88e02cee0d478cbe1ac8052604091d0175105295369e4c41f9bd47611b8a`).

The CUDA build and tests ran on GPU 0, an RTX 2080 Ti (sm_75), with CUDA
12.6. The parity executable SHA-256 was
`2a883068083f6532c049d93ae0737e9d30b1d1770a23f81716464d704d54f822`.
`STRATA_A3B_Q6_GROUPED_EXACT=1` selected the already measured two-warp
grouped path; `STRATA_A3B_Q6_REUSE_EXACT=0/1` selected the old/new weight
expression within it. The newly prepared MTP KV-only experiment remained off
(`STRATA_A3B_MTP_KV_ONLY=0`). All four event runs used this one executable.

The updated fixture tests Q6_K 2048-input matrices with **512, 2,048, 4,096,
8,192 and 248,320 rows**, with 1–10 input columns and both contiguous and
padded input strides. These are A3B dense matrix shapes, but the weights are
synthetic quantized values; the large head repeats 128 independently
quantized rows to avoid a multi-gigabyte float staging matrix. Each result is
compared bitwise against separate one-token ordinary GGML CUDA products.
Both settings passed **100/100** checks. The event runs repeated the same
parity tests, also 100/100 each. Raw logs are `parity-reuse-{0,1}.log` and
`events-{0,1,2,3}-reuse-*.stdout.log` in this folder.

The event microbenchmark made 200 asynchronous GGML graph submissions per
shape between CUDA events after five warmups. Values below are medians of two
runs per setting in 0/1/1/0 order for **three verification columns**. They
include input quantization, Q6_K matvec and graph work; host submission gaps
can affect small shapes. They are not isolated Q6 kernel times or complete
model throughput.

| Q6_K output rows | Reuse off | Reuse on | Median change |
|---:|---:|---:|---:|
| 512 | 9.407 µs | 7.816 µs | 16.9% faster |
| 2,048 | 13.444 µs | 12.482 µs | 7.2% faster |
| 4,096 | 26.220 µs | 19.309 µs | 26.4% faster |
| 8,192 | 35.323 µs | 35.331 µs | no resolved gain |
| 248,320 output head | 939.380 µs | 936.051 µs | no resolved gain |

The 8,192-row and head shapes account for 134.258/269.230 ms of all Q6_K
kernel duration in the available partial 50K output trace. Applying these
five noisy microbenchmark ratios to that trace's per-shape kernel sums would
suggest only about **20.7 ms** saved across the captured Q6 kernels. This is
an exploratory projection, not an actual wall-time measurement or a bound on
full-request speed. The trace covers only 61 complete target windows;
proposals use Q8_0 and cannot benefit from this Q6 change. See the corrected
`bench/results/2026-10-15-a3b-iq3s-f16-native-node/ANALYSIS.md` for the
reproducible attribution SQL and unique Q6 total.

An output improvement to 100 tok/s from the prior 79.35 tok/s F16-tier
baseline would need about 0.67 s saved across 256 output tokens. This probe
does not show a plausible route to that improvement, so the change remains
**off by default**. A 50K/256 API A/B was deliberately not run in this GPU
window; no first-byte, first-model-token, decode-rate or VRAM gain is claimed
for this patch. That GPU window was released for the separate MTP experiment.

Static checks: the patch reverses cleanly from the pinned vendor tree; all
18 patch hashes and all 17 stamped vendor file hashes matched
`.strata-a3b-patch-state.json`. The modified parity fixture built successfully
for CUDA sm_75. Other backends were not built in this window.
