# A3B IQ3_S: Q6_K decode audit and next measurement

This records a read-only audit of the original `G:/Strata` implementation and
measurements saved in `G:/Strata-Moe`. The audit predated the separate,
default-off Q6_K verification patch experiment. The target is the **same**
Unsloth Qwen3.6-35B-A3B UD-IQ3_S GGUF
in both arms (SHA-256 `66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66`),
plus the same Q8_0 MTP head. This mixed GGUF contains 252 Q6_K tensors
(1,894.9 MiB of weights), 80 IQ2_S, 37 IQ3_S and 3 IQ4_XS tensors. Q6_K
is part of the IQ3_S model, not a second model. The tensor counts are visible
in `bench/results/2026-10-15-a3b-iq3s-profile/direct-short.log:62-66`;
the byte count follows from the GGUF header and `tools/gguf_reader.py`'s
Q6_K block geometry (256 values / 210 bytes).

The latest same-executable, same-day 50K-input/256-output K2 A/B in
`bench/results/2026-10-15-a3b-iq3s-strata-ab/README.md` measured these
two-run medians on GPU 0, an RTX 2080 Ti (22,528 MiB):

| Route | First HTTP byte | First model delta | Prefill | Output | GPU 0 sampled peak |
|---|---:|---:|---:|---:|---:|
| GGML experts + grouped Q6_K dense | 15.2 ms | 27.44 s | 1,841.0 tok/s | 67.2 tok/s | 20,499 MiB |
| Strata grouped IQ experts and fused gate/up + same grouped Q6_K dense | 14.8 ms | 27.82 s | 1,815.1 tok/s | 70.4 tok/s | 20,499 MiB |

The Strata arm counted 5,070 original grouped expert CUDA calls during
decode/MTP and zero during prefill. Its output and MTP acceptance differed
from the GGML arm, so the 3.2 tok/s difference cannot be attributed solely
to the expert kernel. These two samples per arm do not resolve a small effect
reliably. At 70.4 tok/s, 256 tokens take about 3.64 s; 100 tok/s would take
2.56 s, requiring about 1.08 s (29.6%) less output wall time.

The older Nsight kernel traces **predate the grouped Q6_K option** and
capture a short, roughly 1K-input/96-output experiment without the 50K/K2
workload. In `bench/results/2026-10-15-a3b-iq3s-profile/short-nomtp-nodes.sqlite`
and its `short-nodes-kernels_cuda_gpu_kern_sum.csv`, Q6_K vector kernels sum
375.95 ms; Q6_K MMQ sums 75.34 ms; GDN sums 75.14 ms. Total GPU kernel
durations sum to 1,428.94 ms. Grouping the Q6_K vector launches by `gridX`
in that SQLite trace gives:

| Matrix rows (`gridX`) | Launches | GPU kernel-duration sum |
|---:|---:|---:|
| 248,320 (output head) | 96 | 75.23 ms |
| 8,192 | 3,800 | 117.01 ms |
| 4,096 | 2,850 | 64.47 ms |
| 2,048 | 7,600 | 84.79 ms |
| 512 | 5,700 | 34.46 ms |

Thus the head is only 75.23/375.95 ms of the *old trace's Q6_K vector
kernel sum*. These profiler sums include profiling overhead and do not equal
decode wall time. The older 50K phase timer in
`bench/results/2026-10-15-a3b-iq3s-q6-grouped/README.md:64-70` assigned
3.707 s of a 4.81 s output phase to target completion/hidden-state
observation, but that was the unmodified exact baseline and includes queued
asynchronous GPU work.

A later **post-grouped-Q6, 50K/K2, original-Strata-experts** partial output
trace is now available at
`bench/results/2026-10-15-a3b-iq3s-f16-native-node/ANALYSIS.md`. With F16
physical KV on an RTX 2080 Ti, its 61 captured target-verification ranges
spent 266.0 ms in Q6_K CUDA vector kernels out of 1,097.5 ms total target
kernel duration. The 61 MTP-proposal ranges instead contain 130.393 ms of
**Q8_0 type-8** vector kernels and no Q6_K. An earlier report mislabeled
that proposal sum as Q6_K. Directly summing **all** Q6_K type-14 launches in
the capture gives 15,472 calls and 269.230 ms, including 161 calls outside
the 61 complete target ranges. These are
profiler-perturbed **kernel-duration sums**, not exclusive wall-time savings.
The 61 windows are only a subset of the 256-output request. The separate
unprofiled two-run median was 79.35 tok/s; the traced 68.7 tok/s must not be
used as the deployment result. At 79.35 tok/s, cutting the 256-token output
from about 3.23 s to 2.56 s requires about 0.67 s saved across the full run.

Original Strata's `src/kernels/cuda/native_mmvq.cu:1810-1917` implements a
packed Q6_K layout specifically for the main and MTP **output heads**;
`src/core/native_head.cpp:67-68` enables it with `STRATA_Q6_PACKED=1`.
Porting only that packed head is unlikely to remove enough of the measured
old trace to reach 100 tok/s. It also adds a packed VRAM copy. Original
`src/prefill/moe_mmq.cu:133,220` routes CUDA Q6_K MMQ through llama.cpp's
MMQ, so calling that path is not a new Q6_K prefill kernel. The earlier A3B
attempt to call Strata's native Q6_K MMVQ changed some float bits against
GGML (`bench/results/2026-10-15-a3b-iq3s-fused-gu/README.md:74-80`),
which breaks the exact target-verification contract used here. Strata's
`native_mmvq.cu:1465` also documents a measured 10–50% slowdown for one
three-column Q6_K variant. Directly substituting that kernel has neither
parity nor speed evidence.

The immediate Q6_K experiment is a default-off split of GGML's existing
`vec_dot_q6_K_q8_1` into one weight-lane load/decode and 2–4 column applies
inside the same exact grouped SM75 kernel. It preserves the two-warp K
partition, per-output accumulation sequence, and final reduction, and adds
no packed copy. The patch must pass real-tensor exact parity before timing.
If Q6_K remains material after that, a packed-layout experiment on 8,192-
and 4,096-row trunk tensors may be considered; those 70 tensors total
721.875 MiB, so a second packed copy could cost up to that much VRAM.
Neither experiment has a measured 50K latency, throughput, or VRAM benefit
yet. Keep the existing configuration as default until a same-binary,
unprofiled 50K/256 A/B establishes one.
