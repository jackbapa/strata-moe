# IQ3_S A3B: same-binary 50K/256 Strata-path check

This is an exploratory **same-day, same-executable** comparison on GPU 0
(RTX 2080 Ti, 22,528 MiB) with a 262,144-token configured context, exact
50,000-token uncached prompt, 256 generated tokens, Q8 KV, and the Q8_0 MTP
head at K=2. All four arms use executable SHA-256
`77afd71f957a3a0fca50568b3e18fcbd584754ff9848a4366c44f116ae198c32`,
the same IQ3_S target SHA-256 `66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66`,
MTP SHA-256 `f6cb995fab78e7284097b2e13508e8daad4ffc6a747b433aef399f4e48524b5a`,
and request SHA-256 `d5f460f2c4f7b5a25c571928ca284a2515483b5d78e067b572697cae4ac7ec90`.
The target GGUF contains its Q6_K dense tensors; Q6_K is **not another model**.

Two-request medians, run sequentially in the table's order:

| Opt-in route | First HTTP byte | First model token | Prefill | Output | Whole GPU 0 sampled peak |
|---|---:|---:|---:|---:|---:|
| GGML experts, grouped Q6_K baseline | 10.26 ms | 28.109 s | 1,792.3 tok/s | 66.45 tok/s | 20,499 MiB |
| Original Strata short expert CUDA path | 14.89 ms | 29.395 s | 1,713.0 tok/s | 66.80 tok/s | 20,499 MiB |
| Short expert path + adapted single-step GDN | 14.68 ms | 28.933 s | 1,738.2 tok/s | 66.65 tok/s | 20,499 MiB |
| Above + original Strata MMQ prefill adapter | 15.43 ms | 28.156 s | 1,789.6 tok/s | 67.40 tok/s | 20,750.5 MiB |

The combined arm actually called the copied original Strata grouped IQ MMQ
wrapper 1,000 times for 16,000,000 routed rows during prompt prefill. It also
called the original short expert CUDA path 5,070 times in output/MTP and none
during prefill. The GDN graph matched once while building the decode graph;
this counter is **not** a CUDA graph replay count. The two combined requests
reported 1,160 MMQ graph builds and 102 fallback graph builds each. Their
full output matched the other two Strata-expert arms byte-for-byte, with
233 MTP proposals and 137 accepted tokens in each run. The baseline answered
differently and accepted 141 of 227 proposals, so its output-speed difference
cannot be assigned to a particular kernel alone.

These arms were not alternated; prefill results drifted during the sequential
sequence. The roughly 1 tok/s output spread is too small for a speed claim.
The copied MMQ adapter used about 252 MiB more sampled whole-card VRAM than
the baseline and did not establish a prefill improvement. A separate
`STRATA_A3B_MTP_DEVICE_PREFILL=1` combined attempt failed on its first 50K
request with `MTP device prefill lost its CUDA hidden input`; it has **no valid
performance result** and is excluded from the table. A fix and parity test are
pending. The original Strata cross-step MTP graph and full target verifier
graph are not present in these arms. The 100 tok/s target remains unmet.

Each `arms/<name>/runNNN.json` records HTTP and model-token timings, output
hash, MTP acceptance, GPU/RAM samples, executable/config/model identities and
the engine log counters; `.events.jsonl` and `.resources.jsonl` retain raw
observations. VRAM is whole-GPU sampling, not process-attributed.

An additional native-expert-only request with `STRATA_A3B_TIMING=1` used the
same old executable and request but was **not** part of the uninstrumented
medians. It generated at 69.0 tok/s and recorded 118 verify windows: 758 ms
MTP proposal, 182 ms target verification submission, 2,574 ms target
observation, 194 ms commit, and 4 ms sampling/emit. Observation synchronizes
previously queued GPU work; its 2,574 ms is **not** the cost of a small CPU
hidden-state copy. These five phases sum to about 3.71 seconds of the output
interval. Even eliminating all proposal time would leave about 2.95 seconds
(roughly 87 tok/s), so the 100 tok/s target also requires improvement in the
target verification/observation path. A current CUDA/NVTX trace is needed to
attribute that time to kernels, graph launch gaps, and synchronization.
