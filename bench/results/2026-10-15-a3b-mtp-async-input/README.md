# A3B IQ3_S MTP K2 pinned-input experiment

On the RTX 2080 Ti GPU 0 (22,528 MiB), this opt-in patch stages the Qwen
MTP one-token hidden, token, position, output-ID, KV-index and attention-mask
inputs in two stable CUDA pinned-host records. It uploads them asynchronously
on the **same GGML CUDA backend** used by the scheduler. The two records stay
valid through the final draft synchronization. It leaves the ordinary input
path selected unless both `STRATA_A3B_MTP_DEVICE_CHAIN_K2=1` and
`STRATA_A3B_MTP_ASYNC_INPUT=1` are set. A second-step graph fallback waits for
the already-queued first step before resetting the scheduler; a failed chain
also synchronizes before its device buffers can be reused. Both fixes are in
the last suffix patch, `a3b/patches/llama-zzzzzzzz-a3b-mtp-async-fallback-sync.patch`.

This follows the original Strata MTP pinned-input idea. It **does not** port
the original round/step/commit CUDA graph or remove the final sampled-token
readback. The normal A3B K2 path still uses GGML draft decode graphs.

The final CUDA sm_75 build passed. On the final diagnostic executable SHA-256
`02d535824cc05233f459179d97dafc7a96d57d3f154a380986ba965cb8cdb046`,
ordinary K2, forced second-step fallback and pinned async K2 each passed all
eight output-limit classes. All 16 serialized draft/target KV/GDN states,
29 proposal-trace lines and eight class fields were identical across the three
arms. The forced fallback log confirms it executed after first-step pinned
DMA. The Nsight reruns' 16 state files per arm also matched the unprofiled
strict files byte-for-byte. `strict-comparison.json` retains the complete
state hashes.

Nsight Systems 2024.4.2 used the same final diagnostic binary with NVTX and
the short strict fixture. Raw `.nsys-rep` and SQLite files, plus the
`analyze_syncs.py` classification and `graph-analysis.json`, are in each arm's
directory. The pre-safety-fix profiles are preserved under
`pre-fallback-fix/` and are not used for the final table.

| Short K2 profile | Ordinary inputs | Forced step-2 fallback | Pinned async |
| --- | ---: | ---: | ---: |
| K2 NVTX ranges | 26 | 26 | 26 |
| `cudaStreamSynchronize` | 624 | 364 | 52 |
| CUDA graph launches | 46 | 0 | 47 |
| CUDA memcpy API calls | 546 | 546 | 546 |
| Sum of K2 host-range durations, ms | 114.446 | 125.333 | 101.497 |
| Sum of stream-sync API durations, ms | 97.042 | 85.060 | 83.067 |

The ordinary and pinned runs have the same 312 H2D input copies. Pinned staging
removes the immediate stream sync after those small H2D copies and the
inter-step 8192-byte D2D copy; the 26 final 4-byte D2H syncs remain. The
forced-fallback arm rebuilds each second-step graph and is only a safety test.
Graph-launch counts differ slightly between independent profile runs because
capture/reuse decisions changed; neither normal arm is one complete original
Strata graph. These profiled short-fixture durations cannot be treated as
50K service throughput.

Six 50,000-prompt-token / 256-output-token requests ran in order
selected, K2 sync, K2 async, K2 async, K2 sync, selected. Every request had
zero cached prompt tokens, the same target and MTP GGUF SHA, the same request
hash, and the same **service** executable SHA-256
`a0a3deaec132200616e7db2c0a18e3cfd719575e660ce6a7d29af58f6c4d9c04`.
Each trial restarted the service. The selected arm used the original selected
config's model and engine arguments, with proposal tracing enabled in all
three arms. GPU memory is observed total GPU 0 usage, not per-process VRAM.

| Service arm (two runs each) | HTTP first byte, ms | First model token, s | Prefill, token/s | Output, token/s | GPU 0 peak, MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected baseline | 9.5 / 9.9 | 28.53 / 29.74 | 1759.4 / 1687.5 | 76.7 / 73.1 | 17119 / 17119 |
| K2 ordinary inputs | 9.6 / 18.4 | 29.22 / 29.85 | 1733.2 / 1696.5 | 76.6 / 72.8 | 17633 / 17633 |
| K2 pinned async | 19.0 / 19.7 | 29.47 / 29.63 | 1710.4 / 1702.1 | 76.1 / 74.7 | 17633 / 17633 |

The K2 ordinary and async medians were 29.535 versus 29.550 seconds to first
model token, and 74.7 versus 75.4 output token/s. With two runs per arm, this
does **not** establish a service-speed improvement. Both K2 arms used 514 MiB
more observed VRAM than the selected baseline, consistent with the separate
approximately 515 MiB Q8_0 draft token table being placed on GPU 0.
The 100 or 200 token/s goal has not been met.

All six final outputs had the same SHA-256. The benchmark parser returned no
proposal-trace hash because it read the engine log before its trace records
were complete. `audit_50k_proposals.py` recovered the records from retained
raw engine logs, splitting at each new window 1. Selected both runs, async
both runs and K2 ordinary run 2 all had the same 119-window trace, offered
238 and accepted 136 tokens. **K2 ordinary run 1 differed at window 7**:
the first proposal was token 0 instead of 108641. It had 121 windows,
offered 242, accepted 134, with no reported K2 fallback. Its final output
still matched. `50k-256/proposal-audit.json` preserves hashes and the first
different records. The cause has not been established; this long-context
proposal difference needs a focused follow-up, and full 50K proposal parity
must not be claimed.

The test runner restored the selected 8096 config after its final run. At
handoff, `/v1/models` reported loaded and `n_ctx=262144` with server PID
39108 and engine PID 42276; `50k-256/restored.json` records the original
config SHA and verifies the child engine environment. The parent test task
then took control of GPU 0 and port 8096 for its next experiment.
