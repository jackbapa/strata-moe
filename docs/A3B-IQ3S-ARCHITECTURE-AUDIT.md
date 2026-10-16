# Strata architecture audit for A3B IQ3_S

The original Strata GPU kernels can be reused selectively, but its Flash-Next
execution graph cannot run A3B unchanged. The current A3B implementation already
replays GGML CUDA graphs and can execute the original Strata grouped IQ kernels.
The clearest remaining GPU candidate is an exact Q6_K dense kernel that reuses
decoded weights across verification columns. This is a proposed adaptation of
Strata's load/apply design, not a measured speed improvement.

This audit is dated 15 October 2026 and covers the source before that new Q6
experiment. Only this document was added. `G:/Strata` remains read-only. The
related [execution record](G:/Strata-Moe/docs/A3B-STRATA-IQ3S-EXECUTION.md) gives
the service history; the tables below distinguish source reuse, measured use,
and proposed work. Earlier profile requirements in
[A3B-Q6-NEXT.md](G:/Strata-Moe/docs/A3B-Q6-NEXT.md) predate the newer 50K node
traces cited here.

## Model and measurement scope

The target is one downloaded Qwen3.6-35B-A3B UD-IQ3_S GGUF, SHA-256
`66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66`.
Its expert gate/up tensors use IQ2_S, its down tensors IQ3_S or IQ4_XS, and
its dense tensors include Q6_K. These are tensors in the same mixed GGUF.
The matching MTP GGUF is Q8_0, SHA-256
`f6cb995fab78e7284097b2e13508e8daad4ffc6a747b433aef399f4e48524b5a`.
The draft proposal's measured `mul_mat_vec_q` kernels are GGML type 8
(Q8_0), rather than Q6_K type 14. Attribute quantization from the kernel
template, not only a model filename or generic `mul_mat_vec_q` name.

The relevant measurements use Windows, GPU 0 RTX 2080 Ti, sm_75, 22,528 MiB,
an exact uncached 50,000-token prompt, 256 output tokens and MTP K2. The F16
tier experiments advertise 262,144 context tokens and allocate 65,536 physical
KV cells for that request. Each cited result records its own executable,
configuration, request and output identities. Results from different
executables or sequential test groups are not a single controlled comparison.

Original model geometry is in
`G:/Strata/include/strata/core/layout.hpp:27`: 48 layers, hidden 2560,
intermediate 640, 512 experts, HC4, GDN and QSA. Its native router is top-10
(`G:/Strata/include/strata/kernels/native_router.hpp:10`). A3B uses 40 layers,
2048/512, 256 experts and top-8; its Qwen model builder marks full attention
every fourth layer and GDN elsewhere
(`G:/Strata-Moe/third_party/llama.cpp/src/models/qwen35moe.cpp:13`).

## What actually executes

| Technique | Evidence and current scope | Port status |
|---|---|---|
| GPU resident expert weights | IQ3_S native-GPU experiments report zero CPU expert rows and zero expert upload MiB. `G:/Strata-Moe/a3b/main.cpp:857` places model layers on one CUDA device. | Used. This placement is the GGML A3B path, not the original RAM/cache scheduler. |
| Target and draft CUDA graphs | `G:/Strata-Moe/third_party/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4518` launches graphs; :4620 begins capture. The 50K graph trace records 367 launches in output, including 111 target-verification and 224 MTP-proposal launches. | Used through GGML. Do not describe the model as missing CUDA graphs. |
| Original Strata grouped IQ expert kernels | `G:/Strata-Moe/a3b/CMakeLists.txt:32` compiles `src/kernels/cuda/iq_kernels.cu`; `a3b/strata_native_experts.cu:127` directly calls `strata::kernels::native_expert_grouped`. A3B uses separate resident gate/up/down tensor pointers and CUDA route grouping. | Ported and actually used in 50K opt-in arms: 5,070 calls per request in the Q8-KV comparison and 5,116 in the later F16 comparison. A config without `--strata-native-experts` does not select it. |
| Original fused expert gate/up | `STRATA_A3B_FUSED_GU=1` adapts the original gate/up, SwiGLU and intermediate-Q8_1 kernel to separate IQ2_S tensors and A3B geometry. | Ported, with real-weight bitwise parity. Isolated expert calls improved 26.5-30.3%; the older full-engine series did not establish an isolated gain. |
| Original prefill IQ MMQ wrapper | The same CMake target compiles `src/prefill/moe_mmq.cu`; `a3b/strata_prefill_mmq_adapter.cu:55` checks A3B IQ types and T9-2048. | Ported and used: 1,000 calls and 16 million routed rows per 50K request. The original wrapper itself uses the pinned GGML MMQ mathematics. It is not a new tensor-core implementation or the full Strata prefill scheduler. |
| Q6 exact grouped dense verification | `G:/Strata-Moe/a3b/backend.cpp:2038` selects ordinary GGML grouped matvec; `ggml-cuda/mmvq.cu:527` preserves the Turing two-warp reduction for 2-4 columns. | Used in the measured configs. This is a local GGML adaptation, not a call to Strata's native Q6 kernel. |
| Quantized CPU expert pool | Original `G:/Strata/src/kernels/cpu/pool.cpp:620` calls actual expert math; `expert.cpp:419` computes gate/up dot products and :429 down. A3B `a3b/cpu_experts.cpp:289` has quantized CPU type-trait dot products. | Ported in A3B, unused in the resident IQ3_S runs. The original CPU pool is a compute pool, not only a copy/scheduling pool. |
| Mapped CPU/GPU expert mailbox | Original `G:/Strata/src/core/verify.cpp:1596` waits for plan A, DMA B and CPU rows M while GPU experts execute. The A3B Q8 branch implements a persistent service and mapped mailbox. | Q8 port exists; mixed-IQ mailbox integration remains unported. A3B `ggml-cuda/strata-moe.cu:344` explicitly requires Q8_0 weights. |
| Host stager, independent DMA issuer and ring | Original `G:/Strata/src/prefill/prefill.cpp:406` owns host copy workers; :2402 starts an issuer. A3B `a3b/backend.cpp:975` has per-task copy-ahead and optional independent issuer. | Partially ported for offloaded weights, not the original all-layer adaptive schedule. Resident IQ3_S weights do not need these uploads. |
| Prefill CPU share | Original `G:/Strata/src/prefill/prefill.cpp:3133` adjusts share from measured CPU/GPU time. A3B `a3b/backend.cpp:1248` assigns complete low-fanout missed experts to CPU. | An adapted opt-in exists. Its policy is not the original measured adaptive policy; it is unused in these resident runs. |
| MTP prefill GPU hidden bridge and K2 device chain | `G:/Strata-Moe/a3b/speculative.cpp:256` and :258 select these opt-ins. The bridge moved all 25 prompt chunks on GPU in the fixed 50K test. K2 connects two independent GGML decode submissions with GPU buffers. | Implemented and exercised. The old K2 `zero cross-step CUDA graph replays` field was hard-coded, not measured. A later 128-cell strict K2 Nsight fixture observed 49 actual graph launches in 26 chains and 624 stream synchronizations; see `G:/Strata-Moe/bench/results/2026-10-15-a3b-mtp-k2-graph-audit/README.md`. K2 is still not the original round/step/commit protocol, and this short trace is not 50K speed evidence. |
| Original MTP round/step and commit scheduling | `G:/Strata/src/core/mtp.cpp:1451` submits a round graph followed by step graphs. `src/core/verify.cpp:2508` has a separate target commit graph. | Requires A3B-specific state and graph integration. Existing individual GGML graphs do not implement this host/control protocol. |
| Shared expert side stream | Original `G:/Strata/src/core/verify.cpp:1424` forks shared FFN and :1541 records its join event. | Not ported as the original scheduling path. Same-card bandwidth and SM contention need measurement before assuming a benefit for fully resident A3B. |
| Fused GDN | `G:/Strata-Moe/third_party/llama.cpp/src/models/qwen35moe.cpp:387` selects the A3B-adapted single-token GDN kernel. A3B uses a different state layout and output gate from the original. | Adapted and parity-tested. It matched zero graphs in the later batched-MTP combined 50K arm, so that arm cannot establish a GDN execution gain. |
| KV formats and capacity tiers | A3B `a3b/main.cpp:875` selects GGML Q8 KV; :667 implements idle context reconstruction in 64K tiers, including an F16-short/Q8-long option. | Used in respective experiments. Tier reconstruction is not the original CUDA VMM page-growth implementation. |
| QSA selected-KV, HC4 and PLE | Original `G:/Strata/src/core/layer.cpp:503` handles QSA and :1332 PLE. A3B's full-attention/GDN graph has neither those operators nor their tensors. | Not applicable as unchanged model math. Their memory-management ideas require a separate A3B design. |

The 50K graph-level SQLite was reopened read-only to check the API counts
above. Its 111 target graph executions occupy 2.647 seconds of a 3.920-second
profiled output interval. The following 114 hidden-observation waits spend
about 2.705 seconds in stream synchronization waiting for target GPU work.
That time is not CPU expert computation. See
[graph analysis](G:/Strata-Moe/bench/results/2026-10-15-a3b-iq3s-nsys-output/ANALYSIS.md).

## Measurements that constrain the next change

The newer F16-tier plus native-IQ node trace covers only 61 complete target
windows after a delayed profiler start. Its kernel sums are not exclusive
wall time or an unprofiled speed benchmark. They identify work inside the
already replayed graphs:

| Captured work | Target kernel time | Draft proposal kernel time |
|---|---:|---:|
| Q6_K dense matvec (type 14) | 266.029 ms | 0 ms |
| Q8_0 dense matvec (type 8) | Included in other target work | 130.393 ms |
| Original IQ gate/up and down | 300.1 ms | Included in other work |
| F16 attention | 155.7 ms | 23.5 ms |
| All kernels | 1,097.5 ms | 187.1 ms |

Q6_K accounts for 24.2% of the target kernel sum, while Q8_0 accounts for
69.7% of the draft proposal kernel sum. Q6_K accounts for zero draft proposal
kernel time. The earlier draft Q6 label came from conflating GGML type 8 with
type 14; the corrected runtime-correlation SQL, unique whole-trace Q6 count
and raw SQLite are in
[F16 native node analysis](G:/Strata-Moe/bench/results/2026-10-15-a3b-iq3s-f16-native-node/ANALYSIS.md).
These percentages justify separate dense-kernel experiments but do not
predict an end-to-end gain.

A second read-only grouping of the Q6 target kernels by launch dimensions
shows why a head-only rewrite is too narrow. Each launch has two 32-thread
warps and the exact 3-column specialization. The output head is the
124,160-block grid, corresponding to 248,320 rows at two rows per block:

| Q6 target gridX | Calls | Kernel sum |
|---:|---:|---:|
| 4,096 | 2,440 | 82.187 ms |
| 1,024 | 4,880 | 62.463 ms |
| 124,160 | 61 | 50.186 ms |
| 2,048 | 1,830 | 40.049 ms |
| 256 | 6,100 | 31.143 ms |

The head is 18.9% of this capture's Q6 target sum. Most measured Q6 time is
in trunk/shared dense matrices. The unprofiled two-run F16 plus native-IQ
arm measured 79.35 output tok/s, 28.252 seconds to first model token,
15.32 ms to first HTTP byte, 1,782.1 prefill tok/s and 17,119 MiB sampled
whole-card peak. This sequential comparison has two repeats, and native IQ
changes the answer and MTP acceptance relative to plain F16. See the
[same-executable record](G:/Strata-Moe/bench/results/2026-10-15-a3b-iq3s-f16-auto-ab/README.md).

## Failed or unselected routes

These records describe the tested settings, not permanent limits on the
underlying techniques. The configurations remain separate opt-ins or the
prototype was removed; no negative result should be silently replaced by a
later unrelated measurement.

| Trial | Measured result | Decision and scope |
|---|---|---|
| Direct original Q6 native MMVQ substitution | For 2048-by-2048 matrices with 1/2/3 columns, 1,150/2,048, 2,238/4,096 and 3,158/6,144 values differed bitwise; maximum absolute difference 1.91e-6. | Prototype removed because it violates the sequential-target exact contract. Small numerical error is not bitwise parity. Source: `G:/Strata-Moe/bench/results/2026-10-15-a3b-iq3s-fused-gu/README.md`. |
| Q8 KV direct-vector verification | Two-run output median 64.9 to 43.8 tok/s; TTFT 28.533 to 28.529 seconds; first HTTP byte 14.52 to 19.68 ms. Answer and acceptance changed. | Switch remains off. Avoiding full-cache dequantization did not compensate for this unbatched attention route on sm_75 at 50K. Source: `G:/Strata-Moe/bench/results/2026-10-15-a3b-iq3s-q8kv-vec-ab/comparison.json`. |
| Q4_0 KV and MTP K4 | Earlier single 50K runs: Q4 KV 52.8 output tok/s; MTP K4 41.8 tok/s, versus adjacent grouped-Q6 K2 rates around 62.6-62.9. | Not selected. Different settings and answers are not isolated kernel comparisons. Source: `G:/Strata-Moe/bench/results/2026-10-15-a3b-iq3s-q6-grouped/README.md`. |
| Exact Q8 dense reuse added to grouped Q6 | Two combined runs gave 67.1 and 63.1 tok/s; an intervening Q6-only run gave 64.5. Answers matched. | No established incremental Q8-reuse gain; selected config left it off. Same Q6 record. |
| All optional ports added to F16 plus native IQ | Native-only median 79.35 versus combined 77.65 tok/s; same answer and 136/238 acceptance. Combined peak rose 534 MiB. It executed 1,000 MMQ calls, 25 bridge calls and 119 K2 calls with zero fallbacks. | Combined route not selected as faster. This does not isolate which optional component caused the difference. GDN matched zero graphs. Source: F16 automatic-tier record above. |
| Target batch 4096 added to F16 plus native IQ | 77.45 versus 79.35 output tok/s; first model token 27.228 versus 28.252 seconds; prefill 1,848.2 versus 1,782.1 tok/s; peak +692 MiB. Answer and acceptance changed. | Kept as a first-token/prefill tradeoff, not the preferred output route. Same record. |
| Initial MTP hidden bridge and initial K2 chain | Earlier attempts failed respectively with lost CUDA hidden input and a CPU GET_ROWS index assertion. | Both were subsequently fixed and completed 50K tests; do not describe the historical failures as current failures. Fixed bridge and K2 result folders retain both scope and performance limits. |

The earlier fused-IQ expert microbenchmark improved complete expert calls
by 26.5-30.3%, with bitwise parity against the original split layout. Its
six-run API series slowed in both arms over time and did not establish a
stable service-level gain. This is an unselected timing experiment, not a
numerical rejection of that fused kernel.

## Proposed exact Q6 GPU adaptation

Original Strata explicitly separates the weight-only part of a dot product
from its activation-dependent part:

- `G:/Strata/src/kernels/cuda/native_mmvq.cu:970` defines
  `Q6KTraits::load` and `apply`.
- `native_mmvq_multi_kernel`, :1214, loads one weight payload per row/block
  before its column loop and calls `apply` for each column.
- `native_mmvq_il_kernel`, :1455, reads interleaved Q8_1 columns, processes
  several output rows per warp and simulates the original four-warp
  accumulation with separate accumulators.

Current A3B exact grouped Q6 keeps GGML's two-warp partition, but
`G:/Strata-Moe/third_party/llama.cpp/ggml/src/ggml-cuda/mmvq.cu:728` calls
the full dot routine once per column. That routine,
`vec_dot_q6_K_q8_1` in `vecdotq.cuh:1019`, includes ql/qh, scale and block
scale reads before calling the dot helper. There is no explicit Q6-specific
load/apply interface there. The compiler may already remove some repeated
loads or expressions; source structure alone does not establish redundant
machine instructions or a speed gain.

The smallest experiment should move only column-independent Q6 payload
work outside the column loop. Keep the existing GGML Q8_1 quantizer,
`vec_dot_q6_K_q8_1_impl_mmvq` expression, kbx/iqs ownership, per-thread
accumulation order, two-warp partial addition and XOR reduction. Keep
existing tensor strides and ordinary graph scheduling. Limit dispatch to
sm_75, Q6_K, no expert IDs/fusion, and T2/3/4 initially; default off.

If that step has a useful measured gain, a second experiment can adapt the
interleaved design to two virtual warps. Map each K chunk to the same
physical thread's logical partition as the current two-warp kernel, retain
two independent sums, add them in the current order and use the same
reduction tree. Start with R1, then measure R2/R4 output rows per warp.
Interleave already quantized Q8_1 bytes rather than change quantization.
The original four-virtual-warp loop cannot be copied unchanged because the
current exact contract uses two warp partitions. The earlier direct Q6
substitution already failed bitwise comparison. Original
`il_arch_ok` at :2690 does not enable the original interleaved route for
sm_75, so its tuning tables are not evidence of a 2080 Ti gain.

Original Q6 planes packing is a separate experiment. The comment at
`G:/Strata/src/kernels/cuda/native_mmvq.cu:1812` targets 2560-wide rows of
2100 bytes with poor word alignment. A3B's 2048-wide rows occupy 1680
bytes, already a multiple of 16. A packed copy might still change access
costs, but the original alignment argument does not establish a benefit
here, and it adds VRAM. The fixed original 640-block persistent grid also
requires local measurement rather than reuse as a tuning constant.

## Interface and validation requirements

Strata's native MMVQ functions take raw weight pointers, prequantized Q8_1
activation pointers, explicit ncols and a CUDA stream. Its verifier owns
stable per-layer scratch and decides when to quantize, interleave, fork and
join. In contrast, current A3B passes ordinary GGML tensors through
`build_native_exact_mm`; CUDA dispatch quantizes the input into backend
scratch and GGML owns allocation, shape-specific capture and replay. A
source-level kernel copy does not transfer that scratch ownership or
producer/consumer schedule.

For the first Q6 experiment, staying inside the current MMVQ dispatch
preserves this interface and avoids a new backend split or synchronization.
Any later shared Q8_1 producer needs a graph-visible dependency and scratch
allocated before capture. A cache keyed only by an input address is unsafe
because buffers are reused across layers and graph replays. Noncontiguous
inputs, padding, capture lifetimes, context rebuilds and fallback behavior
must remain explicit.

GGML already has gate/up/GLU fusion in
`G:/Strata-Moe/third_party/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4061`.
It should not be counted as a Strata transplant. Its matcher depends on
tensor identity and node adjacency. The direct grouped-Q6 branch may
already match some shared FFN cases; other exact reshape paths can disturb
matching. Confirm actual fused-node/kernel use before proposing another
gate/up fusion. Strata's existing `native_mmvq_pair` at
`G:/Strata/src/kernels/cuda/native_mmvq.cu:2417` supports Q8_0, not a ready
Q6 shared-expert pair.

Validation should first compare real Q6 tensors for every relevant shape,
T1-4, contiguous and strided inputs, against separate singleton GGML CUDA
products. Require bitwise equality, including output head and shared FFN,
before MTP strict and complete-model checks. Compare opt-in off/on with one
new executable, fixed model/config/request and alternating repeats. Record
output and proposal hashes, acceptance, first HTTP byte, first model token,
prefill/output rates and whole-card peak VRAM. Use CUDA-event microbenchmarks
and a node trace to attribute changes; do not infer service speed from a
kernel sum. No existing result establishes 100 output tok/s.

Source identities at audit time:

| Source | SHA-256 |
|---|---|
| Original `G:/Strata/src/kernels/cuda/native_mmvq.cu` | `6d367b1c673524ba9237049e00d63eecd8e9e7f0749603c2a8af4d5302f02528` |
| Original `G:/Strata/src/core/verify.cpp` | `0c2762ecd4fa61908019b3d60fc7effa6b72e626a61d9c2abc849f1e1000df0f` |
| Clone `G:/Strata-Moe/third_party/llama.cpp/ggml/src/ggml-cuda/mmvq.cu` | `2ff79da08741a9b19c62ecd63f2e90785b33188eb09c42f21e01707f8adc3d94` |
| Clone `G:/Strata-Moe/a3b/backend.cpp` | `17ac388bb10fea8e038a0c2178d91c9c0f1d906005583e84ab87ba6bd54ca2ff` |
