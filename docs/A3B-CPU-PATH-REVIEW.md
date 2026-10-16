# A3B CPU/GPU expert path: source review

Read-only review on 2026-10-14. No GPU run, CPU expert benchmark, service request
or production configuration change was performed for this review. The original
`G:\Strata` tree was only read. The isolated static/dynamic API experiments are
prepared under `bench/results/2026-10-14-a3b-q8-static-api` and are paused.

The original Strata does compute RAM experts on the CPU during decode. GPU cache
hits run on the GPU. It can assign a fraction of RAM misses to the GPU over PCIe
as well. Its fast path combines that placement with a persistent CPU pool,
shared quantized activations, grouped speculative windows and captured GPU
graphs. Switching an A3B flag to CPU misses alone does not reproduce that design.

## Confirmed differences in the reviewed baseline

| Area | Original Strata source | Legacy A3B source reviewed | Consequence to investigate |
|---|---|---|---|
| CPU activation sharing | `src/core/expert_source.cpp`, `expert_dispatch_multi`: quantize once per token, pass the same activation pointers to its missed experts | `a3b/backend.cpp`, `Task::run`: copy the token activation into each missed expert row; separate gate/up `ggml_mul_mat_id` nodes consume these expanded rows | Repeated copies and activation conversion for the same token; a direct shared Q8 activation path could remove them |
| CPU gate/up work | `ExpertPool::run_split_multi_native` uses one gate/up row phase across experts, followed by intermediate quantization and down phase | Each generic graph runs gate, up, SwiGLU and down as distinct GGML nodes | Extra row mapping, node/barrier overhead; fuse the CPU expert task before changing arithmetic |
| CPU scheduling | `include/strata/kernels/cpu/pool.hpp`: workers use separate physical cores, host placement is explicit, and the host drains tasks too | A3B already owns a persistent `ggml_threadpool`; it takes default affinity parameters | It does not create threads per layer, but does not ensure physical-core placement or reserve a host core |
| CPU/GPU handoff | `src/core/session.cpp`, `session_run_token`: launch a captured token graph; mapped sequence/flag doorbells hand off each layer; one final stream join | `Backend::compute_fused` reads GPU route IDs, copies activation to a temporary host vector, then returns CPU output with GPU scatter and synchronization | More ordinary API calls and layer boundaries; measure route wait, copies, pool work and return separately |
| Parallel work | Original hit/miss plan sends VRAM hits and selected PCIe misses to GPU while the CPU pool handles other misses | A3B launches GPU-hit work before CPU misses and joins it afterwards | Some overlap already exists. There is currently no measured automatic split of missed experts between CPU and transient GPU staging |
| Profile | Original cache starts from a global trained profile and can adapt | A3B has verified Q8 global profiles, but the CLI currently requires GPU-only mode for them | A hybrid comparison cannot silently reuse the production profile with its existing CLI constraints |

Relevant A3B anchors at review time: `Task` near `a3b/backend.cpp:335`, CPU pool
creation near `:576`, route read near `:620`, fused split near `:874`, CPU result
return near `:516`, CLI dependencies in `a3b/main.cpp:157` and profile validation
near `:168`. These may move as the hybrid implementation develops.

There are two important limits to the comparison:

- GGML CPU `MUL_MAT_ID` already groups rows by expert and divides work into row
  chunks (`third_party/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c`,
  `ggml_compute_forward_mul_mat_id`). It is inaccurate to say the existing A3B
  CPU path has no expert grouping. Each gate/up node nevertheless independently
  quantizes the expanded activations, builds its row map and reaches its barrier.
- Original Strata's specialized multi-token IQ kernels are not Q8 gate/up
  kernels. `native_gu_rows` falls back to GGML dot products for Q8_0. An opt-in
  AVX2 Q8_0 down-row path exists, but that alone does not prove an end-to-end
  gain for this model. Its published IQ/Q2 speed cannot be assigned to Q8 A3B.

## Adaptation boundaries

Original CPU pool geometry is fixed at hidden 2560, intermediate 640, experts
512. A3B requires hidden 2048, intermediate 512, experts 256, top 8. The pool's
scratch sizes and row phase bounds must use the A3B dimensions; copying the
original pool unchanged is incorrect.

The Q8 A3B routed expert bundle is 3,342,336 bytes, with all three projections
Q8_0. Keep the official model and strict target/MTP shared-tensor contract. This
review gives no reason to requantize it. Repeated experts in a verification
window should reuse their immutable weight rows across its token activations.

At review time, `--expert-profile`, `--async-expert-events` and positive
`--cache-decay-interval` require `--gpu-only-experts`. Native dense/attention exact
options also exclude the ordinary hybrid mode. `backend.native_exact_verify`
is enabled only for native GPU or GPU-only modes. A new hybrid mode must make
those compatibility choices explicit and validate them; removing GPU-only from
the frozen production JSON is not a valid configuration migration.

CPU and CUDA reductions can round differently. A useful hybrid MTP correctness
test compares verification, rollback and accepted prefixes with a sequential
reference using the same CPU/GPU placement and arithmetic contract. GPU oracle
tests alone cannot certify the CPU expert implementation or a policy that moves
experts between arithmetic paths. Invalid routes and non-finite outputs must
still fail safely.

## Measurement order

First establish a genuine hybrid baseline using the same model, prompts, output
limits, MTP settings and resource accounting as the frozen Q8 report. State any
profile, cache or math-path difference. Record CPU expert rows, GPU hit/miss rows,
distinct experts, expert bytes, CPU pool wall time, routing wait, activation
conversion, output copy/join and MTP accepted drafts. Thread counts should be
measured on the actual 3960X instead of assuming all 24 cores are fastest.

Then evaluate persistent buffers and activation sharing, row phases, physical
core placement, and a measured CPU/PCIe miss split. Captured GPU/doorbell handoff
is a larger architectural step after those costs are separated. Do not infer
that the pool hides all CPU work: the original cache header itself documents a
mostly exposed CPU dependency chain and uses GPU residency to reduce those bytes.

Every end-to-end result still needs TTFB, first output token latency, prefill
throughput, output throughput, RAM/VRAM and MTP acceptance. No speed gain or
100 token/s achievement is established by this source review.

## Opt-in hybrid implementation in progress

The new `--hybrid-experts` selects GPU cache hits with CPU expert misses and
enables fused expert graphs and GPU embedding. It is mutually exclusive with
`--native-gpu` and `--gpu-only-experts`. Hybrid mode now accepts the verified
expert profile, asynchronous events, cache decay and exact GPU dense/attention
options, addressing the earlier CLI limitations described above. The optional
`--cpu-expert-pool` requires hybrid mode and selects the shared-activation Q8 CPU
pool. It does not silently replace the legacy CPU graph path.

The agreed PCIe split interface is `--pcie-frac F`, finite `0 <= F <= 1`, only
valid when explicitly selecting hybrid mode. Zero keeps all misses on the CPU;
one assigns all misses to GPU staging. The fraction applies to distinct missed
experts in a window: all jobs for the same expert have the same destination.
This differs from splitting individual routed token rows. The Python config
generator omits the argument unless explicitly supplied, so legacy args and the
hybrid default remain unchanged. This is an explicit split, without a measured
automatic balancing policy yet.

Work is divided as follows:

- The root agent owns engine CLI/speculative integration, isolated benchmark
  configurations, builds and GPU correctness/performance validation.
- The backend agent owns concurrent CPU/GPU dispatch, cache consistency and
  the distinct-expert PCIe split with bounded GPU staging.
- The original CPU path agent owns the Q8 shared-activation CPU pool and its
  arithmetic/geometry adaptation.
- The latency/report agent owns the Python configuration interface, offline
  option tests, timing/report methodology and this source review.

At this update, Python config option tests pass offline. The new hybrid
implementation, CPU pool and PCIe split still need coordinated build, real Q8
MTP verification, baseline comparison and latency/resource measurement. The
existing frozen Q4/Q8 executables, production configs and reports are retained;
no new speed or memory result should be inferred from adding these flags.
