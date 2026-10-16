# Strata to A3B: source-level status and remaining performance gap

**Historical Q8_0 branch of the work.** This page describes the earlier
full-Q8 expert-cache experiment and its measurements. It does not describe
the currently selected IQ3_S service. For the actual IQ3_S code path, copied
Strata kernel calls, remaining architecture gaps and 50K/256 A/B results,
see [A3B-STRATA-IQ3S-EXECUTION.md](A3B-STRATA-IQ3S-EXECUTION.md) and the
[same-executable five-arm record](../bench/results/2026-10-15-a3b-iq3s-f16-auto-ab/README.md).

The target is a complete Qwen3.5/3.6-35B-A3B implementation in this separate
`Strata-Moe` checkout, using the original Strata execution techniques where
they apply. The original `G:/Strata` checkout is read-only for this work.
Current A3B support runs the official complete Q8_0 GGUF and matching MTP
model on GPU 0 with 262144 context. It does **not** yet meet 100 output
tokens/s or 1500 prefill tokens/s on the local RTX 2080 Ti.

## What the original engine does

`G:/Strata/src/prefill/prefill.cpp` forms a sequence of expert blobs spanning
all layers for each long prompt chunk. Its `Stager` makes pageable blobs
available in pinned memory; a separate issuer thread queues host-to-device
copies into an event-protected ring while layer kernels execute. Expert
products use up to 16 experts per MMQ group, grouping routed rows by expert.
Gate/up share a quantized activation, then SwiGLU runs before another MMQ
product for down. `G:/Strata/src/prefill/moe_mmq.cu` explicitly supports
GGUF Q8_0. For decode, the original CPU expert pool really computes some
nonresident expert matrix products in parallel with GPU work; it is not just
a data-movement thread.

## What the A3B engine currently ports

The A3B loader and trunk are based on the Qwen3.5-MoE model implementation in
the pinned llama.cpp source. The A3B expert backend keeps complete Q8 weights
in host RAM, a bounded GPU cache, and bounded upload banks. Long-prefill
experts execute on GPU. The hybrid short-window path has a CPU expert pool
and GPU doorbell similar in scheduling intent to Strata; MTP K2 is enabled
and target-verified. The opt-in `--prefill-copy-ahead` now ports the original
pageable-to-pinned host stager for 1024+ token chunks. A same-binary 50K
comparison is recorded in
`bench/results/2026-10-15-a3b-q8-strata-stager/README.md`.

The selected long-prefill GPU task uses three GGML `mul_mat_id` products and
SwiGLU. Inspection of the pinned `ggml-cuda.cu` dispatch shows that its
`[hidden, 1, routed_rows]` input enters the quantized MMQ path when enough
rows are present. Therefore the current bottleneck is not simply a missing
MMQ switch. An experimental `--prefill-independent-issuer` now moves each
layer's H2D submissions to a separate host thread. An additional opt-in
`--prefill-packed-gate-up` uploads consecutive Q8 gate/up rows and uses one
GGML MMQ product for both, sharing input quantization. These are not the
original 16-expert grouped CUDA MMQ, large ring, or all-layer copy sequence.
The latter remain unported. A same-prompt 50K trial and an additional
GPU-stream-wait issuer trial showed no clear latency gain for the two new
opt-ins, so the selected service leaves them disabled.
The packed path is numerically close in a 4.19-million-value fixture but
changes the deterministic 256-token generation on the full model. See
`bench/results/2026-10-15-a3b-q8-packed-gu/README.md`.

## Why 3B active parameters do not guarantee 100 tokens/s here

The IQ3_S experiment now also ports the original `native_expert_grouped`
CUDA kernel to A3B's 2048/512 expert geometry and split GGUF tensors. Its
short decode/MTP routes are grouped on CUDA and run within the normal CUDA
graph. This is opt-in and does not change the Q8 prefill/copy pipeline
described above. The [50K paired record](../bench/results/2026-10-15-a3b-iq3s-strata-native/README.md)
reports 57.9 versus 56.1 output tokens/s in one run each; it does not
establish a 100- or 200-token/s implementation.

A Q8_0 expert has three 2048-by-512 matrices. A Q8_0 block stores 32 values
in 34 bytes, so one complete expert is 3,342,336 bytes (3.1875 MiB).
Forty layers with 256 experts hold 31.875 GiB of routed Q8 weights. One
token selecting eight experts in each of forty layers refers to nearly
0.996 GiB of expert weights before cache reuse. The GPU has about 21 GiB
usable VRAM and the measured expert cache is about 9.54 GiB. CPU/GPU
placement, locality, PCIe transfer and CPU miss completion thus directly
affect decode latency; the smaller active parameter count alone does not
remove the memory cost. The original model's quoted speed must be compared
using its actual quantization, hardware, input length and context size.

The existing Nsight decode capture under
`bench/results/2026-10-15-a3b-q8-doorbell-profile/` observed about 947 ms
of GPU kernel time in a 27-token window with profiler overhead. GPU dense Q8
products accounted for about 218 ms, while `mailbox_wait` and merge kernels
accounted for about 181 and 91 ms. These are **kernel sums**, not independent
wall-clock penalties; the profile excludes startup and should not be used as
a throughput benchmark. It does identify two concrete decode targets:
reducing dense trunk work and hiding or shrinking CPU expert completion.

The 50K Q8 prefill moved about 333 GiB of expert weights despite the GPU
cache. Copy-ahead raised measured prefill to 449.8 tokens/s in the first
paired run, but output stayed 39.4 tokens/s. A blind all-expert copy plan
would transfer up to 256 rather than about 110 distinct routed experts per
layer/chunk before cache filtering; porting the issuer must therefore be
route-aware or prove with measurement that extra copies are worthwhile.

## Required work before calling this a full Strata port

1. Port the original grouped Q8 gate/up/down MMQ execution to A3B tensor
   geometry and validate prefill logits and outputs against the existing
   full-Q8 path across short and long chunks. The packed GGML experiment is
   not a substitute for the original grouped kernel.
2. Extend the tested per-layer issuer to a cross-layer schedule with
   event-protected reuse and a deeper ring whose bank count is independent
   of expert compute group size. Measure the extra copied bytes, GPU wait,
   pinned staging, and 50K time together.
3. Profile decode without profiler-induced timing claims, then optimize the
   two measured targets above. Keep the CPU/GPU split configurable; do not
   assert that moving all misses to either device is faster without a paired
   test.
4. Run the final exact binary with the official Q8 + MTP pair at 256K
   context and a counted 50K prompt, reporting first model token, prefill,
   output rate, output parity, peak RAM and VRAM. Repeat enough times to
   explain the observed fast/slow second-request variation.

The local selected Q8 service configuration now enables the measured
copy-ahead option. Its earlier configuration is saved as
`strata-a3b-q8-pre-stager.json`, and the engine default remains off.
CUDA and CPU builds have been checked; HIP and SYCL have not been built on
this NVIDIA Windows host.
