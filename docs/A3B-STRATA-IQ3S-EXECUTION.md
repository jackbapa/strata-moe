# Strata execution path for the A3B IQ3_S model

This document distinguishes the source technique, code that exists in
`G:/Strata-Moe`, and code that the measured service actually executes. The
original `G:/Strata` checkout is read-only. The target is the single
Qwen3.6-35B-A3B UD-IQ3_S mixed-quantization GGUF. Its Q6_K dense tensors are
part of that same file, not another model.

The two models share the broad MoE idea, but their execution graphs differ.
Strata's Flash-Next verifier in `G:/Strata/src/core/verify.cpp` is fixed to its
48-layer, 2560-hidden/640-FFN, 512-expert/top-10 graph, HC4 residual mixing,
QSA and PLE. The A3B target has 40 layers, 2048-hidden/512-FFN, 256
experts/top-8, full attention at every fourth layer and GDN elsewhere. A
Flash-Next verifier graph cannot be relabeled as an A3B verifier; its
attention, recurrent state and residual operations must be implemented for
this model. The current A3B trunk is still built by the pinned Qwen3.5-MoE
llama.cpp model code.

| Strata technique | A3B IQ3_S state on the measured RTX 2080 Ti | Consequence |
|---|---|---|
| RAM expert store, adaptive VRAM cache, CPU expert computation and PCIe copy ring | All IQ3_S target experts fit on GPU 0. The measured 50K service reports zero CPU expert rows and zero expert upload MiB. The A3B CPU quantized expert math exists, but the mapped GPU mailbox still accepts only Q8_0; the full original mixed-IQ offload policy is not ported. | The original CPU pool really computes nonresident expert gate/up/down products (`G:/Strata/src/kernels/cpu/pool.cpp`, `expert.cpp`). These mechanisms have no work in this resident IQ3_S path; forcing a CPU miss or an H2D copy would add work. |
| Grouped quantized gate/up/down expert kernels | `--strata-native-experts` calls the original `native_expert_grouped` CUDA implementation with A3B shapes and GGUF weight pointers for short decode and MTP verification. It is opt-in. | The measured 50K/256 request executed 5,070 Strata calls in decode, zero in prefill. |
| Long-prompt grouped MMQ prefill | The opt-in `--strata-prefill-mmq` adapter now calls the copied original Strata grouped IQ MMQ wrapper from the A3B service. The wrapper itself uses the pinned llama.cpp GGML CUDA MMQ mathematics. The 50K request logged 1,000 actual CUDA calls for 16 million routed rows. | This adapter keeps GGML's A3B routing graph; it is not Strata's cross-layer prefill scheduler or its sm_80 fused IQ prefill kernel. Two sequential 50K runs did not establish a prefill speed gain and sampled whole-card VRAM rose about 252 MiB. |
| Complete target verify and MTP draft CUDA graphs | A 50K/256 Nsight trace confirms 111 `cudaGraphLaunch` calls in 114 target-verification windows and 224 in MTP proposals. Thus GGML **does** replay target and draft model graphs. The normal A3B draft path reads token/hidden on the host between steps. Experimental K2 connects two independent `llama_decode` submissions through GPU token/hidden buffers. Its old `CUDA cross-step graph replays 0` log field was hard-coded; a later short strict K2 Nsight fixture observed 49 actual launches across 26 chain ranges, with 624 stream synchronizations. | Re-wrapping target verification in a CUDA graph would duplicate existing work. The remaining gap is model-specific expert/state scheduling and original Strata's device-connected MTP round/step/commit protocol; original Strata's QSA/HC4 verify and commit graphs cannot execute A3B GDN/KV math unchanged. |
| Fused recurrent GDN | The original kernel has a different state layout and sigmoid closing gate; this A3B GGUF uses SiLU and F32 alpha/beta weights. `STRATA_A3B_GDN_FUSED_STEP=1` selects an adapted CUDA GDN+RMSNorm+SiLU kernel for exactly matched single-token target graphs. | The original kernel cannot be called unchanged. The adapted path passed target-graph parity tests below. Multi-token prefill and MTP verification retain GGML GDN. The two-run 50K service comparison below did not establish an isolated gain. |
| KV growth/streaming | The fixed-Q8 baseline allocates its full 262,144-cell target and MTP Q8 K/V before requests, about 2,992 MiB total. The opt-in tiered route reconstructs idle contexts at 65K/131K/256K while keeping a 262,144-token advertised limit. | The tier is context reconstruction between independent requests, not Strata's CUDA VMM `--kv-grow`. F16 through 131K with Q8 at 256K is an additional opt-in; its 50K and long-boundary measurements are below. |
| PLE and prompt lookup drafting | Flash-Next has PLE and a prompt lookup route; this A3B model has no PLE tensor. | PLE cannot be ported as a model-independent accelerator. A3B currently uses its matching Q8_0 MTP head at K2. |

## Latest exact combined test

The [16 October six-run record](../bench/results/2026-10-16-a3b-iq3s-combined-exact-r3/README.md) used the **same engine binary and IQ3_S/MTP model files** on GPU 0, with uncached 50,000-token input, 256 output tokens, MTP K2 and a configured 262,144-token context. Both arms already used the original Strata grouped IQ expert kernel. The additional fused gate/up and down mode are adaptations of its grouped expert phases; the Q6_K reuse switch is GGML code for a tensor in the **same IQ3_S GGUF**, and the KV-only switch trims unused A3B draft catch-up work. All six outputs and proposal traces matched, and the strict gate matched 16 final state pairs byte for byte.

Three-run medians were **75.2 to 83.2 output tok/s**, **29.385 to 28.575 s** for the first nonempty model token, **1,708.9 to 1,765.0 prompt tok/s**, and **17,119 to 17,103 MiB** of sampled whole-GPU-0 peak. HTTP first byte was 19.25 versus 19.89 ms. The output gain is 10.6% for the **four-switch combination**; it cannot be attributed to the Strata kernel alone or to any single switch. A 16 MiB whole-card VRAM difference is not meaningful memory saving. This is still below the 100 tok/s target. The tested combination is available as the separate opt-in `strata-a3b-iq3s-strata-optimized.json`; the engine defaults remain off.

The resident IQ3_S path does not move experts through the original cross-layer host stager, independent DMA issuer and event-protected ring because its measured expert uploads are zero. `--strata-prefill-mmq` has executed 1,000 adapter calls in a prior 50K test, but was excluded from this selected test after it failed to establish an end-to-end gain and raised observed VRAM. Thus the current service uses Strata expert compute in decode, yet it is **not a complete transplant of the original Strata execution architecture**.

An isolated A3B adaptation of Strata's MTP resident-expert grouping planner
passed 200/200 CUDA route layouts and a real IQ3_S/Q8 draft fixture with eight
matching output classes, 29 identical proposals, 16 byte-identical final
target/draft state pairs and 5,255 actual planner calls per arm. Its
[alternating six-run 50K/256 service comparison](../bench/results/2026-10-16-a3b-iq3s-parallel-plan-ab/exact-ab-report.json)
kept the same executable, models, request, output and MTP proposal trace.
On the RTX 2080 Ti, switching only this planner **reduced** median output
speed from 85.1 to 84.7 tok/s, moved the first nonempty model token from
28.401 to 28.625 s, changed prompt speed from 1,778.6 to 1,766.7 tok/s,
and left sampled whole-GPU peak at 17,103 MiB. HTTP first byte was 20.00
versus 19.47 ms. This planner remains off; a source-level port and exact
state parity did not establish a throughput benefit.

The subsequent [13-run leave-one-out check](../bench/results/2026-10-16-a3b-iq3s-four-switch-ablation/README.md) passed five real-model strict fixtures and retained identical 50K output/proposals across every arm. With the other three switches on, all-on output was 83.0 tok/s over five runs; two-run medians were 82.15 without fused gate/up, 80.7 without 16-lane down, 82.15 without GGML Q6_K reuse, and 79.5 without MTP KV-only. The all-on anchors drifted from 87.5 to about 83 tok/s during the suite, so the report also compares each leave-one-out trial with adjacent anchors. The largest repeatable conditional differences in this small test were from MTP KV-only and Strata down mode; Q6 reuse was the smallest. These are interactions-conditioned observations, not additive per-kernel speedups.

An opt-in A3B draft-vocabulary port now copies 194,208 selected Q8_0 output
rows from the **same matching MTP model** into a derived GGUF; the target
IQ3_S GGUF and its mixed quantization remain unchanged. The new Qwen35MoE
graph currently scatters compact logits back into the full 248,320-token
vector before GPU greedy sampling. The [clean same-binary, alternating
50K/256 A/B](../bench/results/2026-10-16-a3b-draft-vocab-50k-clean-ab/README.md)
found median output **81.9 tok/s full versus 81.7 subset**, first nonempty
model token **28.571 versus 28.663 s**, HTTP first byte **9.34 versus 19.22
ms**, and whole-GPU peak **17,103 versus 16,993 MiB** on the RTX 2080 Ti.
All six output and proposal hashes matched, with 136/238 MTP acceptance.
Output rates drifted across the suite; adjacent-anchor comparisons show no
demonstrated speed gain. The 110 MiB lower peak is useful but not
process-attributed. This subset is research-only and does not change the
selected service. A true compact GPU sampler is being developed separately;
its isolated implementation now has a measured result. With the same derived
I32 draft GGUF in both arms, the [six alternating 50K/256 trials](../bench/results/2026-10-16-a3b-draft-compact-50k-ab/comparison.json)
compared scattering the 194,208 draft logits into the full 248,320-token
vector with directly sampling the compact logits and mapping the full token ID
on GPU. Every arm kept the same target IQ3_S file, output and proposal trace;
the compact arm recorded 238 actual GPU sampled rows per request. Median
output was 85.0 versus 85.4 tok/s, first model token 28.508 versus 28.502 s,
prompt speed 1,768.3 versus 1,763.3 tok/s and whole-GPU peak 16,991 MiB in
both. The 0.4 tok/s difference is below observed run drift and does not
establish a speed gain. This sampler and derived draft remain research-only.

The selected service does not use the original shared-expert side stream
(`G:/Strata/src/core/verify.cpp`) or the device-connected MTP round/step/commit
protocol. The experimental A3B K2 device chain moves token and hidden
data on GPU but submits two separate GGML decode calls. The older `zero
cross-step CUDA graph replays` line was printed as a literal constant and
did not count CUDA graph launches. A short strict K2 Nsight fixture later
measured 49 launches inside 26 chains, not a 50K speed result. The original pipeline across verification
windows requires at least two independent GPUs and is not the single-GPU
execution path here. These gaps should not be described as an absence of CUDA
graphs: the current GGML target and draft graphs are already captured and
replayed. The original QSA selected-KV, PLE and HC4 computation cannot be
applied to the A3B GDN/full-attention graph as generic scheduling features.

An isolated CUDA-graph adaptation of Strata's shared-expert side stream now
forks A3B's independent shared FFN branch and joins before combining it with
the routed expert result. The strict fixture matched eight output classes,
29 proposals and 16 final state pairs, with actual side-stream graph-node
assignments. In the separate [same-binary six-run 50K/256 service test](../bench/results/2026-10-16-a3b-shared-selected-50k-ab-r2/exact-ab-report.json),
both arms produced identical output and MTP proposal SHA with 238 offers and
136 accepts. Median output was **82.6 tok/s both off and on**; first nonempty
model token was 28.545 versus 28.664 s, prompt speed 1,763.7 versus 1,760.3
tok/s and sampled whole-GPU peak 17,103 MiB in both. HTTP first byte was
18.84 versus 10.37 ms, a start-of-response measurement before model text.
This candidate does not establish a throughput or memory benefit on this
sm_75 GPU, so it remains off. The isolated graph patch is not the original
Flash-Next verifier graph, which still has no A3B-native equivalent.

Holding the IQ3_S target, full Q8 MTP draft, four selected switches and
isolated executable fixed, a [six-run fixed-depth K2/K3 test](../bench/results/2026-10-16-a3b-iq3s-k2-k3-ab/exact-ab-report.json)
kept final output identical. K3 naturally changed proposal traces: it offered
317 and accepted 149 tokens (47.0%) versus K2's 238/136 (57.1%). Median
output fell from **83.2 to 79.8 tok/s**, first nonempty model token moved from
28.466 to 28.508 s, prompt speed from 1,771.1 to 1,765.8 tok/s and
whole-GPU peak rose from 17,103 to 17,165 MiB. K3 therefore remains off;
more draft depth alone did not improve this request.

The same-binary, alternating 50K/256 A/B in
`bench/results/2026-10-15-a3b-iq3s-strata-ab/README.md` measured median output
67.2 tok/s with GGML experts and 70.4 tok/s with the Strata short expert path.
The Strata arm's first model delta was 0.38 s later and sampled VRAM was
unchanged. Each arm's output was stable within the arm but differed across
arms, so draft acceptance and arithmetic order also affect this end-to-end
comparison. Neither arm reaches 100 tok/s. A short-kernel microbenchmark or
the presence of source files does not establish a full-engine speedup.

The later same-executable sequential four-arm check is recorded in
`bench/results/2026-10-15-a3b-iq3s-adapted-ab/README.md`. The combined
short-expert, single-step GDN and MMQ adapter arm measured a 28.156 s first
model token, 1,789.6 tok/s prefill, 67.4 tok/s output and 20,750.5 MiB
sampled whole-card peak (two-run medians), against baseline 28.109 s,
1,792.3 tok/s, 66.45 tok/s and 20,499 MiB. The sequential ordering and
different baseline output/MTP acceptance prevent a small speed claim. Its
optional MTP device-prefill bridge failed on the first 50K request and is
excluded from those measurements.

The bridge was subsequently fixed and passed a separate same-binary 50K/256
two-run comparison in `bench/results/2026-10-15-a3b-iq3s-bridge-fixed-ab/README.md`.
It avoided 409,395,200 bytes of whole-chunk hidden download without changing
the answer, MTP acceptance, or sampled 20,499 MiB whole-card peak. The small
sequential timing difference is not established as a stable speed gain.
The current 50K/256 graph-level Nsight trace is in
`bench/results/2026-10-15-a3b-iq3s-nsys-output/`: the 111 target-verification
graph executions took 2.647 s of a 3.920 s output phase. The subsequent
2.707 s host observation intervals are waits for the asynchronously submitted
GPU work, not CPU expert calculation. This establishes where to profile next;
it does not identify a graph-internal kernel until node-level tracing is done.

The subsequent node-level trace (`NODE-ANALYSIS.md` in that result folder)
captured 66 complete target-verification windows. Its leading kernel families
were Q6_K dense products, IQ3_S/IQ2_S expert products, Q8-to-F16 conversion
and F16 attention. The experimental Q8 direct-vector attention removed the
batched attention shape and regressed 50K output from a 64.9 to 43.8 tok/s
two-run median, with little change in first-token latency or VRAM; see
`bench/results/2026-10-15-a3b-iq3s-q8kv-vec-ab/README.md`. It remains off.

The opt-in `--kv-tiered-f16-short` instead uses the existing context tier to
allocate F16 K/V at 65K or 131K cells and Q8_0 at the 256K physical tier.
The same-binary two-run 50K/256 comparison on the RTX 2080 Ti measured
**75.45 tok/s** output, **27.811 s** first model token, **1,808.4 tok/s**
prefill and **17,119 MiB** whole-card peak for the F16 automatic arm,
versus **63.25 tok/s**, **29.203 s**, **1,724.4 tok/s** and **16,591 MiB**
for the 65K Q8 control. The F16 output and MTP acceptance differ, and
first-token/prefill directions reversed in an earlier reverse-order run, so
only the output speed direction is supported by these small experiments.
The separate 131,073-token request really rebuilt and used 262,144 Q8 cells,
then a short request shrank back to 65K F16. See
`bench/results/2026-10-15-a3b-iq3s-f16-auto-ab/README.md` for measurements
and limits. This is a useful Strata-style capacity tactic, not a copy of its
CUDA VMM implementation, and it still does not reach 100 tok/s.

On that same executable and request, enabling the original Strata grouped IQ
expert kernel produced **5,116 actual calls per 50K request** and measured a
**79.35 tok/s** two-run output median at unchanged **17,119 MiB** whole-card
peak. The answer and MTP acceptance changed from the plain F16 arm, so the
3.9 tok/s end-to-end difference is not an isolated kernel speedup. Adding the
original IQ MMQ adapter, GPU prefill hidden bridge, experimental K2 device
chain and single-step GDN switch in one combined arm recorded 1,000 MMQ calls,
25 bridge calls and 119 K2 chain calls with zero fallbacks. Its output matched
the native-expert arm, but the median dropped to **77.65 tok/s** and GPU peak
rose by **534 MiB**. The GDN switch matched zero multi-token graphs and the
K2 log's `zero cross-step CUDA graph replays` value was hard-coded rather
than measured. A later short fixture measured 49 launches in 26 K2 chains;
that separate trace does not remeasure this 50K combined arm. The combined path
is not selected as a faster default.

A separate output-only F16+native Nsight trace in
`bench/results/2026-10-15-a3b-iq3s-f16-native-node/ANALYSIS.md` captured 61
complete target-verification windows. Q8-to-F16 full-cache dequantization
was absent; leading target kernel-duration sums were **266 ms Q6_K dense**,
**300 ms original Strata grouped IQ experts** and **156 ms F16 attention**.
The trace is partial and instrumented. It identifies remaining arithmetic
work but does not supply an unprofiled throughput claim or establish that
100 tok/s is available by any one remaining transplant.

The next opt-in A3B IQ3_S down experiment reuses Strata's 16-lane sub-warp
idea for the model's 512-input down rows. Real GGUF weights from all 40 layers
passed bitwise parity. Three baseline/sub16 alternating 50K/256 service pairs
used the same binary, answer and 136/238 MTP acceptance: median output was
**81.3 versus 83.1 tok/s**, while first model token was **27.814 versus
28.052 s**, prefill **1,815.9 versus 1,803.3 tok/s**, and sampled VRAM was
**17,119 MiB** in both. This small output gain is available through the
separate `strata-a3b-iq3s-q6-kv-tier-f16-auto-native-sub16.json` config; it
is not enabled in the selected default. See
`bench/results/2026-10-15-a3b-iq3s-down-kernels/README.md`.

The original Strata Q6 load/apply idea was also adapted as a default-off
cross-column Q6_K option in the same **IQ3_S GGUF**, with bitwise parity on
real A3B matrix shapes. CUDA-event timing improved some 512/2,048/4,096-row
cases, but the 8,192-row case and 248,320-row head had no confirmed gain.
The partial-trace-weighted estimate was only about 20.7 ms of Q6 kernel
duration; no 50K service gain, first-token effect or VRAM saving is claimed.
See `bench/results/2026-10-15-a3b-iq3s-q6-reuse-exact/README.md`. The
corrected node trace identifies the MTP proposal's 130.393 ms dense work as
**Q8_0 (GGML type 8)**, not Q6_K; this Q6 option cannot accelerate it.

A further default-off adaptation of Strata's MTP KV-only catch-up stops the
A3B draft graph after cache writes and before attention/FFN/head. Real-model
draft and target sequence states matched byte for byte on eight output
limits; all four complete 50K/256 outputs and proposal traces matched.
Short Nsight fixtures confirmed catch-up CUDA kernels **3,767 to 1,422**
and FA-related calls **254 to zero**, with proposal kernels unchanged. In the
unprofiled two-run 50K API comparison, output was **81.10 versus 81.45
tok/s**, first model token **27.616 versus 27.428 s**, and sampled VRAM
**17,119 versus 17,103 MiB**. Individual output runs crossed in direction,
so the new path is not selected as a faster default. The matching GPU native
expert call count fell by 118 per request. See
`bench/results/2026-10-15-a3b-mtp-kv-only/README.md`. These additions still
do not implement the original Flash-Next verifier/commit graph or reach
100 tok/s on this card.

## Optional fused A3B GDN

The adapted kernel in `src/kernels/cuda/fused_gdn.cu` keeps GGML's
`[head][column][row]` F32 recurrent state and A3B's SiLU output gate. The
opt-in Qwen model builder makes the `z` projection available before GDN;
GGML CUDA then fuses only the exact single-token sequence of state update,
cache snapshot, RMSNorm, scale and SiLU gate. The matcher checks A3B head
dimensions, F32 contiguous inputs, tensor edges and memory aliasing. Other
graphs run the existing GGML kernels. The default graph order and execution
path do not change. This GDN computation runs on GPU 0; it does not use CPU
expert computation.

On the RTX 2080 Ti (sm_75) with the 12.73 GiB Qwen3.6-35B-A3B UD-IQ3_S
GGUF, the target executable produced the following same-parameter comparisons
with the switch off and on (15 October 2026):

| Input and output | Opt-in graph hit | First-token logits | Generated text |
|---|---|---|---|
| 9 input tokens, batch 1, 20 output tokens | Yes, including prompt prefill | All 248,320 F32 values bit-identical | Byte-identical |
| 1,063 input tokens, batch 128, 20 output tokens | Yes, in single-token decode | All 248,320 F32 values bit-identical | Byte-identical |
| 9 input tokens, batch 4, MTP K2, 30 output tokens | Yes, in the final prefill chunk; multi-token verification falls back | All 248,320 F32 values bit-identical | Byte-identical; 22 draft tokens offered and 18 accepted on both arms |

The standalone `a3b/gdn_a3b_parity.cu` fixture also checked the service's
separate input-state/cache-output layout: 524,288 state values and 4,096
output values matched an independent GGML GDN transcription bit for bit. Four
successive in-place steps also matched bit for bit. In one GPU-only local
microbenchmark, the adapted fused step took 11.36 microseconds versus 16.58
microseconds for the transcription's separate GDN and normalization launches.
This is a kernel measurement, not end-to-end decode speed or first-token
latency.

Matching GGML CUDA's `-use_fast_math` compilation for this source file was
necessary. Without it, a forced batch-1 prompt produced up to 0.539 F32
logit difference and a different 20-token continuation after repeated steps;
with it, that test's logits and continuation matched exactly. The per-request
`gdn_fused_graph_nodes` counter counts fused nodes during graph construction,
not CUDA Graph replays or kernel executions. `STRATA_A3B_GDN_GRAPH_TRACE=1`
prints the first graph matches for diagnosis. These checks establish actual
graph use and numerical parity on the tested inputs; a 50K first-token,
output-speed and VRAM A/B is still required before making a service-level
performance claim. The later 50K service arm with this option had the same
generated output and MTP acceptance as the native-expert arm without it, but
sequential two-run timings varied too much to demonstrate a benefit.
