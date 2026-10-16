# A3B IQ3_S expert kernels in the 50K output trace

This is a read-only kernel audit of the existing A3B implementation and the
original Strata source. No inference kernel or selected service setting was
changed. The target remains the single Qwen3.6-35B-A3B UD-IQ3_S mixed GGUF;
its Q6_K dense tensors are part of that file. The trace was captured on GPU 0,
an RTX 2080 Ti (sm_75), with a 262,144-token configured context, 50,000
uncached input tokens, 256 output tokens, Q8 K/V, and Q8_0 MTP K2. The raw
export is `bench/results/2026-10-15-a3b-iq3s-nsys-output/output-node-explicit.sqlite`
and the kernel summary is `output-node-stats_cuda_gpu_kern_sum.csv` beside it.

Nsight started after the first streamed output. The capture has 67
`a3b/output/target-verify` ranges, versus 114 in the earlier full phase trace;
its GPU-kernel timestamp span is 2.602 seconds. Thus the following duration
sums cover **only part of output**, include profiler overhead, and must not be
substituted for an unprofiled 256-token throughput measurement. CUDA graph
nodes are visible: more than 97% of the IQ2_S/IQ3_S vector instances have a
`graphNodeId`.

| Kernel family in the partial output capture | GPU kernel-duration sum | Instances |
|---|---:|---:|
| IQ3_S `mul_mat_vec_q`, routed down projections | 273.99 ms | 2,479 |
| IQ2_S `mul_mat_vec_q`, routed gate/up projections | 218.20 ms | 5,360 |
| Q6_K `mul_mat_vec_q`, dense matrices in the same GGUF | 288.27 ms | 16,566 |
| Q8_0 to FP16 dequantization | 257.81 ms | 1,436 |
| FP16 and Q8 attention kernels shown in the summary | 256.78 ms | 844 |

The two main routed-expert vector families total 492.19 ms, about 18.9% of
this 2.602-second captured span. Even **removing all of these measured expert
kernel durations**, while leaving the other work and scheduling unchanged,
would save less than one fifth of that span. That is an optimistic diagnostic
bound, not a measured service speedup: kernel durations can overlap on streams,
and the profiler changes timing. A faster expert kernel alone is not an
evidence-based route from the observed roughly 67-70 output tokens/s to 100.

The applicable original Strata path is **already ported as an opt-in**:
`a3b/strata_native_experts.cu` builds the device-side expert groups and calls
the original `strata::kernels::native_expert_grouped` from
`src/kernels/cuda/iq_kernels.cu`. `NativeExpertLayout::split_storage` lets that
kernel read A3B's separate gate, up and down GGUF tensors without repacking
model weights. `STRATA_A3B_FUSED_GU=1` additionally selects the original
mode-8 gate/up + SwiGLU + Q8_1 fusion for A3B's IQ2_S gate/up, 2048 hidden
and 512 intermediate width. The IQ3_S/IQ4_XS down projection stays in the
original grouped down kernel. The dedicated real-weight parity test covered
layers 0, 19 and 39; isolated expert-call medians fell 26.5-30.3%, but the
existing six-run 50K API series did not establish an end-to-end gain. See
`bench/results/2026-10-15-a3b-iq3s-fused-gu/README.md` for its exact setup.

The other original Strata specialist, `STRATA_EXPERT_V2`, cannot be enabled
for this GGUF. Its dispatch requires IQ3_S gate/up, Q2_0 down, 2560 hidden,
640 intermediate width and a packed expert blob. A3B has IQ2_S gate/up,
IQ3_S or IQ4_XS down, 2048/512, and split GGUF tensors. The S26 kernels use
`Split<21>`/`Split<20>` explicitly and hard-code 2560/640 scratch geometry.
Porting the idea would require new IQ2_S/IQ3_S kernels and parity tests; it is
not a safe flag flip or a small reuse of the existing code. The original
fused IQ prompt path also requires sm_80 or newer, whereas the measured card
is sm_75.

If a new A3B-specific down kernel is attempted, keep it opt-in and first
compare every real layer against the current grouped Strata output, including
the three IQ4_XS down layers. Then compare complete 50K/256 requests with
the same executable, answer hash, MTP acceptance, first HTTP byte, first model
token, output rate and whole-card VRAM peak. The present trace gives no basis
to claim a throughput gain from an untested kernel change.
