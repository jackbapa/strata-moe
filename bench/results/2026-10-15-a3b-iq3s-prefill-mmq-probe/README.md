# A3B IQ3_S long-prefill gate/up MMQ probe

Hardware: GPU 0, NVIDIA RTX 2080 Ti (CUDA sm_75), Windows, CUDA 12.6. The
8096 inference service was stopped for the final runs. Model:
`models/a3b/Qwen3.6-35B-A3B-UD-IQ3_S.gguf` (the same IQ3_S main model used by
the service). Each trial uses one real layer's 256 IQ2_S gate/up experts,
2048 activation tokens, 8 routes/token, and 50 iterations timed with CUDA
events. Results are milliseconds per layer invocation. Both graphs use the
same GGML CUDA MMQ implementation. The second graph packs gate and up into
one quantized tensor, matching Strata's grouped gate/up arrangement; the
first graph runs the two existing `ggml_mul_mat_id` operations. Identical
input, routes, and weight bytes were used. All 16,777,216 output floats per
trial matched bitwise.

| Layer | Separate gate+up MMQ, ABBA trials | Packed gate+up MMQ, ABBA trials | Warmed gain, packing excluded |
| --- | ---: | ---: | ---: |
| 0 | 5.488 / 4.540 ms | 4.370 / 4.355 ms | ~4.0% vs the later separate trial |
| 19 | 4.620 / 4.707 ms | 4.521 / 4.542 ms | ~2.9% vs the adjacent mean |
| 39 | 4.656 / 4.650 ms | 4.528 / 4.527 ms | ~2.7% vs the adjacent mean |

The initial layer-0 separate trial includes GPU clock warmup and should not
be used as the speedup denominator. Packing is excluded from timings and
requires **164 MiB extra per layer** if both original GGUF roles remain
resident. Across all 40 layers this would add about **6.4 GiB**, which is not
available alongside the IQ3_S model, MTP, and 256K KV cache on this GPU.
Packing each layer afresh for every prompt chunk adds another 164 MiB GPU
copy per layer invocation, which this probe does not time. The measured
~3% local upper bound is not evidence of a whole-model gain.

Original Strata's `moe_mmq.cu` supports IQ2_S/IQ3_S/IQ4_XS and invokes
GGML's `mul_mat_q_case`; it adds the expert grouping, gate/up packing, and
staging schedule around that kernel. Current GGML's `mmid.cu` already groups
tokens by expert on GPU, quantizes activations to Q8_1, and runs MMQ on this
sm_75 card. Strata's different fused native IQ prompt kernel in
`moe_fused_iq.cu` requires CUDA sm_80 or newer and cannot execute on the
RTX 2080 Ti. The probe tests the remaining gate/up grouping opportunity in
isolation; it does not claim to be an end-to-end Strata prefill port.

Build: `python tools/a3b.py build --jobs 8`. Run, for example:
`build-a3b/bin/strata-moe-prefill-mmq-probe.exe models/a3b/Qwen3.6-35B-A3B-UD-IQ3_S.gguf 19 2048 50`.
