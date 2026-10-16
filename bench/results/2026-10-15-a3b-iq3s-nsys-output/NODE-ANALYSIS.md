# Partial output CUDA node trace, 50K input

The 50K/256 request used the same IQ3_S A3B GGUF, Q8_0 MTP K2 head,
262,144-token configured context and GPU 0 RTX 2080 Ti as `ANALYSIS.md`.
Nsight Systems 2024.4.2 began node-level collection after the first nonempty
model SSE delta and recorded **66 complete target-verification NVTX ranges**
plus parts of adjacent output. The trigger itself took 1.868 s; profiling
therefore missed early output and perturbed timing. These numbers locate work,
not unprofiled token speed.

Within the captured `a3b/output/target-verify` ranges, GPU kernel durations
sum to 1.584 s. Leading kernel families are:

| Kernel family | Captured duration | Share of target kernel sum |
|---|---:|---:|
| Q6_K `mul_mat_vec_q` (GGML type 14) | 288 ms | 18.2% |
| IQ3_S `mul_mat_vec_q` (type 21) | 270 ms | 17.1% |
| Q8_0 → F16 dequantization | 238 ms | 15.1% |
| IQ2_S `mul_mat_vec_q` (type 22) | 215 ms | 13.6% |
| F16 FlashAttention | 171 ms | 10.8% |

These five families account for about 74.8% of traced target kernel time.
The type numbers are defined in the pinned `ggml.h`; IQ2_S/IQ3_S expert and
Q6_K dense tensors are in the **same** IQ3_S mixed GGUF. F16 FlashAttention
appears alongside Q8_0 dequantization, but kernel names alone do not prove
which exact tensors each dequantization call reads; inspect graph edges and
CUDA selection before changing KV arithmetic. CUDA kernel sums may overlap
across streams and are not exclusive wall-clock percentages.

The earlier full output trace found 111 actual target-verification CUDA Graph
executions taking 2.647 s of a 3.920 s output interval. This partial node
trace explains why copying only a scheduler or adding another graph wrapper
cannot promise 100 tok/s: quantized dense/expert math, conversion, and
attention inside the target graph dominate. It also identifies exact routes
for opt-in parity-checked kernel work. See `output-node-explicit.nsys-rep`,
`output-node-explicit.sqlite`, `output-node-stats_*.csv`, and
`api-node/arms/traced/run001.nsys-control.json` for raw evidence.
