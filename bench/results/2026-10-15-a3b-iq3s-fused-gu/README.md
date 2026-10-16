# A3B IQ3_S: original Strata fused gate/up on separate GGUF tensors

Windows, Threadripper 3960X, RTX 2080 Ti (22,528 MiB, CUDA sm_75), GPU 0,
Qwen3.6-35B-A3B UD-IQ3_S (model SHA-256
`66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66`),
Q8_0 MTP K2. The tested executable SHA-256 was
`92a9e6ccdcfc6db429cd09185a3989cac4ab140f453ee0bdb6297940db58f5ae`.
After removing the non-exact Q6_K prototype, the final rebuilt executable is
`bb9037bbc2977b7bc594fc0dcf696aa39dad7494c96b3a16edd3e828a2714d57`;
the fused IQ3_S code is unchanged.
The original `G:/Strata` checkout was
read only. This experiment changes `G:/Strata-Moe` only.

The original Strata's `native_gu_fused_kernel` combines gate/up products,
SwiGLU and intermediate Q8_1 quantization in one GPU launch. A3B had used
Strata's grouped IQ kernel but could not enter this fused branch because its
GGUF stores gate, up and down in three separate tensors. The new opt-in
`STRATA_A3B_FUSED_GU=1` makes that kernel read the three GPU tensor pointers
directly for A3B's IQ2_S gate/up, 2048/512 expert geometry. The down projection
continues on the original grouped kernel. No CPU expert calculation, PCIe
expert copy or second model-weight copy is involved. The opt-in config is
`strata-a3b-iq3s-fused-gu.json`.

Real GGUF layers 0, 19 and 39 cover IQ2_S gate/up with IQ3_S and IQ4_XS down.
For eight distinct experts and three input tokens each, packed original versus
split fused output differed in **0/49,152 floats on each layer**. The dedicated
CUDA test is `a3b_iq3s_fused_gu_parity`; it skips if the local GGUF is absent.
The 128-token CLI outputs and 50K/256 API outputs were byte-identical with the
switch off and on for their respective identical inputs.

Isolated expert operation, CUDA events, 200 iterations per trial, three
alternating off/on trials. These are kernel-call times, not end-to-end model
speed:

| Real layer | Split original median | Split fused median | Reduction |
|---|---:|---:|---:|
| 0, IQ3_S down | 110.68 us | 79.76 us | 27.9% |
| 19, IQ3_S down | 112.45 us | 82.65 us | 26.5% |
| 39, IQ4_XS down | 100.97 us | 70.42 us | 30.3% |

The API test used 262,144 context, exactly 50,000 counted input tokens,
256 output tokens, Q8 K/V, temperature 0 and seed 12345. Model loading is
excluded; each request reported zero cached prompt tokens. The exact request
SHA-256 was `2ed16fa60f51be9afe54213d3bd8190a5c19f2c6ada95ccaf03a06017f79a910`.
The final output JSON SHA-256 was
`3691e1181739516e37377a2f8c23afd62267779bc8c2e84c0bf55c9c77f61d78`
for both arms. Results are listed in execution order because the later runs
slowed in both arms. HTTP first byte includes the response headers; first
model token is the first nonempty answer delta:

| Order | Fused gate/up | First model token | HTTP first byte | Prefill | Output |
|---:|:---:|---:|---:|---:|---:|
| 1 | off | 27.57 s | 18.1 ms | 1827.2 tok/s | 57.5 tok/s |
| 2 | on | 28.46 s | 18.7 ms | 1779.7 tok/s | 58.7 tok/s |
| 3 | on | 28.54 s | 9.2 ms | 1774.2 tok/s | 58.6 tok/s |
| 4 | on | 29.83 s | 20.4 ms | 1679.9 tok/s | 55.7 tok/s |
| 5 | off | 29.04 s | 11.0 ms | 1737.6 tok/s | 54.4 tok/s |
| 6 | off | 31.38 s | 24.9 ms | 1596.9 tok/s | 53.2 tok/s |

One live sample showed GPU 0 at 82-83 C and 1,740-1,815 MHz during a 50K
run. It does not prove why later runs slowed. The isolated kernel gain is
clear; an end-to-end gain is **not** established by these non-interleaved,
temperature-dependent API runs. The selected service default is unchanged.

The original Strata's fused prompt int8 expert path requires NVIDIA sm_80 or
newer and uses MMQ on sm_75. This 2080 Ti likewise uses GGML MMQ for A3B's
long prompt, which measured about 1,600-1,830 token/s here. A short Nsight
capture requested 96 output tokens but reported only 67 route-plan kernel
instances, so it cannot apportion full-request time. The earlier fuller trace
in `2026-10-15-a3b-iq3s-strata-native` identifies Q6_K dense products and
GDN work as other costs; MTP catch-up and long-context attention also need
separate wall-clock measurements.

A prototype also tried Strata's Q6_K `iq_mmvq` on 2048-by-2048 dense matrices
for one to three verification tokens. Against the current GGML CUDA target
path, 1,150/2,048, 2,238/4,096 and 3,158/6,144 values differed bitwise;
the largest absolute difference was 1.91e-6. That is small numerically but
breaks the exact sequential-target contract used by MTP validation. The
prototype was removed; Q6_K dense execution remains unchanged pending an
exact multi-token kernel with its own parity and timing evidence.

CUDA build passed. All 17 CTest cases passed, including the new
`a3b_iq3s_fused_gu_parity` with the real GGUF. Raw timing,
output, per-request SSE and profiler files are beside this record. HIP and
SYCL are not involved in this CUDA-only change.
