# A3B IQ3_S with Strata's native grouped IQ experts

This is the separate `G:/Strata-Moe` checkout. The original `G:/Strata`
checkout was not edited. The target is Unsloth Qwen3.6-35B-A3B UD-IQ3_S
(SHA-256 `66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66`),
with the ggml-org Q8_0 MTP head. Hardware: Windows, Threadripper 3960X,
128 GiB RAM, one RTX 2080 Ti with 22,528 MiB on GPU 0.

The opt-in `--strata-native-experts` now invokes the original Strata
`native_expert_grouped` IQ kernel for A3B short decode and MTP verification.
It reads IQ2_S gate/up and IQ3_S or IQ4_XS down weights directly from their
three resident GGUF GPU tensors, with no repacked model copy. CUDA forms the
route plan and groups repeated expert IDs across verification tokens. The
operation runs on the ordinary CUDA graph stream; a short target graph has
one split instead of 81 with the first independent-backend integration.
The 50K prompt path still uses the existing GGML MMQ implementation. The
Qwen3.5/3.6 trunk still comes from the pinned llama.cpp model implementation;
this is a measured expert-kernel port, not a complete replacement of the
inference engine.

The original Strata parity and bench tools were generalized from the
Flash-Next 2560/640 geometry to A3B 2048/512. Real layers 0, 19 and 39
passed parity with zero failures; their GPU-vs-float-reference relative
errors were approximately 0.98%, 1.07% and 1.10%. The separate gate/up/down
pointer layout produced bitwise-identical kernel outputs to the original
packed layout on those layers. A 96-output-token short MTP trial produced
identical output before and after CUDA grouping; the 50K/256 output also
matched between the two Strata-kernel integrations. Their text differed
from the GGML baseline, consistent with the changed activation quantization
and arithmetic order. This comparison is not a quality evaluation.

The API benchmark used 262,144 context, exactly 50,000 counted prompt
tokens, 256 output tokens, temperature 0, seed 12345, Q8 K/V, MTP K2, and
no cached prompt tokens. Each row is **one run**, not a stable median.
Model load is excluded. First model token is the first nonempty streamed
answer delta; HTTP first byte is the first socket response including headers.

| Path | First model token | Prefill | Output | HTTP first byte |
|---|---:|---:|---:|---:|
| Same-build GGML baseline | 28.00 s | 1800.8 tok/s | 56.1 tok/s | 11.6 ms |
| Strata kernel, separate expert backend | 27.51 s | 1830.6 tok/s | 55.3 tok/s | 18.7 ms |
| Strata kernel, CUDA graph, grouped MTP routes | 27.62 s | 1833.6 tok/s | 57.9 tok/s | 9.6 ms |

The 57.9 versus 56.1 tok/s difference is too small for a one-run speed claim.
The selected service therefore keeps the tested baseline
`strata-a3b-iq3s.json`; the experiment is saved as
`strata-a3b-iq3s-strata-native.json`. The stable service at 127.0.0.1:8096
was restored after testing. The target of 100 or 200 tok/s was not reached.
The same-build baseline's 256-token answer was byte-identical to the earlier
selected K2 run with the same input and seed.

Nsight Systems, for a separate roughly 1K-prompt/96-output run, summed
268 ms of GGML IQ2_S/IQ3_S/IQ4_XS expert vector kernels against 214 ms of
Strata gate/up/down grouped kernels. These are GPU kernel sums that include
profiler overhead and are not wall-clock throughput. The same trace still
shows substantial Q6_K dense matrix and GDN work. A point sample while the
50K experiment ran showed about 20,493 MiB used on GPU 0; this is not a
peak measurement. The experimental path recorded zero CPU expert rows.

The A3B CUDA build passed. `ctest --test-dir build-a3b --output-on-failure
-j 4` passed 16/16 after this change. HIP and SYCL were not built on this
Windows NVIDIA host. Raw API plans, events, outputs and timing JSON are in
the `api-50k-*` directories next to this file. The original source's native
bench/parity tools and the build logs are retained in the clone.
