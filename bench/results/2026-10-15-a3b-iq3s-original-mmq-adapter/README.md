# A3B IQ3_S: original Strata MMQ split-tensor adapter

This began as an isolated layer probe and now also includes a short-prompt
service-path comparison. It links this repository's byte-identical copy of Strata's
`src/prefill/moe_mmq.cu` to the IQ3_S A3B adapter and current GGML CUDA
backend. The original `G:\Strata` project was not modified.

Hardware: RTX 2080 Ti, CUDA sm_75, GPU 0, CUDA 12.6. Model:
`models/a3b/Qwen3.6-35B-A3B-UD-IQ3_S.gguf`. The model's gate/up weights are
IQ2_S; down is IQ3_S on layers 0/19 and IQ4_XS on layer 39. The reference is
GGML's normal three `ggml_mul_mat_id` operations plus SwiGLU. The adapter
performs GPU route grouping, Q8_1 activation packing, and three calls to
Strata's original grouped IQ MMQ `Context::run`, using the existing separate
GPU-resident gate/up/down tensors. It makes no second weight copy.

Every tested float was bitwise equal to the GGML reference:

| Layer | Tokens | Input lanes | Down type | Compared floats | Differing floats | CUDA adapter calls | Routed rows |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0 | 9 | 1 | IQ3_S | 147,456 | 0 | 1 | 72 |
| 0 | 128 | 1 | IQ3_S | 2,097,152 | 0 | 1 | 1,024 |
| 19 | 128 | 1 | IQ3_S | 2,097,152 | 0 | 1 | 1,024 |
| 39 | 128 | 1 | IQ4_XS | 2,097,152 | 0 | 1 | 1,024 |
| 19 | 128 | 8 | IQ3_S | 2,097,152 | 0 | 1 | 1,024 |
| 0, 19, 39 | 2,048 each | 1 | IQ3_S/IQ4_XS | 33,554,432 each | 0 | 1 each | 16,384 each |

GPU event time per **one expert layer** for 2,048 tokens, 20 warmed repeats
per pass, run in GGML/Strata/Strata/GGML order:

| Layer | GGML first/last (ms) | Strata first/last (ms) |
| ---: | ---: | ---: |
| 0 | 8.5236 / 7.4193 | 7.2491 / 7.1646 |
| 19 | 8.2423 / 7.5239 | 7.3387 / 7.3110 |
| 39 | 9.4105 / 7.2861 | 7.3151 / 7.2732 |

The first GGML pass includes graph setup effects. Its later pass and Strata
are close (roughly 0–3% layer-local difference); these measurements do not
establish an end-to-end benefit. The adapter does not add Strata's packed
gate/up weight layout, whose permanent duplicate would cost about 164 MiB per
layer on this model. It also does not provide Strata's separate expert copy
threads: these weights are already resident on GPU 0.

The original `moe_mmq.cu` supports IQ2_S/IQ3_S/IQ4_XS on sm_75 after its
`fits` tile check. Its native fused IQ prefill kernel in
`moe_fused_iq.cu` requires sm_80+ and cannot run on this card. When compiling
the adapter with the current GGML CUDA library, `GGML_CUDA_USE_GRAPHS` must
match that library: `ggml_backend_cuda_context` has a conditional graph field.
An initial standalone fixture compiled without the macro crashed even in the
GGML oracle; recompilation with the matching macro fixed it. That failed
attempt is excluded from the timing table.

The CMake-built parity fixture accepts `path layer tokens lanes
[bench_repeats]`, for example
`strata-moe-prefill-mmq-adapter-parity <GGUF> 19 2048 1 20`. Its layer-19,
128-token rerun after the service integration still found 0 differences in
2,097,152 floats and one actual CUDA MMQ call for 1,024 routed rows.

## Short-prompt service-path A/B

`short_service_ab.py` runs the **same build and IQ3_S model** with the new
`--strata-prefill-mmq` flag off and on. Both runs use GPU 0, resident IQ3_S
experts, an 864-token prompt, `--batch 512`, one generated token, and no MTP
draft, so the first-token logits comparison isolates the prefill expert path.
See `baseline.stderr.txt`, `strata_prefill.stderr.txt`, and
`short_service_ab.json` for the exact commands and raw output.

| Path | Prefill time | Prefill rate | Actual Strata prefill CUDA calls | Routed rows |
| --- | ---: | ---: | ---: | ---: |
| Existing GGML three-MMID | 0.383 s | 2,256.51 tok/s | 0 | 0 |
| Original Strata grouped MMQ adapter | 0.388 s | 2,226.56 tok/s | 80 | 276,480 |

All **248,320 first-token logits matched bit for bit** (identical SHA-256).
The two prompt chunks each ran 40 expert layers through the Strata MMQ
adapter; the 80 actual CUDA calls and 276,480 routed rows came from the
device-side counters. There were zero fallback graph builds. In this one
short-prompt run the opt-in path was about 1.3% slower. Model load and
generation are excluded from those prefill times; the full wall times are in
the JSON. This run does not measure 50K TTFT, decode speed, MTP, or VRAM.

This direct MMQ path keeps separate resident gate/up/down GGUF tensors and
does not add a packed duplicate or a host expert transfer stage. Original
Strata's wider scheduling and caching architecture requires separate tests;
the short-prompt results above only establish an actual service invocation
and numerical equivalence of this prefill kernel adapter.
The `--batch 2048` adapter reserves about 141 MiB of GPU scratch by allocation
formula (64 MiB gate/up output, 32 MiB hidden activations, about 45 MiB Q8_1
buffers, plus small route maps). This is a code-derived allocation estimate,
not a measured whole-service VRAM delta.
