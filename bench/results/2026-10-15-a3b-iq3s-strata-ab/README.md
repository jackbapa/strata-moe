# IQ3_S A3B: same-model Strata expert A/B on GPU 0

Windows, Threadripper 3960X, NVIDIA RTX 2080 Ti GPU 0 (22,528 MiB),
Qwen3.6-35B-A3B UD-IQ3_S target (`66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66`)
and Q8_0 MTP (`f6cb995fab78e7284097b2e13508e8daad4ffc6a747b433aef399f4e48524b5a`).
The Q6_K dense tensors are part of this one mixed-quantization IQ3_S GGUF;
they are not a second model. The Q6_K optimization uses GGML CUDA, while
the routed expert arm uses Strata's own CUDA kernel.
Both arms used the same executable (`9763195c78c53531c0a6088c720cee9397ea6b697f4f59100b82d40045e9e158`),
262,144 configured context, Q8 K/V, MTP K2, exactly 50,000 counted uncached
input tokens, 256 generated tokens, temperature 0 and seed 12345. The
request template and its SHA-256 are in `request-template.json` and
`identity.json`. Model loading is excluded from the timed request. The order
was baseline, Strata, baseline, Strata, with a service restart for each arm.

| Path | Runs | First HTTP byte median | First model delta median | Prefill median | Output median | GPU 0 sampled peak median |
|---|---:|---:|---:|---:|---:|---:|
| GGML experts + grouped Q6_K dense | 2 | 15.2 ms | 27.44 s | 1841.0 tok/s | 67.2 tok/s | 20,499 MiB |
| Strata grouped IQ experts + fused gate/up + same grouped Q6_K dense | 2 | 14.8 ms | 27.82 s | 1815.1 tok/s | 70.4 tok/s | 20,499 MiB |

The opt-in Strata arm recorded **5,070 original Strata grouped expert CUDA
calls in decode/MTP and zero in prefill per request**. The latter remains the
GGML IQ MMQ path. Thus this is a real Strata expert-kernel integration, but it
is not a complete port of Strata's long-prefill scheduler, cross-MTP-step
device graph or A3B-specific GDN. The 50K/256 result improves observed output
speed by 3.2 tok/s (4.8%), while delaying the first model delta by 0.38 s
and reducing observed prefill speed by 25.9 tok/s. No sampled VRAM saving was
observed. The first HTTP byte is a response-header measurement, not model
completion.

The two arms generated different deterministic outputs; each arm repeated its
own output. MTP acceptance was also different (median 62.1% baseline versus
58.8% Strata). Therefore this paired service experiment measures the complete
configuration effect and **cannot attribute all 3.2 tok/s to the expert
kernel**. Two runs per arm are enough to expose the current direction, but
not to resolve small differences reliably. The target of 100 output tok/s
was not reached.

`tools/bench_a3b_ab.py` saved every response event, output, resource sample,
process command/environment and hashes in `arms/`. GPU memory is whole-card
NVIDIA sampling every 0.5 s, not process-attributed allocator high-water.
The original `G:/Strata` checkout was not modified.
