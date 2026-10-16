# A3B Q8 50K prefill scheduling diagnosis

This experiment used the official Qwen3.6-35B-A3B Q8_0 and official MTP Q8,
GPU 0, FP16 KV, 262144 context, 2048 prompt batch, MTP K2, the same verified
expert profile, and the same synthetic 50000-token prompt with 256 output tokens.
The model files and original `G:/Strata` source were not changed. Both requests
reported exactly 50000 prompt tokens and no prompt-cache reuse. Measurements
exclude model load. The instrumented executable SHA-256 was
`e275ac4dc4e510b6146861d71a7d73418080270d0fe1fc4b7ab0dd44ee2e354e`.

| Staging slots | First model token | Prefill | Output | Expert upload | Staging batches |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 137.369 s | 364.8 tok/s | 37.8 tok/s | 340909.5 MiB | 13806 |
| 32 | 192.267 s | 260.4 tok/s | 36.8 tok/s | 340001.1 MiB | 3819 |

The first HTTP response byte was 19.27/19.49 ms, respectively. It is an early
streaming header, not a model token. The two output files differed, so larger
groups also changed numerical behavior. Each condition has one run only.

The added host wall-time counters report cumulative intervals along the prefill
path. They overlap asynchronously queued GPU work and must **not** be summed as
independent GPU/PCIe durations.

| Staging slots | Route read and preceding GPU wait | `compute_gpu_only` host span | Bank reuse wait, within host span | Task submit, within host span |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 26.539 s | 106.566 s | 62.897 s | 7.721 s |
| 32 | 44.069 s | 141.515 s | 95.639 s | 7.615 s |

The 32-slot condition cut batch count 72.3% while lengthening prefill 40.1%.
This refutes the idea that fewer batches or weight copies alone solve the
bottleneck. The current code ties the size of the upload window to the number
of experts in each generic GGML matrix task. A larger window therefore also
changes GPU work shape and duration. The next implementation should separate
a deep copy-ahead ring from the compute group and use a dedicated Q8 grouped
prefill kernel. The original Strata already separates its large stream-ahead
ring from MMQ/fused compute groups in `src/prefill/prefill.cpp`; its CUDA native
fused kernel does not directly cover a Q8 gate/up pair on this device.

The selected local API service was restored to the original
`strata-a3b-q8.json` and frozen `build-a3b-q8-tuned` executable after testing.
Its 8-slot setting and all production defaults remain unchanged.
