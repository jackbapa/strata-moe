# Q8 A3B experimental issuer and packed gate/up, 2026-10-15

Hardware: Threadripper 3960X, 128 GiB RAM, GPU 0 RTX 2080 Ti (22528 MiB,
PCIe 3.0 x16), Windows. Model: complete official Qwen3.6-35B-A3B Q8_0 GGUF
and official Q8 MTP K2. Every full-model case used context 262144, batch
2048, the same fixed expert profile, temperature 0, seed 12345, exactly
50000 prompt tokens and 256 completion tokens with zero prompt-cache reuse.
The HTTP request hash was `81c405f14c943924ae8c99d25589794a73ca415b5c8acf372b34f84707d096cc`.
Each number below is **one fresh-process request**, not a repeated average.

| Path | First model token | Prefill | Output | Generated text SHA-256 |
| --- | ---: | ---: | ---: | --- |
| Prior host-only copy-ahead, separate earlier trial | 111.38 s | 449.9 tok/s | 39.5 tok/s | `92bf513f71a34c13b1f2b88e05c8b6131f649d0842874c725dc862451d2e9a2c` |
| Independent per-layer H2D issuer | 113.28 s | 442.9 tok/s | 38.6 tok/s | `92bf513f71a34c13b1f2b88e05c8b6131f649d0842874c725dc862451d2e9a2c` |
| Issuer with GPU copy-stream event wait | 112.92 s | 444.2 tok/s | 38.3 tok/s | `92bf513f71a34c13b1f2b88e05c8b6131f649d0842874c725dc862451d2e9a2c` |
| Issuer plus packed Q8 gate/up GGML MMQ | 112.69 s | 444.7 tok/s | 40.4 tok/s | `5405618848aadb2b09bdb627577face8f19342aae495366d75042408649ae9b8` |

The per-layer issuer moves H2D submissions onto a dedicated thread and
protects host and GPU bank reuse with separate completion events. The later
GPU-stream-wait variant queues the bank dependency on the copy stream rather
than blocking its host issuer. Its full result is under
`bench/results/2026-10-15-a3b-q8-async-issuer/measurement/`. It leaves
the original Strata's all-layer weight schedule unimplemented. The packed
path uploads gate and up consecutively and computes them in one GGML Q8 MMQ
product, sharing the input quantization. It is still not Strata's original
16-expert grouped CUDA MMQ pipeline. Neither experiment demonstrates a speed
gain at 50K, and neither reaches 100 output tok/s.

The 1024-token Q8 expert fixture gave identical FNV64 checksums
`1bc5f9cc2f298d31` for the host-only stager and independent issuer. The
packed path had a different checksum. Comparing all 4,194,304 F32 outputs
gave maximum absolute difference `7.394934e-6`, mean absolute difference
`3.4527556e-9`, and RMSE `1.1855292e-7`; all values were finite. The full
model's fixed-sampling text diverged at character 76, so numerical closeness
does not imply identical generated text.

The two new switches are opt-in. `strata-a3b-q8.json` continues to use the
previously measured host-only copy-ahead path. After the experiment, the
production service was restarted on that config and passed a 2048/64 smoke
request: first model token 3.69 s, prefill 617.6 tok/s, output 44.3 tok/s.
All 16 CUDA CTest cases passed; the CPU build succeeded. HIP configuration
could not find a HIP SDK, and SYCL configuration found no SYCL-capable C++
compiler on this host. The original `G:/Strata` checkout was not edited.
