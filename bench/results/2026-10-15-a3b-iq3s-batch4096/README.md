# IQ3_S A3B: 4,096-token prefill batch probe

On GPU 0 RTX 2080 Ti (22,528 MiB), the same executable and 50K/256 request
were compared with `--batch 2048` and `--batch 4096`. Both configured a
262,144-token context, Q8 KV, the same mixed IQ3_S GGUF, and the same Q8_0
MTP K2 head. Two sequential uncached requests per arm are stored in
`../2026-10-15-a3b-iq3s-chain-k2-fixed-ab/arms/baseline` and `arms/batch4096`.

| Prefill batch | First HTTP byte | First model token | Prefill | Output | Whole GPU 0 sampled peak |
|---|---:|---:|---:|---:|---:|
| 2,048 | 9.70 ms | 28.733 s | 1,754.2 tok/s | 64.9 tok/s | 20,499 MiB |
| 4,096 | 15.44 ms | 27.448 s | 1,839.0 tok/s | 64.1 tok/s | 21,767 MiB |

The larger batch used **1,268 MiB more** sampled whole-card VRAM. Its two
answers matched each other but differed from the 2,048 arm; MTP acceptance
was 56.7% versus 62.1%. The changed arithmetic/batching means these are
whole-route results, not an isolated kernel speedup. The two arms ran
sequentially, so the 1.28 s first-model difference needs an alternating
repeat before it can be called stable. Output speed did not improve. The
2,048 default remains in use while the long-context decode bottleneck is
addressed.
