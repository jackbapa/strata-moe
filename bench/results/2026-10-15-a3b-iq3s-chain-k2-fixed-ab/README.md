# IQ3_S A3B: experimental two-decode MTP device chain

This check used the same new executable, GPU 0 RTX 2080 Ti (22,528 MiB),
262,144 configured context, uncached 50,000-token input, 256 output tokens,
Q8 KV, and the single Qwen3.6-35B-A3B UD-IQ3_S mixed GGUF with its Q8_0 MTP
head at K2. The K2 switch also moves the draft's 515.3 MiB Q8 token embedding
from host memory to GPU; the measurements therefore cover the **combined**
change. Q6_K tensors are part of the same target IQ3_S GGUF.

| Route | Runs | First HTTP byte | First model token | Prefill | Output | Whole GPU 0 sampled peak |
|---|---:|---:|---:|---:|---:|---:|
| K2 off | 2 | 9.70 ms | 28.733 s | 1,754.2 tok/s | 64.9 tok/s | 20,499 MiB |
| K2 device chain on | 1 | 19.88 ms | 27.883 s | 1,811.2 tok/s | 66.9 tok/s | 21,013 MiB |

All three requests produced the same output SHA-256
`c1d52a6c58911b9cae7600ccafb992b2dbeeddb245c8050031187a8891068379`
and offered/accepted 227/141 MTP tokens. The enabled 50K request logged 113
K2 calls and zero fallbacks. Its log's **zero cross-step CUDA graph replays**
field was a literal constant, not a measured CUDA count. The chain sends
token and hidden data GPU-to-GPU between two independent GGML `llama_decode`
submissions. It is not the original Strata captured, device-connected MTP
round. A separate 32/64 request also matched the prior answer with 30 K2
calls/zero fallbacks.

One experimental request versus two sequential baselines cannot establish a
speed gain; the two baseline output rates alone ranged from 62.9 to 66.9
tok/s. The K2 route uses 514 MiB more whole-GPU peak in this check and remains
off by default. The earlier attempt without GPU draft embedding crashed at a
CPU `GET_ROWS` index assertion; the opt-in preflight now requires the draft
token table on the same GPU before it changes draft KV. The fixed short and
50K requests completed without that assertion. No >100 tok/s result exists.

The run JSON records model, MTP, executable, config, request and output hashes,
plus raw latency, VRAM and MTP counters. GPU peak is whole-card sampling.
