# A3B IQ3_S Q6 + tiered K/V, isolated 50K trial

Windows, RTX 2080 Ti GPU 0 (sm_75, 22,528 MiB), Qwen3.6-35B-A3B UD-IQ3_S
target and Q8_0 MTP K2. The selected IQ3_S/Q6 config was cloned only for
port 8097 and `--kv-tiered`; the original Strata checkout was not changed.
The server still reported a 262,144-token maximum context. The 50,000-token
prompt plus 256-output request used the 65,536-cell target and MTP contexts.

One measured run, no prompt cache, temperature 0, seed 12345:

| HTTP first byte | First model token | Prefill | Output | GPU0 used after request |
|---:|---:|---:|---:|---:|
| 19.57 ms | 27.473 s | 1829.9 tokens/s | 67.3 tokens/s | 16,591 MiB |

The exact input SHA-256 was
`2ed16fa60f51be9afe54213d3bd8190a5c19f2c6ada95ccaf03a06017f79a910`.
The output JSON SHA-256 was
`d01bcd72056af652f33ed32a582cbcf154f909ff91dbd173cdc790864d7694d1`,
byte-identical to the saved full-256K Q6 run. That saved single run measured
29.12 s first model token, 1736.8 prefill tokens/s, and 63.1 output tokens/s.
These runs are not a paired repeated comparison; they do not establish a speed
gain from tiering.

Startup allocation logs report target+MTP Q8 K/V of 680+68 MiB at 65,536
cells, compared with 2720+272 MiB at 262,144 cells in the full-size baseline.
Target+MTP CUDA compute reserves are 656.46+240.28 MiB, compared with
1744.46+816.28 MiB. Combined logged GPU allocations are 3908 MiB lower;
actual total GPU-use reduction requires paired measurement. Tier switches
destroy and rebuild idle contexts, so a long request above the current tier
may incur one-off setup latency.

A separate boundary smoke then sent exactly 65,537 counted input tokens and
requested one output token. The engine logged a transition from 65,536 to
131,072 cells and completed that request (39.353 s first model delta;
38.980 s prefill, 1681.3 tok/s). The one-token output has no meaningful
decode-rate estimate. A following 32-input/16-output request moved back to
65,536 cells; repeating that short request gave byte-identical output and
zero cached prompt tokens both times. The raw cases are in `boundary-65537/`
and `after-growth-32/`. This tests one growth/shrink boundary and request
state isolation, not a full 256K-token inference or an end-to-end speed gain.

The tiered-KV implementation retains the model weights and rebuilds GGML
contexts between independent requests. It is a separate opt-in memory tactic,
not the original Strata VMM `--kv-grow` implementation. The initial opt-in is
limited to native GPU serving. The model is still IQ3_S mixed quantization;
Q6_K is only the existing GGML exact dense-verification switch for its Q6_K
tensors.
