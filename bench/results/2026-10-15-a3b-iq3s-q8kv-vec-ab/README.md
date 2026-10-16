# Q8 KV direct-vector attention, 50K input

The same `strata-moe.exe` SHA-256 (`66e1abcfdacb8a7089a62298d0b6b67ffddfb8421633d767d3c52c00a1352f0b`) ran two sequential requests per arm on GPU 0, an RTX 2080 Ti with 22,528 MiB VRAM. The target was one `Qwen3.6-35B-A3B-UD-IQ3_S.gguf` mixed-quantization file (SHA-256 `66a3ca88...e89703ec66`) with a separate Q8_0 MTP head. Both arms advertised 262,144 context cells; every run used the same uncached 50,000-token prompt and generated 256 tokens with MTP draft length 2, batch 2,048, and Q8_0 KV. The variant enabled `STRATA_A3B_Q8_KV_VERIFY_VEC=1`; the baseline did not. Other config arguments were identical. The arm output hashes were stable within each arm but differed across arms, so MTP acceptance also changed. Results are medians of two runs, not a broad performance distribution.

| Arm | HTTP first byte | First model token | Prefill | Output | GPU 0 observed peak | MTP accepted/offered |
|---|---:|---:|---:|---:|---:|---:|
| Baseline batched F16 attention after Q8 KV conversion | 14.52 ms | 28.533 s | 1,763.2 tok/s | 64.9 tok/s | 20,513.5 MiB | 141/227 |
| Opt-in singleton Q8 direct-vector attention | 19.68 ms | 28.529 s | 1,761.3 tok/s | 43.8 tok/s | 20,499 MiB | 133/243 |

The direct route reduced observed output speed by **21.1 tok/s (32.5%)**, with no meaningful first-token or VRAM improvement. It remains opt-in and is not recommended for 50K decode. The likely cause is repeated KV scanning for each token in an MTP verification window, compared with the batched Tensor Core path. That mechanism is an inference from the code and these measurements; an output-node trace is still needed to attribute each CUDA kernel in the variant. `native_attention_graphs` alone counts a graph hook before CUDA kernel selection and does not establish that a particular vector kernel ran.

`arms/{baseline,vec}/run*.json` contain individual timing, resource, hash, and acceptance measurements. `arms/*/metadata.json` and `identity.json` contain exact binary, model, config and prompt identities. GPU memory is total GPU 0 usage and may include other processes; it is not per-process attribution. HTTP first byte is an early response byte, **not** the first generated model token.
