# First MTP timing run, 2026-10-14

This is an exploratory single trial per arm. Hardware: Threadripper 3960X, 128 GiB RAM, RTX 2080 Ti GPU 0 reporting 22 GiB, Windows WDDM. Model: Qwen3.6-35B-A3B-UD-Q5_K_M; separate Q8_0 MTP weights. Both arms use the same executable, prompt, 256-token output limit, 4,096 context and 16 GiB GPU budget. The unrelated WSL service on GPU 1 was left running. GPU 0 returned to 100 MiB after the processes exited.

| Metric | Cached experts, no MTP | MTP, three draft tokens |
|---|---:|---:|
| Output tokens | 256 | 256 |
| Decode seconds | 10.203 | 8.285 |
| Decode tokens/s | 25.09 | 30.90 |
| Prefill seconds | 2.112 | 2.014 |
| Peak process working set | 23.183 GiB | 23.778 GiB |
| Peak GPU 0 usage including background | 16,248 MiB | 16,348 MiB |
| Expert cache allocated | 13,546.5 MiB | 11,864.2 MiB |

The MTP arm accepted 164 of 270 proposed tokens (60.74%) in 90 verification windows, producing 2.84 output tokens per window. Draft computation took 0.922 s and draft-state catchup took 0.2774 s. The measured decode rate is 23.2% higher for this one prompt and trial; it is not a general throughput claim.

The first target-token logits match exactly between the two arms. Generated text differs after the first 100 Unicode characters. The full-output hash comparison is therefore false; this timing test alone does not establish token-for-token equivalence or isolate numerical differences from a speculative-decoding defect. The MTP model also reduces the GPU expert cache, changing CPU/GPU expert placement. Further correctness validation is required before treating this result as validated MTP acceleration.

`results.json` records commands, executable SHA-256, host/GPU memory measurements, output hashes and first-token comparison. The stdout and stderr files preserve generated text and the complete engine logs. Float32 logits files are ignored by Git.
