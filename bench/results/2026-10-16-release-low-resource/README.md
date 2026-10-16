# Strata-Moe low-resource release checks (2026-10-16)

Machine: Windows, AMD Ryzen Threadripper 3960X (24 cores / 48 threads), 128 GiB RAM, three NVIDIA RTX 2080 Ti cards. The driver reports 22,528 MiB total for each card. GPU 1 was otherwise idle at 181 MiB used. Both modes used the **same** Unsloth Qwen3.6-35B-A3B UD-IQ3_S file (SHA-256 `66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66`), no MTP, the same seven input token IDs for `Explain mixture of experts briefly.`, greedy sampling, 128 output tokens, 1,024 context cells, batch 32, and a resident engine process. The process was started anew for each request; timing began **after READY**, when the `GEN` request was sent. Model startup is recorded separately. Three complete trials per mode; no trial was omitted.

| Mode | Engine `--vram-mib` | First model token, trials (s) | Output, trials (token/s) | Whole GPU 1 sampled peak |
| --- | ---: | --- | --- | ---: |
| GPU 1 trunk + CPU uncached experts | 2,048 | 0.207, 0.200, 0.204 | 19.94, 20.10, 20.06 | 2,060 MiB; +1,879 MiB over 181 MiB idle |
| CPU only, 16 threads | no GPU | 0.181, 0.168, 0.182 | 11.78, 11.95, 11.96 | 181 MiB idle; +0 MiB |

The fastest completed 2 GiB-budget trial is **20.10 token/s** with **0.200 s** to first token. The CPU-only peak is **11.96 token/s** with **0.182 s** to first token; the lowest CPU first-token latency was **0.168 s** on trial 2. The numbers in README's headline table are maxima/minima **within the measured runs**, so those two columns may come from different trials. Medians are 20.06 and 11.95 token/s, respectively. Process RSS peaks were **12,041 MiB** in 2 GiB mode and **13,231 MiB** on CPU only (the whole-system used-memory peaks were about 37.9 and 39.0 GiB, respectively, including the other resident service). The engine's GPU 1 startup allocation was about 1,690 MiB, with the expert cache deliberately set to zero. This is **not a physical 2 GiB GPU test**. The 2,048 MiB limit is the engine's planner budget, not a hard cap on the whole card's `nvidia-smi` reading; the observed whole-card 2,060 MiB includes the pre-existing 181 MiB baseline and driver allocations.

Reproduce from the repository root after building `build-a3b` and `build-a3b-cpu` and downloading the IQ3_S model:

```powershell
.\.venv\Scripts\python.exe bench\a3b_release_low_resource.py
```

The harness in [`bench/a3b_release_low_resource.py`](../../a3b_release_low_resource.py) records the exact commands, `DONE` timing values, request-to-first-token timing, startup-to-ready time, process RSS, whole-system memory, and GPU 1 samples in [`resident-results.json`](resident-results.json), with engine stderr in `2g-*.txt` and `cpu-*.txt`. The first model token is the first `T <id>` emitted by the resident engine, not the HTTP response first byte. Resources are sampled at 200 ms intervals; brief unsampled spikes are possible. No claims about a real 2 GiB GPU, 50K prompt, 256K context, MTP, model quality, or concurrent requests follow from this short-prompt check. The full-GPU 50K/256 data is separately documented [here](../2026-10-16-a3b-iq3s-combined-exact-r3/README.md).
