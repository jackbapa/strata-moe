# A3B Windows memory audit, 2026-10-14

This is the initial executable before the current memory/performance changes. The executable copy is ignored by Git; its SHA-256 is recorded in `provenance.json`. The unrelated WSL service on GPU 1 was left running throughout. Each audit process was started separately on GPU 0 and exited through `QUIT`; GPU 0 returned to 100 MiB after both runs.

Hardware: Threadripper 3960X, 128 GiB RAM, RTX 2080 Ti GPU 0 reporting 22 GiB, Windows WDDM. Model: Qwen3.6-35B-A3B-UD-Q5_K_M, unchanged quantization. Context 4,096. The request uses nine input tokens and four greedy output tokens; this is a memory test, not a throughput benchmark.

| After generation | No GPU expert cache | Automatic cache under 16 GiB GPU budget |
|---|---:|---:|
| Process working set | 46.791 GiB | 46.800 GiB |
| Process private commitment | 25.662 GiB | 39.009 GiB |
| Estimated resident private regions | 22.464 GiB | 22.470 GiB |
| Estimated resident mapped regions | 24.126 GiB | 24.126 GiB |
| PAGE_READWRITE + PAGE_WRITECOMBINE commitment | 2.501 GiB | 15.815 GiB |
| GPU 0 memory including background | 2,598 MiB | 16,248 MiB |

`VirtualQueryEx` identifies one read-only mapped region of 26,456,195,072 bytes, the page-rounded model file, plus a 23,767,027,712-byte private region matching the routed expert allocation. `QueryWorkingSetEx` samples one page every 64 KiB and estimates approximately 24.13 GiB of mapped model pages remain resident alongside approximately 22.47 GiB of private pages. This explains the large working set even with the GPU expert cache disabled. Disabling the loader's model mmap is a concrete follow-up to test; these baseline measurements do not claim that change has already been measured.

Allocating the GPU expert cache adds approximately 13.35 GiB to private commitment but essentially nothing to the working set. The increase appears in write-combined private regions, consistent with CUDA/WDDM allocation bookkeeping. It should not be counted as another 13.35 GiB of resident RAM in these runs.

The `mapped_files` diagnostic from psutil 7.2.2 reports 686,391,296 bytes for the GGUF. This equals the actual mapped region size modulo 2^32 and contradicts the 64-bit region and working-set queries. Do not use that field to conclude that only 654 MiB of model pages are resident. `Process.memory_info()` working set and the explicit Windows queries provide the evidence used here.

Full snapshots, commands, return codes and engine protocol output are in `cache0.json` and `cache16g.json`; `summary.json` contains the compact comparison. `tools/inspect_a3b_memory.py` reproduces the two sequential launches or inspects a specified PID read-only.
