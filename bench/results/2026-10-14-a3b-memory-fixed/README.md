# Same-executable mmap A/B, 2026-10-14

Hardware and model match `../2026-10-14-a3b-memory-audit/README.md`: Threadripper 3960X, 128 GiB RAM, RTX 2080 Ti GPU 0 reporting 22 GiB, Windows WDDM, unchanged Qwen3.6-35B-A3B-UD-Q5_K_M. The unrelated WSL GPU 1 service remained running. Context 4,096; nine input and four greedy output tokens. All arms use the same newly built executable, a 16 GiB GPU budget and approximately 13,546.5 MiB of GPU expert cache. Each process was exited with `QUIT`, returning GPU 0 to 100 MiB background use.

| After generation | Default file loading | `--mmap` | `--pageable-experts` |
|---|---:|---:|---:|
| Process working set | 23.178 GiB | 46.800 GiB | 23.178 GiB |
| Process private commitment | 39.465 GiB | 39.004 GiB | 39.466 GiB |
| System available RAM | 27.529 GiB | 3.864 GiB | 27.518 GiB |
| GPU 0 memory including background | 16,248 MiB | 16,248 MiB | 16,248 MiB |

Turning off the loader's source-file mmap reduces resident process memory by **23.622 GiB** in this same-executable comparison. The mapped GGUF region disappears; the approximately 22.13 GiB expert allocation remains. This is a resident-memory improvement. Private commitment includes the GPU driver's nonresident allocation bookkeeping and is a different measurement.

Changing expert host memory to pageable does not provide a further resident-memory saving in this test. The four-token requests are too short to draw a throughput conclusion about pinning or mmap. Full 64-bit `VirtualQueryEx` region summaries and 64 KiB `QueryWorkingSetEx` residency samples are in `cache16g*.json`. `summary.json` collects the relevant figures; `provenance.json` records executable/source hashes.

Reproduce with:

```powershell
.venv\Scripts\python.exe tools\inspect_a3b_memory.py --model models\a3b\Qwen3.6-35B-A3B-UD-Q5_K_M.gguf --out bench\results\memory-retest --arms cache16g cache16g-mmap cache16g-pageable
```
