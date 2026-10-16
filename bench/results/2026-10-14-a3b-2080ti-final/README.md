# Qwen3.6 A3B on one RTX 2080 Ti with a 16 GiB planning budget

See [the validation record](../../../docs/A3B-VALIDATION.md) for hardware, method, limits and results.
The physical card reports 22 GiB; no physical 16 GiB card was tested.

Reproduce the three arms from the project root:

```powershell
.venv/Scripts/python.exe tools/bench_a3b.py --model models/a3b/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf --out bench/results/a3b-local --predict 96
```

`results.json` contains timings and one-position logit comparisons. `provenance.json` records hashes.
The `.f32` logits are local generated data and are ignored by Git. The stdout/stderr files are the recorded runs.

The HTTP smoke test uses an already-running local server:

```powershell
.venv/Scripts/python.exe tools/test_a3b_http.py --out bench/results/a3b-local/api-smoke.json
```
