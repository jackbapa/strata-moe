# GPU static/dynamic API experiment preparation

Prepared only. No API test was run from this directory. Static/dynamic speed
experiments are paused while the CPU/GPU expert path is investigated. No number
from the frozen Q8 baseline is a measurement of these prototypes.

`prepare.py` copies the production Q8 configuration into two independent files,
without changing the source configuration. Both use GPU 0, GPU expert math,
the stable broad profile, resident loading, MTP adaptive K6, batch 512,
`build-a3b/bin/strata-moe.exe`, cache decay 0 and reserve 640 MiB. The engine
separately reserves the CUDA static operator's shared scratch. The second config
adds the experimental GPU-managed dynamic cache. They have separate engine logs.

Configs already prepared: `config-static.json` and `config-dynamic.json`.
Their source/config/profile digests are in `preparation.json`. Preparing again
requires a new `--out` directory to preserve this evidence.

When the root agent explicitly resumes this experiment, run one configuration
at a time, starting the service separately. Example commands from the repository
root (not executed here):

```powershell
python -m serve.server --engine strata --config bench/results/2026-10-14-a3b-q8-static-api/config-dynamic.json --port 8096
```

After health reports the model loaded, in another terminal:

```powershell
python bench/results/2026-10-14-a3b-q8-static-api/run-latency.py --config bench/results/2026-10-14-a3b-q8-static-api/config-dynamic.json --out bench/results/2026-10-14-a3b-q8-static-api/measurements-dynamic --phase all
```

Use `config-static.json` with a separate `--out` for the fixed-cache mode.
`--engine-pid` can disambiguate the matching engine process. The runner never
starts, loads, unloads, reloads or stops a model. It refuses a nonempty result
directory. `--describe-only` inspects the plan without HTTP/process/GPU access.

The nine requests are BFS Python code ×3, Chinese database isolation explanation
×3, then neutral input 32/output 256 ×3. These use the same heldout prompts as
the frozen Q8 heldout report, greedy sampling, seed 12345, thinking disabled.
Every repeat is included in its case median; first and later resident requests
remain individually labeled. Separate fresh processes and prototype modes must
not be pooled with the frozen Q8 results.

The runner saves raw SSE, output text, usage, TTFB, first output token latency,
engine prefill/decode rates, client output rate, executable/config/prompt hashes,
the complete engine log and per-request CPU/GPU/route/transfer/MTP counters.
It checks positive static-op calls, zero CPU expert rows, zero decode route
readback, and matching API/log token counts. Resource samples record all GPUs,
process physical working set, Windows process-lifetime working-set peak and
system RAM at one-second intervals. Sampled GPU maxima are not instantaneous
continuous peaks; startup is outside resident-request timing.

Offline verification: Python compilation and `--describe-only` passed. Actual
service behavior, latency, resource use and the 100 token/s goal remain untested
for these configurations.
