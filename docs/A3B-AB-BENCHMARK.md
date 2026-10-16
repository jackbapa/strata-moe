# Same-model A3B A/B measurement

`tools/bench_a3b_ab.py` measures an already-running local IQ3_S service. It
does not start, stop, or reload the model. Use the same GPU 0, target GGUF,
MTP GGUF, 262,144-token context, and fixed 50,000-token/256-output request for
both arms. The tool rejects a process whose command line or A3B environment
does not match its supplied config. It hashes the entire target GGUF, MTP GGUF,
binary, config, and fixed request. The first run saves the request in the suite;
every later arm reuses those exact messages and checks the tokenizer count.
The comparison rejects different executable hashes, and each run rejects
undeclared `STRATA_A3B_*` or `GGML_CUDA_*` switches in the engine environment.

After starting the baseline service separately:

```powershell
python tools/bench_a3b_ab.py run --suite bench/results/iq3s-ab --arm baseline --config strata-a3b-iq3s-q6-grouped.json --repeats 1
```

After switching the service to a Strata-native config that uses the **same
IQ3_S and MTP files**:

```powershell
python tools/bench_a3b_ab.py run --suite bench/results/iq3s-ab --arm strata --config strata-a3b-iq3s-strata-native.json --repeats 1
python tools/bench_a3b_ab.py compare --suite bench/results/iq3s-ab --baseline baseline --variant strata
```

Repeat in alternating order by invoking `run` again with the same arm name;
it appends `run002.json`, `run003.json`, and so on. A same-arm invocation
requires unchanged config and binary hashes. A new executable build needs a
new arm name. `--prepare-only` prepares the exact request and hashes without
inference. Pass `--engine-pid` when more than one matching engine runs.

Each trial records the first HTTP socket byte, the first nonempty streamed
content/reasoning delta, engine prefill and output rates, client-observed output
rate, offered/accepted MTP drafts, output hash, and the model and process
identities. Resource samples every 0.5 seconds record total GPU 0 memory,
engine RSS/private memory, and system RAM. These are **observed** peaks, not
allocator high-water marks; total GPU 0 memory can include other processes.
The first model delta excludes role/progress SSE messages. The client rate can
be affected by MTP burst emission, so use the engine's final output rate as
the primary throughput measure.

The new engine log format gives `A3B Strata grouped CUDA expert calls: N;
request prefill P decode D`. The tool stores `P` and `D` as the actual request
counts. Older logs contain only process-cumulative `N`; for those the request
count is reported as unavailable. The tool rejects a trial without the engine's
matching request performance line, nonzero cached prompt tokens, or an output
count other than the requested 256. Keep other GPU workloads quiet while
measuring, because the GPU memory counter covers the entire card. One run per
arm is exploratory; alternate and repeat before interpreting small differences
in speed or memory.
