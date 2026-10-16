# A3B IQ3_S draft-vocabulary head, clean 50K/256 comparison

16 October 2026, Windows, Threadripper 3960X, RTX 2080 Ti GPU 0 (22,528 MiB
reported), one model request at a time. This is the **same isolated engine
binary** (`a6020a08638e937963679a888084f21b7a8a78b94244695b778e6ce107738d46`)
with the same target IQ3_S GGUF (`66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66`),
256K configured context, 50,000 uncached input tokens, 256 output tokens,
temperature 0, seed 12345 and MTP K2. The arms differ only in the matching
Q8_0 MTP GGUF output head and its log path. Full MTP SHA is
`f6cb995fab78e7284097b2e13508e8daad4ffc6a747b433aef399f4e48524b5a`;
the A3B tokenizer-derived 194,208/248,320-row subset MTP SHA is
`7a0bd6c1a1c8d5037262c52b50375d063e128a254f5f6348a0772543893512ae`.
The target IQ3_S model and quantization did not change. The subset keeps the
full draft input embedding and copies selected Q8 output rows byte-exactly.

The first six-run exploratory suite in the adjacent
`2026-10-16-a3b-draft-vocab-50k-ab/` folder overlapped an isolated CUDA
build from 17:19:18 to 17:23:33 UTC. Its 84.2 versus 81.5 tok/s medians
are **not** used for a speed conclusion. Before this fresh six-run suite,
the restored 8096 service was stopped by its verified PID, GPU 0 returned to
99 MiB, and no `cl`, `nvcc`, `ptxas`, `ninja` or `cmake` process was running.
Other source work held heavy compiles and GPU tests until completion. The
runner alternated full/subset three times, started a fresh service per arm,
checked binary/model/config hashes, copied one request template, measured
each SSE stream and resources, and audited the complete MTP proposal log
after stopping each test engine. The selected 8096 service was restored in
`finally`, verified loaded at context 262,144. Raw trials and hashes are in
`full/`, `subset/`, `comparison.json`, and `runner-state.json`.

| Median of three | Full Q8 draft head | Subset Q8 draft head |
| --- | ---: | ---: |
| HTTP first response byte | 9.34 ms | 19.22 ms |
| First nonempty model token | 28.571 s | 28.663 s |
| Prefill | 1,757.0 tok/s | 1,751.8 tok/s |
| Output | 81.9 tok/s | 81.7 tok/s |
| Whole GPU 0 observed peak | 17,103 MiB | 16,993 MiB |
| MTP offered / accepted | 238 / 136 | 238 / 136 |

All six final output hashes and proposal-trace hashes are identical; each
request reports zero cached prompt tokens. The short real-model strict gate
also matched 16 final state files byte-for-byte, 29 proposal lines and eight
acceptance classes. This checks the sampled Chinese repetitive prompt, not
coverage or response quality for every language and task.

The full-head output rates in chronological order were 85.2, 81.9 and 81.6
tok/s; the subset rates were 83.7, 81.7 and 81.5. There is run-order drift.
Interpolating the full-head rate between adjacent full-head anchors gives
83.55 tok/s around subset run 1 and 81.75 around subset run 2; those subset
runs differed by +0.15 and -0.05 tok/s. The last subset has no following
full-head anchor. The subset therefore has **no demonstrated speed gain or
material first-token gain**. Its 110 MiB lower whole-card peak agrees with
the smaller draft output head but is not process-attributed VRAM. Keep this
version opt-in: it still fills and scatters logits back to the full
248,320-token vocabulary before sampling, unlike original Strata's compact
sampler. A true compact GPU-sampler experiment requires a separate derived
GGUF, independent strict tests and another same-binary 50K comparison.
