# Tuned Q8 resident API measurements — 2026-10-15

The selected mask-reuse plus short-prefill CPU-share cohort reaches
**50.1–65.9 output token/s**. Short input32 first output falls to **0.352 s**;
input2048 remains **537.8 prefill token/s and 3.832 s to first output**.
**100 output token/s and 1500 prefill token/s have not been reached.** This
report separates two twelve-request cohorts and their closed manifests.
Fused MTP rounds are **off** in both. The mask-only cohort below has CPU sharing off.

## Selected cohort: mask reuse plus short-prefill CPU sharing

[Plan](measurements-share/plan.json), [audit PASS](measurements-share/audit.json),
[all trials](measurements-share/case-summary.json). The executable SHA and all
mathematical settings match the mask-only cohort, with the single additional
`--prefill-cpu-share` flag. Engine PID40412; config SHA256
`cc592e88bbc90a5f89c2ac968170857b0980da30fee3ac20d036e70ec7c8395f`.
It is a separate fresh process with zero prior measured requests. Every row
includes all three trials; all three texts within each case match.

| Case | Input tokens | Output tokens, all trials | TTFB ms | First output s | Engine output token/s | Client output token/s | Prefill token/s |
|---|---:|---|---:|---:|---:|---:|---:|
| Heldout Python BFS code | 160 | 512/512/512 | 11.42 | **0.699** | 65.90 | 65.90 | 238.90 |
| Heldout Chinese transactions | 123 | 328/328/328 | 12.05 | **0.584** | 61.40 | 61.79 | 222.70 |
| Neutral short input | 32 | 256/256/256 | 10.86 | **0.352** | 52.30 | 52.15 | 98.40 |
| Neutral long input | 2048 | 256/256/256 | 11.02 | 3.832 | 50.10 | 50.09 | 537.80 |

BFS first request TTFT/output: 0.925s/65.9 token/s; subsequent-two medians:
0.696s/65.95. Short first: 0.356s/52.3; subsequent-two medians: 0.351s/52.15.
Long first: 3.989s/50.2; subsequent-two medians: 3.828s/50.1. Startup remains
outside request timing. The shared configuration is preferred for this machine's
short-input latency; it does not improve every output-rate median.

Compared with mask-only, first output improves from1.312→0.699s (BFS),
1.229→0.584s (Chinese), and0.813→0.352s (short). Long input2048 exceeds the
9–256-token sharing range and has0 prefill-share calls; its timing stays close
to mask-only. Short output rate changes53.0→52.3, long50.2→50.1 token/s.

CPU/GPU placement changes during prefill, so off/on output equality is not
promised. [Actual output comparison](measurements-share/mask-output-comparison.json)
shows all nine BFS/Chinese/short files differ from mask-only, while all three
long files match byte for byte. Chinese output length changes310→328; decode
speed is therefore not measured on identical continuations for that case.
Independent [numeric/full-model validation](../../../docs/A3B-TUNED-VALIDATION.md)
passed before this cohort was started. The new CPU share keeps original Q8 bytes
and native CPU arithmetic; it is not further quantization or skipped experts.

| Case | Prefill share calls per trial | CPU jobs /expert groups per trial | Prefill expert upload MiB, mask→share | Pooled MTP accepted/offered |
|---|---:|---|---|---|
| BFS | 40 | 1435 /1051 | 6237.9→2894.2 | 999/1062 (94.07%) |
| Chinese | 40 | 1479 /1100 | 5877.8→2336.4 | 558/849 (65.72%) |
| Short | 40 | 1207 /914 | 3958.9→1016.8 | 405/717 (56.49%) |
| Long | 0 | 0 /0 | 11796.9→11796.9 | 396/729 (54.32%) |

These prefill CPU jobs remain separate from the decode pool assertion. Every
request still has nonzero CPU/GPU decode rows, pool decode jobs×3=CPU rows,
decode route readbacks0, decode expert-weight uploads0.0 MiB and decays0.

Share-cohort resources:82 one-second samples, no errors; maximum sampled engine
physical RSS **34.1599 GiB**, OS lifetime working-set peak **34.1959 GiB**, final
RSS **34.1538 GiB**. GPU0 sampled maximum/final **22057 MiB**, GPU1 display51 MiB,
GPU2 zero. System RAM sampled maximum50.3060 GiB, final50.2801 GiB, available
77.5906 GiB. Final snapshot UTC2026-10-14 17:20:55.582516 /local01:20:55.
GPU values are sampled maxima, not continuous peaks; private commitment is not RAM.

Its [closed manifest](measurements-share/evidence-manifest.json) independently
hashes46 original artifacts, excluding later live-log appends and derived reports.
Temperature0.7 supplementary requests belong to another directory and are not
included in either greedy cohort's medians.

## Default web sampling: separate temperature0.7 subset

The same resident share service then completed three short and three long
requests using **temperature0.7, top_p0.8, top_k20**, seed12345, thinkingfalse,
maximum256 output tokens. These conditions match the configured web sampling
defaults. They are separate from the temperature0 greedy tables; **65.9 token/s
is the greedy BFS result, not the measured default web speed**.

| Input tokens | Output, all trials | TTFB ms | First output s | Engine output token/s | Client output token/s | Prefill token/s |
|---|---|---:|---:|---:|---:|---:|
| 32 | 256/256/256 | 1.28 | **0.335** | **47.00** | 46.91 | 100.80 |
| 2048 | 256/256/256 | 11.40 | **3.783** | **47.40** | 47.36 | 546.80 |

All three trials enter each median. Short first TTFT is0.592s, subsequent-two
median0.328s; long first3.769s, subsequent-two median3.784s. Pooled MTP
acceptance is390/750 (52.00%) for short and399/726 (54.96%) for long.
This seeded subset does not evaluate sampling diversity or broad quality.

Its [independent six-request audit](measurements-temp07/audit.json) is PASS:
request SHA reconstruction checks temperature/seed/template, timestamped SSE
reproduces text/TTFT/client rate, usage matches DONE-derived timings, every
trial enters the summary, and selected engine log counters match. A separate
engine snapshot selects the six requests after the twelve greedy share requests;
the closed greedy snapshots/manifests remain unchanged. The supplement has its
own25-file [evidence manifest](measurements-temp07/evidence-manifest.json).

The generic supplement runner did not independently fingerprint its process or
sample RAM/VRAM; [service context](measurements-temp07/service-context.json)
comes from the preceding recorded share cohort on the same process. Resource
peaks above belong to the greedy share matrix, not this six-request subset.

## Delivery state

The root agent selected the share settings in `strata-a3b-q8.json`; its args/env
match the still-resident experiment service except the future log path. The
web defaults remain temperature0.7/top_p0.8/top_k20. No restart or idle unload
was performed for delivery. At UTC2026-10-14 17:25:41 (local01:25:41), the
[delivery snapshot](delivery.json) records PID40412, loaded health, physical
RSS34.1478 GiB and GPU0 current22059 MiB. This later current snapshot is distinct
from the earlier sampled matrix maximum22057 MiB.

The earlier GPU-only Q8 config is preserved byte for byte as
`strata-a3b-q8-gpu-compatible.json`; Q4 config/binary hashes remain unchanged.
The delivery file records those hashes and the empty original-Strata tracked
diff. [WSL final state](wsl-final-state.json) reports no running distribution.
These files are outside all closed measurement manifests.

## Completed cohort: attention mask reuse, fixed K2

Evidence: [plan](measurements-mask/plan.json), [audit PASS](measurements-mask/audit.json),
[all trials/first/subsequent summaries](measurements-mask/case-summary.json).
Each row is the median of all three requests, including the first. Inputs
are actual chat-template-inclusive token counts; natural prompts are not padded.

| Case | Input tokens | Output tokens, all trials | TTFB ms | First output s | Engine output token/s | Client output token/s | Prefill token/s |
|---|---:|---|---:|---:|---:|---:|---:|
| Heldout Python BFS code | 160 | 512/512/512 | 11.21 | 1.312 | **65.60** | 65.57 | 124.60 |
| Heldout Chinese transactions | 123 | 310/310/310 | 11.20 | 1.229 | **58.80** | 58.72 | 102.50 |
| Neutral short input | 32 | 256/256/256 | 11.83 | 0.813 | **53.00** | 52.90 | 40.70 |
| Neutral long input | 2048 | 256/256/256 | 11.86 | 3.811 | **50.20** | 50.15 | 542.10 |

BFS first request TTFT/output is 1.615 s/65.6 token/s; subsequent-two medians
are 1.307 s/65.6. Long input first request is 4.035 s/50.2; subsequent-two
medians are 3.804 s/50.2. The table is not three selected warm requests.
There are no earlier measured requests in this process, no model reloads,
no concurrent inference and no excluded inference warmup inside the cohort.

## Identity and hardware

Frozen executable: `build-a3b-q8-tuned/bin/strata-moe.exe`, SHA256
`2da62666608dc8c9005e654bb3d9c370d400b508e6ee5c4f509046ec4801ec62`.
Config SHA256: `12918a76b2f05401374d20c8160f951083bea7cb604c16971649a4dff4bfb8ac`.
Engine PID 25792. The full config and actual process command/environment are
saved in the cohort; the server stays on 127.0.0.1:8096 with idle unload disabled.

Hardware: Threadripper 3960X (24 physical cores), approximately 128 GiB RAM,
GPU 0 RTX 2080 Ti (22528 MiB reported memory, SM75, PCIe 3.0 x16). Model work
uses GPU 0 plus an eight-participant CPU expert pool. GPU cache hits compute
on GPU; nonresident experts in decode compute on CPU through the mailbox.
This is the original-style CPU/GPU expert design requested after GPU-only tests.

The target remains the official Qwen3.6-35B-A3B Q8_0 GGUF, without local
requantization, with all 120 routed expert tensors Q8_0. Target SHA256 is
`d8d7842cc657d720f39546878e431937c84473aa130c36371669ac17c80c7361`.
The 32640 MiB expert weights exceed GPU memory; this is not full-Q8 GPU residency.
The separate original Q4 configuration/executable remain preserved.

Settings: context 4096, main batch 2048, draft microbatch 512, fixed MTP K2,
exact verification, native exact dense/attention, GPU embedding/sampling,
shared MTP weights, deferred catchup, async expert events, pageable experts,
staging 8, reserve 384 MiB, verified broad profile, doorbell, PCIe miss fraction 0
and cache decay 0. SHA-verified shared MTP GPU tensors save 1030.6 MiB.

Actual process environment:

```text
CUDA_VISIBLE_DEVICES=0
STRATA_A3B_ARGMAX_MULTI=1
STRATA_A3B_DENSE_Q8_REUSE=1
STRATA_A3B_EXPERT_Q8_REUSE=0
STRATA_A3B_NATIVE_EXPERT_REUSE=0
STRATA_A3B_ATTENTION_MASK_REUSE=1
```

All requests set temperature 0, seed 12345, thinking false. The configured web
sampling defaults are not used in this matrix. Neither `--prefill-cpu-share`
nor `--mtp-fused-rounds` is enabled; corresponding optional counter lines are absent.

## Same-day comparison and output identity

The [previous fixed-K2 cohort](../2026-10-15-a3b-q8-doorbell-api/README.md)
used executable SHA `465345b125e6cd5907ebd0b544c0e57b4172a3a26ea28f48103c263907795891`.
Both have batch 2048, CPU 8, the same model/profile and the same 5188 cached
experts/16536.8 MiB. This new executable also contains disabled prefill/fused
code. Consequently the following is a descriptive configuration comparison,
**not a pure mask-reuse causal A/B**.

| Case | Previous output token/s | New output token/s | Previous first output s | New first output s |
|---|---:|---:|---:|---:|
| BFS | 63.4 | 65.6 | 1.318 | 1.312 |
| Chinese | 57.0 | 58.8 | 1.235 | 1.229 |
| Short | 51.7 | 53.0 | 0.811 | 0.813 |
| Long | 48.8 | 50.2 | 3.834 | 3.811 |

All twelve actual `.output.json` files were independently hashed and are byte
identical to their corresponding old K2 outputs; the [derived comparison](measurements-mask/baseline-output-comparison.json)
records each SHA. MTP counts and CPU/GPU decode rows also match the old K2 trials.
Within this cohort each case's three output texts match. BFS reaches its
512-token limit, so complete/correct code is not established. Chinese stops
naturally at 310 tokens. The two natural tasks are absent from the five profile
training prompt texts; two tasks do not establish broad quality or generalization.

## Actual expert work and MTP

These counts pool all three trials. Projection rows count gate/up/down separately;
they do not cover all dense trunk/draft GPU operations.

| Case | CPU projection rows | GPU projection rows | Runtime GPU share | MTP accepted/offered | Acceptance |
|---|---:|---:|---:|---|---:|
| BFS | 401625 | 1133415 | 73.84% | 999/1065 | 93.80% |
| Chinese | 161478 | 1039482 | 86.55% | 513/834 | 61.51% |
| Short | 259200 | 751680 | 74.36% | 414/702 | 58.97% |
| Long | 266742 | 787338 | 74.69% | 396/729 | 54.32% |

The printed profile coverage 83.54% is **training routing-score coverage**,
not measured runtime hit rate. All twelve requests exercise both CPU and GPU
decode experts and satisfy pool jobs ×3 = CPU projection rows. Doorbell/static
calls match; **decode route readbacks 0, decode expert-weight upload 0.0 MiB,
decode cache decays 0**. Mailbox activations/results still cross PCIe. Prefill
fallback uploads expert weights, including 3958.9 MiB on the short input;
zero decode expert upload does not mean zero whole-request transfer.

## Physical RAM, VRAM and startup

There are 87 samples at one-second intervals, including before/after snapshots,
with no collector errors. Maximum sampled physical engine RSS is **34.1517 GiB**;
Windows process-lifetime physical working-set peak **34.1874 GiB**; final RSS
**34.1457 GiB**. Maximum sampled system RAM use is **50.2498 GiB**. The final
snapshot has the same system use, with **77.6209 GiB** available of 127.8707 GiB.

GPU 0 maximum sampled and final memory is **22057 MiB**; GPU 1 has 51 MiB display
use with 0% final utilization; GPU 2 has zero use. GPU maxima are periodic
samples, **not continuous instantaneous peaks**. Lifetime working-set peak
includes startup; private/pagefile commitment is not added to physical RAM.
Final snapshot: UTC 2026-10-14 17:04:46.675607 / local 2026-10-15 01:04:46.

Profile full-file hash verification takes 35.940 s and startup graph warmup
3.873 s, outside resident request timing; warmup user KV is cleared. Native
mailbox extra mapped RAM is 40.0 MiB and shared GPU scratch 204.3 MiB, without
a second complete expert-weight copy.

## Audit and further cohorts

Both independent greedy file audits are PASS: complete ordered twelve-request matrices,
all-trial medians, timestamped SSE output reconstruction, usage/token/MTP/DONE-derived
timing agreement, fixed-profile counter ownership, optional feature presence,
actual executable/environment identity and resource sample maxima. The closed
manifest hashes exactly 46 original artifacts (10 base + 3 per trial); derived
audit/report/comparison files and future cohort logs are excluded.

Prefill CPU sharing is off in the mask-only cohort and on in the independently
audited selected cohort above, after its GPU numeric fixtures passed. The full
[numeric validation table](../../../docs/A3B-TUNED-VALIDATION.md) includes54 CPU
bitwise cases,18 CUDA task-layout cases, sanitizer0 errors, full-Q8/MTP strict
rollback and output boundaries,7 CPU CTest passes/6 explicit GPU skips, and33
Python option checks. Earlier failed diagnostics remain linked and preserved.
Future raw results must use another measurement subdirectory and must not
overwrite this manifest or pool requests into these medians.

## Retained test method

This independent directory retains the same twelve requests and timing definitions
as the earlier doorbell matrix: heldout BFS code and Chinese transaction explanation
(three requests each, maximum512 output tokens), neutral input32/output256 ×3 and
input2048/output256 ×3. Every trial enters its case median. The earlier directory,
runner and closed evidence manifests remain unchanged.

The root agent supplied the frozen executable, SHA256 and separate configuration
and ran the service/tests. The runner requires an explicit
`--expected-exe-sha256`, supports `--config`, `--out`, `--url`, and only accepts
`--phase all` for this complete cohort. It never loads/reloads/unloads a model.
The fixed K2/batch 2048/CPU 8/attention-mask-reuse 1 results are listed above.
Prefill CPU sharing passed its separate GPU numeric fixtures before activation.
Fused MTP rounds remain off unless explicitly configured and validated.

```powershell
python bench/results/2026-10-15-a3b-q8-tuned-api/run-latency.py --config <root-config.json> --out <new-directory> --url http://127.0.0.1:8096 --expected-exe-sha256 <frozen-SHA256> --phase all
python bench/results/2026-10-15-a3b-q8-tuned-api/audit.py --results-root <completed-directory>
python bench/results/2026-10-15-a3b-q8-tuned-api/test_offline.py
```

Plans include the actual process environment, including
`STRATA_A3B_ATTENTION_MASK_REUSE`, config/executable/helper fingerprints and optional
feature flags. Timestamped SSE frames, readable SSE, output text, TTFB, first output,
engine/client output rates and prefill counts remain unchanged. Once-per-second RSS
and all-GPU memory samples distinguish sampled maxima from continuous peaks and
Windows lifetime working-set peak from resident request samples.

Optional `A3B prefill CPU share` statistics are prefill-only calls/CPU jobs/experts.
They are separate from the strict decode CPU-pool jobs×3=CPU projection rows check.
Optional fused MTP round time is separate from draft/catchup time. Log presence must
match each feature's config flag. The audit still requires actual nonzero CPU/GPU
decode work, zero decode expert uploads, zero decode route readbacks, fixed-profile
ownership and engine DONE-derived SSE/usage/token/MTP agreement.

Tool preparation and file audit made no API/GPU requests. The root agent executed
the measured matrix. Eight offline tests, Python compilation and describe-only
checks passed; broader model-quality evaluation remains outside this report.
