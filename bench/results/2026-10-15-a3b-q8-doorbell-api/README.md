# Q8 CPU/GPU mailbox API measurements — 2026-10-15

Fixed MTP K2 with batch 2048 is the selected configuration: **48.8–63.4 output token/s**, and **538.8 prefill token/s / 3.834 s to first output** on the 2048-token input. **100 output token/s and 1500 prefill token/s have not been reached.** Earlier GPU-only, ordinary hybrid and static/dynamic experiments remain separate.

## Hardware, model and settings

Hardware: Threadripper 3960X (24 physical cores), approximately 128 GiB RAM, GPU 0 RTX 2080 Ti (22528 MiB reported, SM75, PCIe 3.0 x16). Only GPU 0 performs model work. GPU 1 has 51 MiB display use and GPU 2 has zero use in these samples.

The target is the official `ggml-org/Qwen3.6-35B-A3B-GGUF` Q8_0 model, without local requantization. All 120 routed expert tensors are Q8_0, totaling 32640 MiB, so the whole Q8 model does not reside in GPU memory. Target SHA256: `d8d7842cc657d720f39546878e431937c84473aa130c36371669ac17c80c7361`. The original Q4 configuration and frozen executable are preserved separately.

Both cohorts use `build-a3b-doorbell-compatible/bin/strata-moe.exe`, SHA256 **`465345b125e6cd5907ebd0b544c0e57b4172a3a26ea28f48103c263907795891`**. Each uses a fresh process, then keeps the model resident for twelve requests, with no reload, concurrency, idle unload or excluded inference warmup. Both plans record zero earlier measured requests.

Common settings: context4096, main batch2048, draft microbatch512, exact verification, native exact dense/attention, GPU embedding/sampling, shared MTP weights, deferred catchup, async events, pageable experts, CPU8, staging8, reserve384 MiB, verified broad profile, `--hybrid-doorbell`, PCIe miss fraction0 and cache decay0. GPU computes cache hits; the CPU native pool computes misses. Short-window CUDA operations exchange activations/results through the mailbox, with no layer-by-layer route readback. SHA-verified MTP shared tensors save 1030.6 MiB GPU memory.

Actual process environments: `CUDA_VISIBLE_DEVICES=0`, `STRATA_A3B_ARGMAX_MULTI=1`, `STRATA_A3B_DENSE_Q8_REUSE=1`, `STRATA_A3B_EXPERT_Q8_REUSE=0`, `STRATA_A3B_NATIVE_EXPERT_REUSE=0`. API requests explicitly use temperature0, seed12345 and thinkingfalse; web sampling defaults were not used.

## Fixed K2: selected resident matrix

[Plan](measurements-batch2048/plan.json), config SHA256 `aeeef87363474ae34507ccbf273280928e55afbe033c77cf357b272f94e5cccb`, PID31924. Each row is the median of **all three** requests, including the first. Actual inputs include the applied chat template.

| Case | Input | Output, all trials | TTFB ms | First output s | Engine output token/s | Client output token/s | Prefill token/s |
|---|---:|---|---:|---:|---:|---:|---:|
| Heldout Python BFS code | 160 | 512/512/512 | 11.62 | 1.318 | **63.40** | 63.39 | 124.30 |
| Heldout Chinese transactions | 123 | 310/310/310 | 10.57 | 1.235 | **57.00** | 56.92 | 102.10 |
| Neutral short input | 32 | 256/256/256 | 10.83 | 0.811 | **51.70** | 51.53 | 40.70 |
| Neutral long input | 2048 | 256/256/256 | 11.62 | 3.834 | **48.80** | 48.70 | 538.80 |

BFS first-request TTFT/output: 1.536s/63.1 token/s; subsequent-two medians: 1.315s/63.6. Long-input first: 4.032s/48.8; subsequent-two medians: 3.821s/48.6. [All values](measurements-batch2048/case-summary.json) are retained; these are not three selected warm requests.

Within this cohort, all three texts per case are identical. BFS reaches its 512-token limit, so this does not establish complete or correct code. Chinese stops naturally at310. The two natural tasks are absent from the five profile training prompt texts, but are only a limited heldout sample, not broad quality or profile generalization evidence.

## Adaptive up to K6: separate resident matrix

[Plan](measurements-k6-adaptive/plan.json), config SHA256 `8c7923500946d374ca359753f5cde8d85ae4a378f09b430b76aabbf4bc09db13`, PID21356. Same binary/batch/environment, adaptive MTP capped atK6. All three trials enter each median.

| Case | Input | Output, all trials | TTFB ms | First output s | Engine output token/s | Client output token/s | Prefill token/s |
|---|---:|---|---:|---:|---:|---:|---:|
| Heldout Python BFS code | 160 | 512/512/512 | 10.95 | 1.355 | 61.00 | 60.91 | 120.80 |
| Heldout Chinese transactions | 123 | 336/336/336 | 11.46 | 1.250 | 51.20 | 51.52 | 100.80 |
| Neutral short input | 32 | 256/256/256 | 1.38 | 0.823 | 45.50 | 45.41 | 39.70 |
| Neutral long input | 2048 | 256/256/256 | 11.50 | 3.803 | 46.30 | 46.19 | 543.10 |

K2 has higher median output throughput in all four cases and is retained. K6 long-input prefill is slightly higher. This is **not an isolated causal MTP A/B**: K2 allocates4200 MiB outside expert cache and selects5188 experts/16536.8 MiB; K6 allocates4450 MiB and selects5109/16284.9 MiB. Different placement changes CPU/GPU arithmetic. All four cases' output texts differ between cohorts, including Chinese length310→336. There is no cross-configuration bitwise claim; within K6 each case's three texts are identical.

Profile coverage83.54% (K2) /82.99% (K6) describes **training routing scores only**, not runtime hit rate.

## Actual expert work and MTP acceptance

These sums cover all three K2 trials. Three projection rows represent one expert job; these counters do not cover every dense trunk/draft operation. Runtime GPU share is GPU rows/(CPU+GPU rows).

| Case | CPU projection rows | GPU projection rows | Runtime GPU share | Accepted/offered drafts | MTP acceptance |
|---|---:|---:|---:|---|---:|
| BFS | 401625 | 1133415 | 73.84% | 999/1065 | 93.80% |
| Chinese | 161478 | 1039482 | 86.55% | 513/834 | 61.51% |
| Short | 259200 | 751680 | 74.36% | 414/702 | 58.97% |
| Long | 266742 | 787338 | 74.69% | 396/729 | 54.32% |

K6 pooled acceptance: BFS1169/1385 (84.40%), Chinese488/757 (64.46%), short424/809 (52.41%), long403/731 (55.13%). Adaptive per-K statistics remain in its engine counters.

All24 requests pass counter checks: positive equal doorbell/static calls; actual CPU and GPU decode rows both nonzero; CPU jobs×3=CPU rows; fixed-map ownership matches hits/misses; **decode route readbacks0, reported decode expert-weight uploads0.0 MiB, cache decays0**. Ordinary hybrid calls may be0 on this path.

Mailbox activations/results still cross PCIe: each K2 BFS request reports approximately163.07 MiB to host and348.63 MiB to GPU for the whole request. Prefill fallback uploads expert weights per trial: BFS6237.9 MiB, Chinese5877.8, short3958.9, long11796.9. Zero decode expert upload is not zero whole-request transfer. Short-input first-token delay remains substantial.

| K2 decode CPU phase median ms | Input quantization | Gate/up | Hidden quantization | Down |
|---|---:|---:|---:|---:|
| BFS | 17.128 | 1545.562 | 9.976 | 794.789 |
| Chinese | 10.361 | 629.654 | 4.425 | 325.697 |
| Short | 11.202 | 837.887 | 6.358 | 427.408 |
| Long | 11.707 | 868.137 | 6.594 | 443.557 |

CPU phases overlap GPU work; they cannot simply be added to/subtracted from elapsed decode. Doorbell callback time covers the whole request, pool phases decode-only. No API Nsight timeline was collected.

## RAM, VRAM and startup

| Resource | Fixed K2 | Adaptive up to K6 |
|---|---:|---:|
| One-second samples | 89 | 96 |
| Maximum sampled physical RSS | 34.1542 GiB | 34.1522 GiB |
| Windows lifetime physical working-set peak | 34.1820 GiB | 34.1855 GiB |
| Final physical RSS | 34.1483 GiB | 34.1440 GiB |
| Maximum sampled GPU0 use | 22057 MiB | 22025 MiB |
| Maximum sampled GPU1 /GPU2 use | 51 /0 MiB | 51 /0 MiB |
| Maximum sampled system RAM used | 49.8809 GiB | 49.9328 GiB |
| Final system RAM used | 49.8615 GiB | 49.9115 GiB |

Sampling starts after readiness and includes before/after snapshots. GPU maxima are **sampled use, not continuous instantaneous peaks**. Lifetime working-set peak includes startup; private/pagefile commitment is not added to physical RAM. No collector errors occurred. GPU0 final memory equals its sampled maximum in each cohort. K2 final snapshot: UTC2026-10-14 16:25:15.777281 /local2026-10-15 00:25:15; system total127.8707 GiB, available78.0092 GiB.

Native mailbox extra mapped RAM is40.0 MiB and shared GPU scratch204.3 MiB, without another complete31.875 GiB expert copy. Full-file profile hash takes36.472s/36.125s (K2/K6); startup graph warmup3.968s/3.845s. Both are outside resident timing; warmup user KV is cleared.

## Same-executable native expert reuse CLI comparison

Separate single fresh-process CLI trial per arm: input23/output256, fixedK2, **batch512**, same executable SHA as the API cohorts. Command/prompt/environment match except reuse0/2/4 and prompt-file paths. [Full commands and metrics](native-reuse-comparison.json).

| Native expert reuse | Prefill token/s | Output token/s | Change vs0 | Sampled GPU maximum |
|---|---:|---:|---:|---:|
| 0 | 32.79 | **54.93** | — | 21427 MiB |
| 2 | 32.61 | 53.69 | -2.26% | 21427 MiB |
| 4 | 32.87 | 53.41 | -2.77% | 21427 MiB |

Actual output SHA256 was independently recomputed and identical in all arms: `749833c97c2be16194944623dbf2dd51ac7ede6f4a2c86b5531a27ac968c73b2`. Prompt SHA256: `22f016ed4590a69122b787a8fd497bf128b5da0abb753476b7ef4f39f92d6c46`. Decode routes0/uploads0 for all. Reuse0 is retained because2/4 show no gain. These are single trials, not medians/confidence intervals, and supply no API TTFB/TTFT.

The earlier47.62 token/s used a different executable. **47.62→54.93 cannot be attributed to expert weight reuse.** This three-arm comparison uses one executable.

## Evidence audit

Both [K2 audit](measurements-batch2048/audit.json) and [K6 audit](measurements-k6-adaptive/audit.json) are **PASS**: complete12-response order, all-trial medians, reconstructed timing, usage/MTP/counter agreement, actual executable/environment identity, resource maxima and text repeatability. Each closed manifest hashes exactly46 original artifacts (10 base plus3 per trial), excluding report/audit files and future logs. CLI sources have a [separate closed manifest](native-reuse-evidence-manifest.json).

Root's separate successful correctness logs in `2026-10-14-a3b-q8-validation` are `doorbell-parity-{0,16,256}-v3.log`, `doorbell-parity-128-strided-v3.log`, `doorbell-native-reuse{2,4}-parity-128-strided.log`, `q8-doorbell-mtp-k6-strict.log`, `q8-doorbell-native-reuse4-mtp-k6-strict.log`. They cover independent CPU/GPU arithmetic, invalid/error recovery and MTP rollback, not API performance or broad model quality.

The following tool/method notes remain applicable. All reported cohorts have finished; reproduction must use a fresh output directory and must not overwrite the closed evidence.

The testing/audit tools are retained for reproduction of this matrix.

`run-latency.py` uses the same prompts and socket timing helper as the earlier
hybrid experiment: heldout Python BFS code ×3, heldout Chinese transaction
isolation ×3 (maximum output 512 each), neutral input 32/output 256 ×3, and
neutral input 2048/output 256 ×3. All three values enter each case median; the
first and remaining two requests remain separately visible. It never starts,
loads, reloads, unloads or stops a model.

The root agent supplies a separate `--hybrid-doorbell` configuration. This path
uses CPU experts for cache misses and GPU experts for cache hits, with a fixed
verified profile, PCIe expert fraction 0 and cache decay 0. The tool accepts
fixed or adaptive MTP settings and records the actual config, executable SHA,
engine PID/command line and a whitelist of the process's actual environment,
including `STRATA_A3B_NATIVE_EXPERT_REUSE`. It requires an explicitly supplied
expected executable SHA. Production configuration is not rewritten.

Only after the separate service is ready, the root agent can execute:

```powershell
python bench/results/2026-10-15-a3b-q8-doorbell-api/run-latency.py --config <experiment-config.json> --out <new-result-directory> --url http://127.0.0.1:8096 --expected-exe-sha256 <approved-SHA256> --phase all
```

After all requests complete, file-only audit:

```powershell
python bench/results/2026-10-15-a3b-q8-doorbell-api/audit.py --results-root <completed-result-directory>
```

The runner saves timestamped raw SSE frames in `.events.jsonl`, generated text,
and an additional readable `.sse` copy with normalized LF/frame separators.
It measures first underlying socket response byte (TTFB), first nonempty model
output (TTFT), engine prefill/decode rates and client output token rate. Role,
progress and comment frames do not count as model output. The client rate uses
completion-token usage and first/last nonempty delta times, not the number of
SSE events; MTP can emit bursts. Startup is outside resident request timings.

The engine's stdout `DONE` is normally consumed by the Python server, whose
final SSE exports its times, counts and MTP accepted/offered drafts. The audit
cross-checks those exports with token usage and stderr performance/counters.
If a literal `DONE` is present in the log, its counts are checked too. This is
not a claim that unlogged stdout bytes were independently captured.

Execution assertions require positive matching static/doorbell call counts,
both actual CPU and GPU decode projection rows, CPU pool jobs ×3 matching CPU
projection rows, zero decode route readback, and zero reported decode expert
weight uploads. Mailbox traffic is counted separately from expert uploads.
Ordinary hybrid call counts may be zero. CPU phase times are decode-only;
doorbell callback time and mailbox bytes cover the whole request. Cumulative
process counters are explicitly labeled.

RAM and every GPU's memory are sampled once per second, including before/after.
Reports distinguish current RSS, maximum sampled RSS, Windows process-lifetime
physical working-set peak and maximum sampled GPU memory. A sampled GPU maximum
is not a continuous instantaneous peak. The file audit reconstructs TTFT and
client output rate from SSE, checks the complete matrix/medians/counters/resources,
and writes `audit.json`, `audit-table.md` and hashes for original evidence files.

The same-executable native-expert-reuse CLI comparison above is separate from
the API matrix. Two natural heldout prompts do not establish broad model
quality or cache-profile generalization.

Offline validation: six `test_offline.py` cases, Python compilation and
describe-only mode passed. They make no HTTP/GPU calls. Original Strata,
production configurations and earlier reports were not modified by these tools.
