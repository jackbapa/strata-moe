| Case | Actual input | Actual output (3 trials) | TTFB ms | TTFT s | Engine output t/s | Client output t/s | Prefill t/s |
|---|---:|---|---:|---:|---:|---:|---:|
| heldout-python-bfs | 160 | 512/512/512 | 11.21 | 1.312 | 65.60 | 65.57 | 124.60 |
| heldout-chinese-transactions | 123 | 310/310/310 | 11.20 | 1.229 | 58.80 | 58.72 | 102.50 |
| prompt32-output256 | 32 | 256/256/256 | 11.83 | 0.813 | 53.00 | 52.90 | 40.70 |
| prompt2048-output256 | 2048 | 256/256/256 | 11.86 | 3.811 | 50.20 | 50.15 | 542.10 |

All three trials are included in each median. See case-summary.json for first/subsequent values.
Sampled RSS max: 34.1517 GiB; sampled GPU maxima: {'0': 22057, '1': 51, '2': 0} MiB. These are sampled values.
