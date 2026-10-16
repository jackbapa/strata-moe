| Case | Actual input | Actual output (3 trials) | TTFB ms | TTFT s | Engine output t/s | Client output t/s | Prefill t/s |
|---|---:|---|---:|---:|---:|---:|---:|
| heldout-python-bfs | 160 | 512/512/512 | 10.95 | 1.355 | 61.00 | 60.91 | 120.80 |
| heldout-chinese-transactions | 123 | 336/336/336 | 11.46 | 1.250 | 51.20 | 51.52 | 100.80 |
| prompt32-output256 | 32 | 256/256/256 | 1.38 | 0.823 | 45.50 | 45.41 | 39.70 |
| prompt2048-output256 | 2048 | 256/256/256 | 11.50 | 3.803 | 46.30 | 46.19 | 543.10 |

All three trials are included in each median. See case-summary.json for first/subsequent values.
Sampled RSS max: 34.1522 GiB; sampled GPU maxima: {'0': 22025, '1': 51, '2': 0} MiB. These are sampled values.
