| Case | Actual input | Actual output (3 trials) | TTFB ms | TTFT s | Engine output t/s | Client output t/s | Prefill t/s |
|---|---:|---|---:|---:|---:|---:|---:|
| heldout-python-bfs | 160 | 512/512/512 | 11.62 | 1.318 | 63.40 | 63.39 | 124.30 |
| heldout-chinese-transactions | 123 | 310/310/310 | 10.57 | 1.235 | 57.00 | 56.92 | 102.10 |
| prompt32-output256 | 32 | 256/256/256 | 10.83 | 0.811 | 51.70 | 51.53 | 40.70 |
| prompt2048-output256 | 2048 | 256/256/256 | 11.62 | 3.834 | 48.80 | 48.70 | 538.80 |

All three trials are included in each median. See case-summary.json for first/subsequent values.
Sampled RSS max: 34.1542 GiB; sampled GPU maxima: {'0': 22057, '1': 51, '2': 0} MiB. These are sampled values.
