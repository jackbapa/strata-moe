| Case | Actual input | Actual output (3 trials) | TTFB ms | TTFT s | Engine output t/s | Client output t/s | Prefill t/s |
|---|---:|---|---:|---:|---:|---:|---:|
| heldout-python-bfs | 160 | 512/512/512 | 11.42 | 0.699 | 65.90 | 65.90 | 238.90 |
| heldout-chinese-transactions | 123 | 328/328/328 | 12.05 | 0.584 | 61.40 | 61.79 | 222.70 |
| prompt32-output256 | 32 | 256/256/256 | 10.86 | 0.352 | 52.30 | 52.15 | 98.40 |
| prompt2048-output256 | 2048 | 256/256/256 | 11.02 | 3.832 | 50.10 | 50.09 | 537.80 |

All three trials are included in each median. See case-summary.json for first/subsequent values.
Sampled RSS max: 34.1599 GiB; sampled GPU maxima: {'0': 22057, '1': 51, '2': 0} MiB. These are sampled values.
