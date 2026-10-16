# Two-token MTP draft, 2026-10-14

One exploratory trial, same prompt and executable as `../2026-10-14-a3b-mtp-first/`. Threadripper 3960X, 128 GiB RAM, RTX 2080 Ti GPU 0 reporting 22 GiB, Windows WDDM. Qwen3.6-35B-A3B-UD-Q5_K_M plus separate Q8_0 MTP weights, 4,096 context, 16 GiB GPU budget, 256 output tokens. The unrelated GPU 1 WSL service remained running; GPU 0 returned to 100 MiB afterward.

MTP draft length 2 measured **30.40 tokens/s** versus the earlier same-executable baseline's 25.09 tokens/s (+21.2%). It accepted 147/214 proposed tokens (68.69%), producing 2.393 output tokens per verification window. Draft/catchup computation took 0.7584/0.2828 s. Peak process working set was 23.776 GiB; GPU usage including background peaked at 16,286 MiB.

First target-token logits match baseline exactly. Generated-output SHA-256 differs; the first differing byte is 222. No token-for-token equivalence claim is made. The baseline output hash and first-token comparisons are recorded in `results.json`; stdout/stderr preserve full text and measurements. This is a single timing run, with generated text and expert CPU/GPU placement varying between configurations.

Across these exploratory trials, draft lengths 1/2/3 measured 27.79/30.40/30.90 tokens/s. The 1.6% difference between lengths 2 and 3 is too small to establish a reliable winner from one run each.
