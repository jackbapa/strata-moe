# One-token MTP draft, 2026-10-14

One exploratory trial, same prompt and executable as `../2026-10-14-a3b-mtp-first/`. Threadripper 3960X, 128 GiB RAM, RTX 2080 Ti GPU 0 reporting 22 GiB, Windows WDDM. Qwen3.6-35B-A3B-UD-Q5_K_M plus separate Q8_0 MTP weights, 4,096 context, 16 GiB GPU budget, 256 output tokens. The unrelated GPU 1 WSL service remained running; GPU 0 returned to 100 MiB afterward.

MTP draft length 1 measured **27.79 tokens/s** versus the earlier same-executable baseline's 25.09 tokens/s (+10.8%). It accepted 111/143 proposed tokens (77.62%), producing 1.790 output tokens per verification window. Draft/catchup computation took 0.5531/0.3218 s. Peak process working set was 23.779 GiB; GPU usage including background peaked at 16,296 MiB.

First target-token logits match baseline exactly. Generated-output SHA-256 differs; the first differing byte is 222. No token-for-token equivalence claim is made. The baseline output hash and first-token comparisons are recorded in `results.json`; stdout/stderr preserve full text and measurements. This is a single timing run, with generated text and expert CPU/GPU placement varying between configurations.
