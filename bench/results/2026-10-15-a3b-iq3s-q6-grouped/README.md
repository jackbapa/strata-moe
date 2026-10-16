# IQ3_S A3B: exact grouped Q6_K verification on RTX 2080 Ti

Windows, Threadripper 3960X, NVIDIA RTX 2080 Ti GPU 0 (sm_75, 22,528 MiB),
Qwen3.6-35B-A3B UD-IQ3_S target and Q8_0 MTP K2. Context was 262,144;
the speed request had exactly 50,000 counted prompt tokens and 256 generated
tokens, with no cached prompt tokens, temperature 0 and seed 12345. Its request
SHA-256 was `2ed16fa60f51be9afe54213d3bd8190a5c19f2c6ada95ccaf03a06017f79a910`.
Model loading is excluded. HTTP first byte is the first socket response,
usually response headers; first model token is the first nonempty answer delta.

The original Strata checkout at `G:/Strata` was not changed. In the A3B clone,
`STRATA_A3B_Q6_GROUPED_EXACT=1` makes 2–4-column Q6_K dense verification
use GGML CUDA's grouped MMVQ with the two-warp reduction of its one-column
path on sm_75. The old exact path forms one independent channel per token.
Other shapes, quantization formats and architectures retain that path. GPU 0
does the matrix math; the CPU does not calculate experts. The option is off by
default and has a reproducible pinned llama.cpp patch.

One same-day 50K/256 trial before the change's paired repeat:

| Path | First model token | HTTP first byte | Prefill | Output |
|---|---:|---:|---:|---:|
| Exact baseline, Q8 K/V | 27.90 s | 18.5 ms | 1817.4 tok/s | 56.5 tok/s |
| Grouped Q6 exact, Q8 K/V | 28.88 s | 19.2 ms | 1745.8 tok/s | 62.9 tok/s |

In an immediately adjacent second pair, the exact baseline measured
29.11 s first model token, 19.1 ms HTTP first byte, 1737.5 prompt tok/s and
54.2 output tok/s. The grouped Q6 option measured 29.45 s, 19.2 ms,
1713.4 prompt tok/s and 62.6 output tok/s. Both grouped runs improved output
while first-model-token latency did not improve. These are two single-run pairs,
not a stable population median.
The final rebuilt and locally selected service was measured once more on the
same 50K/256 input: 29.12 s first model token, 9.6 ms HTTP first byte,
1736.8 prefill tok/s and 63.1 output tok/s. Its output JSON hash matched the
same baseline hash below. A later grouped-Q6-only run after the Q8 reuse
experiment gave 28.54 s first model token and 64.5 output tok/s. The four
grouped output rates have median 63.0 tok/s; the two adjacent baseline output
rates have median 55.4 tok/s.

The output JSON files have the same SHA-256,
`d01bcd72056af652f33ed32a582cbcf154f909ff91dbd173cdc790864d7694d1`.
The native-dense parity fixture checks a 2048-by-2048 Q6_K matrix with 2–4
columns, including strided input, against separate one-token GPU products:
every float matched bitwise. A 50,017-token middle-marker retrieval returned
`海蓝-73192` in nine output tokens. This is one quality check, not an accuracy
study.

Negative trials with the same input: Q4_0 K/V gave 28.03 s first model token
and 52.8 tok/s output (its 50,017-token retrieval passed); MTP K4 gave
29.87 s and 41.8 tok/s. The Strata grouped expert + fused gate/up + fused
MTP rounds combination gave 29.38 s and 58.5 tok/s. Omitting exact dense
layout gave 29.38 s and 61.0 tok/s, but changed the answer. The grouped Q6
path is the first tested dense improvement that retained the baseline answer.
The existing Q8_0 dense reuse switch was also combined with grouped Q6.
Two 50K/256 combined runs gave 67.1 and 63.1 tok/s; an immediately
interleaved grouped-Q6-only run gave 64.5 tok/s. All produced the same answer.
This sequence does not establish a Q8 reuse gain, so the selected config leaves
that switch off.
Adding Strata native grouped experts and fused gate/up to grouped Q6 gave
29.86 s first model token and 64.9 output tok/s in one run. Its answer was
identical to the prior Strata-native answer but different from the selected
baseline's answer, so that combination remains a separate opt-in experiment.

With diagnostic phase timing enabled on the unmodified exact baseline, a
1K/256 request output 68.1 tok/s; a later 50K/256 request output 53.2
tok/s. For the 50K output phase (4.81 s), the wall-clock phase sums were:
target verify submit 0.154 s, target completion and hidden-state observation
3.707 s, MTP proposal 0.765 s, commit 0.182 s, sample/emit 0.003 s.
CUDA is asynchronous: the observation timer includes queued target GPU work,
not just a small hidden-state copy. This points to target GPU execution as
the remaining bottleneck. Neither the opt-in result nor any trial reaches
the 100 tok/s goal.

`python tools/a3b.py build --jobs 8` passed after patch registration. The
CUDA build, the 17 existing CTest cases, the explicit grouped-Q6 real-GPU
parity mode and 38 Python config-option tests passed. HIP and SYCL were not
built on this Windows NVIDIA host. The current local 8096 service uses the
opt-in `strata-a3b-iq3s-q6-grouped.json`; its final-build 64-input/16-output
API smoke test passed. `strata-a3b-iq3s.json` remains the unchanged fallback.
