# Qwen3.6 A3B Q8: 256K context and real 50K input

The selected `strata-a3b-q8.json` now sets `--ctx 262144`. The official GGUF
declares `qwen35moe.context_length=262144`; the resident API reports
`max_context=262144`. GPU 0 is the only compute device. The full official Q8_0
weights and official MTP head are unchanged. The default engine remains the
frozen `build-a3b-q8-tuned/bin/strata-moe.exe`, MTP K2, batch 2048, CPU pool 8,
doorbell, mask reuse, and short-prefill CPU sharing. The old 4096 context tuning
is retained in `strata-a3b-q8-4k-compatible.json`.

The test prompt is **synthetic repeated Chinese MoE text**, not a document
retrieval or quality test. `/v1/messages/count_tokens` counted exactly 50000
input tokens, confirmed by each OpenAI response's `usage.prompt_tokens=50000`.
Each request asked for 256 output tokens with temperature 0, seed 12345,
thinking disabled, and `stream_options.include_usage=true`. The cache reports
zero reused input tokens. The first run used a fresh engine; two more used the
same resident process. All three produced the same 256-token output file hash.
Engine load time is outside request times. Raw request/SSE/usage/timing files
are in [first run](measurement) and [resident repeats](repeat).

| FP16 KV run | HTTP first byte | First model output | Prefill | Output | Completion | 
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh 1 | 21.31 ms | 144.377 s | 347.1 token/s | 38.5 token/s | 256 |
| Resident 2 | 18.78 ms | 141.081 s | 355.5 token/s | 38.0 token/s | 256 |
| Resident 3 | 19.14 ms | 248.757 s | 201.1 token/s | 22.7 token/s | 256 |

The three-run medians are 19.14 ms HTTP first byte, 144.377 s first model
output, 347.1 prefill token/s, and 38.0 output token/s. The third identical
request is substantially slower; its cause has **not** been established. The
first two runs do not justify claiming stable 350 token/s. HTTP first byte is
the early streaming response header, not the first model token.

The first run's 52 two-second samples show engine physical RSS peaking at
35.610 GiB, GPU 0 at 21935 MiB, and system physical RAM used at 53.461 GiB.
These are sampled maxima, not continuous or lifetime peaks; the sampler did
not cover every second of model loading and the request.
At startup, the main FP16 KV allocation was 5120 MiB and the MTP KV allocation
was 512 MiB. Remaining GPU expert-cache budget was 9769.7 MiB. Engine counters
record 1000 prefill router readbacks (40 layers × 25 chunks) and
340909.5 MiB = **332.9 GiB of prefill expert-weight upload** for one request.
Decode expert-weight upload was zero; the CPU pool and GPU both computed decode
experts. The 9–256-token CPU prefill share did not apply to 2048-token chunks.
The 332.9 GiB divided by total prefill time is not a PCIe bandwidth measurement,
because GPU computation and scheduling are included in that time.

## Investigated changes

An otherwise identical batch 4096 attempt could not produce a result:
`ggml-cuda/mmid.cu:147` asserted `nbytes_shared <= smpbo` on the first large
prompt graph. The [engine log](batch4096/engine.log) retains this failure. A
4096-token prompt chunk requires fixing this CUDA helper or splitting its work;
simply setting `--batch 4096` is invalid on this build.

An intermediate `--batch 3072` passed a 3072-token smoke request and a full
[50000/256 request](batch3072/measurement). It made 680 prefill route readbacks
(40 layers × 17 chunks), and uploaded 246358.7 MiB = 240.6 GiB of prefill
expert weights, 27.7% less than batch 2048. But its single run after the smoke request was
352.3 prefill token/s, 142.294 s to first output, and 37.0 output token/s:
no compelling speed gain over the first two 2048 runs. GPU expert budget
shrank to 9047.7 MiB; sampled engine RSS reached 36.513 GiB and GPU 0
21975 MiB. The answer text changed with the different prefill math layout.
This one-run comparison does **not** isolate PCIe transfer time: larger graph
work and reduced expert residency may offset the fewer uploads. The stable
batch 2048 remains selected.

An explicit experimental `--kv-q8` flag stores K and V as Q8_0. For 256K, main
KV fell to 2720 MiB and MTP KV to 272 MiB; GPU expert budget increased to
11457.7 MiB. On the same 50000-token prompt, [one fresh run](kvq8/measurement)
gave 18.42 ms HTTP first byte, 160.239 s first model output, 312.8 prefill
token/s, and 39.3 output token/s. Expert-weight upload fell to
317704.5 MiB = 310.3 GiB, but prefill became slower than the first two FP16
runs. The generated text changed. This is a single run, not a quality study;
Q8 KV stays opt-in and the selected service retains FP16 KV. Its 72 sampled
points show GPU 0 at most 21947 MiB, temperature at most 63 °C, and engine
RSS at most 35.626 GiB. See [experimental log](kvq8/engine.log).

## Why the original comparison differs

The [original single-RTX-2080-Ti result](../../../../Strata/bench/results/2026-10-02-community-rtx2080ti-11gb/README.md)
at 256K context, IQ3_S, INT8 KV and KV streaming measured 566 prefill token/s
at 128K input and 466.6 at 250K, with 34.1/31.8 output token/s. That machine
also used a Threadripper 3960X, but ran Linux and different memory clocks.
Another [single 2080 Ti report](../../../../Strata/bench/results/2026-10-03-ashley-2080ti/README.md)
measured about 491 prefill token/s at 128K with IQ3_XXS. The often-cited
approximately 1500 prefill token/s comes from newer GPU measurements in
`G:/Strata/docs/DETAILS.md`; this machine's [three-card report](../../../../Strata/bench/results/2026-10-12-community-3x-rtx-2080-ti-qwen3-8-flash-next-iq3-s/README.md)
used all three RTX 2080 Ti cards and measured 1267 at 16K. Neither is a
one-card Q8 comparison. These are different prompts, model architectures,
quants, context lengths and implementations, not a controlled model-only A/B.

The A3B path still lacks the original's pack-specific prefill pipeline and
automatic streamed expert ring. It stages every missed Q8 expert group to the
GPU in each 2048-token chunk; the original can use more prompt rows per weight
load and has its own optimized pack kernels. Q8 weights also move more bytes
than the original IQ3 packs. The measurements above establish substantial
expert traffic, but cutting it 27.7% with batch 3072 did not materially
improve prompt speed. They do not isolate the full timing gap among matrix
kernels, graph scheduling, attention, and transfer. The target of 100 output token/s
and 1500 prefill token/s remains unmet on this configuration.

Original `G:/Strata` was inspected read-only; all configuration, code and
benchmark artifacts here are in `G:/Strata-Moe`.
