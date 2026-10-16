# A3B Q8 long-prefill host staging, matched 50K test

Hardware: Threadripper 3960X, 128 GiB RAM, GPU 0 RTX 2080 Ti (22528 MiB,
SM75, PCIe 3.0 x16), Windows. Model: the complete official
`ggml-org/Qwen3.6-35B-A3B-GGUF` Q8_0 GGUF and matching official Q8 MTP GGUF;
no weights were quantized or skipped locally. Both conditions used FP16 KV,
262144 context, 2048 prompt batch, the same fixed expert profile, MTP K2,
temperature 0, seed 12345, and one resident server per condition. Every
request reported exactly 50000 prompt and 256 completion tokens, with zero
prompt-cache reuse. The tested executable SHA-256 was
`c37f247b1c6d747632242354a97c480b1e2d5df1aadbd111742a7dae5a801f1b`.

The only engine option changed between conditions was `--prefill-copy-ahead`.
It starts a host worker that packs pageable Q8 expert bytes into bounded pinned
banks while the GPU processes earlier groups. The source model remains in RAM;
all long-prefill expert matrix computation remains on GPU 0. Its two bank
reuse conditions are distinct: host bytes can be overwritten after DMA, and
device weights only after the consuming GPU work. The eight-expert compute
group and copied expert bytes are unchanged. The option is off by default.

| Run in fresh resident process | First model token | Prefill | Output | HTTP first byte |
| --- | ---: | ---: | ---: | ---: |
| Copy ahead, repeat 1 | 111.423 s | 449.8 tok/s | 39.4 tok/s | 19.1 ms |
| Control, repeat 1 | 126.834 s | 395.2 tok/s | 39.2 tok/s | 9.3 ms |
| Copy ahead, repeat 2 | 157.483 s | 317.7 tok/s | 30.4 tok/s | 19.3 ms |
| Control, repeat 2 | 177.307 s | 282.2 tok/s | 30.4 tok/s | 19.1 ms |

The first model-token wait improved by 12.2% and 11.2% in the paired runs;
prefill improved by 13.8% and 12.6%. All four runs produced the same
256-token output for this deterministic prompt. Both second requests were
substantially slower, so the absolute rate is not stable and two pairs are
insufficient for a general speed claim. HTTP first byte only marks early SSE
headers; it is **not** model latency. The output rate did not improve. The
100 tok/s output and 1500 tok/s prefill targets remain unmet.

Each request made 1000 long-prefill route reads (40 layers x 25 chunks), about
13806 staging batches and about 333 GiB of expert uploads. With copy ahead,
the first measured request spent 25.5 s in route read plus preceding GPU wait,
and 81.7 s in the GPU-only host scheduling span, including 67.3 s of device
bank-reuse waiting and 8.5 s of task submission. These intervals contain
asynchronous GPU work and cannot be added as independent device durations.
The control's corresponding spans were 25.2 s, 97.3 s, 53.6 s, and 7.6 s.
The shift in bank wait is expected when the host worker runs ahead; it does
not mean the GPU computation became faster.

The source comparison explains why this is only a partial Strata port.
Original `G:/Strata/src/prefill/prefill.cpp` constructs an all-layer expert
sequence, uses a large event-protected device ring, runs a host stager and an
independent copy issuer, gathers up to 16 experts per MMQ launch, and shares
one activation quantization per group. This A3B engine currently stages
per layer with two banks and an eight-expert GGML task, then invokes
llama.cpp's Q8 `ggml_mul_mat_id` for gate/up/down. Inspection of
`ggml-cuda.cu` confirmed that the present long-prefill tensor layout already
selects its quantized MMQ path; changing that layout would not port the
original grouped pipeline. The unported all-layer issuer, grouped gather,
and shared activation quantization are concrete remaining work.

The tested binary above was rebuilt afterward with only a 1024-token
copy-ahead threshold and argument validation change; the final local binary
SHA-256 is `5e983e360fbfb9152e291692fa023931e58a78128b84d0cfe324ad15cbd8a862`.
That exact hash received a separate full 50K/256 request: first model token
111.383 s, prefill 449.9 tok/s, output 39.5 tok/s, and the same byte-for-byte
output SHA-256 `7c64902d2327d0540442a7b07f783bd812483cf41c55662aaa44d28df10f321e`.
The source configuration `strata-a3b-q8.json` now opts into this verified
binary and copy-ahead path; its earlier contents were saved as
`strata-a3b-q8-pre-stager.json`. This promotion changes the local selected
service configuration, not the default behavior of a newly built engine.
The original `G:/Strata` checkout was not edited.

The final backend was rebuilt with MSVC/CUDA. A dedicated 1024-token,
eight-route-per-token Q8 FFN fixture exercised zero expert cache, multiple
two-slot staging waves, asynchronous GPU events, frozen placement and
repeated requests. Control and copy-ahead processes both passed and produced
the same full-output FNV64 checksum, `1bc5f9cc2f298d31`; each run was
bitwise stable on repeats and after rejected invalid routes. The usual
short-window independent GPU oracle and CPU suite also passed. The long
fixture compares the two execution schedules, while the full-model 50K
request above checks user-visible output parity. Neither result demonstrates
100 tok/s output or a complete port of Strata's grouped Q8 MMQ pipeline.

Inputs and raw server logs are in this directory. The 2048/64 smoke request
reported 3.657 s to first model token, 627.0 prompt tok/s and 44.8 output
tok/s, but has no matched control and is not evidence of a speedup.
