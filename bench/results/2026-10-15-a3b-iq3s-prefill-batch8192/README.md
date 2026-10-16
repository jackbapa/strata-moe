# A3B IQ3_S long-prefill batch-size A/B

This suite tests whether a larger GPU-resident GGML prefill microbatch helps
the 50K first-token path on GPU 0 (RTX 2080 Ti, sm_75). It does **not** claim
that changing `--batch` ports Strata's cross-layer scheduler. The two configs
in this directory differ only in `--batch` (2,048 versus 8,192) and their
private engine-log paths; the selected service config is unchanged. The
benchmark uses the same executable, exact uncached 50,000-token request,
256 output tokens, IQ3_S GGUF and Q8_0 MTP head, and alternates arms twice.

Original Strata's long-prefill issuer in `G:/Strata/src/prefill/prefill.cpp`
prepares CPU-resident expert blobs on a stager thread, submits host-to-device
copies on a copy stream from an independent host thread, and lets the compute
stream wait for copied ring slots. Its grouped gather combines up to 16 copied
experts into one GPU launch and uses one event per group. Those transfers are
useful when experts are **not** resident on the GPU. The selected A3B IQ3_S
service keeps all 40 layers' expert weights on GPU 0, recording zero CPU
expert rows and zero expert upload bytes; a copied ring would have no useful
work in this route. The actual original Strata grouped IQ expert kernel is
used for short target windows through `--strata-native-experts`.

The separately implemented `--strata-prefill-mmq` adapter already made 1,000
original Strata MMQ wrapper calls for 16 million routed rows in a prior 50K
run. Its combined service arm measured 1,776.8 prefill tok/s and 17,653 MiB,
versus 1,782.1 tok/s and 17,119 MiB in the native-expert arm. That combined
arm also changed MTP settings, so the numbers do **not** isolate MMQ's cost.
An 864-token direct adapter comparison found 2,226.56 versus 2,256.51
prefill tok/s with bitwise-identical first-token logits. The original fused
IQ prefill kernel uses `cp.async`, `mma.sync`, and `ldmatrix` and is gated to
sm_80+; it cannot execute unchanged on this sm_75 card. The existing
GGML CUDA backend already groups GPU-resident routes by expert and runs MMQ.

The unprofiled A/B ran on 15 October 2026 in 2,048/8,192/2,048/8,192
order, restarting each service with the same executable SHA-256
`db5c4d92f8ce7b99bf9b376293187e2201270257f77981162ec9d9bec4794f0d`.
The exact request was 50,000 uncached prompt tokens, 256 output tokens,
temperature 0 and seed 12345; context remained configured at 262,144,
physical F16 KV at 65,536, and MTP K2. Each arm used the same IQ3_S target
and Q8_0 draft model. Raw request/event/resource logs and executable,
model, draft and prompt hashes are under `arms/`; `comparison.json` is the
machine-readable summary. The preexisting selected 2,048 config was
restored at port 8096 after measurement, and `/v1/models` again reported
loaded with `n_ctx=262144`.

| Two-run median | HTTP first byte | First model token | Prefill | Output | GPU 0 whole-card peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2,048 batch | 18.10 ms | 28.676 s | 1,758.5 tok/s | 78.25 tok/s | 17,120 MiB |
| 8,192 batch | 19.58 ms | 28.840 s | 1,749.2 tok/s | 79.75 tok/s | 19,196 MiB |

The 8,192 batch consumed **2,076 MiB more sampled whole-card VRAM**.
First-token and prefill differences were small and crossed direction in the
two paired runs: pair 1 was 28.330/1,780.5 versus 28.748/1,756.6;
pair 2 was 29.022/1,736.5 versus 28.933/1,741.7 (seconds/tok/s).
There is no established first-token or prefill gain. The output changed:
2,048 generated SHA-256 `03fc9383...` with MTP accepted/offered **136/238**;
8,192 generated SHA-256 `57a78638...` with **140/229**. Thus the observed
+1.5 tok/s output median cannot be assigned to the prefill scheduling
change; different prefixes and MTP acceptance caused different decode work.
The grouped Strata IQ expert kernel ran in both arms (5,116 versus 4,943
calls/request, altered by the continuation). Neither arm computed experts
on the CPU or copied expert weights from RAM during inference.

The selected service keeps batch 2,048. An 8,192 batch is a scheduling and
capacity tradeoff with no measured 50K first-token benefit here; it is not
a new Strata kernel or the original cross-layer prefill scheduler.
