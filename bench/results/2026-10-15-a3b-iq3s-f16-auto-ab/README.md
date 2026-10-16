# IQ3_S A3B: F16-short/Q8-long automatic KV tier

Both arms used the exact same executable SHA-256 `2d1d9d9119de846b38340b5f484afd8d7f83f2175a6425ffdffd3efb90cc2e49`, the same `Qwen3.6-35B-A3B-UD-IQ3_S.gguf` mixed-quantization target and its separate Q8_0 MTP head, GPU 0 (RTX 2080 Ti, 22,528 MiB), MTP K2, batch 2,048 and batched GGML attention. Each request had the exact same **uncached 50,000-token** prompt, 256 generated tokens, and a **262,144-token configured limit**. The physical tier was 65,536 cells. The automatic arm used `--kv-tiered-f16-short`, hence F16 target/MTP KV at this tier; the control used Q8_0 KV at the same tier. The only intended execution option difference was that KV format. Two runs of the automatic arm preceded two runs of the Q8 control; this is a small sequential comparison, not randomized sampling.

| Two-run median | HTTP first byte | First model token | Prefill | Output | Whole GPU 0 peak | MTP accepted/offered |
|---|---:|---:|---:|---:|---:|---:|
| 65K Q8_0 KV control | 10.48 ms | 29.203 s | 1,724.4 tok/s | 63.25 tok/s | 16,591 MiB | 141/227 |
| 65K F16 KV, automatic long-context Q8 fallback | 14.74 ms | 27.811 s | 1,808.4 tok/s | 75.45 tok/s | 17,119 MiB | 139/232 |

Two more opt-in arms on the **same executable and request** exercised the
copied original Strata kernels and MTP bridge. They were run afterward,
sequentially, twice each:

| Additional arm (two-run median) | HTTP first byte | First model token | Prefill | Output | Whole GPU 0 peak | Actual Strata expert calls/request |
|---|---:|---:|---:|---:|---:|---:|
| F16 tier + original Strata grouped IQ experts | 15.32 ms | 28.252 s | 1,782.1 tok/s | 79.35 tok/s | 17,119 MiB | 5,116 |
| Above + original IQ MMQ adapter + A3B GDN switch + MTP GPU prefill bridge/K2 chain | 14.72 ms | 28.372 s | 1,776.8 tok/s | 77.65 tok/s | 17,653 MiB | 5,116 |

The native-expert arm generated a different answer and accepted 136/238 MTP
draft tokens, so its +3.9 tok/s versus the plain F16 arm is an end-to-end
observation, not an isolated kernel speedup. The combined arm generated the
**same** answer and accepted the same 136/238 as the native-expert arm. It
made 1,000 original Strata IQ MMQ adapter calls for 16 million routed rows,
25 device-prefill bridge calls for 50,000 rows, and 119 K2 device-chain calls
with zero fallbacks. The bridge avoided 409,395,200 bytes of whole-chunk
hidden download. The chain's old **zero cross-step CUDA graph replays** field
was a hard-coded log constant, not an observed count; this comparison cannot
establish graph replay within the K2 branch. Separate Nsight traces show
GGML target and ordinary draft graphs replay. The A3B
single-token GDN switch matched zero graphs in this batched-MTP setup. The
combined arm was 1.7 tok/s slower than the native-expert arm in this small
sample and used 534 MiB more sampled GPU memory. Its lower `mtp_hidden_ms`
counter did not turn into a lower first-model-token or output time. Enabling
every available opt-in is therefore not supported as a faster default.

A fifth arm kept F16-tier + native experts but raised target batch from 2,048
to 4,096 (MTP microbatch stayed 512). Its two-run medians were **9.84 ms**
HTTP first byte, **27.228 s** first model token, **1,848.2 tok/s** prefill,
**77.45 tok/s** output and **17,811 MiB** GPU peak. Against the 2,048 native
arm, the observed first model token was 1.02 s earlier and prefill 66 tok/s
faster, but output 1.9 tok/s slower and VRAM 692 MiB higher. It generated a
different answer and MTP acceptance changed from 136/238 to 132/246. This
batch setting is left as an opt-in first-token/prefill tradeoff, not the
selected output-speed path.

The automatic arm's observed output was **12.2 tok/s faster** and used **528 MiB more** GPU memory at 50K. Its output and MTP acceptance differed from Q8 because changing KV precision changes model arithmetic: hashes were `3ee09940...` for F16 and `c1d52a6c...` for Q8, stable within each arm. The earlier reverse-order comparison on the preceding binary, whose new selector code was not yet linked, measured F16 **72.25** versus Q8 **64.85 tok/s**, again a positive output direction but a smaller +7.4 tok/s. First-token and prefill direction reversed between the two sequential experiments; no stable improvement for those two metrics is established. These end-to-end numbers must not be attributed entirely to one CUDA kernel.

The automatic selector was also exercised with a real **131,073-token** prompt and two generated tokens. It rebuilt the target/MTP contexts at **262,144 physical cells** with Q8_0 KV (2,720 + 272 MiB), completed the request, and recorded a 20,499 MiB whole-card peak. Its 119.625 s first model token and 1,100.8 tok/s prefill are a separate long-input smoke test; two output tokens do not provide a meaningful decode-rate comparison. A following 100-token request rebuilt back to 65,536 cells with F16 KV and completed. Exact logs are `G:/Strata-Moe/a3b-engine-q6-kv-tier-f16-auto.log`; benchmark records are in the sibling `2026-10-15-a3b-iq3s-f16-auto-256k-tier-probe` and `2026-10-15-a3b-iq3s-f16-auto-restore-probe` folders.

For comparison, the prior fixed-256K Q8 service sampled roughly 20,499–20,528 MiB on this 50K input, with about 64.9 tok/s output. The automatic 50K tier used roughly **3.4 GiB less** whole-card memory and was faster in the two sequential comparisons; Q8-only tiering saved about **3.9 GiB** with little observed output difference from fixed Q8. The tier rebuilds idle GGML contexts between independent requests. It is not the original Strata CUDA VMM `--kv-grow`, and it does not implement the original model-specific MTP round/step graph or shared-expert side stream. The 100 tok/s target remains unmet.

The `arms/` subdirectories retain individual HTTP/model-token timing, prefill/decode speed, hashes, resource samples and config/model/executable identities for all five arms. GPU peak is sampled whole-GPU usage, not process-attributed. HTTP first byte is an early protocol byte, **not** the first generated model token. All long-context tests remained on GPU 0 with zero CPU expert rows and zero expert upload MiB.
