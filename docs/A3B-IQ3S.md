# Qwen3.6-35B-A3B IQ3_S on GPU 0

This is the separate `Strata-Moe` implementation. `G:/Strata` is not changed.
The model is [Unsloth's GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF),
revision `a483e9e6cbd595906af30beda3187c2663a1118c`, file
`Qwen3.6-35B-A3B-UD-IQ3_S.gguf` (13,676,723,168 bytes, SHA-256
`66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66`).
The file is downloaded byte-for-byte; the project does not requantize it.
Its GGUF header has 733 tensors. The 40 routed layers use IQ2_S for gate/up;
37 down projections are IQ3_S and three are IQ4_XS. The label `IQ3_S`
describes the mixed quantization recipe, not every tensor's storage type.
The routed expert weights total 11.57 GB, about 2.87 bits/weight.

The default config uses GPU 0, 262,144-token context, 2048-token prompt
chunks, Q8_0 K/V, and the separate Qwen3.6 Q8_0 MTP head at K2. The MTP
file comes from [ggml-org](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF),
revision `baec3ebee244827cda0f4557eafa8b28f7545fa6`. It is a different
quantization from the target's embedding/output tensors, so pointer-sharing
is disabled. The target checks MTP proposals. At this context size the full
model plus FP16 K/V and MTP exceeded the tested GPU's VRAM during MTP context
allocation; Q8 K/V fit. A smaller Q4_0 MTP head from the same source lets
FP16 K/V fit, but left only 562 MiB free on this GPU and had no measured
speed advantage in its one 50K trial. Q8 K/V changes arithmetic and needs
broader quality evaluation than the one retrieval check below.

On the local RTX 2080 Ti with 22,528 MiB VRAM, the selected IQ3_S model fits
in GPU memory with Q8 K/V and MTP. Full GPU residency therefore avoids expert
uploads and CPU expert math entirely. The existing RAM expert cache and
copy-ahead path also accepts the mixed IQ tensors. Its CPU expert pool selects
the correct GGML dot product and activation type separately for gate/up and
down. A forced 4 GiB cache smoke run at 256K context computed both GPU and
CPU experts, including with MTP, but was slower; it is a compatibility path,
not the selected speed path. The Q8-only fixed-profile/doorbell CUDA path
remains Q8-only and is not used for this fully resident model.

On this machine, start the selected service with `START-A3B-IQ3S.bat` or
`start-a3b-iq3s.sh`. These create [the config](../strata-a3b-iq3s.json) if it
is missing. The Q8 model and its existing config remain available. Server
binding is `127.0.0.1:8096`.

The exact 50K input, first-model-token and output measurements are in
[the benchmark record](../bench/results/2026-10-15-a3b-iq3s-first/README.md).
They were made on one GPU; they are not a 100-token/s claim. The repeated
synthetic input measures speed, not task quality. One 50,017-token retrieval
request recovered a marker inserted halfway through the text.

An opt-in port of Strata's original grouped IQ expert CUDA kernel is
described in [the paired native-kernel record](../bench/results/2026-10-15-a3b-iq3s-strata-native/README.md).
It handles the mixed IQ2_S/IQ3_S/IQ4_XS expert tensors without a second
packed model copy. Its single 50K/256 CUDA-graph trial gave 27.62 s to the
first model token and 57.9 output tokens/s, compared with 28.00 s and 56.1
tokens/s for the same-build baseline. The difference needs repeat testing;
the selected service does not enable the experiment by default.

A second opt-in lets the original Strata fused gate/up + SwiGLU + Q8_1 CUDA
kernel read A3B's separate GGUF expert tensors directly. Real IQ3_S layers
passed bitwise parity. Isolated grouped expert calls fell 26.5-30.3% in
three alternating CUDA-event trials on the local RTX 2080 Ti. Six 50K/256
API requests showed substantial run-order variation, so no end-to-end gain
is claimed. The switch, optional config and exact measurements are in
[the fused gate/up record](../bench/results/2026-10-15-a3b-iq3s-fused-gu/README.md).

The next opt-in keeps Q6_K dense verification bitwise-equal to the selected
one-token target while grouping 2-4 verification tokens in one CUDA MMVQ call
on sm_75. Set `STRATA_A3B_Q6_GROUPED_EXACT=1`, or select
`--q6-grouped-exact` when generating a new A3B config. It uses GPU computation
and leaves the existing configuration unchanged. Two 50K/256 runs measured
62.9 and 62.6 output tokens/s, versus adjacent exact baselines of 56.5 and
54.2; the complete answer stayed identical. First-model-token latency did
not improve. The [Q6 experiment record](../bench/results/2026-10-15-a3b-iq3s-q6-grouped/README.md)
also lists Q4 K/V and larger MTP windows that did not help. The 100-token/s
goal remains open.
The local 8096 service currently uses the opt-in
`strata-a3b-iq3s-q6-grouped.json`. The existing startup script and
`strata-a3b-iq3s.json` remain the fallback configuration.
To restart the measured option, use `START-A3B-IQ3S-Q6.bat` on Windows or
`start-a3b-iq3s-q6.sh` on Linux; each creates its own config if missing.
