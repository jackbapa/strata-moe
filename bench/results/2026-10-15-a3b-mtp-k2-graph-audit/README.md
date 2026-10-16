# A3B IQ3_S MTP K2: actual CUDA graph replay and synchronization audit

GPU 0 was the RTX 2080 Ti (22,528 MiB), using the single
`Qwen3.6-35B-A3B-UD-IQ3_S.gguf` target and matching Q8_0 MTP head. The
target's Q6_K tensors are inside that same mixed IQ3_S GGUF, not a separate
model. This audit uses the short 128-cell strict diagnostic fixture, not a
50K latency/throughput benchmark. The diagnostic executable was
`c5a6f0e909629a8da5c40884ec544d99feb84150fe611efd6573c02f0dbae379`.
Its build included the new default-off shared-expert-stream patch; neither
strict arm enabled that switch. CUDA sm_75 was built successfully.

The old service log always printed `CUDA cross-step graph replays 0` as a
literal in `a3b/main.cpp`; it was not a counter. The historical 50K reports
that interpreted this zero as measurement have been corrected. The service
now prints `CUDA graph replays not instrumented`; the benchmark parser treats
the old literal zero as unknown (`null`) and passes its eight unit tests.

The diagnostic previously reset `model_params.tensor_buft_overrides` while
loading the draft. It consequently left the 515 MiB Q8_0 draft token table
on `CUDA_Host`, so every attempted K2 chain returned from its preflight and
measured the ordinary fallback path. `a3b/speculative_test.cpp` now keeps the
target fixture's explicit GPU token-table placement when shared weights are
off. The real K2 test log confirms `token_embd.weight` was overridden to
CUDA0; the earlier invalid Nsight capture is retained separately as
`k2-on-nvtx.*`.

On the corrected diagnostic, independent K2-off and K2-on processes both
passed rollback keep 1/2/3 with zero maximum absolute logit difference and
zero KL divergence. All eight output-limit class lines (1/2/3/4/5/8/16/32)
match exactly. The 29 per-window proposal-trace lines match byte-for-byte,
and all **16** final draft/target sequence-state files compare byte-for-byte
(529,785,008 bytes per arm). `strict-parity.json` has every state SHA-256;
`strict-off.log` and `strict-on.log` retain the complete assertions. This
establishes the corrected fixture's output, proposal and serialized KV/GDN
state parity; it does not prove parity for every prompt or sampling policy.

With `STRATA_A3B_MTP_DEVICE_CHAIN_K2=1` and `STRATA_A3B_NVTX=1`, Nsight
Systems 2024.4.2 recorded **26 real K2 chain ranges**. There were **49
`cudaGraphLaunch` API calls inside those ranges**: 23 ranges contained two,
three contained one. The earlier hard-coded zero
was false as a statement about actual launches. Each range still invokes
two independent GGML `llama_decode` steps; the trace does **not** show one
captured original-Strata round/step graph or GPU-side acceptance/commit.
The complete `k2-real.nsys-rep`, SQLite export, query script and
`graph-analysis.json` are retained.

The same 26 ranges contained **624 `cudaStreamSynchronize` calls**, exactly
24 per range. Their CUDA API duration totals **96.908 ms** within **111.929
ms** of summed K2 host NVTX duration under the profiler. This is a short
profiled fixture and cannot be converted to 50K service speed. CUDA
correlation IDs identify **312 H2D copies**, exactly 12 per K2 range,
with sizes 4/8/16/512/8192 bytes. The timeline shows those small copies immediately followed
by stream synchronizations. That pattern matches the synchronous CUDA tensor
setter in pinned vendor `ggml-cuda.cu:ggml_backend_cuda_buffer_set_tensor`,
which does `cudaMemcpyAsync` on `cudaStreamPerThread` and then
`cudaStreamSynchronize`; A3B's Qwen MTP input token/hidden/position setters
call `ggml_backend_tensor_set`. Nsight did not collect per-call backtraces, so
the exact share of all 624 synchronizations from each input, scheduler, and
final decode wait remains unproven. We do **not** assign them wholesale to
MoE routing or CPU expert computation.

Original Strata `src/core/mtp.cpp:1050-1115` captures separate round and
step graphs with its mapped/pinned input staging; `mtp_select` writes the
sampled token and next-step residual on the GPU. `:1308-1315` launches the
graphs in sequence and synchronizes after the draft chain. A3B K2 already
avoids a host token/hidden getter **between** its two steps, but its GGML
input staging and target acceptance/commit remain separate. A safe next
opt-in would give the one-token Qwen MTP input classes stable pinned buffers
for both steps, resolve the exact CUDA backend for each GPU input, and use
`ggml_backend_tensor_set_async` on the same compute stream for token,
hidden and position uploads. The per-step buffers must live until the final
stream sync; derived temporary 4D positions cannot be passed from a local
vector to an asynchronous copy. CPU-resident mask tensors should keep their
current path. The change needs same-binary off/on strict state/trace checks
and Nsight proof that H2D+sync pairs actually decrease before any speed
claim. It was **not implemented** in this audit.

After the experiment, port 8096 was restored with
`strata-a3b-iq3s-q6-kv-tier-f16-auto-native.json` and the new main executable
SHA-256 `db5c4d92f8ce7b99bf9b376293187e2201270257f77981162ec9d9bec4794f0d`.
`/v1/models` reported the target loaded with `n_ctx=262144`; engine PID was
42128. Original `G:/Strata` tracked source was not edited. No HIP/SYCL build
was run.
