# A3B IQ3_S 50K/K2 decode profiling plan

This is a plan, **not a new performance measurement**. It uses the same
Qwen3.6-35B-A3B UD-IQ3_S GGUF and Q8_0 MTP head as the saved 50,000-input,
256-output A/B. The IQ3_S GGUF is mixed-format: its Q6_K dense tensors are
inside that same target file. The original `G:/Strata` checkout remains
read-only.

## Evidence already available

The same-executable, alternating B-S-B-S run in
`bench/results/2026-10-15-a3b-iq3s-strata-ab/README.md` used an RTX 2080 Ti
(GPU 0, 22,528 MiB), 262,144 context, Q8 K/V, exact 50K uncached input,
MTP K2 and 256 output tokens. Two-run medians were 67.2 tok/s with GGML
experts and 70.4 tok/s with the original Strata grouped IQ experts plus fused
gate/up. First model deltas were 27.44 and 27.82 s; sampled whole-card VRAM
peaks were both 20,499 MiB. The Strata arm counted 5,070 native expert calls
in decode and zero in prefill. Its deterministic answer and MTP acceptance
rate differed from the GGML arm (58.8% versus 62.1%), so the 3.2 tok/s
difference cannot be assigned entirely to one kernel.

`bench/results/2026-10-15-a3b-iq3s-profile/short-nomtp-nodes.sqlite` is a
roughly 1K-input/96-output, **no-MTP, pre-grouped-Q6** Nsight trace. Its
Q6_K vector kernels sum to 375.95 ms, Q6_K MMQ to 75.34 ms, GDN to
75.14 ms, and all GPU kernels to 1,428.94 ms. These are profiled kernel
duration sums, not current decode wall time. They neither locate the
post-grouped-Q6 bottleneck nor measure 50K/K2 behavior. Do not combine
their percentages with the current 70.4 tok/s A/B.

The older `a3b-engine-phases.log` run enabled `STRATA_A3B_TIMING=1` for a
different, pre-grouped-Q6 50K request. Its 4.81 s output phase included
0.765 s proposal, 0.154 s target verify *submission*, 3.707 s target
observation, 0.182 s commit and 0.003 s sample/emit. The observation waits
for earlier asynchronous target GPU work. It is not evidence that a small
host hidden-state copy alone costs 3.707 s. Repeat the timer on the current
binary and config before deciding where to optimize.

At 70.4 tok/s, 256 output tokens take about 3.64 s. Reaching 100 tok/s
requires about 1.08 s, or 29.6%, off that output interval. A faster GDN
step, Q6 head, or expert kernel may be worthwhile, but isolated microbench
percentages cannot be added to promise this gain.

## Required instrumented measurement

1. First run an **unprofiled** 50K/256 request with the current executable,
   same GGUF/MTP SHA-256, exact saved `request-template.json`, config and
   sampling settings. Use `tools/bench_a3b_ab.py` to retain first HTTP byte,
   first nonempty model delta, prefill and output tok/s, MTP offered/accepted,
   answer hash, engine PID/command/environment and sampled GPU 0 usage.
   Alternate configurations in a fresh suite if comparing a new option.
2. Make a copy of the selected JSON config for profiling. Keep model, MTP,
   262,144 context, `--batch 2048`, GPU 0, Q8 K/V, MTP K2 and feature
   switches identical. Use a separate log and local port, and add
   `"STRATA_A3B_TIMING": "1"` to `env`. Do not add an unlisted environment
   switch to a running server: the A/B harness correctly rejects that
   identity mismatch. Record the executable/config hashes.
3. Add a **diagnostic, opt-in NVTX range** around one request's prefill and
   output loop, and nested ranges around MTP proposal, target verification,
   target-hidden observation, sampling and commit. The code boundary for
   these ranges is `a3b/main.cpp::generate`; `MtpDraft::propose` and
   `MtpDraft::process_observed` in `a3b/speculative.cpp` may need finer
   draft/catch-up ranges. Keep the non-diagnostic path unchanged. Without
   ranges or an equivalent timestamp correlation, kernel names alone cannot
   distinguish target from draft work: both models use GGML and may launch
   the same Q6, GDN and attention kernels. A CPU range's duration also need
   not include its asynchronously queued GPU work; use Nsight's CUDA
   launch-to-GPU correlation, not just NVTX host-range elapsed time.
4. Trace a request on the same GPU 0 using the installed Nsight Systems
   2024.4.2 with `--trace=cuda,nvtx --cuda-graph-trace=node`, CPU sampling
   disabled. The workstation profiler traces the Python server's engine
   child. Start collection **after model load and immediately before** the
   exact request, and stop after `[DONE]`. Node granularity is necessary for
   kernel attribution inside CUDA Graph launches; it adds substantial
   profiling overhead. Also take a lower-overhead `graph` trace for replay
   counts/host gaps if needed. Neither profiled tok/s nor profiled TTFT is a
   replacement for step 1's unprofiled latency.
5. Export the `.nsys-rep` to SQLite and save `cuda_api_sum`,
   `cuda_gpu_kern_sum`, `cuda_gpu_mem_time_sum`, `nvtx_kern_sum`,
   `nvtx_pushpop_trace` and `cuda_gpu_trace` reports. Verify the trace has
   one 50K/256 request, full prefill/decode ranges, a CUDA kernel table and
   CUDA graph node activities. If a table is absent, report that limit.

The installed CLI is
`C:/Program Files/NVIDIA Corporation/Nsight Systems 2024.4.2/target-windows-x64/nsys.exe`.
After the root operator has stopped the exact existing engine so the two
22.5-GiB-resident services cannot compete, a PowerShell capture sequence is:

```powershell
$nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.4.2\target-windows-x64\nsys.exe'
$session = 'A3B50KDecode'
$out = 'G:\Strata-Moe\bench\results\a3b-50k-current-profile\nsys-node'
New-Item -ItemType Directory -Force -Path (Split-Path $out) | Out-Null
$profileConfig = 'G:\Strata-Moe\strata-a3b-iq3s-q6-strata-profile.json'
$cfg = Get-Content -Raw 'G:\Strata-Moe\strata-a3b-iq3s-q6-strata-combined.json' |
  ConvertFrom-Json -AsHashtable
$cfg['port'] = 8097
$cfg['log'] = 'G:\Strata-Moe\bench\results\a3b-50k-current-profile\engine.log'
$cfg['env']['STRATA_A3B_TIMING'] = '1'
$cfg | ConvertTo-Json -Depth 20 | Set-Content -Encoding utf8 $profileConfig
$prof = Start-Process -FilePath $nsys -WindowStyle Hidden -PassThru `
  -WorkingDirectory 'G:\Strata-Moe' -ArgumentList @(
    'profile', "--session-new=$session", '--start-later=true',
    '--trace=cuda,nvtx', '--cuda-graph-trace=node',
    '--sample=none', '--cpuctxsw=none', "--output=$out",
    'G:\Strata-Moe\.venv\Scripts\python.exe', '-m', 'serve.server',
    '--engine', 'strata', '--config',
    $profileConfig, '--port', '8097'
  )
# Wait for http://127.0.0.1:8097/v1/models; inspect engine PID, model and log.
& $nsys start "--session=$session"
G:\Strata-Moe\.venv\Scripts\python.exe tools\bench_a3b_ab.py run `
  --suite bench\results\a3b-50k-current-profile\api --arm traced `
  --config $profileConfig --repeats 1
& $nsys stop "--session=$session"
& $nsys shutdown "--session=$session" '--kill=true'
& $nsys export '--type=sqlite' '--output=G:\Strata-Moe\bench\results\a3b-50k-current-profile\nsys-node.sqlite' `
  'G:\Strata-Moe\bench\results\a3b-50k-current-profile\nsys-node.nsys-rep'
& $nsys stats '--report=cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_kern_sum,nvtx_pushpop_trace,cuda_gpu_trace' `
  '--format=csv' '--output=G:\Strata-Moe\bench\results\a3b-50k-current-profile\stats' `
  'G:\Strata-Moe\bench\results\a3b-50k-current-profile\nsys-node.nsys-rep'
```

This sequence is for the root operator to execute and adapt. This audit did
not start it or touch GPU 0. Ensure the profile config's `port` and `log`
match the started server and restore the local 8096 service afterward.

## Attribution and decision rule

Within the **output** NVTX range, report GPU kernel duration and launch count
for each target-verify and MTP-draft/catch-up range separately. Q6_K is
`GGML_TYPE_Q6_K = 14` in `third_party/llama.cpp/ggml/include/ggml.h`;
inspect `mul_mat_vec_q<(ggml_type)14,...>` and grouped Q6 variants by
`gridX` (head versus 8,192/4,096/2,048/512-row trunk). Classify routed
Strata experts by `plan_expert_groups`, `native_gu_fused_kernel`,
`swiglu_q8_1_entries_kernel`, and `native_down_*`; GDN by GGML
`gated_delta_net_cuda` plus its normalization/activation launches, or by
the adapted `gdn_a3b_step_norm_kernel` when its opt-in is on. Include IQ2_S
and IQ3_S MMQ/MMVQ, flash attention, Q8 quantization, transfers and other
kernels. Check `cudaMemcpyAsync`, `cudaStreamSynchronize`, event waits and
`cudaGraphLaunch` in the CUDA API trace, plus gaps between GPU kernels. The
selected native-GPU path should still show zero CPU expert rows and zero
expert upload MiB; H2D/D2H events may reflect tokens/hidden, not experts.

Kernel-duration sums can overlap across streams and include profiling
overhead. Do not divide them by unprofiled wall time as if they were a
single exclusive budget. Report GPU timeline busy/idle/overlap, per-phase
launch latency and host waits alongside totals. Compare a new optimization
against an interleaved, same-binary, unprofiled 50K/256 A/B with identical
request, sampling and context, plus first HTTP byte, first model delta,
output tok/s, prefill tok/s, sampled VRAM and answer/MTP acceptance. A
different answer or acceptance rate makes the speed difference a whole-route
effect, not proof that the changed kernel alone caused it. Only after this
attribution should Q6 trunk packing, GDN fusion, MTP cross-step graph or
another kernel be prioritized.

Nsight CLI behavior and CUDA Graph node-trace overhead: [NVIDIA Nsight
Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).
