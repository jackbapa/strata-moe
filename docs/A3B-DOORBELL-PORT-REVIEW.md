# Original Strata doorbell handoff and a minimal A3B port

Source research on 2026-10-14. No implementation, GPU execution or API request
was performed for this review. Paths beginning `src/` or `include/strata/` below
refer to the read-only original `G:\Strata` tree. A3B/vendor paths refer to
`G:\Strata-Moe`. The original checkout has no `src/core/capture.cpp`; capture
and replay live in `src/core/session.cpp` and `src/core/verify.cpp`.

The useful transplant is the execution handoff: keep CPU expert work alive
while a single captured CUDA graph advances through the model. It does not
require a new cache policy, a different quantization or changing Q8 arithmetic.

## What the original engine actually does

The simple whole-token path is `session_capture_token` (`session.cpp:838`).
For each layer it captures router/input publication, GPU cached experts, a
device wait for the CPU result, a mapped-result copy kernel and the layer's
combine. `session_run_token` (`:912`) launches the token graph once, polls the
published sequence, runs the CPU pool and publishes the corresponding completion
flag. The final stream synchronization happens after all layers.

The current speculative verifier has the more complete three-way protocol:

| Signal | Writer | Meaning | Consumer |
|---|---|---|---|
| `seq` | GPU publication kernel | This layer/group's input and IDs are host-visible | CPU service loop |
| A | CPU planner | GPU cache-hit and PCIe expert plans are ready | GPU plan-copy/hit kernels |
| B | Copy-stream completion callback, or immediate CPU store for no copy | This group's PCIe weights are ready | GPU PCIe expert kernels |
| M (`h_flag_`) | CPU service loop | CPU expert output rows are ready | GPU mapped-row copy/combine |

`src/core/expert_source.cpp:3664` builds the GPU plan first. Cache hits are
grouped by distinct expert. A fraction of distinct misses is selected for PCIe;
every occurrence of one expert has the same destination. The remaining misses
go to the CPU. At `:3790` the host publishes the plan before activation
quantization and CPU expert evaluation.

`Verifier::publish_plan` (`verify.cpp:2432`) executes a host store fence and
raises A. `fetch_dma` (`:2403`) starts copies on a separate nonblocking copy
stream. Its completion host function only raises B. The CPU then quantizes
activations and runs its persistent pool while GPU cache hits and DMA proceed.

The captured post section (`verify.cpp:1594`) waits for A, copies the small plan
from mapped memory and computes cache hits. It waits for B only before the PCIe
expert work, then for M only before consuming CPU rows. Thus CPU arithmetic,
GPU cache-hit arithmetic and expert DMA can overlap. GPU cache-hit and PCIe
expert arithmetic normally follow one another on the compute stream; they are
not three independent GPU compute streams.

After all expert contributions are ready, the original combine preserves the
router weights and layer dependency. CPU work does not generally hide behind
the next layer's attention: that layer depends on the completed current layer.
The verifier optionally pipelines token groups, but that is beyond the minimum
port proposed here.

## Publication and memory ordering

`doorbell_init` (`src/core/layer.cpp:1186`) allocates mapped pinned host memory
and obtains its device aliases. CPU and GPU views point to the same backing
bytes. This mailbox is distinct from the large expert weight arena.

`doorbell_publish_kernel` (`src/kernels/cuda/elementwise.cu:311`) copies input
floats and IDs, optionally router weights, into mapped destinations. Its
threads execute `__threadfence_system()`, join at `__syncthreads()`, and thread
zero publishes a volatile sequence store. The original A3B-sized payload would
be 2048 input floats per token and eight IDs. The result mailbox holds unweighted
expert rows; the existing router-weighted combine can remain on the GPU.

The CPU polls the volatile sequence, computes output, executes a host fence
and `_mm_sfence()`, then writes the completion flag. The GPU wait kernel reads
the volatile flag, executes a system fence after observing completion, and the
next kernel reads the mapped output. A sequence store without the payload
fences is not the original protocol. Natural alignment and single-writer flag
ownership matter; GPU atomics on mapped memory are not universally CPU-atomic.
[NVIDIA mapped-memory documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#mapped-memory)
describes that limitation and the PCIe cost of mapped accesses.

The simple token graph uses a one-thread `doorbell_wait` and
`copy_from_mapped`; the verifier uses monotonic `wait_flag_ge` and can fold the
wait into `copy_rows_from_mapped_after`. GPU-owned output lanes are skipped or
zeroed during the CPU copy so a stale CPU mailbox cannot overwrite them.

This is not zero data transfer. Input, plans and CPU results cross PCIe. Their
publication and consumption are kernels inside the graph, so they avoid a
host-issued copy/synchronization at every layer. The source explicitly records
that replacing the result-copy kernel with a copy-engine node fragmented WDDM
submission (`session.cpp:883`, historical measurement of 67 flushes/token).
That number belongs to the original measurement, not this A3B hardware run.

## Why the loop does not call stream synchronization at each layer

The stream already orders GPU producers, expert consumers and combines. A
mapped sequence tells the CPU when the specific payload is available; A/B/M
tell the GPU when the specific host dependency is available. Waiting for the
entire stream at each handoff would wait for work whose progress itself requires
the host to serve that handoff.

The current verifier launches once, calls `cudaStreamQuery` once to submit work,
then polls memory. It calls the driver again only if a ring remains absent for
about 2 ms, to flush/diagnose stalled submission (`verify.cpp:2171–2228`). Some
older header commentary describes querying every spin; that is not the current
verifier loop. Window completion still synchronizes the compute stream, and
joins the copy stream when used (`:2280–2299`). This also prevents an old B
callback from satisfying the next window's flag.

## Smallest viable port through the existing GGML graph

The current A3B ACCEL fused backend executes host routing/copies and nested GPU
tasks in `Backend::compute_fused`. It is a host scheduler boundary. Adding a
doorbell inside that function alone cannot make the whole model one CUDA graph.

The minimum viable route is an opt-in CUDA-native hybrid expert custom operator,
using the same integration pattern as the existing native CUDA expert marker:

1. Keep the current Qwen graph builder, recurrent state, attention, dense exact
   hooks, head, MTP and GGML scheduler. The new expert op has only GPU input and
   GPU IDs as graph sources. Immutable host Q8 weights and the existing cache
   live behind a prepared opaque descriptor. They are not host graph inputs.
2. Allocate and prepare descriptor/cache/mailbox/CPU scratch outside capture.
   Start one persistent CPU service thread before entering `llama_decode` or
   a verification call. It must be available during first uncaptured execution,
   graph warmup, capture replay and steady-state replay. Starting it after
   `llama_decode` returns would deadlock at the first device wait.
3. First port only `pcie-frac=0` with the cache map fixed for a window. The CUDA
   op enqueues publish, GPU cache-hit work, M wait and mapped CPU-row copy.
   The service thread runs the already implemented CPU pool on the same missed
   experts, then publishes M. Keep each path's Q8 quantizer, reduction, SwiGLU
   and output layout identical to the measured hybrid baseline. The original
   cache/profile policy can update between completed windows.
4. Require the expert op and trunk to remain in the same CUDA scheduler split
   for the intended short-window graph. Physical GPU input/ID residency and
   split count must be checked; the small fixture alone does not establish the
   full-model split count. The existing CUDA backend already performs stable
   graph warmup and capture/replay (`ggml-cuda.cu:4426–4496`). Its provider
   preparation hook runs before capture. Reuse this mechanism rather than
   manually recapturing all 40 layers in a separate model implementation.
5. At the call boundary, wait for the service to become idle while retaining
   its persistent thread, collect counters/errors and let the
   existing decode/MTP code perform acceptance, rollback and state commit.
   Keep target and draft context mailboxes/epochs distinct, or explicitly
   serialize their use. A captured graph retains buffer addresses; those
   buffers must survive all graph executions and graph destruction.

The previously validated Q8 GPU kernel arithmetic and the new CPU pool are
components, not a reason to enable the unrelated static/dynamic cache experiment
flags. The new bridge needs its own opt-in mode/descriptor semantics because
the existing static mode requires GPU-only execution. Invalid IDs and aborted
calls still need the existing non-finite/error guards.

For T <= 8 and broadcast input, one shared mailbox needs at most 64 KiB input,
256 bytes IDs and 512 KiB expert output, plus a small plan and cache-line-sized
flags. These are calculated upper bounds, not measured allocations. Sequential
layers can reuse it because the next publication follows the previous result
copy on the same stream. Keep large weights in ordinary RAM for CPU work;
mapping the entire Q8 expert arena is not required for this handoff.

After that path passes and is measured, wire the existing distinct-expert
PCIe split into A/B/M, keeping current bounded transfer banks and policy.
Do not put expert DMA into the main compute stream before cache hits. If a
window's misses exceed one bank, the captured consumer needs fixed-capacity
batch descriptors and per-bank readiness/reuse acknowledgement; a captured
shape cannot be rebuilt from live host IDs during replay. This is an execution
interface issue for the current split implementation, not a new cache design.

## Windows and shutdown constraints supported by evidence

- Use mapped pinned mailboxes, not managed memory as a substitute. NVIDIA
  documents limited concurrent CPU access to managed allocations on Windows,
  whereas mapped memory is a separate mechanism. Small mailbox mapping failure
  must be reported; it does not justify pinning another full 32 GiB expert
  copy. Large pinned/mapped arenas have separate WDDM allocation observations
  in original `generate.cpp:4430,4485,4702`.
- A CUDA completion host function must not submit CUDA work. The original B
  callback only stores a flag; DMA is submitted by the normal host service
  thread. Calling the nested A3B GPU backend or allocating CUDA buffers inside
  that callback would violate [NVIDIA's callback rules](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#callback-functions-from-streams).
- GPU waits need a host abort/release protocol. Original
  `Verifier::release_gpu_waits` (`:381`) raises A/B/M to `UINT32_MAX`, fences,
  and drains streams with a bounded wait; the verifier's predicates use `>=`.
  A new equality-only wait would not be released by that technique. On a CPU
  error, release pending GPU waits, discard the call's output and drain before
  freeing mailboxes or callbacks. The original source explicitly records a
  Windows GPU-loss incident from spin kernels outliving a stalled process
  (`verify.cpp:2280`).
- Microsoft's [TDR documentation](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/timeout-detection-and-recovery)
  describes a default two-second preemption wait. This does not mean every
  two-second inference call or graph triggers TDR, and the current machine's
  effective settings were not inspected here. It does mean an unbounded
  device spin cannot be treated as harmless. No registry or driver settings
  need to be changed for this proposal.

## Acceptance evidence before a speed claim

Test warmup and repeated graph replay, each supported T, strided IDs, changing
routes, mixed/zero/full GPU cache capacity, duplicates, CPU/GPU lane ownership,
abort/recovery, and long-prefill transitions. CPU and GPU results should match
their corresponding existing arithmetic references. Then test real Q8 MTP
accepted prefixes and rollback with the same placement contract.

Record full-model CUDA split/graph replay counts, host route reads, driver-call
counts, CPU pool time, GPU wait time, transfer bytes and MTP acceptance before
attributing any speed gain to this port. End-to-end TTFB, TTFT, prefill, decode
and actual RAM/VRAM remain required. This review establishes an implementation
path; it does not establish 100 tokens/s or any new benchmark result.
