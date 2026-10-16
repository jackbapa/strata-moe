# Q8_0 MTP output-head study

## Measured hot path

The exact selected model remains `Qwen3.6-35B-A3B-UD-IQ3_S.gguf` plus its
matching `mtp-Qwen3.6-35B-A3B-Q8_0.gguf` head (SHA-256
`f6cb995fab78e7284097b2e13508e8daad4ffc6a747b433aef399f4e48524b5a`).
The Q6_K target tensors live in the **same IQ3_S GGUF**, and the Q6_K
target-only optimization cannot change the Q8_0 MTP proposal kernels.

The previously recorded partial Nsight Systems SQLite capture covers 61
complete `a3b/output/mtp-propose` NVTX windows on GPU 0 (RTX 2080 Ti). A
read-only join from each NVTX range through CUDA runtime correlation to the
kernel records shows:

| MTP Q8_0 MMVQ grid | Calls | GPU kernel-duration sum | Model matrix |
|---|---:|---:|---|
| `(248320,1,1)` | 122 | 119.057 ms | `output.weight` `[2048,248320]` |
| `(2048,1,1)` | 244 | 4.965 ms | MTP projection(s) |
| `(8192,1,1)` | 122 | 4.054 ms | `attn_q.weight` |
| `(512,1,1)` | 488 | 2.316 ms | `attn_k/v` and shared FFN projections |
| **Total** | **976** | **130.392 ms** | SQLite duration rounded per grid |

The output head accounts for **91.3%** (119.057 / 130.393) of captured
proposal Q8_0 kernel time, averaging 0.976 ms per call. All 976 recorded
`mul_mat_vec_q` calls use GGML Q8_0 type 8 and `ncols_dst=1`; the proposal
runs two draft steps in a typical K2 window. This is kernel-duration
attribution, not a complete service wall-time measurement. The trace is
partial, profiler-perturbed and also includes other target/draft work.
`q8_metadata.py` reproduces the metadata and SQL queries without using GPU.

## Applicability of original Strata kernels

`G:/Strata/src/kernels/cuda/native_mmvq.cu:2417`'s `native_mmvq_pair`
requires two equal-shaped weight matrices operating on the same Q8_1 input,
at least two columns, and the exact native layout. None describes the
single-column MTP output head. Likewise A3B's existing opt-in
`STRATA_A3B_DENSE_Q8_REUSE` in GGML `mmvq.cu` handles two to eight
**independent** columns and cannot accelerate the dependent K2 proposal
steps. These are not candidates for the observed 119.057 ms.

Original Strata's `native_q8_0_packed_kernel<1>` at
`G:/Strata/src/kernels/cuda/native_mmvq.cu:1707` is shape-eligible for the
MTP head. It copies the *same* Q8_0 bytes from interleaved 34-byte blocks
to separate `qs` and fp16-scale planes (no requantization). The source
retains the GGUF-layout weights for other kernels. For `[2048,248320]`,
the extra packed allocation is exactly
`248320 * (2048 / 32) * 34 = 540,344,320 bytes = 515.3125 MiB`, plus small
allocation/registry overhead. Initialization requires one GPU repack or
host repack + upload and can increase startup time; it does **not** save
VRAM. The `STRATA_Q8_PACKED=1` result documented in that source (+10–12%
per eligible call, +2.3% end-to-end default) was measured on gfx1151 with
different shapes. It cannot be claimed for sm75 or this 248,320-row head.
`G:/Strata-Moe/src/kernels/cuda/native_mmvq.cu` has the same SHA-256 as
the original source, but the A3B CMake target does **not** currently build or
call this file. Its presence on disk is not an executed Strata kernel.

For the output head's 2048-element input, original packed lane mapping
(`kbx=tid/4+32*i`, `kqs=2*(tid%4)`), four warps, two K-loop iterations,
integer DP4A order, fp16 scale conversion, ascending cross-warp sums and
XOR warp reduction match GGML's **single-column, non-fused, non-ID** sm75
MMVQ source (`G:/Strata-Moe/third_party/llama.cpp/ggml/src/ggml-cuda/mmvq.cu`
and `vecdotq.cuh`). This supports a bitwise-exact hypothesis for finite
real inputs, but compiler flags, launch details and source identities still
need an actual full-vector `memcmp` before runtime use. In particular,
copying the original Strata Q8_1 quantizer is unnecessary: the packed
kernel must consume the *existing GGML-produced* Q8_1 activation buffer.

## Minimal default-off candidate

1. Add an sm75/CUDA-only, explicit `STRATA_A3B_MTP_Q8_PACKED_HEAD=1` setup
   after the draft model loads and before first graph capture. Require the
   matching draft output tensor to be Q8_0 `[2048,248320]`, device 0,
   GPU-resident, unshared and non-aliased; fail closed otherwise. Allocate
   540,344,320 bytes on the same GPU, losslessly repack from the existing
   device weight buffer, synchronize initialization and register the source
   weight pointer. Free the allocation before the draft model and backend
   are destroyed, including exceptional paths. No filesystem model change.
   Add the copied original `native_mmvq.cu` source to the A3B CUDA library
   with the original project's `--use_fast_math` compile setting
   and add a small device repack entry point there; avoid another 515 MiB
   transient host buffer and repeated PCIe download/upload.
2. At `ggml_cuda_mul_mat_vec_q` dispatch, select the original Strata
   `native_q8_0_packed_kernel<1>` arithmetic only when its registered
   pointer/dimensions and GGML's singleton, un-fused, un-routed contiguous
   output-head layout all match. Reuse GGML's quantized Q8_1 activation.
   All other tensor/architecture/batch paths remain the current GGML
   dispatch. The flag defaults off and startup logs packed bytes/count.
3. Before service A/B, compare the entire 248,320-element F32 output
   vector byte-for-byte for many real MTP activations and real output
   weights, including low-amplitude/zero blocks. Record mismatch count and
   first differing bit if any. Compare CUDA-event timings with 50+ paired
   warm repeats on sm75, reporting median/range and the cost of startup
   repack. A mismatch or non-improvement keeps it disabled.
4. Only if parity and kernel speed pass, run same-binary alternating
   off/on 50K exact input / 256 output HTTP A/B with the selected IQ3_S
   target, Q8_0 MTP K2, 262,144 configured context, GPU 0, frozen
   sampling, output/proposal hashes and acceptance, first HTTP byte,
   first model token, prefill and output tok/s, and whole-card VRAM peak.
   Confirm with Nsight that the **original packed CUDA kernel is actually
   launched**, that CUDA graph execution still occurs, and report warm vs cold
   startup separately. The output head is outside prefill and alone
   cannot establish a 100 tok/s outcome.

No source patch has been applied to `third_party/llama.cpp` or `a3b/patches`.
The isolated standalone probe below was compiled and measured on GPU0. It
does not establish runtime GGML parity or service performance.

An isolated CUDA-event preflight source is in `packed_head_probe.cu`.
Its probe-only IQ and PDL host symbol stubs allow standalone linking of the
unmodified copied Strata source; IQ dispatch is never used and PDL is off on
the sm75 probe.
It streams the **real** draft `output.weight` Q8_0 blocks from the GGUF at
file offset **10,944,288**, uses original Strata's `native_q8_0_pack_host`
and compares the full 248,320-element output bit-for-bit for 12 different
finite/zero activation vectors before timing 51 alternating original
GGUF-layout vs packed launches. It reports allocation and startup repack
cost. This probe's control is Strata's native GGUF-layout kernel, **not**
the running GGML path. The probe was built on 2026-10-16 after the other
GPU/build tests released their window, using the target project's copied
Strata source and the following command from `G:/Strata-Moe` (inside an MSVC
`vcvars64.bat` environment):

```powershell
nvcc -std=c++20 -O3 -use_fast_math -arch=sm_75 -Iinclude -Isrc/kernels/cuda `
  src/kernels/cuda/native_mmvq.cu `
  bench/results/2026-10-15-a3b-draft-q8-study/packed_head_probe.cu `
  -o bench/results/2026-10-15-a3b-draft-q8-study/packed_head_probe.exe
bench/results/2026-10-15-a3b-draft-q8-study/packed_head_probe.exe `
  models/a3b/mtp-Qwen3.6-35B-A3B-Q8_0.gguf 10944288
```

The build exited zero. The real-weight probe exited zero on GPU0
(`NVIDIA GeForce RTX 2080 Ti`, sm75). For **12** finite/zero synthetic
Q8_1 activation cases, all **248,320** F32 logits from original Strata
GGUF-layout and original Strata packed kernels matched bit-for-bit. These
activations are not captured from the running GGML draft model.

For **51 alternating paired CUDA-event samples** after eight warm-up pairs:

| Original Strata standalone kernel | Median | Observed range |
|---|---:|---:|
| GGUF-layout Q8_0 | 962.560 µs | 960.832–2795.648 µs |
| Packed Q8_0 | 1159.392 µs | 1149.312–3475.584 µs |

On this RTX 2080 Ti, packed was **20.45% slower by median latency** than
the original Strata GGUF-layout kernel on the same real head. The packed
copy also adds **540,344,320 bytes / 515.3125 MiB** of VRAM. The probe
reported **702.452 ms** to read and pack the GGUF and upload **both** raw
and packed 515.3125-MiB arrays; that is the probe's total dual-copy
initialization and is **not** the isolated incremental packed setup cost
for an already-loaded GGML model.

This fails the improvement gate. Do **not** integrate the packed candidate
or run its 50K service A/B. The current service's GGML Q8_0 MMVQ remains
the selected path. The source-level bitwise hypothesis for a GGML-produced
Q8_1 activation has not been directly measured, because a slower and
VRAM-heavier standalone kernel gives no reason to take that integration risk.
