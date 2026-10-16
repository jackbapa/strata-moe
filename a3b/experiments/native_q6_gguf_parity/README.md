# Isolated native Q6_K parity probe

This optional executable does not link into `strata-moe` and does not change
the selected service. It reads a **Q6_K dense tensor from the existing mixed
IQ3_S GGUF** by GGUF metadata and file offset. Both arms consume those exact
weight bytes and the same Q8_1 activation buffer produced by pinned GGML's
`quantize_row_q8_1_cuda`. One arm calls GGML's `ggml_cuda_op_mul_mat_vec_q`;
the other calls the original Strata `native_q6_k_mmvq`. It checks all float
output bits and reports the maximum absolute difference for 1, 2 and 3
columns, plus CUDA-event median times for Q8 quantization and each dot product.

The fixture requires a 2D Q6_K matrix with `n_in % 512 == 0`. For these rows,
GGML's padded Q8_1 column stride equals the tight stride required by Strata.
The automatic selection chooses the first 2048-wide Q6_K tensor with 512 to
8192 output rows. An exact tensor name may be passed as argument 2. The chosen
tensor name and shape are printed; no payload is rewritten.

Build **later, when GPU 0 and the service are available for a controlled test**:

```powershell
cmake -S G:/Strata-Moe/a3b/experiments/native_q6_gguf_parity -B G:/Strata-Moe/build-native-q6-parity -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build G:/Strata-Moe/build-native-q6-parity --target native_q6_gguf_parity -j 4
```

Run using the **same Q6 grouped/reuse environment switches as the selected
service**, then pass its IQ3_S GGUF path and optionally a tensor name and repeat
count:

```powershell
$env:STRATA_A3B_Q6_GROUPED_EXACT = '1'
G:/Strata-Moe/build-native-q6-parity/native_q6_gguf_parity.exe G:/path/to/Qwen3.6-35B-A3B-UD-IQ3_S.gguf
```

The default repeat count is 40. The reported dot times exclude Q8
quantization; its median is listed separately. These isolated kernel times do
not measure 50K first-token latency or end-to-end output speed. A production
port must also pass complete strict target/MTP validation and same-binary
50K/256 request benchmarks before enabling any alternate kernel.

This source has deliberately **not** been compiled or executed yet.
