# Qwen3.6 A3B：22 GiB 单 GPU 量化选择，2026-10-14

优先测试 **UD-Q4_K_S**。它保留非专家张量的高精度，同时让专家权重和现有 Q8 MTP 有机会全部放入报告 22 GiB 的 RTX 2080 Ti。这里给出的是权重大小核算，尚未测量全 GPU 加载峰值、速度或质量；不能把“能放下”直接当成“达到 100 tokens/s”。

数据来自 [Unsloth 固定版本的 Hugging Face LFS API](https://huggingface.co/api/models/unsloth/Qwen3.6-35B-A3B-GGUF/revision/a483e9e6cbd595906af30beda3187c2663a1118c?blobs=true) 和每个 GGUF 文件开头 12 MiB 的 Range 响应。只下载了头部，没有下载这些完整量化模型。模型固定版本为 `a483e9e6cbd595906af30beda3187c2663a1118c`；完整机器可读结果在 `quant-header-results.json`，重现脚本在本目录。

| 量化文件 | 精确文件字节 | 专家权重字节 | 目标模型 + Q8 MTP 的 GPU 权重 GiB | 22 GiB 留给运行时的 MiB |
|---|---:|---:|---:|---:|
| UD-Q4_K_S | 20,893,015,008 | 18,327,011,328 | 20.2852 | 1,755.98 |
| UD-Q4_K_M | 22,134,528,992 | 19,568,525,312 | 21.4414 | 571.98 |
| UD-IQ4_XS | 17,730,509,792 | 15,164,506,112 | 17.3399 | 4,771.98 |
| 已下载 UD-Q5_K_M | 26,456,194,016 | 23,767,023,616 | 25.4663 | -3,549.48 |

GPU 权重核算沿用本引擎固定 llama.cpp 的输入嵌入放置规则：`token_embd.weight` 留在 CPU。每个目标量化模型的输入嵌入为 540,344,320 字节。现有 Q8 MTP 总张量 1,979,705,344 字节，扣除同大小的 CPU 输入嵌入后，GPU 张量为 1,439,361,024 字节。表格尚未扣除桌面显存、KV/GDN 状态、计算缓冲区、CUDA 工作区和图缓存，因此 Q4_K_M 剩余 572 MiB 很紧，Q4_K_S 更稳。实际模型加载是最终依据。

文件名不足以说明专家精度，头部实际是：

- **UD-Q4_K_S**：117 个专家张量为 Q4_K，3 个为 Q6_K；没有 3 bit 专家。
- **UD-Q4_K_M**：80 个 Q4_K、37 个 Q5_K、3 个 Q6_K。
- **UD-IQ4_XS**：80 个专家张量实际上为 **IQ3_S**，37 个为 IQ4_XS、3 个 Q6_K。它会降低大量专家到 3 bit，不作为本次优先方案。
- 三个候选的非专家部分相同：361 个 F32、251 个 Q8_0，以及 Q6_K 的 `output.weight`。所以 Q4_K_S 已经是保留高精度主干的混合量化，而非把整个模型一律降为 4 bit。

更保守的自定义方案可以保留 Q4_K_S 专家，并从已有 Q5_K_M **直接复制** Q8_0 的 `output.weight`，其余非专家张量保持不变。输出头从 417,177,600 增至 540,344,320 字节，仅增加 123,166,720 字节（117.46 MiB）；目标 + MTP 的 GPU 权重仍只有 21,904,208,384 字节，约 20.400 GiB。这样非专家精度与现有 Q5 模型完全一致。不能把 Q6 重新量化为 Q8 当作精度恢复，必须复制现有高精度来源。此方案这里只核算，尚未生成或评估质量。

四个目标量化 GGUF 的全部 `tokenizer.*` 元数据逐项哈希完全相同，`qwen35moe.*` 架构元数据也完全相同；都没有内置 MTP 张量。因此已有独立 Q8 MTP 可以继续匹配同一词表和架构。量化可能改变 MTP 的接受率，需要重新测量。Q4_K 的 CPU/CUDA 专家算子已在本项目现有 parity 中覆盖；仍需做完整模型运行验证。

推荐下载项：

```text
repo: unsloth/Qwen3.6-35B-A3B-GGUF
revision: a483e9e6cbd595906af30beda3187c2663a1118c
file: Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
bytes: 20893015008
sha256: a8138f183e3993f12cdc23afd2babb8cdb084e64088ce4a256d49101d47b949c
```

来源和兼容性依据：[Qwen 官方模型](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)、[固定版本量化文件](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/a483e9e6cbd595906af30beda3187c2663a1118c)、[ggml-org Q8 MTP](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/blob/baec3ebee244827cda0f4557eafa8b28f7545fa6/mtp-Qwen3.6-35B-A3B-Q8_0.gguf)、[固定 llama.cpp 输入层放置代码](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/llama-model.cpp)、[官方量化工具说明](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)。官方工具支持对指定张量选择精度，并提醒从已量化权重再次量化可能损伤质量；这里优先采用现成权重字节复制。
