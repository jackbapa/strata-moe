# Qwen3.6-35B-A3B 完整 Q8_0 来源与兼容性核验

核验日期：2026-10-14。只读 HF API、HTTP Range 和现有本地 MTP 文件，没有下载整个模型，没有请求 GPU 推理，也没有修改现有 Q4 权重。

建议选择 **ggml-org 的标准 Q8_0**：120/120 个 routed expert 张量都是真实 Q8_0，输出头、输入 embedding 和 output norm 与已有同仓 MTP 的完整 tensor payload SHA-256 严格相同，可以直接共享，无需重组或再次量化。

## 确切来源

| 字段 | 已核验值 |
| --- | --- |
| Repo | `ggml-org/Qwen3.6-35B-A3B-GGUF` |
| Revision | `baec3ebee244827cda0f4557eafa8b28f7545fa6` |
| 文件 | `Qwen3.6-35B-A3B-Q8_0.gguf` |
| 文件字节数 | 36,903,139,328（34.3687 GiB） |
| HF LFS SHA-256 | `d8d7842cc657d720f39546878e431937c84473aa130c36371669ac17c80c7361` |

[固定 revision 下载地址](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/resolve/baec3ebee244827cda0f4557eafa8b28f7545fa6/Qwen3.6-35B-A3B-Q8_0.gguf?download=true)、[HF API](https://huggingface.co/api/models/ggml-org/Qwen3.6-35B-A3B-GGUF/revision/baec3ebee244827cda0f4557eafa8b28f7545fa6?blobs=true)。上面的整文件 SHA 是 HF 发布值；本项研究没有本地完整下载或重新计算该整文件 SHA，下载器仍须在完成后验证。

供比较，Unsloth 相同固定 revision `a483e9e6cbd595906af30beda3187c2663a1118c` 的同名标准 Q8_0 为 36,903,140,320 字节，发布 SHA 为 `d1a395809f65a43a13ad119eb4e7acdef1ac6d68120f39902c8ab96e72794a59`：[Unsloth API](https://huggingface.co/api/models/unsloth/Qwen3.6-35B-A3B-GGUF/revision/a483e9e6cbd595906af30beda3187c2663a1118c?blobs=true)。两个整文件 SHA 不同，本研究不宣称它们的全部权重 payload 一致。另有 `UD-Q8_K_XL`，本次没有依靠其文件名推断量化，也没有选择它。

## 实际 GGUF header

通过验证状态码 206、Content-Range 和长度的 Range 请求读入前 16 MiB，完整解析 GGUF v3 metadata 与 733 个 tensor 描述。tensor 数据起点是 10,989,056 字节。

- 架构 `qwen35moe`，40 层，hidden=2048，256 个专家 / 每 token 选 8 个；模型声明 context=262144，本轮服务 context 另行配置。
- 所有 tensor 类型：432 × Q8_0，301 × F32。
- Routed gate/up/down 专家：**120 × Q8_0**，共 34,225,520,640 字节，即 **31.875 GiB**；没有 Q4、Q5、Q6、IQ3 或其他低于 8 bit 的专家。
- 标准 Q8_0 每 32 个权重占 34 字节，含 FP16 scale。小型 norm、bias 等保留 F32 是更高精度存储，不是低位混合量化。
- Header SHA-256：`52d6d76053e69be2afd77e28adf81a8f01107cd2ba420b222b82225a3f25fb13`。

## 与已有 MTP 的严格共享

已有 MTP：同 repo/revision 的 `mtp-Qwen3.6-35B-A3B-Q8_0.gguf`，1,990,649,632 字节，已记录文件 SHA `f6cb995fab78e7284097b2e13508e8daad4ffc6a747b433aef399f4e48524b5a`。

在 header 类型、shape 比较后，远程三表的全部 1,080,696,832 字节通过 Range 流式读取并分别 SHA-256；再与本地 MTP 对应 tensor 的全部字节比较，三项都相同：

| Tensor | 类型 / shape | payload 字节数 | 两端相同的 SHA-256 |
| --- | --- | ---: | --- |
| `token_embd.weight` | Q8_0 `[2048,248320]` | 540,344,320 | `a1c8a86668c05ad6392b1e55bb7b076e27a0e20eb1e553ab46df4c18f2d0191f` |
| `output.weight` | Q8_0 `[2048,248320]` | 540,344,320 | `d39064714a29049c8afbb96307188dc45828bc6b1140f6b2efe7873ab31ef19d` |
| `output_norm.weight` | F32 `[2048]` | 8,192 | `19f7f87ea054c19dee992e4d1ce5a2d3468b067b9f300fc031e58048f1d750a0` |

新 Q8 的完整 tokenizer metadata 与该 MTP 相同。相较现有 Unsloth Q5/Q4 路径，只有两项 tokenizer metadata 差异：chat template 长度 7,764 对 8,057，以及 padding token ID 248044 对 248055；词表、merge、pre 等其余 tokenizer 字段相同。新 Q8 应单独导出 tokenizer 目录，保留既有 Q4 目录及配置。

## 引擎与内存结论

当前固定 llama.cpp 的 CUDA MMVQ、MMQ 和融合专家分支均有 Q8_0 实现，A3B 的专家布局按 GGUF tensor shape 读取；存储格式与现有同仓 MTP 兼容。实际 GPU-only RAM 专家调度、MTP 逐 token reference、回滚和接受边界仍需要下载后的端到端验证，不能由 header 核验代替。

**34.37 GiB 的 Q8 模型无法完整驻留本机 22 GiB 的 GPU 0。** 仅 routed 专家已有 31.875 GiB。因此应使用 RAM 保存 Q8 专家、GPU 缓存热点，并把未命中完整专家分批传到有界 GPU staging slots 计算；不能沿用一次把所有权重装上 GPU 的 Q4 `--native-gpu` 路径。该路径的实际速度和 RAM RSS 尚未在本项研究中测量，不承诺沿用 Q4 的 102.7 token/s 代码题速度。

完整 API 记录、733 项 tensor 描述、geometry、header/共享三表哈希和比较结果：[API 证据](ggml-org-Qwen3.6-35B-A3B-Q8_0-api.json)、[核验结果](ggml-org-Qwen3.6-35B-A3B-Q8_0-audit.json)。复现脚本：[remote-audit.py](remote-audit.py)。

```powershell
.venv\Scripts\python.exe bench/results/2026-10-14-a3b-q8-research/remote-audit.py --hash-shared
```

该脚本只读约 16 MiB header 与 1.01 GiB 的共享三表，不会下载完整 34.37 GiB 模型。
