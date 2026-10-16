# A3B 单 GPU 常驻推理：完整 API 延迟测试，2026-10-14

本机单张 GPU 0 上，Qwen3.6-35B-A3B 的自然代码题三轮输出中位数为 **102.7 token/s**，自然中文解释题为 **82.2 token/s**；合成中文提示词矩阵为 **73.2–78.2 token/s**。100 token/s 已在该代码题达到，尚未成为所有题型和采样设置的通用速度。2048-token 输入的预填充中位数为 **1815.8–1819.2 token/s**。

这是完整 GPU 常驻路径，模型专家、主干、注意力和 MTP 的矩阵计算都在 GPU 0。没有 CPU 专家计算，没有测试间模型卸载，也没有空闲显存释放。普通 CPU 仍负责 HTTP、分词和调度；贪心选择使用 GPU sampler，非零温度/penalty 的取样器仍在 CPU 选择 token，因此两种采样模式分别报告。

## 硬件、模型和配置

- Windows，AMD Threadripper 3960X，24 物理核心 / 48 逻辑线程，128 GiB RAM。
- 仅 GPU 0：RTX 2080 Ti，计算能力 7.5，驱动报告 22,528 MiB；CUDA 12.6。这是 22 GiB 显存环境，不是物理 16 GiB 卡测试。
- 目标模型：`Qwen3.6-35B-A3B-Strata-Q4_K_S-Q8Head.gguf`。下载的 UD-Q4_K_S 有 117 个 Q4_K 和 3 个 Q6_K 专家张量；只把 `output.weight` 从同 revision 的 UD-Q5_K_M 原样复制为 Q8_0，未重新量化其余张量。输入 embedding 原为 Q8_0。
- 独立 Q8_0 MTP 头；目标与草稿的输入/输出表及 output norm 在类型、形状与 SHA-256 校验后共享 GPU 权重，启动日志报告复用 1,030.6 MiB。
- `--native-gpu --gpu-embedding --gpu-sampling --share-mtp-weights --defer-mtp-catchup --fused-moe --exact-verify --native-attention-exact`。
- MTP draft=2，主模型 batch=1024，MTP microbatch=512，context=4096，`--device 0 --vram-mib 0`，`STRATA_A3B_ARGMAX_MULTI=1`，`idle_unload_s=0`。

启动日志显示目标模型 41/41 层、MTP 42/42 层均 offload 到 GPU，目标与 MTP 的 `graph splits = 1`。其中保留的 `CUDA_Host` 模型缓冲是共享表原加载副本，不是 CPU 专家执行路径。原 Strata 的较大模型确实让 CPU 就地计算部分未命中专家，并与 GPU 缓存和 PCIe 分流并行（[DETAILS 的 CPU 与 PCIe 说明](../../../docs/DETAILS.md)）；本配置的 Q4 模型能全部驻留 GPU，直接消除了该分工及专家传输。

配置、启动证据与文件标识：

- [服务配置快照](server-config.json)、[环境](environment.json)、[启动日志](startup.log)。
- 目标模型 SHA-256：`9a91f467e1dc65f40487cc8f495b29a294a8afdbde7eb22dc4bf746ad350161e`。
- MTP 文件 SHA-256：`f6cb995fab78e7284097b2e13508e8daad4ffc6a747b433aef399f4e48524b5a`。
- 测试引擎 SHA-256：`0ef75ed69ee5fdbde27a0c8f5f2d49a426d7154d56962a7bfa20b426c3f9d2ec`。
- [原样复制配方与逐张量哈希](../../../models/a3b/Qwen3.6-35B-A3B-Strata-Q4_K_S-Q8Head.recipe.json)。

## 计时口径

所有请求串行发送到 `http://127.0.0.1:8096`，模型预先加载，保持常驻。没有静默排除预热请求，每个条件三次全部计入中位数。以下输入 token 数包含聊天模板，以最终 API usage 为准；所有 30 次都与 count endpoint 一致，且 `cached_tokens=0`，没有通过复用提示词 KV 人为缩短 prefill。thinking 关闭，全部请求没有 reasoning 文本。

- **HTTP TTFB**：底层 socket 首次成功读到字节，包含 HTTP 状态行和 headers。服务器可以先回复流式 headers；它不等于模型已经输出。
- **输出 TTFT**：首个含非空 content 或 reasoning 的 SSE delta；角色、进度、心跳不计。本次无 reasoning，因此也等于首个可见文本等待时间。
- **Engine decode**：引擎有效生成 token 数 / `decode_ms`，不计模型加载和 prefill；不把 MTP 草稿尝试数加进输出量。引擎计数可能包含终止 token。
- **客户端 decode**：`(usage.completion_tokens - 1) / (末个非空文本 delta 时间 - 首个非空文本 delta 时间)`。MTP 和 UTF-8 解码会成批发送，因此它与 engine 数字略有差异。
- **Prefill**：引擎实际读入的 prompt token 数 / `prompt_ms`。短输入容易受图准备及固定开销影响，不能从 32-token 数字外推长输入。

## 主矩阵：合成中文提示词，temperature=0

长输入由重复的中性 MoE 资料组成，输出任务为中文解释。它是可复现的合成长提示词实验，另设下方自然代码和中文任务。表内除三次原始 decode 值外，均为包含全部三次的中位数。

| 实际输入 / 输出 token | HTTP TTFB ms | 输出 TTFT s | Prefill token/s | Engine decode token/s，三次 | 中位 decode token/s | 客户端 decode token/s |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| 32 / 256 | 10.61 | 0.1135 | 318.7 | 73.2 / 73.4 / 73.0 | 73.2 | 72.89 |
| 32 / 512 | 10.63 | 0.1121 | 319.7 | 73.7 / 73.7 / 74.2 | 73.7 | 73.60 |
| 512 / 256 | 10.75 | 0.4001 | 1320.6 | 76.9 / 76.2 / 76.9 | 76.9 | 76.59 |
| 512 / 512 | 11.23 | 0.3995 | 1325.4 | 75.9 / 75.7 / 75.9 | 75.9 | 75.74 |
| 2048 / 256 | 11.54 | 1.1417 | 1815.8 | 78.5 / 78.2 / 78.1 | 78.2 | 77.93 |
| 2048 / 512 | 11.24 | 1.1388 | 1819.2 | 77.1 / 77.1 / 77.1 | 77.1 | 76.95 |

本矩阵第一条 32/256 请求的输出 TTFT 是 **0.4600 s**，后两次中位数为 **0.1128 s**；第一条也保留在上述三次统计中。其 prefill 为 75.9 token/s，后两次中位数为 319.35 token/s。模型磁盘加载不在这两个时间内。

| 条件 | 第一次输出 TTFT s | 后两次 TTFT 中位数 s | 第一次 MTP 草稿接受率 |
| --- | ---: | ---: | ---: |
| 32 / 256 | 0.4600 | 0.1128 | 51.39% |
| 32 / 512 | 0.1120 | 0.1123 | 51.90% |
| 512 / 256 | 0.4022 | 0.3945 | 55.83% |
| 512 / 512 | 0.4007 | 0.3941 | 53.97% |
| 2048 / 256 | 1.1417 | 1.1379 | 59.23% |
| 2048 / 512 | 1.1388 | 1.1422 | 57.47% |

这里的“第一次”是每个条件内的请求顺序，不表示每个条件都冷启动。后两次仅有两次，不称作“三轮 warm 中位数”。每个条件的三次输出一致。

[完整 prompt 和计划](main/plan.json)、[18 次原始结果](main/results.json)、[整体/首轮/后续和逐次值](main/case-summary.json)。每次的 `*.events.jsonl` 保留全部 SSE 及时间戳，`*.output.json` 保留文本。

## 自然任务与非零温度

自然代码题要求用 OrderedDict 实现 LRU cache 并写断言；中文题要求解释浏览器访问 HTTPS 的过程。自然题没有填充到指定长度。非零温度两组复用主矩阵相同的 32/2048 合成提示词，明确设置 temperature=0.7、top_p=0.95、top_k=20、seed=12345。

| 任务 | 实际输入 / 输出 token | HTTP TTFB ms | 输出 TTFT s | Prefill token/s | 三次 Engine decode token/s | 中位 / 客户端 decode token/s | MTP 接受率 |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| 英文 LRU 代码，temp=0 | 124 / 512 | 10.45 | 0.2521 | 518.8 | 102.8 / 102.4 / 102.7 | 102.7 / 102.55 | 93.26% |
| 中文 HTTPS 解释，temp=0 | 98 / 273 | 1.33 | 0.1985 | 500.8 | 82.2 / 82.5 / 82.2 | 82.2 / 81.94 | 63.75% |
| 合成短输入，temp=0.7 | 32 / 256 | 10.32 | 0.0888 | 422.7 | 65.4 / 65.4 / 65.2 | 65.4 / 65.17 | 49.61% |
| 合成长输入，temp=0.7 | 2048 / 256 | 11.38 | 1.1163 | 1847.7 | 64.6 / 64.8 / 64.9 | 64.8 / 64.58 | 50.79% |

代码题的较高草稿接受率与 102.7 token/s 同时观测到；这不是所有代码请求的速度保证。其代码块完整，后续复杂度说明在 512-token 上限被截断，finish=`length`。人工阅读代码后执行了原生成断言和独立的非正容量、缺失 key、更新及淘汰检查，均通过：[代码](supplemental/natural-code.generated.py)、[检查记录](supplemental/natural-code-smoke.json)。中文题在 273 token 正常 `stop`。这些是内容与运行 smoke checks，不是 BF16 对比或完整模型质量评分。

自然代码题首次 TTFT 0.5022 s，后两次中位数 0.2459 s；表中 0.2521 s 仍按三次全部统计。各条件三次输出 hash 一致，包括本次固定 seed 的温度 0.7 两组；该有限观察不构成所有随机请求逐字复现的保证。

[四个准确任务及采样参数](supplemental/plan.json)、[12 次原始结果](supplemental/results.json)、[逐次/整体统计](supplemental/case-summary.json)。

另由独立 HTTP 验证保留了一次**默认网页采样**测试：temperature=0.7、top_p=0.8、top_k=20，26-token HTTPS 自然问题、256-token 输出、cache=0。HTTP TTFB **11.60 ms**、输出 TTFT **83.62 ms**，engine decode **77.5 token/s**、客户端 **77.22 token/s**。这是**单次**结果，不混入以上三次中位数：[原始请求与结果](../2026-10-14-a3b-full-gpu-final-cli/default-web-sampling.json)。

## 内存和显存

30 个请求结束后、服务仍驻留时采样，时间为本机 **18:51:22 +08:00**。下表是当时的占用，**不是峰值**。

| 项目 | 常驻快照 |
| --- | ---: |
| A3B 引擎物理 RAM 工作集 RSS | 1.782 GiB |
| GPU 0 总占用，含显示/background | 21,859 / 22,528 MiB |
| GPU 1 总占用 | 185 MiB |
| GPU 2 总占用 | 0 MiB |
| 全系统物理 RAM 已用 / 可用 | 16.77 / 111.10 GiB |

仅存在一个目标 `strata-moe.exe` 引擎，PID 35356。GPU 1/2 没有承担本模型的计算。GPU 0 仍保留模型、上下文、MTP 和工作缓冲；未调用 unload 或空闲释放。Windows 的 private commitment 包含 CUDA/WDDM 的提交，不能把它当作额外的物理 RAM，再与显存相加。[完整快照和进程命令](resident-resource-snapshot.json)。随后 HTTP 与 WSL 检查后的[独立快照](../2026-10-14-a3b-full-gpu-final-cli/resident-final.json)记录引擎 RSS 仍为 1.782 GiB、GPU 0 仍为 21,859 MiB，全系统 RAM 已用为 19.78 GiB；全系统数字属于不同时点，不是本模型额外增长了对应工作集。

## 正确性与可复现范围

本报告只使用通过独立逐 token 参考、GDN 回滚和接受长度检查后的引擎；严格数值测试及早期失败探索保留在 [Native MTP 验证报告](../2026-10-14-a3b-native-mtp-validation/README.md)。同一量化模型上的 max_abs/KL=0 是执行与状态校验，不表示 Q4 相对 BF16 没有量化损失。Qwen3.5 的结构已适配，本次真实权重与全部速度数字都来自明确命名的 Qwen3.6 模型。

原 Strata 的 1500+ prefill 数字有其硬件、量化、输入长度与专用内核条件，不能与本次短提示词混成同条件对比。相较历史的 Q5/RAM 专家路径，本轮同时改变了量化和执行布局，因此不把全部速度差写成单一优化的因果增益。

复现命令（既有服务启动且常驻后）：

```powershell
.venv\Scripts\python.exe tools/bench_a3b_latency.py --port 8096 --out bench/results/2026-10-14-a3b-full-gpu-latency/main --cache-policy resident --repeats 3 --prompt-tokens 32 512 2048 --max-tokens 256 512 --temperature 0 --label full-gpu-mtp-final
.venv\Scripts\python.exe tools/bench_a3b_supplemental.py --port 8096 --out bench/results/2026-10-14-a3b-full-gpu-latency/supplemental --repeats 3 --natural-max-tokens 512 --sample-max-tokens 256
```

[主测速工具](../../../tools/bench_a3b_latency.py)、[补充驱动](../../../tools/bench_a3b_supplemental.py)、[证据文件 SHA-256](manifest.sha256.json)。`G:\Strata` 原项目没有被本测试修改；新增代码、模型、数据和报告都在 `G:\Strata-Moe`。
