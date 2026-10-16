# A3B 单 GPU 实测与实现，2026-10-14

本机配置现为 GPU 0 全驻留、MTP 2、主模型 batch 1024、草稿 microbatch 512、context 4096。
没有 16 GiB 预算，服务 `idle_unload_s=0`，不在空闲时卸载。
原始 `G:\Strata` 未修改；所有代码、模型配方和测试位于 `G:\Strata-Moe`。

完整网页/API 的首字节、首输出 token、预填充和解码实测见
[30 个常驻请求的报告](../bench/results/2026-10-14-a3b-full-gpu-latency/README.md)。
它分别列出合成提示词、自然代码、自然中文和温度采样，不将某一题达到的速度当成所有题的速度。

## 原 Strata 为什么快

原版确实使用 CPU 计算一部分 RAM 专家，不只是保存专家：
`src/core/expert_source.cpp:3667` 定义 CPU / VRAM / PCIe 三类计划，
`3908` 调用 CPU pool，`src/kernels/cpu/pool.cpp:772–781` 计算 gate/up、激活和 down。
另一部分未命中专家通过独立 stream 上传 GPU，与 CPU 和 GPU 命中计算重叠。

全显存时，`src/core/verify.cpp:759–778` 有更直接的路径，绕过 CPU 通知、等待和 PCIe 专家调度；
`2172` 一次提交整个验证窗口的 CUDA 图。
原版还使用完整专家融合、精确多列 MMVQ、MTP、只补算接受前缀，以及大块 GPU 预填充。
prompt lookup 对重复上下文有额外收益，原版的测量限定在相应任务。

当前 A3B 权重能够放下，因此直接采用全 GPU 路径。
模型的注意力、GDN、路由、共享专家、路由专家、embedding gather 和输出投影在 GPU。
CPU 仍承担服务、分词、调度；非中性取样的 temperature / penalty 链保留 CPU 实现。
这里没有 CPU 专家矩阵计算。

## 权重与量化

- 基础：固定 revision 的 Unsloth Qwen3.6-35B-A3B UD-Q4_K_S，117 个路由专家张量 Q4_K、3 个 Q6_K，没有 3 bit 专家。
- 输出头：从已验证 Q5_K_M 文件逐字节复制其 Q8_0 `output.weight`，其余张量保留基础文件字节。
- MTP：对应 checkpoint 的独立 Q8_0 NextN 文件。
- 主文件 SHA-256：`9a91f467e1dc65f40487cc8f495b29a294a8afdbde7eb22dc4bf746ad350161e`。
- MTP SHA-256：`f6cb995fab78e7284097b2e13508e8daad4ffc6a747b433aef399f4e48524b5a`。

`tools/prepare_a3b_model.py` 保留 GGUF 元数据、检查 tokenizer 和几何、重建对齐偏移，记录逐张量/整文件哈希。
它没有解量化再量化，原下载文件保留。此配方没有 BF16 困惑度或任务评分对照，不能称为无损量化。

MTP 与目标的三份公共权重在类型、形状、布局及 SHA-256 都相同后共享 GPU 指针，复用 1030.633 MiB。
安全起见仍保留约 1030.633 MiB 的草稿 CPU 原始副本；这些副本不参与矩阵计算。

## 执行改变

- 原生 GPU 图代替逐层 RAM 专家往返；主模型与 MTP 各为一个 GPU graph split。
- 路由专家验证展开为单列算术；完整 gate/up/激活/down 连续构图，修正 CUDA 将路由数误作 token 列数的融合条件。
- 短窗口 Flash Attention 使用同一图内的单 query 节点，保留单 token 的舍入。K/V 无复制、无逐 query CPU 同步。
- GPU greedy 仅回传 token 与有限值检查，每行 8 字节；大词表两级 argmax 用 `STRATA_A3B_ARGMAX_MULTI=1` 显式启用。
- 草稿只补算接受的前缀；主模型和草稿使用独立 microbatch，减少重复工作区。
- 主干 head 的输出工作区按验证行数规划。
- A3B 的采样协议保留 `top_p=1`、`top_k=0`/宽 top-k、seed 0。原 Flash-Next 的序列化器保持原行为。

图和 CUDA 修改保存于 `a3b/patches/`，构建对固定 llama.cpp pin 幂等应用。
不支持的图变体保留原构图；本机实测仅单序列、F16 KV、CUDA SM75。

## 选择配置的测量

Windows，Threadripper 3960X / 128 GiB RAM，RTX 2080 Ti，驱动报告 22 GiB，CUDA 12.6。
全部测试使用设备 0。以下 CLI 运行保留 OS 文件缓存，每臂重新加载模型；加载与预填充不算入解码时间。

同一 23-token raw 中文提示词，最多 256 输出，实际在 223 个输出 token 后结束。
最终二进制 SHA-256：`0ef75ed69ee5fdbde27a0c8f5f2d49a426d7154d56962a7bfa20b426c3f9d2ec`。

| 同一 Q4/Q8Head，单次最终 CLI 对比 | 输出 tok/s | GPU 总量采样峰值 | 进程 RSS 采样峰值 |
| --- | ---: | ---: | ---: |
| 全 GPU，MTP 关闭 | 79.54 | 20667 MiB | 0.562 GiB |
| 全 GPU，MTP 2 | 83.21 | 21773 MiB | 1.702 GiB |

输出字节 SHA-256 相同；本题 MTP 提升约 4.6%，不是所有任务的提升保证。
[原始命令、输出及资源采样](../bench/results/2026-10-14-a3b-full-gpu-final-cli/results.json)。
Windows private commitment 包含驱动分配，不能与 RSS、显存相加当成物理 RAM。

同一 23-token 题的 MTP 1/2/3/4/6 探索结果为 79.94/82.72/75.73/49.64/39.66 tok/s。
全部输出字节相同。K4/6 为保持逐 token 算术而额外启用 dense 单列布局，权重重复读取更重；接受率下降也限制了收益。
因此当前配置选择 K2。
[全部探索记录](../bench/results/2026-10-14-a3b-native-q4-tuning/environment.json)。

固定 1597-token 合成提示词、128 输出的预填充探索：

| 主 batch / 草稿 microbatch | prefill tok/s | decode tok/s | GPU 采样峰值 |
| --- | ---: | ---: | ---: |
| 512 / 512 | 1079.91 | 86.08 | 21687 MiB |
| 1024 / 512，最终二进制 | 1295.48 | 85.03 | 21801 MiB |
| 1536 / 512，最终二进制 | 1213.86 | 85.29 | 21913 MiB |

每项一次新进程，prefill 含首次图准备。不同 prefill 分块会改变浮点舍入，输出哈希不同。
2048 / 2048 在草稿工作区分配时 OOM；2048 / 512 能完成上下文分配，但推理动态工作区仍 OOM。
这两个失败也保留在 `bench/results/2026-10-14-a3b-prefill-tuning/`，没有作为成功结果。
常驻 API 的长提示词测量单列在完整报告中。

## 正确性和限制

专家、dense 和注意力使用独立逐 token GPU 参考；真实模型 MTP K1/2/3/4/6 的所有接受前缀，logits 均 `max_abs=0, KL=0`。
K4/6 使用 dense exact；K1/2/3 无需它。真实 MTP class 的 1/2/3/4/5/8/16/32 输出边界与顺序 greedy token ID 相同。
真 GPU 草稿、独立 microbatch 2、显式 EOS 匹配统计也重新验证通过。
[数值、初始失败、修复和哈希记录](../bench/results/2026-10-14-a3b-native-mtp-validation/README.md)。

CUDA 六项 CTest 通过；CPU 构建三项通过，三项 GPU 专属检查跳过。采样协议测试通过。
HIP SDK / SYCL 编译器不可用，未验证这两个后端。
Qwen3.5 的相同结构可加载，但本轮端到端权重仅 Qwen3.6。
当前为文本、单序列、4096 context；未加入视觉、跨请求 KV 前缀缓存、prompt lookup 或多 GPU。

另一个 WSL vLLM 已按用户授权停止并释放缓存；8080 不再监听。当前 A3B 服务保持在 8096 驻留。
