# A3B 内存与解码优化，2026-10-14

原始 `G:\Strata` 未修改。下文记录早期 Q5 / 16 GiB 预算阶段；当前全 GPU Q4/Q8Head、MTP 2 的实现和实测见 [A3B-FULL-GPU.md](A3B-FULL-GPU.md)。
当前 `G:\Strata-Moe` 还在 Python 服务加入了仅对 `INFO engine=strata-moe-a3b` 生效的采样协议适配，原 Flash-Next 路径保持原行为。

## 124 GB 内存的来源

排查时 Windows 接近用满 128 GiB RAM，主要进程为：

| 进程 | 物理驻留工作集 |
| --- | ---: |
| WSL `vmmem` | 约 62.4 GiB |
| 旧 A3B `strata-moe.exe` | 约 46.8 GiB |

WSL 的 `t1` 中另有 vLLM，在 `127.0.0.1:8080` 运行 Qwen3.5-35B-A3B GPTQ-INT4，占用 GPU 1。
Linux `free` 报告约 60 GiB buff/cache、约 4.6 GiB used。这些缓存占在虚拟机工作集中，需要由 WSL 回收后才归还 Windows。
初次基准期间保留了它。之后经用户明确授权，已关闭这个 vLLM 服务并清空 WSL 文件缓存；8080 不再监听，GPU 1 降到 176 MiB。
清理后系统占用约 46.85 GiB、可用约 81.02 GiB，WSL 工作集降到约 8.17 GiB，Linux 文件缓存约 65 MiB。
A3B 的 Python launcher 与实际 Python interpreter 属于同一个 8096 服务，推理引擎只有一个。

A3B 自身同时保留了已经复制到 RAM 的专家和仍驻留的源 GGUF 映射页。
64 位 `VirtualQueryEx` / `QueryWorkingSetEx` 采样测得，源 GGUF 映射约 24.1 GiB 驻留。
同一新程序的控制对比为：

| 加载模式，16 GiB 显存预算 | 生成后进程 RSS |
| --- | ---: |
| `--mmap` | 46.800 GiB |
| 默认直接加载 | 23.178 GiB |
| 默认直接加载 + pageable 专家 | 23.178 GiB |

因此默认关闭源文件 mmap，减少约 23.62 GiB 物理驻留。
没有为了降低计数而丢弃专家，也没有把专家降成更低位数。
Windows private commitment 还包含 CUDA/WDDM 分配；它不是物理驻留 RAM，不能与 RSS 或显存简单相加。

证据：[原内存审计](../bench/results/2026-10-14-a3b-memory-audit/README.md)、
[同程序 mmap 对比](../bench/results/2026-10-14-a3b-memory-fixed/README.md)。

## 借鉴原 Strata 的执行方法

原 Strata 的 RAM 专家由 CPU 就地计算；GPU 同时计算缓存专家，并按 CPU/PCIe 带宽分配额外传输。
它按完整专家运行 gate/up/SwiGLU/down，以整验证窗口 CUDA graph 和 MTP 摊薄权重读取、路由发布和同步。
依据是 `src/core/expert_source.cpp`、`src/core/verify.cpp`、`src/kernels/cpu/pool.cpp` 和 `docs/DETAILS.md`。

本轮在 A3B 中落实了：

- 路由和 gather/scatter 索引在计算 stream 异步上传，临时 host 数组保留到工作完成。
- 先提交 GPU 命中，再下载 CPU 未命中输入，扩大两者的重叠时间。
- CPU 后端绑定常驻线程池；短验证窗口保留 CPU/GPU 分工，不触发长提示词的专家流式加载。
- 短窗口按权重复用 Task，允许 CUDA graph 在形状稳定后复用；长提示词仍共享有限 scratch。
- 路由快照只在同一次 backend 图调用内复用，跨 token 重新读取。
- 接入独立训练后的 Q8 MTP 草稿头，目标模型逐行真实取样并验证，接受后回滚 KV 和 Gated DeltaNet 状态。
- 量化专家的多 token 验证使用单列 MMVQ 算术，避免普通多列内核与逐 token 内核的数值差异。
- 主干、MTP、上下文分配后再规划专家缓存，保留 640 MiB 给工作缓冲和 CUDA graph。

仍使用 ggml 分图，尚未完整移植原 Strata 的整层 CUDA graph、完整 CPU 专家融合、RAM 补集存储或多 GPU。
CPU 冷专家仍按矩阵阶段执行，这是后续进一步减少同步与 PCIe 激活往返的方向。

## 重复基准

Windows，Threadripper 3960X，24 inference threads，128 GiB RAM。
一张 RTX 2080 Ti（驱动报告 22 GiB），CUDA 12.6，计算能力 7.5。
本进程预算 16,384 MiB；这不是物理 16 GiB 卡测试。另一个 vLLM 服务保持在 GPU 1。

同一 Qwen3.6-35B-A3B UD-Q5_K_M、同一 23 token 中文 prompt、256 greedy 输出 token、context 4096、batch 128。
每臂三个独立新进程；保留 OS 文件缓存，专家缓存从空开始。时间只含解码，加载与 prompt 单列记录。
各臂均启用 stable graphs、route snapshot、native-layout 和单列验证开关；未启用 MTP 时不产生多 token 验证。

| 路径 | 三次解码 tok/s | 中位数 | 进程峰值 RSS | GPU 峰值总量，含约 100 MiB 背景 |
| --- | --- | ---: | ---: | ---: |
| GPU 主干 + 专家缓存，MTP 关闭 | 26.59 / 26.57 / 26.65 | 26.59 | 23.202 GiB | 16,034 MiB |
| 同配置 + MTP 3 + 单列验证 | 31.89 / 32.24 / 31.92 | 31.92 | 23.841 GiB | 16,258 MiB |

本题中位解码速度提升 20.05%。MTP 每次接受 157/294 个草稿 token（53.40%），98 个验证窗口，平均约 2.61 个输出 token/窗口。
三次 MTP 均实际调用 11,760 次短窗口单列专家计算。各臂内部输出一致；跨臂因缓存容量和 CPU/GPU 路由舍入不同，全文并不完全一致。
首 token target logits 在各次测试中完全一致。以上不是 BF16 质量评测，也不是所有任务的速度保证。

较早探索版本测得 35.66 tok/s，但普通多列专家计算未通过严格逐 token 参考比较。
最终采用较慢但通过该校验的单列路径，不把旧探索速度当作交付值。

命令、完整输出、内存采样、模型和可执行文件 SHA-256：
[最终三轮基准](../bench/results/2026-10-14-a3b-optimized-final/README.md)。

## 数值与运行验证

专家 parity 保留 `2e-5 * (1 + abs(reference))` 容差，覆盖 F32/Q4_K/Q5_K/Q6_K/Q8_0、完整/部分缓存命中、短/长窗口、淘汰和 GPU 调度。
量化 CUDA 的 MMVQ/MMQ/单列布局分别使用独立 native GPU oracle，CPU 使用独立 native CPU oracle。

实际模型的 GDN 回滚测试覆盖接受 0/1/2/3 个草稿 token，再继续解码：
普通多列缓存路径曾测得最大约 0.0205 KL；单列修复后，完整 GPU 命中时，四种情况对独立顺序参考的 logits 均 `max_abs=0, KL=0`。
标准 sequence-state 保存/恢复参考也相等；容差未放宽。
测试范围、失败探索和完整记录见 [MTP 状态验证](../bench/results/2026-10-14-a3b-mtp-validation/README.md)。

另用 2 GiB 缓存，预热后冻结专家 placement，强制混合路径：65,454 行由 GPU、32,466 行由 CPU 计算，验证期间没有上传。
四种接受长度的独立顺序参考与状态恢复参考同样 `max_abs=0, KL=0`。
冻结只用于隔离状态验证；正常服务仍自适应管理专家缓存。

最终补齐服务的 MTP 接受统计、默认单列验证和测试用 admission 开关后，重新构建并复核：
256 token 解码为关闭 MTP 26.50、开启 MTP 31.84 tok/s，输出分别与三轮基准对应臂完全相同。
最终交付二进制、源码哈希及结果见 [交付版本复核](../bench/results/2026-10-14-a3b-delivery-check/README.md)。

本机 `strata-a3b.json` 已启用 MTP 3 与验证过的调优开关，8096 服务已重新启动。
真实网页、OpenAI 非流式/SSE、Anthropic 文本、多请求、1/2/3/4/5/8 token 输出边界、跨多个 prefill block 的长提示词和带 penalty 的随机采样检查完成。
API 的 `draft_n` / `draft_n_accepted` 返回实际 MTP 统计。完整响应在 `api-extended.json`。
相同 seed 的随机采样回复观察到差异；取样器每请求重建，草稿不消耗其 RNG，但自适应 CPU/GPU 分工下不保证逐字复现。
这项观察及首次不适用的重复输出断言记录在 `api-check-notes.md`；没有因此放宽数值或状态测试阈值。

关闭 vLLM 后，又对正在运行的网页/API 做了两次 256 token 检查：temperature=0 为 25.7 tok/s、9.971 秒解码；temperature=0.7 为 29.6 tok/s、8.635 秒解码。
这是已有缓存历史的常驻服务、22 token 的 API 模板提示词，不是上述 23 token 原始模板、新进程三轮基准，不能把两组数字当作相同实验。
完整响应、时序和清理后内存记录见 [关闭 vLLM 后的 API 实测](../bench/results/2026-10-14-a3b-delivery-check/post-vllm-stop.json)。

CUDA 和纯 CPU Release 构建通过。HIP SDK 与 SYCL 编译器在本机不可用，未宣称这两个后端已通过。
Qwen3.5 的结构检查已适配，实际端到端测试仍仅使用命名的 Qwen3.6 权重。

MTP 默认不启用；传 `--mtp-model` 才启用，并默认使用专家单列验证。
本机服务配置会显式记录 MTP 路径、草稿长度与调优开关。所有模型下载固定 revision 与 SHA-256。
