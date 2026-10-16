# 完整 Q8 新题检查：独立服务进程，六个请求

BFS 代码与中文事务隔离题各三次，中位输出速度为 **34.7 / 36.9 token/s**，首输出等待 **1.374 / 0.922 秒**。这两段 prompt 未出现在本次缓存 profile 的五个训练 prompt 中。两题只是有限的新工作负载检查，不是广泛泛化或模型质量评测，100 token/s 仍未达到。

本组与此前 [30 请求候选矩阵](../README.md)来自不同服务进程，结果不池化。GPU 0 / RTX 2080 Ti / SM75 / 驱动 617.42 / PCIe 3.0 ×16，Threadripper 3960X，约 128 GiB RAM，与前组硬件一致。目标仍为官方完整 Q8_0，SHA-256 `d8d7842cc657d720f39546878e431937c84473aa130c36371669ac17c80c7361`。

[artifact-manifest.json](artifact-manifest.json)记录本组六请求的 20 个数据、配置及复现文件的大小和 SHA-256，不包含可继续维护的本页文字。

## 版本和过程

使用冻结的 `build-a3b-q8-compatible/bin/strata-moe.exe`，SHA-256 `fb650108979dca5f6b5d9202409cf63ac73c2d1ea5498826dd213e4ada337456`。保存的 [config.json](config.json) SHA-256 为 `3474d3a417706748b8785536dce1c2c2e6cd28ea1c0058f2c4e4fc992f5c8609`，与 [plan](plan.json) 和 [resource snapshot](resources-final.json) 一致。

数学设置与 30 请求候选相同：context4096、主 batch512、draft microbatch512、adaptive MTP 上限 K6、8 staging slots、GPU-only experts、pageable arena、strict MTP share、deferred catch-up、exact dense/attention、async events、decay4、reserve384 MiB。profile 使用稳定路径 `models/a3b/profiles/qwen3.6-q8-broad-v1.profile`；ARGMAX_MULTI=1、DENSE_Q8_REUSE=1、EXPERT_Q8_REUSE=0。本组没有启用之后新增的 GPU-static 实验架构，不能把本表作为该架构的速度。

`heldout-run.py` 使用与前组相同的 socket/SSE 计时代码。BFS 首轮是新服务的第一个测量请求，模型已加载且 profile 已预载，启动 hash/加载时间不计入 TTFT。其余请求在同一进程连续运行；中文题首轮是新 case，没有重启或清缓存。每次 temperature=0、seed=12345、thinking 关闭、max_tokens512。

## 三次整体中位数

| 新题 | 输入 token | 三次实际输出 token | TTFB ms | TTFT s | 引擎 decode token/s | 客户端 decode token/s | Prefill token/s |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Python BFS 最短路径 | 160 | 512 / 512 / 512 | 11.08 | 1.374 | 34.7 | 34.63 | 117.5 |
| 中文事务隔离 | 123 | 329 / 318 / 329 | 11.03 | 0.922 | 36.9 | 37.35 | 135.4 |

全部六请求收到 `[DONE]`，实际输入数与 count endpoint 相同，cached token=0，没有 reasoning 输出。BFS 全部为 length，中文全部为 stop；BFS 的解释可能受 512 token 预算截断。没有在本轮执行生成代码或给内容质量评分。

| 新题 | 首轮 TTFT s | 后两轮 TTFT 中位数 s | 首轮 decode token/s | 后两轮 decode 中位数 token/s |
| --- | ---: | ---: | ---: | ---: |
| Python BFS | 1.467 | 1.373 | 34.3 | 34.85 |
| 中文事务隔离 | 1.650 | 0.909 | 31.0 | 37.85 |

所有三次均纳入整体中位数，后两轮只包含两次。BFS 三次输出文本 SHA 相同，中文三次不同；这不证明跨任务、缓存历史或硬件的确定性。

TTFB 包括 HTTP headers；TTFT 为首个非空 content/reasoning delta。客户端 decode 使用 `(completion_tokens-1)/(最后非空输出时间-首个非空输出时间)`，MTP 可成批输出，stop token 也可能计入 usage；因此客户端与引擎值分别保留，客户端值偶尔可高于引擎值。定义详见 [测量计划](plan.json) 与 [原始结果](results.json)。

## MTP、真实传输和资源

[engine.log](engine.log) 是复制的历史追加日志，含两次启动。只取 **2026-10-14 22:13:12 的最后一次启动**，其内恰好六个请求，按输入/输出计数与 API 逐条匹配。最早的 K2 baseline 进程被明确排除。[engine-counters.json](engine-counters.json)及 [只读提取脚本](extract-engine-counters.py)保留匹配、逐 K 时间、cache decay 和实际 hit/miss。该进程的全部专家 CPU rows 为 0；路由规划仍由这版既有后端管理。

| 新题 | 合计 accepted / offered | 合并接受率 | Decode hit 中位数 | Prefill 上传 MiB 中位数 | Decode 上传 MiB 中位数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python BFS | 1149 / 1372 | 83.75% | 92.40% | 6,776.6 | 39,875.6 |
| 中文事务隔离 | 572 / 1078 | 53.06% | 96.52% | 4,539.0 | 14,289.6 |

上传包含 persistent admission 与 transient staging，草稿尝试不计作真实输出。三次合计的实际 offered-K 窗口数为：

| 新题 | K0 | K1 | K2 | K3 | K4 | K5 | K6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Python BFS | 10 | 34 | 10 | 114 | 140 | 40 | 36 |
| 中文事务隔离 | 6 | 13 | 235 | 24 | 107 | 13 | 5 |

六请求后的单次 [资源快照](resources-final.json)：引擎 RSS **33.984 GiB**，OS lifetime working-set peak **33.984 GiB**；GPU 0 当前显存 **21,867 / 22,528 MiB**，GPU 1 为 172 MiB 显示使用、GPU 2 为 0。全系统物理 RAM 已用 **49.608 GiB**、可用 **78.262 GiB**。显存是当前采样，不是连续监测峰值；private commitment 不与 RSS 或显存相加。

未进行 BF16 质量对照、长上下文、并发或其他硬件验证。本组保留原始 SSE 和实际输出；不能从两题正常完成推断所有任务质量，也不能将训练 profile 的覆盖率替换成这里实测的命中率。
