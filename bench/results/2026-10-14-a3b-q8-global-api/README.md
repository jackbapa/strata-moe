# Qwen3.6-35B-A3B 完整 Q8：候选服务延迟与吞吐

本版 **尚未达到 100 输出 token/s**。30 个 API 请求已经完成：自然代码题和中文解释题各三次的整体中位数为 34.6 / 35.0 token/s；首输出 token 等待为 1.245 / 0.803 秒。Greedy 矩阵六组中位数为 27.0–38.0 token/s。HTTP 首响应字节通常约 10 毫秒，它包括 headers，不代表模型已经开始输出。

前 30 个主矩阵和 supplemental 请求使用同一常驻进程，不 reload/unload，不排除首轮；后续六个新题请求来自另一个服务进程，单独报告。以下表格均为三次整体中位数；首轮和剩余两次另外列出。原始 SSE、请求/输出与每次计时见 [supplemental/results.json](supplemental/results.json)、[case-summary.json](supplemental/case-summary.json) 和同目录的 `.events.jsonl` / `.output.json`。

[artifact-manifest.json](artifact-manifest.json)记录完整 30 请求 cohort 的文件大小与 SHA-256；后续 held-out 数据不计入这份 manifest。工具的最终离线检查为 options 12 项、profile 14 项、download 9 项及采样协议检查通过，Python 编译检查通过；这些不占用 GPU。

## 硬件、模型与候选配置

- Windows，AMD Threadripper 3960X，约 128 GiB 系统 RAM；GPU 0 是 RTX 2080 Ti / SM75，驱动 617.42 报告显存 22,528 MiB，PCIe 3.0 ×16。同机硬件记录见 [初始 Q8 environment](../2026-10-14-a3b-q8-api-baseline/environment.json)。引擎只看到 `CUDA_VISIBLE_DEVICES=0`。
- 完整官方 Q8_0 GGUF 为 36,903,139,328 bytes，SHA-256 `d8d7842cc657d720f39546878e431937c84473aa130c36371669ac17c80c7361`；120 个 routed tensors 全部 Q8_0。专家在 RAM 保存完整副本，热点及临时专家在 GPU 计算，CPU expert rows 为 0。Q8 无法完整装入这张卡。
- 引擎 SHA-256 `fb650108979dca5f6b5d9202409cf63ac73c2d1ea5498826dd213e4ada337456`。测试 [config.json](config.json) 的 SHA-256 为 `fd166c7c865270345ef805499f661548a43aef102183b820d12f20f07353de47`。
- context=4096，主 batch=512，draft microbatch=512；adaptive MTP 上限 K=6；GPU-only experts，8 staging slots / bank；pageable expert arena，严格共享 Q8 MTP 表，延迟 catch-up，exact dense/attention，async expert events。
- broad-v1 全局训练 profile，cache decay interval=4，reserve=384 MiB。环境变量为 `STRATA_A3B_ARGMAX_MULTI=1`、`STRATA_A3B_DENSE_Q8_REUSE=1`、`STRATA_A3B_EXPERT_Q8_REUSE=0`。实验 expert reuse kernel 保持关闭。
- 这份 JSON 保存 30 请求的候选配置快照。随后生产 `strata-a3b-q8.json` 已采用相同数学参数，将 profile 指向 `models/a3b/profiles/qwen3.6-q8-broad-v1.profile` 稳定路径。冻结的 Q4 配置/二进制另行保留，不用 Q4 结果替代 Q8 结果；不同服务进程的后续结果单列。

启动日志记录完整模型 SHA 校验 36,078.9 ms、profile 预载 warmup 4,094.7 ms；随后清除临时用户 KV，才报告 READY。这些是启动开销，表中 resident TTFT 不包含模型加载或启动 hash。

## 自然任务与 temperature 0.7

自然题 temperature=0；代码题要求实现带断言的 OrderedDict LRU，中文题解释 HTTPS 加载过程。随机采样使用 temperature=0.7、top_p=0.95、top_k=20、seed=12345；它与网页默认 top_p=0.8 不同。全部关闭 thinking，每次真实 prompt token 数与 count endpoint 一致，cached token 为 0，均收到 SSE `[DONE]`。

**自然代码和中文 HTTPS 两题与 broad profile 的训练 prompt 重合，是训练任务复跑；这些提升不能作为独立泛化证据。** 新的 BFS/事务隔离两题另见下面的独立进程检查，不与本节 30 请求合并。

| 任务 | 输入 token | 三次实际输出 token | TTFB ms | TTFT s | 引擎 decode token/s | 客户端 decode token/s | Prefill token/s |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 自然代码 | 124 | 512 / 512 / 512 | 10.22 | 1.245 | 34.6 | 34.57 | 99.9 |
| 中文 HTTPS 解释 | 98 | 281 / 267 / 277 | 11.16 | 0.803 | 35.0 | 34.85 | 124.1 |
| 采样，短输入 | 32 | 256 / 256 / 256 | 10.53 | 0.418 | 33.8 | 33.66 | 79.0 |
| 采样，长输入 | 2048 | 256 / 256 / 256 | 1.79 | 5.185 | 24.9 | 24.81 | 395.4 |

代码题三次达到 length=512，解释可能因预算而截断；中文三次均为 stop。自然任务是实际工作负载检查，没有 BF16 质量分数。三次请求发生在同一进程的连续缓存历史中，“首轮”表示该 case 第一次测量，不是冷磁盘或新进程。

| 任务 | 首轮 TTFT s | 后两轮 TTFT 中位数 s | 首轮 decode token/s | 后两轮 decode 中位数 token/s |
| --- | ---: | ---: | ---: | ---: |
| 自然代码 | 1.363 | 1.242 | 34.6 | 33.9 |
| 中文 HTTPS 解释 | 1.459 | 0.798 | 31.5 | 36.8 |
| 采样，32 输入 | 0.584 | 0.401 | 33.6 | 34.5 |
| 采样，2048 输入 | 5.529 | 5.145 | 24.5 | 25.2 |

MTP 接受率以下按三次合并的实际 accepted/offered 计算；草稿尝试不计为输出 token。自适应策略允许 K=0，仍由目标模型产生输出。

| 任务 | 三次合计 accepted / offered | 合并接受率 |
| --- | ---: | ---: |
| 自然代码 | 1113 / 1240 | 89.76% |
| 中文 HTTPS 解释 | 401 / 823 | 48.72% |
| 采样，32 输入 | 318 / 558 | 56.99% |
| 采样，2048 输入 | 376 / 698 | 53.87% |

## Greedy 主矩阵

32 / 512 / 2048 输入 × 256 / 512 输出 × 三次，共 18 个请求；temperature=0、thinking 关闭、seed=12345。全部请求收到 SSE `[DONE]`，输入计数一致、cached token=0，实际输出均等于各自预算且以 length 结束。原始 [results](matrix/results.json) 与 [case-summary](matrix/case-summary.json) 保留每次完整数值。

| 输入 / 输出 token | TTFB ms | TTFT s | 引擎 decode token/s | 客户端 decode token/s | Prefill token/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 / 256 | 11.02 | 0.418 | 38.0 | 37.89 | 79.0 |
| 32 / 512 | 10.79 | 0.491 | 36.6 | 36.56 | 66.9 |
| 512 / 256 | 11.48 | 1.843 | 34.2 | 34.06 | 278.7 |
| 512 / 512 | 10.86 | 1.874 | 35.3 | 35.24 | 275.2 |
| 2048 / 256 | 11.24 | 5.195 | 27.0 | 26.87 | 395.4 |
| 2048 / 512 | 11.75 | 5.256 | 30.9 | 30.88 | 390.0 |

| 输入 / 输出 token | 首轮 TTFT s | 后两轮 TTFT 中位数 s | 首轮 decode token/s | 后两轮 decode 中位数 token/s |
| --- | ---: | ---: | ---: | ---: |
| 32 / 256 | 0.942 | 0.415 | 38.0 | 39.0 |
| 32 / 512 | 0.365 | 0.501 | 37.5 | 35.95 |
| 512 / 256 | 1.989 | 1.838 | 35.8 | 33.25 |
| 512 / 512 | 1.874 | 1.886 | 35.4 | 35.15 |
| 2048 / 256 | 5.362 | 5.187 | 26.2 | 27.1 |
| 2048 / 512 | 5.183 | 5.310 | 28.6 | 31.1 |

本次六组 greedy case 的三次输出文本均有三种不同 SHA；自然代码为两种，中文为三种。相同 seed/temperature=0 不代表本版跨缓存历史的回复逐字相同。MTP 回滚和 accepted-prefix 数值测试的范围应查独立验证记录，不能从 API 正常结束反推全部长 prefill 分组数值相同。

## 新题检查：单独服务进程

[heldout-run.py](heldout-run.py)的六个新题请求已完成：BFS 最短路径代码与中文事务隔离 prompt 没有出现在本次五个缓存训练 prompt 中。它们使用单独启动的冻结 Q8 兼容二进制，数学配置与前 30 请求相同；新 GPU-static 实验没有启用。完整 [独立报告](heldout/README.md)保存原始数据、计数和资源，结果不与前 30 请求池化。

| 新题（三次中位数） | TTFB ms | TTFT s | Decode token/s | Prefill token/s |
| --- | ---: | ---: | ---: | ---: |
| Python BFS，160 输入 / 512 输出 | 11.08 | 1.374 | 34.7 | 117.5 |
| 中文事务隔离，123 输入 / 318–329 输出 | 11.03 | 0.922 | 36.9 | 135.4 |

首 BFS 请求是该新服务的第一次测量，中文首轮仅为新 case，未重新加载模型。模型加载/profile预载已在请求前完成。两题不足以构成广泛质量或泛化评估，100 token/s 仍未达到。

## 资源快照与专家缓存

[resources-live.json](resources-live.json) 是 supplemental 测试期间的一次采样：

| 指标 | 实测 | 口径 |
| --- | ---: | --- |
| GPU 0 显存 | 21,983 MiB / 22,528 MiB | 单次 live 采样，非监测峰值 |
| 引擎物理 RSS | 34.029 GiB | 当前 working set |
| 引擎 OS lifetime working-set peak | 34.037 GiB | 截至采样时的系统计数 |
| 全系统物理内存已用 | 49.547 GiB | 包括引擎与其他系统进程 |
| 全系统物理内存可用 | 78.324 GiB | 当时可用值 |

private commitment / virtual memory 不是物理 RAM，不能再与 RSS 或显存相加。GPU 1 的约 172 MiB 是同机其他使用，不是这个单卡引擎的模型分片。上述值是中途采样，不能把它当作最终峰值。

完成全部 30 请求后的 [resources-final.json](resources-final.json)另记引擎物理 RSS **34.086 GiB**、OS lifetime working-set peak **34.096 GiB**、GPU 0 当时显存 **22,029 MiB / 22,528 MiB**；全系统物理内存已用约 **49.584 GiB**，可用约 **78.287 GiB**。GPU 1 为显示使用的 172 MiB，GPU 2 为 0；该显存值仍是单次采样，不是连续监测峰值。根代理同时核对 `wsl --list --running`，没有运行中的 WSL 分发。

[broad-v1 provenance](../2026-10-14-a3b-q8-training/broad-v1.profile.provenance.json)记录五个真实训练任务汇总的一个进程累计 TSV。其 preview 分配为 5,271 专家、训练覆盖率 84.1210%；本次实际剩余预算选中 5,273 专家、16,807.7 MiB，启动日志的训练覆盖率约 84.13%。profile SHA-256 为 `10b4803644cecca0e81cca038f5425cf8b1281c8557c0cfca7428ee9d8ba7e8b`。这些覆盖率是训练计数的静态 placement 覆盖，不是运行时命中率，也不是泛化或速度结果。

测试允许 adaptive LFU 替换缓存成员。频率以 Q8.8 路由权重记录，每四个 primary-cache 窗口作一次 0.7 衰减；日志中的 decay sweep 不是专家数或 token 数。实际 decode hit/miss、上传字节及逐 K 窗口耗时需要与各请求顺序对应。路由导出是 lifetime cumulative counts，含启动 dummy token；不能把本进程的连续快照当作独立 trace 再叠加。

已将日志前十二个完成请求按顺序、实际输入/输出计数与 API 逐条核对，结果在 [supplemental/engine-counters.json](supplemental/engine-counters.json)。全部 decode CPU expert rows 为 0。下表同样为每个 case 三次的中位数；上传量包含 persistent admission 和临时 staging 的真实原始权重传输。

| 任务 | Decode 实际 cache hit | Prefill 上传 MiB | Decode 上传 MiB | Decode decay sweeps |
| --- | ---: | ---: | ---: | ---: |
| 自然代码 | 91.55% | 5,979.8 | 41,749.9 | 1,400 |
| 中文 HTTPS 解释 | 96.98% | 3,643.3 | 11,628.0 | 1,480 |
| 采样，32 输入 | 96.79% | 1,992.2 | 10,152.2 | 1,480 |
| 采样，2048 输入 | 91.93% | 22,051.1 | 24,652.1 | 1,240 |

完整 [engine-counters.json](engine-counters.json)把前 30 个顺序请求全部与 API 核对，保留逐次 transfer、decay、各 K 时间、MTP 接受和文本 SHA；可用 [extract_engine_counters.py](extract_engine_counters.py)只读复现。主矩阵同样全部 CPU expert rows 为 0。

| 输入 / 输出 token | 合计 MTP accepted / offered | 合并接受率 | Decode hit 中位数 | Prefill 上传 MiB 中位数 | Decode 上传 MiB 中位数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 / 256 | 346 / 554 | 62.45% | 97.10% | 1,823.2 | 9,189.6 |
| 32 / 512 | 704 / 1531 | 45.98% | 97.70% | 2,301.4 | 16,998.9 |
| 512 / 256 | 332 / 529 | 62.76% | 95.53% | 8,223.8 | 12,663.9 |
| 512 / 512 | 662 / 1091 | 60.68% | 96.20% | 8,395.9 | 23,453.6 |
| 2048 / 256 | 388 / 657 | 59.06% | 91.54% | 21,901.3 | 23,007.4 |
| 2048 / 512 | 825 / 1701 | 48.50% | 95.07% | 22,331.6 | 33,934.1 |

这些数据说明，即使命中率超过 90%，一次生成仍可能上传数十 GiB 专家权重，RAM/GPU 调度的传输成本不能忽略。它是本次性能分析的证据，不能单独证明每一毫秒都消耗在 PCIe 上。

三次合计的自适应窗口数如下。实际 offered K 记录包含接近输出上限时截短的窗口；K=0 的窗口仍运行目标模型。各 K 的累计 round 时间保存在 JSON，不能把窗口数等同于输出 token 数。

| 任务 | K0 | K1 | K2 | K3 | K4 | K5 | K6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 自然代码 | 10 | 20 | 17 | 325 | 35 | 7 | 6 |
| 中文 HTTPS 解释 | 10 | 129 | 194 | 68 | 9 | 6 | 6 |
| 采样，32 输入 | 20 | 360 | 41 | 7 | 6 | 7 | 6 |
| 采样，2048 输入 | 12 | 131 | 208 | 19 | 6 | 8 | 5 |

## 口径与限制

TTFB 来自底层 socket 首响应字节，包含 headers；TTFT 是首个非空 content/reasoning delta，排除角色、进度和心跳。引擎 decode 使用 API 输出计数与引擎 decode 时间；客户端 decode 为 `(completion_tokens - 1) / (最后非空输出 delta 时间 - 首个非空输出 delta 时间)`。MTP 与 detokenizer 可成批输出，usage 也可能计入 stop token，因此客户端值不是逐个模型 token 的独立计时。两种口径分别保留。Prefill 使用引擎处理的实际输入 token/s，没有 KV 复用。

早期 Q8 baseline 的同题自然代码/中文中位数为 22.9 / 24.9 token/s，本候选为 34.6 / 35.0 token/s，分别约提高 51% / 41%。32 输入 / 256 输出的 greedy 中位数则由 39.2 降至 38.0 token/s，约降低 3%；不能概括为所有任务都加速。对照见 [初版 Q8 API](../2026-10-14-a3b-q8-api-baseline/README.md)。两版同时改变了全局缓存、衰减、MTP K 和事件调度，不能把全部差异归因于一个开关；早期版本没有完整长输入矩阵可作同口径对照。Q4 是不同权重精度与驻留方式，其 [完整测试](../2026-10-14-a3b-full-gpu-latency/README.md)独立保留。

未完成 BF16 质量评估、长上下文质量评估、多人并发、Linux/HIP/SYCL 实际推理或其他 GPU 的性能验证；自然题不是通用质量基准。当前优化改善了本机 Q8 的部分任务速度，仍未达到 100 token/s；同样没有把原 Flash-Next 不同硬件/量化的 1500 prefill token/s 作为本模型已达成指标。所有修改和测试位于 `G:\Strata-Moe`，原始 `G:\Strata` 未改动。
