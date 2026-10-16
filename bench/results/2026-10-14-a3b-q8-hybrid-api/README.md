# 完整 Q8 普通混合模式：12 个 API 请求

这组实际使用 CPU/GPU 混合专家计算，**尚未达到 100 输出 token/s 或 1500 prefill token/s**。BFS 新题的三次中位 decode 为 **46.6 token/s**，但 TTFT 为 **1.561 秒**，比先前 GPU-only 新题进程的 1.374 秒更慢。2048 输入的 prefill 为 **244.4 token/s**、TTFT **8.393 秒**，也比先前 GPU-only 的 395.4 token/s、5.195 秒退步。不能只报告 decode 的提升。

这十二请求全部来自 PID 35576 的同一常驻进程，按 BFS、中文事务、短输入、长输入顺序各三次；没有 reload/unload，也没有排除首轮。模型加载、完整目标 SHA 验证 37,036.8 ms 和 profile warmup 4,005.8 ms 已在首次请求前完成，不计入 resident TTFT。本组使用普通混合调度，**没有启用新的 whole-window doorbell、GPU-static 或 GPU-dynamic 实验**，结果也不与那些实验或旧 GPU-only cohort 池化。

[离线审计结果](audit-results.json)独立核对了原始 SSE、输出文件、API 与日志计数、全部中位数及资源样本。[artifact-manifest.json](artifact-manifest.json)记录 41 个证据/复现文件的大小和 SHA-256，排除可继续编辑的本页、manifest 自身及可追加的顶层 engine.log；固定的 [measurements/engine.log](measurements/engine.log)已经纳入。可运行 `python bench/results/2026-10-14-a3b-q8-hybrid-api/audit-results.py` 只读复核已保存测量数据；它不发 HTTP、不访问 GPU，不启动或修改引擎。

## 硬件、权重和冻结配置

Windows，Threadripper 3960X（24 个物理核），约 128 GiB RAM；GPU 0 为 RTX 2080 Ti / SM75、驱动 617.42、PCIe 3.0 ×16，报告 22,528 MiB 显存。配置限定 `CUDA_VISIBLE_DEVICES=0`。

| 文件 | SHA-256 | 验证依据 |
| --- | --- | --- |
| 官方完整 Q8_0 target，36,903,139,328 bytes | `d8d7842cc657d720f39546878e431937c84473aa130c36371669ac17c80c7361` | 执行进程在启动时计算完整模型 SHA；离线审计核对日志与文件大小 |
| 官方 Q8 MTP，1,990,649,632 bytes | `f6cb995fab78e7284097b2e13508e8daad4ffc6a747b433aef399f4e48524b5a` | 离线审计重新计算完整文件 SHA |
| `build-a3b-hybrid-compatible/bin/strata-moe.exe` | `a506677ca04b1c6f643f6c127cba61cd45c84f5d5ad6ddc7a324b8e17fe0c81a` | runner 当时记录，并由离线审计重新核对 |
| [冻结配置](measurements/config.json) | `06998b4b3f56bcdd2f75bb4a3cf54fa926a94a0be56fba14469843aab92c8f23` | 测量快照与顶层实验配置相同 |
| broad-v1 profile | `10b4803644cecca0e81cca038f5425cf8b1281c8557c0cfca7428ee9d8ba7e8b` | 离线审计重新核对 |

模型来源为 `ggml-org/Qwen3.6-35B-A3B-GGUF` 的固定 revision `baec3ebee244827cda0f4557eafa8b28f7545fa6`。专家保存完整原始 Q8 权重，没有路由剪枝或额外权重量化。配置为 context4096、batch512、draft microbatch512、adaptive MTP 上限 K6、strict shared tensors、deferred catch-up、exact dense/attention、async expert events、pageable expert arena、8 staging slots、decay4、reserve384 MiB。实际 profile 缓存为 5,273 个完整专家、16,807.7 MiB。

实验新增 `--hybrid-experts --cpu-expert-pool --threads 8 --pcie-frac 0.25`。CPU 池有八个参与者（含调用线程），共享每个输入的量化结果；GPU 计算缓存命中以及选中的 PCIe miss，CPU 计算其余 miss。分流按独立 missed expert 的首次路由顺序，把最后 `floor(unique_misses × 0.25)` 个整组交给 GPU；同一专家的全部行同归属。小组取整后，实际 GPU miss 行占比并非恒为 25%。环境为 ARGMAX_MULTI=1、DENSE_Q8_REUSE=1、EXPERT_Q8_REUSE=0。生产/Q4 配置没有因这份实验改动。

## 实际请求延迟与吞吐

全部请求为 temperature0、seed12345、thinking 关闭。BFS/中文事务题与先前 GPU-only 新题报告使用相同 prompt，均未出现在五个 profile 训练题中；两题仍不足以构成质量或泛化评测。以下为包含全部三轮的中位数，逐次数据见 [results.json](measurements/results.json) 和 [case-summary.json](measurements/case-summary.json)。

| 任务 | 输入 token | 三次实际输出 token | TTFB ms | TTFT s | 引擎 decode token/s | 客户端 decode token/s | Prefill token/s |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| BFS 最短路径 | 160 | 512 / 512 / 512 | 11.12 | 1.561 | 46.6 | 46.54 | 103.4 |
| 中文事务隔离 | 123 | 324 / 341 / 311 | 11.28 | 1.027 | 37.8 | 38.09 | 121.3 |
| 短输入 | 32 | 256 / 256 / 256 | 10.90 | 0.500 | 39.1 | 38.92 | 65.8 |
| 长输入 | 2048 | 256 / 256 / 256 | 11.38 | 8.393 | 33.5 | 33.34 | 244.4 |

TTFB 是首 HTTP 响应字节，包括 headers；约 11 ms 不表示模型已经输出。TTFT 为首个非空 content/reasoning delta，本组没有 reasoning。客户端 decode 为 `(completion_tokens−1)/(末个非空 delta 时间−首个非空 delta 时间)`；MTP 可以突发输出、usage 可包含停止 token，不能把 SSE frame 数当作 token 数，客户端与引擎数值分别保留。

| 任务 | 首轮 TTFT s | 后两轮 TTFT 中位数 s | 首轮 decode token/s | 后两轮 decode 中位数 token/s |
| --- | ---: | ---: | ---: | ---: |
| BFS | 1.625 | 1.543 | 45.4 | 46.6 |
| 中文事务 | 1.681 | 1.004 | 39.2 | 36.1 |
| 短输入 | 0.800 | 0.493 | 40.2 | 37.6 |
| 长输入 | 8.435 | 8.322 | 33.2 | 33.7 |

离线审计确认十二份 SSE 各有且只有一个 `[DONE]`，没有 error frame；count endpoint、API usage、timings 和引擎 prompt/output 计数逐条相符，`cached_tokens=cache_n=0`。BFS、短/长输入均为 length，中文为 stop。BFS 的代码输出在 512 token 预算内截断，没有在本轮执行代码或评分。三轮文本 SHA 的不同数量依次为 2 / 3 / 3 / 3；相同 greedy seed 不保证跨缓存历史逐字一致。正常完成也不替代独立 MTP 回滚与数值验证。

## 与冻结 GPU-only 结果的描述性比较

对照来自 [GPU-only 新题六请求](../2026-10-14-a3b-q8-global-api/heldout/README.md)及 [GPU-only greedy 矩阵](../2026-10-14-a3b-q8-global-api/README.md)。它们是另一冻结二进制、服务进程和缓存历史；输出/自适应 K 也可能不同，故这不是隔离某一个优化的因果 A/B。

| 任务 | GPU-only → hybrid decode token/s | 观察到的变化 | GPU-only → hybrid TTFT s | GPU-only → hybrid prefill token/s |
| --- | ---: | ---: | ---: | ---: |
| BFS | 34.7 → 46.6 | +34.3% | 1.374 → 1.561 | 117.5 → 103.4 |
| 中文事务 | 36.9 → 37.8 | +2.4% | 0.922 → 1.027 | 135.4 → 121.3 |
| 短输入 | 38.0 → 39.1 | +2.9% | 0.418 → 0.500 | 79.0 → 65.8 |
| 长输入 | 27.0 → 33.5 | +24.1% | 5.195 → 8.393 | 395.4 → 244.4 |

四组 TTFT 均退步；长输入首输出等待增加约 61.6%，prefill 吞吐下降约 38.2%。普通混合模式仍保留每层 host 路由规划/交接，日志显示 81 个 scheduler splits。新的 doorbell 是独立且尚未完成性能测量的后续方案，不能把预期效果填入本表。

## CPU/GPU 实际工作量和传输

[engine-counters.json](measurements/engine-counters.json)按十二请求顺序匹配；每次 `CPU pool jobs × 3 = decode CPU projection rows`，CPU 与 GPU 行数均为正。下表行数和上传量为三轮中位数；CPU 比例按三轮行数合并计算，只涵盖 routed experts 的三个投影，不代表整个模型或 CPU 利用率。

| 任务 | Decode GPU / CPU 投影行 | CPU 行占比 | Prefill 权重上传 MiB | Decode 权重上传 MiB | GPU miss 完整行 / 独立专家（整请求） |
| --- | ---: | ---: | ---: | ---: | ---: |
| BFS | 536,328 / 20,472 | 3.70% | 6,687.4 | 18,411.0 | 536 / 536 |
| 中文事务 | 436,239 / 9,078 | 1.91% | 4,102.3 | 7,073.1 | 62 / 62 |
| 短输入 | 412,002 / 5,616 | 1.46% | 1,781.8 | 4,143.8 | 40 / 40 |
| 长输入 | 375,726 / 17,367 | 4.75% | 31,416.0 | 9,275.6 | 588 / 555 |

上传计数包含 persistent admission 和 transient staging 的实际原始权重传输，没有把 CPU RAM 读取计成 PCIe 权重上传。GPU miss 行/专家和 hybrid calls 是整请求 delta；CPU 池阶段与 decode 行数是 decode-only delta。过程累计 route reads 最终为 63,640，不能把每轮 process-total 直接相加。CPU 输入缓冲最终为 57,344 bytes、累计七次容量增长，没有 pinned fallback。

下表为每请求 decode 中 CPU 池各阶段累计 walltime 的三轮中位数，单位 ms；不包含路由等待、输入 D2H、GPU 或 MTP 运行时间，也不是所有核的 CPU 时间之和。总列先求每次四阶段之和再取中位数，不是前四列中位数相加。

| 任务 | 输入量化 | Gate/Up | 中间量化 | Down | 四阶段合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BFS | 7.083 | 351.691 | 2.224 | 156.267 | 517.233 |
| 中文事务 | 3.668 | 177.970 | 1.165 | 74.158 | 258.535 |
| 短输入 | 2.390 | 119.625 | 0.726 | 46.562 | 168.666 |
| 长输入 | 5.208 | 281.382 | 1.859 | 118.236 | 406.699 |

这些 CPU 阶段可与 GPU 工作重叠，不能用约 0.2–0.5 秒的池时间直接推算整体加速；微基准的改善也不替代本表的 API 结果。三轮合计 MTP accepted/offered 为 BFS 1158/1361（85.08%）、中文 484/984（49.19%）、短输入 405/873（46.39%）、长输入 423/807（52.42%）；草稿尝试未计作真实输出 token。

## 物理内存与显存

采样从模型 READY 后开始，直到十二请求完成，计划间隔一秒，共 130 个样本、无采样错误。原始 [resource-samples.jsonl](measurements/resource-samples.jsonl)、[resource-summary](measurements/resource-summary.json)和 [post 快照](measurements/resources-final.json)保留口径；startup footprint 不在这段周期采样中。

| 指标 | 实测 | 口径 |
| --- | ---: | --- |
| 测量前引擎 RSS | 33.475 GiB | 当前物理 working set |
| 最大采样 RSS | 34.051 GiB | 130 次采样最大值 |
| Windows lifetime working-set peak | 34.059 GiB | OS 记录的进程物理峰值 |
| 完成后引擎 RSS | 33.991 GiB | post 当前值，非峰值 |
| GPU 0 最大采样显存 | 21,895 / 22,528 MiB | 整卡周期采样，非连续瞬时峰值 |
| GPU 0 post 显存 | 21,895 / 22,528 MiB | 仍驻留模型/缓存时的当前值 |
| 全系统最大采样物理 RAM 已用 | 49.818 GiB | 含其他系统进程 |
| post 全系统 RAM 已用 / 可用 | 49.505 / 78.366 GiB | 当前值 |

GPU 1 在采样中为 171 MiB、GPU 2 为 0；本引擎只看到 GPU 0，该背景占用不是模型分片。Private commitment / virtual memory 不是物理 RSS，不能再与 RSS 或显存相加。本组物理内存没有达到 124 GiB；没有额外复制一整套专家权重给 CPU 池。所有资源值仅描述这次冻结普通混合 cohort，不能用于尚未测量的 doorbell 或其他并行服务。
