# Q8 GPU-only 首版 API 基线，2026-10-14

这是优化前候选的九个常驻 API 请求，**不是最终性能或 100 token/s 达标结果**。只测 32/256 和两个自然任务，没有运行 2048 长输入或完整矩阵。候选之后已停机以构建后续优化；本报告保留原始数据，不覆盖现有 Q4 结果。

GPU 0 为 RTX 2080 Ti，驱动报告 22,528 MiB；Threadripper 3960X，128 GiB RAM，Windows/CUDA 12.6。完整 ggml-org Q8_0、同仓 Q8 MTP；context4096、batch512、K2、pageable专家、8 staging slots、native dense/attention exact、ARGMAX_MULTI=1、DENSE_Q8_REUSE=1。专家权重在RAM和GPU缓存之间调度，模型矩阵只由GPU计算。

引擎 SHA-256：`d806fc0a9a30d9718579b151f4a927e2073e86c9659dcebdd35385e7542e6c7a`。全部请求串行、temperature=0、seed=12345、thinking关闭，测试间没有reload/unload。具体配置见[快照](server-config.json)，模型来源与hash见[环境](environment.json)。

## 原始三轮与中位数

| 任务 / 实际输入 | 实际输出token（三次） | HTTP TTFB ms | 输出 TTFT s | Prefill token/s | 三次engine decode token/s | 中位 / 客户端decode token/s |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 合成中文 / 32 | 256 / 256 / 256 | 10.84 | 0.3936 | 83.8 | 32.4 / 39.2 / 41.1 | 39.2 / 39.04 |
| 自然代码 / 124 | 512 / 512 / 512 | 10.73 | 1.0885 | 115.2 | 17.8 / 22.9 / 24.8 | 22.9 / 22.88 |
| 自然中文 / 98 | 297 / 292 / 314 | 10.89 | 1.0258 | 96.8 | 20.6 / 24.9 / 26.4 | 24.9 / 24.80 |

主矩阵第一条请求 TTFT 为3.1577秒，后两次分别0.3878/0.3936秒；第一条仍计入三轮中位数。Prefill对应10.2/85.2/83.8 token/s，不能把32-token预填充数外推成长输入。自然任务沿用Q4报告同一问题，但Q8与Q4不是相同权重/布局的受控单因素对比。

HTTP TTFB包含headers，输出TTFT取首个非空文本/推理delta，排除角色/进度。Engine decode不含加载与prefill、不把草稿尝试算进输出；客户端decode按(usage completion_tokens−1)/(末文本−首文本)计算，存在MTP/UTF-8成批差异。

## MTP与真实传输

| 请求 | MTP接受 / 提议 | 接受率 | Decode GPU cache命中率 | Decode上传MiB | CPU专家rows |
| --- | --- | ---: | ---: | ---: | ---: |
| prompt32-output256-repeat01-resident-first-unknown | 137 / 234 | 58.55% | 94.75% | 16947.9 | 0 |
| prompt32-output256-repeat02-resident-repeat | 133 / 242 | 54.96% | 97.28% | 9444.6 | 0 |
| prompt32-output256-repeat03-resident-repeat | 134 / 240 | 55.83% | 97.65% | 8204.6 | 0 |
| natural-code-repeat01 | 333 / 355 | 93.80% | 76.62% | 111081.2 | 0 |
| natural-code-repeat02 | 334 / 353 | 94.62% | 84.03% | 78670.7 | 0 |
| natural-code-repeat03 | 330 / 358 | 92.18% | 86.23% | 69624.6 | 0 |
| natural-chinese-repeat01 | 156 / 282 | 55.32% | 86.56% | 45941.4 | 0 |
| natural-chinese-repeat02 | 148 / 286 | 51.75% | 91.19% | 32544.4 | 0 |
| natural-chinese-repeat03 | 159 / 310 | 51.29% | 92.56% | 30520.3 | 0 |

所有九次输入count与最终usage相同，cached_tokens=0，没有reasoning，SSE均收到DONE。三个条件的greedy回复均观察到不逐字一致；缓存历史在继续改变，原因尚未定位，不能把它写成重复输出一致或已确定的舍入原因。原文与SSE保留供后续诊断。

## 资源快照

采样时间 `2026-10-14T12:40:18.114764+00:00`，九请求结束后仍常驻，**不是峰值**：

- 引擎PID 33500，RSS 33.713 GiB。
- GPU 0 总占用 21,227 / 22,528 MiB；GPU 1 171 MiB，GPU 2 0 MiB。
- 全系统物理RAM已用 49.08 GiB，可用 78.79 GiB。
- Windows private commitment包含CUDA/WDDM提交，不能与RSS和显存简单相加。

[资源原始快照](resident-resource-snapshot.json)、[冻结引擎日志](engine-baseline.log)、[逐请求传输记录](transfer-per-request.json)、[主三次原始结果](main/results.json)、[自然六次原始结果](natural/results.json)。所有*.events.jsonl保留原始SSE与时间，*.output.json保留回复。

## 后续

这份基线给后续异步桥接与缓存profile优化作参考。最终Q8仍须独立完成数值验证和更全面的TTFB/TTFT/decode/prefill、内存及MTP统计；不把同trace训练与重放的cache覆盖当作泛化能力或100 token/s实测。Q4兼容程序和[Q4完整报告](../2026-10-14-a3b-full-gpu-latency/README.md)保持独立。
