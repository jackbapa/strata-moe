# A3B IQ3_S 可用版本与后续移植交接（2026-10-16）

## 当前可用版本

本轮先冻结已验证的 `Qwen3.6-35B-A3B` 文本服务。以下是当时的本机配置快照；
`strata-a3b-iq3s-strata-optimized.json` 和旧实验启动脚本不随 Git 发布。
新安装使用 [`SETUP.bat`](SETUP.bat)，以后使用 [`START-HERE.bat`](START-HERE.bat) 启动，
参见 [项目 README](README.md)。
截至交接，`http://127.0.0.1:8096/v1/status` 返回 `loaded=true`、
`context.native=262144`、`in_flight=0`。服务只绑定 `127.0.0.1`。
重新启动前先检查 8096 是否已有实例，避免同时加载两份模型。
交接时还用关闭思考的短请求检查了 `/v1/chat/completions`：
“一加一等于几？”返回正文“二”、`finish_reason=stop`，随后服务仍为 `loaded=true`。

| 固定输入 | 当前文件及 SHA-256 |
|---|---|
| 服务配置 | `strata-a3b-iq3s-strata-optimized.json` · `94bb567296d56ebbf5225fb0533f7dd30422f40d0f85e02284e24e551417ac0f` |
| 引擎 | `build-a3b/bin/strata-moe.exe` · `a0a3deaec132200616e7db2c0a18e3cfd719575e660ce6a7d29af58f6c4d9c04` |
| 目标模型 | `models/a3b/Qwen3.6-35B-A3B-UD-IQ3_S.gguf` · `66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66` |
| MTP 草稿 | `models/a3b/mtp-Qwen3.6-35B-A3B-Q8_0.gguf` · `f6cb995fab78e7284097b2e13508e8daad4ffc6a747b433aef399f4e48524b5a` |

目标 GGUF 来自 [Unsloth 的固定版本](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/a483e9e6cbd595906af30beda3187c2663a1118c)，
匹配的 MTP 文件来自 [ggml-org 的固定版本](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/tree/baec3ebee244827cda0f4557eafa8b28f7545fa6)。
**目标模型不是本项目自制的重量化文件。** IQ3_S GGUF 内含 Q6_K 稠密张量；
下文的 Q6_K 优化始终针对同一份 IQ3_S 模型，不是另一套 Q6 模型。

当前配置固定 GPU 0、262,144 token 最大上下文、MTP K2、全 GPU 常驻专家；
50K 输入在短 KV 档使用约 65K 物理 F16 KV。推理矩阵计算在 GPU，
CPU 仍负责服务调度、分词等主机工作，但此路径测到 **0 个 CPU 专家计算行、
0 MiB 专家上传**。原版 Strata 对未驻留专家确实有 CPU 矩阵计算池；
全驻留的本配置没有把专家送去 CPU 算。

## 已完成的真实模型测量

测试机器为 GPU 0 `RTX 2080 Ti`（sm_75，22,528 MiB），单请求、未缓存
50,000 输入 token、256 输出 token、`temperature=0`、`seed=12345`，配置
上下文 262,144。所有显存值是 **GPU 0 整卡采样峰值**，不是本进程独占值。
“HTTP 首字节”指底层 socket 首次读到 HTTP 数据；“首个模型 token”指首个
非空内容或推理 SSE，两者不可混用。输出速度采用引擎报告的生成 token/s。

同二进制、六轮交替的[选中配置对照](bench/results/2026-10-16-a3b-iq3s-combined-exact-r3/exact-ab-report.json)
如下。两臂目标、MTP 模型、请求、最终输出和 MTP 提案均一致；每臂取三轮中位数。

| 指标 | 原对照 | 当前四开关配置 | 变化 |
|---|---:|---:|---:|
| HTTP 首字节 | 19.25 ms | 19.89 ms | +0.64 ms |
| 首个模型 token | 29.385 s | **28.575 s** | −0.810 s |
| 预填充 | 1,708.9 token/s | **1,765.0 token/s** | +56.1 |
| 输出 | 75.2 token/s | **83.2 token/s** | +8.0（10.6%） |
| GPU 0 整卡峰值 | 17,119 MiB | 17,103 MiB | −16 MiB，不能视为显著节约 |

额外四个开关包括 Strata 分组专家 gate/up 与 down 阶段的适配、
GGML 对**同一 IQ3_S 文件内 Q6_K**的精确复用，以及 A3B MTP 的 KV-only
追赶。10.6% 是四项组合的整段收益，不能归给单个内核。
原版 `native_expert_grouped` CUDA 核通过 A3B 形状和 GGUF 权重指针在
decode 中实际执行：50K 请求约 5,000 次调用；prefill 不走这个短序列内核。
`--native-dense-exact` 目前仍由 GGML 稠密算子实现，不能称作原版 Strata
`native_mmvq` 已接入。

为判断“多移植一个原版机制是否一定更快”，又分别做了以下隔离 A/B。
这些试验仍以相同 IQ3_S 目标和 50K/256 请求为准；每行是该试验内部
三轮中位数，**不要把不同试验的绝对速度当作严格横向排名**。

| 单项试验及证据 | 关闭 → 开启输出 | 首个模型 token | GPU 0 峰值 | 结论 |
|---|---:|---:|---:|---|
| [常驻专家并行规划](bench/results/2026-10-16-a3b-iq3s-parallel-plan-ab/exact-ab-report.json) | 85.1 → 84.7 token/s | 28.401 → 28.625 s | 17,103 → 17,103 MiB | 输出与提案一致；无收益，保持关 |
| [共享专家 CUDA 侧流](bench/results/2026-10-16-a3b-shared-selected-50k-ab-r2/exact-ab-report.json) | 82.6 → 82.6 token/s | 28.545 → 28.664 s | 17,103 → 17,103 MiB | 输出与提案一致；无收益，保持关 |
| [紧凑 MTP GPU 采样](bench/results/2026-10-16-a3b-draft-compact-50k-ab/comparison.json) | 85.0 → 85.4 token/s | 28.508 → 28.502 s | 16,991 → 16,991 MiB | 两臂同一派生 Q8 草稿；238 次实际紧凑采样，0.4 token/s 小于运行漂移，保持关 |
| [固定 MTP K2 → K3](bench/results/2026-10-16-a3b-iq3s-k2-k3-ab/exact-ab-report.json) | 83.2 → 79.8 token/s | 28.466 → 28.508 s | 17,103 → 17,165 MiB | K3 提案 317/接受 149，K2 为 238/136；K3 更慢，保持 K2 |

紧凑采样试验的草稿文件由**同一匹配 Q8_0 MTP 文件**逐字节复制选中的
194,208 个输出头行，并添加 I32 映射。目标 IQ3_S 文件完全没变；此派生
草稿只在隔离实验中使用。以上 A/B 的正确性门禁与原始日志都在相应结果目录。
共享侧流第一次跑到第三轮时，基准脚本把逐轮日志路径误算成同臂配置变化；
该轮被门禁拒绝，修正后使用 `-r2` 新目录重跑的六轮结果才列在表中。

## 当前执行架构的真实边界

当前服务**已经使用原版 Strata 分组 IQ 专家 CUDA 数学**，也使用 GPU
采样、MTP K2；GGML 自己已经捕获并回放 target/draft CUDA Graph。
但 A3B 主干仍由固定版本 llama.cpp 的 Qwen3.5-MoE 图搭建，不是原版
Strata 48 层 Flash-Next 验证器图的完整移植。A3B 为 40 层、2048 hidden、
256 专家 top-8，并含 GDN 与周期性完整注意力；原版验证器的 QSA、HC4、
PLE 和状态提交不能只替换常量就运行。原版 CPU 未驻留专家池、跨层上传
线程和拷贝环，对此 **0 上传**的全 GPU 配置没有可节省的工作。
本机 sm_75 也不满足原版部分 sm_80 预填充融合／interleaved 快路的条件。

因此目前**没有完成完整 Strata 执行架构移植，也没有达到 100 token/s**。
不能把代码里存在的实验内核、GGML 的 Q6_K 优化、或一个核的微基准
冒充为选中服务已经执行的原版完整路径。更详细的执行矩阵见
[`docs/A3B-STRATA-IQ3S-EXECUTION.md`](docs/A3B-STRATA-IQ3S-EXECUTION.md)。

## 留存但尚未进入可用版的工作

1. [`a3b/research/full-group-grid/README.md`](a3b/research/full-group-grid/README.md)：
   原版常驻专家网格每组独立发射，A3B 原适配固定 `grid_groups=8`。
   `STRATA_A3B_FULL_GROUP_GRID=1` 已作为默认关闭的源码候选加入，
   40 层逐值对照与六轮 A/B 脚本已备好；**尚未编译或跑 GPU**。
2. [`a3b/work/mtp-fused-chain-k2/README.md`](a3b/work/mtp-fused-chain-k2/README.md)：
   融合首轮与设备内 K2 第二步的独立补丁草案；`git apply --check`、
   脚本语法／dry-run 通过，**未集成、未编译、未验证回滚状态**。
3. [`a3b/experiments/native_q6_gguf_parity/README.md`](a3b/experiments/native_q6_gguf_parity/README.md)：
   同一 IQ3_S GGUF 真实 Q6_K 张量和同一 Q8_1 输入下，原版
   `native_q6_k_mmvq` 对 GGML 的独立逐位／核时间试验。CUDA 编译到
   链接阶段，因缺 `pdl_scope`、`pdl_launch_ok` 两个符号而未生成可执行文件；
   **没有 GPU 结果，更未接入选中服务**。
4. [`a3b/research/mtp-output-profile/README.md`](a3b/research/mtp-output-profile/README.md)：
   Nsight 显式 canary／部分输出采集脚本只完成离线检查，当前模型的
   canary 尚未运行；旧的自动触发采集没有生成有效 `.nsys-rep`，不能用来
   推断整段输出瓶颈。

## 后续顺序与验收

1. **先保持本版可用。** 不改选中 JSON、模型和引擎二进制；任何试验
   使用独立构建和默认关闭开关。测试前核对 8096 空闲，结束后自动恢复
   原服务并核对 `loaded=true`、上下文 262,144。
2. **优先验证原版专家网格差异。** 40 层真实 IQ3_S 权重逐 float 对照；
   再跑真实目标+MTP 的输出、29 条提案和 16 组状态逐字节门禁；通过后
   用同二进制 off/on 交替 50K/256 测首字节、首模型 token、预填充、输出
   和整卡显存。若收益不足或答案变化，继续保持关闭。
3. **修复 Q6 独立程序链接并做核级判断。** 先确定普通原版 Q6_K 核在
   sm_75 上对 GGML Q8_1 输入是否逐位一致，再比较 1／2／3 列核时。
   只有正确且有足够收益时，才做默认关闭的服务接入和 50K A/B。
4. **设备连通 MTP** 要先补齐补丁的回滚／状态检查，检查 CUDA 图缓冲
   生命周期，过严格门禁后才跑同二进制服务对照。K3 已测慢，不再单纯
   增加草稿深度。
5. **完整原生 A3B 验证器** 依照
   [`a3b/research/native-verifier-port/PLAN.md`](a3b/research/native-verifier-port/PLAN.md)
   分阶段实现 A3B 专属权重布局、40 层 GDN／注意力、专家路由、验证与提交、
   NextN round/step 和预填充。每阶段先做逐值／状态正确性，再做相同请求
   的端到端 A/B。此项是较大的新执行引擎工程，不能根据“3B 激活参数”
   预先保证 100 或 200 token/s。

原始 `G:\Strata` 的受版本控制文件在交接前再次执行 `git diff --quiet`
为干净；本轮修改和研究产物仅在 `G:\Strata-Moe`。
