# A3B Q8 原式 CPU/GPU 路径：数值验证记录

本记录对应 2026-10-15 的 mailbox、attention mask 复用及短 prefill CPU
分流。只整理 root 已完成的日志，没有新增 GPU、API、编译或性能请求。
[固定文件清单与 SHA256](../bench/results/2026-10-15-a3b-q8-tuned-api/validation-evidence.json)
包含 14 个明确列出的成功、失败及工具链诊断文件；不含正在运行的服务日志。
性能另见[独立 API 报告](../bench/results/2026-10-15-a3b-q8-tuned-api/README.md)。

使用标准 Q8_0 原始权重，未重新量化。CPU 计算使用独立 ggml Q8_0
quantizer/vec-dot、标量 `std::exp` 的 SwiGLU；GPU 参考使用普通 GGML CUDA
算子。CPU 与 GPU 的算术不承诺彼此逐位相同，改变执行者可能改变文字。
每条路径分别对自己的独立参考核验；这不等于证明所有配置输出相同。

## 已完成的检查

| 检查 | 范围和实际结果 | 证据 |
|---|---|---|
| CPU 原生专家池 | 54 个逐位 oracle case；含 2048/512 几何、广播/8 lanes、padded stride、重复任务、睡眠唤醒及错误恢复；新 T9/32/256 各用 4 个稀疏任务 | [CPU 54 cases](../bench/results/2026-10-14-a3b-q8-validation/cpu-expert-prefill-54-cases.log) |
| prefill 混合分配 | Q8 H2048/F512/E32，T9/32/256 × lanes1/8 × cache0/4/16，共 18 cases；CPU 每例 3 jobs/2 完整专家，GPU 的 task-layout 参考逐位差异实际均为 0；原误差阈值未放宽 | [v4 全矩阵](../bench/results/2026-10-14-a3b-q8-validation/prefill-share-fused-18-cases-v4.log) |
| CUDA 内存检查 | cache4 下 6 个 T/lanes cases，全部 PASS；Compute Sanitizer `ERROR SUMMARY: 0 errors` | [v4 memcheck](../bench/results/2026-10-14-a3b-q8-validation/prefill-share-cache4-memcheck-v4.log) |
| 官方完整 Q8 与 MTP | prefix16 实际触发 prefill 分流：560 calls、22792 CPU jobs、17920 专家组；rollback/snapshot keep1..7 全部 max_abs=0、KL=0、top1 一致且 finite | [完整模型 strict v2](../bench/results/2026-10-14-a3b-q8-validation/q8-doorbell-prefill-share-prefix16-mtp-k6-strict-v2.log) |
| MTP 输出边界 | 同一 strict 日志中 limits1/2/3/4/5/8/16/32 全部 PASS；GPU draft sampler、target sampler、deferred catchup 实际生效 | [完整模型 strict v2](../bench/results/2026-10-14-a3b-q8-validation/q8-doorbell-prefill-share-prefix16-mtp-k6-strict-v2.log) |
| CUDA / CPU 构建 | 两种配置均完成链接；v4 fixture 的再次链接也完成 | [CUDA](../bench/results/2026-10-14-a3b-q8-validation/build-prefill-share-mask-reuse.log)、[v4 fixture](../bench/results/2026-10-14-a3b-q8-validation/build-prefill-reference-v4-retry.log)、[CPU](../bench/results/2026-10-14-a3b-q8-validation/build-cpu-prefill-share-mask.log) |
| CPU CTest | 13 项：7 PASS、6 GPU 项显式 SKIP、0 FAIL；不能把这 6 项写成 CPU 构建已运行 GPU 测试 | [CTest](../bench/results/2026-10-14-a3b-q8-validation/ctest-cpu-prefill-share-mask.log) |
| Python 选项/日志协议 | 33 tests PASS，涵盖可选开关和参数依赖 | [options](../bench/results/2026-10-14-a3b-q8-validation/tools-options-final.log) |

prefill fixture 还检查每个角色上传后的 Q8 原始字节 hash、逐段字节比较、
CPU/GPU 完整专家归属、真实上传字节、命中/缺失、原始 RAM 权重未变，以及
CPU unique-input 量化复用。cache0 使用 8 个 GPU staged 专家组；cache4/16
使用 7 个，CPU 稀有专家始终完整分配，不把同一个专家的 jobs 拆到两路。

CPU 池 54 cases 是严格逐位检查。prefill GPU task-layout 的逐位差异为本次
18 个实际结果的观测值；该 fixture 保留原检查容差
`4e-5 * (1 + abs(CPU_reference))` 和 finite 检查，没有为通过而放宽。
独立完整模型 strict 仍另行验证回滚和 accepted-prefix 延续状态。

## 早期失败与参考图修正

[v2 失败日志](../bench/results/2026-10-14-a3b-q8-validation/prefill-share-fused-18-cases-v2.log)
保留 T9/cache0 的 GPU 参考差异约 `4.2528e-5`。
[v3 失败日志](../bench/results/2026-10-14-a3b-q8-validation/prefill-share-fused-18-cases-v3.log)
和[早期 memcheck](../bench/results/2026-10-14-a3b-q8-validation/prefill-share-cache4-memcheck.log)
也保留，未删除或改名，不列为成功验证。

问题有两个层次。原生 `[top-k,T]` 诊断图收到测试故意重复的 top-k expert ID，
违反 CUDA MMQ compact-index helper 的每 token/expert 唯一性要求，产生非法
访问风险；v4 保留重复 ID 的引擎测试，但将这条不适用的诊断明确打印为
`UNSUPPORTED`，其 NaN 是无诊断结果，不能解释为零误差或 PASS。

另一个问题是普通 CUDA MMQ 的 stream-K 分解取决于 expert tensor `ne2`
及 gathered row 数。全 E32 原生参考与实际 E8 staging/cached task 可以有
不同舍入。v4 新参考在 CPU 独立 gather 输入、复制并核验原 Q8 字节，以
实际 cache/staging 容量建立普通 CUDA FFN，再在 CPU 独立 scatter；它不调用
被测 `Task`、backend 分配或私有 executor。CPU-owned 行仍对独立单 job
serial oracle 检查。flat、singleton-lane 参考保留为布局差异诊断，不再把
这些不同布局强行当作 task 的逐位参考。

v4 的 18 个 task-layout GPU 结果逐位差异均为 0；cache4 的 sanitizer 也为
0 errors。修正的是 oracle 的适用布局与非法诊断输入，没有改 Q8 权重字节、
生产 CPU/GPU 算术或误差阈值。成功日志与早期失败日志均可追溯。

## 验证边界

完整模型 strict 使用一个 prefix16 测试任务，不能代替广泛文本/代码质量
评估，也不是吞吐基准。API 测试另记录每个冻结 engine 的 SHA；测试 fixture
可执行文件为单独构建，不能把 fixture 通过误写成所有生产二进制完全相同。

本机 HIP 构建缺少 `hipConfig.cmake`；SYCL 构建报告编译器不支持 SYCL。
对应[HIP](../bench/results/2026-10-14-a3b-q8-validation/build-hip-unavailable.log)、
[SYCL](../bench/results/2026-10-14-a3b-q8-validation/build-sycl-unavailable.log)
诊断保留，未声称这两后端通过构建或运行。原始 `G:\Strata` 只读，未修改。
