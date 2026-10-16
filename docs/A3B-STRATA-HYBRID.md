# Q8 A3B：按原 Strata 的 CPU/GPU 协作方式运行

当前本机服务已将 context 从 4096 提升到模型声明的 262144。
[50K 实际输入与 256K 显存/延迟报告](../bench/results/2026-10-15-a3b-q8-256k-50k/README.md)
单列三次 FP16 KV 测量、失败的 4096-token 分块尝试和默认关闭的 Q8 KV 对照。
下文短输入表仍是旧 4096-context 配置的结果，不能代表 50K 输入速度。

原 Strata 的 CPU 专家池确实执行矩阵计算，不只是缓存和调度。原项目
`src/kernels/cpu/pool.cpp` 调用 Gate/Up 和 Down 内核；
`docs/DETAILS.md` 的 CPU/PCIe 分配部分说明：GPU 计算显存中的专家，
一部分未命中专家经 PCIe 传入 GPU，其余在 RAM 原地由 CPU 计算，两路并行。
原源码只读核查见 [CPU 路径记录](A3B-CPU-PATH-REVIEW.md)和
[门铃协议记录](A3B-DOORBELL-PORT-REVIEW.md)。本轮没有修改 `G:\Strata`。

`--hybrid-doorbell` 是此项目独立 A3B 引擎的显式实验选项。它使用同一份
标准 Q8_0 权重、固定热点 profile、一个 CPU 服务线程和持久专家行线程池。
短验证窗口内，GPU 发布输入和专家编号后继续计算命中专家，CPU 同时计算
未命中专家，再由 GPU 合并完整结果。此版本的门铃路径固定 PCIe miss
比例为 0；普通 `--hybrid-experts --pcie-frac` 路径保留传入 GPU 的分配方式。
未剪枝、跳过或降低专家位数。

2026-10-15 的 Nsight 检查在 26 个目标验证窗口中看到 24 次已捕获的
整窗图，每次包含全部 40 层门铃；另两次为图预热。约 995 ms 的输出等待
同步包含 GPU 执行时间，不能再与约 947 ms 的 kernel 时间相加。
详见 `bench/results/2026-10-15-a3b-q8-doorbell-profile/`。采集包含 profiler
开销，不用这次采集宣称吞吐提升。

另有两个默认关闭的原项目优化移植：

- `--prefill-cpu-share`：仅对 9–256 token 的 Q8 提示分块，将出现不超过
  两次的完整未命中专家交给 CPU，其余未命中专家仍经有界缓冲传入 GPU。
  大于 256 token 的分块保持 GPU prefill。需要 hybrid 和 CPU pool。
- `--mtp-fused-rounds`：融合上一轮接受前缀的 catch-up 和下一轮首个
  proposal，容量或边界不满足时回退。需要 GPU greedy sampling、deferred
  catch-up、精确 dense/attention 和不超过 7 个草稿。完整模型回滚及输出
  边界测试通过。一个同二进制、同 16000 MiB 专家缓存的 23/256 单次对照
  为关闭 53.56、开启 53.86 token/s，输出 SHA 相同；这点差异不足以选择
  默认开启。融合耗时单列，不重复算入 draft/catch-up。

`STRATA_A3B_ATTENTION_MASK_REUSE=1` 仅复用当前计算图中相同 mask 行的
view，减少逐层重复 H2D 输入；注意力算术不变。环境变量默认关闭，生效
需要新进程，数值测试同时检查 view 共享和逐位输出。

同专家多列 Q8 权重复用也保留为默认关闭的
`STRATA_A3B_NATIVE_EXPERT_REUSE=2/4`。同一新二进制的单次 23/256 对照
关闭/两列/四列分别为 54.93/53.69/53.41 token/s，输出 SHA 一致，因此
本机选关闭。不能把另一旧二进制的速度变化归因于此开关。

使用官方 `ggml-org/Qwen3.6-35B-A3B-GGUF` 的完整 Q8_0 模型和同版本
MTP。120 个 routed tensor 均为 Q8_0，没有重新量化；来源与完整 SHA 见
[Q8 说明](A3B-Q8.md)。CPU 与 GPU 的浮点/激活量化算术不承诺逐位相同。
改变缓存容量会改变执行者，跨配置的输出可能不同；每个配置的目标
验证、回滚、分配和参考计算应分别核验。

硬件为 Threadripper 3960X、128 GiB RAM、GPU 0 RTX 2080 Ti
（驱动报告 22528 MiB，SM75，PCIe 3.0 x16）。只用 GPU 0 计算。
[首轮完整 API 测试](../bench/results/2026-10-15-a3b-q8-doorbell-api/README.md)
使用固定 MTP K2、batch 2048，四组输出中位数为 48.8–63.4 token/s；
2048 token 输入 prefill 538.8 token/s，TTFT 3.834 秒。采样 RAM 峰值
34.1542 GiB、GPU 0 峰值 22057 MiB。100 token/s 目标尚未达到。
后续开关的测量独立记录，不能混合不同配置的最快值。

原有 Q4 与 GPU-only Q8 的配置、模型和冻结二进制保留。新特性默认
关闭；本机服务配置的选择依据同日实测。CUDA 和 CPU 构建分别验证，
本机没有 HIP/SYCL 工具链，未声称这两种后端通过编译或运行。

## 本机最终配置与独立 API 复测

本机选择固定 MTP K2、batch2048、CPU8、mask 复用和短 prefill CPU 分流。
`--mtp-fused-rounds`、native expert reuse 保持关闭。冻结 engine SHA256 为
`2da62666608dc8c9005e654bb3d9c370d400b508e6ee5c4f509046ec4801ec62`。
mask-only 与 mask+share 使用同一个二进制；各自独立新进程完成12请求。
[完整报告](../bench/results/2026-10-15-a3b-q8-tuned-api/README.md)保留逐次
SSE、全部三次中位数、首轮/后两轮、TTFB、客户端速率、资源与 SHA 清单。

下表为选定配置的 temperature0、seed12345、thinkingfalse 测量；每项三次
均计入中位数，不把这些 greedy 数字说成网页默认采样速度。

| 输入场景 | 实际输入/输出 token | TTFB ms | 首输出秒 | 输出 token/s | Prefill token/s |
|---|---|---:|---:|---:|---:|
| BFS 代码 | 160/512 | 11.42 | 0.699 | 65.9 | 238.9 |
| 中文事务解释 | 123/328 | 12.05 | 0.584 | 61.4 | 222.7 |
| 短输入 | 32/256 | 10.86 | 0.352 | 52.3 | 98.4 |
| 长输入 | 2048/256 | 11.02 | 3.832 | 50.1 | 537.8 |

相同二进制关闭 share 时，BFS/中文/短输入的首输出为1.312/1.229/0.813秒，
启用后降到0.699/0.584/0.352秒。短输入 prefill 专家上传从3958.9降到
1016.8 MiB；2048输入不触发分流。短/长 decode 从53.0/50.2变为52.3/50.1
token/s，并非每项速度都提高。CPU/GPU prefill 分配变化使前三种任务的文字
与关闭 share 时不同，中文长度310→328；长输入文字保持逐字节一致。

同一常驻服务随后另做 temperature0.7、top_p0.8、top_k20、seed12345 的
短/长各三次补充，匹配现有网页采样默认，单独审计，不混入上表。

| 实际输入/输出 token | TTFB ms | 首输出秒 | 输出 token/s | Prefill token/s |
|---|---:|---:|---:|---:|
| 32/256 | 1.28 | 0.335 | 47.0 | 100.8 |
| 2048/256 | 11.40 | 3.783 | 47.4 | 546.8 |

100 token/s 输出与1500 token/s prefill 目标均未达到。选定 greedy12请求
的物理 RSS 采样峰值34.1599 GiB、OS lifetime peak34.1959 GiB、GPU0采样
最大22057 MiB；这是定期采样，不是连续瞬时显存峰值。温度0.7补充未另采
RAM/VRAM。交付时后续快照为 RSS34.1478 GiB、GPU0当前22059 MiB，PID40412
保持 loaded/idle0；没有把稍后的当前值混作先前矩阵的峰值。

[数值验证记录](A3B-TUNED-VALIDATION.md)列出54个 CPU逐位 case、18个
prefill CUDA task-layout 参考逐位差异0、cache4 sanitizer0 errors、官方
prefix16/MTP keep1..7 回滚和快照 max_abs0/KL0、输出边界1/2/3/4/5/8/16/32。
CPU CTest为7PASS/6GPUskip/0fail，Python选项33PASS。早期不适用参考图的
失败日志保留，修正布局与重复ID诊断，没有放宽误差阈值。

生产 Q8 配置已采用选定 args/env，原 GPU-only Q8配置逐字节保存在
`strata-a3b-q8-gpu-compatible.json`，原 Q4配置和冻结二进制 hash不变。
8096服务保持驻留，网页采样默认未改，原始Strata未修改；最终WSL状态记录
没有运行中的分发。
