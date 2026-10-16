# A3B MTP 跨步设备图适配：设计与验收关口

**状态：实验 K2 已用 GPU 缓冲区连接两次 draft decode，原版式 round/step/commit 协议及单张跨步 GGML 图尚未实现。** `STRATA_A3B_MTP_DEVICE_PREFILL=1` 在 prompt 阶段把 target NextN hidden 从 GPU 传给 draft 的 catch-up 输入；对应代码在 `a3b/speculative.cpp`、`third_party/llama.cpp/src/llama-context.cpp` 与 `a3b/patches/llama-zy-a3b-mtp-device-prefill.patch`。成功分支省掉整块 target hidden 的 D2H，仅取最终一行供第一次 proposal 使用；不兼容时先安全恢复 host hidden。它不连接两个 proposal step，也不捕获 verify 或 commit。首版在第一个 50K 服务请求上因 GPU hidden 输入丢失而失败；修复后，同二进制两轮 50K/256 对照通过，回答与关闭组逐字节一致，详见 `bench/results/2026-10-15-a3b-iq3s-bridge-fixed-ab/README.md`。开启桥接的首个模型 token 中位数 27.853 s、输出 66.45 tok/s，关闭组为 28.493 s、64.25 tok/s；顺序两轮样本不足以确认稳定提速，显存峰值均为 20,499 MiB。日志 `device prefill calls/rows` 只证明 prompt 传输分支运行，不能证明跨步 CUDA 图执行。

## 从原版 Strata 可复用什么

原版 `src/core/mtp.cpp` 的 `capture_round`、`capture_step` 在固定 GPU 地址上分别捕获 round 和逐步图。`mtp_select` 将本步的 token 与残差写入下一步的 GPU 输入；`draft` 连续提交图，最后同步一次。`src/core/verify.cpp` 另有完整 verify 窗口图及 commit 图。原版也不是一张同时包含 draft 和 verify 的图。可移植的是固定输入/状态地址、设备上的 token 与 hidden 传递、少同步的调度方式。

原版 MTP 权重和算子围绕 hyper-connection 残差、QSA 与它自己的 KV 布局；A3B 的 `third_party/llama.cpp/src/models/qwen35moe.cpp` 则由 GGML 构建单个 NextN attention/MoE 层。不能把原版 `mtp_select`、round 图或 verify 图直接调用在 A3B 张量上。A3B 当前 `MtpDraft::propose` 每步调用 `llama_decode`，之后通过 `llama_get_sampled_token_ith` 和 `llama_get_embeddings_nextn_ith` 取下一步输入；这两个 getter 在 `llama-context.cpp` 同步设备。

## 最小可执行目标：固定 K=2 的一个 GGML draft 图

先只处理 `max_draft >= 2`、greedy GPU draft、一个序列、两个 proposal 的情况；prefill、verify、随机 target sampling、EOS 后丢弃第二 proposal 和所有其他请求继续走现有路径。目标是在一次 `llama_decode` 的 GGML 图内建立以下依赖，然后只读取最终两个 token：

```text
输入 last token、pending hidden、位置 p/p+1、两组预留 KV slot
  -> NextN step 0 -> hidden 0、logits 0 -> 图内精确 greedy token 0
  -> NextN step 1(token 0, hidden 0) -> hidden 1、logits 1
  -> 两列 logits / 两个 sampled token 输出
```

直接把两个已知 token 放进现有 `graph_mtp` 的一个 batch **达不到这个目标**：现有构图在入口一次取得两行 token embedding 和 hidden，第二行不能依赖第一行的采样结果。GGML 自带的 `build_sampling()` 又在模型图末尾应用 sampler，无法自动把第一行的结果接回中间的第二步。

拟修改的边界如下：

1. `third_party/llama.cpp/src/models/qwen35moe.cpp` 与 `src/models/models.h`：把 `graph_mtp` 的单步 NextN 计算提取为可调用两次的构图函数。新图只从 host 接收第 0 行 token/hidden。第 0 步构建完 logits 后，在同一 GGML 图中做与当前 `a3b/speculative.cpp::device_greedy_backend_apply` 相同的 first-max tie 规则和有限值检查；它的 I32 token 供第 1 步 `ggml_get_rows` 查 embedding，第 0 步 `h_nextn` 直接成为第 1 步 hidden。两步各自计算 attention/MoE，最后输出两列 logits。不能只用原始 `ggml_argmax` 而宣称 tie 行为一致。
2. `third_party/llama.cpp/src/llama-graph.h/.cpp`：增加仅供 Qwen35 MTP K2 使用的 graph type 和每步 attention 输入视图。一次 2-token KV context 生成的 K/V 写入索引、RoPE 位置及 `[n_kv, 2]` mask 要切成第 0、1 行。第 0 步的 KV 写入与输出必须在第 1 步的 KV 写入/attention 前形成真实图依赖；仅靠 C++ 建图顺序不够。图要暴露两行 logits/采样 token，且不得把第 1 行的占位 token/hidden 当作模型输入。
3. `third_party/llama.cpp/src/llama-context.h/.cpp`：增加显式的 K2 proposal decode 入口，只在调用期间选择新 graph type；普通 MTP catch-up 仍使用原图。进入图前检查模型、单序列、两行、GPU greedy、`n_outputs_max_per_seq >= 2`，不支持时应在任何 draft KV 变化前回到旧路径。图执行失败后不能静默重试，必须中止该请求。
4. `third_party/llama.cpp/src/llama-kv-cache.cpp`：先验证现有 2-token batch 的 KV slot 与 causal mask 是否可直接分行使用。当前 `set_input_kq_mask_impl` 通过 `p0 > p1` 屏蔽未来位置，理论上第 0 行不应看到第 1 个 slot；必须用真实 A3B KV 布局和数值实验确认。若 `cpy_k/v` 的写入副作用在两步之间缺少图依赖，要在此层显式补齐，不能假设 mask 会保证执行顺序。
5. `a3b/speculative.cpp/.hpp` 与 `a3b/main.cpp`：仅在独立环境开关、`count == 2` 时调用 K2 入口；draft context 的 `n_outputs_max` 与 `n_outputs_max_per_seq` 在此开关下升到 2。读两个 proposal token 后沿用现有 EOS 截断、`trim(draft_, pos)` 回滚和 target verify/commit。记录每请求实际 K2 图调用/回退次数。默认开关关闭，原 proposal 路径保持不变。

一个 GGML 图只说明两步在同一个依赖图内。还要检查 `ggml-cuda.cu` 的真实 CUDA graph capture/replay 计数，不能把 GGML graph 构建数或新增 graph type 当成 CUDA graph 命中。若 CUDA capture 因 MMID/自定义 MoE/KV 写入失效，应分别报告图构建与设备图回放的情况。

## 放行条件

- 用相同 IQ3_S target、Q8_0 MTP 权重与固定 greedy 采样，比较旧路径和 K2 的第 0/1 步 draft logits、token、hidden；报告最大/平均绝对误差和 tie/有限值检查，不只比较最终 target 输出。
- 覆盖 1-token 与超过一个 `--batch` 的 prompt、位置靠近上下文末端、第一 proposal 为 EOS、K2 后 partial/full rejection；对照继续解码 logits 与 draft KV 回滚。任一差异在没有解释前不得默认启用。
- 记录两个真实计数：K2 GGML 图执行数与 CUDA graph replay 数。`replay=0` 时不能称为 CUDA graph 移植。
- 同日同卡同配置比较开关 off/on 的 32、2048、50K 输入，256K context；分别给首个模型 token 延迟、prefill token/s、输出 token/s、显存、MTP 接受率。K2 仅在有明确收益且数值关口通过后考虑扩大 K。

目前完成的是经过 50K 数值和服务测试的 prefill D2D bridge；另有两次独立 `llama_decode` 提交的 K2 device-chain 实验开关。后者在修复草稿 Q8 词嵌入的 GPU 放置后，32/64 与 50K/256 请求均完成，50K 回答及 MTP 接受数与关闭组相同；见 `bench/results/2026-10-15-a3b-iq3s-chain-k2-fixed-ab/README.md`。它多占约 514 MiB 峰值显存，只有一轮 50K 性能样本，不能据此称稳定加速。旧日志的“跨步 CUDA 图回放 0”是硬编码字面量，**不是测量**；后续在 128-cell 严格诊断的 26 个真实 K2 链中，Nsight 观察到 49 次 `cudaGraphLaunch`（23 链各两次，3 链各一次），同时看到 624 次 `cudaStreamSynchronize`，其中多个小 H2D 输入上传后紧跟同步。详见 `bench/results/2026-10-15-a3b-mtp-k2-graph-audit/README.md`。该短诊断不能换算为 50K 性能，但证实 K2 分支确有 GGML 图回放；它仍不是原版完整 round/step/commit 协议。固定单张 K2 图与原版式 verify/commit 状态图均未完成；当前 GGML 目标验证路径本身也会回放 CUDA 图，新增外壳图没有直接依据。
