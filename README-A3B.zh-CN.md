> 此文件记录适配过程中的旧配置与实验。面向新用户的安装、启动和发布测速以 [README.md](README.md) 为准；以下旧启动脚本可能下载其他量化文件，不属于当前一键安装路径。

# Strata-Moe 的 Qwen A3B 路径

这个目录新增了 Qwen3.5-35B-A3B / Qwen3.6-35B-A3B 的独立文本推理引擎 `strata-moe`。
原有 Flash-Next C++ 引擎、安装器和启动配置没有改动。Python 服务仅对 A3B 引擎加入采样协议适配；原 Flash-Next 行为保持不变。`START-HERE.bat` 仍然走原路径；A3B 的已测 IQ3_S 路径使用下文的专门启动脚本。

当前可用版本、完整测量表、未完成的 Strata 架构移植和下一步验收顺序统一记录在 [2026-10-16 交接文档](A3B-IQ3S-HANDOFF-2026-10-16.md)。

2026-10-16 本机 8096 服务已切到单独的 **`strata-a3b-iq3s-strata-optimized.json`**。它仍使用同一个 Qwen3.6-35B-A3B UD-IQ3_S 主模型、匹配 Q8_0 MTP 和 GPU 0，开启原版 Strata 分组 IQ 专家路径及四个额外的默认关闭优化。50K 未缓存输入、256 输出、256K 配置上下文的六轮交替同二进制测试中，输出中位数 **75.2→83.2 token/s**，首个模型 token **29.385→28.575 秒**，预填充 **1708.9→1765.0 token/s**；HTTP 首字节 **19.25→19.89 毫秒**，GPU0 整卡采样峰值 **17119→17103 MiB**（16 MiB 不能算显著节约）。输出和 MTP 提案逐次相同。完整条件及原始结果见[六轮报告](bench/results/2026-10-16-a3b-iq3s-combined-exact-r3/README.md)。从本目录可运行 `START-A3B-IQ3S-OPTIMIZED.bat`，或用 `python -m serve.server --engine strata --config strata-a3b-iq3s-strata-optimized.json --port 8096`；启动前需停止已有 8096 服务。**尚未达到 100 token/s，也不是完整 Strata 执行架构移植。**

随后完成的[13 轮留一消融](bench/results/2026-10-16-a3b-iq3s-four-switch-ablation/README.md)在 50K/256 请求上确认所有输出和提案相同，并验证四开关全开相对各留一组合的条件性收益；全开五轮输出中位数 83.0 token/s。Strata 的 16-lane down 与 MTP KV-only 的差异更明显，GGML Q6_K 复用的贡献约 1 token/s、证据较弱。四开关组合保留为显式配置，未改变引擎默认值。

原版 Strata 的 MTP 缩小词表方案也做了 A3B 专用、默认关闭的移植试验，使用**同一 IQ3_S 主模型**和由匹配 Q8_0 MTP 文件逐行复制出的子集输出头。干净的六轮交替 50K/256 对照中，完整词表与 194,208 行子集的输出中位数分别为 **81.9 与 81.7 token/s**，首个模型 token **28.571 与 28.663 秒**，GPU 0 整卡峰值 **17,103 与 16,993 MiB**。输出及提案哈希相同；当前实现仍把子集 logits 散回完整词表，**没有测出速度收益**，因此未切换 8096 服务。详见[原始数据与测试限制](bench/results/2026-10-16-a3b-draft-vocab-50k-clean-ab/README.md)。

## IQ3_S：本机 GPU 0 的已测配置

此前作为对照的服务配置是 `strata-a3b-iq3s-q6-kv-tier-f16-auto-native.json`。它加载 Unsloth 的 Qwen3.6-35B-A3B UD-IQ3_S 混合量化 GGUF 和独立 Q8_0 MTP 头，配置 262144-token 最大上下文、MTP K2、全 GPU 驻留、原版 Strata 分组 IQ 专家 CUDA 核。模型文件中的 Q6_K 密集层张量仍是**同一个 IQ3_S GGUF 的一部分**，不是另一个 Q6 模型，也没有本项目自制重量化。50K 输入时分配 65K 物理 F16 KV；超过 131K 时切换到 256K 物理 Q8_0 KV。GPU 0 实际计算路由专家，CPU 不执行专家矩阵乘法。

本机 RTX 2080 Ti（22,528 MiB）同一 50K 输入、256-token 输出，在当前二进制且 MTP KV-only 关闭时，两轮中位数为：**HTTP 首字节 14.99 毫秒、首个模型 token 27.616 秒、预填充 1819.65 token/s、输出 81.1 token/s、整卡显存峰值 17,119 MiB**。早一版同配置的两轮输出中位数为 79.35 token/s；不同时间的成绩不能直接归因于某个改动。100 token/s 目标尚未达到；该配置也**不是完整 Strata 执行架构**。原始 JSON、此前五组对照和已移植/未移植机制见[MTP KV-only 同二进制对照](bench/results/2026-10-15-a3b-mtp-kv-only/README.md)、[五组同二进制实测](bench/results/2026-10-15-a3b-iq3s-f16-auto-ab/README.md)及[Strata 执行路径核对](docs/A3B-STRATA-IQ3S-EXECUTION.md)。

从 `G:\Strata-Moe` 启动这个已测配置：

```powershell
.venv\Scripts\python.exe -m serve.server --engine strata --config strata-a3b-iq3s-q6-kv-tier-f16-auto-native.json --port 8096
```

只优先考虑输出速度时，可选 `strata-a3b-iq3s-q6-kv-tier-f16-auto-native-sub16.json`，它通过 `STRATA_A3B_DOWN_MODE=1` 启用 A3B 512 维 IQ3_S down 的 16-lane CUDA 核。同二进制的三组交替 50K/256 中位数：**81.3→83.1 token/s**，答案、MTP 接受数和 17,119 MiB 显存峰值相同；首模型 token **27.814→28.052 秒**、预填充 **1815.9→1803.3 token/s**，因此默认仍关闭。原始记录见[down 核对照](bench/results/2026-10-15-a3b-iq3s-down-kernels/README.md)。

`START-A3B-IQ3S.bat` / `start-a3b-iq3s.sh` 仍是早期的固定 Q8 KV 兼容配置，首次运行会下载并校验模型。其早期 50K/256 基线为首模型 token 28.80 秒、预填充 1749.1 token/s、输出 54.2 token/s，见[早期测量](bench/results/2026-10-15-a3b-iq3s-first/README.md)。原 Q8 模型与 `strata-a3b-q8.json` 保留，可用 `START-A3B-Q8.bat` 启动。

## 单卡全 GPU 路径

显存足够时，`--native-gpu --gpu-embedding` 让主模型全部权重直接驻留所选 GPU。
路由、注意力、Gated DeltaNet、共享/路由专家、输出头和 MTP 的矩阵计算都在 GPU；没有 CPU 冷专家计算或逐层专家回读。
模型装不下时会报分配错误，需要选择较小模型或显式换回下面的 RAM 专家路径。
`--vram-mib 0` 使用实际可用显存；服务设置 `idle_unload_s: 0` 后持续驻留。

本机 0 号 RTX 2080 Ti 的驱动报告 22 GiB。为容纳完整 A3B 与 MTP，新增了可复现配方：
Q4_K_S 专家保留下载文件的 Q4_K/Q6_K，其他主干大矩阵为 Q8_0；将输出头从原 Q6_K 提升为对应 Q5 文件中的 Q8_0。
工具逐字节复制量化张量，保留元数据，不先解量化再量化；输入文件不改动，输出另存并记录逐张量与整文件 SHA-256。
这项配方没有 BF16 质量评测结论，不能称为无损量化。

```powershell
.venv/Scripts/python.exe tools/a3b.py download --quant Q4_K_S
.venv/Scripts/python.exe tools/a3b.py download
.venv/Scripts/python.exe tools/a3b.py download --mtp
.venv/Scripts/python.exe tools/prepare_a3b_model.py --base models/a3b/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf --donor models/a3b/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf --output models/a3b/Qwen3.6-35B-A3B-Strata-Q4_K_S-Q8Head.gguf
```

`--share-mtp-weights` 在形状、类型、布局和内容 SHA-256 全部一致后复用目标模型的 embedding、输出头及归一化权重。
`--gpu-sampling` 将中性 greedy 的选择放到 GPU，每行仅回传 token 与有限值检查；温度采样、penalty 和 logits 诊断仍走原有取样器。
`--defer-mtp-catchup` 只补算接受前缀，`--fused-moe` 保留完整专家 gate/up/激活/down 的连续图。
`--native-dense-exact` 是短窗口 dense 单列算术实验开关，应以对应模型的数值及性能测试决定是否启用。
`--native-attention-exact` 保持短验证窗口的单 query 算术；`--mtp-ubatch` 独立规划草稿 microbatch。
本机全 GPU 配置、测量条件及限制见 [docs/A3B-FULL-GPU.md](docs/A3B-FULL-GPU.md)。

完整标准 Q8_0 使用独立的 `strata-a3b-q8.json`。这条兼容路径按原 Strata 思路，由 GPU 0 计算主干和热点专家，CPU 专家池并行计算冷专家；整窗 CUDA Graph、MTP K2、短提示分担和 mask 复用已验证。上下文为 262144 token，`START-A3B-Q8.bat` 启动后保持常驻；旧 4096 token 调优配置保存在 `strata-a3b-q8-4k-compatible.json`。实际 50000 token 输入结果见 [256K/50K 实测](bench/results/2026-10-15-a3b-q8-256k-50k/README.md)。测量、开关及范围见 [docs/A3B-STRATA-HYBRID.md](docs/A3B-STRATA-HYBRID.md)。先前所有专家在 GPU 计算的 Q8 方案保留在 `strata-a3b-q8-gpu-compatible.json`；已测 Q4 配置、模型和冻结二进制也保留。官方量化来源见 [docs/A3B-Q8.md](docs/A3B-Q8.md)。新特性是显式选项，生成器默认参数保持不变；profile 训练覆盖率、运行命中率与 token/s 分别报告。

所有 graph/CUDA 修改均以 `a3b/patches/` 的补丁记录，由构建脚本对固定 llama.cpp 版本幂等应用。
它们只作用于此项目的独立 A3B 构建。原 `G:\Strata` 没有修改。

## 启动

若要使用旧版 Q4 路径，可在 `G:\Strata-Moe` 下双击 `START-A3B.bat`。首次运行会编译、下载并校验模型、生成独立配置，然后启动原有网页/API：

```text
http://127.0.0.1:8096
http://127.0.0.1:8096/v1
```

旧版 `START-A3B.bat` 的配置使用 Q4_K_S + Q8 输出头、0 号卡全 GPU、MTP 2、主 batch 1024 / 草稿 microbatch 512，持续驻留。
配置不存在时，一键脚本默认下载 `unsloth/Qwen3.6-35B-A3B-GGUF` 的 `Qwen3.6-35B-A3B-UD-Q5_K_M.gguf`，文件 26,456,194,016 字节。
已有配置会直接使用其中的模型，不会要求重新下载默认 Q5 文件。
下载锁定版本和 SHA-256，支持分段续传。模型、构建产物和配置均保存在本项目内。
本机现有 Python 环境、Visual Studio 2022 和 CUDA 12.6 可用于构建；新机器还需要安装这些开发依赖及原项目 Python 服务依赖。

也可以分步运行：

```powershell
.venv/Scripts/python.exe tools/a3b.py build --arch 75
.venv/Scripts/python.exe tools/a3b.py download
.venv/Scripts/python.exe tools/a3b.py run -- --prompt "请用中文解释什么是混合专家模型。" --predict 256
.venv/Scripts/python.exe tools/a3b.py config
.venv/Scripts/python.exe -m serve.server --engine strata --config strata-a3b.json --port 8096
```

`--arch` 是 CUDA 计算能力：RTX 20 为 75，RTX 30 通常为 86，RTX 40 为 89，RTX 50 为 120（后者需要支持它的 CUDA 工具链）。
Linux 使用 `bash start-a3b.sh`，或者手动用 CMake 构建 `a3b/`。

已有单文件 GGUF 可直接传 `--model`，不需要转成 Flash-Next 的 pack：

```powershell
.\build-a3b\bin\strata-moe.exe --model D:\models\Qwen3.5-35B-A3B-Q6_K.gguf --vram-mib 16384 --ctx 4096 --prompt "你好"
.venv/Scripts/python.exe tools/a3b.py config --model D:\models\Qwen3.5-35B-A3B-Q6_K.gguf
```

生成器遇到已有 `strata-a3b.json` 会保留它并报错，避免覆盖用户设置。更换模型时先移动旧配置，或直接编辑其中的模型、tokenizer 路径。
服务默认仅监听 `127.0.0.1`；如果需要其他地址，必须给原 Python 服务配置 API key。

## 内存与专家调度

复用的是 Strata 的分层思路，A3B 的计算图使用项目已下载的 llama.cpp（与 `setup.py` 相同的固定版本）。
新后端位于 `a3b/backend.cpp`，通过 ggml 后端接口和独立构建使用的可复现 graph 补丁接入。

1. 注意力、Gated DeltaNet、路由器、共享专家、输出头和 KV 状态由 GPU 主干处理。
2. 路由专家完整保留在 RAM，优先使用驱动的页锁定内存；驱动无法分配时采用普通 RAM。
3. 上下文、主干及可选 MTP 分配完成后，按真实剩余显存与预算决定热专家缓存大小，并保留 640 MiB 供工作缓冲区、CUDA graph 等使用。
4. 每个专家张量持有固定数量的 GPU 槽位。按访问频率选择晋升和淘汰，频率定期衰减；正在使用的命中槽位不会被覆盖。
5. 解码将路由行分为 GPU 命中与 CPU 未命中。命中路径直接读取 GPU 激活并写回 GPU，分组时使用 GPU gather/scatter；仅 CPU 未命中需要下载输入。GPU 工作提交后，CPU 运行其余行，最后按原 token / 路由位置合并；没有删除专家或改变路由权重。
6. 提示词按块处理，将实际用到的专家分组流经有限 GPU 槽位，最后留下访问较多的专家。计算缓冲区按类型和矩阵形状复用。

`--vram-mib 16384` 可设置本进程的显存规划预算，不要求整份模型装进 16 GiB；默认值现为 `0`，使用可用显存。可用 `--cache-mib N` 进一步缩小专家缓存；`--cache-mib 0` 保留 GPU 主干、全部路由专家交给 CPU。全 GPU 模式不使用此专家缓存。
`--cpu` 则运行原始 llama.cpp 的全 CPU 计算路径，供数值对照。上下文默认 4096，提示词块默认 128，线程默认物理核心数的近似值。
专家仍完整占用系统 RAM，显存缓存是它们的副本；本实现没有移植原 Strata 的 RAM 补集、专家 mmap 或 KV 流式存储。

加载默认关闭源 GGUF 的 mmap，避免已经复制到专家 RAM 的权重在文件映射中再次驻留。
本机同一程序、同一 16 GiB 显存预算实测，`--mmap` 的 RSS 为 46.800 GiB，默认路径为 23.178 GiB。
Windows 的 private commitment 还包括约 16 GiB 驱动分配，不能把它与物理驻留 RAM 或显存相加当成实际内存使用。
详细的 64 位虚拟内存与驻留页记录在 `bench/results/2026-10-14-a3b-memory-fixed/`。

## MTP 加速

现有 Q5_K_M 文件不含训练后的 NextN 权重，需要另下载与主模型对应的 MTP 文件。
Qwen3.6 可使用锁定版本、校验 SHA-256 的独立 Q8_0 草稿头（1,990,649,632 字节），主模型仍保持 Q5/Q6/Q8 混合精度：

```powershell
.venv/Scripts/python.exe tools/a3b.py download --mtp
.venv/Scripts/python.exe tools/a3b.py run -- --mtp-model models/a3b/mtp-Qwen3.6-35B-A3B-Q8_0.gguf --mtp-draft 3 --stable-expert-graphs --cache-graph-routes --exact-verify --prompt "解释混合专家模型" --predict 256
```

启用 MTP 时，目标模型一次验证至多 `--mtp-draft + 1` 个输入 token（上例为 4）；每个输出 token 都从目标 logits 取样，草稿只在与目标取样相同时被接受。
拒绝后回滚注意力 KV 和 Gated DeltaNet 状态，检查回滚是否成功；每个新请求清除目标、草稿与隐状态。
CPU 未命中与 GPU 命中仍分工计算，短验证窗口不会走长提示词的流式专家加载路径。
GPU 上的 MTP 权重和状态也计入 `--vram-mib`，不会额外默认占满另一张卡。

可选调优开关：`--stable-expert-graphs` 为短窗口按权重保留计算图，`--cache-graph-routes` 在同一图调用内复用路由快照。
`--preserve-verify-layout` 为完整 GPU 命中直接引用原输入/输出，跳过 gather/scatter。
`--exact-verify` 对量化专家的短多 token 窗口强制使用单列 MMVQ 算术，并优先于 `--preserve-verify-layout`。
启用 MTP 后，单列专家验证默认开启。它解决了本机 CUDA 多列专家计算的数值差异：完整 GPU 缓存回滚测试的四种接受长度与独立逐 token 参考 logits 均逐 float 相等。
这只是命名模型、量化和测试 fixture 的测量，不承诺所有设备、混合 CPU/GPU 路由或输入都逐字节相同。
这些开关均不改变原 Flash-Next 引擎；MTP 本身与其他调优开关需要显式启用。

创建新网页配置时，可显式使用 `tools/a3b.py config --mtp-model models/a3b/mtp-Qwen3.6-35B-A3B-Q8_0.gguf --mtp-draft 3`。
已有配置则给 `args` 添加同样两个参数；不传 `--mtp-model` 就关闭 MTP。
Qwen3.5 必须配套自己的 MTP 权重，不能仅因结构相同而默认使用 Qwen3.6 的草稿头。

批量验证、不同缓存容量与 CPU/GPU 量化激活舍入可能改变目标 logits，因此不承诺 MTP 开关前后逐字节一致。
缓存跨请求持续调整时，即使使用相同 seed，随机采样也可能因这些舍入差异产生不同回复；seed 固定的是取样器初始化。
独立回滚测试及其测量范围见 `bench/results/2026-10-14-a3b-mtp-validation/README.md`。

```powershell
# 实际模型的零/部分/全接受回滚校验，独立比较后续 logits
.\build-a3b\bin\strata-moe-speculative-test.exe --model models/a3b/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf --device 0 --cached-experts --stable-tasks --preserve-verify-layout --cache-graph-routes --exact-verify --assert-sequential
```

## 量化选择

默认选择 Q5_K_M 是为了先验证一个较保守、与现有 ggml CPU/CUDA 算子兼容的配置，并不表示已经测得它与 BF16 的质量差距。
这个具体 GGUF 的 120 个路由专家张量中，80 个使用 Q5_K、37 个使用 Q6_K、3 个使用 Q8_0。
后端直接复制这些块，不做二次量化或把它们统一降成 2/3 bit。

Q4_K、Q5_K、Q6_K、Q8_0 的专家行计算有独立参考测试。Q6_K / Q8_0 可作为更高精度选项，但会占用更多 RAM，并让相同显存预算缓存更少专家。
不能仅凭 A3B 的名称断言某种位数安全或不安全：本次尚未做 BF16 对照、困惑度或任务评测，因此没有“无损量化”或“质量不变”的结论。

## 验证

```powershell
ctest --test-dir build-a3b --output-on-failure
.\build-a3b\bin\strata-moe-parity.exe --cpu
```

专家测试比较原始 CPU 与原始 GPU 两套独立参考，覆盖：单 token、多 token、共享输入、逐路由输入、混合命中/未命中、槽位淘汰、不同权重间的缓冲区复用，以及 GPU -> RAM 专家 -> GPU 的调度器传输。
CPU 分组路径与原始 CPU 结果一致；混合路径逐行匹配 CPU 或 GPU 参考（容差 `2e-5 * (1 + abs(reference))`）。
CPU 和 GPU 的量化激活舍入不同，所以混合计算不承诺与全 CPU 或全 GPU 输出逐字节一致。

CUDA 和纯 CPU 构建已在本机验证。HIP 配置因缺少 HIP SDK 停止，SYCL 配置因缺少支持 SYCL 的编译器停止；没有宣称 AMD/Intel 或 Linux 已通过实际推理验证。
详细实测记录见 [docs/A3B-VALIDATION.md](docs/A3B-VALIDATION.md)。
内存修复、MTP 及重复性能对比见 [docs/A3B-OPTIMIZATION.md](docs/A3B-OPTIMIZATION.md)。

## 当前范围

- 支持两款 40 层、2048 隐藏维、256 专家、每 token 8 个路由专家的文本模型；加载前检查几何参数。
- 命令行及原有网页、OpenAI/Anthropic 文本 API 使用同一个常驻引擎。
- 服务支持 temperature、top-p、top-k、min-p、seed 和重复/频率/存在惩罚。
- 一次运行一个模型、一个序列、一个 GPU；每次请求从头计算提示词，专家缓存跨请求保留。
- 尚未接入图像/视频、对话状态缓存、多 GPU、动态显存伸缩、运行中的 STOP、logit_bias 或语法约束。不支持的引擎请求选项会报错。
- 全 GPU 路径使用单一 GPU 图；RAM 专家路径仍采用 ggml 分图调度，完整专家融合可显式启用。这里的 MTP 与缓存优化不等于原引擎的完整复刻。
- 本路径要求单文件 GGUF；不直接读取 safetensors，不默认启用超过原生窗口的 RoPE 扩展。

## 源码与模型依据

- 原 Strata：[docs/DETAILS.md](docs/DETAILS.md)、`include/strata/core/expert_cache.hpp`、`src/core/verify.cpp`、`src/prefill/prefill.cpp`。
- [Qwen3.5 官方配置](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json)、[Qwen3.6 官方配置](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json)。两者使用相同的文本结构。
- [固定版本 llama.cpp 的 A3B 计算图](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/models/qwen35moe.cpp)。A3B 使用普通残差和 SiLU 门控，不能直接复用 Flash-Next 的超连接、PLE、稀疏索引器和 sigmoid 输出门控。
- [所用 GGUF 的固定版本](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/a483e9e6cbd595906af30beda3187c2663a1118c)。
- [独立 MTP GGUF 的固定版本](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/tree/baec3ebee244827cda0f4557eafa8b28f7545fa6)。
