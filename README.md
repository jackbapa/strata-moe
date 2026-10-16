# strata-moe

用一张 NVIDIA 显卡和系统内存，在本机运行 **Qwen3.6-35B-A3B**。默认安装会自动读取显卡显存并选择配置；下载的主模型始终是同一个 **Unsloth UD-IQ3_S GGUF**。项目借鉴 [Strata](https://github.com/Niko1221/Strata) 的专家调度，并为 A3B 接入了部分 Strata CUDA 专家核、混合专家路径和 MTP。

> **English:** Run Qwen3.6-35B-A3B locally. On Windows, run `SETUP.bat`; its default `auto` setting chooses a GPU/RAM configuration from detected VRAM. The target is the same IQ3_S GGUF in every configuration. MTP-enabled configurations also download a separate matching Q8_0 draft. [Model links](#模型下载) · [Measurements](#measurements)

## 显存、预留上下文和速度

直接运行 `SETUP.bat` 即使用 **auto**。想按显存选择预留上下文，可参考这一张表：

| 预留上下文 | 建议显存 | 输出速度参考 |
| --- | --- | --- |
| **32K** | **20 GiB+** | 32K 输入：**127 token/s**（22 GiB 卡实测） |
| **64K** | **20 GiB+** | 50K 输入：**122 token/s**（22 GiB 卡实测） |
| **128K** | **22 GiB+** | 长输入未测 |
| **256K** | **22～24 GiB** | 长输入未测 |
| **32K，混合专家** | **16 GiB 预算** | 32K 输入：**47 token/s**（预算模拟） |
| **32K，低显存** | **6 GiB 预算** | 32K 输入：**27 token/s**（预算模拟） |
| **32K，纯 CPU** | **无需显卡** | 32K 输入：**8 token/s**（16 线程实测） |

速度来自预留 256K 的现有测试；16 GiB 和 6 GiB 是在 22 GiB 卡上的预算模拟。`auto` 默认按显存选配置，预留上下文不会按表自动调整；需要时按[安装说明](#安装与启动)修改。

<a id="measurements"></a>
## 🧪 实测速度：预留上下文均为 256K

下表的 **256K 是预留上下文，不是每次实际输入了 256K**。输入长度单列；输出请求为 128 token。首个 token 等待时间从向已经加载好的引擎发出请求算起，不含模型加载，也不是 HTTP 响应首字节。预填充和输出速度按引擎计时。每行展示单次请求（16 GiB／1K 取第二轮），表中的速度**不是中位数**。

| 测试资源 | 预留上下文 | 实际输入 | 首个 token 等待 | 预填充 | 输出 | 资源峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2080 Ti 22 GiB | 256K | 1K | 0.598 s | 1,723 token/s | **137.68 token/s** | 整卡 17,084 MiB |
| 2080 Ti 22 GiB | 256K | 8K | 3.319 s | 2,471 token/s | 137.19 token/s | 整卡 17,102 MiB |
| 2080 Ti 22 GiB | 256K | 32K | 14.941 s | 2,194 token/s | 126.72 token/s | 整卡 17,102 MiB |
| 2080 Ti 22 GiB | 256K | 50K | 25.677 s | 1,948 token/s | 121.62 token/s | 整卡 17,102 MiB |
| 16 GiB 引擎预算模拟 | 256K | 1K | 4.143 s | 247 token/s | 63.50 token/s | 整卡 16,483 MiB；进程 RAM 12,944 MiB |
| 16 GiB 引擎预算模拟 | 256K | 8K | 18.56 s | 441 token/s | 51.47 token/s | 整卡 16,859 MiB；进程 RAM 13,213 MiB |
| 16 GiB 引擎预算模拟 | 256K | 32K | 71.16 s | 461 token/s | 46.99 token/s | 整卡 16,869 MiB；进程 RAM 13,218 MiB |
| 6 GiB 引擎预算模拟 | 256K | 8K | 44.482 s | 184 token/s | 27.29 token/s | 整卡 5,639 MiB；进程 RAM 12,132 MiB |
| 6 GiB 引擎预算模拟 | 256K | 32K | 160.041 s | 205 token/s | 27.41 token/s | 整卡 5,639 MiB；进程 RAM 12,136 MiB |
| 纯 CPU，16 线程 | 256K | 8K | 106.789 s | 77 token/s | 10.36 token/s | 进程 RAM 14,523 MiB |
| 纯 CPU，16 线程 | 256K | 32K | 605.979 s | 54 token/s | 8.22 token/s | 进程 RAM 14,526 MiB |

22 GiB、6 GiB 预算及 CPU 行使用同一段短 token 序列重复填充；16 GiB 行使用另一份文本重复填充，**不能把两组速度直接相减当作配置收益**。6 GiB 行是显式选择的独立低显存测试配置，不代表 4～8 GiB 的 auto 配置。16 GiB 行启用了 MTP，6 GiB 与 CPU 行没有启用。整卡采样包含驱动与其他显存占用；进程 RAM 与整卡显存不能相加。原始条件与 JSON：[22 GiB／6 GiB／CPU 矩阵](bench/results/2026-10-16-256k-context-matrix/README.md)、[16 GiB 混合 + MTP 记录](bench/results/2026-10-16-16g-hybrid-mtp-prose/README.md)。另一次 [50K 输入、256 输出、三轮 API 测试](bench/results/2026-10-16-a3b-iq3s-combined-exact-r3/README.md)采用不同内容，不与上表合并计算。

## 安装与启动

**Windows + NVIDIA 已测试。** 下载或克隆仓库，在项目目录双击 [`SETUP.bat`](SETUP.bat)，或在 PowerShell 运行：

```powershell
.\SETUP.bat
```

保持默认 **auto** 即可。首次运行会准备 Python 虚拟环境、依赖和 CUDA/C++ 构建工具，构建本项目引擎，下载并校验所需模型，随后启动本地网页。之后双击 [`START-HERE.bat`](START-HERE.bat) 复用已有安装。网页是 <http://127.0.0.1:8096/>；OpenAI 兼容 API 是 <http://127.0.0.1:8096/v1>。关闭窗口或按 `Ctrl+C` 停止服务。只安装不启动可运行 `.\SETUP.bat --no-start`；查看自动选档而不下载可运行 `.\SETUP.bat --dry-run`。

安装需要 64 位 Python 3.10+、[CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)（含 `nvcc`）和 [Visual Studio 2022 C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)；脚本会检查并引导安装缺失项。建议至少 32 GiB 系统内存、25 GB 空闲磁盘。仅有 CPU 时仍需 C++ 构建工具。当前一键 GPU 安装面向 Windows/NVIDIA；本项目的 AMD/HIP 与 Linux 安装路径尚未验证。

需要手动调整预留上下文时，先停止服务，编辑安装器生成的 `strata-moe.json`：在 `args` 中把 `--ctx` 后的值改为所需 token 数（32K = `32768`、64K = `65536`、128K = `131072`、256K = `262144`）。设为 32K 或 64K 时，还要从 `args` 删除 `--kv-tiered` 和 `--kv-tiered-f16-short`；引擎不允许这两个选项用于 64K 及以下。保存后运行 `START-HERE.bat`，**不需要重新编译**。输入与输出之和必须留在预留上下文内。再次运行带 `--reconfigure` 的安装器会重写这个本机配置，请先保留自己的修改。

## 模型下载

所有资源档使用**同一个** Qwen3.6-35B-A3B **UD-IQ3_S 主模型**。显存充足时，MTP 会额外加载匹配的 **Q8_0 草稿模型**来提出候选 token，再由 IQ3_S 主模型验证；Q8_0 **没有替换主模型，也不是第二套运行方案**。安装器只下载当前配置需要的文件，模型权重不进入 Git。

| 文件 | 手动下载 | 用途 |
| --- | --- | --- |
| `Qwen3.6-35B-A3B-UD-IQ3_S.gguf`，约 13.68 GB | [Unsloth 固定版本](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-IQ3_S.gguf?download=true) | 所有配置的推理主模型 |
| `mtp-Qwen3.6-35B-A3B-Q8_0.gguf`，约 1.99 GB | [ggml-org 固定版本](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/resolve/baec3ebee244827cda0f4557eafa8b28f7545fa6/mtp-Qwen3.6-35B-A3B-Q8_0.gguf?download=true) | 启用 MTP 时的草稿模型 |

手动下载后，将文件按原名放入 `models/a3b/`，再运行 `SETUP.bat`；脚本会核对大小和 SHA-256，正确时复用。固定版本与校验值见 [`tools/a3b.py`](tools/a3b.py)。需要镜像可设置 `HF_ENDPOINT`。主模型内部本来就含 IQ3_S 专家和 Q6_K 等其他类型张量；它们不是要分别下载的模型。

## 可选配置与项目范围

需要指定显卡可运行 `.\SETUP.bat --device 0`。高级用户可用 `--mode full` 强制 GPU 专家全驻留，或用 `--mode hybrid` 强制 GPU/RAM 专家混合；两者都会开启 MTP。`--mode hybrid-no-mtp` 是旧的 32K 配置；`--mode 6g` 是与上表对应的显式 6 GiB **预算模拟**配置，尚未在物理 6 GiB 卡上验证；`--mode cpu` 使用 CPU。已有配置不会被静默覆盖；切换配置时先停止服务，再运行 `SETUP.bat --mode <名称> --reconfigure --no-start`，之后运行 `START-HERE.bat`。`--rebuild` 才会强制重新编译。服务默认仅监听 `127.0.0.1`；绑定到其他地址前须设置 API key。

本项目的 A3B 引擎在 [`a3b/`](a3b/)；网页和 API 在 [`serve/`](serve/)；构建与下载入口在 [`tools/`](tools/)。部分解码专家计算使用了适配的 Strata CUDA 核，其余路径仍包含 GGML 实现；[技术交接记录](A3B-IQ3S-HANDOFF-2026-10-16.md)说明已完成和待办范围。项目沿用 [Strata](https://github.com/Niko1221/Strata) 的 MIT 许可证，并固定依赖 [llama.cpp/ggml](https://github.com/ggml-org/llama.cpp) 源码。模型、虚拟环境、构建产物和本机配置均被 Git 忽略。
