# strata-moe

**在普通电脑上运行 Qwen3.6-35B-A3B MoE。** 本项目从 [Strata](https://github.com/Niko1221/Strata) 的内存与专家调度思路出发，为 Qwen3.5/3.6 A3B 增加了独立的 C++/CUDA/CPU 推理引擎、MTP 验证路径，以及本地网页和 OpenAI/Anthropic 兼容 API。当前一键安装选择已实测的 **Qwen3.6-35B-A3B UD-IQ3_S**；模型由原发布者提供，我们没有重新量化或把权重放进 Git。可[手动下载 IQ3_S 主模型](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-IQ3_S.gguf?download=true)；全 GPU 与混合 MTP 档另需[匹配的 Q8_0 草稿模型](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/resolve/baec3ebee244827cda0f4557eafa8b28f7545fa6/mtp-Qwen3.6-35B-A3B-Q8_0.gguf?download=true)，保存位置和校验值见下文。

> **English:** Strata-Moe runs the Qwen3.6-35B-A3B MoE model locally. `SETUP.bat` installs the selected engine and downloads only the pinned IQ3_S model and, for full-GPU or hybrid mode, its matching MTP draft. Windows/NVIDIA and CPU paths have been tested. See [Install and start](#安装与启动) and [measured results](#实测速度与资源占用).

## 实测速度与资源占用

测试机：Windows、AMD Ryzen Threadripper 3960X（24 核）、128 GiB RAM、NVIDIA RTX 2080 Ti。GPU 0/1 的驱动各报告 **22,528 MiB** 显存；这是本机的容量，不能当作标准 11 GiB 2080 Ti 的成绩。各行用同一个 13.68 GB IQ3_S 主模型；全 GPU 行另用 Q8_0 MTP K2。**配置上下文上限统一为 262,144 token**，实际输入长度单列。全 GPU 的这些请求最多输入 32,768 token、输出 128 token，引擎实际分配的是 **65,536-cell F16 KV 档**；因此表中的约 17.1 GiB 显存峰值不能当作真正填满 256K 时的峰值。各行都是未缓存、贪心采样、128 输出的单次完整请求；输入为重复同一段 token ID 的合成压力测试，因此数字应结合输入长度和内容阅读，不能当成自然文本的通用速度。

| 资源 | 上下文上限 (token) | 实际输入 (token) | 首字节 (s) | 预填充 (token/s) | 输出 (token/s) | 资源采样峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2080 Ti 22 GiB（GPU 0，全驻留 + MTP K2） | 262,144 | 1,024 | 0.598 | 1,722.75 | **137.68** | 整卡 17,084 MiB；引擎 RSS 1,636 MiB |
| 2080 Ti 22 GiB（GPU 0，全驻留 + MTP K2） | 262,144 | 8,192 | 3.319 | 2,471.49 | 137.19 | 整卡 17,102 MiB；引擎 RSS 1,646 MiB |
| 2080 Ti 22 GiB（GPU 0，全驻留 + MTP K2） | 262,144 | 32,768 | 14.941 | 2,194.36 | 126.72 | 整卡 17,102 MiB；引擎 RSS 1,651 MiB |
| 6 GiB 显存模拟（GPU 1，无 MTP） | 262,144 | 1,024 | 6.788 | 150.99 | 14.44 | 整卡 5,639 MiB；引擎 RSS 12,131 MiB |
| 6 GiB 显存模拟（GPU 1，无 MTP） | 262,144 | 8,192 | 44.482 | 184.19 | 27.29 | 整卡 5,639 MiB；引擎 RSS 12,132 MiB |
| 6 GiB 显存模拟（GPU 1，无 MTP） | 262,144 | 32,768 | 160.041 | 204.76 | 27.41 | 整卡 5,639 MiB；引擎 RSS 12,136 MiB |
| 纯 CPU（16 线程，无 MTP） | 262,144 | 1,024 | 12.439 | 82.76 | 12.21 | 引擎 RSS 14,523 MiB |
| 纯 CPU（16 线程，无 MTP） | 262,144 | 8,192 | 106.789 | 76.75 | 10.36 | 引擎 RSS 14,523 MiB |
| 纯 CPU（16 线程，无 MTP） | 262,144 | 32,768 | 605.979 | 54.08 | 8.22 | 引擎 RSS 14,526 MiB |

表中的“首字节”指从向驻留引擎发送请求，到收到首个模型 token 响应行的时间；**它不是 HTTP 响应首字节时间**。预填充、输出速度根据引擎内部计时，模型加载时间不计入。每行只测一次，**没有可报告的中位数**。[统一 256K 测试方法与逐行 JSON](bench/results/2026-10-16-256k-context-matrix/README.md)保留全部原始测量。另测得全 GPU 的 50K/128 合成重复输入：首 token 25.677 秒、预填充 1,947.98 token/s、输出 121.62 token/s，MTP 草稿 84/84 全部被接受；这不能拿来证明相对旧测试的引擎提速。之前另一种合成内容的 API 请求，50K 输入、256 输出、三轮测试：全 GPU 最快首 token **28.094 s**、最快输出 **86.7 token/s**，输出速度中位数 **83.2 token/s**，HTTP 响应首字节中位数 **19.89 ms**；HTTP 首字节只是流式响应开始，不是模型产生 token。[原始报告与逐轮 JSON](bench/results/2026-10-16-a3b-iq3s-combined-exact-r3/README.md)。

另有[16 GiB 引擎预算的混合 + MTP 初步记录](bench/results/2026-10-16-16g-hybrid-mtp-prose/README.md)：同一张 22 GiB 卡上，以 256K 配置上限、1,024/8,192/32,768 输入分别测到 63.50/51.47/46.99 token/s 输出。它使用另一种重复文本，且是 **16 GiB 预算模拟，不是物理 16 GiB 显卡验证**；不能与上表直接比较提速。此轮按要求停止，没有继续做更长输入或实际 256K 填充测试。

6 GiB 是**引擎显存规划预算**，测试用的仍是本机 22 GiB 卡；整卡采样包含空闲基线和驱动分配，**尚未在物理 6 GiB 显卡上验证**。该档位将 GPU 主干与部分常用专家放进显存，其余专家在系统内存，未命中专家由 CPU 计算；它不是所有专家都在 GPU 计算的流水线。≤64K 物理 KV 档留在 GPU，更长的档位转到系统内存。纯 CPU 档位所有模型计算都在 CPU。RSS 为引擎进程采样峰值，不能与整卡显存或整机已用内存相加。此前 2 GiB 预算、8,192 输入的单次测试仍保留：首 token 159.261 秒、预填充 51.47 token/s、输出 7.01 token/s，见[原始 JSON](bench/results/2026-10-16-256k-context-matrix/2g.json)；它与 6 GiB 行不能混算。

## 安装与启动

**Windows（已测）**：下载或克隆仓库后，在项目目录双击 [`SETUP.bat`](SETUP.bat)。它会找 64 位 Python 3.10+；若没有，会先尝试用 winget 安装 Python 3.12。随后创建本目录的 `.venv`，安装 `requirements.txt` 锁定的 Python 包、CMake 和 Ninja；检查 NVIDIA 驱动与 CUDA `nvcc`、Visual Studio C++ 构建工具，缺少开发工具时沿用原 Strata 的安装引导通过 winget 安装；获取固定版本的 llama.cpp/ggml 源码并打上本项目的 A3B 补丁，构建 `build-a3b/bin/strata-moe.exe`，只下载所选模型，导出 tokenizer，写入本机 `strata-moe.json`，最后启动网页/API。首次下载约 13.68 GB；全 GPU 与混合 MTP 档位还需 1.99 GB MTP 模型，预留至少 25 GB 磁盘空间。下载支持续传和 SHA-256 校验；失败后重新运行同一脚本即可。

CUDA 构建需要 [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)（含 `nvcc`）和 [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) 的 **Desktop development with C++**；安装器会先检查，缺少时提示安装（约 8–10 GB，可能需要管理员确认）。只有 NVIDIA 驱动、没有 CUDA 编译工具时可选纯 CPU 安装；CPU 构建仍需 C++ 构建工具。建议至少 32 GiB 系统 RAM；16 GiB 内存机器尚未验证。当前一键 GPU 安装只针对 NVIDIA；AMD/HIP 和 Linux 尚未做本项目安装实测，不在此承诺可用范围内。

```powershell
# 在项目目录执行；安装完即启动，浏览器打开 http://127.0.0.1:8096/
.\SETUP.bat

# 只安装、编译并下载，不启动服务
.\SETUP.bat --no-start

# 后续启动：复用已有 .venv、编译产物、模型和配置，不重新下载
.\START-HERE.bat
```

安装器根据所选卡报告的显存自动选档：至少 20 GiB 用全 GPU + MTP、8 GiB 至不足 20 GiB 用混合专家 + MTP、4 GiB 至不足 8 GiB 用旧版无 MTP 混合档，更低或没有 NVIDIA 卡用纯 CPU。这些是**选档门槛，不是各档在所有输入长度下实测的最低显存**；8 GiB 混合档尚未在物理 8 GiB 显卡上验证。**6 GiB 预算、256K 上限**的测试档位需要显式选择，因为尚未在物理 6 GiB 显卡上验证。适配的 Strata 分组 IQ CUDA 专家核当前只在 GPU 0 的全驻留档位启用；选择其他 GPU 时回退到 GGML 专家路径。标准 11 GiB RTX 2080 Ti 的实际速度尚未测量，安装不等于性能已验证。你也可以明确选择：

不加 `--mode` 时默认是 `auto`，不是固定的混合模式。本测试机的 GPU 0 报告 22,528 MiB，因此自动选择 `full`。两种主要 GPU 模式的区别：

| 模式 | 专家权重放置和计算 | MTP | 默认上下文上限 |
| --- | --- | --- | ---: |
| `full` | 主模型专家全部驻留 GPU，目标模型在 GPU 上计算；不使用 CPU 专家池 | K2 | 262,144 |
| `hybrid` | 部分专家在 GPU 缓存，其余留在 RAM；缓存未命中时由 CPU 专家池计算一部分，另一部分传到 GPU 计算 | K2 | 262,144 |

`hybrid-no-mtp` 保留旧安装档（32K 上限），供不适合加载草稿模型的机器使用；`6g` 是独立的低显存测试档。新 `hybrid` 启用 `--hybrid-experts`、`--cpu-expert-pool` 和 MTP。`--device 0` 指定 0 号显卡，`--no-start` 只安装、不启动服务。

**`full` 的显存与上下文：** 上下文上限是可接收的最大请求长度，实际 KV 缓存会随“输入 + 输出”按 64K→128K→256K 档分配，而主模型和 MTP 权重不会随上限缩短。现有 64K 物理 KV 档日志分别记录目标权重 **13,032.66 MiB**、MTP 权重 **1,372.68 MiB**，仅这两项已约 **14.1 GiB**，还未计入 KV、计算缓冲和驱动开销。该档实测整卡峰值约 **17.1 GiB**（22 GiB 卡、配置上限 256K）。64K 档目标与草稿 KV 合计 **1,408 MiB**；按当前 F16 短档／Q8_0 长档格式，128K 与 256K 的 KV 理论量约为 2,816／2,992 MiB，**后两档没有整卡峰值实测**。因此降低上限可以避免更大 KV 档，却不能把当前 `full` + MTP 压到 10 GiB：单是权重就超过 10 GiB。20 GiB 是安装器当前的选档门槛；“20 GiB 卡可填满 256K”也尚未经过实测验证。较小显存请选 `hybrid`，不要把 `full` 的选档门槛当成不同上下文的最低需求。

下列命令用于强制指定档位：

```powershell
.\SETUP.bat --mode 6g --device 0 --no-start
.\SETUP.bat --mode cpu --no-start
.\SETUP.bat --mode full --device 0 --no-start
.\SETUP.bat --mode hybrid --device 0 --no-start
.\SETUP.bat --mode hybrid-no-mtp --device 0 --no-start
```

切换档位时，先停止当前服务，再加 `--reconfigure` 重新生成 `strata-moe.json`（它不会默默覆盖用户已有配置）。旧版 `hybrid` 配置不会自动改写；升级为混合 MTP 档时运行 `.\SETUP.bat --mode hybrid --device 0 --reconfigure --no-start`，然后 `.\START-HERE.bat`。例如切换到 CPU：`.\SETUP.bat --mode cpu --reconfigure --no-start`。如果想强制重编译，用 `--rebuild`。想预览选档和下载清单，用 `--dry-run`。

服务只监听 `127.0.0.1:8096`。网页地址为 <http://127.0.0.1:8096/>，OpenAI 兼容 API 根地址为 <http://127.0.0.1:8096/v1>；关闭启动脚本窗口或按 `Ctrl+C` 停止。命令行手动启动与一键脚本一致：

```powershell
.\.venv\Scripts\python.exe -m serve.server --engine strata --config strata-moe.json --port 8096 --open
```

若 8096 端口已被占用，请先关闭已有服务；如要改端口，运行 `.\SETUP.bat --port 8097 --reconfigure --no-start`，再运行 `.\START-HERE.bat --port 8097`。把服务绑定到非本机地址前，必须设置 API key；本项目默认只提供本机访问。

## 模型文件与来源

安装器只会从以下固定版本下载所选档位真正需要的文件，下载到被 Git 忽略的 `models/a3b/`。若手动下载，也把文件按表中原名放到该目录，再运行 `SETUP.bat`；安装器会校验已有文件的大小和 SHA-256，正确时直接复用。`6g`、`hybrid-no-mtp` 和 `cpu` 只需第一行；`full` 与 `hybrid` 还需第二行。tokenizer 从主 GGUF 自带的元数据导出，不另下载一个模型。

| 用途 | 上游文件 | SHA-256 |
| --- | --- | --- |
| 推理主模型，所有档位 | [Unsloth Qwen3.6-35B-A3B UD-IQ3_S](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-IQ3_S.gguf?download=true) | `66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66` |
| MTP 草稿，`full` 与 `hybrid` | [ggml-org 匹配的 Q8_0 MTP](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/resolve/baec3ebee244827cda0f4557eafa8b28f7545fa6/mtp-Qwen3.6-35B-A3B-Q8_0.gguf?download=true) | `f6cb995fab78e7284097b2e13508e8daad4ffc6a747b433aef399f4e48524b5a` |

下载实现与固定版本见 [`tools/a3b.py`](tools/a3b.py)。需要 HTTPS 镜像时可在启动前设置环境变量 `HF_ENDPOINT`；下载后的文件仍按固定 SHA-256 校验。主文件内部有不同 GGUF 张量类型，包括 IQ3_S 专家和 Q6_K 密集张量；Q6_K **不是另一个需要下载的主模型**。不要把 `models/`、`.venv/`、`build*/` 或本机 `strata-moe.json` 提交到 Git。

## 项目结构与范围

- `a3b/`：Qwen A3B 引擎、专家缓存/调度、Strata CUDA 适配核和 MTP 验证。
- `serve/`：本地网页和 API；界面名称与标志已更新为 Strata-Moe。
- `tools/setup_moe.py`：独立安装流程；`tools/a3b.py`：固定版本下载、校验与构建。
- `bench/results/`：测量条件、原始结果和限制；[A3B 交接记录](A3B-IQ3S-HANDOFF-2026-10-16.md)列出已实现与未完成的架构工作。

本项目沿用原 [Strata](https://github.com/Niko1221/Strata) 的 MIT 许可证与版权声明，并使用固定版本的 [llama.cpp/ggml](https://github.com/ggml-org/llama.cpp) 源码作为 A3B 构建依赖。现有默认全 GPU 路径在解码调用了适配的原 Strata 分组 IQ 专家核；预填充及其余算子并非全部由 Strata 内核执行。若要做性能比较，请使用相同模型、请求长度、上下文、采样、显存预算，并分别报告 HTTP 首字节、首个模型 token、预填充、输出速度和资源峰值。
