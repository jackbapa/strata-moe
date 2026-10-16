# 同一引擎、50K 输入：按需 KV 与固定预留对照

2026-10-16，Windows、Ryzen Threadripper 3960X、128 GiB RAM、RTX 2080 Ti GPU 0（22,528 MiB）。主模型 UD-IQ3_S、匹配的 Q8_0 MTP K2、Strata 分组 IQ 专家核、贪心采样。两组均配置 256K，实际输入严格相同的 50,000 token，输出 128 token；按“按需→固定→按需→固定”运行，每次使用独立进程。引擎 SHA-256 均为 `6b1088ab790058ee27f4b830ce9b95bd42f8de0cef5a2b4430ec2713fb897e34`。

按需组保留发布配置的 `--kv-tiered --kv-tiered-f16-short`，本请求实际分配 64K F16 KV。固定组删除这两个参数、添加 `--kv-q8`，提前分配完整 256K Q8 KV。**同时改变了 KV 容量与格式**，此对照不能拆分各自的影响。主模型量化、MTP、专家内核和其他参数均相同。

| 配置 | 输出两轮 | 输出峰值／中位数 | 首个模型 token 最快／中位数 | 预填充中位数 | 整卡显存峰值 |
| --- | --- | ---: | ---: | ---: | ---: |
| 默认按需 KV | 99.84／100.56 token/s | 100.56／100.20 token/s | 32.512／32.650 s | 1,531.90 token/s | 17,063 MiB |
| 固定预留 256K、Q8 KV | 84.22／84.70 token/s | 84.70／84.46 token/s | 32.929／33.074 s | 1,512.42 token/s | 20,443 MiB |

当前同条件对照中，固定组的输出中位数约低 **15.7%**。四个请求的输出 token 哈希一致；均实际输出 128 token、MTP 接受 84/84、解码调用 Strata 分组专家核 1,804 次。首个模型 token 的计时不含加载，也不是 HTTP 首字节。显存为整卡 200 ms 采样。四轮最高采样 SM 时钟均为 1,350 MHz。

旧矩阵的 50K 单次记录是 121.62 token/s、首个模型 token 25.677 s；**本轮默认配置也未复现该速度**，不能把历史差异全部归因于关闭 KV 分档。旧日志生成于本机 09:39，当前引擎文件的修改时间为 09:57；旧矩阵未保存逐轮二进制 SHA 和时钟遥测，无法建立严格的同版本、同时钟比较。现有证据不足以确定其余差异的原因。这次对照只修改测试脚本和说明，未改推理源码、默认安装配置或本机服务配置，也未重编译引擎。

原始数据与参数：[results.json](results.json)；汇总：[summary.json](summary.json)。复现：

```powershell
python bench/a3b_reserved_context.py --contexts 262144 --input 50000 --output 128 --allocation both --trials 2 --folder bench/results/my-50k-kv-ab
```
