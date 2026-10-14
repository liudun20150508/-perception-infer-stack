# 调试日志（能力点「快速开发调试」）

下面全是本次把演示台从零跑通过程中**真实踩到并定位的 bug**，按「现象 → 根因 → 修复 → 验证」记录。
这份清单本身就是"快速调试能力"的证据：多数是从一句报错/一个反常识数字反推到根因。

| # | 现象 | 根因 | 修复 | 验证 |
|---|---|---|---|---|
| 1 | 某组 3 个 `.cu` 编译报 5 个错：`no instance of overloaded function cudaEventCreate` | `CUDA_CHECK` 宏在**调用点**展开出 `cudaError_t e`，与 `bench_cuda` 里的 `cudaEvent_t s, e` 同名遮蔽 | 宏内变量改名 `_rc_` | 编译通过并出数（S3） |
| 2 | SGEMM 的 float4 版运行报 `misaligned address` | 向量化路径对 16B 对齐的假设在 sm_89 上不成立 | **未修，如实登记**（列为待办） | 崩溃可复现（S3c） |
| 3 | FlashAttention `err/tol=12.47` 且比 naive 慢 `0.68x` | 在线 softmax 实现有误 | **未修，如实登记** | 对拍不过（S3c） |
| 4 | `make_corpus.py` 换机报 `FileNotFoundError: <硬编码绝对路径>/...` | 硬编码 macOS 绝对路径 | 改为相对脚本位置 | 语料生成成功 |
| 5 | 自研引擎 mAP 只有 0.5415，明显偏低 | letterbox 输出 **BGR**，而 YOLOv8 要 **RGB** | 去掉 `[::-1]` | mAP50 `0.5415 → 0.6257` |
| 6 | INT8 量化后 mAP **归零** | 整图量化把分类头打崩（max class score `0.x → 0.0`） | 只量化 `Conv` | mAP50 恢复到 0.6038 |
| 7 | 量化模型 ORT 加载报 `INVALID_GRAPH: Unrecognized attribute: axis` | opset 12 不支持 per-channel 的 `DequantizeLinear(axis=)` | `per_channel=False` | 模型可加载 |
| 8 | ORT CUDA EP 加载失败，掉回 CPU（691ms/图） | ORT 1.31 要 CUDA 13、1.20 要 CUDA 12，本机 torch 是 CUDA 11 | 装 `nvidia-*-cu12`，**进程启动前**设 `LD_LIBRARY_PATH`（脚本自举 re-exec） | CUDA EP 生效，p50 回到 ~14ms |
| 9 | NUMA demo 结论反了：本地触碰的内存跑到远端 node | `std::vector<char>(n,1)` **构造时就触碰了全部页**，first-touch 发生在主线程 | 改 `malloc` + 绑定线程内触碰 | node0 触碰 → `N0=65536 页`（干净） |
| 10 | 亲和性 demo 打印的 `Cpus_allowed_list` 一直是 `0-87` | 从**主线程**读，而绑定发生在子线程 | 主线程直接绑定后再读 | `0-87 → 0 → 0-87` |
| 11 | 演示台状态列错位、"通过 0/9" | `record()` 调用参数顺序写反（name/status 颠倒） | 统一 `record "S1 名字" OK "备注"` | 计数正确 |
| 12 | 引擎段的 JSON 报告缺失，但该段仍显示 OK | `KeyError: 'mAP50'`（把 detections 当 metrics）；且脚本只 grep 关键字、**没检查退出码** | 用 `rows` 取指标；S12 增加退出码判定 | 报告生成、失败不再被掩盖 |

## 方法论（可复述）

1. **先看退出码，再看输出**：第 12 条就是"grep 成功掩盖了崩溃"的反面教材。
2. **反常识数字先怀疑测量方法**：第 9 条（本地比远端慢）先怀疑基准代码，而不是先下结论。
3. **区分"确定性证据"与"计时证据"**：共享主机上计时不可信，用 `Cpus_allowed_list`、
   `/proc/self/numa_maps` 这类内核状态做判定。
4. **负结果要留下**：第 2、3 条没修，但登记进 `04_方法与判据/负结果清单`，比假装通过更有价值。
