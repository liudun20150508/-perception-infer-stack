# 已知坑与判定方法

完整清单见仓库 `docs/troubleshooting-log.md`，以下是会反复遇到、且**判定成本低**的几条。

## 1. onnxruntime-gpu 的 CUDA 版本错配

**现象**：推理延迟突然变成几百毫秒/图（掉回 CPU），日志里有
`libcublasLt.so.12: cannot open shared object file`。

**判定**：`python3 -c "import onnxruntime as o;print(o.get_available_providers())"` ——
缺 `CUDAExecutionProvider` 即中招。

**处理**：装对应 CUDA 大版本的运行库，并在**进程启动前**设 `LD_LIBRARY_PATH`。
在 Python 里改 `os.environ` 无效。

## 2. 整图量化把分类头打崩

**现象**：INT8 模型 mAP 归零，但框回归看起来正常。

**判定**：比对量化前后的 `max class score` —— 从 0.x 变 0.0 就是它。

**处理**：`op_types_to_quantize=["Conv"]`，别整图量化。

## 3. opset 与 per-channel 不兼容

**现象**：`INVALID_GRAPH: Unrecognized attribute: axis`。

**判定**：模型 opset 12 + `per_channel=True` 必然触发。

**处理**：导出 opset ≥ 13，或 `per_channel=False`。

## 4. 通道序写错，精度悄悄掉

**现象**：mAP 比预期低一截，但没有报错。

**判定**：检查 letterbox 输出是 RGB 还是 BGR —— YOLO 系要 RGB。改回来 mAP 会明显回升。

## 5. 用计时去证明"确定性"结论

**现象**：共享主机上得到反常识的结果（绑定 CPU 更慢、跨 NUMA 更快）。

**判定**：先怀疑**测量代码**，再怀疑被测对象。

**处理**：这类结论改用内核状态判定 —— `Cpus_allowed_list`、`/proc/self/numa_maps` 等。
