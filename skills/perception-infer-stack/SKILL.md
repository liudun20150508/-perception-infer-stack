---
name: perception-infer-stack
description: Build, run, or review a perception inference stack — CUDA pre/post-processing ops, ONNX Runtime / INT8 engine work, latency-and-mAP evaluation, and acceptance gating. Use when the task involves this repo's pipeline (kernel → engine → evaluation → gates) or when a perception/AD inference stack needs its numbers reproduced or defended. Not for generic CUDA or Python questions unrelated to this pipeline.
metadata:
  short-description: Perception inference stack (kernels → engine → eval → gates)
---

# Perception Infer Stack

一个可复现的感知推理基础设施：CUDA 前后处理算子 → 推理引擎（torch / ONNX-ORT / INT8）→
评估与门禁 → 计算框架（DAG 调度 / 服务 / 多路）。本 skill 让 agent 按**同一套口径**做这件事，
而不是各自发明测法。

## 先定位仓库

仓库根目录 = 含 `scripts/run_all.sh` 的目录。若不确定，直接问用户，别猜。

**仓库不在手边时**：不要凭空复述本 skill 里的数字，先让用户提供仓库或明确说"只要方法论"。

## 工作流

1. **环境**：`bash scripts/env_check.sh` — 缺 `nvcc` 时算子段会跳过，别把它当失败。
2. **取数**：`bash scripts/fetch_data.sh` — 拉公开的 COCO128 与 YOLOv8n 权重（不进仓库）。
3. **跑**：`bash scripts/run_all.sh`（全量）或 `bash scripts/run_all.sh 1 3`（指定段）。
   逐段说明见 [references/pipelines.md](references/pipelines.md)。
4. **判**：`python3 eval/run_gates.py` — 判据定义在 `eval/gates.yaml`，改口径只改 yaml。
5. **报**：指标与门禁状态落在 `report/SUMMARY.md` 与 `report/DASHBOARD.md`。

## 必须守住的五条口径

1. **性能与精度同表**。只给加速比 = 可能把"模型被量化打崩"讲成优化成果。任何后端对比都要
   `p50 / p99 / FPS / mAP50 / mAP50-95` 一起给。
2. **多个后端必须共用同一套前/后处理**。否则比不出后端差异，只能比出测量差异。
3. **CUDA 编译自动探测架构**（读 `nvidia-smi` 的 compute_cap）。写死 `-arch=sm_80` 换机即挂。
4. **`LD_LIBRARY_PATH` 必须在进程启动前设置**。Python 里改 `os.environ` 对已启动的加载器无效；
   参见 `src/engine/run_pipeline.py` 的自举写法。
5. **负结果照报**。跑不通、更慢、掉点超预算，一律如实写进 `docs/limitations.md`，不要从结论里删掉。

## 常见任务的落点

| 任务 | 从哪开始 |
|---|---|
| 加/改一个 CUDA 算子 | `src/kernels/`，照 `perception_ops/` 的结构（CPU 参考 + 自检 + 带宽账） |
| 换检测模型或后端 | `src/engine/yolo_engine.py`（保持 `[(cls,score,x1,y1,x2,y2)]` 契约） |
| 调验收标准 | 只改 `eval/gates.yaml`，不要改代码 |
| 定位变慢 | 先看 `report/S3_engine.txt` 的分阶段耗时，再看 `docs/troubleshooting-log.md` |
| 量化掉点 | 先确认只量化了 `Conv`（见 `references/pitfalls.md`） |

## 详细资料（按需读取）

- [references/pipelines.md](references/pipelines.md) — 每段跑什么、产出什么、怎么单独跑。
- [references/metrics-and-gates.md](references/metrics-and-gates.md) — 指标定义、判据算子语义、怎么加一条判据。
- [references/pitfalls.md](references/pitfalls.md) — 环境与量化的已知坑，附判定方法。
