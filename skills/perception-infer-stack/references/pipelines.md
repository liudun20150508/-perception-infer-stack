# 各段：跑什么 / 产出什么 / 怎么单独跑

| 段 | 内容 | 单独跑 | 产物 |
|---|---|---|---|
| S1 | 感知前后处理算子（letterbox+归一化、NMS） | `bash scripts/run_all.sh 1` | `report/S1_perception.txt` |
| S2 | INT8 GEMM / FP4 反量化 / SGEMM 分块 | `bash scripts/run_all.sh 2` | `report/S2_kernels.txt` |
| S3 | 推理引擎三后端 + mAP | `bash scripts/run_all.sh 3` | `report/S3_engine.txt`、`report/engine/report.json` |
| S3b | 规则引擎判定 | `python3 eval/run_gates.py` | `report/S3_gates.txt` |
| S4 | DAG 执行器 + 多路视频调度 | `bash scripts/run_all.sh 4` | `report/S4_framework.txt` |
| S4b | 推理服务框架（C++17 + ONNXRuntime） | 同上 | `report/S4b_serving.txt` |
| S5 | 训练闭环 smoke（CPU/GPU 均可） | `bash scripts/run_all.sh 5` | `report/S5_train.txt` |
| S6 | OS/体系结构实验 | `bash scripts/run_all.sh 6` | `report/S6_os.txt` |
| S6b | 图形：零依赖光栅化 | 同上 | `report/S6b_render.txt`、`src/graphics/out/*.png` |
| S7 | 重构验收 + 接口契约 | `bash scripts/run_all.sh 7` | `report/S7_refactor.txt`、`report/S7_contract.txt` |

## 依赖关系

- S3 需要 `data/coco128`（`scripts/fetch_data.sh`）与 `ultralytics` / `onnxruntime-gpu`。
- S4b 需要 `cmake` 与 onnxruntime 的 C++ 头/库；缺任一则自动 SKIP（不是失败）。
- S1/S2/S6b 需要 `nvcc` 或 `g++`；缺编译器则 SKIP。
- S5/S6/S7 无 GPU 也能跑。

## 无 GPU 的最小可跑集

```bash
bash scripts/run_all.sh 5 6 7
```

## 输出约定

- 每段 stdout/stderr 落 `report/S*.txt`，便于事后对照与贴进 issue。
- `report/SUMMARY.md` 是权威结论；`report/DASHBOARD.md` 是给非工程读者的指标入口。
