# 感知推理引擎 + 评估系统（能力点「实现 inference engine 并优化」「建立评估系统」）

自研引擎链路 + 三后端对照 + 自研 mAP 评估 + 验收门禁。数据用 COCO128（128 张真图、929 个框）。

```bash
python3 src/engine/run_pipeline.py --dataset data/coco128 --work report/engine --limit 128
```

## 链路

```
图像 --letterbox(自研)--> NCHW --[后端]--> 原始输出 --decode+NMS(自研)--> 检测框
                                   ├ torch-FP32      (PyTorch 直推, baseline)
                                   ├ engine-FP32     (ONNX + onnxruntime CUDA EP)
                                   └ engine-INT8     (Conv-only 静态量化, coco128 校准)
```

三个后端**共用同一套前/后处理**，差异只来自推理后端本身——这才是"引擎"该有的对照口径。

## 实测（L20，见 `report/S3_engine.txt`）

| 后端 | p50 ms | p99 ms | FPS | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| torch-FP32 | 15.09 | 20.33 | 64.4 | 0.6257 | 0.4998 |
| engine-FP32 | 13.60 | 18.92 | 72.6 | 0.6257 | 0.4998 |
| **engine-INT8** | **9.62** | 18.69 | **96.5** | 0.6038 | 0.4656 |

**门禁**：engine-FP32 与 baseline mAP 偏差 `0.0000`（≤0.02）✔；INT8 掉点 `0.0220`（≤0.05）✔；
INT8 相对 FP32 加速 `1.41x`（≥1.10）✔ —— 全过。

## 负结果（诚实的坑，也是排障素材）

1. **整图 INT8 量化把分类头打崩**：`op_types_to_quantize` 不限制时，分类分支 max class score
   从 0.x 直接变 `0.0`，mAP 归零。改成**只量化 Conv** 后恢复到 0.6038。
2. **opset 12 不支持 per-channel**：`per_channel=True` 会生成带 `axis` 属性的
   `DequantizeLinear`，ORT 报 `INVALID_GRAPH`，只能 per-tensor 或把 opset 提到 ≥13。
3. **letterbox 输出 BGR 是错的**：YOLOv8 要 RGB，改回后 mAP50 从 `0.5415` 提到 `0.6257`。
4. **ORT-GPU 版本要对上 CUDA**：1.31 要 CUDA 13、1.20 要 CUDA 12；本机 torch 带的是 CUDA 11，
   故补装 `nvidia-*-cu12` 并在**进程启动前**设 `LD_LIBRARY_PATH`（脚本里用 re-exec 自举）。

## 文件

| 文件 | 作用 |
|---|---|
| `yolo_engine.py` | 自研 letterbox / decode / NMS + 三个推理后端 |
| `coco_eval.py` | 自研 COCO 风格 mAP（101 点插值） |
| `quantize_int8.py` | INT8 静态量化（Conv-only + coco128 校准） |
| `run_pipeline.py` | 编排 + 对照表 + 门禁 + JSON 报告（`work/report.json`） |
