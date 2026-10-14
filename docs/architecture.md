# 架构与设计决策

## 1. 分层

```
应用/服务层     src/framework/serving        推理服务：排队、背压、热更新、看门狗
                src/framework/multi_stream_  多路视频：单实例 + 动态组批
调度层          src/framework/dag_executor   算子 DAG：依赖调度、线程池、内存池
引擎层          src/engine                   前后处理 + 后端（torch / ONNX-ORT / INT8）
算子层          src/kernels                  手写 CUDA：预处理、NMS、量化 GEMM/反量化、分块 GEMM
评估层          eval/ + src/engine/coco_eval 指标采集 + mAP + 门禁规则
```

分层的关键目的：**评估层与实现层解耦**。同一套前/后处理、同一套指标口径，换任何后端都能横比，
否则"优化了"永远说不清是后端变快还是测量口径变了。

## 2. 四个关键决策

### D1. 三个后端共用同一套前/后处理

`src/engine/yolo_engine.py` 里 letterbox / decode / NMS 全部自研，torch、ONNX-FP32、ONNX-INT8
三个后端只用同一个 `run_one()`。这样 p50/p99 的差异**只可能来自后端本身**。

代价：放弃了 ultralytics 里成熟的后处理（含各种边界情况）。收益：口径干净、可解释。

### D2. 精度必须和性能一起报

只报加速比是危险的 —— INT8 实测可以把 YOLO 的分类头整片打崩（mAP 归零）。所以引擎段的输出
永远是一张**联表**：`p50 / p99 / FPS / mAP50 / mAP50-95`，由 `eval/run_gates.py` 卡三条判据。

### D3. 调度与内存分开治

`dag_executor` 里，"什么时候能跑"由依赖计数（入度归零）决定，"跑的时候占多少内存"由内存池
按**最大并发节点数**预分配决定。两件事不耦合，所以改图结构不用改内存策略，反之亦然。

### D4. 结论必须可被第三方复现

- 所有 CUDA 编译**自动探测架构**（读 `nvidia-smi` 的 compute_cap），不写死 `-arch=sm_80`；
- 数据/权重用 `scripts/fetch_data.sh` 拉公开源，不塞进仓库；
- 对外接口写成 `eval/interface_contract.py` 可机器校验；
- 没跑通的（见 `docs/limitations.md`）如实登记，不删。

## 3. 一次请求的完整路径

```
帧到达
  → 前处理：letterbox 缩放 + 归一化（GPU，src/kernels/perception_ops）
  → 批调度：凑批窗口 / 优先级（src/framework/serving）
  → 推理：引擎后端（src/engine）
  → 后处理：decode + NMS（GPU）
  → 指标：各阶段耗时、队列深度、显存水位
  → 门禁：eval/run_gates.py 判定是否达标
```

每个阶段的耗时都被单独记录，所以"端到端变慢"能立刻定位到是哪一段。

## 4. 扩展点

| 想换什么 | 改哪里 |
|---|---|
| 换检测模型 | `src/engine/yolo_engine.py` 的后处理 decode（保持 `[(cls,score,x1,y1,x2,y2)]` 契约） |
| 换推理后端 | 实现一个带 `infer(nchw) -> np.ndarray` 的类，其余不动 |
| 加一个算子到图里 | `src/framework/dag_executor/dag_exec.cpp` 的 `make_graph()` |
| 改验收标准 | `eval/gates.yaml`，不用改代码 |
