# perception-infer-stack

面向**自动驾驶/车端感知**的推理基础设施参考实现：把「感知前后处理算子 → 推理引擎 →
评估系统 → 计算框架 → 端到端评估」做成一条**可一键复现**的链路，并且能作为
**Codex Skill / 知识库 / 规则引擎 / MCP 服务**被其它 agent 直接加载使用。

> 定位：不是论文复现，也不是产品级框架。目标是把感知 infra 的关键能力做成
> **有数字、可复现、可交接**的最小完整实现 —— 每个结论都能在你自己的机器上重跑出来。

---

## 1. 能力矩阵（每条都对应一个可运行的验证）

| 能力 | 实现在哪 | 验证方式 | 实测结果（NVIDIA L20 / CUDA 11.8） |
|---|---|---|---|
| 感知前后处理算子 | `src/kernels/perception_ops` | `bash scripts/run_all.sh 1` | letterbox vs CPU `max\|diff\|=5.96e-8`、快 1176x；NMS 3000 框逐框一致 |
| 量化推理算子 | `src/kernels/{int8_gemm,fp4_dequant}` | `bash scripts/run_all.sh 2` | cuBLAS INT8 `0.165ms`；FP4 反量化 `GPU==CPU 全一致` |
| 手写 CUDA 优化 | `src/kernels/sgemm_attention`（另有 `sparse24/` 源码，需 cuSPARSELt，未纳入一键跑） | `bash scripts/run_all.sh 2` | SGEMM 分块提速；分块/向量化/在线 softmax 三种手法 |
| 推理引擎（自研链路） | `src/engine` | `bash scripts/run_all.sh 3` | torch→ONNX/ORT→INT8：p50 `15.1→14.0→9.6ms`，mAP50 `0.6257→0.6257→0.6038` |
| 评估系统 + 验收门禁 | `src/engine/coco_eval.py`、`eval/` | `bash scripts/run_all.sh 3` | 自研 COCO mAP（101 点插值）+ 三条判据自动 PASS/FAIL |
| 计算框架（DAG 调度） | `src/framework/dag_executor` | `bash scripts/run_all.sh 4` | 依赖调度并行化，内存池 0 未命中 |
| 推理服务框架 | `src/framework/serving` | `bash scripts/run_all.sh 4` | 三段流水线 + 背压 + 零拷贝 + 热更新 + 看门狗 |
| 多路视频调度 | `src/framework/multi_stream_scheduler` | `bash scripts/run_all.sh 4` | 16 路显存 `7.66GB → 0.75GB` |
| 训练闭环 | `src/training/train_smoke.py` | `bash scripts/run_all.sh 5` | 手写 MLA+MoE 架构，GPU 上 loss `7.83 → 0.12` |
| OS / 体系结构 | `src/os` | `bash scripts/run_all.sh 6` | CPU 亲和性（`Cpus_allowed_list`）、伪共享、NUMA first-touch、上下文切换 |
| 图形渲染 | `src/graphics` | `bash scripts/run_all.sh 6` | 零依赖 CPU 光栅化，无头渲染 51928 面网格 |
| 代码重构 | `src/refactor` | `bash scripts/run_all.sh 7` | 重构前后 stdout **逐字节一致** |

---

## 2. 架构

```
                      ┌────────────── 评估与门禁 (eval/) ──────────────┐
                      │  gates.yaml (判据)  →  run_gates.py  →  PASS/FAIL │
                      └────────────────────────┬───────────────────────┘
                                               │ 采集
  图像/传感器帧
      │
      ▼
┌───────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ 前后处理算子   │   │ 推理引擎 src/engine   │   │ 计算框架             │
│ letterbox/NMS │──▶│ torch / ONNX(ORT) /   │──▶│ dag_executor (依赖    │
│ (CUDA)        │   │ INT8 三个后端同链路    │   │ 调度+内存池)         │
└───────────────┘   └──────────────────────┘   │ serving (服务框架)   │
                                                │ multi_stream (多路)  │
                                                └──────────────────────┘
                                               │
                                               ▼
                                     report/ (指标 + 看板 + JSON)
```

设计取舍与关键决策见 [docs/architecture.md](docs/architecture.md)。

---

## 3. 快速开始

```bash
# 环境自检（编译器 / CUDA / Python 依赖）
bash scripts/env_check.sh

# 拉取公开数据与权重（COCO128 样例 + YOLOv8n 权重）
bash scripts/fetch_data.sh

# 全量跑（约 8 分钟），结果写入 report/
bash scripts/run_all.sh

# 只跑某几段
bash scripts/run_all.sh 1 3
```

依赖：Linux + NVIDIA GPU（CUDA ≥ 11.8）、`g++`/`cmake`、Python ≥ 3.10。
Python 侧依赖见 `scripts/env_check.sh` 的输出提示（`torch` / `onnxruntime-gpu` / `ultralytics`）。

部署细节（多机、无 GPU 降级、离线安装）见 [docs/deployment.md](docs/deployment.md)。

---

## 4. 作为 Skill / 知识库 / 规则引擎 / MCP 使用

这个仓库不只是给人看的，也是给 **agent** 用的。四种消费方式：

| 方式 | 入口 | 用途 |
|---|---|---|
| **Codex Skill** | [`skills/perception-infer-stack/SKILL.md`](skills/perception-infer-stack/SKILL.md) | 让 Codex 按本仓库的口径做感知推理工程：选后端、量化、评测、门禁 |
| **知识库** | [`docs/`](docs/) + [`docs/troubleshooting-log.md`](docs/troubleshooting-log.md) | 架构/部署/评测口径/踩坑记录，供检索与引用 |
| **规则引擎** | [`eval/gates.yaml`](eval/gates.yaml) + [`eval/run_gates.py`](eval/run_gates.py) | 把「什么算达标」写成机器可判的规则，CI 里直接卡 |
| **MCP 服务** | [`mcp/server.py`](mcp/server.py) | 把「跑评测、查指标、校契约」暴露成工具给任意 MCP 客户端 |

把 skill 装进 Codex：

```bash
cp -r skills/perception-infer-stack "${CODEX_HOME:-$HOME/.codex}/skills/"
# 之后在 Codex 里按名字调用该 skill
```

---

## 5. 目录结构

```
perception-infer-stack/
├── src/
│   ├── kernels/             # CUDA 算子（感知前后处理 / 量化 / 手写优化）
│   ├── engine/              # 推理引擎链路 + 自研 mAP 评估器
│   ├── framework/           # 计算框架：DAG 执行器 / 服务框架 / 多路调度
│   ├── training/            # 训练闭环 smoke
│   ├── os/                  # OS 与体系结构实验
│   ├── graphics/            # 零依赖软件光栅化器
│   └── refactor/            # 重构对照（行为不变验收）
├── eval/                    # 规则引擎：判据 + 契约校验 + 看板聚合
├── docs/                    # 架构 / 部署 / 评测 / 排障日志 / 限制
├── scripts/                 # 一键脚本：自检 / 取数 / 全量跑
├── skills/                  # Codex skill 包
├── mcp/                     # MCP 服务
├── data/samples/            # 小样例数据（4 张 COCO 图 + 标注）
└── report/                  # 运行产物（指标 / 看板 / 日志）
```

---

## 6. 已知限制

写在 [docs/limitations.md](docs/limitations.md)，包括：未接入 TensorRT、评测数据集为公开小样本、
部分算子仅在单卡验证、图形分支是 CPU 光栅化（非 GPU 图形）等。**宁可能力边界清楚，也不要含糊的"全都支持"。**

## 7. 许可

MIT，见 [LICENSE](LICENSE)。样例图片与模型权重来自公开数据集，遵循其原始许可。
