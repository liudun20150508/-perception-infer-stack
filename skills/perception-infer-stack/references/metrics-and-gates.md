# 指标定义与判据规则

## 指标

| 指标 | 定义 | 位置 |
|---|---|---|
| `p50_ms` / `p99_ms` | 端到端单图延迟中位数 / 尾部 | `report/engine/report.json` |
| `fps` | 1000 / 平均延迟 | 同上 |
| `mAP50` / `mAP50-95` | COCO 口径，101 点插值 | `src/engine/coco_eval.py` |
| 等效带宽 | 算子读写字节 / 耗时 | `src/kernels/perception_ops` |
| 峰值显存 | `torch.cuda.max_memory_allocated` | `src/engine/run_pipeline.py` |

**计时一律取多轮最小值**：共享主机上单次测量不可信（邻居会偷 CPU）。

## 判据（`eval/gates.yaml`）

判据算子语义：

| op | 含义 |
|---|---|
| `abs_diff_le` | `abs(left - right) <= value` |
| `diff_le` | `left - right <= value`（用于"掉点不超过"） |
| `ratio_ge` | `left / right >= value`（用于"加速比不低于"） |
| `le` / `ge` | 单值比较 |
| `has_keys` | `left` 是字典且包含 `value` 列出的键 |

`left` / `right` 是相对报告根的点分路径，例如 `backends.engine-INT8.p50_ms`。

## 加一条判据

```yaml
  - id: my-new-gate
    desc: 说明这条判据在防什么
    left: backends.engine-INT8.fps
    op: ge
    value: 80
```

加完直接 `python3 eval/run_gates.py` 验证；不需要改任何代码。

## 判据设计原则

- **每条判据都要能防止一种具体的坏结果**（例如：少跑一个后端、量化掉点失控、导出改变精度）。
- 阈值要有依据（车端单帧预算、精度预算），不要拍脑袋。
- 判据失败必须让人知道"该回看哪一段"，因此 `id` 与 `desc` 要能自解释。
