# 部署文档

## 1. 环境要求

| 项 | 要求 | 说明 |
|---|---|---|
| OS | Linux x86_64 | 已在 CentOS 7 / 内核 3.10 上验证 |
| GPU | NVIDIA，算力 ≥ 7.5 | 已在 L20（sm_89）验证 |
| 驱动 | ≥ 525 | 需支持 CUDA 12.x 运行时 |
| CUDA Toolkit | ≥ 11.8（提供 `nvcc`） | 算子编译用 |
| C++ 编译器 | `g++` ≥ 9（支持 C++17） | 另需 `cmake` ≥ 3.16（服务框架用） |
| Python | ≥ 3.10 | 引擎/评估/训练用 |

## 2. 一步一步装

```bash
# 1) 取仓库
git clone <repo-url> perception-infer-stack && cd perception-infer-stack

# 2) 自检（会告诉你缺什么）
bash scripts/env_check.sh

# 3) Python 依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install onnx onnxruntime-gpu ultralytics pillow numpy

# 4) 取公开数据与权重
bash scripts/fetch_data.sh

# 5) 全量跑
bash scripts/run_all.sh
```

## 3. 三个已知的部署坑（都踩过，见 `docs/troubleshooting-log.md`）

### 3.1 onnxruntime-gpu 的版本必须和 CUDA 对上

ORT-GPU 每个大版本绑死一个 CUDA 大版本：`1.31 → CUDA 13`、`1.20 → CUDA 12`、`1.17 → CUDA 11`。
如果本机 `torch` 是 cu118（只有 `libcublas.so.11`），装 `onnxruntime-gpu==1.20.2` 会在创建
CUDA EP 时报 `libcublasLt.so.12: cannot open shared object file`，静默退化成 CPU
（表现为单图几百毫秒）。

解决：补装 CUDA 12 的运行库，并**在进程启动前**把路径加进 `LD_LIBRARY_PATH`：

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-runtime-cu12
export LD_LIBRARY_PATH="$(ls -d $CONDA_PREFIX/lib/python3.*/site-packages/nvidia/*/lib | tr '\n' ':')$LD_LIBRARY_PATH"
```

> `os.environ` 在 Python 进程里改 `LD_LIBRARY_PATH` **无效**（动态链接器在进程启动时就读过了）。
> `src/engine/run_pipeline.py` 因此做了自举：检测到缺库就带新环境 `execve` 自己一次。

### 3.2 量化版本要对齐 opset

`per_channel=True` 会在 `DequantizeLinear` 上生成 `axis` 属性，opset 12 的模型会报
`INVALID_GRAPH: Unrecognized attribute: axis`。两条路：把导出 opset 提到 ≥ 13，
或 `per_channel=False`（本仓库默认后者，追求可复现）。

### 3.3 别把量化范围放到整图

`op_types_to_quantize` 不限制时，YOLO 的分类分支会被打崩（max class score → 0，mAP 归零）。
本仓库只量化 `Conv`，见 `src/engine/quantize_int8.py`。

## 4. 无 GPU 机器上的降级

```bash
bash scripts/run_all.sh 5 6 7     # 只跑不需要 GPU 的段：训练(CPU)、OS、图形、重构、规则引擎
```

没有 `nvcc` 时，算子段会自动跳过并给出明确提示，不会假装成功。

## 5. 多机 / 集群

- 每段都通过 `scripts/run_all.sh <编号>` 可单独执行，适合拆到不同节点。
- 所有脚本用**脚本自身路径**定位资源（`cd "$(dirname "$0")"`），不依赖当前工作目录。
- 编译产物落在 `bin/`，`.gitignore` 已忽略，不会污染工作树。

## 6. 接入 CI

```bash
bash scripts/run_all.sh          # 失败段以非 0 退出码体现在 report/SUMMARY.md
python3 eval/run_gates.py        # 规则引擎：判据不满足直接非 0
python3 eval/interface_contract.py   # 对外接口契约
```
