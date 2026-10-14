#!/usr/bin/env bash
# 环境自检：告诉你缺什么、怎么补。不会改任何东西。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ok()   { printf '  [ OK ] %s\n' "$1"; }
miss() { printf '  [MISS] %s  ->  %s\n' "$1" "$2"; }

echo "=== 环境自检 ==="

echo "-- 工具链 --"
command -v g++     >/dev/null && ok "g++ $(g++ -dumpversion)"        || miss "g++" "yum/apt install gcc-c++"
command -v cmake   >/dev/null && ok "cmake $(cmake --version | head -1 | awk '{print $3}')" || miss "cmake" "pip install cmake"
command -v nvcc    >/dev/null && ok "nvcc $(nvcc --version | grep -o 'release [0-9.]*')" || miss "nvcc" "安装 CUDA Toolkit（算子段会跳过）"

echo "-- GPU --"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | sed 's/^/  [ OK ] GPU /'
else
  miss "nvidia-smi" "无 GPU 时只能跑：bash scripts/run_all.sh 5 6 7"
fi

echo "-- Python --"
python3 - <<'PY'
import importlib, sys
mods = [("torch","torch"), ("torchvision","torchvision"), ("onnx","onnx"),
        ("onnxruntime","onnxruntime-gpu"), ("ultralytics","ultralytics"),
        ("PIL","pillow"), ("numpy","numpy"), ("yaml","pyyaml")]
miss = []
for mod, pkg in mods:
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        print(f"  [ OK ] {mod} {v}")
    except Exception:
        print(f"  [MISS] {mod}  ->  pip install {pkg}")
        miss.append(pkg)
try:
    import onnxruntime as ort
    p = ort.get_available_providers()
    print(f"  [INFO] onnxruntime providers: {p}")
    if "CUDAExecutionProvider" not in p:
        print("  [WARN] 没有 CUDA EP，推理引擎会退化到 CPU（延迟会差很多）")
except Exception:
    pass
if miss:
    print("\n  缺失依赖，建议： pip install " + " ".join(sorted(set(miss))))
PY

echo
echo "自检结束。下一步： bash scripts/fetch_data.sh && bash scripts/run_all.sh"
