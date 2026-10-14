#!/usr/bin/env bash
# 拉取 onnxruntime 的 C++ 发行包（pip 装的 python wheel 里**不含** C++ 头文件）。
# 打印两个路径：<include_dir> <lib_file>，供 cmake 用。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VER="${ORT_VER:-1.17.3}"
# 注意：官方 tar 包内的顶层目录就是 onnxruntime-linux-x64-<ver>/，DEST 必须与之一致
DEST="$ROOT/report/onnxruntime-linux-x64-$VER"

if [ ! -f "$DEST/lib/libonnxruntime.so" ]; then
  mkdir -p "$ROOT/report"
  echo "== 下载 onnxruntime C++ $VER" >&2
  python3 - "$VER" "$ROOT/report" <<'PY'
import os, shutil, sys, tarfile, urllib.request
ver, dest = sys.argv[1], sys.argv[2]
name = f"onnxruntime-linux-x64-{ver}"
url = f"https://github.com/microsoft/onnxruntime/releases/download/v{ver}/{name}.tgz"
tgz = os.path.join(dest, f"{name}.tgz")
with urllib.request.urlopen(url) as r, open(tgz, "wb") as f:
    shutil.copyfileobj(r, f)
with tarfile.open(tgz) as t:
    t.extractall(dest)
print("解压完成", file=sys.stderr)
PY
fi

echo "$DEST/include $DEST/lib/libonnxruntime.so"
