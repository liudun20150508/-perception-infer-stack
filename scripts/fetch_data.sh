#!/usr/bin/env bash
# 拉取公开数据与权重（不随仓库分发）：COCO128 小样本 + YOLOv8n 权重
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p data report

COCO_DIR="data/coco128"
if [ -d "$COCO_DIR/images" ]; then
  echo "已存在: $COCO_DIR（跳过）"
else
  echo "== 下载 COCO128（128 张图 + 标注，约 7MB）"
  python3 - <<'PY'
import os, shutil, urllib.request, zipfile
url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
os.makedirs("data", exist_ok=True)
with urllib.request.urlopen(url) as r, open("data/coco128.zip", "wb") as f:
    shutil.copyfileobj(r, f)
with zipfile.ZipFile("data/coco128.zip") as z:      # 压缩包内含顶层 coco128/
    z.extractall("data")
assert os.path.isdir("data/coco128/images"), "解压结果异常"
print("COCO128 就绪 -> data/coco128")
PY
fi

if [ -f "data/yolov8n.pt" ]; then
  echo "已存在: data/yolov8n.pt（跳过）"
else
  echo "== 下载 YOLOv8n 权重（约 6MB）"
  python3 - <<'PY'
import urllib.request, shutil, os
url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
with urllib.request.urlopen(url) as r, open("data/yolov8n.pt", "wb") as f:
    shutil.copyfileobj(r, f)
print("权重就绪 -> data/yolov8n.pt")
PY
fi

echo "完成。数据：$(du -sh data/coco128 2>/dev/null | cut -f1)  权重：$(ls -lh data/yolov8n.pt | awk '{print $5}')"
