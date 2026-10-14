#!/usr/bin/env bash
# 零依赖 CPU 光栅化：把布料引擎导出的真实网格无头渲染成图（PPM -> PNG）
set -euo pipefail
cd "$(dirname "$0")"
OBJ="$(cd "$(dirname "$0")/../.." && pwd)/data/mesh/wedding_dress.obj"
mkdir -p out
g++ -O3 -std=c++17 soft_render.cpp -o soft_render
./soft_render "$OBJ" out/dress_hero.ppm 900 4
python3 - <<'PY'
from PIL import Image
import glob, os
for p in glob.glob(os.path.join(os.path.dirname(__file__) or ".", "out", "*.ppm")):
    Image.open(p).save(p[:-4] + ".png")
    print("PNG:", p[:-4] + ".png", Image.open(p).size)
PY
