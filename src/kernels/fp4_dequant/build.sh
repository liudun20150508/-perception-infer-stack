#!/bin/bash
# 编译 FP4(e2m1fn) 反量化 kernel（A800 = Ampere = sm_80）
set -e
nvcc -O3 -arch=sm_80 -lineinfo fp4_dequant.cu -o fp4_dequant
echo "build OK -> ./fp4_dequant"
