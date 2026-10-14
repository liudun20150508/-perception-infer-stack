#!/bin/bash
# 编译 INT8 GEMM（A800 = Ampere = sm_80）
set -e
nvcc -O3 -arch=sm_80 -lineinfo int8_gemm.cu -lcublas -o int8_gemm
echo "build OK -> ./int8_gemm [M] [N] [K]"
