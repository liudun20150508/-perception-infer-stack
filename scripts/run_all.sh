#!/usr/bin/env bash
# 一键跑全部可运行段落：每段单独落盘、单独判定，任一段失败不阻断其余段。
#
#   bash scripts/run_all.sh         # 全部
#   bash scripts/run_all.sh 1 3     # 只跑指定段
#
# 结果：report/SUMMARY.md、report/DASHBOARD.md 与各段日志。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
REPORT="$ROOT/report"; mkdir -p "$REPORT"
ENGINE_WORK="$REPORT/engine"; mkdir -p "$ENGINE_WORK"
ONLY=("$@")

C_OK=$'\033[32m'; C_NO=$'\033[31m'; C_HD=$'\033[36m'; C_R=$'\033[0m'
banner() { printf '\n%s==================== %s ====================%s\n' "$C_HD" "$1" "$C_R"; }
need() { command -v "$1" >/dev/null 2>&1; }
want() { [ ${#ONLY[@]} -eq 0 ] && return 0; for s in "${ONLY[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

NAME=(); STATUS=(); NOTE=()
record() { NAME+=("$1"); STATUS+=("$2"); NOTE+=("$3"); }

# onnxruntime-gpu 需要 CUDA 12 运行库；必须在进程启动前设置 LD_LIBRARY_PATH
PY_PREFIX="$(python3 -c 'import sys;print(sys.prefix)' 2>/dev/null || echo '')"
if [ -n "$PY_PREFIX" ]; then
  NV_LIBS="$(ls -d "$PY_PREFIX"/lib/python3.*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')"
  [ -n "$NV_LIBS" ] && export LD_LIBRARY_PATH="$NV_LIBS${LD_LIBRARY_PATH:-}"
fi

# ---- S1 感知前后处理算子 ----
if want 1; then
  banner "S1 感知前后处理算子（letterbox + NMS, CUDA）"
  if need nvcc; then
    ( cd src/kernels/perception_ops && bash build_n_run.sh ) 2>&1 | tee "$REPORT/S1_perception.txt"
    [ "${PIPESTATUS[0]}" = 0 ] && record "S1 感知算子" OK "letterbox+NMS 自检 PASS" || record "S1 感知算子" FAIL "见 S1_perception.txt"
  else record "S1 感知算子" SKIP "无 nvcc"; fi
fi

# ---- S2 量化与手写 CUDA 算子 ----
if want 2; then
  banner "S2 CUDA 算子：INT8 GEMM / FP4 反量化 / SGEMM 分块"
  if need nvcc; then
    B="$REPORT/bin"; mkdir -p "$B"
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
    arch="sm_${cc:-8.0}"; arch="${arch/./}"
    {
      echo "### INT8 GEMM (2048^3)"
      nvcc -O3 -arch="$arch" src/kernels/int8_gemm/int8_gemm.cu -lcublas -o "$B/i8" && "$B/i8" 2048 2048 2048
      echo; echo "### FP4(e2m1fn) 反量化"
      nvcc -O3 -arch="$arch" src/kernels/fp4_dequant/fp4_dequant.cu -o "$B/fp4" && "$B/fp4"
      echo; echo "### SGEMM naive -> shared memory tiling"
      cp -r src/kernels/sgemm_attention "$B/sa_tmp"
      sed -i 's/cudaError_t e = (call);/cudaError_t _rc_ = (call);/; s/if (e != cudaSuccess)/if (_rc_ != cudaSuccess)/; s/cudaGetErrorString(e)/cudaGetErrorString(_rc_)/' "$B/sa_tmp/common.h"
      nvcc -O3 -std=c++17 -arch="$arch" "$B/sa_tmp/sgemm.cu" -o "$B/sgemm" && "$B/sgemm"
    } 2>&1 | tee "$REPORT/S2_kernels.txt"
    grep -q "反量化" "$REPORT/S2_kernels.txt" && record "S2 CUDA 算子" OK "INT8 / FP4 / 分块 GEMM" || record "S2 CUDA 算子" FAIL "见 S2_kernels.txt"
  else record "S2 CUDA 算子" SKIP "无 nvcc"; fi
fi

# ---- S3 推理引擎 + 评估系统 ----
if want 3; then
  banner "S3 推理引擎（torch / ONNX-ORT / INT8）+ 评估门禁"
  if need python3; then
    ( cd src/engine && python3 run_pipeline.py --dataset "$ROOT/data/coco128" --work "$ENGINE_WORK" --limit 128 ) \
      > "$REPORT/S3_engine.raw" 2>&1
    s3=$?
    grep -vE "Downloading|WARNING|Ultralytics|summary|^Speed:|Results saved|Requirement|Installing|Successfully|notice|provider_bridge|CreateExecutionProvider|cuStreamGetGreenCtx|Memcpy nodes" \
      "$REPORT/S3_engine.raw" > "$REPORT/S3_engine.txt"
    sed 's/^/  /' "$REPORT/S3_engine.txt"
    [ "$s3" = 0 ] && grep -q "全部通过" "$REPORT/S3_engine.txt" \
      && record "S3 推理引擎" OK "INT8 提速且精度在预算内" || record "S3 推理引擎" FAIL "见 S3_engine.txt"
    banner "S3b 规则引擎判定"
    python3 eval/run_gates.py 2>&1 | tee "$REPORT/S3_gates.txt"
    grep -q "通过  " "$REPORT/S3_gates.txt" || true
  else record "S3 推理引擎" SKIP "无 python3"; fi
fi

# ---- S4 计算框架 ----
if want 4; then
  banner "S4 计算框架：DAG 执行器 + 多路视频调度"
  { ( cd src/framework/dag_executor && bash build_n_run.sh ); echo; ( cd src/framework/multi_stream_scheduler && python3 scheduler_demo.py | tail -22 ); } \
    2>&1 | tee "$REPORT/S4_framework.txt"
  grep -q "DAG 调度并行" "$REPORT/S4_framework.txt" && record "S4 计算框架" OK "依赖调度 + 多路组批" || record "S4 计算框架" FAIL "见 S4_framework.txt"

  banner "S4b 推理服务框架（C++17 + ONNXRuntime）"
  ORT_DIR="$(ls -d "$PY_PREFIX"/lib/python3.*/site-packages/onnxruntime 2>/dev/null | head -1)"
  INC="$(find "$ORT_DIR" -name onnxruntime_cxx_api.h 2>/dev/null | head -1)"
  LIB="$(find "$ORT_DIR" -name 'libonnxruntime.so' 2>/dev/null | head -1)"
  if [ -z "$INC" ] || [ -z "$LIB" ]; then          # pip wheel 不含 C++ 头，改用官方发行包
    CPP="$(bash scripts/fetch_ort_cpp.sh 2>/dev/null || true)"
    if [ -n "$CPP" ]; then
      INC="$(echo "$CPP" | awk '{print $1}')/onnxruntime_cxx_api.h"
      LIB="$(echo "$CPP" | awk '{print $2}')"
    fi
  fi
  if need cmake && [ -n "$INC" ] && [ -n "$LIB" ]; then
    cmake -S src/framework/serving -B "$ENGINE_WORK/serving" -DCMAKE_BUILD_TYPE=Release \
          -DORT_INCLUDE_DIR="$(dirname "$INC")" -DORT_LIBRARY="$LIB" >/dev/null 2>&1 \
      && cmake --build "$ENGINE_WORK/serving" -j"$(nproc)" >/dev/null 2>&1
    if [ -x "$ENGINE_WORK/serving/edge_demo" ]; then
      ( cd src/framework/serving && LD_LIBRARY_PATH="$(dirname "$LIB"):${LD_LIBRARY_PATH:-}" \
          "$ENGINE_WORK/serving/edge_demo" --requests 60 --deadline 20000 ) 2>&1 \
        | grep -v "watchdog:" | tail -30 | tee "$REPORT/S4b_serving.txt" | sed 's/^/  /'
      record "S4b 服务框架" OK "6 场景（吞吐/优先级/超时/热更新/LRU）"
    else record "S4b 服务框架" SKIP "编译未通过"; fi
  else record "S4b 服务框架" SKIP "缺 cmake 或 onnxruntime C++ 头/库"; fi
fi

# ---- S5 训练闭环 ----
if want 5; then
  banner "S5 训练闭环：手写 MLA+MoE 架构收敛 smoke"
  ( cd src/training && python3 train_smoke.py 60 ) 2>&1 | tee "$REPORT/S5_train.txt"
  grep -q "降幅" "$REPORT/S5_train.txt" && record "S5 训练闭环" OK "loss 单调下降" || record "S5 训练闭环" FAIL "见 S5_train.txt"
fi

# ---- S6 OS / 图形 ----
if want 6; then
  banner "S6 OS/体系结构（亲和性 / 伪共享 / NUMA first-touch）"
  ( cd src/os && bash run_os.sh ) 2>&1 | tee "$REPORT/S6_os.txt"
  grep -q "恒定于 cpu0 = 是" "$REPORT/S6_os.txt" && record "S6 OS/体系结构" OK "确定性证据" || record "S6 OS/体系结构" FAIL "见 S6_os.txt"

  banner "S6b 图形：零依赖 CPU 光栅化（无头渲染）"
  if need g++; then
    ( cd src/graphics && bash build_n_run.sh ) 2>&1 | tee "$REPORT/S6b_render.txt"
    grep -q "PNG:" "$REPORT/S6b_render.txt" && record "S6b 图形引擎" OK "无头渲染 -> PNG" || record "S6b 图形引擎" FAIL "见 S6b_render.txt"
  else record "S6b 图形引擎" SKIP "无 g++"; fi
fi

# ---- S7 重构 + 接口契约 ----
if want 7; then
  banner "S7 重构验收（行为不变）"
  ( cd src/refactor && bash verify.sh ) 2>&1 | tee "$REPORT/S7_refactor.txt"
  grep -q "\[PASS\]" "$REPORT/S7_refactor.txt" && record "S7 代码重构" OK "输出逐字节一致" || record "S7 代码重构" FAIL "见 S7_refactor.txt"

fi

# ---------------- 汇总 ----------------
# 契约校验要检查 SUMMARY/DASHBOARD 是否存在，所以先落一次汇总，再校验，再把校验结果并进汇总。
finalize() {
  {
    echo "# 运行结果 $(date -u +%FT%TZ)"
    echo
    echo "| 段落 | 状态 | 结果 |"
    echo "|---|---|---|"
    for i in "${!NAME[@]}"; do echo "| ${NAME[$i]} | ${STATUS[$i]} | ${NOTE[$i]} |"; done
  } > "$REPORT/SUMMARY.md"
  need python3 && python3 eval/make_dashboard.py >/dev/null 2>&1 || true
}
finalize

if want 7; then
  banner "S7b 对外接口契约（校验对外承诺是否可机器验证）"
  python3 eval/interface_contract.py 2>&1 | tee "$REPORT/S7_contract.txt"
  grep -q "全部通过" "$REPORT/S7_contract.txt" \
    && record "S7b 接口契约" OK "可机器校验" || record "S7b 接口契约" FAIL "见 S7_contract.txt"
  finalize
fi

banner "运行结果一览"
printf '  %-22s %-6s %s\n' "段落" "状态" "结果"
for i in "${!NAME[@]}"; do
  st="${STATUS[$i]}"; col="$C_OK"; [ "$st" != "OK" ] && col="$C_NO"
  printf '  %-22s %s%-6s%s %s\n' "${NAME[$i]}" "$col" "$st" "$C_R" "${NOTE[$i]}"
done
okn=$(printf '%s\n' "${STATUS[@]}" 2>/dev/null | grep -c '^OK$' || true)
echo; echo "  通过 ${okn}/${#NAME[@]} 段　日志: report/"
echo "  汇总: report/SUMMARY.md"
echo "  看板: report/DASHBOARD.md"
