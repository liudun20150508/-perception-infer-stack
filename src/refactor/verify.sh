#!/usr/bin/env bash
# 重构验收：断言重构前后 stdout 逐字节一致（行为不变）。
set -uo pipefail
cd "$(dirname "$0")"

python3 before_scheduler_demo.py > /tmp/ref_before.out 2>&1
python3 after_scheduler.py       > /tmp/ref_after.out  2>&1

if diff -q /tmp/ref_before.out /tmp/ref_after.out >/dev/null; then
  echo "[PASS] 重构前后输出逐字节一致（$(wc -l < /tmp/ref_after.out) 行）"
  echo "       坏味道消除：模块常量 14 -> Config；重复代码块 -> SimState.drain()；"
  echo "       排版/话术从 main() 拆出；A/B 只保留策略差异。"
else
  echo "[FAIL] 输出不一致："
  diff /tmp/ref_before.out /tmp/ref_after.out | head -20
  exit 1
fi

echo
echo "代码量对比："
printf "  before %s 行   after %s 行  （新增类型化 Config/SimState，去掉重复）\n" \
  "$(wc -l < before_scheduler_demo.py)" "$(wc -l < after_scheduler.py)"
