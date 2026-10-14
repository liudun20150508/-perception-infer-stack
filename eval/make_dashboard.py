#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把各段产物聚合成一张指标看板 report/DASHBOARD.md。"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "report")


def grep(path, pattern, default="—"):
    p = os.path.join(REP, path)
    if not os.path.exists(p):
        return default
    m = re.search(pattern, open(p, encoding="utf-8", errors="replace").read())
    return m.group(1).strip() if m else default


def rows_from_summary():
    p = os.path.join(REP, "SUMMARY.md")
    rows = []
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            m = re.match(r"\|\s*(S\d[^|]*?)\s*\|\s*(\w+)\s*\|\s*(.*?)\s*\|", line)
            if m:
                rows.append((m.group(1), m.group(2), m.group(3)))
    return rows


def main():
    rows = rows_from_summary()
    ok = sum(1 for _, s, _ in rows if s == "OK")
    out = ["# 指标看板", "",
           f"生成时间：{datetime.now(timezone.utc).isoformat(timespec='seconds')}　"
           f"段落通过：**{ok}/{len(rows)}**", "",
           "## 1. 段落状态", "", "| 段落 | 状态 | 结果 |", "|---|---|---|"]
    for n, s, note in rows:
        out.append(f"| {n} | {'✅' if s == 'OK' else '❌'} {s} | {note} |")

    out += ["", "## 2. 关键指标", "", "| 指标 | 值 | 来源 |", "|---|---|---|"]
    j = os.path.join(REP, "engine", "report.json")
    if os.path.exists(j):
        d = json.load(open(j, encoding="utf-8"))
        for k, v in d.get("backends", {}).items():
            out.append(f"| {k}（p50 / mAP50） | {v['p50_ms']} ms / {v['mAP50']} | S3 |")
        out.append(f"| 规则引擎门禁 | {'PASS' if d.get('gate', {}).get('passed') else 'FAIL'} | S3 |")
    out.append(f"| letterbox 自检 max|diff| | {grep('S1_perception.txt', r'max\|diff\|\s*=\s*(\S+)')} | S1 |")
    out.append(f"| 16 路显存 | {grep('S4_framework.txt', r'显存:\s*(\S+ -> \S+)')} | S4 |")
    out.append(f"| 感知 backbone 耗时 | {grep('S3_engine.txt', r'backbone\s+.*?\s+([\d.]+)\s+[\d.]+%')} ms | S3 |")

    out += ["", "## 3. 口径", "",
            "- 延迟取多轮最小值；精度与性能必须同表出现，详见 `docs/benchmarking.md`。",
            "- 判据由 `eval/gates.yaml` 定义，`eval/run_gates.py` 执行。", ""]
    with open(os.path.join(REP, "DASHBOARD.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("看板: report/DASHBOARD.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
