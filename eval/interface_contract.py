#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
接口契约校验（能力点「沟通与跨组协作」）。

把引擎对**外部团队**的承诺写成可执行的契约：别人拿我们的产物接入他的流水线之前，
先跑这个脚本 —— 契约不满足就红。沟通不靠口头，靠可校验的接口。

契约（见 HANDOFF.md）：
  1. report.json 必含 dataset/images/backends/gate
  2. 每个 backend 必含 p50_ms/p99_ms/fps/mAP50/mAP50-95，且为数值
  3. gate.passed 必须是布尔
  4. 三个后端必须齐全（torch-FP32 / engine-FP32 / engine-INT8）
  5. 交付物存在性：DASHBOARD.md、S12 日志
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "report")
ENGINE_JSON = os.path.join(REPORT, "engine", "report.json")
REQUIRED_BACKENDS = ["torch-FP32", "engine-FP32", "engine-INT8"]
NUM_FIELDS = ["p50_ms", "p99_ms", "fps", "mAP50", "mAP50-95"]

fails = []


def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        fails.append(msg)


def main():
    print("=== 接口契约校验 ===")
    check(os.path.exists(ENGINE_JSON), f"交付物存在: {ENGINE_JSON}")
    if os.path.exists(ENGINE_JSON):
        d = json.load(open(ENGINE_JSON))
        for k in ("dataset", "images", "backends", "gate"):
            check(k in d, f"report.json 含字段 `{k}`")
        be = d.get("backends", {})
        for name in REQUIRED_BACKENDS:
            check(name in be, f"后端 `{name}` 已交付")
        for name, v in be.items():
            for f in NUM_FIELDS:
                check(f in v and isinstance(v[f], (int, float)), f"`{name}.{f}` 为数值")
        check(isinstance(d.get("gate", {}).get("passed"), bool), "`gate.passed` 为布尔")
    for f in ("SUMMARY.md", "DASHBOARD.md"):
        check(os.path.exists(os.path.join(REPORT, f)), f"交付物存在: report/{f}")
    print(f"\n契约校验: {'全部通过 ✔' if not fails else f'{len(fails)} 项未过 ✘'}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
