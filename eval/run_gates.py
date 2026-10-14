#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
规则引擎：按 eval/gates.yaml 判定验收结果。

  python3 eval/run_gates.py [--report <report.json>] [--gates eval/gates.yaml]

退出码 0 = 全部判据满足；非 0 = 有判据未过（可直接用于 CI 卡点）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_yaml(path):
    try:
        import yaml
    except ImportError:
        sys.exit("需要 pyyaml：pip install pyyaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dig(obj, path):
    cur = obj
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


OPS = {
    "abs_diff_le": lambda l, r, v: abs(l - r) <= v,
    "diff_le":     lambda l, r, v: (l - r) <= v,
    "ratio_ge":    lambda l, r, v: (l / r) >= v if r else False,
    "le":          lambda l, r, v: l <= v,
    "ge":          lambda l, r, v: l >= v,
    "has_keys":    lambda l, r, v: isinstance(l, dict) and all(k in l for k in v),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", default=os.path.join(ROOT, "eval/gates.yaml"))
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    spec = load_yaml(a.gates)
    rp = a.report or os.path.join(ROOT, spec.get("report", "report/engine/report.json"))
    if not os.path.isabs(rp):
        rp = os.path.join(ROOT, rp)
    if not os.path.exists(rp):
        sys.exit(f"找不到报告 {rp}（先跑：bash scripts/run_all.sh 3）")
    rep = json.load(open(rp, encoding="utf-8"))

    print(f"规则引擎：{os.path.basename(a.gates)}  报告：{rp}")
    failed = 0
    for g in spec["gates"]:
        op = OPS.get(g["op"])
        if op is None:
            print(f"  [SKIP] {g['id']}: 未知判据算子 {g['op']}")
            continue
        left = dig(rep, g["left"])
        right = dig(rep, g["right"]) if "right" in g else None
        try:
            ok = op(left, right, g["value"])
        except Exception as e:
            ok = False
            print(f"  [ERR ] {g['id']}: {type(e).__name__}: {e}")
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        detail = f"{g['left']}={left}"
        if right is not None:
            detail += f", {g['right']}={right}"
        print(f"  [{mark}] {g['id']}: {g['desc']}  ({detail}, 阈值 {g['op']} {g['value']})")

    total = len(spec["gates"])
    print(f"\n判据：{total - failed}/{total} 通过{'  ✔' if not failed else '  ✘'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
