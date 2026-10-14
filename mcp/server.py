#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
perception-infer-stack 的 MCP 服务（stdio，零第三方依赖）。

把仓库里"能跑的东西"暴露成工具，任何 MCP 客户端（Codex / Claude Desktop / 自研 agent）都能调用：
  · list_segments   列出可运行的段落
  · run_segment     跑某一段（白名单，不接受任意命令）
  · get_metrics     读取评测报告 report/engine/report.json
  · run_gates       执行 eval/gates.yaml 判据
  · check_contract  校验对外接口契约

启动：
  python3 mcp/server.py

配置示例（Codex / Claude Desktop 的 mcpServers）：
  {
    "mcpServers": {
      "perception-infer-stack": {
        "command": "python3",
        "args": ["<repo>/mcp/server.py"],
        "env": {"PERCEPTION_STACK_ROOT": "<repo>"}
      }
    }
  }
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.environ.get(
    "PERCEPTION_STACK_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROTOCOL = "2024-11-05"

# 白名单：段号 -> (说明, 命令)。绝不接受客户端传来的任意命令串。
SEGMENTS = {
    "1": ("感知前后处理算子（letterbox + NMS, CUDA）", ["bash", "scripts/run_all.sh", "1"]),
    "2": ("CUDA 算子：INT8 GEMM / FP4 反量化 / SGEMM 分块", ["bash", "scripts/run_all.sh", "2"]),
    "3": ("推理引擎三后端 + mAP 评估", ["bash", "scripts/run_all.sh", "3"]),
    "4": ("计算框架：DAG 执行器 + 多路调度 + 服务框架", ["bash", "scripts/run_all.sh", "4"]),
    "5": ("训练闭环 smoke", ["bash", "scripts/run_all.sh", "5"]),
    "6": ("OS/体系结构 + 图形光栅化", ["bash", "scripts/run_all.sh", "6"]),
    "7": ("重构验收 + 接口契约", ["bash", "scripts/run_all.sh", "7"]),
}

TOOLS = [
    {"name": "list_segments", "description": "列出可运行的段落编号与说明",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "run_segment", "description": "运行指定段落（白名单段号，如 '3'），返回输出尾部",
     "inputSchema": {"type": "object",
                     "properties": {"segment": {"type": "string", "description": "段号 1-7"}},
                     "required": ["segment"]}},
    {"name": "get_metrics", "description": "读取评测报告（延迟 p50/p99、mAP、门禁状态）",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "run_gates", "description": "执行验收判据 eval/gates.yaml，返回逐条 PASS/FAIL",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "check_contract", "description": "校验对外接口契约",
     "inputSchema": {"type": "object", "properties": {}}},
]


def _run(cmd, timeout=1800):
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + (p.stderr or "")
        return p.returncode, out[-4000:]
    except subprocess.TimeoutExpired:
        return 124, f"超时（>{timeout}s）：{' '.join(cmd)}"
    except FileNotFoundError as e:
        return 127, f"命令不存在：{e}"


def call_tool(name, args):
    if name == "list_segments":
        return "\n".join(f"{k}. {v[0]}" for k, v in sorted(SEGMENTS.items()))
    if name == "run_segment":
        seg = str(args.get("segment", "")).strip()
        if seg not in SEGMENTS:
            return f"段号 {seg!r} 不在白名单：{sorted(SEGMENTS)}"
        desc, cmd = SEGMENTS[seg]
        rc, out = _run(cmd)
        return f"[{desc}]\n退出码 {rc}\n{out}"
    if name == "get_metrics":
        p = os.path.join(ROOT, "report", "engine", "report.json")
        if not os.path.exists(p):
            return "还没有报告。先跑 run_segment('3')。"
        return json.dumps(json.load(open(p, encoding="utf-8")), ensure_ascii=False, indent=2)
    if name == "run_gates":
        rc, out = _run(["python3", "eval/run_gates.py"], timeout=600)
        return f"退出码 {rc}\n{out}"
    if name == "check_contract":
        rc, out = _run(["python3", "eval/interface_contract.py"], timeout=600)
        return f"退出码 {rc}\n{out}"
    return f"未知工具：{name}"


def reply(id_, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": id_}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = req.get("method")
        rid = req.get("id")
        if method == "initialize":
            reply(rid, {"protocolVersion": PROTOCOL,
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "perception-infer-stack", "version": "1.0.0"}})
        elif method in ("notifications/initialized", "initialized"):
            continue                     # 通知，无需响应
        elif method == "tools/list":
            reply(rid, {"tools": TOOLS})
        elif method == "tools/call":
            params = req.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            try:
                text = call_tool(name, args)
                reply(rid, {"content": [{"type": "text", "text": text}], "isError": False})
            except Exception as e:        # 工具异常按 MCP 约定回 isError
                reply(rid, {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}],
                            "isError": True})
        elif rid is not None:
            reply(rid, error={"code": -32601, "message": f"method not found: {method}"})


if __name__ == "__main__":
    main()
