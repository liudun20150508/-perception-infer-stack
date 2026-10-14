# MCP 服务

把仓库能力暴露成 MCP 工具，供 Codex / Claude Desktop / 自研 agent 调用。**零第三方依赖**
（只用标准库 + JSON-RPC over stdio）。

## 工具

| 工具 | 作用 |
|---|---|
| `list_segments` | 列出可运行段落 |
| `run_segment` | 运行某段（**白名单段号**，不接受任意命令） |
| `get_metrics` | 读取 `report/engine/report.json` |
| `run_gates` | 执行 `eval/gates.yaml` 判据 |
| `check_contract` | 校验对外接口契约 |

## 接入

```json
{
  "mcpServers": {
    "perception-infer-stack": {
      "command": "python3",
      "args": ["<repo>/mcp/server.py"],
      "env": { "PERCEPTION_STACK_ROOT": "<repo>" }
    }
  }
}
```

## 手动自测

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"list_segments","arguments":{}}}' \
  | python3 mcp/server.py
```

## 安全边界

- `run_segment` 只接受 `1..7` 的段号，命令在服务端写死，客户端无法注入任意 shell。
- 所有子进程都在仓库根目录下执行，且带超时。
- 不做任何外部写操作（不上传、不推送）。
