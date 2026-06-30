# 阶段三 P2：MCP 接入

## 已实现范围

本阶段把 Agent 工具层扩展为 MCP 兼容的可插拔工具架构：

- `app/services/mcp_client.py`：MCP stdio client adapter，负责读取 Server 配置、发现工具、调用工具。
- `app/services/agent_tools.py`：把 MCP Server 暴露的工具动态转换为 OpenAI-compatible function schema，纳入 LLM planner 白名单。
- `app/services/agent_graph.py`：`meta.agent_tools.available_tools` 会显示本轮已挂载的内置工具 + MCP 工具。
- `config/mcp_servers.example.json`：filesystem / PostgreSQL 两类社区 MCP Server 配置示例。
- `tests/test_mcp_client.py`、`tests/test_agent_tools_mcp.py`：不启动外部 Server 的可复现单测，覆盖配置解析、动态工具解析、执行委托。

默认 `MCP_ENABLED=false`，所以本地未安装 `npx` 或未配置 PostgreSQL MCP Server 时，不影响原有医疗问答链路。

## 启用方式

`.env` 中添加：

```env
MCP_ENABLED=true
MCP_SERVER_CONFIG_PATH=config/mcp_servers.local.json
MCP_TOOL_CACHE_TTL_SECONDS=60
MCP_TOOL_TIMEOUT_SECONDS=8
```

本地配置示例：

```json
{
  "servers": {
    "filesystem": {
      "enabled": true,
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "data", "docs"]
    },
    "postgres": {
      "enabled": false,
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-postgres",
        "postgresql://raguser:ragpass123@localhost:5433/ragqna"
      ]
    }
  }
}
```

Windows 如果 `npx` 不能被 MCP SDK 子进程解析，可以把 `command` 改成 `npx.cmd`。

## 工作流

1. `mcp_client.list_mcp_tools()` 按配置启动 MCP stdio server，并调用 `list_tools()`。
2. 每个 MCP tool 转成 Agent 工具名：`mcp__{server}__{tool}`，例如 `mcp__filesystem__read_file`。
3. `agent_tools.select_tool_calls()` 把内置医疗工具和 MCP 工具一起传给 LLM planner。
4. planner 选择 `mcp__...` 工具时，`execute_tool_call()` 不走内置 handler，而是委托给 `mcp_client.call_mcp_tool_by_agent_name()`。
5. MCP observation 会进入原有 `<工具调用轨迹>` prompt 片段，并出现在 SSE `meta.agent_tools.tool_observations` 中。

## 为什么这样设计

MCP 的价值不是“又写一个工具函数”，而是把工具协议标准化：

- 工具发现：应用不需要手写每个外部工具的 schema，直接读取 MCP Server 的 `list_tools()`。
- 工具执行：应用只实现一次 stdio adapter，后续 filesystem、PostgreSQL、浏览器、GitHub 等 Server 都走同一条调用链。
- 安全边界：Agent 仍只执行白名单里的工具名，filesystem Server 也只暴露配置中的 `data`、`docs` 目录。
- 可降级：SDK 缺失、Server 未配置、Server 超时都会变成 observation，不会打断医疗问答。

## 测试

不启动外部 MCP Server 的单测：

```powershell
uv run pytest tests/test_mcp_client.py tests/test_agent_tools_mcp.py
```

真实 MCP Server 联调：

```powershell
uv run python scripts/list_mcp_tools.py
```

## 面试讲法

可以这样讲：

> 我把 Agent 工具层接入了 MCP。实现上不是把某个文件读取函数硬编码进去，而是写了一个 `mcp_client` adapter：启动社区 MCP Server 后先 `list_tools()` 做工具发现，再把每个 tool 动态转换成 OpenAI-compatible function schema，工具名统一加 `mcp__server__tool` 前缀。LLM planner 看到的是普通 function calling 工具，应用执行时识别 `mcp__` 前缀并委托给 MCP Server。这样 filesystem、PostgreSQL 这类外部能力可以通过配置插拔，工具失败也只作为 observation 返回，不会破坏医疗问答主链路。

如果被追问 MCP 和 Function Calling 的区别：

> Function Calling 是模型和应用之间的工具调用格式，解决“模型怎么提出调用工具”；MCP 是应用和外部工具/数据源之间的标准协议，解决“工具怎么被发现、描述、调用”。我项目里二者是组合关系：MCP tool 被发现后转成 function schema 给模型选择，真正执行时再通过 MCP 协议调用 Server。
