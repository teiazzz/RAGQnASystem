# 阶段三 P0：LangGraph Agent 状态机

## 已实现范围

本阶段把上一项的 Function Calling 工具层接入 LangGraph，形成可观测、可控的 Agent 状态机。

核心代码：

- `app/services/agent_graph.py`：LangGraph 状态图，节点为 `plan -> execute_tools -> review -> finish`。
- `app/services/agent_state.py`：将运行态整理成会话、用户画像、任务、推理、行动、记忆、控制流七类状态快照。
- `app/services/agent_tools.py`：扩展 `AgentToolTrace`，返回 `orchestrator`、`plan`、`graph_events`、`iterations`、`stop_reason` 和控制参数。
- `app/services/chat_service.py`：聊天链路改为调用 `run_agent_graph()`，再按 `final_action` 进入追问、转人工或 RAG 回答。
- `app/api/v1/chat.py`：SSE `meta.agent_tools` 返回 LangGraph 节点轨迹。
- `frontend/src/App.tsx`：管理员调试面板新增 Agent 状态机时间线，可视化查看每个节点、工具调用和 observation。
- `tests/test_agent_graph.py`：覆盖高危转人工路径和 trace 字段。

## 状态设计

`AgentGraphState` 包含：

- `messages`：最近对话和当前用户问题。
- `query` / `entities`：当前独立检索 query 和 NER 实体。
- `plan` / `scratchpad`：计划步骤和工具 observation 摘要。
- `tool_calls` / `pending_tool_calls` / `observations`：工具请求与执行结果。
- `iterations` / `max_iterations` / `timeout_seconds` / `failed_tool_counts`：控制流保护。
- `final_action` / `stop_reason` / `graph_events`：终止动作与可视化轨迹。

## 控制流

1. `plan`：调用工具 planner，优先走原生 function calling，不可用时降级 JSON planner，再降级规则 planner。
2. `execute_tools`：只执行注册表里的白名单工具，逐个记录耗时、状态和 observation。
3. `review`：根据工具结果决定：
   - `escalate_to_human` -> 直接转人工/急诊兜底。
   - `ask_clarification` -> 直接反问。
   - 工具失败且未超迭代 -> 回到 `plan` 重规划。
   - 工具观测已就绪 -> 进入后续 RAG/回答。
4. `finish`：写入最终 `final_action` 和 `stop_reason`。

## 配置

`.env` 可覆盖：

```env
AGENT_GRAPH_ENABLED=true
AGENT_GRAPH_MAX_ITERATIONS=4
AGENT_GRAPH_TIMEOUT_SECONDS=15
AGENT_GRAPH_MAX_REPEATED_TOOL_FAILURES=2
```

## 前端可视化测试

用管理员账号登录前端，发送：

- `突发胸痛伴呼吸困难，怎么办？`：应看到 `plan -> execute_tools -> review -> finish`，`final_action=escalate`。
- `我不舒服`：应看到 `final_action=clarify`。
- `阿司匹林有什么副作用？`：应看到药品工具或本地检索工具调用，再进入 RAG 回答。

右侧“检索调试”面板会展示 Agent 状态机时间线、控制参数、工具调用和 observation。
同一面板也会展示 `AgentState` 七类状态快照，便于解释每轮 Agent 为什么追问、转人工或继续 RAG。
