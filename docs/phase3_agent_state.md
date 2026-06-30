# 阶段三 P0：Agent 状态设计

## 已实现范围

本阶段把 Agent 的运行状态显式建模为可序列化的 `AgentState` 快照，并接入 LangGraph 与前端调试面板。

核心代码：

- `app/services/agent_state.py`：构建七类 AgentState 快照。
- `app/services/agent_graph.py`：每次 LangGraph 执行完成后，把运行态转换为 `agent_tools.state`。
- `app/services/agent_tools.py`：`AgentToolTrace.to_meta()` 返回 `state` 字段。
- `app/api/v1/chat.py`：SSE `meta.agent_tools.state` 返回给前端。
- `frontend/src/App.tsx`：管理员调试面板展示 AgentState 七类状态。
- `tests/test_agent_state.py`：覆盖状态快照构建和意图补全。

## 七类状态

| 状态类别 | 字段 | 用途 |
|---|---|---|
| 会话状态 `conversation` | `messages`、`history_turns`、`current_query`、`original_query`、`standalone_query`、`rewritten_query` | 解决多轮上下文、指代补全和 query 改写可追踪 |
| 用户画像 `user_profile` | `account`、`medical_facts`、`source`、`confidence` | 记录登录用户上下文和本轮提及的健康事实；只标记为 mentioned，不当作确诊病史 |
| 任务状态 `task` | `intent.primary`、`intent.candidates`、`slots`、`missing_slots`、`risk_flags` | 让 Agent 知道当前任务是什么、槽位是否齐全、是否存在高危信号 |
| 推理状态 `reasoning` | `plan`、`scratchpad`、`scratchpad_size` | 保留 Plan-Execute/ReAct 的计划与工具观测摘要 |
| 行动状态 `actions` | `tool_calls`、`observations`、计数 | 记录调用了哪些工具、参数是什么、结果如何 |
| 记忆状态 `memory` | `short_term`、`working`、`long_term` | 短期记忆为最近消息窗口，工作记忆为当前槽位和风险；长期记忆留给 P1 记忆系统 |
| 控制流状态 `control` | `iterations`、`max_iterations`、`timeout_seconds`、`token_budget`、`token_usage`、`max_tool_calls`、`final_action`、`stop_reason` | 防死循环、预算控制、终止原因可解释 |

## 前端可视化

管理员登录后发送任意问题，右侧“检索调试”会出现：

- Agent 状态机时间线：LangGraph 节点执行顺序。
- AgentState：按七类状态展示当前会话、用户画像、任务槽位、推理、行动、记忆、控制流。

可用测试问题：

- `孕妇胸痛2小时很痛，对青霉素过敏，怎么办？`
- `阿司匹林有什么副作用？`
- `我不舒服`

## 面试讲法

可以这样讲：

> 我把 AgentState 拆成 7 类：会话、用户画像、任务、推理、行动、记忆和控制流。会话状态保存原始问题、独立问题和改写 query；任务状态保存 intent、slots、missing_slots 和高危风险；行动状态保存 tool_calls/observations；控制流保存迭代次数、超时、终止原因。医疗画像只记录用户“提到”的事实，不把它当确诊病史，避免 Agent 过度推断。这样做的好处是：多轮问题可追踪、工具调用可复盘、防死循环可解释，前端能直接看到每轮 Agent 的完整状态。
