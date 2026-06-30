# 阶段三 P0：防死循环与控制流保护

## 已实现范围

本阶段在 LangGraph Agent 状态机上补齐控制流保护，防止 Agent 无限规划、重复调用工具或持续消耗 token。

核心代码：

- `app/core/config.py`：新增/配置控制流预算。
- `app/services/agent_graph.py`：在 `plan` 前、`plan` 后、`review` 阶段检查超时和 token 预算。
- `app/services/chat_service.py`：当 `final_action=control_stop` 时，走控制流兜底回复，不继续检索和工具循环。
- `app/services/agent_state.py`：`AgentState.control` 返回迭代、超时、token 预算、工具预算、终止原因。
- `frontend/src/App.tsx`：右侧 AgentState 面板展示 token 用量和预算。
- `tests/test_agent_graph.py`：覆盖 token 预算触发 `control_stop`。

## 保护策略

| 保护项 | 配置 | 行为 |
|---|---|---|
| 最大迭代 | `AGENT_GRAPH_MAX_ITERATIONS=10` | 超过后 `final_action=control_stop` |
| 整体超时 | `AGENT_GRAPH_TIMEOUT_SECONDS=15` | 超时后停止状态机，给用户兜底回复 |
| 单工具超时 | `AGENT_TOOL_TIMEOUT_SECONDS=8` | 单个工具超时记 observation，不阻塞整轮 |
| 重复工具失败 | `AGENT_GRAPH_MAX_REPEATED_TOOL_FAILURES=2` | 同一工具连续失败后触发人工兜底 |
| token 预算 | `AGENT_GRAPH_TOKEN_BUDGET=20000` | 超预算后停止 Agent 循环，并返回控制流停止原因 |

## 前端可视化

管理员登录后，在“检索调试”可以看到：

- `Agent 状态机` 顶部的 `token 当前用量/预算`。
- `AgentState -> 控制流` 中的迭代次数、超时、token 用量、最大工具数、`final_action`、`stop_reason`。

当触发预算保护时，`answer_mode=agent_control_stop`，`final_action=control_stop`，`stop_reason` 会显示具体原因，如 `token_budget_exceeded`。

## 面试讲法

可以这样讲：

> Agent 最怕无限循环和成本失控，所以我把控制流做成显式状态：最大迭代 10 次、整体超时 15 秒、单工具超时 8 秒、同一工具失败 2 次熔断、总 token 预算 20000。每次 LangGraph plan 前、plan 后和工具 review 阶段都会检查预算；一旦超限就把 `final_action` 置为 `control_stop`，不再继续工具循环，并给用户返回兜底说明。所有终止原因都进入 `AgentState.control.stop_reason`，前端可视化可直接复盘。
