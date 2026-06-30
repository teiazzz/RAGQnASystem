# 阶段三 P0：Function Calling / Tool Use

## 已实现范围

本阶段先完成 Agent 工具集设计与轻量执行层；LangGraph 状态机已在
`docs/phase3_langgraph_agent.md` 和 `app/services/agent_graph.py` 中实现。

核心代码：

- `app/services/agent_tools.py`：7 个医疗工具的 schema、description、参数校验、执行函数、LLM 选择与规则降级。
- `app/services/mcp_client.py`：MCP stdio adapter，可把 filesystem / PostgreSQL 等社区 MCP Server 的工具动态挂到 Agent。
- `app/services/llm_service.py`：优先使用 OpenAI-compatible `tools/tool_calls`，不支持时降级到 JSON 规划 prompt。
- `app/services/agent_graph.py`：用 LangGraph 编排 `plan -> execute_tools -> review -> finish` 状态机。
- `app/services/chat_service.py`：在医学问答链路中执行 LangGraph Agent，并把 observation 注入最终 prompt。
- `app/api/v1/chat.py`：SSE `meta.agent_tools` 返回本轮工具调用轨迹。
- `frontend/src/App.tsx`：管理员调试面板展示工具选择、planner、final_action 和 observation。
- `data/agent_eval/tool_use_eval_cases.jsonl` + `scripts/evaluate_agent_tools.py`：工具选择评测集与指标脚本。

## 工具集

| 工具 | 适用场景 |
|---|---|
| `search_knowledge_graph(disease)` | 明确疾病的结构化知识：病因、症状、检查、科室、治疗、并发症等 |
| `search_vector_db(query)` | 口语化医学问题、本地医疗语料、需要引用来源的非结构化资料 |
| `lookup_drug(drug_name)` | 药品用法用量、禁忌、不良反应、相互作用、生产商 |
| `assess_triage(symptoms)` | 推荐科室、紧急程度、是否需要急诊 |
| `web_search(query)` | 最新政策、指南更新、罕见病新资料；当前部署未配置联网，仅返回不可用 observation |
| `escalate_to_human(reason)` | 胸痛、呼吸困难、大出血、疑似中风等高危场景转人工/急诊 |
| `ask_clarification(question)` | 信息不足时先追问，不强行诊断或检索 |
| `mcp__{server}__{tool}(...)` | MCP Server 动态工具，例如 filesystem / PostgreSQL；需要在 `.env` 中启用 MCP |

## 执行逻辑

1. `chat_service.prepare()` 完成多轮 query 补全和医学 query 改写。
2. NER + 意图识别 + Multi-Query + HyDE 并行执行。
3. `agent_tools.run_agent_tools()` 让 LLM 通过原生 `tool_calls` 选择最多 3 个工具；兼容接口不支持时降级为 JSON 规划 prompt；仍失败或未选择时使用规则路由兜底。
4. 如果启用 MCP，`mcp_client` 会先发现社区 MCP Server 工具，并把它们作为 `mcp__server__tool` 动态 schema 提供给 planner。
5. 如果触发 `ask_clarification`，本轮直接反问用户。
6. 如果触发 `escalate_to_human`，本轮直接给出急诊/人工兜底建议。
7. 其他工具 observation 会作为 `<工具调用轨迹>` 注入 prompt，再继续原有 RAG、GraphRAG、冲突消解和引用溯源。

## 配置

`.env` 可覆盖：

```env
AGENT_TOOLS_ENABLED=true
AGENT_TOOL_NATIVE_FUNCTION_CALLING=true
AGENT_TOOL_USE_LLM=true
AGENT_TOOL_MAX_CALLS=3
AGENT_TOOL_TIMEOUT_SECONDS=8
MCP_ENABLED=false
MCP_SERVER_CONFIG_PATH=config/mcp_servers.local.json
```

## 评测

不调用外部 LLM 的可复现工具路由评测：

```powershell
uv run python scripts/evaluate_agent_tools.py --show-failures
```

评测真实 LLM planner：

```powershell
uv run python scripts/evaluate_agent_tools.py --use-llm --show-failures
```

当前指标口径：

- `exact_match`：预测工具集合与期望工具集合完全一致。
- `first_tool_accuracy`：首个工具是否命中期望首工具，用于衡量高危/追问等优先级。
- `tool_precision`：预测工具中有多少是期望工具。
- `tool_recall`：期望工具中有多少被预测出来。

## 面试讲法

可以这样讲：

> 我没有直接把原固定 RAG 流程替换成黑盒 Agent，而是先把医疗能力拆成 7 个互斥工具，全部有 JSON schema 和清晰 description。每轮优先用 OpenAI-compatible tool_calls 让模型选择工具，应用代码只执行注册表里的白名单工具；兼容接口不支持时降级到 JSON 规划 prompt，再失败还有规则路由兜底。现在工具层已经接入 LangGraph，状态图显式分成 plan、execute_tools、review、finish 节点，高危症状会优先转人工/急诊，信息不足会先追问，所有节点和 observation 都进入 SSE trace，前端可以直接看状态机时间线。
