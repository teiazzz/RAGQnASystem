"""LangGraph orchestration for the medical Agent tool loop.

The tool registry lives in :mod:`app.services.agent_tools`. This module adds a
small, explicit state machine around that registry so every turn has observable
control flow: plan -> execute tools -> review -> finish.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from functools import lru_cache
from typing import Any, TypedDict

from app.core.config import settings
from app.services import agent_state, agent_tools, llm_service

try:  # pragma: no cover - exercised in runtime; fallback keeps imports robust
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:  # noqa: BLE001 - optional dependency fallback
    END = START = None
    StateGraph = None
    LANGGRAPH_AVAILABLE = False

logger = logging.getLogger(__name__)


class AgentGraphState(TypedDict, total=False):
    """State carried through the LangGraph Agent turn."""

    messages: list[dict[str, str]]
    query: str
    original_query: str
    standalone_query: str
    rewritten_query: str
    user_profile: dict[str, Any]
    intent_names: list[str]
    raw_intent_response: str
    entities: dict[str, str]
    plan: list[str]
    scratchpad: list[str]
    tool_calls: list[agent_tools.ToolCallRequest]
    pending_tool_calls: list[agent_tools.ToolCallRequest]
    observations: list[agent_tools.ToolObservation]
    failed_tool_counts: dict[str, int]
    planner: str
    planner_error: str
    graph_events: list[dict[str, Any]]
    iterations: int
    max_iterations: int
    started_at: float
    timeout_seconds: float
    next_action: str
    final_action: str
    stop_reason: str


def graph_is_available() -> bool:
    return LANGGRAPH_AVAILABLE and StateGraph is not None


async def run_agent_graph(
    query: str,
    entities: dict[str, str] | None = None,
    messages: list[dict[str, str]] | None = None,
    user_profile: dict[str, Any] | None = None,
    intent_names: list[str] | None = None,
    raw_intent_response: str = "",
    original_query: str | None = None,
    standalone_query: str | None = None,
) -> agent_tools.AgentToolTrace:
    """Run one bounded LangGraph ReAct/Plan-Execute tool loop.

    If LangGraph is unavailable or disabled, this degrades to the existing
    single-pass tool runner while preserving trace metadata.
    """

    entities = entities or {}
    messages = messages or [{"role": "user", "content": query}]
    intent_names = intent_names or []
    original_query = original_query or query
    standalone_query = standalone_query or query
    user_profile = user_profile or {}
    available_tools = await agent_tools.get_available_tool_names()
    if not settings.AGENT_GRAPH_ENABLED or not graph_is_available():
        trace = await agent_tools.run_agent_tools(query, entities)
        trace.orchestrator = "tool_runner_fallback"
        trace.stop_reason = "langgraph_disabled_or_unavailable"
        trace.state_snapshot = agent_state.build_state_snapshot(
            messages=messages,
            query=query,
            entities=entities,
            original_query=original_query,
            standalone_query=standalone_query,
            rewritten_query=query,
            user_profile=user_profile,
            intent_names=intent_names,
            raw_intent_response=raw_intent_response,
            plan=trace.plan,
            scratchpad=[],
            tool_calls=trace.calls,
            observations=trace.observations,
            control=trace.control,
            final_action=trace.final_action,
            stop_reason=trace.stop_reason,
        )
        return trace

    initial_state: AgentGraphState = {
        "messages": messages,
        "query": query,
        "original_query": original_query,
        "standalone_query": standalone_query,
        "rewritten_query": query,
        "user_profile": user_profile,
        "intent_names": intent_names,
        "raw_intent_response": raw_intent_response,
        "entities": entities,
        "plan": [],
        "scratchpad": [],
        "tool_calls": [],
        "pending_tool_calls": [],
        "observations": [],
        "failed_tool_counts": {},
        "planner": "none",
        "planner_error": "",
        "graph_events": [],
        "iterations": 0,
        "max_iterations": settings.AGENT_GRAPH_MAX_ITERATIONS,
        "started_at": time.perf_counter(),
        "timeout_seconds": settings.AGENT_GRAPH_TIMEOUT_SECONDS,
        "next_action": "plan",
        "final_action": "none",
        "stop_reason": "",
    }

    try:
        final_state = await asyncio.wait_for(
            _compiled_agent_graph().ainvoke(initial_state),
            timeout=settings.AGENT_GRAPH_TIMEOUT_SECONDS + 2,
        )
    except TimeoutError:
        logger.warning("LangGraph Agent turn timed out")
        control = _control_meta(0, "graph_timeout")
        return agent_tools.AgentToolTrace(
            available_tools=available_tools,
            orchestrator="langgraph",
            graph_events=[
                _event(
                    "timeout",
                    "error",
                    0,
                    "LangGraph 状态机整体超时，降级继续常规 RAG。",
                    elapsed_ms=0,
                )
            ],
            stop_reason="graph_timeout",
            final_action_override="control_stop",
            control=control,
            state_snapshot=agent_state.build_state_snapshot(
                messages=messages,
                query=query,
                entities=entities,
                original_query=original_query,
                standalone_query=standalone_query,
                rewritten_query=query,
                user_profile=user_profile,
                intent_names=intent_names,
                raw_intent_response=raw_intent_response,
                control=control,
                final_action="control_stop",
                stop_reason="graph_timeout",
            ),
        )
    except Exception as exc:  # noqa: BLE001 - graph failures should not break chat
        logger.warning("LangGraph Agent failed, falling back to tool runner", exc_info=True)
        trace = await agent_tools.run_agent_tools(query, entities)
        trace.orchestrator = "tool_runner_fallback"
        trace.planner_error = f"langgraph failed: {exc}"
        trace.stop_reason = "langgraph_error_fallback"
        trace.state_snapshot = agent_state.build_state_snapshot(
            messages=messages,
            query=query,
            entities=entities,
            original_query=original_query,
            standalone_query=standalone_query,
            rewritten_query=query,
            user_profile=user_profile,
            intent_names=intent_names,
            raw_intent_response=raw_intent_response,
            plan=trace.plan,
            scratchpad=[],
            tool_calls=trace.calls,
            observations=trace.observations,
            control=trace.control,
            final_action=trace.final_action,
            stop_reason=trace.stop_reason,
        )
        return trace

    control = _control_meta(
        int(final_state.get("iterations", 0) or 0),
        final_state.get("stop_reason", ""),
    )
    final_action = final_state.get("final_action", None) or "none"
    return agent_tools.AgentToolTrace(
        calls=final_state.get("tool_calls", []),
        observations=final_state.get("observations", []),
        available_tools=available_tools,
        planner=final_state.get("planner", "none"),
        planner_error=final_state.get("planner_error", ""),
        orchestrator="langgraph",
        plan=final_state.get("plan", []),
        graph_events=final_state.get("graph_events", []),
        iterations=int(final_state.get("iterations", 0) or 0),
        stop_reason=final_state.get("stop_reason", ""),
        final_action_override=final_action,
        control=control,
        state_snapshot=agent_state.build_state_snapshot(
            messages=messages,
            query=query,
            entities=entities,
            original_query=original_query,
            standalone_query=standalone_query,
            rewritten_query=query,
            user_profile=user_profile,
            intent_names=intent_names,
            raw_intent_response=raw_intent_response,
            plan=final_state.get("plan", []),
            scratchpad=final_state.get("scratchpad", []),
            tool_calls=final_state.get("tool_calls", []),
            observations=final_state.get("observations", []),
            control=control,
            final_action=final_action,
            stop_reason=final_state.get("stop_reason", ""),
        ),
    )


@lru_cache(maxsize=1)
def _compiled_agent_graph():
    graph = StateGraph(AgentGraphState)
    graph.add_node("plan", _plan_node)
    graph.add_node("execute_tools", _execute_tools_node)
    graph.add_node("review", _review_node)
    graph.add_node("finish", _finish_node)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan",
        _route_after_plan,
        {"execute_tools": "execute_tools", "review": "review", "finish": "finish"},
    )
    graph.add_edge("execute_tools", "review")
    graph.add_conditional_edges(
        "review",
        _route_after_review,
        {"plan": "plan", "finish": "finish"},
    )
    graph.add_edge("finish", END)
    return graph.compile()


async def _plan_node(state: AgentGraphState) -> dict[str, Any]:
    iteration = int(state.get("iterations", 0) or 0) + 1
    if iteration > int(state.get("max_iterations", settings.AGENT_GRAPH_MAX_ITERATIONS)):
        return _stop_update(state, "max_iterations", "control_stop", iteration - 1)
    budget_stop_reason = _budget_stop_reason(state, phase="before_plan")
    if budget_stop_reason:
        return _stop_update(state, budget_stop_reason, "control_stop", iteration - 1)

    repeated_failure_tool = _repeated_failure_tool(state)
    if repeated_failure_tool:
        calls = [
            agent_tools.ToolCallRequest(
                name="escalate_to_human",
                arguments={
                    "reason": f"{repeated_failure_tool} 连续失败，Agent 已触发人工兜底。"
                },
                reason="同一工具连续失败，按控制流保护转人工。",
                source="langgraph_guard",
            )
        ]
        planner = "langgraph_guard"
        planner_error = state.get("planner_error", "")
    else:
        calls, planner, planner_error = await agent_tools.select_tool_calls(
            state.get("query", ""),
            state.get("entities", {}),
        )
        calls = _drop_duplicate_calls(calls, state)
    budget_stop_reason = _budget_stop_reason(state, phase="after_plan")
    if budget_stop_reason:
        return _stop_update(state, budget_stop_reason, "control_stop", iteration)

    next_plan = state.get("plan") or _build_readable_plan(calls, planner)
    event = _event(
        "plan",
        "ok",
        iteration,
        _plan_summary(calls, planner),
        elapsed_ms=_elapsed_ms(state),
        planner=planner,
        planned_tools=[call.name for call in calls],
    )
    return {
        "iterations": iteration,
        "planner": planner,
        "planner_error": planner_error or state.get("planner_error", ""),
        "pending_tool_calls": calls,
        "tool_calls": [*state.get("tool_calls", []), *calls],
        "plan": next_plan,
        "graph_events": [*state.get("graph_events", []), event],
        "next_action": "execute" if calls else "review",
    }


async def _execute_tools_node(state: AgentGraphState) -> dict[str, Any]:
    calls = state.get("pending_tool_calls", [])
    if not calls:
        return {
            "graph_events": [
                *state.get("graph_events", []),
                _event(
                    "execute_tools",
                    "skipped",
                    int(state.get("iterations", 0) or 0),
                    "没有待执行工具。",
                    elapsed_ms=_elapsed_ms(state),
                ),
            ]
        }

    context = agent_tools.ToolExecutionContext(
        query=state.get("query", ""),
        entities=state.get("entities", {}),
    )
    observations: list[agent_tools.ToolObservation] = []
    failed_tool_counts = dict(state.get("failed_tool_counts", {}))
    for call in calls[: settings.AGENT_TOOL_MAX_CALLS]:
        observation = await agent_tools.execute_tool_call(call, context)
        observations.append(observation)
        if _is_failed_observation(observation):
            failed_tool_counts[call.name] = failed_tool_counts.get(call.name, 0) + 1

    scratchpad = [
        *state.get("scratchpad", []),
        *[
            f"{obs.name}({obs.status}): {obs.content[:220]}"
            for obs in observations
        ],
    ]
    event = _event(
        "execute_tools",
        "ok",
        int(state.get("iterations", 0) or 0),
        f"执行 {len(observations)} 个工具。",
        elapsed_ms=_elapsed_ms(state),
        tools=[
            {
                "name": obs.name,
                "status": obs.status,
                "elapsed_ms": obs.elapsed_ms,
            }
            for obs in observations
        ],
    )
    return {
        "observations": [*state.get("observations", []), *observations],
        "scratchpad": scratchpad,
        "failed_tool_counts": failed_tool_counts,
        "graph_events": [*state.get("graph_events", []), event],
    }


async def _review_node(state: AgentGraphState) -> dict[str, Any]:
    calls = state.get("pending_tool_calls", [])
    observations = state.get("observations", [])
    current_observations = observations[-len(calls) :] if calls else []
    iteration = int(state.get("iterations", 0) or 0)

    final_action = "answer"
    next_action = "finish"
    stop_reason = "tool_observation_ready"

    call_names = {call.name for call in state.get("tool_calls", [])}
    if "escalate_to_human" in call_names:
        final_action = "escalate"
        stop_reason = "escalate_requested"
    elif "ask_clarification" in call_names:
        final_action = "clarify"
        stop_reason = "clarification_requested"
    elif not state.get("tool_calls"):
        final_action = "none"
        stop_reason = "planner_returned_no_tool"
    else:
        budget_stop_reason = _budget_stop_reason(state, phase="after_tools")
        if budget_stop_reason:
            final_action = "control_stop"
            stop_reason = budget_stop_reason
    if (
        final_action == "answer"
        and current_observations
        and all(_is_failed_observation(obs) for obs in current_observations)
    ):
        if iteration < int(state.get("max_iterations", settings.AGENT_GRAPH_MAX_ITERATIONS)):
            next_action = "plan"
            stop_reason = "tool_failed_replan"
        else:
            stop_reason = "tool_failed_max_iterations"

    event = _event(
        "review",
        "ok",
        iteration,
        f"next={next_action}, final_action={final_action}, reason={stop_reason}",
        elapsed_ms=_elapsed_ms(state),
        final_action=final_action,
        next_action=next_action,
        stop_reason=stop_reason,
    )
    return {
        "final_action": final_action,
        "next_action": next_action,
        "stop_reason": stop_reason,
        "pending_tool_calls": [],
        "graph_events": [*state.get("graph_events", []), event],
    }


async def _finish_node(state: AgentGraphState) -> dict[str, Any]:
    event = _event(
        "finish",
        "ok",
        int(state.get("iterations", 0) or 0),
        f"状态机结束：{state.get('stop_reason', '')}",
        elapsed_ms=_elapsed_ms(state),
        final_action=state.get("final_action", "none"),
        stop_reason=state.get("stop_reason", ""),
    )
    return {"graph_events": [*state.get("graph_events", []), event]}


def _route_after_plan(state: AgentGraphState) -> str:
    if state.get("next_action") == "finish":
        return "finish"
    return "execute_tools" if state.get("pending_tool_calls") else "review"


def _route_after_review(state: AgentGraphState) -> str:
    return "plan" if state.get("next_action") == "plan" else "finish"


def _build_readable_plan(
    calls: list[agent_tools.ToolCallRequest],
    planner: str,
) -> list[str]:
    if not calls:
        return [f"planner={planner}: 未选择工具，继续常规 RAG/直接回答。"]
    return [
        f"planner={planner}: 识别用户意图与医疗风险。",
        *[
            f"调用 {call.name}：{call.reason or '获取外部观测'}"
            for call in calls
        ],
        "根据 observation 决定继续 RAG、追问或转人工。",
    ]


def _drop_duplicate_calls(
    calls: list[agent_tools.ToolCallRequest],
    state: AgentGraphState,
) -> list[agent_tools.ToolCallRequest]:
    seen = {_call_key(call) for call in state.get("tool_calls", [])}
    result: list[agent_tools.ToolCallRequest] = []
    for call in calls:
        key = _call_key(call)
        if key in seen:
            continue
        result.append(call)
        seen.add(key)
    return result


def _call_key(call: agent_tools.ToolCallRequest) -> str:
    args = json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)
    return f"{call.name}:{args}"


def _is_failed_observation(obs: agent_tools.ToolObservation) -> bool:
    return obs.status in {"error", "timeout", "missing_argument"}


def _repeated_failure_tool(state: AgentGraphState) -> str:
    threshold = settings.AGENT_GRAPH_MAX_REPEATED_TOOL_FAILURES
    for name, count in state.get("failed_tool_counts", {}).items():
        if count >= threshold:
            return name
    return ""


def _budget_stop_reason(state: AgentGraphState, phase: str) -> str:
    if _time_budget_exceeded(state):
        return f"timeout_{phase}"
    if _token_budget_exceeded():
        return "token_budget_exceeded"
    return ""


def _time_budget_exceeded(state: AgentGraphState) -> bool:
    started_at = float(state.get("started_at", time.perf_counter()) or time.perf_counter())
    timeout_seconds = float(
        state.get("timeout_seconds", settings.AGENT_GRAPH_TIMEOUT_SECONDS)
        or settings.AGENT_GRAPH_TIMEOUT_SECONDS
    )
    return time.perf_counter() - started_at >= timeout_seconds


def _elapsed_ms(state: AgentGraphState) -> int:
    started_at = float(state.get("started_at", time.perf_counter()) or time.perf_counter())
    return max(int((time.perf_counter() - started_at) * 1000), 0)


def _event(
    node: str,
    status: str,
    iteration: int,
    summary: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "node": node,
        "status": status,
        "iteration": iteration,
        "summary": summary,
        **extra,
    }


def _plan_summary(calls: list[agent_tools.ToolCallRequest], planner: str) -> str:
    if not calls:
        return f"{planner} 未选择工具。"
    names = "、".join(call.name for call in calls)
    return f"{planner} 计划调用：{names}。"


def _stop_update(
    state: AgentGraphState,
    stop_reason: str,
    final_action: str,
    iteration: int,
) -> dict[str, Any]:
    return {
        "iterations": max(iteration, 0),
        "pending_tool_calls": [],
        "final_action": final_action,
        "next_action": "finish",
        "stop_reason": stop_reason,
        "graph_events": [
            *state.get("graph_events", []),
            _event(
                "plan",
                "stopped",
                max(iteration, 0),
                f"控制流停止：{stop_reason}",
                elapsed_ms=_elapsed_ms(state),
                stop_reason=stop_reason,
            ),
        ],
    }


def _control_meta(iterations: int, stop_reason: str) -> dict[str, Any]:
    usage = llm_service.current_usage_snapshot()
    return {
        "max_iterations": settings.AGENT_GRAPH_MAX_ITERATIONS,
        "timeout_seconds": settings.AGENT_GRAPH_TIMEOUT_SECONDS,
        "max_repeated_tool_failures": settings.AGENT_GRAPH_MAX_REPEATED_TOOL_FAILURES,
        "token_budget": settings.AGENT_GRAPH_TOKEN_BUDGET,
        "token_usage": usage.to_dict(),
        "iterations": iterations,
        "stop_reason": stop_reason,
    }


def _token_budget_exceeded() -> bool:
    budget = int(settings.AGENT_GRAPH_TOKEN_BUDGET or 0)
    if budget <= 0:
        return False
    return llm_service.current_usage_snapshot().total_tokens >= budget
