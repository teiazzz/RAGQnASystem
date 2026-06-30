"""Structured AgentState snapshot for interviewable Agent design.

The LangGraph runtime state is optimized for control flow. This module builds a
stable, JSON-serializable state snapshot grouped by the seven categories usually
asked in Agent interviews: conversation, user profile, task, reasoning, actions,
memory, and control.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from app.core.config import settings
from app.services import agent_tools

SCHEMA_VERSION = "agent_state.v1"
STATE_CATEGORIES = [
    "conversation",
    "user_profile",
    "task",
    "reasoning",
    "actions",
    "memory",
    "control",
]

SPECIAL_POPULATION_KEYWORDS = {
    "孕妇",
    "怀孕",
    "妊娠",
    "哺乳",
    "儿童",
    "孩子",
    "小孩",
    "婴儿",
    "老人",
    "老年",
}

MENTIONED_CONDITION_KEYWORDS = {
    "高血压",
    "糖尿病",
    "哮喘",
    "冠心病",
    "慢阻肺",
    "肾病",
    "肝病",
    "肿瘤",
    "癌",
}

DURATION_PATTERN = re.compile(r"(\d+\s*(天|日|周|月|年|小时|分钟)|半天|一天|几天|一周|长期|反复|持续)")
SEVERITY_PATTERN = re.compile(r"(剧烈|严重|轻微|明显|加重|缓解|无法忍受|高热|很痛|很疼)")


def build_state_snapshot(
    *,
    messages: list[dict[str, str]],
    query: str,
    entities: dict[str, str] | None = None,
    original_query: str | None = None,
    standalone_query: str | None = None,
    rewritten_query: str | None = None,
    user_profile: dict[str, Any] | None = None,
    intent_names: list[str] | None = None,
    raw_intent_response: str = "",
    plan: list[str] | None = None,
    scratchpad: list[str] | None = None,
    tool_calls: Iterable[agent_tools.ToolCallRequest] | None = None,
    observations: Iterable[agent_tools.ToolObservation] | None = None,
    control: dict[str, Any] | None = None,
    final_action: str = "none",
    stop_reason: str = "",
) -> dict[str, Any]:
    """Build a bounded AgentState snapshot safe for SSE debug metadata."""

    entities = entities or {}
    intent_names = intent_names or []
    plan = plan or []
    scratchpad = scratchpad or []
    calls = list(tool_calls or [])
    obs = list(observations or [])
    control = control or {}
    all_text = " ".join(
        [original_query or "", standalone_query or "", rewritten_query or "", query]
        + [message.get("content", "") for message in messages]
    )
    slots = extract_slots(entities)
    risk_flags = extract_risk_flags(all_text)
    missing_slots = infer_missing_slots(query, slots, risk_flags)

    return {
        "schema_version": SCHEMA_VERSION,
        "categories": STATE_CATEGORIES,
        "conversation": {
            "messages": [_safe_message(message) for message in messages[-8:]],
            "history_turns": max(len(messages) - 1, 0),
            "current_query": query[:500],
            "original_query": (original_query or query)[:500],
            "standalone_query": (standalone_query or query)[:500],
            "rewritten_query": (rewritten_query or query)[:500],
        },
        "user_profile": build_user_profile(
            all_text,
            entities,
            explicit_profile=user_profile,
        ),
        "task": {
            "intent": {
                "primary": intent_names[0] if intent_names else "",
                "candidates": intent_names,
                "raw": raw_intent_response[:600],
            },
            "slots": slots,
            "missing_slots": missing_slots,
            "risk_flags": risk_flags,
        },
        "reasoning": {
            "plan": plan,
            "scratchpad": scratchpad[-8:],
            "scratchpad_size": len(scratchpad),
        },
        "actions": {
            "tool_calls": [call.to_meta() for call in calls],
            "observations": [observation.to_meta() for observation in obs],
            "tool_call_count": len(calls),
            "observation_count": len(obs),
        },
        "memory": {
            "short_term": {
                "messages": [_safe_message(message) for message in messages[-6:]],
                "window_size": min(len(messages), 6),
            },
            "working": {
                "active_slots": slots,
                "risk_flags": risk_flags,
                "pending_missing_slots": missing_slots,
            },
            "long_term": {
                "enabled": False,
                "items": [],
                "note": "长期健康记忆将在阶段三 P1 记忆系统中接入。",
            },
        },
        "control": {
            "iterations": int(control.get("iterations", 0) or 0),
            "max_iterations": int(
                control.get("max_iterations", settings.AGENT_GRAPH_MAX_ITERATIONS) or 0
            ),
            "timeout_seconds": float(
                control.get("timeout_seconds", settings.AGENT_GRAPH_TIMEOUT_SECONDS) or 0
            ),
            "token_budget": int(
                control.get("token_budget", settings.AGENT_GRAPH_TOKEN_BUDGET) or 0
            ),
            "token_usage": control.get(
                "token_usage",
                {
                    "model": settings.DEEPSEEK_MODEL,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0,
                },
            ),
            "max_repeated_tool_failures": int(
                control.get(
                    "max_repeated_tool_failures",
                    settings.AGENT_GRAPH_MAX_REPEATED_TOOL_FAILURES,
                )
                or 0
            ),
            "max_tool_calls": settings.AGENT_TOOL_MAX_CALLS,
            "final_action": final_action,
            "stop_reason": stop_reason or str(control.get("stop_reason") or ""),
        },
    }


def with_task_intents(
    snapshot: dict[str, Any],
    *,
    intent_names: list[str],
    raw_intent_response: str = "",
) -> dict[str, Any]:
    """Return a copy of a state snapshot with resolved intent names filled in."""

    updated = dict(snapshot)
    task = dict(updated.get("task") or {})
    intent = dict(task.get("intent") or {})
    intent["primary"] = intent_names[0] if intent_names else intent.get("primary", "")
    intent["candidates"] = intent_names
    if raw_intent_response:
        intent["raw"] = raw_intent_response[:600]
    task["intent"] = intent
    updated["task"] = task
    return updated


def build_user_profile(
    text: str,
    entities: dict[str, str],
    explicit_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a conservative profile from auth context and mentioned facts."""

    explicit_profile = explicit_profile or {}
    condition_mentions = _dedupe(
        [
            *[
                value
                for key, value in entities.items()
                if key == "疾病" and value
            ],
            *[keyword for keyword in MENTIONED_CONDITION_KEYWORDS if keyword in text],
        ]
    )
    drug_mentions = _dedupe(
        [value for key, value in entities.items() if key == "药品" and value]
    )
    allergy_mentions = _extract_allergy_mentions(text)
    special_population_mentions = _dedupe(
        [keyword for keyword in SPECIAL_POPULATION_KEYWORDS if keyword in text]
    )
    return {
        "account": {
            "user_id": explicit_profile.get("user_id"),
            "role": explicit_profile.get("role", "anonymous"),
            "is_admin": bool(explicit_profile.get("is_admin", False)),
        },
        "medical_facts": {
            "condition_mentions": condition_mentions,
            "drug_mentions": drug_mentions,
            "allergy_mentions": allergy_mentions,
            "special_population_mentions": special_population_mentions,
        },
        "source": "auth_context_and_current_turn_mentions",
        "confidence": "mentioned_not_verified",
    }


def extract_slots(entities: dict[str, str]) -> dict[str, list[str]]:
    """Map project NER labels into Agent task slots."""

    mapping = {
        "疾病": "diseases",
        "疾病症状": "symptoms",
        "药品": "drugs",
        "科目": "departments",
        "检查项目": "checks",
        "治疗方法": "treatments",
        "食物": "foods",
        "药品商": "drug_producers",
    }
    slots: dict[str, list[str]] = {
        "diseases": [],
        "symptoms": [],
        "drugs": [],
        "departments": [],
        "checks": [],
        "treatments": [],
        "foods": [],
        "drug_producers": [],
    }
    for label, slot_name in mapping.items():
        value = entities.get(label)
        if not value:
            continue
        slots[slot_name].extend(_split_entity_value(str(value)))
    return {key: _dedupe(value) for key, value in slots.items()}


def extract_risk_flags(text: str) -> dict[str, list[str]]:
    high_risk = [keyword for keyword in agent_tools.HIGH_RISK_KEYWORDS if keyword in text]
    urgent = [keyword for keyword in agent_tools.URGENT_KEYWORDS if keyword in text]
    return {
        "high_risk_keywords": _dedupe(high_risk),
        "urgent_keywords": _dedupe(urgent),
    }


def infer_missing_slots(
    query: str,
    slots: dict[str, list[str]],
    risk_flags: dict[str, list[str]],
) -> list[str]:
    """Infer missing fields important for safe triage; never blocks answering alone."""

    missing: list[str] = []
    normalized = re.sub(r"\s+", "", query)
    asks_triage = any(keyword in query for keyword in agent_tools.TRIAGE_KEYWORDS)
    has_symptom = bool(slots.get("symptoms")) or any(
        keyword in query
        for keyword in (
            "痛",
            "疼",
            "发烧",
            "发热",
            "咳嗽",
            "头晕",
            "恶心",
            "呕吐",
            "腹泻",
            "胸闷",
            "胸痛",
        )
    )
    if asks_triage or has_symptom or risk_flags.get("high_risk_keywords"):
        if not has_symptom:
            missing.append("symptoms")
        if not DURATION_PATTERN.search(normalized):
            missing.append("duration")
        if not SEVERITY_PATTERN.search(normalized):
            missing.append("severity")
    return missing


def _safe_message(message: dict[str, str]) -> dict[str, str]:
    role = str(message.get("role", ""))[:20]
    content = re.sub(r"\s+", " ", str(message.get("content", ""))).strip()
    return {"role": role, "content": content[:300]}


def _split_entity_value(value: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"[、,，;/；\s]+", value)
        if item.strip()
    ]


def _extract_allergy_mentions(text: str) -> list[str]:
    mentions: list[str] = []
    for match in re.finditer(r"对([\u4e00-\u9fffA-Za-z0-9]{1,20})过敏", text):
        mentions.append(_clean_allergy_mention(match.group(1)))
    for match in re.finditer(r"([\u4e00-\u9fffA-Za-z0-9]{1,20})过敏", text):
        mentions.append(_clean_allergy_mention(match.group(1)))
    return _dedupe(mentions)


def _clean_allergy_mention(text: str) -> str:
    return re.sub(r"^(我|本人|患者|用户)?对", "", text.strip())


def _dedupe(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = str(item).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result
