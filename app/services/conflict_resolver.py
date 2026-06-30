"""多源信息冲突检测与轻量消解。

当前项目的来源主要是 KG、结构化医学语料和用户上传文档。这里先用可测试的规则
覆盖医疗问答中最常见的互斥场景：允许/禁忌、需要/无需、会/不会。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.services.rag_types import RetrievedSource

ConflictStance = Literal["permissive", "restrictive", "positive_fact", "negative_fact"]


AUTHORITY_POLICY = "权威性(KG/指南 > 结构化医学语料 > 上传文档/网页) + 时效性 + 医疗安全"

HIGH_AUTHORITY_HINTS = (
    "official_guideline",
    "clinical_guideline",
    "national_guideline",
    "guideline",
    "指南",
    "卫健委",
    "国家卫生",
    "nhc",
)

AUTHORITY_SCORES = {
    "official_guideline": 100,
    "national_guideline": 98,
    "clinical_guideline": 95,
    "knowledge_graph": 90,
    "structured_medical_corpus": 82,
    "medical_corpus": 78,
    "uploaded_medical_document": 60,
    "uploaded_file": 55,
    "web_page": 40,
    "web": 40,
}

CONFLICT_QUERY_CUES = (
    "能不能",
    "可不可以",
    "能否",
    "是否",
    "可以",
    "能吃",
    "能喝",
    "要不要",
    "该不该",
    "禁忌",
    "忌",
    "不建议",
    "安全吗",
    "风险",
    "副作用",
    "不良反应",
    "相互作用",
    "冲突",
    "会不会",
    "传染",
    "需要吗",
    "饮酒",
    "喝酒",
)

RESTRICTIVE_CUES = (
    "禁用",
    "禁忌",
    "不建议",
    "不推荐",
    "应避免",
    "避免",
    "禁止",
    "不可",
    "不能",
    "不宜",
    "慎用",
)

PERMISSIVE_CUES = (
    "可以",
    "可用于",
    "可使用",
    "可服用",
    "适宜",
    "适合",
    "推荐",
    "宜",
    "能吃",
    "能喝",
)

NEGATIVE_FACT_CUES = (
    "无传染性",
    "不传染",
    "不会传染",
    "不需要",
    "无需",
    "不会导致",
    "没有必要",
)

POSITIVE_FACT_CUES = (
    "有传染性",
    "会传染",
    "需要",
    "应进行",
    "可能导致",
    "可导致",
    "会导致",
)


@dataclass(frozen=True)
class StancedSource:
    source: RetrievedSource
    stance: ConflictStance
    cues: tuple[str, ...]


@dataclass(frozen=True)
class SourceConflict:
    conflict_type: str
    severity: str
    summary: str
    resolution: str
    involved_citation_ids: list[int]
    preferred_citation_id: int | None
    preferred_source_title: str
    policy: str = AUTHORITY_POLICY

    def to_meta(self) -> dict:
        return {
            "conflict_type": self.conflict_type,
            "severity": self.severity,
            "summary": self.summary,
            "resolution": self.resolution,
            "involved_citation_ids": self.involved_citation_ids,
            "preferred_citation_id": self.preferred_citation_id,
            "preferred_source_title": self.preferred_source_title,
            "policy": self.policy,
        }


@dataclass(frozen=True)
class ConflictResolution:
    sources: list[RetrievedSource]
    conflicts: list[SourceConflict]

    @property
    def has_conflict(self) -> bool:
        return bool(self.conflicts)

    @property
    def summary(self) -> str:
        if not self.conflicts:
            return ""
        return "；".join(conflict.summary for conflict in self.conflicts)

    def to_meta(self) -> dict:
        return {
            "has_conflict": self.has_conflict,
            "summary": self.summary,
            "conflicts": [conflict.to_meta() for conflict in self.conflicts],
        }


def empty_conflict_meta() -> dict:
    return {"has_conflict": False, "summary": "", "conflicts": []}


def resolve_source_conflicts(
    query: str, sources: list[RetrievedSource]
) -> ConflictResolution:
    """检测来源分歧；若存在冲突，将优先来源提前并重排 citation_id。"""
    if len(sources) < 2 or not _query_needs_conflict_check(query):
        _assign_citation_ids(sources)
        return ConflictResolution(sources=sources, conflicts=[])

    stanced = [
        stance
        for source in sources[:8]
        if (stance := classify_source_stance(query, source)) is not None
    ]
    if not _has_conflicting_stances(stanced):
        _assign_citation_ids(sources)
        return ConflictResolution(sources=sources, conflicts=[])

    involved_sources = [item.source for item in stanced]
    preferred_source = max(involved_sources, key=source_priority_tuple)
    reordered = _reorder_sources_for_conflict(sources, involved_sources, preferred_source)
    _assign_citation_ids(reordered)
    conflicts = [_build_conflict(query, stanced, preferred_source)]
    return ConflictResolution(sources=reordered, conflicts=conflicts)


def classify_source_stance(
    query: str, source: RetrievedSource
) -> StancedSource | None:
    """识别单条来源在当前问题上的保守/许可或事实正反倾向。"""
    text = f"{source.section}。{source.content}"
    restrictive_hits = _find_cues(text, RESTRICTIVE_CUES)
    text_without_restrictive = _remove_cues(text, restrictive_hits)
    permissive_hits = _find_cues(text_without_restrictive, PERMISSIVE_CUES)

    negative_fact_hits = _find_cues(text, NEGATIVE_FACT_CUES)
    text_without_negative = _remove_cues(text, negative_fact_hits)
    positive_fact_hits = _find_cues(text_without_negative, POSITIVE_FACT_CUES)

    if _query_has_safety_intent(query):
        if restrictive_hits:
            return StancedSource(source, "restrictive", tuple(restrictive_hits))
        if permissive_hits:
            return StancedSource(source, "permissive", tuple(permissive_hits))

    if _query_has_fact_intent(query):
        if negative_fact_hits:
            return StancedSource(source, "negative_fact", tuple(negative_fact_hits))
        if positive_fact_hits:
            return StancedSource(source, "positive_fact", tuple(positive_fact_hits))

    return None


def build_conflict_prompt(resolution: ConflictResolution) -> str:
    if not resolution.has_conflict:
        return ""
    parts = [
        "<注意>检索来源存在互斥或明显分歧。回答时必须先遵循医疗安全原则，"
        f"按{AUTHORITY_POLICY}选择主要依据；不要把低权威来源与高权威来源强行调和。"
        "如果分歧会影响用药、禁忌、饮食或就医决策，必须明确提醒用户以医生/权威指南为准。</注意>"
    ]
    for conflict in resolution.conflicts:
        involved = "、".join(f"[{cid}]" for cid in conflict.involved_citation_ids)
        preferred = (
            f"[{conflict.preferred_citation_id}] {conflict.preferred_source_title}"
            if conflict.preferred_citation_id is not None
            else conflict.preferred_source_title
        )
        parts.append(
            f"<提示>冲突消解：{conflict.conflict_type}；涉及来源：{involved}。"
            f"优先依据：{preferred}。{conflict.resolution}</提示>"
        )
    return "".join(parts)


def source_priority_tuple(source: RetrievedSource) -> tuple[int, int, float]:
    """权威性优先，其次时效性，最后参考检索分数。"""
    score = source.rerank_score or source.fused_score
    return (source_authority_score(source), source_year(source) or 0, score)


def source_authority_score(source: RetrievedSource) -> int:
    text = " ".join(
        [
            source.authority_level or "",
            source.source_type or "",
            source.source_title or "",
            str(source.metadata or ""),
        ]
    ).lower()
    if any(hint.lower() in text for hint in HIGH_AUTHORITY_HINTS):
        return 100
    for key, score in AUTHORITY_SCORES.items():
        if key in text:
            return score
    return 30


def source_year(source: RetrievedSource) -> int | None:
    candidates = [
        source.source_title,
        source.section,
        source.authority_level,
        str(source.metadata or ""),
    ]
    for candidate in candidates:
        match = re.search(r"(19\d{2}|20\d{2})", candidate or "")
        if match:
            year = int(match.group(1))
            if 1900 <= year <= 2100:
                return year
    return None


def _build_conflict(
    query: str, stanced: list[StancedSource], preferred_source: RetrievedSource
) -> SourceConflict:
    conflict_type = _infer_conflict_type(query, stanced)
    involved = [item.source.citation_id or 0 for item in stanced]
    involved = [cid for cid in involved if cid > 0]
    preferred_cid = preferred_source.citation_id
    summary = (
        f"检测到{conflict_type}，已按{AUTHORITY_POLICY}优先采用"
        f"来源[{preferred_cid}] {preferred_source.source_title}。"
    )
    resolution = (
        "答案应呈现分歧点；对可能影响安全的结论，采用更保守、更权威的来源，"
        "并建议用户咨询医生或参考最新指南。"
    )
    return SourceConflict(
        conflict_type=conflict_type,
        severity="high",
        summary=summary,
        resolution=resolution,
        involved_citation_ids=involved,
        preferred_citation_id=preferred_cid,
        preferred_source_title=preferred_source.source_title,
    )


def _has_conflicting_stances(stanced: list[StancedSource]) -> bool:
    stances = {item.stance for item in stanced}
    return (
        {"permissive", "restrictive"}.issubset(stances)
        or {"positive_fact", "negative_fact"}.issubset(stances)
    )


def _query_needs_conflict_check(query: str) -> bool:
    normalized = re.sub(r"\s+", "", query)
    return any(cue in normalized for cue in CONFLICT_QUERY_CUES)


def _query_has_safety_intent(query: str) -> bool:
    normalized = re.sub(r"\s+", "", query)
    return any(
        cue in normalized
        for cue in (
            "能不能",
            "可不可以",
            "可以",
            "能吃",
            "能喝",
            "禁忌",
            "忌",
            "不建议",
            "安全吗",
            "风险",
            "副作用",
            "不良反应",
            "相互作用",
            "饮酒",
            "喝酒",
        )
    )


def _query_has_fact_intent(query: str) -> bool:
    normalized = re.sub(r"\s+", "", query)
    return any(
        cue in normalized
        for cue in ("是否", "会不会", "传染", "需要吗", "要不要", "该不该")
    )


def _infer_conflict_type(query: str, stanced: list[StancedSource]) -> str:
    normalized = re.sub(r"\s+", "", query)
    stances = {item.stance for item in stanced}
    if {"positive_fact", "negative_fact"}.issubset(stances):
        if "传染" in normalized:
            return "传播/传染性结论冲突"
        return "事实性结论冲突"
    if any(cue in normalized for cue in ("药", "服用", "副作用", "不良反应", "相互作用")):
        return "用药安全建议冲突"
    if any(cue in normalized for cue in ("吃", "喝", "饮食", "饮酒", "喝酒")):
        return "饮食/生活方式建议冲突"
    return "医疗建议冲突"


def _find_cues(text: str, cues: tuple[str, ...]) -> list[str]:
    return [cue for cue in cues if cue in text]


def _remove_cues(text: str, cues: list[str]) -> str:
    for cue in cues:
        text = text.replace(cue, "")
    return text


def _reorder_sources_for_conflict(
    sources: list[RetrievedSource],
    involved_sources: list[RetrievedSource],
    preferred_source: RetrievedSource,
) -> list[RetrievedSource]:
    involved_keys = {source.key for source in involved_sources}
    selected_keys: set[str] = set()
    reordered = [preferred_source]
    selected_keys.add(preferred_source.key)

    other_involved = sorted(
        [source for source in involved_sources if source.key != preferred_source.key],
        key=source_priority_tuple,
        reverse=True,
    )
    for source in other_involved:
        if source.key in selected_keys:
            continue
        reordered.append(source)
        selected_keys.add(source.key)

    for source in sources:
        if source.key in selected_keys or source.key in involved_keys:
            continue
        reordered.append(source)
        selected_keys.add(source.key)
    return reordered


def _assign_citation_ids(sources: list[RetrievedSource]) -> None:
    for idx, source in enumerate(sources, start=1):
        source.citation_id = idx
