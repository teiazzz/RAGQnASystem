"""Neo4j GraphRAG path expansion.

GraphRAG here means a retrieval strategy over the existing medical knowledge
graph: anchor on NER entities, expand along schema-approved relationships, then
turn short graph paths into citable RAG evidence.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.services.rag_tokenizer import tokenize

logger = logging.getLogger(__name__)


SUPPORTED_ENTITY_LABELS = {
    "疾病",
    "药品",
    "食物",
    "检查项目",
    "科目",
    "疾病症状",
    "治疗方法",
    "药品商",
}

ONE_HOP_RELATIONS = (
    "疾病使用药品",
    "疾病宜吃食物",
    "疾病忌吃食物",
    "疾病所需检查",
    "疾病所属科目",
    "疾病的症状",
    "治疗的方法",
    "疾病并发疾病",
    "生产",
)

COMPLICATION_EXPANSION_RELATIONS = (
    "疾病所属科目",
    "疾病的症状",
    "疾病所需检查",
    "治疗的方法",
)

SYMPTOM_EXPANSION_RELATIONS = (
    "疾病所属科目",
    "疾病所需检查",
    "治疗的方法",
    "疾病并发疾病",
)

DRUG_EXPANSION_RELATIONS = ("疾病使用药品", "生产")

RELATION_QUERY_CUES = {
    "疾病所属科目": ("科室", "挂什么科", "挂号", "门诊", "属于哪个科", "急诊"),
    "疾病并发疾病": ("并发", "引发", "导致", "合并", "风险"),
    "疾病的症状": ("症状", "表现", "典型表现"),
    "疾病所需检查": ("检查", "检测", "化验", "诊断"),
    "疾病使用药品": ("药", "用药", "药品", "吃什么药"),
    "治疗的方法": ("治疗", "怎么办", "处理", "缓解"),
    "疾病宜吃食物": ("宜吃", "吃什么", "饮食", "适合吃"),
    "疾病忌吃食物": ("忌口", "不能吃", "忌吃", "饮食"),
    "生产": ("生产商", "厂家", "谁生产"),
}

PATH_KNOWLEDGE_PREFIX = "GraphRAG路径证据"


@dataclass(frozen=True)
class GraphRAGEvidence:
    """A short Neo4j path that can be converted into RAG context."""

    anchor_label: str
    anchor_name: str
    node_names: tuple[str, ...]
    node_labels: tuple[str, ...]
    relations: tuple[str, ...]
    score: float

    @property
    def hops(self) -> int:
        return len(self.relations)

    @property
    def signature(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return self.node_names, self.relations

    def render_path(self) -> str:
        parts: list[str] = []
        for idx, name in enumerate(self.node_names):
            label = self.node_labels[idx] if idx < len(self.node_labels) else "实体"
            parts.append(f"{label}:{name}")
            if idx < len(self.relations):
                parts.append(f"-[{self.relations[idx]}]-")
        return " ".join(parts)

    def to_knowledge(self) -> str:
        return (
            f"{PATH_KNOWLEDGE_PREFIX}({self.hops}跳): {self.render_path()}。"
            f"说明：从用户实体“{self.anchor_name}”沿 Neo4j 医疗知识图谱扩展得到，"
            "可作为疾病、症状、并发症、科室、检查、药品等结构化关系依据。"
        )


class GraphRAGService:
    """Schema-aware GraphRAG retriever over the existing Neo4j medical KG."""

    def __init__(self, graph) -> None:
        self.graph = graph

    def build_knowledge(
        self,
        query: str,
        entities: Mapping[str, str],
        limit: int = 8,
        per_entity_limit: int = 12,
    ) -> list[str]:
        """Return GraphRAG path evidence strings for hybrid RAG.

        The service is intentionally conservative: it expands only along known
        medical KG relationships and caps each entity to short paths.
        """
        paths = self.expand(query, entities, limit=limit, per_entity_limit=per_entity_limit)
        return [path.to_knowledge() for path in paths]

    def expand(
        self,
        query: str,
        entities: Mapping[str, str],
        limit: int = 8,
        per_entity_limit: int = 12,
    ) -> list[GraphRAGEvidence]:
        if not entities:
            return []

        evidences: list[GraphRAGEvidence] = []
        for raw_label, raw_name in entities.items():
            label = str(raw_label or "").strip()
            name = str(raw_name or "").strip()
            if not label or not name or label not in SUPPORTED_ENTITY_LABELS:
                continue
            evidences.extend(
                self._expand_entity(query, label, name, per_entity_limit=per_entity_limit)
            )

        deduped = self._dedupe(evidences)
        deduped.sort(key=lambda item: (item.score, -item.hops), reverse=True)
        return deduped[:limit]

    def _expand_entity(
        self,
        query: str,
        label: str,
        name: str,
        per_entity_limit: int,
    ) -> list[GraphRAGEvidence]:
        try:
            if label == "疾病":
                rows = [
                    *self._run_disease_one_hop(name, per_entity_limit),
                    *self._run_disease_complication_paths(name, per_entity_limit),
                ]
            elif label == "疾病症状":
                rows = self._run_symptom_paths(name, per_entity_limit)
            elif label == "药品":
                rows = self._run_drug_paths(name, per_entity_limit)
            else:
                rows = self._run_generic_one_hop(label, name, per_entity_limit)
        except Exception:
            logger.warning("GraphRAG 扩展失败: %s=%s", label, name, exc_info=True)
            return []

        paths = [
            self._evidence_from_row(query, label, name, row)
            for row in rows
            if row.get("nodes") and row.get("relationships")
        ]
        return [path for path in paths if path is not None]

    def _run_disease_one_hop(self, name: str, limit: int) -> list[dict]:
        cypher = """
        MATCH path = (start:`疾病` {名称: $name})-[r]->(end)
        WHERE type(r) IN $relations
        WITH path
        ORDER BY length(path) ASC
        LIMIT $limit
        RETURN
          [node IN nodes(path) | coalesce(node.名称, '')] AS nodes,
          [node IN nodes(path) | labels(node)] AS node_labels,
          [rel IN relationships(path) | type(rel)] AS relationships
        """
        return self._run_paths(cypher, name=name, limit=limit, relations=ONE_HOP_RELATIONS)

    def _run_disease_complication_paths(self, name: str, limit: int) -> list[dict]:
        cypher = """
        MATCH path = (start:`疾病` {名称: $name})-[r1:`疾病并发疾病`]->(mid:`疾病`)-[r2]->(end)
        WHERE type(r2) IN $relations AND end <> start
        WITH path
        ORDER BY length(path) ASC
        LIMIT $limit
        RETURN
          [node IN nodes(path) | coalesce(node.名称, '')] AS nodes,
          [node IN nodes(path) | labels(node)] AS node_labels,
          [rel IN relationships(path) | type(rel)] AS relationships
        """
        return self._run_paths(
            cypher,
            name=name,
            limit=limit,
            relations=COMPLICATION_EXPANSION_RELATIONS,
        )

    def _run_symptom_paths(self, name: str, limit: int) -> list[dict]:
        cypher = """
        MATCH path = (start:`疾病症状` {名称: $name})<-[:`疾病的症状`]-(disease:`疾病`)-[r2]->(end)
        WHERE type(r2) IN $relations
        WITH path
        ORDER BY length(path) ASC
        LIMIT $limit
        RETURN
          [node IN nodes(path) | coalesce(node.名称, '')] AS nodes,
          [node IN nodes(path) | labels(node)] AS node_labels,
          [rel IN relationships(path) | type(rel)] AS relationships
        """
        return self._run_paths(
            cypher,
            name=name,
            limit=limit,
            relations=SYMPTOM_EXPANSION_RELATIONS,
        )

    def _run_drug_paths(self, name: str, limit: int) -> list[dict]:
        cypher = """
        MATCH path = (start:`药品` {名称: $name})-[r]-(end)
        WHERE type(r) IN $relations
        WITH path
        ORDER BY length(path) ASC
        LIMIT $limit
        RETURN
          [node IN nodes(path) | coalesce(node.名称, '')] AS nodes,
          [node IN nodes(path) | labels(node)] AS node_labels,
          [rel IN relationships(path) | type(rel)] AS relationships
        """
        return self._run_paths(cypher, name=name, limit=limit, relations=DRUG_EXPANSION_RELATIONS)

    def _run_generic_one_hop(self, label: str, name: str, limit: int) -> list[dict]:
        safe_label = self._safe_label(label)
        cypher = f"""
        MATCH path = (start:`{safe_label}` {{名称: $name}})-[r]-(end)
        WHERE type(r) IN $relations
        WITH path
        ORDER BY length(path) ASC
        LIMIT $limit
        RETURN
          [node IN nodes(path) | coalesce(node.名称, '')] AS nodes,
          [node IN nodes(path) | labels(node)] AS node_labels,
          [rel IN relationships(path) | type(rel)] AS relationships
        """
        return self._run_paths(cypher, name=name, limit=limit, relations=ONE_HOP_RELATIONS)

    def _run_paths(
        self,
        cypher: str,
        *,
        name: str,
        limit: int,
        relations: Sequence[str],
    ) -> list[dict]:
        cursor = self.graph.run(
            cypher,
            parameters={
                "name": name,
                "limit": max(1, int(limit)),
                "relations": list(relations),
            },
        )
        return cursor.data() or []

    def _evidence_from_row(
        self,
        query: str,
        anchor_label: str,
        anchor_name: str,
        row: Mapping[str, object],
    ) -> GraphRAGEvidence | None:
        node_names = tuple(str(item).strip() for item in row.get("nodes", []) if item)
        relations = tuple(
            str(item).strip() for item in row.get("relationships", []) if item
        )
        if len(node_names) < 2 or not relations:
            return None

        node_labels = self._normalize_node_labels(row.get("node_labels", []), len(node_names))
        score = self._score_path(query, node_names, relations)
        return GraphRAGEvidence(
            anchor_label=anchor_label,
            anchor_name=anchor_name,
            node_names=node_names,
            node_labels=node_labels,
            relations=relations,
            score=score,
        )

    def _score_path(
        self,
        query: str,
        node_names: tuple[str, ...],
        relations: tuple[str, ...],
    ) -> float:
        score = 1.0 if len(relations) == 1 else 0.82
        normalized_query = re.sub(r"\s+", "", query)
        for relation in relations:
            cues = RELATION_QUERY_CUES.get(relation, ())
            if any(cue in normalized_query for cue in cues):
                score += 0.22

        path_tokens = set(tokenize(" ".join([*node_names, *relations])))
        query_tokens = set(tokenize(query))
        if query_tokens:
            score += 0.16 * (len(path_tokens & query_tokens) / len(query_tokens))
        return score

    def _normalize_node_labels(self, labels: object, expected_len: int) -> tuple[str, ...]:
        normalized: list[str] = []
        if isinstance(labels, Sequence) and not isinstance(labels, str):
            for item in labels:
                if isinstance(item, Sequence) and not isinstance(item, str):
                    selected = next(
                        (str(label) for label in item if str(label) in SUPPORTED_ENTITY_LABELS),
                        str(item[0]) if item else "实体",
                    )
                    normalized.append(selected)
                else:
                    normalized.append(str(item))
        while len(normalized) < expected_len:
            normalized.append("实体")
        return tuple(normalized[:expected_len])

    def _dedupe(self, evidences: list[GraphRAGEvidence]) -> list[GraphRAGEvidence]:
        by_signature: dict[tuple[tuple[str, ...], tuple[str, ...]], GraphRAGEvidence] = {}
        for evidence in evidences:
            current = by_signature.get(evidence.signature)
            if current is None or evidence.score > current.score:
                by_signature[evidence.signature] = evidence
        return list(by_signature.values())

    def _safe_label(self, label: str) -> str:
        if label not in SUPPORTED_ENTITY_LABELS:
            raise ValueError(f"Unsupported Neo4j label for GraphRAG: {label}")
        return label
