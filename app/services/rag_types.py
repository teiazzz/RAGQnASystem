"""RAG 检索结果的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievedSource:
    """一次检索召回的候选来源。"""

    chunk_id: int | None
    source_type: str
    source_title: str
    section: str
    content: str
    authority_level: str = "medical_corpus"
    metadata: dict = field(default_factory=dict)
    vector_score: float = 0.0
    bm25_score: float = 0.0
    kg_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0
    citation_id: int | None = None

    @property
    def key(self) -> str:
        if self.chunk_id is not None:
            return f"chunk:{self.chunk_id}"
        return f"{self.source_type}:{self.source_title}:{self.section}:{hash(self.content)}"

    def to_meta(self) -> dict:
        """返回 SSE meta 可直接序列化的来源信息。"""
        return {
            "citation_id": self.citation_id,
            "chunk_id": self.chunk_id,
            "source_type": self.source_type,
            "source_title": self.source_title,
            "section": self.section,
            "authority_level": self.authority_level,
            "score": round(self.rerank_score or self.fused_score, 4),
            "vector_score": round(self.vector_score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "kg_score": round(self.kg_score, 4),
            "content_preview": self.content[:180],
            "content": self.content,
            "metadata": self.metadata,
        }
