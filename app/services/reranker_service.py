"""Reranker 精排服务。"""

from __future__ import annotations

import asyncio
import logging

from app.core.config import settings
from app.services.rag_tokenizer import tokenize
from app.services.rag_types import RetrievedSource

logger = logging.getLogger(__name__)


class RerankerService:
    def __init__(self) -> None:
        self.provider = settings.RERANKER_PROVIDER
        self._model = None
        if self.provider == "sentence_transformers":
            try:
                from sentence_transformers import CrossEncoder  # type: ignore

                self._model = CrossEncoder(settings.RERANKER_MODEL)
                logger.info("Reranker 使用 CrossEncoder: %s", settings.RERANKER_MODEL)
            except Exception:
                logger.warning("CrossEncoder 不可用，降级为本地 reranker", exc_info=True)
                self.provider = "local"

    async def rerank(
        self, query: str, candidates: list[RetrievedSource], top_k: int
    ) -> list[RetrievedSource]:
        if not candidates:
            return []
        if self.provider == "sentence_transformers" and self._model is not None:
            return await asyncio.to_thread(
                self._cross_encoder_rerank, query, candidates, top_k
            )
        return self._local_rerank(query, candidates, top_k)

    def _cross_encoder_rerank(
        self, query: str, candidates: list[RetrievedSource], top_k: int
    ) -> list[RetrievedSource]:
        pairs = [(query, item.content) for item in candidates]
        scores = self._model.predict(pairs)  # type: ignore[union-attr]
        for item, score in zip(candidates, scores, strict=True):
            item.rerank_score = float(score)
        return sorted(candidates, key=lambda item: item.rerank_score, reverse=True)[:top_k]

    def _local_rerank(
        self, query: str, candidates: list[RetrievedSource], top_k: int
    ) -> list[RetrievedSource]:
        query_tokens = set(tokenize(query))
        for item in candidates:
            content_tokens = set(tokenize(item.content))
            overlap = (
                len(query_tokens & content_tokens) / max(len(query_tokens), 1)
                if content_tokens
                else 0.0
            )
            authority_bonus = 0.35 if item.source_type == "kg" else 0.05
            item.rerank_score = (
                0.60 * item.fused_score
                + 0.30 * overlap
                + 0.10 * item.kg_score
                + authority_bonus
            )
        return sorted(candidates, key=lambda item: item.rerank_score, reverse=True)[:top_k]


_reranker_service: RerankerService | None = None


def get_reranker_service() -> RerankerService:
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = RerankerService()
    return _reranker_service
