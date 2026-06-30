"""Embedding 服务。

生产配置可切到 ``sentence_transformers`` + ``BAAI/bge-m3``；默认使用
HashingVectorizer 做离线 fallback，避免开发环境下载大模型后才能验证 RAG 流程。
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
        self.provider = settings.EMBEDDING_PROVIDER
        self.dim = settings.EMBEDDING_DIM
        self._model = None
        self._vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(1, 2),
            n_features=self.dim,
            alternate_sign=False,
            norm="l2",
        )
        if self.provider == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
                logger.info("Embedding 使用 SentenceTransformer: %s", settings.EMBEDDING_MODEL)
            except Exception:
                logger.warning(
                    "SentenceTransformer 不可用，降级为 local_hashing embedding",
                    exc_info=True,
                )
                self.provider = "local_hashing"

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    async def encode(self, text: str) -> list[float]:
        """Backward-compatible single-text embedding API."""
        return await self.embed_query(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_documents_sync, texts)

    def _embed_documents_sync(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.provider == "sentence_transformers" and self._model is not None:
            vectors = self._model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(vectors, dtype=np.float32).tolist()

        matrix = self._vectorizer.transform(texts)
        matrix = normalize(matrix, norm="l2", copy=False)
        return matrix.astype(np.float32).toarray().tolist()


_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
