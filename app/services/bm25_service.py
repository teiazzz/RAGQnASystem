"""轻量 BM25 检索。

用于补足向量检索对药品名、疾病名、数字等精确匹配不稳定的问题。
索引懒加载在进程内缓存；文档入库后调用 ``invalidate`` 重建。
"""

from __future__ import annotations

import asyncio
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import DocumentChunk
from app.services.rag_tokenizer import tokenize


@dataclass(frozen=True)
class BM25Hit:
    chunk_id: int
    score: float


class BM25Index:
    def __init__(self, rows: list[tuple[int, str]]) -> None:
        self.doc_ids = [row[0] for row in rows]
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}
        self.avg_doc_len = 0.0
        self._build(rows)

    def _build(self, rows: list[tuple[int, str]]) -> None:
        doc_freq: Counter[str] = Counter()
        for doc_idx, (_, content) in enumerate(rows):
            counts = Counter(tokenize(content))
            self.doc_len.append(sum(counts.values()) or 1)
            for term, tf in counts.items():
                self.postings[term].append((doc_idx, tf))
                doc_freq[term] += 1

        doc_count = max(len(rows), 1)
        self.avg_doc_len = sum(self.doc_len) / doc_count if rows else 1.0
        self.idf = {
            term: math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def search(self, query: str, top_k: int) -> list[BM25Hit]:
        if not self.doc_ids:
            return []
        k1 = 1.5
        b = 0.75
        scores: defaultdict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_idx, tf in postings:
                dl = self.doc_len[doc_idx]
                denom = tf + k1 * (1 - b + b * dl / self.avg_doc_len)
                scores[doc_idx] += idf * tf * (k1 + 1) / denom

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [BM25Hit(chunk_id=self.doc_ids[idx], score=score) for idx, score in ranked]


class BM25Service:
    def __init__(self) -> None:
        self._index: BM25Index | None = None
        self._doc_count: int | None = None
        self._build_lock = asyncio.Lock()

    def invalidate(self) -> None:
        self._index = None
        self._doc_count = None

    async def search(
        self, session: AsyncSession, query: str, top_k: int
    ) -> list[BM25Hit]:
        index = await self._get_index(session)
        return index.search(query, top_k)

    async def warmup(self, session: AsyncSession) -> None:
        """Build the cached BM25 index before the first user query."""
        await self._get_index(session)

    async def _get_index(self, session: AsyncSession) -> BM25Index:
        doc_count = await session.scalar(select(func.count()).select_from(DocumentChunk))
        doc_count = int(doc_count or 0)
        if self._index is not None and self._doc_count == doc_count:
            return self._index

        async with self._build_lock:
            if self._index is not None and self._doc_count == doc_count:
                return self._index
            rows = await session.execute(
                select(DocumentChunk.id, DocumentChunk.content)
                .order_by(DocumentChunk.id)
                .limit(settings.RAG_BM25_MAX_DOCS)
            )
            self._index = BM25Index([(int(row[0]), str(row[1])) for row in rows.all()])
            self._doc_count = doc_count
        return self._index


_bm25_service: BM25Service | None = None


def get_bm25_service() -> BM25Service:
    global _bm25_service
    if _bm25_service is None:
        _bm25_service = BM25Service()
    return _bm25_service
