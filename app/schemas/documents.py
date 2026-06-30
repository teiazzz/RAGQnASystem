"""文档入库相关 Pydantic 模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class IndexCorpusRequest(BaseModel):
    path: str | None = None
    limit: int | None = Field(default=None, ge=1)
    chunk_size: int = Field(default=500, ge=100, le=2000)
    overlap: int = Field(default=50, ge=0, le=500)


class IndexCorpusResponse(BaseModel):
    records_seen: int
    total_chunks: int
    created_chunks: int
    skipped_chunks: int
    corpus_path: str


class UploadDocumentResponse(BaseModel):
    source_title: str
    total_chunks: int
    created_chunks: int
    skipped_chunks: int


class DocumentStats(BaseModel):
    chunks: int
