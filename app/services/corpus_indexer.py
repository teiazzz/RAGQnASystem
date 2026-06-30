"""医疗语料入库：JSONL → 递归切片 → embedding → document_chunks。"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import DocumentChunk
from app.services.bm25_service import get_bm25_service
from app.services.embedding_service import get_embedding_service
from app.services.text_splitter import RecursiveTextSplitter
from app.services.vector_store import update_pgvector_embedding

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIELD_SECTIONS: tuple[tuple[str, str], ...] = (
    ("desc", "疾病简介"),
    ("cause", "疾病病因"),
    ("prevent", "预防措施"),
    ("symptom", "典型症状"),
    ("easy_get", "易感人群"),
    ("get_way", "传播方式"),
    ("acompany", "并发疾病"),
    ("cure_department", "就诊科室"),
    ("cure_way", "治疗方法"),
    ("cure_lasttime", "治疗周期"),
    ("cured_prob", "治愈概率"),
    ("check", "检查项目"),
    ("do_eat", "宜吃食物"),
    ("not_eat", "忌吃食物"),
    ("recommand_drug", "推荐药品"),
    ("common_drug", "常用药品"),
    ("drug_detail", "药品详情"),
)


@dataclass(frozen=True)
class IndexingResult:
    records_seen: int
    total_chunks: int
    created_chunks: int
    skipped_chunks: int
    corpus_path: str


@dataclass(frozen=True)
class UploadedDocumentResult:
    source_title: str
    total_chunks: int
    created_chunks: int
    skipped_chunks: int


def resolve_corpus_path(path: str | None = None) -> Path:
    corpus_path = Path(path or settings.MEDICAL_CORPUS_PATH)
    if not corpus_path.is_absolute():
        corpus_path = PROJECT_ROOT / corpus_path
    return corpus_path


def iter_medical_records(path: Path, limit: int | None = None) -> Iterable[dict]:
    """逐行读取 medical JSONL；兼容行尾逗号。"""
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.rstrip(",")
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("跳过无法解析的语料行: %s", line[:120])
                continue
            yield record
            seen += 1
            if limit is not None and seen >= limit:
                return


def build_record_sections(record: dict[str, Any]) -> list[tuple[str, str]]:
    name = str(record.get("name") or "").strip()
    if not name:
        return []
    sections: list[tuple[str, str]] = []
    for field, section in FIELD_SECTIONS:
        body = stringify_value(record.get(field))
        if body:
            sections.append((section, f"{name}。{section}：{body}"))
    return sections


def stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "、".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def record_source_id(record: dict[str, Any]) -> str:
    oid = record.get("_id")
    if isinstance(oid, dict) and oid.get("$oid"):
        return str(oid["$oid"])
    return str(record.get("name") or hashlib.sha256(str(record).encode()).hexdigest())


def content_hash(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def index_medical_corpus(
    session: AsyncSession,
    path: str | None = None,
    limit: int | None = None,
    chunk_size: int = 500,
    overlap: int = 50,
    batch_size: int = 64,
) -> IndexingResult:
    """索引本地医疗语料，返回真实写入统计。"""
    corpus_path = resolve_corpus_path(path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"语料文件不存在: {corpus_path}")

    splitter = RecursiveTextSplitter(chunk_size=chunk_size, overlap=overlap)
    embedder = get_embedding_service()
    records_seen = 0
    total_chunks = 0
    created_chunks = 0
    pgvector_ok = settings.ENABLE_PGVECTOR
    batch: list[dict] = []

    async def flush_batch() -> None:
        nonlocal created_chunks, pgvector_ok, batch
        if not batch:
            return
        embeddings = await embedder.embed_documents([item["content"] for item in batch])
        rows = []
        hash_to_embedding: dict[str, list[float]] = {}
        for item, embedding in zip(batch, embeddings, strict=True):
            item = dict(item)
            item["embedding"] = embedding
            rows.append(item)
            hash_to_embedding[item["content_hash"]] = embedding

        stmt = (
            pg_insert(DocumentChunk)
            .values(rows)
            .on_conflict_do_nothing(index_elements=["content_hash"])
            .returning(DocumentChunk.id, DocumentChunk.content_hash)
        )
        result = await session.execute(stmt)
        inserted = result.all()
        created_chunks += len(inserted)

        if pgvector_ok:
            for chunk_id, chunk_hash in inserted:
                try:
                    async with session.begin_nested():
                        await update_pgvector_embedding(
                            session, int(chunk_id), hash_to_embedding[str(chunk_hash)]
                        )
                except Exception:
                    pgvector_ok = False
                    logger.warning(
                        "pgvector 写入失败，后续本次入库仅保留 JSONB embedding",
                        exc_info=True,
                    )
                    break

        await session.commit()
        batch = []

    for record in iter_medical_records(corpus_path, limit=limit):
        records_seen += 1
        source_id = record_source_id(record)
        source_title = str(record.get("name") or source_id)
        metadata = {
            "disease": source_title,
            "category": record.get("category") or [],
            "source_file": str(corpus_path.name),
        }
        for section, text in build_record_sections(record):
            for idx, chunk in enumerate(splitter.split_text(text)):
                total_chunks += 1
                batch.append(
                    {
                        "source_type": "medical_json",
                        "source_id": source_id,
                        "source_title": source_title,
                        "section": section,
                        "chunk_index": idx,
                        "content": chunk,
                        "content_hash": content_hash(source_id, section, str(idx), chunk),
                        "authority_level": "structured_medical_corpus",
                        "meta": metadata,
                    }
                )
                if len(batch) >= batch_size:
                    await flush_batch()

    await flush_batch()
    get_bm25_service().invalidate()
    return IndexingResult(
        records_seen=records_seen,
        total_chunks=total_chunks,
        created_chunks=created_chunks,
        skipped_chunks=total_chunks - created_chunks,
        corpus_path=str(corpus_path),
    )


async def index_text_document(
    session: AsyncSession,
    source_title: str,
    text: str,
    section: str = "上传文档",
    chunk_size: int = 500,
    overlap: int = 50,
) -> UploadedDocumentResult:
    """索引用户上传的纯文本/Markdown/JSON 文档。"""
    source_title = source_title.strip() or "uploaded_document"
    text = text.strip()
    if not text:
        raise ValueError("上传文档内容为空")

    splitter = RecursiveTextSplitter(chunk_size=chunk_size, overlap=overlap)
    chunks = splitter.split_text(text)
    source_id = hashlib.sha256(f"{source_title}|{text[:2048]}".encode("utf-8")).hexdigest()
    rows = [
        {
            "source_type": "uploaded_file",
            "source_id": source_id,
            "source_title": source_title,
            "section": section,
            "chunk_index": idx,
            "content": chunk,
            "content_hash": content_hash(source_id, section, str(idx), chunk),
            "authority_level": "uploaded_medical_document",
            "meta": {"source_file": source_title},
        }
        for idx, chunk in enumerate(chunks)
    ]

    embeddings = await get_embedding_service().embed_documents([row["content"] for row in rows])
    for row, embedding in zip(rows, embeddings, strict=True):
        row["embedding"] = embedding

    stmt = (
        pg_insert(DocumentChunk)
        .values(rows)
        .on_conflict_do_nothing(index_elements=["content_hash"])
        .returning(DocumentChunk.id, DocumentChunk.content_hash)
    )
    result = await session.execute(stmt)
    inserted = result.all()
    hash_to_embedding = {row["content_hash"]: row["embedding"] for row in rows}

    if settings.ENABLE_PGVECTOR:
        for chunk_id, chunk_hash in inserted:
            try:
                async with session.begin_nested():
                    await update_pgvector_embedding(
                        session, int(chunk_id), hash_to_embedding[str(chunk_hash)]
                    )
            except Exception:
                logger.warning("上传文档 pgvector 写入失败，已保留 JSONB embedding", exc_info=True)
                break

    await session.commit()
    get_bm25_service().invalidate()
    created_chunks = len(inserted)
    return UploadedDocumentResult(
        source_title=source_title,
        total_chunks=len(chunks),
        created_chunks=created_chunks,
        skipped_chunks=len(chunks) - created_chunks,
    )
