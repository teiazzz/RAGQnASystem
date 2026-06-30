"""文档/语料入库接口。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func, select

from app.api.deps import SessionDep, get_current_admin
from app.db.models import DocumentChunk, User
from app.schemas.documents import (
    DocumentStats,
    IndexCorpusRequest,
    IndexCorpusResponse,
    UploadDocumentResponse,
)
from app.services.corpus_indexer import index_medical_corpus, index_text_document

router = APIRouter(prefix="/documents", tags=["文档入库"])
AdminUser = Annotated[User, Depends(get_current_admin)]


@router.post("/index-medical-corpus", response_model=IndexCorpusResponse)
async def index_builtin_medical_corpus(
    req: IndexCorpusRequest,
    user: AdminUser,  # noqa: ARG001
    session: SessionDep,
) -> IndexCorpusResponse:
    """索引项目自带 medical JSONL 语料。"""
    result = await index_medical_corpus(
        session,
        path=req.path,
        limit=req.limit,
        chunk_size=req.chunk_size,
        overlap=req.overlap,
    )
    return IndexCorpusResponse(**result.__dict__)


@router.post("/upload", response_model=UploadDocumentResponse)
async def upload_document(
    user: AdminUser,  # noqa: ARG001
    session: SessionDep,
    file: UploadFile = File(...),
) -> UploadDocumentResponse:
    """上传文本类医疗文档并入库。"""
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="仅支持 UTF-8 文本文件",
        ) from exc
    result = await index_text_document(
        session,
        source_title=file.filename or "uploaded_document.txt",
        text=text,
    )
    return UploadDocumentResponse(**result.__dict__)


@router.get("/stats", response_model=DocumentStats)
async def document_stats(user: AdminUser, session: SessionDep) -> DocumentStats:  # noqa: ARG001
    count = await session.scalar(select(func.count()).select_from(DocumentChunk))
    return DocumentStats(chunks=int(count or 0))
