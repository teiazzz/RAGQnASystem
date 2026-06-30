"""用户长期记忆服务：向量检索 + 相关性过滤防止记忆污染。

记忆污染问题的解决方案：
1. **向量相似度过滤**：只召回和当前 query embedding 相似度 > 阈值的记忆
2. **关键词启发式**：优先召回和当前实体/关键词重合的记忆
3. **重要性加权**：高优先级记忆（过敏史、慢性病）降低召回阈值
4. **按对话隔离**：新对话不自动继承旧对话的临时上下文，只召回持久健康档案

面试讲解点：
- 长期记忆用向量检索 + 相似度阈值（默认 0.7）防止无关记忆污染
- 过敏史/慢性病等高危记忆降低召回阈值到 0.5，确保不遗漏
- 短期记忆就是消息滑动窗口，读最近 N 轮
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import Message, UserMemory
from app.services.embedding_service import get_embedding_service
from app.services.rag_tokenizer import tokenize

logger = logging.getLogger(__name__)


@dataclass
class RetrievedMemory:
    """召回的长期记忆，带相似度和来源信息。"""

    id: int
    category: str
    content: str
    importance: int
    similarity: float
    relevance_keywords: list[str] = field(default_factory=list)
    source_conversation_id: int | None = None

    def to_meta(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "content": self.content,
            "importance": self.importance,
            "similarity": self.similarity,
            "relevance_keywords": self.relevance_keywords,
        }


class MemoryService:
    """管理用户长期记忆：创建、向量检索、相关性过滤。"""

    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.default_similarity_threshold = settings.MEMORY_SIMILARITY_THRESHOLD
        self.high_priority_threshold = settings.MEMORY_HIGH_PRIORITY_THRESHOLD

    async def save_memory(
        self,
        session: AsyncSession,
        user_id: int,
        content: str,
        category: str = "health_fact",
        importance: int = 5,
        source_conversation_id: int | None = None,
        relevance_keywords: list[str] | None = None,
    ) -> UserMemory:
        """保存一条长期记忆，自动生成 embedding。"""
        embedding = await self.embedding_service.encode(content)
        if not relevance_keywords:
            relevance_keywords = self._extract_keywords(content)

        memory = UserMemory(
            user_id=user_id,
            content=content,
            category=category,
            importance=importance,
            embedding=embedding,
            relevance_keywords=relevance_keywords,
            source_conversation_id=source_conversation_id,
        )
        session.add(memory)
        await session.commit()
        await session.refresh(memory)
        logger.info(
            "保存用户记忆: user_id=%s, category=%s, importance=%s",
            user_id,
            category,
            importance,
        )
        return memory

    async def retrieve_relevant_memories(
        self,
        session: AsyncSession,
        user_id: int,
        query: str,
        entities: dict[str, str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedMemory]:
        """检索和当前 query 相关的长期记忆（向量相似度 + 关键词过滤）。"""
        entities = entities or {}
        query_embedding = await self.embedding_service.encode(query)
        query_tokens = set(tokenize(query))
        entity_keywords = {
            value.strip()
            for value in entities.values()
            if value and len(value.strip()) >= 2
        }

        # 读取用户所有记忆
        result = await session.execute(
            select(UserMemory).where(UserMemory.user_id == user_id)
        )
        all_memories = result.scalars().all()

        if not all_memories:
            return []

        scored: list[tuple[UserMemory, float]] = []
        for memory in all_memories:
            if not memory.embedding:
                continue

            # 计算向量相似度
            similarity = self._cosine_similarity(query_embedding, memory.embedding)

            # 根据重要性动态调整阈值
            threshold = (
                self.high_priority_threshold
                if memory.importance >= 8
                else self.default_similarity_threshold
            )

            # 关键词启发式加成：如果 query 实体命中记忆关键词，降低阈值
            keyword_boost = 0.0
            if memory.relevance_keywords and entity_keywords:
                memory_keywords = {kw.strip() for kw in memory.relevance_keywords}
                if entity_keywords & memory_keywords:
                    keyword_boost = 0.15
                    logger.debug(
                        "记忆 %s 命中关键词，加成 %.2f", memory.id, keyword_boost
                    )

            adjusted_score = similarity + keyword_boost

            # 过滤低相关性记忆
            if adjusted_score >= threshold:
                scored.append((memory, adjusted_score))

        # 按调整后分数排序，取 top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for memory, score in scored[:top_k]:
            results.append(
                RetrievedMemory(
                    id=memory.id,
                    category=memory.category,
                    content=memory.content,
                    importance=memory.importance,
                    similarity=score,
                    relevance_keywords=memory.relevance_keywords or [],
                    source_conversation_id=memory.source_conversation_id,
                )
            )
        logger.info(
            "用户 %s 召回记忆 %d 条（过滤前 %d 条）",
            user_id,
            len(results),
            len(all_memories),
        )
        return results

    async def load_short_term_memory(
        self,
        session: AsyncSession,
        conversation_id: int | None,
        current_message_id: int | None,
        limit: int = 5,
    ) -> list[dict[str, str]]:
        """短期记忆：读取最近 N 轮对话（滑动窗口）。"""
        if conversation_id is None or current_message_id is None:
            return []

        result = await session.execute(
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.id < current_message_id,
            )
            .order_by(Message.id.desc())
            .limit(limit)
        )
        messages = list(result.scalars().all())
        messages.reverse()

        return [
            {
                "role": message.role,
                "content": self._clean_message_content(message.content),
            }
            for message in messages
            if message.content
        ]

    def build_memory_prompt_context(
        self, memories: list[RetrievedMemory]
    ) -> str:
        """构造记忆上下文注入 prompt。"""
        if not memories:
            return ""

        lines = ["<用户健康档案>"]
        for memory in memories:
            category_label = self._category_label(memory.category)
            lines.append(f"- [{category_label}] {memory.content}")
        lines.append(
            "</用户健康档案>"
            "<注意>上述健康档案来自用户历史记录，回答时需结合当前问题，"
            "不要在无关场景强行使用档案内容；涉及用药、禁忌时务必提醒用户遵医嘱。</注意>"
        )
        return "\n".join(lines)

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """计算余弦相似度。"""
        if len(vec1) != len(vec2):
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)

    def _extract_keywords(self, content: str) -> list[str]:
        """简单启发式抽取关键词（医学实体、疾病、药品）。"""
        medical_patterns = [
            r"过敏",
            r"糖尿病",
            r"高血压",
            r"冠心病",
            r"哮喘",
            r"慢阻肺",
            r"肾病",
            r"肝病",
            r"癌症",
            r"肿瘤",
            r"\w+炎",
            r"\w+病",
            r"\w+症",
            r"青霉素",
            r"头孢",
            r"阿司匹林",
            r"二甲双胍",
            r"胰岛素",
        ]
        keywords = set()
        for pattern in medical_patterns:
            matches = re.findall(pattern, content)
            keywords.update(matches)
        # 限制关键词数量
        return list(keywords)[:10]

    def _clean_message_content(self, content: str) -> str:
        """清理消息内容，截断过长文本。"""
        cleaned = re.sub(r"\s+", " ", content or "").strip()
        return cleaned[:800]

    def _category_label(self, category: str) -> str:
        """记忆类型的中文标签。"""
        labels = {
            "health_fact": "健康事实",
            "allergy": "过敏史",
            "chronic_disease": "慢性病",
            "medication_history": "用药史",
            "decision": "医疗决策",
        }
        return labels.get(category, category)


_memory_service_instance: MemoryService | None = None


def get_memory_service() -> MemoryService:
    """单例模式获取记忆服务。"""
    global _memory_service_instance
    if _memory_service_instance is None:
        _memory_service_instance = MemoryService()
    return _memory_service_instance
