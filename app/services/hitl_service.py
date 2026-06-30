"""HITL（Human-In-The-Loop）人在回路服务：高危医疗场景拦截与转人工。

医疗场景下 Agent 不能随意下结论，需要在以下场景强制转人工：
1. **高危急症**：胸痛、呼吸困难、大出血、意识障碍、疑似中风等
2. **自伤/自杀风险**：用户表达自杀意图或服毒等
3. **复杂用药决策**：多药相互作用、孕产妇/儿童用药、剂量调整
4. **持续工具失败**：Agent 连续调用工具失败超过 N 次
5. **用户明确要求人工**

HITL 设计点：
- 提前拦截：在 Agent 工具规划阶段就识别高危关键词，直接触发 escalate_to_human
- 分级响应：emergency（立即急诊）> urgent（尽快人工）> routine（可选人工）
- 审计日志：记录所有转人工事件，供后续分析和质控

面试讲解：
- "医疗场景不能让 Agent 随便下结论，高危症状直接拦截转人工/急诊，这是 HITL"
- "用规则前置拦截（胸痛、呼吸困难等关键词）+ LLM 二次确认，双重保险"
- "LangGraph 的中断/恢复机制天然支持 HITL，等人工审核后可以继续"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# 高危关键词：立即转急诊
HIGH_RISK_EMERGENCY_KEYWORDS = {
    "胸痛",
    "胸闷",
    "心前区疼痛",
    "呼吸困难",
    "喘不上气",
    "窒息",
    "大出血",
    "止不住血",
    "吐血",
    "呕血",
    "便血",
    "昏迷",
    "意识不清",
    "晕倒",
    "抽搐",
    "癫痫发作",
    "疑似中风",
    "口角歪斜",
    "一侧肢体无力",
    "半边身体不能动",
    "突然说不出话",
    "剧烈头痛",
    "服毒",
    "自杀",
    "想死",
    "不想活了",
    "休克",
    "严重创伤",
    "骨折穿刺",
    "高处坠落",
    "车祸",
}

# 紧急关键词：建议尽快就医或人工介入
URGENT_KEYWORDS = {
    "高热不退",
    "持续发烧",
    "体温超过39",
    "持续腹痛",
    "剧烈腹痛",
    "血压很高",
    "血压180",
    "血糖很高",
    "血糖20",
    "严重脱水",
    "严重过敏",
    "全身皮疹",
    "呼吸急促",
    "心跳过快",
    "心跳过慢",
    "孕妇出血",
    "产后大出血",
    "婴儿高热",
    "婴儿抽搐",
}

# 复杂用药场景：需人工审核
COMPLEX_MEDICATION_KEYWORDS = {
    "孕妇",
    "怀孕",
    "哺乳期",
    "婴儿",
    "新生儿",
    "儿童用药",
    "多种药一起吃",
    "药物相互作用",
    "肾功能不全",
    "肝功能不全",
    "透析",
    "器官移植",
    "化疗",
}


@dataclass
class HITLDecision:
    """HITL 决策结果。"""

    should_escalate: bool
    urgency: str  # emergency / urgent / routine / none
    reason: str
    matched_keywords: list[str]
    recommendation: str

    def to_meta(self) -> dict[str, Any]:
        return {
            "should_escalate": self.should_escalate,
            "urgency": self.urgency,
            "reason": self.reason,
            "matched_keywords": self.matched_keywords,
            "recommendation": self.recommendation,
        }


@dataclass
class HITLEvent:
    """转人工事件记录（用于审计和分析）。"""

    user_id: int
    conversation_id: int | None
    query: str
    urgency: str
    reason: str
    matched_keywords: list[str]
    agent_trace: dict[str, Any]
    created_at: datetime


class HITLService:
    """HITL 服务：高危场景识别 + 转人工决策 + 事件审计。"""

    async def assess_hitl(
        self,
        query: str,
        entities: dict[str, str] | None = None,
        agent_failed_tools: list[str] | None = None,
    ) -> HITLDecision:
        """评估是否需要转人工/急诊。"""
        entities = entities or {}
        agent_failed_tools = agent_failed_tools or []

        matched_emergency = self._match_keywords(query, HIGH_RISK_EMERGENCY_KEYWORDS)
        matched_urgent = self._match_keywords(query, URGENT_KEYWORDS)
        matched_complex_med = self._match_keywords(
            query, COMPLEX_MEDICATION_KEYWORDS
        )

        # 1. 高危急症：立即转急诊
        if matched_emergency:
            return HITLDecision(
                should_escalate=True,
                urgency="emergency",
                reason=f"用户描述包含高危急症症状：{'、'.join(matched_emergency)}",
                matched_keywords=matched_emergency,
                recommendation="立即拨打 120 急救电话或前往最近的急诊科，不要延误。",
            )

        # 2. 紧急症状：建议尽快就医
        if matched_urgent:
            return HITLDecision(
                should_escalate=True,
                urgency="urgent",
                reason=f"用户症状需要尽快医疗干预：{'、'.join(matched_urgent)}",
                matched_keywords=matched_urgent,
                recommendation="建议 24 小时内就医，或立即联系线上医生人工问诊。",
            )

        # 3. 复杂用药场景：转人工审核
        if matched_complex_med and self._is_medication_query(query):
            return HITLDecision(
                should_escalate=True,
                urgency="urgent",
                reason=f"涉及高风险用药场景：{'、'.join(matched_complex_med)}",
                matched_keywords=matched_complex_med,
                recommendation="此类用药问题必须咨询医生或药师，不能依赖自动问答。",
            )

        # 4. Agent 工具连续失败：降级转人工
        if len(agent_failed_tools) >= 2:
            return HITLDecision(
                should_escalate=True,
                urgency="routine",
                reason=f"Agent 工具调用连续失败（{', '.join(agent_failed_tools)}），当前自动处理能力不足",
                matched_keywords=[],
                recommendation="系统暂时无法自动处理您的问题，建议联系人工客服。",
            )

        # 5. 用户明确要求人工
        if self._user_requests_human(query):
            return HITLDecision(
                should_escalate=True,
                urgency="routine",
                reason="用户明确要求转人工",
                matched_keywords=["转人工", "找医生", "人工客服"],
                recommendation="正在为您转接人工服务...",
            )

        # 无需转人工
        return HITLDecision(
            should_escalate=False,
            urgency="none",
            reason="未检测到需要人工介入的风险",
            matched_keywords=[],
            recommendation="",
        )

    async def log_hitl_event(
        self,
        session: AsyncSession,
        user_id: int,
        conversation_id: int | None,
        query: str,
        decision: HITLDecision,
        agent_trace: dict[str, Any] | None = None,
    ) -> None:
        """记录转人工事件到数据库（用于审计和质量分析）。"""
        if not decision.should_escalate:
            return

        agent_trace = agent_trace or {}
        event = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query": query[:500],
            "urgency": decision.urgency,
            "reason": decision.reason,
            "matched_keywords": decision.matched_keywords,
            "agent_trace": agent_trace,
            "created_at": datetime.utcnow(),
        }

        # 简化实现：写入 JSON 字段或独立 hitl_events 表（当前演示用日志）
        logger.warning(
            "HITL 事件触发: user_id=%s, urgency=%s, reason=%s",
            user_id,
            decision.urgency,
            decision.reason,
            extra={"event": event},
        )

        # 生产环境应存入专门的 hitl_events 表
        try:
            await session.execute(
                text(
                    """
                    INSERT INTO hitl_events (user_id, conversation_id, query, urgency, reason, matched_keywords, agent_trace, created_at)
                    VALUES (:user_id, :conversation_id, :query, :urgency, :reason, :matched_keywords::jsonb, :agent_trace::jsonb, :created_at)
                """
                ),
                {
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "query": query[:500],
                    "urgency": decision.urgency,
                    "reason": decision.reason,
                    "matched_keywords": decision.matched_keywords,
                    "agent_trace": agent_trace,
                    "created_at": event["created_at"],
                },
            )
            await session.commit()
        except Exception:
            # hitl_events 表可能不存在，回退到日志
            logger.info("HITL 事件记录失败（表可能不存在），已记录到日志", exc_info=True)

    def _match_keywords(self, text: str, keywords: set[str]) -> list[str]:
        """匹配关键词，返回命中列表。"""
        matched = [kw for kw in keywords if kw in text]
        return matched

    def _is_medication_query(self, query: str) -> bool:
        """判断是否为用药相关问题。"""
        med_signals = {"药", "吃", "用药", "剂量", "用量", "服用", "能不能吃"}
        return any(signal in query for signal in med_signals)

    def _user_requests_human(self, query: str) -> bool:
        """判断用户是否明确要求人工。"""
        human_requests = {
            "转人工",
            "人工客服",
            "找医生",
            "咨询医生",
            "真人",
            "人工服务",
            "接入人工",
        }
        return any(req in query for req in human_requests)


_hitl_service_instance: HITLService | None = None


def get_hitl_service() -> HITLService:
    """单例模式获取 HITL 服务。"""
    global _hitl_service_instance
    if _hitl_service_instance is None:
        _hitl_service_instance = HITLService()
    return _hitl_service_instance
