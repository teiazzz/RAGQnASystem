"""聊天编排服务：意图识别 → NER → 知识图谱检索 → 拼装 prompt → 流式生成。

逻辑抽自基座 ``webui.py`` 的 ``Intent_Recognition()`` 与 ``generate_prompt()``，
剥离全部 Streamlit 依赖，改为异步：

* 意图识别(LLM, async) 与 NER(线程池) **并行**执行；
* KG 查询(py2neo 同步) 通过 ``asyncio.to_thread`` 走线程池；
* prompt 文本与基座保持一致，确保医疗问答质量零回归。

后续阶段二/三：检索升级为向量+KG 混合检索，编排升级为 LangGraph Agent。
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re

from intent_router import execute_intents

from sqlalchemy import select

from app.core.config import settings
from app.db.models import Message
from app.db.session import async_session_maker
from app.services import agent_graph, agent_state, agent_tools, llm_service
from app.services.conflict_resolver import (
    build_conflict_prompt,
    empty_conflict_meta,
    resolve_source_conflicts,
)
from app.services.hitl_service import get_hitl_service
from app.services.hybrid_retriever import build_citation_prompt, get_hybrid_retriever
from app.services.kg_service import get_kg_service
from app.services.memory_service import get_memory_service
from app.services.ner_service import get_ner_service
from app.services.rag_tokenizer import tokenize
from app.services.rag_types import RetrievedSource

logger = logging.getLogger(__name__)


NO_INFO_MARKERS = ("图谱中无信息", "知识库异常", "查找失败", "没有相关信息")

CHITCHAT_PATTERNS = {
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "hello",
    "hi",
    "在吗",
    "谢谢",
    "谢了",
    "早上好",
    "上午好",
    "下午好",
    "晚上好",
    "晚安",
    "再见",
    "拜拜",
    "你是谁",
    "你能做什么",
}

MEDICAL_SIGNAL_KEYWORDS = {
    "疾病",
    "症状",
    "病因",
    "预防",
    "运放",  # 常见输入法误选：预防
    "治疗",
    "治愈",
    "检查",
    "诊断",
    "药",
    "用药",
    "用量",
    "副作用",
    "禁忌",
    "疫苗",
    "预苗",
    "医生",
    "医院",
    "科室",
    "手术",
    "护理",
    "康复",
    "饮食",
    "忌口",
    "过敏",
    "感染",
    "发炎",
    "炎症",
    "发热",
    "发烧",
    "咳嗽",
    "头痛",
    "头疼",
    "胸痛",
    "胸闷",
    "腹痛",
    "腹泻",
    "便秘",
    "恶心",
    "呕吐",
    "头晕",
    "失眠",
    "睡眠",
    "呼吸",
    "心悸",
    "血压",
    "血糖",
    "怀孕",
    "月经",
    "皮肤",
    "肿瘤",
    "癌",
    "感冒",
    "流感",
    "高血压",
    "糖尿病",
}

CONTEXT_DEPENDENT_KEYWORDS = {
    "它",
    "他",
    "她",
    "这个",
    "那个",
    "这些",
    "那些",
    "上述",
    "前面",
    "刚才",
    "该病",
    "该药",
    "这种病",
    "这种药",
    "副作用",
    "禁忌",
    "用量",
    "怎么吃",
    "多久",
    "还要",
    "还有",
}


# 意图识别 prompt：16 类意图 + 10 个 few-shot（含 CoT）。原 webui.py:50-119，{query} 为占位符。
INTENT_PROMPT_TEMPLATE = """
阅读下列提示，回答问题（问题在输入的最后）:
当你试图识别用户问题中的查询意图时，你需要仔细分析问题，并在16个预定义的查询类别中一一进行判断。对于每一个类别，思考用户的问题是否含有与该类别对应的意图。如果判断用户的问题符合某个特定类别，就将该类别加入到输出列表中。这样的方法要求你对每一个可能的查询意图进行系统性的考虑和评估，确保没有遗漏任何一个可能的分类。

**查询类别**
- "查询疾病简介"
- "查询疾病病因"
- "查询疾病预防措施"
- "查询疾病治疗周期"
- "查询治愈概率"
- "查询疾病易感人群"
- "查询疾病所需药品"
- "查询疾病宜吃食物"
- "查询疾病忌吃食物"
- "查询疾病所需检查项目"
- "查询疾病所属科目"
- "查询疾病的症状"
- "查询疾病的治疗方法"
- "查询疾病的并发疾病"
- "查询药品的生产商"

在处理用户的问题时，请按照以下步骤操作：
- 仔细阅读用户的问题。
- 对照上述查询类别列表，依次考虑每个类别是否与用户问题相关。
- 如果用户问题明确或隐含地包含了某个类别的查询意图，请将该类别的描述添加到输出列表中。
- 确保最终的输出列表包含了所有与用户问题相关的类别描述。

以下是一些含有隐晦性意图的例子，每个例子都采用了输入和输出格式，并包含了对你进行思维链形成的提示：
**示例1：**
输入："睡眠不好，这是为什么？"
输出：["查询疾病简介","查询疾病病因"]  # 这个问题隐含地询问了睡眠不好的病因
**示例2：**
输入："感冒了，怎么办才好？"
输出：["查询疾病简介","查询疾病所需药品", "查询疾病的治疗方法"]  # 用户可能既想知道应该吃哪些药品，也想了解治疗方法
**示例3：**
输入："跑步后膝盖痛，需要吃点什么？"
输出：["查询疾病简介","查询疾病宜吃食物", "查询疾病所需药品"]  # 这个问题可能既询问宜吃的食物，也可能在询问所需药品
**示例4：**
输入："我怎样才能避免冬天的流感和感冒？"
输出：["查询疾病简介","查询疾病预防措施"]  # 询问的是预防措施，但因为提到了两种疾病，这里隐含的是对共同预防措施的询问
**示例5：**
输入："头疼是什么原因，应该怎么办？"
输出：["查询疾病简介","查询疾病病因", "查询疾病的治疗方法"]  # 用户询问的是头疼的病因和治疗方法
**示例6：**
输入："如何知道自己是不是有艾滋病？"
输出：["查询疾病简介","查询疾病所需检查项目","查询疾病病因"]  # 用户想知道自己是不是有艾滋病，一定一定要进行相关检查，这是根本性的！其次是查看疾病的病因，看看自己的行为是不是和病因重合。
**示例7：**
输入："我该怎么知道我自己是否得了21三体综合症呢？"
输出：["查询疾病简介","查询疾病所需检查项目","查询疾病病因"]  # 用户想知道自己是不是有21三体综合症，一定一定要进行相关检查(比如染色体)，这是根本性的！其次是查看疾病的病因。
**示例8：**
输入："感冒了，怎么办？"
输出：["查询疾病简介","查询疾病的治疗方法","查询疾病所需药品","查询疾病所需检查项目","查询疾病宜吃食物"]  # 问怎么办，首选治疗方法。然后是要给用户推荐一些药，最后让他检查一下身体。同时，也推荐一下食物。
**示例9：**
输入："癌症会引发其他疾病吗？"
输出：["查询疾病简介","查询疾病的并发疾病","查询疾病简介"]  # 显然，用户问的是疾病并发疾病，随后可以给用户科普一下癌症简介。
**示例10：**
输入："葡萄糖浆的生产者是谁？葡萄糖浆是谁生产的？"
输出：["查询药品的生产商"]  # 显然，用户想要问药品的生产商
通过上述例子，我们希望你能够形成一套系统的思考过程，以准确识别出用户问题中的所有可能查询意图。请仔细分析用户的问题，考虑到其可能的多重含义，确保输出反映了所有相关的查询意图。

**注意：**
- 你的所有输出，都必须在这个范围内上述**查询类别**范围内，不可创造新的名词与类别！
- 参考上述5个示例：在输出查询意图对应的列表之后，请紧跟着用"#"号开始的注释，简短地解释为什么选择这些意图选项。注释应当直接跟在列表后面，形成一条连续的输出。
- 你的输出的类别数量不应该超过5，如果确实有很多个，请你输出最有可能的5个！同时，你的解释不宜过长，但是得富有条理性。

现在，你已经知道如何解决问题了，请你解决下面这个问题并将结果输出！
问题输入："{query}"
输出的时候请确保输出内容都在**查询类别**中出现过。确保输出类别个数**不要超过5个**！确保你的解释和合乎逻辑的！注意，如果用户询问了有关疾病的问题，一般都要先介绍一下疾病，也就是有"查询疾病简介"这个需求。
再次检查你的输出都包含在**查询类别**:"查询疾病简介"、"查询疾病病因"、"查询疾病预防措施"、"查询疾病治疗周期"、"查询治愈概率"、"查询疾病易感人群"、"查询疾病所需药品"、"查询疾病宜吃食物"、"查询疾病忌吃食物"、"查询疾病所需检查项目"、"查询疾病所属科目"、"查询疾病的症状"、"查询疾病的治疗方法"、"查询疾病的并发疾病"、"查询药品的生产商"。
"""


async def recognize_intent(query: str) -> str:
    """调用 LLM 做意图识别，返回原始文本（由 intent_router 解析）。"""
    prompt = INTENT_PROMPT_TEMPLATE.replace("{query}", query)
    resp = await llm_service.generate(prompt)
    logger.debug("意图识别结果: %s", resp)
    return resp


def _clean_query(query: str) -> str:
    """压缩空白，避免把用户换行直接带入检索和 prompt。"""
    return re.sub(r"\s+", " ", query).strip()


def _normalized_short_text(query: str) -> str:
    return re.sub(r"[\s，。！？!?、,.~～]+", "", query.strip().lower())


def _is_chitchat_query(query: str) -> bool:
    """识别寒暄和助手能力询问，这类请求不走 RAG 溯源。"""
    normalized = _normalized_short_text(query)
    if not normalized:
        return False
    if normalized in CHITCHAT_PATTERNS:
        return True
    return len(normalized) <= 14 and any(
        pattern in normalized for pattern in CHITCHAT_PATTERNS
    )


def _looks_like_medical_query(query: str) -> bool:
    """轻量医学信号判断；不确定时再交给 NER。"""
    normalized = _normalized_short_text(query)
    return any(keyword in normalized for keyword in MEDICAL_SIGNAL_KEYWORDS)


def _is_no_info_text(text: str) -> bool:
    return any(marker in text for marker in NO_INFO_MARKERS)


async def _should_use_medical_rag(query: str) -> bool:
    """判断是否需要进入医学 RAG。

    明确的寒暄/普通聊天直接跳过 RAG；疑似医学问题先看关键词，再用 NER 做补充。
    """
    if _is_chitchat_query(query) and not _looks_like_medical_query(query):
        return False
    if _looks_like_medical_query(query):
        return True
    try:
        entities = await get_ner_service().extract(query)
    except Exception:
        logger.warning("NER 路由判断失败，按普通聊天处理", exc_info=True)
        return False
    return bool(entities)


def _may_need_contextual_rewrite(query: str) -> bool:
    """判断当前轮是否可能依赖历史上下文。"""
    normalized = _normalized_short_text(query)
    if not normalized or _is_chitchat_query(query):
        return False
    return len(normalized) <= 24 or any(
        keyword in normalized for keyword in CONTEXT_DEPENDENT_KEYWORDS
    )


async def load_recent_history(
    conversation_id: int | None,
    current_message_id: int | None,
    limit: int = 8,
) -> list[dict[str, str]]:
    """读取当前用户消息之前的最近几轮对话，供多轮 query 改写使用。"""
    if conversation_id is None or current_message_id is None:
        return []
    async with async_session_maker() as session:
        rows = await session.scalars(
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.id < current_message_id,
            )
            .order_by(Message.id.desc())
            .limit(limit)
        )
        messages = list(rows)
    messages.reverse()
    return [
        {
            "role": message.role,
            "content": _clean_query(message.content)[:800],
        }
        for message in messages
        if message.content
    ]


def _format_history_for_rewrite(history: list[dict[str, str]]) -> str:
    lines = []
    for item in history:
        role = "用户" if item["role"] == "user" else "助手"
        lines.append(f"{role}: {item['content']}")
    return "\n".join(lines)


async def rewrite_query_with_history(
    query: str,
    history: list[dict[str, str]],
) -> str:
    """结合最近对话，把省略/指代问题改写成可独立检索的问题。"""
    if not history or not _may_need_contextual_rewrite(query):
        return query

    prompt = (
        "你是多轮医疗 RAG 的查询改写器。请根据最近对话，把当前用户问题改写成"
        "不依赖上下文、可以独立检索的完整问题。"
        "如果当前问题已经完整，或历史无法确定指代对象，原样输出当前问题。"
        "只输出改写后的问题，不要回答问题，不要解释。"
        "不要新增病情、检查结果、药品剂量或治疗建议。\n"
        "示例：\n"
        "历史：用户: 阿司匹林是什么？\n"
        "当前问题：它有什么副作用？\n"
        "输出：阿司匹林有什么副作用？\n\n"
        f"最近对话：\n{_format_history_for_rewrite(history)}\n"
        f"当前问题：{query}\n"
        "输出："
    )
    try:
        rewritten = await llm_service.generate(prompt, temperature=0.0)
    except Exception:
        logger.warning("多轮 query 改写失败，使用当前问题", exc_info=True)
        return query

    rewritten = rewritten.strip().strip("`\"'“”‘’")
    rewritten = re.sub(r"^(输出[:：]|改写后[:：])", "", rewritten).strip()
    rewritten = rewritten.splitlines()[0].strip() if rewritten else ""
    if not rewritten or len(rewritten) > 2000:
        return query
    return rewritten


def _build_direct_chat_prompt(query: str) -> str:
    return (
        "<指令>你是一个医疗健康问答助手，也可以进行自然的日常交流。"
        "当前用户没有提出需要知识库溯源的专业医学问题，请像正常聊天一样简洁、友好地回复。"
        "不要使用引用编号，不要声称查询了知识库。</指令>"
        f"<用户问题>{query}</用户问题>"
    )


def _build_hitl_emergency_prompt(query: str, hitl_decision) -> str:
    """HITL 高危急症拦截 prompt：直接建议急诊，不做诊断。"""
    return (
        "<指令>你是医疗导诊助手。系统 HITL（人在回路）安全机制已触发高危急症拦截。"
        "请直接、简洁地告诉用户：根据描述可能存在急症风险，建议立即拨打 120 急救电话或前往最近的急诊科。"
        "不要给出确定诊断，不要建议复杂自救步骤，不要推荐具体药物或剂量。"
        "保持冷静专业的语气，避免过度恐慌但必须明确风险。不要使用引用编号。</指令>"
        f"<HITL拦截原因>{hitl_decision.reason}</HITL拦截原因>"
        f"<匹配关键词>{'、'.join(hitl_decision.matched_keywords)}</匹配关键词>"
        f"<用户问题>{query}</用户问题>"
    )


def _empty_tool_trace(
    planner: str = "skipped",
    *,
    query: str = "",
    messages: list[dict[str, str]] | None = None,
    user_profile: dict | None = None,
    original_query: str | None = None,
    standalone_query: str | None = None,
) -> agent_tools.AgentToolTrace:
    trace = agent_tools.AgentToolTrace(
        available_tools=list(agent_tools.TOOL_SPECS),
        planner=planner,
        orchestrator="langgraph",
        stop_reason="skipped",
        final_action_override="none",
    )
    trace.state_snapshot = agent_state.build_state_snapshot(
        messages=messages or ([{"role": "user", "content": query}] if query else []),
        query=query,
        original_query=original_query or query,
        standalone_query=standalone_query or query,
        rewritten_query=query,
        user_profile=user_profile or {},
        control=trace.control,
        final_action=trace.final_action,
        stop_reason=trace.stop_reason,
    )
    return trace


def _build_agent_clarification_prompt(
    original_query: str,
    trace: agent_tools.AgentToolTrace,
) -> str:
    tool_context = agent_tools.format_tool_observations_for_prompt(trace)
    return (
        tool_context
        + "<指令>你是医疗导诊助手。Agent 工具判断当前信息不足，不能直接给出诊断或用药建议。"
        "请只向用户提出一个最关键的澄清问题，问题应围绕主要症状、持续时间、严重程度、"
        "年龄/特殊人群或既往病史之一。不要使用引用编号。</指令>"
        f"<用户问题>{original_query}</用户问题>"
    )


def _build_agent_escalation_prompt(
    original_query: str,
    trace: agent_tools.AgentToolTrace,
) -> str:
    tool_context = agent_tools.format_tool_observations_for_prompt(trace)
    return (
        tool_context
        + "<指令>你是医疗导诊助手。Agent 工具已触发高危转人工/急诊兜底。"
        "请直接、简洁地告诉用户可能存在急症风险，建议立即拨打急救电话或前往急诊，"
        "同时避免给出确定诊断、复杂自救步骤或具体用药剂量。不要使用引用编号。</指令>"
        f"<用户问题>{original_query}</用户问题>"
    )


def _build_agent_control_stop_prompt(
    original_query: str,
    trace: agent_tools.AgentToolTrace,
) -> str:
    stop_reason = trace.stop_reason or "control_stop"
    progress = "\n".join(agent_tools.trace_knowledge_items(trace)) or "本轮没有可靠工具结果。"
    return (
        "<指令>你是医疗导诊助手。Agent 控制流保护已触发，不能继续循环调用工具。"
        "请用简短中文告诉用户：当前自动处理已停止，并说明停止原因；"
        "如果涉及急症风险，建议急诊或人工确认；否则建议用户补充关键信息后再问。"
        "不要编造引用来源，不要输出[1]、[2]这类引用编号。</指令>"
        f"<停止原因>{stop_reason}</停止原因>"
        f"<已完成进展>{progress}</已完成进展>"
        f"<用户问题>{original_query}</用户问题>"
    )


def _prepend_tool_context(prompt: str, trace: agent_tools.AgentToolTrace) -> str:
    tool_context = agent_tools.format_tool_observations_for_prompt(trace)
    if not tool_context:
        return prompt
    return tool_context + prompt


def _build_model_fallback_prompt(
    original_query: str,
    standalone_query: str,
    rewritten_query: str,
) -> str:
    rewrite_note = (
        f"系统将用户问题改写为：{rewritten_query}"
        if rewritten_query != original_query
        else "系统未改写用户问题。"
    )
    context_note = (
        f"结合对话历史补全后的独立问题：{standalone_query}"
        if standalone_query != original_query
        else "当前问题不需要历史补全。"
    )
    return (
        "<指令>你是一个谨慎的医疗健康问答助手。"
        "本地知识库和可引用来源不足以直接回答用户问题，必须先说明："
        "“目前的知识储备无法回答该问题，已转为AI模型查询（以下回答没有引用来源）。”"
        "随后基于模型能力给出通用、审慎、可执行的回答。"
        "不要编造引用来源，不要输出[1]、[2]这类引用编号。"
        "涉及急重症、用药剂量、孕产妇、儿童或慢病个体化治疗时，要提醒及时咨询医生或就医。</指令>"
        f"<原始用户问题>{original_query}</原始用户问题>"
        f"<多轮补全说明>{context_note}</多轮补全说明>"
        f"<检索改写说明>{rewrite_note}</检索改写说明>"
    )


async def rewrite_medical_query(query: str) -> str:
    """医学检索 query 改写：纠正明显错别字和口语化表达，不扩写病情。"""
    prompt = (
        "你是医学问答系统的检索查询改写器。请修正用户问题中的明显错别字、"
        "输入法误选和口语化表达，使其更适合医学知识库检索。"
        "必须保留原意，不新增症状、疾病、检查、药品或治疗建议。"
        "如果无需改写，原样输出。只输出改写后的问题，不要解释。\n"
        "示例：感冒怎么运放 -> 感冒怎么预防？\n"
        "示例：高血压平时怎么运放 -> 高血压平时怎么预防？\n"
        f"用户问题：{query}"
    )
    try:
        rewritten = await llm_service.generate(prompt, temperature=0.0)
    except Exception:
        logger.warning("query 改写失败，使用原问题", exc_info=True)
        return query

    rewritten = rewritten.strip().strip("`\"'“”‘’")
    rewritten = re.sub(r"^(改写后[:：]|输出[:：])", "", rewritten).strip()
    rewritten = rewritten.splitlines()[0].strip() if rewritten else ""
    if not rewritten or len(rewritten) > 2000:
        return query
    return rewritten


def parse_multi_query_response(
    response: str,
    original_query: str,
    limit: int | None = None,
) -> list[str]:
    """解析 LLM 输出的 Multi-Query，兼容 JSON、编号和项目符号。"""
    limit = limit or settings.RAG_MULTI_QUERY_COUNT
    raw_items: list[str] = []
    text = response.strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            raw_items.extend(str(item) for item in parsed)
    except json.JSONDecodeError:
        pass

    if not raw_items:
        raw_items = text.splitlines()

    queries: list[str] = []
    seen: set[str] = {original_query}
    for item in raw_items:
        cleaned = _clean_generated_query(item)
        if not cleaned or cleaned in seen:
            continue
        if len(cleaned) > 200:
            continue
        seen.add(cleaned)
        queries.append(cleaned)
        if len(queries) >= limit:
            break
    return queries


def _clean_generated_query(text: str) -> str:
    cleaned = text.strip().strip("`\"'“”‘’，,")
    cleaned = re.sub(r"^\s*[-*]\s+", "", cleaned)
    cleaned = re.sub(r"^\s*\d+[.)、]\s*", "", cleaned)
    cleaned = re.sub(r"^(查询[:：]|问题[:：]|query[:：])", "", cleaned, flags=re.I)
    return cleaned.strip()


async def generate_multi_queries(query: str) -> list[str]:
    """为同一医学意图生成多个检索表达，提升召回覆盖面。"""
    if not settings.RAG_ENABLE_MULTI_QUERY or settings.RAG_MULTI_QUERY_COUNT <= 0:
        return []
    prompt = (
        "你是医疗 RAG 的 Multi-Query 生成器。请围绕用户问题生成"
        f"{settings.RAG_MULTI_QUERY_COUNT} 个不同表达的中文检索 query，用于召回知识库。"
        "要求：保留原意，不新增病情、检查结果、药品剂量或治疗建议；"
        "覆盖同义说法、医学术语和用户可能的口语表达。"
        "只输出 JSON 字符串数组，不要解释。\n"
        "示例输入：阿司匹林有什么副作用？\n"
        "示例输出：[\"阿司匹林不良反应\", \"阿司匹林副作用有哪些\", \"阿司匹林用药风险和禁忌\"]\n"
        f"用户问题：{query}"
    )
    try:
        response = await llm_service.generate(prompt, temperature=0.2)
    except Exception:
        logger.warning("Multi-Query 生成失败，跳过多查询召回", exc_info=True)
        return []
    return parse_multi_query_response(response, query, settings.RAG_MULTI_QUERY_COUNT)


def clean_hyde_document(text: str) -> str:
    """清理 HyDE 假想文档，限制长度，避免污染后续 prompt。"""
    cleaned = text.strip().strip("`")
    cleaned = re.sub(r"^(假想答案[:：]|HyDE[:：]|假想文档[:：])", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[: settings.RAG_HYDE_MAX_CHARS].strip()


async def generate_hyde_document(query: str) -> str:
    """生成 HyDE 假想医学答案，只用于向量召回，不作为最终依据。"""
    if not settings.RAG_ENABLE_HYDE:
        return ""
    prompt = (
        "你是医疗 RAG 的 HyDE 生成器。请基于用户问题写一段可能出现在医学知识库中的"
        "简短假想答案，用于向量检索。"
        "要求：只写和检索相关的医学术语、症状、检查、治疗或风险点；"
        "不要编造具体患者事实，不要给确定诊断，不要添加引用编号。"
        "输出 80-180 字中文文本。\n"
        f"用户问题：{query}"
    )
    try:
        response = await llm_service.generate(prompt, temperature=0.2)
    except Exception:
        logger.warning("HyDE 生成失败，跳过 HyDE 召回", exc_info=True)
        return ""
    return clean_hyde_document(response)


def _source_text(source: RetrievedSource) -> str:
    return f"{source.source_title} {source.section} {source.content}"


def _has_minimum_source_signal(
    query: str, entities: dict[str, str], sources: list[RetrievedSource]
) -> bool:
    """粗判断来源是否至少和问题有实体或词面重合，避免随机向量命中直接进入 RAG。"""
    if not sources:
        return False
    query_tokens = set(tokenize(query))
    entity_values = [value for value in entities.values() if value]
    for source in sources[:5]:
        text = _source_text(source)
        if _is_no_info_text(text):
            continue
        if any(value in text for value in entity_values):
            return True
        content_tokens = set(tokenize(text))
        overlap = len(query_tokens & content_tokens)
        if overlap >= min(4, max(2, len(query_tokens) // 3)):
            return True
    return False


async def _sources_can_answer(query: str, sources: list[RetrievedSource]) -> bool:
    """用 LLM 做一次轻量来源充分性判断，决定是否改走无引用模型兜底。"""
    if not sources:
        return False
    snippets = []
    for source in sources[:5]:
        text = _source_text(source)
        if _is_no_info_text(text):
            continue
        snippets.append(f"[{source.citation_id}] {text[:500]}")
    if not snippets:
        return False

    prompt = (
        "请判断下面的来源是否能够直接回答用户问题。"
        "如果来源只提到相关实体、但没有回答用户问的属性/关系/做法/数据，回答 no。"
        "如果可以回答，回答 yes。只能输出 yes 或 no。\n"
        f"用户问题：{query}\n"
        "来源：\n"
        + "\n".join(snippets)
    )
    try:
        resp = await llm_service.generate(prompt, temperature=0.0)
    except Exception:
        logger.warning("来源充分性判断失败，回退到启发式结果", exc_info=True)
        return True
    normalized = resp.strip().lower()
    if normalized.startswith("yes") or normalized.startswith("是"):
        return True
    if normalized.startswith("no") or normalized.startswith("否"):
        return False
    return "yes" in normalized[:20] or "可以" in normalized[:20]


def _build_prompt_sync(
    intent_response: str, query: str, entities: dict[str, str], kg
) -> tuple[str, list[str], list[str]]:
    """同步拼装最终 prompt（含症状反查 + KG 检索）。

    :return: ``(final_prompt, intent_names, knowledge_list)``
    """
    entities = dict(entities)  # 拷贝，避免修改调用方对象
    yitu: list[str] = []
    knowledge: list[str] = []
    prompt = (
        "<指令>你是一个医疗问答助手。请优先依据给定提示和编号来源回答医学问题；"
        "使用某条编号来源的信息时，在对应句子末尾标注来源编号。"
        "如果当前提示和来源不足以回答，不要编造，应说明当前知识库没有可靠依据，并建议咨询医生。</指令>"
    )
    prompt += "<指令>请仅针对医疗类问题提供简洁、审慎和专业的回答。</指令>"
    if "疾病症状" in entities and "疾病" not in entities:
        res = kg.get_diseases_by_symptom(entities["疾病症状"])
        if len(res) > 0:
            entities["疾病"] = random.choice(res)
            all_en = "、".join(res)
            hint = (
                f"用户有{entities['疾病症状']}的情况，知识库推测其可能是得了{all_en}。"
                "请注意这只是一个推测，你需要明确告知用户这一点。"
            )
            prompt += f"<提示>{hint}</提示>"
            knowledge.append(hint)
    # 表驱动意图路由（intent_router 内部已修复「治疗周期 vs 治疗方法」子串误匹配）
    intent_prompt, intent_names = execute_intents(intent_response, entities, kg)
    yitu.extend(intent_names)
    for item in re.findall(r"<提示>(.*?)</提示>", intent_prompt):
        if len(item) >= 3 and not _is_no_info_text(item):
            prompt += f"<提示>{item}</提示>"
            knowledge.append(item)
    prompt += f"<用户问题>{query}</用户问题>"
    prompt += "<注意>请判断提示和编号来源是否真的覆盖用户问题；没有覆盖时不要硬答，也不要编造引用。</注意>"
    return prompt, yitu, knowledge


async def build_prompt(
    intent_response: str, query: str, entities: dict[str, str]
) -> tuple[str, list[str], list[str]]:
    """异步包装：KG 查询走线程池。"""
    kg = get_kg_service().client
    return await asyncio.to_thread(
        _build_prompt_sync, intent_response, query, entities, kg
    )


async def build_graphrag_knowledge(query: str, entities: dict[str, str]) -> list[str]:
    """沿 Neo4j 医疗图谱扩展短路径，生成 GraphRAG 证据。"""
    if not entities:
        return []
    kg_service = get_kg_service()
    return await asyncio.to_thread(
        kg_service.graphrag.build_knowledge,
        query,
        entities,
    )


def _dedupe_knowledge(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = " ".join((item or "").split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(item)
    return result


async def prepare(
    query: str,
    conversation_id: int | None = None,
    current_message_id: int | None = None,
    user_profile: dict | None = None,
    user_id: int | None = None,
) -> dict:
    """并行执行 NER 与意图识别，再拼装最终 prompt。

    新增记忆系统和 HITL 人在回路集成：
    - 从 user_memories 表召回长期记忆（向量检索 + 相关性过滤）
    - HITL 前置拦截高危症状，优先于 Agent 执行

    :return: ``{"prompt", "entities", "intents", "knowledge", "memories", "hitl_decision"}``
    """
    original_query = _clean_query(query)
    history = await load_recent_history(conversation_id, current_message_id)
    standalone_query = await rewrite_query_with_history(original_query, history)
    agent_messages = [
        *history,
        {"role": "user", "content": standalone_query},
    ]

    # HITL 前置检查：高危症状直接拦截
    hitl_service = get_hitl_service()
    hitl_decision = await hitl_service.assess_hitl(standalone_query)
    if hitl_decision.should_escalate and hitl_decision.urgency == "emergency":
        # 高危急症：跳过 RAG/Agent，直接返回急诊建议
        async with async_session_maker() as session:
            await hitl_service.log_hitl_event(
                session,
                user_id=user_id or 0,
                conversation_id=conversation_id,
                query=original_query,
                decision=hitl_decision,
            )
        return {
            "prompt": _build_hitl_emergency_prompt(original_query, hitl_decision),
            "entities": {},
            "intents": [],
            "knowledge": [f"HITL 高危拦截：{hitl_decision.reason}"],
            "sources": [],
            "memories": [],
            "answer_mode": "hitl_emergency",
            "original_query": original_query,
            "standalone_query": standalone_query,
            "rewritten_query": original_query,
            "multi_queries": [],
            "hyde_document": "",
            "conflict_resolution": empty_conflict_meta(),
            "agent_tools": _empty_tool_trace(query=standalone_query).to_meta(),
            "hitl_decision": hitl_decision.to_meta(),
        }

    if not await _should_use_medical_rag(standalone_query):
        tool_trace = _empty_tool_trace(
            query=standalone_query,
            messages=agent_messages,
            user_profile=user_profile,
            original_query=original_query,
            standalone_query=standalone_query,
        )
        return {
            "prompt": _build_direct_chat_prompt(original_query),
            "entities": {},
            "intents": [],
            "knowledge": [],
            "sources": [],
            "memories": [],
            "answer_mode": "direct_chat",
            "original_query": original_query,
            "standalone_query": standalone_query,
            "rewritten_query": original_query,
            "multi_queries": [],
            "hyde_document": "",
            "conflict_resolution": empty_conflict_meta(),
            "agent_tools": tool_trace.to_meta(),
            "hitl_decision": hitl_decision.to_meta(),
        }

    rewritten_query = await rewrite_medical_query(standalone_query)
    ner = get_ner_service()
    entities, intent_response, multi_queries, hyde_document = await asyncio.gather(
        ner.extract(rewritten_query),
        recognize_intent(rewritten_query),
        generate_multi_queries(rewritten_query),
        generate_hyde_document(rewritten_query),
    )

    # 并行召回长期记忆（如果有 user_id）
    memories = []
    if user_id:
        async with async_session_maker() as session:
            memory_service = get_memory_service()
            memories = await memory_service.retrieve_relevant_memories(
                session,
                user_id=user_id,
                query=rewritten_query,
                entities=entities,
                top_k=3,
            )

    tool_trace = await agent_graph.run_agent_graph(
        rewritten_query,
        entities,
        messages=agent_messages,
        user_profile=user_profile,
        raw_intent_response=intent_response,
        original_query=original_query,
        standalone_query=standalone_query,
    )
    prompt, intents, knowledge = await build_prompt(
        intent_response, rewritten_query, entities
    )
    tool_trace.state_snapshot = agent_state.with_task_intents(
        tool_trace.state_snapshot,
        intent_names=intents,
        raw_intent_response=intent_response,
    )
    tool_knowledge = agent_tools.trace_knowledge_items(tool_trace)

    # 注入长期记忆上下文
    memory_prompt_context = ""
    if memories:
        memory_service = get_memory_service()
        memory_prompt_context = memory_service.build_memory_prompt_context(memories)

    if tool_trace.final_action == "clarify":
        return {
            "prompt": _build_agent_clarification_prompt(original_query, tool_trace),
            "entities": entities,
            "intents": intents,
            "knowledge": tool_knowledge,
            "sources": [],
            "memories": [m.to_meta() for m in memories],
            "answer_mode": "agent_clarification",
            "original_query": original_query,
            "standalone_query": standalone_query,
            "rewritten_query": rewritten_query,
            "multi_queries": multi_queries,
            "hyde_document": hyde_document,
            "conflict_resolution": empty_conflict_meta(),
            "agent_tools": tool_trace.to_meta(),
            "hitl_decision": hitl_decision.to_meta(),
        }
    if tool_trace.final_action == "escalate":
        # Agent 触发转人工：记录 HITL 事件
        async with async_session_maker() as session:
            await hitl_service.log_hitl_event(
                session,
                user_id=user_id or 0,
                conversation_id=conversation_id,
                query=original_query,
                decision=hitl_decision,
                agent_trace=tool_trace.to_meta(),
            )
        return {
            "prompt": _build_agent_escalation_prompt(original_query, tool_trace),
            "entities": entities,
            "intents": intents,
            "knowledge": tool_knowledge,
            "sources": [],
            "memories": [m.to_meta() for m in memories],
            "answer_mode": "agent_escalation",
            "original_query": original_query,
            "standalone_query": standalone_query,
            "rewritten_query": rewritten_query,
            "multi_queries": multi_queries,
            "hyde_document": hyde_document,
            "conflict_resolution": empty_conflict_meta(),
            "agent_tools": tool_trace.to_meta(),
            "hitl_decision": hitl_decision.to_meta(),
        }
    if tool_trace.final_action == "control_stop":
        return {
            "prompt": _build_agent_control_stop_prompt(original_query, tool_trace),
            "entities": entities,
            "intents": intents,
            "knowledge": _dedupe_knowledge(
                [*tool_knowledge, f"Agent控制流停止：{tool_trace.stop_reason}"]
            ),
            "sources": [],
            "memories": [m.to_meta() for m in memories],
            "answer_mode": "agent_control_stop",
            "original_query": original_query,
            "standalone_query": standalone_query,
            "rewritten_query": rewritten_query,
            "multi_queries": multi_queries,
            "hyde_document": hyde_document,
            "conflict_resolution": empty_conflict_meta(),
            "agent_tools": tool_trace.to_meta(),
            "hitl_decision": hitl_decision.to_meta(),
        }

    prompt = _prepend_tool_context(prompt, tool_trace)
    # 注入长期记忆到 prompt（在工具上下文之后）
    if memory_prompt_context:
        prompt = memory_prompt_context + "\n" + prompt
    knowledge = _dedupe_knowledge([*tool_knowledge, *knowledge])
    graphrag_knowledge = await build_graphrag_knowledge(rewritten_query, entities)
    knowledge = _dedupe_knowledge([*knowledge, *graphrag_knowledge])
    async with async_session_maker() as session:
        sources = await get_hybrid_retriever().search(
            session,
            query=rewritten_query,
            retrieval_queries=multi_queries,
            hyde_document=hyde_document,
            entities=entities,
            kg_knowledge=knowledge,
        )

    has_source_signal = _has_minimum_source_signal(rewritten_query, entities, sources)
    if not has_source_signal or not await _sources_can_answer(rewritten_query, sources):
        fallback_note = "目前的知识储备无法回答该问题，已转为AI模型查询（无引用来源）。"
        fallback_prompt = _prepend_tool_context(
            _build_model_fallback_prompt(
                original_query,
                standalone_query,
                rewritten_query,
            ),
            tool_trace,
        )
        return {
            "prompt": fallback_prompt,
            "entities": entities,
            "intents": intents,
            "knowledge": _dedupe_knowledge([*knowledge, fallback_note]),
            "sources": [],
            "memories": [m.to_meta() for m in memories],
            "answer_mode": "model_fallback",
            "original_query": original_query,
            "standalone_query": standalone_query,
            "rewritten_query": rewritten_query,
            "multi_queries": multi_queries,
            "hyde_document": hyde_document,
            "conflict_resolution": empty_conflict_meta(),
            "agent_tools": tool_trace.to_meta(),
            "hitl_decision": hitl_decision.to_meta(),
        }

    conflict_resolution = resolve_source_conflicts(rewritten_query, sources)
    sources = conflict_resolution.sources
    citation_prompt = build_citation_prompt(sources)
    if citation_prompt:
        user_marker = f"<用户问题>{rewritten_query}</用户问题>"
        rewrite_hint = ""
        if rewritten_query != original_query:
            rewrite_hint = (
                f"<提示>用户原问题：{original_query}；系统为提高检索准确性，"
                f"将检索问题改写为：{rewritten_query}。"
                f"多轮补全后的独立问题：{standalone_query}。</提示>"
            )
        conflict_prompt = build_conflict_prompt(conflict_resolution)
        prompt = prompt.replace(
            user_marker,
            rewrite_hint + conflict_prompt + citation_prompt + user_marker,
            1,
        )
        knowledge = [*knowledge, *[f"[{s.citation_id}] {s.content}" for s in sources]]
        if conflict_resolution.has_conflict:
            knowledge.append(f"冲突消解：{conflict_resolution.summary}")

    # 将召回的记忆加入 knowledge 列表（供前端展示）
    for memory in memories:
        knowledge.append(f"[记忆] {memory.content}")

    return {
        "prompt": prompt,
        "entities": entities,
        "intents": intents,
        "knowledge": knowledge,
        "sources": [source.to_meta() for source in sources],
        "memories": [m.to_meta() for m in memories],
        "answer_mode": "rag",
        "original_query": original_query,
        "standalone_query": standalone_query,
        "rewritten_query": rewritten_query,
        "multi_queries": multi_queries,
        "hyde_document": hyde_document,
        "conflict_resolution": conflict_resolution.to_meta(),
        "agent_tools": tool_trace.to_meta(),
        "hitl_decision": hitl_decision.to_meta(),
    }


async def stream_answer(prompt: str):
    """基于最终 prompt 流式生成答案，逐块 yield 文本增量。"""
    async for delta in llm_service.chat_stream(prompt):
        yield delta
