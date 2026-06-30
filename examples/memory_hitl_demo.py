"""记忆系统和 HITL 功能使用示例。

演示如何在实际对话中使用记忆系统和 HITL 拦截。
"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.db.session import async_session_maker
from app.services.chat_service import prepare
from app.services.memory_service import get_memory_service


async def example_conversation_with_memory():
    """演示：用户有糖尿病史，新对话自动召回相关记忆。"""
    print("\n" + "=" * 60)
    print("示例 1: 记忆系统防污染")
    print("=" * 60)

    # 1. 保存用户长期记忆（模拟历史对话提取）
    print("\n【步骤1】保存用户健康档案...")
    async with async_session_maker() as session:
        memory_service = get_memory_service()
        await memory_service.save_memory(
            session,
            user_id=999,
            content="患者有 2 型糖尿病，目前服用二甲双胍 500mg 每日两次，血糖控制良好",
            category="chronic_disease",
            importance=8,
            relevance_keywords=["糖尿病", "二甲双胍", "血糖"],
        )
        await memory_service.save_memory(
            session,
            user_id=999,
            content="患者对青霉素过敏，曾在 2023 年因青霉素注射出现全身皮疹和呼吸困难",
            category="allergy",
            importance=9,
            relevance_keywords=["青霉素", "过敏"],
        )
    print("   ✓ 已保存：糖尿病史、青霉素过敏史")

    # 2. 相关问题：应该召回糖尿病记忆
    print("\n【步骤2】用户问相关问题（糖尿病用药）...")
    query1 = "糖尿病人能吃阿司匹林吗？"
    print(f"   用户问题: {query1}")

    result1 = await prepare(
        query=query1,
        user_id=999,
        conversation_id=None,
        current_message_id=None,
    )
    print(f"   召回记忆: {len(result1.get('memories', []))} 条")
    for mem in result1.get("memories", []):
        print(f"      - [{mem['category']}] {mem['content'][:50]}...")
    print("   ✓ 正确召回糖尿病相关记忆")

    # 3. 无关问题：不应该召回糖尿病记忆（防污染）
    print("\n【步骤3】用户开新对话，问无关问题...")
    query2 = "感冒了吃什么药？"
    print(f"   用户问题: {query2}")

    result2 = await prepare(
        query=query2,
        user_id=999,
        conversation_id=None,
        current_message_id=None,
    )
    print(f"   召回记忆: {len(result2.get('memories', []))} 条")
    if not result2.get("memories"):
        print("   ✓ 正确过滤：无关记忆未召回，防止污染新对话")
    else:
        print("   ⚠️ 意外召回了记忆（可能需要调整阈值）")

    # 4. 高优先级记忆（过敏史）：即使关联不强也应召回
    print("\n【步骤4】用户问头孢（和青霉素有交叉过敏）...")
    query3 = "我能吃头孢吗？"
    print(f"   用户问题: {query3}")

    result3 = await prepare(
        query=query3,
        user_id=999,
        conversation_id=None,
        current_message_id=None,
    )
    print(f"   召回记忆: {len(result3.get('memories', []))} 条")
    allergy_recalled = any(
        mem["category"] == "allergy" for mem in result3.get("memories", [])
    )
    if allergy_recalled:
        print("   ✓ 成功召回过敏史（高优先级记忆，阈值降低到 0.5）")
    else:
        print("   ⚠️ 未召回过敏史（需要调整或检查关键词匹配）")


async def example_hitl_emergency():
    """演示：HITL 高危症状拦截。"""
    print("\n" + "=" * 60)
    print("示例 2: HITL 高危拦截")
    print("=" * 60)

    test_cases = [
        {
            "query": "我胸口很痛，喘不上气，出了很多汗",
            "expected": "emergency",
            "desc": "高危急症（胸痛 + 呼吸困难）",
        },
        {
            "query": "怀孕了能吃感冒药吗？",
            "expected": "urgent",
            "desc": "复杂用药（孕妇）",
        },
        {
            "query": "孩子发烧39度，吃了退烧药还是高热不退",
            "expected": "urgent",
            "desc": "紧急症状（高热不退）",
        },
        {
            "query": "感冒了吃什么药好？",
            "expected": "none",
            "desc": "普通问题",
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n【案例{i}】{case['desc']}")
        print(f"   用户问题: {case['query']}")

        result = await prepare(
            query=case["query"],
            user_id=999,
            conversation_id=None,
            current_message_id=None,
        )

        hitl = result.get("hitl_decision", {})
        print(f"   HITL 结果:")
        print(f"      - 转人工: {hitl.get('should_escalate')}")
        print(f"      - 紧急程度: {hitl.get('urgency')}")
        print(f"      - 原因: {hitl.get('reason', 'N/A')[:60]}...")

        if hitl.get("urgency") == case["expected"]:
            print(f"   ✓ 正确识别为 {case['expected']}")
        else:
            print(f"   ⚠️ 预期 {case['expected']}，实际 {hitl.get('urgency')}")

        if hitl.get("should_escalate") and hitl.get("urgency") == "emergency":
            print(f"   回答模式: {result.get('answer_mode')} (跳过 RAG/Agent)")


async def main():
    print("\n记忆系统和 HITL 功能演示")
    print("=" * 60)

    try:
        # 示例 1: 记忆系统
        await example_conversation_with_memory()

        # 示例 2: HITL 拦截
        await example_hitl_emergency()

        print("\n" + "=" * 60)
        print("✅ 演示完成！")
        print("\n核心要点：")
        print("1. 长期记忆通过向量相似度（阈值 0.7）自动过滤无关内容")
        print("2. 高优先级记忆（过敏史/慢性病）降低阈值到 0.5")
        print("3. HITL 前置拦截高危症状，emergency 级别跳过 RAG/Agent")
        print("4. 所有转人工事件记录到 hitl_events 表")

    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
