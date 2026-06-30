# Phase 3: 记忆系统与 HITL 人在回路实现文档

> **实现日期**: 2026-06-16  
> **对应清单**: 阶段三【P1】记忆系统 + 【P1】HITL人在回路

---

## 一、功能概述

### 1.1 记忆系统（Memory System）

**三层记忆架构**：
- **短期记忆**：最近 N 轮对话（默认 5 轮），滑动窗口，存在 `messages` 表
- **工作记忆**：当前对话的任务状态（意图、实体、工具调用），在 `AgentState` 中
- **长期记忆**：用户健康档案（过敏史、慢性病、用药史），存在 `user_memories` 表

**核心设计**：
- **向量检索 + 相似度过滤**：防止无关记忆污染新对话（阈值 0.7）
- **关键词启发式加成**：实体命中记忆关键词时，相似度 +0.15
- **重要性分级**：高危记忆（过敏史/慢性病 importance≥8）降低阈值到 0.5

### 1.2 HITL 人在回路（Human-In-The-Loop）

**医疗场景强制拦截**：
1. **高危急症**（emergency）：胸痛、呼吸困难、大出血、意识障碍 → 立即急诊
2. **紧急症状**（urgent）：高热不退、严重腹痛、孕妇出血 → 尽快就医/人工
3. **复杂用药**（urgent）：孕产妇、婴幼儿、多药相互作用 → 转人工审核
4. **工具失败**（routine）：Agent 连续失败 ≥2 次 → 人工兜底
5. **用户要求**（routine）：明确说"转人工""找医生"

**安全机制**：
- **前置拦截**：在 `chat_service.prepare()` 入口就 HITL 评估，高危直接返回急诊建议
- **Agent 层拦截**：`escalate_to_human` 工具触发时记录事件
- **审计日志**：所有转人工事件写入 `hitl_events` 表，供质控分析

---

## 二、数据库设计

### 2.1 user_memories 表

```sql
CREATE TABLE user_memories (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    category VARCHAR(32) NOT NULL DEFAULT 'health_fact',
        -- health_fact / allergy / chronic_disease / medication_history / decision
    content TEXT NOT NULL,
    relevance_keywords JSONB,  -- ["糖尿病", "二甲双胍"] 快速过滤用
    embedding JSONB,  -- 向量检索（JSONB保证无pgvector时可降级）
    source_conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
    importance INTEGER NOT NULL DEFAULT 5,  -- 1-10，越高越重要
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(0),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(0)
);
```

**索引**：
- `(user_id, created_at)` — 按用户查询最新记忆
- `(category)` — 按类型筛选（如只查过敏史）

### 2.2 hitl_events 表

```sql
CREATE TABLE hitl_events (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
    query TEXT NOT NULL,
    urgency VARCHAR(32) NOT NULL,  -- emergency / urgent / routine
    reason TEXT NOT NULL,
    matched_keywords JSONB,  -- ["胸痛", "呼吸困难"]
    agent_trace JSONB,  -- Agent 执行轨迹
    human_reviewed BOOLEAN DEFAULT FALSE,
    human_reviewer_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    human_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(0)
);
```

**索引**：
- `(urgency)` — 按紧急程度筛选
- `(human_reviewed)` — 查待审核事件
- `(created_at)` — 时间序列分析

---

## 三、记忆系统实现细节

### 3.1 记忆召回流程

```python
# 1. 用户问"糖尿病人能吃阿司匹林吗？"
query = "糖尿病人能吃阿司匹林吗？"
entities = {"疾病": "糖尿病", "药品": "阿司匹林"}

# 2. 向量检索 + 相关性过滤
memories = await memory_service.retrieve_relevant_memories(
    session,
    user_id=1,
    query=query,
    entities=entities,
    top_k=3,
)
# 返回: [
#   RetrievedMemory(
#       content="患者有2型糖尿病，服用二甲双胍500mg每日两次",
#       category="chronic_disease",
#       similarity=0.82,  # 向量相似度 + 关键词加成
#       importance=8
#   )
# ]

# 3. 注入 prompt
memory_context = memory_service.build_memory_prompt_context(memories)
# 输出:
# <用户健康档案>
# - [慢性病] 患者有2型糖尿病，服用二甲双胍500mg每日两次
# </用户健康档案>
# <注意>回答时需结合当前问题，涉及用药务必提醒遵医嘱。</注意>
```

### 3.2 防污染机制

**问题**：用户开新对话问"感冒吃什么药"，不应该被旧对话的"糖尿病"记忆污染。

**解决**：
1. **向量相似度过滤**：`cosine_similarity(query_embedding, memory_embedding) >= 0.7`
2. **关键词重合判断**：实体命中 `relevance_keywords` 时才加成
3. **高优先级例外**：过敏史/慢性病（importance≥8）阈值降到 0.5，但仍需≥0.5

**示例**：
```python
# 用户问"感冒吃什么药"
query = "感冒吃什么药"
entities = {"疾病": "感冒"}

# 已有记忆："患者有糖尿病，服用二甲双胍"（importance=8）
# 计算相似度：0.35（低于0.5的高优先级阈值）
# 结果：不召回，避免污染 ✓
```

### 3.3 配置参数

```python
# app/core/config.py
MEMORY_SIMILARITY_THRESHOLD: float = 0.70  # 普通记忆阈值
MEMORY_HIGH_PRIORITY_THRESHOLD: float = 0.50  # 高危记忆阈值
MEMORY_SHORT_TERM_WINDOW: int = 5  # 短期记忆窗口
```

---

## 四、HITL 实现细节

### 4.1 拦截流程

```python
# chat_service.prepare() 入口处
hitl_decision = await hitl_service.assess_hitl(query)

if hitl_decision.should_escalate and hitl_decision.urgency == "emergency":
    # 高危急症：跳过 RAG/Agent，直接返回急诊建议
    return {
        "prompt": _build_hitl_emergency_prompt(query, hitl_decision),
        "answer_mode": "hitl_emergency",
        "hitl_decision": hitl_decision.to_meta(),
    }
```

### 4.2 关键词规则

```python
# hitl_service.py
HIGH_RISK_EMERGENCY_KEYWORDS = {
    "胸痛", "呼吸困难", "大出血", "昏迷", "抽搐",
    "疑似中风", "口角歪斜", "一侧肢体无力", "服毒", "自杀"
}

URGENT_KEYWORDS = {
    "高热不退", "持续腹痛", "血压很高", "孕妇出血", "婴儿抽搐"
}

COMPLEX_MEDICATION_KEYWORDS = {
    "孕妇", "哺乳期", "婴儿", "多种药一起吃", "肾功能不全", "化疗"
}
```

### 4.3 HITL 决策示例

| 用户输入 | 匹配关键词 | urgency | should_escalate | 建议 |
|---------|-----------|---------|----------------|------|
| "胸痛喘不上气" | ["胸痛", "呼吸困难"] | emergency | ✓ | 立即拨打120或去急诊 |
| "孩子发烧39度高热不退" | ["高热不退"] | urgent | ✓ | 24小时内就医或人工问诊 |
| "怀孕了能吃感冒药吗" | ["孕妇"] + 用药关键词 | urgent | ✓ | 必须咨询医生或药师 |
| "感冒吃什么药" | 无 | none | ✗ | 正常 RAG 回答 |

---

## 五、集成到 chat_service

### 5.1 修改点

```python
async def prepare(
    query: str,
    conversation_id: int | None = None,
    current_message_id: int | None = None,
    user_profile: dict | None = None,
    user_id: int | None = None,  # 新增：用于召回长期记忆
) -> dict:
    # 1. HITL 前置检查
    hitl_decision = await hitl_service.assess_hitl(query)
    if hitl_decision.should_escalate and hitl_decision.urgency == "emergency":
        # 高危直接返回
        ...

    # 2. 并行召回长期记忆
    if user_id:
        memories = await memory_service.retrieve_relevant_memories(
            session, user_id, query, entities
        )

    # 3. 注入记忆到 prompt
    if memory_prompt_context:
        prompt = memory_prompt_context + "\n" + prompt

    # 4. Agent 转人工时记录 HITL 事件
    if tool_trace.final_action == "escalate":
        await hitl_service.log_hitl_event(...)

    return {
        "memories": [m.to_meta() for m in memories],
        "hitl_decision": hitl_decision.to_meta(),
        ...
    }
```

### 5.2 返回字段新增

```json
{
  "memories": [
    {
      "id": 1,
      "category": "chronic_disease",
      "content": "患者有2型糖尿病...",
      "importance": 8,
      "similarity": 0.82
    }
  ],
  "hitl_decision": {
    "should_escalate": false,
    "urgency": "none",
    "reason": "未检测到需要人工介入的风险"
  }
}
```

---

## 六、测试与验证

### 6.1 运行测试

```bash
cd RAGQnASystem
python tests/test_memory_hitl.py
```

### 6.2 预期输出

```
=== 测试记忆系统 ===
1. 保存用户长期记忆...
   ✓ 保存过敏史记忆 ID=1
   ✓ 保存慢性病记忆 ID=2
   ✓ 保存健康事实记忆 ID=3

2. 测试相关性检索（糖尿病用药问题）...
   召回 1 条记忆:
   - [chronic_disease] 患者有 2 型糖尿病病史... (相似度=0.820)

3. 测试相关性过滤（无关问题）...
   召回 0 条记忆 (预期很少或为空，防止污染)

=== 测试 HITL 人在回路 ===
1. 测试高危急症拦截（胸痛）...
   转人工: True
   紧急程度: emergency
   建议: 立即拨打 120 急救电话...
   ✓ 正确拦截高危急症

✅ 所有测试通过！
```

---

## 七、面试讲解要点（重点背诵）

### 7.1 记忆系统

**问：你的长期记忆怎么防止污染新对话？**

> "我用**向量检索 + 相似度阈值过滤**。每条记忆保存时生成 embedding，新对话召回时计算 `cosine_similarity(query_embedding, memory_embedding)`，只有 ≥0.7 的才召回。过敏史、慢性病等高危记忆（importance≥8）阈值降到 0.5，确保不遗漏。另外还有关键词启发式：如果 NER 实体命中记忆的 `relevance_keywords`，相似度加成 0.15。这样既防止了无关记忆污染，又保证了高优先级信息不漏。"

**问：短期、长期、工作记忆分别是什么？**

> "短期记忆就是消息滑动窗口，保留最近 5 轮对话，存在 `messages` 表；长期记忆是用户健康档案（过敏史、慢性病），存 `user_memories` 表，按需向量检索召回；工作记忆是当前对话的任务状态（意图、实体、工具调用），在 `AgentState` 里，不持久化。"

### 7.2 HITL 人在回路

**问：医疗场景的 HITL 怎么设计？**

> "医疗不能让 Agent 随便下结论，我做了**前置拦截 + Agent 层拦截**两层。前置在 `chat_service.prepare()` 入口用规则匹配高危关键词（胸痛、呼吸困难、大出血等），emergency 级别直接跳过 RAG/Agent，返回急诊建议；Agent 层有 `escalate_to_human` 工具，触发时记录 HITL 事件。所有转人工事件写入 `hitl_events` 表审计，供质控分析哪些场景自动处理不了。分级是 emergency（立即急诊）> urgent（尽快人工）> routine（常规转人工）。"

**问：LangGraph 怎么支持 HITL？**

> "LangGraph 有中断/恢复机制，可以在某个节点 `interrupt_before=['human_review']`，等人工审核后调 `.resume()` 继续。但我们当前用的是更简单的**转人工终止**：高危症状触发 `escalate_to_human` 后直接返回急诊建议，不继续 Agent 循环。未来可以改成暂停等人工，审核通过后继续原任务。"

---

## 八、数据库迁移

```bash
# 应用迁移（创建 user_memories 和 hitl_events 表）
psql -U postgres -d medical_rag -f migrations/003_memory_hitl.sql

# 或在 Python 中用 SQLAlchemy 自动建表
from app.db.base import Base
from app.db.session import engine
Base.metadata.create_all(bind=engine)
```

---

## 九、后续优化方向

1. **自动记忆提取**：对话结束后 LLM 总结关键健康事实，自动存 `user_memories`
2. **记忆遗忘**：低重要性记忆超过 N 天自动删除，避免膨胀
3. **HITL 审核闭环**：`hitl_events` 表的 `human_reviewed` 字段，人工审核后标记，反馈改进拦截规则
4. **LLM 二次确认**：规则匹配到疑似高危后，再用 LLM 判断一次，减少误拦
5. **多模态记忆**：支持存储用户上传的化验单、处方单（OCR 提取关键信息）

---

## 十、已完成清单对照

- [x] **【P1】记忆系统（短期/长期/工作记忆）**
  - ✓ 长期记忆：`user_memories` 表 + 向量检索 + 相关性过滤
  - ✓ 短期记忆：`load_recent_history()` 滑动窗口
  - ✓ 工作记忆：`AgentState` 任务状态
  - ✓ 防污染机制：相似度阈值 0.7，高优先级 0.5
  - 📊 **可讲指标**：相似度阈值、召回条数、过滤前后对比

- [x] **【P1】HITL 人在回路（医疗安全核心）**
  - ✓ 高危关键词前置拦截（emergency/urgent/routine 分级）
  - ✓ `escalate_to_human` 工具集成
  - ✓ `hitl_events` 表审计日志
  - ✓ 急诊建议 prompt（不做诊断）
  - 📊 **可讲指标**：高危拦截次数、转人工率、误拦率（人工复核）

---

**实现完成时间**: 约 2-3 小时  
**代码量**: ~600 行（`memory_service.py` 250行 + `hitl_service.py` 280行 + 集成 70行）  
**面试准备**: 重点背诵第七节"面试讲解要点"
