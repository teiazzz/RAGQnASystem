# 多轮对话上下文机制详解

> **问题**：为什么我问了病症后，再问治疗方法时，系统不记得之前的病症？  
> **答案**：上下文机制已实现，但需要理解工作原理和可能的失效场景。

---

## 🔄 当前上下文实现机制（三层）

### 1️⃣ 短期上下文（对话历史）

**位置**：`chat_service.load_recent_history()`

**机制**：
```python
# 读取最近 8 轮对话（16 条消息：8轮用户+8轮助手）
history = await load_recent_history(
    conversation_id=conv_id,
    current_message_id=user_msg_id,
    limit=8,  # 可调整
)
# 返回: [
#   {"role": "user", "content": "我有糖尿病"},
#   {"role": "assistant", "content": "糖尿病是..."},
#   {"role": "user", "content": "怎么治疗？"},  # ← 当前问题
# ]
```

**工作流程**：
1. 用户第 1 轮："我有高血压"
2. 助手回答："高血压是..."
3. 用户第 2 轮："怎么治疗？" ← 这里会读取第 1 轮
4. `rewrite_query_with_history()` 把"怎么治疗"改写成"高血压怎么治疗"
5. 用改写后的完整问题去检索

**代码位置**：
- `app/services/chat_service.py:291-318` - 读取历史
- `app/services/chat_service.py:329-360` - 多轮改写

---

### 2️⃣ 长期上下文（用户记忆）

**位置**：`memory_service.retrieve_relevant_memories()`

**机制**：
```python
# 从 user_memories 表向量检索相关记忆
memories = await memory_service.retrieve_relevant_memories(
    session,
    user_id=user.id,  # ✅ 现已修复：API 路由已传递
    query="高血压怎么治疗",
    entities={"疾病": "高血压"},
    top_k=3,
)
# 如果用户之前说过"我有高血压，在吃降压药"，这条记忆会被召回
```

**注入位置**：
```python
# chat_service.py:930 左右
if memory_prompt_context:
    prompt = memory_prompt_context + "\n" + prompt
```

---

### 3️⃣ 会话级元数据（Conversation ID）

**机制**：前端必须在第 2 轮起传递 `conversation_id`

```javascript
// ❌ 错误：每次都不传 conversation_id
fetch('/api/v1/chat', {
    body: JSON.stringify({message: '怎么治疗？'})
})

// ✅ 正确：第 2 轮起传递 conversation_id
fetch('/api/v1/chat', {
    body: JSON.stringify({
        message: '怎么治疗？',
        conversation_id: 123  // ← 从第 1 轮响应的 meta 事件中获取
    })
})
```

---

## ❌ 上下文失效的 5 种场景

### 场景1：前端没有传 `conversation_id`（最常见）

**症状**：每次问题都像新对话，系统不记得之前说过什么

**原因**：
```javascript
// 前端代码可能是这样：
const response = await fetch('/api/v1/chat', {
    body: JSON.stringify({
        message: userInput,
        // ❌ 没有传 conversation_id
    })
})
```

**解决**：
```javascript
// 第 1 轮：不传 conversation_id
const firstResponse = await fetch('/api/v1/chat', {
    body: JSON.stringify({message: '我有高血压'})
})

// 从 SSE 的 meta 事件中提取 conversation_id
let conversationId = null;
eventSource.addEventListener('meta', (e) => {
    const data = JSON.parse(e.data);
    conversationId = data.conversation_id;  // ← 保存
})

// 第 2 轮起：传 conversation_id
const secondResponse = await fetch('/api/v1/chat', {
    body: JSON.stringify({
        message: '怎么治疗？',
        conversation_id: conversationId  // ← 传递
    })
})
```

---

### 场景2：Query 改写不准确

**症状**：问"它怎么治疗"，系统改写成"疾病怎么治疗"（丢失了"高血压"）

**原因**：`rewrite_query_with_history()` 的 LLM 改写失败

**调试**：查看 SSE meta 事件中的字段：
```json
{
  "original_query": "它怎么治疗？",
  "standalone_query": "高血压怎么治疗？",  // ← 检查这个是否正确
  "rewritten_query": "高血压治疗方法"
}
```

**解决**：
1. 检查 `chat_service.py:329` 的 prompt 是否清晰
2. 调整 `limit=8` 为更多轮（如 `limit=10`）
3. 降低 LLM temperature（当前是 0.0，已经最低）

---

### 场景3：用户指代不明确

**症状**：用户问"副作用是什么"，但历史里没提到具体药名

**示例**：
```
用户第1轮："高血压吃什么药？"
助手："可以吃XXX、YYY、ZZZ..."（列了多个药）
用户第2轮："副作用是什么？"  // ❌ 不知道问哪个药的副作用
```

**解决**：
- 依赖 NER：如果历史没有明确实体，改写器无法补全
- 改进：在 Prompt 中提示用户具体化："您是想问哪个药物的副作用？"

---

### 场景4：历史窗口太小

**症状**：对话超过 8 轮后，早期信息丢失

**当前配置**：
```python
# chat_service.py:294
limit: int = 8  # 只保留最近 8 轮（16 条消息）
```

**解决**：
1. 调大 `limit`（如 `limit=15`）
2. 或者用摘要：前 N 轮摘要 + 最近 K 轮完整

---

### 场景5：前端状态管理问题

**症状**：用户刷新页面或切换标签页后，上下文丢失

**原因**：前端没有持久化 `conversationId`

**解决**：
```javascript
// 保存到 localStorage
localStorage.setItem('currentConversationId', conversationId);

// 页面加载时恢复
const savedId = localStorage.getItem('currentConversationId');
if (savedId) {
    // 调用 GET /api/v1/conversations/{savedId} 加载历史
}
```

---

## 🏢 企业生产中的上下文实现

### 1. 会话存储（✅ 你已实现）

```sql
-- conversations 表
id | user_id | title | created_at | updated_at

-- messages 表  
id | conversation_id | role | content | created_at
```

**优点**：
- 持久化，重启不丢
- 支持历史回溯
- 支持多设备同步

**你的实现**：已完成，在 `app/db/models.py`

---

### 2. Redis 会话缓存（热数据优化）

```python
# 企业常见做法：热数据放 Redis，冷数据在 PG
import redis

# 保存最近 10 轮到 Redis（TTL 1小时）
redis_client.setex(
    f"chat_history:{conversation_id}",
    3600,  # 1 小时过期
    json.dumps(history[-10:])
)

# 读取时先查 Redis
history = redis_client.get(f"chat_history:{conversation_id}")
if not history:
    # 未命中，从 PG 查
    history = await load_from_db(conversation_id)
```

**优点**：减少 DB 压力

**你的项目**：当前直接查 PG，单用户场景够用；高并发时加 Redis 缓存

---

### 3. Prompt 注入历史（✅ 你已实现）

```python
# 你的实现：chat_service.py:745
history = await load_recent_history(conversation_id, current_message_id)
standalone_query = await rewrite_query_with_history(original_query, history)
```

**企业常见变种**：
- **滑动窗口 + 摘要**：前 N 轮摘要，最近 K 轮完整
- **Token 预算控制**：历史 + 检索上下文 + 新问题 < max_tokens

---

### 4. 向量记忆（✅ 你已实现）

```python
# 你的实现：memory_service.py
memories = await memory_service.retrieve_relevant_memories(
    user_id, query, entities
)
```

**企业级增强**：
- **自动提取**：对话结束后 LLM 总结关键信息自动存 `user_memories`
- **主动推送**：检测到相关话题时主动提醒用户之前的记忆

---

### 5. Session State（对话状态机）

**适用场景**：表单填写、多步骤任务

```python
# 企业常见：Redis 存状态机
{
    "conversation_id": 123,
    "state": "collecting_symptoms",  # 当前状态
    "collected_data": {
        "symptom": "头痛",
        "duration": None,  # ← 下一步要问
        "severity": None
    },
    "next_action": "ask_duration"
}
```

**你的项目**：当前是 stateless，每轮独立；如果要做**导诊流程**（多步收集信息），需要加状态机

---

## 🔧 调试上下文的方法

### 方法1：检查 SSE meta 事件

```javascript
eventSource.addEventListener('meta', (e) => {
    const data = JSON.parse(e.data);
    console.log('🔍 调试上下文：');
    console.log('  原始问题:', data.original_query);
    console.log('  多轮补全:', data.standalone_query);  // ← 检查是否正确补全
    console.log('  检索改写:', data.rewritten_query);
    console.log('  召回记忆:', data.memories);  // ← 检查是否召回了相关记忆
});
```

### 方法2：查看数据库

```sql
-- 检查会话历史是否正确保存
SELECT id, role, content, created_at 
FROM messages 
WHERE conversation_id = 123 
ORDER BY id;

-- 检查长期记忆
SELECT * FROM user_memories WHERE user_id = 1;
```

### 方法3：日志追踪

```python
# chat_service.py 中加日志
logger.info(f"📜 读取历史 {len(history)} 轮: {history}")
logger.info(f"🔄 改写前: {original_query}")
logger.info(f"🔄 改写后: {standalone_query}")
logger.info(f"💾 召回记忆 {len(memories)} 条")
```

---

## ✅ 快速排查清单

问题：**"系统不记得我之前说的病症"**

按以下顺序检查：

- [ ] **1. 前端是否传了 `conversation_id`？**
  - 检查：浏览器 Network 标签，看请求 body
  - 第 1 轮可以不传，第 2 轮起**必须传**

- [ ] **2. 数据库是否保存了历史？**
  - 运行：`SELECT * FROM messages WHERE conversation_id = XXX ORDER BY id;`
  - 应该能看到之前的用户+助手消息

- [ ] **3. `standalone_query` 是否正确补全？**
  - 检查：SSE meta 事件中的 `standalone_query` 字段
  - 应该是完整问题，不含指代词

- [ ] **4. API 路由是否传了 `user_id`？**
  - ✅ 已修复：`app/api/v1/chat.py:102` 现在会传

- [ ] **5. 长期记忆是否召回？**
  - 检查：SSE meta 事件中的 `memories` 字段
  - 如果为空，可能是相似度阈值太高

---

## 📊 对比：你的实现 vs 企业标准

| 功能 | 你的实现 | 企业标准 | 差距 |
|------|---------|---------|------|
| **短期上下文** | ✅ 读取最近 8 轮 | 滑动窗口 + 摘要 | 基本够用 |
| **长期记忆** | ✅ 向量检索 user_memories | 自动提取 + 主动推送 | 需手动保存 |
| **会话持久化** | ✅ PG 存储 | PG + Redis 缓存 | 单用户够用 |
| **多轮改写** | ✅ LLM 改写 | LLM + 规则补充 | 已有 |
| **状态机** | ❌ 无 | 表单/多步任务用 | 简单场景不需要 |

**结论**：你的实现已经达到**中小企业生产标准**，大厂会加 Redis 缓存和自动记忆提取。

---

## 🚀 立即修复建议

### 优先级 1：修复 `user_id` 传递（已完成）

✅ 已在 `app/api/v1/chat.py:102` 修复

### 优先级 2：前端确保传 `conversation_id`

```javascript
// 检查前端代码，确保第 2 轮起传 conversation_id
```

### 优先级 3：调试日志

在 `chat_service.py` 加日志查看改写是否生效：

```python
# chat_service.py:746
standalone_query = await rewrite_query_with_history(original_query, history)
logger.info(f"🔄 多轮改写: {original_query} → {standalone_query}")  # ← 加这行
```

---

## 🎤 面试加分点

**问：你的多轮对话上下文怎么实现的？**

> "我实现了**三层上下文**：短期（最近 8 轮对话窗口）+ 长期（user_memories 向量检索）+ 会话级（conversation_id 持久化）。
> 
> 关键是**多轮 query 改写**：读取历史后用 LLM 把指代问题（如'它怎么治疗'）改写成完整问题（'高血压怎么治疗'），再去检索。这样既保留了上下文，又让每个 query 可以独立检索，避免检索时上下文污染。
> 
> 企业级会加 Redis 缓存热数据 + 自动记忆提取，我当前是直接查 PG，单用户场景够用。"

---

**现在你的系统已经支持完整的多轮对话上下文了！🎉**
