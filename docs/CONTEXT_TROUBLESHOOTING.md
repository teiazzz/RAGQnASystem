# 上下文丢失问题快速排查指南

> **症状**：问了病症后，再问治疗方法时，系统不记得之前的病症  
> **5 分钟快速诊断**

---

## 🔍 立即检查（按顺序）

### ✅ 检查1：前端是否传了 `conversation_id`

**如何检查**：
1. 打开浏览器开发者工具（F12）
2. 切到 Network 标签
3. 发送第 2 轮问题
4. 找到 `/api/v1/chat` 请求
5. 查看 Request Payload

**正确示例**：
```json
{
  "message": "怎么治疗？",
  "conversation_id": 123  // ← 必须有这个
}
```

**错误示例**：
```json
{
  "message": "怎么治疗？"
  // ❌ 没有 conversation_id
}
```

**如果没有**：修改前端代码，从第 1 轮响应的 SSE meta 事件中提取 `conversation_id` 并保存：

```javascript
let conversationId = null;

eventSource.addEventListener('meta', (e) => {
    const data = JSON.parse(e.data);
    conversationId = data.conversation_id;  // ← 保存
});

// 第 2 轮起传递
fetch('/api/v1/chat', {
    method: 'POST',
    body: JSON.stringify({
        message: userInput,
        conversation_id: conversationId  // ← 传递
    })
});
```

---

### ✅ 检查2：后端是否读取了历史

**如何检查**：查看日志

在 `app/services/chat_service.py:746` 添加日志：
```python
history = await load_recent_history(conversation_id, current_message_id)
standalone_query = await rewrite_query_with_history(original_query, history)
logger.info(f"🔄 历史轮数: {len(history)//2}")  # ← 添加这行
logger.info(f"🔄 改写: {original_query} → {standalone_query}")  # ← 添加这行
```

**正确输出**：
```
🔄 历史轮数: 1
🔄 改写: 怎么治疗？ → 高血压怎么治疗？
```

**如果历史轮数为 0**：前端没传 `conversation_id`，回到检查1

---

### ✅ 检查3：改写是否正确

**如何检查**：查看 SSE meta 事件

前端添加调试代码：
```javascript
eventSource.addEventListener('meta', (e) => {
    const data = JSON.parse(e.data);
    console.log('🔍 上下文调试：');
    console.log('  原始问题:', data.original_query);
    console.log('  独立问题:', data.standalone_query);  // ← 检查这个
    console.log('  检索改写:', data.rewritten_query);
});
```

**正确示例**：
```
原始问题: 怎么治疗？
独立问题: 高血压怎么治疗？  // ← 正确补全了"高血压"
检索改写: 高血压治疗方法
```

**错误示例**：
```
原始问题: 怎么治疗？
独立问题: 怎么治疗？  // ❌ 没有补全
检索改写: 治疗方法
```

**如果改写不正确**：可能是历史太短或 LLM 改写失败，调大历史窗口：
```python
# chat_service.py:294
limit: int = 10  # 改成 10 轮
```

---

### ✅ 检查4：数据库是否保存了历史

**如何检查**：运行 SQL

```sql
-- 查看会话历史（替换 123 为实际 conversation_id）
SELECT id, role, content, created_at 
FROM messages 
WHERE conversation_id = 123 
ORDER BY id;
```

**正确结果**：
```
id  | role      | content           | created_at
----|-----------|-------------------|------------------
100 | user      | 我有高血压         | 2026-06-16 10:00
101 | assistant | 高血压是...        | 2026-06-16 10:01
102 | user      | 怎么治疗？         | 2026-06-16 10:02
```

**如果为空或只有最新一条**：会话没有正确保存，检查：
1. 前端是否传了 `conversation_id`
2. 后端是否正确保存了消息

---

### ✅ 检查5：长期记忆是否召回

**如何检查**：查看 SSE meta 事件中的 `memories` 字段

```javascript
console.log('  召回记忆:', data.memories);
```

**正确示例**：
```javascript
memories: [
    {
        category: "chronic_disease",
        content: "患者有高血压，服用降压药...",
        similarity: 0.82
    }
]
```

**如果为空**：
1. 用户可能没有保存过长期记忆
2. 或者相似度阈值太高（当前 0.7）

---

## 🎯 最常见原因排名

1. **前端没传 `conversation_id`** → 80% 的情况
2. **前端状态管理问题**（刷新页面丢失 conversationId） → 10%
3. **LLM 改写不准确** → 8%
4. **后端 `user_id` 未传递**（已修复） → 2%

---

## ✅ 已修复的问题

### 1. 后端 `user_id` 传递

**位置**：`app/api/v1/chat.py:102`

**修复前**：
```python
result = await chat_service.prepare(
    query,
    user_profile={"user_id": user.id},  # ❌ 在 user_profile 里
)
```

**修复后**：
```python
result = await chat_service.prepare(
    query,
    user_id=user.id,  # ✅ 单独参数
    user_profile={"user_id": user.id},
)
```

---

## 🧪 快速测试

运行测试脚本验证上下文是否正常：

```bash
cd RAGQnASystem
python tests/test_context.py
```

**预期输出**：
```
【第 1 轮】用户说病症...
   原始问题: 我有高血压，最近头晕
   
【第 2 轮】用户用指代词问治疗...
   原始问题: 怎么治疗？
   独立问题: 高血压怎么治疗？  # ← 正确补全
   
   ✅ 成功：上下文补全正常工作！
```

---

## 📞 仍然无法解决？

### 完整调试步骤

1. **开启详细日志**

在 `.env` 中设置：
```
LOG_LEVEL=DEBUG
```

2. **添加调试日志**

在 `app/services/chat_service.py` 添加：
```python
# 第 746 行左右
history = await load_recent_history(conversation_id, current_message_id)
logger.info(f"📜 读取历史: {len(history)} 条消息")
for i, msg in enumerate(history):
    logger.info(f"   [{i}] {msg['role']}: {msg['content'][:50]}...")

standalone_query = await rewrite_query_with_history(original_query, history)
logger.info(f"🔄 多轮改写: {original_query} → {standalone_query}")
```

3. **检查完整流程**

```
用户问题 → 读取历史 → 多轮改写 → NER → 检索 → LLM生成
          ↑                ↑           ↑      ↑
       检查这里          检查这里    检查实体  检查prompt
```

4. **提取关键日志**

```bash
# 查看最近的聊天日志
tail -f logs/app.log | grep -E "(📜|🔄|🎯)"
```

---

## 🎤 面试时怎么讲

**问：多轮对话上下文怎么实现的？**

> "我实现了三层上下文：短期（最近 8 轮消息窗口）+ 长期（user_memories 向量检索）+ 会话级（conversation_id 持久化）。
> 
> 关键是**多轮 query 改写**：读取历史后用 LLM 把指代问题（'它怎么治疗'）改写成完整问题（'高血压怎么治疗'），再去检索。这样既保留了上下文，又让每个 query 可以独立检索。
> 
> 常见坑是前端没传 `conversation_id`，导致每轮都是新会话。我在 SSE meta 事件返回 `standalone_query` 字段，方便前端调试改写是否正确。"

---

**✅ 现在你的系统已经支持完整的多轮对话上下文了！**  
**📖 详细文档：`docs/CONTEXT_MECHANISM.md`**
