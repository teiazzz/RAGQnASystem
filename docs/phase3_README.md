# Phase 3: 记忆系统与 HITL 人在回路 - 快速导航

> 📅 **完成日期**: 2026-06-16  
> ⏱️ **开发时长**: 2.5 小时  
> 📝 **代码量**: ~800 行  
> 🎯 **清单对照**: 阶段三【P1】记忆系统 + 【P1】HITL人在回路

---

## 📚 文档导航

| 文档 | 用途 | 适合场景 |
|------|------|---------|
| **[MEMORY_HITL_CHEATSHEET.md](MEMORY_HITL_CHEATSHEET.md)** ⭐ | 面试速查卡（5分钟快速复习） | **面试前必看** |
| [MEMORY_HITL_SUMMARY.md](MEMORY_HITL_SUMMARY.md) | 完整实现总结 | 了解全貌、写简历 |
| [phase3_memory_hitl.md](phase3_memory_hitl.md) | 详细设计文档 | 深入理解实现细节 |
| [phase3_tool_use.md](phase3_tool_use.md) | Function Calling 工具集 | 讲 Agent 工具设计 |
| [phase3_langgraph_agent.md](phase3_langgraph_agent.md) | LangGraph 状态机 | 讲 ReAct/Plan-Execute |
| [phase3_mcp.md](phase3_mcp.md) | MCP 接入 | 讲工具可插拔和协议标准化 |

---

## 🚀 快速开始

### 1. 数据库迁移

```bash
cd RAGQnASystem
psql -U postgres -d medical_rag -f migrations/003_memory_hitl.sql
```

创建表：
- `user_memories` - 用户长期记忆（过敏史、慢性病）
- `hitl_events` - HITL 转人工事件审计

### 2. 运行测试

```bash
python tests/test_memory_hitl.py
```

预期输出：
```
✅ 所有测试通过！
1. 长期记忆用向量检索 + 相似度阈值（0.7）防止污染新对话
2. 高优先级记忆（过敏史/慢性病）阈值降低到 0.5
3. HITL 前置拦截高危症状，emergency 级别跳过 RAG/Agent
```

### 3. 运行演示

```bash
python examples/memory_hitl_demo.py
```

---

## 🎯 核心功能

### ✅ 记忆系统（Memory System）

**解决的问题**：用户开新对话，如何避免被旧对话的无关记忆污染？

**解决方案**：
- 向量检索 + 相似度阈值过滤（普通 0.7，高优先级 0.5）
- 关键词启发式加成（实体命中 +0.15）
- 三层架构：短期（5轮窗口）+ 长期（档案）+ 工作（任务）

**代码位置**：
- `app/services/memory_service.py` - 记忆服务
- `app/db/models.py` - UserMemory 模型

### ✅ HITL 人在回路（Human-In-The-Loop）

**解决的问题**：医疗场景不能让 Agent 随便下诊断，高危症状怎么拦截？

**解决方案**：
- 双层拦截：前置（规则匹配）+ Agent 层（escalate_to_human 工具）
- 三级分级：emergency（立即急诊）> urgent（尽快人工）> routine（常规）
- 审计闭环：hitl_events 表记录完整轨迹

**代码位置**：
- `app/services/hitl_service.py` - HITL 服务
- `app/services/chat_service.py` - 集成到 prepare() 入口

---

## 📊 关键数字（面试必记）

```
0.7  - 普通记忆相似度阈值
0.5  - 高优先级记忆阈值（过敏史/慢性病）
0.15 - 关键词命中加成
5    - 短期记忆窗口（轮数）
19   - 高危急症关键词数
8-10 - 高优先级 importance 范围
```

---

## 🎤 面试必背三句话

### 记忆系统
> "我用**向量检索 + 相似度阈值过滤**。普通记忆阈值 0.7，高优先级（过敏史/慢性病）降到 0.5。关键词命中时相似度加成 0.15。这样既防止无关记忆污染新对话，又保证高优先级信息不漏。"

### HITL
> "医疗不能让 Agent 随便下结论，我做了**前置拦截 + Agent 层拦截**两层。前置在 chat_service 入口用规则匹配高危关键词，emergency 级别直接跳过 RAG/Agent 返回急诊建议；Agent 层有 escalate_to_human 工具。所有转人工事件写入 hitl_events 表审计。"

### 三层记忆
> "短期记忆就是消息滑动窗口（最近 5 轮），长期记忆是用户健康档案（user_memories 表，向量检索召回），工作记忆是当前任务状态（AgentState，不持久化）。"

---

## 🔥 高频追问 + 标准答案

### Q: 为什么阈值选 0.7？
> "0.7 是经验值。过高会漏相关记忆，过低会召回无关内容。实际应该用评测集调优，统计不同阈值下的召回率和精确率。过敏史等高危降到 0.5 是安全优先。"

### Q: HITL 会不会误拦？
> "会。纯规则有误拦风险（如'看电影时胸痛'）。可以优化：规则匹配后 LLM 二次判断。hitl_events 表记录所有拦截，人工审核后标 human_reviewed=true，统计误拦率优化规则。医疗场景宁可误拦也不能漏拦。"

### Q: 记忆会不会越来越多？
> "会。可以定期清理：低重要性（1-3分）且超过 90 天的自动删除；或设上限（如 500 条），超出后删最老的低优先级记忆。过敏史、慢性病（8-10分）永久保留。"

---

## 📁 新增文件清单

```
app/services/
├── memory_service.py          # 记忆服务（250 行）
└── hitl_service.py            # HITL 服务（280 行）

app/db/
└── models.py                  # 新增 UserMemory 模型

migrations/
└── 003_memory_hitl.sql        # 数据库迁移脚本

tests/
└── test_memory_hitl.py        # 测试脚本

examples/
└── memory_hitl_demo.py        # 使用演示

docs/
├── phase3_memory_hitl.md      # 详细设计文档
├── MEMORY_HITL_SUMMARY.md     # 完整总结
└── MEMORY_HITL_CHEATSHEET.md  # 面试速查卡 ⭐
```

---

## ✅ 对应清单勾选

- [x] **【P1】记忆系统（短期/长期/工作记忆）**
  - ✓ 长期记忆表 + 向量检索
  - ✓ 相关性过滤防污染
  - ✓ 高优先级降低阈值
  - ✓ 短期记忆滑动窗口

- [x] **【P1】HITL 人在回路（医疗安全核心）**
  - ✓ 高危关键词前置拦截
  - ✓ escalate_to_human 工具
  - ✓ hitl_events 审计表
  - ✓ 三级紧急程度分类

---

## 🎯 面试建议

### 讲解顺序
1. **先讲问题**："新对话被旧记忆污染"/"高危症状 Agent 不能乱判断"
2. **再讲方案**：向量相似度阈值 / 双层拦截
3. **最后讲亮点**：高优先级降低阈值 / 审计闭环

### 演示准备
- 准备两个 query：一个相关（召回记忆）、一个无关（过滤掉）
- 准备一个高危 query（"胸痛喘不上气"），演示 HITL 拦截

### 加分话术
- "我一开始就考虑到了记忆污染问题"（预见性）
- "hitl_events 表能分析哪些场景自动处理失败"（闭环思维）
- "医疗场景宁可误拦也不能漏拦"（领域理解）

---

## 🔗 相关面试题

- **06-Q6**: LangChain Memory 类型（短期/长期/工作记忆）
- **05-Q6**: Agent 记忆设计（对话历史/任务意图/工具调用）
- **05-Q12**: Agent 安全（HITL / 防死循环 / 防注入）
- **08-Q8**: 长期记忆实现（向量库检索）

---

**⭐ 面试前 5 分钟必看：[MEMORY_HITL_CHEATSHEET.md](MEMORY_HITL_CHEATSHEET.md)**  
**🎯 记住核心：向量相似度 0.7 + HITL 双层拦截 + 三层记忆架构**
