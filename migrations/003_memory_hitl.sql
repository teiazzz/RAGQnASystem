-- Phase 3: 记忆系统与 HITL 人在回路 (Memory & HITL)
-- 新增表：user_memories（长期记忆）、hitl_events（转人工事件审计）

-- 1. 用户长期记忆表
CREATE TABLE IF NOT EXISTS user_memories (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    category VARCHAR(32) NOT NULL DEFAULT 'health_fact',  -- health_fact / allergy / chronic_disease / medication_history / decision
    content TEXT NOT NULL,
    relevance_keywords JSONB,  -- ["糖尿病", "二甲双胍"] 快速过滤用
    embedding JSONB,  -- 向量检索用（JSONB 格式保证可移植性）
    source_conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
    importance INTEGER NOT NULL DEFAULT 5,  -- 1-10，越高越重要
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(0),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(0)
);

CREATE INDEX IF NOT EXISTS ix_user_memories_user_id ON user_memories(user_id);
CREATE INDEX IF NOT EXISTS ix_user_memories_user_id_created ON user_memories(user_id, created_at);
CREATE INDEX IF NOT EXISTS ix_user_memories_category ON user_memories(category);

COMMENT ON TABLE user_memories IS '用户长期记忆：过敏史、慢性病、关键健康事实';
COMMENT ON COLUMN user_memories.category IS '记忆类型：health_fact（健康事实）/ allergy（过敏史）/ chronic_disease（慢性病）/ medication_history（用药史）/ decision（医疗决策）';
COMMENT ON COLUMN user_memories.embedding IS '向量检索用，JSONB 存储确保无 pgvector 扩展时也能降级运行';
COMMENT ON COLUMN user_memories.importance IS '重要性 1-10，过敏史/慢性病等高危记忆设为 8-10，召回阈值会降低';
COMMENT ON COLUMN user_memories.relevance_keywords IS '启发式关键词，用于快速过滤（和向量检索互补）';

-- 2. HITL 转人工事件审计表
CREATE TABLE IF NOT EXISTS hitl_events (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
    query TEXT NOT NULL,
    urgency VARCHAR(32) NOT NULL,  -- emergency / urgent / routine
    reason TEXT NOT NULL,
    matched_keywords JSONB,  -- 命中的高危关键词
    agent_trace JSONB,  -- Agent 完整执行轨迹（用于分析）
    human_reviewed BOOLEAN DEFAULT FALSE,  -- 是否已人工审核
    human_reviewer_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    human_notes TEXT,  -- 人工审核备注
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(0)
);

CREATE INDEX IF NOT EXISTS ix_hitl_events_user_id ON hitl_events(user_id);
CREATE INDEX IF NOT EXISTS ix_hitl_events_urgency ON hitl_events(urgency);
CREATE INDEX IF NOT EXISTS ix_hitl_events_created_at ON hitl_events(created_at);
CREATE INDEX IF NOT EXISTS ix_hitl_events_human_reviewed ON hitl_events(human_reviewed);

COMMENT ON TABLE hitl_events IS 'HITL 转人工事件审计日志，记录所有高危拦截和转人工请求';
COMMENT ON COLUMN hitl_events.urgency IS '紧急程度：emergency（立即急诊）/ urgent（尽快人工）/ routine（常规转人工）';
COMMENT ON COLUMN hitl_events.matched_keywords IS '触发转人工的关键词（如["胸痛", "呼吸困难"]）';
COMMENT ON COLUMN hitl_events.agent_trace IS 'Agent 执行轨迹 JSON，用于分析哪些场景自动处理失败';

-- 可选：如果需要 pgvector，可以动态添加向量列（运行时检测）
-- ALTER TABLE user_memories ADD COLUMN IF NOT EXISTS embedding_vector vector(1024);
-- CREATE INDEX IF NOT EXISTS ix_user_memories_embedding_vector ON user_memories USING hnsw (embedding_vector vector_cosine_ops) WITH (m = 16, ef_construction = 64);
