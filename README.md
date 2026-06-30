<div align="center">

# RAGQnASystem 二次开发版

### 智能导诊与健康问答助手

基于原始医疗知识图谱问答项目二次开发，当前主线已重构为 **FastAPI 后端 + React 前端 + 混合 RAG + 医疗 Agent**。

[快速开始](#快速开始) · [当前能力](#当前能力) · [接口说明](#接口说明) · [项目结构](#项目结构) · [原项目基座](#原项目基座)

</div>

---

## 项目定位

本仓库最初基于医疗知识图谱问答系统改造而来。原项目的核心能力是：从医疗知识图谱中抽取疾病、症状、药品、检查、科室等结构化信息，再结合 NER 和大模型生成医疗问答结果。

我在此基础上进行了二次开发，当前项目重点已经从单体 Streamlit Demo 演进为可前后端分离部署的医疗问答应用：

- 后端使用 `FastAPI` 提供 API 服务，支持 JWT 鉴权、SSE 流式问答、会话持久化和管理员文档入库。
- 数据层新增 `PostgreSQL + pgvector`、`Redis`，保留 `Neo4j` 医疗知识图谱作为结构化知识源。
- RAG 从原来的知识图谱检索扩展为 `向量检索 + BM25 + KG + GraphRAG + Reranker` 的混合检索。
- Agent 层引入工具调用、LangGraph 状态编排、MCP 可插拔工具、长期记忆和 HITL 高危医疗拦截。
- 前端新增 `React + Vite + Ant Design`，原 `Streamlit` 入口仍保留为旧版/调试参考，不再是当前主入口。

> 医疗回答仅用于健康科普和导诊辅助，不能替代医生诊断、处方或急救判断。

---

## 当前能力

| 模块 | 当前实现 |
|---|---|
| 用户与鉴权 | 注册、登录、JWT Bearer Token、默认管理员初始化 |
| 会话系统 | 多轮会话、消息持久化、历史上下文补全、反馈记录 |
| 流式问答 | `/api/v1/chat` 返回 SSE：`meta`、`token`、`done`、`error` |
| 医疗 NER | 复用原项目实体词典，使用 AC 自动机 + TF-IDF 标准化实体 |
| 知识图谱 | 复用原项目 Neo4j 医疗 KG 查询能力，支持疾病属性、关系、药品生产商等 |
| 混合 RAG | pgvector/JSONB 向量检索、BM25、KG 证据融合、Reranker 精排、引用溯源 |
| Query 增强 | 多轮 query 改写、医疗 query 改写、Multi-Query、HyDE |
| GraphRAG | 基于 Neo4j 实体锚点做 1-2 跳路径扩展，生成可引用图谱证据 |
| Agent Tool Use | 内置 KG 查询、向量检索、药品查询、导诊分级、澄清追问、转人工工具 |
| LangGraph | Agent 状态机编排、循环保护、超时和 token budget 控制 |
| MCP | 可选接入外部 MCP stdio server，把外部工具暴露为 Agent 函数 |
| 记忆系统 | 用户长期记忆、短期窗口、相似度阈值过滤，降低跨会话记忆污染 |
| HITL | 高危症状和复杂用药场景前置拦截，支持事件审计表 |
| 前端 | React/Vite/Ant Design，支持登录注册、会话、流式聊天、引用来源、文档入库 |

---

## 技术栈

| 层级 | 技术 |
|---|---|
| 后端 API | FastAPI, Uvicorn, Pydantic v2 |
| 数据库 | PostgreSQL 14 + pgvector, Redis, Neo4j |
| ORM/存储 | SQLAlchemy 2.0 asyncio, asyncpg, JSONB |
| 大模型 | DeepSeek OpenAI 兼容接口，可替换为其他 OpenAI-compatible provider |
| RAG | local hashing embedding 默认兜底，可切换 BGE-M3；BM25；reranker 可切换 bge-reranker |
| Agent | LangGraph, Function Calling, MCP |
| 前端 | React 19, Vite, TypeScript, Ant Design |
| 测试 | pytest, pytest-asyncio, httpx ASGITransport |

---

## 快速开始

### 1. 准备环境

建议使用当前仓库的 `uv.lock` 安装依赖：

```powershell
uv sync
```

前端依赖在 `frontend/` 下单独安装：

```powershell
cd frontend
npm.cmd install
cd ..
```

基础要求：

- Python `>=3.12`
- Node.js / npm
- Docker Desktop
- DeepSeek API Key 或其他 OpenAI 兼容模型服务
- Neo4j 可选但推荐启用；完整 KG/GraphRAG 能力依赖它

### 2. 启动基础设施

仓库内的 `docker-compose.yml` 只启动 PostgreSQL(pgvector) 和 Redis，不包含 Neo4j：

```powershell
docker compose up -d
```

默认端口：

| 服务 | 地址 |
|---|---|
| PostgreSQL | `localhost:5433` |
| Redis | `localhost:6379` |
| FastAPI | `localhost:8000` |
| React Dev Server | `localhost:5173` |

### 3. 配置 `.env`

在项目根目录创建 `.env`：

```env
DATABASE_URL=postgresql+asyncpg://raguser:ragpass123@localhost:5433/ragqna
REDIS_URL=redis://localhost:6379/0

JWT_SECRET=replace-with-a-long-random-string
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

DEEPSEEK_API_KEY=sk-your-key
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com

NEO4J_URL=http://localhost:7474
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4jpass123
NEO4J_DBNAME=neo4j

MEDICAL_CORPUS_PATH=data/medical_new_2.json
ENABLE_PGVECTOR=true
EMBEDDING_PROVIDER=local_hashing
EMBEDDING_DIM=1024
RERANKER_PROVIDER=local
RAG_TOP_K=5
RAG_CANDIDATE_K=50

AGENT_TOOLS_ENABLED=true
AGENT_GRAPH_ENABLED=true
MCP_ENABLED=false

CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
APP_TIMEZONE=Asia/Shanghai
LOG_LEVEL=INFO
```

说明：

- 默认 `local_hashing` embedding 不需要下载模型，适合先跑通链路。
- 如需正式语义检索效果，可安装 `sentence-transformers` 后切换到 `BAAI/bge-m3` 和 `BAAI/bge-reranker-v2-m3`。
- Neo4j 未启动时 `/api/v1/health` 会显示 `degraded`，KG/GraphRAG 能力会降级。

### 4. 启动后端

必须在项目根目录启动，因为 NER 词典和部分旧基座模块使用相对路径：

```powershell
uv run uvicorn app.main:app --reload --port 8000
```

首次启动会自动建表，并初始化默认管理员：

```text
admin / admin123
```

后端入口：

- API 根路径：http://localhost:8000/
- Swagger 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/api/v1/health

如需启用 HITL 审计表，可在后端首次建表后执行：

```powershell
Get-Content migrations\003_memory_hitl.sql | docker exec -i ragqna-postgres psql -U raguser -d ragqna
```

没有执行该迁移时，高危拦截仍会工作，但审计事件会回退到日志记录。

### 5. 索引内置医疗语料

登录获取管理员 token：

```powershell
$token = (Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/api/v1/auth/login" `
  -ContentType "application/x-www-form-urlencoded" `
  -Body "username=admin&password=admin123").access_token
```

建议先小批量入库验证：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/api/v1/documents/index-medical-corpus" `
  -Headers @{ Authorization = "Bearer $token" } `
  -ContentType "application/json" `
  -Body '{"limit":200,"chunk_size":500,"overlap":50}'
```

查看 chunk 数：

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://localhost:8000/api/v1/documents/stats" `
  -Headers @{ Authorization = "Bearer $token" }
```

### 6. 启动前端

```powershell
cd frontend
npm.cmd run dev -- --host 127.0.0.1 --port 5173
```

访问：

```text
http://127.0.0.1:5173/
```

前端默认请求：

```text
http://localhost:8000/api/v1
```

如需覆盖，在 `frontend/.env.local` 中设置：

```env
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

---

## 接口说明

所有业务接口默认前缀为 `/api/v1`。

| 方法 | 路径 | 说明 | 权限 |
|---|---|---|---|
| `GET` | `/health` | PostgreSQL 和 Neo4j 健康检查 | 无 |
| `POST` | `/auth/register` | 注册普通用户 | 无 |
| `POST` | `/auth/login` | OAuth2 表单登录，返回 JWT | 无 |
| `GET` | `/auth/me` | 当前用户信息 | 登录 |
| `POST` | `/chat` | SSE 流式问答 | 登录 |
| `GET` | `/conversations` | 当前用户会话列表 | 登录 |
| `GET` | `/conversations/{conv_id}` | 会话详情与消息历史 | 登录 |
| `POST` | `/documents/index-medical-corpus` | 索引内置医疗 JSON 语料 | 管理员 |
| `POST` | `/documents/upload` | 上传 UTF-8 文本文档并入库 | 管理员 |
| `GET` | `/documents/stats` | 文档 chunk 统计 | 管理员 |
| `POST` | `/feedback` | 对助手回答点赞/点踩 | 登录 |

`/chat` 的 SSE 事件：

| 事件 | 内容 |
|---|---|
| `meta` | 会话 ID、NER 实体、意图、知识片段、引用来源、Agent 工具轨迹、query 改写信息 |
| `token` | 模型生成的增量文本 |
| `done` | 助手消息 ID、token 用量和成本估算 |
| `error` | 流式生成失败信息和已累计 token 用量 |

---

## 项目结构

```text
RAGQnASystem/
├── app/
│   ├── api/v1/              # FastAPI 路由：auth/chat/documents/feedback/health
│   ├── core/                # 配置、安全、生命周期、时间工具
│   ├── db/                  # SQLAlchemy models/session/seed
│   ├── schemas/             # Pydantic 请求/响应模型
│   └── services/            # RAG、KG、LLM、Agent、MCP、Memory、HITL 等服务
├── frontend/                # React + Vite + Ant Design 前端
├── config/                  # MCP 示例配置
├── data/
│   ├── medical_new_2.json   # 内置医疗语料/原 KG 数据源
│   ├── ent_aug/             # 原项目实体词典，规则 NER 依赖
│   └── rag_eval/            # RAG 评测集与报告目录
├── docs/                    # Phase 2/3 设计与测试文档
├── scripts/                 # RAG/Agent 评测和工具脚本
├── tests/                   # pytest 测试
├── migrations/              # 额外 SQL 迁移，如 HITL 审计表
├── build_up_graph.py        # 原项目 Neo4j 建图脚本
├── kg_client.py             # 原项目 KG 查询封装，当前后端继续复用
├── intent_router.py         # 表驱动意图路由，当前后端继续复用
├── rule_ner.py              # 当前后端使用的规则 NER 基座
├── login.py / webui.py      # 原 Streamlit 入口，当前仅作遗留/调试参考
├── docker-compose.yml       # PostgreSQL(pgvector) + Redis
└── pyproject.toml           # 当前二开版 Python 依赖
```

---

## 原项目基座

当前项目仍复用和保留了原始医疗 KG 问答系统的关键资产：

- `data/medical_new_2.json`：医疗知识图谱源数据。
- `data/ent_aug/*.txt`：疾病、症状、药品、食物、检查、科室等实体词典。
- `build_up_graph.py`：把医疗 JSON 数据导入 Neo4j。
- `kg_client.py`：封装疾病属性、关系实体、症状反查疾病、药品生产商查询。
- `intent_router.py`：把 LLM 识别出的意图路由到具体 KG 查询。
- `login.py`、`webui.py`、`ner/`：原 Streamlit/BERT NER 相关代码，当前 FastAPI 主链路不依赖 BERT 权重启动。

如果需要完整 KG/GraphRAG 效果，请单独启动 Neo4j，并用 `build_up_graph.py` 导入医疗图谱。PostgreSQL/Redis 的 `docker-compose.yml` 不会自动创建 Neo4j。

---

## 测试与评测

运行单元测试：

```powershell
uv run pytest
```

常用定向测试：

```powershell
uv run pytest tests/test_auth.py
uv run pytest tests/test_chat.py
uv run pytest tests/test_graphrag_service.py
uv run pytest tests/test_agent_graph.py
uv run pytest tests/test_mcp_client.py
uv run pytest tests/test_memory_hitl.py
```

RAG 链路评测：

```powershell
uv run python scripts/evaluate_rag_p0.py --init-db --top-k 5 --candidate-k 50 --show-failures
```

RAGAS 风格评测：

```powershell
uv run python scripts/evaluate_rag_ragas.py --init-db --top-k 5 --candidate-k 50
```

首次或空库时可加 `--index-corpus` 先索引内置语料。更详细的评测说明见 `docs/phase2_p0_test_guide.md`。

---

## 可选增强配置

### 切换真实 embedding / reranker

默认配置以 `local_hashing` 跑通链路。如果要测试更接近生产的语义召回：

```powershell
uv add sentence-transformers torch
```

`.env` 调整：

```env
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
RERANKER_PROVIDER=sentence_transformers
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

切换 embedding 模型后，需要重新索引 `document_chunks`，否则库里的向量仍是旧模型生成的。

### 启用 MCP

参考 `config/mcp_servers.example.json`：

```env
MCP_ENABLED=true
MCP_SERVER_CONFIG_PATH=config/mcp_servers.example.json
```

MCP server 以 stdio 方式启动，外部命令如 `npx`、具体 MCP server 包需要在本机可用。未配置或启动失败时，Agent 会把失败作为工具观测返回，不会中断整轮对话。

---

## 安全与限制

- 默认管理员 `admin/admin123` 仅用于本地开发，首次启动后应尽快修改或替换初始化逻辑。
- `.env` 中包含数据库密码、JWT 密钥和模型 API Key，不要提交到仓库。
- 当前开发期使用 `metadata.create_all` 自动建表，生产环境建议补齐 Alembic 迁移流程。
- `/documents/upload` 目前只支持 UTF-8 文本文件；PDF、Word 需要先转成 txt/markdown。
- `web_search` 工具当前是占位降级实现，不会真正联网检索。
- 医疗 Agent 会对高危症状触发 HITL/急诊建议，但不能替代线下医生和急救系统。

---

## 致谢与来源

本项目二次开发基于原医疗知识图谱问答思路与公开数据资源：

- 数据集来源：[Open-KG Disease Information](http://data.openkg.cn/dataset/disease-information)
- 参考项目：[RAGOnMedicalKG](https://github.com/liuhuanyong/RAGOnMedicalKG)
- 参考项目：[QASystemOnMedicalKG](https://github.com/liuhuanyong/QASystemOnMedicalKG)

