# 阶段二 P0 测试指南

## 1. 当前完成度判断

阶段二 P0 主体代码已经覆盖，但清单文件还没有勾选，且真实 BGE 模型默认没有启用。

| P0 项 | 当前代码位置 | 状态 |
|---|---|---|
| pgvector + Embedding 向量检索 | `app/db/models.py`, `app/db/session.py`, `app/services/embedding_service.py`, `app/services/vector_store.py` | 已实现。默认 `local_hashing`，可切 `sentence_transformers` + `BAAI/bge-m3` |
| 递归切片 + 参数 | `app/services/text_splitter.py`, `app/services/corpus_indexer.py` | 已实现，默认 `chunk_size=500`, `overlap=50` |
| 向量 + BM25 + KG 三路混合 | `app/services/bm25_service.py`, `app/services/hybrid_retriever.py`, `app/services/chat_service.py` | 已实现。KG 来源由聊天编排先生成后注入混合检索 |
| Reranker 精排 | `app/services/reranker_service.py`, `app/services/hybrid_retriever.py` | 已实现。默认本地精排，可切 CrossEncoder |
| 引用溯源 | `app/services/rag_types.py`, `app/services/hybrid_retriever.py`, `app/api/v1/chat.py` | 已实现。SSE `meta.sources` 返回来源，Prompt 注入 `[1]` 编号来源 |
| GraphRAG 路径证据 | `app/services/graphrag_service.py`, `app/services/kg_service.py`, `app/services/chat_service.py` | 已实现。基于 NER 实体在 Neo4j 上做 schema-aware 1-2 跳扩展，并作为 KG 来源进入混合检索 |

测试结论应分两层写：

- 链路测试：默认配置即可，验证入库、召回、融合、精排、引用来源。
- 效果测试：启用 BGE-M3 和 bge-reranker-v2-m3 后再记录正式 Recall@K。

## 2. 医学文档来源

仓库内已有可直接入库的医学语料：

- `data/medical_new_2.json`：默认入库语料，约 8808 条疾病记录，当前接口默认读取它。
- `data/medical.json`：原始疾病文本源，可作为后续扩充语料。
- `data/rag_eval/sample_medical_upload.txt`：新增的小型 UTF-8 文本，用来测试上传入库。

外部可扩展来源：

- 国家卫健委、医院官网公开健康科普。
- 公开药品说明书字段：用法用量、禁忌、不良反应、相互作用。
- 导诊/分诊资料：症状到科室、急诊高危症状说明。

现有 `/documents/upload` 只支持 UTF-8 文本文件；PDF、Word 需要先转成 txt/markdown。

## 3. 入库测试

启动依赖：

```powershell
cd RAGQnASystem
docker compose up -d
uv run uvicorn app.main:app --reload --port 8000
```

首次启动会创建管理员：`admin / admin123`。

获取管理员 token：

```powershell
$token = (Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/api/v1/auth/login" `
  -ContentType "application/x-www-form-urlencoded" `
  -Body "username=admin&password=admin123").access_token
```

入库内置医学语料。建议先用小批量跑通：

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

上传单个测试文档：

```powershell
curl.exe -X POST "http://localhost:8000/api/v1/documents/upload" `
  -H "Authorization: Bearer $token" `
  -F "file=@data/rag_eval/sample_medical_upload.txt"
```

## 4. 检索与精排指标

新增脚本：

```powershell
uv run python scripts/evaluate_rag_p0.py --init-db --top-k 5 --candidate-k 50 --show-failures
```

输出列含义：

- `vector`：纯向量 Recall@5。
- `bm25`：纯 BM25 Recall@5。
- `hybrid_fused`：向量 + BM25 + KG 加权融合后、不精排的 Recall@5。
- `hybrid_rerank`：候选池 top50 经 Reranker 精排 top5 后的 Recall@5。
- `candidate_pool`：候选池 Recall@50，是 Reranker 能达到的上限。

解读方式：

- `candidate_pool` 低：粗召回不够，优先调 embedding、chunk、BM25、query 改写。
- `candidate_pool` 高但 `hybrid_rerank` 低：候选里有答案，但精排没排上来，调 reranker 或融合权重。
- `hybrid_rerank` 高于 `hybrid_fused`：说明精排有效。
- `bm25` 对药名、疾病名、数字问题高于 `vector`：说明精确匹配补足了向量召回。

Recall@5 计算口径：

```text
Recall@5 = top5 中命中 expected_sources 或 expected_keywords 的问题数 / 总问题数
```

MRR 计算口径：

```text
MRR = 每个问题第一个命中结果的 1/rank 的平均值
```

当前 12 条种子集的本地基线结果：

| 方法 | Recall@5 | MRR |
|---|---:|---:|
| vector | 0.9167 | 0.9167 |
| bm25 | 0.9167 | 0.9167 |
| hybrid_fused | 0.9167 | 0.9167 |
| hybrid_rerank | 1.0000 | 1.0000 |
| candidate_pool | 1.0000 | 1.0000 |

这只能证明链路和小样例有效，不能当最终面试指标。正式指标建议扩到 50-80 条，并启用真实 BGE-M3 / bge-reranker 后重跑。

## 5. 切片参数对比

建议比较三组：

| 组 | chunk_size | overlap |
|---|---:|---:|
| small | 300 | 50 |
| default | 500 | 50 |
| large | 800 | 80 |

注意：当前 `document_chunks` 没有按实验组隔离。比较切片参数时应使用独立测试库，或在一次性测试库里清空 `document_chunks` 后重建索引，否则不同 chunk_size 的结果会混在一起，指标不可信。

记录格式：

```text
chunk_size=300 overlap=50: hybrid_rerank Recall@5=?, MRR=?, avg_latency=?
chunk_size=500 overlap=50: hybrid_rerank Recall@5=?, MRR=?, avg_latency=?
chunk_size=800 overlap=80: hybrid_rerank Recall@5=?, MRR=?, avg_latency=?
```

## 6. 启用真实 BGE 测试

默认依赖里还没有 `sentence-transformers`。要测正式效果，需要安装模型依赖并修改 `.env`：

```powershell
uv add sentence-transformers torch
```

`.env`：

```env
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
RERANKER_PROVIDER=sentence_transformers
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RAG_TOP_K=5
RAG_CANDIDATE_K=50
```

切换 embedding 模型后必须重新入库，否则库里的 embedding 仍是旧模型生成的。

## 7. 引用溯源测试

调用 `/api/v1/chat` 时观察 SSE 的 `meta` 事件：

- `sources[*].citation_id` 应从 1 开始编号。
- `sources[*].source_title` 应有疾病名、上传文档名或 `Neo4j 知识图谱`。
- `sources[*].section` 应能说明来源章节。
- `sources[*].authority_level` 应能区分结构化医学语料、上传文档或 KG。

答案正文应出现 `[1]`、`[2]` 这类引用编号；没有来源支持时，应倾向回答“不确定，建议咨询医生”。

## 8. GraphRAG 路径证据测试

GraphRAG 入口：

- `app/services/graphrag_service.py`：从 NER 命中的疾病、症状、药品等实体出发，按医疗 KG schema 扩展短路径。
- `app/services/chat_service.py`：把 GraphRAG 路径证据合入 `kg_knowledge`，再交给 hybrid retriever 和 reranker。
- `app/services/hybrid_retriever.py`：GraphRAG 来源仍标记为 `source_type=kg`，但 `section` 为 `GraphRAG 路径 N`，`metadata.retrieval_method=graphrag`。

典型路径：

```text
疾病症状:胸痛 -[疾病的症状]- 疾病:冠心病 -[疾病所属科目]- 科目:心血管内科
疾病:高血压 -[疾病并发疾病]- 疾病:脑卒中 -[疾病所属科目]- 科目:神经内科
药品:阿莫西林 -[疾病使用药品]- 疾病:支气管炎
```

单元测试：

```powershell
uv run pytest tests/test_graphrag_service.py
```

面试讲法边界：

- Neo4j 是图数据库/知识图谱存储，不等于完整 GraphRAG。
- 这里新增的是 GraphRAG 检索策略：先实体锚定，再沿图谱关系做 1-2 跳扩展，把路径转成可引用上下文。
- 生产里常见做法不是只选“Neo4j 或向量库”，而是 KG/GraphRAG + vector RAG + BM25 + reranker 的混合检索。微软 GraphRAG 更偏文档自动建图和社区摘要，本项目基于已有医疗 KG 做轻量 GraphRAG，更符合当前基座。

## 9. P1 RAG 评测集 + RAGAS 风格评测

新增 50 条正式种子集：

```text
data/rag_eval/rag_eval_cases.jsonl
```

字段说明：

- `query`：用户问题。
- `expected_sources` / `expected_keywords`：检索层命中口径。
- `reference_answer`：参考答案。
- `answer_keywords`：生成答案应覆盖的关键医学要点。
- `kg_knowledge`：需要 KG 兜底的高危用例。

### 9.1 只测检索和上下文质量

不调用 LLM，成本最低，适合频繁回归：

```powershell
uv run python scripts/evaluate_rag_ragas.py --init-db --top-k 5 --candidate-k 50
```

首次或空库时可加 `--index-corpus`，先入库内置语料：

```powershell
uv run python scripts/evaluate_rag_ragas.py `
  --init-db `
  --index-corpus `
  --index-limit 1000 `
  --top-k 5 `
  --candidate-k 50
```

输出指标：

- `retrieval_recall_at_k`：TopK 是否命中期望来源或关键词。
- `retrieval_mrr`：第一个命中结果的平均倒数排名。
- `context_precision`：相关上下文是否排在更靠前的位置。
- `context_recall`：召回上下文是否覆盖标准答案关键词。
- `avg_latency_ms`：单条检索平均耗时。

### 9.2 测生成答案质量

会调用当前配置的 DeepSeek/OpenAI 兼容模型，建议先用少量用例：

```powershell
uv run python scripts/evaluate_rag_ragas.py `
  --init-db `
  --top-k 5 `
  --candidate-k 50 `
  --generate-answers `
  --max-cases 10
```

额外输出：

- `answer_relevancy_proxy`：答案覆盖 `answer_keywords` 的比例。
- `faithfulness_proxy`：答案中的关键结论是否能被召回上下文支撑。
- `citation_precision`：答案引用编号是否都来自本轮检索来源。
- `hallucination_rate_proxy`：`1 - faithfulness_proxy`，作为幻觉率近似值。

每次运行都会写报告：

```text
data/rag_eval/reports/rag_eval_report_YYYYMMDD_HHMMSS.json
data/rag_eval/reports/ragas_input_YYYYMMDD_HHMMSS.jsonl
```

报告可用于面试展示；`ragas_input_*.jsonl` 是 RAGAS 兼容格式，含
`question / answer / contexts / ground_truth`。

### 9.3 可选真实 RAGAS

如果本地另行安装了 `ragas` 和 `datasets`，并配置好 judge LLM，可加：

```powershell
uv run python scripts/evaluate_rag_ragas.py `
  --generate-answers `
  --max-cases 10 `
  --use-ragas
```

未安装或 judge LLM 未配置时，脚本不会中断，会在报告里的 `ragas.status` 说明原因。生产展示建议同时保留本地 proxy 指标和真实 RAGAS 输出，避免外部 LLM 评测不可复现。
