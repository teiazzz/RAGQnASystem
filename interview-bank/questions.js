window.interviewQuestions = [
  {
    id: 'overview-001',
    stage: '项目总览',
    difficulty: '基础',
    tags: ['项目介绍', 'AI应用', '医疗问答'],
    question: '请用 1-2 分钟介绍一下你的智能导诊与健康问答项目。',
    shortAnswer:
      '基于开源医疗 KG 问答项目做工程化二次开发，核心是 FastAPI + React + 混合 RAG + LangGraph Agent + 医疗安全。',
    answer: `我这个项目是一个智能导诊与健康问答助手，用户可以用自然语言描述症状、疾病、药品或检查问题，系统会给出健康科普、可能关联疾病、推荐科室、紧急程度建议，并尽量标注回答依据。

项目不是从零造医疗知识库，而是基于一个开源医疗知识图谱问答项目做二次开发。原项目主要是 Streamlit 单体、Neo4j 知识图谱、BERT+RNN NER 和固定流程问答。我主要做了几类改造：

第一是工程架构改造，把原 Streamlit 单体拆成 FastAPI 后端和 React 前端，接入 PostgreSQL、JWT、异步接口、SSE 流式输出、Docker Compose。

第二是 RAG 能力升级。原项目只有知识图谱检索，我增加了 pgvector 向量检索、BM25 精确匹配、Neo4j GraphRAG 路径证据，再用 Reranker 精排，并做引用溯源。

第三是 Agent 化。把固定流程改成 LangGraph 状态机，定义了知识图谱查询、向量检索、药品查询、分诊评估、反问澄清、转人工等工具，并做了工具调用 trace 和控制流保护。

第四是医疗场景安全，包括多轮 query 改写、长期记忆、HITL 高危症状转人工、PII 脱敏思路、评测集和 RAGAS 风格评测。整体目标是把一个医疗问答 demo 改造成更接近真实 AI 应用工程的系统。`,
    followUps: [
      '哪些是开源项目已有的，哪些是你新增的？',
      '为什么这个项目能体现 AI 应用开发能力？',
      '如果只讲一个亮点，你会讲哪个？',
    ],
    pitfalls: [
      '不要说知识图谱和 NER 是自己从零训练/构建的。',
      '不要只堆技术名词，要说清楚业务问题和工程取舍。',
    ],
    relatedFiles: ['README.md', 'app/main.py', 'app/services/chat_service.py'],
  },
  {
    id: 'architecture-001',
    stage: '架构改造',
    difficulty: '中等',
    tags: ['FastAPI', 'React', '前后端分离', '异步'],
    question: '你为什么要把原来的 Streamlit 单体重构成 FastAPI + React？',
    shortAnswer:
      'Streamlit 适合 demo，但生产形态需要 API 化、鉴权、会话、异步、流式、可观测和前端交互能力。',
    answer: `原项目的 Streamlit 形态适合本地演示，但它把 UI、流程编排、模型调用和状态管理混在一起，后续接入鉴权、文档入库、会话管理、流式输出和 Agent trace 都会比较困难。

我重构成 FastAPI + React 有几个原因：

第一，FastAPI 更适合做 AI 应用后端。LLM、Embedding、RAG 检索、数据库访问都可以用 async 方式组织，接口可以用 Pydantic 明确请求和响应结构，也方便做 Depends 注入当前用户、DB session、限流器等。

第二，React 前端更适合做复杂交互。比如 SSE 流式逐字渲染、引用来源点击高亮、Agent 状态机时间线、工具调用 observation、文档入库面板，这些在 Streamlit 里可控性较弱。

第三，前后端分离后，系统边界更清楚。后端只暴露 /api/v1/chat、/documents、/auth、/feedback 这些接口，前端只负责交互和可视化，部署时也可以独立扩缩容。

所以这个重构不是为了换框架，而是为了让项目从 demo 形态走向 AI 应用工程形态。`,
    followUps: [
      'FastAPI 的 async 对 LLM 应用有什么价值？',
      'SSE 为什么不直接用 WebSocket？',
      'Pydantic 在项目里解决了什么问题？',
    ],
    pitfalls: [
      '不要把 Streamlit 说得一无是处，它适合原型验证。',
      '不要只说“FastAPI 性能高”，要结合异步 LLM/RAG 链路讲。',
    ],
    relatedFiles: ['app/api/v1/chat.py', 'app/api/deps.py', 'frontend/src/App.tsx'],
  },
  {
    id: 'rag-001',
    stage: 'RAG 检索',
    difficulty: '深挖',
    tags: ['RAG', '混合检索', 'BM25', 'pgvector', 'Neo4j'],
    question: '你这个项目的 RAG 检索链路是怎么设计的？',
    shortAnswer:
      'Query 改写后走向量、BM25、KG/GraphRAG 三路召回，融合候选后 Rerank Top5，再注入带编号来源的 Prompt。',
    answer: `我的 RAG 链路分成几个阶段。

第一步是 query 预处理。多轮场景先把“它有什么副作用”这类问题结合历史改写成独立问题，然后再做医学 query 改写、Multi-Query 和 HyDE，尽量缩小用户口语和医学术语之间的差距。

第二步是多路召回。向量检索负责语义相似，比如“老是头晕想吐”这类口语描述；BM25 负责精确匹配药名、疾病名、数字等词面信息；Neo4j KG/GraphRAG 负责结构化关系，比如疾病到症状、疾病到科室、疾病到检查项目。

第三步是候选融合。三路召回各有分数，我会归一化后融合成候选池，比如 top50。这样避免单一路径失效：纯向量可能漏掉专有名词，纯 BM25 不理解语义，纯 KG 对开放问题覆盖不足。

第四步是 Reranker 精排。粗召回是 Bi-Encoder 或词面匹配，精排用 Cross-Encoder 思路把 query 和文档一起打分，把 top50 精排成 top5。

最后是引用溯源。每个 chunk 带 source_title、section、authority_level、citation_id，Prompt 中注入 [1] [2] 编号来源，要求模型使用来源时标注引用；如果来源不足，就说明没有可靠依据，不强行回答。`,
    followUps: [
      '为什么不用纯向量检索？',
      'BM25 在医疗场景有什么价值？',
      'KG 和 GraphRAG 在你的项目里是什么关系？',
    ],
    pitfalls: [
      '不要把 Neo4j 等同于 GraphRAG；Neo4j 是存储，GraphRAG 是检索策略。',
      '不要只说“混合检索更准”，要解释不同召回路的互补性。',
    ],
    relatedFiles: [
      'app/services/hybrid_retriever.py',
      'app/services/vector_store.py',
      'app/services/bm25_service.py',
      'app/services/graphrag_service.py',
    ],
  },
  {
    id: 'rag-002',
    stage: 'RAG 检索',
    difficulty: '中等',
    tags: ['切片', 'Embedding', 'BGE-M3'],
    question: '你的文档切片策略是怎样的？为什么这么设置？',
    shortAnswer:
      '使用递归切片，默认 chunk_size=500、overlap=50，在语义完整性和检索粒度之间做平衡。',
    answer: `我用的是递归切片，默认 chunk_size 是 500，overlap 是 50。递归切片的好处是尽量按段落、句子、标点边界切，不会像固定长度切片那样容易把一个医学结论切断。

chunk 太小的问题是上下文不足，模型可能只看到症状但看不到建议；chunk 太大的问题是噪声多、向量表示不聚焦，召回后也会占用更多上下文窗口。500 左右对于中文医疗科普、疾病简介、用药说明这种文本比较平衡。

overlap 设置为 50 是为了保留边界上下文。例如一个禁忌说明刚好跨段，overlap 可以降低被切断造成的召回损失。

我不会说这个参数天然最优，正式做法是用评测集比较 300/50、500/50、800/80 等组合，看 Recall@5、MRR、平均延迟和答案质量，再选最稳的一组。`,
    followUps: [
      'chunk 越大越好吗？',
      '切片参数怎么评测？',
      '医学文档和普通 FAQ 的切片有什么不同？',
    ],
    pitfalls: [
      '不要说 chunk_size=500 是绝对最优，它是当前项目基线。',
      '不要忽略 overlap 对边界信息的影响。',
    ],
    relatedFiles: ['app/services/text_splitter.py', 'app/services/corpus_indexer.py'],
  },
  {
    id: 'rag-003',
    stage: 'RAG 检索',
    difficulty: '深挖',
    tags: ['Reranker', 'Bi-Encoder', 'Cross-Encoder'],
    question: '为什么加 Reranker？它和 Embedding 相似度有什么区别？',
    shortAnswer:
      'Embedding 是双塔粗排，速度快但交互弱；Reranker 是 Cross-Encoder 精排，更慢但能更细地判断 query 和文档是否匹配。',
    answer: `Embedding 检索一般是 Bi-Encoder，query 和文档分别编码成向量，再算余弦相似度。它的优点是快，适合大规模召回；缺点是 query 和文档没有充分交互，容易把“相关但不能回答”的内容排到前面。

Reranker 更接近 Cross-Encoder，会把 query 和候选文档一起输入模型，让模型直接判断这段文档能不能回答这个问题。它比向量相似度更准，尤其适合医疗问答中这种细粒度判断：比如用户问“阿司匹林禁忌”，包含“阿司匹林简介”的 chunk 相关但不一定能回答禁忌，Reranker 可以把真正讲禁忌的 chunk 排前。

我的做法不是全量 Rerank，而是先三路粗召回 top50，再 Rerank top5。这样在成本和效果之间平衡。面试中我会强调：Reranker 能提升 topK 质量，但前提是候选池里已经召回了正确答案。如果 candidate_pool Recall@50 很低，Reranker 也救不了，需要回头优化切片、Embedding、BM25 或 query 改写。`,
    followUps: [
      '如果 Reranker 后效果下降怎么办？',
      'Reranker 的延迟怎么控制？',
      'candidate_pool 指标为什么重要？',
    ],
    pitfalls: [
      '不要把 Reranker 当成万能优化，它依赖粗召回候选池。',
      '不要忽略 Cross-Encoder 的延迟成本。',
    ],
    relatedFiles: ['app/services/reranker_service.py', 'app/services/hybrid_retriever.py'],
  },
  {
    id: 'rag-004',
    stage: 'RAG 检索',
    difficulty: '深挖',
    tags: ['GraphRAG', 'Neo4j', '知识图谱'],
    question: '你项目里的 GraphRAG 是怎么做的？和普通 Neo4j 查询有什么区别？',
    shortAnswer:
      'Neo4j 是图存储；GraphRAG 是先实体锚定，再沿医疗 KG schema 做 1-2 跳路径扩展，并把路径证据转成可引用上下文。',
    answer: `我会先区分两个概念：Neo4j 是图数据库，用来存疾病、症状、药品、检查、科室等节点和关系；GraphRAG 是一种检索策略，不是只要用了 Neo4j 就等于做了 GraphRAG。

我的实现是基于 NER 抽出的实体做实体锚定，比如识别到“高血压”“胸痛”“阿司匹林”。然后根据医疗 KG schema 做 1-2 跳路径扩展，比如 症状 -> 疾病 -> 科室，疾病 -> 并发症 -> 科室，药品 -> 相关疾病。扩展出的路径会被格式化成知识证据，带上 source_type=kg、section=GraphRAG 路径、metadata.graph_path，再进入混合检索和引用溯源链路。

普通 Neo4j 查询更像根据固定意图查某个属性或关系，比如查高血压的治疗方法；GraphRAG 更强调围绕实体展开关联路径，为多跳问题、导诊问题提供结构化证据。它和向量 RAG 不是替代关系，而是互补关系。`,
    followUps: [
      '微软 GraphRAG 和你这个有什么不同？',
      'GraphRAG 会不会引入错误路径？',
      '为什么只做 1-2 跳，不做更多跳？',
    ],
    pitfalls: [
      '不要把“用了 Neo4j”直接包装成完整 GraphRAG。',
      '不要无限扩展图路径，跳数越多噪声越大。',
    ],
    relatedFiles: ['app/services/graphrag_service.py', 'app/services/kg_service.py'],
  },
  {
    id: 'rag-005',
    stage: 'RAG 检索',
    difficulty: '中等',
    tags: ['引用溯源', '幻觉控制', 'Prompt'],
    question: '医疗问答里你是怎么做引用溯源和幻觉控制的？',
    shortAnswer:
      'chunk 保留来源元数据，Prompt 注入编号来源并强约束引用；来源不足时走不确定或模型兜底，不伪造引用。',
    answer: `医疗问答不能只追求回答流畅，更重要的是可追溯和不乱答。我主要做了三层控制。

第一层是数据层。每个 chunk 入库时保留 source_title、section、authority_level、content_preview 等元数据。GraphRAG 路径也会保留 graph_path 元数据。

第二层是 Prompt 层。召回后把来源按 [1] [2] 编号注入 prompt，要求模型使用某条来源的信息时，在对应句子末尾标注来源编号。如果来源没有覆盖问题，就明确说当前知识库没有可靠依据，建议咨询医生。

第三层是链路层。我会先判断来源是否有最小词面/实体信号，必要时用轻量 LLM 判断来源是否足以回答。如果不足，就走 model_fallback，并明确告诉用户“以下回答没有引用来源”，不允许模型编造 [1] [2]。

这样做的核心不是完全消灭幻觉，而是把模型回答约束在可验证上下文里，并在不确定时诚实降级。`,
    followUps: [
      '模型仍然乱引用怎么办？',
      'citation_precision 怎么算？',
      '没有来源时为什么还允许模型兜底？',
    ],
    pitfalls: [
      '不要承诺 100% 无幻觉。',
      '不要在没有来源时伪造引用编号。',
    ],
    relatedFiles: ['app/services/rag_types.py', 'app/services/hybrid_retriever.py', 'app/api/v1/chat.py'],
  },
  {
    id: 'query-001',
    stage: '多轮与 Query 改写',
    difficulty: '中等',
    tags: ['多轮对话', 'Query改写', 'HyDE', 'Multi-Query'],
    question: '你怎么解决多轮对话中的指代丢失问题？',
    shortAnswer:
      '读取最近历史，把上下文依赖问题改写成独立 query，再用于 NER、意图识别和检索。',
    answer: `多轮 RAG 不能直接拿用户原问题去检索。比如第一轮问“阿司匹林是什么”，第二轮问“它有什么副作用”，如果直接检索“它有什么副作用”，召回会很差。

我的做法是先读取当前 conversation_id 下最近若干轮历史，然后判断当前问题是否包含“它、这个、该药、副作用、禁忌”等上下文依赖信号。如果需要，就调用小模型或当前 LLM，把问题改写成可以独立检索的问题，比如“阿司匹林有什么副作用”。

后续 NER、意图识别、Multi-Query、HyDE 和 RAG 都基于这个 standalone_query 或 rewritten_query 进行。前端 SSE meta 里会返回 original_query、standalone_query、rewritten_query，方便调试改写是否正确。

需要注意的是，改写不能补造信息。如果历史里列了多个药，用户问“它的副作用”，系统无法确定指代哪个药，就应该触发 ask_clarification，而不是强行猜一个。`,
    followUps: [
      '如果改写错了怎么办？',
      '为什么不直接把完整历史塞进 RAG？',
      'Multi-Query 和 HyDE 分别解决什么问题？',
    ],
    pitfalls: [
      '不要让改写器新增病情、剂量或治疗建议。',
      '不要把历史全文都无脑塞给检索器，容易污染检索。',
    ],
    relatedFiles: ['app/services/chat_service.py', 'docs/CONTEXT_MECHANISM.md'],
  },
  {
    id: 'agent-001',
    stage: 'Agent',
    difficulty: '深挖',
    tags: ['Agent', 'Function Calling', '工具调用'],
    question: '你的 Agent 工具集是怎么设计的？',
    shortAnswer:
      '把医疗能力拆成职责互斥的白名单工具，模型只负责选择工具，应用层负责参数校验和执行。',
    answer: `我没有直接让模型自由调用任意代码，而是把医疗问答能力拆成一组白名单工具。核心工具包括 search_knowledge_graph、search_vector_db、lookup_drug、assess_triage、web_search、ask_clarification、escalate_to_human，另外支持 MCP 动态工具。

设计原则有三个。

第一，工具职责要互斥。比如药品副作用优先 lookup_drug，开放医学资料查 search_vector_db，明确疾病结构化信息查 search_knowledge_graph，症状分诊查 assess_triage。

第二，schema 和 description 要清楚。Function Calling 的效果很依赖工具描述，如果 description 写得含糊，模型容易乱选工具。

第三，应用层必须做白名单和参数校验。模型只提出 tool_call，真正执行的是后端注册表里的 handler。即使模型生成了不存在的工具名，应用也不会执行。

此外我做了降级：优先用 OpenAI-compatible 原生 tool_calls；如果模型或接口不支持，就降级到 JSON planner；再失败还有规则路由兜底。`,
    followUps: [
      '工具粒度怎么确定？',
      '如果模型选错工具怎么办？',
      'Function Calling 和普通 Prompt JSON 有什么区别？',
    ],
    pitfalls: [
      '不要让 LLM 决定执行任意代码。',
      '不要把所有能力塞进一个大工具，工具粒度太粗会降低可控性。',
    ],
    relatedFiles: ['app/services/agent_tools.py', 'docs/phase3_tool_use.md'],
  },
  {
    id: 'agent-002',
    stage: 'Agent',
    difficulty: '深挖',
    tags: ['LangGraph', '状态机', 'ReAct'],
    question: '为什么用 LangGraph？你的 Agent 状态机怎么设计？',
    shortAnswer:
      'LangGraph 把 Agent 流程显式建成 plan -> execute_tools -> review -> finish，方便控制、观测和安全兜底。',
    answer: `我选择 LangGraph 的原因是它比传统黑盒 Agent 循环更可控。医疗场景不能让 Agent 无限思考、无限调用工具，所以我把流程显式拆成状态机。

状态图主要是四个节点：

plan：根据当前 query、实体、历史和用户画像规划要调用哪些工具。

execute_tools：只执行注册表里的白名单工具，记录参数、耗时、状态和 observation。

review：检查工具结果。如果触发 escalate_to_human，就直接转人工或急诊兜底；如果触发 ask_clarification，就反问用户；如果工具失败但未超限，可以回到 plan 重规划；如果 observation 足够，就进入后续 RAG 回答。

finish：写入 final_action、stop_reason 和 graph_events。

前端管理员面板可以看到完整的 Agent 状态机时间线，这对调试非常关键。面试时我会强调：Agent 不只是“会调用工具”，还必须可观测、可终止、可解释。`,
    followUps: [
      'LangGraph 和 LangChain Agent 有什么区别？',
      'ReAct 在你的项目里体现在哪里？',
      '什么时候会触发 clarify 或 escalate？',
    ],
    pitfalls: [
      '不要把 Agent 说成完全自主，医疗场景必须有边界。',
      '不要忽略状态机 trace 的工程价值。',
    ],
    relatedFiles: ['app/services/agent_graph.py', 'app/services/agent_state.py', 'docs/phase3_langgraph_agent.md'],
  },
  {
    id: 'agent-003',
    stage: 'Agent',
    difficulty: '深挖',
    tags: ['AgentState', '状态设计', '可观测'],
    question: 'AgentState 里你保存了哪些状态？为什么要显式建模？',
    shortAnswer:
      '拆成会话、用户画像、任务、推理、行动、记忆、控制流七类，让多轮、工具和终止原因可追踪。',
    answer: `我把 AgentState 拆成七类。

会话状态保存 messages、history_turns、original_query、standalone_query、rewritten_query，用来追踪多轮上下文和 query 改写。

用户画像保存当前登录用户和本轮提及的健康事实。这里我特别注意，只把用户说过的内容标成 mentioned，不把它当成确诊病史，避免过度推断。

任务状态保存 intent、slots、missing_slots、risk_flags，让 Agent 知道当前要做什么、缺什么信息、有没有高危信号。

推理状态保存 plan 和 scratchpad，行动状态保存 tool_calls 和 observations。

记忆状态区分短期、长期和工作记忆。

控制流状态保存 iterations、max_iterations、timeout、token_budget、final_action、stop_reason。

显式建模的价值是可观测和可控。面试官问“为什么这轮转人工”“为什么这轮没检索”“为什么停止了”，我可以从 state 和 trace 里解释，而不是说模型自己决定的。`,
    followUps: [
      '用户画像和长期记忆有什么区别？',
      '为什么医疗画像不能当成确诊事实？',
      'AgentState 是否会持久化？',
    ],
    pitfalls: [
      '不要把 scratchpad 暴露给普通用户。',
      '不要把用户临时提到的症状永久当成病史。',
    ],
    relatedFiles: ['app/services/agent_state.py', 'docs/phase3_agent_state.md'],
  },
  {
    id: 'agent-004',
    stage: 'Agent',
    difficulty: '中等',
    tags: ['防死循环', '成本控制', 'Agent安全'],
    question: '你怎么防止 Agent 无限循环或成本失控？',
    shortAnswer:
      '最大迭代、整体超时、单工具超时、重复失败熔断、token 预算，触发后 control_stop 兜底。',
    answer: `Agent 最大的工程风险之一是无限循环和成本失控，所以我做了显式控制流保护。

具体包括：最大迭代次数，比如 10 次；整体超时，比如 15 秒；单个工具超时，比如 8 秒；同一工具连续失败超过 2 次就熔断；总 token 预算，比如 20000。

这些保护不是只在最后检查，而是在 LangGraph 的 plan 前、plan 后和 review 阶段都会检查。一旦超限，就把 final_action 设置成 control_stop，不继续调用工具，也不继续 RAG 链路，而是给用户一个兜底说明。

所有停止原因都会写入 AgentState.control.stop_reason，比如 token_budget_exceeded、timeout_exceeded、repeated_tool_failure。这样既能保护成本，也方便后续排查是哪类问题导致 Agent 失败。`,
    followUps: [
      '为什么不是让模型自己判断停止？',
      '超时后用户体验怎么处理？',
      'token 预算怎么估算？',
    ],
    pitfalls: [
      '不要只设置 max_iterations，工具超时和重复失败也很重要。',
      '不要让超限后继续生成看似确定的医疗建议。',
    ],
    relatedFiles: ['app/services/agent_graph.py', 'docs/phase3_control_flow_guardrails.md'],
  },
  {
    id: 'memory-001',
    stage: '记忆系统',
    difficulty: '深挖',
    tags: ['Memory', '长期记忆', '向量检索'],
    question: '你的长期记忆怎么防止污染新对话？',
    shortAnswer:
      '向量相似度阈值过滤，普通记忆 0.7，高优先级记忆 0.5，实体关键词命中加 0.15。',
    answer: `长期记忆的核心风险是污染。比如用户之前提到糖尿病，后面问普通感冒用药，如果系统无脑注入糖尿病记忆，回答就会跑偏。

我的做法是每条长期记忆保存内容、category、importance、relevance_keywords 和 embedding。新问题进来后，先对 query 做 embedding，然后和该用户的记忆做相似度计算。

普通记忆只有相似度 >= 0.7 才召回。过敏史、慢性病这种高优先级记忆 importance>=8，会把阈值降到 0.5，因为医疗安全上宁可多召回一点，也不能漏掉关键禁忌。除此之外，如果 NER 实体命中了记忆里的 relevance_keywords，会给相似度加 0.15。

这样可以达到两个目标：问“糖尿病人能吃阿司匹林吗”时召回糖尿病史；问“感冒吃什么药”时，如果和糖尿病记忆相似度很低，就不召回，避免污染。`,
    followUps: [
      '为什么阈值选 0.7？',
      '记忆越来越多怎么办？',
      '为什么不用 pgvector 存 user_memories？',
    ],
    pitfalls: [
      '不要把所有历史都注入 Prompt。',
      '不要忽略过敏史、慢性病这类高优先级记忆的特殊处理。',
    ],
    relatedFiles: ['app/services/memory_service.py', 'migrations/003_memory_hitl.sql'],
  },
  {
    id: 'memory-002',
    stage: '记忆系统',
    difficulty: '中等',
    tags: ['Memory', '短期记忆', '工作记忆'],
    question: '短期记忆、长期记忆、工作记忆分别是什么？',
    shortAnswer:
      '短期是最近对话窗口，长期是用户健康档案，工作记忆是当前 Agent 任务状态。',
    answer: `短期记忆就是最近几轮消息窗口，用于多轮上下文补全，比如把“它有什么副作用”改写成“阿司匹林有什么副作用”。它来自 messages 表，按 conversation_id 读取。

长期记忆是用户健康档案，比如过敏史、慢性病、长期用药史，存在 user_memories 表，按用户维度保存，跨会话可用，但必须通过相似度阈值过滤后才注入。

工作记忆是当前任务的运行状态，比如本轮识别到的实体、意图、槽位、工具调用、风险标记和控制流预算。它在 AgentState 里，主要用于本轮决策和前端调试，通常不作为长期病史持久化。

这三类记忆的生命周期不同：短期面向当前会话，长期面向用户画像，工作记忆面向当前任务。`,
    followUps: [
      '为什么不把所有对话都作为长期记忆？',
      '长期记忆是自动提取还是手动保存？',
      '历史窗口太小怎么办？',
    ],
    pitfalls: [
      '不要混淆 messages 历史和 user_memories 健康档案。',
      '不要把当前推理 scratchpad 当成长期记忆。',
    ],
    relatedFiles: ['app/services/chat_service.py', 'app/services/memory_service.py', 'app/services/agent_state.py'],
  },
  {
    id: 'hitl-001',
    stage: '医疗安全',
    difficulty: '深挖',
    tags: ['HITL', '医疗安全', '转人工'],
    question: '医疗场景的 HITL 人在回路你是怎么设计的？',
    shortAnswer:
      '前置规则拦截 + Agent escalate_to_human 工具 + hitl_events 审计，emergency 直接跳过 RAG/Agent。',
    answer: `医疗场景不能让 Agent 对高危症状自由发挥，所以我设计了两层 HITL。

第一层是前置拦截。在 chat_service.prepare() 入口先做 HITL 评估，匹配胸痛、呼吸困难、大出血、意识障碍、疑似中风等高危关键词。如果 urgency 是 emergency，就直接跳过 RAG 和 Agent，返回“立即拨打 120 或前往急诊”的建议，不做诊断、不推荐药物。

第二层是 Agent 层。Agent 工具集中有 escalate_to_human(reason)，如果工具规划或 review 阶段判断当前问题高危、复杂用药、工具连续失败，就触发转人工兜底。

所有 HITL 事件会写入 hitl_events 表，包括 user_id、conversation_id、query、urgency、reason、matched_keywords、agent_trace、human_reviewed 等字段。这样后续可以做质控：哪些场景自动处理不了，哪些规则误拦，哪些需要优化。

分级上我分 emergency、urgent、routine。emergency 是立即急诊；urgent 是尽快人工或线下医生；routine 是工具失败或用户主动要求转人工。`,
    followUps: [
      'HITL 会不会误拦？',
      '为什么高危问题不先 RAG 检索？',
      'LangGraph 的 interrupt/resume 你用了吗？',
    ],
    pitfalls: [
      '不要让高危急症继续复杂 Agent 循环。',
      '不要给出确定诊断或具体急救用药剂量。',
    ],
    relatedFiles: ['app/services/hitl_service.py', 'app/services/chat_service.py', 'migrations/003_memory_hitl.sql'],
  },
  {
    id: 'security-001',
    stage: '安全与合规',
    difficulty: '中等',
    tags: ['医疗安全', 'Prompt注入', 'PII'],
    question: '医疗 AI 应用里你考虑了哪些安全问题？',
    shortAnswer:
      '幻觉、错误诊断、高危症状、Prompt 注入、PII 隐私、工具越权和多源冲突。',
    answer: `我主要从几个方面考虑安全。

第一是幻觉控制。用 RAG 引用溯源约束回答，没有可靠来源就说明不确定，不编造引用。

第二是医疗边界。System Prompt 明确系统是导诊和健康科普助手，不做确定诊断、不开处方、不替代医生。

第三是高危症状。HITL 前置拦截 emergency 场景，直接建议急诊或人工确认。

第四是 Prompt 注入。用户输入和检索文档都当成数据，不允许覆盖系统指令；工具调用也只走白名单和参数校验，不能由模型执行任意代码。

第五是 PII 和审计。医疗数据里可能有手机号、身份证、病史等敏感信息，进入外部模型前应脱敏，日志里也要避免明文记录。转人工事件需要审计，但也要控制访问权限。

第六是多源冲突。KG、向量语料、外部网页如果结论不一致，需要按权威性和时效性消解，分歧大时呈现不确定或转人工。`,
    followUps: [
      '间接 Prompt 注入是什么？',
      '商用 API 会不会拿用户数据训练？',
      '医疗免责声明是否足够？',
    ],
    pitfalls: [
      '不要只说“加免责声明”就算安全。',
      '不要忽视 RAG 文档中的间接注入风险。',
    ],
    relatedFiles: ['app/services/conflict_resolver.py', 'app/services/hitl_service.py', 'app/services/agent_tools.py'],
  },
  {
    id: 'streaming-001',
    stage: '工程化',
    difficulty: '中等',
    tags: ['SSE', '流式输出', 'FastAPI'],
    question: '你的流式输出是怎么实现的？为什么用 SSE？',
    shortAnswer:
      'FastAPI StreamingResponse 返回 text/event-stream，前端读取 meta/token/done/error 事件；问答场景单向推送用 SSE 更简单。',
    answer: `后端 /chat 接口返回 StreamingResponse，media_type 是 text/event-stream。生成器会依次发几个事件：meta、token、done、error。

meta 事件只发一次，包含 conversation_id、实体、意图、来源、Agent 工具 trace、query 改写信息等。token 事件会多次发送，每次包含增量文本。done 事件包含 message_id 和 token_usage。中间异常会转成 error 事件，让前端能识别流式中途失败。

前端收到响应后用 ReadableStream 读取 SSE 文本，解析 event 和 data，再把 token 追加到当前 assistant message 上。

我选择 SSE 而不是 WebSocket，是因为这个场景主要是服务端向客户端单向推送 token，不需要双向实时通信。SSE 基于 HTTP，部署和调试更简单，也天然支持文本事件。WebSocket 更适合多人协作、实时控制、客户端频繁双向发消息的场景。`,
    followUps: [
      '流式中途报错怎么办？',
      'Nginx 部署 SSE 要注意什么？',
      'StreamingResponse 的 DB session 生命周期有什么坑？',
    ],
    pitfalls: [
      '不要说 SSE 永远比 WebSocket 好，要结合场景。',
      '不要忘记中间错误事件和反向代理 buffering。',
    ],
    relatedFiles: ['app/api/v1/chat.py', 'frontend/src/App.tsx'],
  },
  {
    id: 'cost-001',
    stage: '工程化',
    difficulty: '中等',
    tags: ['Token', '成本优化', '缓存'],
    question: '你在项目里怎么做 Token 统计和成本优化？',
    shortAnswer:
      '每次 LLM 调用累计 token_usage 入库；成本优化靠模型分层、Prompt Caching、语义缓存和 RAG 前置过滤。',
    answer: `Token 统计方面，我在 LLM service 里做 usage accumulator，一次 /chat 请求中多个 LLM 调用会累计 input_tokens、output_tokens、total_tokens、cost。流式结束或出错时，把 usage 写入 token_usage 表，关联 user_id 和 conversation_id。

成本优化可以分几层。

第一是模型分层。query 改写、意图识别、工具规划这类简单任务用便宜小模型；最终医疗回答用能力更强的模型。

第二是缓存。长 System Prompt 或固定上下文可以用 Prompt Caching；相似问题可以做语义缓存，但医疗场景要谨慎，个性化、时效性或高风险问题不能直接缓存命中返回。

第三是检索前置过滤。不是所有寒暄都走 RAG，不是所有问题都需要 Agent；普通聊天直接回答，医疗问题才进入 RAG/Agent 链路。

第四是控制流预算。Agent 有 token_budget 和 max_iterations，避免循环调用造成成本爆炸。

简历里如果写成本降低 60%，面试时要能说明实验口径：同一批 query，优化前后总成本对比，包括模型路由、缓存命中率和平均 token 数。`,
    followUps: [
      '语义缓存为什么在医疗场景有风险？',
      '流式输出如何拿到准确 token？',
      '成本统计粒度按用户还是按接口？',
    ],
    pitfalls: [
      '不要无脑缓存医疗建议。',
      '不要写无法解释实验口径的降本数字。',
    ],
    relatedFiles: ['app/api/v1/chat.py', 'app/services/llm_service.py', 'app/db/models.py'],
  },
  {
    id: 'eval-001',
    stage: '评测',
    difficulty: '深挖',
    tags: ['RAGAS', 'Recall@K', '评测集'],
    question: '你的 RAG 效果是怎么评测的？',
    shortAnswer:
      '构建 query、expected_sources、reference_answer 的评测集，分别测检索层 Recall/MRR 和生成层相关性/忠实性/引用准确率。',
    answer: `我把 RAG 评测分成检索层和生成层。

检索层不调用 LLM，成本低，适合频繁回归。评测集每条包含 query、expected_sources 或 expected_keywords。系统跑检索后看 TopK 是否命中，计算 Recall@5；第一个命中的排名可以算 MRR；还可以算 context_precision 和 context_recall。

生成层会调用当前模型生成答案，再用 reference_answer 和 answer_keywords 做近似评估，或用 RAGAS / LLM-as-Judge 评估 answer_relevancy、faithfulness、citation_precision、hallucination_rate_proxy。

我还会看分阶段指标：vector、bm25、hybrid_fused、hybrid_rerank、candidate_pool。如果 candidate_pool 高但 hybrid_rerank 低，说明精排有问题；如果 candidate_pool 本身低，说明粗召回、切片、Embedding 或 query 改写有问题。

正式面试时，我不会只说“我感觉更准了”，而是说用固定评测集做回归，记录 Recall@5、MRR、答案覆盖率、引用准确率和平均延迟。`,
    followUps: [
      'Recall@5 的计算口径是什么？',
      'RAGAS 的 faithfulness 是什么？',
      'LLM-as-Judge 有什么风险？',
    ],
    pitfalls: [
      '不要用 12 条小样例当最终效果指标。',
      '不要只评测生成答案，不看检索候选池。',
    ],
    relatedFiles: ['scripts/evaluate_rag_p0.py', 'scripts/evaluate_rag_ragas.py', 'data/rag_eval/rag_eval_cases.jsonl'],
  },
  {
    id: 'eval-002',
    stage: '评测',
    difficulty: '中等',
    tags: ['Agent评测', '工具调用', '指标'],
    question: 'Agent 工具调用怎么评测？',
    shortAnswer:
      '用工具选择评测集统计 exact_match、first_tool_accuracy、tool_precision、tool_recall。',
    answer: `Agent 评测不能只看最终回答，因为工具选错但模型碰巧答对也会掩盖问题。我单独建了工具选择评测集，每条 case 包含用户问题和期望工具。

指标上有几个层次：

exact_match 看预测工具集合和期望工具集合是否完全一致。

first_tool_accuracy 看首个工具是否命中，尤其适合高危转人工、澄清提问这类有优先级的场景。

tool_precision 看模型选出来的工具有多少是应该选的，防止乱调工具。

tool_recall 看期望工具中有多少被选中，防止漏调关键工具。

我支持两种评测方式：不调用外部 LLM 的规则路由评测，用于可复现回归；以及真实 LLM planner 评测，用于观察线上模型的工具选择效果。`,
    followUps: [
      '工具调用准确率提高靠什么？',
      '高危问题 first tool 为什么重要？',
      '端到端任务成功率怎么定义？',
    ],
    pitfalls: [
      '不要只用最终答案评估 Agent。',
      '不要忽略首工具优先级。',
    ],
    relatedFiles: ['scripts/evaluate_agent_tools.py', 'data/agent_eval/tool_use_eval_cases.jsonl'],
  },
  {
    id: 'mcp-001',
    stage: 'MCP',
    difficulty: '中等',
    tags: ['MCP', 'Function Calling', '工具协议'],
    question: 'MCP 和 Function Calling 有什么区别？你项目里怎么接的？',
    shortAnswer:
      'Function Calling 是模型到应用的调用格式；MCP 是应用到外部工具/数据源的协议。项目把 MCP tool 转成 function schema 给模型选择。',
    answer: `Function Calling 解决的是模型如何表达“我要调用哪个工具、参数是什么”。它是模型和应用之间的接口格式。

MCP 解决的是应用如何发现、描述和调用外部工具或数据源。它是应用和工具 Server 之间的协议。

我项目里二者是组合关系。mcp_client 读取配置后启动 MCP stdio server，调用 list_tools() 发现工具，然后把每个 MCP tool 转成 OpenAI-compatible function schema，工具名统一加 mcp__server__tool 前缀。

LLM planner 看到的是普通 function calling 工具；真正执行时，应用发现工具名以 mcp__ 开头，就委托给 MCP Server 调用。这样 filesystem、PostgreSQL 等外部能力可以配置化接入，不需要每个工具都在应用里硬编码。

安全上，Agent 仍然只执行白名单里的工具名，MCP Server 也只暴露配置允许的目录或数据源。MCP 失败或超时会变成 observation，不影响医疗问答主链路。`,
    followUps: [
      '为什么 MCP 默认关闭？',
      'MCP 工具怎么做权限控制？',
      'MCP 失败会不会影响主链路？',
    ],
    pitfalls: [
      '不要把 MCP 说成模型能力，它是工具协议。',
      '不要让 MCP filesystem 暴露整个系统目录。',
    ],
    relatedFiles: ['app/services/mcp_client.py', 'docs/phase3_mcp.md', 'config/mcp_servers.example.json'],
  },
  {
    id: 'conflict-001',
    stage: '多源冲突',
    difficulty: '深挖',
    tags: ['冲突消解', '医疗安全', 'RAG'],
    question: '当知识图谱、向量库和模型结果冲突时，你怎么处理？',
    shortAnswer:
      '按安全性、权威性、时效性和来源类型排序；分歧大时呈现不确定或转人工。',
    answer: `医疗多源知识冲突很常见，比如旧科普和新指南不一致、KG 结构化数据缺失、模型常识和本地资料冲突。

我的处理优先级是：安全和合规优先，其次是权威性和时效性，再考虑用户最新意图。具体来源上，指南、官方资料和结构化 KG 通常优先于普通科普网页；更新日期更近的指南优先于旧内容；如果是高危或用药问题，宁可保守，不给确定建议。

系统里会对召回来源做冲突识别和 conflict_resolution meta。如果冲突可消解，就在 prompt 里告诉模型采用哪个来源；如果分歧大或涉及风险，就提示存在信息差异，建议咨询医生或触发人工。

关键点是不要把冲突隐藏起来，让模型自己“编一个看起来合理的答案”。医疗场景里，不确定性本身也应该被明确表达。`,
    followUps: [
      '冲突怎么自动检测？',
      '权威等级怎么定义？',
      '如果用户坚持要明确答案怎么办？',
    ],
    pitfalls: [
      '不要用模型投票替代权威医学来源。',
      '不要在冲突场景下输出绝对化结论。',
    ],
    relatedFiles: ['app/services/conflict_resolver.py', 'app/services/chat_service.py'],
  },
  {
    id: 'backend-001',
    stage: '后端工程',
    difficulty: '中等',
    tags: ['SQLAlchemy', 'PostgreSQL', 'JWT'],
    question: '后端数据层你是怎么设计的？',
    shortAnswer:
      'PostgreSQL 存业务数据和向量数据，SQLAlchemy async 管理会话；JWT 做前后端分离鉴权。',
    answer: `后端数据层主要用 PostgreSQL，原因是它既能存用户、会话、消息、反馈、Token 用量这些业务数据，也能通过 pgvector 存 document_chunks 的向量，减少额外向量库组件。

SQLAlchemy 2.0 async 用 async_sessionmaker 管理会话，FastAPI Depends 注入 session。聊天流式接口有一个细节：StreamingResponse 的生成器执行时，路由依赖注入的 DB session 可能已经释放，所以在 generator 内部持久化 assistant message 和 token_usage 时，会重新打开独立 session。

鉴权方面，原项目是 JSON 文件保存账号，只适合本地 demo。我改成 JWT：登录后签发 access_token，前端存在 localStorage，请求时带 Authorization: Bearer token，后端 Depends(get_current_user) 验签并注入当前用户。

业务表包括 users、conversations、messages、feedback、token_usage，记忆和 HITL 扩展表包括 user_memories、hitl_events。`,
    followUps: [
      '为什么 pgvector 而不是 Milvus？',
      'JWT 和 Session 的区别？',
      '流式接口里 DB session 有什么坑？',
    ],
    pitfalls: [
      '不要说 pgvector 永远优于专业向量库；它适合当前规模和部署复杂度。',
      '不要在流式生成器里复用已关闭的依赖 session。',
    ],
    relatedFiles: ['app/db/models.py', 'app/db/session.py', 'app/core/security.py', 'app/api/v1/chat.py'],
  },
  {
    id: 'frontend-001',
    stage: '前端展示',
    difficulty: '基础',
    tags: ['React', '可视化', 'SSE'],
    question: '前端在这个 AI 项目里做了哪些不只是“页面”的事情？',
    shortAnswer:
      '前端承担了流式渲染、引用跳转、Agent trace 可视化、文档入库和反馈采集。',
    answer: `这个项目的前端不只是一个聊天框。

第一，它处理 SSE 流式响应，把 meta、token、done、error 事件拆开。meta 先更新来源、Agent trace 和 query 改写信息，token 持续追加到回答气泡里。

第二，它做引用来源交互。模型回答里的 [1] [2] 会被渲染成可点击链接，点击后右侧打开来源面板并高亮对应来源，GraphRAG 路径也会用节点和关系展示。

第三，它做管理员调试面板。可以查看实体、意图、回答模式、Multi-Query、HyDE、冲突消解、Agent 状态机时间线、工具调用 observation 和 AgentState。

第四，它支持文档入库和反馈。管理员可以上传 txt/md/jsonl 或索引内置 medical_new_2.json；用户可以对回答点赞/点踩，为后续评测和优化提供数据。

这些交互对 AI 应用很重要，因为 AI 系统需要可解释和可调试，不只是把模型答案显示出来。`,
    followUps: [
      '引用点击是怎么实现的？',
      '为什么要给管理员看 Agent trace？',
      '前端怎么处理流式 error？',
    ],
    pitfalls: [
      '不要低估前端在 AI 应用可观测性中的作用。',
      '不要把调试信息暴露给普通用户。',
    ],
    relatedFiles: ['frontend/src/App.tsx', 'frontend/src/App.css'],
  },
  {
    id: 'tradeoff-001',
    stage: '选型取舍',
    difficulty: '深挖',
    tags: ['技术选型', '项目取舍'],
    question: '你这个项目最大的技术难点是什么？你怎么解决的？',
    shortAnswer:
      '最大难点是医疗 RAG 的召回质量和安全边界，解决方式是 bad case 分析、混合检索、Rerank、引用溯源和 HITL。',
    answer: `我会讲召回质量优化这个难点。

初版只有知识图谱检索，对结构化问题效果还可以，但对用户口语化症状问题，比如“老是头晕想吐挂什么科”，或者药名、禁忌、副作用这类细粒度问题，覆盖不足。

我先做 bad case 分析，把失败分成几类：口语和医学术语差距大；专有名词纯向量召回不稳定；KG 有结构化关系但开放文本不足；召回到相关内容但不能直接回答。

针对这些问题，我加了 pgvector 向量检索补语义，BM25 补药名/疾病名精确匹配，Neo4j GraphRAG 补结构化路径，再用 Reranker 精排 top5。同时做 query 改写、Multi-Query、HyDE 和引用溯源。

所以解决思路不是堆技术，而是先定位失败类型，再用对应技术补短板。`,
    followUps: [
      '有没有一个具体 bad case？',
      '优化前后指标怎么测？',
      '如果只能保留一个优化，你保留哪个？',
    ],
    pitfalls: [
      '不要说“最大难点是技术很多”，要讲具体问题。',
      '不要没有指标或 bad case。',
    ],
    relatedFiles: ['docs/phase2_p0_test_guide.md', 'scripts/evaluate_rag_ragas.py'],
  },
  {
    id: 'future-001',
    stage: '优化方向',
    difficulty: '中等',
    tags: ['优化', '生产化', '规划'],
    question: '如果继续优化这个项目，你会做什么？',
    shortAnswer:
      '优先做自动化评测闭环、自动记忆提取、Prompt 版本管理、Redis 限流缓存、多模态化验单 OCR。',
    answer: `我会按投入产出比排序。

第一是评测闭环自动化。现在有 RAG 和工具调用评测脚本，下一步要接到 CI 或发布流程里，每次改 Prompt、检索权重、Reranker 都跑回归，防止指标劣化。

第二是长期记忆自动提取。现在 user_memories 可以召回，但记忆提取可以更自动化：每轮对话结束后由 LLM 总结过敏史、慢性病、长期用药等候选事实，再通过规则或人工确认后入库。

第三是 Prompt 版本管理和灰度发布。把 System Prompt、RAG Prompt、工具 planner Prompt 抽成版本化 yaml，关键改动走小流量灰度。

第四是工程化增强。Redis 做限流、语义缓存和热会话缓存；LangSmith 或自建 trace 做链路监控；PII 脱敏和权限审计做完整。

第五是多模态扩展。医疗问答常见输入是化验单、处方单、检查报告，可以做 OCR + 结构化抽取 + RAG 解释。`,
    followUps: [
      '哪个优化最有业务价值？',
      '多模态会带来哪些新风险？',
      '自动记忆提取怎么防止写入错误病史？',
    ],
    pitfalls: [
      '不要说“继续换更大模型”作为唯一优化。',
      '不要忽略评测和安全闭环。',
    ],
    relatedFiles: ['docs/phase3_memory_hitl.md', 'docs/phase2_p0_test_guide.md'],
  },
]

window.interviewStages = Array.from(new Set(window.interviewQuestions.map((item) => item.stage)));
window.interviewTags = Array.from(new Set(window.interviewQuestions.flatMap((item) => item.tags))).sort((a, b) => a.localeCompare(b, 'zh-Hans-CN'));
