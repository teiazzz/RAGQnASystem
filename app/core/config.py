"""后端集中配置（pydantic-settings）。

从项目根 ``.env`` 读取所有外部依赖参数。与基座旧的 ``config.py``（frozen dataclass，
供 Streamlit/脚本用）互不干扰：两者都从同一份 ``.env`` 读取，``.env`` 是单一真相源。

用法::

    from app.core.config import settings
    engine = create_async_engine(settings.DATABASE_URL)
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# app/core/config.py -> parents[2] = 项目根，定位 .env，避免依赖运行时 cwd
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",  # .env 含本类未声明的额外键时忽略，不报错
    )

    # --- PostgreSQL（异步驱动 asyncpg）---
    DATABASE_URL: str

    # --- Redis（阶段五限流/缓存用）---
    REDIS_URL: str = "redis://localhost:6379/0"

    # --- JWT 鉴权 ---
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # --- DeepSeek LLM（OpenAI 兼容接口）---
    DEEPSEEK_API_KEY: str
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    LLM_INPUT_COST_PER_1K_TOKENS: float = 0.0
    LLM_OUTPUT_COST_PER_1K_TOKENS: float = 0.0

    # --- Neo4j 知识图谱 ---
    NEO4J_URL: str = "http://localhost:7474"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "neo4jpass123"
    NEO4J_DBNAME: str = "neo4j"

    # --- Phase 2 RAG 检索 ---
    MEDICAL_CORPUS_PATH: str = "data/medical_new_2.json"
    ENABLE_PGVECTOR: bool = True
    EMBEDDING_PROVIDER: str = "local_hashing"  # local_hashing / sentence_transformers
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIM: int = 1024
    RERANKER_PROVIDER: str = "local"  # local / sentence_transformers
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RAG_TOP_K: int = 5
    RAG_CANDIDATE_K: int = 50
    RAG_VECTOR_WEIGHT: float = 0.45
    RAG_BM25_WEIGHT: float = 0.35
    RAG_KG_WEIGHT: float = 0.20
    RAG_VECTOR_SCAN_LIMIT: int = 50000
    RAG_BM25_MAX_DOCS: int = 50000
    RAG_ENABLE_MULTI_QUERY: bool = True
    RAG_MULTI_QUERY_COUNT: int = 3
    RAG_ENABLE_HYDE: bool = True
    RAG_HYDE_MAX_CHARS: int = 1200

    # --- Phase 3 Agent Tool Use ---
    AGENT_TOOLS_ENABLED: bool = True
    AGENT_TOOL_NATIVE_FUNCTION_CALLING: bool = True
    AGENT_TOOL_USE_LLM: bool = True
    AGENT_TOOL_MAX_CALLS: int = 3
    AGENT_TOOL_TIMEOUT_SECONDS: float = 8.0
    AGENT_GRAPH_ENABLED: bool = True
    AGENT_GRAPH_MAX_ITERATIONS: int = 10
    AGENT_GRAPH_TIMEOUT_SECONDS: float = 15.0
    AGENT_GRAPH_MAX_REPEATED_TOOL_FAILURES: int = 2
    AGENT_GRAPH_TOKEN_BUDGET: int = 20000

    # --- Phase 3 MCP Tool Adapter ---
    MCP_ENABLED: bool = False
    MCP_SERVER_CONFIG_PATH: str = ""
    MCP_SERVER_CONFIG_JSON: str = ""
    MCP_TOOL_CACHE_TTL_SECONDS: float = 60.0
    MCP_TOOL_TIMEOUT_SECONDS: float = 8.0

    # --- Phase 3 Memory System ---
    MEMORY_SIMILARITY_THRESHOLD: float = 0.70  # 长期记忆召回相似度阈值
    MEMORY_HIGH_PRIORITY_THRESHOLD: float = 0.50  # 高优先级记忆（过敏史/慢性病）降低阈值
    MEMORY_SHORT_TERM_WINDOW: int = 5  # 短期记忆保留最近 N 轮

    # --- CORS（React 前端 dev 端口）---
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174,http://localhost:3000"

    # --- 时间显示 ---
    APP_TIMEZONE: str = "Asia/Shanghai"

    LOG_LEVEL: str = "INFO"

    @property
    def cors_origins_list(self) -> list[str]:
        """把逗号分隔的 CORS_ORIGINS 拆成列表。"""
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


# 全局单例：模块导入时读取一次 .env
settings = Settings()
