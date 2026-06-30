"""Model Context Protocol client adapter for Agent tools.

The adapter discovers tools from configured MCP stdio servers and exposes them
as OpenAI-compatible function schemas. Runtime failures are returned as tool
observations by ``agent_tools`` instead of breaking the chat turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import settings

try:  # pragma: no cover - covered only when the optional SDK is installed
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    MCP_SDK_AVAILABLE = True
except Exception:  # noqa: BLE001 - optional dependency must not break imports
    ClientSession = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]
    MCP_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MCP_AGENT_PREFIX = "mcp__"
_SAFE_TOOL_NAME = re.compile(r"[^a-zA-Z0-9_-]+")
_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


@dataclass(frozen=True)
class MCPServerConfig:
    """A stdio MCP server process configured by JSON."""

    name: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    enabled: bool = True


@dataclass(frozen=True)
class MCPToolSpec:
    """A discovered MCP tool exposed to the LLM as a function."""

    server_name: str
    tool_name: str
    agent_name: str
    description: str
    input_schema: dict[str, Any]

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.agent_name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def to_meta(self) -> dict[str, Any]:
        return {
            "name": self.agent_name,
            "description": self.description,
            "parameters": self.input_schema,
            "mcp": {
                "server": self.server_name,
                "tool": self.tool_name,
            },
        }


@dataclass(frozen=True)
class MCPToolResult:
    """Result returned by one MCP tool call."""

    status: str
    content: str
    server_name: str
    tool_name: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


_tool_cache_loaded_at: float = 0.0
_tool_cache: list[MCPToolSpec] = []


def is_mcp_tool_name(name: str) -> bool:
    return name.startswith(MCP_AGENT_PREFIX)


def cached_mcp_tools() -> list[MCPToolSpec]:
    """Return the last discovered MCP tools without touching external servers."""

    if not settings.MCP_ENABLED:
        return []
    if not _cache_is_fresh():
        return []
    return list(_tool_cache)


async def list_mcp_tools(force_refresh: bool = False) -> list[MCPToolSpec]:
    """Discover tools from enabled MCP servers.

    Returns an empty list when MCP is disabled, unconfigured, or the Python SDK
    is unavailable. Individual server failures are logged and skipped.
    """

    global _tool_cache, _tool_cache_loaded_at

    if not settings.MCP_ENABLED:
        return []
    if not MCP_SDK_AVAILABLE:
        logger.info("MCP is enabled but the mcp Python SDK is not installed")
        return []
    if not force_refresh and _cache_is_fresh():
        return list(_tool_cache)

    configs = load_server_configs()
    if not configs:
        _tool_cache = []
        _tool_cache_loaded_at = time.monotonic()
        return []

    tools: list[MCPToolSpec] = []
    for config in configs:
        if not config.enabled:
            continue
        try:
            discovered = await asyncio.wait_for(
                _list_tools_from_server(config),
                timeout=settings.MCP_TOOL_TIMEOUT_SECONDS,
            )
            tools.extend(discovered)
        except TimeoutError:
            logger.warning("MCP server %s tool discovery timed out", config.name)
        except Exception:  # noqa: BLE001 - a bad external server should not break chat
            logger.warning("MCP server %s tool discovery failed", config.name, exc_info=True)

    _tool_cache = _dedupe_agent_tools(tools)
    _tool_cache_loaded_at = time.monotonic()
    return list(_tool_cache)


async def call_mcp_tool_by_agent_name(
    agent_name: str,
    arguments: dict[str, Any],
) -> MCPToolResult:
    """Call a discovered MCP tool by its Agent-facing function name."""

    if not settings.MCP_ENABLED:
        return MCPToolResult(
            status="unavailable",
            content="MCP is disabled by configuration.",
            server_name="",
            tool_name=agent_name,
            error="mcp_disabled",
        )
    if not MCP_SDK_AVAILABLE:
        return MCPToolResult(
            status="unavailable",
            content="MCP Python SDK is not installed.",
            server_name="",
            tool_name=agent_name,
            error="mcp_sdk_missing",
        )

    spec = await find_mcp_tool(agent_name)
    if spec is None:
        return MCPToolResult(
            status="error",
            content=f"Unknown MCP tool: {agent_name}",
            server_name="",
            tool_name=agent_name,
            error="unknown_mcp_tool",
        )

    configs = {config.name: config for config in load_server_configs() if config.enabled}
    config = configs.get(spec.server_name)
    if config is None:
        return MCPToolResult(
            status="error",
            content=f"MCP server is not configured: {spec.server_name}",
            server_name=spec.server_name,
            tool_name=spec.tool_name,
            error="mcp_server_missing",
        )

    try:
        return await asyncio.wait_for(
            _call_tool_on_server(config, spec.tool_name, arguments),
            timeout=settings.MCP_TOOL_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        return MCPToolResult(
            status="timeout",
            content="MCP tool call timed out.",
            server_name=spec.server_name,
            tool_name=spec.tool_name,
            error="timeout",
        )
    except Exception as exc:  # noqa: BLE001 - report as observation
        logger.warning("MCP tool call failed: %s", agent_name, exc_info=True)
        return MCPToolResult(
            status="error",
            content=f"MCP tool call failed: {exc}",
            server_name=spec.server_name,
            tool_name=spec.tool_name,
            error=str(exc),
        )


async def find_mcp_tool(agent_name: str) -> MCPToolSpec | None:
    tools = cached_mcp_tools()
    if not tools:
        tools = await list_mcp_tools(force_refresh=True)
    for tool in tools:
        if tool.agent_name == agent_name:
            return tool
    tools = await list_mcp_tools(force_refresh=True)
    return next((tool for tool in tools if tool.agent_name == agent_name), None)


def load_server_configs(
    *,
    config_json: str | None = None,
    config_path: str | None = None,
) -> list[MCPServerConfig]:
    """Load MCP server configs from JSON text or a JSON file.

    Accepted shape:

    ``{"servers": {"filesystem": {"command": "npx", "args": [...]}}}``
    """

    raw = _load_raw_config(config_json=config_json, config_path=config_path)
    if not raw:
        return []

    servers = raw.get("servers", raw)
    items: list[tuple[str, Any]]
    if isinstance(servers, dict):
        items = list(servers.items())
    elif isinstance(servers, list):
        items = [(str(item.get("name") or ""), item) for item in servers if isinstance(item, dict)]
    else:
        return []

    configs: list[MCPServerConfig] = []
    for name, value in items:
        if not isinstance(value, dict):
            continue
        server_name = _safe_segment(str(value.get("name") or name))
        command = _expand_env_value(str(value.get("command") or "")).strip()
        if not server_name or not command:
            continue
        args = tuple(_expand_env_value(str(item)) for item in value.get("args", []) or [])
        env = {
            str(key): _expand_env_value(str(env_value))
            for key, env_value in (value.get("env") or {}).items()
        }
        cwd = value.get("cwd")
        configs.append(
            MCPServerConfig(
                name=server_name,
                command=command,
                args=args,
                env=env,
                cwd=_resolve_cwd(str(cwd)) if cwd else None,
                enabled=bool(value.get("enabled", True)),
            )
        )
    return configs


def make_agent_tool_name(server_name: str, tool_name: str) -> str:
    return f"{MCP_AGENT_PREFIX}{_safe_segment(server_name)}__{_safe_segment(tool_name)}"[:64]


async def _list_tools_from_server(config: MCPServerConfig) -> list[MCPToolSpec]:
    async with _mcp_session(config) as session:
        response = await session.list_tools()
        tools = []
        for tool in getattr(response, "tools", []) or []:
            tool_name = str(getattr(tool, "name", "") or "")
            if not tool_name:
                continue
            description = str(getattr(tool, "description", "") or "").strip()
            tools.append(
                MCPToolSpec(
                    server_name=config.name,
                    tool_name=tool_name,
                    agent_name=make_agent_tool_name(config.name, tool_name),
                    description=(
                        f"[MCP:{config.name}] {description}"
                        if description
                        else f"[MCP:{config.name}] External MCP tool {tool_name}."
                    ),
                    input_schema=_normalize_input_schema(
                        getattr(tool, "inputSchema", None)
                        or getattr(tool, "input_schema", None)
                    ),
                )
            )
        return tools


async def _call_tool_on_server(
    config: MCPServerConfig,
    tool_name: str,
    arguments: dict[str, Any],
) -> MCPToolResult:
    async with _mcp_session(config) as session:
        result = await session.call_tool(tool_name, arguments=arguments)
        is_error = bool(getattr(result, "isError", False) or getattr(result, "is_error", False))
        content = _mcp_content_to_text(getattr(result, "content", []) or [])
        return MCPToolResult(
            status="error" if is_error else "ok",
            content=content or "(MCP tool returned no text content)",
            server_name=config.name,
            tool_name=tool_name,
            data={"mcp": True, "is_error": is_error},
            error="mcp_tool_error" if is_error else None,
        )


def _mcp_session(config: MCPServerConfig):
    if StdioServerParameters is None or stdio_client is None or ClientSession is None:
        raise RuntimeError("mcp Python SDK is not installed")
    env = {**os.environ, **config.env} if config.env else None
    params = StdioServerParameters(
        command=config.command,
        args=list(config.args),
        env=env,
        cwd=config.cwd,
    )

    class _SessionContext:
        async def __aenter__(self):
            self._stdio = stdio_client(params)
            read, write = await self._stdio.__aenter__()
            self._session = ClientSession(read, write)
            session = await self._session.__aenter__()
            await session.initialize()
            return session

        async def __aexit__(self, exc_type, exc, tb):
            await self._session.__aexit__(exc_type, exc, tb)
            await self._stdio.__aexit__(exc_type, exc, tb)

    return _SessionContext()


def _load_raw_config(
    *,
    config_json: str | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    raw_json = config_json if config_json is not None else settings.MCP_SERVER_CONFIG_JSON
    if raw_json:
        try:
            loaded = json.loads(raw_json)
            return loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            logger.warning("Invalid MCP_SERVER_CONFIG_JSON")
            return {}

    raw_path = config_path if config_path is not None else settings.MCP_SERVER_CONFIG_PATH
    if not raw_path:
        return {}
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        logger.info("MCP config file does not exist: %s", path)
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        logger.warning("Invalid MCP config file: %s", path)
        return {}


def _normalize_input_schema(schema: Any) -> dict[str, Any]:
    if hasattr(schema, "model_dump"):
        schema = schema.model_dump(exclude_none=True)
    if not isinstance(schema, dict):
        schema = {}
    normalized = dict(schema)
    normalized.setdefault("type", "object")
    if not isinstance(normalized.get("properties"), dict):
        normalized["properties"] = {}
    if not isinstance(normalized.get("required"), list):
        normalized["required"] = []
    return normalized


def _mcp_content_to_text(content: list[Any]) -> str:
    parts: list[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            parts.append(text)
            continue
        if hasattr(item, "model_dump"):
            parts.append(json.dumps(item.model_dump(exclude_none=True), ensure_ascii=False))
        else:
            parts.append(str(item))
    return "\n".join(part for part in parts if part).strip()


def _expand_env_value(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return os.environ.get(key) or str(getattr(settings, key, ""))

    return _ENV_PATTERN.sub(replace, value)


def _resolve_cwd(cwd: str) -> str:
    path = Path(_expand_env_value(cwd))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def _safe_segment(value: str) -> str:
    return _SAFE_TOOL_NAME.sub("_", value.strip()).strip("_")


def _cache_is_fresh() -> bool:
    if not _tool_cache:
        return False
    ttl = float(settings.MCP_TOOL_CACHE_TTL_SECONDS or 0)
    if ttl <= 0:
        return False
    return time.monotonic() - _tool_cache_loaded_at <= ttl


def _dedupe_agent_tools(tools: list[MCPToolSpec]) -> list[MCPToolSpec]:
    result: list[MCPToolSpec] = []
    seen: set[str] = set()
    for tool in tools:
        if tool.agent_name in seen:
            continue
        result.append(tool)
        seen.add(tool.agent_name)
    return result
