"""List tools discovered from configured MCP servers."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services import mcp_client


async def main() -> None:
    tools = await mcp_client.list_mcp_tools(force_refresh=True)
    print(
        json.dumps(
            [tool.to_meta() for tool in tools],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
