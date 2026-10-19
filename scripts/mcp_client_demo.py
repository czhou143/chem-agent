"""启动本地 MCP 子进程，列出工具，再计算阿司匹林分子量；无需 API key。"""

import asyncio
from datetime import timedelta
import json
from pathlib import Path
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    server_path = Path(__file__).resolve().parents[1] / "src" / "mcp_server.py"
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
        env={"PYTHONUTF8": "1", "OPENBLAS_NUM_THREADS": "1"},
    )
    # stdio_client 自动启动、连接并清理服务进程，不需要先手动启动服务。
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30)) as session:
            await session.initialize()
            available = await session.list_tools()
            print("tools/list:")
            for tool in available.tools:
                print(f"  - {tool.name}")

            result = await session.call_tool(
                "mol_weight", {"smiles": "CC(=O)Oc1ccccc1C(=O)O"}
            )
            if result.isError:
                raise RuntimeError(f"MCP tool call failed: {result.content}")
            print("tools/call mol_weight:")
            print(json.dumps(result.structuredContent, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
