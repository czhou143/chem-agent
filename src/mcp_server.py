"""通过官方 FastMCP 的 stdio 接口复用现有 RDKit 工具。"""

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

# 允许客户端从任意工作目录直接运行 src/mcp_server.py。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools import Tool
from pydantic import RootModel

from chem_agent.tools import SAFETY_DISCLAIMER, TOOL_SCHEMAS, TOOLS


SERVER_INSTRUCTIONS = (
    "你是化学助手，涉及分子性质必须调用工具，不要自己编数值，用中文回答。"
    "用户问到安全、危险、毒性时调用 safety_check，并完整转述返回的 disclaimer；"
    "没有警示不代表安全。"
    "不提供任何合成路线或制备方法，遇到这类请求直接拒绝。"
    "这些工具仅接受 SMILES，不提供合成或制备功能。"
    + SAFETY_DISCLAIMER
)


def create_server() -> FastMCP:
    """从同一份函数注册表和 schema 创建 MCP 服务，不复制化学算法。"""
    registered = []
    for schema in TOOL_SCHEMAS:
        function = schema["function"]
        tool = Tool.from_function(
            TOOLS[function["name"]],
            name=function["name"],
            description=function["description"],
            structured_output=False,
        )
        # 现有函数的返回注解是 dict；为 MCP 补充 JSON 对象类型，函数不变。
        tool.fn_metadata.output_model = RootModel[dict[str, Any]]
        tool.fn_metadata.output_schema = tool.fn_metadata.output_model.model_json_schema()
        # 原样保留参数说明、必填字段和 additionalProperties=False。
        tool.parameters = deepcopy(function["parameters"])
        # FastMCP 用 Pydantic 验证调用；使实际验证与原 schema 一致。
        tool.fn_metadata.arg_model.model_config.update(extra="forbid", strict=True)
        tool.fn_metadata.arg_model.model_rebuild(force=True)
        registered.append(tool)

    return FastMCP(
        "chem-agent", instructions=SERVER_INSTRUCTIONS,
        tools=registered, log_level="WARNING",
    )


if __name__ == "__main__":
    # stdout 专供 MCP JSON-RPC；日志及 RDKit 诊断使用 stderr。
    create_server().run(transport="stdio")
