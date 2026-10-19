"""真实 stdio 协议集成测试；仅本地 RDKit，不需要模型或密钥。"""

import asyncio
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from chem_agent import tools


ROOT = Path(__file__).resolve().parents[1]
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
TNT = "Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]"
NAMES = {"mol_weight", "logp", "formula", "tpsa", "hbd_hba", "lipinski", "similarity", "safety_check"}
ARGUMENTS = {
    name: ({"smiles_a": "CCO", "smiles_b": "CCCO"} if name == "similarity"
           else {"smiles": TNT if name == "safety_check" else ASPIRIN})
    for name in NAMES
}
INVALID_ARGUMENTS = [{}, {"smiles": 123}, {"smiles": None}, {"smiles": "CCO", "route": "unexpected"}]


@pytest.fixture(scope="module")
def mcp_exchange(tmp_path_factory):
    """在同一个事件循环中完成启动、握手、调用和清理，复用返回记录。"""
    async def collect():
        # 禁止子进程导入 OpenAI，并从无 .env 的临时目录启动。
        launcher = (
            "import os, runpy, sys; os.environ.pop('LLM_API_KEY', None); "
            "sys.modules['openai'] = None; runpy.run_path(sys.argv[1], run_name='__main__')"
        )
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-c", launcher, str(ROOT / "src" / "mcp_server.py")],
            cwd=tmp_path_factory.mktemp("mcp-workdir"),
            env={"PYTHONUTF8": "1", "OPENBLAS_NUM_THREADS": "1"},
        )
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30)) as session:
                initialized = await session.initialize()
                listed = await session.list_tools()
                results = {}
                for name, arguments in ARGUMENTS.items():
                    results[name] = await session.call_tool(name, arguments)
                invalid = {}
                for name in ("mol_weight", "safety_check", "similarity"):
                    arguments = ({"smiles_a": "CCO", "smiles_b": "invalid"}
                                 if name == "similarity" else {"smiles": "invalid"})
                    invalid[name] = await session.call_tool(name, arguments)
                rejected = [await session.call_tool("mol_weight", args) for args in INVALID_ARGUMENTS]
                ethanol = await session.call_tool("safety_check", {"smiles": "CCO"})
                unknown = await session.call_tool("synthesis", {"smiles": "CCO"})
                return initialized, listed, results, invalid, rejected, ethanol, unknown

    return asyncio.run(collect())


@pytest.fixture(scope="module")
def mcp_aspirin_mol_weight(mcp_exchange):
    """返回通过 MCP 调用 mol_weight(阿司匹林) 取得的实际结果。"""
    return mcp_exchange[2]["mol_weight"].structuredContent


def test_mcp_tools_list_exactly_eight(mcp_exchange):
    names = [tool.name for tool in mcp_exchange[1].tools]
    assert len(names) == 8
    assert set(names) == NAMES
    assert "safety_check" in names


@pytest.mark.parametrize("schema", tools.TOOL_SCHEMAS, ids=lambda item: item["function"]["name"])
def test_mcp_preserves_schema_descriptions_and_parameters(mcp_exchange, schema):
    function = schema["function"]
    actual = next(tool for tool in mcp_exchange[1].tools if tool.name == function["name"])
    assert actual.description == function["description"]
    assert actual.inputSchema == function["parameters"]


@pytest.mark.parametrize("name", sorted(NAMES))
def test_mcp_results_match_existing_tools(mcp_exchange, name):
    result = mcp_exchange[2][name]
    assert not result.isError
    expected = tools.TOOLS[name](**ARGUMENTS[name])
    assert result.structuredContent == expected
    assert json.loads(result.content[0].text) == expected


def test_mcp_aspirin_weight(mcp_aspirin_mol_weight):
    assert round(mcp_aspirin_mol_weight["mol_weight"], 2) == 180.16


@pytest.mark.parametrize("name", ["mol_weight", "safety_check", "similarity"])
def test_mcp_invalid_smiles_returns_error(mcp_exchange, name):
    result = mcp_exchange[3][name]
    assert "error" in result.structuredContent


@pytest.mark.parametrize("index", range(len(INVALID_ARGUMENTS)))
def test_mcp_rejects_arguments_outside_schema(mcp_exchange, index):
    assert mcp_exchange[4][index].isError


def test_mcp_safety_disclaimer_and_clean_molecule(mcp_exchange):
    assert mcp_exchange[5].structuredContent == {
        "alerts": [], "alert_count": 0, "disclaimer": tools.SAFETY_DISCLAIMER,
    }
    assert mcp_exchange[2]["safety_check"].structuredContent["disclaimer"] == tools.SAFETY_DISCLAIMER


def test_mcp_instructions_require_safety_and_refuse_synthesis(mcp_exchange):
    instructions = mcp_exchange[0].instructions
    assert tools.SAFETY_DISCLAIMER in instructions
    assert "safety_check" in instructions
    assert "不提供任何合成路线或制备方法" in instructions
    assert "直接拒绝" in instructions
    assert mcp_exchange[6].isError


def test_mcp_demo_runs_without_key_from_another_directory(tmp_path):
    environment = {key: value for key, value in os.environ.items() if key != "LLM_API_KEY"}
    environment.update(PYTHONUTF8="1", OPENBLAS_NUM_THREADS="1")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "mcp_client_demo.py")],
        cwd=tmp_path, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=45,
    )
    assert completed.returncode == 0, completed.stderr
    names = {line.removeprefix("  - ") for line in completed.stdout.splitlines() if line.startswith("  - ")}
    assert names == NAMES
    assert json.loads(completed.stdout.splitlines()[-1]) == {"mol_weight": 180.16}
