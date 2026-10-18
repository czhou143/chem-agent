"""工具层的回归测试：给定已知输入，检查数值、错误处理和工具声明。"""

import inspect
import json

import pytest

from chem_agent.tools import (
    TOOLS,
    TOOL_SCHEMAS,
    formula,
    hbd_hba,
    lipinski,
    logp,
    mol_weight,
    similarity,
    tpsa,
)


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
SINGLE_MOLECULE_TOOLS = (mol_weight, logp, formula, tpsa, hbd_hba, lipinski)


def assert_error(result):
    """非法输入应返回能解释问题的字典，而不是让程序抛异常。"""
    assert isinstance(result, dict)
    assert isinstance(result.get("error"), str)
    assert result["error"].strip()


def test_aspirin_mol_weight():
    # approx 用于比较浮点数；同时检查输出确实保留了两位小数。
    value = mol_weight(ASPIRIN)["mol_weight"]
    assert value == pytest.approx(180.16, abs=0.005)
    assert value == round(value, 2)


def test_aspirin_formula():
    assert formula(ASPIRIN) == {"formula": "C9H8O4"}


def test_aspirin_logp():
    assert logp(ASPIRIN)["logp"] == pytest.approx(1.3101, abs=0.005)


def test_aspirin_tpsa():
    assert tpsa(ASPIRIN)["tpsa"] == pytest.approx(63.60, abs=0.005)


def test_aspirin_hydrogen_bond_counts():
    result = hbd_hba(ASPIRIN)
    assert result == {"hbd": 1, "hba": 3}
    assert type(result["hbd"]) is int
    assert type(result["hba"]) is int


def test_aspirin_lipinski():
    result = lipinski(ASPIRIN)
    assert result["mol_weight"] == pytest.approx(180.16, abs=0.005)
    assert result["logp"] == pytest.approx(1.3101, abs=0.005)
    assert result["hbd"] == 1
    assert result["hba"] == 3
    assert result["violations"] == 0
    assert result["lipinski_pass"] is True


def test_lipinski_allows_one_violation_as_in_demo():
    """项目示例的判定约定：违反不超过一项仍记为通过。"""
    result = lipinski("C" * 20)
    assert result["mol_weight"] < 500
    assert result["logp"] > 5
    assert result["violations"] == 1
    assert result["lipinski_pass"] is True


def test_lipinski_rejects_two_violations():
    result = lipinski("C" * 40)
    assert result["mol_weight"] > 500
    assert result["logp"] > 5
    assert result["violations"] == 2
    assert result["lipinski_pass"] is False


@pytest.mark.parametrize(
    "bad_smiles",
    ["not-a-smiles", "", "   ", None, 42, "C(C)(C)(C)(C)C", "CCO garbage"],
)
def test_all_single_molecule_tools_return_error(bad_smiles):
    for tool in SINGLE_MOLECULE_TOOLS:
        assert_error(tool(bad_smiles))


def test_same_molecule_similarity_is_one():
    assert similarity(ASPIRIN, ASPIRIN)["similarity"] == pytest.approx(1.0)


def test_equivalent_smiles_similarity_is_one():
    # 同一个分子可以有多种 SMILES 写法，相似度应按分子结构计算。
    equivalent_aspirin = "O=C(O)c1ccccc1OC(C)=O"
    assert similarity(ASPIRIN, equivalent_aspirin)["similarity"] == pytest.approx(1.0)


def test_similarity_is_symmetric_for_different_molecules():
    forward = similarity(ASPIRIN, "CCO")["similarity"]
    backward = similarity("CCO", ASPIRIN)["similarity"]
    assert 0 <= forward < 1
    assert forward == pytest.approx(backward)


@pytest.mark.parametrize(
    ("smiles_a", "smiles_b"),
    [
        ("not-a-smiles", ASPIRIN),
        (ASPIRIN, "not-a-smiles"),
        ("", ASPIRIN),
        (ASPIRIN, None),
    ],
)
def test_similarity_returns_error_for_invalid_input(smiles_a, smiles_b):
    assert_error(similarity(smiles_a, smiles_b))


def test_tool_schemas_match_registered_functions():
    """声明的工具必须真的存在，且参数名与 Python 函数签名一致。"""
    expected_names = {
        "mol_weight", "logp", "formula", "tpsa", "hbd_hba", "lipinski", "similarity"
    }
    assert set(TOOLS) == expected_names
    assert isinstance(TOOL_SCHEMAS, list)
    json.dumps(TOOL_SCHEMAS)  # schema 应能转换为 JSON，不含 Python 专有对象。

    names = [schema["function"]["name"] for schema in TOOL_SCHEMAS]
    assert len(names) == len(set(names)) == len(TOOLS)
    assert set(names) == set(TOOLS)

    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        definition = schema["function"]
        assert definition["strict"] is True
        assert definition["description"].strip()
        function = TOOLS[definition["name"]]
        assert callable(function)
        assert function.__name__ == definition["name"]

        parameters = definition["parameters"]
        argument_names = set(inspect.signature(function).parameters)
        assert parameters["type"] == "object"
        assert parameters["additionalProperties"] is False
        assert set(parameters["properties"]) == argument_names
        assert set(parameters["required"]) == argument_names
        assert len(parameters["required"]) == len(argument_names)
        for parameter in parameters["properties"].values():
            assert parameter["type"] == "string"
