"""只用真实本地 RDKit 目录验证结构警示；不调用模型或网络。"""

import json

import pytest

from chem_agent import agent, tools


TNT = "Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]"
EXPECTED_DISCLAIMER = "仅基于结构的规则粗筛（药物化学警示库），不是毒性或爆炸性评估，请以 MSDS/SDS 为准"
EXPECTED_SOURCES = {
    "BRENK": "Brenk et al., ChemMedChem 2008, doi:10.1002/cmdc.200700139",
    "NIH": "Jadhav et al., J Med Chem 2010, doi:10.1021/jm901070c",
}


def test_safety_tnt_nitro():
    result = tools.safety_check(TNT)
    assert set(result) == {"alerts", "alert_count", "disclaimer"}
    hits = {(alert["catalog"], alert["name"]) for alert in result["alerts"]}
    assert ("BRENK", "nitro_group") in hits
    assert ("NIH", "trinitro_aromatic") in hits
    assert result["alert_count"] == len(result["alerts"]) == len(hits)
    assert result["disclaimer"] == EXPECTED_DISCLAIMER
    for alert in result["alerts"]:
        assert set(alert) == {"name", "catalog", "source"}
        assert alert["source"] == EXPECTED_SOURCES[alert["catalog"]]
    json.dumps(result, ensure_ascii=False)


def test_safety_ethanol_clean():
    assert tools.safety_check("CCO") == {
        "alerts": [], "alert_count": 0, "disclaimer": EXPECTED_DISCLAIMER
    }


@pytest.mark.parametrize("smiles", ["not_a_smiles", "", "   ", None, 42, "C1CC", "CCO garbage"])
def test_safety_invalid(smiles):
    result = tools.safety_check(smiles)
    assert set(result) == {"error"}
    assert isinstance(result["error"], str) and result["error"].strip()


def test_safety_acetyl_chloride_acyl_halide():
    result = tools.safety_check("CC(=O)Cl")
    assert {(alert["catalog"], alert["name"]) for alert in result["alerts"]} == {
        ("BRENK", "acid_halide"), ("NIH", "carbonyl_halide")
    }
    # 同一官能团命中两个目录，是两条警示，不是两个酰卤基团。
    assert result["alert_count"] == 2


def test_safety_methyl_azide():
    result = tools.safety_check("CN=[N+]=[N-]")
    hits = {(alert["catalog"], alert["name"]) for alert in result["alerts"]}
    assert ("BRENK", "Azido_group") in hits
    assert ("NIH", "azide") in hits


def test_safety_returns_fresh_results_for_each_call():
    expected = tools.safety_check(TNT)
    changed = tools.safety_check(TNT)
    changed["alerts"][0]["name"] = "caller mutation"
    changed["alerts"].clear()
    assert tools.safety_check(TNT) == expected


def test_safety_schema_and_registry_match():
    names = [schema["function"]["name"] for schema in tools.TOOL_SCHEMAS]
    assert len(names) == len(set(names)) == len(tools.TOOLS) == 8
    assert set(names) == set(tools.TOOLS)
    assert tools.TOOLS["safety_check"] is tools.safety_check
    schema = next(item["function"] for item in tools.TOOL_SCHEMAS if item["function"]["name"] == "safety_check")
    assert schema["parameters"]["required"] == ["smiles"]
    assert all(word in schema["description"] for word in ("BRENK", "NIH", "disclaimer", "MSDS/SDS"))


def test_safety_system_prompt_requires_disclaimer_and_refuses_synthesis():
    prompt = agent.SYSTEM_PROMPT
    assert all(word in prompt for word in ("安全", "危险", "毒性", "safety_check", "disclaimer"))
    assert "不提供任何合成路线或制备方法" in prompt
    assert "直接拒绝" in prompt
