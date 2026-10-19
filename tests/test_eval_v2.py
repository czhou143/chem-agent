"""v2 题集及历史数据保护；只调用本地 RDKit，不调用模型。"""

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

import pytest

from chem_agent.tools import TOOLS
from eval import build_questions as builder


ROOT = Path(__file__).resolve().parents[1]
# S3 开始前的历史内容指纹。只在计算指纹时统一换行，兼容 Windows/Linux checkout。
# 生成器执行前后另用原始 bytes 比较，不能借换行归一化掩盖写入操作。
HISTORICAL_SHA256 = {
    "eval/questions.jsonl": "043b52b2cad0d103e75ef5debba2dfde259eafe90af52b6cf6ce05be32899fc8",
    "eval/results/20261018_221950_033160.jsonl": "1c4a0c5a03ba6b4ef0770ba45c390c52431919e42845935ac0eb58ce0941718b",
    "eval/results/summary.md": "35efae3943cb4c73a3eabd1143edbc9e35d5ac1dbed71ec376c94e8152a84f98",
}


@pytest.fixture(scope="module")
def v2_rows():
    lines = (ROOT / "eval/questions_v2.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 60 and all(line.strip() for line in lines)
    return [json.loads(line) for line in lines]


def test_v2_keeps_original_40_rows_and_adds_required_categories(v2_rows):
    original = (ROOT / "eval/questions.jsonl").read_text(encoding="utf-8").splitlines()
    assert v2_rows[:40] == [json.loads(line) for line in original]
    assert all(set(row) == builder.FIELDS for row in v2_rows[:40])
    assert Counter(row["category"] for row in v2_rows[40:]) == {
        "safety": 8, "multi_step": 6, "confusable": 6,
    }
    assert [row["id"] for row in v2_rows] == [f"q{i:03d}" for i in range(1, 61)]
    assert len({row["question"] for row in v2_rows}) == 60
    for row in v2_rows[40:]:
        fields = builder.FIELDS | {"category"}
        if row["category"] == "multi_step":
            fields.add("expected_tools")
            assert len(row["expected_tools"]) == 2
            assert row["expected_tool"] == row["expected_tools"][0]
            assert all(name in TOOLS for name in row["expected_tools"])
        assert set(row) == fields
        assert row["expected_tool"] in TOOLS


def test_all_new_answers_match_tools_on_smiles_in_the_question(v2_rows):
    # 从题干独立取输入，避免只拿生成器和自己比较，漏掉题干/计算输入不一致。
    for row in v2_rows[40:]:
        smiles = re.findall(r"SMILES：`([^`]+)`", row["question"])
        if row["category"] == "multi_step":
            assert len(smiles) == 2
            actual = [TOOLS[name](value) for name, value in zip(row["expected_tools"], smiles)]
        else:
            assert len(smiles) == 1
            actual = TOOLS[row["expected_tool"]](smiles[0])
            if row["category"] == "safety":
                actual = sorted(alert["name"] for alert in actual["alerts"])
        assert row["expected_value"] == actual, row["id"]


def test_safety_includes_clean_ethanol_and_glucose(v2_rows):
    for molecule in ("乙醇", "葡萄糖"):
        row = next(row for row in v2_rows[40:48] if molecule in row["question"])
        assert row["expected_tool"] == "safety_check"
        assert row["expected_value"] == []


def test_saved_v2_matches_generation_and_passes_validation(v2_rows):
    assert builder.build_questions_v2() == v2_rows
    assert builder.validate_questions_v2(v2_rows) is None


@pytest.mark.parametrize("change", ["answer", "order", "missing_field", "category", "non_object"])
def test_v2_validator_rejects_corruption(v2_rows, change):
    changed = deepcopy(v2_rows)
    if change == "answer":
        changed[40]["expected_value"].append("invented_alert")
    elif change == "order":
        changed[48]["expected_tools"].reverse()
    elif change == "missing_field":
        del changed[59]["tolerance"]
    elif change == "category":
        changed[59]["category"] = "safety"
    else:
        changed[40] = None
    with pytest.raises(ValueError):
        builder.validate_questions_v2(changed)


@pytest.mark.parametrize("relative_path,expected_hash", HISTORICAL_SHA256.items())
def test_historical_question_and_result_files_are_unchanged(relative_path, expected_hash):
    content = (ROOT / relative_path).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(content).hexdigest() == expected_hash


@pytest.mark.parametrize("check_only", [False, True])
def test_generator_preserves_v1_and_old_results_and_reports_hash(tmp_path, monkeypatch, capsys, check_only):
    for relative_path in (*HISTORICAL_SHA256, "eval/questions_v2.jsonl"):
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative_path).read_bytes())
    snapshots = {name: (tmp_path / name).read_bytes() for name in HISTORICAL_SHA256}
    v2_path = tmp_path / "eval/questions_v2.jsonl"
    original_v2 = v2_path.read_bytes()
    if not check_only:
        v2_path.unlink()  # 真正验证首次生成，而不是只校验仓库里已有的文件。
    monkeypatch.setattr(builder, "OUTPUT_PATH", tmp_path / "eval/questions.jsonl")
    monkeypatch.setattr(builder, "V2_OUTPUT_PATH", v2_path)
    monkeypatch.setattr(sys, "argv", ["build_questions.py", *(["--check"] if check_only else [])])

    assert builder.main() == 0

    for name, before in snapshots.items():
        assert (tmp_path / name).read_bytes() == before
    assert v2_path.read_bytes() == original_v2
    assert b"\r\n" not in v2_path.read_bytes()
    output = capsys.readouterr().out
    assert "v1 校验通过" in output and "40 题，未改写" in output and "60 题" in output
    assert hashlib.sha256(v2_path.read_bytes()).hexdigest() in output


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_check_detects_changed_answers_without_rewriting_files(tmp_path, monkeypatch, capsys, version):
    v1_path, v2_path = tmp_path / "questions.jsonl", tmp_path / "questions_v2.jsonl"
    v1_path.write_bytes((ROOT / "eval/questions.jsonl").read_bytes())
    v2_path.write_bytes((ROOT / "eval/questions_v2.jsonl").read_bytes())
    target = v1_path if version == "v1" else v2_path
    target.write_text(target.read_text(encoding="utf-8").replace("180.16", "999.99", 1), encoding="utf-8")
    before = {path: path.read_bytes() for path in (v1_path, v2_path)}
    monkeypatch.setattr(builder, "OUTPUT_PATH", v1_path)
    monkeypatch.setattr(builder, "V2_OUTPUT_PATH", v2_path)
    monkeypatch.setattr(sys, "argv", ["build_questions.py", "--check"])

    assert builder.main() == 1
    assert "校验失败" in capsys.readouterr().err
    assert all(path.read_bytes() == content for path, content in before.items())
