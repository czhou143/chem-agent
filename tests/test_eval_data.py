"""评估题库测试：检查结构、覆盖范围，以及能否由真实工具重新生成。"""

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import re

import pytest

from chem_agent.tools import TOOLS
from eval.build_questions import build_questions, validate_questions


DATA_PATH = Path(__file__).resolve().parents[1] / "eval" / "questions.jsonl"
FIELDS = {"id", "question", "expected_tool", "expected_value", "tolerance"}


@pytest.fixture(scope="module")
def rows():
    """JSONL 每一行都是独立的 JSON 对象，不应有空行。"""
    lines = DATA_PATH.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 40
    assert all(line.strip() for line in lines)
    return [json.loads(line) for line in lines]


def test_questions_have_required_fields_and_unique_ids(rows):
    ids = []
    for row in rows:
        assert isinstance(row, dict)
        assert set(row) == FIELDS
        assert isinstance(row["id"], str) and row["id"].strip()
        assert isinstance(row["question"], str)
        assert re.search(r"[\u4e00-\u9fff]", row["question"])
        ids.append(row["id"])
    assert len(set(ids)) == len(ids) == 40


def test_tool_names_and_question_distribution(rows):
    # null 表示非法输入题；这些题不要求模型选择某个特定工具。
    valid_rows = [row for row in rows if row["expected_tool"] is not None]
    invalid_rows = [row for row in rows if row["expected_tool"] is None]
    counts = Counter(row["expected_tool"] for row in valid_rows)

    assert len(valid_rows) == 36
    assert len(invalid_rows) == 4
    # v1 的 40 道题只覆盖原有 7 个工具；新工具将在后续题集验证。
    assert set(counts) == {"mol_weight", "logp", "formula", "tpsa", "hbd_hba", "lipinski", "similarity"}
    assert set(counts) <= set(TOOLS)
    assert all(name in TOOLS for name in counts)
    assert all(5 <= count <= 6 for count in counts.values())


def test_expected_values_and_tolerances_are_well_formed(rows):
    for row in rows:
        expected = row["expected_value"]
        assert isinstance(expected, dict) and expected
        if row["expected_tool"] is None:
            assert isinstance(expected.get("error"), str)
            assert expected["error"].strip()
        else:
            assert "error" not in expected

        tolerance = row["tolerance"]
        # bool 在 Python 中也属于 int，因此这里明确排除 True/False。
        assert type(tolerance) in (int, float)
        assert math.isfinite(tolerance) and tolerance >= 0


def test_saved_questions_match_fresh_tool_calculations(rows):
    # 不在测试里手填标准答案，直接重新调用生成器比较完整题库。
    assert rows == build_questions()


def test_validator_accepts_saved_questions(rows):
    assert validate_questions(rows) is None


@pytest.mark.parametrize(
    "change",
    ["missing_field", "unknown_tool", "changed_answer", "duplicate_id", "changed_question"],
)
def test_validator_rejects_corrupted_questions(rows, change):
    changed = deepcopy(rows)
    target = next(row for row in changed if row["expected_tool"] == "mol_weight")

    if change == "missing_field":
        del target["expected_value"]
    elif change == "unknown_tool":
        target["expected_tool"] = "not_a_registered_tool"
    elif change == "changed_answer":
        # 偏移量超过该题容差，避免浮点比较掩盖被篡改的答案。
        target["expected_value"]["mol_weight"] += max(1.0, 2 * target["tolerance"])
    elif change == "duplicate_id":
        changed[1]["id"] = changed[0]["id"]
    elif change == "changed_question":
        target["question"] = "请计算 SMILES：CC 的分子量。"

    with pytest.raises(ValueError):
        validate_questions(changed)
