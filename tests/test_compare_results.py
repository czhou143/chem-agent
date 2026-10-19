"""对比表从结果文件计算，并拒绝不具可比性的输入。"""

from copy import deepcopy
import json

import pytest

from eval.compare_results import compare_results
from eval.run_eval import score_response


def records():
    rows = []
    for index in range(2):
        row = {"id": f"q{index}", "question": f"乙醇分子量问题 {index}",
               "expected_tool": "mol_weight", "expected_value": {"mol_weight": 46.07}, "tolerance": 0.01,
               "mode": "live", "answer": "乙醇分子量为 46.07。", "error": None,
               "tool_calls": [{"name": "mol_weight", "arguments": {"smiles": "CCO"},
                               "result": {"mol_weight": 46.07}}],
               "usage_complete": True, "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
               "run_config": {"model": "test-model", "base_url": "test-url", "max_output_tokens": 2048,
                              "thinking": "disabled", "tool_schemas_sha256": "tools",
                              "questions_sha256": "questions", "system_prompt_sha256": "before"}}
        row.update(score_response(row, row))
        rows.append(row)
    return rows


def write(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_comparison_reports_decrease_and_sums_only_file_values(tmp_path):
    before, after = records(), records()
    for row in after:
        row.update(prompt_tokens=200, completion_tokens=20, total_tokens=220)
        row["run_config"]["system_prompt_sha256"] = "after"
    after[1]["tool_calls"][0]["result"]["mol_weight"] = 99.0
    after[1].update(score_response(after[1], after[1]))
    a, b = tmp_path / "before.jsonl", tmp_path / "after.jsonl"
    write(a, before)
    write(b, after)
    original = {path: path.read_bytes() for path in (a, b)}
    report = compare_results(a, b)
    assert "| 总准确率 | 100.00%（2/2） | 50.00%（1/2） |" in report
    assert "| 工具选择准确率 | 100.00%（2/2） | 100.00%（2/2） |" in report
    assert "| total_tokens | 220 | 440 |" in report
    assert "**660**" in report and "-50.00 个百分点" in report
    assert "q1" in report and "after.jsonl:2" in report
    assert all(path.read_bytes() == content for path, content in original.items())


@pytest.mark.parametrize("change", ["dry_run", "unknown_usage", "changed_model", "changed_question", "wrong_score", "wrong_total"])
def test_comparison_rejects_invalid_inputs(tmp_path, change):
    before, after = records(), deepcopy(records())
    if change == "dry_run":
        after[0]["mode"] = "dry-run"
    elif change == "unknown_usage":
        after[0]["usage_complete"] = False
    elif change == "changed_model":
        after[0]["run_config"]["model"] = "other-model"
    elif change == "changed_question":
        after[0]["question"] = "different"
    elif change == "wrong_score":
        after[0]["correct"] = False
    else:
        after[0]["total_tokens"] = 999
    a, b = tmp_path / "before.jsonl", tmp_path / "after.jsonl"
    write(a, before)
    write(b, after)
    with pytest.raises(ValueError):
        compare_results(a, b)
