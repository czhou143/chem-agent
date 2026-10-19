"""评分器的离线测试：替换 agent.run，不调用真实模型或使用真实密钥。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

from chem_agent import agent
from eval import run_eval


def question_row(question_id="sample-1", *, invalid=False):
    """小型测试题避免依赖完整 40 题或真实模型的表达方式。"""
    return {
        "id": question_id,
        "question": "请计算乙醇的分子量，SMILES：`CCO`。",
        "expected_tool": None if invalid else "mol_weight",
        "expected_value": {"error": "非法 SMILES"} if invalid else {"mol_weight": 46.07},
        "tolerance": 0.01,
    }


def response(*, value=46.07, name="mol_weight", answer="乙醇分子量为 46.07。"):
    return {
        "answer": answer,
        "tool_calls": [
            {"name": name, "arguments": {"smiles": "CCO"}, "result": {"mol_weight": value}}
        ],
        "steps": 2,
    }


def write_questions(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("actual", "expected", "tolerance", "matches"),
    [
        (10.25, 10.0, 0.25, True),  # 容差边界应包含在内。
        (10.25001, 10.0, 0.25, False),
        (1_000_000_000_001.0, 1_000_000_000_000.0, 0.01, False),
        (2.01, 2, 0.1, False),  # 氢键计数等整数不允许被浮点容差放过。
        (1, True, 0.0, False),  # Python 的 1 == True，但化学结果不能混同。
        ("C9H8O4", "C9H8O4", 0.0, True),
        ("C9H8O4 ", "C9H8O4", 0.0, False),
        ({"pass": True, "values": [2, {"x": 10.25}]},
         {"pass": True, "values": [2, {"x": 10.0}]}, 0.25, True),
        ({"pass": False}, {"pass": True}, 1.0, False),
        ("46.07", 46.07, 0.01, False),
    ],
)
def test_matches_expected_uses_explicit_types_and_absolute_tolerance(
    actual, expected, tolerance, matches
):
    assert run_eval.matches_expected(actual, expected, tolerance) is matches


def test_nonfinite_numbers_are_not_accepted():
    for actual, expected in [
        (float("nan"), 1.0),
        (1.0, float("nan")),
        (float("inf"), float("inf")),
        (float("-inf"), 1.0),
    ]:
        assert not run_eval.matches_expected(actual, expected, 0.01)


def test_result_structure_must_match():
    assert not run_eval.matches_expected({"hbd": 1}, {"hbd": 1, "hba": 2}, 0)
    assert not run_eval.matches_expected({"hbd": 1, "extra": 2}, {"hbd": 1}, 0)
    assert not run_eval.matches_expected([1, 2], [2, 1], 0)
    assert not run_eval.matches_expected([1], [1, 2], 0)
    assert not run_eval.matches_expected([1, 2], {"hbd": 1, "hba": 2}, 0)


@pytest.mark.parametrize(
    ("answer", "invalid"),
    [
        ("输入的 SMILES 无效，无法解析，请检查结构。", True),
        ("这不是有效的 SMILES。", True),
        ("该 SMILES 非法，不能用于计算分子性质。", True),
        ("", False),
        ("这是有效的 SMILES。", False),
        ("这不是无效输入，可以用于计算。", False),
        ("如果无效请告知。", False),
        ("这不是一个无效的 SMILES。", False),
        ("该 SMILES 有效，没有无效或非法的问题。", False),
        ("该字符串不是合法的 SMILES。", True),
        ("空字符串不能构成有效的 SMILES。", True),
        ("并不能说该输入无效。", False),
        ("无法确认该 SMILES 是否无效。", False),
    ],
)
def test_invalid_answer_requires_an_actual_invalid_input_statement(answer, invalid):
    assert run_eval.answer_reports_invalid(answer) is invalid


def test_correct_first_tool_and_result_pass():
    scored = run_eval.score_response(question_row(), response())
    assert scored["tool_correct"] is True
    assert scored["value_correct"] is True
    assert scored["correct"] is True
    assert isinstance(scored["reason"], str)


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "   \n",
        "已达到 max_steps=5 的请求上限，尚未获得最终回答；已保留执行过的工具结果。",
    ],
)
def test_missing_final_answer_fails_overall_even_with_correct_tool_result(answer):
    result = response(answer=answer)
    result["steps"] = 5

    scored = run_eval.score_response(question_row(), result)

    # 两项子分数保留，便于区分工具计算成功和整个回答任务完成。
    assert scored["tool_correct"] is True
    assert scored["value_correct"] is True
    assert scored["correct"] is False
    assert scored["reason"]


def test_final_answer_at_fifth_step_is_not_mistaken_for_step_limit_failure():
    result = response()
    result["steps"] = 5

    scored = run_eval.score_response(question_row(), result)

    assert scored["tool_correct"] is True
    assert scored["value_correct"] is True
    assert scored["correct"] is True


def test_later_correct_tool_cannot_hide_a_wrong_first_tool():
    result = response()
    result["tool_calls"].insert(
        0, {"name": "formula", "arguments": {"smiles": "CCO"}, "result": {"formula": "C2H6O"}}
    )
    scored = run_eval.score_response(question_row(), result)
    assert scored["tool_correct"] is False
    assert scored["value_correct"] is False
    assert scored["correct"] is False


def test_correct_tool_with_incorrect_value_fails_value_score():
    scored = run_eval.score_response(question_row(), response(value=60.0))
    assert scored["tool_correct"] is True
    assert scored["value_correct"] is False
    assert scored["correct"] is False


def test_answer_alone_does_not_replace_a_required_tool_call():
    result = {"answer": "乙醇分子量为 46.07。", "tool_calls": [], "steps": 1}
    scored = run_eval.score_response(question_row(), result)
    assert scored["tool_correct"] is False
    assert scored["value_correct"] is False
    assert scored["correct"] is False


def test_invalid_question_scores_the_answer_with_or_without_tool_calls():
    row = question_row(invalid=True)
    for calls in ([], [{"name": "mol_weight", "arguments": {}, "result": {"error": "parse failed"}}]):
        result = {"answer": "输入的 SMILES 无效，请检查后重试。", "tool_calls": calls, "steps": 1}
        scored = run_eval.score_response(row, result)
        assert scored["tool_correct"] is True
        assert scored["value_correct"] is True
        assert scored["correct"] is True

    # 仅有工具错误还不够；用户必须从最终回答知道自己的输入无效。
    result["answer"] = "计算已结束。"
    scored = run_eval.score_response(row, result)
    assert scored["tool_correct"] is False
    assert scored["value_correct"] is False
    assert scored["correct"] is False


def test_evaluate_one_records_exception_as_failure(monkeypatch):
    mocked = Mock(side_effect=RuntimeError("模拟模型暂不可用"))
    monkeypatch.setattr(agent, "run", mocked)
    row = question_row()

    record = run_eval.evaluate_one(row)

    mocked.assert_called_once_with(row["question"])
    assert record["id"] == row["id"]
    assert record["question"] == row["question"]
    assert record["mode"] == "live"
    assert "模拟模型暂不可用" in record["error"]
    assert record["tool_correct"] is record["value_correct"] is record["correct"] is False


def test_exception_keys_are_redacted_from_records_reports_and_console(tmp_path, monkeypatch, capsys):
    # 全部为人为构造的测试值，不能读取或依赖使用者的真实密钥。
    active_key = "example-test-key-without-common-prefix"
    other_key = "sk-0123456789abcdef0123456789abcdef"
    monkeypatch.setenv("LLM_API_KEY", active_key)
    mocked = Mock(side_effect=RuntimeError(f"模拟认证故障：{active_key}，备用密钥 {other_key}"))
    monkeypatch.setattr(agent, "run", mocked)
    questions_path = tmp_path / "questions.jsonl"
    write_questions(questions_path, [question_row()])

    summary = run_eval.run_evaluation(questions_path, tmp_path / "results")

    assert summary["total"] == 1
    assert summary["correct"] == 0
    records_text = Path(summary["records_path"]).read_text(encoding="utf-8")
    report_text = Path(summary["summary_path"]).read_text(encoding="utf-8")
    captured = capsys.readouterr()
    saved_or_printed = records_text + report_text + captured.out + captured.err
    assert active_key not in saved_or_printed
    assert other_key not in saved_or_printed
    record = json.loads(records_text.strip())
    assert "模拟认证故障" in record["error"]
    assert record["correct"] is False


def test_run_continues_after_one_question_fails_and_saves_reports(tmp_path, monkeypatch):
    rows = [question_row("failed-question"), question_row("passed-question")]
    questions_path = tmp_path / "questions.jsonl"
    write_questions(questions_path, rows)
    mocked = Mock(side_effect=[RuntimeError("模拟网络故障"), response()])
    monkeypatch.setattr(agent, "run", mocked)

    summary = run_eval.run_evaluation(questions_path, tmp_path / "results")

    assert mocked.call_count == 2
    assert summary["total"] == 2
    assert summary["correct"] == summary["tool_correct"] == summary["value_correct"] == 1
    assert summary["accuracy"] == summary["tool_accuracy"] == summary["value_accuracy"] == 0.5
    assert summary["categories"]
    records_path = Path(summary["records_path"])
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    assert [record["id"] for record in records] == [row["id"] for row in rows]
    assert records[0]["correct"] is False
    assert records[1]["correct"] is True
    assert all(record["mode"] == "live" for record in records)
    assert all(field in records[1] for field in ("answer", "tool_calls", "steps"))
    report = Path(summary["summary_path"]).read_text(encoding="utf-8")
    assert "failed-question" in report
    assert "mol_weight" in report
    assert "50.00%" in report or "50.0%" in report or "50%" in report


def test_limit_only_evaluates_first_n_questions(tmp_path, monkeypatch):
    rows = [question_row(f"question-{index}") for index in range(3)]
    questions_path = tmp_path / "questions.jsonl"
    write_questions(questions_path, rows)
    mocked = Mock(return_value=response())
    monkeypatch.setattr(agent, "run", mocked)

    summary = run_eval.run_evaluation(questions_path, tmp_path / "results", limit=1)

    mocked.assert_called_once_with(rows[0]["question"])
    assert summary["total"] == summary["correct"] == 1
    records = Path(summary["records_path"]).read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    assert json.loads(records[0])["id"] == rows[0]["id"]


def test_dry_run_never_calls_agent_or_overwrites_live_report(tmp_path, monkeypatch):
    questions_path = tmp_path / "questions.jsonl"
    rows = [question_row(), question_row("invalid-question", invalid=True)]
    rows[1]["question"] = "SMILES：`not-a-smiles`，请检查输入。"
    original_rows = deepcopy(rows)
    write_questions(questions_path, rows)
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    live_report = output_dir / "summary.md"
    live_report.write_text("真实评估已有结果", encoding="utf-8")
    forbidden = Mock(side_effect=AssertionError("dry-run 不应调用真实 Agent"))
    monkeypatch.setattr(agent, "run", forbidden)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    summary = run_eval.run_evaluation(questions_path, output_dir, dry_run=True)

    forbidden.assert_not_called()
    assert summary["total"] == 2
    assert Path(summary["summary_path"]).parent == output_dir / "dry_run"
    assert Path(summary["records_path"]).parent == output_dir / "dry_run"
    assert live_report.read_text(encoding="utf-8") == "真实评估已有结果"
    records = [
        json.loads(line)
        for line in Path(summary["records_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    assert all(record["mode"] == "dry-run" for record in records)
    assert [record["expected_value"] for record in records] == [row["expected_value"] for row in rows]
    assert rows == original_rows


def multi_question(*, repeated_tool=False):
    row = question_row("multi-question")
    row.update(category="multi_step", expected_tool="formula",
               expected_tools=["formula", "mol_weight"],
               expected_value=[{"formula": "C2H6O"}, {"mol_weight": 46.07}])
    if repeated_tool:
        row.update(expected_tool="logp", expected_tools=["logp", "logp"],
                   expected_value=[{"logp": 3.0}, {"logp": 1.0}])
    return row


def multi_response(row):
    return {"answer": "两次计算已完成。", "steps": 3,
            "tool_calls": [{"name": name, "arguments": {}, "result": deepcopy(value)}
                           for name, value in zip(row["expected_tools"], row["expected_value"])]}


@pytest.mark.parametrize("repeated_tool", [False, True])
def test_multi_tool_ordered_results_pass(repeated_tool):
    row = multi_question(repeated_tool=repeated_tool)
    scored = run_eval.score_response(row, multi_response(row))
    assert scored["tool_correct"] is scored["value_correct"] is scored["correct"] is True


@pytest.mark.parametrize("change", ["reversed", "missing", "extra", "wrong_name", "malformed"])
def test_multi_tool_sequence_must_match_exactly(change):
    row = multi_question()
    result = multi_response(row)
    calls = result["tool_calls"]
    if change == "reversed":
        calls.reverse()
    elif change == "missing":
        calls.pop()
    elif change == "extra":
        calls.append(deepcopy(calls[-1]))
    elif change == "wrong_name":
        calls[-1]["name"] = "tpsa"
    else:
        result["tool_calls"] = None
    scored = run_eval.score_response(row, result)
    assert scored["tool_correct"] is False
    assert scored["correct"] is False


@pytest.mark.parametrize("change", ["outside_tolerance", "missing_result", "tool_error", "string_mismatch"])
def test_multi_tool_checks_each_result(change):
    row = multi_question()
    result = multi_response(row)
    calls = result["tool_calls"]
    if change == "outside_tolerance":
        calls[1]["result"]["mol_weight"] += 0.02
    elif change == "missing_result":
        del calls[1]["result"]
    elif change == "tool_error":
        calls[1]["result"] = {"error": "计算失败"}
    else:
        calls[0]["result"]["formula"] = "C2H6"
    scored = run_eval.score_response(row, result)
    assert scored["tool_correct"] is True
    assert scored["value_correct"] is scored["correct"] is False


def test_multi_tool_accepts_tolerance_and_rejects_swapped_repeated_tool_values():
    row = multi_question(repeated_tool=True)
    result = multi_response(row)
    result["tool_calls"][1]["result"]["logp"] += row["tolerance"]
    assert run_eval.score_response(row, result)["correct"] is True
    result["tool_calls"].reverse()
    scored = run_eval.score_response(row, result)
    assert scored["tool_correct"] is True  # 两次名称相同，必须靠逐项结果发现交换。
    assert scored["value_correct"] is scored["correct"] is False


@pytest.mark.parametrize("expected,actual,correct", [
    ([], {"alerts": []}, True),
    (["a", "b"], {"alerts": [{"name": "b"}, {"name": "a"}]}, True),
    (["a", "a"], {"alerts": [{"name": "a"}]}, False),
    (["a"], {"alerts": [{"name": "b"}]}, False),
    ([], {"alerts": [{"name": "a"}]}, False),
    ([], {}, False),
    ([], {"error": "输入无效", "alerts": []}, False),
    (["a"], {"alerts": [{"catalog": "BRENK"}]}, False),
])
def test_safety_scoring_compares_alert_names(expected, actual, correct):
    row = question_row()
    row.update(expected_tool="safety_check", expected_value=expected, tolerance=0)
    result = response(name="safety_check")
    result["tool_calls"][0]["result"] = actual
    assert run_eval.score_response(row, result)["correct"] is correct


@pytest.mark.parametrize("change", ["names_string", "unknown", "missing_value", "wrong_first", "single", "non_dict_value"])
def test_loader_rejects_malformed_multi_tool_rows(tmp_path, change):
    row = multi_question()
    if change == "names_string":
        row["expected_tools"] = "formula"
    elif change == "unknown":
        row["expected_tools"][1] = "unknown_tool"
    elif change == "missing_value":
        row["expected_value"].pop()
    elif change == "wrong_first":
        row["expected_tool"] = "mol_weight"
    elif change == "single":
        row["expected_tools"].pop()
    else:
        row["expected_value"][1] = 46.07
    path = tmp_path / "questions.jsonl"
    write_questions(path, [row])
    with pytest.raises(ValueError, match="多工具"):
        run_eval._load_questions(path)


def test_v2_dry_run_saves_separate_report_and_never_calls_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "EVAL_DIR", tmp_path)
    forbidden = Mock(side_effect=AssertionError("不能调用真实 API"))
    monkeypatch.setattr(agent, "run", forbidden)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    results = tmp_path / "results"
    results.mkdir()
    old_report = results / "summary.md"
    old_report.write_bytes(b"existing v1 report\r\n")
    question_path = tmp_path / "questions_v2.jsonl"
    safety = question_row("safety-question")
    safety.update(category="safety", expected_tool="safety_check", expected_value=[])
    rows = [multi_question(), safety]
    write_questions(question_path, rows)

    summary = run_eval.run_evaluation(question_path, dry_run=True)

    forbidden.assert_not_called()
    assert summary["correct"] == summary["total"] == 2
    assert set(summary["categories"]) == {"multi_step", "safety"}
    assert Path(summary["summary_path"]).parent == results / "questions_v2" / "dry_run"
    assert old_report.read_bytes() == b"existing v1 report\r\n"
    records = [json.loads(line) for line in Path(summary["records_path"]).read_text(encoding="utf-8").splitlines()]
    assert [call["name"] for call in records[0]["tool_calls"]] == rows[0]["expected_tools"]
    assert records[1]["tool_calls"][0]["result"]["alerts"] == []
    assert "不代表模型准确率" in Path(summary["summary_path"]).read_text(encoding="utf-8")


@pytest.mark.parametrize("custom_questions", [False, True])
def test_cli_questions_option_defaults_to_v1(monkeypatch, capsys, custom_questions):
    mocked = Mock(return_value={"total": 2, "accuracy": 1.0, "tool_accuracy": 1.0,
                               "summary_path": "mock/summary.md"})
    monkeypatch.setattr(run_eval, "run_evaluation", mocked)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    argv = ["run_eval.py", "--dry-run", "--limit", "2"]
    if custom_questions:
        argv.extend(["--questions", "eval/questions_v2.jsonl"])
    monkeypatch.setattr(sys, "argv", argv)
    assert run_eval.main() == 0
    kwargs = mocked.call_args.kwargs
    expected_path = Path("eval/questions_v2.jsonl") if custom_questions else run_eval.EVAL_DIR / "questions.jsonl"
    assert kwargs["questions_path"] == expected_path
    assert kwargs["dry_run"] is True and kwargs["limit"] == 2
    assert "完成 2 题" in capsys.readouterr().out
