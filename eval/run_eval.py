"""逐题评测 Agent；--dry-run 使用模拟答案，只检查流程，不衡量模型能力。"""

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from unittest.mock import Mock

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chem_agent import agent
from chem_agent.tools import TOOLS


EVAL_DIR = Path(__file__).resolve().parent
REQUIRED_FIELDS = {"id", "question", "expected_tool", "expected_value", "tolerance"}
CATEGORY_NAMES = {
    "mol_weight": "分子量", "logp": "LogP", "formula": "分子式",
    "tpsa": "TPSA", "hbd_hba": "氢键供体/受体", "lipinski": "Lipinski",
    "similarity": "相似度", "invalid_smiles": "非法 SMILES",
    "safety": "安全检查", "multi_step": "多步骤", "confusable": "易混淆",
}


def matches_expected(actual, expected, tolerance: float) -> bool:
    """递归比较工具结果：浮点数使用绝对容差，计数、文本和布尔值精确匹配。"""
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(matches_expected(actual[key], value, tolerance)
                        for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(matches_expected(a, b, tolerance) for a, b in zip(actual, expected)))
    # bool 是 int 的子类，必须先判断，避免把 True 当作 1。
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, str):
        return isinstance(actual, str) and actual == expected
    if expected is None:
        return actual is None
    if isinstance(expected, (int, float)):
        if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                or not math.isfinite(actual) or not math.isfinite(expected)):
            return False
        if isinstance(expected, int):
            return actual == expected
        if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
                or not math.isfinite(tolerance) or tolerance < 0):
            return False
        difference = abs(actual - expected)
        # 只修正容差边界的浮点舍入误差，不按结果的大小放宽容差。
        return difference <= tolerance or (
            tolerance > 0 and math.isclose(difference, tolerance, rel_tol=1e-12, abs_tol=0)
        )
    return False


def answer_reports_invalid(answer: str) -> bool:
    """中文规则判定：要求明确断言无效，排除条件句、疑问和否定无效的表达。"""
    if not isinstance(answer, str):
        return False
    answer = re.sub(r"```.*?```", "", answer, flags=re.S).replace("**", "")
    for sentence in re.split(r"[。！？!?;；\n]", answer):
        # 引用题目中的“如果输入无效”不代表模型已判断输入无效。
        sentence = re.sub(r"(?:如果|假如|假设|倘若|若|一旦)[^，,]*", "", sentence)
        for clause in re.split(r"[，,]", sentence):
            if re.search(r"是否|不确定|不能判断|无法判断|无法确认|不能确认|不能说|不能认为|不代表|不意味着|可能|也许", clause):
                continue
            if re.search(r"(?:并非|不是|并不|不属于|没有(?:发现|出现)?|不存在)\s*(?:一个)?\s*(?:无效|非法|不合法|不能解析|无法解析)", clause):
                continue
            if re.search(
                r"无效|非法|不合法|不有效|(?:不是|并非|不属于|不能构成|不构成)\s*(?:一个)?\s*(?:有效|合法)"
                r"|无法解析|不能解析|解析失败|无法被.*解析|不能被.*解析"
                r"|不符合.*SMILES|SMILES.*(?:格式错误|语法错误)", clause, re.I
            ):
                return True
    return False


def _result_matches(tool_name, actual, expected, tolerance) -> bool:
    """v2 安全题只比较警示名称；其余题仍递归比较完整工具结果。"""
    if tool_name == "safety_check" and isinstance(expected, list):
        if not isinstance(actual, dict) or "error" in actual or not isinstance(actual.get("alerts"), list):
            return False
        if not all(isinstance(alert, dict) and isinstance(alert.get("name"), str)
                   for alert in actual["alerts"]):
            return False
        # 名称顺序没有化学意义，但重复名称仍保留，不能用 set 丢掉数量。
        return sorted(alert["name"] for alert in actual["alerts"]) == sorted(expected)
    return matches_expected(actual, expected, tolerance)


def score_response(question_row: dict, response: dict) -> dict:
    """工具选择与结果独立打分；总通过还要求 Agent 已给出最终回答。"""
    answer = response.get("answer", "")
    calls = response.get("tool_calls", [])
    completed = (isinstance(answer, str) and bool(answer.strip())
                 and not answer.startswith("已达到 max_steps=")
                 and answer.strip() != "模型未返回最终回答。")
    reasons = []
    if "expected_tools" in question_row:
        expected_tools = question_row["expected_tools"]
        valid_calls = isinstance(calls, list) and all(isinstance(call, dict) for call in calls)
        actual_names = [call.get("name") for call in calls] if valid_calls else []
        tool_correct = valid_calls and actual_names == expected_tools
        expected_values = question_row["expected_value"]
        value_correct = (valid_calls and len(calls) == len(expected_values)
                         and all("result" in call and _result_matches(name, call["result"], value,
                                                                      question_row["tolerance"])
                                 for name, call, value in zip(expected_tools, calls, expected_values)))
        if not tool_correct:
            reasons.append(f"工具调用序列应为 {expected_tools}，实际为 {actual_names}")
        if not value_correct:
            reasons.append("多工具结果未逐项匹配期望值")
    elif question_row["expected_tool"] is None:
        tool_correct = value_correct = answer_reports_invalid(answer)
        if not tool_correct:
            reasons.append("回答没有明确说明输入无效")
    else:
        first = calls[0] if isinstance(calls, list) and calls and isinstance(calls[0], dict) else {}
        tool_correct = first.get("name") == question_row["expected_tool"]
        value_correct = "result" in first and _result_matches(
            question_row["expected_tool"], first["result"],
            question_row["expected_value"], question_row["tolerance"]
        )
        if not tool_correct:
            reasons.append(f"首个工具应为 {question_row['expected_tool']}，实际为 {first.get('name')}")
        if not value_correct:
            reasons.append("首个工具结果与期望值不匹配")
    if not completed:
        reasons.append("Agent 未取得最终回答")
    return {
        "tool_correct": tool_correct, "value_correct": value_correct,
        "correct": bool(completed and tool_correct and value_correct),
        "reason": "；".join(reasons) if reasons else "通过",
    }


def _safe_error(exc: Exception) -> str:
    """保留便于排错的说明，去掉密钥和常见认证头；不保存 traceback。"""
    message = str(exc)
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    message = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", message)
    message = re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", message)
    return f"{type(exc).__name__}: {message[:500]}"


def evaluate_one(question_row: dict, dry_run: bool = False) -> dict:
    """单题边界：任何调用/评分异常都记作失败，让批量评测能够继续。"""
    start = time.perf_counter()
    record = {**deepcopy(question_row), "mode": "dry-run" if dry_run else "live",
              "answer": "", "tool_calls": [], "steps": 0, "error": None}
    try:
        if dry_run:
            # 用题集答案构造 mock，仅验证读题、评分和写报告的管道。
            name = question_row["expected_tool"]
            if "expected_tools" in question_row:
                mock_calls = [{"name": tool, "arguments": {}, "result": deepcopy(value)}
                              for tool, value in zip(question_row["expected_tools"], question_row["expected_value"])]
            elif name is None:
                mock_calls = []
            else:
                mock_result = deepcopy(question_row["expected_value"])
                if name == "safety_check" and isinstance(mock_result, list):
                    mock_result = {"alerts": [{"name": alert} for alert in mock_result]}
                mock_calls = [{"name": name, "arguments": {}, "result": mock_result}]
            mock_run = Mock(return_value={
                "answer": "输入的 SMILES 无效。" if name is None else "离线模拟：已获得工具结果。",
                "tool_calls": mock_calls,
                "steps": 0,
            })
            response = mock_run(question_row["question"])
        else:
            response = agent.run(question_row["question"])
        if not isinstance(response, dict):
            raise TypeError("Agent 返回值必须是字典")
        for field in ("answer", "tool_calls", "steps"):
            record[field] = response[field]
        record.update(score_response(question_row, response))
        # 序列化错误也属于单题错误，不能让它中断后续问题。
        json.dumps(record, ensure_ascii=False, allow_nan=False)
    except Exception as exc:
        record.update(answer="", tool_calls=[], steps=0, tool_correct=False,
                      value_correct=False, correct=False, reason="单题执行异常", error=_safe_error(exc))
    record["duration_s"] = round(time.perf_counter() - start, 3)
    return record


def _load_questions(path: Path) -> list[dict]:
    rows = []
    seen = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict) or not REQUIRED_FIELDS <= row.keys():
            raise ValueError(f"第 {number} 行缺少题目字段")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in seen:
            raise ValueError(f"第 {number} 行的 id 为空或重复")
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError(f"第 {number} 行的问题为空")
        if row["expected_tool"] is not None and (
            not isinstance(row["expected_tool"], str) or row["expected_tool"] not in TOOLS
        ):
            raise ValueError(f"第 {number} 行有未知的 expected_tool")
        if "category" in row and (not isinstance(row["category"], str) or not row["category"].strip()):
            raise ValueError(f"第 {number} 行的 category 必须是非空字符串")
        if "expected_tools" in row:
            names, values = row["expected_tools"], row["expected_value"]
            if (not isinstance(names, list) or len(names) < 2
                    or not all(isinstance(name, str) and name in TOOLS for name in names)
                    or row["expected_tool"] != names[0]
                    or not isinstance(values, list) or len(values) != len(names)
                    or not all(isinstance(value, dict) and value and "error" not in value for value in values)):
                raise ValueError(f"第 {number} 行的多工具名称和结果必须逐项对应，expected_tool 必须等于首项")
        elif row["expected_tool"] == "safety_check" and isinstance(row["expected_value"], list):
            if not all(isinstance(name, str) and name for name in row["expected_value"]):
                raise ValueError(f"第 {number} 行的安全警示名称必须为非空字符串")
        tolerance = row["tolerance"]
        if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
                or not math.isfinite(tolerance) or tolerance < 0):
            raise ValueError(f"第 {number} 行的 tolerance 必须是非负有限数值")
        json.dumps(row, allow_nan=False)
        rows.append(row)
        seen.add(row["id"])
    if not rows:
        raise ValueError("题集为空")
    return rows


def _metrics(records: list[dict]) -> dict:
    total = len(records)
    counts = {key: sum(record[key] for record in records)
              for key in ("correct", "tool_correct", "value_correct")}
    return {"total": total, **counts, "accuracy": counts["correct"] / total,
            "tool_accuracy": counts["tool_correct"] / total,
            "value_accuracy": counts["value_correct"] / total}


def _write_summary(summary: dict, records: list[dict], path: Path) -> None:
    mode = "离线模拟（mock，不代表模型准确率）" if summary["mode"] == "dry-run" else "真实 API 评测"
    lines = ["# 化学 Agent 评测报告", "", f"- 模式：{mode}",
             f"- 开始时间：{summary['started_at']}；完成时间：{summary['finished_at']}",
             f"- 模型：`{agent.MODEL}`；接口：`{agent.BASE_URL}`",
             f"- 题集 SHA256：`{summary['questions_sha256']}`",
             f"- 题集：`{summary['questions_path']}`",
             f"- 每题详情：[{Path(summary['records_path']).name}]({Path(summary['records_path']).name})",
             f"- 题数：{summary['total']}；异常：{summary['errors']}",
             f"- 总准确率：**{summary['accuracy']:.2%}**（{summary['correct']}/{summary['total']}）",
             f"- 工具选择准确率：**{summary['tool_accuracy']:.2%}**（{summary['tool_correct']}/{summary['total']}）",
             f"- 数值/结果准确率：**{summary['value_accuracy']:.2%}**（{summary['value_correct']}/{summary['total']}）",
             "", "## 评分口径", "",
             "单工具题检查第一条工具调用。含 expected_tools 的多工具题要求完整调用序列严格一致（包括重复工具）；少调、多调、倒序均不通过，并逐项比较对应结果。",
             "安全题的名称列表与实际 alerts 中的 name 比较，忽略排列顺序但保留重复数量；其他结果递归比较，浮点数用绝对容差，整数、字符串、布尔值精确匹配。",
             "非法 SMILES 题允许调用工具，也允许直接拒绝；两项得分均要求最终回答明确说明输入无效。此项使用中文规则匹配，排除条件句及否定表达，仍需结合原文复核边界措辞。",
             "总准确率要求两项都通过且已获得最终回答；单题异常计为失败。所有题目（包括非法输入和异常）均计入分母。工具结果得分不等于核验了最终回答中的每个数值。",
             "离线模拟直接使用期望值构造返回结果，不调用模型或 RDKit，不能作为真实评测成绩。",
             "", "## 按类别准确率", "",
             "| 类别 | 题数 | 总通过 | 总准确率 | 工具选择准确率 | 数值/结果准确率 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for category, values in summary["categories"].items():
        lines.append(f"| {CATEGORY_NAMES.get(category, category)} (`{category}`) | {values['total']} | "
                     f"{values['correct']} | {values['accuracy']:.2%} | {values['tool_accuracy']:.2%} | "
                     f"{values['value_accuracy']:.2%} |")
    lines.extend(["", "## 错题列表", ""])
    failures = [record for record in records if not record["correct"]]
    if not failures:
        lines.append("无。")
    for record in failures:
        question = record["question"].replace("\n", " ")
        lines.extend([f"- **{record['id']}**：{question}",
                      f"  - 原因：{record['reason']}"])
        if record["error"]:
            lines.append(f"  - 异常：{record['error']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(questions_path=EVAL_DIR / "questions.jsonl",
                   output_dir=None, limit=None, dry_run=False) -> dict:
    """顺序运行并即时保存每题详情，返回汇总数字及输出文件路径。"""
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit 必须是大于 0 的整数")
    questions_path = Path(questions_path)
    rows = _load_questions(questions_path)
    if limit is not None:
        rows = rows[:limit]
    if output_dir is None:
        output_dir = EVAL_DIR / "results"
        if questions_path.resolve() != (EVAL_DIR / "questions.jsonl").resolve():
            output_dir /= questions_path.stem
    output_dir = Path(output_dir)
    if dry_run:
        output_dir /= "dry_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone(timedelta(hours=8)))
    records_path = output_dir / (started.strftime("%Y%m%d_%H%M%S_%f") + ".jsonl")
    records = []
    with records_path.open("x", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows, 1):
            record = evaluate_one(row, dry_run=dry_run)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            records.append(record)
            state = "PASS" if record["correct"] else "FAIL"
            print(f"[{index}/{len(rows)}] {row['id']} {state} ({record['duration_s']:.2f}s)", flush=True)
    categories = {}
    for record in records:
        category = record.get("category") or record["expected_tool"] or "invalid_smiles"
        categories.setdefault(category, []).append(record)
    summary = {**_metrics(records), "mode": "dry-run" if dry_run else "live",
               "categories": {key: _metrics(values) for key, values in categories.items()},
               "errors": sum(record["error"] is not None for record in records),
               "started_at": started.isoformat(timespec="seconds"),
               "finished_at": datetime.now(started.tzinfo).isoformat(timespec="seconds"),
               "questions_sha256": hashlib.sha256(questions_path.read_bytes()).hexdigest(),
               "questions_path": str(questions_path.resolve()),
               "records_path": str(records_path), "summary_path": str(output_dir / "summary.md")}
    _write_summary(summary, records, Path(summary["summary_path"]))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="只运行前 N 题，N 必须大于 0")
    parser.add_argument("--dry-run", action="store_true", help="使用 mock，不联网；成绩不代表模型能力")
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.jsonl", help="题集路径，默认仍为 v1")
    parser.add_argument("--output-dir", type=Path, help="报告目录；其他题集默认存入 results/<题集名>/")
    args = parser.parse_args()
    if not args.dry_run and not os.environ.get("LLM_API_KEY", "").strip():
        print("错误：缺少环境变量 LLM_API_KEY，请设置密钥或使用 --dry-run。", file=sys.stderr)
        return 1
    try:
        summary = run_evaluation(questions_path=args.questions, output_dir=args.output_dir,
                                 limit=args.limit, dry_run=args.dry_run)
    except (OSError, ValueError, TypeError) as exc:
        print(f"评测无法启动或报告无法保存：{_safe_error(exc)}", file=sys.stderr)
        return 1
    print(f"完成 {summary['total']} 题；总准确率 {summary['accuracy']:.2%}；"
          f"工具选择准确率 {summary['tool_accuracy']:.2%}")
    print(f"报告：{summary['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
