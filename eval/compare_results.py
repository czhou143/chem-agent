"""从两份已保存的真实结果生成对比表；不调用 Agent，不改写原始结果。"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run_eval import CATEGORY_NAMES, TOKEN_FIELDS, score_response


def load_records(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("结果必须非空且 id 唯一")
    for row in rows:
        if row["mode"] != "live":
            raise ValueError("前后对比必须使用真实评测结果，不能使用 dry-run")
        if not row["usage_complete"]:
            raise ValueError(f"{row['id']} 用量不完整，不能生成完整 token 对比")
        if not all(type(row[field]) is int and row[field] >= 0 for field in TOKEN_FIELDS):
            raise ValueError("token 计数必须为非负整数")
        if row["total_tokens"] != row["prompt_tokens"] + row["completion_tokens"]:
            raise ValueError("token 总数不一致")
        if row["error"] is None:
            rescored = score_response(row, row)
            if any(row[key] != rescored[key] for key in ("correct", "tool_correct", "value_correct")):
                raise ValueError(f"{row['id']} 保存的分数与当前评分规则不一致")
    return rows


def metrics(rows):
    n = len(rows)
    return {"n": n, "correct": sum(row["correct"] for row in rows),
            "tool_correct": sum(row["tool_correct"] for row in rows),
            **{field: sum(row[field] for row in rows) for field in TOKEN_FIELDS}}


def compare_results(before_path, after_path):
    before_path, after_path = Path(before_path), Path(after_path)
    before, after = load_records(before_path), load_records(after_path)
    question_fields = ("id", "question", "expected_tool", "expected_tools", "expected_value", "tolerance", "category")
    if len(before) != len(after) or any(
        any(a.get(key) != b.get(key) for key in question_fields) for a, b in zip(before, after)
    ):
        raise ValueError("两轮必须使用内容、顺序相同的题集")
    fixed_config = ("model", "base_url", "max_output_tokens", "thinking", "tool_schemas_sha256", "questions_sha256")
    configs = [row["run_config"] for row in before + after]
    if any(any(config[key] != configs[0][key] for key in fixed_config) for config in configs):
        raise ValueError("两轮模型、请求设置、工具描述或题集不一致，不能作为单处 prompt 迭代对比")

    a, b = metrics(before), metrics(after)
    def accuracy(values, field):
        return f"{values[field] / values['n']:.2%}（{values[field]}/{values['n']}）"

    lines = ["# v2 改进前后对比", "", "所有计数均从下列两份真实 JSONL 结果重新计算；不使用试跑数据或估算值。", ""]
    for label, path, records in (("改进前", before_path, before), ("改进后", after_path, after)):
        lines.extend([f"- {label}：`{path.as_posix()}`；{len(records)} 行；SHA256 `{hashlib.sha256(path.read_bytes()).hexdigest()}`",
                      f"  - system prompt SHA256：`{records[0]['run_config']['system_prompt_sha256']}`"])
    lines.extend(["", "| 指标 | 改进前 | 改进后 |", "| --- | ---: | ---: |",
                  f"| 总准确率 | {accuracy(a, 'correct')} | {accuracy(b, 'correct')} |",
                  f"| 工具选择准确率 | {accuracy(a, 'tool_correct')} | {accuracy(b, 'tool_correct')} |",
                  *[f"| {field} | {a[field]} | {b[field]} |" for field in TOKEN_FIELDS],
                  "", f"两轮合计 token：**{a['total_tokens'] + b['total_tokens']}**（不含 5 题试跑）。", "",
                  "| 类别 | 题数 | 改进前总准确率 | 改进后总准确率 | 改进前工具准确率 | 改进后工具准确率 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    groups = []
    for records in (before, after):
        grouped = defaultdict(list)
        for row in records:
            grouped[row.get("category") or row["expected_tool"] or "invalid_smiles"].append(row)
        groups.append(grouped)
    for category in groups[0]:
        first, second = metrics(groups[0][category]), metrics(groups[1][category])
        lines.append(f"| {CATEGORY_NAMES.get(category, category)} | {first['n']} | "
                     f"{accuracy(first, 'correct')} | {accuracy(second, 'correct')} | "
                     f"{accuracy(first, 'tool_correct')} | {accuracy(second, 'tool_correct')} |")
    lines.extend(["", "## 错题定位", "", "| 轮次 | id | 原始文件与行号 | 评分器记录 |", "| --- | --- | --- | --- |"])
    for label, path, records in (("改进前", before_path, before), ("改进后", after_path, after)):
        failed = [(number, row) for number, row in enumerate(records, 1) if not row["correct"]]
        if not failed:
            lines.append(f"| {label} | 无 | — | 全部通过 |")
        for number, row in failed:
            reason = row["reason"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {label} | {row['id']} | `{path.as_posix()}:{number}` | {reason} |")
    delta = (b["correct"] / b["n"] - a["correct"] / a["n"]) * 100
    lines.extend(["", f"总准确率变化：{delta:+.2f} 个百分点。", "",
                  "同一题集每种设置只运行一次；结果受模型采样影响，不能据此认定因果效果或泛化能力。评分检查工具调用和结果，不逐句核验自然语言回答。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = compare_results(args.before, args.after)
    # 拒绝覆盖任何现有文件，尤其不能误覆盖原始结果。
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    print(f"已从两份真实结果生成对比：{args.output}")


if __name__ == "__main__":
    main()
