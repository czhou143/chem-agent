"""运行三类化学问题，按顺序展示 Agent 实际记录的工具调用和最终回答。

从项目根目录运行：python examples/demo.py
需要环境变量 LLM_API_KEY；导入本模块不会发起模型请求。
"""

import json
import os
from pathlib import Path
import sys
from textwrap import indent

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chem_agent import agent


QUESTIONS = (
    ("分子量", "阿司匹林的 SMILES 是 CC(=O)Oc1ccccc1C(=O)O，请计算它的分子量。"),
    ("分子式", "咖啡因的 SMILES 是 Cn1c(=O)c2c(ncn2C)n(C)c1=O，请计算它的分子式。"),
    ("相似度", "请计算乙醇（CCO）与 1-丙醇（CCCO）的 Morgan 指纹 Tanimoto 相似度。"),
)


def print_result(result: dict) -> None:
    """直接展示实际返回值，不重新计算或自行生成答案。"""
    print("\n  工具调用记录")
    if not result["tool_calls"]:
        print("    本题未调用工具。")
    for index, call in enumerate(result["tool_calls"], 1):
        print(f"\n    [{index}] {call['name']}")
        for label, field in (("参数", "arguments"), ("结果", "result")):
            formatted = json.dumps(call[field], ensure_ascii=False, indent=2)
            print(f"      {label}：")
            print(indent(formatted, "        "))
    print("\n  最终回答")
    print(indent(result["answer"], "    "))
    print(f"\n  模型请求次数：{result['steps']} | 工具调用次数：{len(result['tool_calls'])}")


def main() -> int:
    if not os.environ.get("LLM_API_KEY", "").strip():
        print("缺少环境变量 LLM_API_KEY，请先在当前终端设置 DeepSeek API 密钥。", file=sys.stderr)
        return 1

    print("Chem Agent 演示 | DeepSeek + RDKit", flush=True)
    failed = False
    for index, (category, question) in enumerate(QUESTIONS, 1):
        print("\n" + "=" * 64)
        print(f"  问题 {index}/{len(QUESTIONS)} · {category}")
        print("=" * 64)
        print(f"  {question}")
        print("  正在请求模型并执行工具……", flush=True)
        try:
            result = agent.run(question)
        except (ValueError, RuntimeError) as exc:
            # Agent 对 API 错误已作脱敏；单题失败仍展示后续问题。
            print(f"\n  本题失败：{exc}", flush=True)
            failed = True
            continue
        print_result(result)
        if (not result["answer"].strip() or result["answer"].startswith("已达到 max_steps=")
                or result["answer"] == "模型未返回最终回答。"):
            failed = True
    print("\n" + "=" * 64)
    print("演示结束；有问题未完成，请检查上方说明。" if failed else "三个问题均已获得最终回答。")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
