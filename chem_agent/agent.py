"""手写 function-calling 循环：模型选择工具，Python 执行 RDKit 计算。

运行：python -m chem_agent.agent "阿司匹林的分子量是多少"
密钥只从环境变量 LLM_API_KEY 读取；导入本模块不会发起网络请求。
"""

import argparse
import json
import os
import sys

from openai import APIStatusError, OpenAI, OpenAIError

from . import tools


BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"
SYSTEM_PROMPT = (
    "你是化学助手，用中文回答。涉及分子性质必须调用提供的工具，"
    "不要自己编造分子量、LogP、分子式、TPSA、氢键数、Lipinski 结果或相似度数值。"
    "根据工具实际返回的结果作答；工具报错时说明原因或修正参数后重试。"
    "如果不能确定用户指的是哪种分子，请先澄清，不要猜测结构。"
)


def _execute_tool(name: str, raw_arguments: str) -> tuple[object, dict]:
    """解析模型给出的 JSON 参数，只执行 TOOLS 中登记过的函数。"""
    try:
        arguments = json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError):
        return raw_arguments, {"error": "工具参数不是有效的 JSON，请重新提供参数。"}

    if not isinstance(arguments, dict):
        return arguments, {"error": "工具参数必须是 JSON 对象，例如 {\"smiles\": \"CCO\"}。"}

    function = tools.TOOLS.get(name)
    if function is None:
        return arguments, {"error": f"未知工具：{name}，请使用提供的工具列表。"}

    try:
        # **arguments 把字典展开成关键字参数，例如 mol_weight(smiles="CCO")。
        result = function(**arguments)
    except TypeError:
        result = {"error": f"工具 {name} 的参数不匹配，请检查参数名和必填参数。"}
    except Exception as exc:
        # 工具是循环的边界：单次工具失败也要回传结果，给模型纠错机会。
        result = {"error": f"工具 {name} 执行失败（{type(exc).__name__}）。"}
    return arguments, result


def run(question: str, max_steps: int = 5) -> dict:
    """回答问题，返回 answer、tool_calls 和 steps。

    steps 表示模型请求次数，一轮可以调用多个工具。max_steps 必须为正整数；
    达到上限时保留工具记录并明确说明未取得最终回答，不追加额外请求。
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question 必须是非空问题字符串。")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps 必须是大于 0 的整数。")

    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "缺少环境变量 LLM_API_KEY，请先在当前终端设置 DeepSeek API 密钥再运行；"
            "如果刚设置了 Windows 用户环境变量，请重新打开终端。"
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]
    history = []
    # 设置超时并关闭 SDK 隐式重试，让请求次数与 max_steps 的含义一致。
    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=30.0, max_retries=0)
    try:
        for step in range(1, max_steps + 1):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools.TOOL_SCHEMAS,
                stream=False,
            )
            if not response.choices:
                raise RuntimeError("DeepSeek 返回了空响应，没有可读取的回答。")
            message = response.choices[0].message

            if not message.tool_calls:
                return {
                    "answer": message.content or "模型未返回最终回答。",
                    "tool_calls": history,
                    "steps": step,
                }

            # 必须先保存 assistant 的工具请求，再追加每条匹配的 tool 结果。
            messages.append(message.model_dump(exclude_none=True))
            for call in message.tool_calls:
                name = call.function.name
                arguments, result = _execute_tool(name, call.function.arguments)
                history.append({"name": name, "arguments": arguments, "result": result})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        return {
            "answer": f"已达到 max_steps={max_steps} 的请求上限，尚未获得最终回答；"
            "已保留执行过的工具结果。",
            "tool_calls": history,
            "steps": max_steps,
        }
    except APIStatusError as exc:
        raise RuntimeError(
            f"DeepSeek 请求失败（HTTP {exc.status_code}），请检查 API 密钥、余额、"
            f"模型名称 {MODEL} 和服务状态。"
        ) from None
    except OpenAIError as exc:
        # 不原样打印 SDK 错误，以免响应或请求信息中意外带出凭据。
        raise RuntimeError(
            f"DeepSeek 请求失败（{type(exc).__name__}），请检查网络和服务状态。"
        ) from None
    finally:
        client.close()


def main() -> int:
    """命令行入口：输出完整 JSON，方便同时检查答案与工具调用记录。"""
    parser = argparse.ArgumentParser(description="使用 DeepSeek 和本地 RDKit 工具回答化学问题")
    parser.add_argument("question", help="要询问的化学问题；含空格时请加引号")
    parser.add_argument("--max-steps", type=int, default=5, help="最多请求模型的次数，默认 5")
    args = parser.parse_args()
    try:
        result = run(args.question, max_steps=args.max_steps)
    except (ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
