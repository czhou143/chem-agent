"""离线测试 Agent 循环：替换模型客户端，RDKit 工具仍在本地真实计算。"""

from copy import deepcopy
import json
from unittest.mock import MagicMock

from openai import OpenAIError
from openai.types.chat import ChatCompletion
import pytest

from chem_agent import agent, tools


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def model_response(content=None, tool_calls=None, usage=None):
    """构造与 SDK 相同的响应对象，不发送网络请求。"""
    return ChatCompletion(
        id="mock-completion",
        created=0,
        model=agent.MODEL,
        usage=usage,
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                },
            }
        ],
    )


def tool_call(name, arguments, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


@pytest.fixture
def mock_model(monkeypatch):
    """预先安排模型回答，并保存每次请求当时的完整消息。"""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    client = MagicMock()
    constructor = MagicMock(return_value=client)
    monkeypatch.setattr(agent, "OpenAI", constructor)
    requests = []

    def configure(*responses):
        pending = iter(responses)

        def create(**kwargs):
            # 主循环会继续修改 messages；必须复制才能验证每一轮的真实内容。
            requests.append(deepcopy(kwargs))
            response = next(pending)
            if isinstance(response, Exception):
                raise response
            return response

        client.chat.completions.create.side_effect = create
        return constructor, client, requests

    return configure


def test_tool_result_is_sent_back_before_final_answer(mock_model):
    arguments = {"smiles": ASPIRIN}
    answer = "阿司匹林的分子量约为 180.16。"
    constructor, client, requests = mock_model(
        model_response(
            tool_calls=[tool_call("mol_weight", json.dumps(arguments), "call_weight")]
        ),
        model_response(answer),
    )

    result = agent.run("阿司匹林的分子量是多少？")

    assert {key: result[key] for key in ("answer", "tool_calls", "steps")} == {
        "answer": answer,
        "tool_calls": [
            {
                "name": "mol_weight",
                "arguments": arguments,
                "result": {"mol_weight": 180.16},
            }
        ],
        "steps": 2,
    }
    assert constructor.call_args.kwargs["api_key"] == "test-key"
    assert constructor.call_args.kwargs["base_url"] == "https://api.deepseek.com"
    assert client.chat.completions.create.call_count == 2
    for request in requests:
        assert request["model"] == "deepseek-flash"
        assert request["max_tokens"] == 2048
        assert request["extra_body"] == {"thinking": {"type": "disabled"}}
        assert request["tools"] == tools.TOOL_SCHEMAS

    first_messages = requests[0]["messages"]
    assert first_messages[0]["role"] == "system"
    assert first_messages[-1] == {
        "role": "user", "content": "阿司匹林的分子量是多少？"
    }
    assert not any(message["role"] == "tool" for message in first_messages)

    second_messages = requests[1]["messages"]
    assistant_message, tool_message = second_messages[-2:]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["tool_calls"][0]["id"] == "call_weight"
    assert assistant_message["tool_calls"][0]["function"]["name"] == "mol_weight"
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_weight"
    assert json.loads(tool_message["content"]) == {"mol_weight": 180.16}


def test_safety_result_and_instructions_are_sent_to_model(mock_model):
    answer = "命中酰卤相关警示。" + tools.SAFETY_DISCLAIMER
    _, _, requests = mock_model(
        model_response(tool_calls=[tool_call("safety_check", '{"smiles": "CC(=O)Cl"}', "call_safety")]),
        model_response(answer),
    )
    result = agent.run("乙酰氯有危险吗？SMILES：CC(=O)Cl")
    assert result["answer"] == answer
    assert result["tool_calls"][0]["name"] == "safety_check"
    prompt = requests[0]["messages"][0]["content"]
    assert "直接拒绝" in prompt and "disclaimer" in prompt
    message = requests[1]["messages"][-1]
    assert message["role"] == "tool" and message["tool_call_id"] == "call_safety"
    assert json.loads(message["content"]) == tools.safety_check("CC(=O)Cl")


def test_max_steps_stops_without_an_extra_model_request(mock_model):
    _, client, requests = mock_model(
        model_response(tool_calls=[tool_call("formula", '{"smiles": "CCO"}', "call_1")]),
        model_response(tool_calls=[tool_call("formula", '{"smiles": "O"}', "call_2")]),
    )

    result = agent.run("继续检查分子式", max_steps=2)

    assert result["steps"] == 2
    assert client.chat.completions.create.call_count == len(requests) == 2
    assert [call["result"] for call in result["tool_calls"]] == [
        {"formula": "C2H6O"}, {"formula": "H2O"}
    ]
    assert isinstance(result["answer"], str) and result["answer"].strip()
    assert any(word in result["answer"] for word in ("上限", "最大", "未完成"))


def test_multiple_tools_in_one_round_keep_matching_ids(mock_model):
    _, _, requests = mock_model(
        model_response(
            tool_calls=[
                tool_call("mol_weight", json.dumps({"smiles": ASPIRIN}), "call_weight"),
                tool_call("formula", json.dumps({"smiles": ASPIRIN}), "call_formula"),
            ]
        ),
        model_response("分子量为 180.16，分子式为 C9H8O4。"),
    )

    result = agent.run("阿司匹林的分子量和分子式是多少？")

    assert result["steps"] == 2  # steps 是模型请求次数，不是执行工具的次数。
    assert [call["name"] for call in result["tool_calls"]] == ["mol_weight", "formula"]
    tool_messages = [m for m in requests[1]["messages"] if m["role"] == "tool"]
    assert [(m["tool_call_id"], json.loads(m["content"])) for m in tool_messages] == [
        ("call_weight", {"mol_weight": 180.16}),
        ("call_formula", {"formula": "C9H8O4"}),
    ]


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("mol_weight", "not-json"),
        ("mol_weight", '[]'),
        ("mol_weight", 'null'),
        ("unknown_tool", '{"smiles": "CCO"}'),
        ("mol_weight", '{}'),
        ("mol_weight", '{"smiles": "CCO", "unexpected": 1}'),
        ("mol_weight", '{"smiles": "not-a-smiles"}'),
    ],
)
def test_bad_tool_calls_return_error_to_model(mock_model, name, arguments):
    _, _, requests = mock_model(
        model_response(tool_calls=[tool_call(name, arguments)]),
        model_response("工具输入有误，请提供有效输入。"),
    )

    result = agent.run("检查输入")

    assert result["steps"] == 2
    assert len(result["tool_calls"]) == 1
    record = result["tool_calls"][0]
    assert record["name"] == name
    assert "arguments" in record
    assert isinstance(record["result"].get("error"), str)
    assert record["result"]["error"].strip()
    tool_message = requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"
    assert json.loads(tool_message["content"]) == record["result"]


@pytest.mark.parametrize("key", [None, "", "   "])
def test_missing_api_key_explains_which_variable_to_set(mock_model, monkeypatch, key):
    constructor, client, _ = mock_model()
    if key is None:
        monkeypatch.delenv("LLM_API_KEY", raising=False)
    else:
        monkeypatch.setenv("LLM_API_KEY", key)

    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        agent.run("你好")

    constructor.assert_not_called()
    client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("question", [None, "", "   "])
def test_empty_question_is_rejected_before_model_request(mock_model, question):
    _, client, _ = mock_model()
    with pytest.raises(ValueError):
        agent.run(question)
    client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("max_steps", [0, -1, 1.5, True])
def test_max_steps_must_be_a_positive_integer(mock_model, max_steps):
    _, client, _ = mock_model()
    with pytest.raises(ValueError):
        agent.run("你好", max_steps=max_steps)
    client.chat.completions.create.assert_not_called()


def test_sdk_failure_gives_clear_error_without_exposing_api_key(mock_model):
    _, client, _ = mock_model(OpenAIError("mock transport failure: test-key"))

    with pytest.raises(RuntimeError) as error:
        agent.run("你好")

    assert str(error.value).strip()
    assert "test-key" not in str(error.value)
    assert client.chat.completions.create.call_count == 1


def token_usage(prompt, completion, hit):
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion, "prompt_cache_hit_tokens": hit,
            "prompt_cache_miss_tokens": prompt - hit}


def test_usage_sums_all_model_requests_and_keeps_raw_usage(mock_model):
    first, second = token_usage(100, 10, 20), token_usage(200, 30, 80)
    mock_model(model_response(tool_calls=[tool_call("formula", '{"smiles":"CCO"}')], usage=first),
               model_response("乙醇分子式为 C2H6O。", usage=second))
    usage = agent.run("乙醇的分子式？")["usage"]
    assert usage["prompt_tokens"] == 300
    assert usage["completion_tokens"] == 40
    assert usage["total_tokens"] == 340
    assert usage["prompt_cache_hit_tokens"] == 100
    assert usage["prompt_cache_miss_tokens"] == 200
    assert usage["complete"] is usage["cache_complete"] is True
    assert len(usage["requests"]) == 2
    assert usage["requests"][0]["usage"]["prompt_tokens"] == 100


def test_usage_at_max_steps_is_not_lost(mock_model):
    mock_model(model_response(tool_calls=[tool_call("formula", '{"smiles":"CCO"}')],
                               usage=token_usage(100, 10, 0)))
    result = agent.run("分子式？", max_steps=1)
    assert result["usage"]["total_tokens"] == 110
    assert result["usage"]["complete"] is True


def test_partial_usage_survives_later_api_failure(mock_model):
    mock_model(model_response(tool_calls=[tool_call("formula", '{"smiles":"CCO"}')],
                               usage=token_usage(100, 10, 0)), OpenAIError("test-key"))
    with pytest.raises(agent.AgentRunError) as error:
        agent.run("分子式？")
    assert error.value.usage["total_tokens"] == 110
    assert error.value.usage["complete"] is False
    assert len(error.value.usage["requests"]) == 2
    assert error.value.tool_calls[0]["result"] == {"formula": "C2H6O"}
    assert "test-key" not in str(error.value)


def test_missing_usage_is_explicitly_incomplete(mock_model):
    mock_model(model_response("你好"))
    assert agent.run("你好")["usage"]["complete"] is False


def test_budget_refusal_stops_before_network_request(mock_model):
    _, client, _ = mock_model()
    budget = MagicMock()
    budget.reserve_request.side_effect = agent.BudgetExceeded("预算不足")
    with pytest.raises(agent.AgentRunError) as error:
        agent.run("你好", budget=budget)
    assert error.value.budget_exceeded is True
    client.chat.completions.create.assert_not_called()
