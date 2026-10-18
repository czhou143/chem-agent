"""离线测试 Agent 循环：替换模型客户端，RDKit 工具仍在本地真实计算。"""

from copy import deepcopy
import json
from unittest.mock import MagicMock

from openai import OpenAIError
from openai.types.chat import ChatCompletion
import pytest

from chem_agent import agent, tools


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def model_response(content=None, tool_calls=None):
    """构造与 SDK 相同的响应对象，不发送网络请求。"""
    return ChatCompletion(
        id="mock-completion",
        created=0,
        model="deepseek-chat",
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

    assert result == {
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
        assert request["model"] == "deepseek-chat"
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
