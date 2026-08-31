from anvil.llm.parse import (
    complete_without_tools,
    parse_assistant_choice,
    parse_tool_call,
    parse_tool_calls,
    parse_usage,
)


def test_valid_tool_call_round_trip() -> None:
    call = parse_tool_call("c1", "read_file", '{"path": "a.py"}')
    assert call.parse_error is False
    assert call.arguments == {"path": "a.py"}
    assert call.id == "c1"


def test_empty_object_is_not_a_parse_error() -> None:
    call = parse_tool_call("c1", "list_dir", "{}")
    assert call.parse_error is False
    assert call.arguments == {}


def test_malformed_json_sets_parse_error() -> None:
    call = parse_tool_call("c1", "list_dir", "{")
    assert call.parse_error is True
    assert call.arguments == {}


def test_json_array_arguments_are_a_parse_error() -> None:
    call = parse_tool_call("c1", "list_dir", "[1, 2]")
    assert call.parse_error is True


def test_dict_arguments_are_accepted() -> None:
    call = parse_tool_call("c1", "read_file", {"path": "a.py"})
    assert call.parse_error is False
    assert call.arguments["path"] == "a.py"


def test_empty_function_name_is_skipped() -> None:
    calls = parse_tool_calls(
        [
            {"id": "x", "function": {"name": "", "arguments": "{}"}},
            {"id": "y", "function": {"name": "list_dir", "arguments": "{}"}},
            {"function": {"arguments": "{}"}},
            "not-an-object",
        ]
    )
    assert len(calls) == 1
    assert calls[0].id == "y"
    assert calls[0].name == "list_dir"


def test_missing_id_is_filled() -> None:
    calls = parse_tool_calls([{"function": {"name": "list_dir", "arguments": "{}"}}])
    assert calls[0].id == "call_0"


def test_usage_falls_back_to_sum() -> None:
    usage = parse_usage({"prompt_tokens": 10, "completion_tokens": 4})
    assert usage.total_tokens == 14
    assert parse_usage(None).total_tokens == 0
    assert parse_usage({"prompt_tokens": "nope"}).prompt_tokens == 0
    detailed = parse_usage(
        {
            "prompt_tokens": 8,
            "completion_tokens": 20,
            "total_tokens": 28,
            "completion_tokens_details": {"reasoning_tokens": 12},
        }
    )
    assert detailed.reasoning_tokens == 12


def test_choice_extracts_finish_reason_and_tools() -> None:
    response = parse_assistant_choice(
        {
            "finish_reason": "tool_calls",
            "message": {
                "content": "",
                "reasoning_content": "look around",
                "tool_calls": [
                    {
                        "id": "1",
                        "function": {"name": "list_dir", "arguments": "{}"},
                    }
                ],
            },
        },
        {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
    )
    assert response.finish_reason == "tool_calls"
    assert response.reasoning_content == "look around"
    assert response.tool_calls[0].name == "list_dir"
    assert response.usage.prompt_tokens == 8


def test_finish_reason_is_ignored_when_tools_are_absent() -> None:
    assert complete_without_tools("tool_calls") == "completed"
    assert complete_without_tools("stop") == "completed"
    assert complete_without_tools(None) == "completed"
    assert complete_without_tools("length") == "length"
    assert complete_without_tools("max_tokens") == "length"
