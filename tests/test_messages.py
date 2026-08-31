from anvil.llm.types import Message, ToolCall


def test_assistant_payload_keeps_reasoning_when_tools_are_used() -> None:
    message = Message(
        role="assistant",
        content="",
        reasoning_content="I should read the file first.",
        tool_calls=[
            ToolCall(id="call_1", name="read_file", arguments={"path": "a.py"}, arguments_raw='{"path":"a.py"}')
        ],
    )
    payload = message.to_openai(include_reasoning=True)
    assert payload["reasoning_content"] == "I should read the file first."
    assert payload["tool_calls"][0]["function"]["name"] == "read_file"


def test_empty_object_arguments_are_not_parse_errors() -> None:
    from anvil.llm.openai_compat import _tool_call_from_parts

    call = _tool_call_from_parts("c1", "list_dir", "{}")
    assert call.parse_error is False
    assert call.arguments == {}


def test_malformed_arguments_set_parse_error() -> None:
    from anvil.llm.openai_compat import _tool_call_from_parts

    call = _tool_call_from_parts("c1", "list_dir", "{")
    assert call.parse_error is True


def test_from_record_preserves_parse_error() -> None:
    message = Message.from_record(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "list_dir", "arguments": "{"}}
            ],
        }
    )
    assert message is not None
    assert message.tool_calls
    assert message.tool_calls[0].parse_error is True
    assert message.tool_calls[0].name == "list_dir"


def test_missing_reasoning_is_serialized_as_empty_string() -> None:
    message = Message(role="assistant", content="done")
    payload = message.to_openai(include_reasoning=True)
    assert payload["reasoning_content"] == ""
