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


def test_missing_reasoning_is_serialized_as_empty_string() -> None:
    message = Message(role="assistant", content="done")
    payload = message.to_openai(include_reasoning=True)
    assert payload["reasoning_content"] == ""
