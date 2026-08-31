from anvil.llm.parse import parse_assistant_choice, parse_tool_call, parse_tool_calls, parse_usage
from anvil.llm.types import LLMResponse, Message, ToolCall, Usage

__all__ = [
    "DeepSeekClient",
    "LLMError",
    "LLMResponse",
    "Message",
    "ToolCall",
    "Usage",
    "parse_assistant_choice",
    "parse_tool_call",
    "parse_tool_calls",
    "parse_usage",
]


def __getattr__(name: str):
    if name in {"DeepSeekClient", "LLMError"}:
        from anvil.llm.openai_compat import DeepSeekClient, LLMError

        return DeepSeekClient if name == "DeepSeekClient" else LLMError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
