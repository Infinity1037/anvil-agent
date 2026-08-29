from anvil.agent.context import PLACEHOLDER, ContextManager, clip_text, estimate_tokens
from anvil.llm.types import Message


def test_clip_text_keeps_head_and_tail() -> None:
    text = "A" * 5000 + "MIDDLE" + "B" * 5000
    clipped = clip_text(text, limit=8000)
    assert clipped.startswith("A" * 100)
    assert clipped.endswith("B" * 100)
    assert "truncated" in clipped
    assert "MIDDLE" not in clipped


def test_compact_shrinks_old_tool_results() -> None:
    messages = [Message(role="user", content="fix tests")]
    for i in range(10):
        messages.append(Message(role="assistant", content=f"step {i}"))
        messages.append(Message(role="tool", content="x" * 4000, tool_call_id=f"c{i}"))
    manager = ContextManager(budget=2000)
    compacted = manager.prepare(messages)
    assert estimate_tokens(compacted) < estimate_tokens(messages)
    huge = [m for m in compacted if m.role == "tool" and m.content and len(m.content) >= 4000]
    assert len(huge) <= 1
    assert any(m.role == "tool" and m.content == PLACEHOLDER for m in compacted) or any(
        m.content and "compacted" in m.content for m in compacted
    )
