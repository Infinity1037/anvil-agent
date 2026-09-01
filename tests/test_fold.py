from anvil.agent.context import ContextSnapshot
from anvil.agent.loop import CompactResult
from anvil.llm.types import Usage
from anvil.tui.chrome import footer_text, status_plain
from anvil.tui.fold import (
    THINKING_PREVIEW,
    TOOL_PREVIEW,
    preview_limit,
    visible_body,
    wrap_to_width,
)
from anvil.tui.app import STREAM_FLUSH_S, coalesce_events, stream_flush_delay
from anvil.tui.markdown import (
    render_assistant,
    render_markdown,
    render_reply,
    with_message_prefix,
)
from anvil.tui.widgets import _thinking_text, render_user
from anvil.events import AgentEvent
from anvil.ui.format import compact_result_text, context_badge, context_report, short_tokens


def test_context_display_derives_live_ratio_from_token_counts() -> None:
    snapshot = ContextSnapshot(
        estimated_tokens=42_000,
        budget=100_000,
        history_messages=30,
        view_messages=12,
        covered_count=18,
        calibrated=True,
    )
    assert context_badge(snapshot) == "ctx ≈42% (42k/100k)"
    report = context_report(snapshot, Usage(prompt_tokens=128_400, completion_tokens=9_700))
    assert "remaining ≈58k" in report
    assert "30 full / 12 active" in report
    assert "checkpoint covers 17 historical" in report
    assert "128.4k input / 9.7k output" in report
    assert short_tokens(1_250_000) == "1.2m"


def test_context_display_separates_context_window_from_agent_budget() -> None:
    snapshot = ContextSnapshot(
        estimated_tokens=42_000,
        budget=160_000,
        history_messages=30,
        view_messages=12,
        calibrated=True,
        context_window=200_000,
        output_limit=32_000,
        compaction_threshold=160_000,
    )
    assert context_badge(snapshot) == "ctx ≈21% (42k/200k)"
    report = context_report(snapshot, Usage())
    assert "context window 200k tokens" in report
    assert "agent view budget 160k tokens" in report
    assert "per-turn output cap 32k tokens" in report
    assert "auto compact near 160k tokens" in report
    assert "remaining ≈118k" in report


def test_compaction_result_text_is_observable_without_exposing_summary() -> None:
    text = compact_result_text(CompactResult("compacted", 62_100, 13_400, 34))
    assert "≈62.1k → 13.4k" in text
    assert "33" in text
    assert "完整历史" in text


def test_live_thinking_shows_tail_under_spinner() -> None:
    full = "one\ntwo\nthree\nfour\nfive"
    shown = visible_body("thinking", full, expanded=False, live=True)
    assert shown.startswith("thinking…")
    assert shown.endswith("four\nfive")
    assert "one" not in shown
    assert "three" not in shown
    assert "Ctrl+O" not in shown


def test_finalized_thinking_shows_head_and_hint() -> None:
    full = "one\ntwo\nthree\nfour"
    shown = visible_body("thinking", full, expanded=False, live=False)
    assert shown.startswith("one\ntwo")
    assert "Ctrl+O to expand" in shown
    assert "four" not in shown
    rest = len(full.splitlines()) - THINKING_PREVIEW
    assert f"{rest} more lines" in shown


def test_expanded_thinking_is_the_same_card_full_text() -> None:
    full = "one\ntwo\nthree\nfour"
    shown = visible_body("thinking", full, expanded=True, live=False)
    assert shown == full
    assert "Ctrl+O" not in shown


def test_short_thinking_is_never_truncated() -> None:
    full = "one\ntwo"
    assert visible_body("thinking", full, expanded=False, live=False) == full
    live = visible_body("thinking", full, expanded=False, live=True)
    assert live == "thinking…\none\ntwo"


def test_empty_live_thinking_is_just_the_label() -> None:
    assert visible_body("thinking", "", expanded=False, live=True) == "thinking…"


def test_tool_preview_is_shorter_than_shell() -> None:
    lines = "\n".join(f"L{i:02d}" for i in range(20))
    tool = visible_body("read_file", lines, expanded=False, live=False)
    shell = visible_body("run_shell", lines, expanded=False, live=False)
    assert tool.count("\n") == TOOL_PREVIEW
    assert "Ctrl+O to expand" in tool
    assert "L03" not in tool
    assert preview_limit("run_shell") > preview_limit("read_file")
    assert preview_limit("write_file") > preview_limit("run_shell")
    assert "L07" in shell
    assert "L08" not in shell.split("Ctrl+O")[0]


def test_expanded_tool_shows_full_diff() -> None:
    diff = "Wrote 10 bytes\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new"
    assert visible_body("write_file", diff, expanded=True, live=False) == diff


def test_wide_terminal_uses_full_width_for_thinking() -> None:
    paragraph = (
        'The user just said "你好" (hello in Chinese). This is a simple greeting. '
        "I should respond politely and briefly, perhaps asking what they'd like "
        "help with. No need to call tools yet."
    )
    shown = visible_body("thinking", paragraph, expanded=False, live=False, width=160)
    assert "tools yet" in shown
    assert "Ctrl+O to expand" not in shown
    long = ("Need a plan. " * 40).strip()
    folded = visible_body("thinking", long, expanded=False, live=False, width=160)
    assert "Ctrl+O to expand" in folded
    assert folded.count("\n") == THINKING_PREVIEW


def test_long_paragraph_is_truncated_by_visual_width() -> None:
    paragraph = (
        'The user just said "你好" (Hello in Chinese). This is a greeting, '
        "not a task. I should respond concisely in Chinese and perhaps ask "
        "what they would like help with, or wait."
    )
    assert "\n" not in paragraph
    shown = visible_body("thinking", paragraph, expanded=False, live=False, width=48)
    assert "Ctrl+O to expand" in shown
    content_lines = [line for line in shown.split("\n") if not line.startswith("…")]
    assert len(content_lines) == THINKING_PREVIEW
    assert all(len(line) <= 48 for line in content_lines)
    expanded = visible_body("thinking", paragraph, expanded=True, live=False, width=48)
    assert expanded == paragraph
    assert "Ctrl+O" not in expanded


def test_wrap_to_width_handles_cjk() -> None:
    rows = wrap_to_width("你好世界", 4)
    assert rows == ["你好", "世界"]


def test_wrap_breaks_on_spaces_not_mid_word() -> None:
    rows = wrap_to_width("I should respond concisely in Chinese", 18)
    assert "should" in " ".join(rows)
    assert not any(row.rstrip().endswith("shou") for row in rows)


def test_user_message_indents_shift_enter_lines() -> None:
    rendered = render_user("你好\n你是谁")
    lines = rendered.plain.split("\n")
    assert lines[0].startswith("› ")
    assert "你好" in lines[0]
    assert lines[1].startswith("  ")
    assert "你是谁" in lines[1]
    assert not lines[1].lstrip().startswith("›")


def test_render_reply_bold_and_code() -> None:
    rendered = render_reply("我是 **Anvil**，目录是 `broken_ledger`。")
    plain = rendered.plain
    assert "**" not in plain
    assert "`" not in plain
    assert "Anvil" in plain
    assert "broken_ledger" in plain


def test_render_reply_turns_lists_into_bullets() -> None:
    plain = render_reply("- 30x20 网格\n- Esc 退出").plain
    assert "• " in plain
    assert plain.splitlines()[0].startswith("• ")
    assert "-" not in plain.splitlines()[0]


def test_render_markdown_drops_fences_and_left_aligns_headings() -> None:
    source = "## 运行方式\n\n- R 重新开始\n\n```\npython snake.py\n```\n"
    plain = render_markdown(source, width=48).plain
    assert "```" not in plain
    assert "python snake.py" in plain
    assert "运行方式" in plain
    heading = next(line for line in plain.splitlines() if "运行方式" in line)
    assert not heading.startswith("  ")
    assert "• " in plain or "R 重新开始" in plain


def test_assistant_prefix_is_a_bullet_on_the_first_line() -> None:
    body = render_markdown("Hello! I am **Anvil**.", width=40)
    prefixed = with_message_prefix(body)
    lines = prefixed.plain.split("\n")
    assert lines[0].startswith("● ")
    assert "Anvil" in lines[0]
    assert "**" not in prefixed.plain


def test_live_and_final_plain_layout_match() -> None:
    source = (
        "我可以做这些事：\n\n"
        "- **读代码**：查看文件\n"
        "- **改代码**：修复 `bug`\n\n"
        "比如你现在这个 `broken_ledger` 项目，如果里面有`bug`，可以修好。\n"
    )
    live = render_assistant(source, live=True, width=56)
    final = render_assistant(source, live=False, width=56)
    assert live.plain == final.plain
    assert "`" not in live.plain
    assert "**" not in live.plain
    assert "broken_ledger" in live.plain
    assert "bug" in live.plain


def test_live_and_final_fence_plain_match() -> None:
    source = "见下面：\n\n```python\nprint(1)\n```\n\n- 继续\n"
    live = render_assistant(source, live=True, width=48)
    final = render_assistant(source, live=False, width=48)
    assert live.plain == final.plain
    assert "print(1)" in live.plain
    assert "```" not in live.plain
    assert "继续" in live.plain


def test_open_fence_does_not_leave_raw_backticks() -> None:
    source = "example:\n\n```python\nprint(1)\n"
    live = render_assistant(source, live=True, width=48)
    assert "print(1)" in live.plain
    assert "```" not in live.plain


def test_assistant_wrap_keeps_hanging_indent() -> None:
    source = (
        "比如你现在这个 broken_ledger 项目（名字像是坏掉的账本），"
        "如果里面有bug，我可以帮你找到并修好。有什么具体任务吗？"
    )
    prefixed = with_message_prefix(render_assistant(source, live=False, width=40))
    lines = [line for line in prefixed.plain.split("\n") if line.strip()]
    assert lines[0].startswith("● ")
    assert len(lines) > 1
    for line in lines[1:]:
        assert line.startswith("  ")


def test_finalized_thinking_visual_keeps_a_label() -> None:
    plain = _thinking_text("thinking\none\ntwo").plain
    lines = plain.split("\n")
    assert lines[0] == "thinking"
    assert lines[1].startswith("  ")
    assert "one" in lines[1]


def test_status_line_changes_with_expand_and_busy() -> None:
    idle = status_plain(expanded=False, busy=False)
    assert "ctrl+o" in idle
    assert "展开" in idle
    assert "?" in idle
    assert "收起" in status_plain(expanded=True, busy=False)
    assert "停止" in status_plain(expanded=False, busy=True)
    assert "离开底部" in status_plain(expanded=False, busy=True, follow=False)
    assert "再按一次" in status_plain(expanded=False, busy=False, quit_armed=True)
    assert "等待确认" in status_plain(expanded=False, busy=True, approving=True)
    assert "ctrl+e" in status_plain(expanded=False, busy=True, approving=True)
    assert "输入原因" in status_plain(
        expanded=False, busy=True, approving=True, approval_reason=True
    )
    footer = footer_text(
        identity="Anvil  scripted  max  ask  ledger",
        expanded=False,
        busy=False,
    )
    assert "scripted" in footer.plain
    assert "ctrl+o" in footer.plain.lower()
    assert "\n" in footer.plain


def test_stream_flush_delay_first_and_urgent_are_immediate() -> None:
    assert stream_flush_delay(1.0, None, urgent=False) == 0.0
    assert stream_flush_delay(1.0, 0.99, urgent=True) == 0.0


def test_stream_flush_delay_waits_out_the_window() -> None:
    remaining = stream_flush_delay(1.02, 1.0, urgent=False)
    assert abs(remaining - (STREAM_FLUSH_S - 0.02)) < 1e-9
    assert stream_flush_delay(1.0 + STREAM_FLUSH_S, 1.0, urgent=False) == 0.0
    assert stream_flush_delay(1.0 + STREAM_FLUSH_S + 0.01, 1.0, urgent=False) == 0.0


def test_coalesce_merges_adjacent_same_kind_deltas() -> None:
    events = [
        AgentEvent("delta", {"kind": "reasoning", "text": "a"}),
        AgentEvent("delta", {"kind": "reasoning", "text": "b"}),
        AgentEvent("delta", {"kind": "content", "text": "c"}),
        AgentEvent("delta", {"kind": "content", "text": "d"}),
        AgentEvent("assistant", {"content": "cd"}),
    ]
    merged = coalesce_events(events)
    assert len(merged) == 3
    assert merged[0].payload["text"] == "ab"
    assert merged[1].payload["text"] == "cd"
    assert merged[2].kind == "assistant"
