"""Assistant markdown for the transcript.

Streaming and finalized replies share one parser. An unclosed fence is
virtually closed so it cannot swallow later lines. Syntax highlighting
runs only after the message is complete. Wrap happens at content width
(minus the bullet) so Textual does not re-break the line.
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from io import StringIO

from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import CodeBlock, Heading, Markdown
from rich.syntax import Syntax
from rich.text import Text

_MD_INLINE = re.compile(r"\*\*(.+?)\*\*|`([^`]+)`")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_UL = re.compile(r"^(\s*)[-*]\s+(.*)$")
_OL = re.compile(r"^(\s*)(\d+)[.)]\s+(.*)$")

# "● " and the hanging indent on later lines.
PREFIX_WIDTH = 2
_PARSE_WIDTH = 4096
_markdown_live: ContextVar[bool] = ContextVar("markdown_live", default=False)


class LeftHeading(Heading):
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        text = self.text.copy()
        text.justify = "left"
        text.stylize("bold")
        yield text


class TightCodeBlock(CodeBlock):
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        code = str(self.text).rstrip()
        if _markdown_live.get():
            yield Text(code)
            return
        yield Syntax(
            code,
            self.lexer_name or "text",
            theme=self.theme,
            word_wrap=True,
            padding=0,
            background_color="default",
        )


class TranscriptMarkdown(Markdown):
    elements = {
        **Markdown.elements,
        "heading_open": LeftHeading,
        "fence": TightCodeBlock,
        "code_block": TightCodeBlock,
    }


def fence_is_open(text: str) -> bool:
    n = sum(1 for line in text.splitlines() if line.strip().startswith("```"))
    return n % 2 == 1


def render_reply(text: str) -> Text:
    """Line-wise fallback: **bold**, `code`, lists, ATX headings. Fences stay literal."""
    out = Text()
    for index, line in enumerate((text or "").splitlines()):
        if index:
            out.append("\n")
        heading = _HEADING.match(line)
        if heading:
            out.append(heading.group(2), style="bold")
            continue
        unordered = _UL.match(line)
        if unordered:
            out.append(unordered.group(1))
            out.append("• ", style="bold")
            _append_inline(out, unordered.group(2))
            continue
        ordered = _OL.match(line)
        if ordered:
            out.append(ordered.group(1))
            out.append(f"{ordered.group(2)}. ", style="bold")
            _append_inline(out, ordered.group(3))
            continue
        _append_inline(out, line)
    return out


def render_markdown(text: str, *, width: int = 80, live: bool = False) -> Text:
    """Parse markdown and wrap to ``width`` (content cells, no bullet)."""
    parsed = parse_markdown(text, live=live)
    if width <= 0:
        return parsed
    return wrap_rich_text(parsed, max(1, width))


def render_assistant(text: str, *, live: bool = False, width: int = 80) -> Text:
    """Widget-width render used by the transcript. Leaves room for ``● ``."""
    content_width = max(1, (width or 80) - PREFIX_WIDTH)
    return render_markdown(text, width=content_width, live=live)


def parse_markdown(text: str, *, live: bool = False) -> Text:
    source = _prepare_source(text)
    if not source:
        return Text()
    buf = StringIO()
    console = Console(
        file=buf,
        width=_PARSE_WIDTH,
        force_terminal=True,
        color_system="truecolor",
        highlight=False,
        legacy_windows=False,
        soft_wrap=True,
    )
    token = _markdown_live.set(live)
    try:
        console.print(TranscriptMarkdown(source, code_theme="ansi_dark", hyperlinks=False))
    except Exception:
        return render_reply(source)
    finally:
        _markdown_live.reset(token)
    return _squeeze_lines(Text.from_ansi(buf.getvalue()))


def wrap_rich_text(body: Text, width: int) -> Text:
    """Fold each logical line to ``width`` cells, including CJK runs."""
    width = max(1, width)
    console = Console(
        width=width,
        force_terminal=True,
        highlight=False,
        legacy_windows=False,
    )
    out = Text()
    lines = body.split("\n")
    for index, line in enumerate(lines):
        if index:
            out.append("\n")
        if not line.plain:
            continue
        rows = line.wrap(console, width, overflow="fold")
        if not rows:
            continue
        for row_index, row in enumerate(rows):
            if row_index:
                out.append("\n")
            out.append_text(row)
    return out


def with_message_prefix(body: Text, mark: str = "● ", style: str = "bold") -> Text:
    lines = body.split("\n")
    while lines and not lines[0].plain.strip():
        lines.pop(0)
    while lines and not lines[-1].plain.strip():
        lines.pop()
    out = Text()
    indent = " " * len(mark)
    for index, line in enumerate(lines):
        line.rstrip()
        if index:
            out.append("\n")
            out.append(indent)
        else:
            out.append(mark, style=style)
        out.append_text(line)
    return out


def _prepare_source(text: str) -> str:
    source = text or ""
    if fence_is_open(source):
        source = source.rstrip() + "\n```"
    return source.rstrip()


def _append_inline(out: Text, line: str) -> None:
    pos = 0
    for match in _MD_INLINE.finditer(line):
        if match.start() > pos:
            out.append(line[pos : match.start()])
        if match.group(1) is not None:
            out.append(match.group(1), style="bold")
        else:
            out.append(match.group(2), style="cyan")
        pos = match.end()
    out.append(line[pos:])


def _squeeze_lines(rendered: Text) -> Text:
    lines: list[Text] = []
    blank = 0
    for line in rendered.split("\n"):
        line.rstrip()
        if not line.plain.strip():
            blank += 1
            if blank == 1:
                lines.append(Text())
            continue
        blank = 0
        lines.append(line)
    while lines and not lines[0].plain.strip():
        lines.pop(0)
    while lines and not lines[-1].plain.strip():
        lines.pop()
    out = Text()
    for index, line in enumerate(lines):
        if index:
            out.append("\n")
        out.append_text(line)
    return out
