from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from anvil.tui.cards import (
    body_is_truncated,
    key_argument,
    render_body,
    render_card,
)
from anvil.tui.fold import (
    THINKING_PREVIEW,
    preview_limit,
    visible_body,
    wrap_to_width,
)
from anvil.tui.markdown import (
    render_assistant,
    with_message_prefix,
)

_THINKING_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def welcome_content(
    *,
    workspace: str,
    model: str,
    effort: str,
    perm: str,
    context: str,
    skills: int = 0,
) -> Text:
    """Compact startup identity and environment summary."""
    body = Text(no_wrap=True, overflow="ellipsis")
    body.append("Local coding agent", style="dim")
    body.append("\n\n")
    body.append(workspace, style="bold #67e8f9")
    body.append("\n")
    fields = [model, effort, perm, context]
    if skills > 0:
        fields.append(f"{skills} skill" + ("s" if skills != 1 else ""))
    for index, field in enumerate(fields):
        if index:
            body.append("  ·  ", style="dim")
        body.append(field)
    return body


class WelcomeBlock(Static):
    """Scrollable startup card; it leaves the viewport as the transcript grows."""

    DEFAULT_CSS = """
    WelcomeBlock {
        height: auto;
        width: 92;
        max-width: 100%;
        margin: 0 0 1 0;
        padding: 0 1;
        background: #111111;
        border: round #3a3a3a;
        border-title-color: #67e8f9;
        border-title-style: bold;
    }
    """

    def __init__(
        self,
        *,
        version: str,
        workspace: str,
        model: str,
        effort: str,
        perm: str,
        context: str,
        skills: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            welcome_content(
                workspace=workspace,
                model=model,
                effort=effort,
                perm=perm,
                context=context,
                skills=skills,
            ),
            markup=False,
            **kwargs,
        )
        self.border_title = Text.assemble(
            (" ANVIL ", "bold #67e8f9"),
            (f" v{version} ", "dim"),
        )


class FoldBlock(Static):
    """Transcript card whose body grows/shrinks in place when expanded changes."""

    DEFAULT_CSS = """
    FoldBlock {
        height: auto;
        width: 1fr;
        margin-top: 1;
        padding: 0;
    }
    """

    def __init__(
        self,
        kind: str,
        summary: str = "",
        arguments: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(markup=False, **kwargs)
        self.kind = kind
        self.arguments = dict(arguments or {})
        self.summary = summary or key_argument(kind, self.arguments)
        self.full = ""
        self.live = kind == "thinking"
        self.expanded = False
        self.ok: bool | None = None
        self._spin = 0
        self._spin_timer = None
        self.add_class(kind)
        self.refresh_body()

    def on_mount(self) -> None:
        if self.kind == "thinking" and self.live:
            self._spin_timer = self.set_interval(0.08, self._spin_tick)

    def on_unmount(self) -> None:
        self._stop_spin()

    def _spin_tick(self) -> None:
        if not self.live:
            self._stop_spin()
            return
        self._spin = (self._spin + 1) % len(_THINKING_SPINNER)
        self.refresh_body()

    def _stop_spin(self) -> None:
        timer = self._spin_timer
        self._spin_timer = None
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass

    def on_resize(self, event) -> None:  # noqa: ARG002
        self.refresh_body()

    def _wrap_width(self) -> int:
        width = self.size.width
        if width <= 0:
            try:
                width = self.app.size.width - 8
            except Exception:
                width = 80
        if self.kind == "thinking":
            return max(16, width - 2)
        return max(16, width)

    def shown(self) -> str:
        if self.kind == "thinking":
            return visible_body(
                self.kind,
                self.full,
                expanded=self.expanded,
                live=self.live,
                width=self._wrap_width(),
            )
        return render_body(
            self.kind,
            arguments=self.arguments,
            result=self.full,
            live=self.live,
            expanded=self.expanded,
            ok=self.ok,
        ).plain

    def has_hidden_lines(self) -> bool:
        if self.kind == "thinking":
            text = (self.full or "").rstrip()
            if not text:
                return False
            return len(wrap_to_width(text, self._wrap_width())) > preview_limit(self.kind)
        return body_is_truncated(
            self.kind,
            arguments=self.arguments,
            result=self.full,
            ok=self.ok,
        )

    def append_full(self, chunk: str) -> None:
        self.full += chunk
        self.refresh_body()

    def set_full(self, text: str) -> None:
        self.full = text
        self.refresh_body()

    def set_result(self, content: str, ok: bool) -> None:
        self.full = content
        self.ok = ok
        self.live = False
        self.refresh_body()

    def finalize(self) -> None:
        if not self.live:
            return
        self.live = False
        self._stop_spin()
        self.refresh_body()

    def set_expanded(self, value: bool) -> None:
        if self.expanded == value:
            return
        self.expanded = value
        self.refresh_body()

    def refresh_body(self) -> None:
        if self.kind == "thinking":
            self.update(_thinking_text(self._thinking_display()))
            return
        self.update(
            render_card(
                self.kind,
                arguments=self.arguments,
                result=self.full,
                live=self.live,
                expanded=self.expanded,
                ok=self.ok,
                summary=self.summary,
            )
        )

    def _thinking_display(self) -> str:
        if self.live:
            body = self.shown() or "thinking…"
            if not self.expanded:
                rows = body.split("\n")
                while len(rows) < 1 + THINKING_PREVIEW:
                    rows.append("")
                body = "\n".join(rows)
            if body.startswith("thinking…"):
                frame = _THINKING_SPINNER[self._spin % len(_THINKING_SPINNER)]
                return f"{frame} {body}"
            return body
        if self.expanded and self.full.strip():
            content = "\n".join(wrap_to_width(self.full.rstrip(), self._wrap_width()))
        else:
            content = visible_body(
                "thinking",
                self.full,
                expanded=False,
                live=False,
                width=self._wrap_width(),
            )
        return f"thinking\n{content}" if content else "thinking"


class UserBlock(Static):
    DEFAULT_CSS = """
    UserBlock {
        height: auto;
        width: 1fr;
        margin-top: 1;
        margin-bottom: 0;
    }
    """

    def __init__(self, text: str, **kwargs) -> None:
        super().__init__(markup=False, **kwargs)
        self.update(render_user(text))


class AssistantBlock(Static):
    DEFAULT_CSS = """
    AssistantBlock {
        height: auto;
        width: 1fr;
        margin-top: 1;
        margin-bottom: 0;
    }
    """

    def __init__(self, text: str = "", **kwargs) -> None:
        super().__init__(markup=False, **kwargs)
        self._text = ""
        self._final = False
        if text:
            self.set_text(text)

    def on_resize(self, event) -> None:  # noqa: ARG002
        self._refresh()

    def append(self, chunk: str) -> None:
        if not chunk:
            return
        self._final = False
        self._text += chunk
        self._refresh()

    def set_text(self, text: str) -> None:
        self._text = text
        self._final = True
        self._refresh()

    def _wrap_width(self) -> int:
        width = self.size.width
        if width <= 0:
            try:
                width = self.app.size.width - 8
            except Exception:
                width = 80
        return max(24, width)

    def _refresh(self) -> None:
        if not self._text:
            self.update("")
            return
        body = render_assistant(
            self._text,
            live=not self._final,
            width=self._wrap_width(),
        )
        self.update(with_message_prefix(body))


class NoticeBlock(Static):
    DEFAULT_CSS = """
    NoticeBlock {
        height: auto;
        width: 1fr;
        margin: 0 0 1 0;
    }
    """

    def __init__(self, text: str, *, style: str = "dim", **kwargs) -> None:
        super().__init__(markup=False, **kwargs)
        self.update(Text(text, style=style))


def render_user(text: str) -> Text:
    """Bullet on the first line; later Shift+Enter lines indent to match."""
    out = Text()
    for index, line in enumerate((text or "").split("\n")):
        if index:
            out.append("\n")
            out.append("  ")
        else:
            out.append("› ", style="bold cyan")
        out.append(line, style="bold")
    return out


def _thinking_text(body: str) -> Text:
    renderable = Text()
    lines = body.split("\n")
    for index, line in enumerate(lines):
        if index:
            renderable.append("\n")
        if index == 0 and "thinking" in line:
            renderable.append(line, style="dim italic")
            continue
        indented = f"  {line}" if line else ""
        style = "dim" if line.startswith("… (") else "dim italic"
        renderable.append(indented, style=style)
    return renderable
