import os
from pathlib import Path

import pytest

from anvil.skills import MAX_SKILL_BODY_CHARS, SkillStore


def _write_skill(
    root: Path,
    name: str = "test-first",
    *,
    description: str = "Use when adding or changing tested behavior.",
    body: str = "Run tests before and after. Task: $ARGUMENTS",
    extra: str = "",
) -> Path:
    folder = root / ".agents" / "skills" / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "SKILL.md"
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n{extra}---\n\n{body}\n",
        encoding="utf-8",
    )
    return path


def test_discovery_exposes_metadata_but_not_instructions(tmp_path: Path) -> None:
    _write_skill(tmp_path, body="PRIVATE-BODY-RULE")
    store = SkillStore(tmp_path)

    assert [item.name for item in store.skills] == ["test-first"]
    assert "Use when adding" in store.prompt_catalog()
    assert "PRIVATE-BODY-RULE" not in store.prompt_catalog()


def test_load_renders_arguments_without_granting_allowed_tools(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        body="Keep the marker. Request: $ARGUMENTS",
        extra="allowed-tools: run_shell write_file\n",
    )
    loaded = SkillStore(tmp_path).load("test-first", "add parser")

    assert loaded.ok
    assert "add parser" in loaded.content
    assert "Keep the marker" in loaded.content
    assert "never grants permissions" not in loaded.content
    assert "normal approved tools" in loaded.content


def test_changed_skill_is_rejected_until_catalog_refresh(tmp_path: Path) -> None:
    path = _write_skill(tmp_path, body="old instructions")
    store = SkillStore(tmp_path)
    path.write_text(
        "---\nname: test-first\ndescription: changed\n---\nnew instructions\n",
        encoding="utf-8",
    )

    result = store.load("test-first")
    assert not result.ok
    assert result.error_code == "stale_skill"


@pytest.mark.parametrize(
    "text",
    [
        "name: missing-fence\ndescription: bad\n",
        "---\nname: Wrong\ndescription: bad\n---\nbody\n",
        "---\nname: test-first\ndescription: !!python/object:bad {}\n---\nbody\n",
        "---\nname: test-first\n---\nbody\n",
    ],
)
def test_invalid_skill_files_are_ignored(tmp_path: Path, text: str) -> None:
    folder = tmp_path / ".agents" / "skills" / "test-first"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(text, encoding="utf-8")

    store = SkillStore(tmp_path)
    assert store.skills == ()
    assert len(store.issues) == 1


def test_oversized_skill_body_is_ignored(tmp_path: Path) -> None:
    _write_skill(tmp_path, body="x" * (MAX_SKILL_BODY_CHARS + 1))
    store = SkillStore(tmp_path)
    assert store.skills == ()
    assert "instructions exceed" in store.issues[0]


def test_manual_only_skill_is_not_advertised_to_model(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        extra="disable-model-invocation: true\n",
    )
    store = SkillStore(tmp_path)
    assert len(store.skills) == 1
    assert not store.available
    assert "test-first" not in store.prompt_catalog()
    assert store.load("test-first").ok
    assert store.load_for_model("test-first").error_code == "manual_skill"


def test_flow_skill_is_rejected(tmp_path: Path) -> None:
    _write_skill(tmp_path, extra="type: flow\n")
    store = SkillStore(tmp_path)
    assert store.skills == ()
    assert "unsupported skill type" in store.issues[0]


def test_skill_symlink_cannot_escape_workspace(tmp_path: Path, tmp_path_factory) -> None:
    outside = tmp_path_factory.mktemp("outside-skill")
    target = _write_skill(outside)
    folder = tmp_path / ".agents" / "skills" / "test-first"
    folder.mkdir(parents=True)
    try:
        os.symlink(target, folder / "SKILL.md")
    except OSError:
        pytest.skip("symlink creation is unavailable")

    store = SkillStore(tmp_path)
    assert store.skills == ()
    assert "escapes" in store.issues[0]
