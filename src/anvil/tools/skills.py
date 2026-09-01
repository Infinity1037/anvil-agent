from __future__ import annotations

from anvil.skills import SkillStore
from anvil.tools.base import ToolSpec


def make_load_skill(store: SkillStore) -> ToolSpec:
    return ToolSpec(
        name="load_skill",
        description=(
            "Load one available project Skill by exact name. Use it before acting when "
            "the current task matches a Skill description or the user explicitly requests one. "
            "It returns guidance only and never grants permissions or executes resources."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Exact Skill name from the project skill catalog.",
                },
                "arguments": {
                    "type": "string",
                    "description": "Optional user request or arguments for $ARGUMENTS.",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        handler=store.load_for_model,
        parallel_safe=True,
        available=lambda: store.available,
    )
