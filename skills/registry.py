"""Skill registry: the complete tool surface exposed to the brain.
Each skill module defines SCHEMA (JSON schema for the VLM tool call) and
run(rt, world, **args) -> dict result. Objects are referenced by display
name (or SoM marker resolution happens above this layer, in the brain).
"""

from __future__ import annotations

from skills import grasp, home, look_at, place, reach

_MODULES = [reach, grasp, place, look_at, home]

SKILLS = {m.SCHEMA["name"]: (m.SCHEMA, m.run) for m in _MODULES}


def resolve_object(world, name: str) -> str | None:
    """Display name -> body name (case-insensitive)."""
    for bname, display in world.objects.items():
        if display.lower() == name.strip().lower():
            return bname
    return None


def execute(name: str, rt, world, **args) -> dict:
    if name not in SKILLS:
        return {"ok": False, "error": f"unknown skill '{name}'"}
    try:
        return SKILLS[name][1](rt, world, **args)
    except TypeError as e:
        return {"ok": False, "error": f"bad arguments: {e}"}


def tool_schemas() -> list[dict]:
    return [schema for schema, _ in SKILLS.values()]
