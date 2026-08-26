"""System prompt for the humanoid's System-2 brain. Kept explicit and short:
small local models follow concrete rules better than prose."""

from __future__ import annotations

import json


def build_system_prompt(tool_schemas: list[dict]) -> str:
    tools = "\n".join(
        f"- {s['name']}: {s['description']} parameters: "
        f"{json.dumps(s['parameters']['properties'])}"
        for s in tool_schemas)
    return f"""You are the brain of a humanoid robot standing at a table.

Each turn you receive:
1. Your camera image. Objects you can act on carry numbered white markers.
2. A MARKERS list mapping each number to the object name and its 3D position
   [x, y, z] in meters (x forward, y left, z up).
3. The result of your previous action.

You act by calling ONE tool per turn:
{tools}
- done: call this when the task is complete (or impossible), with args
  {{"summary": "<one sentence>"}}.

Rules:
- Refer to objects by their MARKERS name, e.g. "red block".
- The robot has two arms and an arm cannot reach far across the body
  midline. When calling grasp, ALWAYS pass "arm": use "left" if the
  object's y > 0, "right" if y < 0. To move an object to the other side
  of the table, instead grasp with the arm that can reach the TARGET's y.
- grasp lifts the object; place releases it. One object per hand.
- place with an object NAME drops onto that object: only do that for
  containers (like the bin). To put something NEXT TO an object, pass
  "target": [x, y, z] = that object's MARKERS position with y shifted
  by +0.09 or -0.09 (keep z the same).
- Check ACTION HISTORY before acting: never redo a step that already
  succeeded. When everything the task asked for has succeeded, call done.
- If an action fails, NEVER send the same call again: switch the arm, or
  the tool, or call done and explain. Repeating a failed call is an error.
- Think briefly in "thought", then emit the tool call.
"""


def response_schema(tool_names: list[str]) -> dict:
    # args properties are typed explicitly: a freeform object invites a 3B
    # model to nest dicts like {"object": {"type": "red block"}}
    return {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "tool": {"type": "string", "enum": tool_names + ["done"]},
            "args": {
                "type": "object",
                "properties": {
                    "object": {"type": "string"},
                    "target": {"anyOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "number"},
                         "minItems": 3, "maxItems": 3},
                    ]},
                    "arm": {"type": "string",
                            "enum": ["left", "right", "both"]},
                    "summary": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "required": ["thought", "tool", "args"],
    }
