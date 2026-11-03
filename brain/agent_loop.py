"""The System-2 loop: perceive (SoM-annotated camera frame) -> plan (local
VLM tool call) -> act (skill execution) -> observe (result), until done.
"""

from __future__ import annotations

import io
import json

import numpy as np
from PIL import Image

from brain.som import annotate
from brain.system_prompt import build_system_prompt, response_schema
from brain.vlm_client import OllamaVLM
from skills.registry import execute, tool_schemas


def _png(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


# Minimal tool surface: a 3B planner stays on the rails with few choices.
# `reach` is subsumed by grasp/place, and `look_at` only invites stalls —
# the head camera already sees the whole workspace and every marker ships
# with its 3D position. Both remain available as skills.
DEFAULT_TOOLS = ("walk_to", "grasp", "place", "home")


class Agent:
    def __init__(self, rt, world, rig, vlm: OllamaVLM | None = None,
                 allowed_tools: tuple = DEFAULT_TOOLS, context: str = "full"):
        self.rt, self.world, self.rig = rt, world, rig
        # "full": the whole chat so far; "last": system prompt + the newest state message only.
        # The state message already carries the action history and the last result, so "last"
        # loses nothing the planner needs and keeps the prompt near 1.3k tokens instead of 4.5k.
        assert context in ("full", "last")
        self.context = context
        self.vlm = vlm or OllamaVLM()
        # A small tool surface keeps a 3B planner on the rails (reach is
        # subsumed by grasp/place and only invites loops), and it must match
        # what *this body* can do: handing a fixed-base robot a walk_to it
        # can never complete just burns turns on a tool that always fails.
        schemas = [s for s in tool_schemas()
                   if s["name"] in allowed_tools
                   and (s["name"] != "walk_to" or rt.loco is not None)]
        self.tools = [s["name"] for s in schemas]
        self.system = build_system_prompt(schemas)
        self.schema = response_schema(self.tools)
        self.transcript: list[dict] = []

    def _table_name(self, pos) -> str:
        t = self.world.table_of(pos)
        for name, table in self.world.tables.items():
            if t is not None and table is t:
                return name
        return "unknown"

    def _perceive(self) -> tuple[bytes, dict]:
        rgb = self.rig.rgb(self.world.data)
        seg = self.rig.segmentation(self.world.data)
        annotated, markers = annotate(rgb, seg, self.world.model,
                                      self.world.data, self.world.objects)
        return _png(annotated), markers

    def _resolve_marker_refs(self, args: dict, markers: dict) -> dict:
        """Allow the VLM to say "3" or 3 where an object name is expected."""
        out = dict(args)
        for key in ("object", "target"):
            v = out.get(key)
            if isinstance(v, dict):  # e.g. {"name": "green block", ...}
                v = v.get("name", v.get("object", v))
                out[key] = v
            if isinstance(v, int) or (isinstance(v, str) and v.strip().isdigit()):
                m = markers.get(int(v))
                if m:
                    out[key] = m["name"]
        return out

    def run(self, command: str, max_steps: int = 10,
            on_step=None) -> list[dict]:
        messages = [{"role": "system", "content": self.system}]
        last_result: dict | None = None
        last_call = None
        repeats = 0
        history: list[str] = []
        for step in range(max_steps):
            png, markers = self._perceive()
            if step == 0 and not markers:
                # every scene has objects in view; an empty list means the renderer is broken
                # (e.g. a previous episode's mujoco.Renderer still alive in this process)
                raise RuntimeError("perception returned no markers on the first frame")
            marker_txt = json.dumps(
                {mid: {"name": m["name"], "position": m["pos"],
                       "arm": self.rt.nearest_arm(m["pos"]),
                       "table": self._table_name(m["pos"])}
                 for mid, m in markers.items()})
            parts = [f"TASK: {command}", f"MARKERS: {marker_txt}"]
            if history:
                parts.append("ACTION HISTORY: " + "; ".join(history))
            if last_result is not None:
                parts.append(f"PREVIOUS ACTION RESULT: {json.dumps(last_result)}")
            # only the newest frame goes to the model: stale views waste
            # ~400 context tokens each and mislead the planner
            for m in messages:
                m.pop("images", None)
            messages.append({"role": "user", "content": "\n".join(parts),
                             "images": [png]})

            sent = messages if self.context == "full" else [messages[0], messages[-1]]
            content, stats = self.vlm.chat(sent, schema=self.schema)
            try:
                call = json.loads(content)
            except json.JSONDecodeError:
                call = {"thought": "unparseable", "tool": "done",
                        "args": {"summary": "brain emitted invalid JSON"}}
            messages.append({"role": "assistant", "content": content})

            entry = {"step": step, "thought": call.get("thought", ""),
                     "tool": call["tool"], "args": call.get("args", {}),
                     "vlm_seconds": stats["seconds"]}
            if call.get("thought") == "unparseable":
                entry["raw"] = content[:2000]
            if call["tool"] == "done":
                entry["result"] = {"ok": True, "summary":
                                   call.get("args", {}).get("summary", "")}
                self.transcript.append(entry)
                if on_step:
                    on_step(entry)
                break
            args = self._resolve_marker_refs(call.get("args", {}), markers)

            # Hard loop-breaker: a small model will happily repeat a call
            # forever — a failing one, or a harmless "successful" one that
            # makes no progress (look_at spam). Count consecutive identical
            # calls; the counter only resets when the call actually changes.
            sig = (call["tool"], json.dumps(args, sort_keys=True))
            if sig == last_call:
                repeats += 1
            else:
                last_call, repeats = sig, 0

            failed_before = (last_result is not None
                             and not last_result.get("ok", False))
            if repeats >= 2:
                entry["result"] = {"ok": False,
                                   "summary": "aborted: brain repeated the "
                                              "same action without progress"}
                self.transcript.append(entry)
                if on_step:
                    on_step(entry)
                break
            if repeats and failed_before:
                entry["result"] = {"ok": False, "error":
                                   "REFUSED: identical action already failed. "
                                   "You MUST change the arm or the tool, or "
                                   "call done."}
                last_result = {"tool": call["tool"], **entry["result"]}
                history.append(f"{call['tool']}(...) -> REFUSED (repeat)")
                self.transcript.append(entry)
                if on_step:
                    on_step(entry)
                continue

            if on_step:
                on_step(entry)
            result = execute(call["tool"], self.rt, self.world, **args)
            entry["result"] = result
            last_result = {"tool": call["tool"], **result}
            history.append(f"{call['tool']}({json.dumps(args)}) -> "
                           + ("OK" if result.get("ok")
                              else f"FAILED: {result.get('error', '')}"))
            self.transcript.append(entry)
        return self.transcript
