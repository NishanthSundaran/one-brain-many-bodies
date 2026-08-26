import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bodies.body import Body
from brain.agent_loop import Agent
from brain.system_prompt import build_system_prompt, response_schema
from sim.cameras import CameraRig
from sim.world import World
from skills.registry import tool_schemas
from skills.runtime import Runtime


class ScriptedVLM:
    """Stands in for Ollama: replays canned tool calls."""

    def __init__(self, calls):
        self.calls = list(calls)
        self.prompts = []

    def chat(self, messages, schema=None, timeout=0):
        self.prompts.append(messages[-1]["content"])
        call = self.calls.pop(0) if self.calls else \
            {"thought": "", "tool": "done", "args": {"summary": "out of script"}}
        return json.dumps(call), {"seconds": 0.0, "eval_count": 0}


@pytest.fixture()
def stack():
    world = World(Body(REPO / "bodies" / "g1" / "config.yaml"))
    rt = Runtime(world)
    rig = CameraRig(world.model, width=320, height=240)
    yield rt, world, rig
    rig.close()


def test_prompt_and_schema_shape():
    schemas = tool_schemas()
    prompt = build_system_prompt(schemas)
    for s in schemas:
        assert s["name"] in prompt
    schema = response_schema([s["name"] for s in schemas])
    assert "done" in schema["properties"]["tool"]["enum"]
    assert schema["properties"]["args"]["additionalProperties"] is False


def test_agent_executes_and_terminates(stack):
    rt, world, rig = stack
    vlm = ScriptedVLM([
        {"thought": "", "tool": "grasp",
         "args": {"object": "green block", "arm": "right"}},
        {"thought": "", "tool": "place", "args": {"target": "bin"}},
        {"thought": "", "tool": "done", "args": {"summary": "done"}},
    ])
    transcript = Agent(rt, world, rig, vlm=vlm).run("green block into bin")
    assert [e["tool"] for e in transcript] == ["grasp", "place", "done"]
    assert all(e["result"]["ok"] for e in transcript)
    bb, gp = world.object_pos("obj_bin"), world.object_pos("obj_green_block")
    assert abs(gp[0] - bb[0]) < 0.07 and abs(gp[1] - bb[1]) < 0.07


def test_agent_breaks_repetition_loop(stack):
    rt, world, rig = stack
    bad = {"thought": "", "tool": "grasp",
           "args": {"object": "green block", "arm": "left"}}  # fails: wrong arm
    vlm = ScriptedVLM([bad] * 6)
    transcript = Agent(rt, world, rig, vlm=vlm).run("loop forever")
    assert len(transcript) < 6
    assert "aborted" in transcript[-1]["result"]["summary"]


def test_agent_breaks_no_progress_loop(stack):
    """Repeating a call that *succeeds* but changes nothing (home spam)
    must also terminate — the counter resets only when the call changes."""
    rt, world, rig = stack
    spam = {"thought": "", "tool": "home", "args": {"arm": "both"}}
    vlm = ScriptedVLM([spam] * 8)
    transcript = Agent(rt, world, rig, vlm=vlm).run("spin forever")
    assert len(transcript) <= 3
    assert "aborted" in transcript[-1]["result"]["summary"]


def test_marker_reference_resolution(stack):
    rt, world, rig = stack
    vlm = ScriptedVLM([
        {"thought": "", "tool": "grasp", "args": {"object": "2", "arm": "right"}},
        {"thought": "", "tool": "done", "args": {"summary": "x"}},
    ])
    transcript = Agent(rt, world, rig, vlm=vlm).run("grasp marker 2")
    assert transcript[0]["result"]["ok"]  # marker 2 = green block, right arm
    assert rt.held("right") == "obj_green_block"


def test_action_history_in_prompt(stack):
    rt, world, rig = stack
    vlm = ScriptedVLM([
        {"thought": "", "tool": "home", "args": {}},
        {"thought": "", "tool": "done", "args": {"summary": "x"}},
    ])
    Agent(rt, world, rig, vlm=vlm).run("go home")
    assert "ACTION HISTORY" in vlm.prompts[-1]
    assert "home" in vlm.prompts[-1]
