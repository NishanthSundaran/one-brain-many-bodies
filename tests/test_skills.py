import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bodies.body import Body
from sim.world import World
from skills.registry import SKILLS, execute, tool_schemas
from skills.runtime import Runtime


@pytest.fixture(scope="module")
def rt_world():
    world = World(Body(REPO / "bodies" / "g1" / "config.yaml"))
    return Runtime(world), world


def test_registry_complete():
    assert set(SKILLS) == {"reach", "grasp", "place", "look_at",
                           "home", "walk_to"}
    for schema in tool_schemas():
        assert schema["name"] and schema["parameters"]["type"] == "object"


def test_unknown_skill_and_bad_args(rt_world):
    rt, world = rt_world
    assert not execute("fly", rt, world)["ok"]
    assert not execute("grasp", rt, world, wrong_arg=1)["ok"]
    assert not execute("grasp", rt, world, object="purple whale")["ok"]
    assert not execute("place", rt, world, target="bin")["ok"]  # holding nothing


def test_pick_and_place_into_bin(rt_world):
    rt, world = rt_world
    world.reset()
    res = execute("grasp", rt, world, object="green block")
    assert res["ok"], res
    assert rt.held(res["arm"]) == "obj_green_block"
    res = execute("place", rt, world, target="bin")
    assert res["ok"], res
    bb = world.object_pos("obj_bin")
    gp = world.object_pos("obj_green_block")
    assert abs(gp[0] - bb[0]) < 0.07 and abs(gp[1] - bb[1]) < 0.07
    assert 0.72 < gp[2] < 0.82
    assert rt.held(res["arm"]) is None


def test_home_returns_arm(rt_world):
    rt, world = rt_world
    res = execute("home", rt, world)
    assert res["ok"]
    # arms hang again: EEs below shoulder height
    assert rt.ee_pos("left")[2] < 0.75 and rt.ee_pos("right")[2] < 0.75
