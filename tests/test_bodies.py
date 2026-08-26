import pathlib
import sys

import mujoco
import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bodies.body import Body, EE_SITE, HEAD_CAM
from sim.world import World


@pytest.fixture(scope="module")
def world():
    return World(Body(REPO / "bodies" / "g1" / "config.yaml"))


def test_compiles_fixed_base(world):
    # 29 body joints + 14 finger joints; free joints belong to scene objects
    assert world.model.nu == 43
    robot_free = [j for j in range(world.model.njnt)
                  if world.model.joint(j).type[0] == mujoco.mjtJoint.mjJNT_FREE]
    assert len(robot_free) == 3  # red block, green block, blue cup


def test_interface_injected(world):
    for site in EE_SITE.values():
        assert world.model.site(site).id >= 0
    assert world.model.cam(HEAD_CAM).id >= 0


def test_limits_and_actuators_derived(world):
    body = world.body
    assert len(body.joint_limits) == 43
    for lo, hi in body.joint_limits.values():
        assert lo < hi
    assert len(body.actuator_of) == 43
    for side, joints in body.arm_joints.items():
        assert len(joints) == 7


def test_objects_registered_and_on_table(world):
    assert len(world.objects) == 4
    for bname in world.objects:
        x, y, z = world.object_pos(bname)
        assert 0.70 < z < 0.80, f"{bname} not on table: z={z}"


def test_stands_stably(world):
    world.reset()
    torso = world.model.body("torso_link").id
    z0 = world.data.xpos[torso][2]
    world.step(2.0)
    assert abs(world.data.xpos[torso][2] - z0) < 0.005
