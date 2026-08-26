import pathlib
import sys

import mujoco
import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bodies.body import Body
from controllers.diff_ik import DiffIK
from sim.world import World


@pytest.fixture(scope="module")
def world():
    return World(Body(REPO / "bodies" / "g1" / "config.yaml"), objects=[])


@pytest.mark.parametrize("side", ["left", "right"])
def test_qdot_realizes_requested_twist(world, side):
    # The home pose is near-singular (arm hanging: sigma_3 ~ 0.02), so use a
    # generic elbow-raised configuration for the realization check.
    body, model, data = world.body, world.model, world.data
    world.reset(settle=0)
    sign = 1.0 if side == "left" else -1.0
    generic = [-0.6, sign * 0.4, 0.2, 0.9, 0.1, 0.1, 0.1]
    for j, q in zip(body.arm_joints[side], generic):
        data.qpos[model.joint(j).qposadr[0]] = q
    mujoco.mj_forward(model, data)

    # small lambda: this checks the mapping math; the runtime default (0.05)
    # deliberately trades tracking in weak directions for robustness
    ik = DiffIK(model, f"{side}_ee", body.arm_joints[side], damping=0.01)
    twist = np.array([0.05, -0.03, 0.04, 0, 0, 0])
    qdot = ik.solve(world.data, twist)
    # map back through the Jacobian: achieved twist ~ requested (DLS tolerance)
    jacp = np.zeros((3, world.model.nv))
    jacr = np.zeros((3, world.model.nv))
    mujoco.mj_jacSite(world.model, world.data, jacp, jacr, ik.site_id)
    achieved = np.vstack([jacp, jacr])[:, ik.dof_ids] @ qdot
    assert np.linalg.norm(achieved[:3] - twist[:3]) < 0.01
    world.reset()


def test_converges_to_nearby_target(world):
    world.reset(settle=0)
    body, model, data = world.body, world.model, world.data
    ik = DiffIK(model, "right_ee", body.arm_joints["right"])
    qpos_ids = ik.qpos_ids
    target = data.site_xpos[ik.site_id] + np.array([0.03, -0.05, 0.06])
    for _ in range(400):  # pure kinematic integration at 100 Hz
        err = target - data.site_xpos[ik.site_id]
        qdot = ik.solve(data, np.r_[2.0 * err, np.zeros(3)])
        data.qpos[qpos_ids] += qdot * 0.01
        mujoco.mj_forward(model, data)
    assert np.linalg.norm(target - data.site_xpos[ik.site_id]) < 0.005
    world.reset()
