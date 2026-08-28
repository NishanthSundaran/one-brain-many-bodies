"""Both reaching controllers, same targets, same physics, same metric: in
the open, and then at a workbench.

demos/09 answers "which one gets the hand there". This adds two questions
that matter more for grasping: does it move like a person doing it, and does
it still work in the scene a grasp actually happens in? The learned policy
was trained in an empty world. A bench is not an obstacle it has to avoid so
much as contact dynamics it has never felt.
"""

from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bodies.body import Body                                     # noqa: E402
from controllers.reach_policy import ReachPolicy                 # noqa: E402
from eval.human_motion import score, table                       # noqa: E402
from sim.world import World                                      # noqa: E402
from skills.runtime import Runtime                               # noqa: E402

TARGETS = [(0.25, -0.18, 0.12), (0.32, -0.20, 0.16), (0.30, -0.30, 0.10)]
CFG = REPO / "bodies" / "g1_walk" / "config.yaml"


class Bare:
    """The world the reach policy was trained in: a floor and nothing else."""

    def __init__(self):
        self.body = Body(CFG)
        w = self.body.spec.worldbody
        w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[8, 8, 0.1])
        self.model = self.body.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.body.bind(self.model)
        self.body.reset(self.data)
        mujoco.mj_forward(self.model, self.data)


def setup(method, scene):
    world = Bare() if scene == "open" else World(Body(CFG))
    rt = Runtime(world)
    if method == "rl":
        # A policy is only valid for the plant it learned on: left on the
        # manipulation arm gains, with the stance controller still writing
        # leg targets underneath it, this one falls on every target. It also
        # starts every training episode at the body's home stance at rest.
        rt._set_arm_gain(1.0)
        rt._standing = False
        rt.loco = None
        rt.stance = None
        rt.body.reset(world.data)
        mujoco.mj_forward(world.model, world.data)
    return world, rt


def main():
    rows, summary = {}, []
    for scene in ("open", "bench"):
        for method in ("ik", "rl"):
            for i, L in enumerate(TARGETS, 1):
                world, rt = setup(method, scene)
                hand = world.model.site("right_ee").id
                tgt = rt.to_world(np.array(L[:2]))
                tgt[2] = world.data.xpos[rt.root_id][2] + L[2]
                if method == "ik":
                    rt.move_arm_ready("right")
                    with rt.trace("right") as path:
                        rt.reach_joint_space("right", tgt)
                        rt.step(0.4)
                else:
                    pol = ReachPolicy(
                        world.model, REPO / "third_party/policies/g1_reach.pt",
                        rt.body.cfg["locomotion"]["stand_angles"])
                    pol.aim(tgt, world.data.site_xpos[hand].copy())
                    path = []
                    for _ in range(int(4.4 / rt.dt)):
                        pol.step(world.data)
                        mujoco.mj_step(world.model, world.data)
                        path.append(world.data.site_xpos[hand].copy())
                P = np.array(path)
                err = float(np.linalg.norm(tgt - P[-1]))
                upright = bool(world.data.xpos[rt.root_id][2] > 0.55)
                m = score(P, rt.dt, target=tgt)
                rows[f"{scene}/{method.upper()} target {i}"] = m
                summary.append((scene, method, err, upright, m["human"]))

    print(table(rows))
    print()
    print("| scene | controller | mean error | within 6 cm | upright | "
          "human motion |\n|---|---|---|---|---|---|")
    for scene in ("open", "bench"):
        for method in ("ik", "rl"):
            rs = [r for r in summary if r[0] == scene and r[1] == method]
            name = {"ik": "deterministic IK",
                    "rl": "learned whole-body"}[method]
            print(f"| {scene} | {name} | {np.mean([r[2] for r in rs]):.3f} m "
                  f"| {sum(r[2] < 0.06 for r in rs)}/{len(rs)} "
                  f"| {sum(r[3] for r in rs)}/{len(rs)} "
                  f"| {sum(r[4] for r in rs)}/{len(rs)} |")


if __name__ == "__main__":
    main()
