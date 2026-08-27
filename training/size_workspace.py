"""Measure what the robot can actually reach, and size its workspace to that.

Written after guessing wrong twice. The first estimate maximised reach over a
whole height band and was ~8 cm optimistic; the second forgot that the policy
chooses its own stance height, which changes the answer again. So this
measures in the stance the trained policy actually adopts, conditioned on both
the height *and* the lateral offset where objects really sit, and prints the
bench geometry that follows.
"""

import pathlib
import sys
import warnings

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

import mujoco  # noqa: E402

from bodies.body import Body  # noqa: E402
from sim.world import World  # noqa: E402
from skills.runtime import Runtime  # noqa: E402

CHEST_CLEARANCE = 0.32   # measured: where the chest first touches a table edge
MARGIN = 0.05            # keep the target this far inside the reach limit


def main():
    world = World(Body(REPO / "bodies" / "g1_walk" / "config.yaml"))
    rt = Runtime(world)
    m, d = world.model, world.data
    pelvis_z = float(d.xpos[rt.root_id][2])
    base = rt.base_xy.copy()
    print(f"stance the policy adopts: pelvis {pelvis_z:.3f} m")

    arm, ik = rt.arm["right"], rt.ik["right"]
    saved = d.qpos.copy()
    rng = np.random.default_rng(0)
    pts = []
    for _ in range(30000):
        d.qpos[arm["qpos"]] = rng.uniform(arm["limits"][:, 0],
                                          arm["limits"][:, 1])
        mujoco.mj_forward(m, d)
        p = d.site_xpos[ik.site_id]
        pts.append([p[0] - base[0], p[1] - base[1], p[2]])
    d.qpos[:] = saved
    mujoco.mj_forward(m, d)
    pts = np.array(pts)

    print(f"\nright-hand forward reach (m), by object height and offset:")
    print(f"{'height':>8} " + " ".join(f"{'y=' + f'{y:+.2f}':>9}"
                                       for y in (-0.12, -0.16, -0.20)))
    best = (0.0, None, None)
    for z in np.arange(0.70, 1.02, 0.04):
        row = f"{z:8.2f}"
        for y in (-0.12, -0.16, -0.20):
            sel = pts[(np.abs(pts[:, 2] - z) < 0.03)
                      & (np.abs(pts[:, 1] - y) < 0.03)]
            if len(sel):
                r = float(sel[:, 0].max())
                row += f" {r:9.3f}"
                if r > best[0]:
                    best = (r, z, y)
            else:
                row += "        --"
        print(row)

    reach, z_best, y_best = best
    print(f"\nbest: {reach:.3f} m at height {z_best:.2f}, offset {y_best:+.2f}")
    target_x = reach - MARGIN
    print(f"usable object distance (with {MARGIN:.2f} m margin): {target_x:.3f} m")
    if target_x < CHEST_CLEARANCE:
        print(f"WARNING: chest needs {CHEST_CLEARANCE:.2f} m clearance — a "
              f"tabletop cannot fit; use a narrow bench the robot stands "
              f"against.")
    print("\nsuggested scene block:")
    print(f"  table_top: {z_best - 0.03:.2f}")
    print(f"  table_x: {target_x:.2f}")
    print(f"  table_half: [0.10, 0.30]")
    print(f"  objects:")
    print(f"    green block: [{target_x:.2f}, {y_best:.2f}]")
    print(f"    red block: [{target_x:.2f}, {-y_best:.2f}]")
    print(f"    bin: [{target_x - 0.02:.2f}, 0.00]")


if __name__ == "__main__":
    main()
