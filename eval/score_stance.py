"""Does the robot stand on its feet, or dance on them?

The stance policy is scored on how still it holds the pelvis, and it is good
at that. It gets there by chattering its feet about twenty times a second,
which no amount of pelvis-drift measurement reveals and which is the first
thing anyone notices watching the recording. So the feet get measured too,
along with the posture the legs settle into.

Run it several times before believing any single row. Standing balance here
is chaotic: the same policy measured 21.5 and 17.5 foot lifts per second on
two runs, which was enough to make a retrained policy look like an
improvement it was not (see the README).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scipy.spatial.transform import Rotation                     # noqa: E402

from bodies.body import Body                                     # noqa: E402
from controllers.stance import StancePolicy                      # noqa: E402
from sim.world import World                                      # noqa: E402
from skills.registry import execute                              # noqa: E402
from skills.runtime import Runtime                               # noqa: E402

CFG = REPO / "bodies" / "g1_walk" / "config.yaml"
POLICIES = {"stance policy": "g1_stance.pt"}


def foot_stats(hist, dt):
    H = np.array(hist)
    if len(H) < 10:
        return None
    dur = len(H) * dt
    lifts = sum(
        int(np.sum(np.diff((H[:, k] > H[:, k].min() + 0.01).astype(int)) == 1))
        for k in (0, 1))
    return {"lifts_per_s": lifts / max(dur, 1e-9),
            "range_cm": float((H.max(0) - H.min(0)).max() * 100),
            "seconds": dur}


def posture_sample(world):
    """What the standing pose actually is.

    The reward specifies the pelvis: where it is, how high, how upright. It
    says nothing about the legs, so the policy is free to hold that pelvis up
    from any leg configuration it likes.

    Read the knee number against the right baseline. This body's own home
    pose is already a 38 degree crouch, which is how the G1 ships and not
    something a controller chose, so the crouch is not the policy's doing.
    What is the policy's doing is the *asymmetry*: 53 degrees on one knee
    against 36 on the other, which is what reads as shifting its weight
    about, and a 10 degree forward lean the home pose does not have.
    """
    d, m = world.data, world.model
    pel = m.body("pelvis").id
    pitch = Rotation.from_matrix(d.xmat[pel].reshape(3, 3)).as_euler(
        "xyz", degrees=True)[1]
    knees = [float(np.degrees(d.qpos[m.joint(f"{s}_knee_joint").qposadr[0]]))
             for s in ("left", "right")]
    return (float(d.xpos[pel][2]), float(pitch), max(knees),
            abs(knees[0] - knees[1]))


def summarise(samples):
    """Averaged over the window, not read off one frame.

    A single snapshot of a controller that is still moving is a sample of
    noise: read at two different instants, the same shipped policy gave 1 and
    17 degrees of knee asymmetry. Only the mean over the window says anything.
    """
    a = np.array(samples)
    return {"pelvis_h": a[:, 0].mean(), "pitch": a[:, 1].mean(),
            "knee": a[:, 2].mean(), "knee_asym": a[:, 3].mean(),
            "knee_asym_max": a[:, 3].max()}


def trial(checkpoint):
    world = World(Body(CFG))
    rt = Runtime(world)
    rt.stance = StancePolicy(world.model, REPO / "third_party/policies" / checkpoint,
                             rt.body.cfg["locomotion"]["stand_angles"])
    rt.stand_still()
    lf = world.model.body("left_ankle_roll_link").id
    rf = world.model.body("right_ankle_roll_link").id

    idle, reaching, poses = [], [], []

    def watch_idle(t):
        idle.append((world.data.xpos[lf][2], world.data.xpos[rf][2]))
        poses.append(posture_sample(world))

    rt.on_frame = watch_idle
    home = rt.base_xy.copy()
    rt.step(4.0)
    drift_idle = float(np.linalg.norm(rt.base_xy - home))
    pose = summarise(poses)

    rt.on_frame = lambda t: reaching.append(
        (world.data.xpos[lf][2], world.data.xpos[rf][2]))
    got = execute("grasp", rt, world, object="green block")
    rt.on_frame = None
    return {
        "posture": pose,
        "idle": foot_stats(idle, rt.dt),
        "reach": foot_stats(reaching, rt.dt),
        "drift_idle": drift_idle,
        "drift_reach": float(np.linalg.norm(rt.base_xy - home)),
        "grasped": bool(got.get("ok")),
        "upright": bool(world.data.xpos[rt.root_id][2] > 0.55),
    }


def main():
    print("| policy | foot lifts, idle | foot lifts, reaching | drift idle | "
          "drift through reach | knee bend | knee asymmetry | pelvis pitch | "
          "grasped |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, ck in POLICIES.items():
        r = trial(ck)
        p = r["posture"]
        print(f"| {name} | {r['idle']['lifts_per_s']:.1f}/s | "
              f"{r['reach']['lifts_per_s']:.1f}/s | "
              f"{r['drift_idle'] * 100:.1f} cm | "
              f"{r['drift_reach'] * 100:.1f} cm | "
              f"{p['knee']:.0f} deg | {p['knee_asym']:.0f} deg "
              f"(max {p['knee_asym_max']:.0f}) | "
              f"{p['pitch']:+.0f} deg | "
              f"{'yes' if r['grasped'] else 'NO'} |")
    print("| *this body's home pose* | 0/s | 0/s | | | 38 deg | 0 deg | "
          "0 deg | |")
    print("| *a person* | ~0/s | ~0/s | | | ~5 deg | ~0 deg | ~0 deg | |")


if __name__ == "__main__":
    main()
