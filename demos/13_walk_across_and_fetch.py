"""Walk across the room to the bench, pick the block up, drop it in the bin.

Demo 08 starts at the bench and never travels further than 0.3 m, because
the stance policy that moves it falls at 0.5 m and beyond. Here the robot
starts 2.8 m away and covers the distance under the walking policy, steered
by controllers/navigator.py, then hands the legs to the stance policy for
the last few decimetres and the manipulation:

  walk      gait_travel to a point about a metre short of the bench. The
            walking policy cannot stop on a mark: told to stop, it keeps
            going for ~0.9 m (eval/score_handover.py), so stopping has to be
            decided early.
  handover  gait_stop: command eased to zero, legs given to the stance
            policy once the body is slow with both feet down.
  position  step_to the work spot, a short stance-policy move.
  fetch     the existing grasp and place skills.

Every trial is scored and recorded from the same rollout.

    python demos/13_walk_across_and_fetch.py                # one trial, GIF
    python demos/13_walk_across_and_fetch.py --trials 10    # success rate
"""

import argparse
import pathlib
import sys
from multiprocessing import Pool

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402

from bodies.body import Body  # noqa: E402
from controllers.locomotion import POLICY_JOINTS  # noqa: E402
from sim.cameras import CameraRig  # noqa: E402
from sim.world import World  # noqa: E402
from skills.registry import execute  # noqa: E402
from skills.runtime import Runtime  # noqa: E402

CFG = REPO / "bodies" / "g1_walk" / "config.yaml"
RECORD_FPS = 10
PLAY_FPS = 20
# The right hand reaches 0.20 to 0.35 m right of the pelvis and 0.26 to
# 0.38 m ahead; nearer the midline the upper arm hits the torso. The block
# and bin sit diagonally inside that band, relative to the work spot.
# Everything sits mid-bench, 0.3 m in from the end it used to be at: there
# the right foot stood beside a bench leg and kicked it on every step.
WORK_SPOT = np.array([0.0, 0.15])
LINE_Y = 0.22                             # stops ~0.07 m right of its line
START = np.array([-2.8, LINE_Y])          # 2.8 m back, on the line
STOP_DECIDED = np.array([-1.05, LINE_Y])
STOP_WAIT = 3.0
HOLD = "walk"
WALK_ONLY = False
PLANT = False
IK_LIMITS = True
STEPPER = None
READY_PATH = True
VIEW = "front"   # "side" shows the whole walk from 3.8 m
LOOKAHEAD = 1.0
SETTLE = 1.0
# One stance reaches both, so the robot does not step across between
# the pick and the drop: the bin sits nearer the body (about 0.25 m to
# the right, where the hand reaches at drop height) and the block further
# out (0.35 m, reachable at bench height). Measured 16/20 against 13/20
# with a step between them, and in about 35 s instead of 60.
BIN_X, BIN_Y = 0.35, -0.095
OBJECTS = [
    {"name": "green block", "kind": "box", "size": [0.025, 0.025, 0.025],
     "rgba": [0.15, 0.65, 0.2, 1], "pos": [0.29, -0.20]},
    {"name": "bin", "kind": "bin", "size": [0.065, 0.065, 0.035],
     "rgba": [0.45, 0.45, 0.5, 1], "pos": [BIN_X, BIN_Y]},
]
JITTER = 0.02
# block position relative to the pelvis the robot lines up to
LINE_UP = np.array([0.27, -0.35])
PLACE_UP = np.array([0.30, -0.30])
RESTEP = False
# the line-up puts the block about 0.35 m right of the pelvis, where the
# arm reaches without rolling into the torso; grasp must not step it back
APPROACH_LATERAL = 0.46
LINEUP_STEP = True   # step to LINE_UP after the walk stops
# after the release: lower the empty arm back to the side, so the robot
# does not finish with its hand held up where the retract left it
END_POSE = "home"
END_TIME = 3.0
# human arm motion: minimum-jerk profile, via poses passed through without
# stopping (needs a stance policy trained against it: training/arm_traj.npz)
SMOOTH_ARM = True
# --live: run in MuJoCo's interactive window at this multiple of real time
LIVE = None


def caption(frame, lines):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:
        font = ImageFont.load_default()
    h = 8 + 19 * len(lines)
    draw.rectangle([0, frame.shape[0] - h, frame.shape[1], frame.shape[0]],
                   fill=(0, 0, 0))
    for i, (text, color) in enumerate(lines):
        draw.text((8, frame.shape[0] - h + 3 + 19 * i), text, fill=color,
                  font=font)
    return np.asarray(img)


def trial(args):
    seed, record, stance, policy = args
    rng = np.random.default_rng(seed)
    body = Body(CFG)
    if stance:
        body.cfg["locomotion"]["stance_policy"] = stance
    if STEPPER:
        body.cfg["locomotion"]["step_policy"] = STEPPER
    hold_walk = bool(policy)   # the walker stops itself; HOLD picks who holds
    if policy:
        body.cfg["locomotion"]["policy"] = policy
        body.cfg["locomotion"]["hold_with"] = "walk_policy"
        # The upper arm meets the torso at ~0.06 rad of inward shoulder
        # roll (the joint allows 2.25); without this the IK picks poses
        # jammed against the chest and the hand lands 10 to 30 cm off.
        # Not in the body config: demo 08's placement relies on the old IK.
        if IK_LIMITS:
            body.cfg["arms"]["left"]["ik_limits"] = {
                "left_shoulder_roll_joint": [0.0, 2.25]}
            body.cfg["arms"]["right"]["ik_limits"] = {
                "right_shoulder_roll_joint": [-2.25, 0.0]}
        # arm gravity fed forward, so the wrist does not sag off target
        body.cfg["locomotion"]["arm_gravity_comp"] = True
        body.cfg["locomotion"]["plant_to_manipulate"] = PLANT
        if READY_PATH:
            # the default ready pose puts the wrist into the bench top at
            # this standing height, and the swing to it from a hanging arm
            # sweeps through the bench edge. Fold the elbow first (hand up
            # by the chest), swing the upper arm back and out, then come in
            # over the bench: 2.8 cm clearance at the closest, where the
            # earlier single waypoint dragged the fingers 4.3 cm into the
            # bench top (checked kinematically from the real standing pose)
            body.cfg["arms"]["right"]["ready_via"] = [
                [0.5, -0.5, -0.02, 1.9, 0.03, 0.07, 0.1],
                [0.5, -1.2, -0.6, 0.6, 0.4, 1.2, 0.5]]
            body.cfg["arms"]["right"]["ready"] = [
                -0.39, -1.04, -0.27, -0.61, 0.55, 1.61, 0.31]
    world = World(body, objects=OBJECTS, start_xy=START)
    m, d = world.model, world.data
    for n in POLICY_JOINTS:
        d.qpos[m.joint(n).qposadr[0]] += rng.uniform(-JITTER, JITTER)
    mujoco.mj_forward(m, d)

    state = {"frames": [], "next_t": 0.0, "label": "walking to the bench"}
    rig = CameraRig(m, width=620, height=440) if record else None

    def on_frame(t):
        if t >= state["next_t"]:
            z = d.xpos[rt.root_id][2]
            if VIEW == "front":
                # the demo 04 view of the bench, pulled back and following
                # the robot while it is still walking in
                far = max(0.0, -float(d.xpos[rt.root_id][0]))
                rx, ry = d.xpos[rt.root_id][:2]
                k = min(1.0, far / 2.5)
                frame = rig.third_person(
                    d, azimuth=-155, distance=1.9 + 1.6 * k, elevation=-16,
                    lookat=((1 - k) * 0.2 + k * (rx + 0.6) * 0.8,
                            (1 - k) * 0.0 + k * ry, 0.8 - 0.1 * k))
            else:
                frame = rig.third_person(d, azimuth=90, distance=3.8,
                                         lookat=(-1.2, 0.1, 0.6), elevation=-10)
            state["frames"].append(caption(frame, [
                (state["label"], (120, 220, 255)),
                (f"t {t:4.1f} s   pelvis {z:.2f} m", (225, 225, 225)),
            ]))
            state["next_t"] += 1.0 / RECORD_FPS

    rt = Runtime(world, start="walk")
    if SMOOTH_ARM:
        rt.arm_profile, rt.blend_vias = "min_jerk", True
    p = rng.uniform(-np.pi, np.pi)
    rt.loco.phase = np.array([p, np.fmod(p + 2 * np.pi, 2 * np.pi) - np.pi])
    out = {"seed": seed, "stage": "walk"}
    # Bench watch, every physics step of the trial: any robot-bench contact,
    # and (sampled) the closest any robot geom comes to the bench.
    bench = [g for g in range(m.ngeom)
             if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
             .startswith("table")]
    robot_root = m.body_rootid[rt.root_id]
    mine = [g for g in range(m.ngeom)
            if m.body_rootid[m.geom_bodyid[g]] == robot_root
            and m.geom_contype[g] | m.geom_conaffinity[g]]
    out["bench_contacts"] = 0
    out["bench_clearance"] = np.inf
    out["bench_force_max"] = 0.0
    out["torso_tilt_max"] = 0.0   # degrees off vertical, while standing
    torso = m.body("torso_link").id
    f6 = np.zeros(6)
    nstep = {"n": 0}

    def watch(t):
        for i, c in enumerate(d.contact[:d.ncon]):
            if (c.geom1 in bench) != (c.geom2 in bench) and (
                    c.geom1 in mine or c.geom2 in mine):
                out["bench_contacts"] += 1
                mujoco.mj_contactForce(m, d, i, f6)
                out["bench_force_max"] = max(out["bench_force_max"],
                                             float(f6[0]))
                break
        if rt._standing and nstep["n"] % 25 == 0:
            out["torso_tilt_max"] = max(out["torso_tilt_max"], float(
                np.degrees(np.arccos(np.clip(d.xmat[torso][8], -1, 1)))))
        nstep["n"] += 1
        if rt._standing and nstep["n"] % 25 == 0:
            for g in mine:
                for b in bench:
                    dist = mujoco.mj_geomDistance(m, d, g, b, 0.3, None)
                    out["bench_clearance"] = min(out["bench_clearance"], dist)
        if record:
            on_frame(t)

    rt.on_frame = watch
    if LIVE:
        # Interactive window for presenting: physics paced to the wall
        # clock, the current step shown in the corner. Drag to orbit,
        # right-drag to pan, scroll to zoom.
        import time
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(m, d, show_left_ui=False,
                                              show_right_ui=False)
        viewer.cam.azimuth, viewer.cam.elevation = -155, -16
        viewer.cam.distance, viewer.cam.lookat[:] = 3.2, (-0.3, 0.0, 0.8)
        live = {"t0": time.time(), "next": 0.0}
        record_watch = watch

        def watch(t):
            record_watch(t)
            if t < live["next"]:
                return
            live["next"] = t + 1.0 / 60.0
            ahead = t / LIVE - (time.time() - live["t0"])
            if ahead > 0:
                time.sleep(ahead)
            try:
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,
                                  mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  state["label"], f"t {t:4.1f} s"))
            except Exception:
                pass
            viewer.sync()
        rt.on_frame = watch
        out["viewer"] = viewer
    block0 = world.object_pos("obj_green_block").copy()

    def finish(stage):
        out["stage"] = stage
        out["upright"] = not rt.fallen()
        out["t"] = rt._t
        if record:
            state["label"] = stage
            if not (WALK_ONLY and stage == "stopped at the bench"):
                rt.step(1.0)   # walk-only already watched it for 4 s
            out["frames"] = state["frames"]
        return out

    # follow the line y = STOP_DECIDED[1]: aim 1 m ahead on it, so the
    # steering never reaches its freeze radius and lateral drift is corrected
    def aim(r):
        return np.array([r.base_xy[0] + LOOKAHEAD, STOP_DECIDED[1]])
    if not rt.gait_travel(aim, arrive=0.05,
                          until=lambda r: r.base_xy[0] >= STOP_DECIDED[0]):
        return finish("fell walking" if rt.fallen() else "walk timeout")
    out["decided_at"] = rt.base_xy.round(3).tolist()
    state["label"] = "stopping"
    if hold_walk:
        # the fine-tuned walking policy stops by itself and keeps the legs
        v0 = rt.loco.command.copy()
        tick = rt.loco.n_substeps * rt.dt
        for k in range(1, 31):
            rt.loco.set_command(*(v0 * (1 - k / 30)))
            rt.step(tick)
        waited = 0.0
        while waited < STOP_WAIT:
            rt.step(tick)
            waited += tick
            if np.linalg.norm(d.qvel[:2]) < 0.05 or rt.fallen():
                break
        out["stop_wait"] = round(waited, 2)
        if HOLD == "stance":
            # arms frozen where they are, then the stance policy takes the
            # legs holding this spot (the walker has already stopped)
            for arm in rt.arm.values():
                d.ctrl[arm["act"]] = [d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
                                      for i in arm["act"]]
            rt.hold_with_walk = False
        rt.stand_still()
    else:
        out["handover"] = rt.gait_stop()
    if rt.fallen():
        return finish("fell stopping")
    rt.step(SETTLE)
    if rt.fallen():
        return finish("fell after stopping")
    out["stopped_at"] = rt.base_xy.round(3).tolist()
    yaw_stop = rt.base_yaw
    out["work_spot_error"] = float(np.linalg.norm(rt.base_xy - WORK_SPOT))
    if WALK_ONLY:
        state["label"] = "standing at the bench"
        # scored over everything the GIF shows after the stop, so its
        # last frame is covered by the same numbers
        # foot lifts while standing: a foot rising over 1 cm off the floor
        feet = [m.body(f).id for f in ("left_ankle_roll_link",
                                       "right_ankle_roll_link")]
        z0 = [d.xpos[f][2] for f in feet]
        up, lifts = [False, False], 0
        for _ in range(200):
            rt.step(0.02)
            for k, f in enumerate(feet):
                now = d.xpos[f][2] > z0[k] + 0.01
                lifts += now and not up[k]
                up[k] = now
        out["foot_lifts"] = int(lifts)
        out["drift"] = float(np.linalg.norm(rt.base_xy - out["stopped_at"]))
        out["turn_deg"] = float(np.degrees(rt._wrap(rt.base_yaw - yaw_stop)))
        if rt.fallen():
            return finish("fell standing")
        return finish("stopped at the bench")

    if not hold_walk or (HOLD == "stance" and LINEUP_STEP):
        state["label"] = "stepping up to the bench"
        # line up so the block sits where the right hand reaches without
        # rolling the upper arm into the torso (0.25 to 0.40 m right)
        spot = (WORK_SPOT if not hold_walk else
                world.object_pos("obj_green_block")[:2] - LINE_UP)
        # one stride at most: the stepper cannot make corrections under
        # about 5 cm (it treats them as arrived), and retrying them is
        # what kept its feet shuffling at the bench
        placed = rt.step_to(spot, yaw=0.0, tol=0.05, max_steps=2)
        out["positioned"] = bool(placed)
        if rt.fallen():
            return finish("fell positioning")

    state["label"] = "picking up the block"
    if HOLD == "stance":
        rt.approach_lateral = APPROACH_LATERAL
    got = execute("grasp", rt, world, object="green block")
    out["grasp"] = bool(got.get("ok"))
    out["grasp_error"] = got.get("error")
    if rt.fallen() or not got.get("ok"):
        return finish("fell grasping" if rt.fallen() else "grasp failed")

    state["label"] = "dropping it in the bin"
    if hold_walk and HOLD == "stance" and RESTEP:
        # the bin sits a hand's width further right than the block; step
        # across so it lands where the block did, instead of reaching for
        # it at the edge of the arm's range
        rt.step_to(world.object_pos("obj_bin")[:2] - PLACE_UP, yaw=0.0,
                   tol=0.04)
        if rt.fallen():
            return finish("fell stepping to the bin")
    put = execute("place", rt, world, target="bin")
    out["place"] = bool(put.get("ok"))
    out["place_error"] = put.get("error")
    if put.get("ok") and END_POSE and not rt.fallen():
        state["label"] = "lowering the arm"
        a = rt.arm[put["arm"]]
        if a["home_ctrl"] is not None:
            # back the way it came: to the ready pose over the bench, then
            # the ready route in reverse. Straight down to hanging drags
            # the hand across the bench top.
            rt._interp_ctrl(put["arm"], body.cfg["arms"][put["arm"]]["ready"],
                            END_TIME * 0.4)
            rt.move_arm_home(put["arm"], duration=END_TIME * 1.2)
    rt.step(1.5)
    end = world.object_pos("obj_green_block")
    bin_xy = world.object_pos("obj_bin")[:2]
    # resting inside: within the walls, below the rim, and let go of (a
    # block still held over the bin passes the first two)
    bin_z = world.object_pos("obj_bin")[2]
    out["in_bin"] = bool(np.all(np.abs(end[:2] - bin_xy) < 0.065)
                         and end[2] > world.table_top
                         and end[2] - bin_z < 0.06
                         and not any(rt.held(s) for s in rt.arm))
    out["block_moved"] = float(np.linalg.norm(end[:2] - block0[:2]))
    out["block_vs_bin"] = (end - world.object_pos("obj_bin")).round(3).tolist()
    if rt.fallen():
        return finish("fell placing")
    return finish("done" if out["in_bin"] else "missed the bin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--stance", default=None,
                    help="stance checkpoint, relative to the repo")
    ap.add_argument("--record", default="all", choices=["all", "first", "none"])
    ap.add_argument("--policy", default=None,
                    help="fine-tuned walking policy (.pt) that stops and "
                         "holds by itself; no stance handover")
    ap.add_argument("--hold", default="walk", choices=["walk", "stance"],
                    help="with --policy: who holds the robot while it works")
    ap.add_argument("--decide", type=float, default=None,
                    help="x where stopping is decided")
    ap.add_argument("--outdir", default=None,
                    help="where per-trial GIFs go (default demos/media)")
    ap.add_argument("--plant", action="store_true",
                    help="stiffen the legs while the arm works (measured "
                         "worse: shoves the pelvis forward 0.1 m)")
    ap.add_argument("--walk-only", action="store_true",
                    help="walk, stop at the work spot, stand 3 s; no fetch")
    ap.add_argument("--stepper", default=None,
                    help="stance checkpoint that takes the short line-up "
                         "steps, when --stance keeps its feet planted")
    ap.add_argument("--no-ik-limits", action="store_true",
                    help="let the IK roll the upper arm into the torso")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="with --live: which trial seed to run (0 = 9000)")
    ap.add_argument("--live", type=float, nargs="?", const=1.0, default=None,
                    help="watch it in MuJoCo's interactive window, paced to "
                         "real time (optionally a speed-up, e.g. --live 2)")
    ap.add_argument("--view", default="front", choices=["front", "side"],
                    help="front: follow the robot in, then frame the bench "
                         "like demo 04; side: the whole walk from 3.8 m")
    a = ap.parse_args()
    global HOLD, WALK_ONLY, PLANT, IK_LIMITS, STEPPER, VIEW, LIVE
    VIEW = a.view
    LIVE = a.live
    STEPPER = a.stepper
    IK_LIMITS = not a.no_ik_limits
    HOLD = a.hold
    WALK_ONLY = a.walk_only
    PLANT = a.plant
    if a.decide is not None:
        STOP_DECIDED[0] = a.decide
    seeds = [9000 + 23 * k for k in range(a.trials)]
    rec = [a.record == "all" or (a.record == "first" and k == 0)
           for k in range(a.trials)]
    if LIVE:
        # one trial, in this process, with the window; no GIF written
        r = trial((9000 + 23 * a.seed_offset, False, a.stance, a.policy))
        viewer = r.pop("viewer")
        print({k: v for k, v in r.items()})
        print("done. close the window to exit")
        while viewer.is_running():
            viewer.sync()
            import time
            time.sleep(0.05)
        return
    with Pool(min(10, a.trials), maxtasksperchild=1) as pool:
        rows = pool.map(trial, [(s, r, a.stance, a.policy)
                                for s, r in zip(seeds, rec)])
    media = REPO / "demos" / "media"
    ok = clean = 0
    for r in rows:
        success = r["upright"] and (
            r["stage"] == "done" or (WALK_ONLY and r["stage"] == "stopped at the bench"
                                     and r["work_spot_error"] < 0.15
                                     and r["drift"] < 0.05
                                     and abs(r["turn_deg"]) < 5.0
                                     and r["foot_lifts"] == 0
                                     and r["bench_contacts"] == 0))
        ok += success
        # clean: also never touched the bench (any body part, hands too)
        # and stood up straight (torso within 10 deg of vertical)
        clean += success and r["bench_contacts"] == 0 and (
            r["torso_tilt_max"] < 10.0)
        frames = r.pop("frames", None)
        print({k: v for k, v in r.items()})
        if frames:
            name = ("13_walk_and_stop.gif" if WALK_ONLY and a.trials == 1
                    else "13_walk_across_and_fetch.gif" if a.trials == 1
                    else f"13_trial_{r['seed']}.gif")
            path = (media if a.trials == 1 else
                    pathlib.Path(a.outdir or media)) / name
            imageio.mimsave(path, frames, fps=PLAY_FPS, loop=0)
            print(f"  wrote {path} ({len(frames)} frames)")
    goal = ("stopped within 0.15 m of the work spot, then held it (under "
            "5 cm and 5 deg over 4 s, no foot lifted, bench untouched), "
            "upright" if WALK_ONLY
            else "block in the bin, robot upright at the end")
    print(f"\n{goal}: {ok}/{len(rows)}")
    print(f"... and clean (bench never touched, torso within 10 deg of "
          f"vertical): {clean}/{len(rows)}")


if __name__ == "__main__":
    main()
