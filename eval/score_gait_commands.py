"""Which commands can the borrowed walking policy actually follow?

The pretrained joystick policy (MuJoCo Playground) walks straight well
(demos/07) and has never been steered to a point on this body: whenever a
stance policy is loaded, Runtime.walk_to navigates with the stance policy
instead. Before building a navigator on top of it, map the command envelope
it can follow: hold a constant (vx, wz) for 5 s from a clean walking start,
five jittered trials per command, and record what the robot really did.

Two plants, because a policy is only valid for the plant it learned on:

  A  this project's g1_walk body as it runs today: 44 colliding geoms,
     including shins, knees, hips, pelvis and arms
  B  the same body with collisions filtered the way Playground trained it:
     only the feet touch the floor (and each other). Legs that swing close
     during a turn pass through each other, as they did in training.

Gains, damping, armature, Euler integration, timestep (0.002 s) and solver
iterations already match training (bodies/g1_walk/config.yaml). Foot contact
geometry is near-equivalent: four point spheres per foot here, one box per
foot in training, corners within a few millimetres, same friction.

Steps are counted with a 4 cm lift threshold and a 0.1 s debounce. A 1-2 cm
threshold counts the stance policy's tremor as steps, which once reported a
cadence 3.6x human that was really 1.4x.

    python eval/score_gait_commands.py            # both plants, full grid
    python eval/score_gait_commands.py --plants B --trials 2   # quick look
"""

import argparse
import json
import pathlib
import sys
from multiprocessing import Pool

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402

from bodies.body import Body  # noqa: E402
from controllers.locomotion import POLICY_JOINTS, LocomotionPolicy  # noqa: E402

CFG = REPO / "bodies" / "g1_walk" / "config.yaml"
VX = (0.0, 0.25, 0.5, 0.75)
WZ = (-1.0, -0.5, 0.0, 0.5, 1.0)
SETTLE_S = 1.5
HOLD_S = 5.0
JITTER = 0.02
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
LIFT = 0.04
DEBOUNCE = 0.1


def fallen(data, pelvis):
    """Same check as demos/12: low, or tipped past ~45 degrees."""
    R = data.xmat[pelvis].reshape(3, 3)
    return data.xpos[pelvis][2] < 0.55 or R[2, 2] < 0.7


def build(plant):
    body = Body(CFG)
    w = body.spec.worldbody
    w.add_light(pos=[2, -2, 3], dir=[-0.4, 0.4, -1], diffuse=[0.7] * 3)
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
               size=[30, 30, 0.1])
    model = body.spec.compile()
    data = mujoco.MjData(model)
    body.bind(model)
    if plant == "B":
        keep = {model.body(n).id for n in FEET}
        floor = model.geom("floor").id
        for g in range(model.ngeom):
            if g != floor and model.geom_bodyid[g] not in keep:
                model.geom_contype[g] = 0
                model.geom_conaffinity[g] = 0
    return body, model, data


def trial(args):
    plant, vx, wz, seed = args
    rng = np.random.default_rng(seed)
    body, model, data = build(plant)
    body.reset(data)
    for n in POLICY_JOINTS:
        data.qpos[model.joint(n).qposadr[0]] += rng.uniform(-JITTER, JITTER)
    mujoco.mj_forward(model, data)

    cfg = body.cfg["locomotion"]
    pol = LocomotionPolicy(model, REPO / cfg["policy"], cfg["stand_angles"],
                           cfg["ctrl_dt"], cfg["action_scale"],
                           gait_freq=cfg["gait_freq"])
    p = rng.uniform(-np.pi, np.pi)
    pol.phase = np.array([p, np.fmod(p + 2 * np.pi, 2 * np.pi) - np.pi])

    dt = model.opt.timestep
    pelvis = model.body("pelvis").id
    feet = [model.body(n).id for n in FEET]

    def yaw():
        R = data.xmat[pelvis].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    pol.set_command(0.0, 0.0, 0.0)
    for _ in range(int(SETTLE_S / dt)):
        pol.step(data)
        mujoco.mj_step(model, data)
        if fallen(data, pelvis):
            return {"plant": plant, "vx": vx, "wz": wz, "seed": seed,
                    "fell_at": 0.0, "settle_fall": True}

    pol.set_command(vx, 0.0, wz)
    y0 = prev_yaw = yaw()
    turned, fwd, fell_at = 0.0, 0.0, None
    heights = []
    n = int(HOLD_S / dt)
    for i in range(n):
        xy0 = data.xpos[pelvis][:2].copy()
        pol.step(data)
        mujoco.mj_step(model, data)
        y = yaw()
        turned += (y - prev_yaw + np.pi) % (2 * np.pi) - np.pi
        prev_yaw = y
        d = data.xpos[pelvis][:2] - xy0
        fwd += d[0] * np.cos(y) + d[1] * np.sin(y)
        heights.append([data.xpos[f][2] for f in feet])
        if fallen(data, pelvis):
            fell_at = (i + 1) * dt
            break

    T = (len(heights)) * dt
    H = np.array(heights)
    steps = 0
    for k in range(2):
        up = H[:, k] > np.percentile(H[:, k], 5) + LIFT
        last = -1e9
        for j in np.flatnonzero(np.diff(up.astype(int)) == 1):
            if (j - last) * dt > DEBOUNCE:
                steps += 1
                last = j
    return {"plant": plant, "vx": vx, "wz": wz, "seed": seed,
            "fell_at": fell_at, "settle_fall": False,
            "vx_real": fwd / T, "wz_real": turned / T,
            "steps_per_s": steps / T, "yaw_total": turned, "t": T}


def summarise(rows, trials):
    out = {}
    for r in rows:
        out.setdefault((r["plant"], r["vx"], r["wz"]), []).append(r)
    table = []
    for (plant, vx, wz), rs in sorted(out.items()):
        falls = sum(r["fell_at"] is not None for r in rs)
        ok = [r for r in rs if not r.get("settle_fall")]
        vr = np.array([r["vx_real"] for r in ok]) if ok else np.array([np.nan])
        wr = np.array([r["wz_real"] for r in ok]) if ok else np.array([np.nan])
        st = np.array([r["steps_per_s"] for r in ok]) if ok else np.array([np.nan])
        table.append({
            "plant": plant, "vx": vx, "wz": wz, "falls": falls,
            "trials": len(rs),
            "vx_real": (float(vr.mean()), float(vr.std())),
            "wz_real": (float(wr.mean()), float(wr.std())),
            "steps": float(st.mean()),
            "sign_ok": (wz == 0 or bool(np.all(np.sign(wr) == np.sign(wz))))})
    return table


def print_tables(table):
    for plant in sorted({t["plant"] for t in table}):
        print(f"\n### plant {plant}\n")
        print("| vx cmd | wz cmd | falls | vx real | wz real | steps/s | turns right way |")
        print("|---|---|---|---|---|---|---|")
        for t in table:
            if t["plant"] != plant:
                continue
            print(f"| {t['vx']:.2f} | {t['wz']:+.1f} | {t['falls']}/{t['trials']} "
                  f"| {t['vx_real'][0]:.2f} +/- {t['vx_real'][1]:.2f} "
                  f"| {t['wz_real'][0]:+.2f} +/- {t['wz_real'][1]:.2f} "
                  f"| {t['steps']:.1f} | {'yes' if t['sign_ok'] else 'NO'} |")


def gate(table):
    """Gate 1, fixed before running: some plant has a fall-free envelope
    containing vx >= 0.3 with |wz| >= 0.5 in both directions."""
    verdict = {}
    for plant in sorted({t["plant"] for t in table}):
        safe = {(t["vx"], t["wz"]) for t in table
                if t["plant"] == plant and t["falls"] == 0}
        left = any(v >= 0.3 and w >= 0.5 for v, w in safe)
        right = any(v >= 0.3 and w <= -0.5 for v, w in safe)
        verdict[plant] = left and right
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plants", default="AB")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    jobs = [(p, vx, wz, 1000 * k + int(vx * 100) * 10 + int((wz + 1) * 2))
            for p in a.plants for vx in VX for wz in WZ
            for k in range(a.trials)]
    with Pool() as pool:
        rows = pool.map(trial, jobs)
    table = summarise(rows, a.trials)
    print_tables(table)
    v = gate(table)
    print("\nGate 1 (fall-free vx>=0.3 with |wz|>=0.5 both ways):",
          {p: ("PASS" if ok else "fail") for p, ok in v.items()})
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(
            {"rows": rows, "table": table, "gate": v}, indent=1, default=str))


if __name__ == "__main__":
    main()
