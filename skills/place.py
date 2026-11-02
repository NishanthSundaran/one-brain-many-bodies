import numpy as np

SCHEMA = {
    "name": "place",
    "description": "Place the held object onto a named object (e.g. 'bin') or "
                   "at a 3D position. Releases the object and retracts.",
    "parameters": {
        "type": "object",
        "properties": {
            "target": {"type": ["string", "array"],
                       "description": "object name or [x, y, z]"},
            "arm": {"type": "string", "enum": ["left", "right"]},
        },
        "required": ["target"],
    },
}

HOVER = 0.10
DROP = 0.07
DROP_CONTAINER = 0.07   # just over the rim: dropped from higher it bounces out
RETRACT = 0.15
OVER_TOL = 0.02     # held object this close to a container's centre: drop
OVER_MAX = 0.04     # ... and never let go further out than this
OVER_TRIES = 6
OVER_GAIN = 0.6


CLEARANCE = 0.10


def _avoid_objects(rt, world, tpos, held):
    """Collision-aware placement: if a coordinate target would drop the
    object onto another one, shift it sideways to a clear spot."""
    note = None
    for bname in world.objects:
        if bname == held or bname not in world.graspable:
            continue
        obs = world.object_pos(bname)
        d = tpos[:2] - obs[:2]
        dist = np.linalg.norm(d)
        if dist < CLEARANCE - 1e-9:
            direction = d / dist if dist > 1e-6 else np.array([0.0, 1.0])
            tpos = tpos.copy()
            tpos[:2] = obs[:2] + direction * CLEARANCE
            note = (f"placement shifted to {tpos.round(3).tolist()} to avoid "
                    f"dropping onto {world.objects[bname]}")
    return tpos, note


def run(rt, world, target, arm=None):
    from skills.registry import resolve_object
    note = None
    drop = DROP
    if isinstance(target, str):
        bname = resolve_object(world, target)
        if bname is None:
            return {"ok": False, "error": f"no object named '{target}'"}
        tpos = world.object_pos(bname)
        drop = DROP_CONTAINER
    else:
        tpos = np.asarray(target, dtype=float)
        held_body = next((rt.held(s) for s in rt.arm if rt.held(s)), None)
        tpos, note = _avoid_objects(rt, world, tpos, held_body)

    if arm is None:
        arm = next((s for s in rt.arm if rt.held(s)), None)
    if arm is None or not rt.held(arm):
        return {"ok": False, "error": "not holding anything"}
    held_name = world.objects[rt.held(arm)]
    if not rt.arm_can_reach(arm, tpos):
        # same graceful degradation as an out-of-reach servo, but decided
        # up front: set the object down mid-table without any flailing
        down_spot = rt.set_down_spot(world)
        rt.square_up()
        rt.plant()
        rt.servo_ee(arm, [down_spot[0], down_spot[1], down_spot[2] + HOVER],
                    speed=0.18, tol=0.05, timeout=4.0)
        rt.servo_ee(arm, down_spot, speed=0.1, tol=0.03, timeout=3.0)
        rt.set_hand(arm, "open")
        held = rt.held(arm)
        rt.release(arm)
        rt.step(0.3)
        rt.recover(arm)
        set_down = world.object_pos(held).round(3).tolist() if held else None
        return {"ok": False, "arm": arm,
                "error": f"target y={tpos[1]:.2f} is outside the {arm} arm's "
                         f"workspace; {held_name} was set down at {set_down}. "
                         f"Grasp it with the other arm and place it."}

    rt.square_up()
    rt.plant()
    hover_h = drop + 0.05
    aim = None
    if rt.loco is not None and isinstance(target, str):
        # Aim what is held, not the hand, when dropping into a container. The grasp on a balancing body
        # accepts a palm up to 8 cm off the object's centre, and the hold
        # keeps that offset, so a hand centred over the bin drops the block
        # beside it (measured: 9.5 cm short, on the bin's wall, 8 of 8).
        # Height too: the palm hovers above what it holds, so the hand
        # "just over the rim" puts the block below it, into the wall.
        held_body = rt.held(arm)
        aim = tpos.copy()
        grip = rt.ee_pos(arm) - world.object_pos(held_body)
        tpos = tpos + grip
    elif rt.loco is None:
        # Same idea for a fixed base, sideways only (the drop heights are tuned
        # for the hand): a grasp taken off-centre, as after a handover at the
        # midline, keeps its offset in the hold, and aiming the hand then
        # set the object down ~5 cm from the target (measured on cups).
        grip = rt.ee_pos(arm) - world.object_pos(rt.held(arm))
        tpos = tpos + np.array([grip[0], grip[1], 0.0])
    # safe-height waypoint: travel above tabletop clutter, then descend
    patience = 16.0 if rt.loco is not None else 4.0
    balancing = rt.loco is not None
    if balancing:
        # Same reason as grasp: solve and move once, rather than running a
        # Cartesian servo whose constant re-planning is a disturbance the
        # legs cannot settle against.
        ok = rt.reach_joint_space(arm, tpos + [0, 0, hover_h])
    else:
        # One blended motion over and down to the release point: the drop
        # used to be a separate 3 cm primitive with its own speed profile.
        ok = rt.reach_via(arm, [[tpos[0], tpos[1], tpos[2] + hover_h + 0.08],
                                tpos + [0, 0, hover_h],
                                tpos + [0, 0, drop]])
    if not ok:
        # graceful degradation: don't strand the object in a stuck hand.
        # Set it down at a neutral mid-table spot both arms can reach and
        # tell the brain the single next step that finishes the job.
        held = rt.held(arm)
        down_spot = rt.set_down_spot(world)
        rt.servo_ee(arm, [down_spot[0], down_spot[1], down_spot[2] + HOVER],
                    speed=0.18, tol=0.05, timeout=4.0)
        rt.servo_ee(arm, down_spot, speed=0.1, tol=0.03, timeout=3.0)
        rt.set_hand(arm, "open")
        rt.release(arm)
        rt.step(0.3)
        rt.recover(arm)
        set_down = world.object_pos(held).round(3).tolist() if held else None
        return {"ok": False, "arm": arm,
                "error": f"target out of reach for the {arm} arm; "
                         f"{held_name} was set down at {set_down}. "
                         f"Grasp it with the other arm and place it."}
    if balancing:
        rt.reach_joint_space(arm, tpos + [0, 0, drop], duration=1.2,
                             seed="current")
    if balancing and aim is not None:
        # Release on where the object is, not where the hand was sent. The
        # joint-space reach accepts a hand up to 10 cm off, and the bin is
        # 13 cm across, so the block was let go over the rim or past the
        # bench edge (measured 8 to 17 cm off centre in every miss). Nudge
        # the hand by the object's own error until it is over the middle.
        for _ in range(OVER_TRIES):
            off = aim[:2] - world.object_pos(rt.held(arm))[:2]
            if np.linalg.norm(off) < OVER_TOL:
                break
            # Damped and then settled: the base sways a few centimetres
            # with every arm move, and full-size corrections chased that
            # sway back and forth (measured: +/-3 to 6 cm, never converging).
            rt.reach_joint_space(arm, rt.ee_pos(arm) + [*(OVER_GAIN * off), 0.0],
                                 duration=1.0, tol=0.01, seed="current",
                                 passes=1)
            rt.step(0.4)
        off = aim[:2] - world.object_pos(rt.held(arm))[:2]
        if np.linalg.norm(off) > OVER_MAX:
            return {"ok": False, "arm": arm,
                    "error": f"could not get {held_name} over {target} "
                             f"({np.linalg.norm(off):.2f} m off); still "
                             f"holding it"}
    rt.set_hand(arm, "open")
    rt.release(arm)
    rt.step(0.4)  # let it settle
    if balancing:
        rt.reach_joint_space(arm, rt.ee_pos(arm) + [0, 0, RETRACT],
                             duration=1.2, seed="current")
    else:
        rt.reach_line(arm, rt.ee_pos(arm) + [0, 0, RETRACT])
    result = {"ok": True, "arm": arm, "placed": held_name}
    if note:
        result["note"] = note
    return result
