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
RETRACT = 0.15


def run(rt, world, target, arm=None):
    from skills.registry import resolve_object
    if isinstance(target, str):
        bname = resolve_object(world, target)
        if bname is None:
            return {"ok": False, "error": f"no object named '{target}'"}
        tpos = world.object_pos(bname)
    else:
        tpos = np.asarray(target, dtype=float)

    if arm is None:
        arm = next((s for s in rt.arm if rt.held(s)), None)
    if arm is None or not rt.held(arm):
        return {"ok": False, "error": "not holding anything"}
    held_name = world.objects[rt.held(arm)]

    rt.square_up()
    # safe-height waypoint: travel above tabletop clutter, then descend
    rt.servo_ee(arm, [tpos[0], tpos[1], tpos[2] + HOVER + 0.08],
                speed=0.18, tol=0.05, timeout=4.0)
    if not rt.servo_ee(arm, tpos + [0, 0, HOVER]):
        # graceful degradation: don't strand the object in a stuck hand.
        # Set it down at a neutral mid-table spot both arms can reach and
        # tell the brain the single next step that finishes the job.
        held = rt.held(arm)
        drop = [world.table_x - 0.08, 0.0, world.table_top + DROP]
        rt.servo_ee(arm, [drop[0], drop[1], drop[2] + HOVER], speed=0.18,
                    tol=0.05, timeout=4.0)
        rt.servo_ee(arm, drop, speed=0.1, tol=0.03, timeout=3.0)
        rt.release(arm)
        rt.step(0.3)
        rt.recover(arm)
        set_down = world.object_pos(held).round(3).tolist() if held else None
        return {"ok": False, "arm": arm,
                "error": f"target out of reach for the {arm} arm; "
                         f"{held_name} was set down at {set_down}. "
                         f"Grasp it with the other arm and place it."}
    rt.servo_ee(arm, tpos + [0, 0, DROP], speed=0.08)
    rt.release(arm)
    rt.step(0.4)  # let it settle
    rt.servo_ee(arm, rt.ee_pos(arm) + [0, 0, RETRACT])
    return {"ok": True, "arm": arm, "placed": held_name}
