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

    # safe-height waypoint: travel above tabletop clutter, then descend
    rt.servo_ee(arm, [tpos[0], tpos[1], tpos[2] + HOVER + 0.08],
                speed=0.18, tol=0.05, timeout=4.0)
    if not rt.servo_ee(arm, tpos + [0, 0, HOVER]):
        return {"ok": False, "arm": arm, "error": "could not reach above target"}
    rt.servo_ee(arm, tpos + [0, 0, DROP], speed=0.08)
    rt.release(arm)
    rt.step(0.4)  # let it settle
    rt.servo_ee(arm, rt.ee_pos(arm) + [0, 0, RETRACT])
    return {"ok": True, "arm": arm, "placed": held_name}
