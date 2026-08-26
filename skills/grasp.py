import numpy as np

SCHEMA = {
    "name": "grasp",
    "description": "Pick up a named object: approach from above, attach, lift.",
    "parameters": {
        "type": "object",
        "properties": {
            "object": {"type": "string",
                       "description": "object name, e.g. 'red block'"},
            "arm": {"type": "string", "enum": ["left", "right"]},
        },
        "required": ["object"],
    },
}

APPROACH = 0.12
TOUCH = 0.055   # above object center: hover just over the top surface,
LIFT = 0.15     # the weld does the holding (honest v1 limitation)
ATTACH_DIST = 0.075


def run(rt, world, object, arm=None):
    from skills.registry import resolve_object
    bname = resolve_object(world, object)
    if bname is None:
        return {"ok": False, "error": f"no object named '{object}'"}
    if bname not in world.graspable:
        return {"ok": False, "error": f"'{object}' is not graspable"}
    pos = world.object_pos(bname)
    arm = arm or rt.nearest_arm(pos)
    if rt.held(arm):
        return {"ok": False, "error": f"{arm} hand is already holding "
                                      f"{world.objects[rt.held(arm)]}"}

    rt.move_arm_ready(arm)
    # safe-height waypoint: travel above tabletop clutter, then descend
    rt.servo_ee(arm, [pos[0], pos[1], pos[2] + APPROACH + 0.06],
                speed=0.18, tol=0.05, timeout=4.0)
    if not rt.servo_ee(arm, pos + [0, 0, APPROACH]):
        return {"ok": False, "arm": arm, "error": "could not reach above object"}
    rt.servo_ee(arm, world.object_pos(bname) + [0, 0, TOUCH], speed=0.08)
    if np.linalg.norm(rt.ee_pos(arm) - world.object_pos(bname)) > ATTACH_DIST:
        return {"ok": False, "arm": arm, "error": "hand did not get close enough"}
    rt.attach(arm, bname)
    rt.servo_ee(arm, world.object_pos(bname) + [0, 0, LIFT])
    return {"ok": True, "arm": arm, "holding": object}
