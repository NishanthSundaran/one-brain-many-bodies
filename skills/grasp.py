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
CLEAR = 0.03    # palm hovers this far above the object's top surface;
LIFT = 0.15     # the weld does the holding (honest v1 limitation)
ATTACH_TOP_DIST = 0.055


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
    if not rt.arm_can_reach(arm, pos):
        return {"ok": False, "arm": arm,
                "error": f"'{object}' is outside the {arm} arm's workspace; "
                         f"use the other arm"}

    rt.square_up()
    rt.plant()          # no-op unless this body plants to manipulate
    rt.move_arm_ready(arm)
    # A balancing robot holds its ground to about +/-0.2 m, so the target
    # drifts in and out of the arm's envelope. The servo is closed loop on
    # the object in world coordinates, so the answer is patience: keep
    # tracking and let the base wander back into range.
    patience = 16.0 if rt.loco is not None else 4.0
    rt.servo_ee(arm, [pos[0], pos[1], pos[2] + APPROACH + 0.06],
                speed=0.18, tol=0.05, timeout=patience)
    if not rt.servo_ee(arm, pos + [0, 0, APPROACH],
                       timeout=patience):
        rt.recover(arm)
        return {"ok": False, "arm": arm, "error": "could not reach above object"}
    half = world.object_half.get(bname, 0.03)
    rt.servo_ee(arm, world.object_pos(bname) + [0, 0, half + CLEAR], speed=0.08)
    top = world.object_pos(bname) + [0, 0, half]
    if np.linalg.norm(rt.ee_pos(arm) - top) > ATTACH_TOP_DIST:
        rt.recover(arm)
        return {"ok": False, "arm": arm, "error": "hand did not get close enough"}
    rt.attach(arm, bname)
    rt.set_hand(arm, "closed")
    rt.servo_ee(arm, world.object_pos(bname) + [0, 0, LIFT])
    return {"ok": True, "arm": arm, "holding": object}
