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
    rt.plant()          # stop stepping; stiffen into stance to work
    if rt.reposition_for(pos):        # walked? re-read the object and arm
        pos = world.object_pos(bname)
        arm = rt.nearest_arm(pos)
    rt.move_arm_ready(arm)
    # safe-height waypoint: travel above tabletop clutter, then descend
    rt.servo_ee(arm, [pos[0], pos[1], pos[2] + APPROACH + 0.06],
                speed=0.18, tol=0.05, timeout=4.0)
    if not rt.servo_ee(arm, pos + [0, 0, APPROACH]):
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
