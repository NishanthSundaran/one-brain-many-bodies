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
HOVER_GRIP = 0.08       # palm clearance above the object when balancing
ATTACH_TOP_BALANCING = 0.11


def run(rt, world, object, arm=None):
    arm_given = arm is not None
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
    rt.approach_object(pos)   # stand, and step if the target is out of range
    rt.plant()                # no-op unless this body plants to manipulate
    pos = world.object_pos(bname)
    arm = arm if arm_given else rt.nearest_arm(pos)
    rt.move_arm_ready(arm)
    # A balancing robot holds its ground to about +/-0.2 m, so the target
    # drifts in and out of the arm's envelope. The servo is closed loop on
    # the object in world coordinates, so the answer is patience: keep
    # tracking and let the base wander back into range.
    patience = 16.0 if rt.loco is not None else 4.0
    # A balancing robot should not reach high: lifting the hand well above
    # the shoulder pitches the centre of mass forward and it leans back out
    # of its own reach. It also stands at a table it can see over, so it does
    # not need the clutter-clearing height a fixed base does.
    balancing = rt.loco is not None
    approach_h = 0.06 if balancing else APPROACH
    travel_h = approach_h + (0.02 if balancing else 0.06)
    if balancing:
        # Come in from above, then straight down: reaching directly to the
        # grip point sweeps the hand in sideways and knocks the object off
        # the bench. The descent reseeds from the current pose so it stays a
        # small motion rather than a swing to a new solution branch.
        half_h = world.object_half.get(bname, 0.03)
        # Hover the palm clear of the object rather than on it: the palm
        # keeps its collision geometry (so hands cannot pass through benches)
        # and its underside reaches a few centimetres below the wrist site,
        # so a 3 cm clearance drives it into the object and shoves it away.
        # One move, straight to the grip pose. Now that the hand cannot
        # disturb the object there is nothing to approach around, and every
        # extra arm move is a disturbance the legs have to absorb.
        ok = rt.reach_joint_space(arm, pos + [0, 0, half_h + HOVER_GRIP])
    else:
        rt.servo_ee(arm, [pos[0], pos[1], pos[2] + travel_h],
                    speed=0.18, tol=0.05, timeout=patience)
        ok = rt.servo_ee(arm, pos + [0, 0, approach_h], timeout=patience)
    if not ok:
        rt.recover(arm)
        return {"ok": False, "arm": arm, "error": "could not reach above object"}
    half = world.object_half.get(bname, 0.03)
    if not balancing:
        rt.servo_ee(arm, world.object_pos(bname) + [0, 0, half + CLEAR],
                    speed=0.08)
    top = world.object_pos(bname) + [0, 0, half]
    ee = rt.ee_pos(arm)
    if balancing:
        # Judge the grasp on lateral alignment, not straight-line distance:
        # the palm deliberately hovers above the object (its collision
        # geometry would otherwise shove it), so height is by design and
        # only sideways error means a miss.
        aligned = (np.linalg.norm(ee[:2] - top[:2]) < 0.08
                   and -0.02 < ee[2] - top[2] < 0.18)
    else:
        aligned = np.linalg.norm(ee - top) <= ATTACH_TOP_DIST
    if not aligned:
        rt.recover(arm)
        return {"ok": False, "arm": arm, "error": "hand did not get close enough"}
    rt.attach(arm, bname)
    rt.set_hand(arm, "closed")
    rt.servo_ee(arm, world.object_pos(bname) + [0, 0, LIFT])
    return {"ok": True, "arm": arm, "holding": object}
