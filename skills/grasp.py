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

    rt.gaze_at(pos)          # look at it before reaching for it
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
        # One blended motion over the clutter and all the way down to the
        # grip point, rather than up-stop-across-stop-down-stop-down. The
        # final descent used to be its own primitive, which made it a 4 cm
        # movement with its own acceleration and braking phase, the tell
        # that reads as a robot placing its hand rather than a person
        # reaching for something. See Runtime.reach_via.
        half_top = world.object_half.get(bname, 0.03) + CLEAR
        ok = rt.reach_via(arm, [[pos[0], pos[1], pos[2] + travel_h],
                                pos + [0, 0, approach_h],
                                world.object_pos(bname) + [0, 0, half_top]])
    if not ok:
        rt.recover(arm)
        return {"ok": False, "arm": arm, "error": "could not reach above object"}
    half = world.object_half.get(bname, 0.03)
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
    if balancing:
        # Lift to a fixed height above the object, not LIFT above the hand:
        # the palm is already hovering ~11 cm up, so lifting from there ends
        # a hand's width higher than this body can afford. Reaching high
        # pitches a balancing robot's mass forward, and it goes over, which
        # it duly did, after an otherwise clean fetch.
        # A balancing robot gets the gentle tracker: half the feedback gain,
        # a lower speed cap and no corrective submovement. The settings that
        # make a fixed-base reach crisp are a disturbance the legs cannot
        # absorb while also carrying something, and it walks away off course.
        rt.reach_line(arm, world.object_pos(bname) + [0, 0, LIFT],
                      gain=3.0, cap=0.6, correct=False)
    else:
        # Lift straight up from where the hand actually is, not up a column
        # above the object's nominal centre: any lateral grasp error would
        # otherwise be smuggled into the lift, curving it sideways.
        rt.reach_line(arm, rt.ee_pos(arm) + [0, 0, LIFT])
    return {"ok": True, "arm": arm, "holding": object}
