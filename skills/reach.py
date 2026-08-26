import numpy as np

SCHEMA = {
    "name": "reach",
    "description": "Move one hand to a 3D position in the robot frame (meters).",
    "parameters": {
        "type": "object",
        "properties": {
            "position": {"type": "array", "items": {"type": "number"},
                         "minItems": 3, "maxItems": 3},
            "arm": {"type": "string", "enum": ["left", "right"]},
        },
        "required": ["position"],
    },
}


def run(rt, world, position, arm=None):
    position = np.asarray(position, dtype=float)
    arm = arm or rt.nearest_arm(position)
    if rt.held(arm) is None:
        rt.move_arm_ready(arm)
    ok = rt.servo_ee(arm, position)
    return {"ok": bool(ok), "arm": arm,
            "ee_position": rt.ee_pos(arm).round(3).tolist()}
