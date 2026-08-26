SCHEMA = {
    "name": "home",
    "description": "Return an arm (or both) to the resting pose.",
    "parameters": {
        "type": "object",
        "properties": {
            "arm": {"type": "string", "enum": ["left", "right", "both"]},
        },
    },
}


def run(rt, world, arm="both"):
    sides = ["left", "right"] if arm == "both" else [arm]
    for side in sides:
        rt.move_arm_home(side)
    return {"ok": True, "arm": arm}
