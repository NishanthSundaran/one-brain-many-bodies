import numpy as np

SCHEMA = {
    "name": "look_at",
    "description": "Turn the torso/gaze toward a named object to center it "
                   "in the camera view.",
    "parameters": {
        "type": "object",
        "properties": {"object": {"type": "string"}},
        "required": ["object"],
    },
}


def run(rt, world, object):
    from skills.registry import resolve_object
    bname = resolve_object(world, object)
    if bname is None:
        return {"ok": False, "error": f"no object named '{object}'"}
    gaze_joint = rt.body.cfg.get("gaze_joint")
    if gaze_joint is None:
        return {"ok": True, "note": "this body has a fixed gaze; no motion"}
    pos = world.object_pos(bname)
    yaw = float(np.arctan2(pos[1], pos[0]))
    lo, hi = rt.body.joint_limits[gaze_joint]
    yaw = float(np.clip(yaw, lo, hi))
    act = rt.body.actuator_of[gaze_joint]
    start = float(rt.data.ctrl[act])
    n = max(1, int(0.8 / rt.dt))
    for i in range(n):
        s = (i + 1) / n
        rt.data.ctrl[act] = (1 - s) * start + s * yaw
        rt.step(rt.dt)
    return {"ok": True, "yaw_rad": round(yaw, 3)}
