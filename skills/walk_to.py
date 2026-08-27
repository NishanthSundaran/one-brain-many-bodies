import numpy as np

SCHEMA = {
    "name": "walk_to",
    "description": "Walk to a table (or the table an object is on) and stand "
                   "in front of it. Use this when something is out of reach.",
    "parameters": {
        "type": "object",
        "properties": {
            "target": {"type": "string",
                       "description": "table name (e.g. 'side table') or "
                                      "the name of an object on it"},
        },
        "required": ["target"],
    },
}


def run(rt, world, target):
    from skills.registry import resolve_object
    if rt.loco is None:
        return {"ok": False,
                "error": "this body has no locomotion policy; it works from "
                         "a fixed station"}
    name = str(target).strip().lower()

    tables = {k.lower().replace("_", " "): k for k in world.tables}
    if name in tables:
        table = world.tables[tables[name]]
    else:
        bname = resolve_object(world, target)
        if bname is None:
            return {"ok": False,
                    "error": f"no table or object named '{target}'"}
        table = world.table_of(world.object_pos(bname))
        if table is None:
            return {"ok": False,
                    "error": f"'{target}' is not on any table"}

    if np.linalg.norm(rt.base_xy - table["station"]) < 0.05:
        return {"ok": True, "note": "already standing there"}

    rt.home()      # arms down before walking
    rt.unplant()      # hand the legs back to the walking policy
    rt.walk_again()
    arrived = rt.walk_to(table["station"][0], table["station"][1],
                         face=table["xy"])
    upright = bool(rt.data.xpos[rt.root_id][2] > 0.55)
    if not (arrived and upright):
        return {"ok": False, "arm": None,
                "error": ("fell while walking" if not upright else
                          "could not reach the station"),
                "stopped_at": np.round(rt.base_xy, 2).tolist()}
    return {"ok": True, "standing_at": np.round(rt.base_xy, 2).tolist(),
            "facing": str(target)}
