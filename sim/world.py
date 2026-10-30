"""Tabletop world builder: composes a Body's spec with floor, light, table,
and manipulable objects, then compiles. Object bodies are named `obj_*` and
registered with display names the brain can refer to.
"""

from __future__ import annotations

import mujoco
import numpy as np

TABLE_TOP = 0.72
TABLE_X = 0.42   # table center, robot faces +X
STANDOFF = 0.28  # how far clear of a table edge the robot stands to work

DEFAULT_OBJECTS = [
    {"name": "red block", "kind": "box", "size": [0.025, 0.025, 0.025],
     "rgba": [0.85, 0.15, 0.15, 1], "pos": [0.34, 0.16]},
    {"name": "green block", "kind": "box", "size": [0.025, 0.025, 0.025],
     "rgba": [0.15, 0.65, 0.2, 1], "pos": [0.30, -0.10]},
    {"name": "blue cup", "kind": "cylinder", "size": [0.03, 0.05],
     "rgba": [0.15, 0.35, 0.85, 1], "pos": [0.40, 0.10]},
    {"name": "bin", "kind": "bin", "size": [0.065, 0.065, 0.035],
     "rgba": [0.45, 0.45, 0.5, 1], "pos": [0.38, -0.22]},
]


class World:
    def __init__(self, body, objects: list[dict] | None = None,
                 extra_tables: list[dict] | None = None,
                 start_xy=None):
        self.body = body
        # Where a mobile robot starts, if not on the primary table's station.
        # Used to start a fetch from across the room.
        self.start_xy = None if start_xy is None else np.asarray(
            start_xy, dtype=float)[:2]
        scene_cfg = body.cfg.get("scene", {})
        self.table_top = scene_cfg.get("table_top", TABLE_TOP)
        self.table_x = scene_cfg.get("table_x", TABLE_X)
        self.table_half = np.asarray(scene_cfg.get("table_half", [0.25, 0.4]),
                                     dtype=float)
        self.object_layout = scene_cfg.get("objects")
        spec = body.spec
        w = spec.worldbody

        w.add_light(pos=[0.9, -0.9, 2.2], dir=[-0.3, 0.35, -1], diffuse=[0.6, 0.6, 0.6],
                    ambient=[0.35, 0.35, 0.35], specular=[0.05, 0.05, 0.05])
        w.add_light(pos=[-1.0, 1.0, 2.0], dir=[0.4, -0.4, -1], diffuse=[0.25, 0.25, 0.25],
                    specular=[0, 0, 0])
        w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[6, 6, 0.1], rgba=[0.32, 0.36, 0.42, 1])

        # Each table carries the station the robot works from: the spot it
        # stands on and the direction it faces. The primary table's station
        # is the robot's start pose (where the arm workspace is tuned);
        # others are derived by approaching from the side nearest the start.
        self._add_table(w, "table", [self.table_x, 0.0], self.table_top,
                        self.table_half)
        self.tables = {"table": {"xy": np.array([self.table_x, 0.0]),
                                 "top": self.table_top,
                                 "half": self.table_half,
                                 "station": np.zeros(2)}}
        for t in (extra_tables or []):
            name = t["name"]
            half = np.asarray(t.get("half", [0.22, 0.3]), dtype=float)
            top = t.get("top", self.table_top)
            xy = np.asarray(t["xy"], dtype=float)
            self._add_table(w, name, xy, top, half)
            if "station" in t:
                station = np.asarray(t["station"], dtype=float)
            else:
                away = -xy  # approach from the side facing the origin
                if abs(away[0]) >= abs(away[1]):
                    off = np.array([np.sign(away[0]) * (half[0] + STANDOFF), 0.0])
                else:
                    off = np.array([0.0, np.sign(away[1]) * (half[1] + STANDOFF)])
                station = xy + off
            self.tables[name] = {"xy": xy, "top": top, "half": half,
                                 "station": station}

        self.objects: dict[str, str] = {}  # body name -> display name
        self.graspable: list[str] = []     # subset with free joints
        self.object_half: dict[str, float] = {}  # half-height, for grasp/place
        if objects is None:
            template = {o["name"]: o for o in DEFAULT_OBJECTS}
            if self.object_layout:
                # A body's scene lists exactly what is on its bench: keeping
                # the default props as well leaves objects sitting where the
                # arm cannot reach, or inside each other.
                objects = [dict(template[n], pos=p)
                           for n, p in self.object_layout.items()]
            else:
                objects = [dict(o) for o in DEFAULT_OBJECTS]
        for obj in objects:
            self._add_object(w, obj)

        # The hold is a weld, not friction (see README), so the hand does not
        # need to touch what it picks up — and if it does, the palm shoves
        # the object off the bench on the way in. Exclude hand<->object
        # contact and let the weld do its job. The palm still collides with
        # benches and the floor, so hands cannot pass through the world.
        for side, arm in body.cfg["arms"].items():
            hand = arm.get("hand", {})
            prefix = hand.get("link_prefix")
            links = [arm["ee_parent"]]
            if prefix:
                links += [b.name for b in spec.bodies
                          if b.name.startswith(prefix)]
            for link in links:
                for bname in self.graspable:
                    spec.add_exclude(name=f"nc_{link}_{bname}",
                                     bodyname1=link, bodyname2=bname)

        # Pre-compiled (inactive) hand<->object welds: MuJoCo equalities must
        # exist at compile time; grasp/release toggles them at runtime.
        for side, arm in body.cfg["arms"].items():
            for bname in self.graspable:
                spec.add_equality(
                    name=f"grasp_{side}_{bname}",
                    type=mujoco.mjtEq.mjEQ_WELD,
                    objtype=mujoco.mjtObj.mjOBJ_BODY,
                    name1=arm["ee_parent"], name2=bname,
                    active=False)

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        body.bind(self.model)
        self.reset()

    def _add_table(self, w, name, xy, top, half) -> None:
        leg_h = (top - 0.04) / 2
        w.add_geom(name=f"{name}_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[half[0], half[1], 0.02], pos=[xy[0], xy[1], top - 0.02],
                   rgba=[0.55, 0.42, 0.28, 1])
        for i, (dx, dy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
            w.add_geom(name=f"{name}_leg{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[0.02, 0.02, leg_h],
                       pos=[xy[0] + dx * (half[0] - 0.04),
                            xy[1] + dy * (half[1] - 0.04), leg_h],
                       rgba=[0.4, 0.3, 0.2, 1])

    def _add_object(self, w, obj: dict) -> None:
        bname = "obj_" + obj["name"].replace(" ", "_")
        x, y = obj["pos"]
        kind, size, rgba = obj["kind"], obj["size"], obj["rgba"]
        if kind == "bin":  # static open container: floor + 4 walls
            b = w.add_body(name=bname, pos=[x, y, obj.get("z", self.table_top)])
            sx, sy, sz = size
            t = 0.005
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, t],
                       pos=[0, 0, t], rgba=rgba)
            for pos, s in [([sx, 0, sz], [t, sy, sz]), ([-sx, 0, sz], [t, sy, sz]),
                           ([0, sy, sz], [sx, t, sz]), ([0, -sy, sz], [sx, t, sz])]:
                b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=s, pos=pos, rgba=rgba)
        else:
            half_h = size[-1]
            base_z = obj.get("z", self.table_top)
            b = w.add_body(name=bname, pos=[x, y, base_z + half_h + 0.002])
            b.add_freejoint()
            gtype = (mujoco.mjtGeom.mjGEOM_BOX if kind == "box"
                     else mujoco.mjtGeom.mjGEOM_CYLINDER)
            b.add_geom(type=gtype, size=size, rgba=rgba, mass=0.1)
            self.graspable.append(bname)
        self.object_half[bname] = float(size[-1])
        self.objects[bname] = obj["name"]

    def reset(self, settle: float = 0.5) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.body.reset(self.data)
        if self.start_xy is not None and self.body.mobile:
            root = self.model.body(self.body.cfg["root_body"]).id
            adr = self.model.jnt_qposadr[self.model.body_jntadr[root]]
            self.data.qpos[adr:adr + 2] = self.start_xy
        mujoco.mj_forward(self.model, self.data)
        # A mobile body must never free-run: with no controller in the loop
        # it simply collapses, and the locomotion policy then has to recover
        # from a heap on the floor. The Runtime settles it under the policy.
        if settle and not self.body.mobile:
            self.step(settle)

    def step(self, seconds: float) -> None:
        for _ in range(int(round(seconds / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)

    def object_pos(self, body_name: str) -> np.ndarray:
        return self.data.xpos[self.model.body(body_name).id].copy()

    def table_of(self, pos) -> dict | None:
        """Which table a world point sits on (None if it is on none)."""
        p = np.asarray(pos, dtype=float)
        for t in self.tables.values():
            if (abs(p[0] - t["xy"][0]) <= t["half"][0] + 0.05
                    and abs(p[1] - t["xy"][1]) <= t["half"][1] + 0.05
                    and abs(p[2] - t["top"]) < 0.25):
                return t
        return None
