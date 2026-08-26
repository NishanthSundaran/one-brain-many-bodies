"""Tabletop world builder: composes a Body's spec with floor, light, table,
and manipulable objects, then compiles. Object bodies are named `obj_*` and
registered with display names the brain can refer to.
"""

from __future__ import annotations

import mujoco
import numpy as np

TABLE_TOP = 0.72
TABLE_X = 0.42  # table center, robot faces +X

DEFAULT_OBJECTS = [
    {"name": "red block", "kind": "box", "size": [0.025, 0.025, 0.025],
     "rgba": [0.85, 0.15, 0.15, 1], "pos": [0.38, 0.14]},
    {"name": "green block", "kind": "box", "size": [0.025, 0.025, 0.025],
     "rgba": [0.15, 0.65, 0.2, 1], "pos": [0.33, -0.08]},
    {"name": "blue cup", "kind": "cylinder", "size": [0.03, 0.05],
     "rgba": [0.15, 0.35, 0.85, 1], "pos": [0.47, 0.03]},
    {"name": "bin", "kind": "bin", "size": [0.07, 0.07, 0.035],
     "rgba": [0.45, 0.45, 0.5, 1], "pos": [0.44, -0.24]},
]


class World:
    def __init__(self, body, objects: list[dict] | None = None):
        self.body = body
        spec = body.spec
        w = spec.worldbody

        w.add_light(pos=[0.9, -0.9, 2.2], dir=[-0.3, 0.35, -1], diffuse=[0.6, 0.6, 0.6],
                    ambient=[0.35, 0.35, 0.35], specular=[0.05, 0.05, 0.05])
        w.add_light(pos=[-1.0, 1.0, 2.0], dir=[0.4, -0.4, -1], diffuse=[0.25, 0.25, 0.25],
                    specular=[0, 0, 0])
        w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[3, 3, 0.1], rgba=[0.32, 0.36, 0.42, 1])

        leg_h = (TABLE_TOP - 0.04) / 2
        w.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.25, 0.4, 0.02], pos=[TABLE_X, 0, TABLE_TOP - 0.02],
                   rgba=[0.55, 0.42, 0.28, 1])
        for i, (dx, dy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
            w.add_geom(name=f"table_leg{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[0.02, 0.02, leg_h],
                       pos=[TABLE_X + dx * 0.21, dy * 0.36, leg_h],
                       rgba=[0.4, 0.3, 0.2, 1])

        self.objects: dict[str, str] = {}  # body name -> display name
        for obj in (DEFAULT_OBJECTS if objects is None else objects):
            self._add_object(w, obj)

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        body.bind(self.model)
        self.reset()

    def _add_object(self, w, obj: dict) -> None:
        bname = "obj_" + obj["name"].replace(" ", "_")
        x, y = obj["pos"]
        kind, size, rgba = obj["kind"], obj["size"], obj["rgba"]
        if kind == "bin":  # static open container: floor + 4 walls
            b = w.add_body(name=bname, pos=[x, y, TABLE_TOP])
            sx, sy, sz = size
            t = 0.005
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, t],
                       pos=[0, 0, t], rgba=rgba)
            for pos, s in [([sx, 0, sz], [t, sy, sz]), ([-sx, 0, sz], [t, sy, sz]),
                           ([0, sy, sz], [sx, t, sz]), ([0, -sy, sz], [sx, t, sz])]:
                b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=s, pos=pos, rgba=rgba)
        else:
            half_h = size[-1]
            b = w.add_body(name=bname, pos=[x, y, TABLE_TOP + half_h + 0.002])
            b.add_freejoint()
            gtype = (mujoco.mjtGeom.mjGEOM_BOX if kind == "box"
                     else mujoco.mjtGeom.mjGEOM_CYLINDER)
            b.add_geom(type=gtype, size=size, rgba=rgba, mass=0.1)
        self.objects[bname] = obj["name"]

    def reset(self, settle: float = 0.5) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.body.reset(self.data)
        mujoco.mj_forward(self.model, self.data)
        if settle:
            self.step(settle)

    def step(self, seconds: float) -> None:
        for _ in range(int(round(seconds / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)

    def object_pos(self, body_name: str) -> np.ndarray:
        return self.data.xpos[self.model.body(body_name).id].copy()
