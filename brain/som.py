"""Set-of-Mark grounding: overlay stable numbered markers on the objects the
robot can act on. The VLM refers to marker ids; the simulator resolves each id
to an exact 3D pose. Marker ids are assigned once per world (registry order),
so an object keeps its number across frames even when briefly occluded.
"""

from __future__ import annotations

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MIN_PIXELS = 40


def geoms_of(model: mujoco.MjModel, body_name: str) -> np.ndarray:
    bid = model.body(body_name).id
    adr, num = model.body_geomadr[bid], model.body_geomnum[bid]
    return np.arange(adr, adr + num)


def annotate(rgb: np.ndarray, seg: np.ndarray, model: mujoco.MjModel,
             data: mujoco.MjData, objects: dict[str, str]) -> tuple[np.ndarray, dict]:
    """Returns (annotated RGB, markers). markers[id] = {name, body, pos, pixel};
    only objects visible in this frame appear."""
    geom_mask = seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    ids = seg[:, :, 0]

    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()

    markers: dict[int, dict] = {}
    for marker_id, (body_name, display) in enumerate(objects.items(), start=1):
        mask = geom_mask & np.isin(ids, geoms_of(model, body_name))
        n = int(mask.sum())
        if n < MIN_PIXELS:
            continue
        vs, us = np.nonzero(mask)
        u, v = float(us.mean()), float(vs.mean())
        r = 11
        draw.ellipse([u - r, v - r, u + r, v + r], fill=(255, 255, 255),
                     outline=(0, 0, 0), width=2)
        draw.text((u, v), str(marker_id), fill=(0, 0, 0), font=font, anchor="mm")
        markers[marker_id] = {
            "name": display,
            "body": body_name,
            "pos": data.xpos[model.body(body_name).id].round(4).tolist(),
            "pixel": (round(u), round(v)),
        }
    return np.asarray(img), markers
