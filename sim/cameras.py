"""Offscreen rendering: RGB and segmentation frames from any model camera
(or a free MjvCamera), sharing one EGL context pair per world.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


class CameraRig:
    def __init__(self, model: mujoco.MjModel, width: int = 640, height: int = 480):
        self.model = model
        self._rgb = mujoco.Renderer(model, height=height, width=width)
        self._seg = mujoco.Renderer(model, height=height, width=width)
        self._seg.enable_segmentation_rendering()

    def rgb(self, data, camera="head_cam") -> np.ndarray:
        self._rgb.update_scene(data, camera=camera)
        return self._rgb.render()

    def segmentation(self, data, camera="head_cam") -> np.ndarray:
        """(H, W, 2) int32: [:, :, 0] = object id, [:, :, 1] = mjtObj type."""
        self._seg.update_scene(data, camera=camera)
        return self._seg.render()

    def third_person(self, data, azimuth=150.0, distance=2.0,
                     lookat=(0.35, 0.0, 0.75), elevation=-20.0) -> np.ndarray:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.azimuth, cam.distance, cam.elevation = azimuth, distance, elevation
        self._rgb.update_scene(data, camera=cam)
        return self._rgb.render()

    def close(self) -> None:
        self._rgb.close()
        self._seg.close()
