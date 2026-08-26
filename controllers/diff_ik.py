"""Damped-least-squares differential IK: 6D site twist -> joint velocities
for an arbitrary joint subset. Embodiment-agnostic: everything comes from the
compiled model (site name + joint names).
"""

from __future__ import annotations

import mujoco
import numpy as np


class DiffIK:
    def __init__(self, model: mujoco.MjModel, site: str, joints: list[str],
                 damping: float = 0.05, angular_weight: float = 0.3):
        self.model = model
        self.site_id = model.site(site).id
        self.dof_ids = np.array([model.joint(j).dofadr[0] for j in joints])
        self.qpos_ids = np.array([model.joint(j).qposadr[0] for j in joints])
        self.damping = damping
        # linear tracking is the priority; a full-weight angular constraint
        # fights translation on a 7-dof arm and starves both
        self.W = np.diag([1.0] * 3 + [angular_weight] * 3)

    def solve(self, data: mujoco.MjData, twist: np.ndarray) -> np.ndarray:
        """Joint velocities (len(joints),) realizing the world-frame twist
        [v; w] at the site, via weighted DLS: J^T (J J^T + lambda^2 I)^-1 twist."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self.site_id)
        J = self.W @ np.vstack([jacp, jacr])[:, self.dof_ids]  # 6 x n
        JJt = J @ J.T + (self.damping ** 2) * np.eye(6)
        return J.T @ np.linalg.solve(JJt, self.W @ np.asarray(twist, dtype=float))
