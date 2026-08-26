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

    def solve(self, data: mujoco.MjData, twist: np.ndarray,
              posture: np.ndarray | None = None,
              posture_gain: float = 1.0) -> np.ndarray:
        """Joint velocities (len(joints),) realizing the world-frame twist
        [v; w] at the site, via weighted DLS: J^T (J J^T + lambda^2 I)^-1 twist.
        If `posture` (a preferred joint configuration) is given, a secondary
        task pulls toward it in the Jacobian nullspace — this steers redundant
        arms away from self-collision-prone configurations."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self.site_id)
        J = self.W @ np.vstack([jacp, jacr])[:, self.dof_ids]  # 6 x n
        J_pinv = J.T @ np.linalg.inv(J @ J.T + (self.damping ** 2) * np.eye(6))
        qdot = J_pinv @ (self.W @ np.asarray(twist, dtype=float))
        if posture is not None:
            q = data.qpos[self.qpos_ids]
            # exact-rank projector: the damped J_pinv leaks the secondary task
            # into task space, which stalls fine servoing near tolerance
            N = np.eye(len(self.dof_ids)) - np.linalg.pinv(J, rcond=1e-2) @ J
            qdot += N @ (posture_gain * (np.asarray(posture) - q))
        return qdot
