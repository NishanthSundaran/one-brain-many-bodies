"""Steering for the borrowed walking policy, kept inside the commands it can
actually follow.

Measured by eval/score_gait_commands.py, five trials per command on open
floor, from a clean walking start:

  - commanded zero forward speed does not stop it: it still walks
    0.26-0.52 m/s
  - at zero or low forward speed, right turns fall: 2/5 at vx 0 with
    wz -0.5 and -1.0, 1/5 at vx 0.25 with wz -1.0 (left turns: 0/5)
  - at vx 0.5, every yaw rate up to +/-1.0 rad/s is fall-free (0/5 each)
  - realised yaw rate is about half the command, so the tightest circle the
    robot can walk is about a metre across

The old pursuit eased forward speed to zero near the target while asking for
up to 1.1 rad/s of yaw: the exact corner that falls, while the robot kept
moving at ~0.4 m/s on a ~1 m circle. That is the "orbits the target and
eventually falls" the README describes. It was not the policy failing; it
was the controller asking for commands the policy never learned.

So: never command below VX_TRAVEL while travelling, steer only with yaw rate,
cap it at WZ_MAX, hold yaw at zero until the robot is up to speed, and
slew-limit both so a command never jumps. Stopping is not this module's job:
the policy cannot stop itself, so arrival hands over to the stance policy.

Inside FREEZE_R of the target the yaw command eases to zero. A robot a
couple of decimetres off its line otherwise sees the bearing swing towards
90 degrees in the last half metre and gets a full-rate turn just before the
handover; the one repeatable walking fall in eval/score_handover.py was a
right turn there. Line up early, walk the last stretch straight.
"""

from __future__ import annotations

import numpy as np

VX_TRAVEL = 0.5     # m/s: slowest command at which every turn rate was safe
WZ_MAX = 1.0        # rad/s: 0/5 falls both ways at VX_TRAVEL
YAW_GAIN = 1.5      # rad/s of command per rad of bearing error
DVX_MAX = 1.0       # m/s per s
DWZ_MAX = 2.0       # rad/s per s
YAW_TAU = 0.0       # s: heading low-pass time constant
FREEZE_R = 0.8      # m: no steering this close to the target


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class GaitNavigator:
    def __init__(self, gain: float = YAW_GAIN, tau: float = YAW_TAU):
        self.gain = gain
        self.tau = tau
        self.cmd = np.zeros(2)          # vx, wz currently being commanded
        self.yaw = None                 # heading, low-passed over the stride

    def reset(self, vx: float = 0.0, wz: float = 0.0) -> None:
        self.cmd[:] = (vx, wz)
        self.yaw = None

    def command(self, base_xy, base_yaw: float, target,
                dt: float) -> tuple[float, float]:
        if self.yaw is None or self.tau <= 0:
            self.yaw = float(base_yaw)
        else:
            self.yaw += dt / (self.tau + dt) * _wrap(base_yaw - self.yaw)
        err = np.asarray(target, dtype=float)[:2] - np.asarray(base_xy)[:2]
        bearing = _wrap(float(np.arctan2(err[1], err[0])) - self.yaw)
        steer = (self.cmd[0] >= VX_TRAVEL - 1e-6
                 and float(np.linalg.norm(err)) > FREEZE_R)
        want = np.array([
            VX_TRAVEL,
            float(np.clip(self.gain * bearing, -WZ_MAX, WZ_MAX))
            if steer else 0.0])
        step = np.array([DVX_MAX, DWZ_MAX]) * dt
        self.cmd += np.clip(want - self.cmd, -step, step)
        return float(self.cmd[0]), float(self.cmd[1])


def arrived(base_xy, base_yaw: float, target, arrive: float) -> bool:
    """Within `arrive` of the target, or level with it: once steering has
    frozen, a robot slightly off its line passes beside the target instead
    of through the arrival circle, and should stop there, not walk on."""
    err = np.asarray(target, dtype=float)[:2] - np.asarray(base_xy)[:2]
    d = float(np.linalg.norm(err))
    ahead = float(err @ np.array([np.cos(base_yaw), np.sin(base_yaw)]))
    return d < arrive or (d < FREEZE_R and ahead < arrive)
