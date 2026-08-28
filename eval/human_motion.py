"""How human does a hand movement look? A few numbers, from the motor-control
literature, that a reaching person satisfies and a naive robot does not.

The reference is Flash & Hogan (1985): unconstrained human point-to-point
reaches follow a near-straight hand path with a single-peaked, symmetric,
bell-shaped speed profile. That gives cheap, controller-agnostic tests:

    straightness   path length / straight-line distance   human 1.0 - 1.15
    bow            greatest deviation from the line       human < ~12 mm
    peak_ratio     peak speed / mean speed                human ~1.8 (2.0 for
                                                          an ideal min-jerk)
    peak_at        when peak speed occurs, as a fraction  human 0.30 - 0.60
                   of the movement
    overshoot      how far past the target it goes        human < 6%
    dwell          fraction of the movement spent halted  human ~0
                   in its own interior

A lunge shows up as peak_at near 0, dithering as a high peak_ratio, and
stop-and-go via-point motion as dwell.

Reaches that round a via point are judged differently. Such a path is
deliberately curved, so straightness says nothing about it; what makes it
human is that it passes the waypoint without stopping there, in one
continuous movement rather than two.
"""

from __future__ import annotations

import numpy as np

PEAK_RATIO = (1.5, 2.6)
PEAK_AT = (0.30, 0.60)
MAX_STRAIGHTNESS = 1.15
# The published straightness figures come from unconstrained reaches of
# 20-40 cm. A ratio is the wrong yardstick below that: 10 mm of bow is 4% of
# a 25 cm reach and 20% of a 5 cm one, and people are visibly *less* straight
# on small adjusting movements, not more. Either test may vouch for a reach.
MAX_BOW = 0.012       # m
MAX_OVERSHOOT = 0.06  # fraction of the distance
MAX_VIA_MISS = 0.04   # m: how near the waypoint the hand must actually pass
MAX_DWELL = 0.08      # fraction of the interior spent below SLOW
# An absolute threshold, not a fraction of peak speed. Human reaching is a
# fast primary movement followed by a *slow* corrective one, so a speed
# relative to the peak flags the corrective phase (the very thing that makes
# the motion human) as a defect. What is not human is coming to a dead stop
# in the middle, and that is what this catches.
SLOW = 0.03           # m/s


def score(path, dt: float, settle: float = 0.02, target=None, via=None) -> dict:
    """Score one hand trajectory. `path` is (N, 3) in metres, `dt` seconds.

    `target` is where the motion was told to end and `via` any interior
    waypoints it was told to round. Trailing samples where the hand has
    essentially stopped are trimmed: a long settle tail drags the mean speed
    down and would flatter any controller with a peak/mean ratio it did not
    earn.
    """
    nan = {"straightness": float("nan"), "peak_ratio": float("nan"),
           "peak_at": float("nan"), "human": False}
    p = np.asarray(path, dtype=float)
    if len(p) < 3:
        return {**nan, "samples": len(p)}
    speed = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
    moving = np.flatnonzero(speed > settle)
    if len(moving) < 2:
        return {**nan, "samples": len(p)}
    lo, hi = moving[0], moving[-1] + 1
    speed, p = speed[lo:hi], p[lo:hi + 1]
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)

    end = np.asarray(target, dtype=float) if target is not None else p[-1]
    seg = end - p[0]
    direct = float(np.linalg.norm(seg))
    out = {
        "distance": direct,
        "straightness": float(step.sum() / max(direct, 1e-6)),
        "peak_ratio": float(speed.max() / max(speed.mean(), 1e-6)),
        "peak_at": float(np.argmax(speed) / max(len(speed) - 1, 1)),
        "peak_speed": float(speed.max()),
        "samples": int(len(speed)),
    }

    # A movement that pauses inside itself is two movements. Measured over
    # the interior only, since every reach starts and ends at rest.
    a, b = int(0.1 * len(speed)), int(0.9 * len(speed))
    interior = speed[a:b] if b > a else speed
    out["dwell"] = float(np.mean(interior < SLOW))

    if direct > 1e-6:
        u = seg / direct
        along = (p - p[0]) @ u
        out["bow"] = float(np.linalg.norm(
            (p - p[0]) - np.outer(along, u), axis=1).max())
        out["overshoot"] = float(max(0.0, along.max() / direct - 1.0))
    else:
        out["bow"], out["overshoot"] = 0.0, 0.0

    if via:
        out["via_miss"] = float(max(
            np.linalg.norm(p - np.asarray(w, dtype=float), axis=1).min()
            for w in via))
        shape_ok = out["via_miss"] <= MAX_VIA_MISS
    else:
        shape_ok = (out["bow"] <= MAX_BOW
                    or out["straightness"] <= MAX_STRAIGHTNESS)

    out["human"] = bool(
        shape_ok
        and out["dwell"] <= MAX_DWELL
        and out["overshoot"] <= MAX_OVERSHOOT
        and PEAK_RATIO[0] <= out["peak_ratio"] <= PEAK_RATIO[1]
        and PEAK_AT[0] <= out["peak_at"] <= PEAK_AT[1])
    return out


def table(rows: dict[str, dict]) -> str:
    """Render scored motions as a markdown table, human range in the footer."""
    head = ("| motion | distance | shape | overshoot | dwell | peak/mean | "
            "peak at | human |\n|---|---|---|---|---|---|---|---|\n")
    body = ""
    for name, m in rows.items():
        if "via_miss" in m:
            shape = f"via miss {m['via_miss'] * 1000:.0f} mm"
        else:
            shape = f"bow {m['bow'] * 1000:.0f} mm ({m['straightness']:.2f}x)"
        body += (f"| {name} | {m['distance']:.2f} m | {shape} | "
                 f"{m['overshoot'] * 100:.0f}% | {m['dwell'] * 100:.0f}% | "
                 f"{m['peak_ratio']:.2f} | {m['peak_at'] * 100:.0f}% | "
                 f"{'yes' if m['human'] else 'NO'} |\n")
    return head + body + ("| *human* | | <12 mm or <1.15x; via <40 mm | <6% | "
                          "<8% | 1.5-2.6 | 30-60% | |\n")
