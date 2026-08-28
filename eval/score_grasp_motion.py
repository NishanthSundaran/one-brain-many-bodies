"""Score every hand movement of a real grasp-and-place against the human
reaching signatures in eval/human_motion.py.

An isolated reach is the easy case. What matters for "does this robot move
like a person" is every segment the skill actually runs, so this wraps the
runtime's motion primitives and scores whatever the real grasp/place code
asks for: approach, descent, lift, transport, and retreat.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bodies.body import Body                                     # noqa: E402
from eval.human_motion import score, table                       # noqa: E402
from sim.world import World                                      # noqa: E402
from skills.registry import execute                              # noqa: E402
from skills.runtime import Runtime                               # noqa: E402


def instrument(rt, rows, names):
    """Trace and score every hand motion the skills ask for.

    `_track` is the single chokepoint every Cartesian motion goes through
    (reach_line and reach_via both delegate to it), so wrapping it catches
    the real movements without depending on which entry point a skill used.
    Only outermost calls are scored: a reach includes its own corrective
    submovement, and scoring the two halves separately would judge neither.
    """
    depth = [0]

    def wrap(name, target_arg):
        original = getattr(rt, name, None)
        if original is None:
            return

        def wrapped(side, *a, _orig=original, _name=name, **kw):
            if depth[0]:
                return _orig(side, *a, **kw)
            depth[0] += 1
            try:
                with rt.trace(side) as path:
                    out = _orig(side, *a, **kw)
            finally:
                depth[0] -= 1
            target = np.asarray(a[target_arg], dtype=float)
            label = names.pop(0) if names else f"{_name} #{len(rows) + 1}"
            m = score(np.array(path), rt.dt, target=target,
                      via=kw.get("via"))
            if not np.isnan(m["straightness"]):
                rows[label] = m
            return out

        setattr(rt, name, wrapped)

    wrap("_track", 1)              # path, target -> target is arg 1
    wrap("reach_joint_space", 0)
    wrap("servo_ee", 0)


def main():
    body = Body(REPO / "bodies" / "g1" / "config.yaml")
    world = World(body)
    rt = Runtime(world)
    rt.home()

    rows: dict[str, dict] = {}
    names = ["grasp: reach to the object", "grasp: lift",
             "place: carry to the bin", "place: retreat"]
    instrument(rt, rows, names)

    g = execute("grasp", rt, world, object="green block")
    p = execute("place", rt, world, target="bin")
    print(f"grasp -> {g}\nplace -> {p}\n")
    print(table(rows))
    bad = [n for n, m in rows.items() if not m["human"]]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} segments human-like")
    for n in bad:
        m = rows[n]
        print(f"  NOT human: {n}: straightness {m['straightness']:.2f}, "
              f"peak/mean {m['peak_ratio']:.2f}, peak at {m['peak_at']*100:.0f}%")


if __name__ == "__main__":
    main()
