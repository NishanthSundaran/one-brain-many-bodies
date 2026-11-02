"""Seeded tabletop tasks for post-training the planner, with a success check
read off the physics state and a privileged expert that picks the next call.

Every task is a pure function of its seed: the object set, where each object
stands, the command, and what counts as success. Every task is feasible by
construction: objects stand only where `posttrain/reach_map.json` (measured
by sweep_reach.py) says their own arm can grasp them, and a move across the
body midline is only asked of a cup, the one object both hands can reach at
the midline. The expert sees the true simulator state (the planner gets the
camera image and the marker list), so its episodes are the ceiling and, once
verified in physics, the supervised training data.

Families, chosen so that each needs a different piece of judgement:
    bin       "Put the X in the bin."                       pick the right arm
    two_bin   "Put the X and the Y in the bin."            two in a row
    next_to   "Put the X next to the Y."                   a coordinate target
    side      "Move the X to the left side of the table."  a handover (cups)
    absent    "Put the purple block in the bin." (no such  should only say done
               object on the table)
    already   "Put the X in the bin." (X already in it)    should only say done
"""

from __future__ import annotations

import json
import pathlib
import random

import numpy as np

TABLE_TOP = 0.72
X_RANGE = (0.28, 0.35)   # cups reach to 0.38, but a grasp at the edge cannot carry to the bin
Y_ABS = (0.06, 0.27)
MIN_GAP = 0.08           # centre to centre: objects must not start touching
BIN_CLEAR = 0.10         # per axis from the bin centre: bin half-width 0.065 + object + margin
BIN_XY = {"left": (0.32, 0.21), "right": (0.32, -0.21)}
HANDOVER = (0.29, 0.0)   # cups only: both arms grasp a cup here (reach map)
HANDOVER_BIAS = 0.0      # biasing toward the receiver measured worse (87 vs 94 of 100)

PALETTE = [
    ("red block", "box", [0.85, 0.15, 0.15, 1]),
    ("green block", "box", [0.15, 0.65, 0.2, 1]),
    ("yellow block", "box", [0.92, 0.8, 0.1, 1]),
    ("orange block", "box", [0.95, 0.5, 0.1, 1]),
    ("blue cup", "cylinder", [0.15, 0.35, 0.85, 1]),
    ("white cup", "cylinder", [0.9, 0.9, 0.9, 1]),
]
KIND = {n: k for n, k, _ in PALETTE}
ABSENT_NAMES = ["purple block", "black cup", "pink block", "brown cup"]
SIZE = {"box": [0.025, 0.025, 0.025], "cylinder": [0.03, 0.05]}

FAMILIES = ["bin", "two_bin", "next_to", "side", "absent", "already"]
WEIGHTS = [0.25, 0.15, 0.2, 0.2, 0.1, 0.1]

NEXT_TO_DIST = (0.05, 0.16)   # m, XY centre distance that counts as "next to"
NEXT_TO_START = 0.17          # a next_to task must not start satisfied
SIDE_MARGIN = 0.06            # |y| past the midline that counts as "on that side"
MOVED_TOL = 0.03              # m, how far an object may drift and still count as untouched

_GRID = None


def _grid() -> dict:
    global _GRID
    if _GRID is None:
        rows = json.load(open(pathlib.Path(__file__).with_name("reach_map.json")))
        _GRID = {(round(x, 2), round(y, 2), arm, kind): ok for x, y, arm, kind, ok, _ in rows}
    return _GRID


def reachable(kind: str, arm: str, x: float, y: float) -> bool:
    """Conservative: all four surrounding grid points of the measured map grasp OK."""
    g = _grid()
    x0 = 0.24 + 0.02 * np.floor((x - 0.24) / 0.02 + 1e-9)
    y0 = -0.30 + 0.03 * np.floor((y + 0.30) / 0.03 + 1e-9)
    corners = [(round(a, 2), round(b, 2)) for a in (x0, x0 + 0.02) for b in (y0, y0 + 0.03)]
    return all(g.get((a, b, arm, kind), False) for a, b in corners)


def at_handover(p) -> bool:
    return bool(np.hypot(p[0] - HANDOVER[0], p[1] - HANDOVER[1]) < 0.04)


def handover_for(receiver: str) -> tuple[float, float]:
    return (HANDOVER[0], HANDOVER[1] + (HANDOVER_BIAS if receiver == "left" else -HANDOVER_BIAS))


def side_of(y: float) -> str:
    return "left" if y > 0 else "right"


def place_z(kind: str) -> float:
    return round(TABLE_TOP + SIZE[kind][-1] + 0.005, 3)


def _clear(x, y, pts, bin_xy, gap=MIN_GAP) -> bool:
    if abs(x - bin_xy[0]) < BIN_CLEAR and abs(y - bin_xy[1]) < BIN_CLEAR:
        return False
    return all(np.hypot(x - px, y - py) >= gap for px, py in pts)


def _free_spot(rng, kind, arm, pts, bin_xy, near=None, ring=(0.07, 0.12), tries=400):
    """A reachable, clear spot for `kind` held by `arm`; within `ring` of `near` if given."""
    for _ in range(tries):
        if near is None:
            x = rng.uniform(*X_RANGE)
            y = (1 if arm == "left" else -1) * rng.uniform(*Y_ABS)
        else:
            a, r = rng.uniform(0, 2 * np.pi), rng.uniform(*ring)
            x, y = near[0] + r * np.cos(a), near[1] + r * np.sin(a)
        x, y = round(float(x), 3), round(float(y), 3)
        # 0.105: clear of skills/place.py CLEARANCE (0.10), or the skill shifts the target itself
        if reachable(kind, arm, x, y) and _clear(x, y, pts, bin_xy, gap=0.105):
            return [x, y]
    return None


def sample_task(seed: int) -> dict:
    """A task as plain data: objects (World format), command, family, goal."""
    rng = random.Random(seed)
    family = rng.choices(FAMILIES, WEIGHTS)[0]    # fixed first, so rejections keep the mix
    for _ in range(2000):
        t = _try_sample(rng, seed, family)
        if t is not None:
            return t
    raise RuntimeError(f"seed {seed}: no feasible task")


def _try_sample(rng: random.Random, seed: int, family: str) -> dict | None:
    bin_side = rng.choice(["left", "right"])
    bin_xy = BIN_XY[bin_side]
    picks = rng.sample(PALETTE, 3)
    pts = []
    for name, kind, _ in picks:
        spot = None
        for _ in range(400):
            x = rng.uniform(*X_RANGE)
            y = rng.choice([-1, 1]) * rng.uniform(*Y_ABS)
            if reachable(kind, side_of(y), x, y) and _clear(x, y, pts, bin_xy):
                spot = (round(x, 3), round(y, 3))
                break
        if spot is None:
            return None
        pts.append(spot)
    objects = [{"name": n, "kind": k, "size": SIZE[k], "rgba": c, "pos": list(p)}
               for (n, k, c), p in zip(picks, pts)]
    pos = {o["name"]: o["pos"] for o in objects}
    objects.append({"name": "bin", "kind": "bin", "size": [0.065, 0.065, 0.035],
                    "rgba": [0.45, 0.45, 0.5, 1], "pos": list(bin_xy)})
    names = [n for n, _, _ in picks]
    handover_free = _clear(*HANDOVER, pts, bin_xy)

    def to_bin_ok(n):  # same side as the bin, or a cup that can be handed over
        return side_of(pos[n][1]) == bin_side or (KIND[n] == "cylinder" and handover_free)

    if family == "bin":
        ok = [n for n in names if to_bin_ok(n)]
        if not ok:
            return None
        x = rng.choice(ok)
        command, goal = f"Put the {x} in the bin.", {"in_bin": [x]}
    elif family == "two_bin":
        ok = [n for n in names if to_bin_ok(n)]
        if len(ok) < 2:
            return None
        x, y = rng.sample(ok, 2)
        command, goal = f"Put the {x} and the {y} in the bin.", {"in_bin": [x, y]}
    elif family == "next_to":
        pairs = [(a, b) for a in names for b in names if a != b
                 and side_of(pos[a][1]) == side_of(pos[b][1])
                 and np.hypot(pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]) > NEXT_TO_START]
        if not pairs:
            return None
        x, y = rng.choice(pairs)
        others = [p for n, p in pos.items() if n != x]
        spot = _free_spot(rng, KIND[x], side_of(pos[x][1]), others, bin_xy, near=pos[y])
        if spot is None:
            return None
        command, goal = f"Put the {x} next to the {y}.", {"next_to": [x, y], "spot": spot}
    elif family == "side":
        cups = [n for n in names if KIND[n] == "cylinder"]
        if not cups or not handover_free:
            return None
        x = rng.choice(cups)
        there = "right" if side_of(pos[x][1]) == "left" else "left"
        others = [p for n, p in pos.items() if n != x] + [HANDOVER]
        spot = _free_spot(rng, KIND[x], there, others, bin_xy)
        if spot is None:
            return None
        command, goal = f"Move the {x} to the {there} side of the table.", {"side": [x, there], "spot": spot}
    elif family == "absent":
        x = rng.choice(ABSENT_NAMES)
        command, goal = f"Put the {x} in the bin.", {"nothing": True}
    else:  # already: one object starts inside the bin
        x = rng.choice(names)
        for o in objects:
            if o["name"] == x:
                o["pos"] = list(bin_xy)
                o["z"] = TABLE_TOP + 0.01
        command, goal = f"Put the {x} in the bin.", {"in_bin": [x], "nothing": True}
    return {"seed": seed, "family": family, "command": command, "goal": goal,
            "bin_side": bin_side, "objects": objects}


# ---------------------------------------------------------------- checking

def _pos(world, name: str) -> np.ndarray:
    from skills.registry import resolve_object
    return world.object_pos(resolve_object(world, name))


def in_bin(world, name: str) -> bool:
    p, b = _pos(world, name), _pos(world, "bin")
    return bool(abs(p[0] - b[0]) < 0.07 and abs(p[1] - b[1]) < 0.07
                and TABLE_TOP < p[2] < TABLE_TOP + 0.10)


def on_table(world, name: str) -> bool:
    return bool(TABLE_TOP < _pos(world, name)[2] < TABLE_TOP + 0.08)


def snapshot(world) -> dict:
    return {n: world.object_pos(b).copy() for b, n in world.objects.items()}


def check(task: dict, world, start: dict, rt=None) -> dict:
    """Success against the physics state. `start` is snapshot() at reset.
    Returns {"success": bool, "why": str}. Holding anything at the end fails."""
    g = task["goal"]
    now = snapshot(world)
    if rt is not None and any(rt.held(s) for s in ("left", "right")):
        return {"success": False, "why": "still holding an object"}
    if g.get("nothing"):
        moved = [n for n, p in start.items() if np.linalg.norm(now[n] - p) > MOVED_TOL]
        if moved:
            return {"success": False, "why": f"moved {moved}"}
    if "in_bin" in g:
        missing = [x for x in g["in_bin"] if not in_bin(world, x)]
        if missing:
            return {"success": False, "why": f"not in bin: {missing}"}
    if "next_to" in g:
        x, y = g["next_to"]
        d = float(np.hypot(*(now[x] - now[y])[:2]))
        if not (on_table(world, x) and NEXT_TO_DIST[0] <= d <= NEXT_TO_DIST[1]):
            return {"success": False, "why": f"{x} is {d:.2f} m from {y}"}
    if "side" in g:
        x, side = g["side"]
        y = now[x][1]
        ok = y > SIDE_MARGIN if side == "left" else y < -SIDE_MARGIN
        if not (ok and on_table(world, x)):
            return {"success": False, "why": f"{x} ended at y={y:.2f}"}
    moving = set(g.get("in_bin", []))
    for key in ("next_to", "side"):
        if key in g:
            moving.add(g[key][0])
    knocked = [n for n, p in start.items() if n not in moving and n != "bin"
               and np.linalg.norm(now[n] - p) > MOVED_TOL]
    if knocked:
        return {"success": False, "why": f"disturbed {knocked}"}
    return {"success": True, "why": "ok"}


# ---------------------------------------------------------------- expert

def _call(thought: str, tool: str, **args) -> dict:
    return {"thought": thought, "tool": tool, "args": args}


def _xyz(xy, kind) -> list:
    return [round(float(xy[0]), 3), round(float(xy[1]), 3), place_z(kind)]


def expert(task: dict, world, rt) -> dict:
    """Next call from the true state: what a planner that reasons correctly
    about arms, reach and the task would send."""
    g = task["goal"]
    held = {s: world.objects[rt.held(s)] for s in ("left", "right") if rt.held(s)}

    if g.get("nothing") and "in_bin" not in g:
        x = task["command"].removeprefix("Put the ").removesuffix(" in the bin.")
        return _call(f"No marker is a {x}, so there is nothing to put in the bin.",
                     "done", summary=f"There is no {x} on the table, so nothing was moved.")

    if "in_bin" in g:
        bin_arm = task["bin_side"]
        for side, name in held.items():
            if side == bin_arm:
                return _call(f"The {side} hand holds the {name} and the bin is on the {side} side.",
                             "place", target="bin", arm=side)
            return _call(f"The bin is on the {bin_arm} side, out of the {side} arm's reach; "
                         f"set the {name} down at the middle so the {bin_arm} arm can take it.",
                         "place", target=_xyz(handover_for(bin_arm), KIND[name]), arm=side)
        todo = [x for x in g["in_bin"] if not in_bin(world, x)]
        if not todo:
            return _call("Every object the task names is in the bin.", "done",
                         summary="Done: " + " and ".join(f"the {x}" for x in g["in_bin"]) + " in the bin.")
        x = todo[0]
        p = _pos(world, x)
        if at_handover(p) or reachable(KIND[x], bin_arm, p[0], p[1]):
            return _call(f"The {x} is within the {bin_arm} arm's reach, like the bin.",
                         "grasp", object=x, arm=bin_arm)
        arm = side_of(p[1])
        return _call(f"The {x} is on the {arm} side and the bin on the {bin_arm} side; "
                     f"pick it up with the {arm} arm first.", "grasp", object=x, arm=arm)

    if "next_to" in g:
        x, y = g["next_to"]
        for side, name in held.items():
            return _call(f"Set the {name} down beside the {y}.", "place",
                         target=_xyz(g["spot"], KIND[name]), arm=side)
        px, py = _pos(world, x), _pos(world, y)
        if np.hypot(*(px - py)[:2]) <= NEXT_TO_DIST[1] and on_table(world, x):
            return _call(f"The {x} is now next to the {y}.", "done", summary=f"The {x} is next to the {y}.")
        arm = side_of(px[1])
        return _call(f"The {x} is on the {arm} side; grasp it with the {arm} arm.",
                     "grasp", object=x, arm=arm)

    if "side" in g:
        x, there = g["side"]
        for side, name in held.items():
            if side == there:
                return _call(f"The {there} hand holds the {name}; set it down on the {there} side.",
                             "place", target=_xyz(g["spot"], KIND[name]), arm=side)
            return _call(f"The {side} arm cannot reach the {there} side; set the {name} down "
                         f"at the middle for the {there} arm.", "place",
                         target=_xyz(handover_for(there), KIND[name]), arm=side)
        p = _pos(world, x)
        on_target = p[1] > SIDE_MARGIN if there == "left" else p[1] < -SIDE_MARGIN
        if on_target and on_table(world, x):
            return _call(f"The {x} is on the {there} side now.", "done",
                         summary=f"The {x} is on the {there} side of the table.")
        if at_handover(p) or reachable(KIND[x], there, p[0], p[1]):
            return _call(f"The {x} is at the middle, within the {there} arm's reach.",
                         "grasp", object=x, arm=there)
        arm = side_of(p[1])
        return _call(f"The {x} is on the {arm} side; pick it up with the {arm} arm.",
                     "grasp", object=x, arm=arm)
    raise ValueError(task)
