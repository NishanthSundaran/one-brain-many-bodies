"""Results table for the planner post-training study, from posttrain/results/*.jsonl.

Usage: python posttrain/summarize.py
"""

from __future__ import annotations

import collections
import json
import pathlib
from math import comb

RES = pathlib.Path(__file__).resolve().parent / "results"
RUNS = [("base", "Qwen2.5-VL-3B base"), ("sft_r1_final", "SFT"), ("dagger_r1", "SFT + DAgger"),
        ("grpo_r1", "SFT + DAgger + GRPO"), ("expert", "privileged expert (ceiling)")]
FAMILIES = ["bin", "two_bin", "next_to", "side", "absent", "already"]


def load(name: str) -> dict[int, dict]:
    f = RES / ("eval_base_final.jsonl" if name == "base" else f"eval_{name}.jsonl")
    return {r["seed"]: r for r in map(json.loads, open(f))}


def mcnemar(a: dict, b: dict) -> tuple[int, int, float]:
    only_a = sum(a[s]["success"] and not b[s]["success"] for s in a)
    only_b = sum(b[s]["success"] and not a[s]["success"] for s in a)
    n, k = only_a + only_b, min(only_a, only_b)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return only_a, only_b, p


def main() -> None:
    runs = {n: load(n) for n, _ in RUNS}
    head = "| planner | success | " + " | ".join(FAMILIES) + " | failed calls | calls per solved task |"
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    for n, label in RUNS:
        rs = list(runs[n].values())
        by = collections.defaultdict(lambda: [0, 0])
        for r in rs:
            by[r["family"]][0] += r["success"]
            by[r["family"]][1] += 1
        solved = [r for r in rs if r["success"]]
        failed = sum(r.get("failed_calls", 0) for r in rs) if n != "expert" else "n/a"
        calls = (f"{sum(r['n_calls'] for r in solved) / len(solved):.2f}"
                 if solved and "n_calls" in solved[0] else "n/a")
        fam = " | ".join(f"{by[f][0]}/{by[f][1]}" for f in FAMILIES)
        print(f"| {label} | **{len(solved)}/100** | {fam} | {failed} | {calls} |")
    print()
    for a, b in [("sft_r1_final", "base"), ("dagger_r1", "sft_r1_final"), ("grpo_r1", "dagger_r1")]:
        x, y, p = mcnemar(runs[a], runs[b])
        print(f"{a} vs {b}: solved only by {a} {x}, only by {b} {y}, exact McNemar p = {p:.2g}")
    vlm = {n: [r["vlm_s"] / max(1, r["n_calls"]) for r in runs[n].values()] for n, _ in RUNS[:4]}
    print("mean planner latency per call (s):", {n: round(sum(v) / len(v), 2) for n, v in vlm.items()})


if __name__ == "__main__":
    main()
