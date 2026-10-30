#!/usr/bin/env bash
# Live demo for presenting: the G1 walks to the bench, picks the block and
# drops it in the bin, in MuJoCo's interactive window, paced to real time.
#   ./demos/live_demo.sh          real time
#   ./demos/live_demo.sh 2        twice real time
#   SEED=3 ./demos/live_demo.sh   another of the 20 scored trials
# Drag to orbit, right-drag to pan, scroll to zoom. Close the window to exit.
cd "$(dirname "$0")/.." || exit 1
exec env -u PYTHONPATH .venv/bin/python demos/13_walk_across_and_fetch.py \
    --policy third_party/policies/g1_walk_human.pt --decide -0.37 \
    --hold stance --stance third_party/policies/g1_stance_human2.pt \
    --seed-offset "${SEED:-0}" --live "${1:-1}"
