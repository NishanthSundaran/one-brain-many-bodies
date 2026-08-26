# One Brain, Many Bodies

**One brain for every humanoid.** A hierarchical humanoid autonomy stack in MuJoCo: a *local* open-source VLM (no cloud APIs) plans at ~1 Hz via tool-calling, and a compliant, embodiment-agnostic control stack executes at high rate. The same brain, with the same tool interface, drives different humanoid bodies (Unitree G1, Unitree H1) — only a small per-robot config file changes.

> Work in progress. Milestone demos land as GIFs below as they are built.

## Architecture

```
  "put the red block in the bin"        (user, natural language)
              |
  local VLM agent (Qwen2.5-VL-3B via Ollama, ~1 Hz)
  sees Set-of-Mark annotated camera frames, calls tools
              |
  skills: reach / grasp / place / look_at / walk_to   (deterministic mid-level API)
              |
  compliant whole-body control (admittance + differential IK, ~500 Hz)
              |
  Unitree G1  --or--  Unitree H1        (MuJoCo, model swapped by config only)
```

## Milestones

| # | Demo | Status |
|---|------|--------|
| 0 | G1 standing in MuJoCo | ✅ `demos/01_stand.py` |
| 1 | Head-camera view with Set-of-Mark object markers | ✅ `demos/02_som_view.py` |
| 2 | Compliant arm: drag the hand, tanh-damped admittance response | ✅ `demos/03_compliant_arm.py` |
| 3 | Scripted pick-and-place skills | ✅ `demos/04_pick_place.py` |
| 4 | Local VLM brain executes a typed command | ⬜ |
| 5 | **Same brain, two bodies: G1 and H1 side by side** | ⬜ |
| 6 | Walk-and-fetch | ⬜ |
| 7 | ISO/TS 15066-style safety layer + evaluation | ⬜ |

## Setup

```bash
python3 -m virtualenv .venv && source .venv/bin/activate
pip install -e .
# robot models (sparse checkout, ~140 MB):
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git third_party/menagerie
git -C third_party/menagerie sparse-checkout set unitree_g1 unitree_h1 robotis_op3
# first demo:
python demos/01_stand.py
# tests (clear PYTHONPATH if a ROS environment is sourced in your shell):
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```

Runs entirely on a laptop (developed on an RTX 3060 Mobile, 6 GB VRAM).

## Honest limitations

- Grasping is a weld constraint in v1, not contact-rich manipulation.
- The VLM chooses *what* and *where* by marker ID; exact 3D poses come from the simulator (Set-of-Mark grounding), not monocular estimation.
- Locomotion starts as kinematic base motion; a pretrained RL policy is a stretch goal.

## License

MIT. Robot models are from [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Unitree G1: Apache-2.0, Unitree H1: BSD-3, Robotis OP3: Apache-2.0) and are not redistributed here.
