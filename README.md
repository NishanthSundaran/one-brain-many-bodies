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
| 4 | Local VLM brain executes a typed command | ✅ `demos/05_vlm_brain.py` |
| 5 | **Same brain, two bodies: G1 and H1 side by side** | ✅ `demos/06_cross_embodiment.py` |
| 6 | **Physics-based walking** (learned policy, real balance) | ✅ `demos/07_physics_walk.py` |
| 6b | **Walk over and fetch it, balancing throughout** | ✅ `demos/08_walk_and_fetch.py` |
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

## Locomotion

The G1 walks under a pretrained RL policy (MJX/PPO, from MuJoCo Playground)
running in closed loop at 50 Hz on the real IMU and joint state. Nothing
about the base is scripted — no glide, no animation, no weld — so a wrong
command makes it fall over. Two things were needed to run someone else's
policy on this body:

- **Matching the plant.** A policy is only valid for the dynamics it learned
  on. Menagerie ships the G1 with `kp=500` position actuators; the policy was
  trained at `kp=75/20/2` with explicit joint damping and armature under an
  Euler integrator. Under the stiffer defaults it falls instantly, so
  `bodies/g1_walk/config.yaml` carries the training values and
  `Body.apply_locomotion_dynamics` applies them after the model compiles.
- **Station keeping.** It is a *locomotion* policy: commanded zero velocity,
  it still creeps forward at ~0.25 m/s. Standing on a spot is therefore its
  own closed-loop problem, solved by a PI controller on the base pose whose
  integral term learns the creep (`Runtime._hold_station`).

### Standing still is its own skill

The borrowed walking policy cannot manipulate: told to hold position it
creeps forward at ~0.25 m/s, wanders in a ±0.2 m band, and treats a sustained
arm reach as something to step away from — it falls, every time. Freezing the
legs is worse, because a frozen robot cannot absorb the momentum of its own
arm.

So `training/train_stance.py` trains one. The arms are the *environment*: on
every episode they are driven through the poses a real reach visits, and the
policy (legs and waist, 15 DoF) is rewarded for keeping the pelvis on its
mark. PPO on CPU, ~3,300 control-steps/s across 14 workers — about half an
hour per run on a laptop, no GPU involved.

| | walking policy | trained stance policy |
|---|---|---|
| hold a mark, idle | ±0.2 m | **1.4 cm** |
| hold through a full arm reach | falls | **3 cm**, stays upright |
| heading drift | uncontrolled | **2°** |
| place the base on a spot | 0.2-2 m out | **2-4 cm** |

Three lessons cost real time and are worth stating plainly:

- **A reward that saturates teaches nothing.** The first drift term was
  `exp(-(drift/0.05)^2)`, which is numerically zero past 0.1 m, so a policy
  sitting at 0.3 m got no gradient telling it 0.15 m was better. It plateaued
  for 2M steps. A linear term alongside the Gaussian fixed it.
- **Train on the poses you will actually use.** Sampling arm disturbances
  symmetrically around the arms' resting pose covered ±1.1 rad of shoulder
  pitch; a real reach goes to -1.2. The policy balanced against everything it
  had seen and fell the moment the arm went somewhere new.
- **Position is not enough — hold the heading too.** With no yaw term the
  robot slowly turned on the spot, which puts the bench where its arms are
  not, and is exactly as fatal to a reach as drifting away from it.

### How human is the gait?

`demos/07_physics_walk.py` walks the same robot twice and measures it. Tuning
the policy's gait-frequency input and letting it keep its arms roughly doubled
the stride and the arm swing:

| metric | before | after | human |
|---|---|---|---|
| speed (m/s) | 0.52 | 0.89 | 1.2-1.4 (0.9-1.2 Froude-scaled) |
| cadence (steps/s) | 2.9 | 2.6 | 1.8-2.0 (2.2-2.5 scaled) |
| stride (m) | 0.18 | 0.35 | ~0.26 for this leg length |
| foot clearance (m) | 0.08 | 0.08 | 0.02-0.05 |
| pelvis bob (m) | 0.06 | 0.10 | 0.04-0.05 |
| arm swing | 6° | 15° | 25-35° |

Cadence is compared after Froude scaling: a 0.65 m leg does not walk at a
human's 1.9 steps/s. Below 1.2 Hz the policy leaves the distribution it was
trained in and falls over, so that is the floor.

Stride, cadence and arm swing are now in a human range; **foot clearance,
pelvis bounce and torso lean are not** — the robot still lifts its feet about
twice as high as a person and bounces twice as much. Those are properties of
the policy's learned gait, not parameters, so closing that gap means training
a policy against human motion capture (AMP / motion-imitation style rewards)
rather than tuning this one.

## Honest limitations

- Grasping: both robots' fingers visibly close around objects, but the hold
  itself is a weld constraint and the *finger* collisions are disabled, so
  finger closure is cosmetic rather than contact-rich manipulation. Palms
  keep their collision geometry, so hands cannot pass through bins or tables.
- The H1 ships with bare ball wrists; this repo bolts Sharpa Wave five-finger
  hands (22 DoF each) onto it at load time via `hand_attach` in its body
  config — a demonstration of the body layer composing hardware, not a
  configuration Unitree ships.
- The VLM chooses *what* and *where* by marker ID; exact 3D poses come from the simulator (Set-of-Mark grounding), not monocular estimation.
- Locomotion starts as kinematic base motion; a pretrained RL policy is a stretch goal.

## License

MIT. Robot models are from [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Unitree G1: Apache-2.0, Unitree H1: BSD-3, Robotis OP3: Apache-2.0) and are not redistributed here.
