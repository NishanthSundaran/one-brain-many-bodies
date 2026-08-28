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
| 6c | **Reaching: solved vs learned, side by side** | ✅ `demos/09_reach_ik_vs_rl.py` |
| 6d | **Human-motion metrics + IK vs RL for grasping** | ✅ `eval/` |
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
| hold through a scripted arm disturbance | falls | **3 cm**, stays upright |
| heading drift | uncontrolled | **2°** |
| place the base on a spot | 0.2-2 m out | **2-4 cm** |

Those are training-environment numbers and the second one does not survive
contact with the real skill. Running an actual grasp instead of a scripted
arm sweep (`eval/score_stance.py`, five different start states) the base
drifts **29 +/- 8 cm**, not 3 cm, and the grasp succeeds **3 times in 5**.
The disturbance a real reach applies is not the disturbance the policy was
trained against: a grasp approaches, descends, closes, lifts and carries a
load, over several seconds, and the training sweep does none of that.

### Standing still, measured at the feet

The stance policy holds the pelvis beautifully and gets there by dancing.
`eval/score_stance.py` measures what the pelvis metric cannot see:

| | measured | a person |
|---|---|---|
| foot lifts, standing idle | **21.5 /s** | ~0 |
| foot lifts, during a reach | **20.8 /s** | ~0 |
| knee flexion | 50° | ~5° (this body's home pose is already 38°) |
| knee asymmetry | 9° mean, 22° peak | ~0° |
| pelvis pitch | -9° | ~0° |

With the joints frozen and no policy running, the plant produces 0.5 foot
lifts per second, so the chatter is learned behaviour, not a contact-solver
artefact. The cause is visible in the reward: holding the pelvis still is
worth 3.5, and keeping a foot on the floor was worth 0.2 and asked only that
*one* foot be down. Dancing is the cheapest way to satisfy that.

**Putting the feet in the reward did not fix it, and that is worth
recording.** Two retrains, one gentle (+0.8 both feet down, -3.0 foot height,
-0.3 foot speed) and one aggressive (+2.0 / -12.0 / -1.0), both trained to
convergence:

| | reaching chatter | drift through a reach | grasped |
|---|---|---|---|
| shipped | 17.6 +/- 8.3 /s | **29.1 +/- 8.0 cm** | 3/5 |
| gentle foot reward | 14.8 +/- 7.3 /s | 61.5 +/- 57.3 cm | 3/5 |
| aggressive foot reward | (fails the task) | | 0/1 |

The gentle version looked like a clear win on a single run (21.5 to 17.5
foot lifts per second, drift 31 cm to 23 cm) and that was noise: repeated
across five start states the difference falls inside one standard deviation,
and one run drifted 1.7 m. The aggressive version chatters least of all and
fails the grasp outright, which makes sense once stated plainly: a foot the
policy may never lift is a foot it cannot catch itself with.

So the checkpoint that ships is the original. The lesson is not "foot rewards
do not work", it is that **standing balance here is chaotic enough that a
single run cannot tell you anything**, and that hand-shaping a posture term
by term is the wrong instrument for "stand like a person". A motion-capture
reference specifies that in one tracking term instead of five weights tuned
against each other. That is the same conclusion the gait reached about foot
clearance, now reached twice.

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

### Does it move like a person? A number, not an opinion

"Human-like" is easy to claim and easy to fool yourself about, so it is
measured. `eval/human_motion.py` scores a hand trajectory against the
signatures of human reaching (Flash & Hogan 1985): a near-straight path with
a single-peaked, bell-shaped speed profile.

| what it measures | human |
|---|---|
| straightness (path / direct distance), or absolute bow on short moves | <1.15x, or <12 mm |
| for a reach around a via point, how near it passes the waypoint | <40 mm |
| overshoot past the target | <6% |
| dwell: fraction of the motion halted in its own interior | <8% |
| peak speed / mean speed | 1.5 to 2.6 |
| when peak speed occurs | 30 to 60% |

`eval/score_grasp_motion.py` runs the real grasp and place skills and scores
every motion they ask for. Three of the four now pass every criterion:

| motion | shape | overshoot | dwell | peak/mean | peak at | human |
|---|---|---|---|---|---|---|
| reach to the object | via miss 12 mm | 4% | 4% | 2.32 | 58% | yes |
| lift | bow 20 mm (1.01x) | 0% | 0% | 1.94 | 56% | yes |
| carry to the bin | via miss 13 mm | 0% | 0% | 1.78 | 73% | no |
| retreat | bow 28 mm (1.09x) | 0% | 0% | 1.66 | 46% | yes |

Four things had to change, and each was found by measuring rather than by
watching:

- **Plan the trajectory, do not chase the target.** A position servo gets a
  straight line for free but peaks at 83% of the motion, accelerating into
  the target where a person decelerates into it. Planning minimum jerk in
  time and tracking it moves the peak to the middle.
- **Blend through via points.** Reaching over clutter as "up, stop, across,
  stop, down" is three movements. One Catmull-Rom curve on one minimum-jerk
  clock is one movement, and the terminal descent folds into it rather than
  being its own 4 cm primitive.
- **Give position priority over orientation near the target.** Position and
  palm orientation compete for the same joint velocity in one damped
  least-squares solve. The palm's canonical orientation is not always
  reachable, so its term never converges and sits at its rate cap; once the
  speed profile decelerates, that constant angular demand is the largest
  thing left in the command. Measured: the last stretch of every reach walked
  *away* from the target, 30 mm out to 60 mm. Fading orientation out over the
  last 6 cm took landing error from 60 mm to under 3 mm.
- **Let the correction run straight out of the reach.** Human reaching is a
  fast primary movement plus a short slow correction, not one ballistic
  throw. Pausing between the two is the stop-and-go that reads as a machine,
  and scaling the correction to the error keeps it a coda rather than a third
  of the movement.

The one still failing peaks at 73%: carrying a load, the arm lags its plan
and is still speeding up past the middle. Lead compensation is the textbook
answer and was tried; it made this motion worse, not better.

### Solved or learned? Two reaching controllers, measured

`demos/09_reach_ik_vs_rl.py` gives both the same three targets and the same
physics. Left: measured IK, solved offline, arm only. Right: a policy trained
here (`training/train_reach.py`) driving arm, waist and legs together, 22
joints, so it can lean and shift its stance.

Retraining the policy to track a minimum-jerk reference rather than to
minimise final distance roughly doubled its accuracy:

| | within 6 cm | mean error |
|---|---|---|
| deterministic IK | 0/3 | 0.075 m |
| learned, rewarded on final distance | 3/3 | 0.046 m |
| learned, rewarded on tracking a human trajectory | **3/3** | **0.024 m** |

That is worth stating plainly, because it is the opposite of the usual
trade-off: asking the policy to move the way a person does made it *more*
accurate, not less. Rewarded only on where the hand ends up, it lunged,
hitting peak speed in the first 5% of the motion and tracing five times the
direct distance before settling. Correct, and nothing like a person.

### So which one should grasp?

`eval/score_reach_controllers.py` runs both, on the same targets, in two
scenes: the empty world the policy trained in, and at the bench where
grasping actually happens.

| scene | controller | mean error | within 6 cm | upright |
|---|---|---|---|---|
| open | deterministic IK | 0.408 m | 2/3 | 2/3 |
| open | learned whole-body | **0.022 m** | **3/3** | **3/3** |
| bench | deterministic IK | **0.082 m** | 1/3 | **3/3** |
| bench | learned whole-body | 0.516 m | 1/3 | 2/3 |

**Grasping should use IK.** Not because it reaches better (in the open the
policy is nearly twenty times more accurate) but because grasping happens at
a bench, and the policy has never seen one. It was trained in an empty world,
so a bench is not an obstacle it avoids badly, it is contact dynamics it has
never felt, and its accuracy collapses from 22 mm to half a metre. The IK
path goes the other way: the bench is where it is at its best.

The recommendation, then, is not "IK is better". It is:

- **IK for the grasp itself**, in the scene it was tuned for, where it is
  also the only one of the two that currently produces human-shaped hand
  paths (3 of 4 motions pass every criterion above, against 0 of 6 for the
  policy).
- **The learned policy for whole-body reaching in the open**, where a target
  may be anywhere and the arm alone cannot get there.
- And the un-human motion of the policy is not a fact about RL. It is a fact
  about its reward and its training scene, both of which are fixable: the
  same change that made it track a human trajectory already halved its error.
  Training it in a scene with a bench is the obvious next step.

### Human arm motion costs a balancing robot its balance

On the fixed-base body the arms move on a minimum-jerk profile. On the
balancing body they do not, and that is a measured decision rather than an
oversight. The stance policy learned to reject arm disturbances shaped like
the smoothstep profile it was trained against; swapping in minimum jerk at
the same duration makes the robot reach correctly and then fall over, on
every target. Stretching the move to 1.75x reaches better in isolation (3/3
upright at 63 mm, against 2/3 for the profile it ships with) but costs the
walk-and-fetch, which needs the arm quick enough not to hold a load out
through a long move.

So a balancing body's arms move gently and slightly robotically, on purpose.
Moving them like a person's means retraining the stance policy against
minimum-jerk disturbances: a training job, not a tuning one, and the same
conclusion the gait work reached about foot clearance.

### One policy for both jobs — an honest negative result

A single policy for walking *and* manipulating was trained
(`training/train_locomanip.py`, 9M steps, velocity commands including "stand
exactly here", arms moving throughout) and measured worse at each job than
the two specialists:

| | walking policy | stance policy | unified |
|---|---|---|---|
| hold still, arms idle | ±0.2 m | **0.014 m** | 0.207 m |
| hold through a reach | falls | **0.03 m** | 0.384 m |
| walk 5 s at 0.5 m/s | **2.9 m** | shuffles | 0.35 m |
| walk-and-fetch | — | **succeeds** | fails |

It learned to stay upright with the arms moving, but neither holds a mark nor
travels. Its fall rate was still 0.42 and its return still climbing when the
run ended, so this is under-training, not impossibility — one network learning
velocity tracking, precise position holding *and* arm-disturbance rejection
needs far more than the 9M steps that were plenty for either alone. The
checkpoint and trainer are kept; `unified_policy` in the body config switches
it on.

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
- **Walking and precise placement are done by different controllers.** The
  pretrained policy has the better gait but cannot be steered to a spot —
  commanded to a target it orbits and eventually falls, landing a metre or
  two out. Navigation in `demos/08` therefore uses the stance policy, moving
  its hold point a stride at a time: slower, and accurate to a few
  centimetres. Getting one policy to do both is the honest next step.
- The workspace is tuned tightly around the measured reach envelope (~0.43 m
  at bench height in the stance the policy adopts). Objects inboard of the
  shoulder line make the IK roll the upper arm into the torso, and objects
  beyond ~0.40 m are simply out of range; there is little slack either way.

## License

MIT. Robot models are from [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Unitree G1: Apache-2.0, Unitree H1: BSD-3, Robotis OP3: Apache-2.0) and are not redistributed here.
