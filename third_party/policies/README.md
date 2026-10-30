# Pretrained locomotion policies

`g1_policy.onnx` — Unitree G1 joystick locomotion policy from
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
(`mujoco_playground/experimental/sim2sim/onnx/g1_policy.onnx`, Apache-2.0),
trained with MJX/PPO on the same Menagerie G1 used here.

Interface: observation 103 floats, output 29 continuous actions applied as
`ctrl = action * 0.5 + default_angles` at 50 Hz. Observation layout:
`[local_linvel(3), gyro(3), projected_gravity(3), command(3),
joint_angles - default(29), joint_velocities(29), last_action(29),
gait_phase(4)]` in the canonical G1 joint order (legs, waist, arms).

## Trained in this repo

| file | what | trained with |
|---|---|---|
| `g1_walk_human.pt` | walk and stop, human cadence, action filter (used by demo 13) | `training/train_walkstop.py` |
| `g1_walk_arms.pt` | earlier walk and stop policy, used to record the arm and stop data sets | `training/train_walkstop.py` |
| `g1_stance_human2.pt` | stand and manipulate, trained from `g1_walk_human` stop states (used by demo 13) | `training/train_stance.py` |
| `g1_stance_human.pt` | previous stance round (C) | `training/train_stance.py` |
| `g1_stance_humanA.pt` | stance round A, used to record `training/arm_traj.npz` | `training/train_stance.py` |

Per-iteration checkpoints are not committed.
