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
