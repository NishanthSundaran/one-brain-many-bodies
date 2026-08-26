"""Milestone 0: load the Unitree G1 in MuJoCo, hold the `stand` keyframe
under physics, and record an orbiting-camera GIF to demos/media/01_stand.gif.
"""

import os
import pathlib

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco

REPO = pathlib.Path(__file__).resolve().parents[1]
SCENE = REPO / "third_party" / "menagerie" / "unitree_g1" / "scene.xml"

WIDTH, HEIGHT = 480, 360
FPS = 24
DURATION_S = 5.0


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    key = model.key("stand")
    mujoco.mj_resetDataKeyframe(model, data, key.id)
    data.ctrl[:] = key.ctrl

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.0, 0.0, 0.75)
    cam.distance = 2.2
    cam.elevation = -12.0

    steps_per_frame = max(1, round(1.0 / (FPS * model.opt.timestep)))
    n_frames = int(DURATION_S * FPS)

    frames = []
    for i in range(n_frames):
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        cam.azimuth = 120.0 + 360.0 * i / n_frames
        renderer.update_scene(data, cam)
        frames.append(renderer.render())

    out = REPO / "demos" / "media" / "01_stand.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=FPS, loop=0)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), "
          f"final pelvis height {data.qpos[2]:.3f} m")


if __name__ == "__main__":
    main()
