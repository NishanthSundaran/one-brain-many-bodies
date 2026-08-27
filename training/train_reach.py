"""Train the single policy that reaches with its whole body.

Usage: python training/train_reach.py [iters] [workers] [steps_per_iter]
"""

import pathlib
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import training.ppo as ppo_mod  # noqa: E402
from training.ppo import ActorCritic, VecEnv, train  # noqa: E402

OUT = REPO / "third_party" / "policies" / "g1_reach.pt"


def _worker(remote, seed, body):
    import pathlib as _p
    import sys as _s
    _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
    from training.reach_env import ReachEnv

    env = ReachEnv(seed=seed, body=body)
    obs = env.reset()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            obs, rew, done, info = env.step(data)
            if done:
                info = {**info, "episode_end": True}
                obs = env.reset()
            remote.send((obs, rew, done, info))
        elif cmd == "reset":
            remote.send(env.reset())
        elif cmd == "dims":
            remote.send((env.obs_dim, env.act_dim))
        elif cmd == "close":
            remote.close()
            return


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    per_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 128

    ppo_mod._worker = _worker          # same PPO, different environment
    vec = VecEnv(workers)
    policy = ActorCritic(vec.obs_dim, vec.act_dim)
    warm = REPO / "third_party" / "policies" / "g1_reach_warm.pt"
    if warm.exists():
        policy.load_state_dict(torch.load(
            warm, map_location="cpu", weights_only=False)["state"])
        print(f"warm started from {warm.name}")
    print(f"obs {vec.obs_dim}  act {vec.act_dim}  workers {workers}  "
          f"iters {iters}  ({iters * per_iter * workers / 1e6:.1f}M steps)")

    t0 = time.time()
    norm_ref = [None]
    base_norm = ppo_mod.RunningNorm

    class Capturing(base_norm):
        def __init__(self, dim):
            super().__init__(dim)
            norm_ref[0] = self

    ppo_mod.RunningNorm = Capturing
    best = [-1e9]

    def report(it, ret, fell, drift):
        if it % 20 == 0 or it == iters - 1:
            print(f"iter {it:4d} | steps {(it+1)*per_iter*vec.n/1e6:5.2f}M "
                  f"| return {ret:8.1f} | fall rate {fell:4.2f} "
                  f"| hand-to-target {drift:.3f} m | {time.time()-t0:5.0f}s", flush=True)
        if ret > best[0] and it > 30:
            best[0] = ret
            torch.save({"state": policy.state_dict(), "obs_dim": vec.obs_dim,
                        "act_dim": vec.act_dim, "mean": norm_ref[0].mean,
                        "var": norm_ref[0].var}, OUT)

    try:
        norm = train(vec, policy, iters=iters, on_iter=report,
                     steps_per_iter=per_iter)
    finally:
        ppo_mod.RunningNorm = base_norm
        vec.close()
    torch.save({"state": policy.state_dict(), "obs_dim": vec.obs_dim,
                "act_dim": vec.act_dim, "mean": norm.mean, "var": norm.var},
               OUT)
    print(f"saved {OUT} after {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
