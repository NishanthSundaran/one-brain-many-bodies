"""Train the stance policy and save it for the runtime to load.

Usage: python training/train_stance.py [iters] [workers]
"""

import pathlib
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.ppo import ActorCritic, VecEnv, train  # noqa: E402

OUT = REPO / "third_party" / "policies" / "g1_stance.pt"


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    per_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    vec = VecEnv(workers)
    policy = ActorCritic(vec.obs_dim, vec.act_dim)
    warm = REPO / "third_party" / "policies" / "g1_stance_warm.pt"
    if warm.exists():   # continue a previous run instead of starting over
        policy.load_state_dict(torch.load(
            warm, map_location="cpu", weights_only=False)["state"])
        print(f"warm started from {warm.name}")
    print(f"obs {vec.obs_dim}  act {vec.act_dim}  workers {workers}  "
          f"iters {iters}")
    t0 = time.time()
    best = [-1e9]

    def report(it, ret, fell, drift):
        if it % 10 == 0 or it == iters - 1:
            done = (it + 1) * per_iter * vec.n
            print(f"iter {it:4d} | steps {done/1e6:5.2f}M | return {ret:8.1f} "
                  f"| fall rate {fell:4.2f} | drift {drift:.3f} m "
                  f"| {time.time()-t0:5.0f}s", flush=True)
        if ret > best[0] and it > 20:
            best[0] = ret
            torch.save({"state": policy.state_dict(),
                        "obs_dim": vec.obs_dim, "act_dim": vec.act_dim,
                        "mean": norm_ref[0].mean, "var": norm_ref[0].var},
                       OUT)

    norm_ref = [None]
    # train() builds the normaliser; capture it so checkpoints carry it
    import training.ppo as ppo_mod
    orig = ppo_mod.RunningNorm

    class Capturing(orig):
        def __init__(self, dim):
            super().__init__(dim)
            norm_ref[0] = self

    ppo_mod.RunningNorm = Capturing
    try:
        norm = train(vec, policy, iters=iters, on_iter=report,
                     steps_per_iter=per_iter)
    finally:
        ppo_mod.RunningNorm = orig
        vec.close()

    torch.save({"state": policy.state_dict(), "obs_dim": vec.obs_dim,
                "act_dim": vec.act_dim, "mean": norm.mean, "var": norm.var},
               OUT)
    print(f"saved {OUT} after {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
