"""Fine-tune the walking policy to stop on a zero command (walk_stop_env).

PPO with the network on the GPU and MuJoCo in CPU worker processes. The
actor starts as an exact copy of g1_policy.onnx; a fresh critic is warmed
up first with the actor frozen, so early updates are not driven by a
random value function. While the command is non-zero the actor is anchored
to the original policy (squared distance between action means), so walking
is not traded away for stopping.

    python training/train_walkstop.py [iters] [workers]
Checkpoints: third_party/policies/walkstop/it{N}.pt, every 50 iterations.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from controllers.walk_actor import WalkActor  # noqa: E402

OUT = REPO / "third_party" / "policies" / os.environ.get("WALKSTOP_OUT", "walkstop")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _worker(remote, seed):
    sys.path.insert(0, str(REPO))
    from training.walk_stop_env import WalkStopEnv
    env = WalkStopEnv(seed=seed)
    o, p = env.reset()
    remote.send((o, p))
    while True:
        cmd, u = remote.recv()
        if cmd == "close":
            return
        o, p, r, done, info = env.step(u)
        if done:
            info = {**info, "end": True}
            o, p = env.reset()
        remote.send((o, p, r, done, info))


class Vec:
    def __init__(self, n, seed0=0):
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for i in range(n):
            a, b = ctx.Pipe()
            pr = ctx.Process(target=_worker, args=(b, seed0 + i), daemon=True)
            pr.start(); b.close()
            self.remotes.append(a); self.procs.append(pr)
        first = [r.recv() for r in self.remotes]
        self.obs = np.stack([f[0] for f in first])
        self.priv = np.stack([f[1] for f in first])

    def step(self, U):
        for r, u in zip(self.remotes, U):
            r.send(("step", u))
        out = [r.recv() for r in self.remotes]
        o, p, rew, done, info = zip(*out)
        self.obs, self.priv = np.stack(o), np.stack(p)
        return np.array(rew, np.float32), np.array(done), info

    def close(self):
        for r in self.remotes:
            r.send(("close", None))


class Critic(nn.Module):
    def __init__(self, actor: WalkActor, priv_dim=2):
        super().__init__()
        self.register_buffer("m", actor.obs_mean.clone())
        self.register_buffer("s", actor.obs_inv_std.clone())
        self.net = nn.Sequential(nn.Linear(103 + priv_dim, 512), nn.ELU(),
                                 nn.Linear(512, 256), nn.ELU(),
                                 nn.Linear(256, 1))

    def forward(self, obs, priv):
        return self.net(torch.cat([(obs - self.m) * self.s, priv], -1)).squeeze(-1)


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    T = int(os.environ.get("WALKSTOP_STEPS", 128))
    lr = float(os.environ.get("WALKSTOP_LR", 3e-5))
    beta = float(os.environ.get("WALKSTOP_ANCHOR", 1.0))
    warmup = int(os.environ.get("WALKSTOP_WARMUP", 40))
    gamma, lam, clip, epochs, mbs = 0.99, 0.95, 0.2, 5, 4
    OUT.mkdir(parents=True, exist_ok=True)

    actor = WalkActor.from_onnx(REPO / "third_party/policies/g1_policy.onnx").to(DEV)
    warm = os.environ.get("WALKSTOP_WARM")
    if warm:   # continue from a fine-tuned checkpoint; the anchor stays the original
        actor.load_state_dict(torch.load(REPO / warm, map_location=DEV)["actor"])
    anchor = WalkActor.from_onnx(REPO / "third_party/policies/g1_policy.onnx").to(DEV).eval()
    for q in anchor.parameters():
        q.requires_grad_(False)
    log_std = nn.Parameter(torch.full((29,), np.log(float(os.environ.get("WALKSTOP_STD", 0.10))), device=DEV))
    critic = Critic(actor).to(DEV)
    opt_a = torch.optim.Adam(list(actor.parameters()) + [log_std], lr=lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=3e-4)

    vec = Vec(workers)
    print(f"device {DEV} ({torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'cpu'})"
          f"  workers {workers}  steps/iter {T * workers}  lr {lr}  anchor {beta}",
          flush=True)
    t0 = time.time()
    ep_ret = np.zeros(workers)
    hist = {"ret": [], "fell": [], "stop_speed": [], "drift": [], "fell_walk": []}
    last_drift = np.zeros(workers)
    reach = np.zeros(workers)
    stop_speeds = [[] for _ in range(workers)]

    for it in range(iters):
        O, P, U, LP, V, R, D, W = ([] for _ in range(8))
        for _ in range(T):
            o = torch.as_tensor(vec.obs, device=DEV)
            p = torch.as_tensor(vec.priv, device=DEV)
            with torch.no_grad():
                mu = actor.loc(o)
                dist = torch.distributions.Normal(mu, log_std.exp())
                u = dist.sample()
                lp = dist.log_prob(u).sum(-1)
                v = critic(o, p)
            walking = (vec.priv[:, 1] < 0.5).astype(np.float32)
            rew, done, infos = vec.step(u.cpu().numpy())
            O.append(o); P.append(p); U.append(u); LP.append(lp); V.append(v)
            R.append(rew); D.append(done.astype(np.float32)); W.append(walking)
            ep_ret += rew
            for i, inf in enumerate(infos):
                if inf["stopped"]:
                    stop_speeds[i].append(inf["speed"])
                if inf["arms"]:
                    last_drift[i] = inf["drift"]
                reach[i] = inf.get("reach", 0.0)
                if inf.get("end"):
                    hist["ret"].append(ep_ret[i]); ep_ret[i] = 0
                    hist["fell"].append(float(inf["fell"]))
                    hist["fell_walk"].append(float(inf.get("fell_walking", 0)))
                    s = stop_speeds[i]
                    if len(s) > 100:      # mean speed from 1 s after the stop
                        hist["stop_speed"].append(float(np.mean(s[50:])))
                    stop_speeds[i] = []
                    if last_drift[i] > 0:
                        hist["drift"].append(last_drift[i]); last_drift[i] = 0
        with torch.no_grad():
            last_v = critic(torch.as_tensor(vec.obs, device=DEV),
                            torch.as_tensor(vec.priv, device=DEV)).cpu().numpy()
        V_np = torch.stack(V).cpu().numpy()
        R = np.array(R); D = np.array(D)
        adv = np.zeros_like(R); g = 0.0
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else V_np[t + 1]
            delta = R[t] + gamma * nv * (1 - D[t]) - V_np[t]
            g = delta + gamma * lam * (1 - D[t]) * g
            adv[t] = g
        ret = adv + V_np
        b_o = torch.cat(O); b_p = torch.cat(P); b_u = torch.cat(U)
        b_lp = torch.cat(LP)
        b_adv = torch.as_tensor(adv.reshape(-1), device=DEV)
        b_ret = torch.as_tensor(ret.reshape(-1), device=DEV)
        b_w = torch.as_tensor(np.array(W).reshape(-1), device=DEV)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        with torch.no_grad():
            b_mu0 = anchor.loc(b_o)

        n = b_o.shape[0]
        train_actor = it >= warmup
        for _ in range(epochs):
            perm = torch.randperm(n, device=DEV)
            for mb in perm.chunk(mbs):
                vloss = ((critic(b_o[mb], b_p[mb]) - b_ret[mb]) ** 2).mean()
                opt_c.zero_grad(); vloss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0); opt_c.step()
                if not train_actor:
                    continue
                mu = actor.loc(b_o[mb])
                dist = torch.distributions.Normal(mu, log_std.exp())
                ratio = (dist.log_prob(b_u[mb]).sum(-1) - b_lp[mb]).exp()
                pg = -torch.min(ratio * b_adv[mb],
                                ratio.clamp(1 - clip, 1 + clip) * b_adv[mb]).mean()
                anc = (((mu - b_mu0[mb]) ** 2).sum(-1) * b_w[mb]).mean()
                loss = pg + beta * anc - 0.001 * dist.entropy().sum(-1).mean()
                opt_a.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + [log_std], 0.5)
                opt_a.step()

        if it % 10 == 0 or it == iters - 1:
            f = lambda k: np.mean(hist[k][-60:]) if hist[k] else float("nan")
            print(f"iter {it:4d} | {(it + 1) * T * workers / 1e6:5.2f}M steps "
                  f"| return {f('ret'):7.1f} | fall {f('fell'):.2f} "
                  f"| fall walking {f('fell_walk'):.3f} "
                  f"| speed after stop {f('stop_speed'):.3f} m/s "
                  f"| reach drift {f('drift'):.3f} m | reach {reach.mean():.2f} "
                  f"| std {log_std.exp().mean().item():.3f} "
                  f"| {'actor' if train_actor else 'critic warmup'} "
                  f"| {time.time() - t0:5.0f}s", flush=True)
        if (it + 1) % 50 == 0 or it == iters - 1:
            torch.save({"actor": {k: v.cpu() for k, v in actor.state_dict().items()},
                        "iter": it + 1,
                        "act_filter": float(os.environ.get("WALKSTOP_ACT_FILTER", 1.0)),
                        "gait_freq": os.environ.get("WALKSTOP_DEPLOY_FREQ")},
                       OUT / f"it{it + 1}.pt")
    vec.close()


if __name__ == "__main__":
    main()
