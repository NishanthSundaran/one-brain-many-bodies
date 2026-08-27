"""A compact PPO, sized for this job: a 15-DoF stance task on CPU.

Deliberately small — the environment runs at ~16k physics steps/s per core,
so the bottleneck is rollout collection, not the network. Worker processes
each own a MuJoCo model and step it independently.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        def trunk():
            return nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ELU(),
                nn.Linear(hidden, hidden), nn.ELU(),
                nn.Linear(hidden, 128), nn.ELU())
        self.pi = trunk()
        self.vf = trunk()
        self.mu = nn.Linear(128, act_dim)
        self.v = nn.Linear(128, 1)
        self.log_std = nn.Parameter(-0.7 * torch.ones(act_dim))
        for layer in (self.mu,):
            nn.init.orthogonal_(layer.weight, 0.01)
            nn.init.zeros_(layer.bias)

    def dist(self, obs):
        mu = self.mu(self.pi(obs))
        return torch.distributions.Normal(mu, self.log_std.exp())

    def value(self, obs):
        return self.v(self.vf(obs)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        d = self.dist(obs)
        a = d.mean if deterministic else d.sample()
        return a, d.log_prob(a).sum(-1), self.value(obs)


def _worker(remote, seed, body):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from training.stance_env import StanceEnv

    env = StanceEnv(seed=seed, body=body)
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


class VecEnv:
    def __init__(self, n: int, body: str = "g1_walk", seed0: int = 0):
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for i in range(n):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, seed0 + i, body),
                            daemon=True)
            p.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(p)
        self.n = n
        self.remotes[0].send(("dims", None))
        self.obs_dim, self.act_dim = self.remotes[0].recv()

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return np.stack([r.recv() for r in self.remotes])

    def step(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(("step", a))
        out = [r.recv() for r in self.remotes]
        obs, rew, done, info = zip(*out)
        return np.stack(obs), np.array(rew, np.float32), np.array(done), info

    def close(self):
        for r in self.remotes:
            r.send(("close", None))
        for p in self.procs:
            p.join(timeout=2)


class RunningNorm:
    """Observation normalisation — a stance task has wildly different scales
    (gravity ~1, joint velocities ~10) and PPO will not learn without it."""

    def __init__(self, dim):
        self.mean = np.zeros(dim, np.float64)
        self.var = np.ones(dim, np.float64)
        self.count = 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        self.var = (m_a + m_b + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10)


def train(vec, policy, steps_per_iter=64, iters=300, gamma=0.99, lam=0.95,
          clip=0.2, lr=3e-4, epochs=4, minibatches=4, on_iter=None):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    norm = RunningNorm(vec.obs_dim)
    obs = vec.reset()
    norm.update(obs)
    ep_ret = np.zeros(vec.n)
    recent, falls, drifts = [], [], []

    for it in range(iters):
        O, A, LP, V, R, D = [], [], [], [], [], []
        for _ in range(steps_per_iter):
            nobs = norm(obs)
            t = torch.as_tensor(nobs, dtype=torch.float32)
            a, lp, v = policy.act(t)
            an = a.numpy()
            nxt, rew, done, infos = vec.step(np.clip(an, -1, 1))
            O.append(nobs); A.append(an); LP.append(lp.numpy())
            V.append(v.numpy()); R.append(rew); D.append(done.astype(np.float32))
            ep_ret += rew
            for i, inf in enumerate(infos):
                if inf.get("episode_end"):
                    recent.append(ep_ret[i]); ep_ret[i] = 0.0
                    falls.append(float(inf["fell"])); drifts.append(inf["drift"])
            obs = nxt
        norm.update(obs)

        with torch.no_grad():
            last_v = policy.value(
                torch.as_tensor(norm(obs), dtype=torch.float32)).numpy()
        O = np.array(O); A = np.array(A); LP = np.array(LP)
        V = np.array(V); R = np.array(R); D = np.array(D)
        adv = np.zeros_like(R); gae = 0.0
        for t in reversed(range(steps_per_iter)):
            nv = last_v if t == steps_per_iter - 1 else V[t + 1]
            delta = R[t] + gamma * nv * (1 - D[t]) - V[t]
            gae = delta + gamma * lam * (1 - D[t]) * gae
            adv[t] = gae
        ret = adv + V
        b_obs = torch.as_tensor(O.reshape(-1, vec.obs_dim), dtype=torch.float32)
        b_act = torch.as_tensor(A.reshape(-1, vec.act_dim), dtype=torch.float32)
        b_lp = torch.as_tensor(LP.reshape(-1), dtype=torch.float32)
        b_adv = torch.as_tensor(adv.reshape(-1), dtype=torch.float32)
        b_ret = torch.as_tensor(ret.reshape(-1), dtype=torch.float32)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        n = b_obs.shape[0]
        idx = np.arange(n)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for mb in np.array_split(idx, minibatches):
                mb = torch.as_tensor(mb)
                dist = policy.dist(b_obs[mb])
                lp = dist.log_prob(b_act[mb]).sum(-1)
                ratio = (lp - b_lp[mb]).exp()
                pg = -torch.min(ratio * b_adv[mb],
                                torch.clamp(ratio, 1 - clip, 1 + clip) * b_adv[mb]).mean()
                vloss = ((policy.value(b_obs[mb]) - b_ret[mb]) ** 2).mean()
                loss = pg + 0.5 * vloss - 0.004 * dist.entropy().sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.8)
                opt.step()

        if on_iter and recent:
            on_iter(it, float(np.mean(recent[-40:])),
                    float(np.mean(falls[-40:])), float(np.mean(drifts[-40:])))
    return norm
