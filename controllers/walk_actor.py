"""The walking policy as a PyTorch module, so it can be fine-tuned.

g1_policy.onnx is a Brax PPO export: fixed observation normalisation, an
MLP 103-512-256-128-58 with SiLU, and an output split into 29 action means
and 29 scale parameters; the action is tanh(mean). `from_onnx` copies the
weights exactly, and the runtime loads either file (LocomotionPolicy).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

ONNX_KEYS = {
    "mean": "mlp_1/sub/ReadVariableOp:0",
    "inv_std": "ConstantFolding/mlp_1/truediv_recip:0",
    "layers": [f"mlp_1/MLP_0_1/hidden_{i}_1" for i in range(4)],
}


class WalkActor(nn.Module):
    def __init__(self, obs_dim: int = 103, act_dim: int = 29):
        super().__init__()
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_inv_std", torch.ones(obs_dim))
        dims = [obs_dim, 512, 256, 128]
        self.trunk = nn.Sequential(*[m for a, b in zip(dims, dims[1:])
                                     for m in (nn.Linear(a, b), nn.SiLU())])
        self.head = nn.Linear(128, 2 * act_dim)
        self.act_dim = act_dim

    def loc(self, obs: torch.Tensor) -> torch.Tensor:
        """Pre-tanh action mean."""
        x = (obs - self.obs_mean) * self.obs_inv_std
        return self.head(self.trunk(x))[..., :self.act_dim]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.loc(obs))

    @classmethod
    def from_onnx(cls, path) -> "WalkActor":
        import onnx
        from onnx import numpy_helper
        g = onnx.load(str(path)).graph
        w = {t.name: numpy_helper.to_array(t) for t in g.initializer}
        net = cls()
        with torch.no_grad():
            net.obs_mean.copy_(torch.as_tensor(w[ONNX_KEYS["mean"]]))
            net.obs_inv_std.copy_(torch.as_tensor(w[ONNX_KEYS["inv_std"]]))
            linears = [m for m in net.trunk if isinstance(m, nn.Linear)]
            linears.append(net.head)
            for lin, key in zip(linears, ONNX_KEYS["layers"]):
                lin.weight.copy_(torch.as_tensor(
                    w[f"{key}/Cast/ReadVariableOp:0"].T))
                lin.bias.copy_(torch.as_tensor(
                    w[f"{key}/BiasAdd/ReadVariableOp:0"]))
        return net


class TorchSession:
    """Drop-in for the onnxruntime session LocomotionPolicy calls."""

    def __init__(self, path):
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        self.net = WalkActor()
        self.net.load_state_dict(ck["actor"] if "actor" in ck else ck)
        self.net.eval()
        # trained-with settings travel with the weights (absent = original)
        self.act_filter = float(ck.get("act_filter", 1.0)) if "actor" in ck else 1.0
        f = ck.get("gait_freq") if "actor" in ck else None
        self.gait_freq = float(f) if f else None

    def run(self, _names, feeds):
        with torch.no_grad():
            a = self.net(torch.as_tensor(np.asarray(feeds["obs"]),
                                         dtype=torch.float32))
        return [a.numpy()]
