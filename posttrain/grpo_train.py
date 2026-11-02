"""GRPO on whole episodes, with the reward read from physics.

For each training task, the current policy (Qwen2.5-VL-3B 4-bit + LoRA, starting
from the SFT adapter) plays G episodes in the real agent loop with sampled,
schema-constrained decoding. Each episode gets one reward:

    r = success (from the physics check) - FAIL_COST * failed_calls - STEP_COST * calls

and its advantage is r minus the group mean, over the group std. Groups where
every episode scored the same carry no signal and are skipped. Each optimiser
step uses fresh on-policy samples once, so the PPO ratio is 1 and the GRPO loss
reduces to -A * mean token log-prob over every planner call of the episode.
No KL term: the learning rate is small and the run is short; held-out success
is checked against the SFT model after training.

Training tasks come from seeds 20000 and up (SFT used 10000-10599, eval uses 0-99).

Usage (training venv):
  python posttrain/grpo_train.py --init ~/ml_data/obmb_post/ckpt/sft_r1/final \
      --out ~/ml_data/obmb_post/ckpt/grpo_r1 --groups 60
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import random
import signal
import sys
import time

import torch
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (build_token_enforcer_tokenizer_data,
                                                        build_transformers_prefix_allowed_tokens_fn)
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from brain.agent_loop import Agent  # noqa: E402
from posttrain.run_expert import EpisodeTimeout, _alarm, build  # noqa: E402
from posttrain.tasks import check, sample_task, snapshot  # noqa: E402
from sim.cameras import CameraRig  # noqa: E402

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_PIXELS = 560 * 400
FAIL_COST = 0.1
STEP_COST = 0.02
EPISODE_LIMIT_S = 300


class PolicyVLM:
    """Stands in for OllamaVLM inside the agent loop: samples from the policy under the
    JSON schema and records exactly what it saw and what it emitted."""

    def __init__(self, processor, model, tok_data, temperature: float):
        self.p, self.m, self.tok_data, self.temperature = processor, model, tok_data, temperature
        self.calls: list[dict] = []

    def encode(self, messages):
        msgs, images = [], []
        for m in messages:
            parts = []
            for b in m.get("images") or []:
                images.append(Image.open(io.BytesIO(b)).convert("RGB"))
                parts.append({"type": "image"})
            parts.append({"type": "text", "text": m["content"]})
            msgs.append({"role": m["role"], "content": parts})
        text = self.p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.p(text=[text], images=images or None, return_tensors="pt")

    def chat(self, messages, schema=None, timeout=None):
        t0 = time.time()
        enc = self.encode(messages)
        kw = {}
        if schema:
            kw["prefix_allowed_tokens_fn"] = build_transformers_prefix_allowed_tokens_fn(
                self.tok_data, JsonSchemaParser(schema))
        with torch.no_grad():
            out = self.m.generate(**enc.to(self.m.device), max_new_tokens=256, do_sample=True,
                                  temperature=self.temperature, top_p=1.0, **kw)
        gen = out[0, enc["input_ids"].shape[1]:].cpu()
        self.calls.append({"enc": {k: v.cpu() for k, v in enc.items()}, "gen": gen})
        return self.p.tokenizer.decode(gen, skip_special_tokens=True), {"seconds": time.time() - t0}


def play(task, vlm) -> dict:
    world, rt = build(task)
    rig = CameraRig(world.model, width=560, height=400)
    agent = Agent(rt, world, rig, vlm=vlm, context="last")
    start = snapshot(world)
    vlm.calls = []
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(EPISODE_LIMIT_S)
    try:
        transcript = agent.run(task["command"], max_steps=10)
        ok = check(task, world, start, rt)["success"]
    except EpisodeTimeout:
        transcript, ok = agent.transcript, False
    finally:
        signal.alarm(0)
    failed = sum(1 for e in transcript if e["tool"] != "done" and not (e.get("result") or {}).get("ok", False))
    reward = float(ok) - FAIL_COST * failed - STEP_COST * len(transcript)
    return {"success": ok, "failed": failed, "n": len(transcript), "reward": reward,
            "calls": list(vlm.calls), "tools": [e["tool"] for e in transcript]}


def call_logp(model, call) -> torch.Tensor:
    """Mean log-prob of the emitted tokens of one planner call under the current policy."""
    enc, gen = call["enc"], call["gen"]
    dev = model.device
    gen = gen.to(dev)
    inputs = {k: v.to(dev) for k, v in enc.items()}
    inputs["input_ids"] = torch.cat([inputs["input_ids"], gen[None]], dim=1)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    k = gen.shape[0]
    logits = model(**inputs, logits_to_keep=k + 1).logits[0, :-1].float()
    return torch.log_softmax(logits, -1).gather(1, gen[:, None]).squeeze(1).mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="SFT adapter to start from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", type=int, default=60, help="informative groups to train on")
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed0", type=int, default=20000)
    a = ap.parse_args()
    out = pathlib.Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    random.seed(0)

    processor = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16,
                                               bnb_4bit_use_double_quant=True))
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, str(pathlib.Path(a.init).expanduser()), is_trainable=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    vlm = PolicyVLM(processor, model, build_token_enforcer_tokenizer_data(processor.tokenizer), a.temperature)

    log = open(out / "log.jsonl", "a")
    seed, trained, tried = a.seed0, 0, 0
    while trained < a.groups:
        task = sample_task(seed)
        seed += 1
        tried += 1
        t0 = time.time()
        model.eval()
        eps = [play(task, vlm) for _ in range(a.G)]
        rewards = torch.tensor([e["reward"] for e in eps])
        rec = {"seed": task["seed"], "family": task["family"], "rewards": [round(r, 3) for r in rewards.tolist()],
               "success": [e["success"] for e in eps], "tools": [e["tools"] for e in eps],
               "rollout_s": round(time.time() - t0, 1)}
        if rewards.std() < 1e-6:
            rec["skipped"] = True
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            continue
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        model.train()
        opt.zero_grad(set_to_none=True)
        n_calls = sum(len(e["calls"]) for e in eps)
        t1 = time.time()
        for e, A in zip(eps, adv.tolist()):
            if abs(A) < 1e-6:
                continue
            for c in e["calls"]:
                loss = -A * call_logp(model, c) / n_calls
                loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        trained += 1
        rec.update({"adv": [round(x, 2) for x in adv.tolist()], "grad_norm": round(gn, 3),
                    "train_s": round(time.time() - t1, 1), "group": trained, "tried": tried})
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if trained % 20 == 0:
            model.save_pretrained(str(out / f"group{trained}"))
    model.save_pretrained(str(out / "final"))
    processor.save_pretrained(str(out / "final"))
    print("saved", out / "final", flush=True)


if __name__ == "__main__":
    main()
