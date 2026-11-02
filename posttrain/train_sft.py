"""Supervised fine-tuning (QLoRA) of Qwen2.5-VL-3B-Instruct on the expert's
planner calls, collected by collect_sft.py. Runs in the training venv
(~/ml_data/venvs/obmb-post), on a 6 GB GPU: the base model is loaded in 4-bit
NF4 and only LoRA adapters on the language model are trained. The loss covers
the expert's answer only, not the prompt.

Usage: python posttrain/train_sft.py --data ~/ml_data/obmb_post/sft --out ~/ml_data/obmb_post/ckpt/sft_r1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import random

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import (AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration,
                          Trainer, TrainingArguments)

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_PIXELS = 560 * 400   # the agent's camera frame: no extra downscaling


class Rows(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Collator:
    def __init__(self, processor):
        self.p = processor

    def __call__(self, batch):
        assert len(batch) == 1, "variable-length multimodal rows: batch 1 + gradient accumulation"
        r = batch[0]
        msgs = r["messages"]
        image = Image.open(r["image"]).convert("RGB")
        full = self.p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt = self.p.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        enc = self.p(text=[full], images=[image], return_tensors="pt")
        n_prompt = self.p(text=[prompt], images=[image], return_tensors="pt")["input_ids"].shape[1]
        labels = enc["input_ids"].clone()
        labels[:, :n_prompt] = -100           # learn the answer only
        enc["labels"] = labels
        return enc


def load_rows(data_dir: str):
    rows = []
    for f in sorted(glob.glob(os.path.join(os.path.expanduser(data_dir), "sft_*.jsonl"))):
        rows += [json.loads(l) for l in open(f)]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only this many rows")
    a = ap.parse_args()

    rows = load_rows(a.data)
    seeds = sorted({r["seed"] for r in rows})
    random.Random(0).shuffle(seeds)
    val_seeds = set(seeds[: max(1, len(seeds) // 10)])     # split by episode, not by step
    train = [r for r in rows if r["seed"] not in val_seeds]
    val = [r for r in rows if r["seed"] in val_seeds]
    if a.limit:
        train, val = train[: a.limit], val[: max(1, a.limit // 10)]
    print(f"{len(rows)} rows from {len(seeds)} episodes: train {len(train)}, val {len(val)}", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16,
                                               bnb_4bit_use_double_quant=True))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      # language model only; the vision tower stays frozen
                      target_modules=r".*model\.language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=a.accum, lr_scheduler_type="cosine", warmup_ratio=0.05,
        bf16=True, logging_steps=10, eval_strategy="steps", eval_steps=100,
        save_strategy="steps", save_steps=100, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, dataloader_num_workers=2, report_to=[], seed=0)
    trainer = Trainer(model=model, args=args, train_dataset=Rows(train), eval_dataset=Rows(val),
                      data_collator=Collator(processor))
    trainer.train()
    final = pathlib.Path(a.out) / "final"
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
