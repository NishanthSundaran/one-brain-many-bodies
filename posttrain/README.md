# Planner post-training: SFT, DAgger and GRPO for the Qwen2.5-VL-3B brain

The brain of this project is a 3B vision-language model that reads a camera frame with numbered
markers and calls tools (`grasp`, `place`, `done`) to carry out a table-top command on the Unitree G1.
Out of the box it solves 23 of 100 held-out tasks. After supervised fine-tuning on a privileged
expert, one DAgger round and a GRPO run with rewards from physics, it solves 92 of 100, two short of the
expert that taught it. Everything runs on one laptop GPU (RTX 3060, 6 GB).

| planner | success | bin | two_bin | next_to | side | absent | already | failed calls | calls per solved task |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-3B base | **23/100** | 13/28 | 0/16 | 1/25 | 0/16 | 1/4 | 8/11 | 418 | 5.09 |
| SFT | **83/100** | 22/28 | 11/16 | 20/25 | 15/16 | 4/4 | 11/11 | 46 | 4.07 |
| SFT + DAgger | **92/100** | 26/28 | 14/16 | 22/25 | 15/16 | 4/4 | 11/11 | 28 | 4.15 |
| SFT + DAgger + GRPO | **92/100** | 26/28 | 14/16 | 22/25 | 15/16 | 4/4 | 11/11 | 27 | 4.14 |
| privileged expert (ceiling) | **94/100** | 27/28 | 14/16 | 24/25 | 14/16 | 4/4 | 11/11 | n/a | 4.29 |

Paired on the same 100 tasks (exact McNemar): SFT beats base 61 to 1 (p = 3e-17); DAgger beats SFT
11 to 2 (p = 0.022); GRPO and DAgger solve exactly the same tasks (0 to 0). Regenerate with
`python posttrain/summarize.py`; every episode, with its full tool-call transcript, is in `results/`.

## The benchmark

`tasks.py` samples a seeded scene (three objects from four blocks and two cups, plus a bin) and a command
from six families:

* **bin** and **two_bin**: put one or two named objects in the bin;
* **next_to**: put X next to Y (centres 5 to 16 cm apart);
* **side**: move X to the left or right half of the table;
* **absent**: the named object is not on the table, so the right answer is to do nothing;
* **already**: the goal already holds, so the right answer is `done` at once.

Success is read from the physics state at the end (`check`), never from what the model says.
Anything still in a hand, or any other object pushed more than 3 cm, is a failure.

Every task is feasible by construction, and that took measuring. A sweep (`sweep_reach.py`) of real
grasps over a 9 x 21 grid gave the reach map: the fixed-base G1 grasps blocks only in a narrow band
(x 0.28 to 0.32 m, |y| at least 0.06 m), while both arms reach a cup at the midline. So a cup can be
handed across the table (one arm sets it down at (0.29, 0), the other picks it up) and a block cannot.
The sampler only asks for cross-table moves of cups.

The expert (`expert` in `tasks.py`) reads the true state and returns the next tool call with a short
reason: which arm, when to hand over, and which free spot to place on. It solves 94/100 eval tasks;
the other 6 are physics failures (three cups that miss the bin, one cup that ends on the wrong side after
a handover, two neighbours nudged during a place).

Eval uses seeds 0 to 99. Training never sees them: SFT uses 10000+, DAgger 30000+, GRPO 20000+.

## What each stage does

**Base.** Qwen2.5-VL-3B-Instruct in 4-bit NF4, served by `serve_vlm.py` behind the same Ollama
`/api/chat` interface the agent always used, with JSON-schema constrained decoding and greedy
decoding for eval. It succeeds where the task is one obvious grasp and place. It fails on everything
that needs arm reasoning: placing with the empty hand, grasping a block the arm cannot reach, never
using the handover, "fixing" tasks that are already done, and moving some other object when the named
one is absent. 418 of its calls fail.

**SFT** (`collect_sft.py`, `train_sft.py`). The real agent loop runs with the expert answering in
place of the VLM, so each example is exactly the prompt the planner sees (system prompt, marked camera
frame, marker list with positions and nearest arm, action history, last result) paired with the
expert's call. 575 of 600 training episodes succeed in physics; their 2432 steps are the data.
QLoRA (rank 16, language model only, vision tower frozen), loss on the answer tokens only,
2 epochs, validation split by episode. About 4 hours.

**DAgger** (`collect_dagger.py`). SFT data only covers states on the expert's own path. When the SFT
planner slips (a handover set down a few cm off, a place that fails), it lands in states it was never
shown and tends to repeat the failing call or declare `done`. One DAgger round: the SFT planner drives
250 training episodes (temperature 0.2), the expert labels every state it reaches, and training
continues from the SFT adapter on old plus new data (3529 rows, 1 epoch). That fixed 11 tasks and broke 2.

**GRPO** (`grpo_train.py`). Whole episodes in the real loop, 4 samples per task at temperature 1.0,
reward = physics success minus 0.1 per failed call minus 0.02 per call, advantage normalised within the
group, one on-policy update per group (so the PPO ratio is 1), no KL term, LoRA lr 2e-5. Run on the
three families that still failed (bin, two_bin, next_to): 147 tasks rolled out, 97 skipped because
all four samples scored the same, 50 updates, 4.9 hours.

## GRPO did not move the held-out result, and why

The GRPO adapter differs from the DAgger one by 2.5% in weight norm, and its greedy choice is
identical on all 100 eval tasks. Three reasons, all visible in `results/grpo_r1_train_log.jsonl`:

1. **Little left to win.** After DAgger the planner fails 8 tasks, and the expert fails 4 of
   those 8 too. That leaves 4 failures a perfect planner could fix, out of 100.
2. **Most groups carry no signal.** The planner is confident: 66% of groups had four identical
   rewards, so each update rests on a handful of disagreeing samples.
3. **The remaining errors are rare, specific states.** Two are next_to placements where the expert
   places on a free spot it computes from privileged state, while the planner has to infer one from
   marker positions. The other two are recoveries after a failed physical place. Fifty updates spread
   over random training tasks barely visit such states.

A larger GRPO budget aimed at those states (next_to only, more samples per task) is the obvious next
experiment. On this benchmark, at this budget, the on-policy correction that worked was DAgger.

## Engineering notes that changed the numbers

* **Context.** The agent used to resend the whole chat each step, up to 4.5k tokens, which did not
  fit SFT in 6 GB. The newest state message already contains the action history and the last result,
  so the planner now gets the system prompt plus that one message (`Agent(context="last")`, about
  1.3k tokens). All rows of the table use it.
* **Bounded thought.** Under greedy decoding the SFT model sometimes looped one sentence inside
  `thought` until the 256 token limit, so the JSON never closed (7 of its first 21 failures). The
  schema now caps `thought` at 200 characters (the expert never needs more than 125). It applies to
  every planner in the table; SFT went from 79 to 83 with it.
* **A stale renderer blinds the next episode.** If an episode's `mujoco.Renderer` is still alive when
  the next one starts, the new segmentation comes back empty and the planner gets `MARKERS: {}`. The
  first DAgger collection was ruined this way (955 of 960 states blind, policy "success" fell to 29%).
  Renderers are now closed every episode and the agent raises on an empty first frame.
* **Server memory.** The JSON-schema enforcer cached every prefix it had seen and grew the server to
  17 GB of RAM. It is now built per request from cached tokenizer tables; RSS stays near 2.2 GB.
* **Answer-only logits.** Full-vocabulary logits over the whole prompt cost 0.9 GB on their own. The
  trainer asks for logits on the answer positions only (`logits_to_keep`).

## Reproduce

Two environments: the repo's `.venv` (simulation, agent, scoring) and a training venv with
torch + CUDA, transformers 4.57, peft, bitsandbytes, lm-format-enforcer and mujoco.

```bash
# expert ceiling and SFT data (sim venv)
python posttrain/run_expert.py --seeds 0 100 --out posttrain/results/eval_expert.jsonl
python posttrain/collect_sft.py --seeds 10000 10200 --out DATA/sft      # and 10200, 10400
# SFT (training venv)
python posttrain/train_sft.py --data DATA/sft --out CKPT/sft_r1
# serve a planner, then score it (sim venv)
python posttrain/serve_vlm.py --greedy --adapter CKPT/sft_r1/final
python posttrain/run_agent.py --seeds 0 100 --host http://localhost:11500 --out runs/eval_sft.jsonl
# DAgger: serve without --greedy, collect, continue training
python posttrain/collect_dagger.py --seeds 30000 30250 --out DATA/dagger
python posttrain/train_sft.py --data DATA/sft DATA/dagger --init CKPT/sft_r1/final --epochs 1 --lr 1e-4 --out CKPT/dagger_r1
# GRPO (training venv; runs the sim in process)
python posttrain/grpo_train.py --init CKPT/dagger_r1/final --out CKPT/grpo_r1 --groups 50 --lr 2e-5 --families next_to two_bin bin
```

## Limitations

* One benchmark, one robot, 100 eval tasks. A 2-task difference is noise at this size; the
  SFT and DAgger gains are not.
* Seed sets are disjoint, but the task distribution is the same for training and eval. This shows the
  planner learned this task family well, not that it transfers to new objects or commands.
* The expert is privileged (true state, a precomputed free spot for next_to), so imitation is
  bounded by it, and the planner still has to infer from markers what the expert just reads.
* Latency is about 4 to 5 s per call on the 6 GB GPU with constrained decoding through HF
  transformers; this study did not optimise serving.
