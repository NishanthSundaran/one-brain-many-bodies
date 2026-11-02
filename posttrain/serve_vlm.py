"""Serve Qwen2.5-VL-3B (4-bit, optionally with a LoRA adapter) behind the subset
of Ollama's /api/chat that brain/vlm_client.py uses, with the same structured
output: when the request carries a JSON schema as `format`, decoding is
constrained to it (lm-format-enforcer). Base, SFT and GRPO models are all
evaluated through this one server, so they differ only in their weights.

Usage (training venv): python posttrain/serve_vlm.py [--adapter DIR] [--port 11500]
Then: OllamaVLM(host="http://localhost:11500")
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_PIXELS = 560 * 400


def load(adapter: str | None):
    processor = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16,
                                               bnb_4bit_use_double_quant=True))
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return processor, model.eval()


def to_qwen(messages):
    """Ollama-style messages (content str, images [b64]) -> Qwen chat + PIL images."""
    out, images = [], []
    for m in messages:
        parts = []
        for b in m.get("images") or []:
            images.append(Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB"))
            parts.append({"type": "image"})
        parts.append({"type": "text", "text": m["content"]})
        out.append({"role": m["role"], "content": parts})
    return out, images


class Server:
    def __init__(self, processor, model, temperature: float, greedy: bool = False):
        self.p, self.m, self.temperature, self.greedy = processor, model, temperature, greedy
        self._fns = {}

    def chat(self, req: dict) -> dict:
        t0 = time.time()
        msgs, images = to_qwen(req["messages"])
        text = self.p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.p(text=[text], images=images or None, return_tensors="pt").to(self.m.device)
        kw = {}
        if req.get("format"):
            key = json.dumps(req["format"], sort_keys=True)
            if key not in self._fns:
                self._fns[key] = build_transformers_prefix_allowed_tokens_fn(
                    self.p.tokenizer, JsonSchemaParser(req["format"]))
            kw["prefix_allowed_tokens_fn"] = self._fns[key]
        temp = 0.0 if self.greedy else req.get("options", {}).get("temperature", self.temperature)
        with torch.no_grad():
            out = self.m.generate(**enc, max_new_tokens=256, do_sample=temp > 0,
                                  temperature=max(temp, 1e-5), top_p=1.0, **kw)
        gen = out[0, enc["input_ids"].shape[1]:]
        content = self.p.tokenizer.decode(gen, skip_special_tokens=True)
        return {"message": {"role": "assistant", "content": content},
                "total_duration": int((time.time() - t0) * 1e9), "eval_count": int(gen.shape[0])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--port", type=int, default=11500)
    ap.add_argument("--temperature", type=float, default=0.2)   # matches brain/vlm_client.py
    ap.add_argument("--greedy", action="store_true",
                    help="evaluation: ignore the client's temperature and decode greedily")
    a = ap.parse_args()
    torch.manual_seed(0)
    srv = Server(*load(a.adapter), temperature=a.temperature, greedy=a.greedy)

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/api/chat":
                self.send_error(404)
                return
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps(srv.chat(req)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    print(f"serving {MODEL} adapter={a.adapter} greedy={a.greedy} on :{a.port}", flush=True)
    HTTPServer(("127.0.0.1", a.port), H).serve_forever()   # one request at a time: one GPU model


if __name__ == "__main__":
    main()
