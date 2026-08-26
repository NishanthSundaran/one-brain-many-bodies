"""Local VLM client (Ollama native API, stdlib-only). The brain runs fully
on-device: no cloud APIs. Structured outputs (a JSON schema passed as
`format`) constrain the small model to emit exactly one well-formed tool call.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request


class OllamaVLM:
    def __init__(self, model: str = "qwen2.5vl:3b",
                 host: str = "http://localhost:11434",
                 keep_alive: str = "15m", num_ctx: int = 8192):
        self.model, self.host, self.keep_alive = model, host, keep_alive
        self.num_ctx = num_ctx

    def chat(self, messages: list[dict], schema: dict | None = None,
             timeout: float = 120.0) -> tuple[str, dict]:
        """messages: [{role, content, images?: [png_bytes]}]. Returns
        (content, stats). With `schema`, content is schema-valid JSON."""
        payload_msgs = []
        for m in messages:
            out = {"role": m["role"], "content": m["content"]}
            if m.get("images"):
                out["images"] = [base64.b64encode(img).decode()
                                 for img in m["images"]]
            payload_msgs.append(out)
        payload = {"model": self.model, "messages": payload_msgs,
                   "stream": False, "keep_alive": self.keep_alive,
                   "options": {"temperature": 0.2, "num_ctx": self.num_ctx}}
        if schema is not None:
            payload["format"] = schema
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                r = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Ollama {e.code}: {e.read().decode()[:500]}") from e
        stats = {"seconds": round(r.get("total_duration", 0) / 1e9, 2),
                 "eval_count": r.get("eval_count")}
        return r["message"]["content"], stats
