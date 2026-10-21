"""Generation backends for open-weight models. All decode greedily with a fixed seed.

hf     : transformers, CPU or one GPU. Slow but runs anywhere.
vllm   : vLLM offline engine for GPU runs (the plan's main harness path).
ollama : local Ollama server (GGUF, quantized). CPU-friendly baseline only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class HFBackend:
    def __init__(self, model: str, revision: str | None, seed: int, max_new_tokens: int, **_):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.seed, self.max_new_tokens = seed, max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model, revision=revision)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=dtype,
                                                          device_map="auto" if torch.cuda.is_available() else None)
        self.model.eval()

    def generate(self, messages: list[dict]) -> Generation:
        torch = self.torch
        torch.manual_seed(self.seed)
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True)
        ids = {k: v.to(self.model.device) for k, v in ids.items()}
        t0 = time.monotonic()
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        new = out[0, ids["input_ids"].shape[1]:]
        return Generation(self.tok.decode(new, skip_special_tokens=True), int(ids["input_ids"].shape[1]),
                          int(new.shape[0]), time.monotonic() - t0)


class VLLMBackend:
    def __init__(self, model: str, revision: str | None, seed: int, max_new_tokens: int, **_):
        from vllm import LLM, SamplingParams

        self.llm = LLM(model=model, revision=revision, seed=seed)
        self.params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, seed=seed)

    def generate(self, messages: list[dict]) -> Generation:
        t0 = time.monotonic()
        out = self.llm.chat(messages, self.params, use_tqdm=False)[0]
        return Generation(out.outputs[0].text, len(out.prompt_token_ids), len(out.outputs[0].token_ids),
                          time.monotonic() - t0)


class OllamaBackend:
    def __init__(self, model: str, seed: int, max_new_tokens: int, url: str = "http://localhost:11434", **_):
        import requests

        self.requests, self.model, self.url = requests, model, url
        self.options = {"temperature": 0, "seed": seed, "num_predict": max_new_tokens, "num_ctx": 8192}

    def generate(self, messages: list[dict]) -> Generation:
        t0 = time.monotonic()
        r = self.requests.post(f"{self.url}/api/chat", timeout=1800, json={
            "model": self.model, "messages": messages, "stream": False, "options": self.options})
        r.raise_for_status()
        d = r.json()
        return Generation(d["message"]["content"], d.get("prompt_eval_count", 0), d.get("eval_count", 0),
                          time.monotonic() - t0)


BACKENDS = {"hf": HFBackend, "vllm": VLLMBackend, "ollama": OllamaBackend}
