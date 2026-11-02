"""LLM providers behind one interface.
complete() returns a Completion; stream() yields str deltas and finally a Completion with usage."""
import asyncio
import json
import re
import zlib
from dataclasses import dataclass

import httpx

from . import config as cfg


class ProviderError(Exception):
    pass


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int


def est_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def _client(transport=None) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(cfg.PROVIDER_TIMEOUT), transport=transport)


# ---------------------------------------------------------------- mock (offline demo + tests)
FACTS = {"france": "Paris", "germany": "Berlin", "india": "New Delhi", "japan": "Tokyo",
         "italy": "Rome", "spain": "Madrid", "canada": "Ottawa", "brazil": "Brasilia"}
LATENCY = {"mock-fast": 0.03, "mock-smart": 0.09, "mock-flaky": 0.01}
_MATH = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/x])\s*(-?\d+(?:\.\d+)?)")


def _mock_answer(model: str, prompt: str) -> str:
    p = prompt.lower()
    # mock-fast is deterministically wrong on about a third of prompts so evals have something to find
    wrong = model == "mock-fast" and zlib.crc32(p.encode()) % 3 == 0
    m = _MATH.search(p)
    if m:
        a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
        r = {"+": a + b, "-": a - b, "*": a * b, "x": a * b, "/": a / b if b else 0}[op]
        r = r + 1 if wrong else r
        return f"The answer is {int(r) if r == int(r) else round(r, 2)}."
    if "capital" in p:
        for country, cap in FACTS.items():
            if country in p:
                return "I am not sure." if wrong else f"The capital of {country.title()} is {cap}."
    if "sentiment" in p:
        pos = any(w in p for w in ("love", "great", "amazing", "excellent", "happy"))
        neg = any(w in p for w in ("hate", "terrible", "awful", "worst", "sad"))
        label = "Positive" if pos and not neg else "Negative" if neg else "Neutral"
        if wrong:
            label = {"Positive": "Negative", "Negative": "Positive"}.get(label, "Positive")
        return f"Sentiment: {label}."
    return f"[{model}] You asked: {prompt[-120:]}"


class MockProvider:
    name = "mock"

    async def _text(self, model, messages) -> str:
        await asyncio.sleep(LATENCY.get(model, 0.05))
        if model == "mock-flaky":
            raise ProviderError("mock-flaky: simulated upstream outage (HTTP 503)")
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return _mock_answer(model, last_user)

    async def complete(self, model, messages, temperature, max_tokens) -> Completion:
        text = await self._text(model, messages)
        return Completion(text, est_tokens(" ".join(m["content"] for m in messages)), est_tokens(text))

    async def stream(self, model, messages, temperature, max_tokens):
        text = await self._text(model, messages)
        for piece in re.findall(r"\S+\s*", text):
            await asyncio.sleep(0.01)
            yield piece
        yield Completion(text, est_tokens(" ".join(m["content"] for m in messages)), est_tokens(text))


# ---------------------------------------------------------------- OpenAI-compatible (OpenAI, Groq, ...)
class OpenAICompatProvider:
    def __init__(self, name: str, base_url: str, api_key: str, transport=None):
        self.name, self.base, self.key, self.transport = name, base_url.rstrip("/"), api_key, transport

    def _req(self, model, messages, temperature, max_tokens, stream):
        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if stream:
            body.update(stream=True, stream_options={"include_usage": True})
        return f"{self.base}/chat/completions", {"Authorization": f"Bearer {self.key}"}, body

    async def complete(self, model, messages, temperature, max_tokens) -> Completion:
        url, headers, body = self._req(model, messages, temperature, max_tokens, False)
        try:
            async with _client(self.transport) as c:
                r = await c.post(url, headers=headers, json=body)
        except httpx.HTTPError as e:
            raise ProviderError(f"{self.name}: {type(e).__name__}: {e}") from e
        if r.status_code >= 400:
            raise ProviderError(f"{self.name}: HTTP {r.status_code} {r.text[:200]}")
        d = r.json()
        text = (d["choices"][0]["message"].get("content")) or ""
        u = d.get("usage") or {}
        return Completion(text, u.get("prompt_tokens") or est_tokens(json.dumps(messages)),
                          u.get("completion_tokens") or est_tokens(text))

    async def stream(self, model, messages, temperature, max_tokens):
        url, headers, body = self._req(model, messages, temperature, max_tokens, True)
        parts, usage = [], {}
        try:
            async with _client(self.transport) as c:
                async with c.stream("POST", url, headers=headers, json=body) as r:
                    if r.status_code >= 400:
                        raise ProviderError(f"{self.name}: HTTP {r.status_code} {(await r.aread())[:200]!r}")
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        ev = json.loads(data)
                        usage = ev.get("usage") or usage
                        for ch in ev.get("choices") or []:
                            delta = (ch.get("delta") or {}).get("content")
                            if delta:
                                parts.append(delta)
                                yield delta
        except httpx.HTTPError as e:
            raise ProviderError(f"{self.name}: {type(e).__name__}: {e}") from e
        text = "".join(parts)
        yield Completion(text, usage.get("prompt_tokens") or est_tokens(json.dumps(messages)),
                         usage.get("completion_tokens") or est_tokens(text))


# ---------------------------------------------------------------- Anthropic (Claude)
class AnthropicProvider:
    name = "anthropic"
    url = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, transport=None):
        self.key, self.transport = api_key, transport

    def _req(self, model, messages, temperature, max_tokens, stream):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        body = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
                "messages": [m for m in messages if m["role"] != "system"]}
        if system:
            body["system"] = system
        if stream:
            body["stream"] = True
        return {"x-api-key": self.key, "anthropic-version": "2023-06-01"}, body

    async def complete(self, model, messages, temperature, max_tokens) -> Completion:
        headers, body = self._req(model, messages, temperature, max_tokens, False)
        try:
            async with _client(self.transport) as c:
                r = await c.post(self.url, headers=headers, json=body)
        except httpx.HTTPError as e:
            raise ProviderError(f"anthropic: {type(e).__name__}: {e}") from e
        if r.status_code >= 400:
            raise ProviderError(f"anthropic: HTTP {r.status_code} {r.text[:200]}")
        d = r.json()
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        u = d.get("usage") or {}
        return Completion(text, u.get("input_tokens") or est_tokens(json.dumps(messages)),
                          u.get("output_tokens") or est_tokens(text))

    async def stream(self, model, messages, temperature, max_tokens):
        headers, body = self._req(model, messages, temperature, max_tokens, True)
        parts, tin, tout = [], 0, 0
        try:
            async with _client(self.transport) as c:
                async with c.stream("POST", self.url, headers=headers, json=body) as r:
                    if r.status_code >= 400:
                        raise ProviderError(f"anthropic: HTTP {r.status_code} {(await r.aread())[:200]!r}")
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        ev = json.loads(line[5:].strip())
                        t = ev.get("type")
                        if t == "message_start":
                            tin = (ev.get("message", {}).get("usage") or {}).get("input_tokens", 0)
                        elif t == "content_block_delta" and ev["delta"].get("type") == "text_delta":
                            parts.append(ev["delta"]["text"])
                            yield ev["delta"]["text"]
                        elif t == "message_delta":
                            tout = (ev.get("usage") or {}).get("output_tokens", tout)
                        elif t == "error":
                            raise ProviderError(f"anthropic: {ev.get('error')}")
        except httpx.HTTPError as e:
            raise ProviderError(f"anthropic: {type(e).__name__}: {e}") from e
        text = "".join(parts)
        yield Completion(text, tin or est_tokens(json.dumps(messages)), tout or est_tokens(text))


def build_providers() -> dict:
    p: dict = {"mock": MockProvider()}
    if cfg.ANTHROPIC_API_KEY:
        p["anthropic"] = AnthropicProvider(cfg.ANTHROPIC_API_KEY)
    if cfg.OPENAI_API_KEY:
        p["openai"] = OpenAICompatProvider("openai", "https://api.openai.com/v1", cfg.OPENAI_API_KEY)
    if cfg.GROQ_API_KEY:
        p["groq"] = OpenAICompatProvider("groq", "https://api.groq.com/openai/v1", cfg.GROQ_API_KEY)
    return p


PROVIDERS = build_providers()
