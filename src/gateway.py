"""Core gateway logic: caching (exact + similarity), provider chain with failover, circuit breaker, cost."""
import hashlib
import json
import re
import time

from . import config as cfg
from .pricing import MODELS, cost_usd
from .providers import PROVIDERS, Completion, ProviderError
from .semantic import SemanticIndex


class AllProvidersFailed(Exception):
    def __init__(self, attempts: list):
        super().__init__("all providers failed")
        self.attempts = attempts


def available_models() -> list[str]:
    return [m for m, i in MODELS.items() if i["provider"] in PROVIDERS]


def auto_chain() -> list[str]:
    chain = [m for m in cfg.AUTO_CHAIN if m in MODELS and MODELS[m]["provider"] in PROVIDERS]
    return chain or ["mock-smart"]


def resolve_chain(model: str, fallbacks: list[str] | None) -> list[str]:
    if model == "auto":
        return auto_chain()
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'")
    avail = set(available_models())
    if model not in avail:
        raise ValueError(f"Model '{model}' is unavailable: its provider has no API key configured")
    chain = [model]
    for m in (fallbacks if fallbacks is not None else auto_chain()):
        if m not in MODELS:
            raise ValueError(f"Unknown fallback model '{m}'")
        if m in avail and m not in chain:
            chain.append(m)
    return chain


class Breaker:
    """Skip a model for `cooldown` seconds after `threshold` consecutive failures."""

    def __init__(self, threshold: int = 3, cooldown: float = 30.0):
        self.threshold, self.cooldown = threshold, cooldown
        self.fails: dict[str, int] = {}
        self.until: dict[str, float] = {}

    def is_open(self, m: str) -> bool:
        return self.until.get(m, 0) > time.monotonic()

    def ok(self, m: str) -> None:
        self.fails[m] = 0

    def fail(self, m: str) -> None:
        self.fails[m] = self.fails.get(m, 0) + 1
        if self.fails[m] >= self.threshold:
            self.until[m] = time.monotonic() + self.cooldown
            self.fails[m] = 0


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def cache_key(model, messages, temperature, max_tokens) -> str:
    blob = json.dumps([model, temperature, max_tokens, [(m["role"], _ws(m["content"])) for m in messages]])
    return hashlib.sha256(blob.encode()).hexdigest()


class Gateway:
    def __init__(self, store):
        self.store = store
        self.semantic = SemanticIndex(cfg.SEMANTIC_THRESHOLD)
        self.breaker = Breaker()

    # ---- cache
    def _scope(self, model, messages, temperature):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return hashlib.sha1(f"{model}|{temperature}|{system}".encode()).hexdigest(), text

    async def _lookup(self, model, messages, temperature, max_tokens):
        raw = await self.store.get("c:" + cache_key(model, messages, temperature, max_tokens))
        if raw:
            return json.loads(raw), "exact"
        scope, text = self._scope(model, messages, temperature)
        sk = self.semantic.find(scope, text)
        if sk and (raw := await self.store.get("c:" + sk)):
            return json.loads(raw), "semantic"
        return None, "none"

    async def _save(self, model, messages, temperature, max_tokens, res) -> None:
        key = cache_key(model, messages, temperature, max_tokens)
        payload = {k: res[k] for k in ("content", "provider", "model", "input_tokens", "output_tokens", "cost_usd")}
        await self.store.set("c:" + key, json.dumps(payload), cfg.CACHE_TTL)
        scope, text = self._scope(model, messages, temperature)
        self.semantic.add(scope, text, key)

    # ---- results
    def _hit(self, payload, kind, requested, t0) -> dict:
        return {"content": payload["content"], "provider": payload["provider"], "model": payload["model"],
                "requested_model": requested, "cached": kind, "input_tokens": payload["input_tokens"],
                "output_tokens": payload["output_tokens"], "cost_usd": 0.0, "saved_usd": payload["cost_usd"],
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "fallback_used": False, "attempts": []}

    def _result(self, m, comp: Completion, requested, chain, t0, attempts) -> dict:
        return {"content": comp.text, "provider": MODELS[m]["provider"], "model": m, "requested_model": requested,
                "cached": "none", "input_tokens": comp.input_tokens, "output_tokens": comp.output_tokens,
                "cost_usd": round(cost_usd(m, comp.input_tokens, comp.output_tokens), 8), "saved_usd": 0.0,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "fallback_used": m != chain[0],
                "attempts": attempts}

    # ---- non-streaming
    async def complete(self, *, model, messages, temperature=0.0, max_tokens=512, use_cache=True, fallbacks=None) -> dict:
        chain = resolve_chain(model, fallbacks)
        t0 = time.perf_counter()
        cacheable = use_cache and temperature <= 0.3
        if cacheable:
            hit, kind = await self._lookup(model, messages, temperature, max_tokens)
            if hit:
                return self._hit(hit, kind, model, t0)
        attempts: list[dict] = []
        for m in chain:
            if self.breaker.is_open(m):
                attempts.append({"model": m, "ok": False, "error": "circuit open after repeated failures"})
                continue
            try:
                comp = await PROVIDERS[MODELS[m]["provider"]].complete(m, messages, temperature, max_tokens)
            except ProviderError as e:
                self.breaker.fail(m)
                attempts.append({"model": m, "ok": False, "error": str(e)[:200]})
                continue
            self.breaker.ok(m)
            attempts.append({"model": m, "ok": True})
            res = self._result(m, comp, model, chain, t0, attempts)
            if cacheable:
                await self._save(model, messages, temperature, max_tokens, res)
            return res
        raise AllProvidersFailed(attempts)

    # ---- streaming: yields {"type": "delta"|"done"|"error", ...}
    async def stream(self, *, model, messages, temperature=0.0, max_tokens=512, use_cache=True, fallbacks=None):
        chain = resolve_chain(model, fallbacks)
        t0 = time.perf_counter()
        cacheable = use_cache and temperature <= 0.3
        if cacheable:
            hit, kind = await self._lookup(model, messages, temperature, max_tokens)
            if hit:
                for piece in re.findall(r"\S+\s*", hit["content"]):
                    yield {"type": "delta", "text": piece}
                yield {"type": "done", **self._hit(hit, kind, model, t0)}
                return
        attempts: list[dict] = []
        for m in chain:
            if self.breaker.is_open(m):
                attempts.append({"model": m, "ok": False, "error": "circuit open after repeated failures"})
                continue
            started, comp = False, None
            try:
                async for item in PROVIDERS[MODELS[m]["provider"]].stream(m, messages, temperature, max_tokens):
                    if isinstance(item, str):
                        started = True
                        yield {"type": "delta", "text": item}
                    else:
                        comp = item
            except ProviderError as e:
                self.breaker.fail(m)
                attempts.append({"model": m, "ok": False, "error": str(e)[:200]})
                if started:  # cannot switch providers mid-answer
                    yield {"type": "error", "error": "stream interrupted", "attempts": attempts}
                    return
                continue
            if comp is None:
                attempts.append({"model": m, "ok": False, "error": "stream ended without a completion"})
                continue
            self.breaker.ok(m)
            attempts.append({"model": m, "ok": True})
            res = self._result(m, comp, model, chain, t0, attempts)
            if cacheable:
                await self._save(model, messages, temperature, max_tokens, res)
            yield {"type": "done", **res}
            return
        yield {"type": "error", "error": "all providers failed", "attempts": attempts}
