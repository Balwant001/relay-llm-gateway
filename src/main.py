import asyncio
import json
import re
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import func, select

from . import config as cfg
from . import metrics
from .db import (ApiKey, EvalRun, Prompt, SessionLocal, UsageLog, get_prompt, init_db, run, write_log,
                 write_logs)
from .evals import pick_winner, score, summarize, percentile
from .gateway import AllProvidersFailed, Gateway, available_models, resolve_chain
from .pricing import MODELS
from .providers import PROVIDERS
from .security import KeyInfo, clear_key_cache, hash_key, new_key, require_admin, require_key
from .state import state
from .store import make_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    state.store = make_store()
    state.gateway = Gateway(state.store)
    yield


app = FastAPI(title="Relay: LLM Gateway", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=cfg.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "Retry-After"])


# ------------------------------------------------------------------ schemas
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)


class PromptRef(BaseModel):
    name: str
    version: int | None = None
    variables: dict[str, str] = {}


class ChatBody(BaseModel):
    model: str = "auto"
    messages: list[Message] | None = None
    prompt: PromptRef | None = None
    fallbacks: list[str] | None = None
    temperature: float = Field(0.0, ge=0, le=2)
    max_tokens: int = Field(512, ge=1, le=4096)
    stream: bool = False
    use_cache: bool = True

    @model_validator(mode="after")
    def need_input(self):
        if not self.messages and not self.prompt:
            raise ValueError("Provide 'messages' or 'prompt'")
        return self


class KeyBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    rate_limit_per_min: int = Field(cfg.DEFAULT_RATE_LIMIT, ge=1, le=100000)
    monthly_budget_usd: float | None = Field(None, ge=0)


class PromptBody(BaseModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[\w.\-]+$")
    template: str = Field(min_length=1, max_length=20000)


class Variant(BaseModel):
    label: str = Field(min_length=1, max_length=40)
    model: str
    template: str | None = None
    prompt: PromptRef | None = None


class Case(BaseModel):
    input: str = Field(min_length=1, max_length=5000)
    expected: str | list[str]


class EvalBody(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    variants: list[Variant] = Field(min_length=2, max_length=4)
    cases: list[Case] = Field(min_length=1, max_length=50)
    scoring: Literal["contains", "exact", "f1"] = "contains"
    max_tokens: int = Field(256, ge=1, le=2048)


# ------------------------------------------------------------------ helpers
_VAR = re.compile(r"{{\s*(\w+)\s*}}")


def render(template: str, variables: dict) -> str:
    missing = set(_VAR.findall(template)) - set(variables)
    if missing:
        raise HTTPException(422, f"Missing prompt variables: {sorted(missing)}")
    return _VAR.sub(lambda m: str(variables[m.group(1)]), template)


async def build_messages(body: ChatBody):
    msgs = [m.model_dump() for m in (body.messages or [])]
    name = ver = None
    if body.prompt:
        p = await run(get_prompt, body.prompt.name, body.prompt.version)
        if not p:
            raise HTTPException(404, "Prompt not found")
        msgs.append({"role": "user", "content": render(p["template"], body.prompt.variables)})
        name, ver = p["name"], p["version"]
    return msgs, name, ver


def log_kwargs(key_id, source, requested, res, status="ok", pname=None, pver=None, error=None) -> dict:
    base = dict(key_id=key_id, source=source, requested_model=requested, status=status,
                prompt_name=pname, prompt_version=pver, error=error)
    if not res:
        return {**base, "cache": "none", "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                "saved_usd": 0.0, "latency_ms": 0.0, "fallback_used": False}
    return {**base, "model": res["model"], "provider": res["provider"], "cache": res["cached"],
            "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],
            "cost_usd": res["cost_usd"], "saved_usd": res["saved_usd"], "latency_ms": res["latency_ms"],
            "fallback_used": res["fallback_used"]}


# ------------------------------------------------------------------ public API
@app.get("/health")
async def health():
    return {"status": "ok", "store": state.store.kind, "providers": sorted(PROVIDERS)}


@app.get("/v1/models")
async def models():
    return [{"model": m, "provider": MODELS[m]["provider"], "input_per_1m_usd": MODELS[m]["in"],
             "output_per_1m_usd": MODELS[m]["out"]} for m in available_models()]


@app.post("/v1/chat")
async def chat(body: ChatBody, request: Request, background: BackgroundTasks, key: KeyInfo = Depends(require_key)):
    msgs, pname, pver = await build_messages(body)
    rl = getattr(request.state, "rl", {})
    args = dict(model=body.model, messages=msgs, temperature=body.temperature, max_tokens=body.max_tokens,
                use_cache=body.use_cache, fallbacks=body.fallbacks)
    try:
        resolve_chain(body.model, body.fallbacks)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if body.stream:
        async def sse():
            final, err = None, None
            async for ev in state.gateway.stream(**args):
                if ev["type"] == "done":
                    final = ev
                elif ev["type"] == "error":
                    err = ev
                yield f"data: {json.dumps(ev)}\n\n"
            if final:
                metrics.observe(final)
                await run(write_log, **log_kwargs(key.id, "api", body.model, final, "ok", pname, pver))
            else:
                metrics.observe(None, "error", body.model)
                await run(write_log, **log_kwargs(key.id, "api", body.model, None, "error", pname, pver,
                                                  (err or {}).get("error")))
        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={**rl, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    try:
        res = await state.gateway.complete(**args)
    except AllProvidersFailed as e:
        metrics.observe(None, "error", body.model)
        await run(write_log, **log_kwargs(key.id, "api", body.model, None, "error", pname, pver, "all providers failed"))
        raise HTTPException(502, {"error": "all providers failed", "attempts": e.attempts})
    metrics.observe(res)
    background.add_task(write_log, **log_kwargs(key.id, "api", body.model, res, "ok", pname, pver))
    return JSONResponse(res, headers=rl)


@app.get("/v1/usage")
async def my_usage(key: KeyInfo = Depends(require_key)):
    def q():
        with SessionLocal() as s:
            rows = s.execute(select(UsageLog.model, func.count(), func.sum(UsageLog.cost_usd),
                                    func.sum(UsageLog.saved_usd), func.sum(UsageLog.input_tokens),
                                    func.sum(UsageLog.output_tokens)).where(
                UsageLog.key_id == key.id, UsageLog.source == "api").group_by(UsageLog.model)).all()
        return rows
    rows = await run(q)
    by_model = [{"model": r[0] or "error", "requests": r[1], "cost_usd": round(r[2] or 0, 6),
                 "saved_usd": round(r[3] or 0, 6), "input_tokens": r[4] or 0, "output_tokens": r[5] or 0} for r in rows]
    return {"key": key.name, "requests": sum(r["requests"] for r in by_model),
            "cost_usd": round(sum(r["cost_usd"] for r in by_model), 6),
            "saved_usd": round(sum(r["saved_usd"] for r in by_model), 6), "by_model": by_model}


# ------------------------------------------------------------------ admin: keys
@app.post("/admin/keys", dependencies=[Depends(require_admin)])
async def create_key(body: KeyBody):
    raw = new_key()

    def q():
        with SessionLocal() as s:
            k = ApiKey(name=body.name, prefix=raw[:10], key_hash=hash_key(raw),
                       rate_limit_per_min=body.rate_limit_per_min, monthly_budget_usd=body.monthly_budget_usd)
            s.add(k)
            s.commit()
            return k.id
    kid = await run(q)
    return {"id": kid, "name": body.name, "key": raw, "note": "Copy this key now. It is shown only once.",
            "rate_limit_per_min": body.rate_limit_per_min, "monthly_budget_usd": body.monthly_budget_usd}


@app.get("/admin/keys", dependencies=[Depends(require_admin)])
async def list_keys():
    def q():
        with SessionLocal() as s:
            keys = s.scalars(select(ApiKey).order_by(ApiKey.id.desc())).all()
            agg = {r[0]: (r[1], r[2]) for r in s.execute(select(
                UsageLog.key_id, func.count(), func.sum(UsageLog.cost_usd)).group_by(UsageLog.key_id)).all()}
        return [{"id": k.id, "name": k.name, "prefix": k.prefix, "rate_limit_per_min": k.rate_limit_per_min,
                 "monthly_budget_usd": k.monthly_budget_usd, "is_active": k.is_active,
                 "requests": agg.get(k.id, (0, 0))[0], "cost_usd": round(agg.get(k.id, (0, 0))[1] or 0, 6),
                 "created_at": k.created_at.isoformat()} for k in keys]
    return await run(q)


@app.post("/admin/keys/{key_id}/revoke", dependencies=[Depends(require_admin)])
async def revoke_key(key_id: int):
    def q():
        with SessionLocal() as s:
            k = s.get(ApiKey, key_id)
            if not k:
                return False
            k.is_active = False
            s.commit()
            return True
    if not await run(q):
        raise HTTPException(404, "Key not found")
    clear_key_cache()
    return {"id": key_id, "revoked": True}


# ------------------------------------------------------------------ admin: stats and logs
@app.get("/admin/stats", dependencies=[Depends(require_admin)])
async def stats(sample: int = 5000):
    def q():
        with SessionLocal() as s:
            return s.scalars(select(UsageLog).where(UsageLog.source == "api").order_by(UsageLog.id.desc())
                             .limit(min(sample, 20000))).all()
    rows = await run(q)
    ok = [r for r in rows if r.status == "ok"]
    hits = [r for r in ok if r.cache != "none"]
    live = [r for r in ok if r.cache == "none"]
    cost, saved = sum(r.cost_usd for r in ok), sum(r.saved_usd for r in ok)
    by_provider: dict[str, dict] = {}
    for r in ok:
        d = by_provider.setdefault(r.provider or "?", {"requests": 0, "cost_usd": 0.0})
        d["requests"] += 1
        d["cost_usd"] = round(d["cost_usd"] + r.cost_usd, 8)
    lat = [r.latency_ms for r in ok]
    return {"sample_size": len(rows), "requests": len(rows), "ok": len(ok), "errors": len(rows) - len(ok),
            "cache_hits": len(hits), "exact_hits": sum(1 for r in hits if r.cache == "exact"),
            "semantic_hits": sum(1 for r in hits if r.cache == "semantic"),
            "cache_hit_rate": round(len(hits) / len(ok), 4) if ok else 0.0,
            "cost_usd": round(cost, 6), "saved_usd": round(saved, 6),
            "savings_pct": round(saved / (cost + saved), 4) if (cost + saved) else 0.0,
            "p50_latency_ms": percentile(lat, 50), "p95_latency_ms": percentile(lat, 95),
            "avg_cached_latency_ms": round(sum(r.latency_ms for r in hits) / len(hits), 1) if hits else 0.0,
            "avg_live_latency_ms": round(sum(r.latency_ms for r in live) / len(live), 1) if live else 0.0,
            "fallbacks": sum(1 for r in ok if r.fallback_used), "by_provider": by_provider}


@app.get("/admin/logs", dependencies=[Depends(require_admin)])
async def logs(limit: int = 50):
    def q():
        with SessionLocal() as s:
            names = {k.id: k.name for k in s.scalars(select(ApiKey)).all()}
            rows = s.scalars(select(UsageLog).order_by(UsageLog.id.desc()).limit(min(limit, 500))).all()
        return [{"id": r.id, "time": r.created_at.isoformat(), "key": names.get(r.key_id, "-"), "source": r.source,
                 "requested_model": r.requested_model, "model": r.model, "provider": r.provider, "cache": r.cache,
                 "tokens": r.input_tokens + r.output_tokens, "cost_usd": r.cost_usd, "saved_usd": r.saved_usd,
                 "latency_ms": r.latency_ms, "status": r.status, "fallback_used": r.fallback_used,
                 "prompt": f"{r.prompt_name}@v{r.prompt_version}" if r.prompt_name else None, "error": r.error}
                for r in rows]
    return await run(q)


# ------------------------------------------------------------------ admin: prompts (versioned)
@app.post("/admin/prompts", dependencies=[Depends(require_admin)])
async def create_prompt(body: PromptBody):
    def q():
        with SessionLocal() as s:
            last = s.scalar(select(func.max(Prompt.version)).where(Prompt.name == body.name)) or 0
            p = Prompt(name=body.name, version=last + 1, template=body.template)
            s.add(p)
            s.commit()
            return p.version
    return {"name": body.name, "version": await run(q)}


@app.get("/admin/prompts", dependencies=[Depends(require_admin)])
async def list_prompts():
    def q():
        with SessionLocal() as s:
            rows = s.scalars(select(Prompt).order_by(Prompt.name, Prompt.version.desc()).limit(200)).all()
        return [{"name": p.name, "version": p.version, "template": p.template, "created_at": p.created_at.isoformat()}
                for p in rows]
    return await run(q)


# ------------------------------------------------------------------ admin: evals
@app.post("/admin/evals", dependencies=[Depends(require_admin)])
async def create_eval(body: EvalBody):
    templates = []
    for v in body.variants:
        try:
            resolve_chain(v.model, [])
        except ValueError as e:
            raise HTTPException(400, str(e))
        if v.prompt:
            p = await run(get_prompt, v.prompt.name, v.prompt.version)
            if not p:
                raise HTTPException(404, f"Prompt '{v.prompt.name}' not found")
            t = p["template"]
        else:
            t = v.template or "{{input}}"
        templates.append(t if re.search(r"{{\s*input\s*}}", t) else t + "\n\n{{input}}")

    sem = asyncio.Semaphore(5)

    async def one(vi: int, ci: int) -> dict:
        v, case = body.variants[vi], body.cases[ci]
        content = render(templates[vi], {"input": case.input})
        async with sem:
            try:
                res = await state.gateway.complete(model=v.model, messages=[{"role": "user", "content": content}],
                                                   temperature=0.0, max_tokens=body.max_tokens, use_cache=False,
                                                   fallbacks=[])
                return {"variant": vi, "case": ci, "output": res["content"], "res": res, "error": None,
                        "score": score(body.scoring, res["content"], case.expected), "latency_ms": res["latency_ms"],
                        "cost_usd": res["cost_usd"]}
            except AllProvidersFailed as e:
                return {"variant": vi, "case": ci, "output": "", "res": None, "score": 0.0, "latency_ms": 0.0,
                        "cost_usd": 0.0, "error": e.attempts[-1]["error"] if e.attempts else "failed"}

    results = await asyncio.gather(*[one(vi, ci) for vi in range(len(body.variants)) for ci in range(len(body.cases))])
    logs_ = [log_kwargs(None, "eval", body.variants[r["variant"]].model, r["res"], "ok" if r["res"] else "error",
                        error=r["error"]) for r in results]
    await run(write_logs, logs_)
    for r in results:
        metrics.observe(r["res"], "ok" if r["res"] else "error", body.variants[r["variant"]].model)

    summaries = [summarize(v.label, v.model, [r for r in results if r["variant"] == i]) for i, v in enumerate(body.variants)]
    winner = pick_winner(summaries)
    cases = [{"input": c.input, "expected": c.expected,
              "results": [{"variant": body.variants[r["variant"]].label, "output": r["output"], "score": r["score"],
                           "error": r["error"]} for r in results if r["case"] == ci]} for ci, c in enumerate(body.cases)]
    out = {"name": body.name, "scoring": body.scoring, "variants": summaries, "winner": winner, "cases": cases}

    def save():
        with SessionLocal() as s:
            e = EvalRun(name=body.name, spec=body.model_dump_json(), results=json.dumps(out))
            s.add(e)
            s.commit()
            return e.id
    out["id"] = await run(save)
    return out


@app.get("/admin/evals", dependencies=[Depends(require_admin)])
async def list_evals():
    def q():
        with SessionLocal() as s:
            rows = s.scalars(select(EvalRun).order_by(EvalRun.id.desc()).limit(30)).all()
        res = []
        for e in rows:
            d = json.loads(e.results)
            res.append({"id": e.id, "name": e.name, "created_at": e.created_at.isoformat(), "winner": d.get("winner"),
                        "variants": d.get("variants", [])})
        return res
    return await run(q)


@app.get("/metrics")
async def prometheus():
    body, ctype = metrics.render()
    return Response(body, media_type=ctype)
