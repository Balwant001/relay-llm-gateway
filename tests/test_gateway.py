import json
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["ADMIN_KEY"] = "test-admin"

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.providers import OpenAICompatProvider, AnthropicProvider

ADMIN = {"X-Admin-Key": "test-admin"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def make_key(client, **kw):
    r = client.post("/admin/keys", json={"name": "t", **kw}, headers=ADMIN)
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['key']}"}


def chat(client, headers, content, **kw):
    return client.post("/v1/chat", json={"model": "mock-smart", "messages": [{"role": "user", "content": content}], **kw},
                       headers=headers)


def test_health_and_models(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "mock-smart" in [m["model"] for m in client.get("/v1/models").json()]


def test_auth_required(client):
    assert chat(client, {}, "hi").status_code == 401
    assert chat(client, {"Authorization": "Bearer nope"}, "hi").status_code == 401
    assert client.get("/admin/stats").status_code == 401


def test_chat_and_exact_cache(client):
    h = make_key(client)
    a = chat(client, h, "What is the capital of France?").json()
    assert "Paris" in a["content"] and a["cached"] == "none" and a["cost_usd"] > 0
    b = chat(client, h, "What is the capital of France?").json()
    assert b["cached"] == "exact" and b["cost_usd"] == 0 and b["saved_usd"] == a["cost_usd"]


def test_semantic_cache_hits_typos_but_not_different_numbers(client):
    h = make_key(client)
    chat(client, h, "What is the capital of Japan?")
    assert chat(client, h, "What is the capital of Japn?").json()["cached"] == "semantic"
    chat(client, h, "What is 2 + 2?")
    assert chat(client, h, "What is 2 + 3?").json()["cached"] == "none"


def test_use_cache_false_skips_cache(client):
    h = make_key(client)
    chat(client, h, "What is the capital of Italy?")
    assert chat(client, h, "What is the capital of Italy?", use_cache=False).json()["cached"] == "none"


def test_fallback_when_primary_fails(client):
    h = make_key(client)
    r = chat(client, h, "What is 5 + 5?", model="mock-flaky").json()
    assert r["model"] == "mock-smart" and r["fallback_used"] is True
    assert r["attempts"][0]["ok"] is False and r["attempts"][-1]["ok"] is True


def test_all_providers_failed_returns_502(client):
    h = make_key(client)
    r = chat(client, h, "hello", model="mock-flaky", fallbacks=[])
    assert r.status_code == 502 and r.json()["detail"]["attempts"]


def test_unknown_model_400(client):
    assert chat(client, make_key(client), "hi", model="gpt-9000").status_code == 400


def test_rate_limit(client):
    h = make_key(client, rate_limit_per_min=3)
    codes = [chat(client, h, f"What is {i} + 1?").status_code for i in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3] == 429
    assert "retry-after" in chat(client, h, "x").headers


def test_budget_blocks_with_402(client):
    h = make_key(client, monthly_budget_usd=0)
    assert chat(client, h, "hello").status_code == 402


def test_revoked_key_rejected(client):
    r = client.post("/admin/keys", json={"name": "temp"}, headers=ADMIN).json()
    h = {"Authorization": f"Bearer {r['key']}"}
    assert chat(client, h, "hi").status_code == 200
    client.post(f"/admin/keys/{r['id']}/revoke", headers=ADMIN)
    assert chat(client, h, "hi again").status_code == 401


def test_streaming_sse(client):
    h = make_key(client)
    with client.stream("POST", "/v1/chat", headers=h, json={
            "model": "mock-fast", "stream": True, "messages": [{"role": "user", "content": "What is 6 * 7?"}]}) as r:
        events = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ")]
    assert events[-1]["type"] == "done" and events[-1]["output_tokens"] > 0
    assert "".join(e["text"] for e in events if e["type"] == "delta") == events[-1]["content"]


def test_streaming_fallback_before_first_token(client):
    h = make_key(client)
    with client.stream("POST", "/v1/chat", headers=h, json={
            "model": "mock-flaky", "stream": True, "messages": [{"role": "user", "content": "What is 1 + 1?"}]}) as r:
        events = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ")]
    assert events[-1]["type"] == "done" and events[-1]["fallback_used"] is True


def test_prompt_versioning_and_rendering(client):
    assert client.post("/admin/prompts", json={"name": "qa", "template": "Answer: {{q}}"}, headers=ADMIN).json()["version"] == 1
    assert client.post("/admin/prompts", json={"name": "qa", "template": "Be brief. {{q}}"}, headers=ADMIN).json()["version"] == 2
    h = make_key(client)
    r = client.post("/v1/chat", headers=h, json={"model": "mock-smart", "prompt": {
        "name": "qa", "version": 1, "variables": {"q": "What is 3 + 4?"}}}).json()
    assert "7" in r["content"]
    bad = client.post("/v1/chat", headers=h, json={"model": "mock-smart", "prompt": {"name": "qa"}})
    assert bad.status_code == 422


def test_usage_and_admin_stats(client):
    h = make_key(client)
    chat(client, h, "What is the capital of Spain?")
    chat(client, h, "What is the capital of Spain?")
    u = client.get("/v1/usage", headers=h).json()
    assert u["requests"] == 2 and u["saved_usd"] > 0
    s = client.get("/admin/stats", headers=ADMIN).json()
    assert s["requests"] >= 2 and s["cache_hits"] >= 1 and 0 < s["savings_pct"] <= 1
    assert client.get("/admin/logs", headers=ADMIN).json()[0]["status"] in ("ok", "error")


CASES = [{"input": f"What is {a} + {b}?", "expected": str(a + b)} for a, b in
         [(12, 7), (45, 18), (3, 9), (20, 22), (8, 8), (31, 11), (50, 25), (6, 14)]]


def test_eval_harness_compares_models(client):
    r = client.post("/admin/evals", headers=ADMIN, json={"name": "fast-vs-smart", "variants": [
        {"label": "fast", "model": "mock-fast", "template": "Answer concisely: {{input}}"},
        {"label": "smart", "model": "mock-smart", "template": "Answer concisely: {{input}}"}], "cases": CASES})
    assert r.status_code == 200
    d = r.json()
    smart = next(v for v in d["variants"] if v["label"] == "smart")
    fast = next(v for v in d["variants"] if v["label"] == "fast")
    assert smart["mean_score"] == 1.0 and fast["mean_score"] < 1.0
    assert d["winner"] == "smart" and len(d["cases"]) == len(CASES)
    assert client.get("/admin/evals", headers=ADMIN).json()[0]["name"] == "fast-vs-smart"


def test_metrics_endpoint(client):
    assert "gateway_requests_total" in client.get("/metrics").text


# ---- real provider adapters, exercised against fake HTTP transports
def test_openai_compatible_adapter_parses_response_and_stream():
    def handler(request: httpx.Request):
        body = json.loads(request.content)
        if body.get("stream"):
            sse = ('data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
                   'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
                   'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
                   'data: [DONE]\n\n')
            return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Hello"}}],
                                         "usage": {"prompt_tokens": 5, "completion_tokens": 2}})

    p = OpenAICompatProvider("openai", "https://x.test/v1", "k", transport=httpx.MockTransport(handler))
    msgs = [{"role": "user", "content": "hi"}]
    c = asyncio.run(p.complete("gpt-4o-mini", msgs, 0, 10))
    assert (c.text, c.input_tokens, c.output_tokens) == ("Hello", 5, 2)

    async def collect():
        return [x async for x in p.stream("gpt-4o-mini", msgs, 0, 10)]
    items = asyncio.run(collect())
    assert items[:2] == ["Hel", "lo"] and items[-1].text == "Hello" and items[-1].output_tokens == 2


def test_anthropic_adapter_parses_response_and_http_errors():
    def ok(request: httpx.Request):
        assert request.headers["x-api-key"] == "k" and json.loads(request.content)["system"] == "be nice"
        return httpx.Response(200, json={"content": [{"type": "text", "text": "Hi!"}],
                                         "usage": {"input_tokens": 7, "output_tokens": 3}})
    p = AnthropicProvider("k", transport=httpx.MockTransport(ok))
    c = asyncio.run(p.complete("claude-sonnet-4-6", [{"role": "system", "content": "be nice"},
                                                     {"role": "user", "content": "hi"}], 0, 10))
    assert (c.text, c.input_tokens, c.output_tokens) == ("Hi!", 7, 3)

    from src.providers import ProviderError
    bad = AnthropicProvider("k", transport=httpx.MockTransport(lambda r: httpx.Response(529, text="overloaded")))
    with pytest.raises(ProviderError):
        asyncio.run(bad.complete("claude-sonnet-4-6", [{"role": "user", "content": "hi"}], 0, 10))
