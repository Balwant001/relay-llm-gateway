from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUESTS = Counter("gateway_requests_total", "Gateway requests", ["provider", "model", "cache", "status"])
LATENCY = Histogram("gateway_latency_seconds", "End-to-end latency", ["cache"],
                    buckets=(.01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30))
COST = Counter("gateway_cost_usd_total", "Spend in USD", ["model"])
SAVED = Counter("gateway_saved_usd_total", "Spend avoided by the cache in USD")
TOKENS = Counter("gateway_tokens_total", "Tokens processed", ["direction"])
FALLBACKS = Counter("gateway_fallbacks_total", "Requests served by a fallback model")


def observe(res: dict | None, status: str = "ok", requested: str = "") -> None:
    if res is None:
        REQUESTS.labels("none", requested or "unknown", "none", status).inc()
        return
    REQUESTS.labels(res["provider"], res["model"], res["cached"], status).inc()
    LATENCY.labels(res["cached"]).observe(res["latency_ms"] / 1000)
    COST.labels(res["model"]).inc(res["cost_usd"])
    SAVED.inc(res["saved_usd"])
    TOKENS.labels("input").inc(res["input_tokens"])
    TOKENS.labels("output").inc(res["output_tokens"])
    if res["fallback_used"]:
        FALLBACKS.inc()


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
