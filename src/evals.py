"""Evaluation harness: scoring and summaries for comparing prompts/models on a test set."""
import re


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def score(mode: str, output: str, expected) -> float:
    exp = expected if isinstance(expected, list) else [expected]
    o = _norm(output)
    if mode == "exact":
        return 1.0 if o == _norm(exp[0]) else 0.0
    if mode == "f1":
        a, b = o.split(), _norm(exp[0]).split()
        common = sum(min(a.count(w), b.count(w)) for w in set(a))
        if not common:
            return 0.0
        p, r = common / len(a), common / len(b)
        return round(2 * p * r / (p + r), 4)
    return 1.0 if all(_norm(e) in o for e in exp) else 0.0  # contains


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))], 1)


def summarize(label: str, model: str, rows: list[dict]) -> dict:
    n = len(rows) or 1
    lat = [r["latency_ms"] for r in rows if not r["error"]]
    return {"label": label, "model": model, "cases": len(rows),
            "mean_score": round(sum(r["score"] for r in rows) / n, 4),
            "pass_rate": round(sum(1 for r in rows if r["score"] >= 1.0) / n, 4),
            "avg_latency_ms": round(sum(lat) / len(lat), 1) if lat else 0.0,
            "p95_latency_ms": percentile(lat, 95),
            "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 8),
            "errors": sum(1 for r in rows if r["error"])}


def pick_winner(summaries: list[dict]) -> str | None:
    best = sorted(summaries, key=lambda s: (-s["mean_score"], s["total_cost_usd"], s["avg_latency_ms"]))
    if len(best) > 1 and best[0]["mean_score"] == best[1]["mean_score"] and best[0]["total_cost_usd"] == best[1]["total_cost_usd"]:
        return None
    return best[0]["label"] if best else None
