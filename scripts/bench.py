"""Zero-install load test: python scripts/bench.py --url http://localhost:8000 --key gw_xxx --requests 300 --concurrency 20
Sends a mix of repeated prompts (cache hits) and unique prompts (cache misses) and prints latency percentiles."""
import argparse
import asyncio
import random
import statistics
import time

import httpx

PROMPTS = ["What is the capital of France?", "What is the capital of Japan?", "What is 12 * 7?",
           "What is the capital of India?", "What is 45 + 18?"]


async def main(a):
    lat: dict[str, list[float]] = {"none": [], "exact": [], "semantic": []}
    errors = 0
    sem = asyncio.Semaphore(a.concurrency)
    async with httpx.AsyncClient(base_url=a.url, timeout=60, headers={"Authorization": f"Bearer {a.key}"}) as c:
        async def one(i):
            nonlocal errors
            text = random.choice(PROMPTS) if random.random() < a.repeat_ratio else f"What is {i} + {random.randint(1, 999)}?"
            async with sem:
                t = time.perf_counter()
                r = await c.post("/v1/chat", json={"model": a.model, "messages": [{"role": "user", "content": text}]})
                ms = (time.perf_counter() - t) * 1000
            if r.status_code == 200:
                lat[r.json()["cached"]].append(ms)
            else:
                errors += 1
        t0 = time.perf_counter()
        await asyncio.gather(*[one(i) for i in range(a.requests)])
        wall = time.perf_counter() - t0
    allv = sorted(x for v in lat.values() for x in v)
    q = lambda xs, p: round(xs[min(len(xs) - 1, int(p / 100 * len(xs)))], 1) if xs else 0
    print(f"requests={a.requests} concurrency={a.concurrency} wall={wall:.1f}s throughput={a.requests / wall:.1f} req/s errors={errors}")
    print(f"overall   p50={q(allv, 50)}ms p95={q(allv, 95)}ms p99={q(allv, 99)}ms")
    for k, v in lat.items():
        if v:
            print(f"{k:9} n={len(v):4} mean={statistics.mean(v):.1f}ms p95={q(sorted(v), 95)}ms")
    hits = len(lat["exact"]) + len(lat["semantic"])
    print(f"cache hit rate = {hits / max(1, len(allv)):.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--key", required=True)
    ap.add_argument("--model", default="mock-smart")
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--repeat-ratio", type=float, default=0.6)
    asyncio.run(main(ap.parse_args()))
