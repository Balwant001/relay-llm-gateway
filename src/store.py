"""Key-value store for the response cache and rate-limit counters.
Redis when REDIS_URL is set, otherwise an in-process dict with TTLs."""
import time

from . import config as cfg


class MemoryStore:
    kind = "memory"

    def __init__(self, max_items: int = 5000):
        self._d: dict[str, tuple[float, str]] = {}
        self._max = max_items

    def _purge(self):
        now = time.monotonic()
        for k in [k for k, (exp, _) in self._d.items() if exp <= now]:
            del self._d[k]
        while len(self._d) > self._max:
            self._d.pop(next(iter(self._d)))

    async def get(self, key: str):
        item = self._d.get(key)
        if not item or item[0] <= time.monotonic():
            self._d.pop(key, None)
            return None
        return item[1]

    async def set(self, key: str, value: str, ttl: int):
        self._d[key] = (time.monotonic() + ttl, value)
        if len(self._d) > self._max:
            self._purge()

    async def incr(self, key: str, ttl: int) -> int:
        cur = await self.get(key)
        n = int(cur or 0) + 1
        exp = self._d[key][0] if cur is not None else time.monotonic() + ttl
        self._d[key] = (exp, str(n))
        return n


class RedisStore:
    kind = "redis"

    def __init__(self, url: str):
        import redis.asyncio as redis

        self.r = redis.from_url(url, decode_responses=True)

    async def get(self, key: str):
        return await self.r.get(key)

    async def set(self, key: str, value: str, ttl: int):
        await self.r.set(key, value, ex=ttl)

    async def incr(self, key: str, ttl: int) -> int:
        n = await self.r.incr(key)
        if n == 1:
            await self.r.expire(key, ttl)
        return int(n)


def make_store():
    return RedisStore(cfg.REDIS_URL) if cfg.REDIS_URL else MemoryStore()
