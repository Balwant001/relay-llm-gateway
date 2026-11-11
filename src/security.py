import hashlib
import secrets
import time
from dataclasses import dataclass

from fastapi import Header, HTTPException, Request

from . import config as cfg
from .db import lookup_key, month_spend, run
from .state import state


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def new_key() -> str:
    return "gw_" + secrets.token_urlsafe(24)


@dataclass
class KeyInfo:
    id: int
    name: str
    rate_limit: int
    budget: float | None


_key_cache: dict[str, tuple[float, KeyInfo | None]] = {}
_spend_cache: dict[int, tuple[float, float]] = {}


def clear_key_cache() -> None:
    _key_cache.clear()
    _spend_cache.clear()


async def require_key(request: Request, authorization: str | None = Header(None),
                      x_api_key: str | None = Header(None)) -> KeyInfo:
    token = x_api_key
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(401, "Missing API key. Send 'Authorization: Bearer <key>' or 'X-API-Key'.")

    h, now = hash_key(token), time.monotonic()
    cached = _key_cache.get(h)
    if cached and cached[0] > now:
        info = cached[1]
    else:
        row = await run(lookup_key, h)
        info = KeyInfo(row["id"], row["name"], row["rate_limit"], row["budget"]) if row else None
        _key_cache[h] = (now + 15, info)
    if info is None:
        raise HTTPException(401, "Invalid or revoked API key")

    # fixed-window rate limit, one counter per key per minute
    n = await state.store.incr(f"rl:{info.id}:{int(time.time() // 60)}", 70)
    rl = {"X-RateLimit-Limit": str(info.rate_limit), "X-RateLimit-Remaining": str(max(0, info.rate_limit - n))}
    if n > info.rate_limit:
        raise HTTPException(429, "Rate limit exceeded", headers={**rl, "Retry-After": str(60 - int(time.time()) % 60)})

    if info.budget is not None:
        c = _spend_cache.get(info.id)
        if not c or c[0] <= now:
            c = (now + 10, await run(month_spend, info.id))
            _spend_cache[info.id] = c
        if c[1] >= info.budget:
            raise HTTPException(402, "Monthly budget exhausted for this key", headers=rl)
    request.state.rl = rl
    return info


async def require_admin(x_admin_key: str | None = Header(None)) -> None:
    if not cfg.ADMIN_KEY:
        raise HTTPException(503, "Admin access is not configured")

    if not x_admin_key or not secrets.compare_digest(
        x_admin_key, cfg.ADMIN_KEY
    ):
        raise HTTPException(401, "Invalid admin key")