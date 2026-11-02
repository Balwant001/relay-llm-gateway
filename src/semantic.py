"""Similarity cache index. Uses hashed character-trigram vectors: cheap, dependency-free, and good at
catching typos and small rewordings. For true semantic matching swap _vec() for a real embedding model."""
import math
import re
import zlib
from collections import deque


def _vec(text: str) -> dict[int, float]:
    t = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()
    t = f" {t} "
    v: dict[int, float] = {}
    for i in range(len(t) - 2):
        h = zlib.crc32(t[i:i + 3].encode()) % 2048
        v[h] = v.get(h, 0) + 1
    n = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {k: x / n for k, x in v.items()}


def _cos(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(k, 0.0) for k, x in a.items())


_NUM = re.compile(r"\d+(?:\.\d+)?")


class SemanticIndex:
    def __init__(self, threshold: float, max_per_scope: int = 300):
        self.threshold = threshold
        self.max = max_per_scope
        self.scopes: dict[str, deque] = {}

    def add(self, scope: str, text: str, key: str) -> None:
        self.scopes.setdefault(scope, deque(maxlen=self.max)).append((_vec(text), _NUM.findall(text), key))

    def find(self, scope: str, text: str):
        entries = self.scopes.get(scope)
        if not entries:
            return None
        v, nums = _vec(text), _NUM.findall(text)
        best_score, best_key = 0.0, None
        for vec, n2, key in entries:
            if n2 != nums:  # safety: "2+2" must never hit the cache entry for "2+3"
                continue
            s = _cos(v, vec)
            if s > best_score:
                best_score, best_key = s, key
        return best_key if best_score >= self.threshold else None
