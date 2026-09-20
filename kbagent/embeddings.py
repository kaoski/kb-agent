"""Embedders behind one interface.

The default `HashEmbedder` needs no network and no API key, so the walking
skeleton runs and is testable offline. Swap to `LiteLLMEmbedder` (one config
change) for real semantic embeddings across any provider.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Deterministic bag-of-words hashing into a fixed vector.

    Not semantic — it captures lexical overlap only — but it is real,
    dependency-free, and good enough to prove the retrieval loop end to end.
    Replace with a real embedder for production recall.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _TOKEN.findall(text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm:
            vec = [v / norm for v in vec]
        return vec


class LiteLLMEmbedder:
    """Real embeddings through the LiteLLM gateway, so the provider is a
    config value (Voyage, OpenAI, a local model, ...)."""

    def __init__(self, model: str = "voyage/voyage-3", dim: int = 1024) -> None:
        self.model = model
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        import litellm  # imported lazily so the offline path needs no install

        resp = litellm.embedding(model=self.model, input=[text])
        return resp["data"][0]["embedding"]


def build_embedder(config: dict) -> Embedder:
    backend = (config or {}).get("backend", "hash")
    if backend == "hash":
        return HashEmbedder(dim=config.get("dim", 256))
    if backend == "litellm":
        return LiteLLMEmbedder(
            model=config.get("model", "voyage/voyage-3"),
            dim=config.get("dim", 1024),
        )
    raise ValueError(f"Unknown embedder backend: {backend}")
