from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from ..config import EMBEDDING_DIM, EMBEDDING_MODEL


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


def _l2_normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class HashEmbedder:
    """Deterministic hashing embedder, used when sentence-transformers isn't installed."""

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 7) & 1 else -1.0
        return _l2_normalize(vec)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class PubMedBERTEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.dim = EMBEDDING_DIM
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> list[list[float]]:
        self._ensure()
        return self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True).tolist()


def get_embedder(kind: str = "auto") -> Embedder:
    if kind == "hash":
        return HashEmbedder()
    if kind == "pubmedbert":
        return PubMedBERTEmbedder()
    if kind == "auto":
        try:
            import sentence_transformers  # noqa: F401
            return PubMedBERTEmbedder()
        except ImportError:
            return HashEmbedder()
    raise ValueError(f"unknown embedder kind: {kind!r}")
