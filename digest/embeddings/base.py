"""Embeddings plumbing (Phase 6, Appendix C 96): the Embedder protocol, the error type, and
vector helpers. Vectors are normalised to unit length and cached as base64 float16, and fresh
vectors go through the same round trip, so a cached run and a fresh run score identically."""

from __future__ import annotations

import base64
import binascii
import math
import struct
from collections.abc import Sequence
from typing import Protocol

MAX_DIM = 4096


class EmbeddingError(Exception):
    """An embeddings call or its answer was unusable. The message is a fixed, secret-free text."""


class Embedder(Protocol):
    provider: str
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in order. Raises EmbeddingError."""


def _finite(values: Sequence[object]) -> list[float]:
    out: list[float] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, int | float):
            raise EmbeddingError("embeddings: the vector holds a value that is not a number")
        try:
            number = float(v)  # a huge JSON integer raises OverflowError
        except OverflowError:
            raise EmbeddingError("embeddings: a number in the vector is too large") from None
        if not math.isfinite(number):
            raise EmbeddingError("embeddings: the vector holds a value that is not a number")
        out.append(number)
    return out


def normalize(vec: Sequence[object]) -> list[float]:
    """Unit-length copy. Rejects empty, oversized, non-numeric, non-finite and zero vectors."""
    if not 1 <= len(vec) <= MAX_DIM:
        raise EmbeddingError("embeddings: unexpected vector size")
    values = _finite(vec)
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0.0 or not math.isfinite(norm):
        raise EmbeddingError("embeddings: the vector has no usable length")
    return [v / norm for v in values]


def pack(unit: Sequence[float]) -> str:
    """Base64 of little-endian float16 (2 bytes per dimension)."""
    return base64.b64encode(struct.pack(f"<{len(unit)}e", *unit)).decode("ascii")


def unpack(data: str, dim: int) -> list[float] | None:
    """The vector, or None if the text is not exactly `dim` finite float16 values."""
    try:
        raw = base64.b64decode(data.encode("ascii"), validate=True)
    except (binascii.Error, UnicodeEncodeError):
        return None
    if len(raw) != dim * 2:
        return None
    values = list(struct.unpack(f"<{dim}e", raw))
    return values if all(math.isfinite(v) for v in values) else None


def quantize(vec: Sequence[object]) -> list[float]:
    """normalize, then the exact values a cached copy would give back."""
    unit = normalize(vec)
    restored = unpack(pack(unit), len(unit))
    if restored is None:  # cannot happen for unit vectors; kept as a guard
        raise EmbeddingError("embeddings: the vector could not be stored")
    return restored


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; 0.0 when the sizes differ or either vector has no length."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))
