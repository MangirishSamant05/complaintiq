"""Duplicate detection.

Exact duplicates : identical text after normalisation -> same 64-bit hash.
Near duplicates  : MinHash signatures + LSH banding -> groups of almost-identical
                   complaints (e.g. the same template with a few words changed).
"""
from __future__ import annotations

import re
import zlib
from collections import defaultdict

import numpy as np
import pandas as pd

_AMOUNT = re.compile(r"\{\$[\d,]*\.?\d*\}")   # CFPB writes amounts as {$500.00}
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DIGITS = re.compile(r"\d+")
_REDACTED = re.compile(r"\bx{2,}\b")           # XX, XXXX (after lowercasing)
_SPACES = re.compile(r"\s+")

_MERSENNE_PRIME = (1 << 31) - 1


# ---------------------------------------------------------------- normalisation
def normalize_text(text: str) -> str:
    """Lowercase, mask amounts/numbers/redactions, drop punctuation."""
    text = _AMOUNT.sub(" amt ", text.lower())
    text = _NON_ALNUM.sub(" ", text)
    text = _DIGITS.sub("0", text)
    text = _REDACTED.sub("xx", text)
    return _SPACES.sub(" ", text).strip()


def normalize_series(texts: pd.Series) -> pd.Series:
    """Same as normalize_text, applied to a whole column."""
    out = texts.str.lower()
    out = out.str.replace(_AMOUNT, " amt ", regex=True)
    out = out.str.replace(_NON_ALNUM, " ", regex=True)
    out = out.str.replace(_DIGITS, "0", regex=True)
    out = out.str.replace(_REDACTED, "xx", regex=True)
    return out.str.replace(_SPACES, " ", regex=True).str.strip()


def text_hash(normalized: pd.Series) -> pd.Series:
    """Deterministic 64-bit hash per row (same text -> same number, every run)."""
    return pd.Series(pd.util.hash_pandas_object(normalized, index=False).to_numpy(),
                     index=normalized.index, dtype="uint64")


# ---------------------------------------------------------------- MinHash
class MinHasher:
    """Turns a text into `num_perm` numbers whose agreement rate between two
    texts estimates their Jaccard similarity (overlap of word n-grams)."""

    def __init__(self, num_perm: int = 64, shingle_words: int = 3, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.shingle_words = shingle_words
        self.a = rng.integers(1, _MERSENNE_PRIME, size=num_perm, dtype=np.int64)
        self.b = rng.integers(0, _MERSENNE_PRIME, size=num_perm, dtype=np.int64)

    def shingles(self, normalized: str) -> np.ndarray:
        words = normalized.split()
        k = self.shingle_words
        if len(words) <= k:
            grams = {" ".join(words)}
        else:
            grams = {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}
        return np.fromiter((zlib.crc32(g.encode()) % _MERSENNE_PRIME for g in grams),
                           dtype=np.int64, count=len(grams))

    def signature(self, normalized: str) -> np.ndarray:
        x = self.shingles(normalized)
        # one hash function per row: (a*x + b) mod p, keep the minimum per row
        return ((self.a[:, None] * x[None, :] + self.b[:, None]) % _MERSENNE_PRIME).min(axis=1)

    def signatures(self, normalized_texts) -> np.ndarray:
        return np.vstack([self.signature(t) for t in normalized_texts])


def estimated_similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))


# ---------------------------------------------------------------- LSH grouping
class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:          # path compression
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


def near_duplicate_groups(signatures: np.ndarray, bands: int = 16,
                          threshold: float = 0.7, max_pairwise: int = 200) -> np.ndarray:
    """Return a group id per row; rows in the same group are near-duplicates.

    LSH banding: split each signature into `bands` slices. Two texts that share
    ANY identical slice become candidates; candidates are then confirmed by
    their estimated similarity >= threshold. This avoids comparing all pairs.
    """
    n, num_perm = signatures.shape
    if num_perm % bands:
        raise ValueError(f"num_perm ({num_perm}) must be divisible by bands ({bands})")
    rows = num_perm // bands
    uf = _UnionFind(n)

    for band in range(bands):
        block = np.ascontiguousarray(signatures[:, band * rows:(band + 1) * rows])
        buckets: dict[bytes, list[int]] = defaultdict(list)
        for i, row in enumerate(block):
            buckets[row.tobytes()].append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            idx = np.array(members)
            if len(idx) <= max_pairwise:
                # compare every pair inside a small bucket
                sig = signatures[idx]
                sims = (sig[:, None, :] == sig[None, :, :]).mean(axis=2)
                left, right = np.nonzero(np.triu(sims >= threshold, k=1))
                for a, b in zip(idx[left], idx[right]):
                    uf.union(int(a), int(b))
            else:
                # huge bucket (a big template family): compare to its first member
                sims = (signatures[idx[1:]] == signatures[idx[0]]).mean(axis=1)
                for j in idx[1:][sims >= threshold]:
                    uf.union(int(idx[0]), int(j))

    roots = np.array([uf.find(i) for i in range(n)])
    return pd.factorize(roots)[0]               # compact ids 0..k-1