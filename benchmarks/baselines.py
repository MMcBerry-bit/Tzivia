# ===================================================
# Baseline retrieval systems, all sharing the same turn-list input shape:
#   turns: list of {"speaker": str, "text": str}  (LoCoMo)
#       or {"role": str, "content": str}           (LongMemEval)
# normalized to {"speaker": str, "text": str} by normalize_turns() below.
#
# Every baseline exposes .index(turns) and .retrieve(question, token_budget)
# -> (context_str, latency_s, tokens_injected), so the QA runner can treat
# them identically to the Tzivia system.
# ===================================================
from __future__ import annotations

import time

from rank_bm25 import BM25Okapi
from ollama_client import estimate_tokens


def normalize_turns(turns: list) -> list[dict]:
    out = []
    for t in turns:
        if "speaker" in t:
            out.append({"speaker": t["speaker"], "text": t.get("text", "")})
        else:
            out.append({"speaker": t.get("role", "unknown"), "text": t.get("content", "")})
    return out


def _turn_str(t: dict) -> str:
    return f"{t['speaker']}: {t['text']}"


class FullContextBaseline:
    """Baseline A: everything, verbatim, in the prompt. Only usable where
    the corpus is small enough to be tractable on CPU-only local inference
    (see benchmarks/report.md for which runs actually used this)."""
    name = "full_context"

    def __init__(self):
        self._turns: list[dict] = []

    def index(self, turns: list) -> float:
        t0 = time.time()
        self._turns = normalize_turns(turns)
        return time.time() - t0

    def retrieve(self, question: str, token_budget: int | None = None) -> tuple[str, float, int]:
        t0 = time.time()
        context = "\n".join(_turn_str(t) for t in self._turns)
        elapsed = time.time() - t0
        return context, elapsed, estimate_tokens(context)


class BM25Baseline:
    """Baseline C: BM25 keyword retrieval, one turn = one document, top-k
    turns within the given token budget (same budget Tzivia's recall used,
    for an apples-to-apples tokens-injected comparison)."""
    name = "bm25"

    def __init__(self):
        self._turns: list[dict] = []
        self._bm25: BM25Okapi | None = None

    def index(self, turns: list) -> float:
        t0 = time.time()
        self._turns = normalize_turns(turns)
        tokenized = [_turn_str(t).lower().split() for t in self._turns]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        return time.time() - t0

    def retrieve(self, question: str, token_budget: int = 500) -> tuple[str, float, int]:
        t0 = time.time()
        if not self._bm25:
            return "", time.time() - t0, 0
        scores = self._bm25.get_scores(question.lower().split())
        ranked = sorted(range(len(self._turns)), key=lambda i: scores[i], reverse=True)
        lines: list[str] = []
        tokens_used = 0
        for i in ranked:
            line = _turn_str(self._turns[i])
            line_tokens = estimate_tokens(line)
            if tokens_used + line_tokens > token_budget and lines:
                break
            lines.append(line)
            tokens_used += line_tokens
        context = "\n".join(lines)
        elapsed = time.time() - t0
        return context, elapsed, estimate_tokens(context)


class NaiveRAGBaseline:
    """Baseline B: fixed-size chunks (one turn = one chunk), same
    all-MiniLM-L6-v2 embeddings Tzivia uses, top-k by cosine similarity
    within the same token budget."""
    name = "naive_rag"

    _shared_encoder = None  # class-level cache, same pattern as Tzivia's VectorIndex

    def __init__(self):
        self._turns: list[dict] = []
        self._embeddings = None

    @classmethod
    def _get_encoder(cls):
        if cls._shared_encoder is None:
            from sentence_transformers import SentenceTransformer
            import os
            # Same model file Tzivia uses (baked into the server image);
            # falls back to the by-name download on a dev machine.
            cls._shared_encoder = SentenceTransformer(
                os.environ.get("TZIVIA_MODEL_DIR") or "all-MiniLM-L6-v2")
        return cls._shared_encoder

    def index(self, turns: list) -> float:
        t0 = time.time()
        self._turns = normalize_turns(turns)
        encoder = self._get_encoder()
        texts = [_turn_str(t) for t in self._turns]
        self._embeddings = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False) if texts else None
        return time.time() - t0

    def retrieve(self, question: str, token_budget: int = 500) -> tuple[str, float, int]:
        t0 = time.time()
        if self._embeddings is None or len(self._turns) == 0:
            return "", time.time() - t0, 0
        import numpy as np
        encoder = self._get_encoder()
        q_vec = encoder.encode([question], convert_to_numpy=True, show_progress_bar=False)[0]
        sims = self._embeddings @ q_vec / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(q_vec) + 1e-8
        )
        ranked = sims.argsort()[::-1]
        lines: list[str] = []
        tokens_used = 0
        for i in ranked:
            line = _turn_str(self._turns[i])
            line_tokens = estimate_tokens(line)
            if tokens_used + line_tokens > token_budget and lines:
                break
            lines.append(line)
            tokens_used += line_tokens
        context = "\n".join(lines)
        elapsed = time.time() - t0
        return context, elapsed, estimate_tokens(context)


class OracleContextBaseline:
    """LongMemEval-specific: only the gold evidence session(s), not the
    full haystack. A cheap ceiling baseline (small enough for CPU) showing
    what accuracy looks like if retrieval were perfect -- the counterpart
    to the (infeasible-here) full-context baseline."""
    name = "oracle_context"

    def __init__(self):
        self._turns: list[dict] = []

    def index(self, turns: list) -> float:
        t0 = time.time()
        self._turns = normalize_turns(turns)
        return time.time() - t0

    def retrieve(self, question: str, token_budget: int | None = None) -> tuple[str, float, int]:
        t0 = time.time()
        context = "\n".join(_turn_str(t) for t in self._turns)
        elapsed = time.time() - t0
        return context, elapsed, estimate_tokens(context)
