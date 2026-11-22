# ===================================================
# Mem0 wrapped in the same index(turns)/retrieve(question, token_budget)
# interface as everything else.
#   - LLM (memory extraction + updates): BENCH_LLM=openai -> the SAME OpenAI
#     model that answers and judges for every system; otherwise local Ollama.
#   - Embedder: HuggingFace all-MiniLM-L6-v2 (the SAME embedding model Tzivia
#     and the naive-RAG baseline use, so the comparison isolates the memory
#     architecture, not the embedding choice).
#   - Vector store: Qdrant local mode (hybrid semantic + BM25), no server;
#     spaCy installed for Mem0's entity/lemma handling.
# One fresh Mem0 instance (isolated Chroma dir) per item, mirroring Tzivia's
# per-item throwaway-account isolation. Turns are added two at a time (one
# exchange), as Mem0's own LoCoMo evaluation feeds conversation messages.
# Mem0's OpenAI calls are metered into the shared benchmark spend cap.
# ===================================================
from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid

import ollama_client
from ollama_client import estimate_tokens

_PATCHED = False


def _meter_openai():
    """Count Mem0's own OpenAI usage toward BENCH_MAX_USD and stop at the cap."""
    global _PATCHED
    if _PATCHED:
        return
    import openai.resources.chat.completions as cc
    orig = cc.Completions.create

    def create(self, *a, **kw):
        import openai
        ollama_client._check_budget()
        delay = 2.0
        for attempt in range(10):          # ride out rate limits instead of losing the save
            try:
                resp = orig(self, *a, **kw)
                break
            except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
                    openai.InternalServerError) as e:
                if "insufficient_quota" in str(e) or attempt == 9:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60)
        u = getattr(resp, "usage", None)
        if u is not None:
            ollama_client.charge(int(getattr(u, "prompt_tokens", 0) or 0),
                                 int(getattr(u, "completion_tokens", 0) or 0))
        return resp

    cc.Completions.create = create
    _PATCHED = True


def build_config(persist_dir: str) -> dict:
    if ollama_client.USE_OPENAI:
        llm = {"provider": "openai",
               "config": {"model": ollama_client.OPENAI_MODEL, "temperature": 0.0}}
    else:
        llm = {"provider": "ollama",
               "config": {"model": "llama3.1:8b", "temperature": 0.0,
                          "ollama_base_url": ollama_client.OLLAMA_BASE}}
    return {
        "llm": llm,
        "embedder": {"provider": "huggingface",
                     "config": {"model": os.environ.get("TZIVIA_MODEL_DIR")
                                or "sentence-transformers/all-MiniLM-L6-v2"}},
        # Qdrant (local, on-disk): Mem0's recommended store, and the one that gives
        # it hybrid semantic + BM25 keyword search (fastembed). Chroma has no
        # keyword search, which would handicap Mem0.
        "vector_store": {"provider": "qdrant",
                         "config": {"collection_name": "bench", "path": persist_dir,
                                    "embedding_model_dims": 384}},
    }


class Mem0System:
    name = "mem0"

    def __init__(self):
        self._mem = None
        self._dir: str | None = None
        self._user_id = "bench_user"

    def index(self, turns: list) -> float:
        from mem0 import Memory
        if ollama_client.USE_OPENAI:
            _meter_openai()
        self._dir = tempfile.mkdtemp(prefix="mem0_bench_")
        self._mem = Memory.from_config(build_config(self._dir))
        self._user_id = f"bench_{uuid.uuid4().hex[:8]}"
        t0 = time.time()
        msgs = []
        for t in turns:
            speaker = t.get("speaker", t.get("role", "unknown"))
            text = t.get("text", t.get("content", ""))
            if str(text).strip():
                msgs.append({"role": "user", "content": f"{speaker}: {text}"})
        for i in range(0, len(msgs), 2):
            self._mem.add(msgs[i:i + 2], user_id=self._user_id)
            if i % 100 == 0:
                print(f"[mem0] added {i + 2}/{len(msgs)} messages in {time.time() - t0:.0f}s", flush=True)
        return time.time() - t0

    def retrieve(self, question: str, token_budget: int = 500) -> tuple[str, float, int]:
        t0 = time.time()
        results = self._mem.search(question, top_k=10, filters={"user_id": self._user_id})
        elapsed = time.time() - t0
        memories = results.get("results", results) if isinstance(results, dict) else results
        lines, tokens_used = [], 0
        for m in memories:
            line = m.get("memory", "") if isinstance(m, dict) else str(m)
            lt = estimate_tokens(line)
            if tokens_used + lt > token_budget and lines:
                break
            lines.append(line)
            tokens_used += lt
        context = "\n".join(lines)
        return context, elapsed, estimate_tokens(context)

    def close(self):
        self._mem = None
        if self._dir:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None
