# ===================================================
# Minimal Ollama HTTP client for the benchmark harness.
#
# The one thing that matters here: Ollama's default context window
# (num_ctx) is small (commonly 2048-4096 tokens) regardless of what the
# underlying model actually supports. If a prompt exceeds num_ctx, Ollama
# SILENTLY truncates it from the front -- no error, no warning -- which
# would make a "full-context" baseline look artificially bad (the model
# genuinely never sees most of the conversation) rather than measuring
# what it's supposed to measure. Every call here computes an estimated
# prompt token count and sets num_ctx explicitly above it, logging a loud
# warning (not silently proceeding) if the required context would be
# large enough to risk memory pressure on this machine (16GB RAM, no GPU).
# ===================================================
from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error

# OLLAMA_BASE_URL lets the model run in a separate machine/container (so the
# LLM and the memory system under test don't share one memory limit).
OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_URL = OLLAMA_BASE + "/api/generate"

# Rough, conservative token estimate (chars/4) -- consistent with the
# same heuristic tzivia_mcp.py already uses (_estimate_tokens), so context
# sizing and reported token counts use the same yardstick throughout this
# comparison.
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

# Hard ceiling this machine should attempt: 16GB total RAM, no GPU. Above
# this, KV cache + model weights risk real memory pressure. Calls that
# would need more than this raise instead of silently truncating.
MAX_SAFE_NUM_CTX = 40000

_WARNED_LARGE_CTX: set[int] = set()


def _pick_num_ctx(prompt_tokens: int) -> int:
    """Smallest 'clean' context window that comfortably fits the prompt
    plus room for the answer, never silently truncating."""
    needed = prompt_tokens + 1024  # headroom for the response itself
    for candidate in (2048, 4096, 8192, 16384, 32768, MAX_SAFE_NUM_CTX):
        if candidate >= needed:
            return candidate
    raise ValueError(
        f"Prompt needs ~{needed} tokens of context, which exceeds this "
        f"machine's safe ceiling ({MAX_SAFE_NUM_CTX}) -- refusing to run "
        f"rather than risk OOM or a silent truncation. Shrink the prompt "
        f"or the subset."
    )


# -- OpenAI backend (BENCH_LLM=openai) ------------------------------------------
# One hosted model answers AND judges for every system, so the only thing that
# differs between systems is the context their memory layer returns. Spend is
# metered from the API's own usage numbers and capped (BENCH_MAX_USD): past the
# cap every call raises, and the run stops instead of overspending.
import threading as _threading

USE_OPENAI = os.environ.get("BENCH_LLM", "").lower() == "openai"
OPENAI_MODEL = os.environ.get("BENCH_OPENAI_MODEL", "gpt-4o-mini")
# USD per 1M tokens (input, output); override with BENCH_PRICE_IN / BENCH_PRICE_OUT.
_PRICE_IN = float(os.environ.get("BENCH_PRICE_IN", "0.15"))
_PRICE_OUT = float(os.environ.get("BENCH_PRICE_OUT", "0.60"))
MAX_USD = float(os.environ.get("BENCH_MAX_USD", "25"))
_spend = {"usd": 0.0, "in": 0, "out": 0, "calls": 0}
_spend_lock = _threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


def spend() -> dict:
    with _spend_lock:
        return dict(_spend)


def charge(tokens_in: int, tokens_out: int) -> None:
    """Record usage from any OpenAI caller (answers, judge, Mem0)."""
    with _spend_lock:
        _spend["in"] += tokens_in
        _spend["out"] += tokens_out
        _spend["calls"] += 1
        _spend["usd"] += tokens_in / 1e6 * _PRICE_IN + tokens_out / 1e6 * _PRICE_OUT


def _check_budget():
    if spend()["usd"] >= MAX_USD:
        raise BudgetExceeded(f"benchmark spend cap reached (${MAX_USD:.2f})")


def _openai_chat(prompt: str, temperature: float, timeout: float) -> dict:
    _check_budget()
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("BENCH_LLM=openai but OPENAI_API_KEY is not set")
    body = json.dumps({"model": OPENAI_MODEL, "temperature": temperature,
                       "messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
    delay = 2.0
    for attempt in range(8):
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body,
                                     method="POST", headers={"Content-Type": "application/json",
                                                             "Authorization": f"Bearer {key}"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=min(timeout, 180)) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:300]
            if e.code in (429, 500, 502, 503, 504) and "insufficient_quota" not in msg and attempt < 7:
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise RuntimeError(f"OpenAI HTTP {e.code}: {msg}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 7:
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise RuntimeError(f"OpenAI unreachable: {e}") from e
    usage = result.get("usage") or {}
    charge(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))
    text = result["choices"][0]["message"].get("content") or ""
    return {"text": text, "elapsed_s": time.time() - t0, "num_ctx_used": None,
            "prompt_tokens_est": estimate_tokens(prompt), "model": OPENAI_MODEL}


def generate(model: str, prompt: str, temperature: float = 0.0,
             timeout: float = 1800.0) -> dict:
    """Call Ollama's /api/generate with num_ctx sized to the actual prompt,
    KV cache quantized to q8_0 to reduce RAM pressure for the larger
    contexts (LoCoMo full-context sample), and full timing.

    Returns {"text": ..., "elapsed_s": ..., "num_ctx_used": ..., "prompt_tokens_est": ...}
    """
    if USE_OPENAI:
        return _openai_chat(prompt, temperature, timeout)
    prompt_tokens = estimate_tokens(prompt)
    num_ctx = _pick_num_ctx(prompt_tokens)

    if num_ctx >= 16384 and num_ctx not in _WARNED_LARGE_CTX:
        _WARNED_LARGE_CTX.add(num_ctx)
        print(f"[ollama_client] WARNING: prompt needs num_ctx={num_ctx} "
              f"(~{prompt_tokens} est. tokens) -- large context window, "
              f"expect slow CPU prompt processing and real RAM pressure "
              f"on a 16GB machine.", flush=True)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            # Pin the weights in RAM. Under a tight container memory limit
            # the model was being paged out and re-read for every token
            # (observed: ~40 s per generated token on 8 GB / 8 vCPU).
            **({"use_mlock": True} if os.environ.get("OLLAMA_USE_MLOCK") == "1" else {}),
        },
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL} -- is `ollama serve` "
            f"running? ({e})"
        ) from e
    elapsed = time.time() - t0

    return {
        "text": result.get("response", ""),
        "elapsed_s": elapsed,
        "num_ctx_used": num_ctx,
        "prompt_tokens_est": prompt_tokens,
        "model": model,
    }


def is_reachable() -> bool:
    if USE_OPENAI:
        return bool(os.environ.get("OPENAI_API_KEY"))
    try:
        req = urllib.request.Request(OLLAMA_BASE + "/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False
