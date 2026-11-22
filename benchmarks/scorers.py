# ===================================================
# Judge-free scoring (exact-match, token-F1) plus the local-LLM judge.
#
# exact-match/F1 are computed with zero LLM calls -- a cross-check against
# judge-model noise, and the same family of metric the original LoCoMo
# paper itself reports (F1/BLEU-1 alongside their LLM-judge score).
# ===================================================
from __future__ import annotations

import re
import string
from collections import Counter

from ollama_client import generate


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


JUDGE_PROMPT_TEMPLATE = """You are grading a question-answering system. Given the question, the reference (gold) answer, and the system's predicted answer, judge whether the predicted answer is factually correct and equivalent in meaning to the reference answer. Minor phrasing differences are fine; the core fact must match.

Question: {question}
Reference answer: {gold}
Predicted answer: {prediction}

Respond with exactly one word: CORRECT or INCORRECT."""


def llm_judge(question: str, gold: str, prediction: str, model: str) -> dict:
    """Local-model judge. Returns {"correct": bool, "raw": str, "elapsed_s": float}.
    Any non-"CORRECT"-leading response is scored INCORRECT (fails closed,
    not open, on ambiguous judge output)."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, gold=gold, prediction=prediction)
    result = generate(model, prompt, temperature=0.0)
    raw = result["text"].strip()
    correct = raw.upper().startswith("CORRECT")
    return {
        "correct": correct,
        "raw": raw,
        "elapsed_s": result["elapsed_s"],
        "num_ctx_used": result["num_ctx_used"],
    }
