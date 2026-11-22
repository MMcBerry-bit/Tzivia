# ===================================================
# QA pipeline: index -> retrieve -> answer (local LLM) -> score
# (exact-match, token-F1, local-LLM judge). One shared runner for every
# system so the prompts, models, and scoring are byte-identical across
# Tzivia, the baselines, and Mem0 -- the only thing that differs is what
# each system's .retrieve() returns as context.
# ===================================================
from __future__ import annotations

import time

from ollama_client import generate, estimate_tokens
from scorers import exact_match, token_f1, llm_judge

ANSWER_PROMPT_TEMPLATE = """Answer the question using ONLY the context below. If the context does not contain the answer, say "I don't know" rather than guessing. Be concise -- answer in as few words as possible.

Context:
{context}

Question: {question}

Answer:"""


def answer_question(system, question: str, gold_answer: str,
                     answer_model: str, judge_model: str,
                     token_budget: int = 500) -> dict:
    """Assumes system.index(turns) was already called ONCE by the caller --
    this only does retrieve+answer+score for a single question, so a
    corpus is never re-ingested per question (that bug cost real wall-clock
    time before it was caught: see benchmarks/report.md)."""
    context, retrieve_latency, tokens_injected = system.retrieve(question, token_budget)

    prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, question=question)
    prompt_tokens = estimate_tokens(prompt)

    t0 = time.time()
    gen = generate(answer_model, prompt, temperature=0.0)
    answer_latency = time.time() - t0
    prediction = gen["text"].strip()

    em = exact_match(prediction, gold_answer)
    f1 = token_f1(prediction, gold_answer)
    judge = llm_judge(question, gold_answer, prediction, judge_model)

    return {
        "system": system.name,
        "question": question,
        "gold_answer": gold_answer,
        "prediction": prediction,
        "exact_match": em,
        "token_f1": round(f1, 4),
        "judge_correct": judge["correct"],
        "judge_raw": judge["raw"],
        "retrieve_latency_s": round(retrieve_latency, 4),
        "answer_latency_s": round(answer_latency, 4),
        "judge_latency_s": round(judge["elapsed_s"], 4),
        "tokens_injected": tokens_injected,
        "answer_prompt_tokens_est": prompt_tokens,
        "answer_num_ctx_used": gen["num_ctx_used"],
        "judge_num_ctx_used": judge["num_ctx_used"],
    }
