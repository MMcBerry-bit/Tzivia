# Tzivia LoCoMo benchmark: test setup

These are the files used in the November 2026 LoCoMo comparison on
[tzivia.dev/benchmarks](https://tzivia.dev/benchmarks). They cover how Mem0
was configured, the prompts every system got, and how answers were graded.
Tzivia's engine is closed source and is not included.

| File | What it shows |
|---|---|
| `mem0_system.py` | The Mem0 setup: open-source `mem0ai`, gpt-4o-mini for memory extraction (temperature 0), all-MiniLM-L6-v2 embeddings, local Qdrant with hybrid semantic + BM25 search, `top_k=10`, turns added two at a time. |
| `qa_runner.py` | The answer prompt. Every system goes through this one function; the only thing that differs between systems is the context its `retrieve()` returns. |
| `scorers.py` | The judge prompt (`JUDGE_PROMPT_TEMPLATE`), plus exact-match and token-F1. The judge's answer counts as correct only if it starts with `CORRECT`. |
| `baselines.py` | Naive RAG (same MiniLM embeddings, one vector per turn), BM25, and whole-conversation baselines. |
| `ollama_client.py` | Model routing. With `BENCH_LLM=openai`, gpt-4o-mini answers, judges and runs Mem0's extraction, all at temperature 0. |

## Protocol

- Dataset: [LoCoMo](https://github.com/snap-research/locomo) `locomo10.json`,
  all 10 conversations. Question categories 1-4 (1,540 questions); category 5
  (adversarial) is excluded, as in the Mem0 paper.
- Each system indexes each conversation once, from identical input: every
  turn in order, with its session date in the text.
- Every system gets the same retrieval budget (about 500 tokens of context),
  the same answer prompt, the same answer model (gpt-4o-mini) and the same
  judge (gpt-4o-mini), all at temperature 0.
- Mem0 here is the open-source library run by us, not Mem0's hosted platform.
  Mem0 publishes higher LoCoMo scores from its own setup.

## Caveats

- LLM-judge scores vary by about 1-2 points between runs.
- The per-question outputs of the November run were not kept. They will be
  published here with the next full run, along with a Mem0 column for the
  question-type table from that same run.
