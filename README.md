# Tzivia

**Middleware that makes LLMs faster and cheaper.**

Tzivia is a persistent memory and reasoning layer that attaches to any large language model. Instead of re-feeding entire context windows on every call, Tzivia retrieves only the relevant knowledge chains — cutting search space, latency, and token cost by orders of magnitude.

> ⚡ **~3,500 tokens/sec** ingestion: a 1,000,000-token corpus saved in under 5 minutes
> 🎯 **50.1% vs 44.6%** against Mem0 on the full LoCoMo long-term memory benchmark, with **$0** spent saving memories
> 🧠 **50ms** average recall (65ms p95) from a **1,000,000-token** memory

---

## The Problem

Every LLM call today pays the same tax: context has to be stuffed into the prompt, tokenized, and attended over from scratch. As knowledge bases grow, this gets slower and more expensive linearly — or worse. Retrieval-augmented approaches help, but most treat memory as a flat vector store with no structure, no reasoning, and no persistence of *insight*.

## The Approach

Tzivia models memory as a typed, decaying knowledge graph with a dedicated routing algorithm on top:

- **ARIA routing algorithm** — sub-linear traversal that reaches relevant nodes without scanning the full store.
- **4D vector scoring (Vec4)** — every node is scored across four independent dimensions: frequency, accuracy, connectivity, and inference weight. Retrieval balances all four rather than relying on similarity alone.
- **Typed semantic relationships** — 10 distinct edge types (not just "related to"), enabling a formal logic engine to reason over the graph with defined inference rules.
- **Synthesis nodes** — a persistent insight layer that doesn't decay. Discovered patterns are promoted into durable knowledge instead of being recomputed every session.
- **MCP server architecture** — Tzivia attaches to LLMs through the Model Context Protocol, making it a drop-in memory layer for any compatible model.

## Benchmarks

| Metric | Result | Condition |
|---|---|---|
| LoCoMo accuracy | **50.1%** (Mem0 44.6%, naive RAG 35.7%, full context 53.3%) | 10 conversations, 1,540 questions, gpt-4o-mini answer + judge |
| Recall latency | **50ms** avg / **65ms** p95 | 1,000,000-token memory, 8 vCPU / 8 GB |
| Ingestion | **~3,500 tokens/sec** | 1,000,000-token corpus, same server |
| Cost to save | **$0** (no LLM calls) | Mem0 calls an LLM on every message |

*Measured November 2026. LoCoMo: every system got the same questions, answer model and judge, with about 500 tokens of context per question; Mem0 is the open-source `mem0ai` library run by us, not Mem0's hosted platform or published setup. Full results: [tzivia.dev/benchmarks](https://tzivia.dev/benchmarks).*

## Applied Result: DNA Decay Finder

To validate the graph-decay mechanics on real data, Tzivia's decay model was applied to cancer genomics. The **DNA decay finder** detects mutation hotspots and was validated against the public **TCGA** cancer genomics dataset — demonstrating that the same routing and decay machinery that accelerates LLM memory generalizes to structured scientific data.

## Architecture at a Glance

```
        ┌─────────────┐
  LLM ──│ MCP Server  │── attachment point
        └──────┬──────┘
               │
        ┌──────▼───────────────────────────┐
        │  ARIA Routing  (sub-linear)       │
        ├──────────────────────────────────┤
        │  Typed Knowledge Graph            │
        │   • Vec4-scored nodes             │
        │   • 10 typed relationship edges   │
        │   • Logic engine (formal rules)   │
        ├──────────────────────────────────┤
        │  Synthesis Layer  (no decay)      │
        └──────────────────────────────────┘
               │
        ┌──────▼──────┐
        │   SQLite    │  local persistence
        └─────────────┘
```

- **SQLite persistence** — zero-configuration local storage; the full graph lives in a single file
- **Single-tenant by design** — your memory graph stays on your machine

## Roadmap

- PostgreSQL / Supabase backend for hosted, multi-tenant deployments
- Zero-knowledge encryption for multi-tenant mode
- Real-time 3D knowledge-graph visualization (Fibonacci-sphere layout, force-directed)

## Install

**Tzivia-AI v5.1.0 is out (Windows).**

1. Download `Tzivia-AI-Setup-5.1.0.exe` from the [latest release](https://github.com/MMcBerry-bit/Tzivia/releases/latest) and run it.
2. Connect your AI app:
   - **Claude Desktop:** download `tzivia-mcp-5.1.0.mcpb` from the same release and double-click it.
   - **Any MCP client:** Tzivia is listed in the official MCP Registry as `io.github.MMcBerry-bit/tzivia`.
3. Ask your AI to remember something, then ask it to recall it.

More at [tzivia.dev](https://tzivia.dev).

## Status

**v5.1.0 released November 6, 2026.** Free for personal use under the [Tzivia-AI EULA](https://tzivia.dev/terms). Core components — ARIA routing, Vec4 scoring, the typed edge store, the logic engine, and the validated DNA decay finder — are working and benchmarked.

## About

Built by **Matthew McBerry**, founder & CEO of **Dorcas Innovations**.

The full engine is under active development and kept private as core IP. This repository is a public overview — for a live demo, benchmark reproduction, or partnership inquiries, reach out.

📧 matthew@tzivia.dev

---

*Tzivia — persistent memory and reasoning for any LLM.*
