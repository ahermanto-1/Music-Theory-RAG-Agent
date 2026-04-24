# Technical Architecture

## System Overview

The Music Theory RAG Agent is a conversational question-answering system built around a self-corrective retrieval pipeline. It routes each query to the appropriate data source (vector knowledge base, web search, or direct generation), maintains a rolling conversation window, and scores answer quality automatically.

---

## Component Map

```
┌─────────────────────────────────────────────────────────────┐
│                        app.py (Streamlit UI)                │
│  - Renders chat history                                     │
│  - Passes completed turns (messages[:-1]) as history        │
│  - Displays source badge (Knowledge base / Web / Direct)    │
└───────────────────────────┬─────────────────────────────────┘
                            │ run(query, history)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                        agent.py                             │
│                                                             │
│  1. _route(query, history)                                  │
│       └─► local model classifies: RETRIEVE | WEB_SEARCH |  │
│           DIRECT                                            │
│                                                             │
│  2a. RETRIEVE path                                          │
│       search_knowledge_base(query)    [retriever.py]        │
│       _grade(query, chunks)                                 │
│       if not relevant:                                      │
│           _rewrite_query(query, history)                    │
│           search_knowledge_base(rewritten)                  │
│       _generate(query, chunks, history, source="retrieve")  │
│                                                             │
│  2b. WEB_SEARCH path                                        │
│       web_search(query)               [web_search.py]       │
│       _generate(query, chunks, history, source="web_search")│
│                                                             │
│  2c. DIRECT path                                            │
│       _generate(query, [], history, source="direct")        │
│                                                             │
│  Returns: AgentResult(answer, chunks, rewritten, source)    │
└──────────┬──────────────────────────┬───────────────────────┘
           │                          │
           ▼                          ▼
┌──────────────────┐       ┌──────────────────────┐
│   retriever.py   │       │    web_search.py      │
│                  │       │                       │
│  SentenceTransf. │       │  TavilyClient         │
│  → embed query   │       │  → search(query)      │
│  Pinecone index  │       │  → returns [{text,    │
│  → top-5 chunks  │       │     source, score}]   │
└──────────────────┘       └──────────────────────┘
```

---

## Data Models

### `AgentResult` (agent.py)

```python
@dataclass
class AgentResult:
    answer:    str            # final generated answer
    chunks:    list[dict]     # passages used for generation (empty for DIRECT)
    rewritten: str | None     # rewritten query if self-correction fired, else None
    source:    str            # "retrieve" | "web_search" | "direct"
```

### Chunk dict (shared shape across retriever.py and web_search.py)

```python
{
    "text":    str,    # passage content
    "source":  str,    # filename (KB) or URL (web)
    "chapter": str,    # chapter heading (KB) or "" (web)
    "score":   float,  # cosine similarity (KB) or Tavily relevance (web)
}
```

Keeping the same shape across both sources means `_format_context()` and `_generate()` in `agent.py` work without branching on source type.

---

## Agent Decision Logic

### Router (`_route`)

Single local-model call via `_call`. Receives the query and completed conversation history (all turns before the current message). Returns one of three tokens:

| Token | When used |
| --- | --- |
| `RETRIEVE` | Music theory concept, technique, notation — likely in a textbook |
| `WEB_SEARCH` | Specific artist, gear, album, live news — outside textbook scope |
| `DIRECT` | Greeting, follow-up reference, summary request, conversational reply |

Falls back to `RETRIEVE` if the model returns an unexpected value. Bias is intentionally toward `RETRIEVE` — the prompt instructs the model to use `WEB_SEARCH` and `DIRECT` only when clearly warranted.

### Relevance grader (`_grade`)

Used only on the `RETRIEVE` path. Single local-model call via `_call`. Returns `True` if the retrieved chunks are sufficient to answer the query, `False` if not.

### Query rewriter (`_rewrite_query`)

Fires only when `_grade` returns `False`. Receives the original query and conversation history (to resolve pronouns and references). Rewrites into technical music theory vocabulary that better matches textbook language in the Pinecone index.

---

## Conversation History

History is managed by `app.py` in `st.session_state.messages`, a list of `{role, content, chunks, rewritten, source}` dicts.

On each turn, `app.py` passes `messages[:-1]` (all completed turns, excluding the current user message) to `run()`. This prevents the current query from appearing twice — once in history and once as the `query` argument — which would confuse the router. Inside `run()`, history is further truncated to the last `HISTORY_WINDOW = 10` messages before use.

History flows into three places:

| Where | How |
| --- | --- |
| `_route` | Embedded as a plain-text transcript in the prompt; helps detect follow-ups |
| `_rewrite_query` | Embedded as plain text; resolves pronouns before re-searching ("it", "that one") |
| `_generate` | Passed as actual `user`/`assistant` message turns to the chat completions API |

`_generate` is the critical path for response quality. It uses `_call_messages`, which builds a proper multi-turn messages list — system prompt first, then alternating `user`/`assistant` turns from history, then the current user message — and sends it to the local model. This gives the model the conversation context as native chat turns rather than embedded text, which local models track significantly more reliably.

Only the `content` field of each history entry is forwarded — `chunks`, `rewritten`, and `source` are UI metadata only.

---

## LLM Calls Per Turn

| Path | Calls | Purpose |
| --- | --- | --- |
| RETRIEVE (relevant on first try) | 3 | route + grade + generate |
| RETRIEVE (self-correction fired) | 4 | route + grade + rewrite + generate |
| WEB_SEARCH | 2 | route + generate |
| DIRECT | 2 | route + generate |

All calls go to a local model served by LM Studio via an OpenAI-compatible endpoint. Two call patterns exist:

- `_call(prompt)` — single `user` message; used for routing, grading, and query rewriting, where conversation history is embedded in the prompt text.
- `_call_messages(messages)` — full message list with `system`, `user`, and `assistant` turns; used only in `_generate` so the model receives proper chat context.

---

## External Services

| Service | SDK | Purpose | Key env var |
| --- | --- | --- | --- |
| LM Studio (local) | `openai` (OpenAI-compatible) | All LLM calls via local model | `LM_STUDIO_BASE_URL`, `LM_STUDIO_MODEL` |
| Pinecone | `pinecone` | Vector database for KB chunks | `PINECONE_API_KEY` |
| Tavily | `tavily-python` | Web search | `TAVILY_API_KEY` |
| HuggingFace (local) | `sentence-transformers` | Embedding model (all-MiniLM-L6-v2) | — |

**LM Studio setup:** Load any OpenAI-compatible model in LM Studio and start the local server (default `http://localhost:1234`). Set `LM_STUDIO_BASE_URL` and `LM_STUDIO_MODEL` in `.env`; the agent uses the OpenAI Python SDK pointed at the local endpoint. No API key is required — `lm-studio` is passed as a placeholder.

The embedding model also runs locally — no API call or key required. It is loaded once as a module-level singleton in `retriever.py` and reused across queries.

---

## Evaluation Pipeline (`eval_pipeline.py`)

Runs two synthetic questions through `agent.run()` with no history (each question is independent). Scores answers using DeepEval with the configured local model as the judge.

| Metric | What it measures | Threshold |
| --- | --- | --- |
| Faithfulness | Answer stays within retrieved context | ≥ 0.8 |
| Answer Relevancy | Answer addresses the question | ≥ 0.8 |

Failures are appended to `failures.csv`. Pipeline exits with code `1` on any failure, making it compatible with GitHub Actions as a quality gate.

---

## File Reference

| File | Role |
| --- | --- |
| `agent.py` | Orchestrator: router, grader, rewriter, generator, `AgentResult` |
| `retriever.py` | Pinecone vector search; embedding singleton |
| `web_search.py` | Tavily web search; `TavilyClient` singleton |
| `ingest.py` | One-time ETL: PDF → Markdown → chunks → Pinecone |
| `app.py` | Streamlit chat UI; session state; source badge rendering |
| `eval_pipeline.py` | DeepEval quality gate; Gemma judge wrapper |
| `requirements.txt` | Python dependencies |
| `.env` | API keys (not committed) |
