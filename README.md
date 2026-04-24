# Music Theory RAG Agent

---

## What does this do?

This is a chatbot trained on music theory textbooks. If you ask a question like *"How do I play a G major scale on guitar?"* the agent searches the textbooks for relevant passages to provide you with grounded answers.

The technical name for this approach is **RAG** (Retrieval Augmented Generation). Rather than relying on an AI's general training data, RAG forces the AI to answer from a specific knowledge base. Think of it like giving a lawyer the exact contract to reference before they advise you, rather than asking them to go from memory.

The agent also knows when not to use the knowledge base. If you ask about a specific artist, a piece of gear, or something that wouldn't appear in a textbook, it searches the web instead. And if you ask a follow-up like *"can you summarise what we just discussed?"*, it answers directly from conversation memory.

Everything runs locally, so you don't need to pay for LLM tokens. The language model is served through **LM Studio** on your machine. The only external services are Pinecone (vector database) and Tavily (web search), which provide generous free tiers for tinkering around. 

---

## The files that make it work


| File               | What it does                                                                                      |
| ------------------ | ------------------------------------------------------------------------------------------------- |
| `ingest.py`        | One-time setup: reads the PDF textbooks and stores them in a searchable database (Pinecone)       |
| `retriever.py`     | Search function: finds the most relevant passages for a given question                            |
| `web_search.py`    | Web search tool: fetches live results from the web when the knowledge base isn't enough (Tavily)  |
| `agent.py`         | The coordinator: routes each question, runs the full workflow, and maintains conversation context |
| `app.py`           | The chat interface: a Streamlit web app with conversation history                                 |
| `eval_pipeline.py` | Quality gate: automatically checks whether the answers are good                                   |


---

## How it works, step by step

### Step 1 — Ingestion (one-time setup)

Before the system can answer questions, it needs to read and index the textbooks. This happens once, before the system is used.

```
PDF textbooks
     ↓
Parse into text
     ↓
Split into small chunks of ~400 words each,
with slight overlap so nothing gets cut off mid-thought
     ↓
Convert each chunk into an embedding (a list of numbers
that represents its meaning)
     ↓
Store all embeddings + their original text in Pinecone
(a database that stores embeddings)
```

The database is now ready to be searched. This step doesn't need to be repeated unless you want to update the knowledge base.

---

### Step 2 — Routing

When a user asks a question, the agent first decides the best way to answer it. This is called routing.

```
User's question + completed conversation history (prior turns only)
     ↓
Ask local model: what kind of query is this?
     ↓
     ├── RETRIEVE   → search the music theory knowledge base
     ├── WEB_SEARCH → search the web via Tavily
     └── DIRECT     → answer from conversation history and general knowledge
```

**Why routing?** Not every question needs a database lookup. Asking *"what guitar did James Hetfield use to record Ride the Lightning?"* requires information that no textbook would contain. Asking *"thanks, that makes sense"* requires neither. Routing ensures each query takes the most efficient and appropriate path.

---

### Step 3 — Retrieval or web search

**Knowledge base path (RETRIEVE):**

The agent runs a self-correcting retrieval loop against the Pinecone knowledge base.

```
User's question
     ↓
Retrieve top 5 chunks (nearest-neighbour search by embedding similarity)
     ↓
Ask local model: "Are these chunks actually useful for answering the question?"
     ↓
     ├── YES → go straight to generating the answer
     │
     └── NO  → ask local model to rewrite the question using more technical
               language (e.g. "why does this chord sound sad?" →
               "minor third interval emotional character")
                    ↓
               Retrieve again with the rewritten question
                    ↓
               Generate answer from the new chunks
```

**Web search path (WEB_SEARCH):**

The agent passes the query directly to Tavily, which returns clean text snippets from live web pages — no HTML scraping required.

---

### Step 4 — Answer generation

Regardless of the retrieval path (or none if the agent decides it didn't need info from the database or the web), the final generation step works the same way. The local model receives:

1. A system prompt containing its role and any retrieved context
2. Alternating user/assistant turns from conversation history (up to the last 10 messages)
3. The current user question as the final user turn

History is passed as proper chat turns — not embedded text — so the model can reliably track what was discussed earlier. It is instructed to answer using the provided context. If the context doesn't fully cover the question, it says so rather than guessing. Follow-up questions like *"what about the one you mentioned earlier?"* are resolved naturally without the user needing to repeat themselves.

---

## The evaluation pipeline

After the agent produces an answer, `eval_pipeline.py` acts as an automated quality gate. It runs two test questions through the full system and scores the answers on two criteria:

**Faithfulness** — Did the answer stick to what the retrieved passages actually said? A score below 0.8 means the AI may have gone beyond (or contradicted) its sources.

**Answer Relevancy** — Did the answer actually address the question asked? A low score means the answer was technically grounded but missed the point.

Both metrics are scored by the same local model acting as an impartial judge. If either score falls below 0.8, the pipeline:

- Logs the failure to a `failures.csv` file (with the question, score, and reason)
- Exits with an error code — which would cause a CI/CD pipeline (like GitHub Actions) to flag the build as failed

This means quality regressions — where a code change accidentally makes answers worse — get caught automatically before they reach users.

---

## Why this approach matters


| Problem                                         | How this system solves it                                                                                        |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| AI makes things up ("hallucination")            | The AI is forced to answer from specific passages, not general training                                          |
| Hard to update AI knowledge                     | You update the database, not the model. No retraining required                                                   |
| Can't cite sources                              | Every answer is grounded in trackable chunks with source and chapter metadata                                    |
| User language ≠ technical language              | The self-correction step bridges everyday questions to textbook terminology, allowing for better database search |
| Some questions need live information            | The router sends current-events queries to web search instead of the knowledge base                              |
| No conversational memory                        | Completed turns are passed as proper chat messages, enabling reliable follow-up questions                        |
| No way to know if quality degrades with changes | The eval pipeline scores every answer automatically and flags regressions                                        |


---

## The data flow in one sentence

The user's question is classified by a router → the router picks the right source (knowledge base, web, or memory) → retrieved passages are put in the system prompt, prior conversation turns are passed as chat messages → the local model reads the full context and writes an answer → a badge in the UI tells the user where the answer came from.

---

## Quick start

**Prerequisites:** Python 3.11+, [LM Studio](https://lmstudio.ai) with a model loaded and the local server running, a [Pinecone](https://pinecone.io) Serverless index named `music-theory` (dimension `384`, metric `cosine`), a [Tavily API key](https://tavily.com).

```bash
# 1. Install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Set up environment variables
cp .env.example .env
# Required: PINECONE_API_KEY, TAVILY_API_KEY
# LM Studio (defaults shown — change if your setup differs):
#   LM_STUDIO_BASE_URL=http://localhost:1234/v1
#   LM_STUDIO_MODEL=local-model

# 3. Start LM Studio and load your model, then enable the local server
#    (LM Studio → Local Server tab → Start Server)

# 4. Add your PDF textbooks to the data/ folder, then ingest them
python3 ingest.py

# 5. Run the chat interface
streamlit run app.py

# 6. Or ask a single question directly from the terminal
python3 agent.py "How do I build a minor 7th chord on guitar?"

# 7. Run the quality gate
python3 eval_pipeline.py
```

