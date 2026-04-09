# Music Theory RAG Agent
### A plain-English guide for Product Managers

---

## What does this do?

This is a question-answering system trained on music theory textbooks. You ask it a question in plain English — like *"How do I play a G major scale on guitar?"* — and it finds the most relevant passages from the books, then uses an AI to write you a clear answer grounded in those passages.

The technical name for this approach is **RAG** — Retrieval-Augmented Generation. Rather than relying on an AI's general training data (which can be vague or wrong), RAG forces the AI to answer from a specific, trusted knowledge base. Think of it like giving a lawyer the exact contract to reference before they advise you, rather than asking them to go from memory.

---

## The three files that make it work

| File | Plain-English job |
|---|---|
| `ingest.py` | One-time setup: reads the PDF textbooks and stores them in a searchable database |
| `retriever.py` | Search function: finds the most relevant passages for a given question |
| `agent.py` | The coordinator: runs the full workflow end-to-end |
| `eval_pipeline.py` | Quality gate: automatically checks whether the answers are good |

---

## How it works, step by step

### Step 1 — Ingestion (one-time setup)

Before the system can answer questions, it needs to read and index the textbooks. This happens once, before the system is used.

```
PDF textbooks
     ↓
Parse into text (like copy-pasting a PDF)
     ↓
Split into small passages ("chunks") of ~400 words each,
with slight overlap so nothing gets cut off mid-thought
     ↓
Convert each chunk into a "fingerprint" (a list of numbers
that represents its meaning — called an embedding)
     ↓
Store all fingerprints + their original text in Pinecone
(a database built for this kind of search)
```

The database is now ready to be searched. This step doesn't need to be repeated unless the textbooks change.

---

### Step 2 — Retrieval

When a user asks a question, the system needs to find the most relevant passages. It does this with pure maths — no AI involved at this stage.

```
User's question
     ↓
Convert question into its own fingerprint (same method as the chunks)
     ↓
Compare the question's fingerprint against all stored chunk fingerprints
     ↓
Return the top 5 closest matches
```

This is similar to how Spotify finds songs that "sound like" a song you like — it's matching on meaning, not just keywords.

---

### Step 3 — The self-correcting agent

This is where the AI enters. The coordinator (`agent.py`) runs the question through a quality check before generating an answer.

```
User's question
     ↓
Retrieve top 5 chunks (Step 2 above)
     ↓
Ask Gemini: "Are these chunks actually useful for answering the question?"
     ↓
     ├── YES → go straight to generating the answer
     │
     └── NO  → ask Gemini to rewrite the question using more
               technical language (e.g. "G major scale" → "major
               scale interval pattern on guitar fretboard")
                    ↓
               Retrieve again with the rewritten question
                    ↓
               Generate answer from the new chunks
```

**Why the self-correction step?** Users ask questions in everyday language. Textbooks use technical terms. A question like *"why does this chord sound sad?"* might not match passages that talk about *"minor third intervals"* — even though they're about the same thing. The rewrite bridges that gap.

The AI makes **at most 3 calls** per question: one to grade, one optional rewrite, one to generate.

---

### Step 4 — Answer generation

The AI receives:
1. The user's original question
2. The retrieved passages (formatted as numbered, labelled text)

It's instructed to answer using only those passages. If the passages don't fully cover the question, it says so rather than guessing.

---

## The evaluation pipeline

After the agent produces an answer, `eval_pipeline.py` acts as an automated quality gate. It runs two test questions through the full system and scores the answers on two criteria:

**Faithfulness** — Did the answer stick to what the retrieved passages actually said? A score below 0.8 means the AI may have gone beyond (or contradicted) its sources.

**Answer Relevancy** — Did the answer actually address the question asked? A low score means the answer was technically grounded but missed the point.

Both metrics are scored by the same Gemini model acting as an impartial judge. If either score falls below 0.8, the pipeline:
- Logs the failure to a `failures.csv` file (with the question, score, and reason)
- Exits with an error code — which would cause a CI/CD pipeline (like GitHub Actions) to flag the build as failed

This means quality regressions — where a code change accidentally makes answers worse — get caught automatically before they reach users.

---

## Why this approach matters (PM takeaways)

| Problem | How RAG solves it |
|---|---|
| AI makes things up ("hallucination") | The AI is forced to answer from specific passages, not general training |
| Hard to update AI knowledge | You update the database, not the model — no retraining required |
| Can't cite sources | Every answer is grounded in trackable chunks with source + chapter metadata |
| User language ≠ technical language | The self-correction step bridges everyday questions to textbook terminology |
| No way to know if quality degrades | The eval pipeline scores every answer automatically and flags regressions |

---

## The data flow in one sentence

The user's question becomes a fingerprint → the fingerprint finds matching passages → the passages get pasted into a prompt → Gemini reads the prompt and writes an answer → a second Gemini call scores whether the answer was faithful and relevant.

Nothing is saved between steps. All data lives in memory while the script runs, then disappears. The only permanent storage is the Pinecone database (the indexed textbooks) and the `failures.csv` log.

---

## Quick start (for developers)

**Prerequisites:** Python 3.11+, a [Pinecone](https://pinecone.io) Serverless index named `music-theory` (dimension `384`, metric `cosine`), a [Gemini API key](https://aistudio.google.com/app/apikey).

```bash
# 1. Install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Set up environment variables
cp .env.example .env   # then fill in your API keys

# 3. Add your PDF textbooks to the data/ folder, then ingest them
python3 ingest.py

# 4. Run the chat interface
streamlit run app.py

# 5. Or ask a question directly from the terminal
python3 agent.py "How do I build a minor 7th chord on guitar?"

# 6. Run the quality gate
python3 eval_pipeline.py
```
