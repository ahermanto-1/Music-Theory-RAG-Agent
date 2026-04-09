"""
agent.py — Self-corrective RAG agent using Gemini and a Pinecone knowledge base.

Workflow:
  1. Retrieve top-5 chunks for the user query.
  2. Grade relevance with Gemini (YES / NO).
  3a. YES → generate answer from retrieved context.
  3b. NO  → rewrite query with Gemini, retrieve again, then generate.
"""

import os

from dotenv import load_dotenv
from google import genai

from retriever import search_knowledge_base

load_dotenv()

_MODEL = "gemma-3-27b-it"
_gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[{i}] {chunk['source']}"
        if chunk["chapter"]:
            header += f" — {chunk['chapter']}"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)


def _call(prompt: str) -> str:
    response = _gemini.models.generate_content(model=_MODEL, contents=prompt)
    return response.text.strip()


# ── Core steps ────────────────────────────────────────────────────────────────

def _grade(query: str, chunks: list[dict]) -> bool:
    """Ask Gemini whether the retrieved context is sufficient to answer the query."""
    prompt = f"""You are evaluating whether retrieved context is sufficient to answer a music theory question.

User query: {query}

Retrieved context:
{_format_context(chunks)}

Does this retrieved context contain the specific musical steps or guitar fretboard logic to answer the user query? Reply exactly with YES or NO."""

    answer = _call(prompt).upper()
    return answer.startswith("YES")


def _rewrite_query(original_query: str) -> str:
    """Ask Gemini to rewrite the query to be more specific for better retrieval."""
    prompt = f"""You are helping improve a search query for a music theory knowledge base.

The following query did not retrieve useful results:
"{original_query}"

Rewrite it to be more specific — focus on technical music theory terms, guitar fretboard positions, intervals, scales, or chord construction that would appear in a music theory textbook.

Reply with only the rewritten query, no explanation."""

    return _call(prompt)


def _generate(query: str, chunks: list[dict]) -> str:
    """Generate a final answer grounded in the retrieved context."""
    prompt = f"""You are a music theory tutor. Answer the user's question using only the provided context.
If the context does not fully cover the question, say so and answer as best you can from what is provided.

User question: {query}

Context:
{_format_context(chunks)}

Answer:"""

    return _call(prompt)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run(query: str) -> tuple[str, list[dict], str | None]:
    """
    Self-corrective RAG workflow:
      - Retrieve → Grade → (rewrite + re-retrieve if needed) → Generate.

    Returns:
        answer        — final generated answer
        chunks        — the chunks used for generation
        rewritten     — the rewritten query if correction occurred, else None
    """
    print(f"\nQuery: {query}")

    # Step 1: initial retrieval
    print("Retrieving context ...")
    chunks = search_knowledge_base(query)

    # Step 2: grade relevance
    print("Grading relevance ...")
    relevant = _grade(query, chunks)

    rewritten = None
    if not relevant:
        # Step 3b: rewrite and retrieve once more
        rewritten = _rewrite_query(query)
        print(f"Context insufficient — rewriting query:\n  {rewritten}")
        chunks = search_knowledge_base(rewritten)

    # Step 3a / final: generate answer
    print("Generating answer ...")
    return _generate(query, chunks), chunks, rewritten


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = input("Ask a music theory question: ").strip()

    try:
        answer, chunks, rewritten = run(query)
        if rewritten:
            print(f"\n[Query rewritten to: {rewritten}]")
        print(f"\n{answer}")
    except Exception as e:
        print(f"Error: {e}")
