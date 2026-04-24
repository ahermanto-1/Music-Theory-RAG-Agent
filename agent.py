"""
agent.py — Self-corrective RAG agent using Gemini and a Pinecone knowledge base.

Workflow:
  1. Route the query: RETRIEVE | WEB_SEARCH | DIRECT
  2a. RETRIEVE  → search Pinecone → grade relevance → (rewrite + re-retrieve if needed) → generate
  2b. WEB_SEARCH → search the web → generate
  2c. DIRECT    → generate from conversation history + LLM knowledge (no retrieval)
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

from retriever import search_knowledge_base
from web_search import web_search

load_dotenv()

_MODEL = os.environ.get("LM_STUDIO_MODEL", "local-model")
_client = OpenAI(
    base_url=os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
    api_key="lm-studio",  # LM Studio ignores the key but the SDK requires a value
)

HISTORY_WINDOW = 10  # number of recent messages passed as context


# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    answer: str
    chunks: list[dict] = field(default_factory=list)  # empty for DIRECT
    rewritten: str | None = None
    source: str = "retrieve"  # "retrieve" | "web_search" | "direct"
    steps: list[str] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[{i}] {chunk['source']}"
        if chunk["chapter"]:
            header += f" — {chunk['chapter']}"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)


def _format_history(history: list[dict]) -> str:
    """Format conversation history as a readable transcript for prompts."""
    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _call(prompt: str) -> str:
    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def _call_messages(messages: list[dict]) -> str:
    """Call the model with a proper multi-turn message list."""
    response = _client.chat.completions.create(
        model=_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content.strip()


# ── Router ────────────────────────────────────────────────────────────────────

def _route(query: str, history: list[dict]) -> str:
    """
    Classify the query into one of three routing intents:
      RETRIEVE   — answerable from a music theory knowledge base (books, theory texts)
      WEB_SEARCH — requires current/external information not in the knowledge base
      DIRECT     — conversational or meta (greetings, follow-ups, summaries)

    Returns the intent as a string: "RETRIEVE", "WEB_SEARCH", or "DIRECT".
    """
    history_block = ""
    if history:
        history_block = f"""
Conversation so far:
{_format_history(history)}
"""

    prompt = f"""You are a routing agent for a music theory assistant that has access to two tools:
1. A knowledge base of music theory textbooks (scales, chords, intervals, harmony, counterpoint, guitar theory, etc.)
2. Web search (for current events, specific artists, gear, recordings, or anything outside textbook scope)

{history_block}
New user query: "{query}"

Classify this query into exactly one of:
- RETRIEVE   — the knowledge base likely contains the answer (music theory concepts, techniques, notation)
- WEB_SEARCH — requires up-to-date or external information (specific artists, albums, gear, news, tabs for specific songs)
- DIRECT     — purely conversational, a follow-up referring to something just discussed, a greeting, or a summary request

Reply with only one word: RETRIEVE, WEB_SEARCH, or DIRECT."""

    result = _call(prompt).upper().strip()

    # Normalise — fall back to RETRIEVE if the model produces unexpected output
    if result not in {"RETRIEVE", "WEB_SEARCH", "DIRECT"}:
        return "RETRIEVE"
    return result


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


def _rewrite_query(original_query: str, history: list[dict]) -> str:
    """Rewrite the query to improve Pinecone retrieval, using conversation history to resolve references."""
    history_block = ""
    if history:
        history_block = f"""
Conversation so far:
{_format_history(history)}
"""

    prompt = f"""You are helping improve a search query for a music theory knowledge base.
{history_block}
The following query did not retrieve useful results:
"{original_query}"

Rewrite it to be more specific — focus on technical music theory terms, guitar fretboard positions, intervals, scales, or chord construction that would appear in a music theory textbook. If the query contains pronouns or references to earlier conversation, resolve them into explicit terms.

Reply with only the rewritten query, no explanation."""

    return _call(prompt)


def _generate(query: str, chunks: list[dict], history: list[dict], source: str) -> str:
    """Generate a final answer using proper multi-turn messages for reliable history tracking."""
    if source == "direct":
        system_content = (
            "You are a music theory tutor having a conversation with a student. "
            "Respond naturally, drawing on the conversation history where relevant. "
            "If the student is asking for a summary of what was discussed, provide one."
        )
        user_content = query
    else:
        source_label = "music theory knowledge base" if source == "retrieve" else "web search results"
        system_content = (
            f"You are a music theory tutor. Answer the user's question using the provided {source_label}. "
            "If the context does not fully cover the question, say so and answer as best you can from what is provided.\n\n"
            f"Context:\n{_format_context(chunks)}"
        )
        user_content = query

    messages: list[dict] = [{"role": "system", "content": system_content}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_content})

    return _call_messages(messages)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run(query: str, history: list[dict] | None = None) -> AgentResult:
    """
    Route → retrieve/search/direct → generate.

    Args:
        query   — the user's current message
        history — list of {role, content} dicts (full history; last 10 used)

    Returns an AgentResult with answer, chunks, rewritten query, and source label.
    """
    # Trim history to the last HISTORY_WINDOW messages
    history = (history or [])[-HISTORY_WINDOW:]

    steps: list[str] = []

    steps.append(f'Received query: "{query}"')

    # Step 1: Route
    steps.append("Routing query…")
    intent = _route(query, history)
    steps.append(f"Intent: {intent}")

    # Step 2: Retrieve or search
    if intent == "DIRECT":
        steps.append("Answering directly — no retrieval needed.")
        answer = _generate(query, [], history, source="direct")
        return AgentResult(answer=answer, source="direct", steps=steps)

    if intent == "WEB_SEARCH":
        steps.append("Searching the web via Tavily…")
        chunks = web_search(query)
        steps.append(f"Retrieved {len(chunks)} web result(s).")
        steps.append("Generating answer from web results…")
        answer = _generate(query, chunks, history, source="web_search")
        return AgentResult(answer=answer, chunks=chunks, source="web_search", steps=steps)

    # RETRIEVE path
    steps.append("Searching the knowledge base…")
    chunks = search_knowledge_base(query)
    steps.append(f"Retrieved {len(chunks)} chunk(s) from knowledge base.")

    steps.append("Grading relevance of retrieved context…")
    relevant = _grade(query, chunks)
    steps.append(f"Relevance grade: {"sufficient" if relevant else "insufficient"}.")

    rewritten = None
    if not relevant:
        steps.append("Context insufficient — rewriting query for better retrieval…")
        rewritten = _rewrite_query(query, history)
        steps.append(f'Rewritten query: "{rewritten}"')
        steps.append("Re-searching knowledge base with rewritten query…")
        chunks = search_knowledge_base(rewritten)
        steps.append(f"Retrieved {len(chunks)} chunk(s) after rewrite.")

    steps.append("Generating final answer…")
    answer = _generate(query, chunks, history, source="retrieve")
    return AgentResult(answer=answer, chunks=chunks, rewritten=rewritten, source="retrieve", steps=steps)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = input("Ask a music theory question: ").strip()

    try:
        result = run(query)
        if result.rewritten:
            print(f"\n[Query rewritten to: {result.rewritten}]")
        print(f"\nSource: {result.source}")
        print(f"\n{result.answer}")
    except Exception as e:
        print(f"Error: {e}")
