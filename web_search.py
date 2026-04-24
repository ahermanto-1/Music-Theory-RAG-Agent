"""
web_search.py — Web search tool for the Music Theory RAG Agent.

Returns results in the same shape as retriever.search_knowledge_base() so that
_format_context() and _generate() in agent.py work without modification:
  [{"text": str, "source": str, "chapter": str, "score": float}, ...]

Usage:
    from web_search import web_search
    results = web_search("Ichika Nito guitar tuning")
"""

import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

_TOP_K = 5
_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TAVILY_API_KEY is not set. Add it to your .env file. "
                "Get a free key at https://tavily.com"
            )
        _client = TavilyClient(api_key=api_key)
    return _client


def web_search(query: str, top_k: int = _TOP_K) -> list[dict]:
    """
    Search the web for the given query.

    Returns up to top_k results, each a dict with keys:
        text    — extracted page content (clean text, no HTML)
        source  — URL of the result
        chapter — empty string (no chapter concept for web results)
        score   — Tavily relevance score (0–1)

    Raises:
        ValueError: if query is empty.
        RuntimeError: if TAVILY_API_KEY is missing or the API call fails.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    client = _get_client()

    try:
        response = client.search(
            query=query,
            max_results=top_k,
            search_depth="basic",
            include_answer=False,
        )
    except Exception as e:
        raise RuntimeError(f"Tavily search failed: {e}") from e

    results = []
    for r in response.get("results", []):
        text = r.get("content", "").strip()
        if not text:
            continue
        results.append({
            "text": text,
            "source": r.get("url", ""),
            "chapter": "",
            "score": round(float(r.get("score", 0.0)), 4),
        })

    return results


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "circle of fifths music theory"
    print(f"Searching: {query}\n")
    results = web_search(query)
    for i, r in enumerate(results, 1):
        print(f"── Result {i} (score: {r['score']}) ──")
        print(f"Source : {r['source']}")
        print(f"Text   : {r['text'][:300]}{'...' if len(r['text']) > 300 else ''}")
        print()
