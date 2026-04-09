"""
retriever.py — Query the Pinecone knowledge base with a natural language string.
"""

import os

from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone.exceptions import PineconeApiException
from sentence_transformers import SentenceTransformer

load_dotenv()

_EMBED_MODEL = "all-MiniLM-L6-v2"
_PINECONE_INDEX_HOST = "https://music-theory-j7x145p.svc.aped-4627-b74a.pinecone.io"
_TOP_K = 5

# Module-level singletons — loaded once on first call, reused on subsequent calls
_model: SentenceTransformer | None = None
_index = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_EMBED_MODEL)
    return _model


def _get_index():
    global _index
    if _index is None:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        _index = pc.Index(host=_PINECONE_INDEX_HOST)
    return _index


def search_knowledge_base(query: str) -> list[dict]:
    """
    Embed query and return the top 5 most relevant chunks from the knowledge base.

    Each result is a dict with keys: text, chapter, source, score.

    Raises:
        ValueError: if query is empty.
        RuntimeError: if Pinecone is unreachable or times out.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    query_vector = _get_model().encode(query).tolist()

    try:
        response = _get_index().query(
            vector=query_vector,
            top_k=_TOP_K,
            include_metadata=True,
        )
    except PineconeApiException as e:
        raise RuntimeError(f"Pinecone API error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Could not reach Pinecone index: {e}") from e

    return [
        {
            "text": match.metadata.get("text", ""),
            "chapter": match.metadata.get("chapter", ""),
            "source": match.metadata.get("source", ""),
            "score": round(match.score, 4),
        }
        for match in response.matches
    ]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What is a major scale?"
    results = search_knowledge_base(query)
    for i, r in enumerate(results, 1):
        print(f"\n── Result {i} (score: {r['score']}) ──")
        print(f"Source : {r['source']}")
        print(f"Chapter: {r['chapter']}")
        print(f"Text   : {r['text'][:300]}{'...' if len(r['text']) > 300 else ''}")
