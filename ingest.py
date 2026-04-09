"""
ingest.py — ETL pipeline: PDFs → Markdown chunks → Pinecone embeddings

Reads PDFs from PDF_DIR, parses them with docling, splits into token-bounded
chunks (with overlap) using the embedding model's own tokenizer, embeds with
sentence-transformers, and upserts to a Pinecone Serverless index.
"""

import hashlib
import os
import re
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()

PDF_DIR = Path("data")
PINECONE_INDEX_HOST = "https://music-theory-j7x145p.svc.aped-4627-b74a.pinecone.io"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 400                          # model hard limit is 512; leave headroom for special tokens
OVERLAP_TOKENS = int(MAX_TOKENS * 0.15)  # 60 tokens (~15%)
UPSERT_BATCH = 100

# ── PDF → Markdown ────────────────────────────────────────────────────────────

def convert_pdf(pdf_path: Path) -> str:
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown()


# ── Chunking ──────────────────────────────────────────────────────────────────

def _make_chunk(text: str, chapter: str | None, source: str) -> dict:
    chunk_id = hashlib.md5(f"{source}:{text}".encode()).hexdigest()
    return {"id": chunk_id, "text": text, "chapter": chapter or "", "source": source}


def _token_chunks(text: str, tokenizer, max_tokens: int, overlap: int) -> list[str]:
    """
    Sliding-window split of text into pieces of up to max_tokens tokens,
    each window overlapping the previous by overlap tokens.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if not token_ids:
        return []

    pieces: list[str] = []
    start = 0
    step = max_tokens - overlap

    while start < len(token_ids):
        window = token_ids[start : start + max_tokens]
        pieces.append(tokenizer.decode(window, skip_special_tokens=True))
        if start + max_tokens >= len(token_ids):
            break
        start += step

    return pieces


def chunk_markdown(
    markdown: str,
    source: str,
    tokenizer,
    max_tokens: int = MAX_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> list[dict]:
    """
    Parse markdown into token-bounded chunks with overlap.
    - H1 headings set the chapter metadata; they are not included in chunk text.
    - Text under each chapter is tokenized and split with a sliding window.
    - Chapter boundaries are never crossed by a chunk.
    """
    chunks: list[dict] = []
    current_chapter: str | None = None
    section_blocks: list[str] = []

    def flush_section():
        if not section_blocks:
            return
        section_text = "\n\n".join(section_blocks)
        for piece in _token_chunks(section_text, tokenizer, max_tokens, overlap):
            if piece.strip():
                chunks.append(_make_chunk(piece, current_chapter, source))

    for block in re.split(r"\n{2,}", markdown.strip()):
        block = block.strip()
        if not block:
            continue

        if re.match(r"^# ", block):
            flush_section()
            section_blocks = []
            current_chapter = block[2:].strip()
            continue

        section_blocks.append(block)

    flush_section()
    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> list[dict]:
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


# ── Pinecone upsert ───────────────────────────────────────────────────────────

def _batched(items: list, n: int) -> Iterator[list]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def upsert(chunks: list[dict], index) -> None:
    vectors = [
        {
            "id": c["id"],
            "values": c["embedding"],
            "metadata": {
                "text": c["text"],
                "chapter": c["chapter"],
                "source": c["source"],
            },
        }
        for c in chunks
    ]
    for batch in _batched(vectors, UPSERT_BATCH):
        index.upsert(vectors=batch)
    print(f"  Upserted {len(vectors)} vectors")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.environ["PINECONE_API_KEY"]
    # PINECONE_ENVIRONMENT is loaded from .env but Serverless uses the host URL directly
    _ = os.getenv("PINECONE_ENVIRONMENT")

    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {PDF_DIR.resolve()}")
    print(f"Found {len(pdf_paths)} PDF(s) in {PDF_DIR.resolve()}")

    print(f"Loading tokenizer and embedding model: {EMBED_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    embed_model = SentenceTransformer(EMBED_MODEL)

    print("Connecting to Pinecone ...")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(host=PINECONE_INDEX_HOST)

    all_chunks: list[dict] = []

    for pdf_path in pdf_paths:
        print(f"\nParsing: {pdf_path.name}")
        markdown = convert_pdf(pdf_path)
        chunks = chunk_markdown(markdown, source=pdf_path.name, tokenizer=tokenizer)
        print(f"  {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nEmbedding {len(all_chunks)} total chunks ...")
    all_chunks = embed_chunks(all_chunks, embed_model)

    print("Upserting to Pinecone ...")
    upsert(all_chunks, index)

    print("\nDone.")


if __name__ == "__main__":
    main()
