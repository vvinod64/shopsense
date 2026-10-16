"""Qdrant Cloud index over the policy corpus with Gemini embeddings.

Both renderings of every policy are indexed: the Markdown files (header-aware chunks) and the PDFs (page
text, whose headings the renderer flattens). Each chunk records its format. Retrieval defaults to the
Markdown chunks, and the retrieval evaluation measures both formats.
"""
import hashlib
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue, PayloadSchemaType, PointStruct, VectorParams,
)

from shopsense.config import CORPUS_DIR, EMBED_CACHE_DIR

# The embedding model is a setting: any LiteLLM embedding model (e.g. "openai/text-embedding-3-small") works.
# Vectors from different models cannot be mixed, so each non-default model gets its own collection; build it
# with `python -m shopsense.m3.rag` after changing the setting.
DEFAULT_EMBED_MODEL = "gemini-embedding-001"
EMBED_MODEL = os.getenv("SHOPSENSE_EMBED_MODEL") or DEFAULT_EMBED_MODEL
COLLECTION = ("shopsense_policies" if EMBED_MODEL == DEFAULT_EMBED_MODEL
              else "shopsense_policies__" + re.sub(r"[^a-z0-9]+", "_", EMBED_MODEL.lower()).strip("_"))
HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
CHUNK_SIZE, CHUNK_OVERLAP = 900, 120
BATCH, PAUSE = 90, 60                      # free-tier embedding limit per minute
UPSERT_BATCH = 20                          # small batches avoid Qdrant Cloud write timeouts
PDF_CID_OFFSET = 10_000                    # PDF chunk ids start here; Markdown ids are positions 0..n


# Corpus to chunks

@lru_cache(maxsize=1)
def load_chunks() -> tuple:
    """Chunk every markdown document in corpus/index.json.

    Headers are split first so each chunk stays within one policy section and keeps its section path;
    long sections are split further. A chunk's cid is its position, which is also its Qdrant point id.
    """
    index = json.loads((CORPUS_DIR / "index.json").read_text(encoding="utf-8"))
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=False)
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    chunks = []
    for doc in index:
        md_path = CORPUS_DIR / "markdown" / f"{doc['slug']}.md"
        if not md_path.exists():          # a document that only exists as a PDF (e.g. a scan): see load_pdf_chunks
            continue
        sections = size_splitter.split_documents(
            header_splitter.split_text(md_path.read_text(encoding="utf-8")))
        for i, section in enumerate(sections):
            text = section.page_content.strip()
            # Skip title-only chunks with no policy text.
            if not text or all(line.lstrip().startswith("#") for line in text.splitlines() if line.strip()):
                continue
            chunks.append({
                "cid": len(chunks),
                "text": text,
                "source": md_path.name,
                "title": doc["title"],
                "doc_type": doc["section"],      # policy | operations | compliance | category_policy
                "section": " > ".join(v for k, v in section.metadata.items() if k.startswith("h")),
                "chunk_index": i,
                "format": "markdown",
            })
    return tuple(chunks)


INGEST_WARNINGS: list = []   # documents or pages the PDF path could not read, and what was used instead
MIN_PDF_CHARS = 200          # less text than this from a whole PDF means no usable text layer


def _pdf_pages(pdf_path) -> list:
    """(page number, text) for every page with text; a page that fails to extract is skipped and logged."""
    from pypdf import PdfReader
    pages = []
    for page_no, page in enumerate(PdfReader(pdf_path).pages, start=1):
        try:
            # The PDF font has no rupee glyph: "₹2,000" is extracted as "■2,000".
            pages.append((page_no, re.sub(r"■(?=\s?\d)", "₹", page.extract_text() or "")))
        except Exception as e:
            INGEST_WARNINGS.append(f"{pdf_path.name} page {page_no}: {type(e).__name__}, page skipped")
    return pages


OCR_PROMPT = ("This is a scanned policy document. Transcribe all of its text exactly, in reading order, page by "
              "page. Start each page with a line '=== page N ==='. Keep every number, currency sign (₹, $), "
              "percentage, clause number and table row; do not summarise, translate or add anything.")


def ocr_pdf(pdf_path) -> list:
    """(page number, text) for a PDF without a text layer, read by the multimodal model (LLM_MODEL)."""
    import base64
    import litellm
    from shopsense.config import LLM_MODEL, LLM_TIMEOUT_SECONDS
    data = base64.b64encode(Path(pdf_path).read_bytes()).decode("ascii")
    resp = litellm.completion(model=LLM_MODEL, temperature=0, num_retries=3, timeout=max(LLM_TIMEOUT_SECONDS, 180),
                              messages=[{"role": "user", "content": [
                                  {"type": "text", "text": OCR_PROMPT},
                                  {"type": "file", "file": {"file_data": f"data:application/pdf;base64,{data}"}}]}])
    text = resp.choices[0].message.content or ""
    parts = re.split(r"^=+\s*page\s+(\d+)\s*=+\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if len(parts) < 3:
        return [(1, text.strip())] if text.strip() else []
    return [(int(parts[i]), parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2) if parts[i + 1].strip()]


@lru_cache(maxsize=1)
def load_pdf_chunks() -> tuple:
    """Chunk the PDF rendering of every document page by page. `source` stays the logical document name
    (category-electronics.md) so filters and citations work the same; `file` is the PDF.

    One bad file does not stop ingestion. A PDF that cannot be opened, or has no text layer (a scan), uses the
    same document's Markdown chunks when there are any (`extraction: markdown-fallback`); a scan with no
    Markdown version is read by OCR (`extraction: ocr`). If OCR also fails, the document is skipped and logged.
    """
    index = json.loads((CORPUS_DIR / "index.json").read_text(encoding="utf-8"))
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    INGEST_WARNINGS.clear()
    chunks = []
    for doc in index:
        pdf_path = CORPUS_DIR / "pdf" / f"{doc['slug']}.pdf"
        try:
            pages = _pdf_pages(pdf_path)
        except Exception as e:
            pages, reason = [], f"{type(e).__name__}"
        else:
            reason = "no text layer"
        extraction = None
        if sum(len(t.strip()) for _, t in pages) < MIN_PDF_CHARS:
            fallback = [c for c in load_chunks() if c["source"] == f"{doc['slug']}.md"]
            if fallback:
                INGEST_WARNINGS.append(f"{pdf_path.name}: {reason}, used {len(fallback)} Markdown chunks instead")
                for c in fallback:
                    chunks.append({**c, "cid": PDF_CID_OFFSET + len(chunks), "file": pdf_path.name,
                                   "format": "pdf", "extraction": "markdown-fallback"})
                continue
            try:
                pages, extraction = ocr_pdf(pdf_path), "ocr"
                INGEST_WARNINGS.append(f"{pdf_path.name}: {reason}, no Markdown version, read {len(pages)} page(s) by OCR")
            except Exception as e:
                INGEST_WARNINGS.append(f"{pdf_path.name}: {reason}, OCR failed ({type(e).__name__}), document skipped")
                continue
        for page_no, page_text in pages:
            for i, text in enumerate(splitter.split_text(page_text)):
                if text.strip():
                    chunks.append({"cid": PDF_CID_OFFSET + len(chunks), "text": text.strip(),
                                   "source": f"{doc['slug']}.md", "file": pdf_path.name, "title": doc["title"],
                                   "doc_type": doc["section"], "section": f"page {page_no}", "chunk_index": i,
                                   "format": "pdf", **({"extraction": extraction} if extraction else {})})
    return tuple(chunks)


def all_chunks() -> tuple:
    return load_chunks() + load_pdf_chunks()


@lru_cache(maxsize=1)
def _by_cid() -> dict:
    return {c["cid"]: c for c in all_chunks()}


def chunk(cid: int) -> dict:
    return _by_cid()[cid]


def corpus_fingerprint() -> str:
    """Hash of all chunk text (both formats), stored on each point to detect a stale index."""
    h = hashlib.sha1(EMBED_MODEL.encode())
    for c in all_chunks():
        h.update(c["text"].encode("utf-8"))
    return h.hexdigest()[:16]


# Embeddings

class LiteLLMEmbeddings:
    """Embeddings from any LiteLLM provider, with the embed_documents / embed_query interface used here."""

    def __init__(self, model: str):
        self.model = model

    def _embed(self, texts: list) -> list:
        import litellm
        resp = litellm.embedding(model=self.model, input=texts)
        return [d["embedding"] for d in sorted(resp.data, key=lambda d: d["index"])]

    def embed_documents(self, texts: list) -> list:
        return self._embed(texts)

    def embed_query(self, text: str) -> list:
        return self._embed([text])[0]


@lru_cache(maxsize=1)
def _embedder():
    if EMBED_MODEL == DEFAULT_EMBED_MODEL or EMBED_MODEL.removeprefix("models/").startswith("gemini-embedding"):
        # Gemini through its own client keeps the document/query task types the index was built with
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    return LiteLLMEmbeddings(EMBED_MODEL)


def _cache_path(text: str, kind: str):
    # Documents and queries use different task types, so the kind is part of the cache key.
    key = hashlib.sha1(f"{EMBED_MODEL}::{kind}::{text}".encode("utf-8")).hexdigest()
    return EMBED_CACHE_DIR / f"{key}.json"


def _cache_get(text: str, kind: str) -> Optional[list]:
    p = _cache_path(text, kind)
    return json.loads(p.read_text()) if p.exists() else None


def _cache_put(text: str, kind: str, vec: list) -> None:
    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(text, kind).write_text(json.dumps(vec))


def _with_backoff(fn):
    for attempt in range(5):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) and attempt < 4:
                time.sleep(2 ** attempt)
            else:
                raise


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed documents, using the disk cache and throttled batches for misses."""
    out = [_cache_get(t, "doc") for t in texts]
    todo = [i for i, v in enumerate(out) if v is None]
    if todo:
        print(f"  {len(todo)} chunks to embed ({len(texts) - len(todo)} from cache).")
    for b in range(0, len(todo), BATCH):
        idx = todo[b:b + BATCH]
        vecs = _with_backoff(lambda: _embedder().embed_documents([texts[i] for i in idx]))
        for i, v in zip(idx, vecs):
            out[i] = v
            _cache_put(texts[i], "doc", v)
        if b + BATCH < len(todo) and not EMBED_MODEL.startswith("ollama"):   # local models have no rate limit
            print(f"  embedded {b + len(idx)}/{len(todo)}; pausing {PAUSE}s (rate limit)")
            time.sleep(PAUSE)
    return out


def embed_query(text: str) -> list[float]:
    v = _cache_get(text, "query")
    if v is None:
        v = _with_backoff(lambda: _embedder().embed_query(text))
        _cache_put(text, "query", v)
    return v


# Qdrant Cloud

@lru_cache(maxsize=1)
def client() -> QdrantClient:
    url, key = os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY")
    if not url or not key:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in .env")
    return QdrantClient(url=url, api_key=key, timeout=60)


def _index_is_current(c: QdrantClient, n_chunks: int, fingerprint: str) -> bool:
    if not c.collection_exists(COLLECTION):
        return False
    if c.count(COLLECTION, exact=True).count != n_chunks:
        return False
    probe = c.retrieve(COLLECTION, ids=[0], with_payload=["corpus_fingerprint"])
    return bool(probe) and probe[0].payload.get("corpus_fingerprint") == fingerprint


def build_index(force_rebuild: bool = False) -> int:
    """Rebuild the collection when the corpus changed or force_rebuild is set. Returns the chunk count."""
    chunks = all_chunks()
    fingerprint = corpus_fingerprint()
    c = client()
    if not force_rebuild and _index_is_current(c, len(chunks), fingerprint):
        print(f"Collection '{COLLECTION}' is current ({len(chunks)} chunks).")
        return len(chunks)

    vecs = embed_texts([ch["text"] for ch in chunks])
    if c.collection_exists(COLLECTION):
        c.delete_collection(COLLECTION)
    c.create_collection(COLLECTION, vectors_config=VectorParams(size=len(vecs[0]), distance=Distance.COSINE))
    # Qdrant Cloud filters only on indexed payload fields.
    for field in ("doc_type", "source", "format"):
        c.create_payload_index(COLLECTION, field_name=field, field_schema=PayloadSchemaType.KEYWORD)

    points = [PointStruct(id=ch["cid"], vector=vec, payload={**ch, "corpus_fingerprint": fingerprint})
              for ch, vec in zip(chunks, vecs)]
    for i in range(0, len(points), UPSERT_BATCH):
        c.upsert(COLLECTION, points=points[i:i + UPSERT_BATCH], wait=True)
    print(f"Indexed {len(points)} chunks into Qdrant Cloud (dim={len(vecs[0])}).")
    return len(points)


def retrieve(query: str, k: int = 4, doc_type: Optional[str] = None, source: Optional[str] = None,
             fmt: Optional[str] = "markdown") -> list:
    """Dense search, optionally filtered by doc_type or source, over one format (None = both)."""
    must = [FieldCondition(key="format", match=MatchValue(value=fmt))] if fmt else []
    if doc_type:
        must.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))
    if source:
        must.append(FieldCondition(key="source", match=MatchValue(value=source)))
    points = client().query_points(
        collection_name=COLLECTION, query=embed_query(query),
        query_filter=Filter(must=must) if must else None, limit=k, with_payload=True,
    ).points
    return [{
        "cid": p.id,
        "score": round(p.score, 4),
        "source": p.payload.get("source"),
        "title": p.payload.get("title"),
        "doc_type": p.payload.get("doc_type"),
        "section": p.payload.get("section"),
        "chunk_index": p.payload.get("chunk_index"),
        "format": p.payload.get("format"),
        "text": p.payload.get("text", ""),
    } for p in points]


# Agent tool

@tool
def search_policy(query: str, k: int = 4) -> str:
    """Search Kartway policy documents for passages relevant to the query.
    Use for ANY policy question: return windows, restocking fees, warranty periods,
    shipping SLAs, refund caps, category-specific rules (electronics/apparel/groceries etc).
    Returns top k matching passages with source filenames. Cite the source in your response."""
    hits = retrieve(query, k=k)
    if not hits:
        return "No matching policy passages found."
    return json.dumps([{k_: h[k_] for k_ in ("source", "section", "score", "text")} for h in hits],
                      indent=2, ensure_ascii=False)


# Self-check and smoke test

def self_check():
    """Offline check that the chunker covers all 14 documents with stable ids."""
    chunks = load_chunks()
    index = json.loads((CORPUS_DIR / "index.json").read_text(encoding="utf-8"))
    assert {c["source"] for c in chunks} == {f"{d['slug']}.md" for d in index}, "every document must be chunked"
    assert [c["cid"] for c in chunks] == list(range(len(chunks))), "cid must equal position"
    assert max(len(c["text"]) for c in chunks) <= CHUNK_SIZE, "no chunk may exceed CHUNK_SIZE"
    assert all(not c["text"].startswith("#") or "\n" in c["text"] for c in chunks), "no header-only chunks"
    assert {c["doc_type"] for c in chunks} == {"policy", "operations", "compliance", "category_policy"}
    pdf = load_pdf_chunks()
    assert {c["source"] for c in pdf} == {c["source"] for c in chunks}, "every PDF must be chunked"
    assert len({c["cid"] for c in all_chunks()}) == len(chunks) + len(pdf), "chunk ids must be unique"
    assert max(len(c["text"]) for c in pdf) <= CHUNK_SIZE
    print(f"M3 RAG self-check: PASS - {len(chunks)} Markdown and {len(pdf)} PDF chunks from {len(index)} documents.")


def smoke_test():
    n = build_index()
    cnt = client().count(COLLECTION, exact=True).count
    pdf_hit = retrieve("What is the return window for electronics?", k=1, fmt="pdf")
    assert pdf_hit and pdf_hit[0]["format"] == "pdf", pdf_hit
    assert cnt == n, f"expected {n} points, found {cnt}"
    print(f"PASS - Qdrant Cloud holds {cnt} vectors.\n")

    queries = [
        ("What is the return window for electronics?", None),
        ("Can I return groceries?", None),
        ("What is the auto-approval refund cap?", None),
        ("How many days until a shipment is declared lost in transit?", None),
        ("What is the return window?", "category_policy"),
    ]
    for q, doc_type in queries:
        print(f"Q: {q}" + (f"   [doc_type={doc_type}]" if doc_type else ""))
        hits = retrieve(q, k=2, doc_type=doc_type)
        for h in hits:
            print(f"  [{h['cid']}] {h['score']:.3f} {h['source']} :: {h['section'][:60]}")
        if doc_type:
            assert all(h["doc_type"] == doc_type for h in hits), "filter must restrict doc_type"
        print()


if __name__ == "__main__":
    import sys
    self_check()
    if "--offline" not in sys.argv:
        smoke_test()
