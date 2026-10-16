"""Hybrid retrieval: dense search fused with BM25 using Reciprocal Rank Fusion.

BM25 matches exact tokens such as amounts, durations and order references; RRF scores a chunk as
the sum of 1 / (c + rank) over both lists.
"""
import json
import re
from functools import lru_cache
from typing import Optional

from langchain_core.tools import tool
from rank_bm25 import BM25Okapi

from shopsense.m3.rag import all_chunks, chunk, retrieve as dense_retrieve

RRF_C = 60


def _tokenize(text: str) -> list:
    """Lowercase word and number tokens, keeping the rupee sign attached to amounts."""
    return re.findall(r"[a-z0-9₹]+", text.lower())


@lru_cache(maxsize=None)
def _bm25(fmt: str) -> tuple:
    """BM25 over one format's chunks (the same chunks as the Qdrant index), with their cids."""
    chunks = [c for c in all_chunks() if c["format"] == fmt]
    return BM25Okapi([_tokenize(c["text"]) for c in chunks]), chunks


def _hit(cid: int, **scores) -> dict:
    c = chunk(cid)
    return {"cid": cid, "source": c["source"], "title": c["title"], "doc_type": c["doc_type"],
            "section": c["section"], "chunk_index": c["chunk_index"], "text": c["text"], **scores}


def sparse_retrieve(query: str, k: int = 10, doc_type: Optional[str] = None, fmt: str = "markdown") -> list:
    """BM25 retrieval; chunks sharing no token with the query are dropped."""
    bm25, chunks = _bm25(fmt)
    scores = bm25.get_scores(_tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hits = [_hit(chunks[i]["cid"], score=round(float(scores[i]), 3)) for i in order
            if scores[i] > 0 and (doc_type is None or chunks[i]["doc_type"] == doc_type)]
    return hits[:k]


def reciprocal_rank_fusion(ranked_lists: list, k: Optional[int] = None, c: int = RRF_C) -> list:
    """Fuse [(name, hits), ...] by RRF score, keeping each list's rank."""
    scores, ranks = {}, {}
    for name, hits in ranked_lists:
        for rank, h in enumerate(hits, start=1):
            scores[h["cid"]] = scores.get(h["cid"], 0.0) + 1.0 / (c + rank)
            ranks.setdefault(h["cid"], {})[name] = rank
    fused = [_hit(cid, rrf_score=round(s, 5), ranks=ranks[cid])
             for cid, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
    return fused[:k] if k else fused


def hybrid_retrieve(query: str, k: int = 10, pool: int = 10, doc_type: Optional[str] = None,
                    fmt: str = "markdown") -> list:
    """Dense and BM25 top-`pool` lists over one format, fused by RRF; returns the top k."""
    return reciprocal_rank_fusion([
        ("dense", dense_retrieve(query, k=pool, doc_type=doc_type, fmt=fmt)),
        ("bm25", sparse_retrieve(query, k=pool, doc_type=doc_type, fmt=fmt)),
    ], k=k)


@tool
def search_policy_hybrid(query: str, k: int = 4) -> str:
    """Search Kartway policy documents using HYBRID retrieval (dense + BM25 fused via RRF).
    Good for queries with exact numbers, amounts or codes ('₹2000 refund cap', '15 days').
    Returns top k passages with source filenames. Cite the source in your response."""
    hits = hybrid_retrieve(query, k=k)
    if not hits:
        return "No matching policy passages found."
    return json.dumps([{"source": h["source"], "section": h["section"], "rrf_score": h["rrf_score"],
                        "text": h["text"]} for h in hits], indent=2, ensure_ascii=False)


def self_check():
    """Offline check of RRF and BM25 exact-token matching."""
    a, b, c_, d = ({"cid": i} for i in range(4))
    fused = reciprocal_rank_fusion([("x", [a, b, c_]), ("y", [a, c_, d])], k=4)
    assert [h["cid"] for h in fused][0] == 0, "chunk ranked first by both lists must fuse to #1"
    assert {h["cid"] for h in fused} == {0, 1, 2, 3}, "fusion keeps every id seen"
    top = sparse_retrieve("₹2,000 automated cap", k=3)
    assert top and top[0]["source"] == "refund-authority.md", f"BM25 should hit the refund matrix: {top[:1]}"
    print("M4 hybrid self-check: PASS - RRF fuses correctly; BM25 finds the ₹2,000 cap by exact token.")


def smoke_test():
    queries = [
        "What happens if my package doesn't arrive?",
        "₹2000 refund cap",
        "lithium battery",
        "electronics 15 days",
        "how long do I have to return apparel",
    ]
    for q in queries:
        print(f"\nQ: {q}")
        for name, hits in [("dense", dense_retrieve(q, k=3)), ("bm25", sparse_retrieve(q, k=3)),
                           ("hybrid", hybrid_retrieve(q, k=3))]:
            print(f"  {name:<6} " + " | ".join(f"[{h['cid']}] {h['source']}" for h in hits))


if __name__ == "__main__":
    import sys
    self_check()
    if "--offline" not in sys.argv:
        smoke_test()
