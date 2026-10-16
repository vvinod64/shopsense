"""Cross-encoder reranking of hybrid retrieval candidates."""
import json
from typing import Optional

from sentence_transformers import CrossEncoder
from langchain_core.tools import tool

from shopsense.m4.hybrid import hybrid_retrieve

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # local model, about 90 MB
_reranker = None


def reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def retrieve_and_rerank(query: str, top_k_retrieve: int = 10, top_k_final: int = 4, fmt: str = "markdown") -> list:
    """Retrieve top_k_retrieve candidates by hybrid search over one format and rerank them to top_k_final."""
    candidates = hybrid_retrieve(query, k=top_k_retrieve, fmt=fmt)
    if not candidates:
        return []
    pairs = [(query, c.get("text", "")) for c in candidates]
    scores = reranker().predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k_final]


# Agent tool

@tool
def search_policy_production(query: str, k: int = 4) -> str:
    """Production-grade policy search: HYBRID retrieval (dense + BM25) with
    CROSS-ENCODER RERANKING. Use this for the most accurate policy lookups,
    especially for high-stakes decisions. Returns top k passages with source
    filenames and rerank scores. Always cite the source in your response."""
    hits = retrieve_and_rerank(query, top_k_retrieve=10, top_k_final=k)
    if not hits:
        return "No matching policy passages found."
    simplified = []
    for h in hits:
        simplified.append({
            "source": h.get("source"),
            "rerank_score": round(h.get("rerank_score", 0.0), 3),
            "rrf_score": h.get("rrf_score"),
            "text": h.get("text", "")[:600],
        })
    return json.dumps(simplified, indent=2, default=str)


# Smoke test

def smoke_test():
    print("Loading reranker (first run downloads ~90MB)...")
    reranker()
    print("Reranker ready.\n")

    queries = [
        "What is the refund cap for the automated system?",
        "How many days to return electronics?",
        "What happens if my grocery order is spoiled?",
        "Furniture restocking fee for change of mind",
        "Lithium battery damaged, what do I do?",
    ]

    for q in queries:
        print(f"\n{'='*70}\nQ: {q}\n{'='*70}")
        hits = retrieve_and_rerank(q, top_k_retrieve=10, top_k_final=3)
        print("\n--- TOP 3 (rerank order) ---")
        for i, h in enumerate(hits, 1):
            print(f"\n  #{i} [rerank={round(h.get('rerank_score', 0.0), 3)}] [rrf={h.get('rrf_score')}] {h.get('source')}")
            print(f"      {h.get('text', '')[:200]}...")


if __name__ == "__main__":
    smoke_test()