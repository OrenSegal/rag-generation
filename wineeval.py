"""Retrieval eval over top_rated_wines.csv: BM25 vs a dense model vs RRF hybrid.

Run: uv run python wineeval.py   (writes results.md)
"""
import csv
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
MODEL = "all-MiniLM-L6-v2"  # the model the course notebooks use
GENERIC = {"wine", "red", "white"}  # too common to make a query "literal"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_corpus(path: Path = ROOT / "top_rated_wines.csv") -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))  # doc id = row index; names are not unique


def doc_text(row: dict[str, str]) -> str:
    return f"{row['name']} | {row['region']} | {row['variety']} | {row['notes']}"


def relevant(rows: list[dict[str, str]], filters: dict[str, str]) -> set[int]:
    """Ids of rows where every filter value is a case-insensitive substring of that field."""
    return {i for i, r in enumerate(rows)
            if all(v.lower() in r[k].lower() for k, v in filters.items())}


def query_kind(q: str, filters: dict[str, str]) -> str:
    """'literal' if the query shares a non-generic token with a label value, else 'paraphrase'."""
    label_tokens = {t for v in filters.values() for t in tokenize(v)} - GENERIC
    return "literal" if label_tokens & set(tokenize(q)) else "paraphrase"


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = np.array([sum(tf.values()) for tf in self.tfs], dtype=float)
        self.avgdl = self.lens.mean()
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.tfs))
        norm = self.k1 * (1 - self.b + self.b * self.lens / self.avgdl)
        for t in set(tokenize(query)):
            if t not in self.idf:
                continue
            tf = np.array([d.get(t, 0) for d in self.tfs], dtype=float)
            s += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
        return s


def rank(scores: Sequence[float] | np.ndarray) -> list[int]:
    # stable sort so ties break by doc id, deterministically
    return list(np.argsort(-np.asarray(scores), kind="stable"))


def rrf(rankings: Iterable[Sequence[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank), rank from 1."""
    fused = Counter()
    for r in rankings:
        for pos, d in enumerate(r, start=1):
            fused[d] += 1.0 / (k + pos)
    return sorted(fused, key=lambda d: (-fused[d], d))


def recall_at_k(ranking: Sequence[int], rel: set[int], k: int) -> float:
    """Capped recall: hits in top k / min(k, |rel|), so 1.0 is reachable for large label sets."""
    return len(set(ranking[:k]) & rel) / min(k, len(rel))


def reciprocal_rank(ranking: Sequence[int], rel: set[int]) -> float:
    for pos, d in enumerate(ranking, start=1):
        if d in rel:
            return 1.0 / pos
    return 0.0


def bootstrap_ci(values: Sequence[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile 95% CI of the mean, resampling queries."""
    v = np.asarray(values, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(v), (n, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def dense_rankings(docs: list[str], queries: list[str]) -> list[list[int]] | None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    m = SentenceTransformer(MODEL)
    D = m.encode(docs, normalize_embeddings=True, batch_size=64)
    Q = m.encode(queries, normalize_embeddings=True)
    return [rank(D @ q) for q in Q]  # exhaustive cosine, same as an in-memory Qdrant search


METRICS = {"recall@5": lambda r, rel: recall_at_k(r, rel, 5),
           "recall@10": lambda r, rel: recall_at_k(r, rel, 10),
           "MRR": reciprocal_rank}


def fmt(vals: Sequence[float]) -> str:
    lo, hi = bootstrap_ci(vals)
    return f"{np.mean(vals):.2f} ({lo:.2f} to {hi:.2f})"


def main() -> None:
    rows = load_corpus()
    docs = [doc_text(r) for r in rows]
    qs = json.loads((ROOT / "queries.json").read_text())
    rels = [relevant(rows, q["filters"]) for q in qs]
    empty = [q["q"] for q, r in zip(qs, rels) if not r]
    if empty:
        sys.exit(f"queries.json: these queries match no document, fix their filters: {empty}")
    kinds = [query_kind(q["q"], q["filters"]) for q in qs]
    texts = [q["q"] for q in qs]

    bm25 = BM25(docs)
    runs = {"BM25": [rank(bm25.scores(t)) for t in texts]}
    dense = dense_rankings(docs, texts)
    if dense is None:
        print("dense: sentence-transformers not installed, not run", file=sys.stderr)
    else:
        runs[f"dense ({MODEL})"] = dense
        runs["hybrid (RRF, k=60)"] = [rrf([b, d]) for b, d in zip(runs["BM25"], dense)]

    scores = {name: {m: [f(r, rel) for r, rel in zip(rs, rels)] for m, f in METRICS.items()}
              for name, rs in runs.items()}
    sizes = sorted(len(r) for r in rels)
    out = [f"{len(qs)} queries ({kinds.count('literal')} literal, {kinds.count('paraphrase')} paraphrase), "
           f"{len(docs)} docs, relevant per query min {sizes[0]} / median {sizes[len(sizes) // 2]} / max {sizes[-1]}. "
           "Mean with bootstrap 95% CI over queries.", ""]
    for subset in ["all", "literal", "paraphrase"]:
        keep = [i for i, k in enumerate(kinds) if subset == "all" or k == subset]
        out += [f"**{subset}** (n={len(keep)})", "", "| retriever | recall@5 | recall@10 | MRR |", "|---|---|---|---|"]
        for name, s in scores.items():
            out.append(f"| {name} | " + " | ".join(fmt([s[m][i] for i in keep]) for m in METRICS) + " |")
        for name, s in list(scores.items())[1:]:  # paired difference: same queries, so CIs are tighter
            out.append(f"| {name} minus BM25 | " + " | ".join(
                fmt([s[m][i] - scores["BM25"][m][i] for i in keep]) for m in METRICS) + " |")
        out.append("")
    (ROOT / "results.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
