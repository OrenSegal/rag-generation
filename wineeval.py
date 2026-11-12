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


# Labels are filters on name, region and variety, and doc_text() indexes those fields.
# These two indexes show how much of each score comes from them.
ABLATIONS = {
    "notes only": lambda row: row["notes"],
    "metadata only (name | region | variety)": lambda row: f"{row['name']} | {row['region']} | {row['variety']}",
}


Filters = dict[str, str | list[str]]


def alternatives(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else value


def relevant(rows: list[dict[str, str]], filters: Filters) -> set[int]:
    """Ids of rows where every filter matches its field: a case-insensitive substring, or any one of a list."""
    return {i for i, r in enumerate(rows)
            if all(any(v.lower() in r[k].lower() for v in alternatives(vs)) for k, vs in filters.items())}


def independent_groups(rels: list[set[int]]) -> int:
    """Number of groups of queries after joining any two whose relevant sets share a row."""
    groups: list[set[int]] = []
    for rel in rels:
        merged = set(rel)
        rest = []
        for g in groups:
            if g & merged:
                merged |= g
            else:
                rest.append(g)
        groups = rest + [merged]
    return len(groups)


def query_kind(q: str, filters: Filters) -> str:
    """'literal' if the query shares a non-generic token with a label value, else 'paraphrase'."""
    label_tokens = {t for vs in filters.values() for v in alternatives(vs) for t in tokenize(v)} - GENERIC
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


def sign_flip_p(diffs: Sequence[float], n: int = 20000, seed: int = 0) -> float:
    """Two-sided paired randomization test of mean(diffs) = 0: flip each query's sign at random."""
    d = np.asarray(diffs, dtype=float)
    flips = np.random.default_rng(seed).choice([-1.0, 1.0], (n, len(d)))
    return float((1 + (np.abs(flips @ d) >= abs(d.sum()) - 1e-12).sum()) / (n + 1))


def holm(ps: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the order given."""
    adjusted, running = [0.0] * len(ps), 0.0
    for step, i in enumerate(sorted(range(len(ps)), key=lambda i: ps[i])):
        running = max(running, min(1.0, ps[i] * (len(ps) - step)))
        adjusted[i] = running
    return adjusted


def mde(diffs: Sequence[float]) -> float:
    """Smallest true mean difference a paired test detects with 80% power at two-sided 5%, normal approximation."""
    return float((1.96 + 0.84) * np.std(diffs, ddof=1) / math.sqrt(len(diffs)))


def dense_rankings(docs: list[str], queries: list[str]) -> list[list[int]] | None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    m = SentenceTransformer(MODEL)
    D = m.encode(docs, normalize_embeddings=True, batch_size=64)
    Q = m.encode(queries, normalize_embeddings=True)
    return [rank(D @ q) for q in Q]  # exhaustive cosine, same as an in-memory Qdrant search


# "capped recall" and not "recall": for a label with at least k rows it is precision@k
METRICS = {"capped recall@5": lambda r, rel: recall_at_k(r, rel, 5),
           "capped recall@10": lambda r, rel: recall_at_k(r, rel, 10),
           "MRR": reciprocal_rank}


def fmt(vals: Sequence[float]) -> str:
    lo, hi = bootstrap_ci(vals)
    return f"{np.mean(vals):.2f} ({lo:.2f} to {hi:.2f})"


Scores = dict[str, dict[str, list[float]]]


def table(title: str, scores: Scores, keep: list[int]) -> list[str]:
    out = [f"**{title}** (n={len(keep)})", "", "| retriever | " + " | ".join(METRICS) + " |", "|---|---|---|---|"]
    for name, s in scores.items():
        out.append(f"| {name} | " + " | ".join(fmt([s[m][i] for i in keep]) for m in METRICS) + " |")
    for name, s in list(scores.items())[1:]:  # paired difference: same queries, so CIs are tighter
        out.append(f"| {name} minus BM25 | " + " | ".join(
            fmt([s[m][i] - scores["BM25"][m][i] for i in keep]) for m in METRICS) + " |")
    return out + [""]


def render(header: str, kinds: list[str], scores: Scores, ablations: dict[str, Scores]) -> str:
    """The text of results.md, from per-query scores alone (scores.json holds exactly these arguments)."""
    out = [header, ""]
    for subset in ["all", "literal", "paraphrase"]:
        out += table(subset, scores, [i for i, k in enumerate(kinds) if subset == "all" or k == subset])
    for index, s in ablations.items():
        out += table(f"all, index = {index}", s, list(range(len(kinds))))
    return "\n".join(out + corrected(kinds, scores))


def corrected(kinds: list[str], scores: Scores) -> list[str]:
    """Every 'minus BM25' cell of the three main tables, tested and Holm-adjusted as one family."""
    cells = []
    for subset in ["all", "literal", "paraphrase"]:
        keep = [i for i, k in enumerate(kinds) if subset == "all" or k == subset]
        for name, s in list(scores.items())[1:]:
            for m in METRICS:
                cells.append((subset, name, m, [s[m][i] - scores["BM25"][m][i] for i in keep]))
    if not cells:
        return []
    ps = [sign_flip_p(d) for *_, d in cells]
    out = [f"**paired differences from BM25, corrected for {len(cells)} comparisons**", "",
           "Sign-flip randomization test (20,000 flips, seed 0), then Holm over all rows. "
           "MDE: smallest true difference detectable with 80% power at two-sided 5% (normal approximation).", "",
           "| queries | comparison | metric | mean difference | p | Holm p | MDE |", "|---|---|---|---|---|---|---|"]
    for (subset, name, m, d), p, hp in zip(cells, ps, holm(ps)):
        shown = "<0.0001" if p < 0.0001 else f"{p:.4f}"
        out.append(f"| {subset} (n={len(d)}) | {name} minus BM25 | {m} | {np.mean(d):+.2f} | {shown} | {hp:.3f} | {mde(d):.2f} |")
    return out + [""]


def evaluate(docs: list[str], texts: list[str], rels: list[set[int]]) -> tuple[Scores, bool]:
    """Per-query scores of every retriever over one index, and whether the dense model ran."""
    bm25 = BM25(docs)
    runs = {"BM25": [rank(bm25.scores(t)) for t in texts]}
    dense = dense_rankings(docs, texts)
    if dense is not None:
        runs[f"dense ({MODEL})"] = dense
        runs["hybrid (RRF, k=60)"] = [rrf([b, d]) for b, d in zip(runs["BM25"], dense)]
    return ({name: {m: [f(r, rel) for r, rel in zip(rs, rels)] for m, f in METRICS.items()}
             for name, rs in runs.items()}, dense is not None)


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

    scores, ran_dense = evaluate(docs, texts, rels)
    if not ran_dense:
        print("dense: sentence-transformers not installed, not run", file=sys.stderr)
    ablations = {name: evaluate([text(r) for r in rows], texts, rels)[0] for name, text in ABLATIONS.items()}
    sizes = sorted(len(r) for r in rels)
    header = (f"{len(qs)} queries ({kinds.count('literal')} literal, {kinds.count('paraphrase')} paraphrase), "
              f"{len(docs)} docs, relevant per query min {sizes[0]} / median {sizes[len(sizes) // 2]} / max {sizes[-1]}. "
              f"{independent_groups(rels)} independent groups of queries once queries whose relevant sets share a row are joined. "
              "Mean with bootstrap 95% CI over queries. "
              "capped recall@k = relevant rows in the top k / min(k, relevant rows); when a query has at least k relevant rows "
              f"this is precision@k, which is the case for {sum(n >= 5 for n in sizes)} queries at k=5 "
              f"and {sum(n >= 10 for n in sizes)} at k=10. MRR uses the full ranking.")
    if ran_dense:  # a BM25-only run must not overwrite the committed dense scores
        (ROOT / "scores.json").write_text(json.dumps(
            {"header": header, "queries": texts, "kinds": kinds, "scores": scores, "ablations": ablations}, indent=1) + "\n")
    text = render(header, kinds, scores, ablations)
    (ROOT / "results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
