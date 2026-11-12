# Retrieval eval over a wine dataset

[![test](https://github.com/OrenSegal/rag-generation/actions/workflows/test.yml/badge.svg)](https://github.com/OrenSegal/rag-generation/actions/workflows/test.yml)

Scope: retrieval evaluation only. Nothing is generated.

A small, labelled retrieval eval: 55 queries over 1,365 wine reviews, scoring
BM25, one dense embedding model (`all-MiniLM-L6-v2`, run locally) and a
reciprocal-rank-fusion hybrid on recall@5, recall@10 and MRR, with bootstrap
95% intervals over queries. It grew out of the RAG lab from Coursera's
[Introduction to Retrieval Augmented Generation](https://www.coursera.org/learn/introduction-to-rag)
([course notebooks](#course-notebooks) below); the lab showed a search that
looked right, and this measures how often it is.

## Results

Each document is `name | region | variety | notes`. Mean over queries, bootstrap
95% CI in brackets. The "minus BM25" rows are paired differences on the same
queries. Full per-stratum tables: [`results.md`](./results.md).

| retriever, all 55 queries | recall@5 | recall@10 | MRR |
|---|---|---|---|
| BM25 | 0.66 (0.55 to 0.77) | 0.66 (0.55 to 0.77) | 0.71 (0.59 to 0.82) |
| dense (all-MiniLM-L6-v2) | 0.58 (0.49 to 0.69) | 0.56 (0.47 to 0.65) | 0.74 (0.64 to 0.85) |
| hybrid (RRF, k=60) | 0.68 (0.59 to 0.78) | 0.67 (0.58 to 0.77) | 0.81 (0.71 to 0.90) |
| dense minus BM25 | -0.08 (-0.16 to 0.00) | -0.10 (-0.19 to -0.01) | 0.03 (-0.04 to 0.11) |
| hybrid minus BM25 | 0.02 (-0.03 to 0.08) | 0.01 (-0.04 to 0.07) | 0.10 (0.02 to 0.18) |

| recall@5 / MRR by query kind | BM25 | dense | hybrid |
|---|---|---|---|
| literal (n=36): query names the label | 0.91 / 0.95 | 0.73 / 0.91 | 0.87 / 0.95 |
| paraphrase (n=19): it doesn't | 0.18 / 0.24 | 0.31 / 0.42 | 0.33 / 0.55 |

What it shows:

- **The answer depends on the query mix.** When the query contains the label
  term ("Pomerol", "Barossa Valley Shiraz"), BM25 has the best recall and the dense model
  is clearly worse on recall (-0.18 recall@5, CI -0.26 to -0.11). When it
  doesn't ("Tuscan red wine", "bubbly for a celebration"), BM25 mostly fails
  and both dense and hybrid do better, though with 19 queries the dense
  recall@5 gain has an interval that includes zero.
- **Hybrid is the safe default here, not a clear winner.** Over all queries it
  ties BM25 on recall (both intervals include zero) and improves MRR by 0.10
  (0.02 to 0.18). On literal queries it gives up a little recall to BM25
  (-0.04, -0.09 to -0.01).
- **The "all" row is a property of this query set, not of the retrievers.**
  It is about two thirds literal, so it leans toward BM25. A different mix
  would move it.
- This agrees in direction with the retrieval benchmark in
  [rootcause](https://github.com/OrenSegal/rootcause), where BM25 beat a
  transformer model on queries full of exact tokens; it is a different corpus
  and labels, so the numbers aren't comparable.

## How the labels were made, and what they favor

- **Queries** (`queries.json`) were written for this eval, drafted with an AI
  assistant and checked against the data. Each has a filter over the CSV's
  metadata, e.g. `{"region": "Pauillac", "variety": "Red Wine"}`.
- **Relevant** = every row where each filter value is a case-insensitive
  substring of that field (`relevant()` in `wineeval.py`); a filter value can
  be a list, which matches any one of its entries. Labels never come
  from the tasting notes. Sets range from 3 to 91 rows (median 14).
- **Query kind is computed, not hand-assigned:** a query is *literal* if it
  shares a token with a label value (ignoring "wine", "red", "white"),
  otherwise *paraphrase*.
- **Bias.** Region and name are part of the indexed text, so literal queries
  are close to string matching, which favors BM25 by construction. Paraphrase
  queries can still share words with relevant documents ("Willamette" for an
  Oregon label). Labels are only as good as the metadata: a Shiraz whose name
  doesn't say Shiraz counts as not relevant (the Grenache label has 10 rows
  named Grenache; 42 more rows mention Grenache only in their notes and count
  as not relevant), and the Portugal label includes Madeira. `variety` is coarse (Red/White/Sparkling/Collectible): Red and White
  are only used together with another filter, and Sparkling & Champagne (35
  rows) is the one label used on its own. Three label sets are queried twice, once
  literally and once as a paraphrase, and others overlap or nest (Rioja and
  Priorat inside "Spanish red", Barolo inside the nebbiolo-villages query). Joining every pair of
  queries whose relevant sets share a row leaves 34 independent groups, not
  55, so the intervals are somewhat narrower than they should be.
- **Metrics.** recall@k is capped: hits in the top k divided by
  min(k, |relevant|), so 1.0 is reachable when a label has 70 rows. MRR uses the
  full ranking. Intervals are percentile bootstraps over queries (2,000
  resamples, fixed seed).
- **Not tuned.** BM25 uses k1=1.5, b=0.75; RRF uses k=60; one embedding model,
  no reranker. Latency was not measured.

## Limitations

- **Small n, wide intervals.** 55 queries over 52 distinct label sets (3 sets
  are queried twice), in 34 independent groups. The paraphrase stratum has 19 queries; at that size
  dense minus BM25 recall@5 of +0.13 has an interval that includes zero
  (-0.02 to 0.27), while hybrid minus BM25 (+0.15, 0.04 to 0.27) clears zero.
  "No difference" in this eval means "not detectable at
  this n", not "equal".
- **Author-built queries.** I wrote the queries (drafted with an AI assistant)
  and chose the label filters, knowing the corpus. Nobody else labelled
  anything, and no query came from real users.
- **Many intervals, no correction.** The tables show 45 intervals (3 strata,
  5 rows, 3 metrics) at 95% each, with no multiple-comparison adjustment.
  Read single intervals that barely clear zero (for example literal hybrid
  minus BM25 recall@5, -0.09 to -0.01) as weak evidence.
- **The strata are a post-hoc lens.** The literal/paraphrase rule was written
  together with the queries, not pre-registered.
- **What it does not show:** anything about generation or answer quality,
  latency, other embedding models, rerankers, tuned BM25 or RRF parameters, or
  any corpus other than these 1,365 reviews.

Design choices and their alternatives are in [`docs/decisions.md`](./docs/decisions.md).

## Quickstart

```bash
uv sync --extra dense          # numpy + sentence-transformers (pulls in torch)
uv run python wineeval.py      # prints the tables and writes results.md
uv run pytest                  # metric, fusion and label tests; no network
uvx ruff@0.9.9 check .         # lint, same as CI
```

The first `wineeval.py` run downloads `all-MiniLM-L6-v2` (about 90 MB) from
Hugging Face; after that it uses the cached copy (set `HF_HUB_OFFLINE=1` to skip the hub check). Without the `dense` extra it runs
BM25 only and says so. The results above took about three minutes on an
Apple M2 CPU. BM25 and the hybrid are deterministic; dense scores can differ
in the last decimal across torch versions and hardware.

## Course notebooks

The original lab, completed and run end-to-end: embeddings with Sentence
Transformers, an in-memory Qdrant index and a semantic search. Install with
`uv sync --extra notebooks`.

1. [Managing data](./examples/1-managing-data/example.ipynb): load the CSV with pandas.
2. [Embeddings](./examples/2-embeddings/embeddings.ipynb): encode the notes and search them in Qdrant.
3. [Applied RAG](./examples/3-applied-rag/embeddings.ipynb): retrieve wines and pass them to an LLM.

Two fixes were needed on a current stack: `qdrant-client`'s `.search()` was
replaced by `.query_points()`, and the old `pandas==1.3.5` pin no longer
builds on Python 3.12. The notebooks index the notes only (the applied-RAG one over a
random 700-row sample), so their output is not comparable with the eval above. The last cell
of the applied-RAG notebook needs a local LLM endpoint (Llamafile or any
OpenAI-compatible server) and was not run here. [`lab.md`](./lab.md) is the
course's practice brief. Template:
[alfredodeza/learn-retrieval-augmented-generation](https://github.com/alfredodeza/learn-retrieval-augmented-generation).

## License

MIT
