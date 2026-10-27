55 queries (36 literal, 19 paraphrase), 1365 docs, relevant per query min 3 / median 14 / max 91. Mean with bootstrap 95% CI over queries.

**all** (n=55)

| retriever | recall@5 | recall@10 | MRR |
|---|---|---|---|
| BM25 | 0.66 (0.55 to 0.77) | 0.66 (0.55 to 0.77) | 0.71 (0.59 to 0.82) |
| dense (all-MiniLM-L6-v2) | 0.58 (0.49 to 0.69) | 0.55 (0.46 to 0.64) | 0.74 (0.64 to 0.85) |
| hybrid (RRF, k=60) | 0.68 (0.59 to 0.78) | 0.67 (0.58 to 0.77) | 0.81 (0.71 to 0.90) |
| dense (all-MiniLM-L6-v2) minus BM25 | -0.08 (-0.16 to 0.00) | -0.11 (-0.20 to -0.02) | 0.03 (-0.04 to 0.11) |
| hybrid (RRF, k=60) minus BM25 | 0.02 (-0.03 to 0.08) | 0.01 (-0.04 to 0.07) | 0.10 (0.02 to 0.18) |

**literal** (n=36)

| retriever | recall@5 | recall@10 | MRR |
|---|---|---|---|
| BM25 | 0.91 (0.85 to 0.97) | 0.92 (0.87 to 0.97) | 0.95 (0.90 to 1.00) |
| dense (all-MiniLM-L6-v2) | 0.73 (0.63 to 0.83) | 0.68 (0.58 to 0.77) | 0.91 (0.82 to 0.99) |
| hybrid (RRF, k=60) | 0.87 (0.81 to 0.94) | 0.87 (0.80 to 0.93) | 0.95 (0.88 to 1.00) |
| dense (all-MiniLM-L6-v2) minus BM25 | -0.18 (-0.26 to -0.11) | -0.24 (-0.34 to -0.16) | -0.04 (-0.13 to 0.04) |
| hybrid (RRF, k=60) minus BM25 | -0.04 (-0.09 to -0.01) | -0.06 (-0.11 to -0.01) | -0.01 (-0.07 to 0.06) |

**paraphrase** (n=19)

| retriever | recall@5 | recall@10 | MRR |
|---|---|---|---|
| BM25 | 0.18 (0.06 to 0.32) | 0.16 (0.05 to 0.30) | 0.24 (0.10 to 0.42) |
| dense (all-MiniLM-L6-v2) | 0.31 (0.16 to 0.47) | 0.31 (0.18 to 0.45) | 0.42 (0.24 to 0.62) |
| hybrid (RRF, k=60) | 0.33 (0.21 to 0.46) | 0.31 (0.19 to 0.45) | 0.55 (0.35 to 0.76) |
| dense (all-MiniLM-L6-v2) minus BM25 | 0.13 (-0.02 to 0.27) | 0.14 (0.00 to 0.29) | 0.18 (0.07 to 0.33) |
| hybrid (RRF, k=60) minus BM25 | 0.15 (0.04 to 0.27) | 0.15 (0.06 to 0.26) | 0.30 (0.15 to 0.48) |
